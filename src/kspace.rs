//! Acquisition stage — per-slice k-space: EPI geometric distortion, T2* relaxation, eddy
//! currents, Nyquist ghosting, partial Fourier, Gibbs ringing, spikes, multi-coil combine, and
//! GRAPPA. Faithful port of the core DFT in `Algorithms/itkKspaceImageFilter.cpp:452`:
//!
//! ```text
//! kspace[kx,ky] = (1/N) Σ_{x,y} f(x,y)·exp( i·2π·( kx·x + ky·y + φ ) ),   φ = fmap(x,y)·t(ky)
//! image[x,y]    =        Σ_{kx,ky} kspace[kx,ky]·exp( -i·2π·( kx·x + ky·y ) )
//! ```
//! `f = Σ_c comp_c · exp(-tRf/T2_c − |t|/tInhom) · signal_scale`. The `φ = fmap·t(ky)` term (the
//! readout time increases with the PE line) is what warps EPI along the phase-encode axis — the
//! same physics that makes AP/PA reverse-PE pairs distort oppositely (the DRBUDDI/topup target).
//!
//! The transform is exact — the literal O(N³) sum, not an approximation of it — so it stays
//! directly comparable against Fiberfox. It is organised for speed without changing a term: the
//! per-line factors are split into static (hoisted), affine-in-ky (advanced by per-voxel rotors),
//! per-line scalars (relaxation) and per-axis separable (the eddy polynomial), leaving an inner
//! loop of complex multiply-adds; the x-DFT of each line and the 2-D reconstruction are FFTs
//! (`rustfft`, behind the `kspace` feature) or twiddle-table sums (default, std-only) — same
//! numbers either way (`restructured_forward_matches_the_literal_sum`,
//! `fft_inverse_matches_the_direct_dft`). Under `kspace` the y-sum with the fieldmap is also
//! O(N log N): read as geometry it is a source warp `y − sny·τ·fmap`, evaluated by a type-1
//! NUFFT (`nufft.rs`, ~1e-13); only the legacy gradient-model eddy polynomial, whose time profile
//! is not affine in ky, keeps the O(N³) rotor path.

use crate::phase::{PhaseModel, ShotPhase};
use crate::readout::{Readout, SingleShotEpi};
use std::f64::consts::TAU;

#[derive(Clone, Copy)]
pub(crate) struct C {
    pub(crate) re: f64,
    pub(crate) im: f64,
}
impl C {
    pub(crate) const ZERO: C = C { re: 0.0, im: 0.0 };
    #[inline]
    #[cfg_attr(feature = "kspace", allow(dead_code))]
    pub(crate) fn cis(theta: f64) -> C {
        C { re: theta.cos(), im: theta.sin() }
    }
    #[inline]
    pub(crate) fn add(self, o: C) -> C {
        C { re: self.re + o.re, im: self.im + o.im }
    }
    #[inline]
    pub(crate) fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    #[inline]
    pub(crate) fn scale(self, s: f64) -> C {
        C { re: self.re * s, im: self.im * s }
    }
    #[inline]
    pub(crate) fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// Deterministic Gaussian source (SplitMix64 + Box–Muller), std-only so k-space stays dep-free.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// standard normal
    pub fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// How partial Fourier drops phase-encode lines.
///
/// These differ by more than an off-by-one, and the difference matters for anything evaluating
/// PF-aware unringing (RPG and friends), so it is an explicit choice rather than a hidden quirk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialFourierMode {
    /// Fiberfox's rule, ported as-is. On an even matrix it additionally **preserves line zero**,
    /// because that line's Nyquist conjugate is absent. The consequence is that nominal "6/8" is
    /// not 75%: at `ny = 32` it keeps 25 of 32 lines, **78.1%**, and leaves one isolated line
    /// disconnected from the acquired block. Faithful to the reference implementation, and the
    /// default so existing behaviour is unchanged.
    FiberfoxCompatible,
    /// Conventional contiguous zero-filled partial Fourier: keep exactly `round(ny * pf)`
    /// consecutive lines from one end. This is what a scanner produces, and it is the more
    /// relevant condition for benchmarking PF-aware reconstruction -- especially now the object is
    /// genuinely complex and has no Hermitian symmetry to exploit. The kept lines are the ones
    /// acquired *before* the centre, so timing is unchanged (the centre is reached as late as at
    /// full Fourier).
    Contiguous,
    /// What a scanner does (TRXScan's default; ported from TRXScan `main`): the train **starts
    /// late** and skips the first `ny - round(ny*pf)` lines it would have acquired, so the k-space
    /// centre is reached `ny*(1-pf)` line times sooner and the readout is that much shorter.
    /// Exactly `round(ny*pf)` consecutive lines are kept: the centre plus the side acquired *after*
    /// it. `t_echo` is still the caller's number. The eddy-decay clock (time since the readout
    /// began) counts from the first acquired line ([`LineTiming::for_acquisition`]). Defined by the
    /// 2D EPI train's line order, so the 3D readouts refuse it.
    Scanner,
}

/// How many lines at the start of the EPI train a [`PartialFourierMode::Scanner`] acquisition
/// skips (0 for the other modes and at full Fourier).
pub fn pf_skipped_lines(ny: usize, acq: &Acquisition) -> usize {
    if acq.partial_fourier < 1.0 && acq.pf_mode == PartialFourierMode::Scanner {
        ny - ((ny as f64 * acq.partial_fourier).round() as usize).min(ny)
    } else {
        0
    }
}

/// Scanner-side reconstruction apodization. Distinct transfer functions with distinct PSFs, so
/// they are named rather than hidden behind one ambiguous scalar.
///
/// `None` is the default and is what the algorithm-validation fixture uses: Kellner/`mrdegibbs`
/// assume an unapodized rectangular window. Note this is the default *window*; TRXScan's shipping
/// acquisition default remains 6/8 partial Fourier, and the full-Fourier benchmark fixture sets
/// `partial_fourier = 1.0` explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KspaceWindow {
    None,
    Tukey { alpha: f64 },
    Hann,
    Fermi { radius: f64, width: f64 },
}

impl KspaceWindow {
    /// Window value at normalized frequency `(kx, ky)`, each in `[-0.5, 0.5]`.
    ///
    /// **These are RADIAL windows**, `r = 2*hypot(kx, ky)`, not the more common separable
    /// Cartesian form `W(kx)*W(ky)`. Radial is isotropic in-plane, which suits a benchmark whose
    /// edges are not axis-aligned; a scanner reproducing a specific vendor filter may need the
    /// separable convention instead.
    pub fn at(&self, kx: f64, ky: f64) -> f64 {
        let r = 2.0 * (kx * kx + ky * ky).sqrt(); // 0 at centre, 1 at the band edge
        match *self {
            KspaceWindow::None => 1.0,
            KspaceWindow::Hann => {
                if r >= 1.0 { 0.0 } else { 0.5 * (1.0 + (std::f64::consts::PI * r).cos()) }
            }
            KspaceWindow::Tukey { alpha } => {
                let a = alpha.clamp(0.0, 1.0);
                if r <= 1.0 - a {
                    1.0
                } else if r >= 1.0 {
                    0.0
                } else {
                    0.5 * (1.0 + (std::f64::consts::PI * (r - (1.0 - a)) / a.max(1e-12)).cos())
                }
            }
            KspaceWindow::Fermi { radius, width } => {
                1.0 / (1.0 + ((r - radius) / width.max(1e-12)).exp())
            }
        }
    }
}

/// Acquisition parameters for the k-space stage.
#[derive(Debug, Clone)]
pub struct Acquisition {
    pub t_line: f64,      // ms per PE line
    pub t_echo: f64,      // ms
    pub t_inhom: f64,     // ms, T2* inhomogeneity time
    pub signal_scale: f64,
    pub reverse_phase: bool,
    pub do_distortions: bool,
    pub do_relaxation: bool,
    /// Per-component variance of the reconstructed complex image at **full sampling, single coil,
    /// pre-combination**: `Var(Re n) = Var(Im n) = noise_variance`, so `E[|n|^2] = 2*noise_variance`.
    /// The per-k-space-sample variance is derived from the reconstruction normalization; the
    /// combined multi-coil variance emerges from the coil model as `V / sum_c s_c^2`.
    pub noise_variance: f64,
    pub partial_fourier: f64, // fraction of PE lines acquired (1.0 = full); skips low-ky lines
    /// Which PF line-dropping rule to use. See [`PartialFourierMode`]: the default is NOT exactly
    /// `partial_fourier` of the lines on an even matrix.
    pub pf_mode: PartialFourierMode,
    pub ghost_offset: f64,    // Nyquist ghost: kx offset (±) on odd/even PE lines (0 = off)
    pub eddy_strength: f64,   // linear (in-plane) eddy-current phase scale (0 = off)
    pub eddy_quad: f64,       // quadratic (x²,y²,z²) eddy-current phase scale (0 = off)
    pub eddy_phase: f64,      // eddy OBJECT-phase ramp: rad per unit (bvec·bval) per acquired voxel;
                              // direction- and b-dependent, imprints reconstructed phase (0 = off)
    pub eddy_tau: f64,        // eddy-current decay time (ms)
    pub n_spikes: usize,      // random k-space spikes per slice (0 = off) → herringbone
    pub spike_amplitude: f64, // spike magnitude as a fraction of the peak k-space sample
    pub window: KspaceWindow, // reconstruction apodization; None = unapodized rectangular
    pub n_coils: usize,       // receiver coils (1 = uniform single coil); ring-arranged sensitivities
    pub accel: usize,         // GRAPPA acceleration R (1 = fully sampled); undersamples PE lines
    pub acs_lines: usize,     // GRAPPA autocalibration lines (fully-sampled central PE band)
    pub seed: u64,            // mixed into every derived per-slice seed; 0 reproduces legacy output
    pub echo: EchoFormation,  // how the echo forms: spin echo (default) or gradient echo
}

/// How the echo forms (P5 addendum, part A). `Spin`: static off-resonance is refocused at the
/// echo, so `T2'` decays as `exp(-|t|/T2')` about it and the fieldmap phase is `2 pi fmap t`, `t`
/// from the echo. `Gradient`: nothing is refocused, so `T2'` decays from the RF,
/// `exp(-trf/T2')`, and the fieldmap phase is `2 pi fmap (TE + t)`, whose static part
/// `2 pi fmap TE` joins the object phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EchoFormation {
    #[default]
    Spin,
    Gradient,
}

/// Spatial sensitivity of `coil` (of `n_coils` arranged in a ring) at continuous position
/// `(x, y)` in ACQUIRED-voxel units, on an `nx` x `ny` acquired matrix. Uniform for a single coil;
/// otherwise a Gaussian falloff from the coil position (distinct per coil — the spatial diversity
/// GRAPPA/SENSE need). Not normalized; the Roemer combine handles the overall scale.
///
/// The coordinates are continuous and acquired-voxel-based on purpose. This function is called
/// from two places that walk DIFFERENT grids — the forward model walks the oversampled simulation
/// grid, the Roemer combine walks the acquired matrix — and it must describe the same physical
/// field to both, or the combine divides by sensitivities the signal was never multiplied by.
/// Taking `(x: usize, nx: usize)` and centring on `nx/2` did not: at oversampling `o` the forward
/// model's field was displaced by `(o-1)/(2o)` acquired voxels relative to the combine's, because
/// the sim cells composing an acquired voxel are centred on it (`xoff = (o-1)/2`) rather than
/// starting at it. Sub-voxel and negligible for these very smooth Gaussians — and exactly zero for
/// the canonical Gibbs benchmark, which forces `n_coils = 1` — but wrong, and it would stop being
/// negligible the moment anyone supplied a sharper sensitivity map.
pub(crate) fn coil_sensitivity(coil: usize, n_coils: usize, x: f64, y: f64, nx: usize, ny: usize) -> f64 {
    if n_coils <= 1 {
        return 1.0;
    }
    let (cx, cy) = (nx as f64 / 2.0, ny as f64 / 2.0);
    let r = 0.6 * nx.max(ny) as f64;
    let ang = TAU * coil as f64 / n_coils as f64;
    let (px, py) = (cx + r * ang.cos(), cy + r * ang.sin());
    let d2 = (x - px).powi(2) + (y - py).powi(2);
    let sigma = 0.9 * nx.max(ny) as f64;
    (-d2 / (2.0 * sigma * sigma)).exp() + 0.15
}

impl Default for Acquisition {
    fn default() -> Self {
        Acquisition {
            t_line: 1.0,
            t_echo: 90.0,
            t_inhom: 50.0,
            signal_scale: 100.0,
            reverse_phase: false,
            do_distortions: true,
            do_relaxation: true,
            noise_variance: 0.0,
            partial_fourier: 1.0,
            pf_mode: PartialFourierMode::FiberfoxCompatible,
            ghost_offset: 0.0,
            eddy_strength: 0.0,
            eddy_quad: 0.0,
            eddy_phase: 0.0,
            eddy_tau: 70.0,
            n_spikes: 0,
            spike_amplitude: 1.0,
            window: KspaceWindow::None,
            n_coils: 1,
            accel: 1,
            acs_lines: 24,
            seed: 0,
            echo: EchoFormation::Spin,
        }
    }
}

/// Per-PE-line readout times (ms): `t` from max echo, `tRf` from RF, `tRead` from the last
/// preparation gradient (drives eddy-current decay).
fn line_times(epi: &SingleShotEpi) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (nx, ny) = (epi.kx_max, epi.ky_max);
    let xs = nx / 2;
    let (mut t, mut trf, mut tread) = (vec![0.0; ny], vec![0.0; ny], vec![0.0; ny]);
    for tick in 0..nx * ny {
        let (kx, ky) = epi.kspace_index(tick);
        if kx == xs {
            t[ky] = epi.time_from_max_echo(tick);
            trf[ky] = epi.time_from_rf(tick);
            tread[ky] = epi.time_from_prep_gradient(tick);
        }
    }
    (t, trf, tread)
}

/// Per-phase-encode-line timing and readout polarity, the table the forward model reads (P5
/// addendum, "Generalizing the 2D forward"). Indexed by `ky`. `t_ms` from the echo centre (fieldmap
/// phase, `T2'`), `trf_ms` from the RF (T2), `tread_ms` from the last prep gradient (eddy decay),
/// `polarity` `+1`/`-1` the readout direction of the line (the Nyquist ghost's sign).
#[derive(Debug, Clone, PartialEq)]
pub struct LineTiming {
    pub t_ms: Vec<f64>,
    pub trf_ms: Vec<f64>,
    pub tread_ms: Vec<f64>,
    pub polarity: Vec<i8>,
}

impl LineTiming {
    /// The single-shot EPI's table: [`line_times`] unchanged, and the polarity alternating with
    /// `ky`, which is what the ghost has always used.
    pub fn from_epi(epi: &SingleShotEpi) -> LineTiming {
        let (t_ms, trf_ms, tread_ms) = line_times(epi);
        let polarity = (0..t_ms.len()).map(|ky| if ky % 2 == 1 { -1 } else { 1 }).collect();
        LineTiming { t_ms, trf_ms, tread_ms, polarity }
    }

    /// The table [`build_coil_kspace`] uses for an [`Acquisition`] on an `nx x ny` matrix. Under
    /// [`PartialFourierMode::Scanner`] the readout starts `skip` lines into the nominal train, so
    /// the eddy-decay clock (`tread_ms`) starts there; `t_ms` and `trf_ms` are unchanged.
    pub fn for_acquisition(acq: &Acquisition, nx: usize, ny: usize) -> LineTiming {
        let mut lt = LineTiming::from_epi(&SingleShotEpi {
            kx_max: nx,
            ky_max: ny,
            t_line: acq.t_line,
            t_echo: acq.t_echo,
            reverse_phase: acq.reverse_phase,
        });
        let offset = pf_skipped_lines(ny, acq) as f64 * acq.t_line;
        if offset > 0.0 {
            for v in lt.tread_ms.iter_mut() {
                *v -= offset;
            }
        }
        lt
    }
}

/// The per-line timing of the single-shot EPI readout an [`Acquisition`] implies on an
/// `nx` x `ny` matrix: what the forward model uses for relaxation, T2* decay and eddy decay
/// ([`LineTiming::for_acquisition`]), with the acquisition order. Public so front ends can
/// annotate k-space figures with the simulator's own timing (ported from TRXScan `main`).
#[derive(Debug, Clone, PartialEq)]
pub struct EpiTiming {
    /// ms from the k-space centre (echo) at the centre of each ky line, indexed by ky.
    pub t_ms: Vec<f64>,
    /// ms since the RF pulse, indexed by ky (`t_echo + t_ms`).
    pub t_rf_ms: Vec<f64>,
    /// ms since the readout started (drives eddy-current decay), indexed by ky. With
    /// [`PartialFourierMode::Scanner`] the clock starts at the first *acquired* line, so the
    /// skipped lines carry negative values.
    pub t_read_ms: Vec<f64>,
    /// ky lines in the order they are acquired (`ny` entries).
    pub order: Vec<usize>,
    /// ms per PE line.
    pub t_line: f64,
    /// Echo time (ms).
    pub t_echo: f64,
    /// ms per k-space sample (`t_line / nx`).
    pub dt: f64,
}

/// Readout timing per ky line for `acq` on an `nx` x `ny` matrix. See [`EpiTiming`].
pub fn epi_timing(nx: usize, ny: usize, acq: &Acquisition) -> EpiTiming {
    let epi = SingleShotEpi {
        kx_max: nx, ky_max: ny, t_line: acq.t_line, t_echo: acq.t_echo, reverse_phase: acq.reverse_phase,
    };
    let lt = LineTiming::for_acquisition(acq, nx, ny);
    let order = (0..ny).map(|line| epi.kspace_index(line * nx).1).collect();
    EpiTiming { t_ms: lt.t_ms, t_rf_ms: lt.trf_ms, t_read_ms: lt.tread_ms, order, t_line: acq.t_line, t_echo: acq.t_echo,
                dt: epi.dt() }
}

/// The full k-space trajectory: `(kx, ky, ms from the echo)` for every sample tick, in
/// acquisition order (`nx * ny` entries). Partial Fourier / GRAPPA line skipping is a property
/// of [`sampling_mask`], not of the trajectory.
pub fn epi_trajectory(nx: usize, ny: usize, acq: &Acquisition) -> Vec<(usize, usize, f64)> {
    let epi = SingleShotEpi {
        kx_max: nx, ky_max: ny, t_line: acq.t_line, t_echo: acq.t_echo, reverse_phase: acq.reverse_phase,
    };
    (0..nx * ny)
        .map(|tick| {
            let (kx, ky) = epi.kspace_index(tick);
            (kx, ky, epi.time_from_max_echo(tick))
        })
        .collect()
}

/// Per-compartment T2 (or T2') for ONE slice. Goes in [`SliceInput`].
#[derive(Debug, Clone, Copy)]
pub enum T2Slice<'a> {
    Uniform(f32),
    /// `snx * sny`, layout `x + snx*y`. Same convention as [`SliceInput::fmap`].
    Map(&'a [f32]),
}

/// Per-compartment T2 (or T2') for the WHOLE volume. Goes to the entry points, which cut each
/// z slice themselves.
#[derive(Debug, Clone, Copy)]
pub enum T2Volume<'a> {
    Uniform(f32),
    /// `snx * sny * nz`, layout `x + snx*(y + sny*z)`.
    Map(&'a [f32]),
}

/// Every map value must be strictly positive; `f32::INFINITY` is allowed and means no decay.
/// Zero is forbidden: `-trf/0` is `-inf` when `trf > 0` (harmless) but `NaN` when `trf == 0`,
/// and `0 * NaN` poisons the whole Fourier sum.
pub fn validate_t2_map(m: &[f32]) -> Result<(), String> {
    for (i, &v) in m.iter().enumerate() {
        if v.is_nan() || v <= 0.0 {
            return Err(format!("T2 map value {v} at index {i} is not strictly positive"));
        }
    }
    Ok(())
}

/// `trf = t_echo + t` must be positive on every acquired line. A negative `trf` is a readout
/// that begins before the excitation, and the forward model answers it with signal growth.
/// Partial Fourier helps only in Scanner mode, which skips the first lines of the train; the
/// Fiberfox and Contiguous modes drop lines this trajectory reads LAST.
pub fn validate_acquisition_timing(acq: &Acquisition, nx: usize, ny: usize) -> Result<(), String> {
    // The same table `build_coil_kspace` reads, or the timing this validates is not the timing
    // that runs.
    let trf_ms = LineTiming::for_acquisition(acq, nx, ny).trf_ms;
    let mask = sampling_mask(nx, ny, acq);
    let mut worst = f64::INFINITY;
    for ky in 0..ny {
        if mask[ky * nx] && trf_ms[ky] < worst {
            worst = trf_ms[ky];
        }
    }
    if worst <= 0.0 {
        let need = acq.t_echo - worst;
        return Err(format!(
            "trf = {worst:.3} ms <= 0 on an acquired line: the readout starts before the \
             excitation. Raise t_echo above {need:.3} ms or shorten t_line. Partial Fourier \
             helps only in Scanner mode (it skips the first lines); the Fiberfox and Contiguous \
             modes drop lines this trajectory reads last."));
    }
    Ok(())
}

/// Check every `Map` in a per-compartment volume list: right length, strictly positive values.
fn validate_t2_volumes(what: &str, vols: &[T2Volume], nvox: usize) {
    for (c, v) in vols.iter().enumerate() {
        if let T2Volume::Map(m) = v {
            assert_eq!(m.len(), nvox, "{what} map for compartment {c} is not on the simulation grid");
            if let Err(e) = validate_t2_map(m) {
                panic!("{what} map for compartment {c}: {e}");
            }
        }
    }
}

/// Cut one z slice out of each per-compartment volume.
fn t2_slices<'a>(vols: &'a [T2Volume<'a>], z: usize, nplane: usize) -> Vec<T2Slice<'a>> {
    vols.iter()
        .map(|v| match v {
            T2Volume::Uniform(s) => T2Slice::Uniform(*s),
            T2Volume::Map(m) => T2Slice::Map(&m[z * nplane..(z + 1) * nplane]),
        })
        .collect()
}

/// Everything one slice's forward model needs. The simulation grid (`sim`) and the acquired
/// matrix (`acq_matrix`) are distinct: the object lives on the finer grid, and only the central
/// `acq_matrix` block of its k-space is evaluated (spec 3.1).
pub struct SliceInput<'a> {
    /// Per-compartment images on the SIM grid, layout `x + snx*y`.
    pub compartments: &'a [&'a [f32]],
    /// Per-compartment T2 (ms): one scalar, or one map on the SIM grid. Any `Map` here or in
    /// `t_inhom` takes the slice off the per-line-scalar (NUFFT) path.
    pub t2: &'a [T2Slice<'a>],
    /// Per-compartment T2' inhomogeneity time (ms), same shape as `t2`; `None` falls back to
    /// `Acquisition::t_inhom` for every compartment. `f32::INFINITY` means no inhomogeneity decay.
    pub t_inhom: Option<&'a [T2Slice<'a>]>,
    /// Off-resonance field (Hz) on the SIM grid.
    pub fmap: &'a [f32],
    /// Pre-readout object phase (radians) on the SIM grid. Ignored until Task 6.
    pub phase0: Option<&'a [f64]>,
    /// `[snx, sny]` — simulation grid, in-plane.
    pub sim: [usize; 2],
    /// `[nx, ny]` — acquired matrix.
    pub acq_matrix: [usize; 2],
    pub z: usize,
    pub nz: usize,
    /// Per-volume eddy drive in the legacy model's units, `bvec * bval` for diffusion.
    /// `None` disables eddy for this volume, reproducing the old `bval ~= 0` branch.
    /// `Some([0.0; 3])` does NOT: an FSL b0 row (`bval = 5`, zero `bvec`) is `Some`, keeps
    /// `do_eddy` true with identity rotors, and keeps the NUFFT path disabled.
    pub eddy_drive: Option<[f64; 3]>,
    /// Per-volume preparation-gradient drive for the phase model: `(magnitude, direction)`. The
    /// direction is NOT required to be normalized — `PrepPhase::shot` normalizes, and moving that
    /// division to the caller changes floating-point operation order and breaks bit-identity.
    /// `None` disables the prep phase term for this volume. Informational at slice level: the
    /// shot is realised at the entry point and arrives here through `phase0`.
    pub prep_drive: Option<(f64, [f64; 3])>,
    pub slice_seed: u64,
    /// Optional per-volume linear eddy shear `[a_x, a_y, a_z]` (dimensionless: PE shift in acquired
    /// voxels per acquired voxel of position). Replays a real DIFFPREP/TORTOISE eddy estimate
    /// (`--eddy-trace`), applied as a geometric PE distortion `shift = a·pos`, INSTEAD of the
    /// gradient-model `--eddy` — real eddy does not track the diffusion-gradient direction.
    pub eddy_lin: Option<[f64; 3]>,
}

/// A rectangle with sub-voxel-positioned edges on all four sides, with exact fractional occupancy
/// in every boundary voxel.
///
/// Unlike [`step_hires`], this has edges along BOTH axes, so an artifact that acts on only one --
/// partial Fourier, which zero-fills phase-encode lines -- actually shows up. A readout-only step
/// leaves the whole PF axis of a factor grid inert.
pub fn box_hires(snx: usize, sny: usize, x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<f32> {
    let cover = |lo: f64, hi: f64, a: f64, b: f64| -> f64 {
        (hi.min(b) - lo.max(a)).clamp(0.0, 1.0)
    };
    let mut v = vec![0.0f32; snx * sny];
    for y in 0..sny {
        let fy = cover(y as f64, y as f64 + 1.0, y0, y1);
        if fy <= 0.0 {
            continue;
        }
        for x in 0..snx {
            let fx = cover(x as f64, x as f64 + 1.0, x0, x1);
            v[x + snx * y] = (fx * fy) as f32;
        }
    }
    v
}

/// Step edge at continuous position `edge` (sim-voxel units) with exact fractional occupancy in
/// the boundary voxel. This is the same partial-volume representation the path-length rasterizer
/// produces for real anatomy, so it is a production code path, not a test-only fixture.
pub fn step_hires(snx: usize, sny: usize, edge: f64) -> Vec<f32> {
    let mut v = vec![0.0f32; snx * sny];
    for x in 0..snx {
        let (lo, hi) = (x as f64, x as f64 + 1.0);
        let frac = if hi <= edge {
            0.0
        } else if lo >= edge {
            1.0
        } else {
            hi - edge
        };
        for y in 0..sny {
            v[x + snx * y] = frac as f32;
        }
    }
    v
}

/// Build the acquired k-space for one coil, complete with the ringing mask, spikes and thermal
/// noise, but before GRAPPA, reconstruction and coil combination. Shared by [`simulate_slice`] and
/// [`simulate_slice_kspace`] so tests can inspect the coefficients before reconstruction.
/// Which k-space samples are actually acquired: partial Fourier plus GRAPPA undersampling.
/// Layout `kx + nx*ky`.
///
/// This is the single source of truth for both the forward build and the noise, so signal and
/// noise can never disagree about what was sampled. The original defect (section 1, finding 2.5)
/// was exactly that disagreement: noise was added after the line skip, populating k-space that
/// was never acquired.
pub fn sampling_mask(nx: usize, ny: usize, acq: &Acquisition) -> Vec<bool> {
    let ys = ny / 2;
    let accel = acq.accel.max(1);
    let acs_half = (acq.acs_lines / 2) as i64;
    let mut m = vec![false; nx * ny];
    // Scanner mode: exactly round(ny*pf) consecutive lines; the FIRST lines of the train are
    // skipped. The forward train starts at high ky, so the skipped block is the high-ky end;
    // reversed polarity starts at low ky and skips there. Contiguous mode keeps the same count
    // but drops the LAST lines instead (the low-ky end; high-ky when the polarity is reversed).
    let keep_n = (ny as f64 * acq.partial_fourier).round() as usize;
    for kyi in 0..ny {
        if acq.partial_fourier < 1.0 && acq.pf_mode == PartialFourierMode::Scanner {
            let skip = if acq.reverse_phase { kyi < ny - keep_n } else { kyi >= keep_n };
            if skip {
                continue;
            }
        } else if acq.partial_fourier < 1.0 && acq.pf_mode == PartialFourierMode::Contiguous {
            let skip = if acq.reverse_phase { kyi >= keep_n } else { kyi < ny - keep_n };
            if skip {
                continue;
            }
        } else if acq.partial_fourier < 1.0 {
            let skip = if acq.reverse_phase {
                kyi as f64 > (ny as f64 * acq.partial_fourier).ceil()
            } else {
                (kyi as f64) < (ny as f64 * (1.0 - acq.partial_fourier)).floor()
                    && (kyi > 0 || ny % 2 == 1)
            };
            if skip {
                continue;
            }
        }
        let acquired = accel <= 1
            || (kyi as i64 - ys as i64).abs() <= acs_half
            || kyi % accel == ys % accel;
        if !acquired {
            continue;
        }
        for kxi in 0..nx {
            m[kxi + nx * kyi] = true;
        }
    }
    m
}

/// The eddy eligibility decision, in one place. `None` disables; `Some` enables even for a
/// zero vector (the FSL b0 row, which must keep the NUFFT path disabled as it is today); a zero
/// `eddy_strength` disables regardless.
pub(crate) fn eddy_enabled(acq: &Acquisition, drive: Option<[f64; 3]>) -> bool {
    acq.eddy_strength != 0.0 && drive.is_some()
}

fn build_coil_kspace(inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize) -> Vec<C> {
    let [nx, ny] = inp.acq_matrix;
    build_coil_kspace_timed(inp, acq, coil, ncoils, &LineTiming::for_acquisition(acq, nx, ny), None)
}

/// [`build_coil_kspace`] with the line-timing table given. `log_amp` (the 3D path's voxel mode,
/// `kspace3d`) is a per-voxel log-amplitude on the simulation slice, added inside each
/// compartment's decay exponent so the echo amplitude and the within-echo decay are one
/// exponential (P5 addendum, "Echo amplitudes"); it forces the per-voxel path. The 2D path passes
/// `None` and computes exactly what it computed before.
pub(crate) fn build_coil_kspace_timed(
    inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize, timing: &LineTiming, log_amp: Option<&[f64]>,
) -> Vec<C> {
    let [snx, sny] = inp.sim;
    let [nx, ny] = inp.acq_matrix;
    assert!(
        snx % nx == 0 && sny % ny == 0,
        "sim grid must be an integer multiple of the acquired matrix"
    );
    assert!(
        timing.t_ms.len() == ny && timing.trf_ms.len() == ny && timing.tread_ms.len() == ny && timing.polarity.len() == ny,
        "line-timing table must have one entry per phase-encode line"
    );
    let (z, nz) = (inp.z, inp.nz);
    let (compartments, t2, fmap) = (inp.compartments, inp.t2, inp.fmap);
    let (t_ms, trf_ms, tread_ms) = (&timing.t_ms, &timing.trf_ms, &timing.tread_ms);
    let gradient = inp.eddy_drive.unwrap_or([0.0; 3]);
    // eddy currents affect volumes with a prep gradient only (`None` = the old b0 branch)
    let do_eddy = eddy_enabled(acq, inp.eddy_drive);
    let do_eddy_phase = acq.eddy_phase != 0.0 && inp.eddy_drive.is_some();
    let trt_s = acq.t_line * ny as f64 / 1000.0; // total readout time (s), for the shift<->phase map
    // acquired-matrix centres (k-space indexing) and sim-grid centres (image indexing)
    // Centred k-space indexing: the acquired band is [-n/2, n/2-1], asymmetric about k=0 by one
    // sample. This is deliberate, not an off-by-one -- real even-matrix Cartesian acquisitions
    // cover exactly this range. It gives a real object a small deterministic imaginary component,
    // which is NOT object phase; see `phase.rs` for that. Pinned by
    // `even_matrix_window_asymmetry_is_intentional`.
    let (xs, ys, zs) = (nx / 2, ny / 2, nz / 2);
    let (ox, oy) = (snx / nx, sny / ny); // in-plane oversampling factors
    // The sim-grid centre is the IMAGE of the acquired centre, o*(n/2), not snx/2: for an odd
    // acquired matrix those differ by one sim cell, i.e. (1/o) of an acquired voxel in-plane.
    let (sxs, sys) = (ox * xs, oy * ys);
    // Half-cell alignment. Sim cell `x` covers [x, x+1) in sim units, so its centre is at x+0.5;
    // acquired cell `X` covers o cells and is centred at o*X + o/2. Aligning sample index `x` with
    // `o*X` — as a bare index substitution does — therefore misregisters the object against the
    // reconstruction grid by (o-1)/2 sim cells, i.e. (o-1)/(2o) of an ACQUIRED voxel: 0.44 voxels
    // at o=8. Since this whole model turns on sub-voxel edge position, that shift is fatal and the
    // acquired k-space is measured from the acquired grid's centre instead. Exactly zero at o=1.
    let (xoff, yoff) = ((ox as f64 - 1.0) / 2.0, (oy as f64 - 1.0) / 2.0);
    let at = |x: usize, y: usize| x + snx * y; // SIM-grid image index
    let kat = |kx: usize, ky: usize| kx + nx * ky; // acquired k-space / acquired-image index
    let nvox = snx * sny;

    // Which samples are acquired (partial Fourier + GRAPPA undersampling). Single source of
    // truth, shared with the noise below so signal and noise cannot disagree.
    let mask = sampling_mask(nx, ny, acq);

    // ---- forward: k[kx,ky] = (1/N) Σ_x e^{i2π kx x} Σ_y mod_ky(x,y) e^{i2π ky y} ----
    //
    // The same O(N³) sum Fiberfox evaluates, but organised by how each factor of mod_ky depends
    // on the PE line, so that the inner loop is pure complex arithmetic with no transcendentals:
    //
    //   static      amp(r)·e^{iφ0(r)}: signal_scale, coil sensitivity, object phase, the eddy
    //               object-phase ramp — evaluated once per slice and coil;
    //   affine      e^{i2π(rate(r)·t(ky) + ky_norm·(y−c))}: the fieldmap (rate = fmap) and the
    //               replayed linear eddy shear (rate = a·pos/TRT) together with the y-DFT kernel.
    //               `t` is affine in the line index per line parity (`readout.rs`), so this
    //               factor advances between acquired lines by a per-voxel ROTOR that depends only
    //               on the step (Δky, parity) — a complex multiply per voxel per line. The rotors
    //               are memoised per step and checked against the actual Δt, so an exotic
    //               readout timing degrades to the closed form rather than to a wrong answer;
    //   per line    the compartment relaxations e^{-tRf/T2_c − |t|/tInhom}: one scalar per
    //               compartment per line;
    //   separable   the gradient-model eddy polynomial × exp(-tRead/τ)·t: e^{iθ_x(x)}·e^{iθ_y(y)}
    //               ·e^{iθ_z}, i.e. snx + sny cis per line, not snx·sny.
    //
    // Round-off of the rotor recurrence is ~ny·ε; the state is re-anchored to the closed form
    // every `REANCHOR` acquired lines. Pinned against the literal per-line sum by
    // `restructured_forward_matches_the_literal_sum`.
    const REANCHOR: usize = 32;
    let zc = z as f64 - zs as f64;
    let zc_c = z as f64 - (nz as f64 - 1.0) / 2.0;
    let mut amp = vec![0.0f64; nvox];
    let mut phi0 = vec![0.0f64; nvox];
    let mut rate = vec![0.0f64; nvox];
    for y in 0..sny {
        for x in 0..snx {
            let i = at(x, y);
            // TWO coordinate frames, named apart on purpose. Both are in acquired-voxel
            // UNITS and both carry the half-cell registration (`xoff = (o-1)/2`), but they
            // have DIFFERENT ORIGINS, and a quantity evaluated in the wrong one is wrong by
            // half a FOV rather than by a sub-voxel amount:
            //
            //   xa — ABSOLUTE on the acquired grid, 0 .. nx-1, averaging to exactly `v` over
            //        the o sim cells of acquired voxel `v`. This is the frame the Roemer
            //        combine walks (`x in 0..nx`), so anything the combine must agree with —
            //        `coil_sensitivity`, whose coil ring is centred on nx/2 — uses it.
            //   xc — CENTRED on the acquired image, -nx/2 .. nx/2-1. The eddy polynomial is
            //        an expansion about the image centre and needs this one.
            //
            // Conflating them is not hypothetical: an earlier revision passed `xc` to
            // `coil_sensitivity`, which put the forward model's sensitivity field half a FOV
            // from the combine's at EVERY oversampling factor, o = 1 included (28% of peak).
            // Pinned end-to-end by
            // `multicoil_roemer_reproduces_the_single_coil_image_at_every_oversampling`.
            let (xa, ya) = ((x as f64 - xoff) / ox as f64, (y as f64 - yoff) / oy as f64);
            amp[i] = acq.signal_scale * coil_sensitivity(coil, ncoils, xa, ya, nx, ny);
            let (xc, yc) = (
                (x as f64 - sxs as f64 - xoff) / ox as f64,
                (y as f64 - sys as f64 - yoff) / oy as f64,
            );
            // Pre-readout object phase, already in radians (outside the TAU factor).
            let mut p0 = inp.phase0.map_or(0.0, |p| p[i]);
            if do_eddy_phase {
                // Eddy OBJECT-phase ramp: constant across the readout (NOT ∝ ky), so it
                // imprints the reconstructed object phase rather than distorting geometry.
                // Direction- and b-dependent (∝ gradient = bvec·bval), it reproduces the
                // per-volume phase-ramp variation real DWI shows (∝ gradient direction).
                // z centred on the volume (zc is slice-index-from-start, not centred).
                p0 += acq.eddy_phase * (gradient[0] * xc + gradient[1] * yc + gradient[2] * zc_c);
            }
            if acq.echo == EchoFormation::Gradient && acq.do_distortions {
                // Unrefocused off-resonance accrued from the RF to the echo: static per voxel,
                // so it joins the object phase. B0 only: the replayed eddy shear that shares
                // `rate` below is a readout-gradient effect and accrues nothing before the readout.
                p0 += TAU * fmap[i] as f64 * (acq.t_echo / 1000.0);
            }
            phi0[i] = p0;
            let mut r = if acq.do_distortions { fmap[i] as f64 } else { 0.0 };
            if let Some(a) = inp.eddy_lin {
                // Replay a real per-volume linear eddy shear as a PE geometric shift: a
                // phase (a·pos / TRT)·t behaves like a fieldmap of that value, giving
                // shift = a·pos acquired voxels (a is dimensionless shift-per-position).
                r += (a[0] * xc + a[1] * yc + a[2] * zc) / trt_s;
            }
            rate[i] = r;
        }
    }
    // Centred, half-cell-registered sim-grid coordinates in ACQUIRED-voxel units, per axis.
    let xc_of = |x: usize| (x as f64 - sxs as f64 - xoff) / ox as f64;
    let yc_of = |y: usize| (y as f64 - sys as f64 - yoff) / oy as f64;
    let ky_norm_of = |kyi: usize| (kyi as f64 - ys as f64) / sny as f64;

    // Rotor state: amp·e^{i(φ0 + 2π(rate·t + ky_norm·(y−c)))} for the current line.
    let (mut st_re, mut st_im) = (vec![0.0f64; nvox], vec![0.0f64; nvox]);
    let set_state = |kyi: usize, st_re: &mut [f64], st_im: &mut [f64]| {
        let t = t_ms[kyi] / 1000.0;
        let kyn = ky_norm_of(kyi);
        for y in 0..sny {
            let yy = y as f64 - sys as f64 - yoff;
            for x in 0..snx {
                let i = at(x, y);
                let ph = phi0[i] + TAU * (rate[i] * t + kyn * yy);
                st_re[i] = amp[i] * ph.cos();
                st_im[i] = amp[i] * ph.sin();
            }
        }
    };
    // Memoised rotors, keyed by (Δky, parity of the source line); each remembers the (Δt, Δky_norm)
    // it was built for so a mismatching step falls back to the closed form.
    struct Rotor { key: (usize, usize), dt: f64, dkyn: f64, re: Vec<f64>, im: Vec<f64> }
    let mut rotors: Vec<Rotor> = Vec::new();

    // Both terms of exp(-trf/T2 - |t|/T2') must be per-line scalars for the factoring — and
    // therefore the NUFFT path — to survive. One Map anywhere takes the whole slice to the rotor
    // path with per-voxel relaxation; Uniform compartments in a mixed slice keep their scalar.
    let uniform = t2.iter().all(|s| matches!(s, T2Slice::Uniform(_)))
        && inp.t_inhom.is_none_or(|ti| ti.iter().all(|s| matches!(s, T2Slice::Uniform(_))))
        && log_amp.is_none();
    if let Some(la) = log_amp {
        assert_eq!(la.len(), snx * sny, "log-amplitude map is not on the simulation slice");
    }
    // Compartment relaxation weights for this line (uniform case only).
    let mut rel = vec![1.0f64; compartments.len()];
    // Separable gradient-model eddy factors.
    let (mut ex_re, mut ex_im) = (vec![1.0f64; snx], vec![0.0f64; snx]);
    let (mut ey_re, mut ey_im) = (vec![1.0f64; sny], vec![0.0f64; sny]);
    // y-sum result and x-stage inputs.
    let (mut g_re, mut g_im) = (vec![0.0f64; snx], vec![0.0f64; snx]);
    let mut xstage = XStage::new(snx, nx, xs, sxs, xoff, acq.ghost_offset);

    // ---- NUFFT y-stage (feature `kspace`): the fieldmap as geometry ----
    // With t affine in the line index per parity, t(ky) = τ·ky + t0 + δ·[ky odd], the phase
    // 2π(rate·t + ky_norm·(y−c)) is 2π(ky−ys)·v/sny + const, with the SOURCE displaced to
    // v = (y − c) + sny·τ·rate — every voxel contributes from its distorted PE position. The
    // per-line y-sum then collapses to one type-1 NUFFT per column (per compartment, since the
    // relaxation is an output-side scalar; per parity, since δ adds a static phase to the odd
    // lines). The gradient-model eddy polynomial has a non-affine time profile and keeps the
    // rotor path. The timing model is checked, not assumed.
    #[cfg(feature = "kspace")]
    let nufft_rows: Option<Vec<(Vec<f64>, Vec<f64>)>> = (!do_eddy && ny >= 3 && uniform).then(|| {
        let tau = (t_ms[2] - t_ms[0]) / 2.0;
        let t0 = t_ms[0];
        let delta = t_ms[1] - (tau + t0);
        let tmax = t_ms.iter().fold(0.0f64, |m, t| m.max(t.abs())).max(1e-300);
        // the per-parity weights below also assume the readout polarity is the line parity's
        let affine = (0..ny).all(|k| {
            let pred = tau * k as f64 + t0 + if k % 2 == 1 { delta } else { 0.0 };
            (t_ms[k] - pred).abs() <= 1e-9 * tmax && timing.polarity[k] == if k % 2 == 1 { -1 } else { 1 }
        });
        if !affine {
            return None;
        }
        let (tau_s, delta_s) = (tau / 1000.0, delta / 1000.0);
        let t_centre = (tau * ys as f64 + t0) / 1000.0; // even-line time at the centre line
        let ncomp = compartments.len();
        let mut nufft = crate::nufft::Nufft1::new(sny, -(ys as i64), ny, 1e-13);
        // per-column buffers: positions, then (compartment, parity) weight vectors
        let mut pos = vec![0.0f64; sny];
        let mut wre = vec![vec![0.0f64; sny]; 2 * ncomp];
        let mut wim = vec![vec![0.0f64; sny]; 2 * ncomp];
        let mut out: Vec<(Vec<f64>, Vec<f64>)> = (0..2 * ncomp).map(|_| (Vec::new(), Vec::new())).collect();
        // rows[c] = (re, im) of the y-summed line, layout x + snx*kyi
        let mut rows: Vec<(Vec<f64>, Vec<f64>)> = (0..ncomp).map(|_| (vec![0.0; snx * ny], vec![0.0; snx * ny])).collect();
        for x in 0..snx {
            for y in 0..sny {
                let i = at(x, y);
                pos[y] = (y as f64 - sys as f64 - yoff) + sny as f64 * tau_s * rate[i];
                let ph = phi0[i] + TAU * rate[i] * t_centre;
                let (c0, s0) = (ph.cos(), ph.sin());
                let phd = TAU * rate[i] * delta_s;
                let (cd, sd) = (phd.cos(), phd.sin());
                // odd-line phase = even phase + δ term
                let (c1, s1) = (c0 * cd - s0 * sd, c0 * sd + s0 * cd);
                for (c, comp) in compartments.iter().enumerate() {
                    let a = amp[i] * comp[i] as f64;
                    wre[2 * c][y] = a * c0;
                    wim[2 * c][y] = a * s0;
                    wre[2 * c + 1][y] = a * c1;
                    wim[2 * c + 1][y] = a * s1;
                }
            }
            let weights: Vec<(&[f64], &[f64])> = (0..2 * ncomp).map(|q| (wre[q].as_slice(), wim[q].as_slice())).collect();
            nufft.run(&pos, &weights, &mut out);
            for c in 0..ncomp {
                for kyi in 0..ny {
                    let q = 2 * c + (kyi % 2);
                    rows[c].0[x + snx * kyi] = out[q].0[kyi];
                    rows[c].1[x + snx * kyi] = out[q].1[kyi];
                }
            }
        }
        Some(rows)
    }).flatten();
    #[cfg(not(feature = "kspace"))]
    let nufft_rows: Option<Vec<(Vec<f64>, Vec<f64>)>> = None;

    let mut kspace = vec![C::ZERO; nx * ny];
    let mut prev: Option<usize> = None;
    let mut since_anchor = 0usize;
    for kyi in 0..ny {
        // Not acquired (partial Fourier, or GRAPPA undersampling): this k-space row stays
        // zero and, critically, receives no noise either.
        if !mask[nx * kyi] {
            continue;
        }
        let t = t_ms[kyi] / 1000.0; // seconds
        let trf = trf_ms[kyi];
        // Advance the affine phase to this line: closed form on the first line and every
        // REANCHOR lines, otherwise one rotor multiply per voxel.
        if nufft_rows.is_none() { match prev {
            Some(p) if since_anchor < REANCHOR => {
                let key = (kyi - p, p % 2);
                let (dt, dkyn) = (t - t_ms[p] / 1000.0, ky_norm_of(kyi) - ky_norm_of(p));
                let idx = rotors.iter().position(|r| r.key == key).unwrap_or_else(|| {
                    let (mut re, mut im) = (vec![0.0f64; nvox], vec![0.0f64; nvox]);
                    for y in 0..sny {
                        let yy = y as f64 - sys as f64 - yoff;
                        for x in 0..snx {
                            let i = at(x, y);
                            let ph = TAU * (rate[i] * dt + dkyn * yy);
                            re[i] = ph.cos();
                            im[i] = ph.sin();
                        }
                    }
                    rotors.push(Rotor { key, dt, dkyn, re, im });
                    rotors.len() - 1
                });
                let r = &rotors[idx];
                if (r.dt - dt).abs() <= 1e-12 * dt.abs().max(1e-300) && r.dkyn == dkyn {
                    for i in 0..nvox {
                        let (a, b) = (st_re[i], st_im[i]);
                        st_re[i] = a * r.re[i] - b * r.im[i];
                        st_im[i] = a * r.im[i] + b * r.re[i];
                    }
                    since_anchor += 1;
                } else {
                    set_state(kyi, &mut st_re, &mut st_im);
                    since_anchor = 0;
                }
            }
            _ => {
                set_state(kyi, &mut st_re, &mut st_im);
                since_anchor = 0;
            }
        }
        prev = Some(kyi); }

        if uniform {
            for (c, w) in rel.iter_mut().enumerate() {
                *w = if acq.do_relaxation {
                    let T2Slice::Uniform(t2c) = t2[c] else { unreachable!("uniform branch") };
                    let tic = match inp.t_inhom.map(|ti| ti[c]) {
                        None => acq.t_inhom,
                        Some(T2Slice::Uniform(v)) => v as f64,
                        Some(T2Slice::Map(_)) => unreachable!("uniform branch"),
                    };
                    match acq.echo {
                        EchoFormation::Spin => (-trf / t2c as f64 - t.abs() * 1000.0 / tic).exp(),
                        EchoFormation::Gradient => (-trf / t2c as f64 - trf / tic).exp(),
                    }
                } else {
                    1.0
                };
            }
        }
        // Gradient-model eddy field growing through the readout: linear (g·pos) plus a
        // quadratic (g·pos²) term — the polynomial eddy/TORTOISE fit — times
        // exp(-tRead/τ)·t (itkKspaceImageFilter.cpp:354), so it grows with ky → geometric
        // DISTORTION. The polynomial is a sum over axes, so its phase factor is separable.
        let mut ez = (1.0f64, 0.0f64);
        if do_eddy {
            let eddy_decay = (-tread_ms[kyi] / acq.eddy_tau).exp() * t;
            for x in 0..snx {
                let xc = xc_of(x);
                let ph = TAU * (acq.eddy_strength * gradient[0] * xc + acq.eddy_quad * gradient[0] * xc * xc) * eddy_decay;
                ex_re[x] = ph.cos();
                ex_im[x] = ph.sin();
            }
            for y in 0..sny {
                let yc = yc_of(y);
                let ph = TAU * (acq.eddy_strength * gradient[1] * yc + acq.eddy_quad * gradient[1] * yc * yc) * eddy_decay;
                ey_re[y] = ph.cos();
                ey_im[y] = ph.sin();
            }
            let ph = TAU * (acq.eddy_strength * gradient[2] * zc + acq.eddy_quad * gradient[2] * zc * zc) * eddy_decay;
            ez = (ph.cos(), ph.sin());
        }

        // inner y-sum over the SIM grid → g(x): Σ_y (Σ_c rel_c comp_c) · state · e_y
        g_re.iter_mut().for_each(|v| *v = 0.0);
        g_im.iter_mut().for_each(|v| *v = 0.0);
        if let Some(rows) = &nufft_rows {
            for (c, (rre, rim)) in rows.iter().enumerate() {
                let row = &rre[snx * kyi..snx * (kyi + 1)];
                let rowi = &rim[snx * kyi..snx * (kyi + 1)];
                for x in 0..snx {
                    g_re[x] += rel[c] * row[x];
                    g_im[x] += rel[c] * rowi[x];
                }
            }
        } else {
        for y in 0..sny {
            let row = snx * y;
            let (eyr, eyi) = (ey_re[y], ey_im[y]);
            for x in 0..snx {
                let i = row + x;
                let mut w = 0.0f64;
                for (c, comp) in compartments.iter().enumerate() {
                    let r = if uniform {
                        rel[c]
                    } else if acq.do_relaxation {
                        let t2c = match t2[c] {
                            T2Slice::Uniform(v) => v as f64,
                            T2Slice::Map(m) => m[i] as f64,
                        };
                        let tic = match inp.t_inhom.map(|ti| ti[c]) {
                            None => acq.t_inhom,
                            Some(T2Slice::Uniform(v)) => v as f64,
                            Some(T2Slice::Map(m)) => m[i] as f64,
                        };
                        match (acq.echo, log_amp) {
                            (EchoFormation::Spin, None) => (-trf / t2c - t.abs() * 1000.0 / tic).exp(),
                            (EchoFormation::Gradient, None) => (-trf / t2c - trf / tic).exp(),
                            (EchoFormation::Spin, Some(la)) => (la[i] - trf / t2c - t.abs() * 1000.0 / tic).exp(),
                            (EchoFormation::Gradient, Some(la)) => (la[i] - trf / t2c - trf / tic).exp(),
                        }
                    } else {
                        1.0
                    };
                    w += r * comp[i] as f64;
                }
                let (sr, si) = (st_re[i], st_im[i]);
                let (mr, mi) = if do_eddy { (sr * eyr - si * eyi, sr * eyi + si * eyr) } else { (sr, si) };
                g_re[x] += w * mr;
                g_im[x] += w * mi;
            }
        }
        }
        if do_eddy {
            for x in 0..snx {
                let (a, b) = (g_re[x], g_im[x]);
                let (fr, fi) = (ex_re[x] * ez.0 - ex_im[x] * ez.1, ex_re[x] * ez.1 + ex_im[x] * ez.0);
                g_re[x] = a * fr - b * fi;
                g_im[x] = a * fi + b * fr;
            }
        }
        // x-DFT evaluated only at the acquired kx band, with the Nyquist (N/2) ghost: an
        // alternating readout-line kx offset (gradient-delay mismatch).
        let ghost_shift = if timing.polarity[kyi] < 0 { -acq.ghost_offset } else { acq.ghost_offset };
        xstage.run(&g_re, &g_im, ghost_shift, &mut kspace[nx * kyi..nx * (kyi + 1)]);
    }
    let n_inv = 1.0 / nvox as f64;
    for k in kspace.iter_mut() {
        k.re *= n_inv;
        k.im *= n_inv;
    }
    let _ = kat;

    // Spikes: overwrite random k-space points with a fraction of the peak sample → herringbone
    // ripple in the image (itkKspaceImageFilter.cpp:502).
    if acq.n_spikes > 0 {
        let (mut peak, mut peak_mag) = (C::ZERO, 0.0);
        for k in &kspace {
            let m = k.abs();
            if m > peak_mag {
                peak_mag = m;
                peak = *k;
            }
        }
        let spike = peak.scale(acq.spike_amplitude);
        // Spikes land only on ACQUIRED samples. Choosing arbitrary matrix coordinates could put
        // a spike on a partial-Fourier or GRAPPA-omitted line, which the scanner never read.
        let acquired: Vec<usize> = (0..nx * ny).filter(|&i| mask[i]).collect();
        if !acquired.is_empty() {
            let mut rng = Rng(inp.slice_seed ^ 0xA5A5_1234_5678_9ABC);
            for _ in 0..acq.n_spikes {
                let pick = acquired[(rng.next_u64() as usize) % acquired.len()];
                kspace[pick] = spike;
            }
        }
    }

    // Complex k-space noise, on ACQUIRED samples only.
    //
    // `noise_variance` is the PER-COMPONENT variance of the reconstructed complex image under
    // full sampling, single-coil, pre-combination: Var(Re n) = Var(Im n) = noise_variance, so
    // E[|n|^2] = 2*noise_variance. The per-sample variance follows from the reconstruction
    // normalization, which uses the ACQUIRED matrix, never the simulation grid.
    //
    // Masking alone produces the sqrt(f) scaling; there is deliberately NO additional
    // sampled-fraction factor, which would double-count it and give SD proportional to f.
    if acq.noise_variance > 0.0 {
        let mut rng = Rng((inp.slice_seed ^ (coil as u64).wrapping_mul(0x9E37_79B9)) | 1);
        let sigma = (acq.noise_variance / (nx * ny) as f64).sqrt();
        for (i, k) in kspace.iter_mut().enumerate() {
            if mask[i] {
                k.re += rng.gauss() * sigma;
                k.im += rng.gauss() * sigma;
            }
        }
    }

    kspace
}


/// The x-stage of the forward model: one PE line's y-summed row `g[x]` (sim grid, `snx` points)
/// → its `nx` acquired kx samples, `kx_norm = (kxi − xs + ghost)/snx`, measured from the acquired
/// grid's centre `sxs + xoff`. The Nyquist-ghost offset is an off-grid frequency shift, i.e. a
/// linear phase on `g`; the centre offset is a linear phase on the output; between them sits a
/// plain length-`snx` e^{+i2π} transform (rustfft's unnormalised inverse), of which the bins
/// `(kxi − xs) mod snx` are kept. Without the `kspace` feature the identical sum is evaluated
/// from precomputed twiddle tables (one per ghost polarity).
struct XStage {
    snx: usize,
    nx: usize,
    #[cfg(feature = "kspace")]
    xs: usize,
    #[cfg(feature = "kspace")]
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
    #[cfg(feature = "kspace")]
    buf: Vec<rustfft::num_complex::Complex<f64>>,
    #[cfg(feature = "kspace")]
    scratch: Vec<rustfft::num_complex::Complex<f64>>,
    /// Input ghost ramps `e^{i2π(±ghost)(x − sxs − xoff)/snx}` for [+ghost, −ghost].
    #[cfg(feature = "kspace")]
    ramp: [Vec<(f64, f64)>; 2],
    /// Output centre phase `e^{−i2π m (sxs + xoff)/snx}`, m = kxi − xs.
    #[cfg(feature = "kspace")]
    centre: Vec<(f64, f64)>,
    /// Twiddles `e^{i2π (m ± ghost)(x − sxs − xoff)/snx}`, layout `x + snx*kxi`, per polarity.
    #[cfg(not(feature = "kspace"))]
    tw: [Vec<(f64, f64)>; 2],
}

impl XStage {
    fn new(snx: usize, nx: usize, xs: usize, sxs: usize, xoff: f64, ghost: f64) -> XStage {
        let c = sxs as f64 + xoff;
        #[cfg(feature = "kspace")]
        {
            let fft = rustfft::FftPlanner::<f64>::new().plan_fft_inverse(snx);
            let scratch = vec![rustfft::num_complex::Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
            let ramp = [ghost, -ghost].map(|gh| {
                (0..snx).map(|x| { let p = TAU * gh * (x as f64 - c) / snx as f64; (p.cos(), p.sin()) }).collect()
            });
            let centre = (0..nx)
                .map(|kxi| { let p = -TAU * (kxi as f64 - xs as f64) * c / snx as f64; (p.cos(), p.sin()) })
                .collect();
            XStage { snx, nx, xs, fft, buf: vec![rustfft::num_complex::Complex::new(0.0, 0.0); snx], scratch, ramp, centre }
        }
        #[cfg(not(feature = "kspace"))]
        {
            let tw = [ghost, -ghost].map(|gh| {
                let mut t = Vec::with_capacity(snx * nx);
                for kxi in 0..nx {
                    let kx_norm = (kxi as f64 - xs as f64 + gh) / snx as f64;
                    for x in 0..snx {
                        let p = TAU * kx_norm * (x as f64 - c);
                        t.push((p.cos(), p.sin()));
                    }
                }
                t
            });
            let _ = xs;
            XStage { snx, nx, tw }
        }
    }

    /// `ghost_shift` is `±ghost` for this line's parity; `out` receives the nx samples (unnormalised).
    fn run(&mut self, g_re: &[f64], g_im: &[f64], ghost_shift: f64, out: &mut [C]) {
        let pol = if ghost_shift < 0.0 { 1 } else { 0 };
        #[cfg(feature = "kspace")]
        {
            for x in 0..self.snx {
                let (rr, ri) = self.ramp[pol][x];
                self.buf[x] = rustfft::num_complex::Complex::new(g_re[x] * rr - g_im[x] * ri, g_re[x] * ri + g_im[x] * rr);
            }
            self.fft.process_with_scratch(&mut self.buf, &mut self.scratch);
            for kxi in 0..self.nx {
                let m = kxi as i64 - self.xs as i64;
                let bin = m.rem_euclid(self.snx as i64) as usize;
                let (cr, ci) = self.centre[kxi];
                let v = self.buf[bin];
                out[kxi] = C { re: v.re * cr - v.im * ci, im: v.re * ci + v.im * cr };
            }
        }
        #[cfg(not(feature = "kspace"))]
        {
            let tw = &self.tw[pol];
            for kxi in 0..self.nx {
                let row = &tw[self.snx * kxi..self.snx * (kxi + 1)];
                let (mut ar, mut ai) = (0.0f64, 0.0f64);
                for x in 0..self.snx {
                    let (tr, ti) = row[x];
                    ar += g_re[x] * tr - g_im[x] * ti;
                    ai += g_re[x] * ti + g_im[x] * tr;
                }
                out[kxi] = C { re: ar, im: ai };
            }
        }
    }
}

/// The acquired k-space of the first coil, before reconstruction. Layout `kx + nx*ky`.
/// Used by the convergence tests (spec 4.1.6); not part of the simulation path.
pub fn simulate_slice_kspace(inp: &SliceInput, acq: &Acquisition) -> Vec<(f64, f64)> {
    build_coil_kspace(inp, acq, 0, acq.n_coils.max(1))
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}

/// Sample a [`PhaseModel`] onto one slice of the simulation grid.
///
/// Positions are in **acquired-voxel units** from the FOV centre, so the same coefficients describe
/// the same physical field at any oversampling factor `o`. The half-cell offset matches the forward
/// transform's registration convention (see [`simulate_slice`]).
pub fn phase_slice(
    model: &PhaseModel,
    shot: &ShotPhase,
    snx: usize,
    sny: usize,
    o: usize,
    z: usize,
    nz: usize,
) -> Vec<f64> {
    let (sxs, sys) = (snx as f64 / 2.0, sny as f64 / 2.0);
    let off = (o as f64 - 1.0) / 2.0;
    let zc = z as f64 - nz as f64 / 2.0;
    let mut v = vec![0.0f64; snx * sny];
    for y in 0..sny {
        for x in 0..snx {
            let r = [
                (x as f64 - sxs - off) / o as f64,
                (y as f64 - sys - off) / o as f64,
                zc,
            ];
            v[x + snx * y] = model.at(r, shot);
        }
    }
    v
}

/// Which intermediates [`simulate_slice_full`] should retain besides the combined image (ported
/// from TRXScan `main`). Everything off is what the production path uses; each flag costs one
/// `n_coils * nx * ny` complex buffer per slice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SliceCapture {
    /// Per-coil k-space as acquired: after spikes and k-space noise, BEFORE GRAPPA, unwindowed;
    /// un-acquired lines (partial Fourier, undersampling) are exactly zero.
    pub acquired: bool,
    /// Per-coil k-space as reconstructed: after GRAPPA and the reconstruction window — what is
    /// inverse-transformed.
    pub reconstructed: bool,
    /// Per-coil complex images, before the Roemer combine.
    pub coil_images: bool,
}

impl SliceCapture {
    pub const ALL: SliceCapture = SliceCapture { acquired: true, reconstructed: true, coil_images: true };
}

/// One slice's reconstruction with the requested intermediates. Complex values are `[re, im]`
/// pairs (the memory layout of `numpy.complex128`); all 2-D buffers are `kx + nx*ky` /
/// `x + nx*y`.
#[derive(Debug, Clone)]
pub struct SliceRecon {
    pub nx: usize,
    pub ny: usize,
    pub n_coils: usize,
    /// Which k-space samples were acquired (= [`sampling_mask`]).
    pub mask: Vec<bool>,
    /// Per coil; see [`SliceCapture::acquired`].
    pub acquired: Option<Vec<Vec<[f64; 2]>>>,
    /// Per coil; see [`SliceCapture::reconstructed`].
    pub reconstructed: Option<Vec<Vec<[f64; 2]>>>,
    /// Per coil; see [`SliceCapture::coil_images`].
    pub coil_images: Option<Vec<Vec<[f64; 2]>>>,
    /// Per coil, the real sensitivity on the acquired grid (what the combine divides by).
    pub sensitivities: Vec<Vec<f64>>,
    /// The Roemer-combined complex image, before the `f32` cast [`simulate_slice`] applies.
    pub combined: Vec<[f64; 2]>,
}

/// Simulate one slice: compartment images on the SIM grid (each `snx*sny`, layout `x + snx*y`) →
/// complex image on the ACQUIRED matrix (`nx*ny`). `t2` is the per-compartment T2 (ms), scalar
/// or map, as is the optional `t_inhom`; `fmap` is
/// the off-resonance field (Hz), same sim-grid layout. Only the central `nx*ny` block of the sim
/// grid's k-space is evaluated, so truncation to the nominal band happens during the forward
/// transform rather than by discarding a computed k-space (spec 3.1).
pub fn simulate_slice(inp: &SliceInput, acq: &Acquisition) -> Vec<(f32, f32)> {
    simulate_slice_full(inp, acq, SliceCapture::default())
        .combined
        .iter()
        .map(|c| (c[0] as f32, c[1] as f32))
        .collect()
}

/// [`simulate_slice`] with the intermediates selected by `cap` retained. The combined image is
/// identical to `simulate_slice`'s before its `f32` cast; capturing changes no arithmetic.
pub fn simulate_slice_full(inp: &SliceInput, acq: &Acquisition, cap: SliceCapture) -> SliceRecon {
    // Build each coil's k-space (undersampled for GRAPPA when accel>1), reconstruct, then combine.
    let ncoils = acq.n_coils.max(1);
    let mut coil_kspace: Vec<Vec<C>> = Vec::with_capacity(ncoils);
    for coil in 0..ncoils {
        coil_kspace.push(build_coil_kspace(inp, acq, coil, ncoils));
    }
    reconstruct_coils_full(coil_kspace, acq, inp.acq_matrix, cap)
}

/// One slice's (or partition's) reconstruction from its coils' acquired k-spaces: GRAPPA, the
/// window, the inverse transform and the Roemer combine, returning the combined complex image
/// on the acquired matrix in `f64`. [`simulate_slice`]'s reconstruction, moved here unchanged so
/// the 3D path (`kspace3d`) reconstructs each partition by the same code.
pub(crate) fn reconstruct_coils(coil_kspace: Vec<Vec<C>>, acq: &Acquisition, acq_matrix: [usize; 2]) -> Vec<C> {
    reconstruct_coils_full(coil_kspace, acq, acq_matrix, SliceCapture::default())
        .combined
        .into_iter()
        .map(|c| C { re: c[0], im: c[1] })
        .collect()
}

/// [`reconstruct_coils`] with the intermediates selected by `cap` retained.
fn reconstruct_coils_full(mut coil_kspace: Vec<Vec<C>>, acq: &Acquisition, acq_matrix: [usize; 2], cap: SliceCapture)
    -> SliceRecon
{
    let [nx, ny] = acq_matrix;
    let (xs, ys) = (nx / 2, ny / 2);
    let kat = |kx: usize, ky: usize| kx + nx * ky; // acquired k-space / acquired-image index
    let pairs = |v: &[C]| -> Vec<[f64; 2]> { v.iter().map(|c| [c.re, c.im]).collect() };
    let ncoils = acq.n_coils.max(1);
    let accel = acq.accel.max(1);
    assert_eq!(coil_kspace.len(), ncoils, "one k-space per coil");
    let acquired = cap.acquired.then(|| coil_kspace.iter().map(|k| pairs(k)).collect());

    // GRAPPA: fill the un-acquired PE lines with a kernel calibrated on the ACS band across coils.
    if accel > 1 {
        grappa_reconstruct(&mut coil_kspace, nx, ny, ys, accel, acq.acs_lines);
    }

    // Reconstruction window: after GRAPPA, before the inverse transform, so it acts on
    // originally-acquired and GRAPPA-synthesized lines alike -- and on the noise those lines
    // already carry: K_filtered = W(k) * [K_signal(k) + n(k)].
    //
    // The position matters. Replacing the old zero_ringing block in situ would have kept the
    // order signal -> band limitation -> noise, recreating the very defect this work removes:
    // filtered signal combined with unfiltered noise (section 1, finding 2.5).
    if acq.window != KspaceWindow::None {
        for ks in coil_kspace.iter_mut() {
            for kyi in 0..ny {
                for kxi in 0..nx {
                    let kx = (kxi as f64 - xs as f64) / nx as f64;
                    let ky = (kyi as f64 - ys as f64) / ny as f64;
                    let w = acq.window.at(kx, ky);
                    let k = &mut ks[kxi + nx * kyi];
                    k.re *= w;
                    k.im *= w;
                }
            }
        }
    }
    let reconstructed = cap.reconstructed.then(|| coil_kspace.iter().map(|k| pairs(k)).collect());

    // inverse each coil, then phase-preserving Roemer combine with the known sensitivities:
    //   combined = Σ_c image_c · sens_c / Σ_c sens_c²  (real sensitivities here)
    let sensitivities: Vec<Vec<f64>> = (0..ncoils)
        .map(|coil| (0..nx * ny).map(|i| coil_sensitivity(coil, ncoils, (i % nx) as f64, (i / nx) as f64, nx, ny)).collect())
        .collect();
    let mut wsum = vec![C::ZERO; nx * ny];
    let mut ssum = vec![0.0f64; nx * ny];
    let mut coil_images: Option<Vec<Vec<[f64; 2]>>> = cap.coil_images.then(Vec::new);
    for (coil, ks) in coil_kspace.iter().enumerate() {
        #[cfg(feature = "kspace")]
        let img = inverse_2d_fft(ks, nx, ny, xs, ys);
        #[cfg(not(feature = "kspace"))]
        let img = inverse_2d(ks, nx, ny, xs, ys);
        for y in 0..ny {
            for x in 0..nx {
                let i = kat(x, y);
                let s = sensitivities[coil][i];
                wsum[i] = wsum[i].add(img[i].scale(s));
                ssum[i] += s * s;
            }
        }
        if let Some(ci) = coil_images.as_mut() {
            ci.push(pairs(&img));
        }
    }
    let combined = (0..nx * ny)
        .map(|i| {
            let s = ssum[i].max(1e-12);
            [wsum[i].re / s, wsum[i].im / s]
        })
        .collect();
    SliceRecon {
        nx, ny, n_coils: ncoils, mask: sampling_mask(nx, ny, acq), acquired, reconstructed, coil_images, sensitivities,
        combined,
    }
}

/// Exact FFT reconstruction (feature `kspace`, backed by the well-tested `rustfft` crate).
/// Bit-for-bit equivalent to [`inverse_2d`] up to floating-point round-off: the centred convention
/// (`xs = nx/2`, `ys = ny/2`) is a fftshift, realised as linear-phase modulations of the input and
/// output plus a constant phase, so only the O(N log N) transform is delegated to rustfft.
#[cfg(feature = "kspace")]
fn inverse_2d_fft(kspace: &[C], nx: usize, ny: usize, xs: usize, ys: usize) -> Vec<C> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let mut planner = FftPlanner::<f64>::new();
    let fx = planner.plan_fft_forward(nx);
    let fy = planner.plan_fft_forward(ny);
    let (nxf, nyf, xsf, ysf) = (nx as f64, ny as f64, xs as f64, ys as f64);
    // 1) pre-modulate: K'[kx,ky] = K · cis(2π(kx·xs/nx + ky·ys/ny))
    let mut buf: Vec<Complex<f64>> = (0..nx * ny)
        .map(|i| {
            let (kx, ky) = ((i % nx) as f64, (i / nx) as f64);
            let ph = TAU * (kx * xsf / nxf + ky * ysf / nyf);
            let (c, sn) = (ph.cos(), ph.sin());
            Complex::new(kspace[i].re * c - kspace[i].im * sn, kspace[i].re * sn + kspace[i].im * c)
        })
        .collect();
    // 2) forward FFT along x (rows are contiguous), then along y (strided columns)
    for ky in 0..ny {
        fx.process(&mut buf[ky * nx..(ky + 1) * nx]);
    }
    let mut col = vec![Complex::new(0.0, 0.0); ny];
    for kx in 0..nx {
        for y in 0..ny {
            col[y] = buf[y * nx + kx];
        }
        fy.process(&mut col);
        for y in 0..ny {
            buf[y * nx + kx] = col[y];
        }
    }
    // 3) post-modulate: out = F · cis(2π(x·xs/nx + y·ys/ny)) · cis(-2π(xs²/nx + ys²/ny))
    let const_ph = -TAU * (xsf * xsf / nxf + ysf * ysf / nyf);
    (0..nx * ny)
        .map(|i| {
            let (x, y) = ((i % nx) as f64, (i / nx) as f64);
            let ph = TAU * (x * xsf / nxf + y * ysf / nyf) + const_ph;
            let (c, sn) = (ph.cos(), ph.sin());
            C { re: buf[i].re * c - buf[i].im * sn, im: buf[i].re * sn + buf[i].im * c }
        })
        .collect()
}

/// Inverse 2D DFT (separable direct sums): complex k-space → complex image, centred convention.
#[cfg(any(not(feature = "kspace"), test))]
fn inverse_2d(kspace: &[C], nx: usize, ny: usize, xs: usize, ys: usize) -> Vec<C> {
    let at = |x: usize, y: usize| x + nx * y;
    let mut h = vec![C::ZERO; nx * ny];
    for kxi in 0..nx {
        for y in 0..ny {
            let mut acc = C::ZERO;
            for kyi in 0..ny {
                let ky_norm = (kyi as f64 - ys as f64) / ny as f64;
                acc = acc.add(kspace[at(kxi, kyi)].mul(C::cis(-TAU * ky_norm * (y as f64 - ys as f64))));
            }
            h[at(kxi, y)] = acc;
        }
    }
    let mut out = vec![C::ZERO; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut acc = C::ZERO;
            for kxi in 0..nx {
                let kx_norm = (kxi as f64 - xs as f64) / nx as f64;
                acc = acc.add(h[at(kxi, y)].mul(C::cis(-TAU * kx_norm * (x as f64 - xs as f64))));
            }
            out[at(x, y)] = acc;
        }
    }
    out
}

/// Solve the Hermitian normal-equations system `H x = b` (complex, N×N) by Gaussian elimination
/// with partial pivoting. `H = AᴴA` is positive-definite after Tikhonov regularization.
fn solve_hermitian(mut h: Vec<Vec<C>>, mut b: Vec<C>) -> Vec<C> {
    let n = b.len();
    let recip = |d: C| {
        let m2 = d.re * d.re + d.im * d.im;
        if m2 < 1e-18 { C::ZERO } else { C { re: d.re / m2, im: -d.im / m2 } }
    };
    for col in 0..n {
        let (mut piv, mut best) = (col, h[col][col].abs());
        for r in (col + 1)..n {
            let m = h[r][col].abs();
            if m > best { best = m; piv = r; }
        }
        h.swap(col, piv);
        b.swap(col, piv);
        let dinv = recip(h[col][col]);
        for r in (col + 1)..n {
            let f = h[r][col].mul(dinv);
            for c in col..n {
                let s = f.mul(h[col][c]);
                h[r][c] = h[r][c].add(s.scale(-1.0));
            }
            let sb = f.mul(b[col]);
            b[r] = b[r].add(sb.scale(-1.0));
        }
    }
    let mut x = vec![C::ZERO; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s = s.add(h[col][c].mul(x[c]).scale(-1.0));
        }
        x[col] = s.mul(recip(h[col][col]));
    }
    x
}

/// GRAPPA: fill un-acquired PE lines in-place. Calibrates a kernel (2 PE source lines spaced `R`,
/// × 3 readout points, × all coils) on the fully-sampled central ACS band, then applies it to
/// synthesize each missing line per coil. The g-factor noise amplification emerges automatically
/// from reconstructing undersampled noisy multi-coil data.
fn grappa_reconstruct(coil: &mut [Vec<C>], nx: usize, ny: usize, ys: usize, accel: usize, acs_lines: usize) {
    let nc = coil.len();
    if nc == 0 || accel <= 1 {
        return;
    }
    let at = |x: usize, y: usize| x + nx * y;
    let dkx: [i64; 3] = [-1, 0, 1];
    let nfeat = 2 * dkx.len() * nc;
    let sidx = |line: usize, ki: usize, ci: usize| (line * dkx.len() + ki) * nc + ci;
    let acs_half = (acs_lines / 2) as i64;
    let (acs_lo, acs_hi) = ((ys as i64 - acs_half).max(0) as usize, ((ys as i64 + acs_half) as usize).min(ny - 1));

    let feature = |coil: &[Vec<C>], s0: usize, s1: usize, kx: usize| -> Vec<C> {
        let mut f = vec![C::ZERO; nfeat];
        for (li, &sl) in [s0, s1].iter().enumerate() {
            for (ki, &dk) in dkx.iter().enumerate() {
                let kxs = (kx as i64 + dk).rem_euclid(nx as i64) as usize;
                for ci in 0..nc {
                    f[sidx(li, ki, ci)] = coil[ci][at(kxs, sl)];
                }
            }
        }
        f
    };

    for d in 1..accel {
        // calibrate: accumulate AᴴA (shared) and Aᴴb (per output coil) over ACS positions
        let mut aha = vec![vec![C::ZERO; nfeat]; nfeat];
        let mut ahb = vec![vec![C::ZERO; nfeat]; nc];
        for b in acs_lo..acs_hi {
            let (s0, s1, tgt_ky) = (b, b + accel, b + d);
            if s1 > acs_hi || tgt_ky > acs_hi {
                continue;
            }
            for kx in 0..nx {
                let f = feature(coil, s0, s1, kx);
                for i in 0..nfeat {
                    let fic = C { re: f[i].re, im: -f[i].im }; // conj
                    for j in 0..nfeat {
                        aha[i][j] = aha[i][j].add(fic.mul(f[j]));
                    }
                    for co in 0..nc {
                        ahb[co][i] = ahb[co][i].add(fic.mul(coil[co][at(kx, tgt_ky)]));
                    }
                }
            }
        }
        for (i, row) in aha.iter_mut().enumerate() {
            row[i] = row[i].add(C { re: 1e-4, im: 0.0 }); // Tikhonov
        }
        let weights: Vec<Vec<C>> = (0..nc).map(|co| solve_hermitian(aha.clone(), ahb[co].clone())).collect();

        // synthesize every missing line at this offset (outside the ACS)
        for ky in 0..ny {
            if (ky as i64 - ys as i64).rem_euclid(accel as i64) != d as i64 {
                continue;
            }
            if (ky as i64 - ys as i64).abs() <= acs_half || ky < d || ky + accel - d >= ny {
                continue;
            }
            let (s0, s1) = (ky - d, ky - d + accel);
            for kx in 0..nx {
                let f = feature(coil, s0, s1, kx);
                for co in 0..nc {
                    let mut v = C::ZERO;
                    for i in 0..nfeat {
                        v = v.add(weights[co][i].mul(f[i]));
                    }
                    coil[co][at(kx, ky)] = v;
                }
            }
        }
    }
}

/// Run the k-space acquisition over a whole 4D clean-signal volume: every (volume, slice) through
/// [`simulate_slice`]. `clean` is the clean signal in `(x+nx*(y+ny*z))*n_volumes + g` layout; `fmap`
/// is the off-resonance field (Hz) on the same grid. v1 treats the mixed signal as one compartment
/// with an effective T2 (`t2_eff`, ms) — per-tissue T2 for realistic b0 contrast is a refinement.
/// Returns `(magnitude, phase)` 4D arrays (phase in radians, atan2), same layout — the complex pair
/// needed for BIDS `part-mag`/`part-phase` and complex denoisers. Parallel over volumes with `par`.
/// The production acquisition: a finer object, the nominal k-space band, reconstruction at the
/// acquisition matrix (spec 3.1), with an object phase model applied before encoding (spec 3.2).
///
/// `images` and `fmap` are on the **simulation** grid `sim_dims = [nx*o, ny*o, nz]`; the output is
/// on `acq_dims = [nx, ny, nz]`. `o` is derived and must divide both in-plane axes.
///
/// This is the path that makes ringing intrinsic. [`simulate_acquisition_legacy`] does not.
///
/// **Simulate a whole series in ONE call.** Every random stream here is keyed on the volume
/// index `g` *within this call*: `slice_seed` (k-space noise and spikes), the image-space noise
/// stream, and the per-shot phase realization. Calling this once per volume gives every volume
/// the same `g = 0` and the same noise, which for ASL would make control minus label cancel the
/// noise exactly and produce impossibly clean perfusion maps. A caller that must make more than
/// one call — an ASL series plus a separate M0 scan, say — passes a different `seed` to each,
/// because `seed` is the only thing distinguishing them.
#[allow(clippy::too_many_arguments)]
pub fn simulate_acquisition_oversampled(
    sim_dims: [usize; 3],
    acq_dims: [usize; 3],
    n_volumes: usize,
    images: &[Vec<f32>],
    // Per-compartment T2 (ms), scalar or a map on the simulation grid.
    t2: &[T2Volume],
    fmap: &[f32],
    // Per-compartment T2' (ms), scalar or map; `None` uses `Acquisition::t_inhom` throughout.
    t_inhom: Option<&[T2Volume]>,
    acq: &Acquisition,
    // Per-volume eddy drive (`SliceInput::eddy_drive`); `None` disables eddy for that volume.
    eddy_drive: &[Option<[f64; 3]>],
    // Per-volume prep-gradient drive for `phase.prep` (`SliceInput::prep_drive`), individually
    // optional so a series can carry a prep gradient on some volumes and not others.
    prep_drive: &[Option<(f64, [f64; 3])>],
    phase: &PhaseModel,
    seed: u64,
    // Optional per-voxel per-component noise SD on the ACQUIRED grid (`x + nx*(y + ny*z)`). When
    // given, complex Gaussian noise of this SD is added to the reconstructed complex image, before
    // the magnitude/phase split — so magnitude and phase share the SAME noise realization, and the
    // written SD map is the exact ground truth for a denoiser's estimated noise level. This is the
    // image-space, spatially-varying counterpart to `Acquisition::noise_variance` (uniform, k-space).
    noise_sigma: Option<&[f32]>,
    // Optional per-volume linear eddy shear `[a_x,a_y,a_z]` (one per gradient) to REPLAY a real
    // DIFFPREP eddy estimate as a geometric PE distortion (see `SliceInput::eddy_lin`).
    eddy_trace: Option<&[[f64; 3]]>,
) -> (Vec<f32>, Vec<f32>) {
    acquire_volumes(sim_dims, acq_dims, n_volumes, images, t2, fmap, t_inhom, acq, eddy_drive, prep_drive, phase,
                    seed, seed, noise_sigma, eddy_trace)
}

/// The 2D acquisition's inputs, by name (TRXScan re-sync): what [`simulate_acquisition_oversampled`]
/// takes positionally. `images` and `fmap` are on the simulation grid `sim_dims = [nx*o, ny*o, nz]`
/// (images `(x + snx*(y + sny*z))*n_volumes + g`); the output is on `acq_dims = [nx, ny, nz]`.
#[derive(Clone, Copy)]
pub struct AcquisitionInput<'a> {
    pub sim_dims: [usize; 3],
    pub acq_dims: [usize; 3],
    pub n_volumes: usize,
    pub images: &'a [Vec<f32>],
    /// Per-compartment T2 (ms), scalar or a map on the simulation grid.
    pub t2: &'a [T2Volume<'a>],
    pub fmap: &'a [f32],
    /// Per-compartment T2' (ms); `None` uses `Acquisition::t_inhom` throughout.
    pub t_inhom: Option<&'a [T2Volume<'a>]>,
    /// Per-volume eddy drive (`SliceInput::eddy_drive`).
    pub eddy_drive: &'a [Option<[f64; 3]>],
    /// Per-volume prep-gradient drive for `phase.prep` (`SliceInput::prep_drive`).
    pub prep_drive: &'a [Option<(f64, [f64; 3])>],
    pub phase: &'a PhaseModel,
    /// Mixed into every per-slice seed; see [`simulate_acquisition_oversampled`].
    pub seed: u64,
    /// Per-voxel image-space noise SD on the acquired grid; see [`simulate_acquisition_oversampled`].
    pub noise_sigma: Option<&'a [f32]>,
    /// Per-volume linear eddy shear to replay; see [`simulate_acquisition_oversampled`].
    pub eddy_trace: Option<&'a [[f64; 3]]>,
}

/// Options for [`simulate_acquisition_complex`] beyond the object and protocol (ported from
/// TRXScan `main`). `Default` is exactly what [`simulate_acquisition_oversampled`] does.
#[derive(Clone, Copy, Default)]
pub struct AcquisitionOptions<'a> {
    /// Full-FOV z index of each local slice (length `acq_dims[2]`), so a slab or a single slice
    /// sees the same eddy z-terms, object phase, per-slice seeds and image-noise stream as the
    /// full-volume run. The data (images, fieldmap, T2 maps) stay local. `None` = `0..nz`.
    pub slice_z: Option<&'a [usize]>,
    /// Full-FOV slice count when `slice_z` is given. `None` = `acq_dims[2]`.
    pub nz_full: Option<usize>,
    /// LOCAL slice indices whose k-space is retained (in this order). `None` = none.
    pub kspace_slices: Option<&'a [usize]>,
    /// Which per-coil intermediates to keep for those slices.
    pub capture: SliceCapture,
    /// Per-volume echo time (ms), length `n_volumes`; `None` = `acq.t_echo` for every volume. Each
    /// volume's timing is checked as [`validate_acquisition_timing`] checks one.
    pub t_echo_per_volume: Option<&'a [f64]>,
    /// Called after each volume completes with `(done, total)`. Volumes run in parallel under
    /// `par`, so it must be `Sync`; completion order is not volume order.
    pub progress: Option<&'a (dyn Fn(usize, usize) + Sync)>,
}

/// K-space and per-coil intermediates for the captured slices. Complex arrays are `[re, im]`
/// pairs in C order over the axes named; `nsl = slices.len()`.
#[derive(Debug, Clone)]
pub struct KspaceCapture {
    /// The local slice indices captured, in array order.
    pub slices: Vec<usize>,
    pub nx: usize,
    pub ny: usize,
    pub n_coils: usize,
    /// `[ky, kx]`, shared by every slice and volume (= [`sampling_mask`]).
    pub mask: Vec<bool>,
    /// `[coil, y, x]` real coil sensitivities on the acquired grid.
    pub sensitivities: Vec<f32>,
    /// `[vol, slice, coil, ky, kx]`; see [`SliceCapture::acquired`].
    pub acquired: Option<Vec<[f32; 2]>>,
    /// `[vol, slice, coil, ky, kx]`; see [`SliceCapture::reconstructed`].
    pub reconstructed: Option<Vec<[f32; 2]>>,
    /// `[vol, slice, coil, y, x]`; see [`SliceCapture::coil_images`].
    pub coil_images: Option<Vec<[f32; 2]>>,
    /// `[vol, slice, y, x]`: the Roemer-combined complex image before any image-space noise.
    pub combined: Vec<[f32; 2]>,
}

/// The complex acquisition output: real and imaginary parts in the 4-D layout
/// `(x + nx*(y + ny*z))*n_volumes + g`, after image-space noise, plus the optional k-space capture.
#[derive(Debug, Clone)]
pub struct AcquisitionOutput {
    pub re: Vec<f32>,
    pub im: Vec<f32>,
    pub kspace: Option<KspaceCapture>,
}

/// [`simulate_acquisition_oversampled`] returning the complex image and, on request, the k-space
/// of selected slices, per-volume echo times, slab-aware slice indexing and a progress callback
/// (ported from TRXScan `main`). With default options its magnitude and phase are
/// `simulate_acquisition_oversampled`'s, bit for bit.
pub fn simulate_acquisition_complex(inp: &AcquisitionInput, acq: &Acquisition, opts: &AcquisitionOptions)
    -> AcquisitionOutput
{
    acquire_volumes_complex(inp, acq, opts, inp.seed, inp.seed)
}

/// The body of [`simulate_acquisition_oversampled`] with its one seed split in two: the
/// **excitation** seed draws the per-shot phase realization (`PrepPhase::shot`), the **receiver**
/// seed the k-space noise and spikes (`slice_seed`) and the image-space noise. The public entries
/// pass the same seed to both, except [`simulate_acquisition_echoes`], whose echoes share one
/// excitation and draw independent receiver noise. Magnitude and phase of
/// [`acquire_volumes_complex`], with the same `f32` arithmetic as before it existed.
#[allow(clippy::too_many_arguments)]
fn acquire_volumes(
    sim_dims: [usize; 3],
    acq_dims: [usize; 3],
    n_volumes: usize,
    images: &[Vec<f32>],
    t2: &[T2Volume],
    fmap: &[f32],
    t_inhom: Option<&[T2Volume]>,
    acq: &Acquisition,
    eddy_drive: &[Option<[f64; 3]>],
    prep_drive: &[Option<(f64, [f64; 3])>],
    phase: &PhaseModel,
    excitation_seed: u64,
    receiver_seed: u64,
    noise_sigma: Option<&[f32]>,
    eddy_trace: Option<&[[f64; 3]]>,
) -> (Vec<f32>, Vec<f32>) {
    let inp = AcquisitionInput {
        sim_dims, acq_dims, n_volumes, images, t2, fmap, t_inhom, eddy_drive, prep_drive, phase, seed: receiver_seed,
        noise_sigma, eddy_trace,
    };
    let out = acquire_volumes_complex(&inp, acq, &AcquisitionOptions::default(), excitation_seed, receiver_seed);
    let n = out.re.len();
    let (mut mag, mut ph) = (vec![0.0f32; n], vec![0.0f32; n]);
    for i in 0..n {
        let (re, im) = (out.re[i], out.im[i]);
        mag[i] = (re * re + im * im).sqrt();
        ph[i] = im.atan2(re);
    }
    (mag, ph)
}

/// The acquisition's one body: every (volume, slice) through [`simulate_slice_full`], complex out.
/// `inp.seed` is not read; the two seeds are explicit (see [`acquire_volumes`]).
fn acquire_volumes_complex(
    inp: &AcquisitionInput,
    acq: &Acquisition,
    opts: &AcquisitionOptions,
    excitation_seed: u64,
    receiver_seed: u64,
) -> AcquisitionOutput {
    let AcquisitionInput {
        sim_dims, acq_dims, n_volumes, images, t2, fmap, t_inhom, eddy_drive, prep_drive, phase, noise_sigma, eddy_trace, ..
    } = *inp;
    let [snx, sny, nz] = sim_dims;
    let [nx, ny, nzo] = acq_dims;
    assert_eq!(nz, nzo, "slice count must match; z is never oversampled");
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple of the acquired matrix");
    let o = snx / nx;
    assert_eq!(o, sny / ny, "oversampling must match on both axes");
    let (nvox_sim, nvox_acq) = (snx * sny * nz, nx * ny * nz);
    let ncomp = images.len();
    for im in images {
        assert_eq!(im.len(), nvox_sim * n_volumes, "compartment image is not on the simulation grid");
    }
    assert_eq!(fmap.len(), nvox_sim, "fieldmap is not on the simulation grid");
    assert_eq!(t2.len(), ncomp, "t2 has {} entries for {ncomp} compartments", t2.len());
    validate_t2_volumes("T2", t2, nvox_sim);
    if let Some(ti) = t_inhom {
        assert_eq!(ti.len(), ncomp, "t_inhom has {} entries for {ncomp} compartments", ti.len());
        validate_t2_volumes("T2'", ti, nvox_sim);
    }
    match opts.t_echo_per_volume {
        None => validate_acquisition_timing(acq, nx, ny).unwrap_or_else(|e| panic!("{e}")),
        Some(t) => {
            assert_eq!(t.len(), n_volumes, "t_echo_per_volume must have one entry per volume");
            for (g, &te) in t.iter().enumerate() {
                validate_acquisition_timing(&Acquisition { t_echo: te, ..acq.clone() }, nx, ny)
                    .unwrap_or_else(|e| panic!("volume {g}: {e}"));
            }
        }
    }
    assert_eq!(eddy_drive.len(), n_volumes,
               "eddy_drive has {} entries for {} volumes", eddy_drive.len(), n_volumes);
    assert_eq!(prep_drive.len(), n_volumes,
               "prep_drive has {} entries for {} volumes", prep_drive.len(), n_volumes);
    if let Some(tr) = eddy_trace {
        assert_eq!(tr.len(), n_volumes,
                   "eddy_trace has {} entries for {} volumes", tr.len(), n_volumes);
    }
    if let Some(sz) = opts.slice_z {
        assert_eq!(sz.len(), nz, "slice_z must name every local slice");
    }
    let nz_full = opts.nz_full.unwrap_or(nz);
    if let Some(sz) = opts.slice_z {
        assert!(sz.iter().all(|&z| z < nz_full), "slice_z entries must be below nz_full {nz_full}");
    }
    let global_z = |zl: usize| opts.slice_z.map_or(zl, |s| s[zl]);
    let captured: Vec<usize> = opts.kspace_slices.map(|s| s.to_vec()).unwrap_or_default();
    for &zl in &captured {
        assert!(zl < nz, "kspace_slices index {zl} out of range for {nz} slices");
    }
    let done = std::sync::atomic::AtomicUsize::new(0);

    let per_vol = |g: usize| -> (Vec<f32>, Vec<f32>, Vec<SliceRecon>) {
        let acq_owned;
        let acq_g: &Acquisition = match opts.t_echo_per_volume {
            Some(t) => {
                acq_owned = Acquisition { t_echo: t[g], ..acq.clone() };
                &acq_owned
            }
            None => acq,
        };
        let (mut re_out, mut im_out) = (vec![0.0f32; nvox_acq], vec![0.0f32; nvox_acq]);
        let mut recons = Vec::with_capacity(captured.len());
        let mut cslices = vec![vec![0.0f32; snx * sny]; ncomp];
        let mut fslice = vec![0.0f32; snx * sny];
        for zl in 0..nz {
            // the data are local; everything keyed on position (seeds, shot phase, eddy z-terms,
            // the object phase) is the full-FOV slice's
            let z = global_z(zl);
            for y in 0..sny {
                for x in 0..snx {
                    let vox = x + snx * (y + sny * zl);
                    for (c, img) in images.iter().enumerate() {
                        cslices[c][x + snx * y] = img[vox * n_volumes + g];
                    }
                    fslice[x + snx * y] = fmap[vox];
                }
            }
            let refs: Vec<&[f32]> = cslices.iter().map(|v| v.as_slice()).collect();
            let t2_slice = t2_slices(t2, zl, snx * sny);
            let ti_slice = t_inhom.map(|ti| t2_slices(ti, zl, snx * sny));
            let shot = match (&phase.prep, prep_drive[g]) {
                (Some(p), Some((mag, dir))) => p.shot(mag, dir, g, z, excitation_seed),
                _ => ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] },
            };
            let phi = phase_slice(phase, &shot, snx, sny, o, z, nz_full);
            let slice_seed = (g as u64)
                .wrapping_mul(0x100_0001)
                .wrapping_add(z as u64)
                .wrapping_mul(0x9E37)
                ^ receiver_seed;
            let want = captured.contains(&zl);
            let cap = if want { opts.capture } else { SliceCapture::default() };
            let out = simulate_slice_full(
                &SliceInput {
                    compartments: &refs,
                    t2: &t2_slice,
                    t_inhom: ti_slice.as_deref(),
                    fmap: &fslice,
                    phase0: Some(&phi),
                    sim: [snx, sny],
                    acq_matrix: [nx, ny],
                    z,
                    nz: nz_full,
                    eddy_drive: eddy_drive[g],
                    prep_drive: prep_drive[g],
                    slice_seed,
                    eddy_lin: eddy_trace.map(|tr| tr[g]),
                },
                acq_g,
                cap,
            );
            for y in 0..ny {
                for x in 0..nx {
                    let c = out.combined[x + nx * y];
                    let (mut re, mut im) = (c[0] as f32, c[1] as f32);
                    let vox = x + nx * (y + ny * zl);
                    if let Some(ns) = noise_sigma {
                        let sd = ns[vox] as f64;
                        if sd > 0.0 {
                            // deterministic per (volume, FULL-FOV voxel); parallel-safe (per_vol is over g)
                            let vox_g = x + nx * (y + ny * z);
                            let mut rng = Rng(
                                receiver_seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                    ^ (vox_g as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1,
                            );
                            re += (rng.gauss() * sd) as f32;
                            im += (rng.gauss() * sd) as f32;
                        }
                    }
                    re_out[vox] = re;
                    im_out[vox] = im;
                }
            }
            if want {
                recons.push(out);
            }
        }
        if let Some(p) = opts.progress {
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            p(n, n_volumes);
        }
        (re_out, im_out, recons)
    };

    #[cfg(feature = "par")]
    let vols: Vec<(Vec<f32>, Vec<f32>, Vec<SliceRecon>)> = {
        use rayon::prelude::*;
        (0..n_volumes).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<(Vec<f32>, Vec<f32>, Vec<SliceRecon>)> = (0..n_volumes).map(per_vol).collect();

    let (mut red, mut imd) = (vec![0.0f32; nvox_acq * n_volumes], vec![0.0f32; nvox_acq * n_volumes]);
    for (g, (r, i, _)) in vols.iter().enumerate() {
        for vox in 0..nvox_acq {
            red[vox * n_volumes + g] = r[vox];
            imd[vox * n_volumes + g] = i[vox];
        }
    }

    let kspace = if captured.is_empty() {
        None
    } else {
        // per_vol keeps the captured slices in local-slice order; emit them in the requested order
        let nsl = captured.len();
        let ncoils = acq.n_coils.max(1);
        let per_slice = nx * ny;
        let f = |v: &[f64; 2]| [v[0] as f32, v[1] as f32];
        let mut acquired = opts.capture.acquired.then(|| Vec::with_capacity(n_volumes * nsl * ncoils * per_slice));
        let mut reconstructed = opts.capture.reconstructed.then(|| Vec::with_capacity(n_volumes * nsl * ncoils * per_slice));
        let mut coil_images = opts.capture.coil_images.then(|| Vec::with_capacity(n_volumes * nsl * ncoils * per_slice));
        let mut combined = Vec::with_capacity(n_volumes * nsl * per_slice);
        let mut sorted_local: Vec<usize> = captured.clone();
        sorted_local.sort_unstable();
        sorted_local.dedup();
        let pos_in_run = |zl: usize| sorted_local.iter().position(|&s| s == zl).expect("a captured slice");
        for (_, _, recons) in &vols {
            for &zl in &captured {
                let r = &recons[pos_in_run(zl)];
                if let Some(a) = acquired.as_mut() {
                    for coil in r.acquired.as_ref().expect("acquired captured") {
                        a.extend(coil.iter().map(f));
                    }
                }
                if let Some(a) = reconstructed.as_mut() {
                    for coil in r.reconstructed.as_ref().expect("reconstructed captured") {
                        a.extend(coil.iter().map(f));
                    }
                }
                if let Some(a) = coil_images.as_mut() {
                    for coil in r.coil_images.as_ref().expect("coil images captured") {
                        a.extend(coil.iter().map(f));
                    }
                }
                combined.extend(r.combined.iter().map(f));
            }
        }
        let first = &vols[0].2[0];
        let sensitivities = first.sensitivities.iter().flat_map(|c| c.iter().map(|&v| v as f32)).collect();
        Some(KspaceCapture {
            slices: captured, nx, ny, n_coils: ncoils, mask: first.mask.clone(), sensitivities, acquired, reconstructed,
            coil_images, combined,
        })
    };
    AcquisitionOutput { re: red, im: imd, kspace }
}

/// The receiver-seed salt of echo `e` in [`simulate_acquisition_echoes`]: zero for the first echo,
/// so a one-echo call is [`simulate_acquisition_oversampled`] bit for bit. Public so a caller can
/// record the receiver seeds it produced.
pub fn echo_salt(e: usize) -> u64 {
    (e as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
}

/// A multi-echo series in one call: each excitation is read at every `echo_times_ms[e]` (ms) by its
/// own readout, and echo `e` is [`simulate_acquisition_oversampled`] with
/// `Acquisition { t_echo: echo_times_ms[e], ..acq.clone() }` on `images_per_echo[e]`.
///
/// The echoes of a volume share one excitation: the per-shot phase realization (`phase.prep`) is
/// drawn from `seed` itself for every echo. Each echo's receiver noise (k-space noise and spikes,
/// and the image-space `noise_sigma` noise) is drawn from `seed ^ echo_salt(e)`, with
/// `echo_salt(0) = 0`, so the echoes' noise is independent and a call with one echo at
/// `acq.t_echo` is bit-identical to `simulate_acquisition_oversampled`. The echo train's own k-space
/// trajectory is not modeled: the echoes are independent 2D readouts at their own echo times.
///
/// `images_per_echo[e]` holds echo `e`'s compartment images (callers that apply no echo-time decay
/// themselves pass the same images for every echo). Returns `(magnitude, phase)` per echo, each in
/// the layout of `simulate_acquisition_oversampled`.
#[allow(clippy::too_many_arguments)]
pub fn simulate_acquisition_echoes(
    sim_dims: [usize; 3],
    acq_dims: [usize; 3],
    n_volumes: usize,
    images_per_echo: &[&[Vec<f32>]],
    t2: &[T2Volume],
    fmap: &[f32],
    t_inhom: Option<&[T2Volume]>,
    acq: &Acquisition,
    echo_times_ms: &[f64],
    eddy_drive: &[Option<[f64; 3]>],
    prep_drive: &[Option<(f64, [f64; 3])>],
    phase: &PhaseModel,
    seed: u64,
    noise_sigma: Option<&[f32]>,
    eddy_trace: Option<&[[f64; 3]]>,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    for (e, w) in echo_times_ms.windows(2).enumerate() {
        assert!(w[0] < w[1], "echo times must increase strictly: echo {e} at {} ms, echo {} at {} ms", w[0], e + 1, w[1]);
    }
    echoes_with_salt(sim_dims, acq_dims, n_volumes, images_per_echo, t2, fmap, t_inhom, acq, echo_times_ms,
                     eddy_drive, prep_drive, phase, seed, noise_sigma, eddy_trace, echo_salt, |_| 0)
}

/// [`simulate_acquisition_echoes`] with the per-echo salts of the receiver and excitation seeds as
/// parameters, so the tests can show what each salt does (no public switch).
#[allow(clippy::too_many_arguments)]
fn echoes_with_salt(
    sim_dims: [usize; 3],
    acq_dims: [usize; 3],
    n_volumes: usize,
    images_per_echo: &[&[Vec<f32>]],
    t2: &[T2Volume],
    fmap: &[f32],
    t_inhom: Option<&[T2Volume]>,
    acq: &Acquisition,
    echo_times_ms: &[f64],
    eddy_drive: &[Option<[f64; 3]>],
    prep_drive: &[Option<(f64, [f64; 3])>],
    phase: &PhaseModel,
    seed: u64,
    noise_sigma: Option<&[f32]>,
    eddy_trace: Option<&[[f64; 3]]>,
    receiver_salt: fn(usize) -> u64,
    excitation_salt: fn(usize) -> u64,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    assert!(!echo_times_ms.is_empty(), "simulate_acquisition_echoes needs at least one echo");
    assert_eq!(images_per_echo.len(), echo_times_ms.len(),
               "images_per_echo has {} entries for {} echoes", images_per_echo.len(), echo_times_ms.len());
    echo_times_ms
        .iter()
        .zip(images_per_echo)
        .enumerate()
        .map(|(e, (&te, images))| {
            let acq_e = Acquisition { t_echo: te, ..acq.clone() };
            acquire_volumes(sim_dims, acq_dims, n_volumes, images, t2, fmap, t_inhom, &acq_e, eddy_drive, prep_drive,
                            phase, seed ^ excitation_salt(e), seed ^ receiver_salt(e), noise_sigma, eddy_trace)
        })
        .collect()
}

/// **Legacy path: no intrinsic Gibbs ringing and no object phase.**
///
/// Runs the acquisition with the object already on the reconstruction matrix (`o = 1`) and
/// `phase0: None`. At `o = 1` the forward and inverse transforms are an exact round trip, so this
/// produces **zero** ringing (measured: +0.0000% overshoot on a step edge) and a real-valued image.
///
/// Retained only for backward comparison. Production callers must use
/// [`simulate_acquisition_oversampled`], which is what gives ringing intrinsic to the acquisition
/// and a genuinely complex object.
#[allow(clippy::too_many_arguments)]
pub fn simulate_acquisition_legacy(
    dims: [usize; 3],
    n_volumes: usize,
    images: &[Vec<f32>],
    t2: &[T2Volume],
    fmap: &[f32],
    t_inhom: Option<&[T2Volume]>,
    acq: &Acquisition,
    gradients: &[[f64; 3]],
) -> (Vec<f32>, Vec<f32>) {
    let [nx, ny, nz] = dims;
    let nvox = nx * ny * nz;
    let ncomp = images.len();
    assert_eq!(t2.len(), ncomp, "t2 has {} entries for {ncomp} compartments", t2.len());
    validate_t2_volumes("T2", t2, nvox);
    if let Some(ti) = t_inhom {
        assert_eq!(ti.len(), ncomp, "t_inhom has {} entries for {ncomp} compartments", ti.len());
        validate_t2_volumes("T2'", ti, nvox);
    }
    validate_acquisition_timing(acq, nx, ny).unwrap_or_else(|e| panic!("{e}"));
    // returns (magnitude, phase) for one volume
    let per_vol = |g: usize| -> (Vec<f32>, Vec<f32>) {
        let (mut mag, mut phase) = (vec![0.0f32; nvox], vec![0.0f32; nvox]);
        let mut cslices = vec![vec![0.0f32; nx * ny]; ncomp];
        let mut fslice = vec![0.0f32; nx * ny];
        // split the scaled gradient into a unit direction and a magnitude; the forward model
        // recombines them, so this preserves the previous behaviour exactly
        let gr = gradients[g];
        let bval = (gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]).sqrt();
        let bvec =
            if bval > 1e-12 { [gr[0] / bval, gr[1] / bval, gr[2] / bval] } else { [0.0; 3] };
        // (g/|g|)*|g| is not bit-equal to g: keep forming the product the forward model formed.
        let eddy_drive = if bval.abs() > 1e-9 {
            Some([bvec[0] * bval, bvec[1] * bval, bvec[2] * bval])
        } else {
            None
        };
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let vox = x + nx * (y + ny * z);
                    for (c, img) in images.iter().enumerate() {
                        cslices[c][x + nx * y] = img[vox * n_volumes + g];
                    }
                    fslice[x + nx * y] = fmap[vox];
                }
            }
            let refs: Vec<&[f32]> = cslices.iter().map(|v| v.as_slice()).collect();
            let t2_slice = t2_slices(t2, z, nx * ny);
            let ti_slice = t_inhom.map(|ti| t2_slices(ti, z, nx * ny));
            let seed = (g as u64).wrapping_mul(0x100_0001).wrapping_add(z as u64).wrapping_mul(0x9E37)
                .wrapping_add(acq.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let inp = SliceInput {
                compartments: &refs,
                t2: &t2_slice,
                t_inhom: ti_slice.as_deref(),
                fmap: &fslice,
                phase0: None,
                // v1 acquisition path: the object is already on the acquired matrix (o = 1).
                sim: [nx, ny],
                acq_matrix: [nx, ny],
                z,
                nz,
                eddy_drive,
                prep_drive: None,
                slice_seed: seed,
                eddy_lin: None,
            };
            let out = simulate_slice(&inp, acq);
            for y in 0..ny {
                for x in 0..nx {
                    let (re, im) = out[x + nx * y];
                    let vox = x + nx * (y + ny * z);
                    mag[vox] = (re * re + im * im).sqrt();
                    phase[vox] = im.atan2(re);
                }
            }
        }
        (mag, phase)
    };

    #[cfg(feature = "par")]
    let vols: Vec<(Vec<f32>, Vec<f32>)> = {
        use rayon::prelude::*;
        (0..n_volumes).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<(Vec<f32>, Vec<f32>)> = (0..n_volumes).map(per_vol).collect();

    let (mut magd, mut phased) = (vec![0.0f32; nvox * n_volumes], vec![0.0f32; nvox * n_volumes]);
    for (g, (m, p)) in vols.iter().enumerate() {
        for vox in 0..nvox {
            magd[vox * n_volumes + g] = m[vox];
            phased[vox * n_volumes + g] = p[vox];
        }
    }
    (magd, phased)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal per-line sum the production forward was restructured from: every
    /// factor re-evaluated for every (voxel, line). Kept verbatim as the oracle for
    /// `restructured_forward_matches_the_literal_sum`; not a code path.
    #[allow(clippy::needless_range_loop)]
    fn reference_coil_kspace(inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize) -> Vec<C> {
        let [snx, sny] = inp.sim;
        let [nx, ny] = inp.acq_matrix;
        assert!(
            snx % nx == 0 && sny % ny == 0,
            "sim grid must be an integer multiple of the acquired matrix"
        );
        let (z, nz) = (inp.z, inp.nz);
        let (compartments, t2, fmap) = (inp.compartments, inp.t2, inp.fmap);
        let epi = SingleShotEpi {
            kx_max: nx,
            ky_max: ny,
            t_line: acq.t_line,
            t_echo: acq.t_echo,
            reverse_phase: acq.reverse_phase,
        };
        let (t_ms, trf_ms, tread_ms) = line_times(&epi);
        let gradient = inp.eddy_drive.unwrap_or([0.0; 3]);
        // eddy currents affect volumes with a prep gradient only (`None` = the old b0 branch)
        let do_eddy = eddy_enabled(acq, inp.eddy_drive);
        let do_eddy_phase = acq.eddy_phase != 0.0 && inp.eddy_drive.is_some();
        let do_eddy_trace = inp.eddy_lin.is_some();
        let trt_s = acq.t_line * ny as f64 / 1000.0; // total readout time (s), for the shift<->phase map
        // acquired-matrix centres (k-space indexing) and sim-grid centres (image indexing)
        // Centred k-space indexing: the acquired band is [-n/2, n/2-1], asymmetric about k=0 by one
        // sample. This is deliberate, not an off-by-one -- real even-matrix Cartesian acquisitions
        // cover exactly this range. It gives a real object a small deterministic imaginary component,
        // which is NOT object phase; see `phase.rs` for that. Pinned by
        // `even_matrix_window_asymmetry_is_intentional`.
        let (xs, ys, zs) = (nx / 2, ny / 2, nz / 2);
        let (ox, oy) = (snx / nx, sny / ny); // in-plane oversampling factors
        // The sim-grid centre is the IMAGE of the acquired centre, o*(n/2), not snx/2: for an odd
        // acquired matrix those differ by one sim cell, i.e. (1/o) of an acquired voxel in-plane.
        let (sxs, sys) = (ox * xs, oy * ys);
        // Half-cell alignment. Sim cell `x` covers [x, x+1) in sim units, so its centre is at x+0.5;
        // acquired cell `X` covers o cells and is centred at o*X + o/2. Aligning sample index `x` with
        // `o*X` — as a bare index substitution does — therefore misregisters the object against the
        // reconstruction grid by (o-1)/2 sim cells, i.e. (o-1)/(2o) of an ACQUIRED voxel: 0.44 voxels
        // at o=8. Since this whole model turns on sub-voxel edge position, that shift is fatal and the
        // acquired k-space is measured from the acquired grid's centre instead. Exactly zero at o=1.
        let (xoff, yoff) = ((ox as f64 - 1.0) / 2.0, (oy as f64 - 1.0) / 2.0);
        let at = |x: usize, y: usize| x + snx * y; // SIM-grid image index
        let kat = |kx: usize, ky: usize| kx + nx * ky; // acquired k-space / acquired-image index

        // Which samples are acquired (partial Fourier + GRAPPA undersampling). Single source of
        // truth, shared with the noise below so signal and noise cannot disagree.
        let mask = sampling_mask(nx, ny, acq);
        // ---- forward: build k-space, factored as  Σ_x e^{..kx x}[ Σ_y mod(x,y) e^{..ky y} ] ----
        // mod(x,y) depends on the PE line ky (through φ and relaxation), so the y-sum is recomputed
        // per ky, but that keeps the whole build at O(N³).
        let mut kspace = vec![C::ZERO; nx * ny];
        let n_inv = 1.0 / (snx * sny) as f64;
        for kyi in 0..ny {
            // Not acquired (partial Fourier, or GRAPPA undersampling): this k-space row stays
            // zero and, critically, receives no noise either.
            if !mask[nx * kyi] {
                continue;
            }
            let t = t_ms[kyi] / 1000.0; // seconds
            let trf = trf_ms[kyi];
            // Divide by the SIM extent: the loop still runs over the ny ACQUIRED lines, but each sits
            // at absolute sim index sys - ys + kyi, whose normalized frequency is (kyi - ys)/sny.
            let ky_norm = (kyi as f64 - ys as f64) / sny as f64;
            // Nyquist (N/2) ghost: alternating readout-line kx offset (gradient-delay mismatch).
            let ghost_shift = if kyi % 2 == 1 { -acq.ghost_offset } else { acq.ghost_offset };
            // eddy-current decay for this PE line: exp(-tRead/τ)·t (itkKspaceImageFilter.cpp:354)
            let eddy_decay = if do_eddy { (-tread_ms[kyi] / acq.eddy_tau).exp() * t } else { 0.0 };

            // modulated image for this PE line
            let mut modimg = vec![C::ZERO; snx * sny];
            for y in 0..sny {
                for x in 0..snx {
                    let mut f_real = 0.0f64;
                    for (c, comp) in compartments.iter().enumerate() {
                        let mut v = comp[at(x, y)] as f64;
                        if acq.do_relaxation {
                            let t2c = match t2[c] {
                                T2Slice::Uniform(u) => u as f64,
                                T2Slice::Map(m) => m[at(x, y)] as f64,
                            };
                            let tic = match inp.t_inhom.map(|ti| ti[c]) {
                                None => acq.t_inhom,
                                Some(T2Slice::Uniform(u)) => u as f64,
                                Some(T2Slice::Map(m)) => m[at(x, y)] as f64,
                            };
                            // T2' from the echo (spin echo) or from the RF (gradient echo)
                            let inhom_ms = match acq.echo {
                                EchoFormation::Spin => t.abs() * 1000.0,
                                EchoFormation::Gradient => trf,
                            };
                            v *= (-(trf as f64) / t2c - inhom_ms / tic).exp();
                        }
                        f_real += v;
                    }
                    // TWO coordinate frames, named apart on purpose. Both are in acquired-voxel
                    // UNITS and both carry the half-cell registration (`xoff = (o-1)/2`), but they
                    // have DIFFERENT ORIGINS, and a quantity evaluated in the wrong one is wrong by
                    // half a FOV rather than by a sub-voxel amount:
                    //
                    //   xa — ABSOLUTE on the acquired grid, 0 .. nx-1, averaging to exactly `v` over
                    //        the o sim cells of acquired voxel `v`. This is the frame the Roemer
                    //        combine walks (`x in 0..nx`), so anything the combine must agree with —
                    //        `coil_sensitivity`, whose coil ring is centred on nx/2 — uses it.
                    //   xc — CENTRED on the acquired image, -nx/2 .. nx/2-1. The eddy polynomial is
                    //        an expansion about the image centre and needs this one.
                    //
                    // Conflating them is not hypothetical: an earlier revision passed `xc` to
                    // `coil_sensitivity`, which put the forward model's sensitivity field half a FOV
                    // from the combine's at EVERY oversampling factor, o = 1 included (28% of peak).
                    // Pinned end-to-end by
                    // `multicoil_roemer_reproduces_the_single_coil_image_at_every_oversampling`.
                    let (xa, ya) = (
                        (x as f64 - xoff) / ox as f64,
                        (y as f64 - yoff) / oy as f64,
                    );
                    f_real *= acq.signal_scale * coil_sensitivity(coil, ncoils, xa, ya, nx, ny);
                    // B0 phase accrues over `t` from the echo (spin echo) or `TE + t` from the RF
                    // (gradient echo)
                    let t_b0 = match acq.echo {
                        EchoFormation::Spin => t,
                        EchoFormation::Gradient => t + acq.t_echo / 1000.0,
                    };
                    let mut phi = if acq.do_distortions { fmap[at(x, y)] as f64 * t_b0 } else { 0.0 };
                    // Pre-readout object phase, already in radians, so it is added outside the TAU
                    // factor that scales the distortion/eddy term.
                    let mut phi0 = inp.phase0.map_or(0.0, |p| p[at(x, y)]);
                    if do_eddy || do_eddy_phase || do_eddy_trace {
                        // centre on the sim grid, then express in ACQUIRED voxel units so that the
                        // eddy scales keep their meaning independent of oversampling
                        let (xc, yc, zc) = (
                            (x as f64 - sxs as f64 - xoff) / ox as f64,
                            (y as f64 - sys as f64 - yoff) / oy as f64,
                            z as f64 - zs as f64,
                        );
                        if do_eddy {
                            // gradient-dependent field growing through the readout: linear (g·pos)
                            // plus a quadratic (g·pos²) term — the polynomial eddy/TORTOISE fit. This
                            // grows with the readout time (eddy_decay ∝ ky) → geometric DISTORTION.
                            let lin = gradient[0] * xc + gradient[1] * yc + gradient[2] * zc;
                            let quad =
                                gradient[0] * xc * xc + gradient[1] * yc * yc + gradient[2] * zc * zc;
                            phi += (acq.eddy_strength * lin + acq.eddy_quad * quad) * eddy_decay;
                        }
                        if do_eddy_phase {
                            // Eddy OBJECT-phase ramp: constant across the readout (NOT ∝ ky), so it
                            // imprints the reconstructed object phase rather than distorting geometry.
                            // Direction- and b-dependent (∝ gradient = bvec·bval), it reproduces the
                            // per-volume phase-ramp variation real DWI shows (∝ gradient direction).
                            // z centred on the volume (zc above is slice-index-from-start, not centred).
                            let zc_c = z as f64 - (inp.nz as f64 - 1.0) / 2.0;
                            phi0 += acq.eddy_phase
                                * (gradient[0] * xc + gradient[1] * yc + gradient[2] * zc_c);
                        }
                        if let Some(a) = inp.eddy_lin {
                            // Replay a real per-volume linear eddy shear as a PE geometric shift: a
                            // phase (a·pos / TRT)·t behaves like a fieldmap of that value, giving
                            // shift = a·pos acquired voxels (a is dimensionless shift-per-position).
                            let shear = a[0] * xc + a[1] * yc + a[2] * zc;
                            phi += shear * (t / trt_s);
                        }
                    }
                    modimg[at(x, y)] = C::cis(TAU * phi + phi0).scale(f_real);
                }
            }
            // inner y-sum over the SIM grid → g(x), then an x-DFT evaluated only at acquired kx
            let mut g = vec![C::ZERO; snx];
            for x in 0..snx {
                let mut acc = C::ZERO;
                for y in 0..sny {
                    let ph = C::cis(TAU * ky_norm * (y as f64 - sys as f64 - yoff));
                    acc = acc.add(modimg[at(x, y)].mul(ph));
                }
                g[x] = acc;
            }
            for kxi in 0..nx {
                let kx_norm = (kxi as f64 - xs as f64 + ghost_shift) / snx as f64;
                let mut acc = C::ZERO;
                for x in 0..snx {
                    acc = acc.add(g[x].mul(C::cis(TAU * kx_norm * (x as f64 - sxs as f64 - xoff))));
                }
                kspace[kat(kxi, kyi)] = acc.scale(n_inv);
            }
        }
        kspace
    }

    fn mag(v: &[(f32, f32)]) -> Vec<f32> {
        v.iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect()
    }

    /// The pre-existing single-compartment call shape, on a `SliceInput` whose simulation grid
    /// equals the acquired matrix (`o = 1`, `phase0: None`) — i.e. the old behaviour exactly.
    #[allow(clippy::too_many_arguments)]
    fn slice1(
        img: &[f32],
        t2: &[T2Slice],
        fmap: &[f32],
        nx: usize,
        ny: usize,
        z: usize,
        nz: usize,
        acq: &Acquisition,
        gradient: [f64; 3],
        slice_seed: u64,
    ) -> Vec<(f32, f32)> {
        let bval = (gradient[0].powi(2) + gradient[1].powi(2) + gradient[2].powi(2)).sqrt();
        let bvec = if bval > 1e-12 {
            [gradient[0] / bval, gradient[1] / bval, gradient[2] / bval]
        } else {
            [0.0; 3]
        };
        let eddy_drive = if bval.abs() > 1e-9 {
            Some([bvec[0] * bval, bvec[1] * bval, bvec[2] * bval])
        } else {
            None
        };
        let comps: [&[f32]; 1] = [img];
        simulate_slice(
            &SliceInput {
                compartments: &comps,
                t2,
                t_inhom: None,
                fmap,
                phase0: None,
                sim: [nx, ny],
                acq_matrix: [nx, ny],
                z,
                nz,
                eddy_drive,
                prep_drive: None,
                slice_seed,
                eddy_lin: None,
            },
            acq,
        )
    }

    fn phantom(nx: usize, ny: usize) -> Vec<f32> {
        // a bright square in the middle
        let mut v = vec![0.0f32; nx * ny];
        for y in ny / 3..2 * ny / 3 {
            for x in nx / 3..2 * nx / 3 {
                v[x + nx * y] = 1.0;
            }
        }
        v
    }

    #[test]
    fn roundtrip_recovers_image_when_no_distortion() {
        let (nx, ny) = (16, 16);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let acq = Acquisition {
            signal_scale: 1.0,
            do_distortions: false,
            do_relaxation: false,
            ..Default::default()
        };
        let out = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &acq, [0.0, 0.0, 0.0], 0));
        let err: f32 = img.iter().zip(&out).map(|(a, b)| (a - b).abs()).sum::<f32>() / (nx * ny) as f32;
        assert!(err < 1e-4, "roundtrip mean abs err {err}");
    }

    #[test]
    fn uniform_fieldmap_shifts_along_phase_encode() {
        // A uniform off-resonance shifts the whole image along PE (y) by ~fmap·N_pe·t_line pixels.
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let f0 = 60.0f32; // Hz off-resonance
        let fmap = vec![f0; nx * ny];
        // predicted PE shift ≈ f0 · t_line[s] · N_pe = 60 · 0.003 · 24 ≈ 4.3 px
        let acq = Acquisition {
            t_line: 3.0, // ms
            signal_scale: 1.0,
            do_distortions: true,
            do_relaxation: false,
            reverse_phase: false,
            ..Default::default()
        };
        let out = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &acq, [0.0, 0.0, 0.0], 0));
        // centroid of the bright region should move in y vs the undistorted image
        let cy = |v: &[f32]| {
            let (mut sw, mut sy) = (0.0f64, 0.0f64);
            for y in 0..ny {
                for x in 0..nx {
                    let w = v[x + nx * y] as f64;
                    sw += w;
                    sy += w * y as f64;
                }
            }
            sy / sw.max(1e-9)
        };
        let acq0 = Acquisition { do_distortions: false, ..acq.clone() };
        let base = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &acq0, [0.0, 0.0, 0.0], 0));
        let shift = cy(&out) - cy(&base);
        assert!(shift.abs() > 1.5, "expected a clear PE shift (~4px), got {shift}");
        // reverse phase-encode flips the distortion direction
        let acq_rev = Acquisition { reverse_phase: true, ..acq.clone() };
        let rev = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &acq_rev, [0.0, 0.0, 0.0], 0));
        let shift_rev = cy(&rev) - cy(&base);
        assert!(shift * shift_rev < 0.0, "AP/PA should distort oppositely: {shift} vs {shift_rev}");
    }

    #[test]
    fn partial_fourier_changes_the_image_but_preserves_brain() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let full = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let pf = Acquisition { partial_fourier: 0.6, ..full.clone() };
        let a = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &full, [0.0, 0.0, 0.0], 0));
        let b = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &pf, [0.0, 0.0, 0.0], 0));
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / (nx * ny) as f32;
        assert!(diff > 1e-3, "partial Fourier should alter the image, diff {diff}");
        // most of the signal energy is still there (PF keeps the central k-space)
        let (ea, eb): (f32, f32) = (a.iter().sum(), b.iter().sum());
        assert!((eb / ea - 1.0).abs() < 0.5, "PF shouldn't destroy the brain: {ea} vs {eb}");
    }

    #[test]
    fn nyquist_ghost_leaks_signal_into_background() {
        // off-centre source so its N/2 ghost lands in otherwise-empty background
        let (nx, ny) = (24, 24);
        let mut img = vec![0.0f32; nx * ny];
        for y in 2..6 {
            for x in 10..14 {
                img[x + nx * y] = 1.0;
            }
        }
        let fmap = vec![0.0f32; nx * ny];
        let base = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let ghost = Acquisition { ghost_offset: 0.5, ..base.clone() };
        let a = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &base, [0.0, 0.0, 0.0], 0));
        let b = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &ghost, [0.0, 0.0, 0.0], 0));
        // background region ~half-FOV away in the PE(y) direction from the source
        let bg = |v: &[f32]| {
            let mut s = 0.0f32;
            for y in 14..18 {
                for x in 10..14 {
                    s += v[x + nx * y];
                }
            }
            s
        };
        assert!(bg(&b) > bg(&a) + 0.1, "ghost should add background signal: {} vs {}", bg(&a), bg(&b));
    }

    #[test]
    fn eddy_currents_affect_dwi_but_not_b0() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let base = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let eddy = Acquisition { eddy_strength: 5.0, ..base.clone() };
        // DWI volume (gradient along x): eddy shears the image
        let grad = [1.0, 0.0, 0.0];
        let a = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &base, grad, 0));
        let b = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &eddy, grad, 0));
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / (nx * ny) as f32;
        assert!(diff > 1e-3, "eddy should shear the DWI, diff {diff}");
        // b0 (zero gradient): eddy must have no effect
        let z0 = [0.0, 0.0, 0.0];
        let a0 = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &base, z0, 0));
        let b0 = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &eddy, z0, 0));
        let diff0: f32 = a0.iter().zip(&b0).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff0 < 1e-6, "eddy must not touch b0, diff {diff0}");
    }

    #[test]
    fn spikes_ripple_into_the_background() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let base = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let spiky = Acquisition { n_spikes: 3, spike_amplitude: 1.0, ..base.clone() };
        let a = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &base, [0.0, 0.0, 0.0], 0));
        let b = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &spiky, [0.0, 0.0, 0.0], 0));
        let corner = |v: &[f32]| {
            let mut s = 0.0f32;
            for y in 0..3 {
                for x in 0..3 {
                    s += v[x + nx * y];
                }
            }
            s
        };
        assert!(corner(&b) > corner(&a) + 0.1, "spikes should ripple into background: {} vs {}", corner(&a), corner(&b));
    }


    #[test]
    fn multicoil_rss_recovers_the_brain() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let acq = Acquisition { n_coils: 4, signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let out = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &acq, [0.0, 0.0, 0.0], 0));
        let center = out[12 + nx * 12];
        let corner = out[1 + nx * 1];
        assert!(center > 0.1 && center > corner, "RSS should recover the brain: center {center} corner {corner}");
    }

    /// END-TO-END: a Roemer combine of N coils must reproduce the single-coil image, at every
    /// oversampling factor.
    ///
    /// This is the test that actually exercises the production coordinate transform. The
    /// helper-level test below checks the arithmetic of the transform the forward model OUGHT to
    /// use; only this one goes through `build_coil_kspace` and `simulate_slice`, so only this one
    /// can catch the forward model passing a coordinate in the wrong frame.
    ///
    /// It exists because that is exactly what happened: the coil call was given the eddy
    /// polynomial's CENTRED coordinate (`(x - sxs - xoff)/o`, spanning [-nx/2, nx/2)) while the
    /// combine passed absolute indices (0..nx) into a function whose coil ring is centred on
    /// nx/2. A half-FOV origin error, present even at o = 1, which every existing multi-coil test
    /// missed because they only assert broad properties like "centre brighter than corner".
    ///
    /// The combine is exact only for sensitivities band-limited within the acquired band; these
    /// Gaussians (sigma = 0.9 max(nx,ny)) are very smooth but not exactly so, hence a tolerance
    /// rather than equality. Measured: 0.0 at o = 1 (the two grids coincide, so the combine
    /// divides out exactly), 1.1e-3 at o = 2 and 1.0e-3 at o = 4 — against 0.28 unregistered.
    #[test]
    fn multicoil_roemer_reproduces_the_single_coil_image_at_every_oversampling() {
        let (nx, ny) = (32usize, 32usize);
        for o in [1usize, 2, 4] {
            let (snx, sny) = (nx * o, ny * o);
            // Edges off the voxel grid so the object is not accidentally band-limited.
            let q = nx as f64 / 4.0;
            let img = box_hires(
                snx, sny,
                (q + 0.5) * o as f64, (3.0 * q + 0.5) * o as f64,
                (q + 0.5) * o as f64, (3.0 * q + 0.5) * o as f64,
            );
            let fmap = vec![0.0f32; snx * sny];
            let comps: [&[f32]; 1] = [&img];
            let inp = step_input(&comps, &fmap, None, snx, sny, nx, ny);

            let one = simulate_slice(&inp, &Acquisition { n_coils: 1, ..clean() });
            let many = simulate_slice(&inp, &Acquisition { n_coils: 6, ..clean() });

            let peak = one.iter().map(|&(r, i)| (r * r + i * i).sqrt()).fold(0.0f32, f32::max);
            assert!(peak > 0.5, "o={o}: degenerate reference, peak {peak}");
            let worst = one
                .iter()
                .zip(&many)
                .map(|(&(ar, ai), &(br, bi))| ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt())
                .fold(0.0f32, f32::max);
            assert!(
                worst / peak < 0.02,
                "o={o}: 6-coil Roemer combine differs from single coil by {:.4} of peak \
                 -- the forward model and the combine are not sampling one sensitivity field",
                worst / peak
            );
        }
    }

    /// The SUB-VOXEL half of the registration, at helper level.
    ///
    /// Scope, stated explicitly because getting this wrong is what let a half-FOV origin error
    /// ship: this test evaluates `coil_sensitivity` directly on the coordinate expression
    /// `(v*o + i - xoff)/o`. It therefore pins that expression's arithmetic — that the `o` sim
    /// cells of acquired voxel `v` are centred ON `v`, so their sensitivities average to the
    /// value the combine uses there — and NOT that the forward model passes that expression.
    /// A test that evaluates the transform it wishes the caller used cannot catch the caller
    /// using a different one. `multicoil_roemer_reproduces_the_single_coil_image_at_every_
    /// oversampling` above covers that, through the production call path.
    ///
    /// The sub-voxel error this one is about: the original code passed sim indices against
    /// `(snx, sny)`, centring the ring on `snx/2` = `nx/2 - (o-1)/(2o)` in acquired-voxel terms,
    /// a quarter voxel from the combine's `nx/2` at o = 2.
    #[test]
    fn coil_sensitivity_is_registered_identically_on_both_grids() {
        let (nx, ny, ncoils) = (32usize, 32usize, 4usize);
        for o in [1usize, 2, 3, 4, 8] {
            let (xoff, yoff) = ((o as f64 - 1.0) / 2.0, (o as f64 - 1.0) / 2.0);
            for coil in 0..ncoils {
                for v in [0usize, 1, 7, nx / 2, nx - 1] {
                    // Mean over the sim cells of acquired voxel (v, v), in acquired-voxel units.
                    let mut acc = 0.0;
                    for i in 0..o {
                        for j in 0..o {
                            let xc = (v * o + i) as f64 / o as f64 - xoff / o as f64;
                            let yc = (v * o + j) as f64 / o as f64 - yoff / o as f64;
                            acc += coil_sensitivity(coil, ncoils, xc, yc, nx, ny);
                        }
                    }
                    let sim_mean = acc / (o * o) as f64;
                    let combined = coil_sensitivity(coil, ncoils, v as f64, v as f64, nx, ny);
                    // A Gaussian is not linear, so the block mean is not EXACTLY the centre
                    // value. Its curvature over one acquired voxel puts the residual at most
                    // 7.9e-5 across this sweep; the tolerance sits an order of magnitude above
                    // that and two below the unregistered error asserted just below, so the test
                    // separates the two rather than merely accepting the current numbers.
                    assert!(
                        (sim_mean - combined).abs() < 1e-3,
                        "o={o} coil={coil} v={v}: sim-grid mean {sim_mean} vs combine {combined}"
                    );

                    // The OLD convention, reconstructed here so the test cannot pass vacuously:
                    // sim index against `(snx, sny)`. Every length in `coil_sensitivity` scales
                    // with the matrix, so that is exactly this field evaluated at `x / o` — i.e.
                    // displaced by `xoff / o = (o-1)/(2o)` acquired voxels. Worst case over this
                    // sweep is 1.15e-2, 146x the registered residual.
                    let mut old = 0.0;
                    for i in 0..o {
                        for j in 0..o {
                            old += coil_sensitivity(
                                coil, ncoils,
                                (v * o + i) as f64 / o as f64,
                                (v * o + j) as f64 / o as f64,
                                nx, ny,
                            );
                        }
                    }
                    let old_mean = old / (o * o) as f64;
                    if o > 1 && v != nx / 2 {
                        assert!(
                            (old_mean - combined).abs() > (sim_mean - combined).abs(),
                            "o={o} coil={coil} v={v}: unregistered sampling was no worse \
                             ({old_mean} vs registered {sim_mean}, combine {combined})"
                        );
                    }
                }
            }
        }
    }

    /// The half-cell registration itself, stated as arithmetic: the sim cells of acquired voxel
    /// `v` have centroid exactly `v` in acquired-voxel units, at every oversampling factor. This
    /// is the invariant every acquired-grid quantity evaluated on the sim grid depends on.
    #[test]
    fn sim_cells_of_an_acquired_voxel_are_centred_on_it() {
        for o in [1usize, 2, 3, 4, 8, 16] {
            let xoff = (o as f64 - 1.0) / 2.0;
            for v in 0..5usize {
                let mean: f64 = (0..o)
                    .map(|i| ((v * o + i) as f64 - xoff) / o as f64)
                    .sum::<f64>()
                    / o as f64;
                assert!((mean - v as f64).abs() < 1e-12, "o={o} v={v}: centroid {mean}");
            }
        }
    }

    #[test]
    fn grappa_unfolds_undersampled_multicoil_data() {
        let (nx, ny) = (32, 32);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let full = Acquisition { n_coils: 8, signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let accel = Acquisition { accel: 2, acs_lines: 16, ..full.clone() };
        let a = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &full, [0.0, 0.0, 0.0], 0));
        let g = mag(&slice1(&img, &[T2Slice::Uniform(100.0)], &fmap, nx, ny, 0, 1, &accel, [0.0, 0.0, 0.0], 0));
        // noise-free: GRAPPA should recover the fully-sampled image closely (aliasing unfolded)
        let num: f32 = a.iter().zip(&g).map(|(x, y)| (x - y).abs()).sum();
        let den: f32 = a.iter().sum::<f32>().max(1e-6);
        let rel = num / den;
        assert!(rel < 0.15, "GRAPPA should reconstruct the image, rel err {rel}");
    }
    use crate::analytic::truncated_step_profile;

    /// Acquisition with every artifact off: pure finite-Fourier acquisition.
    fn clean() -> Acquisition {
        Acquisition {
            signal_scale: 1.0,
            do_distortions: false,
            do_relaxation: false,
            noise_variance: 0.0,
            partial_fourier: 1.0,
            pf_mode: PartialFourierMode::FiberfoxCompatible,
            ghost_offset: 0.0,
            eddy_strength: 0.0,
            n_coils: 1,
            accel: 1,
            ..Acquisition::default()
        }
    }

    /// Build a step-edge SliceInput on the sim grid, optionally with an object phase field.
    fn step_input<'a>(
        comps: &'a [&'a [f32]],
        fmap: &'a [f32],
        phase0: Option<&'a [f64]>,
        snx: usize, sny: usize, nx: usize, ny: usize,
    ) -> SliceInput<'a> {
        SliceInput {
            compartments: comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap, phase0,
            sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
            eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None
        }
    }

    #[test]
    fn noise_lands_only_on_acquired_samples() {
        // The original defect (finding 2.5): noise populated k-space that was never acquired,
        // so the signal was band-limited while the noise stayed white to the full Nyquist.
        let (nx, ny) = (24usize, 24usize);
        let acq = Acquisition { partial_fourier: 0.75, noise_variance: 1.0, ..clean() };
        let mask = sampling_mask(nx, ny, &acq);
        assert!(mask.iter().any(|b| !b), "pf=0.75 must leave some samples unacquired");
        let empty = vec![0.0f32; nx * ny];
        let fmap = vec![0.0f32; nx * ny];
        let comps: [&[f32]; 1] = [&empty];
        let k = simulate_slice_kspace(
            &SliceInput {
                compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
                sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                eddy_drive: None, prep_drive: None, slice_seed: 9, eddy_lin: None
            },
            &acq,
        );
        for i in 0..nx * ny {
            if !mask[i] {
                assert_eq!((k[i].0, k[i].1), (0.0, 0.0), "unacquired sample {i} carries noise");
            }
        }
    }

    #[test]
    fn noise_sd_scales_as_sqrt_sampled_fraction() {
        // Scoped: GRAPPA disabled, no window, identical per-sample thermal variance (spec 4.1.8).
        // `noise_variance` is the PER-COMPONENT variance of the reconstructed image at full
        // sampling, so the full-sampling SD is sqrt(noise_variance) = 1.0 here.
        let (nx, ny) = (32usize, 32usize);
        let sd_at = |pf: f64| -> f64 {
            let acq = Acquisition { partial_fourier: pf, noise_variance: 1.0, ..clean() };
            let empty = vec![0.0f32; nx * ny];
            let fmap = vec![0.0f32; nx * ny];
            let comps: [&[f32]; 1] = [&empty];
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
                    sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    eddy_drive: None, prep_drive: None, slice_seed: 3, eddy_lin: None
                },
                &acq,
            );
            let v: f64 = out.iter().map(|p| (p.0 as f64).powi(2)).sum::<f64>() / (nx * ny) as f64;
            v.sqrt()
        };
        let full = sd_at(1.0);
        assert!((full - 1.0).abs() < 0.15, "full-sampling per-component SD {full:.3}, expected ~1.0");
        let frac = |pf: f64| {
            let acq = Acquisition { partial_fourier: pf, ..clean() };
            let m = sampling_mask(nx, ny, &acq);
            m.iter().filter(|b| **b).count() as f64 / (nx * ny) as f64
        };
        for pf in [0.75, 0.5] {
            let (sd, f) = (sd_at(pf), frac(pf));
            let ratio = sd / full;
            assert!(
                (ratio - f.sqrt()).abs() < 0.1,
                "pf={pf}: SD ratio {ratio:.3}, expected sqrt(sampled fraction {f:.3}) = {:.3}",
                f.sqrt()
            );
        }
    }

    #[test]
    fn noise_covariance_matches_the_sampling_mask() {
        // Spec 4.1.7a, scoped to GRAPPA disabled and window None: for white noise on mask M, the
        // reconstructed image noise autocovariance is the inverse DFT of M, up to scale.
        let (nx, ny) = (16usize, 16usize);
        let acq = Acquisition { partial_fourier: 0.75, noise_variance: 1.0, ..clean() };
        let mask = sampling_mask(nx, ny, &acq);
        let empty = vec![0.0f32; nx * ny];
        let fmap = vec![0.0f32; nx * ny];
        let (trials, lags) = (240usize, 4usize);
        let mut meas = vec![0.0f64; lags];
        for t in 0..trials {
            let comps: [&[f32]; 1] = [&empty];
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
                    sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    eddy_drive: None, prep_drive: None, slice_seed: 1000 + t as u64, eddy_lin: None
                },
                &acq,
            );
            // Lag along the PHASE-ENCODE axis. Partial Fourier undersamples ky only, so a
            // readout-direction lag cannot see it: the skipped lines contribute uniformly to
            // every kx lag and cancel when normalizing by lag 0. Measuring along kx made this
            // test pass even with the mask guard removed.
            for (d, m) in meas.iter_mut().enumerate() {
                let mut acc = 0.0;
                for y in 0..ny - d {
                    for x in 0..nx {
                        acc += out[x + nx * y].0 as f64 * out[x + nx * (y + d)].0 as f64;
                    }
                }
                *m += acc / (nx * (ny - d)) as f64;
            }
        }
        for m in meas.iter_mut() {
            *m /= trials as f64;
        }
        // Prediction: inverse DFT of the mask along ky.
        let mut pred = vec![0.0f64; lags];
        for (d, p) in pred.iter_mut().enumerate() {
            let mut acc = 0.0;
            for kyi in 0..ny {
                for kxi in 0..nx {
                    if mask[kxi + nx * kyi] {
                        let ky = (kyi as f64 - (ny / 2) as f64) / ny as f64;
                        acc += (TAU * ky * d as f64).cos();
                    }
                }
            }
            *p = acc;
        }
        for d in 1..lags {
            let (a, b) = (meas[d] / meas[0], pred[d] / pred[0]);
            assert!((a - b).abs() < 0.08, "lag {d}: measured {a:.3}, mask prediction {b:.3}");
        }
    }

    #[test]
    fn partial_fourier_ringing_is_asymmetric_about_the_edge() {
        // Zero-filled PF (no homodyne/POCS) rings asymmetrically along PE, unlike the symmetric
        // full-Fourier case. 6/8 is the shipping default, so this is the primary configuration.
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let edge = (ny as f64 / 2.0 + 0.5) * o as f64;
        let mut img = vec![0.0f32; snx * sny];
        for y in 0..sny {
            let f = if (y as f64 + 1.0) <= edge {
                0.0
            } else if y as f64 >= edge {
                1.0
            } else {
                y as f64 + 1.0 - edge
            };
            for x in 0..snx {
                img[x + snx * y] = f as f32;
            }
        }
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let run = |pf: f64| {
            simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None
                },
                &Acquisition { partial_fourier: pf, ..clean() },
            )
        };
        let col = nx / 2;
        let asym = |v: &Vec<(f32, f32)>| {
            let below: f64 = (1..6).map(|d| (v[col + nx * (ny / 2 - d)].0 as f64).powi(2)).sum();
            let above: f64 = (1..6).map(|d| (v[col + nx * (ny / 2 + d)].0 as f64 - 1.0).powi(2)).sum();
            (above / below.max(1e-12)).ln().abs()
        };
        let full = run(1.0);
        for pf in [0.75, 0.875] {
            let p = run(pf);
            assert!(
                asym(&p) > asym(&full),
                "pf={pf} should ring more asymmetrically than full Fourier: {} vs {}",
                asym(&p), asym(&full)
            );
        }
        let e_full: f32 = full.iter().map(|v| v.0.abs()).sum();
        let e_pf: f32 = run(0.75).iter().map(|v| v.0.abs()).sum();
        assert!((e_pf / e_full - 1.0).abs() < 0.5, "PF energy {e_pf} vs full {e_full}");
    }

    #[test]
    fn even_matrix_window_asymmetry_is_intentional() {
        // The acquired band is [-n/2, n/2-1]: asymmetric about k=0 by one sample, exactly as real
        // even-matrix Cartesian acquisitions are. Retained deliberately (spec 3.4), so pin it.
        let n = 32i64;
        let (lo, hi) = (-(n / 2), n / 2 - 1);
        assert_eq!((lo, hi), (-16, 15));
        assert_eq!((hi - lo + 1) as usize, n as usize, "the band must hold exactly n samples");
        assert_eq!(lo.abs() - hi.abs(), 1, "one extra negative-frequency sample, by convention");

        // Consequence: a real object acquires a small imaginary component. It is deterministic and
        // is NOT realistic object phase -- that is what `phase.rs` supplies.
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let out = simulate_slice(&step_input(&comps, &fmap, None, snx, sny, nx, ny), &clean());
        let mr = out.iter().map(|p| p.0.abs()).fold(0.0f32, f32::max);
        let mi = out.iter().map(|p| p.1.abs()).fold(0.0f32, f32::max);
        let residual = (mi / mr) as f64;
        assert!(residual < 0.05, "asymmetry residual should stay small: {residual}");
        assert!(residual > 1e-4, "a real object should still show the asymmetry: {residual}");
    }

    #[test]
    fn window_filters_signal_and_noise_together() {
        // The original defect (finding 2.5) was band-limited signal with unfiltered noise.
        // A reconstruction window must multiply both, so noise-only data must be suppressed too.
        let (nx, ny) = (32usize, 32usize);
        let empty = vec![0.0f32; nx * ny];
        let fmap = vec![0.0f32; nx * ny];
        let sd = |w: KspaceWindow| -> f64 {
            let comps: [&[f32]; 1] = [&empty];
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
                    sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    eddy_drive: None, prep_drive: None, slice_seed: 5, eddy_lin: None
                },
                &Acquisition { noise_variance: 1.0, window: w, ..clean() },
            );
            (out.iter().map(|p| (p.0 as f64).powi(2)).sum::<f64>() / (nx * ny) as f64).sqrt()
        };
        let unwindowed = sd(KspaceWindow::None);
        let hann = sd(KspaceWindow::Hann);
        assert!(hann < 0.8 * unwindowed, "a window must attenuate noise too: {hann:.3} vs {unwindowed:.3}");
    }

    #[test]
    fn window_reduces_ringing_below_the_unapodized_case() {
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let peak = |w: KspaceWindow| {
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None
                },
                &Acquisition { window: w, ..clean() },
            );
            (nx / 2 + 1..nx).map(|x| out[x + nx * (ny / 2)].0 as f64).fold(f64::MIN, f64::max)
        };
        let (plain, hann) = (peak(KspaceWindow::None), peak(KspaceWindow::Hann));
        assert!(hann < plain, "apodization must reduce overshoot: {hann:.4} vs {plain:.4}");
        assert!(plain > 1.05, "unapodized case should show real Gibbs overshoot: {plain:.4}");
    }

    #[test]
    fn global_phase_rotation_is_exact() {
        // Rotating the object by exp(i*alpha) must rotate the reconstructed image by exactly
        // exp(i*alpha) and leave its magnitude untouched. A gauge invariance no phase-histogram
        // statistic can substitute for (spec 4.1.5).
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let acq = clean();
        let base = simulate_slice(&step_input(&comps, &fmap, None, snx, sny, nx, ny), &acq);
        let alpha = 0.7f64;
        let field = vec![alpha; snx * sny];
        let rot = simulate_slice(&step_input(&comps, &fmap, Some(&field), snx, sny, nx, ny), &acq);
        let (sa, ca) = alpha.sin_cos();
        for i in 0..nx * ny {
            let (re, im) = (base[i].0 as f64, base[i].1 as f64);
            let (er, ei) = (re * ca - im * sa, re * sa + im * ca);
            assert!((rot[i].0 as f64 - er).abs() < 1e-6, "re mismatch at {i}");
            assert!((rot[i].1 as f64 - ei).abs() < 1e-6, "im mismatch at {i}");
            let m0 = (re * re + im * im).sqrt();
            let m1 = ((rot[i].0 as f64).powi(2) + (rot[i].1 as f64).powi(2)).sqrt();
            assert!((m0 - m1).abs() < 1e-6, "magnitude changed at {i}");
        }
    }

    #[test]
    fn object_phase_puts_ringing_in_both_channels() {
        // The original defect: with a real object the image is real to machine precision, so all
        // ringing sits in Re and part-phase is degenerate. With object phase it must not.
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let acq = clean();
        let ratio = |v: &Vec<(f32, f32)>| {
            let mr = v.iter().map(|p| p.0.abs()).fold(0.0f32, f32::max);
            let mi = v.iter().map(|p| p.1.abs()).fold(0.0f32, f32::max);
            (mi / mr) as f64
        };
        // Without object phase the image is real APART FROM the deliberate even-N band
        // asymmetry (spec 3.4), whose residual `even_matrix_window_asymmetry_is_intentional`
        // bounds at 0.05. Use that same bound here: a tighter one would contradict it.
        let none = simulate_slice(&step_input(&comps, &fmap, None, snx, sny, nx, ny), &acq);
        let r_none = ratio(&none);
        assert!(r_none < 0.05, "no-phase Im/Re should be only the even-N residual: {r_none}");
        let ramp: Vec<f64> = (0..snx * sny)
            .map(|i| 0.9 * ((i % snx) as f64 - snx as f64 / 2.0) / snx as f64)
            .collect();
        let withph = simulate_slice(&step_input(&comps, &fmap, Some(&ramp), snx, sny, nx, ny), &acq);
        let r_ph = ratio(&withph);
        assert!(r_ph > 0.1, "phase should move energy into Im: {r_ph}");
        // And the effect must dominate the asymmetry residual, not merely exceed a threshold.
        assert!(r_ph > 5.0 * r_none, "object phase {r_ph} vs even-N residual {r_none}");
    }

    #[test]
    fn crop_reproduces_the_analytic_profile_across_subvoxel_offsets() {
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let fmap = vec![0.0f32; snx * sny];
        let acq = clean();

        for i in 0..16 {
            let delta = i as f64 / 16.0;
            // Edge at (nx/2 + delta) acquired voxels, expressed in sim-voxel units.
            let edge = (nx as f64 / 2.0 + delta) * o as f64;
            let img = step_hires(snx, sny, edge);
            let comps: [&[f32]; 1] = [&img];
            let inp = SliceInput {
                compartments: &comps,
                t2: &[T2Slice::Uniform(100.0)],
                t_inhom: None,
                fmap: &fmap,
                phase0: None,
                sim: [snx, sny],
                acq_matrix: [nx, ny],
                z: 0,
                nz: 1,
                eddy_drive: None,
                prep_drive: None,
                slice_seed: 0,
                eddy_lin: None,
            };
            let out = simulate_slice(&inp, &acq);

            let expect = truncated_step_profile(nx, (nx as f64 / 2.0 + delta) / nx as f64);
            let row = ny / 2;
            let worst = (0..nx)
                .map(|x| (out[x + nx * row].0 as f64 - expect[x]).abs())
                .fold(0.0f64, f64::max);
            assert!(worst < 5e-3, "offset {delta}: max profile deviation {worst:.4e}");
        }
    }

    #[test]
    fn ringing_is_intrinsic_without_any_ringing_parameter() {
        // A sub-voxel-positioned edge must now ring: the old exact-DFT round-trip is gone.
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let fmap = vec![0.0f32; snx * sny];
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let comps: [&[f32]; 1] = [&img];
        let inp = SliceInput {
            compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
            sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
            eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None
        };
        let out = simulate_slice(&inp, &clean());
        let row = ny / 2;
        let peak = (nx / 2 + 1..nx).map(|x| out[x + nx * row].0 as f64).fold(f64::MIN, f64::max);
        assert!(peak > 1.05, "expected intrinsic overshoot, got peak {peak:.4}");
    }

    #[test]
    fn acquired_band_converges_with_simulation_resolution() {
        // Spec 4.1.6: adequacy is convergence of the ACQUIRED coefficients, not smoothness of the
        // object. A sharp edge is deliberately not band-limited; that is not a defect.
        let (nx, ny) = (16usize, 16usize);
        let acq = clean();
        let k_at = |o: usize| -> Vec<(f64, f64)> {
            let (snx, sny) = (nx * o, ny * o);
            let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.37) * o as f64);
            let fmap = vec![0.0f32; snx * sny];
            let comps: [&[f32]; 1] = [&img];
            simulate_slice_kspace(
                &SliceInput {
                    compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None
                },
                &acq,
            )
        };
        let rel = |a: &[(f64, f64)], b: &[(f64, f64)]| {
            let num: f64 =
                a.iter().zip(b).map(|(p, q)| (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sum();
            let den: f64 = b.iter().map(|q| q.0 * q.0 + q.1 * q.1).sum::<f64>().max(1e-30);
            (num / den).sqrt()
        };
        let (k2, k4, k8) = (k_at(2), k_at(4), k_at(8));
        let (e2, e4) = (rel(&k2, &k4), rel(&k4, &k8));
        assert!(e4 < e2, "error must shrink with o: e(2->4)={e2:.4e}, e(4->8)={e4:.4e}");
        // Report o_min for the production default (spec 4.1.10): the smallest o under tolerance.
        let eps = 1e-3;
        let o_min = if e2 < eps { 2 } else if e4 < eps { 4 } else { 8 };
        println!("o_min at eps={eps}: {o_min}  (e2={e2:.3e}, e4={e4:.3e})");
        assert!(e4 < 1e-2, "o=4 should be within 1% of o=8: {e4:.4e}");
    }

    #[test]
    fn production_path_produces_intrinsic_ringing_and_complex_phase() {
        // Regression guard for the defect this test exists because of: the shipped CLI ran the
        // o=1 legacy path, where the forward/inverse transforms are an exact round trip, so after
        // zero_ringing was removed it produced +0.0000% overshoot and no object phase.
        use crate::phase::PhaseModel;
        let (nx, ny, nz, o) = (32usize, 32usize, 1usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny * nz];
        let acq = Acquisition {
            signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default()
        };
        let model = PhaseModel::hbcd_like();
        let (mag, ph) = simulate_acquisition_oversampled(
            [snx, sny, nz], [nx, ny, nz], 1, &[img], &[T2Volume::Uniform(100.0)], &fmap, None, &acq,
            &[Some([1000.0, 0.0, 0.0])], &[Some((1000.0, [1.0, 0.0, 0.0]))], &model, 7,
            None,
            None,
        );
        let row: Vec<f64> = (nx / 2 + 1..nx).map(|x| mag[(x + nx * (ny / 2)) * 1] as f64).collect();
        let over = row.iter().cloned().fold(f64::MIN, f64::max) - 1.0;
        assert!(over > 0.05, "production path must ring intrinsically, got {:+.4}%", over * 100.0);

        // and the object must actually be complex: phase varies across the bright region
        let bright: Vec<f64> = (nx / 2 + 2..nx - 2)
            .map(|x| ph[(x + nx * (ny / 2)) * 1] as f64)
            .collect();
        let spread = bright.iter().cloned().fold(f64::MIN, f64::max)
            - bright.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 0.1, "object phase should vary across the image, spread {spread}");
    }

    #[test]
    fn legacy_path_is_documented_as_ringing_free() {
        // Pins WHY the legacy path must not be the production one, so nobody re-wires it.
        let (nx, ny, nz) = (32usize, 32usize, 1usize);
        let mut img = vec![0.0f32; nx * ny * nz];
        for y in 0..ny {
            for x in nx / 2..nx {
                img[x + nx * y] = 1.0;
            }
        }
        let acq = Acquisition {
            signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default()
        };
        let (mag, _) = simulate_acquisition_legacy(
            [nx, ny, nz], 1, &[img], &[T2Volume::Uniform(100.0)], &vec![0.0f32; nx * ny * nz], None, &acq,
            &[[0.0, 0.0, 0.0]],
        );
        let over = (nx / 2 + 1..nx)
            .map(|x| mag[(x + nx * (ny / 2)) * 1] as f64)
            .fold(f64::MIN, f64::max) - 1.0;
        assert!(over.abs() < 1e-4, "legacy path is an exact round trip by construction: {over}");
    }

    #[test]
    fn partial_fourier_keeps_the_same_line_count_in_both_pe_polarities() {
        // AP and PA must acquire equally many PE lines, or a reverse-PE pair is not comparable --
        // which is the whole point of simulating one (topup / DRBUDDI).
        for (ny, pf) in [(32usize, 0.75f64), (32, 0.875), (64, 0.75), (64, 0.875), (30, 0.75)] {
            let mk = |rev: bool| {
                let a = Acquisition { partial_fourier: pf, reverse_phase: rev, ..Default::default() };
                let m = sampling_mask(16, ny, &a);
                (0..ny).filter(|&ky| m[16 * ky]).count()
            };
            let (f, r) = (mk(false), mk(true));
            let want = (ny as f64 * pf).round() as usize;
            assert_eq!(f, r, "ny={ny} pf={pf}: forward keeps {f} lines, reverse {r}");
            assert!(
                (f as i64 - want as i64).abs() <= 1,
                "ny={ny} pf={pf}: kept {f} lines, expected about {want}"
            );
        }
    }

    #[test]
    fn spikes_land_only_on_acquired_samples() {
        let (nx, ny) = (24usize, 24usize);
        let acq = Acquisition {
            signal_scale: 1.0, do_distortions: false, do_relaxation: false,
            partial_fourier: 0.75, n_spikes: 40, spike_amplitude: 1.0, ..Default::default()
        };
        let mask = sampling_mask(nx, ny, &acq);
        let img = vec![1.0f32; nx * ny];
        let comps: [&[f32]; 1] = [&img];
        let k = simulate_slice_kspace(
            &SliceInput {
                compartments: &comps, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &vec![0.0f32; nx * ny], phase0: None,
                sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                eddy_drive: None, prep_drive: None, slice_seed: 5, eddy_lin: None
            },
            &acq,
        );
        for i in 0..nx * ny {
            if !mask[i] {
                assert_eq!((k[i].0, k[i].1), (0.0, 0.0), "spike landed on unacquired sample {i}");
            }
        }
    }

    #[test]
    fn contiguous_partial_fourier_keeps_exactly_the_nominal_fraction() {
        // The default Fiberfox rule does NOT: on an even matrix it preserves line zero because
        // that line's Nyquist conjugate is absent, so nominal 6/8 is 25/32 = 78.1%, with one
        // isolated line detached from the acquired block. Contiguous mode is what a scanner
        // produces and what a PF-aware unringing benchmark should default to comparing against.
        for &(ny, pf) in &[(32usize, 0.75f64), (32, 0.875), (64, 0.75), (140, 0.75)] {
            let count = |mode, rev| {
                let a = Acquisition {
                    partial_fourier: pf, pf_mode: mode, reverse_phase: rev, ..Default::default()
                };
                let m = sampling_mask(8, ny, &a);
                (0..ny).filter(|&k| m[8 * k]).count()
            };
            let want = (ny as f64 * pf).round() as usize;
            for rev in [false, true] {
                assert_eq!(
                    count(PartialFourierMode::Contiguous, rev), want,
                    "ny={ny} pf={pf} reverse={rev}: contiguous must keep exactly {want} lines"
                );
            }
            // and the two modes must actually differ on an even matrix
            if ny % 2 == 0 {
                assert!(
                    count(PartialFourierMode::FiberfoxCompatible, false) > want,
                    "ny={ny} pf={pf}: the Fiberfox rule should keep MORE than {want}"
                );
            }
        }
    }

    #[test]
    fn contiguous_partial_fourier_lines_are_actually_contiguous() {
        let a = Acquisition {
            partial_fourier: 0.75, pf_mode: PartialFourierMode::Contiguous, ..Default::default()
        };
        let ny = 32;
        let m = sampling_mask(8, ny, &a);
        let kept: Vec<usize> = (0..ny).filter(|&k| m[8 * k]).collect();
        assert!(!kept.is_empty());
        assert_eq!(
            kept.last().unwrap() - kept[0] + 1, kept.len(),
            "contiguous mode must leave no gaps, got {kept:?}"
        );
    }
    /// The line-timing table the forward reads is exactly what `line_times` gave it before P5,
    /// bit for bit, and its polarity is the line parity the ghost always used.
    #[test]
    fn line_timing_table_is_line_times_bit_for_bit() {
        for &(nx, ny) in &[(17usize, 70usize), (16, 40), (8, 9), (64, 64), (1, 1)] {
            for reverse_phase in [false, true] {
                for (t_line, t_echo) in [(1.0, 90.0), (0.5, 30.0), (0.37, 12.0)] {
                    let acq = Acquisition { t_line, t_echo, reverse_phase, ..Acquisition::default() };
                    let epi = SingleShotEpi { kx_max: nx, ky_max: ny, t_line, t_echo, reverse_phase };
                    let (t, trf, tread) = line_times(&epi);
                    let tab = LineTiming::for_acquisition(&acq, nx, ny);
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&tab.t_ms), bits(&t));
                    assert_eq!(bits(&tab.trf_ms), bits(&trf));
                    assert_eq!(bits(&tab.tread_ms), bits(&tread));
                    assert!(tab.polarity.iter().enumerate().all(|(k, &p)| p == if k % 2 == 1 { -1 } else { 1 }));
                }
            }
        }
    }

    /// The production forward is the literal per-line sum reorganised (static factors hoisted,
    /// the affine phase advanced by memoised rotors, the eddy polynomial factored per axis, the
    /// x-DFT done by FFT / twiddle table). Every effect, alone and combined, must reproduce the
    /// literal sum to round-off — including the rotor re-anchoring (ny > REANCHOR), the odd-line
    /// dwell offset, GRAPPA strides, partial Fourier in both modes and polarities, odd matrices
    /// and o = 1 as well as o = 2.
    #[test]
    fn restructured_forward_matches_the_literal_sum() {
        let mut rng = Rng(0xC0FFEE);
        let unit = |r: &mut Rng| r.unit();
        for &(nx, ny, o) in &[(17usize, 70usize, 2usize), (16, 40, 1), (8, 9, 3)] {
            let (snx, sny) = (nx * o, ny * o);
            let n = snx * sny;
            let mut comps: Vec<Vec<f32>> = Vec::new();
            for c in 0..3 {
                comps.push((0..n).map(|i| {
                    let (x, y) = ((i % snx) as f64 / snx as f64, (i / snx) as f64 / sny as f64);
                    let blob = (-(((x - 0.5).powi(2) + (y - 0.45).powi(2)) / (0.03 * (c + 1) as f64))).exp();
                    (blob + 0.05 * unit(&mut rng)) as f32
                }).collect());
            }
            let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
            let t2 = [T2Slice::Uniform(70.0f32), T2Slice::Uniform(100.0), T2Slice::Uniform(2000.0)];
            // fieldmap with a large range (±300 Hz over a 40–70 ms readout: ±20 cycles at the edge)
            let fmap: Vec<f32> = (0..n).map(|i| {
                let (x, y) = ((i % snx) as f64 / snx as f64, (i / snx) as f64 / sny as f64);
                (300.0 * ((3.0 * x).sin() * (2.0 * y + 1.0).cos()) + 20.0 * unit(&mut rng)) as f32
            }).collect();
            let phase0: Vec<f64> = (0..n).map(|_| TAU * unit(&mut rng)).collect();
            let full = Acquisition {
                do_distortions: true, do_relaxation: true, signal_scale: 100.0,
                ..Acquisition::default()
            };
            // eddy_drive is bvec*bval of the old table; every drive there had bval 1.0 or 0.0.
            let cases: Vec<(&str, Acquisition, Option<[f64; 3]>, Option<[f64; 3]>, usize, usize)> = vec![
                ("clean", Acquisition { do_distortions: false, do_relaxation: false, ..full.clone() }, None, None, 0, 1),
                ("distortion+relaxation", full.clone(), None, None, 0, 1),
                ("reverse", Acquisition { reverse_phase: true, ..full.clone() }, None, None, 0, 1),
                ("ghost", Acquisition { ghost_offset: 0.015, ..full.clone() }, None, None, 0, 1),
                ("eddy-poly", Acquisition { eddy_strength: 3.0, eddy_quad: 0.4, eddy_tau: 70.0, ..full.clone() },
                    Some([0.3, -0.8, 0.5]), None, 0, 1),
                ("eddy-phase", Acquisition { eddy_phase: 0.2, ..full.clone() }, Some([0.6, 0.6, 0.5]), None, 0, 1),
                ("eddy-trace", full.clone(), Some([0.6, 0.6, 0.5]), Some([0.03, -0.05, 0.02]), 0, 1),
                ("pf-fiberfox", Acquisition { partial_fourier: 0.75, ..full.clone() }, None, None, 0, 1),
                ("pf-contiguous-reverse", Acquisition { partial_fourier: 0.75, pf_mode: PartialFourierMode::Contiguous,
                    reverse_phase: true, ..full.clone() }, None, None, 0, 1),
                ("grappa-coils", Acquisition { accel: 2, acs_lines: 6, n_coils: 4, ..full.clone() }, None, None, 2, 4),
                ("grappa3", Acquisition { accel: 3, acs_lines: 8, n_coils: 4, ..full.clone() }, None, None, 1, 4),
                ("everything", Acquisition { ghost_offset: 0.02, eddy_strength: 2.0, eddy_quad: 0.3, eddy_phase: 0.1,
                    partial_fourier: 0.8, accel: 2, acs_lines: 8, n_coils: 3, ..full.clone() },
                    Some([0.5, 0.5, 0.7]), Some([0.02, 0.04, -0.01]), 0, 3),
                // gradient echo (P5 part A): decay from the RF and the static fmap*TE phase
                ("ge-clean", Acquisition { echo: EchoFormation::Gradient, do_distortions: false, do_relaxation: false,
                    ..full.clone() }, None, None, 0, 1),
                ("ge-distortion+relaxation", Acquisition { echo: EchoFormation::Gradient, ..full.clone() }, None, None, 0, 1),
                ("ge-everything", Acquisition { echo: EchoFormation::Gradient, ghost_offset: 0.02, eddy_strength: 2.0,
                    eddy_quad: 0.3, eddy_phase: 0.1, partial_fourier: 0.8, accel: 2, acs_lines: 8, n_coils: 3,
                    reverse_phase: true, ..full.clone() },
                    Some([0.5, 0.5, 0.7]), Some([0.02, 0.04, -0.01]), 0, 3),
            ];
            for (name, acq, eddy_drive, eddy_lin, coil, ncoils) in cases {
                let inp = SliceInput {
                    compartments: &comp_refs, t2: &t2, t_inhom: None, fmap: &fmap, phase0: Some(&phase0),
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 3, nz: 9, eddy_drive, prep_drive: None, slice_seed: 0, eddy_lin,
                };
                let a = build_coil_kspace(&inp, &acq, coil, ncoils);
                let b = reference_coil_kspace(&inp, &acq, coil, ncoils);
                let peak = b.iter().map(|k| k.abs()).fold(0.0, f64::max);
                let worst = a.iter().zip(&b).map(|(p, q)| (p.re - q.re).hypot(p.im - q.im)).fold(0.0, f64::max);
                assert!(peak > 0.0, "{name}: empty reference");
                assert!(worst <= 1e-10 * peak, "{name} @ {nx}x{ny} o={o}: max |Δ| {worst:.3e} vs peak {peak:.3e}");
            }
        }
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn fft_inverse_matches_the_direct_dft() {
        // random k-space of a typical acquired size; the FFT recon must equal the direct DFT.
        let (nx, ny) = (84, 100);
        let mut rng = Rng(0xC0FFEE);
        let ks: Vec<C> = (0..nx * ny).map(|_| C { re: rng.gauss(), im: rng.gauss() }).collect();
        let (xs, ys) = (nx / 2, ny / 2);
        let direct = inverse_2d(&ks, nx, ny, xs, ys);
        let fft = inverse_2d_fft(&ks, nx, ny, xs, ys);
        let mut max = 0.0f64;
        for (a, b) in direct.iter().zip(&fft) {
            max = max.max((a.re - b.re).abs()).max((a.im - b.im).abs());
        }
        // amplitudes are O(nx*ny) after an unnormalized inverse, so a ~1e-8 relative tolerance
        assert!(max < 1e-6 * (nx * ny) as f64, "FFT vs direct DFT max abs diff {max}");
    }
    #[cfg(feature = "kspace")]
    #[test]
    #[ignore = "timing micro-benchmark; run explicitly with --ignored"]
    fn bench_inverse_direct_vs_fft() {
        // Load-independent micro-benchmark of the reconstruction transform at the acquired size.
        let (nx, ny) = (85usize, 128usize);
        let mut rng = Rng(1);
        let ks: Vec<C> = (0..nx * ny).map(|_| C { re: rng.gauss(), im: rng.gauss() }).collect();
        let (xs, ys) = (nx / 2, ny / 2);
        let n = 3000;
        let mut sink = 0.0f64;
        let t0 = std::time::Instant::now();
        for _ in 0..n { sink += inverse_2d(&ks, nx, ny, xs, ys)[0].re; }
        let td = t0.elapsed();
        let t1 = std::time::Instant::now();
        for _ in 0..n { sink += inverse_2d_fft(&ks, nx, ny, xs, ys)[0].re; }
        let tf = t1.elapsed();
        println!(
            "inverse {nx}x{ny}: direct {:.2} us/call, fft {:.2} us/call, speedup {:.1}x (sink {sink:.1})",
            td.as_micros() as f64 / n as f64, tf.as_micros() as f64 / n as f64,
            td.as_secs_f64() / tf.as_secs_f64());
    }

    #[test]
    fn eddy_drive_some_reproduces_the_bvec_bval_product() {
        // Criterion 4. The expected values are this slice's first eight k-space coefficients as
        // produced by the pre-change bvec/bval path (bvec [0.6, 0.8, 0.0], bval 1000) at tag
        // `p0-move-complete` plus the Task 7 rename. Bit equality, not a tolerance: the drive is
        // the same product the old code formed internally, so the arithmetic must be the same.
        // A nonzero check would pass with eddy ignored entirely. One reference per build shape:
        // the `kspace` feature swaps the x-stage to rustfft, which differs in the last bits
        // from the std twiddle-table sum, before and after this change alike.
        const EXPECTED_BITS_STD: [(u64, u64); 8] = [(0xbf85ec1cb1daa8d4, 0x3fa5ffff06e41c35), (0xbf3ac84a01a68680, 0x3f817f28896f61be), (0xbf83c1e814334026, 0xbfb080ac7ad7ff65), (0xbf7548454640584a, 0xbf8d9e9dd3358a7c), (0x3fb6e06347299105, 0x3fc30724fd363632), (0x3fc9ee837d7455c5, 0x3fcc89859a43e2be), (0x3fc5f2c7a85a66e2, 0x3fc03bf44552eb93), (0x3f8f6e71db6b4c1c, 0x3f7da1ff8d601054)];
        const EXPECTED_BITS_FFT: [(u64, u64); 8] = [(0xbf85ec1cb1daa8b8, 0x3fa5ffff06e41c36), (0xbf3ac84a01a68879, 0x3f817f28896f61c4), (0xbf83c1e81433403f, 0xbfb080ac7ad7ff64), (0xbf75484546405882, 0xbf8d9e9dd3358a91), (0x3fb6e06347299108, 0x3fc30724fd363630), (0x3fc9ee837d7455c7, 0x3fcc89859a43e2ba), (0x3fc5f2c7a85a66e4, 0x3fc03bf44552eb91), (0x3f8f6e71db6b4c24, 0x3f7da1ff8d60105a)];
        let expected = if cfg!(feature = "kspace") { EXPECTED_BITS_FFT } else { EXPECTED_BITS_STD };
        let (nx, ny) = (16, 16);
        let comps = [box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap = vec![0.0f32; nx * ny];
        let acq = Acquisition { eddy_strength: 0.05, ..Default::default() };
        let (bval, bvec) = (1000.0f64, [0.6f64, 0.8, 0.0]);
        let inp = SliceInput {
            compartments: &comp_refs, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
            sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
            eddy_drive: Some([bvec[0] * bval, bvec[1] * bval, bvec[2] * bval]),
            prep_drive: None, slice_seed: 7, eddy_lin: None,
        };
        let got = simulate_slice_kspace(&inp, &acq);
        for (i, (re, im)) in got.iter().take(8).enumerate() {
            assert_eq!((re.to_bits(), im.to_bits()), expected[i], "coefficient {i} moved");
        }
    }

    #[test]
    fn eddy_drive_none_disables_but_zero_vector_does_not() {
        // Criterion 5: None and Some([0,0,0]) are DIFFERENT inputs. The FSL b0 row
        // (bval=5, bvec=[0,0,0]) is Some([0,0,0]) and must keep eddy "on" with a zero gradient,
        // because that is what disables the NUFFT path today.
        let (nx, ny) = (16, 16);
        let comps = [box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap = vec![3.0f32; nx * ny];
        let acq = Acquisition { eddy_strength: 0.05, ..Default::default() };

        let make = |drive| SliceInput {
            compartments: &comp_refs, t2: &[T2Slice::Uniform(100.0)], t_inhom: None, fmap: &fmap, phase0: None,
            sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
            eddy_drive: drive, prep_drive: None, slice_seed: 7, eddy_lin: None,
        };
        let none = simulate_slice(&make(None), &acq);
        let zero = simulate_slice(&make(Some([0.0; 3])), &acq);

        // Identical numerically: a zero gradient multiplies by identity rotors.
        for ((a, b), (c, d)) in none.iter().zip(zero.iter()) {
            assert!((a - c).abs() < 1e-6 && (b - d).abs() < 1e-6,
                    "zero-vector drive should match None numerically on this slice");
        }
        // But they must not be collapsed into the same input. The eligibility decision lives in
        // `eddy_enabled` so it can be pinned here: `is_some()`, never "is the vector zero".
        assert!(eddy_enabled(&acq, Some([0.0; 3])), "a zero-vector drive must keep eddy enabled");
        assert!(!eddy_enabled(&acq, None), "None must disable eddy");
        assert!(!eddy_enabled(&Acquisition::default(), Some([1.0; 3])), "eddy_strength 0 disables");
    }

    /// Change 1's first map test: a Map filled with one constant reproduces Uniform of that
    /// constant. L2 relative over the coefficients, not coefficient-wise: a near-zero coefficient
    /// makes a relative bound meaningless. Not bit-exact either: under `kspace` the Uniform run
    /// takes the NUFFT path and the Map run the rotor path.
    /// A single-slice `SliceInput` for the map-path tests. A closure cannot return a struct that
    /// borrows its own argument, so this is a fn with the lifetime spelled out.
    fn mk_slice<'a>(
        comps: &'a [&'a [f32]], t2: &'a [T2Slice<'a>], t_inhom: Option<&'a [T2Slice<'a>]>,
        fmap: &'a [f32], nx: usize, ny: usize, seed: u64,
    ) -> SliceInput<'a> {
        SliceInput {
            compartments: comps, t2, t_inhom, fmap, phase0: None,
            sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
            eddy_drive: None, prep_drive: None, slice_seed: seed, eddy_lin: None,
        }
    }

    #[test]
    fn constant_map_reproduces_uniform() {
        let (nx, ny) = (16, 16);
        let comps = [box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap = vec![0.0f32; nx * ny];
        let acq = Acquisition::default();
        let constant = 100.0f32;
        let map = vec![constant; nx * ny];

        let t2u = [T2Slice::Uniform(constant)];
        let t2m = [T2Slice::Map(&map)];
        let u = simulate_slice_kspace(&mk_slice(&comp_refs, &t2u, None, &fmap, nx, ny, 3), &acq);
        let m = simulate_slice_kspace(&mk_slice(&comp_refs, &t2m, None, &fmap, nx, ny, 3), &acq);

        let (mut num, mut den) = (0.0f64, 0.0f64);
        for ((ur, ui), (mr, mi)) in u.iter().zip(m.iter()) {
            num += (ur - mr).powi(2) + (ui - mi).powi(2);
            den += ur.powi(2) + ui.powi(2);
        }
        let l2_rel = (num / den.max(1e-300)).sqrt();
        assert!(l2_rel < 1e-12, "constant map vs uniform: L2 relative {l2_rel:e}");
    }

    /// Change 1's second map test: the restructured forward against the literal per-line sum
    /// with a varying T2 map and a varying T2' map.
    #[test]
    fn varying_t2_map_matches_the_literal_sum() {
        let (nx, ny) = (12, 12);
        let comps = [box_hires(nx, ny, 3.0, 9.0, 3.0, 9.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap: Vec<f32> = (0..nx * ny).map(|i| 3.0 * ((i % 5) as f32 - 2.0)).collect();
        let map: Vec<f32> = (0..nx * ny).map(|i| 60.0 + (i % 7) as f32 * 10.0).collect();
        let tmap: Vec<f32> = (0..nx * ny).map(|i| 30.0 + (i % 4) as f32 * 15.0).collect();
        let acq = Acquisition::default();
        let inp = SliceInput {
            compartments: &comp_refs, t2: &[T2Slice::Map(&map)], t_inhom: Some(&[T2Slice::Map(&tmap)]),
            fmap: &fmap, phase0: None, sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
            eddy_drive: None, prep_drive: None, slice_seed: 5, eddy_lin: None,
        };
        let got = simulate_slice_kspace(&inp, &acq);
        // (inp, acq, coil, ncoils) -> Vec<C>; single coil here.
        let want = reference_coil_kspace(&inp, &acq, 0, 1);
        let peak = want.iter().map(|k| k.abs()).fold(0.0, f64::max);
        assert!(peak > 0.0);
        for ((gr, gi), w) in got.iter().zip(want.iter()) {
            assert!((gr - w.re).abs() < 1e-10 * peak && (gi - w.im).abs() < 1e-10 * peak,
                    "restructured forward diverged from the literal sum");
        }
    }

    /// Gradient echo (P5 part A) with per-voxel T2 and T2' maps against the literal sum.
    #[test]
    fn gradient_echo_maps_match_the_literal_sum() {
        let (nx, ny) = (12, 12);
        let comps = [box_hires(nx, ny, 3.0, 9.0, 3.0, 9.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap: Vec<f32> = (0..nx * ny).map(|i| 3.0 * ((i % 5) as f32 - 2.0)).collect();
        let map: Vec<f32> = (0..nx * ny).map(|i| 60.0 + (i % 7) as f32 * 10.0).collect();
        let tmap: Vec<f32> = (0..nx * ny).map(|i| 30.0 + (i % 4) as f32 * 15.0).collect();
        let acq = Acquisition { echo: EchoFormation::Gradient, ..Acquisition::default() };
        let inp = SliceInput {
            compartments: &comp_refs, t2: &[T2Slice::Map(&map)], t_inhom: Some(&[T2Slice::Map(&tmap)]),
            fmap: &fmap, phase0: None, sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
            eddy_drive: None, prep_drive: None, slice_seed: 5, eddy_lin: None,
        };
        let got = simulate_slice_kspace(&inp, &acq);
        let want = reference_coil_kspace(&inp, &acq, 0, 1);
        let peak = want.iter().map(|k| k.abs()).fold(0.0, f64::max);
        assert!(peak > 0.0);
        for ((gr, gi), w) in got.iter().zip(want.iter()) {
            assert!((gr - w.re).abs() < 1e-10 * peak && (gi - w.im).abs() < 1e-10 * peak);
        }
    }

    /// The gradient echo's two departures from the spin echo, each isolated on the acquired
    /// k-space without assuming when the centre line is read: with a uniform fieldmap and no
    /// decay, every sample is the spin echo's times `exp(i 2 pi f TE)` (the distortion term is
    /// shared, whatever the line times); with decay and no fieldmap, moving TE scales every line
    /// by `exp(-dTE (1/T2 + 1/T2'))`.
    #[test]
    fn gradient_echo_adds_the_static_phase_and_decays_from_the_rf() {
        let (nx, ny) = (16, 16);
        let comps = [box_hires(2 * nx, 2 * ny, 9.0, 23.0, 7.0, 21.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let phase0: Vec<f64> = (0..4 * nx * ny).map(|i| 0.3 * ((i % 11) as f64 - 5.0)).collect();
        let f = 20.0f32;
        let fmap = vec![f; 4 * nx * ny];
        let t2 = [T2Slice::Uniform(80.0)];
        let zero = vec![0.0f32; 4 * nx * ny];
        let with_fmap = SliceInput {
            compartments: &comp_refs, t2: &t2, t_inhom: None, fmap: &fmap, phase0: Some(&phase0), sim: [2 * nx, 2 * ny],
            acq_matrix: [nx, ny], z: 0, nz: 1, eddy_drive: None, prep_drive: None, slice_seed: 1, eddy_lin: None,
        };
        let without = SliceInput { fmap: &zero, ..with_fmap };
        let base = Acquisition { do_relaxation: false, t_echo: 40.0, t_line: 0.5, ..Acquisition::default() };
        let spin = simulate_slice_kspace(&with_fmap, &base);
        let grad = simulate_slice_kspace(&with_fmap, &Acquisition { echo: EchoFormation::Gradient, ..base.clone() });
        let rot = TAU * f as f64 * 0.040;
        let (c, s) = (rot.cos(), rot.sin());
        let peak = spin.iter().map(|(a, b)| a.hypot(*b)).fold(0.0, f64::max);
        assert!(peak > 0.0);
        for ((sr, si), (gr, gi)) in spin.iter().zip(&grad) {
            let (wr, wi) = (sr * c - si * s, sr * s + si * c);
            assert!((gr - wr).hypot(gi - wi) <= 1e-12 * peak, "static phase: {gr},{gi} vs {wr},{wi}");
        }
        // decay from the RF: T2 80 ms, T2' (the acquisition scalar) 50 ms
        let decaying = Acquisition { echo: EchoFormation::Gradient, do_relaxation: true, t_inhom: 50.0, ..base.clone() };
        let k1 = simulate_slice_kspace(&without, &decaying);
        let k2 = simulate_slice_kspace(&without, &Acquisition { t_echo: 70.0, ..decaying.clone() });
        let factor = (-30.0 * (1.0 / 80.0 + 1.0 / 50.0f64)).exp();
        let peak = k1.iter().map(|(a, b)| a.hypot(*b)).fold(0.0, f64::max);
        for ((ar, ai), (br, bi)) in k1.iter().zip(&k2) {
            assert!((br - factor * ar).hypot(bi - factor * ai) <= 1e-12 * peak, "decay from the RF");
        }
        // the spin echo does not decay with T2' as the echo moves: the same TE shift scales by T2 only
        let spin_decaying = Acquisition { echo: EchoFormation::Spin, ..decaying.clone() };
        let s1 = simulate_slice_kspace(&without, &spin_decaying);
        let s2 = simulate_slice_kspace(&without, &Acquisition { t_echo: 70.0, ..spin_decaying.clone() });
        let t2_only = (-30.0f64 / 80.0).exp();
        for ((ar, ai), (br, bi)) in s1.iter().zip(&s2) {
            assert!((br - t2_only * ar).hypot(bi - t2_only * ai) <= 1e-12 * peak, "spin echo decay");
        }
    }

    /// The static gradient-echo term is B0 only: the replayed eddy shear shares `rate` but is a
    /// readout-gradient effect, so with no fieldmap and no decay the gradient echo is the spin
    /// echo exactly, shear and all.
    #[test]
    fn gradient_echo_static_phase_excludes_the_eddy_shear() {
        let (nx, ny) = (16, 16);
        let comps = [box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap = vec![0.0f32; nx * ny];
        let t2 = [T2Slice::Uniform(80.0)];
        let inp = SliceInput {
            compartments: &comp_refs, t2: &t2, t_inhom: None, fmap: &fmap, phase0: None, sim: [nx, ny],
            acq_matrix: [nx, ny], z: 2, nz: 5, eddy_drive: None, prep_drive: None, slice_seed: 1,
            eddy_lin: Some([0.05, -0.03, 0.02]),
        };
        let spin = Acquisition { do_relaxation: false, ..Acquisition::default() };
        let a = simulate_slice_kspace(&inp, &spin);
        let b = simulate_slice_kspace(&inp, &Acquisition { echo: EchoFormation::Gradient, ..spin.clone() });
        assert_eq!(a, b);
    }

    /// The NUFFT path (uniform relaxation) and the rotor path (a constant map) agree under the
    /// gradient echo, static phase and decay from the RF included.
    #[test]
    fn gradient_echo_nufft_and_rotor_paths_agree() {
        let (nx, ny) = (16, 16);
        let comps = [box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap: Vec<f32> = (0..nx * ny).map(|i| 15.0 + 5.0 * ((i % 7) as f32 - 3.0)).collect();
        let acq = Acquisition { echo: EchoFormation::Gradient, ..Acquisition::default() };
        let map = vec![100.0f32; nx * ny];
        let u = simulate_slice_kspace(&mk_slice(&comp_refs, &[T2Slice::Uniform(100.0)], None, &fmap, nx, ny, 3), &acq);
        let m = simulate_slice_kspace(&mk_slice(&comp_refs, &[T2Slice::Map(&map)], None, &fmap, nx, ny, 3), &acq);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for ((ur, ui), (mr, mi)) in u.iter().zip(m.iter()) {
            num += (ur - mr).powi(2) + (ui - mi).powi(2);
            den += ur.powi(2) + ui.powi(2);
        }
        let l2_rel = (num / den.max(1e-300)).sqrt();
        assert!(l2_rel < 1e-12, "gradient echo, NUFFT vs rotor: L2 relative {l2_rel:e}");
    }

    /// A readout that starts before the excitation. TRXScan never hits this at TE 88 ms; ASL
    /// will, at TE near 12 ms with a 64-line readout.
    #[test]
    fn negative_trf_on_an_acquired_line_is_rejected() {
        let acq = Acquisition { t_echo: 2.0, t_line: 1.0, partial_fourier: 1.0, ..Default::default() };
        let err = validate_acquisition_timing(&acq, 64, 64).unwrap_err();
        assert!(err.contains("t_echo"), "the error must name t_echo: {err}");
        // Partial Fourier removes the LATE lines, so it cannot rescue this.
        let pf = Acquisition { partial_fourier: 0.5, ..acq.clone() };
        assert!(validate_acquisition_timing(&pf, 64, 64).is_err());
        // The default (TE 90 ms, 1 ms/line) is fine at 64 lines.
        assert!(validate_acquisition_timing(&Acquisition::default(), 64, 64).is_ok());
    }

    #[test]
    fn zero_in_a_t2_map_is_rejected() {
        assert!(validate_t2_map(&[100.0, 0.0, 50.0]).is_err());
        assert!(validate_t2_map(&[100.0, f32::INFINITY, 50.0]).is_ok());
        assert!(validate_t2_map(&[100.0, f32::NAN, 50.0]).is_err());
        assert!(validate_t2_map(&[100.0, -1.0, 50.0]).is_err());
    }

    /// A per-compartment Uniform T2' must be honoured on the fast (uniform) path, not only in the
    /// map branch: class mode passes Uniform per label and TRXScan's gate cannot see this.
    #[test]
    fn uniform_t_inhom_overrides_the_acquisition_scalar() {
        let (nx, ny) = (16, 16);
        let comps = [box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap = vec![0.0f32; nx * ny];
        let t2 = [T2Slice::Uniform(100.0)];
        let ti = [T2Slice::Uniform(20.0)];
        let a50 = Acquisition { t_inhom: 50.0, ..Default::default() };
        let a20 = Acquisition { t_inhom: 20.0, ..Default::default() };
        let base = simulate_slice_kspace(&mk_slice(&comp_refs, &t2, None, &fmap, nx, ny, 3), &a50);
        let over = simulate_slice_kspace(&mk_slice(&comp_refs, &t2, Some(&ti), &fmap, nx, ny, 3), &a50);
        let direct = simulate_slice_kspace(&mk_slice(&comp_refs, &t2, None, &fmap, nx, ny, 3), &a20);
        assert!(base.iter().zip(&over).any(|(p, q)| p != q), "the override must change the output");
        for (p, q) in over.iter().zip(&direct) {
            assert_eq!(p.0.to_bits(), q.0.to_bits());
            assert_eq!(p.1.to_bits(), q.1.to_bits());
        }
    }

    #[test]
    fn per_volume_slice_lengths_are_validated_before_the_loop() {
        let dims = [4usize, 4, 1];
        let images = vec![vec![0.0f32; 16 * 2]];
        let t2 = [T2Volume::Uniform(100.0)];
        let fmap = vec![0.0f32; 16];
        let acq = Acquisition::default();
        let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
        // The panic MESSAGE is asserted, not the panic: an out-of-bounds `tr[g]` already
        // panicked before any validation existed, so `is_err()` would prove nothing.
        fn message(r: std::thread::Result<(Vec<f32>, Vec<f32>)>) -> String {
            let e = r.expect_err("must panic");
            e.downcast_ref::<String>().cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        }
        // Two volumes, but only one drive entry.
        let r = std::panic::catch_unwind(|| {
            simulate_acquisition_oversampled(
                dims, dims, 2, &images, &t2, &fmap, None, &acq,
                &[None], &[None, None], &phase, 1, None, None)
        });
        assert!(message(r).contains("eddy_drive has 1 entries for 2 volumes"));

        let r = std::panic::catch_unwind(|| {
            simulate_acquisition_oversampled(
                dims, dims, 2, &images, &t2, &fmap, None, &acq,
                &[None, None], &[None], &phase, 1, None, None)
        });
        assert!(message(r).contains("prep_drive has 1 entries for 2 volumes"));

        let r = std::panic::catch_unwind(|| {
            simulate_acquisition_oversampled(
                dims, dims, 2, &images, &t2, &fmap, None, &acq,
                &[None, None], &[None, None], &phase, 1, None, Some(&[[0.0; 3]]))
        });
        assert!(message(r).contains("eddy_trace has 1 entries for 2 volumes"));
    }
}

/// `simulate_acquisition_echoes` (P6 addendum, part C), tested through the wrapper's outputs.
#[cfg(test)]
mod echo_tests {
    use super::*;

    const N: usize = 32; // acquired matrix
    const O: usize = 2; // oversampling
    const NZ: usize = 2;
    const NV: usize = 2;

    fn dims() -> ([usize; 3], [usize; 3]) {
        ([N * O, N * O, NZ], [N, N, NZ])
    }

    /// A smooth disc with a bright off-centre spot, the same in every volume, plus a second
    /// compartment at half weight.
    fn images() -> Vec<Vec<f32>> {
        let ([sx, sy, nz], _) = dims();
        let mut a = vec![0.0f32; sx * sy * nz * NV];
        for z in 0..nz {
            for y in 0..sy {
                for x in 0..sx {
                    let (dx, dy) = (x as f64 - sx as f64 / 2.0, y as f64 - sy as f64 / 2.0);
                    let r = (dx * dx + dy * dy).sqrt() / (0.35 * sx as f64);
                    let spot = (-((dx - 8.0).powi(2) + (dy + 5.0).powi(2)) / 30.0).exp();
                    let v = if r < 1.0 { 1.0 - 0.5 * r * r } else { 0.0 } + spot + 0.1 * z as f64;
                    for g in 0..NV {
                        a[(x + sx * (y + sy * z)) * NV + g] = v as f32;
                    }
                }
            }
        }
        let b = a.iter().map(|v| 0.5 * v).collect();
        vec![a, b]
    }

    fn fmap(zero: bool) -> Vec<f32> {
        let ([sx, sy, nz], _) = dims();
        (0..sx * sy * nz).map(|i| if zero { 0.0 } else { 3.0 * ((i % sx) as f32 / sx as f32 - 0.5) }).collect()
    }

    fn t2() -> Vec<T2Volume<'static>> {
        vec![T2Volume::Uniform(80.0), T2Volume::Uniform(160.0)]
    }

    fn base() -> Acquisition {
        Acquisition { t_echo: 30.0, t_line: 0.5, signal_scale: 1.0, ..Acquisition::default() }
    }

    type Drives = (Vec<Option<[f64; 3]>>, Vec<Option<(f64, [f64; 3])>>);

    fn drives() -> Drives {
        (vec![None; NV], vec![Some((800.0, [1.0, 0.3, 0.0])); NV])
    }

    /// One call of the single-echo entry.
    fn single(acq: &Acquisition, imgs: &[Vec<f32>], fm: &[f32], seed: u64, sigma: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let (s, a) = dims();
        let (eddy, prep) = drives();
        simulate_acquisition_oversampled(s, a, NV, imgs, &t2(), fm, None, acq, &eddy, &prep, &PhaseModel::hbcd_like(),
                                         seed, sigma, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn echoes(acq: &Acquisition, imgs: &[Vec<f32>], fm: &[f32], tes: &[f64], seed: u64, sigma: Option<&[f32]>,
              rsalt: Option<fn(usize) -> u64>, xsalt: fn(usize) -> u64) -> Vec<(Vec<f32>, Vec<f32>)> {
        let (s, a) = dims();
        let (eddy, prep) = drives();
        let per: Vec<&[Vec<f32>]> = tes.iter().map(|_| imgs).collect();
        match rsalt {
            // the public entry unless a test overrides a salt
            None if xsalt(1) == 0 => simulate_acquisition_echoes(s, a, NV, &per, &t2(), fm, None, acq, tes, &eddy, &prep,
                                                                 &PhaseModel::hbcd_like(), seed, sigma, None),
            r => echoes_with_salt(s, a, NV, &per, &t2(), fm, None, acq, tes, &eddy, &prep, &PhaseModel::hbcd_like(),
                                  seed, sigma, None, r.unwrap_or(echo_salt), xsalt),
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn complex(m: &[f32], p: &[f32]) -> Vec<(f64, f64)> {
        m.iter().zip(p).map(|(&m, &p)| (m as f64 * (p as f64).cos(), m as f64 * (p as f64).sin())).collect()
    }

    /// The noise residual (noise-on minus noise-off) as interleaved real and imaginary parts.
    fn residual(on: &(Vec<f32>, Vec<f32>), off: &(Vec<f32>, Vec<f32>)) -> Vec<f64> {
        complex(&on.0, &on.1).iter().zip(complex(&off.0, &off.1)).flat_map(|(a, b)| [a.0 - b.0, a.1 - b.1]).collect()
    }

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            ab += (x - ma) * (y - mb);
            aa += (x - ma) * (x - ma);
            bb += (y - mb) * (y - mb);
        }
        ab / (aa * bb).sqrt()
    }

    fn zero(_: usize) -> u64 {
        0
    }

    #[test]
    fn one_echo_is_the_single_echo_entry_bit_for_bit() {
        // k-space noise, image-space noise, spikes, a fieldmap and a prepared phase model with a
        // drive on every volume: everything the two seeds feed
        let acq = Acquisition { noise_variance: 0.01, n_spikes: 2, spike_amplitude: 0.5, ..base() };
        let imgs = images();
        let fm = fmap(false);
        let sigma = vec![0.05f32; N * N * NZ];
        let want = single(&acq, &imgs, &fm, 11, Some(&sigma));
        let got = echoes(&acq, &imgs, &fm, &[acq.t_echo], 11, Some(&sigma), None, zero);
        assert_eq!(got.len(), 1);
        assert_eq!(bits(&got[0].0), bits(&want.0));
        assert_eq!(bits(&got[0].1), bits(&want.1));
        // and the same holds for gradient echo, whose static fieldmap phase depends on TE
        let ge = Acquisition { echo: EchoFormation::Gradient, ..acq };
        let want = single(&ge, &imgs, &fm, 11, Some(&sigma));
        let got = echoes(&ge, &imgs, &fm, &[ge.t_echo], 11, Some(&sigma), None, zero);
        assert_eq!(bits(&got[0].0), bits(&want.0));
        assert_eq!(bits(&got[0].1), bits(&want.1));
    }

    #[test]
    fn noise_off_each_echo_is_the_single_entry_at_its_echo_time() {
        let acq = base();
        let (imgs, fm) = (images(), fmap(false));
        for echo in [EchoFormation::Spin, EchoFormation::Gradient] {
            let acq = Acquisition { echo, ..acq.clone() };
            let got = echoes(&acq, &imgs, &fm, &[30.0, 55.0], 4, None, None, zero);
            for (e, te) in [30.0, 55.0].into_iter().enumerate() {
                let want = single(&Acquisition { t_echo: te, ..acq.clone() }, &imgs, &fm, 4, None);
                assert_eq!(bits(&got[e].0), bits(&want.0), "{echo:?} echo {e} magnitude");
                assert_eq!(bits(&got[e].1), bits(&want.1), "{echo:?} echo {e} phase");
            }
            // the echoes differ (the decay and, for gradient echo, the fieldmap phase are applied)
            assert_ne!(bits(&got[0].0), bits(&got[1].0));
        }
    }

    /// Each receiver channel alone: the two echoes' noise residuals are uncorrelated with the
    /// echo salt and identical without it (equal echo times, the negative control).
    #[test]
    fn echoes_draw_independent_receiver_noise() {
        let (imgs, fm) = (images(), fmap(false));
        let sigma = vec![0.05f32; N * N * NZ];
        let kspace = Acquisition { noise_variance: 0.0025, ..base() };
        let image = base();
        for (name, on, sig) in [("k-space", &kspace, None), ("image-space", &image, Some(sigma.as_slice()))] {
            let tes = [30.0, 55.0];
            let noisy = echoes(on, &imgs, &fm, &tes, 9, sig, None, zero);
            let clean = echoes(&base(), &imgs, &fm, &tes, 9, None, None, zero);
            let (r0, r1) = (residual(&noisy[0], &clean[0]), residual(&noisy[1], &clean[1]));
            let n = r0.len() as f64;
            assert!(r0.iter().any(|v| v.abs() > 1e-3), "{name}: no noise in echo 0");
            let r = corr(&r0, &r1);
            assert!(r.abs() < 3.0 / n.sqrt(), "{name}: echo residuals correlated, r = {r}");
            // negative control: equal echo times, the receiver salt forced to zero: the same noise
            let same = echoes(on, &imgs, &fm, &[30.0, 30.0], 9, sig, Some(zero), zero);
            let clean_same = echoes(&base(), &imgs, &fm, &[30.0, 30.0], 9, None, Some(zero), zero);
            assert_eq!(bits(&same[0].0), bits(&same[1].0), "{name}: unsalted echoes differ");
            let r = corr(&residual(&same[0], &clean_same[0]), &residual(&same[1], &clean_same[1]));
            assert!(r > 0.999, "{name}: unsalted residuals not identical, r = {r}");
            // and with the salt at equal echo times the noise is independent: the salt does it
            let salted = echoes(on, &imgs, &fm, &[30.0, 30.0], 9, sig, Some(echo_salt), zero);
            let r = corr(&residual(&salted[0], &clean_same[0]), &residual(&salted[1], &clean_same[1]));
            assert!(r.abs() < 3.0 / n.sqrt(), "{name}: salted equal-TE residuals correlated, r = {r}");
        }
    }

    /// The echoes of a volume share the excitation's shot phase: with one uniform compartment, no
    /// fieldmap and spin echo, echo 1 is echo 0 times exp(-dTE/T2), phase and all. Salting the
    /// excitation seed per echo (the negative control) breaks it.
    #[test]
    fn echoes_share_the_shot_phase() {
        let one = vec![images().remove(0)];
        let fm = fmap(true);
        let acq = base();
        let (s, a) = dims();
        let (eddy, prep) = drives();
        let run = |xsalt: fn(usize) -> u64| {
            echoes_with_salt(s, a, NV, &[&one, &one], &[T2Volume::Uniform(80.0)], &fm, None, &acq, &[30.0, 55.0],
                             &eddy, &prep, &PhaseModel::hbcd_like(), 21, None, None, echo_salt, xsalt)
        };
        let worst = |out: &[(Vec<f32>, Vec<f32>)]| {
            let (c0, c1) = (complex(&out[0].0, &out[0].1), complex(&out[1].0, &out[1].1));
            let f = (-25.0f64 / 80.0).exp();
            let peak = c0.iter().map(|c| c.0.hypot(c.1)).fold(0.0, f64::max);
            c0.iter().zip(&c1).map(|(a, b)| (b.0 - f * a.0).hypot(b.1 - f * a.1)).fold(0.0, f64::max) / peak
        };
        let shared = worst(&run(zero));
        // f32 magnitude and phase: half an ulp each, at phases up to pi (ulp 2^-22)
        let tol = 4.0 * 2f64.powi(-24);
        assert!(shared < tol, "shared shot phase: worst relative error {shared:e} >= {tol:e}");
        let salted = worst(&run(echo_salt));
        assert!(salted > 1e3 * tol, "salting the excitation seed should change the shot phase: {salted:e}");
    }

    #[test]
    #[should_panic(expected = "echo times must increase strictly")]
    fn echo_times_must_increase() {
        let (imgs, fm) = (images(), fmap(true));
        echoes(&base(), &imgs, &fm, &[30.0, 30.0], 1, None, None, zero);
    }
}

/// Scanner partial Fourier (TRXScan re-sync, Task 2), ported from TRXScan `main`.
#[cfg(test)]
mod scanner_pf_tests {
    use super::*;
    use crate::readout::Readout;

    /// The lines in acquisition order, from the train itself.
    fn order(nx: usize, ny: usize, acq: &Acquisition) -> Vec<usize> {
        let epi = SingleShotEpi { kx_max: nx, ky_max: ny, t_line: acq.t_line, t_echo: acq.t_echo, reverse_phase: acq.reverse_phase };
        (0..nx * ny).map(|tick| epi.kspace_index(tick)).filter(|&(kx, _)| kx == nx / 2).map(|(_, ky)| ky).collect()
    }

    fn kept(nx: usize, ny: usize, acq: &Acquisition) -> Vec<usize> {
        let m = sampling_mask(nx, ny, acq);
        (0..ny).filter(|&ky| m[nx * ky]).collect()
    }

    /// TRXScan main's `scanner_partial_fourier_skips_the_first_lines_and_reaches_the_centre_sooner`,
    /// the order from the train and the timing from `LineTiming::for_acquisition`: exactly
    /// `round(ny*pf)` contiguous lines with the centre, the skipped ones the FIRST acquired, the
    /// eddy clock starting at the first acquired line (the centre reached `skip` lines sooner, the
    /// echo-relative times unchanged); Contiguous keeps the same count, drops the LAST lines and
    /// leaves the timing alone.
    #[test]
    fn scanner_partial_fourier_skips_the_first_lines_and_reaches_the_centre_sooner() {
        let (nx, ny, pf) = (16usize, 32usize, 0.75f64);
        for reverse in [false, true] {
            let acq = Acquisition { partial_fourier: pf, pf_mode: PartialFourierMode::Scanner, reverse_phase: reverse,
                                    t_line: 2.0, ..Default::default() };
            let full = Acquisition { partial_fourier: 1.0, ..acq.clone() };
            let k = kept(nx, ny, &acq);
            assert_eq!(k.len(), (ny as f64 * pf).round() as usize);
            assert!(k.windows(2).all(|w| w[1] == w[0] + 1), "kept lines are contiguous");
            assert!(k.contains(&(ny / 2)), "the centre line is kept");
            let ord = order(nx, ny, &acq);
            let skip = pf_skipped_lines(ny, &acq);
            assert_eq!(skip, ny / 4);
            assert!(ord[..skip].iter().all(|ky| !k.contains(ky)), "first {skip} lines skipped: {:?}", &ord[..skip]);
            assert!(ord[skip..].iter().all(|ky| k.contains(ky)));
            let (tp, tf) = (LineTiming::for_acquisition(&acq, nx, ny), LineTiming::for_acquisition(&full, nx, ny));
            assert_eq!((&tp.t_ms, &tp.trf_ms, &tp.polarity), (&tf.t_ms, &tf.trf_ms, &tf.polarity));
            let c = ny / 2;
            assert!((tp.tread_ms[c] - (tf.tread_ms[c] - skip as f64 * 2.0)).abs() < 1e-9);
            let first = ord[skip];
            assert!(tp.tread_ms[first] > 0.0 && tp.tread_ms[first] < 2.0, "the first acquired line reads at the start of its own train");
            let contiguous = Acquisition { pf_mode: PartialFourierMode::Contiguous, ..acq.clone() };
            let kc = kept(nx, ny, &contiguous);
            assert_eq!(kc.len(), k.len());
            assert!(ord[ny - skip..].iter().all(|ky| !kc.contains(ky)), "Contiguous drops the last lines");
            assert_eq!(LineTiming::for_acquisition(&contiguous, nx, ny), tf);
            assert_eq!(pf_skipped_lines(ny, &contiguous), 0);
            // the fall-through trap: Scanner has its own mask, not Fiberfox's
            let fiberfox = Acquisition { pf_mode: PartialFourierMode::FiberfoxCompatible, ..acq.clone() };
            assert_ne!(sampling_mask(nx, ny, &fiberfox), sampling_mask(nx, ny, &acq));
            // at full Fourier every mode is the same mask and timing
            for mode in [PartialFourierMode::FiberfoxCompatible, PartialFourierMode::Contiguous, PartialFourierMode::Scanner] {
                let a = Acquisition { pf_mode: mode, ..full.clone() };
                assert_eq!(sampling_mask(nx, ny, &a), sampling_mask(nx, ny, &full), "{mode:?}");
                assert_eq!(LineTiming::for_acquisition(&a, nx, ny), tf, "{mode:?}");
                assert_eq!(pf_skipped_lines(ny, &a), 0);
            }
        }
    }

    /// The timing check reads Scanner's mask: an echo too early for the first lines of the full
    /// train passes once Scanner skips them, and still fails under Contiguous and Fiberfox, which
    /// keep those lines.
    #[test]
    fn scanner_partial_fourier_relaxes_the_timing_check() {
        let (nx, ny) = (16usize, 32usize);
        let early = Acquisition { partial_fourier: 0.75, pf_mode: PartialFourierMode::Scanner, t_line: 2.0, t_echo: 20.0,
                                  ..Default::default() };
        assert!(validate_acquisition_timing(&Acquisition { partial_fourier: 1.0, ..early.clone() }, nx, ny).is_err());
        validate_acquisition_timing(&early, nx, ny).unwrap();
        for mode in [PartialFourierMode::Contiguous, PartialFourierMode::FiberfoxCompatible] {
            let e = validate_acquisition_timing(&Acquisition { pf_mode: mode, ..early.clone() }, nx, ny).unwrap_err();
            assert!(e.contains("Scanner"), "{e}");
        }
    }
}

/// EPI timing and the per-slice capture (TRXScan re-sync, Task 3), ported from TRXScan `main` at
/// the slice level; the acquisition-level capture tests come with the complex entry point.
#[cfg(test)]
mod capture_tests {
    use super::*;

    /// One slice, two compartments at o = 2, every in-plane artifact on (noise, partial Fourier,
    /// ghosting, eddy, a spike, four coils, GRAPPA).
    fn artifact_slice() -> (Vec<Vec<f32>>, Vec<f32>, Acquisition) {
        let (snx, sny) = (32usize, 32usize);
        let a = box_hires(snx, sny, 6.0, 20.0, 8.0, 24.0);
        let b: Vec<f32> = box_hires(snx, sny, 12.0, 26.0, 4.0, 18.0).iter().map(|v| 0.5 * v).collect();
        let fmap: Vec<f32> = (0..snx * sny).map(|i| 3.0 * (((i % snx) as f64 - 16.0) / 16.0) as f32).collect();
        let acq = Acquisition {
            signal_scale: 1.0, noise_variance: 1e-4, partial_fourier: 0.75, pf_mode: PartialFourierMode::Contiguous,
            ghost_offset: 0.01, eddy_strength: 0.01, eddy_quad: 0.002, eddy_phase: 2e-5, n_spikes: 1,
            spike_amplitude: 0.2, n_coils: 4, accel: 2, acs_lines: 8, seed: 11, ..Default::default()
        };
        (vec![a, b], fmap, acq)
    }

    fn input<'a>(comps: &'a [&'a [f32]], t2: &'a [T2Slice<'a>], fmap: &'a [f32]) -> SliceInput<'a> {
        SliceInput {
            compartments: comps, t2, t_inhom: None, fmap, phase0: None, sim: [32, 32], acq_matrix: [16, 16], z: 1, nz: 3,
            eddy_drive: Some([0.0, 600.0, 800.0]), prep_drive: None, slice_seed: 17, eddy_lin: None,
        }
    }

    /// Capturing changes no arithmetic: the combined image is the same with every capture on, and
    /// `simulate_slice` is its `f32` cast.
    #[test]
    fn capturing_changes_no_arithmetic() {
        let (imgs, fmap, acq) = artifact_slice();
        let comps: Vec<&[f32]> = imgs.iter().map(|v| v.as_slice()).collect();
        let t2 = [T2Slice::Uniform(80.0), T2Slice::Uniform(60.0)];
        let inp = input(&comps, &t2, &fmap);
        let plain = simulate_slice_full(&inp, &acq, SliceCapture::default());
        let all = simulate_slice_full(&inp, &acq, SliceCapture::ALL);
        assert_eq!(plain.combined, all.combined);
        assert!(plain.acquired.is_none() && plain.reconstructed.is_none() && plain.coil_images.is_none());
        let s = simulate_slice(&inp, &acq);
        assert!(s.iter().zip(&plain.combined).all(|(a, c)| a.0 == c[0] as f32 && a.1 == c[1] as f32));
        assert_eq!((all.nx, all.ny, all.n_coils), (16, 16, 4));
        assert_eq!(all.acquired.as_ref().unwrap().len(), 4);
    }

    /// The captured reconstructed k-space inverts to the captured coil images, which combine with
    /// the captured sensitivities to the combined image.
    #[test]
    fn captured_reconstructed_kspace_inverts_to_combined() {
        let (imgs, fmap, acq) = artifact_slice();
        let comps: Vec<&[f32]> = imgs.iter().map(|v| v.as_slice()).collect();
        let t2 = [T2Slice::Uniform(80.0), T2Slice::Uniform(60.0)];
        let r = simulate_slice_full(&input(&comps, &t2, &fmap), &acq, SliceCapture::ALL);
        let (nx, ny) = (r.nx, r.ny);
        let (rec, cimg) = (r.reconstructed.as_ref().unwrap(), r.coil_images.as_ref().unwrap());
        let mut wsum = vec![C::ZERO; nx * ny];
        let mut ssum = vec![0.0f64; nx * ny];
        for coil in 0..r.n_coils {
            let ks: Vec<C> = rec[coil].iter().map(|v| C { re: v[0], im: v[1] }).collect();
            let img = inverse_2d(&ks, nx, ny, nx / 2, ny / 2);
            for i in 0..nx * ny {
                let ci = cimg[coil][i];
                assert!((img[i].re - ci[0]).abs() < 1e-9 && (img[i].im - ci[1]).abs() < 1e-9, "coil image mismatch");
                let sv = r.sensitivities[coil][i];
                wsum[i] = wsum[i].add(img[i].scale(sv));
                ssum[i] += sv * sv;
            }
        }
        for i in 0..nx * ny {
            let c = r.combined[i];
            let (re, im) = (wsum[i].re / ssum[i].max(1e-12), wsum[i].im / ssum[i].max(1e-12));
            assert!((re - c[0]).abs() < 1e-9 && (im - c[1]).abs() < 1e-9, "combined mismatch at {i}");
        }
    }

    /// The captured acquired k-space is pre-GRAPPA (zero off the mask), the mask is the sampling
    /// mask, and GRAPPA fills un-acquired samples in the reconstructed k-space.
    #[test]
    fn acquired_kspace_is_pre_grappa_and_mask_is_the_sampling_mask() {
        let (imgs, fmap, acq) = artifact_slice();
        let comps: Vec<&[f32]> = imgs.iter().map(|v| v.as_slice()).collect();
        let t2 = [T2Slice::Uniform(80.0), T2Slice::Uniform(60.0)];
        let r = simulate_slice_full(&input(&comps, &t2, &fmap), &acq, SliceCapture::ALL);
        assert_eq!(r.mask, sampling_mask(r.nx, r.ny, &acq));
        let (a, rec) = (r.acquired.as_ref().unwrap(), r.reconstructed.as_ref().unwrap());
        let mut filled = 0;
        for coil in 0..r.n_coils {
            for (i, &m) in r.mask.iter().enumerate() {
                if !m {
                    assert_eq!(a[coil][i], [0.0, 0.0], "un-acquired sample must be zero in `acquired`");
                    if rec[coil][i] != [0.0, 0.0] {
                        filled += 1;
                    }
                }
            }
        }
        assert!(filled > 0, "GRAPPA must have synthesised un-acquired lines in `reconstructed`");
    }

    /// TRXScan main's `epi_timing_agrees_with_line_times_and_the_trajectory`, against
    /// `LineTiming::for_acquisition`, and with Scanner partial Fourier's clock.
    #[test]
    fn epi_timing_agrees_with_line_times_and_the_trajectory() {
        for reverse in [false, true] {
            for pf in [1.0, 0.75] {
                let acq = Acquisition { t_line: 0.8, t_echo: 75.0, reverse_phase: reverse, partial_fourier: pf,
                                        pf_mode: PartialFourierMode::Scanner, ..Default::default() };
                let (nx, ny) = (12usize, 10usize);
                let t = epi_timing(nx, ny, &acq);
                let lt = LineTiming::for_acquisition(&acq, nx, ny);
                assert_eq!((&t.t_ms, &t.t_rf_ms, &t.t_read_ms), (&lt.t_ms, &lt.trf_ms, &lt.tread_ms));
                let mut seen = t.order.clone();
                seen.sort_unstable();
                assert_eq!(seen, (0..ny).collect::<Vec<_>>(), "order must visit every line once");
                for w in t.order.windows(2) {
                    assert!(t.t_read_ms[w[1]] > t.t_read_ms[w[0]], "readout time must increase along the acquisition order");
                }
                for ky in 0..ny {
                    assert!((t.t_rf_ms[ky] - (t.t_echo + t.t_ms[ky])).abs() < 1e-12);
                }
                assert!((t.dt - 0.8 / nx as f64).abs() < 1e-15);
                let traj = epi_trajectory(nx, ny, &acq);
                assert_eq!(traj.len(), nx * ny);
                assert!(traj.windows(2).all(|w| w[1].2 > w[0].2));
                assert_eq!(traj[0].1, t.order[0]);
            }
        }
    }
}

/// The complex entry point and its options (TRXScan re-sync, Task 4), ported from TRXScan `main`
/// with the diffusion drives explicit: eddy drive `bvec * bval`, prep drive `(bval, bvec)`.
#[cfg(test)]
mod complex_tests {
    use super::*;

    const EDDY: [Option<[f64; 3]>; 2] = [Some([0.0, 0.0, 0.0]), Some([600.0, 800.0, 0.0])];
    const PREP: [Option<(f64, [f64; 3])>; 2] = [Some((0.0, [0.0, 0.0, 0.0])), Some((1000.0, [0.6, 0.8, 0.0]))];

    /// Images, fieldmap, acquisition, simulation and acquired dimensions.
    type Fixture = (Vec<Vec<f32>>, Vec<f32>, Acquisition, [usize; 3], [usize; 3]);

    /// TRXScan main's `artifact_fixture`: 3 slices, 2 compartments at o = 2, every in-plane artifact
    /// on, distinct per slice; two volumes.
    fn artifact_fixture() -> Fixture {
        let (nx, ny, nz, o) = (16usize, 16usize, 3usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let ngrad = 2;
        let mut fiber = vec![0.0f32; snx * sny * nz * ngrad];
        let mut gm = vec![0.0f32; snx * sny * nz * ngrad];
        let mut fmap = vec![0.0f32; snx * sny * nz];
        for z in 0..nz {
            let sh = z as f64 * 3.0;
            let a = box_hires(snx, sny, 6.0 + sh, 20.0 + sh, 8.0, 24.0 - sh);
            let b = box_hires(snx, sny, 12.0, 26.0, 4.0 + sh, 18.0);
            for y in 0..sny {
                for x in 0..snx {
                    let i = x + snx * y;
                    let vox = x + snx * (y + sny * z);
                    for g in 0..ngrad {
                        let att = if g == 0 { 1.0 } else { 0.6 };
                        fiber[vox * ngrad + g] = a[i] * att;
                        gm[vox * ngrad + g] = 0.5 * b[i] * att;
                    }
                    fmap[vox] = 3.0 * ((x as f64 - 16.0) / 16.0) as f32 + z as f32;
                }
            }
        }
        let acq = Acquisition {
            signal_scale: 1.0, noise_variance: 1e-4, partial_fourier: 0.75, pf_mode: PartialFourierMode::Contiguous,
            ghost_offset: 0.01, eddy_strength: 0.01, eddy_quad: 0.002, eddy_phase: 2e-5, n_spikes: 1,
            spike_amplitude: 0.2, n_coils: 4, accel: 2, acs_lines: 8, seed: 11, ..Default::default()
        };
        (vec![fiber, gm], fmap, acq, [snx, sny, nz], [nx, ny, nz])
    }

    const T2: [T2Volume<'static>; 2] = [T2Volume::Uniform(80.0), T2Volume::Uniform(60.0)];

    fn input<'a>(images: &'a [Vec<f32>], fmap: &'a [f32], phase: &'a PhaseModel, noise_sigma: Option<&'a [f32]>,
                 sim: [usize; 3], acq: [usize; 3]) -> AcquisitionInput<'a> {
        AcquisitionInput {
            sim_dims: sim, acq_dims: acq, n_volumes: 2, images, t2: &T2, fmap, t_inhom: None, eddy_drive: &EDDY,
            prep_drive: &PREP, phase, seed: 5, noise_sigma, eddy_trace: None,
        }
    }

    fn mag_phase(out: &AcquisitionOutput) -> (Vec<f32>, Vec<f32>) {
        let mag = out.re.iter().zip(&out.im).map(|(re, im)| (re * re + im * im).sqrt()).collect();
        let ph = out.re.iter().zip(&out.im).map(|(re, im)| im.atan2(*re)).collect();
        (mag, ph)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// TRXScan main's `simulate_acquisition_is_bit_identical_to_the_per_slice_loop`: the literal
    /// loop the acquisition used to be (`simulate_slice` per (g, z), the image-space noise keyed
    /// on the voxel, then `f32` magnitude and phase) against the complex entry point, sample for
    /// sample; and `simulate_acquisition_oversampled` is the complex output's magnitude and phase.
    #[test]
    fn simulate_acquisition_is_bit_identical_to_the_per_slice_loop() {
        let (images, fmap, acq, sim, acqd) = artifact_fixture();
        let [snx, sny, nz] = sim;
        let [nx, ny, _] = acqd;
        let o = snx / nx;
        let phase = PhaseModel::hbcd_like();
        let ns: Vec<f32> = (0..nx * ny * nz).map(|v| 0.01 * ((v % 7) as f32)).collect();
        let inp = input(&images, &fmap, &phase, Some(&ns), sim, acqd);
        let (mag, ph) = mag_phase(&simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions::default()));
        let (om, op) = simulate_acquisition_oversampled(sim, acqd, 2, &images, &T2, &fmap, None, &acq, &EDDY, &PREP, &phase,
                                                        5, Some(&ns), None);
        assert_eq!((bits(&mag), bits(&ph)), (bits(&om), bits(&op)));
        let ngrad = 2;
        for g in 0..ngrad {
            for z in 0..nz {
                let mut cs = vec![vec![0.0f32; snx * sny]; images.len()];
                let mut fs = vec![0.0f32; snx * sny];
                for y in 0..sny {
                    for x in 0..snx {
                        let vox = x + snx * (y + sny * z);
                        for (c, img) in images.iter().enumerate() {
                            cs[c][x + snx * y] = img[vox * ngrad + g];
                        }
                        fs[x + snx * y] = fmap[vox];
                    }
                }
                let refs: Vec<&[f32]> = cs.iter().map(|v| v.as_slice()).collect();
                let (pm, pd) = PREP[g].unwrap();
                let shot = phase.prep.as_ref().unwrap().shot(pm, pd, g, z, 5);
                let phi = phase_slice(&phase, &shot, snx, sny, o, z, nz);
                let slice_seed = (g as u64).wrapping_mul(0x100_0001).wrapping_add(z as u64).wrapping_mul(0x9E37) ^ 5;
                let t2s = [T2Slice::Uniform(80.0), T2Slice::Uniform(60.0)];
                let out = simulate_slice(
                    &SliceInput {
                        compartments: &refs, t2: &t2s, t_inhom: None, fmap: &fs, phase0: Some(&phi), sim: [snx, sny],
                        acq_matrix: [nx, ny], z, nz, eddy_drive: EDDY[g], prep_drive: PREP[g], slice_seed, eddy_lin: None,
                    },
                    &acq,
                );
                for y in 0..ny {
                    for x in 0..nx {
                        let (mut re, mut im) = out[x + nx * y];
                        let vox = x + nx * (y + ny * z);
                        let sd = ns[vox] as f64;
                        if sd > 0.0 {
                            let mut rng = Rng(5 ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                              ^ (vox as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1);
                            re += (rng.gauss() * sd) as f32;
                            im += (rng.gauss() * sd) as f32;
                        }
                        let i = vox * ngrad + g;
                        assert_eq!(mag[i].to_bits(), (re * re + im * im).sqrt().to_bits(), "mag at g={g} z={z} x={x} y={y}");
                        assert_eq!(ph[i].to_bits(), im.atan2(re).to_bits(), "phase at g={g} z={z} x={x} y={y}");
                    }
                }
            }
        }
    }

    /// TRXScan main's `a_single_slice_run_is_bit_identical_to_its_slice_of_the_full_run`: with
    /// `slice_z`/`nz_full` the eddy z-terms, object phase, per-slice seeds and the noise-map stream
    /// see the slice's full-FOV position; without them it is not the same slice.
    #[test]
    fn a_single_slice_run_is_bit_identical_to_its_slice_of_the_full_run() {
        let (images, fmap, acq, sim, acqd) = artifact_fixture();
        let [snx, sny, nz] = sim;
        let [nx, ny, _] = acqd;
        let phase = PhaseModel::hbcd_like();
        let ns: Vec<f32> = (0..nx * ny * nz).map(|v| 0.02 * ((v % 5) as f32)).collect();
        let full = simulate_acquisition_complex(&input(&images, &fmap, &phase, Some(&ns), sim, acqd), &acq,
                                                &AcquisitionOptions::default());
        let (zl, ngrad) = (1usize, 2usize);
        let sub_images: Vec<Vec<f32>> =
            images.iter().map(|img| img[snx * sny * zl * ngrad..snx * sny * (zl + 1) * ngrad].to_vec()).collect();
        let sub_fmap = fmap[snx * sny * zl..snx * sny * (zl + 1)].to_vec();
        let sub_ns = ns[nx * ny * zl..nx * ny * (zl + 1)].to_vec();
        let sub = input(&sub_images, &sub_fmap, &phase, Some(&sub_ns), [snx, sny, 1], [nx, ny, 1]);
        let z_of = [zl];
        let one = simulate_acquisition_complex(&sub, &acq,
                                               &AcquisitionOptions { slice_z: Some(&z_of), nz_full: Some(nz), ..Default::default() });
        for y in 0..ny {
            for x in 0..nx {
                for g in 0..ngrad {
                    let (vf, vs) = ((x + nx * (y + ny * zl)) * ngrad + g, (x + nx * y) * ngrad + g);
                    assert_eq!(full.re[vf].to_bits(), one.re[vs].to_bits(), "re at x={x} y={y} g={g}");
                    assert_eq!(full.im[vf].to_bits(), one.im[vs].to_bits(), "im at x={x} y={y} g={g}");
                }
            }
        }
        let naive = simulate_acquisition_complex(&sub, &acq, &AcquisitionOptions::default());
        assert_ne!(naive.re, one.re);
    }

    /// TRXScan main's `per_volume_te_equals_separate_runs`; and a per-volume echo time too early
    /// for the readout is refused, naming the volume.
    #[test]
    fn per_volume_te_equals_separate_runs() {
        let (images, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::hbcd_like();
        let inp = input(&images, &fmap, &phase, None, sim, acqd);
        let tes = [70.0, 110.0];
        let both = simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions { t_echo_per_volume: Some(&tes), ..Default::default() });
        let nvox = acqd[0] * acqd[1] * acqd[2];
        for (g, te) in tes.iter().enumerate() {
            let one = simulate_acquisition_complex(&inp, &Acquisition { t_echo: *te, ..acq.clone() }, &AcquisitionOptions::default());
            for vox in 0..nvox {
                assert_eq!(both.re[vox * 2 + g].to_bits(), one.re[vox * 2 + g].to_bits());
                assert_eq!(both.im[vox * 2 + g].to_bits(), one.im[vox * 2 + g].to_bits());
            }
        }
        let same = simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions::default());
        assert_ne!(same.re, both.re, "a different TE must change the relaxation weighting");
        let early = [70.0, 1.0];
        let r = std::panic::catch_unwind(|| {
            simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions { t_echo_per_volume: Some(&early), ..Default::default() })
        });
        let e = r.expect_err("a 1 ms echo");
        let text = e.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(text.contains("volume 1") && text.contains("before the excitation"), "{text}");
    }

    /// TRXScan main's `capturing_kspace_changes_no_arithmetic`: capture leaves the image alone, and
    /// `combined` for each captured slice is that slice's pre-noise image, in the requested order.
    #[test]
    fn capturing_kspace_changes_no_arithmetic() {
        let (images, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::hbcd_like();
        let inp = input(&images, &fmap, &phase, None, sim, acqd);
        let plain = simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions::default());
        let slices = [1usize, 0];
        let cap = simulate_acquisition_complex(&inp, &acq,
                                               &AcquisitionOptions { kspace_slices: Some(&slices), capture: SliceCapture::ALL, ..Default::default() });
        assert_eq!((bits(&plain.re), bits(&plain.im)), (bits(&cap.re), bits(&cap.im)));
        assert!(plain.kspace.is_none());
        let k = cap.kspace.expect("k-space requested");
        assert_eq!(k.slices, vec![1, 0]);
        let (nx, ny, ncoils, ngrad, nsl) = (k.nx, k.ny, k.n_coils, 2, 2);
        assert_eq!(k.combined.len(), ngrad * nsl * nx * ny);
        assert_eq!(k.acquired.as_ref().unwrap().len(), ngrad * nsl * ncoils * nx * ny);
        assert_eq!(k.sensitivities.len(), ncoils * nx * ny);
        for g in 0..ngrad {
            for (si, &zl) in k.slices.iter().enumerate() {
                for y in 0..ny {
                    for x in 0..nx {
                        let c = k.combined[((g * nsl + si) * ny + y) * nx + x];
                        let vox = x + nx * (y + ny * zl);
                        assert_eq!(c[0], plain.re[vox * ngrad + g]);
                        assert_eq!(c[1], plain.im[vox * ngrad + g]);
                    }
                }
            }
        }
    }

    /// TRXScan main's `captured_reconstructed_kspace_inverts_to_combined`, on the `f32` capture.
    #[test]
    fn captured_reconstructed_kspace_inverts_to_combined() {
        let (images, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::hbcd_like();
        let inp = input(&images, &fmap, &phase, None, sim, acqd);
        let slices = [2usize];
        let k = simulate_acquisition_complex(&inp, &acq,
                                             &AcquisitionOptions { kspace_slices: Some(&slices), capture: SliceCapture::ALL, ..Default::default() })
            .kspace.unwrap();
        let (nx, ny, nc) = (k.nx, k.ny, k.n_coils);
        let (rec, cimg) = (k.reconstructed.as_ref().unwrap(), k.coil_images.as_ref().unwrap());
        for g in 0..2 {
            let mut wsum = vec![C::ZERO; nx * ny];
            let mut ssum = vec![0.0f64; nx * ny];
            for coil in 0..nc {
                let base = (g * nc + coil) * nx * ny;
                let ks: Vec<C> = rec[base..base + nx * ny].iter().map(|v| C { re: v[0] as f64, im: v[1] as f64 }).collect();
                let img = inverse_2d(&ks, nx, ny, nx / 2, ny / 2);
                for i in 0..nx * ny {
                    let ci = cimg[base + i];
                    assert!((img[i].re - ci[0] as f64).abs() < 1e-4 && (img[i].im - ci[1] as f64).abs() < 1e-4, "coil image mismatch");
                    let s = k.sensitivities[coil * nx * ny + i] as f64;
                    wsum[i] = wsum[i].add(img[i].scale(s));
                    ssum[i] += s * s;
                }
            }
            for i in 0..nx * ny {
                let c = k.combined[g * nx * ny + i];
                let (re, im) = (wsum[i].re / ssum[i].max(1e-12), wsum[i].im / ssum[i].max(1e-12));
                assert!((re - c[0] as f64).abs() < 1e-4 && (im - c[1] as f64).abs() < 1e-4, "combined at g={g} i={i}");
            }
        }
    }

    /// TRXScan main's `acquired_kspace_is_pre_grappa_and_mask_is_the_sampling_mask`.
    #[test]
    fn acquired_kspace_is_pre_grappa_and_mask_is_the_sampling_mask() {
        let (images, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::none();
        let inp = input(&images, &fmap, &phase, None, sim, acqd);
        let slices = [0usize];
        let k = simulate_acquisition_complex(&inp, &acq,
                                             &AcquisitionOptions { kspace_slices: Some(&slices), capture: SliceCapture::ALL, ..Default::default() })
            .kspace.unwrap();
        assert_eq!(k.mask, sampling_mask(k.nx, k.ny, &acq));
        let (a, rec) = (k.acquired.as_ref().unwrap(), k.reconstructed.as_ref().unwrap());
        let mut filled = 0;
        for coil in 0..k.n_coils {
            for (j, &m) in k.mask.iter().enumerate() {
                let i = coil * k.nx * k.ny + j;
                if !m {
                    assert_eq!(a[i], [0.0, 0.0], "un-acquired sample must be zero in `acquired`");
                    if rec[i] != [0.0, 0.0] {
                        filled += 1;
                    }
                }
            }
        }
        assert!(filled > 0, "GRAPPA must have synthesised un-acquired lines in `reconstructed`");
    }

    /// The progress callback sees every volume once, with the total.
    #[test]
    fn progress_counts_every_volume() {
        let (images, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::none();
        let inp = input(&images, &fmap, &phase, None, sim, acqd);
        let seen = std::sync::Mutex::new(Vec::new());
        let cb = |done: usize, total: usize| seen.lock().unwrap().push((done, total));
        simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions { progress: Some(&cb), ..Default::default() });
        let mut v = seen.into_inner().unwrap();
        v.sort_unstable();
        assert_eq!(v, vec![(1, 2), (2, 2)]);
    }
}
