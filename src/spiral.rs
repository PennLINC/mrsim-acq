//! The stack-of-spirals forward (P5 addendum, part C, "Forward and reconstruction").
//!
//! Per slice, compartment and coil, in the Cartesian forward's coordinates, Fourier sign and
//! normalization: positions `(xc, yc)` centred and half-cell registered in acquired-voxel units,
//! frequencies `k_j` in cycles/FOV, the **positive** exponent with each component divided by its
//! matrix size, `1/nvox`:
//!
//! ```text
//! K(k_j) = (1/nvox) sum_r amp_q(r) exp(i phi0(r)) sum_c m_c(r) w_c(r, tau_j) exp(i 2 pi f(r) tau_j)
//!          exp(+i 2 pi (kx_j xc / nx + ky_j yc / ny))
//! ```
//!
//! with `w_c = exp(ln A(r) - tau/T2_c(r) - tau/T2'_c(r))` under relaxation (spiral-out starts at the
//! echo, so `tau >= 0` and the in-echo decay is monotonic) and the fieldmap only with distortions.
//! [`spiral_kspace_exact`] is the exact sum: the oracle, `O(samples x voxels)`, for tests.

use std::f64::consts::TAU;

use crate::kspace::{coil_sensitivity, Acquisition, EchoFormation, SliceInput, T2Slice, C};

/// The per-voxel static factors of the forward, formed as the Cartesian forward forms them
/// (`kspace::build_coil_kspace_timed`): `amp = signal_scale x` coil `coil`'s sensitivity on the
/// absolute acquired grid, `phi0` the object phase, the centred coordinates `(xc, yc)`, and the
/// off-resonance rate (Hz).
pub(crate) struct Statics {
    pub amp: Vec<f64>,
    pub phi0: Vec<f64>,
    pub xc: Vec<f64>,
    pub yc: Vec<f64>,
    pub rate: Vec<f64>,
}

pub(crate) fn statics(inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize) -> Statics {
    let [snx, sny] = inp.sim;
    let [nx, ny] = inp.acq_matrix;
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple of the acquired matrix");
    assert_eq!(acq.echo, EchoFormation::Spin, "the spiral train is a spin-echo train");
    assert!(inp.eddy_drive.is_none() && inp.eddy_lin.is_none(), "the eddy model is not available for spirals");
    let (ox, oy) = (snx / nx, sny / ny);
    let (sxs, sys) = (ox * (nx / 2), oy * (ny / 2));
    let (xoff, yoff) = ((ox as f64 - 1.0) / 2.0, (oy as f64 - 1.0) / 2.0);
    let n = snx * sny;
    let mut s = Statics { amp: vec![0.0; n], phi0: vec![0.0; n], xc: vec![0.0; n], yc: vec![0.0; n], rate: vec![0.0; n] };
    for y in 0..sny {
        for x in 0..snx {
            let i = x + snx * y;
            let (xa, ya) = ((x as f64 - xoff) / ox as f64, (y as f64 - yoff) / oy as f64);
            s.amp[i] = acq.signal_scale * coil_sensitivity(coil, ncoils, xa, ya, nx, ny);
            s.xc[i] = (x as f64 - sxs as f64 - xoff) / ox as f64;
            s.yc[i] = (y as f64 - sys as f64 - yoff) / oy as f64;
            s.phi0[i] = inp.phase0.map_or(0.0, |p| p[i]);
            s.rate[i] = if acq.do_distortions { inp.fmap[i] as f64 } else { 0.0 };
        }
    }
    s
}

/// `1/T2 + 1/T2'` (1/ms) of compartment `c` at voxel `i`.
pub(crate) fn decay_rate(inp: &SliceInput, acq: &Acquisition, c: usize, i: usize) -> f64 {
    let t2 = match inp.t2[c] { T2Slice::Uniform(v) => v as f64, T2Slice::Map(m) => m[i] as f64 };
    let tp = match inp.t_inhom.map(|ti| ti[c]) {
        None => acq.t_inhom,
        Some(T2Slice::Uniform(v)) => v as f64,
        Some(T2Slice::Map(m)) => m[i] as f64,
    };
    1.0 / t2 + 1.0 / tp
}

/// The decay of compartment `c` at voxel `i` and time `tau_ms` from the echo, with the optional
/// log-amplitude map, as one exponent.
pub(crate) fn decay(inp: &SliceInput, acq: &Acquisition, log_amp: Option<&[f64]>, c: usize, i: usize, tau_ms: f64) -> f64 {
    if !acq.do_relaxation {
        return 1.0;
    }
    (log_amp.map_or(0.0, |la| la[i]) - tau_ms * decay_rate(inp, acq, c, i)).exp()
}

/// The time-segmented forward's inputs for a one-compartment slice: the static image
/// `amp exp(i phi0) m (exp(ln A))` and the complex rates `(-d, 2 pi f)` (1/ms, rad/ms). With
/// `decay_inside` (`voxel` mode) `d = 1/T2 + 1/T2'` per voxel under relaxation and `log_amp` joins
/// the static amplitude; otherwise (`class` mode) `d = 0` and the decay is the caller's per-sample
/// weight.
pub(crate) fn segment_inputs(inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize, log_amp: Option<&[f64]>,
                             decay_inside: bool) -> (Vec<C>, Vec<(f64, f64)>) {
    assert_eq!(inp.compartments.len(), 1, "one compartment per segmented forward");
    let st = statics(inp, acq, coil, ncoils);
    let m = inp.compartments[0];
    let n = m.len();
    let mut x = vec![C::ZERO; n];
    let mut rate = vec![(0.0, 0.0); n];
    for i in 0..n {
        let mut a = st.amp[i] * m[i] as f64;
        let mut d = 0.0;
        if decay_inside && acq.do_relaxation {
            a *= log_amp.map_or(1.0, |la| la[i].exp());
            d = decay_rate(inp, acq, 0, i);
        }
        x[i] = C::cis(st.phi0[i]).scale(a);
        rate[i] = (-d, TAU * st.rate[i] / 1000.0);
    }
    (x, rate)
}

/// The exact-sum spiral forward of one slice and coil at frequencies `k` (cycles/FOV) sampled at
/// `tau_ms` from the echo (one per frequency). `log_amp` (per voxel on the simulation slice) is
/// added inside every compartment's decay exponent, as the 3D path's voxel mode does.
pub(crate) fn spiral_kspace_exact(
    inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize, k: &[[f64; 2]], tau_ms: &[f64],
    log_amp: Option<&[f64]>,
) -> Vec<C> {
    assert_eq!(k.len(), tau_ms.len(), "one time per sample");
    let [snx, sny] = inp.sim;
    let [nx, ny] = inp.acq_matrix;
    let nvox = snx * sny;
    if let Some(la) = log_amp {
        assert_eq!(la.len(), nvox, "log-amplitude map is not on the simulation slice");
    }
    let st = statics(inp, acq, coil, ncoils);
    let mut out = vec![C::ZERO; k.len()];
    for (j, (kj, &tau)) in k.iter().zip(tau_ms).enumerate() {
        let mut acc = C::ZERO;
        for i in 0..nvox {
            let mut w = 0.0;
            for (c, comp) in inp.compartments.iter().enumerate() {
                if comp[i] != 0.0 {
                    w += comp[i] as f64 * decay(inp, acq, log_amp, c, i, tau);
                }
            }
            if w == 0.0 {
                continue;
            }
            let ph = st.phi0[i] + TAU * (st.rate[i] * tau / 1000.0 + kj[0] * st.xc[i] / nx as f64 + kj[1] * st.yc[i] / ny as f64);
            let a = st.amp[i] * w;
            acc = acc.add(C { re: a * ph.cos(), im: a * ph.sin() });
        }
        out[j] = acc.scale(1.0 / nvox as f64);
    }
    out
}

/// The time-segmented type-2 NUFFT forward (feature `kspace`): `exp(rate(r) tau)` interpolated
/// over the plan's `L` segment times, so one slice's samples are `L` type-2 NUFFTs of
/// `x(r) exp(rate(r) tau_l)` combined per sample by `b_l(tau_j)`.
#[cfg(feature = "kspace")]
pub(crate) struct SegmentedForward {
    nufft: crate::nufft::Nufft2,
    snx: usize,
    sny: usize,
    /// The samples (cycles/FOV), all interleaves, and each one's sample-time index.
    k: Vec<[f64; 2]>,
    tau_idx: Vec<usize>,
    /// `exp(-i 2 pi (kx xoff / snx + ky yoff / sny))`: the half-cell registration the integer
    /// NUFFT modes leave out, times `1/nvox`.
    stat: Vec<C>,
    pub plan: crate::tseg::TsegPlan,
}

#[cfg(feature = "kspace")]
impl SegmentedForward {
    /// The forward for samples `k` whose sample times are `tau_ms[tau_idx[j]]`, on the simulation
    /// slice `sim` over the acquired matrix `acq_matrix`, certified over `rect`.
    pub(crate) fn new(k: &[[f64; 2]], tau_idx: &[usize], tau_ms: &[f64], t_ms: f64, sim: [usize; 2],
                      acq_matrix: [usize; 2], rect: crate::tseg::RateRect) -> Result<SegmentedForward, String> {
        let [snx, sny] = sim;
        let [nx, ny] = acq_matrix;
        let (ox, oy) = (snx / nx, sny / ny);
        let (sxs, sys) = (ox * (nx / 2), oy * (ny / 2));
        let (xoff, yoff) = ((ox as f64 - 1.0) / 2.0, (oy as f64 - 1.0) / 2.0);
        let plan = crate::tseg::plan(tau_ms, t_ms, rect)?;
        let nufft = crate::nufft::Nufft2::new([snx, sny], [-(sxs as i64), -(sys as i64)], [snx, sny], 1e-12);
        let ninv = 1.0 / (snx * sny) as f64;
        let stat = k.iter().map(|p| C::cis(-TAU * (p[0] * xoff / snx as f64 + p[1] * yoff / sny as f64)).scale(ninv)).collect();
        Ok(SegmentedForward { nufft, snx, sny, k: k.to_vec(), tau_idx: tau_idx.to_vec(), stat, plan })
    }

    /// The samples of the image `x` (simulation slice, `x + snx y`) under the per-voxel complex
    /// rates `rate = (-d, 2 pi f)` (1/ms, rad/ms).
    pub(crate) fn apply(&self, x: &[C], rate: &[(f64, f64)]) -> Vec<C> {
        let nvox = self.snx * self.sny;
        assert!(x.len() == nvox && rate.len() == nvox, "image and rates on the simulation slice");
        let l = self.plan.l;
        let mut out = vec![C::ZERO; self.k.len()];
        let mut seg = vec![(0.0, 0.0); nvox];
        for (i, &tl) in self.plan.tau_l.iter().enumerate() {
            for ((s, xv), &(re, im)) in seg.iter_mut().zip(x).zip(rate) {
                let e = C::cis(im * tl).scale((re * tl).exp());
                let v = xv.mul(e);
                *s = (v.re, v.im);
            }
            let ks = self.nufft.type2(&seg, &self.k, 1);
            for (j, (o, (re, im))) in out.iter_mut().zip(ks).enumerate() {
                let (br, bi) = self.plan.b[self.tau_idx[j] * l + i];
                *o = o.add(C { re: br * re - bi * im, im: br * im + bi * re });
            }
        }
        for (o, s) in out.iter_mut().zip(&self.stat) {
            *o = o.mul(*s);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kspace::{build_coil_kspace_timed, LineTiming};

    fn slice_input<'a>(comps: &'a [&'a [f32]], t2: &'a [T2Slice<'a>], fmap: &'a [f32], phase0: &'a [f64], sim: [usize; 2],
                       acq_matrix: [usize; 2]) -> SliceInput<'a> {
        SliceInput {
            compartments: comps, t2, t_inhom: None, fmap, phase0: Some(phase0), sim, acq_matrix, z: 0, nz: 1,
            eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None,
        }
    }

    #[test]
    fn exact_spiral_forward_at_cartesian_frequencies_is_the_cartesian_forward() {
        // a displaced complex point (and a second, weaker one), oversampling 1 and 2, two coils,
        // a signal_scale: the spiral forward evaluated at the Cartesian grid's frequencies
        // (kx - nx/2, ky - ny/2) equals the Cartesian forward, which fixes its sign and scale
        for o in [1usize, 2] {
            let (nx, ny) = (12usize, 12usize);
            let (snx, sny) = (nx * o, ny * o);
            let mut img = vec![0.0f32; snx * sny];
            img[(3 * o + 1) + snx * (8 * o)] = 1.0;
            img[(9 * o) + snx * (2 * o + o / 2)] = 0.37;
            let phase0: Vec<f64> = (0..snx * sny).map(|i| 0.3 + 0.01 * i as f64).collect();
            let fmap = vec![0.0f32; snx * sny];
            let comps = [img.as_slice()];
            let t2 = [T2Slice::Uniform(f32::INFINITY)];
            let inp = slice_input(&comps, &t2, &fmap, &phase0, [snx, sny], [nx, ny]);
            let acq = Acquisition { do_relaxation: false, noise_variance: 0.0, n_spikes: 0, signal_scale: 2.5, n_coils: 2,
                                    ..Acquisition::default() };
            let ks: Vec<[f64; 2]> = (0..ny).flat_map(|ky| (0..nx).map(move |kx| [kx as f64 - (nx / 2) as f64, ky as f64 - (ny / 2) as f64])).collect();
            let taus = vec![0.0; nx * ny];
            for q in 0..2 {
                let cart = build_coil_kspace_timed(&inp, &acq, q, 2, &LineTiming::for_acquisition(&acq, nx, ny), None);
                let sp = spiral_kspace_exact(&inp, &acq, q, 2, &ks, &taus, None);
                let peak = cart.iter().fold(0.0f64, |m, c| m.max(c.abs()));
                for (a, b) in cart.iter().zip(&sp) {
                    assert!((a.re - b.re).abs() <= 1e-12 * peak && (a.im - b.im).abs() <= 1e-12 * peak,
                            "o {o} coil {q}: ({}, {}) vs ({}, {})", a.re, a.im, b.re, b.im);
                }
            }
        }
    }

    #[test]
    fn exact_spiral_forward_decay_and_off_resonance() {
        // one voxel: the sample is amp exp(i phi0) exp(ln A - tau/T2 - tau/T2') exp(i 2 pi f tau)
        // at k = 0, with each factor switched by its flag
        let (nx, o) = (8usize, 1usize);
        let mut img = vec![0.0f32; nx * nx];
        img[3 + nx * 5] = 1.0;
        let phase0 = vec![0.4; nx * nx];
        let fmap = vec![40.0f32; nx * nx];
        let t2m = vec![80.0f32; nx * nx];
        let comps = [img.as_slice()];
        let t2 = [T2Slice::Map(&t2m)];
        let tim = vec![50.0f32; nx * nx];
        let ti = [T2Slice::Map(&tim)];
        let mut inp = slice_input(&comps, &t2, &fmap, &phase0, [nx * o, nx * o], [nx, nx]);
        inp.t_inhom = Some(&ti);
        let la = vec![-0.2; nx * nx];
        let acq = Acquisition { do_relaxation: true, do_distortions: true, noise_variance: 0.0, n_spikes: 0,
                                ..Acquisition::default() };
        let tau = 3.0;
        let k = spiral_kspace_exact(&inp, &acq, 0, 1, &[[0.0, 0.0]], &[tau], Some(&la))[0];
        let amp = acq.signal_scale * coil_sensitivity(0, 1, 3.0, 5.0, nx, nx) / (nx * nx) as f64;
        let mag = amp * (-0.2 - tau / 80.0 - tau / 50.0f64).exp();
        let ph = 0.4 + TAU * 40.0 * tau / 1000.0;
        assert!((k.re - mag * ph.cos()).abs() < 1e-15 && (k.im - mag * ph.sin()).abs() < 1e-15, "{} {}", k.re, k.im);
        let off = Acquisition { do_relaxation: false, do_distortions: false, ..acq };
        let k0 = spiral_kspace_exact(&inp, &off, 0, 1, &[[0.0, 0.0]], &[tau], Some(&la))[0];
        assert!((k0.re - amp * 0.4f64.cos()).abs() < 1e-15 && (k0.im - amp * 0.4f64.sin()).abs() < 1e-15);
    }

    /// A smooth test slice: Gaussian blobs, a 50 Hz fieldmap, independent T2, T2' and ln A maps and
    /// a nonconstant object phase, on `n o x n o`.
    #[cfg(feature = "kspace")]
    #[allow(clippy::type_complexity)]
    fn smooth_maps(n: usize, o: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>, Vec<f64>) {
        let s = n * o;
        let (mut img, mut fmap, mut t2, mut tp, mut ph, mut la) =
            (vec![0.0f32; s * s], vec![0.0f32; s * s], vec![0.0f32; s * s], vec![0.0f32; s * s], vec![0.0; s * s], vec![0.0; s * s]);
        for y in 0..s {
            for x in 0..s {
                let (u, v) = ((x as f64 + 0.5) / s as f64, (y as f64 + 0.5) / s as f64);
                let i = x + s * y;
                let g = |cx: f64, cy: f64, w: f64| (-((u - cx).powi(2) + (v - cy).powi(2)) / (2.0 * w * w)).exp();
                img[i] = (g(0.4, 0.45, 0.08) + 0.6 * g(0.62, 0.6, 0.06)) as f32;
                fmap[i] = (-20.0 + 50.0 * u * v) as f32;
                t2[i] = (40.0 + 80.0 * v) as f32;
                tp[i] = (30.0 + 60.0 * (1.0 - u)) as f32;
                ph[i] = 0.7 * u - 1.1 * v * v;
                la[i] = -0.1 - 0.3 * u;
            }
        }
        (img, fmap, t2, tp, ph, la)
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn segmented_forward_matches_the_exact_sum() {
        // a complete 32 x 32 trajectory at oversample 2: voxel mode (decay inside, independent maps)
        // and class mode (uniform relaxation as a per-sample weight outside), two coils, a
        // signal_scale, to 1e-6 of peak
        use crate::tseg::RateRect;
        let (n, o) = (32usize, 2usize);
        let s = n * o;
        let tr = crate::readout::spiral_trajectory(n, n, 4, 4.0, 0.008).unwrap();
        let ns = tr.n_samples();
        let tau_idx: Vec<usize> = (0..tr.k.len()).map(|j| j % ns).collect();
        let taus: Vec<f64> = tau_idx.iter().map(|&j| tr.tau_ms[j]).collect();
        let (img, fmap, t2m, tpm, ph, la) = smooth_maps(n, o);
        let comps = [img.as_slice()];
        let acq = Acquisition { do_relaxation: true, do_distortions: true, noise_variance: 0.0, n_spikes: 0, n_coils: 2,
                                signal_scale: 3.0, ..Acquisition::default() };
        // voxel
        let t2 = [T2Slice::Map(&t2m)];
        let ti = [T2Slice::Map(&tpm)];
        let mut inp = slice_input(&comps, &t2, &fmap, &ph, [s, s], [n, n]);
        inp.t_inhom = Some(&ti);
        let d: Vec<f64> = (0..s * s).map(|i| decay_rate(&inp, &acq, 0, i)).collect();
        let rect = RateRect { d: [d.iter().cloned().fold(f64::MAX, f64::min), d.iter().cloned().fold(0.0, f64::max)],
                              f: [fmap.iter().fold(f32::MAX, |a, &b| a.min(b)) as f64, fmap.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64] };
        let fwd = SegmentedForward::new(&tr.k, &tau_idx, &tr.tau_ms, 4.0, [s, s], [n, n], rect).unwrap();
        println!("voxel: L {} m {} B {:.2} bound {:.2e}", fwd.plan.l, fwd.plan.m, fwd.plan.b_sum, fwd.plan.bound);
        for q in 0..2 {
            let (x, r) = segment_inputs(&inp, &acq, q, 2, Some(&la), true);
            let a = fwd.apply(&x, &r);
            let b = spiral_kspace_exact(&inp, &acq, q, 2, &tr.k, &taus, Some(&la));
            let peak = b.iter().fold(0.0f64, |m, c| m.max(c.abs()));
            let err = a.iter().zip(&b).fold(0.0f64, |m, (p, q)| m.max((p.re - q.re).hypot(p.im - q.im)));
            println!("voxel coil {q}: max error {:.2e} of peak", err / peak);
            assert!(err <= 1e-6 * peak, "voxel coil {q}: {}", err / peak);
        }
        // class: uniform T2 and T2', the decay a per-sample weight
        let t2u = [T2Slice::Uniform(70.0)];
        let tiu = [T2Slice::Uniform(45.0)];
        let mut inp = slice_input(&comps, &t2u, &fmap, &ph, [s, s], [n, n]);
        inp.t_inhom = Some(&tiu);
        let fwd = SegmentedForward::new(&tr.k, &tau_idx, &tr.tau_ms, 4.0, [s, s], [n, n], RateRect { d: [0.0, 0.0], f: rect.f }).unwrap();
        println!("class: L {} m {} B {:.2} bound {:.2e}", fwd.plan.l, fwd.plan.m, fwd.plan.b_sum, fwd.plan.bound);
        for q in 0..2 {
            let (x, r) = segment_inputs(&inp, &acq, q, 2, None, false);
            let a: Vec<C> = fwd.apply(&x, &r).into_iter().zip(&taus).map(|(c, &t)| c.scale((-t / 70.0 - t / 45.0).exp())).collect();
            let b = spiral_kspace_exact(&inp, &acq, q, 2, &tr.k, &taus, None);
            let peak = b.iter().fold(0.0f64, |m, c| m.max(c.abs()));
            let err = a.iter().zip(&b).fold(0.0f64, |m, (p, q)| m.max((p.re - q.re).hypot(p.im - q.im)));
            println!("class coil {q}: max error {:.2e} of peak", err / peak);
            assert!(err <= 1e-6 * peak, "class coil {q}: {}", err / peak);
        }
    }

    /// The Task 12 feasibility benchmark (run once, numbers recorded in the plan's Measurements):
    /// one asl001-sized volume, 64 x 64 x 20 at oversample 2, six compartments, one coil, a 50 Hz
    /// fieldmap range and the phantom's decay range, class and voxel mode.
    #[cfg(feature = "kspace")]
    #[test]
    #[ignore]
    fn feasibility_benchmark_asl001() {
        use crate::tseg::RateRect;
        use std::time::Instant;
        let (n, o, nz, ncomp, etl) = (64usize, 2usize, 20usize, 6usize, 20usize);
        let s = n * o;
        let tr = crate::readout::spiral_trajectory(n, n, 8, 4.0, 0.004).unwrap();
        let ns = tr.n_samples();
        let idx: Vec<usize> = (0..tr.k.len()).map(|j| j % ns).collect();
        let (img, _, t2m, tpm, ph, la) = smooth_maps(n, o);
        let fmap: Vec<f32> = (0..s * s).map(|i| (-25.0 + 50.0 * ((i % s) as f64 / s as f64)) as f32).collect();
        let acq = Acquisition { do_relaxation: true, do_distortions: true, noise_variance: 0.0, n_spikes: 0, n_coils: 1,
                                signal_scale: 1.0, ..Acquisition::default() };
        let comps = [img.as_slice()];
        // class: uniform relaxation per compartment, the frequency interval only
        let t0 = Instant::now();
        let fwd = SegmentedForward::new(&tr.k, &idx, &tr.tau_ms, 4.0, [s, s], [n, n], RateRect { d: [0.0, 0.0], f: [-25.0, 25.0] }).unwrap();
        let cert_class = t0.elapsed();
        let t2u = [T2Slice::Uniform(80.0)];
        let tiu = [T2Slice::Uniform(50.0)];
        let mut inp = slice_input(&comps, &t2u, &fmap, &ph, [s, s], [n, n]);
        inp.t_inhom = Some(&tiu);
        let t1 = Instant::now();
        for _z in 0..nz {
            for _c in 0..ncomp {
                let (x, r) = segment_inputs(&inp, &acq, 0, 1, None, false);
                std::hint::black_box(fwd.apply(&x, &r));
            }
        }
        let fwd_class = t1.elapsed();
        println!("class: L {} m {} B {:.2} bound {:.2e}; certification {:.2?}; forward {:.2?} ({} slices x {} compartments)",
                 fwd.plan.l, fwd.plan.m, fwd.plan.b_sum, fwd.plan.bound, cert_class, fwd_class, nz, ncomp);
        // voxel: the decay-frequency rectangle, the forward once per echo
        let t2 = [T2Slice::Map(&t2m)];
        let ti = [T2Slice::Map(&tpm)];
        let mut inp = slice_input(&comps, &t2, &fmap, &ph, [s, s], [n, n]);
        inp.t_inhom = Some(&ti);
        let d: Vec<f64> = (0..s * s).map(|i| decay_rate(&inp, &acq, 0, i)).collect();
        let rect = RateRect { d: [d.iter().cloned().fold(f64::MAX, f64::min), d.iter().cloned().fold(0.0, f64::max)], f: [-25.0, 25.0] };
        let t0 = Instant::now();
        let fwd = SegmentedForward::new(&tr.k, &idx, &tr.tau_ms, 4.0, [s, s], [n, n], rect).unwrap();
        let cert_voxel = t0.elapsed();
        let t1 = Instant::now();
        let per = 20; // one echo's slices x compartments timed, scaled to the train
        for _ in 0..per {
            let (x, r) = segment_inputs(&inp, &acq, 0, 1, Some(&la), true);
            std::hint::black_box(fwd.apply(&x, &r));
        }
        let one = t1.elapsed() / per as u32;
        println!("voxel: rect {rect:?} L {} m {} B {:.2} bound {:.2e}; certification {:.2?}; one forward {:.2?}, volume {:.1?} \
                  ({} echoes x {} slices x {} compartments)", fwd.plan.l, fwd.plan.m, fwd.plan.b_sum, fwd.plan.bound, cert_voxel,
                 one, one * (etl * nz * ncomp) as u32, etl, nz, ncomp);
        // reconstruction: density weights and the eigenvalue bound once, then one least squares per partition
        let t0 = Instant::now();
        let g = crate::grid_recon::SpiralRecon::new(&tr.k, n, crate::kspace::KspaceWindow::None);
        let dcf = t0.elapsed();
        let dd = vec![C { re: 1.0, im: 0.0 }; tr.k.len()];
        let t1 = Instant::now();
        for _ in 0..nz {
            std::hint::black_box(g.image(&dd));
        }
        println!("reconstruction: setup (density weights, eigenvalue bound) {:.2?}, {} partitions {:.2?}", dcf, nz, t1.elapsed());
    }
}
