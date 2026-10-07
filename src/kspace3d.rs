//! The 3D echo-train acquisition (P5 addendum, part B, "The acquisition stage"; part D's
//! carriers).
//!
//! A segmented spin-echo train reads one kz partition per refocused echo. Each echo is a spin echo,
//! so within an echo the 2D spin-echo forward holds with `t` from that echo's centre; across echoes
//! only a per-echo amplitude changes. So, per volume and coil:
//!
//! 1. each compartment's 2D k-space of each slice, from the existing forward with the within-echo
//!    line timing and **no relaxation** (`class` compartments: every relaxation input uniform), or
//!    once per echo with the per-voxel decay `ln A_e(r) - t/T2(r) - |t|/T2'(r)` as one exponent
//!    (`voxel` compartments);
//! 2. a z-DFT per compartment, in the centred asymmetric convention the in-plane axes use;
//! 3. the weighted compartment sum, `W_c(p, ky) = exp(ln A_e(p)(c) - t/T2_c - |t|/T2'_c)` times
//!    the per-shot line weights (physiology, shot gains), each line taken from the images of the
//!    pose its shot had;
//! 4. spikes and noise per acquired 3D sample;
//!
//! then an inverse z-DFT (`1/nz`) and each partition through the 2D reconstruction
//! (`kspace::reconstruct_coils`).
//!
//! Each compartment's 2D k-space comes from its own call of the 2D forward rather than from a
//! per-compartment accumulator inside it, so the 2D path is untouched (its rotor setup is repeated
//! per compartment, a cost the 3D path accepts).

use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::epg::epg_cpmg;
use crate::kspace::{
    build_coil_kspace_timed, phase_slice, reconstruct_coils, sampling_mask, Acquisition, EchoFormation, LineTiming,
    Rng, SliceInput, T2Slice, T2Volume, C,
};
use crate::phase::{PhaseModel, ShotPhase};
use crate::kspace::echo_salt;
use crate::readout::{ge3d_lines, grase_lines, EchoTrain, ExcitationTrain, Ge3dReadout, Ge3dTable, Grase3dTable, Readout3d};

/// The salt of the 3D path's noise and spike streams ("3DREAD").
pub const SEED_SALT_3D: u64 = 0x3344_5245_4144;

/// Per-`(volume, shot, compartment)` scalars multiplying every line of that shot in that
/// compartment: the physiological factors and the shot gains (part D), composed by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct LineWeights {
    pub n_shots: usize,
    pub n_compartments: usize,
    /// `w[(g * n_shots + s) * n_compartments + c]`.
    pub w: Vec<f64>,
}

impl LineWeights {
    fn at(&self, g: usize, s: usize, c: usize) -> f64 {
        self.w[(g * self.n_shots + s) * self.n_compartments + c]
    }
}

/// The images one set of shots of one volume saw (part D, per-shot motion): per compartment, one
/// volume on the simulation grid (`x + snx*(y + sny*z)`).
#[derive(Debug, Clone, PartialEq)]
pub struct ShotSet {
    pub shots: Vec<usize>,
    pub images: Vec<Vec<f32>>,
}

/// How one compartment's decay enters: as a line weight (every relaxation input uniform), or per
/// voxel inside the forward, once per echo.
enum Mode {
    Class { ln_a: Vec<f64>, t2_ms: f64, t2p_ms: f64 },
    Voxel,
}

/// A slice of a per-compartment volume list.
fn slice_of<'a>(v: &'a T2Volume<'a>, z: usize, nplane: usize) -> T2Slice<'a> {
    match v {
        T2Volume::Uniform(s) => T2Slice::Uniform(*s),
        T2Volume::Map(m) => T2Slice::Map(&m[z * nplane..(z + 1) * nplane]),
    }
}

/// `exp(ln_a - t/T2 - |t|/T2')`, one exponent, with infinite times meaning no decay.
fn line_weight(ln_a: f64, t_ms: f64, t2_ms: f64, t2p_ms: f64) -> f64 {
    let d2 = if t2_ms.is_infinite() { 0.0 } else { t_ms / t2_ms };
    let dp = if t2p_ms.is_infinite() { 0.0 } else { t_ms.abs() / t2p_ms };
    (ln_a - d2 - dp).exp()
}

/// The centred z-DFT kernel `exp(+i 2 pi (p - nz/2)(z - nz/2) / nz)`.
fn zkernel(nz: usize) -> Vec<C> {
    let zs = (nz / 2) as f64;
    let mut k = vec![C::ZERO; nz * nz];
    for p in 0..nz {
        for z in 0..nz {
            let th = TAU * (p as f64 - zs) * (z as f64 - zs) / nz as f64;
            k[p * nz + z] = C { re: th.cos(), im: th.sin() };
        }
    }
    k
}

/// Everything the 3D forward needs, resolved once per call.
struct Plan<'a> {
    snx: usize,
    sny: usize,
    nz: usize,
    nx: usize,
    ny: usize,
    o: usize,
    ncomp: usize,
    acq: &'a Acquisition,
    table: Grase3dTable,
    within: LineTiming,
    modes: Vec<Mode>,
    /// Voxel-mode `ln A_e` maps per compartment per echo, on the simulation grid (`None` for class).
    ln_a_maps: Vec<Option<Vec<Vec<f64>>>>,
    t2: &'a [T2Volume<'a>],
    t_inhom: Option<&'a [T2Volume<'a>]>,
    fmap: &'a [f32],
    phase: &'a PhaseModel,
    zk: Vec<C>,
    mask: Vec<bool>,
}

impl Plan<'_> {
    /// The 3D k-space of every compartment from one set of images (`img(c, vox)`), coil `q`,
    /// weighted by `W_c(p, ky)` but not by the line weights: `out[c][(p * ny + ky) * nx + kx]`.
    fn compartments<F: Fn(usize, usize) -> f32>(&self, img: F, q: usize, ncoils: usize) -> Vec<Vec<C>> {
        let (snx, sny, nz, nx, ny) = (self.snx, self.sny, self.nz, self.nx, self.ny);
        let nplane = snx * sny;
        let norel = Acquisition { do_relaxation: false, noise_variance: 0.0, n_spikes: 0, ..self.acq.clone() };
        let rel = Acquisition { do_relaxation: true, noise_variance: 0.0, n_spikes: 0, ..self.acq.clone() };
        let zero_shot = ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] };
        let phis: Vec<Vec<f64>> = (0..nz).map(|z| phase_slice(self.phase, &zero_shot, snx, sny, self.o, z, nz)).collect();
        let mut out = vec![vec![C::ZERO; nz * ny * nx]; self.ncomp];
        for c in 0..self.ncomp {
            let plane = |z: usize| -> Vec<f32> { (0..nplane).map(|i| img(c, z * nplane + i)).collect() };
            let one = |z: usize, pl: &[f32], acq: &Acquisition, la: Option<&[f64]>| -> Vec<C> {
                let refs = [pl];
                let t2s = [slice_of(&self.t2[c], z, nplane)];
                let tis = self.t_inhom.map(|ti| [slice_of(&ti[c], z, nplane)]);
                let inp = SliceInput {
                    compartments: &refs, t2: &t2s, t_inhom: tis.as_ref().map(|a| a.as_slice()),
                    fmap: &self.fmap[z * nplane..(z + 1) * nplane], phase0: Some(&phis[z]), sim: [snx, sny],
                    acq_matrix: [nx, ny], z, nz, eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None,
                };
                build_coil_kspace_timed(&inp, acq, q, ncoils, &self.within, la)
            };
            match &self.modes[c] {
                Mode::Class { ln_a, t2_ms, t2p_ms } => {
                    let k2: Vec<Vec<C>> = (0..nz).map(|z| one(z, &plane(z), &norel, None)).collect();
                    for p in 0..nz {
                        for ky in 0..ny {
                            let line = self.table.line(p, ky);
                            let w = if self.acq.do_relaxation {
                                line_weight(ln_a[line.echo - 1], line.t_ms, *t2_ms, *t2p_ms)
                            } else {
                                1.0
                            };
                            for kx in 0..nx {
                                let mut acc = C::ZERO;
                                for (z, k2z) in k2.iter().enumerate() {
                                    acc = acc.add(k2z[ky * nx + kx].mul(self.zk[p * nz + z]));
                                }
                                out[c][(p * ny + ky) * nx + kx] = acc.scale(w);
                            }
                        }
                    }
                }
                Mode::Voxel => {
                    let maps = self.ln_a_maps[c].as_ref().expect("voxel compartments carry their maps");
                    let planes: Vec<Vec<f32>> = (0..nz).map(plane).collect();
                    for e in 1..=self.table.lines.iter().map(|l| l.echo).max().unwrap_or(0) {
                        let ps: Vec<usize> = (0..nz).filter(|&p| self.table.line(p, 0).echo == e).collect();
                        if ps.is_empty() {
                            continue;
                        }
                        let k2: Vec<Vec<C>> = (0..nz)
                            .map(|z| one(z, &planes[z], &rel, Some(&maps[e - 1][z * nplane..(z + 1) * nplane])))
                            .collect();
                        for &p in &ps {
                            for ky in 0..ny {
                                for kx in 0..nx {
                                    let mut acc = C::ZERO;
                                    for (z, k2z) in k2.iter().enumerate() {
                                        acc = acc.add(k2z[ky * nx + kx].mul(self.zk[p * nz + z]));
                                    }
                                    out[c][(p * ny + ky) * nx + kx] = acc;
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The acquired 3D k-space of volume `g`, coil `q`, before spikes and noise: the compartment
    /// sum with the line weights, each line from the images its shot saw.
    #[allow(clippy::too_many_arguments)]
    fn kspace(&self, images: &[Vec<f32>], n_volumes: usize, g: usize, q: usize, ncoils: usize,
              lw: Option<&LineWeights>, sets: &[ShotSet]) -> Vec<C> {
        let (nz, nx, ny) = (self.nz, self.nx, self.ny);
        let base = self.compartments(|c, v| images[c][v * n_volumes + g], q, ncoils);
        let moved: Vec<Vec<Vec<C>>> = sets.iter().map(|s| self.compartments(|c, v| s.images[c][v], q, ncoils)).collect();
        let mut k = vec![C::ZERO; nz * ny * nx];
        for p in 0..nz {
            for ky in 0..ny {
                let line = self.table.line(p, ky);
                let src = sets.iter().position(|s| s.shots.contains(&line.shot)).map_or(&base, |i| &moved[i]);
                for (c, kc) in src.iter().enumerate() {
                    let w = lw.map_or(1.0, |lw| lw.at(g, line.shot, c));
                    for kx in 0..nx {
                        let i = (p * ny + ky) * nx + kx;
                        k[i] = k[i].add(kc[i].scale(w));
                    }
                }
                if !self.mask[ky * nx] {
                    for kx in 0..nx {
                        k[(p * ny + ky) * nx + kx] = C::ZERO;
                    }
                }
            }
        }
        k
    }
}

/// One coil's acquired 3D k-space of volume `g` (layout `(p * ny + ky) * nx + kx`) to its 2D
/// k-spaces per slice: spikes and noise per acquired sample on the `(volume, partition)` streams of
/// `seed` (part B, "Seeds"), then the inverse z-DFT normalized by `1/nz`. Shared by the GRASE and
/// the gradient-echo trains (P7).
#[allow(clippy::too_many_arguments)]
fn partitions_to_slices(k: &mut [C], mask: &[bool], zk: &[C], dims: [usize; 3], g: usize, q: usize, seed: u64,
                        acq: &Acquisition, sigma: f64) -> Vec<Vec<C>> {
    let [nx, ny, nz] = dims;
    for p in 0..nz {
        let part = &mut k[p * ny * nx..(p + 1) * ny * nx];
        let pseed = (g as u64).wrapping_mul(0x100_0001).wrapping_add(p as u64).wrapping_mul(0x9E37) ^ seed ^ SEED_SALT_3D;
        if acq.n_spikes > 0 {
            let (mut peak, mut peak_mag) = (C::ZERO, 0.0);
            for c in part.iter() {
                if c.abs() > peak_mag {
                    peak_mag = c.abs();
                    peak = *c;
                }
            }
            let acquired: Vec<usize> = (0..nx * ny).filter(|&i| mask[i]).collect();
            for pick in spike_picks(pseed, q, &acquired, acq.n_spikes) {
                part[pick] = peak.scale(acq.spike_amplitude);
            }
        }
        if acq.noise_variance > 0.0 {
            let mut rng = Rng((pseed ^ (q as u64).wrapping_mul(0x9E37_79B9)) | 1);
            for (i, c) in part.iter_mut().enumerate() {
                if mask[i] {
                    c.re += rng.gauss() * sigma;
                    c.im += rng.gauss() * sigma;
                }
            }
        }
    }
    // inverse z-DFT, normalized by 1/nz: per slice z, a 2D k-space
    let mut slices = vec![vec![C::ZERO; ny * nx]; nz];
    for (z, sl) in slices.iter_mut().enumerate() {
        for p in 0..nz {
            let kc = zk[p * nz + z];
            let conj = C { re: kc.re, im: -kc.im };
            for i in 0..ny * nx {
                sl[i] = sl[i].add(k[p * ny * nx + i].mul(conj));
            }
        }
        for c in sl.iter_mut() {
            *c = c.scale(1.0 / nz as f64);
        }
    }
    slices
}

/// Every slice through the 2D coil reconstruction: `coil_parts[q][z]` to the complex volume,
/// `vox = x + nx*(y + ny*z)`.
fn reconstruct_slices(coil_parts: &[Vec<Vec<C>>], acq: &Acquisition, dims: [usize; 3]) -> Vec<(f64, f64)> {
    let [nx, ny, nz] = dims;
    let mut out = vec![(0.0, 0.0); nx * ny * nz];
    for z in 0..nz {
        let coils: Vec<Vec<C>> = coil_parts.iter().map(|cp| cp[z].clone()).collect();
        let img = reconstruct_coils(coils, acq, [nx, ny]);
        for (i, c) in img.iter().enumerate() {
            out[i + nx * ny * z] = (c.re, c.im);
        }
    }
    out
}

/// The acquired samples one partition's spikes overwrite, on the stream of `(volume, partition)`
/// (`pseed`) and coil `q` (part B, "Seeds").
fn spike_picks(pseed: u64, q: usize, acquired: &[usize], n_spikes: usize) -> Vec<usize> {
    let mut rng = Rng(pseed ^ 0xA5A5_1234_5678_9ABC ^ (q as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    (0..n_spikes).map(|_| acquired[(rng.next_u64() as usize) % acquired.len()]).collect()
}

/// The checks and resolution shared by the entry points; panics name the offending input.
#[allow(clippy::too_many_arguments)]
fn plan<'a>(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, images: &[Vec<f32>], t2: &'a [T2Volume<'a>],
    t1: Option<&'a [T2Volume<'a>]>, fmap: &'a [f32], t_inhom: Option<&'a [T2Volume<'a>]>, acq: &'a Acquisition,
    train: &EchoTrain, readout: &Readout3d, line_weights: Option<&LineWeights>, shot_images: Option<&[Vec<ShotSet>]>,
    phase: &'a PhaseModel,
) -> Plan<'a> {
    let [snx, sny, nz] = sim_dims;
    let [nx, ny, nzo] = acq_dims;
    assert_eq!(nz, nzo, "partition count must match; z is never oversampled");
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple of the acquired matrix");
    let o = snx / nx;
    assert_eq!(o, sny / ny, "oversampling must match on both axes");
    let nvox = snx * sny * nz;
    let ncomp = images.len();
    for im in images {
        assert_eq!(im.len(), nvox * n_volumes, "compartment image is not on the simulation grid");
    }
    assert_eq!(fmap.len(), nvox, "fieldmap is not on the simulation grid");
    assert_eq!(t2.len(), ncomp, "t2 has {} entries for {ncomp} compartments", t2.len());
    assert_eq!(acq.echo, EchoFormation::Spin, "the 3D echo trains are spin-echo trains");
    assert!(acq.eddy_strength == 0.0 && acq.eddy_quad == 0.0 && acq.eddy_phase == 0.0,
            "the eddy model is not available in 3D (an echo-dependent eddy evolution breaks the z factorization)");
    let Readout3d::Grase { reverse_phase, .. } = *readout else {
        panic!("the spiral path is not implemented yet (P5 milestone C, Task 13)");
    };
    assert_eq!(reverse_phase, acq.reverse_phase, "the readout's phase-encode sign must be the acquisition's");
    let table = grase_lines(train, readout, ny, nz).unwrap_or_else(|e| panic!("{e}"));
    let within = table.within_echo_timing();
    if let Some(lw) = line_weights {
        assert!(lw.n_shots == table.n_shots && lw.n_compartments == ncomp && lw.w.len() == n_volumes * table.n_shots * ncomp,
                "line weights are not (volume, shot, compartment)");
    }
    if let Some(si) = shot_images {
        assert_eq!(si.len(), n_volumes, "shot images per volume");
        for sets in si {
            for s in sets {
                assert!(s.images.len() == ncomp && s.images.iter().all(|i| i.len() == nvox), "shot images are not per compartment");
                assert!(s.shots.iter().all(|&x| x < table.n_shots), "a shot index beyond the train's {}", table.n_shots);
            }
        }
    }
    let (modes, ln_a_maps) = resolve_modes(t2, t1, t_inhom, acq, train, nvox);
    Plan {
        snx, sny, nz, nx, ny, o, ncomp, acq, table, within, modes, ln_a_maps, t2, t_inhom, fmap, phase,
        zk: zkernel(nz), mask: sampling_mask(nx, ny, acq),
    }
}

/// Each compartment's decay mode and the voxel-mode `ln A_e` maps (per echo, simulation grid),
/// shared by the GRASE and spiral paths; panics name the offending input.
#[allow(clippy::type_complexity)]
fn resolve_modes(t2: &[T2Volume], t1: Option<&[T2Volume]>, t_inhom: Option<&[T2Volume]>, acq: &Acquisition,
                 train: &EchoTrain, nvox: usize) -> (Vec<Mode>, Vec<Option<Vec<Vec<f64>>>>) {
    let ncomp = t2.len();
    let needs_t1 = acq.do_relaxation && train.refocusing_deg != 180.0;
    assert!(!needs_t1 || t1.is_some(), "refocusing below 180 degrees needs T1 (stimulated echoes)");
    if let Some(t1) = t1 {
        assert_eq!(t1.len(), ncomp, "t1 has {} entries for {ncomp} compartments", t1.len());
    }
    let t1_of = |c: usize| t1.map(|t| t[c]).unwrap_or(T2Volume::Uniform(f32::INFINITY));
    let ti_of = |c: usize| t_inhom.map(|t| t[c]).unwrap_or(T2Volume::Uniform(acq.t_inhom as f32));
    let etl = train.etl;
    let mut cache: HashMap<(u32, u32), Vec<f64>> = HashMap::new();
    let mut epg = |t1: f32, t2: f32| -> Vec<f64> {
        cache.entry((t1.to_bits(), t2.to_bits()))
            .or_insert_with(|| epg_cpmg(etl, train.esp_ms, train.refocusing_deg, t1 as f64, t2 as f64))
            .clone()
    };
    let mut modes = Vec::with_capacity(ncomp);
    let mut ln_a_maps = Vec::with_capacity(ncomp);
    for (c, &t2c) in t2.iter().enumerate() {
        match (t2c, t1_of(c), ti_of(c)) {
            (T2Volume::Uniform(a), T2Volume::Uniform(b), T2Volume::Uniform(p)) => {
                modes.push(Mode::Class { ln_a: epg(b, a), t2_ms: a as f64, t2p_ms: p as f64 });
                ln_a_maps.push(None);
            }
            _ if !acq.do_relaxation => {
                modes.push(Mode::Class { ln_a: vec![0.0; etl], t2_ms: f64::INFINITY, t2p_ms: f64::INFINITY });
                ln_a_maps.push(None);
            }
            (t2v, t1v, _) => {
                let at = |v: &T2Volume, i: usize| match v { T2Volume::Uniform(s) => *s, T2Volume::Map(m) => m[i] };
                let mut maps = vec![vec![0.0f64; nvox]; etl];
                for i in 0..nvox {
                    let la = epg(at(&t1v, i), at(&t2v, i));
                    for (m, &l) in maps.iter_mut().zip(&la) {
                        m[i] = l;
                    }
                }
                modes.push(Mode::Voxel);
                ln_a_maps.push(Some(maps));
            }
        }
    }
    (modes, ln_a_maps)
}

/// The acquired 3D k-space per coil of volume `g`, after the line weights and the sampling mask
/// and before spikes, noise and reconstruction: the observation point of the tests that assert on
/// acquired samples. Layout `(p * ny + ky) * nx + kx`, `f64`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn kspace_3d_observed(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, images: &[Vec<f32>], t2: &[T2Volume],
    t1: Option<&[T2Volume]>, fmap: &[f32], t_inhom: Option<&[T2Volume]>, acq: &Acquisition, train: &EchoTrain,
    readout: &Readout3d, line_weights: Option<&LineWeights>, shot_images: Option<&[Vec<ShotSet>]>, phase: &PhaseModel,
    g: usize,
) -> Vec<Vec<(f64, f64)>> {
    let pl = plan(sim_dims, acq_dims, n_volumes, images, t2, t1, fmap, t_inhom, acq, train, readout, line_weights,
                  shot_images, phase);
    let ncoils = acq.n_coils.max(1);
    let sets = shot_images.map_or(&[][..], |si| &si[g][..]);
    (0..ncoils)
        .map(|q| pl.kspace(images, n_volumes, g, q, ncoils, line_weights, sets).into_iter().map(|c| (c.re, c.im)).collect())
        .collect()
}

/// The reconstructed complex 3D image of every volume, `f64`, layout `(vox, g)` with
/// `vox = x + nx*(y + ny*z)`: [`simulate_acquisition_3d`] before its `f32` cast.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_acquisition_3d_complex(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, images: &[Vec<f32>], t2: &[T2Volume],
    t1: Option<&[T2Volume]>, fmap: &[f32], t_inhom: Option<&[T2Volume]>, acq: &Acquisition, train: &EchoTrain,
    readout: &Readout3d, line_weights: Option<&LineWeights>, shot_images: Option<&[Vec<ShotSet>]>, phase: &PhaseModel,
    seed: u64,
) -> Vec<Vec<(f64, f64)>> {
    if let Readout3d::Spiral { .. } = readout {
        #[cfg(feature = "kspace")]
        return spiral_volumes(sim_dims, acq_dims, n_volumes, images, t2, t1, fmap, t_inhom, acq, train, readout,
                              line_weights, shot_images, phase, seed);
        #[cfg(not(feature = "kspace"))]
        panic!("spiral readouts need the `kspace` feature (the time-segmented NUFFT forward)");
    }
    let pl = plan(sim_dims, acq_dims, n_volumes, images, t2, t1, fmap, t_inhom, acq, train, readout, line_weights,
                  shot_images, phase);
    let (nz, nx, ny) = (pl.nz, pl.nx, pl.ny);
    let ncoils = acq.n_coils.max(1);
    let sigma = (acq.noise_variance / (nx * ny) as f64).sqrt();
    let per_vol = |g: usize| -> Vec<(f64, f64)> {
        let sets = shot_images.map_or(&[][..], |si| &si[g][..]);
        // per coil: the acquired 3D k-space, spikes and noise per partition, then the inverse z-DFT
        let mut coil_parts: Vec<Vec<Vec<C>>> = Vec::with_capacity(ncoils);
        for q in 0..ncoils {
            let mut k = pl.kspace(images, n_volumes, g, q, ncoils, line_weights, sets);
            coil_parts.push(partitions_to_slices(&mut k, &pl.mask, &pl.zk, [nx, ny, nz], g, q, seed, acq, sigma));
        }
        reconstruct_slices(&coil_parts, acq, [nx, ny, nz])
    };
    #[cfg(feature = "par")]
    let vols: Vec<Vec<(f64, f64)>> = {
        use rayon::prelude::*;
        (0..n_volumes).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<Vec<(f64, f64)>> = (0..n_volumes).map(per_vol).collect();
    vols
}

/// The 3D echo-train acquisition of a series: one call per series (the seeds are keyed on volume
/// and partition, P0 change 5a). Compartment images, T2, T2' and the fieldmap as
/// [`crate::kspace::simulate_acquisition_oversampled`] takes them; `t1` per compartment, required
/// when the refocusing angle is below 180 degrees; `line_weights` and `shot_images` are part D's.
/// Returns magnitude and phase, `f32`, layout `vox * n_volumes + g`.
#[allow(clippy::too_many_arguments)]
pub fn simulate_acquisition_3d(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, images: &[Vec<f32>], t2: &[T2Volume],
    t1: Option<&[T2Volume]>, fmap: &[f32], t_inhom: Option<&[T2Volume]>, acq: &Acquisition, train: &EchoTrain,
    readout: &Readout3d, line_weights: Option<&LineWeights>, shot_images: Option<&[Vec<ShotSet>]>, phase: &PhaseModel,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let vols = simulate_acquisition_3d_complex(sim_dims, acq_dims, n_volumes, images, t2, t1, fmap, t_inhom, acq, train,
                                               readout, line_weights, shot_images, phase, seed);
    let nvox = acq_dims.iter().product::<usize>();
    let (mut mag, mut ph) = (vec![0.0f32; nvox * n_volumes], vec![0.0f32; nvox * n_volumes]);
    for (g, v) in vols.iter().enumerate() {
        for (vox, &(re, im)) in v.iter().enumerate() {
            let (re, im) = (re as f32, im as f32);
            mag[vox * n_volumes + g] = (re * re + im * im).sqrt();
            ph[vox * n_volumes + g] = im.atan2(re);
        }
    }
    (mag, ph)
}

// ---- the 3D gradient-echo train (P7 addendum, part C) ----

/// One volume's input to [`simulate_acquisition_3d_ge`], built on demand by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct GeVolume {
    /// Per compartment, one volume on the simulation grid (`x + snx*(y + sny*z)`); an empty image
    /// is a compartment this volume does not use (its weights must be zero).
    pub images: Vec<Vec<f32>>,
    /// `w[(p * ky_segments + sy) * n_compartments + c]`: the scalar multiplying compartment `c`'s
    /// object for the excitation that reads partition `p` at in-plane segment `sy`. It carries the
    /// longitudinal state of the train: its approach to the steady state, the label's depletion and
    /// kinetics, the per-shot physiology and gains.
    pub weights: Vec<f64>,
    /// P5's per-shot images, for per-shot motion; `None` when every shot sees `images`.
    pub shot_images: Option<Vec<ShotSet>>,
}

/// The gradient-echo decay of a line read `trf_ms` after its excitation,
/// `exp(-trf (1/T2 + 1/T2'))`, with infinite times meaning no decay.
fn ge_line_weight(trf_ms: f64, t2_ms: f64, t2p_ms: f64) -> f64 {
    let d2 = if t2_ms.is_infinite() { 0.0 } else { trf_ms / t2_ms };
    let dp = if t2p_ms.is_infinite() { 0.0 } else { trf_ms / t2p_ms };
    (-d2 - dp).exp()
}

/// Everything the gradient-echo train's forward needs, resolved once per call.
struct GePlan<'a> {
    snx: usize,
    sny: usize,
    nz: usize,
    nx: usize,
    ny: usize,
    o: usize,
    ncomp: usize,
    acq: &'a Acquisition,
    train: &'a ExcitationTrain,
    ky_segments: usize,
    table: Ge3dTable,
    /// Per compartment: `Some((T2, T2'))` where every relaxation input is uniform (the decay is a
    /// line weight), `None` where it is applied per voxel inside the forward.
    class: Vec<Option<(f64, f64)>>,
    t2: &'a [T2Volume<'a>],
    t_inhom: Option<&'a [T2Volume<'a>]>,
    fmap: &'a [f32],
    phase: &'a PhaseModel,
    zk: Vec<C>,
    mask: Vec<bool>,
}

impl GePlan<'_> {
    /// Each used compartment's 3D k-space at echo `e` from one set of images (`img(c, vox)`),
    /// coil `q`, with its decay but not its weights: `out[c][(p * ny + ky) * nx + kx]`. Each
    /// excitation's echo is a gradient echo at `TE_e`: the 2D forward with gradient-echo formation
    /// at `t_echo = TE_e` adds the static `2 pi fmap TE_e` and the readout's off-resonance; the decay
    /// `exp(-(TE_e + t)(1/T2 + 1/T2'))` is a line weight (class) or the forward's own (voxel).
    fn compartments<F: Fn(usize, usize) -> f32>(&self, img: F, used: &[bool], q: usize, ncoils: usize, e: usize) -> Vec<Vec<C>> {
        let (snx, sny, nz, nx, ny) = (self.snx, self.sny, self.nz, self.nx, self.ny);
        let nplane = snx * sny;
        let te = self.train.echo_times_ms[e];
        let base = Acquisition {
            echo: EchoFormation::Gradient, t_echo: te, noise_variance: 0.0, n_spikes: 0, ..self.acq.clone()
        };
        let norel = Acquisition { do_relaxation: false, ..base.clone() };
        let rel = Acquisition { do_relaxation: true, ..base };
        let t = &self.table.block.t_ms;
        let within = LineTiming {
            t_ms: t.clone(),
            trf_ms: t.iter().map(|x| te + x).collect(),
            tread_ms: t.clone(),
            polarity: self.table.block.polarity.clone(),
        };
        let zero_shot = ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] };
        let phis: Vec<Vec<f64>> = (0..nz).map(|z| phase_slice(self.phase, &zero_shot, snx, sny, self.o, z, nz)).collect();
        let mut out = vec![Vec::new(); self.ncomp];
        for c in (0..self.ncomp).filter(|&c| used[c]) {
            let voxel = self.class[c].is_none() && self.acq.do_relaxation;
            let acq_c = if voxel { &rel } else { &norel };
            let k2: Vec<Vec<C>> = (0..nz).map(|z| {
                let pl: Vec<f32> = (0..nplane).map(|i| img(c, z * nplane + i)).collect();
                let refs = [&pl[..]];
                let t2s = [slice_of(&self.t2[c], z, nplane)];
                let tis = self.t_inhom.map(|ti| [slice_of(&ti[c], z, nplane)]);
                let inp = SliceInput {
                    compartments: &refs, t2: &t2s, t_inhom: tis.as_ref().map(|a| a.as_slice()),
                    fmap: &self.fmap[z * nplane..(z + 1) * nplane], phase0: Some(&phis[z]), sim: [snx, sny],
                    acq_matrix: [nx, ny], z, nz, eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None,
                };
                build_coil_kspace_timed(&inp, acq_c, q, ncoils, &within, None)
            }).collect();
            let mut kc = vec![C::ZERO; nz * ny * nx];
            for p in 0..nz {
                for ky in 0..ny {
                    let w = match (self.class[c], self.acq.do_relaxation) {
                        (Some((t2, t2p)), true) => ge_line_weight(te + t[ky], t2, t2p),
                        _ => 1.0,
                    };
                    for kx in 0..nx {
                        let mut acc = C::ZERO;
                        for (z, k2z) in k2.iter().enumerate() {
                            acc = acc.add(k2z[ky * nx + kx].mul(self.zk[p * nz + z]));
                        }
                        kc[(p * ny + ky) * nx + kx] = acc.scale(w);
                    }
                }
            }
            out[c] = kc;
        }
        out
    }

    /// The acquired 3D k-space of one volume at echo `e`, coil `q`, before spikes and noise: per
    /// line, the compartment sum weighted by the excitation that reads it, from the images its shot
    /// saw.
    fn kspace(&self, vol: &GeVolume, q: usize, ncoils: usize, e: usize) -> Vec<C> {
        let (nz, nx, ny, ncomp) = (self.nz, self.nx, self.ny, self.ncomp);
        let used: Vec<bool> = vol.images.iter().map(|i| !i.is_empty()).collect();
        let base = self.compartments(|c, v| vol.images[c][v], &used, q, ncoils, e);
        let sets = vol.shot_images.as_deref().unwrap_or(&[]);
        let moved: Vec<Vec<Vec<C>>> = sets.iter().map(|s| self.compartments(|c, v| s.images[c][v], &used, q, ncoils, e)).collect();
        let mut k = vec![C::ZERO; nz * ny * nx];
        for p in 0..nz {
            for ky in 0..ny {
                if !self.mask[ky * nx] {
                    continue;
                }
                let line = self.table.line(p, ky);
                let sy = ky % self.ky_segments;
                let src = sets.iter().position(|s| s.shots.contains(&line.shot)).map_or(&base, |i| &moved[i]);
                for (c, kc) in src.iter().enumerate().filter(|(c, _)| used[*c]) {
                    let w = vol.weights[(p * self.ky_segments + sy) * ncomp + c];
                    for kx in 0..nx {
                        let i = (p * ny + ky) * nx + kx;
                        k[i] = k[i].add(kc[i].scale(w));
                    }
                }
            }
        }
        k
    }
}

/// The reconstructed complex volumes of a 3D gradient-echo series, `f64`, per volume per echo,
/// layout `vox = x + nx*(y + ny*z)`: [`simulate_acquisition_3d_ge`] before its `f32` cast.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn simulate_acquisition_3d_ge_complex(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, t2: &[T2Volume], fmap: &[f32],
    t_inhom: Option<&[T2Volume]>, acq: &Acquisition, train: &ExcitationTrain, readout: &Ge3dReadout,
    volume: &(dyn Fn(usize) -> GeVolume + Sync), phase: &PhaseModel, seed: u64,
) -> Vec<Vec<Vec<(f64, f64)>>> {
    let [snx, sny, nz] = sim_dims;
    let [nx, ny, nzo] = acq_dims;
    assert_eq!(nz, nzo, "partition count must match; z is never oversampled");
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple of the acquired matrix");
    let o = snx / nx;
    assert_eq!(o, sny / ny, "oversampling must match on both axes");
    let nvox = snx * sny * nz;
    let ncomp = t2.len();
    assert_eq!(fmap.len(), nvox, "fieldmap is not on the simulation grid");
    if let Some(ti) = t_inhom {
        assert_eq!(ti.len(), ncomp, "t_inhom has {} entries for {ncomp} compartments", ti.len());
    }
    assert_eq!(acq.echo, EchoFormation::Gradient, "the 3D gradient-echo train is a gradient-echo acquisition");
    assert!(acq.eddy_strength == 0.0 && acq.eddy_quad == 0.0 && acq.eddy_phase == 0.0,
            "the eddy model is not available in 3D (an excitation-dependent eddy evolution breaks the z factorization)");
    assert!(acq.partial_fourier >= 1.0, "partial Fourier is not available on the 3D gradient-echo train");
    assert_eq!(readout.reverse_phase, acq.reverse_phase, "the readout's phase-encode sign must be the acquisition's");
    assert!(!train.echo_times_ms.is_empty(), "a 3D gradient-echo train needs at least one echo time");
    let table = ge3d_lines(train, readout, ny, nz).unwrap_or_else(|e| panic!("{e}"));
    let class: Vec<Option<(f64, f64)>> = (0..ncomp).map(|c| {
        let ti = t_inhom.map_or(T2Volume::Uniform(acq.t_inhom as f32), |t| t[c]);
        match (t2[c], ti) {
            (T2Volume::Uniform(a), T2Volume::Uniform(b)) => Some((a as f64, b as f64)),
            _ => None,
        }
    }).collect();
    let n_exc_lines = nz * readout.ky_segments * ncomp;
    let pl = GePlan {
        snx, sny, nz, nx, ny, o, ncomp, acq, train, ky_segments: readout.ky_segments, table, class, t2, t_inhom, fmap,
        phase, zk: zkernel(nz), mask: sampling_mask(nx, ny, acq),
    };
    let ncoils = acq.n_coils.max(1);
    let sigma = (acq.noise_variance / (nx * ny) as f64).sqrt();
    let per_vol = |g: usize| -> Vec<Vec<(f64, f64)>> {
        let vol = volume(g);
        assert_eq!(vol.images.len(), ncomp, "volume {g}: {} compartment images for {ncomp} compartments", vol.images.len());
        for (c, im) in vol.images.iter().enumerate() {
            assert!(im.is_empty() || im.len() == nvox, "volume {g}: compartment {c} is not on the simulation grid");
        }
        assert_eq!(vol.weights.len(), n_exc_lines, "volume {g}: weights are not (partition, ky segment, compartment)");
        for (c, im) in vol.images.iter().enumerate() {
            if im.is_empty() {
                assert!((0..nz * readout.ky_segments).all(|x| vol.weights[x * ncomp + c] == 0.0),
                        "volume {g}: compartment {c} has no image but a nonzero weight");
            }
        }
        if let Some(sets) = &vol.shot_images {
            for s in sets {
                assert!(s.images.len() == ncomp && s.images.iter().zip(&vol.images).all(|(a, b)| a.len() == b.len()),
                        "volume {g}: shot images are not the volume's compartments");
                assert!(s.shots.iter().all(|&x| x < pl.table.n_shots), "a shot index beyond the train's {}", pl.table.n_shots);
            }
        }
        (0..train.echo_times_ms.len()).map(|e| {
            // the receiver noise of echo e on its own streams (P6's echo salt; echo 0 unsalted)
            let seed_e = seed ^ echo_salt(e);
            let coil_parts: Vec<Vec<Vec<C>>> = (0..ncoils).map(|q| {
                let mut k = pl.kspace(&vol, q, ncoils, e);
                partitions_to_slices(&mut k, &pl.mask, &pl.zk, [nx, ny, nz], g, q, seed_e, acq, sigma)
            }).collect();
            reconstruct_slices(&coil_parts, acq, [nx, ny, nz])
        }).collect()
    };
    #[cfg(feature = "par")]
    let vols: Vec<Vec<Vec<(f64, f64)>>> = {
        use rayon::prelude::*;
        (0..n_volumes).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<Vec<Vec<(f64, f64)>>> = (0..n_volumes).map(per_vol).collect();
    vols
}

/// The 3D gradient-echo stack-of-EPI acquisition of a series (P7 addendum, part C): one call per
/// series, each volume's images and per-excitation weights built on demand by `volume(g)` (called
/// once per volume, from the worker that simulates it), so memory scales with the volumes in
/// flight. The noise and spike streams are keyed on `(volume, partition)` and salted per echo.
/// Returns per echo magnitude and phase, `f32`, layout `vox * n_volumes + g`.
#[allow(clippy::too_many_arguments)]
pub fn simulate_acquisition_3d_ge(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, t2: &[T2Volume], fmap: &[f32],
    t_inhom: Option<&[T2Volume]>, acq: &Acquisition, train: &ExcitationTrain, readout: &Ge3dReadout,
    volume: &(dyn Fn(usize) -> GeVolume + Sync), phase: &PhaseModel, seed: u64,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let vols = simulate_acquisition_3d_ge_complex(sim_dims, acq_dims, n_volumes, t2, fmap, t_inhom, acq, train, readout,
                                                  volume, phase, seed);
    let nvox = acq_dims.iter().product::<usize>();
    (0..train.echo_times_ms.len()).map(|e| {
        let (mut mag, mut ph) = (vec![0.0f32; nvox * n_volumes], vec![0.0f32; nvox * n_volumes]);
        for (g, v) in vols.iter().enumerate() {
            for (vox, &(re, im)) in v[e].iter().enumerate() {
                let (re, im) = (re as f32, im as f32);
                mag[vox * n_volumes + g] = (re * re + im * im).sqrt();
                ph[vox * n_volumes + g] = im.atan2(re);
            }
        }
        (mag, ph)
    }).collect()
}

// ---- the stack of spirals (P5 addendum, part C) ----

/// One slice's certified time segmentation, as the spiral path plans it.
#[derive(Debug, Clone, PartialEq)]
pub struct SpiralSegmentation {
    pub z: usize,
    /// `true` for the voxel-mode plan (decay inside), `false` for the class-mode one.
    pub voxel: bool,
    pub l: usize,
    pub m: usize,
    pub b_sum: f64,
    pub bound: f64,
}

/// Per slice, the off-resonance interval (Hz) the spiral forward must cover, and the decay interval
/// (1/ms) of its voxel-mode compartments (`None` when there are none).
#[cfg(feature = "kspace")]
#[allow(clippy::too_many_arguments)]
fn spiral_rects(snx: usize, sny: usize, nz: usize, fmap: &[f32], t2: &[T2Volume], t_inhom: Option<&[T2Volume]>,
                acq: &Acquisition, modes: &[Mode]) -> Vec<([f64; 2], Option<[f64; 2]>)> {
    let nplane = snx * sny;
    let at = |v: &T2Volume, i: usize| match v { T2Volume::Uniform(s) => *s as f64, T2Volume::Map(m) => m[i] as f64 };
    (0..nz)
        .map(|z| {
            let sl = &fmap[z * nplane..(z + 1) * nplane];
            let f = if acq.do_distortions {
                [sl.iter().fold(f32::MAX, |a, &b| a.min(b)) as f64, sl.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64]
            } else {
                [0.0, 0.0]
            };
            let mut d: Option<[f64; 2]> = None;
            for (c, mode) in modes.iter().enumerate() {
                if !matches!(mode, Mode::Voxel) {
                    continue;
                }
                for i in z * nplane..(z + 1) * nplane {
                    // as `spiral::decay_rate` forms it
                    let tp = t_inhom.map_or(acq.t_inhom, |ti| at(&ti[c], i));
                    let r = 1.0 / at(&t2[c], i) + 1.0 / tp;
                    d = Some(d.map_or([r, r], |[a, b]| [a.min(r), b.max(r)]));
                }
            }
            (f, d)
        })
        .collect()
}

/// Everything the spiral path needs, resolved once per series.
#[cfg(feature = "kspace")]
struct SpiralPlan<'a> {
    snx: usize,
    sny: usize,
    nz: usize,
    n: usize,
    o: usize,
    ncomp: usize,
    acq: &'a Acquisition,
    table: crate::readout::Spiral3dTable,
    modes: Vec<Mode>,
    ln_a_maps: Vec<Option<Vec<Vec<f64>>>>,
    t2: &'a [T2Volume<'a>],
    t_inhom: Option<&'a [T2Volume<'a>]>,
    fmap: &'a [f32],
    phase: &'a PhaseModel,
    zk: Vec<C>,
    /// Samples per partition (all interleaves) and per interleaf.
    nsamp: usize,
    ns: usize,
    class_fwd: Vec<Option<crate::spiral::SegmentedForward>>,
    voxel_fwd: Vec<Option<crate::spiral::SegmentedForward>>,
    recon: crate::grid_recon::SpiralRecon,
}

#[cfg(feature = "kspace")]
impl SpiralPlan<'_> {
    /// Every compartment's 3D samples from one set of images (`img(c, vox)`), coil `q`, weighted by
    /// the echo amplitude and in-echo decay but not by the line weights: `out[c][p * nsamp + j]`.
    fn compartments<F: Fn(usize, usize) -> f32>(&self, img: F, q: usize, ncoils: usize) -> Vec<Vec<C>> {
        use crate::spiral::segment_inputs;
        let (snx, sny, nz, n) = (self.snx, self.sny, self.nz, self.n);
        let nplane = snx * sny;
        let zero_shot = ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] };
        let phis: Vec<Vec<f64>> = (0..nz).map(|z| phase_slice(self.phase, &zero_shot, snx, sny, self.o, z, nz)).collect();
        let mut out = vec![vec![C::ZERO; nz * self.nsamp]; self.ncomp];
        for (c, oc) in out.iter_mut().enumerate() {
            let planes: Vec<Vec<f32>> = (0..nz).map(|z| (0..nplane).map(|i| img(c, z * nplane + i)).collect()).collect();
            let one = |z: usize, la: Option<&[f64]>, voxel: bool| -> Vec<C> {
                let refs = [planes[z].as_slice()];
                let t2s = [slice_of(&self.t2[c], z, nplane)];
                let tis = self.t_inhom.map(|ti| [slice_of(&ti[c], z, nplane)]);
                let inp = SliceInput {
                    compartments: &refs, t2: &t2s, t_inhom: tis.as_ref().map(|a| a.as_slice()),
                    fmap: &self.fmap[z * nplane..(z + 1) * nplane], phase0: Some(&phis[z]), sim: [snx, sny],
                    acq_matrix: [n, n], z, nz, eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None,
                };
                let (x, r) = segment_inputs(&inp, self.acq, q, ncoils, la, voxel);
                let fwd = if voxel { &self.voxel_fwd[z] } else { &self.class_fwd[z] };
                fwd.as_ref().expect("a plan for every slice of the mode").apply(&x, &r)
            };
            let zsum = |k2: &[Vec<C>], p: usize, j: usize| {
                let mut acc = C::ZERO;
                for (z, k2z) in k2.iter().enumerate() {
                    acc = acc.add(k2z[j].mul(self.zk[p * nz + z]));
                }
                acc
            };
            match &self.modes[c] {
                Mode::Class { ln_a, t2_ms, t2p_ms } => {
                    let k2: Vec<Vec<C>> = (0..nz).map(|z| one(z, None, false)).collect();
                    for p in 0..nz {
                        let e = self.table.echo[p];
                        for j in 0..self.nsamp {
                            let w = if self.acq.do_relaxation {
                                line_weight(ln_a[e - 1], self.table.traj.tau_ms[j % self.ns], *t2_ms, *t2p_ms)
                            } else {
                                1.0
                            };
                            oc[p * self.nsamp + j] = zsum(&k2, p, j).scale(w);
                        }
                    }
                }
                Mode::Voxel => {
                    let maps = self.ln_a_maps[c].as_ref().expect("voxel compartments carry their maps");
                    for e in 1..=self.table.echo.iter().copied().max().unwrap_or(0) {
                        let ps: Vec<usize> = (0..nz).filter(|&p| self.table.echo[p] == e).collect();
                        if ps.is_empty() {
                            continue;
                        }
                        let k2: Vec<Vec<C>> =
                            (0..nz).map(|z| one(z, Some(&maps[e - 1][z * nplane..(z + 1) * nplane]), true)).collect();
                        for &p in &ps {
                            for j in 0..self.nsamp {
                                oc[p * self.nsamp + j] = zsum(&k2, p, j);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The acquired 3D samples of volume `g`, coil `q`, before noise: the compartment sum with the
    /// line weights, each sample from the images its shot saw. Layout `p * nsamp + j`.
    #[allow(clippy::too_many_arguments)]
    fn samples(&self, images: &[Vec<f32>], n_volumes: usize, g: usize, q: usize, ncoils: usize,
               lw: Option<&LineWeights>, sets: &[ShotSet]) -> Vec<C> {
        let base = self.compartments(|c, v| images[c][v * n_volumes + g], q, ncoils);
        let moved: Vec<Vec<Vec<C>>> = sets.iter().map(|s| self.compartments(|c, v| s.images[c][v], q, ncoils)).collect();
        let mut k = vec![C::ZERO; self.nz * self.nsamp];
        for p in 0..self.nz {
            for j in 0..self.nsamp {
                let shot = self.table.shot(p, j / self.ns);
                let src = sets.iter().position(|s| s.shots.contains(&shot)).map_or(&base, |i| &moved[i]);
                let i = p * self.nsamp + j;
                for (c, kc) in src.iter().enumerate() {
                    let w = lw.map_or(1.0, |lw| lw.at(g, shot, c));
                    k[i] = k[i].add(kc[i].scale(w));
                }
            }
        }
        k
    }
}

/// The spiral path's checks and resolution; panics name the offending input.
#[cfg(feature = "kspace")]
#[allow(clippy::too_many_arguments)]
fn spiral_plan<'a>(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, images: &[Vec<f32>], t2: &'a [T2Volume<'a>],
    t1: Option<&'a [T2Volume<'a>]>, fmap: &'a [f32], t_inhom: Option<&'a [T2Volume<'a>]>, acq: &'a Acquisition,
    train: &EchoTrain, readout: &Readout3d, line_weights: Option<&LineWeights>, shot_images: Option<&[Vec<ShotSet>]>,
    phase: &'a PhaseModel,
) -> SpiralPlan<'a> {
    use crate::tseg::RateRect;
    let [snx, sny, nz] = sim_dims;
    let [nx, ny, nzo] = acq_dims;
    assert_eq!(nz, nzo, "partition count must match; z is never oversampled");
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple of the acquired matrix");
    let o = snx / nx;
    assert_eq!(o, sny / ny, "oversampling must match on both axes");
    let nvox = snx * sny * nz;
    let ncomp = images.len();
    for im in images {
        assert_eq!(im.len(), nvox * n_volumes, "compartment image is not on the simulation grid");
    }
    assert_eq!(fmap.len(), nvox, "fieldmap is not on the simulation grid");
    assert_eq!(t2.len(), ncomp, "t2 has {} entries for {ncomp} compartments", t2.len());
    assert_eq!(acq.echo, EchoFormation::Spin, "the 3D echo trains are spin-echo trains");
    assert!(acq.eddy_strength == 0.0 && acq.eddy_quad == 0.0 && acq.eddy_phase == 0.0,
            "the eddy model is not available in 3D (an echo-dependent eddy evolution breaks the z factorization)");
    assert!(acq.accel <= 1, "GRAPPA has no spiral meaning in this model");
    assert!(acq.partial_fourier >= 1.0, "partial Fourier has no spiral meaning in this model");
    assert!(acq.ghost_offset == 0.0, "Nyquist ghosting has no spiral meaning in this model");
    assert!(acq.n_spikes == 0, "spikes have no spiral meaning in this model");
    let table = crate::readout::spiral_lines(train, readout, nx, ny, nz).unwrap_or_else(|e| panic!("{e}"));
    if let Some(lw) = line_weights {
        assert!(lw.n_shots == table.n_shots && lw.n_compartments == ncomp && lw.w.len() == n_volumes * table.n_shots * ncomp,
                "line weights are not (volume, shot, compartment)");
    }
    if let Some(si) = shot_images {
        assert_eq!(si.len(), n_volumes, "shot images per volume");
        for sets in si {
            for s in sets {
                assert!(s.images.len() == ncomp && s.images.iter().all(|i| i.len() == nvox), "shot images are not per compartment");
                assert!(s.shots.iter().all(|&x| x < table.n_shots), "a shot index beyond the train's {}", table.n_shots);
            }
        }
    }
    let (modes, ln_a_maps) = resolve_modes(t2, t1, t_inhom, acq, train, nvox);
    let ns = table.traj.n_samples();
    let nsamp = table.traj.k.len();
    let tau_idx: Vec<usize> = (0..nsamp).map(|j| j % ns).collect();
    let t_ms = table.traj.readout_ms;
    let rects = spiral_rects(snx, sny, nz, fmap, t2, t_inhom, acq, &modes);
    let any_class = modes.iter().any(|m| matches!(m, Mode::Class { .. }));
    let mk = |rect: RateRect| {
        crate::spiral::SegmentedForward::new(&table.traj.k, &tau_idx, &table.traj.tau_ms, t_ms, [snx, sny], [nx, ny], rect)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let class_fwd = rects.iter().map(|(f, _)| any_class.then(|| mk(RateRect { d: [0.0, 0.0], f: *f }))).collect();
    let voxel_fwd = rects.iter().map(|(f, d)| d.map(|d| mk(RateRect { d, f: *f }))).collect();
    let recon = crate::grid_recon::SpiralRecon::new(&table.traj.k, nx, acq.window);
    SpiralPlan {
        snx, sny, nz, n: nx, o, ncomp, acq, table, modes, ln_a_maps, t2, t_inhom, fmap, phase, zk: zkernel(nz), nsamp, ns,
        class_fwd, voxel_fwd, recon,
    }
}

/// The certified segmentations a spiral series will use, per slice and mode, or the error naming
/// the rate rectangle that cannot be certified: what a caller checks and records before
/// simulating. The arguments are [`simulate_acquisition_3d`]'s (images are not needed).
#[cfg(feature = "kspace")]
#[allow(clippy::too_many_arguments)]
pub fn spiral_segmentation(
    sim_dims: [usize; 3], acq_dims: [usize; 3], t2: &[T2Volume], t1: Option<&[T2Volume]>, fmap: &[f32],
    t_inhom: Option<&[T2Volume]>, acq: &Acquisition, train: &EchoTrain, readout: &Readout3d,
) -> Result<Vec<SpiralSegmentation>, String> {
    use crate::tseg::RateRect;
    let [snx, sny, nz] = sim_dims;
    let [nx, ny, _] = acq_dims;
    let table = crate::readout::spiral_lines(train, readout, nx, ny, nz)?;
    let (modes, _) = resolve_modes(t2, t1, t_inhom, acq, train, snx * sny * nz);
    let rects = spiral_rects(snx, sny, nz, fmap, t2, t_inhom, acq, &modes);
    let any_class = modes.iter().any(|m| matches!(m, Mode::Class { .. }));
    let mut out = Vec::new();
    for (z, (f, d)) in rects.iter().enumerate() {
        let mut add = |voxel: bool, rect: RateRect| -> Result<(), String> {
            let p = crate::tseg::plan(&table.traj.tau_ms, table.traj.readout_ms, rect).map_err(|e| format!("slice {z}: {e}"))?;
            out.push(SpiralSegmentation { z, voxel, l: p.l, m: p.m, b_sum: p.b_sum, bound: p.bound });
            Ok(())
        };
        if any_class {
            add(false, RateRect { d: [0.0, 0.0], f: *f })?;
        }
        if let Some(d) = d {
            add(true, RateRect { d: *d, f: *f })?;
        }
    }
    Ok(out)
}

/// The spiral path of [`simulate_acquisition_3d_complex`].
#[cfg(feature = "kspace")]
#[allow(clippy::too_many_arguments)]
fn spiral_volumes(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize, images: &[Vec<f32>], t2: &[T2Volume],
    t1: Option<&[T2Volume]>, fmap: &[f32], t_inhom: Option<&[T2Volume]>, acq: &Acquisition, train: &EchoTrain,
    readout: &Readout3d, line_weights: Option<&LineWeights>, shot_images: Option<&[Vec<ShotSet>]>, phase: &PhaseModel,
    seed: u64,
) -> Vec<Vec<(f64, f64)>> {
    let pl = spiral_plan(sim_dims, acq_dims, n_volumes, images, t2, t1, fmap, t_inhom, acq, train, readout, line_weights,
                         shot_images, phase);
    let (nz, n, nsamp) = (pl.nz, pl.n, pl.nsamp);
    let ncoils = acq.n_coils.max(1);
    let sigma = (acq.noise_variance / (n * n) as f64).sqrt();
    let per_vol = |g: usize| -> Vec<(f64, f64)> {
        let sets = shot_images.map_or(&[][..], |si| &si[g][..]);
        let mut coil_parts: Vec<Vec<Vec<C>>> = Vec::with_capacity(ncoils);
        for q in 0..ncoils {
            let mut k = pl.samples(images, n_volumes, g, q, ncoils, line_weights, sets);
            if acq.noise_variance > 0.0 {
                for p in 0..nz {
                    let pseed = (g as u64).wrapping_mul(0x100_0001).wrapping_add(p as u64).wrapping_mul(0x9E37) ^ seed ^ SEED_SALT_3D;
                    let mut rng = Rng((pseed ^ (q as u64).wrapping_mul(0x9E37_79B9)) | 1);
                    for c in k[p * nsamp..(p + 1) * nsamp].iter_mut() {
                        c.re += rng.gauss() * sigma;
                        c.im += rng.gauss() * sigma;
                    }
                }
            }
            // inverse z-DFT, normalized by 1/nz: per slice z, its spiral samples
            let mut slices = vec![vec![C::ZERO; nsamp]; nz];
            for (z, sl) in slices.iter_mut().enumerate() {
                for p in 0..nz {
                    let kc = pl.zk[p * nz + z];
                    let conj = C { re: kc.re, im: -kc.im };
                    for (s, kv) in sl.iter_mut().zip(&k[p * nsamp..(p + 1) * nsamp]) {
                        *s = s.add(kv.mul(conj));
                    }
                }
                for c in sl.iter_mut() {
                    *c = c.scale(1.0 / nz as f64);
                }
            }
            coil_parts.push(slices);
        }
        let mut out = vec![(0.0, 0.0); n * n * nz];
        for z in 0..nz {
            let coils: Vec<Vec<C>> = coil_parts.iter().map(|cp| cp[z].clone()).collect();
            let img = pl.recon.reconstruct(&coils, acq);
            for (i, c) in img.iter().enumerate() {
                out[i + n * n * z] = (c.re, c.im);
            }
        }
        out
    };
    #[cfg(feature = "par")]
    let vols: Vec<Vec<(f64, f64)>> = {
        use rayon::prelude::*;
        (0..n_volumes).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<Vec<(f64, f64)>> = (0..n_volumes).map(per_vol).collect();
    vols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kspace::simulate_acquisition_oversampled;
    use crate::readout::KzOrder;

    /// A smooth blob per compartment, varying across volumes and slices, on a `snx x sny x nz`
    /// grid, in the entry points' `vox * n_volumes + g` layout.
    fn blobs(snx: usize, sny: usize, nz: usize, nv: usize, ncomp: usize) -> Vec<Vec<f32>> {
        let nvox = snx * sny * nz;
        (0..ncomp)
            .map(|c| {
                let mut v = vec![0.0f32; nvox * nv];
                for vox in 0..nvox {
                    let (x, y, z) = (vox % snx, (vox / snx) % sny, vox / (snx * sny));
                    let (fx, fy, fz) = (x as f64 / snx as f64, y as f64 / sny as f64, z as f64 / nz as f64);
                    let b = (-((fx - 0.5 - 0.05 * c as f64).powi(2) + (fy - 0.45).powi(2)) / 0.04).exp()
                        * (1.0 + 0.5 * (TAU * fz).sin());
                    for g in 0..nv {
                        v[vox * nv + g] = ((1.0 + 0.1 * g as f64) * b) as f32;
                    }
                }
                v
            })
            .collect()
    }

    fn train(nz: usize, kz_segments: usize, order: KzOrder, esp_ms: f64, b: f64) -> EchoTrain {
        EchoTrain { etl: nz / kz_segments, esp_ms, refocusing_deg: b, kz_order: order, kz_segments, refocusing_time_ms: 2.0 }
    }

    fn peak(v: &[(f64, f64)]) -> f64 {
        v.iter().map(|(a, b)| a.hypot(*b)).fold(0.0, f64::max)
    }

    fn maxdiff(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x.0 - y.0).hypot(x.1 - y.1)).fold(0.0, f64::max)
    }

    fn zero_shot() -> ShotPhase {
        ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] }
    }

    #[test]
    fn with_nothing_to_distinguish_partitions_3d_is_2d() {
        let (nx, ny, nz, o, nv, ncomp) = (16usize, 16usize, 6usize, 2usize, 2usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let images = blobs(snx, sny, nz, nv, ncomp);
        let fmap = vec![0.0f32; snx * sny * nz];
        let t2 = vec![T2Volume::Uniform(80.0); ncomp];
        let phase = PhaseModel::none();
        for (n_coils, accel) in [(1usize, 1usize), (3, 2)] {
            let acq = Acquisition { do_relaxation: false, n_coils, accel, acs_lines: 6, signal_scale: 100.0,
                                    ..Acquisition::default() };
            let (m2, p2) = simulate_acquisition_oversampled([snx, sny, nz], [nx, ny, nz], nv, &images, &t2, &fmap, None,
                                                            &acq, &vec![None; nv], &vec![None; nv], &phase, 9, None, None);
            let pk = m2.iter().fold(0.0f32, |a, b| a.max(*b)) as f64;
            for ky_segments in [1usize, 2, 4] {
                for kz_segments in [1usize, 2] {
                    for order in [KzOrder::Centric, KzOrder::Linear] {
                        let tr = train(nz, kz_segments, order, 20.0, 180.0);
                        let ro = Readout3d::Grase { ky_segments, t_line_ms: 0.5, reverse_phase: false };
                        let (m3, p3) = simulate_acquisition_3d([snx, sny, nz], [nx, ny, nz], nv, &images, &t2, None, &fmap,
                                                               None, &acq, &tr, &ro, None, None, &phase, 9);
                        let mut worst = 0.0f64;
                        for i in 0..m2.len() {
                            let (a, b) = (m2[i] as f64, m3[i] as f64);
                            let (pa, pb) = (p2[i] as f64, p3[i] as f64);
                            let d = (a * pa.cos() - b * pb.cos()).hypot(a * pa.sin() - b * pb.sin());
                            worst = worst.max(d);
                        }
                        assert!(worst <= 1e-5 * pk, "coils {n_coils} R {accel} ky {ky_segments} kz {kz_segments} \
                                                    {order:?}: {worst:e} of {pk:e}");
                    }
                }
            }
        }
    }

    #[test]
    fn ghost_polarity_follows_the_acquisition_index() {
        let (nx, ny, nz, o) = (12usize, 16usize, 4usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let images = blobs(snx, sny, nz, 1, 1);
        let fmap = vec![0.0f32; snx * sny * nz];
        let t2 = [T2Volume::Uniform(80.0)];
        for reverse_phase in [false, true] {
            let acq = Acquisition { do_relaxation: false, ghost_offset: 0.05, reverse_phase, ..Acquisition::default() };
            let tr = train(nz, 1, KzOrder::Centric, 20.0, 180.0);
            let ro = Readout3d::Grase { ky_segments: 2, t_line_ms: 0.5, reverse_phase };
            let got = kspace_3d_observed([snx, sny, nz], [nx, ny, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr, &ro,
                                         None, None, &PhaseModel::none(), 0).remove(0);
            // independently: the polarity by acquisition index within each interleaved block, the
            // 2D forward per slice with that table, then the z-DFT
            let epi = ny / 2;
            let mut want_t = LineTiming { t_ms: vec![0.0; ny], trf_ms: vec![0.0; ny], tread_ms: vec![0.0; ny], polarity: vec![0; ny] };
            for ky in 0..ny {
                let j = ky / 2;
                let a = if reverse_phase { j } else { epi - 1 - j };
                want_t.t_ms[ky] = (a as f64 - (epi as f64 - 1.0) / 2.0) * 0.5;
                want_t.trf_ms[ky] = want_t.t_ms[ky];
                want_t.tread_ms[ky] = want_t.t_ms[ky];
                want_t.polarity[ky] = if a % 2 == 0 { 1 } else { -1 };
            }
            let parity_t = LineTiming { polarity: (0..ny).map(|k| if k % 2 == 1 { -1 } else { 1 }).collect(), ..want_t.clone() };
            let direct = |tim: &LineTiming| -> Vec<(f64, f64)> {
                let nplane = snx * sny;
                let k2: Vec<Vec<C>> = (0..nz).map(|z| {
                    let pl: Vec<f32> = (0..nplane).map(|i| images[0][z * nplane + i]).collect();
                    let phi = phase_slice(&PhaseModel::none(), &zero_shot(), snx, sny, o, z, nz);
                    let refs = [pl.as_slice()];
                    let t2s = [T2Slice::Uniform(80.0)];
                    let inp = SliceInput { compartments: &refs, t2: &t2s, t_inhom: None, fmap: &fmap[z * nplane..(z + 1) * nplane],
                                           phase0: Some(&phi), sim: [snx, sny], acq_matrix: [nx, ny], z, nz, eddy_drive: None,
                                           prep_drive: None, slice_seed: 0, eddy_lin: None };
                    build_coil_kspace_timed(&inp, &acq, 0, 1, tim, None)
                }).collect();
                let zs = (nz / 2) as f64;
                let mut out = vec![(0.0, 0.0); nz * ny * nx];
                for p in 0..nz {
                    for i in 0..ny * nx {
                        let mut acc = C::ZERO;
                        for (z, k) in k2.iter().enumerate() {
                            let th = TAU * (p as f64 - zs) * (z as f64 - zs) / nz as f64;
                            acc = acc.add(k[i].mul(C { re: th.cos(), im: th.sin() }));
                        }
                        out[p * ny * nx + i] = (acc.re, acc.im);
                    }
                }
                out
            };
            let want = direct(&want_t);
            let wrong = direct(&parity_t);
            let pk = peak(&want);
            assert!(maxdiff(&got, &want) <= 1e-10 * pk, "reverse {reverse_phase}: {:e}", maxdiff(&got, &want));
            assert!(maxdiff(&got, &wrong) > 1e-3 * pk, "polarity by ky parity must differ: {:e}", maxdiff(&got, &wrong));
        }
    }

    /// The y centroid of one slice's magnitude, in acquired voxels.
    fn centroid_y(mag: &[f32], nx: usize, ny: usize, z: usize) -> f64 {
        let (mut s, mut sy) = (0.0f64, 0.0f64);
        for y in 0..ny {
            for x in 0..nx {
                let m = mag[x + nx * (y + ny * z)] as f64;
                s += m;
                sy += m * y as f64;
            }
        }
        sy / s
    }

    #[test]
    fn off_resonance_shifts_the_way_the_2d_epi_does() {
        let (nx, ny, nz, o) = (16usize, 32usize, 2usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let nvox = snx * sny * nz;
        let mut img = vec![0.0f32; nvox];
        for (vox, v) in img.iter_mut().enumerate() {
            let (x, y) = (vox % snx, (vox / snx) % sny);
            if (12..20).contains(&x) && (36..44).contains(&y) {
                *v = 1.0;
            }
        }
        let images = vec![img];
        // 62.5 Hz with 0.5 ms lines over 32 lines: phase 2 pi f t_line per line, a shift of
        // f t_line ny = exactly one voxel
        let fmap = vec![62.5f32; nvox];
        let t2 = [T2Volume::Uniform(80.0)];
        for reverse_phase in [false, true] {
            let acq = Acquisition { do_relaxation: false, reverse_phase, t_line: 0.5, t_echo: 30.0, ..Acquisition::default() };
            let none = Acquisition { do_distortions: false, ..acq.clone() };
            let run2 = |a: &Acquisition| simulate_acquisition_oversampled([snx, sny, nz], [nx, ny, nz], 1, &images, &t2, &fmap, None,
                                                                         a, &[None], &[None], &PhaseModel::none(), 1, None, None).0;
            let tr = train(nz, 1, KzOrder::Centric, 30.0, 180.0);
            let ro = Readout3d::Grase { ky_segments: 1, t_line_ms: 0.5, reverse_phase };
            let run3 = |a: &Acquisition| simulate_acquisition_3d([snx, sny, nz], [nx, ny, nz], 1, &images, &t2, None, &fmap, None,
                                                                 a, &tr, &ro, None, None, &PhaseModel::none(), 1).0;
            // the 2D EPI's direction (its centroid also carries the off-resonance Nyquist ghost
            // of its odd-line time offset, so only its sign is used)
            let s2 = centroid_y(&run2(&acq), nx, ny, 0) - centroid_y(&run2(&none), nx, ny, 0);
            assert!(s2.abs() > 0.5, "reverse {reverse_phase}: 2D shift {s2}");
            let dir: isize = if s2 > 0.0 { 1 } else { -1 };
            // the GRASE block has no odd-line offset: the image is the undistorted one moved by
            // exactly one voxel in that direction
            let (d, u) = (run3(&acq), run3(&none));
            let pk = u.iter().fold(0.0f32, |a, b| a.max(*b));
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let src = ((y as isize - dir).rem_euclid(ny as isize)) as usize;
                        let (a, b) = (d[x + nx * (y + ny * z)], u[x + nx * (src + ny * z)]);
                        assert!((a - b).abs() <= 1e-5 * pk, "reverse {reverse_phase} ({x},{y},{z}): {a} vs {b}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_one_partition_slab_spreads_by_the_kz_modulation() {
        let (nx, ny, nz, z0) = (8usize, 8usize, 12usize, 4usize);
        let nvox = nx * ny * nz;
        let img: Vec<f32> = (0..nvox).map(|vox| if vox / (nx * ny) == z0 { 1.0 } else { 0.0 }).collect();
        let images = vec![img];
        let fmap = vec![0.0f32; nvox];
        let t2 = [T2Volume::Uniform(60.0)];
        let ti = [T2Volume::Uniform(f32::INFINITY)];
        let centre = (nx / 2) + nx * (ny / 2);
        let mut peaks = Vec::new();
        for order in [KzOrder::Centric, KzOrder::Linear] {
            let tr = train(nz, 1, order, 15.0, 180.0);
            let ro = Readout3d::Grase { ky_segments: 1, t_line_ms: 0.5, reverse_phase: false };
            let run = |a: &Acquisition| simulate_acquisition_3d_complex([nx, ny, nz], [nx, ny, nz], 1, &images, &t2, None, &fmap,
                                                                       Some(&ti), a, &tr, &ro, None, None, &PhaseModel::none(), 0)
                .remove(0);
            let off = run(&Acquisition { do_relaxation: false, ..Acquisition::default() });
            let on = run(&Acquisition::default());
            let m0 = off[centre + nx * ny * z0];
            // a uniform slab has only the in-plane DC line, so the profile is
            // (1/nz) sum_p W(p, ky_c) exp(i 2 pi (p - pc)(z0 - z)/nz)
            let tab = grase_lines(&tr, &ro, ny, nz).unwrap();
            let pc = (nz / 2) as f64;
            let mut worst = 0.0f64;
            for z in 0..nz {
                let (mut re, mut im) = (0.0, 0.0);
                for p in 0..nz {
                    let w = (-tab.line(p, ny / 2).trf_ms / 60.0).exp();
                    let th = TAU * (p as f64 - pc) * (z0 as f64 - z as f64) / nz as f64;
                    re += w * th.cos() / nz as f64;
                    im += w * th.sin() / nz as f64;
                }
                let want = (m0.0 * re - m0.1 * im, m0.0 * im + m0.1 * re);
                let got = on[centre + nx * ny * z];
                worst = worst.max((got.0 - want.0).hypot(got.1 - want.1));
            }
            assert!(worst <= 1e-9 * m0.0.hypot(m0.1), "{order:?}: {worst:e}");
            let p = on[centre + nx * ny * z0];
            peaks.push(p.0.hypot(p.1));
        }
        // centric reads the kz centre first, with the least decay: a higher peak than linear
        assert!(peaks[0] > peaks[1], "{peaks:?}");
    }

    #[test]
    fn three_d_noise_is_the_2d_noise_over_root_nz() {
        let (nx, ny, nz, nseed) = (32usize, 32usize, 16usize, 6u64);
        let nvox = nx * ny * nz;
        let images = vec![vec![0.0f32; nvox]];
        let fmap = vec![0.0f32; nvox];
        let t2 = [T2Volume::Uniform(80.0)];
        let acq = Acquisition { noise_variance: 1.0, do_relaxation: false, ..Acquisition::default() };
        let tr = train(nz, 1, KzOrder::Centric, 20.0, 180.0);
        let ro = Readout3d::Grase { ky_segments: 1, t_line_ms: 0.5, reverse_phase: false };
        let (mut s3, mut s2, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for seed in 0..nseed {
            let v3 = simulate_acquisition_3d_complex([nx, ny, nz], [nx, ny, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr, &ro,
                                                     None, None, &PhaseModel::none(), seed).remove(0);
            let (m2, p2) = simulate_acquisition_oversampled([nx, ny, nz], [nx, ny, nz], 1, &images, &t2, &fmap, None, &acq,
                                                            &[None], &[None], &PhaseModel::none(), seed, None, None);
            for i in 0..nvox {
                s3 += v3[i].0 * v3[i].0;
                let re2 = m2[i] as f64 * (p2[i] as f64).cos();
                s2 += re2 * re2;
                n += 1.0;
            }
        }
        let (sd3, sd2) = ((s3 / n).sqrt(), (s2 / n).sqrt());
        let ratio = sd3 / sd2;
        let want = 1.0 / (nz as f64).sqrt();
        println!("3D/2D noise SD: {ratio:.4} (want {want:.4}); 2D SD {sd2:.4}");
        assert!((ratio / want - 1.0).abs() < 0.03, "{ratio} vs {want}");
    }

    #[test]
    fn shots_and_their_weights() {
        let (nx, ny, nz) = (16usize, 16usize, 4usize);
        let nvox = nx * ny * nz;
        // an object in the central half of the phase-encode field of view, so the FOV/2 ghost
        // does not overlap it
        let img: Vec<f32> = (0..nvox)
            .map(|vox| {
                let (x, y) = (vox % nx, (vox / nx) % ny);
                if (5..11).contains(&x) && (6..10).contains(&y) { 1.0 + 0.1 * x as f32 } else { 0.0 }
            })
            .collect();
        let images = vec![img.clone()];
        let fmap = vec![0.0f32; nvox];
        let t2 = [T2Volume::Uniform(80.0)];
        let acq = Acquisition { do_relaxation: false, ..Acquisition::default() };
        let tr = train(nz, 1, KzOrder::Centric, 20.0, 180.0);
        let grase = |ky_segments: usize| Readout3d::Grase { ky_segments, t_line_ms: 0.5, reverse_phase: false };
        let run = |ky_segments: usize, lw: Option<&LineWeights>| {
            simulate_acquisition_3d_complex([nx, ny, nz], [nx, ny, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr,
                                            &grase(ky_segments), lw, None, &PhaseModel::none(), 0).remove(0)
        };
        // the number of shots changes nothing when nothing differs between them
        let one = run(1, None);
        for s in [2usize, 4] {
            assert!(maxdiff(&one, &run(s, None)) <= 1e-12 * peak(&one), "{s} shots");
        }
        // two interleaved segments weighted w1, w2: a FOV/2 ghost of |w1 - w2| / (w1 + w2)
        let (w1, w2) = (1.0, 0.7);
        let v = run(2, Some(&LineWeights { n_shots: 2, n_compartments: 1, w: vec![w1, w2] }));
        let (x, y, z) = (8usize, 8usize, 1usize);
        let main = v[x + nx * (y + ny * z)];
        let ghost = v[x + nx * ((y + ny / 2) % ny + ny * z)];
        let ratio = ghost.0.hypot(ghost.1) / main.0.hypot(main.1);
        assert!((ratio - (w1 - w2) / (w1 + w2)).abs() < 1e-9, "{ratio}");
        // a dropped shot: its lines exactly zero at the observation point, the others unchanged
        let obs = |lw: Option<&LineWeights>, sets: Option<&[Vec<ShotSet>]>| {
            kspace_3d_observed([nx, ny, nz], [nx, ny, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr, &grase(2), lw, sets,
                               &PhaseModel::none(), 0).remove(0)
        };
        let full = obs(None, None);
        let dropped = obs(Some(&LineWeights { n_shots: 2, n_compartments: 1, w: vec![1.0, 0.0] }), None);
        // per-shot images: doubled images for shot 1 double exactly its lines
        let doubled: Vec<f32> = img.iter().map(|v| 2.0 * v).collect();
        let moved = obs(None, Some(&[vec![ShotSet { shots: vec![1], images: vec![doubled] }]]));
        for p in 0..nz {
            for ky in 0..ny {
                for kx in 0..nx {
                    let i = (p * ny + ky) * nx + kx;
                    if ky % 2 == 1 {
                        assert_eq!(dropped[i], (0.0, 0.0));
                        assert!((moved[i].0 - 2.0 * full[i].0).abs() <= 1e-12 && (moved[i].1 - 2.0 * full[i].1).abs() <= 1e-12);
                    } else {
                        assert_eq!(dropped[i], full[i]);
                        assert_eq!(moved[i], full[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn extreme_relaxation_times_give_finite_images() {
        let (nx, ny, nz) = (8usize, 8usize, 4usize);
        let nvox = nx * ny * nz;
        let images = blobs(nx, ny, nz, 1, 1);
        let fmap = vec![0.0f32; nvox];
        let tr = train(nz, 1, KzOrder::Centric, 10.0, 130.0);
        let ro = Readout3d::Grase { ky_segments: 1, t_line_ms: 0.5, reverse_phase: false };
        let tiny: Vec<f32> = (0..nvox).map(|i| if i % 2 == 0 { 1e-3 } else { 50.0 }).collect();
        for (t2, t1) in [(vec![T2Volume::Uniform(1e-3)], vec![T2Volume::Uniform(1000.0)]),
                         (vec![T2Volume::Uniform(f32::INFINITY)], vec![T2Volume::Uniform(f32::INFINITY)]),
                         (vec![T2Volume::Map(&tiny)], vec![T2Volume::Uniform(1000.0)])] {
            let v = simulate_acquisition_3d_complex([nx, ny, nz], [nx, ny, nz], 1, &images, &t2, Some(&t1), &fmap, None,
                                                    &Acquisition::default(), &tr, &ro, None, None, &PhaseModel::none(), 0);
            assert!(v[0].iter().all(|(a, b)| a.is_finite() && b.is_finite()));
        }
    }

    #[test]
    fn a_constant_map_reproduces_class_mode() {
        let (nx, ny, nz, o) = (12usize, 12usize, 6usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let nvox = snx * sny * nz;
        let images = blobs(snx, sny, nz, 1, 2);
        let fmap: Vec<f32> = (0..nvox).map(|i| 10.0 * ((i % snx) as f32 / snx as f32 - 0.5)).collect();
        let (t2m, t1m, tim) = (vec![70.0f32; nvox], vec![1300.0f32; nvox], vec![40.0f32; nvox]);
        let tr = train(nz, 2, KzOrder::Centric, 12.0, 130.0);
        let ro = Readout3d::Grase { ky_segments: 2, t_line_ms: 0.4, reverse_phase: false };
        let acq = Acquisition::default();
        let class = [T2Volume::Uniform(70.0), T2Volume::Uniform(120.0)];
        let class_t1 = [T2Volume::Uniform(1300.0), T2Volume::Uniform(1650.0)];
        let class_ti = [T2Volume::Uniform(40.0), T2Volume::Uniform(40.0)];
        let voxel = [T2Volume::Map(&t2m), T2Volume::Uniform(120.0)];
        let voxel_t1 = [T2Volume::Map(&t1m), T2Volume::Uniform(1650.0)];
        let voxel_ti = [T2Volume::Map(&tim), T2Volume::Uniform(40.0)];
        let t0 = std::time::Instant::now();
        let a = simulate_acquisition_3d_complex([snx, sny, nz], [nx, ny, nz], 1, &images, &class, Some(&class_t1), &fmap,
                                                Some(&class_ti), &acq, &tr, &ro, None, None, &PhaseModel::none(), 0).remove(0);
        let ta = t0.elapsed();
        let t0 = std::time::Instant::now();
        let b = simulate_acquisition_3d_complex([snx, sny, nz], [nx, ny, nz], 1, &images, &voxel, Some(&voxel_t1), &fmap,
                                                Some(&voxel_ti), &acq, &tr, &ro, None, None, &PhaseModel::none(), 0).remove(0);
        let tb = t0.elapsed();
        println!("class {ta:?}, voxel {tb:?}");
        let pk = peak(&a);
        let d = maxdiff(&a, &b);
        assert!(d <= 1e-6 * pk, "{d:e} of {pk:e}");
    }

    // ---- spirals ----

    #[cfg(feature = "kspace")]
    fn spiral_ro() -> Readout3d {
        Readout3d::Spiral { interleaves: 2, readout_ms: 4.0, dwell_ms: 0.01, radial_oversampling: 1.2 }
    }

    #[cfg(feature = "kspace")]
    fn spiral_fmap(snx: usize, sny: usize, nz: usize) -> Vec<f32> {
        (0..snx * sny * nz).map(|v| {
            let (x, y, z) = (v % snx, (v / snx) % sny, v / (snx * sny));
            (-15.0 + 30.0 * x as f64 / snx as f64 + 5.0 * z as f64 - 3.0 * y as f64 / sny as f64) as f32
        }).collect()
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn spiral_with_nothing_to_distinguish_partitions_3d_is_the_2d_spiral() {
        // relaxation off: every partition's weights are 1, so the z-DFT and its inverse cancel and
        // each slice's image is the 2D spiral pipeline's (segmented forward, least squares) on that
        // slice, fieldmap and coils included
        use crate::grid_recon::SpiralRecon;
        use crate::readout::spiral_trajectory;
        use crate::spiral::{segment_inputs, SegmentedForward};
        use crate::tseg::RateRect;
        let (n, nz, o, ncomp) = (16usize, 4usize, 2usize, 2usize);
        let s = n * o;
        let images = blobs(s, s, nz, 1, ncomp);
        let fmap = spiral_fmap(s, s, nz);
        let t2 = vec![T2Volume::Uniform(80.0); ncomp];
        let acq = Acquisition { do_relaxation: false, do_distortions: true, n_coils: 2, signal_scale: 100.0, ..Acquisition::default() };
        let tr = train(nz, 1, KzOrder::Centric, 12.0, 180.0);
        let phase = PhaseModel::none();
        let out = simulate_acquisition_3d_complex([s, s, nz], [n, n, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr,
                                                  &spiral_ro(), None, None, &phase, 3);
        let traj = spiral_trajectory(n, n, 2, 4.0, 0.01, 1.2).unwrap();
        let idx: Vec<usize> = (0..traj.k.len()).map(|j| j % traj.n_samples()).collect();
        let rec = SpiralRecon::new(&traj.k, n, acq.window);
        let nplane = s * s;
        let mut want = vec![(0.0, 0.0); n * n * nz];
        for z in 0..nz {
            let sl = &fmap[z * nplane..(z + 1) * nplane];
            let f = [sl.iter().fold(f32::MAX, |a, &b| a.min(b)) as f64, sl.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64];
            let fwd = SegmentedForward::new(&traj.k, &idx, &traj.tau_ms, 4.0, [s, s], [n, n], RateRect { d: [0.0, 0.0], f }).unwrap();
            let phi = phase_slice(&phase, &zero_shot(), s, s, o, z, nz);
            let coils: Vec<Vec<C>> = (0..2).map(|q| {
                let mut acc = vec![C::ZERO; traj.k.len()];
                for im in &images {
                    let pl: Vec<f32> = im[z * nplane..(z + 1) * nplane].to_vec();
                    let refs = [pl.as_slice()];
                    let t2s = [T2Slice::Uniform(80.0)];
                    let inp = SliceInput { compartments: &refs, t2: &t2s, t_inhom: None, fmap: sl, phase0: Some(&phi), sim: [s, s],
                                           acq_matrix: [n, n], z, nz, eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None };
                    let (x, r) = segment_inputs(&inp, &acq, q, 2, None, false);
                    for (a, b) in acc.iter_mut().zip(fwd.apply(&x, &r)) {
                        *a = a.add(b);
                    }
                }
                acc
            }).collect();
            for (i, c) in rec.reconstruct(&coils, &acq).iter().enumerate() {
                want[i + n * n * z] = (c.re, c.im);
            }
        }
        let pk = peak(&want);
        let d = maxdiff(&out[0], &want);
        assert!(d <= 1e-9 * pk, "{d:e} of {pk:e}");
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn spiral_class_and_voxel_modes_agree() {
        // the same uniform relaxation as uniform volumes (class: decay outside the segmentation)
        // and as maps (voxel: decay inside, ln A per echo), refocusing below 180 degrees
        let (n, nz, o, ncomp) = (16usize, 4usize, 2usize, 2usize);
        let s = n * o;
        let nvox = s * s * nz;
        let images = blobs(s, s, nz, 1, ncomp);
        let fmap = spiral_fmap(s, s, nz);
        let acq = Acquisition { do_relaxation: true, do_distortions: true, signal_scale: 100.0, t_inhom: 50.0, ..Acquisition::default() };
        let tr = train(nz, 2, KzOrder::Centric, 12.0, 130.0);
        let phase = PhaseModel::none();
        let (t2a, t2b, t1a, t1b) = (vec![80.0f32; nvox], vec![120.0f32; nvox], vec![1300.0f32; nvox], vec![1600.0f32; nvox]);
        let tpm = vec![50.0f32; nvox];
        let class_t2 = [T2Volume::Uniform(80.0), T2Volume::Uniform(120.0)];
        let class_t1 = [T2Volume::Uniform(1300.0), T2Volume::Uniform(1600.0)];
        let vox_t2 = [T2Volume::Map(&t2a), T2Volume::Map(&t2b)];
        let vox_t1 = [T2Volume::Map(&t1a), T2Volume::Map(&t1b)];
        let vox_ti = [T2Volume::Map(&tpm), T2Volume::Map(&tpm)];
        let a = simulate_acquisition_3d_complex([s, s, nz], [n, n, nz], 1, &images, &class_t2, Some(&class_t1), &fmap, None,
                                                &acq, &tr, &spiral_ro(), None, None, &phase, 3);
        let b = simulate_acquisition_3d_complex([s, s, nz], [n, n, nz], 1, &images, &vox_t2, Some(&vox_t1), &fmap, Some(&vox_ti),
                                                &acq, &tr, &spiral_ro(), None, None, &phase, 3);
        let pk = peak(&a[0]);
        let d = maxdiff(&a[0], &b[0]);
        println!("spiral class vs voxel: {:.2e} of peak", d / pk);
        assert!(d <= 1e-6 * pk, "{d:e} of {pk:e}");
        let segs = spiral_segmentation([s, s, nz], [n, n, nz], &vox_t2, Some(&vox_t1), &fmap, Some(&vox_ti), &acq, &tr,
                                       &spiral_ro()).unwrap();
        assert_eq!(segs.len(), nz);
        assert!(segs.iter().all(|g| g.voxel && g.bound < 1e-7 && g.l <= 64));
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn spiral_3d_is_linear_and_line_weights_scale_their_shots() {
        let (n, nz, o, ncomp) = (16usize, 4usize, 2usize, 2usize);
        let s = n * o;
        let nvox = s * s * nz;
        let images = blobs(s, s, nz, 1, ncomp);
        let fmap = spiral_fmap(s, s, nz);
        let t2 = vec![T2Volume::Uniform(80.0), T2Volume::Uniform(110.0)];
        let t1 = vec![T2Volume::Uniform(1400.0); ncomp];
        let acq = Acquisition { do_relaxation: true, do_distortions: true, n_coils: 2, signal_scale: 100.0, ..Acquisition::default() };
        let tr = train(nz, 2, KzOrder::Centric, 12.0, 150.0);
        let phase = PhaseModel::none();
        let run = |ims: &[Vec<f32>], lw: Option<&LineWeights>| {
            simulate_acquisition_3d_complex([s, s, nz], [n, n, nz], 1, ims, &t2, Some(&t1), &fmap, None, &acq, &tr, &spiral_ro(),
                                            lw, None, &phase, 3).remove(0)
        };
        // linearity in the images
        let half: Vec<Vec<f32>> = images.iter().map(|v| v.iter().map(|x| x * 0.3).collect()).collect();
        let rest: Vec<Vec<f32>> = images.iter().zip(&half).map(|(v, h)| v.iter().zip(h).map(|(a, b)| a - b).collect()).collect();
        let (full, a, b) = (run(&images, None), run(&half, None), run(&rest, None));
        let sum: Vec<(f64, f64)> = a.iter().zip(&b).map(|(p, q)| (p.0 + q.0, p.1 + q.1)).collect();
        let pk = peak(&full);
        assert!(maxdiff(&full, &sum) <= 1e-6 * pk, "{:e}", maxdiff(&full, &sum) / pk);
        // 2 interleaves x 2 kz segments = 4 shots; weight 3 on every shot is three times the image
        let lw = LineWeights { n_shots: 4, n_compartments: ncomp, w: vec![3.0; 4 * ncomp] };
        let tripled = run(&images, Some(&lw));
        let want: Vec<(f64, f64)> = full.iter().map(|p| (3.0 * p.0, 3.0 * p.1)).collect();
        assert!(maxdiff(&tripled, &want) <= 1e-9 * peak(&want));
        // a weight on one shot only changes the image
        let mut w1 = vec![1.0; 4 * ncomp];
        w1[2 * ncomp] = 0.5;
        let one = run(&images, Some(&LineWeights { n_shots: 4, n_compartments: ncomp, w: w1 }));
        assert!(maxdiff(&one, &full) > 1e-3 * pk);
        let _ = nvox;
    }

    #[test]
    fn spike_streams_are_per_coil() {
        // (volume, partition, coil) streams: the same seed and coil repeat, another coil differs
        let acquired: Vec<usize> = (0..256).collect();
        let (a, b, c) = (spike_picks(77, 0, &acquired, 8), spike_picks(77, 0, &acquired, 8), spike_picks(77, 1, &acquired, 8));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn spiral_shots_select_exactly_their_samples() {
        // on the acquired samples, with 3 interleaves x 2 kz segments (unequal, so a swapped
        // shot formula would show): a zero weight on one shot removes exactly that shot's samples
        // (interleaf s at the partitions of kz segment sz), and a shot set with doubled images
        // doubles exactly its own
        let (n, nz, o, ncomp) = (16usize, 4usize, 2usize, 2usize);
        let s = n * o;
        let images = blobs(s, s, nz, 1, ncomp);
        let fmap = spiral_fmap(s, s, nz);
        let t2 = vec![T2Volume::Uniform(80.0), T2Volume::Uniform(110.0)];
        let acq = Acquisition { do_relaxation: true, do_distortions: true, signal_scale: 100.0, ..Acquisition::default() };
        let tr = train(nz, 2, KzOrder::Centric, 12.0, 180.0);
        let ro = Readout3d::Spiral { interleaves: 3, readout_ms: 4.0, dwell_ms: 0.01, radial_oversampling: 1.2 };
        let phase = PhaseModel::none();
        let pl = spiral_plan([s, s, nz], [n, n, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr, &ro, None, None, &phase);
        assert_eq!(pl.table.n_shots, 6);
        let base = pl.samples(&images, 1, 0, 0, 1, None, &[]);
        let shot_of = |i: usize| pl.table.shot(i / pl.nsamp, (i % pl.nsamp) / pl.ns);
        // zero shot 3 = interleaf 1, kz segment 1
        let mut w = vec![1.0; 6 * ncomp];
        w[3 * ncomp] = 0.0;
        w[3 * ncomp + 1] = 0.0;
        let zeroed = pl.samples(&images, 1, 0, 0, 1, Some(&LineWeights { n_shots: 6, n_compartments: ncomp, w }), &[]);
        let mut hit = 0;
        for i in 0..base.len() {
            if shot_of(i) == 3 {
                hit += 1;
                assert!(zeroed[i].re == 0.0 && zeroed[i].im == 0.0, "sample {i}");
                assert!((i % pl.nsamp) / pl.ns == 1 && pl.table.kz_segment[i / pl.nsamp] == 1);
            } else {
                assert!(zeroed[i].re == base[i].re && zeroed[i].im == base[i].im, "sample {i}");
            }
        }
        assert_eq!(hit, pl.ns * nz / 2, "one interleaf over half the partitions");
        // shot 4 (interleaf 2, kz segment 0) sees doubled images
        let doubled: Vec<Vec<f32>> = images.iter().map(|v| v.iter().map(|x| 2.0 * x).collect()).collect();
        let sets = [ShotSet { shots: vec![4], images: doubled }];
        let moved = pl.samples(&images, 1, 0, 0, 1, None, &sets);
        for i in 0..base.len() {
            let f = if shot_of(i) == 4 { 2.0 } else { 1.0 };
            assert!(moved[i].re == f * base[i].re && moved[i].im == f * base[i].im, "sample {i}");
        }
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn spiral_noise_is_seeded_and_measured() {
        let (n, nz, o) = (16usize, 4usize, 2usize);
        let s = n * o;
        let images = vec![vec![0.0f32; s * s * nz]];
        let fmap = vec![0.0f32; s * s * nz];
        let t2 = [T2Volume::Uniform(80.0)];
        let acq = Acquisition { do_relaxation: false, noise_variance: 4.0, ..Acquisition::default() };
        let tr = train(nz, 1, KzOrder::Centric, 12.0, 180.0);
        let phase = PhaseModel::none();
        let run = |seed| simulate_acquisition_3d_complex([s, s, nz], [n, n, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr,
                                                         &spiral_ro(), None, None, &phase, seed).remove(0);
        let (a, b, c) = (run(5), run(5), run(6));
        assert_eq!(a, b);
        assert_ne!(a, c);
        let sd = (a.iter().map(|p| p.0 * p.0).sum::<f64>() / a.len() as f64).sqrt();
        // Cartesian 3D: per-sample SD sqrt(var / n^2), an unnormalized n^2 sum, 1/sqrt(nz) from the z-DFT
        let cart = (acq.noise_variance / (n * n) as f64).sqrt() * n as f64 / (nz as f64).sqrt();
        println!("spiral 3D noise SD {sd:.4}, Cartesian 3D {cart:.4}, ratio {:.4}", sd / cart);
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn spiral_refusals() {
        let (n, nz, o) = (16usize, 4usize, 2usize);
        let s = n * o;
        let images = vec![vec![0.0f32; s * s * nz]];
        let fmap = vec![0.0f32; s * s * nz];
        let t2 = [T2Volume::Uniform(80.0)];
        let tr = train(nz, 1, KzOrder::Centric, 12.0, 180.0);
        let phase = PhaseModel::none();
        for (acq, msg) in [
            (Acquisition { n_spikes: 2, ..Acquisition::default() }, "spikes"),
            (Acquisition { accel: 2, ..Acquisition::default() }, "GRAPPA"),
            (Acquisition { partial_fourier: 0.75, ..Acquisition::default() }, "partial Fourier"),
            (Acquisition { ghost_offset: 0.1, ..Acquisition::default() }, "ghosting"),
        ] {
            let r = std::panic::catch_unwind(|| {
                simulate_acquisition_3d_complex([s, s, nz], [n, n, nz], 1, &images, &t2, None, &fmap, None, &acq, &tr,
                                                &spiral_ro(), None, None, &phase, 1)
            });
            let e = r.expect_err(msg);
            let text = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap();
            assert!(text.contains(msg), "{text}");
        }
        // a fieldmap range no segmentation within 64 can certify is reported by name
        let wild: Vec<f32> = (0..s * s * nz).map(|v| if v % 2 == 0 { -9000.0 } else { 9000.0 }).collect();
        let acq = Acquisition { do_distortions: true, ..Acquisition::default() };
        let e = spiral_segmentation([s, s, nz], [n, n, nz], &t2, None, &wild, None, &acq, &tr, &spiral_ro()).unwrap_err();
        assert!(e.contains("slice 0") && e.contains("64 segments"), "{e}");
    }

    // ---- P7 part C: the 3D gradient-echo train

    /// A fieldmap with a linear gradient and a bump (Hz), not constant, on the simulation grid.
    fn varying_fmap(snx: usize, sny: usize, nz: usize) -> Vec<f32> {
        (0..snx * sny * nz).map(|v| {
            let (x, y, z) = (v % snx, (v / snx) % sny, v / (snx * sny));
            let (fx, fy) = (x as f64 / snx as f64 - 0.5, y as f64 / sny as f64 - 0.5);
            (25.0 * fx - 10.0 * fy + 15.0 * (-(fx * fx + fy * fy) / 0.02).exp() + 2.0 * z as f64) as f32
        }).collect()
    }

    fn ge_train3(kz_segments: usize, tes: Vec<f64>) -> ExcitationTrain {
        ExcitationTrain { kz_segments, kz_order: KzOrder::Linear, exc_spacing_ms: 60.0, echo_times_ms: tes, excitation_time_ms: 2.0 }
    }

    fn ge_acq(n_coils: usize) -> Acquisition {
        Acquisition { echo: EchoFormation::Gradient, n_coils, signal_scale: 100.0, t_inhom: 45.0, ..Acquisition::default() }
    }

    /// The volume of `img` (one volume, voxel layout) with weight `w(p, sy, c)`.
    fn ge_volume(img: &[Vec<f32>], nz: usize, ky_segments: usize, w: impl Fn(usize, usize, usize) -> f64) -> GeVolume {
        let ncomp = img.len();
        let mut weights = vec![0.0; nz * ky_segments * ncomp];
        for p in 0..nz {
            for sy in 0..ky_segments {
                for c in 0..ncomp {
                    weights[(p * ky_segments + sy) * ncomp + c] = w(p, sy, c);
                }
            }
        }
        GeVolume { images: img.to_vec(), weights, shot_images: None }
    }

    /// The reference: per slice and coil, the 2D forward with gradient-echo formation at `TE`, its
    /// own decay at `trf = TE + t` (the block's line times), the fieldmap's static and readout
    /// phase; the 3D k-space as the z-DFT of those, each partition times `w(p)`, back by the inverse
    /// z-DFT; each slice through the 2D reconstruction. No code of the 3D gradient-echo path.
    #[allow(clippy::too_many_arguments)]
    fn ge_reference(img: &[f32], t2: f32, ti: f32, fmap: &[f32], acq: &Acquisition, te: f64, ro: &Ge3dReadout,
                    dims: [usize; 3], o: usize, w: &dyn Fn(usize) -> f64) -> Vec<(f64, f64)> {
        let [nx, ny, nz] = dims;
        let (snx, sny) = (nx * o, ny * o);
        let nplane = snx * sny;
        let block = crate::readout::grase_block(ny, ro.ky_segments, ro.t_line_ms, ro.reverse_phase).unwrap();
        let timing = LineTiming {
            t_ms: block.t_ms.clone(), trf_ms: block.t_ms.iter().map(|t| te + t).collect(), tread_ms: block.t_ms.clone(),
            polarity: block.polarity.clone(),
        };
        let a = Acquisition { t_echo: te, do_relaxation: true, noise_variance: 0.0, n_spikes: 0, ..acq.clone() };
        let phase = PhaseModel::none();
        let ncoils = acq.n_coils.max(1);
        let zk = zkernel(nz);
        let mut coil_parts = Vec::new();
        for q in 0..ncoils {
            let k2: Vec<Vec<C>> = (0..nz).map(|z| {
                let pl = &img[z * nplane..(z + 1) * nplane];
                let phi = phase_slice(&phase, &zero_shot(), snx, sny, o, z, nz);
                let refs = [pl];
                let t2s = [T2Slice::Uniform(t2)];
                let tis = [T2Slice::Uniform(ti)];
                let inp = SliceInput {
                    compartments: &refs, t2: &t2s, t_inhom: Some(&tis), fmap: &fmap[z * nplane..(z + 1) * nplane],
                    phase0: Some(&phi), sim: [snx, sny], acq_matrix: [nx, ny], z, nz, eddy_drive: None, prep_drive: None,
                    slice_seed: 0, eddy_lin: None,
                };
                build_coil_kspace_timed(&inp, &a, q, ncoils, &timing, None)
            }).collect();
            let mut slices = vec![vec![C::ZERO; ny * nx]; nz];
            for (z, sl) in slices.iter_mut().enumerate() {
                for p in 0..nz {
                    let kc = zk[p * nz + z];
                    let conj = C { re: kc.re, im: -kc.im };
                    for i in 0..ny * nx {
                        let mut kp = C::ZERO;
                        for (zz, k2z) in k2.iter().enumerate() {
                            kp = kp.add(k2z[i].mul(zk[p * nz + zz]));
                        }
                        sl[i] = sl[i].add(kp.scale(w(p)).mul(conj));
                    }
                }
                for c in sl.iter_mut() {
                    *c = c.scale(1.0 / nz as f64);
                }
            }
            coil_parts.push(slices);
        }
        let mut out = vec![(0.0, 0.0); nx * ny * nz];
        for z in 0..nz {
            let coils: Vec<Vec<C>> = coil_parts.iter().map(|cp| cp[z].clone()).collect();
            for (i, c) in reconstruct_coils(coils, acq, [nx, ny]).iter().enumerate() {
                out[i + nx * ny * z] = (c.re, c.im);
            }
        }
        out
    }

    /// One shot, constant weights, two echoes, a varying fieldmap: each echo equals the
    /// reference, in class mode (uniform T2) and voxel mode (a T2 map). A fieldmap phase taken as
    /// a line weight (its mean, applied uniformly) is far from it: the static phase is per voxel.
    #[test]
    fn ge3d_matches_the_reference_forward() {
        let (nx, ny, nz, o) = (16usize, 16usize, 4usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let nvox = snx * sny * nz;
        let img: Vec<f32> = blobs(snx, sny, nz, 1, 1).remove(0);
        let fmap = varying_fmap(snx, sny, nz);
        let ro = Ge3dReadout { ky_segments: 1, t_line_ms: 0.4, reverse_phase: false };
        let tes = vec![12.0, 30.0];
        let tr = ge_train3(1, tes.clone());
        let t2map = vec![80.0f32; nvox];
        let phase = PhaseModel::none();
        for n_coils in [1usize, 3] {
            let acq = ge_acq(n_coils);
            for (mode, t2v) in [("class", T2Volume::Uniform(80.0)), ("voxel", T2Volume::Map(&t2map))] {
                let t2 = [t2v];
                let ti = [T2Volume::Uniform(45.0)];
                let vol = |_g: usize| ge_volume(std::slice::from_ref(&img), nz, 1, |_, _, _| 0.7);
                let got = simulate_acquisition_3d_ge_complex([snx, sny, nz], [nx, ny, nz], 1, &t2, &fmap, Some(&ti), &acq,
                                                             &tr, &ro, &vol, &phase, 5);
                for (e, &te) in tes.iter().enumerate() {
                    let want = ge_reference(&img, 80.0, 45.0, &fmap, &acq, te, &ro, [nx, ny, nz], o, &|_| 0.7);
                    let d = maxdiff(&got[0][e], &want);
                    assert!(d <= 1e-9 * peak(&want), "{mode} coils {n_coils} echo {e}: {d:e} of {:e}", peak(&want));
                }
            }
            // the negative control: the fieldmap's phase as one line weight (the mean) misses it
            let zero = vec![0.0f32; nvox];
            let mean = fmap.iter().map(|&f| f as f64).sum::<f64>() / nvox as f64;
            let flat = ge_reference(&img, 80.0, 45.0, &zero, &acq, 30.0, &ro, [nx, ny, nz], o, &|_| 0.7);
            let rot = C { re: (TAU * mean * 0.030).cos(), im: (TAU * mean * 0.030).sin() };
            let as_weight: Vec<(f64, f64)> = flat.iter().map(|&(a, b)| { let c = C { re: a, im: b }.mul(rot); (c.re, c.im) }).collect();
            let want = ge_reference(&img, 80.0, 45.0, &fmap, &acq, 30.0, &ro, [nx, ny, nz], o, &|_| 0.7);
            assert!(maxdiff(&as_weight, &want) > 0.05 * peak(&want), "the fieldmap's phase is not per voxel here");
        }
    }

    /// Weights that vary across partitions blur through-plane: the result is the inverse z-DFT of
    /// `w_p` times the z-DFT of the object (the reference with `w(p)`), and differs from constant
    /// weights.
    #[test]
    fn ge3d_weights_across_partitions_blur_through_plane() {
        let (nx, ny, nz, o) = (12usize, 12usize, 6usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let img: Vec<f32> = blobs(snx, sny, nz, 1, 1).remove(0);
        let fmap = varying_fmap(snx, sny, nz);
        let ro = Ge3dReadout { ky_segments: 1, t_line_ms: 0.4, reverse_phase: false };
        let tr = ge_train3(2, vec![12.0]);
        let acq = ge_acq(2);
        let w = |p: usize| 1.0 + 0.5 * (p as f64 * 1.3).cos();
        let vol = |_g: usize| ge_volume(std::slice::from_ref(&img), nz, 1, |p, _, _| w(p));
        let got = simulate_acquisition_3d_ge_complex([snx, sny, nz], [nx, ny, nz], 1, &[T2Volume::Uniform(80.0)], &fmap,
                                                     Some(&[T2Volume::Uniform(45.0)]), &acq, &tr, &ro, &vol, &PhaseModel::none(), 5);
        let want = ge_reference(&img, 80.0, 45.0, &fmap, &acq, 12.0, &ro, [nx, ny, nz], o, &w);
        assert!(maxdiff(&got[0][0], &want) <= 1e-9 * peak(&want));
        let flat = ge_reference(&img, 80.0, 45.0, &fmap, &acq, 12.0, &ro, [nx, ny, nz], o, &|_| 1.0);
        assert!(maxdiff(&flat, &want) > 0.01 * peak(&want));
    }

    /// Echo 0 of a two-echo call is the one-echo call bit for bit (noise and spikes on); echo 1
    /// with noise off is the one-echo call at its TE; the noise of the two echoes is uncorrelated.
    #[test]
    fn ge3d_echoes() {
        let (nx, ny, nz, o, nv) = (12usize, 12usize, 4usize, 2usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let imgs = blobs(snx, sny, nz, nv, 1);
        let one_vol = |g: usize| -> Vec<f32> { (0..snx * sny * nz).map(|v| imgs[0][v * nv + g]).collect() };
        let fmap = varying_fmap(snx, sny, nz);
        let ro = Ge3dReadout { ky_segments: 2, t_line_ms: 0.4, reverse_phase: false };
        let t2 = [T2Volume::Uniform(80.0)];
        let run = |tes: Vec<f64>, acq: &Acquisition| {
            let vol = |g: usize| ge_volume(&[one_vol(g)], nz, 2, |p, sy, _| 0.5 + 0.1 * p as f64 + 0.05 * sy as f64);
            simulate_acquisition_3d_ge([snx, sny, nz], [nx, ny, nz], nv, &t2, &fmap, None, acq, &ge_train3(2, tes), &ro,
                                       &vol, &PhaseModel::none(), 77)
        };
        let noisy = Acquisition { noise_variance: 0.5, n_spikes: 2, ..ge_acq(2) };
        let two = run(vec![12.0, 30.0], &noisy);
        let one = run(vec![12.0], &noisy);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!((bits(&two[0].0), bits(&two[0].1)), (bits(&one[0].0), bits(&one[0].1)));
        let quiet = ge_acq(2);
        let q2 = run(vec![12.0, 30.0], &quiet);
        let q1 = run(vec![30.0], &quiet);
        assert_eq!((bits(&q2[1].0), bits(&q2[1].1)), (bits(&q1[0].0), bits(&q1[0].1)));
        // the noise residuals of echoes 0 and 1 at the same echo time: uncorrelated
        let n_only = Acquisition { noise_variance: 0.5, ..ge_acq(2) };
        let a = run(vec![12.0, 12.0], &n_only);
        let b = run(vec![12.0, 12.0], &quiet);
        let resid = |e: usize| -> Vec<(f64, f64)> {
            (0..a[e].0.len()).map(|i| {
                let c = |m: f32, p: f32| (m as f64 * (p as f64).cos(), m as f64 * (p as f64).sin());
                let (x, y) = (c(a[e].0[i], a[e].1[i]), c(b[e].0[i], b[e].1[i]));
                (x.0 - y.0, x.1 - y.1)
            }).collect()
        };
        let (r0, r1) = (resid(0), resid(1));
        let dot: f64 = r0.iter().zip(&r1).map(|(u, v)| u.0 * v.0 + u.1 * v.1).sum();
        let n0: f64 = r0.iter().map(|u| u.0 * u.0 + u.1 * u.1).sum();
        let n1: f64 = r1.iter().map(|u| u.0 * u.0 + u.1 * u.1).sum();
        let corr = dot / (n0 * n1).sqrt();
        assert!(n0 > 0.0 && corr.abs() < 3.0 / ((2 * r0.len()) as f64).sqrt(), "echo noise correlation {corr}");
    }

    /// `volume(g)` is called once per volume and its result, under one thread and under many, equals
    /// the same call fed precomputed volumes, bit for bit (noise and spikes on).
    #[test]
    fn ge3d_volumes_on_demand() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (nx, ny, nz, o, nv) = (12usize, 12usize, 4usize, 2usize, 5usize);
        let (snx, sny) = (nx * o, ny * o);
        let imgs = blobs(snx, sny, nz, nv, 2);
        let pre: Vec<GeVolume> = (0..nv).map(|g| {
            let im: Vec<Vec<f32>> = imgs.iter().map(|c| (0..snx * sny * nz).map(|v| c[v * nv + g]).collect()).collect();
            ge_volume(&im, nz, 1, |p, _, c| 0.3 + 0.1 * (p + c + g) as f64)
        }).collect();
        let fmap = varying_fmap(snx, sny, nz);
        let t2 = [T2Volume::Uniform(80.0), T2Volume::Uniform(120.0)];
        let ro = Ge3dReadout { ky_segments: 1, t_line_ms: 0.4, reverse_phase: false };
        let acq = Acquisition { noise_variance: 0.2, n_spikes: 1, ..ge_acq(2) };
        let calls = AtomicUsize::new(0);
        let vol = |g: usize| { calls.fetch_add(1, Ordering::SeqCst); pre[g].clone() };
        let run = || simulate_acquisition_3d_ge([snx, sny, nz], [nx, ny, nz], nv, &t2, &fmap, None, &acq,
                                                &ge_train3(1, vec![12.0]), &ro, &vol, &PhaseModel::none(), 3);
        let a = run();
        assert_eq!(calls.load(Ordering::SeqCst), nv);
        #[cfg(feature = "par")]
        let b = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(run);
        #[cfg(not(feature = "par"))]
        let b = run();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!((bits(&a[0].0), bits(&a[0].1)), (bits(&b[0].0), bits(&b[0].1)));
        // an unused compartment (empty image, zero weights) is skipped and changes nothing
        let pre2: Vec<GeVolume> = pre.iter().map(|v| {
            let mut images = v.images.clone();
            images.push(Vec::new());
            let mut weights = Vec::new();
            for x in v.weights.chunks(2) {
                weights.extend_from_slice(x);
                weights.push(0.0);
            }
            GeVolume { images, weights, shot_images: None }
        }).collect();
        let t2b = [T2Volume::Uniform(80.0), T2Volume::Uniform(120.0), T2Volume::Uniform(60.0)];
        let c = simulate_acquisition_3d_ge([snx, sny, nz], [nx, ny, nz], nv, &t2b, &fmap, None, &acq,
                                           &ge_train3(1, vec![12.0]), &ro, &|g| pre2[g].clone(), &PhaseModel::none(), 3);
        assert_eq!((bits(&a[0].0), bits(&a[0].1)), (bits(&c[0].0), bits(&c[0].1)));
    }

    /// The refusals: a compartment count other than `t2`'s, a spin-echo acquisition, the eddy model,
    /// partial Fourier, a weight on a compartment without an image.
    #[test]
    fn ge3d_refusals() {
        let (n, nz, o) = (12usize, 4usize, 2usize);
        let s = n * o;
        let img = vec![1.0f32; s * s * nz];
        let fmap = vec![0.0f32; s * s * nz];
        let ro = Ge3dReadout { ky_segments: 1, t_line_ms: 0.4, reverse_phase: false };
        let tr = ge_train3(1, vec![12.0]);
        let t2 = [T2Volume::Uniform(80.0)];
        let fail = |acq: Acquisition, v: GeVolume, msg: &str| {
            let r = std::panic::catch_unwind(|| {
                simulate_acquisition_3d_ge_complex([s, s, nz], [n, n, nz], 1, &t2, &fmap, None, &acq, &tr, &ro, &|_| v.clone(),
                                                   &PhaseModel::none(), 1)
            });
            let e = r.expect_err(msg);
            let text = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap();
            assert!(text.contains(msg), "{text}");
        };
        let ok = ge_volume(std::slice::from_ref(&img), nz, 1, |_, _, _| 1.0);
        fail(ge_acq(1), ge_volume(&[img.clone(), img.clone()], nz, 1, |_, _, _| 1.0), "compartment images");
        fail(Acquisition { echo: EchoFormation::Spin, ..ge_acq(1) }, ok.clone(), "gradient-echo acquisition");
        fail(Acquisition { eddy_strength: 1.0, ..ge_acq(1) }, ok.clone(), "eddy");
        fail(Acquisition { partial_fourier: 0.75, ..ge_acq(1) }, ok.clone(), "partial Fourier");
        fail(ge_acq(1), GeVolume { images: vec![Vec::new()], ..ok.clone() }, "no image but a nonzero weight");
    }
}
