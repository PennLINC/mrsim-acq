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
use crate::readout::{grase_lines, EchoTrain, Grase3dTable, Readout3d};

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
    Plan {
        snx, sny, nz, nx, ny, o, ncomp, acq, table, within, modes, ln_a_maps, t2, t_inhom, fmap, phase,
        zk: zkernel(nz), mask: sampling_mask(nx, ny, acq),
    }
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
                    let acquired: Vec<usize> = (0..nx * ny).filter(|&i| pl.mask[i]).collect();
                    let mut rng = Rng(pseed ^ 0xA5A5_1234_5678_9ABC);
                    for _ in 0..acq.n_spikes {
                        let pick = acquired[(rng.next_u64() as usize) % acquired.len()];
                        part[pick] = peak.scale(acq.spike_amplitude);
                    }
                }
                if acq.noise_variance > 0.0 {
                    let mut rng = Rng((pseed ^ (q as u64).wrapping_mul(0x9E37_79B9)) | 1);
                    for (i, c) in part.iter_mut().enumerate() {
                        if pl.mask[i] {
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
                    let kc = pl.zk[p * nz + z];
                    let conj = C { re: kc.re, im: -kc.im };
                    for i in 0..ny * nx {
                        sl[i] = sl[i].add(k[p * ny * nx + i].mul(conj));
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
            let img = reconstruct_coils(coils, acq, [nx, ny]);
            for (i, c) in img.iter().enumerate() {
                out[i + nx * ny * z] = (c.re, c.im);
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
}
