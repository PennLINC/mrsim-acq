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

/// The decay of compartment `c` at voxel `i` and time `tau_ms` from the echo, with the optional
/// log-amplitude map, as one exponent.
pub(crate) fn decay(inp: &SliceInput, acq: &Acquisition, log_amp: Option<&[f64]>, c: usize, i: usize, tau_ms: f64) -> f64 {
    if !acq.do_relaxation {
        return 1.0;
    }
    let t2 = match inp.t2[c] { T2Slice::Uniform(v) => v as f64, T2Slice::Map(m) => m[i] as f64 };
    let tp = match inp.t_inhom.map(|ti| ti[c]) {
        None => acq.t_inhom,
        Some(T2Slice::Uniform(v)) => v as f64,
        Some(T2Slice::Map(m)) => m[i] as f64,
    };
    (log_amp.map_or(0.0, |la| la[i]) - tau_ms / t2 - tau_ms / tp).exp()
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
}
