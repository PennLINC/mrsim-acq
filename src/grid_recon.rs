//! The spiral reconstruction (P5 addendum, part C, "Forward and reconstruction", as amended after
//! the feasibility benchmark).
//!
//! Per coil and partition, density-weighted least squares on the `n x n` acquired grid,
//! `min sum_j w_j |(A x)_j - d_j|^2`, with `A` the forward of an object on the acquired grid (the
//! type-2 NUFFT in the Cartesian forward's convention, `1/n^2`) and `A^H` its adjoint (the type-1
//! NUFFT with the opposite sign, `img(X) = sum_j r_j exp(-i 2 pi k_j . (X - n/2) / n)`,
//! deapodized). The solver is the Chebyshev semi-iteration on `A^H W A x = A^H W d` from zero over
//! `[lambda_hi / LS_KAPPA, lambda_hi]`, [`LS_ITERATIONS`] iterations, with `lambda_hi` from a
//! fixed power iteration: every coefficient depends only on the trajectory, so the reconstruction
//! is linear in the data (the linearity identity needs that; conjugate gradients is not), and its
//! residual polynomial is bounded by 1 on `[0, lambda_hi]`. Then the Roemer combine of
//! `kspace::reconstruct_coils`.
//!
//! The density weights are Pipe and Menon's (1999) fixed point in operator form,
//! `w <- w / |A A^H w|`, [`DCF_ITERATIONS`] iterations from `w = 1`, scaled so that a constant
//! object with no off-resonance and no decay grids (`A^H W d` up to the scale) to its Cartesian
//! value at the image centre. The in-plane window multiplies the samples before the least squares
//! by `KspaceWindow::at` of their frequency, the radial function the Cartesian path applies on its
//! grid.
//!
//! Gridding alone (the first iterate, up to a scale) misses the declared accuracy on the designed
//! trajectories, whose interleaves are exactly Nyquist-spaced radially: 4e-2 of peak on a
//! band-limited object, 0.3-0.5 on a uniform one (plan Measurements, Task 12).
#![cfg(feature = "kspace")]

use std::f64::consts::TAU;

use crate::kspace::{coil_sensitivity, Acquisition, KspaceWindow, C};
use crate::nufft::Nufft2;

/// Pipe-Menon iterations (fixed, recorded).
pub const DCF_ITERATIONS: usize = 10;
/// Chebyshev iterations of the least squares (fixed, recorded).
pub const LS_ITERATIONS: usize = 40;
/// The ratio of the Chebyshev interval's ends.
pub const LS_KAPPA: f64 = 30.0;
/// Power iterations for the largest eigenvalue of the normal operator, and the margin over it.
pub const POWER_ITERATIONS: usize = 50;
pub const LAMBDA_MARGIN: f64 = 1.1;
/// The NUFFT's tolerance.
const NUFFT_TOL: f64 = 1e-12;

/// Pipe-Menon density weights (unnormalized) of the samples `k` for the gridding operator `nufft`.
pub fn pipe_menon(nufft: &Nufft2, k: &[[f64; 2]], iterations: usize) -> Vec<f64> {
    let mut w = vec![1.0; k.len()];
    for _ in 0..iterations {
        let wts: Vec<(f64, f64)> = w.iter().map(|&x| (x, 0.0)).collect();
        let img = nufft.type1(k, &wts, -1);
        let back = nufft.type2(&img, k, 1);
        for (wi, (re, im)) in w.iter_mut().zip(back) {
            *wi /= re.hypot(im);
        }
    }
    w
}

/// The 1D Dirichlet kernel of a constant object on `n` acquired voxels at frequency `k`:
/// `(1/n) sum_x exp(i 2 pi k (x - n/2) / n)`.
fn dirichlet(k: f64, n: usize) -> C {
    let mut acc = C::ZERO;
    for x in 0..n {
        acc = acc.add(C::cis(TAU * k * (x as f64 - (n / 2) as f64) / n as f64));
    }
    acc.scale(1.0 / n as f64)
}

/// The spiral reconstruction of one trajectory (all interleaves of one partition), built once per
/// series.
pub struct SpiralRecon {
    n: usize,
    k: Vec<[f64; 2]>,
    /// Density weights, normalized.
    w: Vec<f64>,
    /// The window per sample.
    win: Vec<f64>,
    nufft: Nufft2,
    /// The Chebyshev interval's upper end.
    pub lambda_hi: f64,
}

impl SpiralRecon {
    pub fn new(k: &[[f64; 2]], n: usize, window: KspaceWindow) -> SpiralRecon {
        let h = (n / 2) as i64;
        let nufft = Nufft2::new([n, n], [-h, -h], [n, n], NUFFT_TOL);
        let mut w = pipe_menon(&nufft, k, DCF_ITERATIONS);
        // a constant object's centre pixel: sum_j w_j K(k_j), with K its exact samples
        let centre = k.iter().zip(&w).fold(C::ZERO, |acc, (p, &wj)| acc.add(dirichlet(p[0], n).mul(dirichlet(p[1], n)).scale(wj)));
        let s = 1.0 / centre.re;
        for wj in w.iter_mut() {
            *wj *= s;
        }
        let win = k.iter().map(|p| window.at(p[0] / n as f64, p[1] / n as f64)).collect();
        let mut r = SpiralRecon { n, k: k.to_vec(), w, win, nufft, lambda_hi: 0.0 };
        // the largest eigenvalue of A^H W A, from a fixed start vector
        let mut v: Vec<C> = (0..n * n).map(|i| C { re: 1.0 + (i % 7) as f64 * 0.1, im: (i % 3) as f64 * 0.1 }).collect();
        let mut lam = 0.0;
        for _ in 0..POWER_ITERATIONS {
            let nv = norm(&v);
            let u = r.normal(&v);
            let nu = norm(&u);
            lam = nu / nv;
            v = u.into_iter().map(|c| c.scale(1.0 / nu)).collect();
        }
        r.lambda_hi = LAMBDA_MARGIN * lam;
        r
    }

    /// The density weights (normalized).
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// `A x`: the samples of an object on the acquired grid.
    fn forward(&self, x: &[C]) -> Vec<C> {
        let sc = 1.0 / (self.n * self.n) as f64;
        let c: Vec<(f64, f64)> = x.iter().map(|c| (c.re, c.im)).collect();
        self.nufft.type2(&c, &self.k, 1).into_iter().map(|(re, im)| C { re: re * sc, im: im * sc }).collect()
    }

    /// `A^H W r`.
    fn adjoint(&self, r: &[C]) -> Vec<C> {
        let sc = 1.0 / (self.n * self.n) as f64;
        let wts: Vec<(f64, f64)> = r.iter().zip(&self.w).map(|(c, &w)| (c.re * w * sc, c.im * w * sc)).collect();
        self.nufft.type1(&self.k, &wts, -1).into_iter().map(|(re, im)| C { re, im }).collect()
    }

    fn normal(&self, x: &[C]) -> Vec<C> {
        self.adjoint(&self.forward(x))
    }

    /// The gridding image `n^2 A^H W d` (no window): the least squares' first direction, kept for
    /// the tests that measure gridding against it.
    pub(crate) fn gridding(&self, d: &[C]) -> Vec<C> {
        let s = (self.n * self.n) as f64;
        self.adjoint(d).into_iter().map(|c| c.scale(s)).collect()
    }

    /// One coil's image on the `n x n` acquired grid (layout `x + n y`): the windowed samples'
    /// density-weighted least squares by the Chebyshev semi-iteration.
    pub(crate) fn image(&self, d: &[C]) -> Vec<C> {
        assert_eq!(d.len(), self.k.len(), "one value per sample");
        let dw: Vec<C> = d.iter().zip(&self.win).map(|(c, &w)| c.scale(w)).collect();
        let (hi, lo) = (self.lambda_hi, self.lambda_hi / LS_KAPPA);
        let (theta, delta) = ((hi + lo) / 2.0, (hi - lo) / 2.0);
        let sigma1 = theta / delta;
        let mut rho = 1.0 / sigma1;
        let mut x = vec![C::ZERO; self.n * self.n];
        let mut r = self.adjoint(&dw);
        let mut dir: Vec<C> = r.iter().map(|c| c.scale(1.0 / theta)).collect();
        for _ in 0..LS_ITERATIONS {
            for (xi, di) in x.iter_mut().zip(&dir) {
                *xi = xi.add(*di);
            }
            let nd = self.normal(&dir);
            for (ri, ni) in r.iter_mut().zip(&nd) {
                *ri = C { re: ri.re - ni.re, im: ri.im - ni.im };
            }
            let rho1 = 1.0 / (2.0 * sigma1 - rho);
            for (di, ri) in dir.iter_mut().zip(&r) {
                *di = di.scale(rho1 * rho).add(ri.scale(2.0 * rho1 / delta));
            }
            rho = rho1;
        }
        x
    }

    /// Every coil's image, Roemer-combined with the known sensitivities as
    /// `kspace::reconstruct_coils` combines them.
    pub(crate) fn reconstruct(&self, coil_samples: &[Vec<C>], acq: &Acquisition) -> Vec<C> {
        let n = self.n;
        let ncoils = acq.n_coils.max(1);
        assert_eq!(coil_samples.len(), ncoils, "one sample set per coil");
        let mut wsum = vec![C::ZERO; n * n];
        let mut ssum = vec![0.0f64; n * n];
        for (coil, d) in coil_samples.iter().enumerate() {
            let img = self.image(d);
            for y in 0..n {
                for x in 0..n {
                    let s = coil_sensitivity(coil, ncoils, x as f64, y as f64, n, n);
                    let i = x + n * y;
                    wsum[i] = wsum[i].add(img[i].scale(s));
                    ssum[i] += s * s;
                }
            }
        }
        (0..n * n)
            .map(|i| {
                let s = ssum[i].max(1e-12);
                C { re: wsum[i].re / s, im: wsum[i].im / s }
            })
            .collect()
    }
}

fn norm(v: &[C]) -> f64 {
    v.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kspace::{SliceInput, T2Slice};
    use crate::readout::spiral_trajectory;
    use crate::spiral::{segment_inputs, spiral_kspace_exact, SegmentedForward};
    use crate::tseg::RateRect;

    fn inp<'a>(comps: &'a [&'a [f32]], t2: &'a [T2Slice<'a>], fmap: &'a [f32], ph: &'a [f64], n: usize, o: usize) -> SliceInput<'a> {
        SliceInput {
            compartments: comps, t2, t_inhom: None, fmap, phase0: Some(ph), sim: [n * o, n * o], acq_matrix: [n, n], z: 0,
            nz: 1, eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None,
        }
    }

    fn plain() -> Acquisition {
        Acquisition { do_relaxation: false, do_distortions: false, noise_variance: 0.0, n_spikes: 0, n_coils: 1,
                      signal_scale: 1.0, ..Acquisition::default() }
    }

    /// Gaussian blobs (spectrum below 1e-13 of its peak at k_max) with a smooth phase.
    fn blobs(n: usize) -> (Vec<f32>, Vec<f64>) {
        let c = n as f64 / 32.0;
        let obj = (0..n * n).map(|i| {
            let (x, y) = ((i % n) as f64, (i / n) as f64);
            let g = |cx: f64, cy: f64, s: f64| (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * s * s)).exp();
            (g(13.0 * c, 15.0 * c, 2.5 * c) + 0.7 * g(20.0 * c, 19.0 * c, 3.0 * c)) as f32
        }).collect();
        let ph = (0..n * n).map(|i| 0.02 * (i % n) as f64 - 0.03 * (i / n) as f64).collect();
        (obj, ph)
    }

    /// A deterministic uniform stream on (-1/2, 1/2).
    fn uniform(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        }
    }

    #[test]
    fn reconstructs_a_band_limited_object_and_a_uniform_one() {
        // the declared tolerances (part C): band-limited 1e-2 of peak, uniform 1e-3 at the centre,
        // at oversample 1 where the data are exactly A x; on asl001's in-plane design (64 x 64,
        // 8 interleaves, 4 ms, 4 us) and a 32 x 32 one; gridding's numbers printed beside
        for &(n, il, dw) in &[(32usize, 4usize, 0.008f64), (64, 8, 0.004)] {
            let tr = spiral_trajectory(n, n, il, 4.0, dw).unwrap();
            let taus: Vec<f64> = (0..tr.k.len()).map(|j| tr.tau_ms[j % tr.n_samples()]).collect();
            let rec = SpiralRecon::new(&tr.k, n, KspaceWindow::None);
            let acq = plain();
            let t2 = [T2Slice::Uniform(f32::INFINITY)];
            let fmap = vec![0.0f32; n * n];
            let (obj, ph) = blobs(n);
            let comps = [obj.as_slice()];
            let d = spiral_kspace_exact(&inp(&comps, &t2, &fmap, &ph, n, 1), &acq, 0, 1, &tr.k, &taus, None);
            let peak = obj.iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
            let err = |img: &[C]| (0..n * n).fold(0.0f64, |m, i| {
                let want = C::cis(ph[i]).scale(obj[i] as f64);
                m.max((img[i].re - want.re).hypot(img[i].im - want.im))
            }) / peak;
            let (e_ls, e_grid) = (err(&rec.reconstruct(std::slice::from_ref(&d), &acq)), err(&rec.gridding(&d)));
            println!("{n} x {n}: band-limited: least squares {e_ls:.2e}, gridding {e_grid:.2e} of peak");
            assert!(e_ls <= 1e-2, "{e_ls}");
            let one = vec![1.0f32; n * n];
            let zero = vec![0.0f64; n * n];
            let comps = [one.as_slice()];
            let d = spiral_kspace_exact(&inp(&comps, &t2, &fmap, &zero, n, 1), &acq, 0, 1, &tr.k, &taus, None);
            let c = (n / 2) * (n + 1);
            let g = rec.gridding(&d);
            assert!((g[c].re - 1.0).abs() < 1e-9 && g[c].im.abs() < 1e-9, "the normalization: {} {}", g[c].re, g[c].im);
            let img = rec.reconstruct(&[d], &acq);
            let centre = (img[c].re - 1.0).hypot(img[c].im);
            let fov = img.iter().fold(0.0f64, |m, v| m.max((v.re - 1.0).hypot(v.im)));
            let gfov = g.iter().fold(0.0f64, |m, v| m.max((v.re - 1.0).hypot(v.im)));
            println!("{n} x {n}: uniform: least squares {centre:.2e} at the centre, {fov:.2e} over the FOV; gridding {gfov:.2e}");
            assert!(centre <= 1e-3, "{centre}");
        }
    }

    #[test]
    fn the_reconstruction_is_linear_and_reports_its_noise() {
        let n = 32;
        let tr = spiral_trajectory(n, n, 4, 4.0, 0.008).unwrap();
        let rec = SpiralRecon::new(&tr.k, n, KspaceWindow::Hann);
        let mut g = uniform(99);
        let d1: Vec<C> = tr.k.iter().map(|_| C { re: g(), im: g() }).collect();
        let d2: Vec<C> = tr.k.iter().map(|_| C { re: g(), im: g() }).collect();
        let sum: Vec<C> = d1.iter().zip(&d2).map(|(a, b)| a.add(*b)).collect();
        let (a, b, ab) = (rec.image(&d1), rec.image(&d2), rec.image(&sum));
        let peak = ab.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        let res = (0..n * n).fold(0.0f64, |m, i| m.max((ab[i].re - a[i].re - b[i].re).hypot(ab[i].im - a[i].im - b[i].im)));
        assert!(res <= 1e-12 * peak, "{}", res / peak);
        // unit-variance noise per sample component: the image noise SD against the Cartesian
        // path's n (an unnormalized sum of n^2 such samples); measured, not asserted (part C)
        let rec = SpiralRecon::new(&tr.k, n, KspaceWindow::None);
        let (mut acc, trials) = (0.0, 8);
        for _ in 0..trials {
            let noise: Vec<C> = tr.k.iter().map(|_| C { re: (g() + g() + g() + g()) * 3f64.sqrt(), im: (g() + g() + g() + g()) * 3f64.sqrt() }).collect();
            acc += rec.image(&noise).iter().map(|c| c.re * c.re).sum::<f64>() / (n * n) as f64;
        }
        println!("noise SD ratio, spiral / Cartesian: {:.4}", (acc / trials as f64).sqrt() / n as f64);
    }

    #[test]
    fn uniform_off_resonance_matches_the_reconstruction_of_the_exact_sum() {
        // the whole pipeline (segmented forward, reconstruction) against the reconstruction of
        // exact-sum samples, with a uniform 30 Hz off-resonance, oversample 2
        let (n, o) = (32, 2);
        let s = n * o;
        let tr = spiral_trajectory(n, n, 4, 4.0, 0.008).unwrap();
        let ns = tr.n_samples();
        let idx: Vec<usize> = (0..tr.k.len()).map(|j| j % ns).collect();
        let taus: Vec<f64> = idx.iter().map(|&j| tr.tau_ms[j]).collect();
        let obj: Vec<f32> = (0..s * s)
            .map(|i| (-((((i % s) as f64 - 27.0).powi(2) + ((i / s) as f64 - 33.0).powi(2)) / 60.0)).exp() as f32)
            .collect();
        let fmap = vec![30.0f32; s * s];
        let ph = vec![0.0; s * s];
        let t2 = [T2Slice::Uniform(f32::INFINITY)];
        let comps = [obj.as_slice()];
        let input = inp(&comps, &t2, &fmap, &ph, n, o);
        let acq = Acquisition { do_distortions: true, ..plain() };
        let fwd = SegmentedForward::new(&tr.k, &idx, &tr.tau_ms, 4.0, [s, s], [n, n], RateRect { d: [0.0, 0.0], f: [30.0, 30.0] }).unwrap();
        let (x, r) = segment_inputs(&input, &acq, 0, 1, None, false);
        let a = fwd.apply(&x, &r);
        let b = spiral_kspace_exact(&input, &acq, 0, 1, &tr.k, &taus, None);
        let rec = SpiralRecon::new(&tr.k, n, KspaceWindow::None);
        let (ia, ib) = (rec.reconstruct(&[a], &acq), rec.reconstruct(&[b], &acq));
        let peak = ib.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        let err = ia.iter().zip(&ib).fold(0.0f64, |m, (p, q)| m.max((p.re - q.re).hypot(p.im - q.im)));
        assert!(err <= 1e-6 * peak, "{}", err / peak);
    }
}
