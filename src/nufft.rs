//! One-dimensional type-1 NUFFT on a periodic interval — nonuniform sources, uniform integer
//! frequencies:
//!
//! ```text
//! F[k] = Σ_j w_j · exp(+i·2π·k·v_j / n),     k = k_lo .. k_lo + n_out,   v_j ∈ ℝ (period n)
//! ```
//!
//! This is the y-sum of the k-space forward model once the fieldmap is read as geometry: with the
//! readout time affine in the PE line, `fmap(r)·t(ky)` is a displacement of the source along y,
//! `v = (y − c) + sny·τ·fmap`, and the per-line sum over voxels becomes one transform of the
//! displaced object (`kspace.rs`). Gridding after Greengard & Lee (2004) with the "exponential of
//! semicircle" kernel of Barnett, Magland & af Klinteberg (2019) at oversampling σ = 2: each source
//! is spread onto `w` fine cells, one FFT (`rustfft`), and the kernel's own transform is divided
//! out. The accuracy is set by `w` (≈ 10^{−(w−1)}); it is not an expansion whose cost grows with
//! the fieldmap range. Several weight vectors sharing the same positions are transformed
//! together, so the kernel is evaluated once per source. Pinned against the literal sum by
//! `matches_the_direct_sum`.
#![cfg(feature = "kspace")]

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::TAU;
use std::sync::Arc;

pub struct Nufft1 {
    nf: usize,
    w: usize,
    beta: f64,
    k_lo: i64,
    n_out: usize,
    fft: Arc<dyn Fft<f64>>,
    scratch: Vec<Complex<f64>>,
    /// 1/ψ̂(k) for the `n_out` output frequencies.
    corr: Vec<f64>,
    /// Kernel values of the current source, reused across weight vectors.
    kern: Vec<f64>,
    grid: Vec<Vec<Complex<f64>>>,
}

impl Nufft1 {
    /// Period `n` (cells), outputs at integer frequencies `k_lo .. k_lo + n_out` (which must lie
    /// within `[-n/2, n/2]`), tolerance `tol` (relative to the total weight; 1e-12 → w = 13).
    pub fn new(n: usize, k_lo: i64, n_out: usize, tol: f64) -> Nufft1 {
        assert!(n_out > 0 && k_lo >= -(n as i64) / 2 && k_lo + n_out as i64 - 1 <= (n as i64) / 2,
            "output band must sit inside the coarse Nyquist band");
        let w = ((-tol.log10()).ceil() as usize + 1).clamp(2, 16);
        let beta = 2.30 * w as f64;
        let nf = 2 * n;
        let fft = FftPlanner::<f64>::new().plan_fft_inverse(nf);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        // ψ̂(k) = Σ_m ψ(m) e^{i2π k m/nf} over the kernel's support (real, symmetric).
        let half = w as f64 / 2.0;
        let psi = |z: f64| if z.abs() < 1.0 { (beta * ((1.0 - z * z).sqrt() - 1.0)).exp() } else { 0.0 };
        let mut corr = Vec::with_capacity(n_out);
        for j in 0..n_out {
            let k = (k_lo + j as i64) as f64;
            let mut s = psi(0.0);
            let mut m = 1.0;
            while m < half {
                s += 2.0 * psi(m / half) * (TAU * k * m / nf as f64).cos();
                m += 1.0;
            }
            corr.push(1.0 / s);
        }
        Nufft1 { nf, w, beta, k_lo, n_out, fft, scratch, corr, kern: vec![0.0; w], grid: Vec::new() }
    }

    /// `pos[j]` in cells (any real; periodic modulo `n`). `weights`: any number of `(re, im)`
    /// vectors of the same length as `pos`. `out[q]` receives the `n_out` complex outputs of
    /// weight vector `q` as `(re, im)`.
    pub fn run(&mut self, pos: &[f64], weights: &[(&[f64], &[f64])], out: &mut [(Vec<f64>, Vec<f64>)]) {
        let nw = weights.len();
        assert_eq!(out.len(), nw);
        let (nf, w, half) = (self.nf, self.w, self.w as f64 / 2.0);
        self.grid.resize_with(nw, Vec::new);
        for g in self.grid.iter_mut() {
            g.clear();
            g.resize(nf, Complex::new(0.0, 0.0));
        }
        let beta = self.beta;
        for (j, &v) in pos.iter().enumerate() {
            // fine-grid position, wrapped into [0, nf)
            let p = (2.0 * v).rem_euclid(nf as f64);
            let m0 = (p - half).floor() as i64 + 1;
            for i in 0..w {
                let z = (m0 as f64 + i as f64 - p) / half;
                self.kern[i] = if z.abs() < 1.0 { (beta * ((1.0 - z * z).sqrt() - 1.0)).exp() } else { 0.0 };
            }
            for (q, &(re, im)) in weights.iter().enumerate() {
                let (wr, wi) = (re[j], im[j]);
                let g = &mut self.grid[q];
                for i in 0..w {
                    let m = (m0 + i as i64).rem_euclid(nf as i64) as usize;
                    g[m].re += wr * self.kern[i];
                    g[m].im += wi * self.kern[i];
                }
            }
        }
        for (q, g) in self.grid.iter_mut().enumerate() {
            self.fft.process_with_scratch(g, &mut self.scratch);
            let (ore, oim) = &mut out[q];
            ore.clear();
            oim.clear();
            for j in 0..self.n_out {
                let bin = (self.k_lo + j as i64).rem_euclid(nf as i64) as usize;
                ore.push(g[bin].re * self.corr[j]);
                oim.push(g[bin].im * self.corr[j]);
            }
        }
    }

    pub fn kernel_width(&self) -> usize { self.w }
}

/// `1/ψ̂(m)` for the integer modes `m_lo .. m_lo + m_out` on a fine grid of `nf` cells (the
/// kernel's own transform, sampled as [`Nufft1`] samples it).
fn deapodization(nf: usize, w: usize, beta: f64, m_lo: i64, m_out: usize) -> Vec<f64> {
    let half = w as f64 / 2.0;
    let psi = |z: f64| if z.abs() < 1.0 { (beta * ((1.0 - z * z).sqrt() - 1.0)).exp() } else { 0.0 };
    (0..m_out)
        .map(|j| {
            let k = (m_lo + j as i64) as f64;
            let mut s = psi(0.0);
            let mut m = 1.0;
            while m < half {
                s += 2.0 * psi(m / half) * (TAU * k * m / nf as f64).cos();
                m += 1.0;
            }
            1.0 / s
        })
        .collect()
}

/// The two-dimensional NUFFT pair of P5's stack of spirals (addendum, part C): nonuniform points
/// `u_j` (periodic with period `n` per axis) and uniform integer modes `m` in a band inside the
/// coarse Nyquist band:
///
/// ```text
/// type 1:  f[m] = Σ_j w_j · exp(s·i·2π·(u_j0·m0/n0 + u_j1·m1/n1))       (spread, FFT, deapodize)
/// type 2:  g_j  = Σ_m c_m · exp(s·i·2π·(u_j0·m0/n0 + u_j1·m1/n1))       (deapodize, FFT, interpolate)
/// ```
///
/// with `s = ±1`. The spiral forward is a type 2 (`u` the k-space samples, `m` the image), and the
/// gridding reconstruction a type 1 of the opposite sign; type 2 with sign `s` is exactly the
/// adjoint of type 1 with sign `-s`. The same exponential-of-semicircle kernel, oversampling
/// `σ = 2` and width rule as [`Nufft1`]. Pinned against direct sums by
/// `two_d_pair_matches_the_direct_sums`.
pub struct Nufft2 {
    n: [usize; 2],
    nf: [usize; 2],
    w: usize,
    beta: f64,
    m_lo: [i64; 2],
    m_out: [usize; 2],
    corr: [Vec<f64>; 2],
    plans: [[Arc<dyn Fft<f64>>; 2]; 2],
}

impl Nufft2 {
    /// Period `n` per axis, modes `m_lo .. m_lo + m_out` per axis (inside `[-n/2, n/2]`),
    /// tolerance `tol` as [`Nufft1`]'s.
    pub fn new(n: [usize; 2], m_lo: [i64; 2], m_out: [usize; 2], tol: f64) -> Nufft2 {
        for a in 0..2 {
            assert!(m_out[a] > 0 && m_lo[a] >= -(n[a] as i64) / 2 && m_lo[a] + m_out[a] as i64 - 1 <= (n[a] as i64) / 2,
                    "mode band must sit inside the coarse Nyquist band");
        }
        let w = ((-tol.log10()).ceil() as usize + 1).clamp(2, 16);
        let beta = 2.30 * w as f64;
        let nf = [2 * n[0], 2 * n[1]];
        let mut planner = FftPlanner::<f64>::new();
        let plans = [
            [planner.plan_fft_forward(nf[0]), planner.plan_fft_inverse(nf[0])],
            [planner.plan_fft_forward(nf[1]), planner.plan_fft_inverse(nf[1])],
        ];
        let corr = [deapodization(nf[0], w, beta, m_lo[0], m_out[0]), deapodization(nf[1], w, beta, m_lo[1], m_out[1])];
        Nufft2 { n, nf, w, beta, m_lo, m_out, corr, plans }
    }

    pub fn kernel_width(&self) -> usize {
        self.w
    }

    /// The first fine cell and the `w` kernel weights of a point at period-`n` coordinate `u`.
    fn kernel(&self, a: usize, u: f64, k: &mut [f64]) -> i64 {
        let half = self.w as f64 / 2.0;
        let p = (2.0 * u).rem_euclid(self.nf[a] as f64);
        let m0 = (p - half).floor() as i64 + 1;
        for (i, ki) in k.iter_mut().enumerate().take(self.w) {
            let z = (m0 as f64 + i as f64 - p) / half;
            *ki = if z.abs() < 1.0 { (self.beta * ((1.0 - z * z).sqrt() - 1.0)).exp() } else { 0.0 };
        }
        m0
    }

    /// In-place 2D FFT of a `nf0 x nf1` grid (layout `c0 + nf0 c1`), sign `+1` (inverse, `e^{+i}`)
    /// or `-1`.
    fn fft2(&self, g: &mut [Complex<f64>], sign: i8) {
        let [nf0, nf1] = self.nf;
        let dir = usize::from(sign > 0);
        let p0 = &self.plans[0][dir];
        let mut scratch = vec![Complex::new(0.0, 0.0); p0.get_inplace_scratch_len()];
        for row in g.chunks_exact_mut(nf0) {
            p0.process_with_scratch(row, &mut scratch);
        }
        let p1 = &self.plans[1][dir];
        let mut col = vec![Complex::new(0.0, 0.0); nf1];
        let mut scratch = vec![Complex::new(0.0, 0.0); p1.get_inplace_scratch_len()];
        for c0 in 0..nf0 {
            for c1 in 0..nf1 {
                col[c1] = g[c0 + nf0 * c1];
            }
            p1.process_with_scratch(&mut col, &mut scratch);
            for c1 in 0..nf1 {
                g[c0 + nf0 * c1] = col[c1];
            }
        }
    }

    /// The fine-grid bin of mode `m` on axis `a`.
    fn bin(&self, a: usize, m: i64) -> usize {
        m.rem_euclid(self.nf[a] as i64) as usize
    }

    /// Type 1: the modes `f[m]`, layout `(m0 - m_lo0) + m_out0 (m1 - m_lo1)`.
    pub fn type1(&self, u: &[[f64; 2]], wts: &[(f64, f64)], sign: i8) -> Vec<(f64, f64)> {
        assert_eq!(u.len(), wts.len());
        let [nf0, nf1] = self.nf;
        let mut g = vec![Complex::new(0.0, 0.0); nf0 * nf1];
        let (mut k0, mut k1) = (vec![0.0; self.w], vec![0.0; self.w]);
        for (uj, &(wr, wi)) in u.iter().zip(wts) {
            let a0 = self.kernel(0, uj[0], &mut k0);
            let a1 = self.kernel(1, uj[1], &mut k1);
            for (i1, &v1) in k1.iter().enumerate() {
                let c1 = (a1 + i1 as i64).rem_euclid(nf1 as i64) as usize;
                for (i0, &v0) in k0.iter().enumerate() {
                    let c0 = (a0 + i0 as i64).rem_euclid(nf0 as i64) as usize;
                    let v = v0 * v1;
                    let cell = &mut g[c0 + nf0 * c1];
                    cell.re += wr * v;
                    cell.im += wi * v;
                }
            }
        }
        self.fft2(&mut g, sign);
        let mut out = Vec::with_capacity(self.m_out[0] * self.m_out[1]);
        for j1 in 0..self.m_out[1] {
            let b1 = self.bin(1, self.m_lo[1] + j1 as i64);
            for j0 in 0..self.m_out[0] {
                let b0 = self.bin(0, self.m_lo[0] + j0 as i64);
                let c = self.corr[0][j0] * self.corr[1][j1];
                let v = g[b0 + nf0 * b1];
                out.push((v.re * c, v.im * c));
            }
        }
        out
    }

    /// Type 2: the values `g_j` at the points, from modes `c` in type 1's layout.
    pub fn type2(&self, c: &[(f64, f64)], u: &[[f64; 2]], sign: i8) -> Vec<(f64, f64)> {
        assert_eq!(c.len(), self.m_out[0] * self.m_out[1]);
        let [nf0, nf1] = self.nf;
        let mut g = vec![Complex::new(0.0, 0.0); nf0 * nf1];
        for j1 in 0..self.m_out[1] {
            let b1 = self.bin(1, self.m_lo[1] + j1 as i64);
            for j0 in 0..self.m_out[0] {
                let b0 = self.bin(0, self.m_lo[0] + j0 as i64);
                let s = self.corr[0][j0] * self.corr[1][j1];
                let (re, im) = c[j0 + self.m_out[0] * j1];
                g[b0 + nf0 * b1] = Complex::new(re * s, im * s);
            }
        }
        self.fft2(&mut g, sign);
        let (mut k0, mut k1) = (vec![0.0; self.w], vec![0.0; self.w]);
        u.iter()
            .map(|uj| {
                let a0 = self.kernel(0, uj[0], &mut k0);
                let a1 = self.kernel(1, uj[1], &mut k1);
                let (mut re, mut im) = (0.0, 0.0);
                for (i1, &v1) in k1.iter().enumerate() {
                    let c1 = (a1 + i1 as i64).rem_euclid(nf1 as i64) as usize;
                    for (i0, &v0) in k0.iter().enumerate() {
                        let c0 = (a0 + i0 as i64).rem_euclid(nf0 as i64) as usize;
                        let v = v0 * v1;
                        let cell = g[c0 + nf0 * c1];
                        re += cell.re * v;
                        im += cell.im * v;
                    }
                }
                (re, im)
            })
            .collect()
    }

    /// The period per axis.
    pub fn period(&self) -> [usize; 2] {
        self.n
    }
}

/// Reference: the literal sum, O(n_src · n_out).
#[cfg(test)]
fn direct(n: usize, k_lo: i64, n_out: usize, pos: &[f64], re: &[f64], im: &[f64]) -> Vec<(f64, f64)> {
    (0..n_out)
        .map(|j| {
            let k = (k_lo + j as i64) as f64;
            let (mut ar, mut ai) = (0.0, 0.0);
            for (jj, &v) in pos.iter().enumerate() {
                let ph = TAU * k * v / n as f64;
                let (c, s) = (ph.cos(), ph.sin());
                ar += re[jj] * c - im[jj] * s;
                ai += re[jj] * s + im[jj] * c;
            }
            (ar, ai)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kspace::Rng;

    #[test]
    fn matches_the_direct_sum() {
        let mut rng = Rng(42);
        for &(n, k_lo, n_out, tol) in &[(48usize, -12i64, 24usize, 1e-12), (140, -35, 70, 1e-12), (16, -8, 16, 1e-12), (64, -16, 32, 1e-8)] {
            let nsrc = 3 * n;
            let pos: Vec<f64> = (0..nsrc).map(|_| (rng.unit() - 0.5) * 3.0 * n as f64).collect();
            let re: Vec<f64> = (0..nsrc).map(|_| rng.unit() - 0.5).collect();
            let im: Vec<f64> = (0..nsrc).map(|_| rng.unit() - 0.5).collect();
            let re2: Vec<f64> = (0..nsrc).map(|_| rng.unit()).collect();
            let im2: Vec<f64> = vec![0.0; nsrc];
            let mut nufft = Nufft1::new(n, k_lo, n_out, tol);
            let mut out = vec![(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
            nufft.run(&pos, &[(&re, &im), (&re2, &im2)], &mut out);
            for (q, (r, i)) in [(&re, &im), (&re2, &im2)].iter().enumerate() {
                let want = direct(n, k_lo, n_out, &pos, r, i);
                let total: f64 = r.iter().zip(i.iter()).map(|(a, b)| a.hypot(*b)).sum();
                let worst = want.iter().zip(out[q].0.iter().zip(&out[q].1))
                    .map(|(&(wr, wi), (&or, &oi))| (wr - or).hypot(wi - oi)).fold(0.0, f64::max);
                assert!(worst <= 10.0 * tol * total, "n={n} q={q}: max err {worst:.3e} vs {:.3e} allowed (total {total:.3e})", 10.0 * tol * total);
            }
        }
    }

    /// The 2D pair's direct sums: `Σ_j w_j e^{s i 2π u_j·m/n}` (type 1) and its transpose (type 2).
    fn direct2(n: [usize; 2], m_lo: [i64; 2], m_out: [usize; 2], u: &[[f64; 2]], sign: f64)
        -> Vec<Vec<(f64, f64)>> {
        // e[m][j] = exp(s i 2π u_j·m/n), m in type-1 layout
        let mut e = Vec::with_capacity(m_out[0] * m_out[1]);
        for j1 in 0..m_out[1] {
            for j0 in 0..m_out[0] {
                let (m0, m1) = ((m_lo[0] + j0 as i64) as f64, (m_lo[1] + j1 as i64) as f64);
                e.push(u.iter().map(|uj| {
                    let ph = sign * TAU * (uj[0] * m0 / n[0] as f64 + uj[1] * m1 / n[1] as f64);
                    (ph.cos(), ph.sin())
                }).collect());
            }
        }
        e
    }

    fn cmul((a, b): (f64, f64), (c, d): (f64, f64)) -> (f64, f64) {
        (a * c - b * d, a * d + b * c)
    }

    #[test]
    fn two_d_pair_matches_the_direct_sums() {
        let mut rng = Rng(77);
        for &(n, m_lo, m_out) in &[([16usize, 16usize], [-8i64, -8i64], [16usize, 16usize]), ([24, 20], [-6, -10], [12, 20]),
                                   ([9, 12], [-4, -6], [9, 12])] {
            let np = 300;
            let u: Vec<[f64; 2]> = (0..np).map(|_| [(rng.unit() - 0.5) * n[0] as f64, (rng.unit() - 0.5) * n[1] as f64]).collect();
            let w: Vec<(f64, f64)> = (0..np).map(|_| (rng.unit() - 0.5, rng.unit() - 0.5)).collect();
            let c: Vec<(f64, f64)> = (0..m_out[0] * m_out[1]).map(|_| (rng.unit() - 0.5, rng.unit() - 0.5)).collect();
            let nu = Nufft2::new(n, m_lo, m_out, 1e-12);
            for sign in [1i8, -1] {
                let e = direct2(n, m_lo, m_out, &u, sign as f64);
                // type 1
                let got = nu.type1(&u, &w, sign);
                let total: f64 = w.iter().map(|(a, b)| a.hypot(*b)).sum();
                for (m, em) in e.iter().enumerate() {
                    let want = em.iter().zip(&w).fold((0.0, 0.0), |acc, (&x, &y)| { let p = cmul(x, y); (acc.0 + p.0, acc.1 + p.1) });
                    assert!((got[m].0 - want.0).hypot(got[m].1 - want.1) <= 1e-10 * total, "type 1 {n:?} s {sign} m {m}");
                }
                // type 2
                let got = nu.type2(&c, &u, sign);
                let total: f64 = c.iter().map(|(a, b)| a.hypot(*b)).sum();
                for j in 0..np {
                    let want = e.iter().zip(&c).fold((0.0, 0.0), |acc, (em, &cm)| { let p = cmul(em[j], cm); (acc.0 + p.0, acc.1 + p.1) });
                    assert!((got[j].0 - want.0).hypot(got[j].1 - want.1) <= 1e-10 * total, "type 2 {n:?} s {sign} j {j}");
                }
            }
            // the adjoint identity: <type2_s(c), y> = <c, type1_{-s}(y)>
            let y: Vec<(f64, f64)> = (0..np).map(|_| (rng.unit() - 0.5, rng.unit() - 0.5)).collect();
            let a = nu.type2(&c, &u, 1);
            let b = nu.type1(&u, &y, -1);
            let dot = |p: &[(f64, f64)], q: &[(f64, f64)]| p.iter().zip(q).fold((0.0, 0.0), |acc, (&x, &(qr, qi))| {
                let t = cmul(x, (qr, -qi));
                (acc.0 + t.0, acc.1 + t.1)
            });
            let (l, r) = (dot(&a, &y), dot(&c, &b));
            let scale = a.iter().map(|z| z.0.hypot(z.1)).sum::<f64>() * y.iter().map(|z| z.0.hypot(z.1)).sum::<f64>();
            assert!((l.0 - r.0).hypot(l.1 - r.1) <= 1e-10 * scale, "adjoint {n:?}: {l:?} vs {r:?}");
        }
    }

    #[test]
    fn two_d_integer_points_reduce_to_the_dft() {
        let n = [8usize, 6usize];
        let u: Vec<[f64; 2]> = (0..6).flat_map(|b| (0..8).map(move |a| [a as f64 - 4.0, b as f64 - 3.0])).collect();
        let w: Vec<(f64, f64)> = (0..48).map(|i| (((i as f64) * 0.37).sin(), 0.0)).collect();
        let nu = Nufft2::new(n, [-4, -3], [8, 6], 1e-12);
        let got = nu.type1(&u, &w, 1);
        let e = direct2(n, [-4, -3], [8, 6], &u, 1.0);
        for (m, em) in e.iter().enumerate() {
            let want = em.iter().zip(&w).fold((0.0, 0.0), |acc, (&x, &y)| { let p = cmul(x, y); (acc.0 + p.0, acc.1 + p.1) });
            assert!((got[m].0 - want.0).abs() < 1e-10 && (got[m].1 - want.1).abs() < 1e-10, "mode {m}");
        }
    }

    #[test]
    fn integer_positions_reduce_to_the_dft() {
        // Sources on the grid: the NUFFT must reproduce a plain DFT of the periodic sequence.
        let n = 32;
        let pos: Vec<f64> = (0..n).map(|y| y as f64).collect();
        let re: Vec<f64> = (0..n).map(|y| ((y as f64) * 0.3).sin()).collect();
        let im = vec![0.0; n];
        let mut nufft = Nufft1::new(n, -16, 32, 1e-12);
        let mut out = vec![(Vec::new(), Vec::new())];
        nufft.run(&pos, &[(&re, &im)], &mut out);
        let want = direct(n, -16, 32, &pos, &re, &im);
        for (j, &(wr, wi)) in want.iter().enumerate() {
            assert!((wr - out[0].0[j]).abs() < 1e-10 && (wi - out[0].1[j]).abs() < 1e-10, "k index {j}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::kspace::Rng;

    /// Timing only: one slice-column's worth of work at production size (sny=256, ny=128, 6 weight
    /// vectors), split into spreading and FFT. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "timing micro-benchmark; run explicitly with --ignored"]
    fn bench_column() {
        let mut rng = Rng(7);
        let (n, n_out, nw, cols) = (256usize, 128usize, 6usize, 170usize);
        let pos: Vec<f64> = (0..n).map(|y| y as f64 - 128.0 + 30.0 * (rng.unit() - 0.5)).collect();
        let ws: Vec<(Vec<f64>, Vec<f64>)> = (0..nw).map(|_| ((0..n).map(|_| rng.unit()).collect(), (0..n).map(|_| rng.unit()).collect())).collect();
        let weights: Vec<(&[f64], &[f64])> = ws.iter().map(|(a, b)| (a.as_slice(), b.as_slice())).collect();
        let mut nufft = Nufft1::new(n, -64, n_out, 1e-13);
        let mut out: Vec<(Vec<f64>, Vec<f64>)> = (0..nw).map(|_| (Vec::new(), Vec::new())).collect();
        let t = std::time::Instant::now();
        for _ in 0..cols { nufft.run(&pos, &weights, &mut out); }
        let per_slice = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("NUFFT y-stage per slice ({cols} columns, w={}): {per_slice:.2} ms", nufft.kernel_width());
    }
}
