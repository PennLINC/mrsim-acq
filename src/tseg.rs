//! Time segmentation with a certified error (P5 addendum, part C, "Forward and reconstruction").
//!
//! The spiral forward needs `exp(z(r) tau)` for every voxel's complex rate `z = -d + i 2 pi f`
//! (`d >= 0` the decay rate, `f` the off-resonance) at every sample time `tau` in `[0, T]`. It is
//! interpolated in `tau` over `L` segment times `tau_l` uniform on `[0, T]`,
//!
//! ```text
//! exp(z tau) ~ sum_l b_l(tau) exp(z tau_l)
//! ```
//!
//! so the forward is `L` type-2 NUFFTs of `x(r) exp(z(r) tau_l)` combined per sample by `b(tau)`.
//! The interpolators are the least-squares ones (Sutton, Noll and Fessler 2003): for each `tau`,
//! `b(tau)` minimizes the error over a tensor Chebyshev grid of degree `m` on the rate rectangle
//! `[d_min, d_max] x 2 pi [f_min, f_max]` (minimum norm when the grid has fewer points than `L`).
//!
//! The error `e(z, tau) = exp(z tau) - sum_l b_l(tau) exp(z tau_l)` is entire in `z` with every
//! `n`-th derivative bounded, for `Re z <= 0`, by `T^n (1 + B)`, `B = max_tau sum_l |b_l(tau)|`. With
//! `P` the tensor Chebyshev interpolant on the grid, `Lambda_m <= 1 + (2/pi) ln(m + 1)` its
//! Lebesgue constant and `rem(R) = 2 (R T / 4)^(m+1) (1 + B) / (m + 1)!` the one-dimensional
//! remainder on an interval of length `R`, every rate in the rectangle has
//!
//! ```text
//! |e| <= |P e| + |e - P e| <= Lambda_m^2 max_grid |e| + rem(R_d) + Lambda_m rem(R_f)
//! ```
//!
//! over every sample time, plus `4 L eps (1 + B)` for the rounding of evaluating the interpolant. `m` is the smallest degree making the remainder terms below
//! [`REMAINDER_TARGET`]; `L` starts at `ceil(T (f_max - f_min)) + 2` and doubles until the bound is
//! below [`BOUND_TARGET`], past [`L_MAX`] an error. An axis of zero length has one grid point, no
//! remainder and no Lebesgue factor; `class` mode's rectangle is the frequency interval alone (its
//! decay is a per-sample weight outside the sum).
#![cfg(feature = "kspace")]

use std::f64::consts::{PI, TAU};

/// The certified bound the plan must reach.
pub const BOUND_TARGET: f64 = 1e-7;
/// The remainder terms' share, which selects `m`.
pub const REMAINDER_TARGET: f64 = 5e-8;
/// The largest `L` before the segmentation is refused.
pub const L_MAX: usize = 64;
const M_MAX: usize = 80;
/// Singular values below this fraction of the largest are dropped from the least-squares fit
/// (the bound is computed from the fit actually made, so this affects only efficiency).
const RCOND: f64 = 1e-13;

/// The rates the segmentation must cover: decay `d` (1/ms, `>= 0`) and off-resonance `f` (Hz).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateRect {
    pub d: [f64; 2],
    pub f: [f64; 2],
}

/// A certified time segmentation for one set of sample times.
#[derive(Debug, Clone, PartialEq)]
pub struct TsegPlan {
    pub l: usize,
    pub m: usize,
    /// Segment times (ms).
    pub tau_l: Vec<f64>,
    /// `b[j * l + i]`: sample `j`'s coefficient of segment `i`.
    pub b: Vec<(f64, f64)>,
    /// `max_tau sum_l |b_l(tau)|`.
    pub b_sum: f64,
    /// The largest error on the fitting grid, over every sample time.
    pub grid_err: f64,
    /// The certified bound on `|e|` over the whole rectangle and every sample time.
    pub bound: f64,
}

/// `m + 1` Chebyshev points of the first kind on `[a, b]` (one point when `a == b`).
fn cheb(a: f64, b: f64, m: usize) -> Vec<f64> {
    if b <= a {
        return vec![a];
    }
    (0..=m).map(|k| 0.5 * (a + b) + 0.5 * (b - a) * ((2 * k + 1) as f64 * PI / (2 * (m + 1)) as f64).cos()).collect()
}

fn lebesgue(m: usize) -> f64 {
    1.0 + 2.0 / PI * ((m + 1) as f64).ln()
}

/// `2 (R T / 4)^(m+1) (1 + B) / (m + 1)!`, zero for a zero-length axis.
fn remainder(r: f64, t: f64, m: usize, b_sum: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    let x = r * t / 4.0;
    let mut v = 2.0 * (1.0 + b_sum);
    for k in 1..=m + 1 {
        v *= x / k as f64;
    }
    v
}

/// A truncated SVD of a real `m x n` matrix, kept factored: `A ~ sum_k sig_k left_k right_k^T`
/// over the singular values above `RCOND` of the largest. The least-squares (minimum-norm)
/// solution is applied as `sum_k right_k (left_k . y) / sig_k`, never through an explicit
/// pseudo-inverse: at the condition numbers of the segmentation (`1e11` and more) an explicit
/// pseudo-inverse has entries near `1 / sig_min`, and the cancellation in applying it leaves
/// residuals near `eps / sig_min`, while in factored form the rounding error lies along the small
/// singular directions and `A` maps it back down.
struct Lsq {
    left: Vec<Vec<f64>>,
    right: Vec<Vec<f64>>,
    sig: Vec<f64>,
}

impl Lsq {
    /// By one-sided Jacobi (Hestenes) on the narrower side: a wide matrix is transposed first, so
    /// no column has to converge to zero.
    fn new(cols: &[Vec<f64>]) -> Lsq {
        let n = cols.len();
        let m = cols.first().map_or(0, |c| c.len());
        if n > m {
            // A^T = U S V^T  =>  A = V S U^T
            let rows: Vec<Vec<f64>> = (0..m).map(|i| cols.iter().map(|c| c[i]).collect()).collect();
            let t = Lsq::new(&rows);
            return Lsq { left: t.right, right: t.left, sig: t.sig };
        }
        let mut u: Vec<Vec<f64>> = cols.to_vec();
        let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect()).collect();
        for _sweep in 0..100 {
            let mut rotated = false;
            for p in 0..n {
                for q in p + 1..n {
                    let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                    for i in 0..m {
                        alpha += u[p][i] * u[p][i];
                        beta += u[q][i] * u[q][i];
                        gamma += u[p][i] * u[q][i];
                    }
                    if gamma == 0.0 || gamma.abs() <= 1e-15 * alpha.sqrt() * beta.sqrt() {
                        continue;
                    }
                    rotated = true;
                    let zeta = (beta - alpha) / (2.0 * gamma);
                    let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = c * t;
                    for i in 0..m {
                        let (a, b) = (u[p][i], u[q][i]);
                        u[p][i] = c * a - s * b;
                        u[q][i] = s * a + c * b;
                    }
                    for i in 0..n {
                        let (a, b) = (v[p][i], v[q][i]);
                        v[p][i] = c * a - s * b;
                        v[q][i] = s * a + c * b;
                    }
                }
            }
            if !rotated {
                break;
            }
        }
        let norms: Vec<f64> = u.iter().map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
        let smax = norms.iter().fold(0.0f64, |a, &b| a.max(b));
        let mut out = Lsq { left: Vec::new(), right: Vec::new(), sig: Vec::new() };
        for k in 0..n {
            if norms[k] > RCOND * smax && norms[k] > 0.0 {
                out.left.push(u[k].iter().map(|x| x / norms[k]).collect());
                out.right.push(v[k].clone());
                out.sig.push(norms[k]);
            }
        }
        out
    }

    fn solve(&self, y: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.right.first().map_or(0, |r| r.len())];
        for ((l, r), &s) in self.left.iter().zip(&self.right).zip(&self.sig) {
            let c = l.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / s;
            for (xi, ri) in x.iter_mut().zip(r) {
                *xi += c * ri;
            }
        }
        x
    }
}

/// The fit at fixed `L` and `m`: coefficients, `B`, the largest grid error.
fn fit(tau: &[f64], t: f64, l: usize, rates: &[(f64, f64)]) -> (Vec<f64>, Vec<(f64, f64)>, f64, f64) {
    let tau_l: Vec<f64> = if l == 1 { vec![0.0] } else { (0..l).map(|i| t * i as f64 / (l - 1) as f64).collect() };
    let g = rates.len();
    // the real embedding [[Re A, -Im A], [Im A, Re A]] of A[g][i] = exp(z_g tau_i), by columns
    let mut cols = vec![vec![0.0; 2 * g]; 2 * l];
    for (i, &ti) in tau_l.iter().enumerate() {
        for (k, &(re, im)) in rates.iter().enumerate() {
            let mag = (re * ti).exp();
            let (c, s) = ((im * ti).cos() * mag, (im * ti).sin() * mag);
            cols[i][k] = c;
            cols[i][g + k] = s;
            cols[l + i][k] = -s;
            cols[l + i][g + k] = c;
        }
    }
    let p = Lsq::new(&cols);
    let (mut b, mut b_sum, mut err) = (Vec::with_capacity(tau.len() * l), 0.0f64, 0.0f64);
    let mut y = vec![0.0; 2 * g];
    for &tj in tau {
        for (k, &(re, im)) in rates.iter().enumerate() {
            let mag = (re * tj).exp();
            y[k] = (im * tj).cos() * mag;
            y[g + k] = (im * tj).sin() * mag;
        }
        let x = p.solve(&y);
        let mut s = 0.0;
        for i in 0..l {
            b.push((x[i], x[l + i]));
            s += x[i].hypot(x[l + i]);
        }
        b_sum = b_sum.max(s);
        for k in 0..g {
            let (mut re, mut im) = (y[k], y[g + k]);
            for i in 0..2 * l {
                re -= cols[i][k] * x[i];
                im -= cols[i][g + k] * x[i];
            }
            err = err.max(re.hypot(im));
        }
    }
    (tau_l, b, b_sum, err)
}

/// The certified segmentation of `exp(z tau)` at the sample times `tau` (ms, in `[0, t_ms]`) over
/// the rectangle `rect`. An error, naming the rectangle, when `L` would exceed [`L_MAX`].
pub fn plan(tau: &[f64], t_ms: f64, rect: RateRect) -> Result<TsegPlan, String> {
    assert!(rect.d[0] >= 0.0 && rect.d[1] >= rect.d[0] && rect.f[1] >= rect.f[0], "rate rectangle {rect:?}");
    assert!(tau.iter().all(|&x| (0.0..=t_ms).contains(&x)), "sample times outside [0, T]");
    let r_d = rect.d[1] - rect.d[0];
    let r_f = TAU * (rect.f[1] - rect.f[0]) / 1000.0; // rad/ms
    let mut l = ((t_ms / 1000.0) * (rect.f[1] - rect.f[0])).ceil() as usize + 2;
    loop {
        if l > L_MAX {
            return Err(format!(
                "the time segmentation cannot certify {BOUND_TARGET:e} within {L_MAX} segments over decay rates \
                 {:.4}..{:.4} 1/ms and off-resonance {:.2}..{:.2} Hz on a {t_ms} ms readout",
                rect.d[0], rect.d[1], rect.f[0], rect.f[1]));
        }
        // the smallest m whose remainder could meet the target with B = 0, then upward
        let rem_total = |m: usize, b_sum: f64| {
            let lam_d = if r_d > 0.0 { lebesgue(m) } else { 1.0 };
            remainder(r_d, t_ms, m, b_sum) + lam_d * remainder(r_f, t_ms, m, b_sum)
        };
        let mut m = 1;
        while m < M_MAX && rem_total(m, 0.0) >= REMAINDER_TARGET {
            m += 1;
        }
        loop {
            let ds = cheb(rect.d[0], rect.d[1], m);
            let fs = cheb(TAU * rect.f[0] / 1000.0, TAU * rect.f[1] / 1000.0, m);
            let rates: Vec<(f64, f64)> = ds.iter().flat_map(|&d| fs.iter().map(move |&w| (-d, w))).collect();
            let (tau_l, b, b_sum, err) = fit(tau, t_ms, l, &rates);
            let rem = rem_total(m, b_sum);
            if rem >= REMAINDER_TARGET && m < M_MAX {
                m += 1;
                continue;
            }
            let lam_d = if r_d > 0.0 { lebesgue(m) } else { 1.0 };
            let lam_f = if r_f > 0.0 { lebesgue(m) } else { 1.0 };
            // plus the rounding of evaluating the interpolant itself (L terms of size up to B)
            let bound = lam_d * lam_f * err + rem + 4.0 * (l as f64) * f64::EPSILON * (1.0 + b_sum);
            if bound < BOUND_TARGET {
                return Ok(TsegPlan { l, m, tau_l, b, b_sum, grid_err: err, bound });
            }
            break;
        }
        l *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The actual error of a plan at rate `(-d, w)` (w in rad/ms), over every sample time.
    fn actual(p: &TsegPlan, tau: &[f64], d: f64, w: f64) -> f64 {
        let mut e = 0.0f64;
        for (j, &t) in tau.iter().enumerate() {
            let (mut re, mut im) = ((-d * t).exp() * (w * t).cos(), (-d * t).exp() * (w * t).sin());
            for (i, &tl) in p.tau_l.iter().enumerate() {
                let (br, bi) = p.b[j * p.l + i];
                let (cr, ci) = ((-d * tl).exp() * (w * tl).cos(), (-d * tl).exp() * (w * tl).sin());
                re -= br * cr - bi * ci;
                im -= br * ci + bi * cr;
            }
            e = e.max(re.hypot(im));
        }
        e
    }

    #[test]
    fn least_squares_and_minimum_norm() {
        // overdetermined: the normal equations hold; underdetermined: the minimum-norm solution
        let a = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -1.0, 2.0, 0.0]];
        let y = [1.0, 0.0, -1.0, 2.0];
        let x = Lsq::new(&a).solve(&y);
        for col in &a {
            let r: f64 = (0..4).map(|i| col[i] * (y[i] - a[0][i] * x[0] - a[1][i] * x[1])).sum();
            assert!(r.abs() < 1e-12, "{r}");
        }
        // one equation x0 + x1 = 2: minimum norm (1, 1)
        let x = Lsq::new(&[vec![1.0], vec![1.0]]).solve(&[2.0]);
        assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 1.0).abs() < 1e-14);
        // the segmentation's own conditioning (singular values down to 1e-11 of the largest): a
        // right-hand side in the range is fitted to rounding, which an explicit pseudo-inverse
        // misses by 4.5e-6
        let (t, l) = (4.0, 24usize);
        let ws = cheb(TAU * -100.0 / 1000.0, TAU * 150.0 / 1000.0, 14);
        let g = ws.len();
        let tau_l: Vec<f64> = (0..l).map(|i| t * i as f64 / (l - 1) as f64).collect();
        let mut cols = vec![vec![0.0; 2 * g]; 2 * l];
        for (i, &ti) in tau_l.iter().enumerate() {
            for (k, &w) in ws.iter().enumerate() {
                let (c, s) = ((w * ti).cos(), (w * ti).sin());
                cols[i][k] = c;
                cols[i][g + k] = s;
                cols[l + i][k] = -s;
                cols[l + i][g + k] = c;
            }
        }
        let xt: Vec<f64> = (0..2 * l).map(|i| ((i * 7 % 11) as f64 - 5.0) / 5.0).collect();
        let y: Vec<f64> = (0..2 * g).map(|k| (0..2 * l).map(|i| cols[i][k] * xt[i]).sum()).collect();
        let x = Lsq::new(&cols).solve(&y);
        let res = (0..2 * g).map(|k| (y[k] - (0..2 * l).map(|i| cols[i][k] * x[i]).sum::<f64>()).abs()).fold(0.0, f64::max);
        assert!(res < 1e-12, "{res:e}");
    }

    #[test]
    fn the_certified_bound_is_never_exceeded_between_grid_points() {
        // fieldmap and decay stress cases, among them the second review's 100 to 1100 s^-1 decay
        // range; actual errors on a dense sampling of the rectangle, off the grid
        let t = 4.0;
        let tau: Vec<f64> = (0..200).map(|j| (j as f64 + 0.5) * t / 200.0).collect();
        let cases = [
            RateRect { d: [0.0, 0.0], f: [-25.0, 25.0] },
            RateRect { d: [0.0, 0.0], f: [-100.0, 150.0] },
            RateRect { d: [0.01, 0.07], f: [-25.0, 25.0] },
            RateRect { d: [0.1, 1.1], f: [0.0, 0.0] },
            RateRect { d: [0.1, 1.1], f: [-50.0, 50.0] },
            RateRect { d: [0.0, 0.0], f: [10.0, 10.0] },
        ];
        for rect in cases {
            let p = plan(&tau, t, rect).unwrap_or_else(|e| panic!("{rect:?}: {e}"));
            assert!(p.bound < BOUND_TARGET && p.l <= L_MAX);
            let mut worst = 0.0f64;
            for a in 0..=23 {
                for c in 0..=37 {
                    let d = rect.d[0] + (rect.d[1] - rect.d[0]) * a as f64 / 23.0;
                    let f = rect.f[0] + (rect.f[1] - rect.f[0]) * c as f64 / 37.0;
                    worst = worst.max(actual(&p, &tau, d, TAU * f / 1000.0));
                }
            }
            println!("{rect:?}: L {} m {} B {:.3} grid {:.2e} bound {:.2e} actual {:.2e}", p.l, p.m, p.b_sum, p.grid_err,
                     p.bound, worst);
            assert!(worst <= p.bound, "{rect:?}: actual {worst:e} above the bound {:e}", p.bound);
        }
    }

    #[test]
    fn a_rectangle_too_wide_is_refused() {
        let tau: Vec<f64> = (0..50).map(|j| (j as f64 + 0.5) * 0.4).collect();
        let e = plan(&tau, 20.0, RateRect { d: [0.0, 0.0], f: [-2000.0, 2000.0] }).unwrap_err();
        assert!(e.contains("64 segments") && e.contains("-2000"), "{e}");
    }
}
