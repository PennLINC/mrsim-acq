//! CPMG echo amplitudes by the extended phase graph (P5 addendum, part B, "Echo amplitudes").
//! Pure std.
//!
//! A 90-degree excitation about x, then refocusing pulses of angle `b` about y (the CPMG phase) at
//! echo spacing `ESP`, each flanked by equal crusher gradients, with T1 recovery and T2 decay
//! between pulses and the pulses instantaneous. The echo `e` amplitude is the `F+_0` state at the
//! echo, normalized so that the excitation leaves amplitude 1. Below 180 degrees stimulated echoes
//! contribute (T1 enters through the `Z` states); at 180 degrees the amplitude is
//! `exp(-e ESP / T2)`.
//!
//! The amplitudes are returned as logarithms, `-inf` for an exactly zero amplitude, so that a line
//! weight can be formed as one exponent with the within-echo decay (the addendum's rule: separate
//! factors can give `0 * inf`).
//!
//! Notation follows Weigel, "Extended phase graphs: dephasing, RF pulses, and echoes - pure and
//! simple", JMRI 2015.

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };
    fn new(re: f64, im: f64) -> Cx {
        Cx { re, im }
    }
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
    fn mul(self, o: Cx) -> Cx {
        Cx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, s: f64) -> Cx {
        Cx::new(self.re * s, self.im * s)
    }
    fn conj(self) -> Cx {
        Cx::new(self.re, -self.im)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// The phase graph: `fp[k]` = `F+_k`, `fm[k]` = `F-_k`, `z[k]` = `Z_k`, `k = 0..n`.
struct Graph {
    fp: Vec<Cx>,
    fm: Vec<Cx>,
    z: Vec<Cx>,
}

impl Graph {
    /// An RF pulse of flip `alpha` (rad) about the axis at phase `phi` (rad) from x (Weigel eq. 15).
    fn pulse(&mut self, alpha: f64, phi: f64) {
        let (c2, s2, sa, ca) = ((alpha / 2.0).cos().powi(2), (alpha / 2.0).sin().powi(2), alpha.sin(), alpha.cos());
        let e = |a: f64| Cx::new(a.cos(), a.sin());
        let (e2p, e2m, e1p, e1m) = (e(2.0 * phi), e(-2.0 * phi), e(phi), e(-phi));
        let i = Cx::new(0.0, 1.0);
        let mi = Cx::new(0.0, -1.0);
        for k in 0..self.fp.len() {
            let (p, m, z) = (self.fp[k], self.fm[k], self.z[k]);
            self.fp[k] = p.scale(c2).add(e2p.mul(m).scale(s2)).add(mi.mul(e1p).mul(z).scale(sa));
            self.fm[k] = e2m.mul(p).scale(s2).add(m.scale(c2)).add(i.mul(e1m).mul(z).scale(sa));
            self.z[k] = mi.mul(e1m).mul(p).scale(sa / 2.0).add(i.mul(e1p).mul(m).scale(sa / 2.0)).add(z.scale(ca));
        }
    }

    /// Free relaxation over a time with factors `e1 = exp(-dt/T1)`, `e2 = exp(-dt/T2)`; `Z_0`
    /// recovers toward the equilibrium 1.
    fn relax(&mut self, e1: f64, e2: f64) {
        for k in 0..self.fp.len() {
            self.fp[k] = self.fp[k].scale(e2);
            self.fm[k] = self.fm[k].scale(e2);
            self.z[k] = self.z[k].scale(e1);
        }
        self.z[0] = self.z[0].add(Cx::new(1.0 - e1, 0.0));
    }

    /// One unit of gradient dephasing: `F+` up one order, `F-` down one, `F+_0` from `F-_0`.
    fn shift(&mut self) {
        let n = self.fp.len();
        for k in (1..n).rev() {
            self.fp[k] = self.fp[k - 1];
        }
        for k in 0..n - 1 {
            self.fm[k] = self.fm[k + 1];
        }
        self.fm[n - 1] = Cx::ZERO;
        self.fp[0] = self.fm[0].conj();
    }
}

/// `exp(-dt / t)` for a time constant `t` in ms that may be infinite (no relaxation).
fn decay(dt: f64, t: f64) -> f64 {
    if t.is_infinite() { 1.0 } else { (-dt / t).exp() }
}

/// `ln A_e`, `e = 1..=n_echoes`, of a CPMG train: echo spacing `esp_ms`, refocusing angle
/// `refocusing_deg` in `(0, 180]`, relaxation times in ms (positive, `INFINITY` allowed).
pub fn epg_cpmg(n_echoes: usize, esp_ms: f64, refocusing_deg: f64, t1_ms: f64, t2_ms: f64) -> Vec<f64> {
    assert!(esp_ms > 0.0 && esp_ms.is_finite(), "echo spacing {esp_ms} ms");
    assert!(refocusing_deg > 0.0 && refocusing_deg <= 180.0, "refocusing angle {refocusing_deg}");
    assert!(t1_ms > 0.0 && t2_ms > 0.0, "relaxation times must be positive ({t1_ms}, {t2_ms})");
    // Each echo period adds one order before and one after its pulse.
    let n = 2 * n_echoes + 2;
    let mut g = Graph { fp: vec![Cx::ZERO; n], fm: vec![Cx::ZERO; n], z: vec![Cx::ZERO; n] };
    g.z[0] = Cx::new(1.0, 0.0);
    g.pulse(std::f64::consts::FRAC_PI_2, 0.0);
    let norm = g.fp[0].abs();
    let half = esp_ms / 2.0;
    let (e1, e2) = (decay(half, t1_ms), decay(half, t2_ms));
    let beta = refocusing_deg.to_radians();
    (0..n_echoes)
        .map(|_| {
            g.relax(e1, e2);
            g.shift();
            g.pulse(beta, std::f64::consts::FRAC_PI_2);
            g.shift();
            g.relax(e1, e2);
            let a = g.fp[0].abs() / norm;
            if a == 0.0 { f64::NEG_INFINITY } else { a.ln() }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, TAU};

    /// An independent isochromat simulation of the same train: `n_iso` spins dephased uniformly
    /// over one crusher cycle, rotation matrices for the pulses, Bloch relaxation between them.
    fn isochromats(n_echoes: usize, esp: f64, b_deg: f64, t1: f64, t2: f64, n_iso: usize) -> Vec<f64> {
        let rot_x = |m: [f64; 3], a: f64| [m[0], m[1] * a.cos() - m[2] * a.sin(), m[1] * a.sin() + m[2] * a.cos()];
        let rot_y = |m: [f64; 3], a: f64| [m[0] * a.cos() + m[2] * a.sin(), m[1], -m[0] * a.sin() + m[2] * a.cos()];
        let rot_z = |m: [f64; 3], a: f64| [m[0] * a.cos() - m[1] * a.sin(), m[0] * a.sin() + m[1] * a.cos(), m[2]];
        let (e1, e2) = (decay(esp / 2.0, t1), decay(esp / 2.0, t2));
        let relax = |m: [f64; 3]| [m[0] * e2, m[1] * e2, m[2] * e1 + 1.0 - e1];
        let mut spins: Vec<[f64; 3]> = vec![rot_x([0.0, 0.0, 1.0], FRAC_PI_2); n_iso];
        let b = b_deg.to_radians();
        (0..n_echoes)
            .map(|_| {
                for (j, m) in spins.iter_mut().enumerate() {
                    let th = TAU * j as f64 / n_iso as f64;
                    *m = relax(rot_z(rot_y(rot_z(relax(*m), th), b), th));
                }
                let (sx, sy) = spins.iter().fold((0.0, 0.0), |(a, b), m| (a + m[0], b + m[1]));
                (sx / n_iso as f64).hypot(sy / n_iso as f64)
            })
            .collect()
    }

    #[test]
    fn perfect_refocusing_is_the_t2_decay() {
        for (t1, t2) in [(1330.0, 80.0), (830.0, 70.0), (3000.0, 2000.0), (f64::INFINITY, 100.0)] {
            let la = epg_cpmg(30, 12.0, 180.0, t1, t2);
            for (i, l) in la.iter().enumerate() {
                let want = -((i + 1) as f64) * 12.0 / t2;
                // the amplitude, not its logarithm, to 1e-12 relative
                assert!((l - want).exp_m1().abs() <= 1e-12, "echo {}: {l} vs {want}", i + 1);
            }
        }
        // no decay at all
        assert!(epg_cpmg(10, 12.0, 180.0, f64::INFINITY, f64::INFINITY).iter().all(|l| *l == 0.0));
    }

    #[test]
    fn reduced_refocusing_matches_an_isochromat_simulation() {
        // GM, WM, CSF, blood (3 T, ms)
        for (t1, t2) in [(1330.0, 80.0), (830.0, 70.0), (3000.0, 2000.0), (1650.0, 165.0)] {
            for b in [130.0, 111.0, 160.0] {
                let la = epg_cpmg(30, 13.4, b, t1, t2);
                let iso = isochromats(30, 13.4, b, t1, t2, 2001);
                // n_iso uniformly spaced isochromats reproduce every dephasing order below n_iso
                // exactly (the orders here reach 60), so the two methods agree to round-off; 1e-9
                // leaves margin (measured: within 1e-11)
                for (e, (l, a)) in la.iter().zip(&iso).enumerate() {
                    assert!((l.exp() - a).abs() <= 1e-9, "b {b} T1 {t1} T2 {t2} echo {}: {} vs {a}", e + 1, l.exp());
                }
                // the first echoes are where reduced refocusing differs most from T2 decay
                assert!(la[0].exp() < 0.99 * (-13.4f64 / t2).exp(), "b {b}: no stimulated-echo signature");
            }
        }
    }

    #[test]
    fn extreme_relaxation_stays_finite() {
        let la = epg_cpmg(20, 10.0, 130.0, 1000.0, 1e-3);
        assert!(la.iter().all(|l| !l.is_nan() && *l <= 0.0), "{la:?}");
        let la = epg_cpmg(20, 10.0, 180.0, 1e-3, 1e-3);
        assert!(la.iter().all(|l| !l.is_nan()), "{la:?}");
    }
}
