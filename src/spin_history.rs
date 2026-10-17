//! Spin history of slice-selective excitations under head motion (P9 addendum, part B).
//!
//! The longitudinal magnetization of tissue that moves through the slice direction between
//! excitations: carried into a slab not yet fired, it is excited twice in one repetition; carried out
//! of one, it is skipped. This module steps the history through every excitation of a series and
//! reports, per excitation, the difference between the magnetization the excited tissue has and a
//! reference the caller gives (the closed form a static head would have), as images on the
//! excitation's slices. It knows nothing of ASL or diffusion; pure std, `par` across source planes.
//!
//! **State.** Every tracked source voxel (a class other than [`NO_CLASS`]) carries its z extent,
//! fixed in the head, as contiguous pieces: each piece holds `mz`, its longitudinal magnetization
//! just after its last excitation (normalised to `M0 = 1`), and `t`, that excitation's time. An
//! excitation that covers part of a voxel splits a piece at the slab's edge, and adjacent pieces in
//! the same state merge, so the history is exact along z (a static head on slabs that cut the voxels
//! keeps two pieces per cut voxel, each excited by its own slab). A voxel holds at most
//! [`MAX_PIECES`]; beyond that its two closest-in-state neighbouring pieces merge into their
//! length-weighted mean (counted in [`SpinHistory::merges`]).
//!
//! **Between excitations.** `Mz(T)` replays the series from `(mz, t)`: recovery toward 1 with the
//! voxel's `T1`, each [`Pulse::Inversion`] in `(t, T]` scaling by `1 - 2 epsilon`, each
//! [`Pulse::Saturation`] in `(t, T]` setting zero. A pulse never modifies the stored state, only an
//! excitation does. `T1 = 0` gives zero.
//!
//! **The slab.** A box: slice `k` excites the scanner-frame slice coordinate `[k - 1/2, k + 1/2)` over
//! the whole plane. A voxel's z extent is the interval of its own z length (in slab units) about its
//! moved centre; its tilt under rotation is ignored.
//!
//! **An excitation** at time `T`, transform `m` (world space, the head moves by `p' = m p`) and slices
//! `S`: each piece inside slice `k`'s slab adds `l V weight (Mz(T) - mz_ref(T1))` to `num` and
//! `l V weight mz_ref(T1)` to `den` of its voxel's class, at the in-plane cell of slice `k` nearest the
//! voxel's moved centre (`l` the piece's fraction of the voxel, `V` the voxel's volume over the slab
//! cell's), then holds `mz = 0`, `t = T` (a 90 degree excitation).

/// The class of a source voxel that is not tracked.
pub const NO_CLASS: u8 = u8::MAX;

/// The most pieces a voxel holds before its closest pair merges.
pub const MAX_PIECES: usize = 8;

/// A slab edge this close to a voxel's end (in voxel lengths) is taken as the end.
const SNAP: f64 = 1e-6;

/// The tissue in the head frame: a grid, and per voxel `T1` (seconds), weight and class.
pub struct Source<'a> {
    pub dims: [usize; 3],
    pub v2w: [[f64; 4]; 4],
    pub t1: &'a [f32],
    pub weight: &'a [f32],
    pub class: &'a [u8],
    pub n_classes: usize,
}

/// The slice stack in the scanner frame: the grid whose z index is the slice.
#[derive(Debug, Clone, Copy)]
pub struct Slabs {
    pub dims: [usize; 3],
    pub v2w: [[f64; 4]; 4],
}

/// A nonselective pulse on the series clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pulse {
    /// Scales `Mz` by `1 - 2 epsilon`.
    Inversion { t: f64, epsilon: f64 },
    /// Sets `Mz` to zero.
    Saturation { t: f64 },
}

impl Pulse {
    fn time(&self) -> f64 {
        match *self {
            Pulse::Inversion { t, .. } | Pulse::Saturation { t } => t,
        }
    }
}

/// One excited slice's terms, `[class][x + nx*y]` on the slab grid.
#[derive(Debug, Clone)]
pub struct SliceTerms {
    pub slice: usize,
    pub num: Vec<Vec<f32>>,
    pub den: Vec<Vec<f32>>,
}

/// A voxel's z extent as pieces: piece `i` covers `[start[i], start[i + 1])` of `[0, 1)`.
#[derive(Debug, Clone, Copy)]
struct Pieces {
    n: u8,
    start: [f32; MAX_PIECES],
    mz: [f32; MAX_PIECES],
    t: [f64; MAX_PIECES],
}

impl Pieces {
    /// One piece, never excited: fully relaxed.
    fn relaxed() -> Pieces {
        Pieces { n: 1, start: [0.0; MAX_PIECES], mz: [1.0; MAX_PIECES], t: [f64::NEG_INFINITY; MAX_PIECES] }
    }
    fn end(&self, i: usize) -> f32 {
        if i + 1 < self.n as usize { self.start[i + 1] } else { 1.0 }
    }
    /// Split at `u` (strictly inside a piece); false when the voxel is full.
    fn split(&mut self, u: f32) -> bool {
        let n = self.n as usize;
        let Some(i) = (0..n).find(|&i| self.start[i] < u && u < self.end(i)) else { return true };
        if n == MAX_PIECES {
            return false;
        }
        for j in (i + 1..n).rev() {
            self.start[j + 1] = self.start[j];
            self.mz[j + 1] = self.mz[j];
            self.t[j + 1] = self.t[j];
        }
        self.start[i + 1] = u;
        self.mz[i + 1] = self.mz[i];
        self.t[i + 1] = self.t[i];
        self.n += 1;
        true
    }
    fn remove(&mut self, i: usize) {
        let n = self.n as usize;
        for j in i..n - 1 {
            self.start[j] = self.start[j + 1];
            self.mz[j] = self.mz[j + 1];
            self.t[j] = self.t[j + 1];
        }
        self.n -= 1;
    }
    /// Merge adjacent pieces in the same state.
    fn coalesce(&mut self) {
        let mut i = 1;
        while i < self.n as usize {
            if self.mz[i] == self.mz[i - 1] && self.t[i] == self.t[i - 1] {
                self.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

/// The tracked voxels of one source z plane and their pieces.
struct Plane {
    z: usize,
    /// `(x, y)` of each tracked voxel.
    xy: Vec<(usize, usize)>,
    t1: Vec<f32>,
    weight: Vec<f32>,
    class: Vec<u8>,
    pieces: Vec<Pieces>,
}

/// The history of one series.
pub struct SpinHistory {
    v2w: [[f64; 4]; 4],
    n_classes: usize,
    slabs: Slabs,
    w2v_slab: [[f64; 4]; 4],
    pulses: Vec<Pulse>,
    planes: Vec<Plane>,
    /// The source grid's in-plane size (a per-shot weight is indexed like the source grid).
    src_nx: usize,
    src_ny: usize,
    /// A voxel's z length in slab units, and its volume over a slab cell's.
    vox_len: f64,
    vox_vol: f64,
    merges: u64,
}

fn det3(m: &[[f64; 4]; 4]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn affine_inverse(m: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let lin = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
    let li = crate::mat::inverse3(&lin).expect("an invertible grid affine");
    let it = crate::mat::matvec(&li, [m[0][3], m[1][3], m[2][3]]);
    [
        [li[0][0], li[0][1], li[0][2], -it[0]],
        [li[1][0], li[1][1], li[1][2], -it[1]],
        [li[2][0], li[2][1], li[2][2], -it[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mul4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

fn apply4(m: &[[f64; 4]; 4], p: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    ]
}

fn col_norm(m: &[[f64; 4]; 4], c: usize) -> f64 {
    (m[0][c] * m[0][c] + m[1][c] * m[1][c] + m[2][c] * m[2][c]).sqrt()
}

/// `Mz` at `t_to` from `mz` just after `t_from`, through the pulses in `(t_from, t_to]` (sorted).
#[inline]
fn replay(mz: f64, t_from: f64, t_to: f64, t1: f64, pulses: &[Pulse]) -> f64 {
    if t1 <= 0.0 {
        return 0.0;
    }
    let recover = |m: f64, dt: f64| if dt.is_infinite() { 1.0 } else { 1.0 - (1.0 - m) * (-dt / t1).exp() };
    let start = pulses.partition_point(|p| p.time() <= t_from);
    let (mut m, mut t) = (mz, t_from);
    for p in &pulses[start..] {
        let tp = p.time();
        if tp > t_to {
            break;
        }
        m = recover(m, tp - t);
        m = match *p {
            Pulse::Inversion { epsilon, .. } => m * (1.0 - 2.0 * epsilon),
            Pulse::Saturation { .. } => 0.0,
        };
        t = tp;
    }
    recover(m, t_to - t)
}

/// The terms one plane contributes: `(slice index in the shot, class, cell, num, den)`.
type Contribution = (usize, usize, usize, f64, f64);

impl SpinHistory {
    /// `pulses` in any order.
    pub fn new(source: Source, slabs: Slabs, mut pulses: Vec<Pulse>) -> SpinHistory {
        let [nx, ny, nz] = source.dims;
        let n = nx * ny * nz;
        assert!(source.t1.len() == n && source.weight.len() == n && source.class.len() == n, "one value per source voxel");
        pulses.sort_by(|a, b| a.time().total_cmp(&b.time()));
        let mut planes = Vec::new();
        for z in 0..nz {
            let mut pl = Plane { z, xy: Vec::new(), t1: Vec::new(), weight: Vec::new(), class: Vec::new(), pieces: Vec::new() };
            for y in 0..ny {
                for x in 0..nx {
                    let i = x + nx * (y + ny * z);
                    let c = source.class[i];
                    if c == NO_CLASS {
                        continue;
                    }
                    assert!((c as usize) < source.n_classes, "class {c} of voxel {i} is not below {}", source.n_classes);
                    pl.xy.push((x, y));
                    pl.t1.push(source.t1[i]);
                    pl.weight.push(source.weight[i]);
                    pl.class.push(c);
                    pl.pieces.push(Pieces::relaxed());
                }
            }
            if !pl.xy.is_empty() {
                planes.push(pl);
            }
        }
        let vox_len = col_norm(&source.v2w, 2) / col_norm(&slabs.v2w, 2);
        let vox_vol = det3(&source.v2w).abs() / det3(&slabs.v2w).abs();
        SpinHistory {
            v2w: source.v2w, n_classes: source.n_classes, slabs, w2v_slab: affine_inverse(&slabs.v2w), pulses, planes,
            src_nx: nx, src_ny: ny, vox_len, vox_vol, merges: 0,
        }
    }

    /// How often a voxel was full and merged its closest pieces (zero in every test case; a large
    /// count means the history is being averaged).
    pub fn merges(&self) -> u64 {
        self.merges
    }

    /// The fictitious volume before the series: each `(time, slices)` shot excites at `transform`,
    /// with no terms read.
    pub fn prime(&mut self, shots: &[(f64, &[usize])], transform: [[f64; 4]; 4]) {
        for &(t, slices) in shots {
            self.step(t, transform, slices, None, None);
        }
    }

    /// One shot: the terms of each of `slices` against `mz_ref(T1)`, then the excitation. `weight`
    /// replaces the source weight for this shot only (indexed like the source grid).
    pub fn excite(&mut self, t: f64, transform: [[f64; 4]; 4], slices: &[usize], mz_ref: &(dyn Fn(f64) -> f64 + Sync),
                  weight: Option<&[f32]>) -> Vec<SliceTerms> {
        self.step(t, transform, slices, Some(mz_ref), weight).expect("terms are read")
    }

    fn step(&mut self, t_now: f64, transform: [[f64; 4]; 4], slices: &[usize], mz_ref: Option<&(dyn Fn(f64) -> f64 + Sync)>,
            weight: Option<&[f32]>) -> Option<Vec<SliceTerms>> {
        let [snx, sny, snz] = self.slabs.dims;
        let cells = snx * sny;
        // source voxel coordinates -> slab voxel coordinates
        let m = mul4(&self.w2v_slab, &mul4(&transform, &self.v2w));
        let mut in_shot = vec![usize::MAX; snz];
        for (k, &z) in slices.iter().enumerate() {
            assert!(z < snz, "slice {z} is outside the {snz} slabs");
            in_shot[z] = k;
        }
        let (len, vol, ncls) = (self.vox_len, self.vox_vol, self.n_classes);
        let (nx_src, ny_src) = (self.src_nx, self.src_ny);
        let pulses = &self.pulses;
        let in_shot = &in_shot;
        // a plane's z range in slab units: its corners, plus half a voxel
        let plane_range = |z: usize| -> (f64, f64) {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for &(x, y) in &[(-0.5, -0.5), (nx_src as f64 - 0.5, -0.5), (-0.5, ny_src as f64 - 0.5), (nx_src as f64 - 0.5, ny_src as f64 - 0.5)] {
                let p = apply4(&m, [x, y, z as f64]);
                lo = lo.min(p[2]);
                hi = hi.max(p[2]);
            }
            (lo - len / 2.0, hi + len / 2.0)
        };
        let per_plane = |pl: &mut Plane| -> (Option<Vec<Contribution>>, u64) {
            let (lo, hi) = plane_range(pl.z);
            let k0 = (lo + 0.5).floor().max(0.0) as isize;
            let k1 = (hi + 0.5).floor().min(snz as f64 - 1.0) as isize;
            if k1 < k0 || !(k0..=k1).any(|k| in_shot[k as usize] != usize::MAX) {
                return (None, 0);
            }
            let mut out = Vec::new();
            let mut merges = 0u64;
            for (j, &(x, y)) in pl.xy.iter().enumerate() {
                let p = apply4(&m, [x as f64, y as f64, pl.z as f64]);
                let a = p[2] - len / 2.0;
                let (ka, kb) = ((a + 0.5).floor(), (a + len + 0.5).floor());
                let t1 = pl.t1[j] as f64;
                let w_src = match weight {
                    Some(w) => w[x + nx_src * (y + ny_src * pl.z)] as f64,
                    None => pl.weight[j] as f64,
                };
                let r = mz_ref.map(|f| f(t1));
                let cell = {
                    let (cx, cy) = (p[0].round(), p[1].round());
                    (cx >= 0.0 && cy >= 0.0 && cx < snx as f64 && cy < sny as f64).then(|| cx as usize + snx * cy as usize)
                };
                let pc = &mut pl.pieces[j];
                let mut k = ka;
                while k <= kb {
                    if k >= 0.0 && (k as usize) < snz && in_shot[k as usize] != usize::MAX {
                        // the slab's extent in the voxel's own coordinate, snapped to the voxel's
                        // ends within SNAP (rounding must not leave slivers on aligned grids)
                        let snap = |u: f64| -> f32 { if u < SNAP { 0.0 } else if u > 1.0 - SNAP { 1.0 } else { u as f32 } };
                        let u0 = snap((k - 0.5 - a) / len);
                        let u1 = snap((k + 0.5 - a) / len);
                        if u1 > u0 {
                            for u in [u0, u1] {
                                if u > 0.0 && u < 1.0 && !pc.split(u) {
                                    merges += 1;
                                    merge_closest(pc);
                                    pc.split(u);
                                }
                            }
                            let mut cache: Option<(f32, f64, f64)> = None;
                            for i in 0..pc.n as usize {
                                let (s, e) = (pc.start[i], pc.end(i));
                                if s < u0 || e > u1 {
                                    continue;
                                }
                                let mz_t = match cache {
                                    Some((cm, ct, v)) if cm == pc.mz[i] && ct == pc.t[i] => v,
                                    _ => {
                                        let v = replay(pc.mz[i] as f64, pc.t[i], t_now, t1, pulses);
                                        cache = Some((pc.mz[i], pc.t[i], v));
                                        v
                                    }
                                };
                                if let (Some(r), Some(cell)) = (r, cell) {
                                    let f = (e - s) as f64 * vol * w_src;
                                    out.push((in_shot[k as usize], pl.class[j] as usize, cell, f * (mz_t - r), f * r));
                                }
                                pc.mz[i] = 0.0;
                                pc.t[i] = t_now;
                            }
                            pc.coalesce();
                        }
                    }
                    k += 1.0;
                }
            }
            (Some(out), merges)
        };
        #[cfg(feature = "par")]
        let parts: Vec<(Option<Vec<Contribution>>, u64)> = {
            use rayon::prelude::*;
            self.planes.par_iter_mut().map(per_plane).collect()
        };
        #[cfg(not(feature = "par"))]
        let parts: Vec<(Option<Vec<Contribution>>, u64)> = self.planes.iter_mut().map(per_plane).collect();
        self.merges += parts.iter().map(|p| p.1).sum::<u64>();
        mz_ref?;
        // accumulate in f64, in plane order (deterministic with or without `par`)
        let mut acc: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> =
            slices.iter().map(|_| (vec![vec![0.0; cells]; ncls], vec![vec![0.0; cells]; ncls])).collect();
        for (part, _) in parts {
            for (k, c, cell, nm, dn) in part.into_iter().flatten() {
                acc[k].0[c][cell] += nm;
                acc[k].1[c][cell] += dn;
            }
        }
        let to32 = |v: Vec<Vec<f64>>| -> Vec<Vec<f32>> { v.into_iter().map(|c| c.into_iter().map(|x| x as f32).collect()).collect() };
        Some(slices.iter().zip(acc).map(|(&z, (nm, dn))| SliceTerms { slice: z, num: to32(nm), den: to32(dn) }).collect())
    }
}

/// A full voxel: merge the neighbouring pair whose merging averages least, the state difference
/// weighted by the pair's reduced length `la lb / (la + lb)`, into their length-weighted mean. It
/// picks thin slivers first: under a drift, the voxels at the ends of the slice stack collect one
/// sliver per volume, carried out of the stack and never excited again.
fn merge_closest(pc: &mut Pieces) {
    let n = pc.n as usize;
    let len = |i: usize| (pc.end(i) - pc.start[i]) as f64;
    let cost = |i: usize| {
        let d = (pc.mz[i] - pc.mz[i - 1]).abs() as f64 + (pc.t[i] - pc.t[i - 1]).abs().min(1e6);
        d * len(i) * len(i - 1) / (len(i) + len(i - 1))
    };
    let best = (1..n).min_by(|&a, &b| cost(a).total_cmp(&cost(b))).expect("a full voxel has pieces to merge");
    let (la, lb) = (pc.end(best - 1) - pc.start[best - 1], pc.end(best) - pc.start[best]);
    pc.mz[best - 1] = (pc.mz[best - 1] * la + pc.mz[best] * lb) / (la + lb);
    pc.t[best - 1] = pc.t[best - 1].max(pc.t[best]);
    pc.remove(best);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::{Pose, IDENTITY_MATRIX};

    const TR: f64 = 4.6;
    const T1: f64 = 1.33;
    const T_SIG: f64 = 3.6;
    const DT: f64 = 0.04;

    /// A uniform source of `nz` 1 mm planes, `n` x `n` in-plane, and slabs of 5 mm whose slice `k`
    /// covers source planes `5k .. 5k + 4` when `shift` is 0 (mm, the slab origin's offset).
    fn setup(n: usize, nz: usize, shift: f64) -> (Vec<f32>, Vec<f32>, Vec<u8>, Slabs) {
        let nv = n * n * nz;
        let mut s = IDENTITY_MATRIX;
        s[2][2] = 5.0;
        s[2][3] = 2.0 + shift;
        (vec![T1 as f32; nv], vec![1.0; nv], vec![0; nv], Slabs { dims: [n, n, nz / 5], v2w: s })
    }

    fn shot_t(v: usize, s: usize) -> f64 {
        v as f64 * TR + T_SIG + DT * s as f64
    }

    /// Run `nvol` volumes of sequential single-slice shots, shot `s` of volume `v` moved by
    /// `pose(v, s)`; per volume and slice, the summed `(num, den)`. `presat` puts a saturation at
    /// each repetition's start. Also returns the merge count.
    fn run(n: usize, nz: usize, shift: f64, nvol: usize, presat: bool, pose: impl Fn(usize, usize) -> Pose)
           -> (Vec<Vec<(f64, f64)>>, u64) {
        let (t1, w, c, slabs) = setup(n, nz, shift);
        let ns = slabs.dims[2];
        let pulses: Vec<Pulse> = if presat { (0..nvol).map(|v| Pulse::Saturation { t: v as f64 * TR }).collect() } else { Vec::new() };
        let mut h = SpinHistory::new(Source { dims: [n, n, nz], v2w: IDENTITY_MATRIX, t1: &t1, weight: &w, class: &c, n_classes: 1 },
                                     slabs, pulses);
        let center = [n as f64 / 2.0, n as f64 / 2.0, nz as f64 / 2.0];
        let mat = |p: Pose| if p == Pose::IDENTITY { IDENTITY_MATRIX } else { p.to_matrix(center) };
        let slices: Vec<Vec<usize>> = (0..ns).map(|k| vec![k]).collect();
        let prime: Vec<(f64, &[usize])> = (0..ns).map(|s| (shot_t(0, s) - TR, slices[s].as_slice())).collect();
        h.prime(&prime, mat(pose(0, 0)));
        let mut out = vec![vec![(0.0, 0.0); ns]; nvol];
        for (v, vol) in out.iter_mut().enumerate() {
            for (s, x) in vol.iter_mut().enumerate() {
                let t_read = T_SIG + DT * s as f64;
                let r = move |t1: f64| if presat { 1.0 - (-t_read / t1).exp() } else { 1.0 - (-TR / t1).exp() };
                let terms = h.excite(shot_t(v, s), mat(pose(v, s)), &slices[s], &r, None);
                *x = (terms[0].num[0].iter().map(|x| *x as f64).sum(), terms[0].den[0].iter().map(|x| *x as f64).sum());
            }
        }
        (out, h.merges())
    }

    fn ratio(x: (f64, f64)) -> f64 {
        (x.1 + x.0) / x.1
    }

    #[test]
    fn a_static_head_needs_no_correction() {
        for (presat, shift) in [(false, 0.0), (true, 0.0), (false, 0.3), (true, 1.7)] {
            let (out, merges) = run(4, 20, shift, 3, presat, |_, _| Pose::IDENTITY);
            assert_eq!(merges, 0);
            for (v, vol) in out.iter().enumerate() {
                for (s, x) in vol.iter().enumerate() {
                    // with the slabs shifted, the stack's first and last slices hold a part of a
                    // voxel the stack never excites otherwise: compare the interior
                    if shift != 0.0 && (s == 0 || s == vol.len() - 1) {
                        continue;
                    }
                    assert!(x.1 > 0.0 && x.0.abs() <= 1e-6 * x.1, "presat {presat} shift {shift} volume {v} slice {s}: {x:?}");
                }
            }
        }
    }

    #[test]
    fn in_plane_motion_needs_no_correction() {
        let p = Pose { rot_deg: [0.0, 0.0, 3.0], trans_mm: [1.7, -0.6, 0.0] };
        let (out, _) = run(6, 20, 0.0, 3, false, |v, s| if v == 1 && s >= 2 { p } else { Pose::IDENTITY });
        for vol in &out {
            for x in vol {
                assert!(x.0.abs() <= 1e-6 * x.1, "{x:?}");
            }
        }
    }

    /// A jump of a tenth of a slab (0.5 mm) toward the next slab at shot 2 of volume 1: that slab's
    /// tenth was excited `DT` earlier. The opposite jump skips a tenth of slab 2, brighter in the
    /// next volume unless that volume is presaturated.
    #[test]
    fn a_through_plane_jump_double_excites_and_skips() {
        let e_tr = 1.0 - (-TR / T1).exp();
        let up = Pose { trans_mm: [0.0, 0.0, 0.5], ..Pose::IDENTITY };
        let down = Pose { trans_mm: [0.0, 0.0, -0.5], ..Pose::IDENTITY };
        let jump = |p: Pose| move |v: usize, s: usize| if v == 1 && s >= 2 { p } else { Pose::IDENTITY };
        // double excitation
        let (out, merges) = run(4, 20, 0.0, 3, false, jump(up));
        assert_eq!(merges, 0);
        let want = 1.0 - 0.1 * (1.0 - (1.0 - (-DT / T1).exp()) / e_tr);
        assert!((ratio(out[1][2]) - want).abs() < 1e-5, "{} vs {want}", ratio(out[1][2]));
        assert!((want - 0.9031).abs() < 1e-4);
        for s in [0, 1] {
            assert!(out[1][s].0.abs() <= 1e-6 * out[1][s].1, "slices before the jump: {:?}", out[1][s]);
        }
        // the next slab's tenth from below was last excited a repetition plus DT ago: a hair brighter
        let r3 = ratio(out[1][3]);
        let want3 = 1.0 + 0.1 * ((1.0 - (-(TR + DT) / T1).exp()) / e_tr - 1.0);
        assert!((r3 - want3).abs() < 1e-6, "{r3} vs {want3}");
        // with presaturation, the dip against its own reference
        let (out_p, _) = run(4, 20, 0.0, 3, true, jump(up));
        let ref_p = 1.0 - (-(T_SIG + 2.0 * DT) / T1).exp();
        let want_p = 1.0 - 0.1 * (1.0 - (1.0 - (-DT / T1).exp()) / ref_p);
        assert!((ratio(out_p[1][2]) - want_p).abs() < 1e-5, "{} vs {want_p}", ratio(out_p[1][2]));
        // skipping, seen in the next volume
        let (out, _) = run(4, 20, 0.0, 3, false, jump(down));
        let want = 1.0 + 0.1 * ((1.0 - (-2.0 * TR / T1).exp()) / e_tr - 1.0);
        assert!((ratio(out[2][2]) - want).abs() < 1e-5, "{} vs {want}", ratio(out[2][2]));
        let (out_p, _) = run(4, 20, 0.0, 3, true, jump(down));
        assert!(out_p[2][2].0.abs() <= 1e-6 * out_p[2][2].1, "presaturation erases it: {:?}", out_p[2][2]);
    }

    /// A slow drift (0.3 mm per repetition, continuous): small corrections, smooth from volume to
    /// volume, with no merges.
    #[test]
    fn a_slow_drift_changes_smoothly() {
        let pose = |v: usize, s: usize| Pose { trans_mm: [0.0, 0.0, 0.3 * shot_t(v, s) / TR], ..Pose::IDENTITY };
        let (out, merges) = run(4, 20, 0.0, 10, false, pose);
        assert_eq!(merges, 0);
        for s in 1..3 {
            for v in 2..out.len() {
                let (a, b) = (ratio(out[v - 1][s]), ratio(out[v][s]));
                assert!((a - b).abs() < 1e-4 && (b - 1.0).abs() < 2e-3, "slice {s} volume {v}: {a} -> {b}");
            }
        }
    }

    #[test]
    fn weights_and_t1_zero() {
        let (t1, w, c, slabs) = setup(2, 10, 0.0);
        let mut t1b = t1.clone();
        let mut wb = w.clone();
        // voxel (0, 0, 2): T1 0; voxel (1, 0, 2): weight 3
        t1b[2 * 4] = 0.0;
        wb[1 + 2 * 4] = 3.0;
        let mut h = SpinHistory::new(Source { dims: [2, 2, 10], v2w: IDENTITY_MATRIX, t1: &t1b, weight: &wb, class: &c, n_classes: 1 },
                                     slabs, Vec::new());
        let r = |t1: f64| if t1 > 0.0 { 0.5 } else { 0.0 };
        // never excited: Mz = 1; slice 0 holds planes 0-4, each a fifth of the slab cell
        let a = h.excite(1.0, IDENTITY_MATRIX, &[0], &r, None);
        let cell = |x: usize, y: usize| x + 2 * y;
        assert!((a[0].den[0][cell(0, 0)] - 0.5 * 4.0 * 0.2).abs() < 1e-6, "four planes of T1 > 0: {}", a[0].den[0][cell(0, 0)]);
        assert!((a[0].num[0][cell(0, 0)] - 0.5 * 4.0 * 0.2).abs() < 1e-6, "T1 = 0 adds nothing to num either");
        assert!((a[0].den[0][cell(1, 0)] - 0.5 * (4.0 + 3.0) * 0.2).abs() < 1e-6, "weight-summed: {}", a[0].den[0][cell(1, 0)]);
        assert!((a[0].num[0][cell(1, 0)] - 0.5 * (4.0 + 3.0) * 0.2).abs() < 1e-6, "Mz = 1 against 0.5");
        // a per-call weight for one shot, the source weight on the next
        let doubled = vec![2.0f32; 40];
        let b = h.excite(1.0, IDENTITY_MATRIX, &[1], &r, Some(&doubled));
        let c2 = h.excite(1.0, IDENTITY_MATRIX, &[1], &r, None);
        assert!((b[0].den[0][cell(1, 1)] - 2.0 * c2[0].den[0][cell(1, 1)]).abs() < 1e-6);
    }

    #[test]
    fn pieces_split_coalesce_and_merge_when_full() {
        let mut p = Pieces::relaxed();
        assert!(p.split(0.25) && p.split(0.75) && p.n == 3);
        assert!(p.split(0.25) && p.n == 3, "an existing edge is not split again");
        p.mz[1] = 0.0;
        p.t[1] = 5.0;
        p.coalesce();
        assert_eq!(p.n, 3);
        p.mz[1] = 1.0;
        p.t[1] = f64::NEG_INFINITY;
        p.coalesce();
        assert_eq!(p.n, 1);
        for i in 1..MAX_PIECES {
            assert!(p.split(i as f32 / MAX_PIECES as f32));
            p.mz[i] = i as f32;
        }
        assert!(!p.split(0.99), "full");
        merge_closest(&mut p);
        assert_eq!(p.n as usize, MAX_PIECES - 1);
    }

    /// The cost on the motivating size (P9 plan, Task 6): `--release --features par -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn measure_the_motivating_series() {
        let (nx, ny, nz) = (197usize, 233usize, 100usize);
        let nv = nx * ny * nz;
        let t1 = vec![1.33f32; nv];
        let w = vec![1.0f32; nv];
        let c: Vec<u8> = (0..nv).map(|i| if i % 3 == 0 { 0 } else { NO_CLASS }).collect();
        let mut s = IDENTITY_MATRIX;
        s[0][0] = 197.0 / 128.0;
        s[1][1] = 233.0 / 136.0;
        s[2][2] = 5.0;
        s[2][3] = 2.0;
        let slabs = Slabs { dims: [128, 136, 20], v2w: s };
        let start = std::time::Instant::now();
        let mut h = SpinHistory::new(
            Source { dims: [nx, ny, nz], v2w: IDENTITY_MATRIX, t1: &t1, weight: &w, class: &c, n_classes: 1 }, slabs, Vec::new());
        let built = start.elapsed();
        let slices: Vec<Vec<usize>> = (0..20).map(|k| vec![k]).collect();
        let r = |t1: f64| 1.0 - (-4.6 / t1).exp();
        let center = [98.0, 116.0, 50.0];
        for v in 0..40 {
            for (k, sl) in slices.iter().enumerate() {
                let p = Pose { trans_mm: [0.0, 0.0, 0.5 * (v as f64 + k as f64 / 20.0) / 40.0], rot_deg: [0.5, 0.0, 0.3] };
                let _ = h.excite(v as f64 * 4.6 + 3.6 + 0.04 * k as f64, p.to_matrix(center), sl, &r, None);
            }
        }
        println!("spin history: built in {:.2?}, 800 excitations in {:.2?}, {} merges", built, start.elapsed() - built, h.merges());
    }
}
