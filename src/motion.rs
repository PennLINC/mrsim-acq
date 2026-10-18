//! Head motion: rigid transforms applied to the fibers (and mask) per volume — or, for the new
//! within-volume feature, per slice-group.
//!
//! Port of `SimulateMotion` (`itkTractsToDWIImageFilter.cpp:1456`). Modes:
//! - **Random** (`:1486`): each listed volume gets an independent uniform draw in `[-amp,+amp]`
//!   per axis; fibers+mask reset to baseline between moved volumes (`:1461`) → intermittent motion.
//! - **Linear** (`:1508`): constant increment `amp / n_moved` per listed volume; accumulates and
//!   persists → monotonic drift.
//! - **Trajectory** (new): an explicit per-volume (or per slice-group) 6-DOF pose — the thing
//!   Fiberfox can't take from its CLI. Enables drift+sneeze, real motion traces, and multiband
//!   within-volume motion.
//!
//! The applied per-unit transform must also be handed to [`crate::kspace`] so the fieldmap warp
//! tracks the moved head (Fiberfox: `SetTranslation`/`SetRotationMatrix`, `:270`).
//!
//! ## Applying the pose — reuse `trx-rs`, don't hand-roll the point loop
//! `trx-rs` already transforms streamlines: `apply_transform_in_place(&mut Tractogram, &TransformChain)`
//! (`../rust/trx-rs/src/transform.rs:26`, rayon-parallel with its `parallel` feature). A `TransformChain`
//! is built from `itk_transforms_rs::{Affine3, TransformChain}` via `chain.push_affine(Affine3::from_matrix(m))`
//! (`transform.rs:100`). So this module only needs to **compute** the per-unit pose matrix `m`
//! (`nalgebra::Matrix4<f64>`); trx-rs applies it — and the same chain composes with the ITK/H5
//! registration warps you already use.
//!
//! Poses are **absolute**, so apply each to a *fresh copy of the baseline streamlines* (clone →
//! `apply_transform`), mirroring Fiberfox's reset-to-baseline between moved volumes (`:1461`);
//! do not accumulate in place.

use crate::Vec3;

#[derive(Debug, Clone)]
pub enum MotionMode {
    Off,
    /// per-axis amplitude (mm) + (deg), impulses that return to baseline
    Random { trans_mm: Vec3, rot_deg: Vec3, volumes: Vec<usize> },
    /// per-axis end-of-scan totals (mm) + (deg), monotonic drift
    Linear { trans_mm: Vec3, rot_deg: Vec3, volumes: Vec<usize> },
    /// explicit pose per acquisition unit (volume, or slice-group when within_volume)
    Trajectory { poses: Vec<Pose> },
}

/// A rigid pose: rotation (deg, per axis) + translation (mm).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pose {
    pub rot_deg: Vec3,
    pub trans_mm: Vec3,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { rot_deg: [0.0; 3], trans_mm: [0.0; 3] };

    /// 4×4 rigid transform (world mm) about `center`, faithful to
    /// `FiberBundle::TransformFibers` (`mitkFiberBundle.cpp:1632`): `p' = R·(p−c) + c + t`, with
    /// `R = Rz·Ry·Rx` in degrees. Hand this to trx-rs via `Affine3::from_matrix` (real impl builds
    /// a `nalgebra::Matrix4`).
    pub fn to_matrix(&self, center: crate::Vec3) -> [[f64; 4]; 4] {
        let r = crate::mat::rotation_zyx_deg(self.rot_deg[0], self.rot_deg[1], self.rot_deg[2]);
        // translation part = c − R·c + t   (so that center maps to center + t)
        let rc = crate::mat::matvec(&r, center);
        let tx = center[0] - rc[0] + self.trans_mm[0];
        let ty = center[1] - rc[1] + self.trans_mm[1];
        let tz = center[2] - rc[2] + self.trans_mm[2];
        [
            [r[0][0], r[0][1], r[0][2], tx],
            [r[1][0], r[1][1], r[1][2], ty],
            [r[2][0], r[2][1], r[2][2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// The inverse of [`Pose::to_matrix`]: the pose whose matrix about `center` is `m`, for a rigid
    /// `m` with `R = Rz·Ry·Rx` and `|ry| < 90°` (the branch `to_matrix` produces for such angles).
    pub fn from_matrix(m: [[f64; 4]; 4], center: Vec3) -> Pose {
        let r = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
        let ry = (-r[2][0]).clamp(-1.0, 1.0).asin();
        let rx = r[2][1].atan2(r[2][2]);
        let rz = r[1][0].atan2(r[0][0]);
        // translation part = c − R·c + t, so t = m[:,3] − c + R·c
        let rc = crate::mat::matvec(&r, center);
        Pose {
            rot_deg: [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()],
            trans_mm: [m[0][3] - center[0] + rc[0], m[1][3] - center[1] + rc[1], m[2][3] - center[2] + rc[2]],
        }
    }
}

/// The 4×4 identity: a shot transform that moves nothing.
pub const IDENTITY_MATRIX: [[f64; 4]; 4] =
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// The pose of a trajectory at time `t`: linear in each of the six parameters between the nodes
/// `(node_times[i], poses[i])` (times non-decreasing), held at the first node before it and at the
/// last after it. Where two nodes share a time, the later one holds from that time on.
pub fn sample_trajectory(node_times: &[f64], poses: &[Pose], t: f64) -> Pose {
    assert_eq!(node_times.len(), poses.len(), "one time per pose");
    let n = poses.len();
    if n == 0 {
        return Pose::IDENTITY;
    }
    if t <= node_times[0] {
        return poses[0];
    }
    if t >= node_times[n - 1] {
        return poses[n - 1];
    }
    // the last node at or before t; t < the last time, so i + 1 exists
    let i = node_times.iter().rposition(|&x| x <= t).expect("t is past the first node");
    let (t0, t1) = (node_times[i], node_times[i + 1]);
    if t1 <= t0 {
        return poses[i + 1];
    }
    let f = (t - t0) / (t1 - t0);
    let (a, b) = (poses[i], poses[i + 1]);
    let lerp = |x: f64, y: f64| x + (y - x) * f;
    Pose {
        rot_deg: [lerp(a.rot_deg[0], b.rot_deg[0]), lerp(a.rot_deg[1], b.rot_deg[1]), lerp(a.rot_deg[2], b.rot_deg[2])],
        trans_mm: [lerp(a.trans_mm[0], b.trans_mm[0]), lerp(a.trans_mm[1], b.trans_mm[1]), lerp(a.trans_mm[2], b.trans_mm[2])],
    }
}

/// Deterministic, seedable PRNG (SplitMix64) so motion is reproducible without pulling `rand` into
/// the default build. (Fiberfox uses a Mersenne Twister; we only need *our own* reproducibility.)
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0,1)
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// uniform in [-amp, amp]  (matches Fiberfox `GetVariateWithClosedRange(2·amp) − amp`)
    fn signed(&mut self, amp: f64) -> f64 {
        2.0 * amp * self.unit() - amp
    }
}

fn as_flags(volumes: &[usize], n_units: usize) -> Vec<bool> {
    let mut f = vec![false; n_units];
    for &v in volumes {
        if v < n_units {
            f[v] = true;
        }
    }
    f
}

/// Resolve a motion mode into the per-unit **absolute** pose sequence — the ground truth to apply
/// (clone-baseline-then-transform) and to write to a motion TSV. Mirrors `SimulateMotion`
/// (`itkTractsToDWIImageFilter.cpp:1456`).
pub fn resolve_poses(mode: &MotionMode, n_units: usize, seed: u64) -> Vec<Pose> {
    let mut poses = vec![Pose::IDENTITY; n_units];
    match mode {
        MotionMode::Off => {}
        MotionMode::Random { trans_mm, rot_deg, volumes } => {
            // independent draw per listed volume; unlisted stay at baseline (reset-to-baseline).
            let mut rng = SplitMix64(seed.wrapping_add(0x1234_5678));
            for &v in volumes {
                if v >= n_units {
                    continue;
                }
                poses[v] = Pose {
                    rot_deg: [rng.signed(rot_deg[0]), rng.signed(rot_deg[1]), rng.signed(rot_deg[2])],
                    trans_mm: [rng.signed(trans_mm[0]), rng.signed(trans_mm[1]), rng.signed(trans_mm[2])],
                };
            }
        }
        MotionMode::Linear { trans_mm, rot_deg, volumes } => {
            // constant increment per listed volume; accumulates and persists (monotonic drift).
            let n = volumes.len().max(1) as f64;
            let flags = as_flags(volumes, n_units);
            let mut counter = 0.0_f64;
            for v in 0..n_units {
                if flags[v] {
                    counter += 1.0;
                }
                let k = counter;
                poses[v] = Pose {
                    rot_deg: [rot_deg[0] / n * k, rot_deg[1] / n * k, rot_deg[2] / n * k],
                    trans_mm: [trans_mm[0] / n * k, trans_mm[1] / n * k, trans_mm[2] / n * k],
                };
            }
        }
        MotionMode::Trajectory { poses: p } => {
            for (i, slot) in poses.iter_mut().enumerate() {
                if let Some(x) = p.get(i) {
                    *slot = *x;
                }
            }
        }
    }
    poses
}

/// Parse a qsiprep/`eddy`-style confounds TSV — columns `trans_x/y/z` (mm) and `rot_x/y/z`
/// (**radians**), one row per volume — into an absolute per-volume [`Pose`] trajectory (rotations
/// converted to the degrees `Pose` uses). `n/a` cells map to 0. This is the "real motion trace" path.
pub fn load_motion_tsv(path: &std::path::Path) -> Result<Vec<Pose>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty TSV")?;
    let cols: Vec<&str> = header.split('\t').collect();
    let find = |name: &str| {
        cols.iter().position(|&c| c == name).ok_or_else(|| format!("column '{name}' missing"))
    };
    let (tx, ty, tz) = (find("trans_x")?, find("trans_y")?, find("trans_z")?);
    let (rx, ry, rz) = (find("rot_x")?, find("rot_y")?, find("rot_z")?);
    let deg = 180.0 / std::f64::consts::PI;
    let mut poses = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let v = |i: usize| f.get(i).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(0.0);
        poses.push(Pose {
            trans_mm: [v(tx), v(ty), v(tz)],
            rot_deg: [v(rx) * deg, v(ry) * deg, v(rz) * deg],
        });
    }
    Ok(poses)
}

/// A timed motion trace (P10): rigid poses at sample times, seconds on the caller's acquisition
/// clock. Read with [`load_motion_trace`]; the pose at any time is [`sample_trajectory`] over the
/// samples ([`Trace::pose_at`]). Whether the trace covers the times a run samples depends on the
/// run, so [`Trace::check_covers`] is the caller's to call with them.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    /// Sample times (s), strictly increasing.
    pub times: Vec<f64>,
    pub poses: Vec<Pose>,
}

/// The clock of a trace file without a `time` column (a BIDS-physio-style sidecar's
/// `SamplingFrequency` and `StartTime`): sample `i` at `start_s + i / frequency_hz`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub frequency_hz: f64,
    pub start_s: f64,
}

impl Trace {
    /// The trace's pose at `t`.
    pub fn pose_at(&self, t: f64) -> Pose {
        sample_trajectory(&self.times, &self.poses, t)
    }

    /// The longest interval between two samples (s).
    pub fn max_interval(&self) -> f64 {
        self.times.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max)
    }

    /// Every one of `times` inside the trace's span, or an error naming the span `what` needs and the
    /// trace's. Never extrapolate: holding the last pose past the trace would invent stillness.
    pub fn check_covers(&self, times: &[f64], what: &str) -> Result<(), String> {
        if times.is_empty() {
            return Ok(());
        }
        let (lo, hi) = times.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| (a.min(t), b.max(t)));
        let (first, last) = (self.times[0], self.times[self.times.len() - 1]);
        if lo < first || hi > last {
            return Err(format!(
                "{what}: the run samples the trace from {lo} s to {hi} s, but it spans {first} s to {last} s; a timed \
                 trace must cover every time it is sampled at (it is never extrapolated)"));
        }
        Ok(())
    }
}

/// Whether a motion TSV's header has a `time` column (P10): with the caller's sidecar read, how a
/// timed trace is told from an untimed one before either loader runs.
pub fn has_time_column(path: &std::path::Path) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let header = text.lines().map(|l| l.trim_end_matches('\r')).find(|l| !l.trim().is_empty()).unwrap_or("");
    Ok(header.split('\t').any(|c| c.trim() == "time"))
}

/// Read a timed motion trace (P10): the columns `trans_x/y/z` (mm) and `rot_x/y/z` (radians, held
/// in degrees) of [`load_motion_tsv`], with times from a `time` column (s) or from `sampling`, never
/// both and never neither. Strict, unlike [`load_motion_tsv`], which stays the reader of untimed
/// traces: every cell present and finite (`n/a` is 0 in the six pose columns), times strictly
/// increasing, at least two samples. The rotations are converted with [`load_motion_tsv`]'s
/// arithmetic, so the same cells give the same poses bit for bit.
pub fn load_motion_trace(path: &std::path::Path, sampling: Option<Sampling>) -> Result<Trace, String> {
    let what = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {what}: {e}"))?;
    let mut lines = text.lines().map(|l| l.trim_end_matches('\r')).filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().ok_or_else(|| format!("{what}: empty TSV"))?.split('\t').map(str::trim).collect();
    let find = |name: &str| header.iter().position(|&c| c == name);
    let mut idx = [0usize; 6];
    for (k, name) in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"].iter().enumerate() {
        idx[k] = find(name).ok_or_else(|| format!("{what}: column '{name}' missing"))?;
    }
    let time_col = find("time");
    match (time_col, sampling) {
        (Some(_), Some(_)) => {
            return Err(format!("{what}: a time column and a sampling rate (a sidecar SamplingFrequency): two clocks for one trace"))
        }
        (None, None) => return Err(format!("{what}: no time column and no sampling rate; not a timed trace")),
        (None, Some(s)) => {
            if !(s.frequency_hz.is_finite() && s.frequency_hz > 0.0) {
                return Err(format!("{what}: SamplingFrequency {} is not a positive number", s.frequency_hz));
            }
            if !s.start_s.is_finite() {
                return Err(format!("{what}: StartTime {} is not finite", s.start_s));
            }
        }
        (Some(_), None) => {}
    }
    let names = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"];
    let deg = 180.0 / std::f64::consts::PI;
    let (mut times, mut poses) = (Vec::new(), Vec::new());
    for (i, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split('\t').map(str::trim).collect();
        let cell = |c: usize, name: &str| cells.get(c).copied().ok_or_else(|| format!("{what} row {i}: column {name} is missing"));
        let mut v = [0.0f64; 6];
        for k in 0..6 {
            let s = cell(idx[k], names[k])?;
            v[k] = if s == "n/a" {
                0.0
            } else {
                match s.parse::<f64>() {
                    Ok(x) if x.is_finite() => x,
                    _ => return Err(format!("{what} row {i}: {} = {s:?} is not a finite number or n/a", names[k])),
                }
            };
        }
        let t = match (time_col, sampling) {
            (Some(c), _) => {
                let s = cell(c, "time")?;
                match s.parse::<f64>() {
                    Ok(x) if x.is_finite() => x,
                    _ => return Err(format!("{what} row {i}: time = {s:?} is not a finite number")),
                }
            }
            (None, Some(sm)) => sm.start_s + i as f64 / sm.frequency_hz,
            (None, None) => unreachable!("refused above"),
        };
        if let Some(&prev) = times.last() {
            if t <= prev {
                return Err(format!("{what} row {i}: time {t} s does not increase (the row before is at {prev} s)"));
            }
        }
        times.push(t);
        poses.push(Pose { trans_mm: [v[0], v[1], v[2]], rot_deg: [v[3] * deg, v[4] * deg, v[5] * deg] });
    }
    if times.len() < 2 {
        return Err(format!("{what}: a timed trace needs at least two samples, it has {}", times.len()));
    }
    Ok(Trace { times, poses })
}

/// Trilinear sample of `vol` (layout `x + nx*(y + ny*z)`) at continuous voxel coordinate `p`;
/// zero outside the grid. Public because consumers' own warps (TRXScan's `gnl`) use it.
#[inline]
pub fn trilinear(vol: &[f32], dims: [usize; 3], p: Vec3) -> f32 {
    let [nx, ny, nz] = dims;
    let (x, y, z) = (p[0], p[1], p[2]);
    if x < 0.0 || y < 0.0 || z < 0.0 || x > (nx - 1) as f64 || y > (ny - 1) as f64 || z > (nz - 1) as f64 {
        return 0.0;
    }
    let (x0, y0, z0) = (x.floor() as usize, y.floor() as usize, z.floor() as usize);
    let (x1, y1, z1) = ((x0 + 1).min(nx - 1), (y0 + 1).min(ny - 1), (z0 + 1).min(nz - 1));
    let (fx, fy, fz) = (x - x0 as f64, y - y0 as f64, z - z0 as f64);
    let at = |xi: usize, yi: usize, zi: usize| vol[xi + nx * (yi + ny * zi)] as f64;
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let c00 = lerp(at(x0, y0, z0), at(x1, y0, z0), fx);
    let c10 = lerp(at(x0, y1, z0), at(x1, y1, z0), fx);
    let c01 = lerp(at(x0, y0, z1), at(x1, y0, z1), fx);
    let c11 = lerp(at(x0, y1, z1), at(x1, y1, z1), fx);
    lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz) as f32
}

/// The FOV centre in world coordinates (the rotation centre for a pose on this grid).
pub fn fov_center(dims: [usize; 3], v2w: [[f64; 4]; 4]) -> Vec3 {
    use crate::mat;
    let [nx, ny, nz] = dims;
    let lin = linear3(v2w);
    let c = mat::matvec(&lin, [(nx as f64 - 1.0) / 2.0, (ny as f64 - 1.0) / 2.0, (nz as f64 - 1.0) / 2.0]);
    [c[0] + v2w[0][3], c[1] + v2w[1][3], c[2] + v2w[2][3]]
}

fn linear3(v2w: [[f64; 4]; 4]) -> crate::mat::Mat3 {
    [
        [v2w[0][0], v2w[0][1], v2w[0][2]],
        [v2w[1][0], v2w[1][1], v2w[1][2]],
        [v2w[2][0], v2w[2][1], v2w[2][2]],
    ]
}

/// Resample one 3D scalar volume by an absolute rigid pose (head moves by `p' = R(p−c)+c+t`,
/// rotation about the FOV centre). Backward-maps each output voxel and trilinearly samples `src`.
pub fn resample_by_pose(src: &[f32], dims: [usize; 3], v2w: [[f64; 4]; 4], pose: Pose) -> Vec<f32> {
    use crate::mat;
    let [nx, ny, nz] = dims;
    let nvox = nx * ny * nz;
    if pose == Pose::IDENTITY {
        return src.to_vec();
    }
    let lin = linear3(v2w);
    let origin = [v2w[0][3], v2w[1][3], v2w[2][3]];
    let linv = match mat::inverse3(&lin) {
        Some(m) => m,
        None => return src.to_vec(),
    };
    let c = fov_center(dims, v2w);
    let rt = mat::transpose(&mat::rotation_zyx_deg(pose.rot_deg[0], pose.rot_deg[1], pose.rot_deg[2]));
    let t = pose.trans_mm;
    let mut out = vec![0.0f32; nvox];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let w = mat::matvec(&lin, [x as f64, y as f64, z as f64]);
                let wv = [w[0] + origin[0], w[1] + origin[1], w[2] + origin[2]];
                let d = [wv[0] - c[0] - t[0], wv[1] - c[1] - t[1], wv[2] - c[2] - t[2]];
                let pr = mat::matvec(&rt, d);
                let hv = mat::matvec(&linv, [pr[0] + c[0] - origin[0], pr[1] + c[1] - origin[1], pr[2] + c[2] - origin[2]]);
                out[x + nx * (y + ny * z)] = trilinear(src, dims, hv);
            }
        }
    }
    out
}

/// Fast (approximate) motion: resample each volume of each compartment image by its pose. This
/// moves the head geometrically but does NOT rotate the fiber orientations relative to the
/// gradients — use TRXScan's `compartments::generate_compartments_moving` for the faithful
/// (re-simulate-from-moved-streamlines) version. Fieldmap stays fixed (scanner space).
pub fn apply_motion(images: &mut [Vec<f32>], dims: [usize; 3], ngrad: usize, v2w: [[f64; 4]; 4], poses: &[Pose]) {
    let nvox = dims[0] * dims[1] * dims[2];
    for g in 0..ngrad {
        let pose = poses.get(g).copied().unwrap_or(Pose::IDENTITY);
        if pose == Pose::IDENTITY {
            continue;
        }
        for img in images.iter_mut() {
            let src: Vec<f32> = (0..nvox).map(|v| img[v * ngrad + g]).collect();
            let moved = resample_by_pose(&src, dims, v2w, pose);
            for v in 0..nvox {
                img[v * ngrad + g] = moved[v];
            }
        }
    }
}

/// Multiband slice schedule: acquisition order of shots → the anatomical slices excited in each.
/// `mb` slices (separated by `nz/mb`) fire simultaneously per shot; `n_shots = nz/mb` shots per
/// volume, acquired sequentially or interleaved (even groups then odd).
pub fn slice_schedule(nz: usize, mb: usize, interleaved: bool) -> Vec<Vec<usize>> {
    let mb = mb.max(1);
    let n_groups = nz / mb; // simultaneously-excited group count = shots per volume
    if n_groups == 0 {
        return vec![(0..nz).collect()];
    }
    let group_slices = |gi: usize| -> Vec<usize> { (0..mb).map(|j| gi + j * n_groups).filter(|&z| z < nz).collect() };
    let order: Vec<usize> = if interleaved {
        (0..n_groups).step_by(2).chain((1..n_groups).step_by(2)).collect()
    } else {
        (0..n_groups).collect()
    };
    order.into_iter().map(group_slices).collect()
}

/// The shot schedule a set of per-slice excitation offsets describes (P9): slices whose offsets are
/// bit-identical form one shot, the shots in increasing offset, the slices of a shot ascending.
/// Any slice order and any grouping; for the schedules [`slice_schedule`] represents, the same
/// shots in the same order.
pub fn slice_schedule_from_times(offsets: &[f64]) -> Vec<Vec<usize>> {
    let mut times: Vec<f64> = offsets.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| a.to_bits() == b.to_bits());
    times.iter().map(|t| (0..offsets.len()).filter(|&z| offsets[z].to_bits() == t.to_bits()).collect()).collect()
}

/// The given slices of `src` resampled by the world-space rigid transform `m` (the head moves by
/// `p' = m·p`): each output voxel is backward-mapped through `m⁻¹` and trilinearly sampled, zero
/// outside the grid. The result holds the slices in the order given, each `x + nx*y`. The identity
/// copies them exactly. `m` folds the rotation centre into its translation, so for the same pose this
/// agrees with [`resample_by_pose`] to rounding, not bit for bit.
pub fn resample_slices_by_matrix(src: &[f32], dims: [usize; 3], v2w: [[f64; 4]; 4], m: [[f64; 4]; 4],
                                 slices: &[usize]) -> Vec<f32> {
    use crate::mat;
    let [nx, ny, _] = dims;
    let slab = nx * ny;
    let mut out = vec![0.0f32; slab * slices.len()];
    if m == IDENTITY_MATRIX {
        for (k, &z) in slices.iter().enumerate() {
            out[k * slab..(k + 1) * slab].copy_from_slice(&src[z * slab..(z + 1) * slab]);
        }
        return out;
    }
    let lin = linear3(v2w);
    let origin = [v2w[0][3], v2w[1][3], v2w[2][3]];
    let Some(linv) = mat::inverse3(&lin) else {
        for (k, &z) in slices.iter().enumerate() {
            out[k * slab..(k + 1) * slab].copy_from_slice(&src[z * slab..(z + 1) * slab]);
        }
        return out;
    };
    // rigid inverse: m⁻¹·w = Rᵀ(w − t)
    let rt = mat::transpose(&[[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]);
    let t = [m[0][3], m[1][3], m[2][3]];
    for (k, &z) in slices.iter().enumerate() {
        for y in 0..ny {
            for x in 0..nx {
                let w = mat::matvec(&lin, [x as f64, y as f64, z as f64]);
                let d = [w[0] + origin[0] - t[0], w[1] + origin[1] - t[1], w[2] + origin[2] - t[2]];
                let pr = mat::matvec(&rt, d);
                let hv = mat::matvec(&linv, [pr[0] - origin[0], pr[1] - origin[1], pr[2] - origin[2]]);
                out[k * slab + x + nx * y] = trilinear(src, dims, hv);
            }
        }
    }
    out
}

/// Per-shot rigid motion and dropout on the per-compartment signal, in place (P9). For each volume
/// `g` and shot `k`, the slices `schedule[k]` (local indices) are resampled from the volume's
/// pre-motion signal by the world-space transform `transforms[g][k]`, then multiplied by each factor
/// of `attenuation[g][k]` in order (one per event, as [`apply_multiband_motion_slab`] applies them).
/// A shot with the identity transform and no factor is left untouched. Build the transforms about
/// the whole slice stack's field-of-view centre, so a slab and the full volume move alike.
pub fn apply_shot_transforms(
    images: &mut [Vec<f32>],
    dims: [usize; 3],
    ngrad: usize,
    v2w: [[f64; 4]; 4],
    schedule: &[Vec<usize>],
    transforms: &[Vec<[[f64; 4]; 4]>],
    attenuation: &[Vec<Vec<f32>>],
) {
    let [nx, ny, nz] = dims;
    let (slab, nvox) = (nx * ny, nx * ny * nz);
    assert_eq!(transforms.len(), ngrad, "one transform list per volume");
    assert_eq!(attenuation.len(), ngrad, "one attenuation list per volume");
    for g in 0..ngrad {
        assert_eq!(transforms[g].len(), schedule.len(), "one transform per shot");
        assert_eq!(attenuation[g].len(), schedule.len(), "one factor list per shot");
        let moved = transforms[g].iter().any(|m| *m != IDENTITY_MATRIX);
        for img in images.iter_mut() {
            if moved {
                let orig: Vec<f32> = (0..nvox).map(|v| img[v * ngrad + g]).collect();
                for (k, m) in transforms[g].iter().enumerate() {
                    if *m == IDENTITY_MATRIX || schedule[k].is_empty() {
                        continue;
                    }
                    let rs = resample_slices_by_matrix(&orig, dims, v2w, *m, &schedule[k]);
                    for (j, &z) in schedule[k].iter().enumerate() {
                        for xy in 0..slab {
                            img[(z * slab + xy) * ngrad + g] = rs[j * slab + xy];
                        }
                    }
                }
            }
            for (k, factors) in attenuation[g].iter().enumerate() {
                for &a in factors {
                    for &z in &schedule[k] {
                        for xy in 0..slab {
                            img[(z * slab + xy) * ngrad + g] *= a;
                        }
                    }
                }
            }
        }
    }
}

/// A synthetic within-volume bulk-motion event: at `shot` of `volume` the head jumps (geometric)
/// and the diffusion signal of that shot's slices drops out (`severity` 0..1).
#[derive(Debug, Clone, Copy)]
pub struct MotionEvent {
    pub volume: usize,
    pub shot: usize,
    pub severity: f32,
    pub jump_mm: [f64; 3],
    pub jump_deg: [f64; 3],
}

/// Ground truth of a dropped shot (for scoring `eddy --repol` / SHORELine outlier detection).
#[derive(Debug, Clone)]
pub struct DroppedShot {
    pub volume: usize,
    pub shot: usize,
    pub slices: Vec<usize>,
    pub attenuation: f32,
}

/// How a dropped shot's signal is attenuated. Diffusion scales with b-value and exempts b0;
/// a sequence without a diffusion weighting has nothing to scale with.
#[derive(Debug, Clone)]
pub enum DropoutLaw {
    /// `1 - severity * (drive / drive_max)`, with `drive < floor` exempt. Diffusion passes
    /// b-values and a floor of 50.0, reproducing the previous b0-exempt behavior exactly
    /// (`drive_max` is the same `fold(0, max)` the scheme's `b_max` was).
    Scaled { drive: Vec<f64>, floor: f64 },
    /// `1 - severity * (drive / max)` with a given normaliser, `drive < floor` exempt, and nothing
    /// attenuated when `max <= 0`; a volume missing from `drive` has drive 0. TRXScan's diffusion
    /// dropout divides by the caller's `b_max` this way.
    ScaledTo { drive: Vec<f64>, max: f64, floor: f64 },
    /// `1 - severity`, applied to every volume alike.
    Uniform,
}

impl DropoutLaw {
    /// Multiplier applied to volume `volume`'s dropped shot for an event of `severity`.
    pub fn attenuation(&self, volume: usize, severity: f32) -> f32 {
        match self {
            DropoutLaw::Uniform => 1.0 - severity,
            DropoutLaw::Scaled { drive, floor } => {
                let d = drive.get(volume).copied().unwrap_or(0.0);
                let d_max = drive.iter().cloned().fold(0.0f64, f64::max);
                if d < *floor || d_max <= 0.0 { 1.0 } else { 1.0 - severity * (d / d_max) as f32 }
            }
            DropoutLaw::ScaledTo { drive, max, floor } => {
                let d = drive.get(volume).copied().unwrap_or(0.0);
                if d < *floor || *max <= 0.0 { 1.0 } else { 1.0 - severity * (d / max) as f32 }
            }
        }
    }
}

/// Apply multiband within-volume motion + slice dropout to the per-compartment signal in place.
/// For each event: (1) the head jumps for that shot and the ones after it within the volume (a
/// per-shot 3D resample of the affected slices — the within-volume wobble), and (2) the shot's
/// slices' signal is attenuated by `law` (for diffusion, `1 − severity·(b/b_max)`, b0 exempt;
/// higher b drops harder) — the phenomenological dropout `eddy --repol` targets. Returns the
/// dropped-shot ground truth. Runs after volume-level motion, before k-space.
#[allow(clippy::too_many_arguments)]
pub fn apply_multiband_motion(
    images: &mut [Vec<f32>],
    dims: [usize; 3],
    ngrad: usize,
    v2w: [[f64; 4]; 4],
    mb: usize,
    interleaved: bool,
    law: &DropoutLaw,
    events: &[MotionEvent],
) -> Vec<DroppedShot> {
    apply_multiband_motion_slab(images, dims, ngrad, v2w, mb, interleaved, law, events, None, None)
}

/// [`apply_multiband_motion`] on a slab of a larger volume (ported from TRXScan `main`):
/// `slice_z` gives the full-FOV z index of each local slice and `nz_full` the full slice count, so
/// the multiband shot schedule (and therefore which shots touch which slices) is the full
/// volume's. Shots whose slices lie outside the slab are skipped; the returned
/// [`DroppedShot::slices`] are full-FOV indices. The geometric jump resamples within the slab, so
/// keep a few slices of context around the ones of interest.
#[allow(clippy::too_many_arguments)]
pub fn apply_multiband_motion_slab(
    images: &mut [Vec<f32>],
    dims: [usize; 3],
    ngrad: usize,
    v2w: [[f64; 4]; 4],
    mb: usize,
    interleaved: bool,
    law: &DropoutLaw,
    events: &[MotionEvent],
    slice_z: Option<&[usize]>,
    nz_full: Option<usize>,
) -> Vec<DroppedShot> {
    let [nx, ny, nz] = dims;
    let nvox = nx * ny * nz;
    let nz_full = nz_full.unwrap_or(nz);
    if let Some(sz) = slice_z {
        assert_eq!(sz.len(), nz, "slice_z must name every local slice");
    }
    // full-FOV schedule, then each shot's slices mapped to LOCAL indices (dropping any outside)
    let schedule_full = slice_schedule(nz_full, mb, interleaved);
    let local_of = |zg: usize| -> Option<usize> {
        match slice_z {
            None => (zg < nz).then_some(zg),
            Some(sz) => sz.iter().position(|&z| z == zg),
        }
    };
    let schedule: Vec<Vec<usize>> =
        schedule_full.iter().map(|shot| shot.iter().filter_map(|&z| local_of(z)).collect()).collect();
    let n_shots = schedule.len();
    let at3 = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
    let mut gt = Vec::new();

    for g in 0..ngrad {
        let evs: Vec<&MotionEvent> = events.iter().filter(|e| e.volume == g && e.shot < n_shots).collect();
        if evs.is_empty() {
            continue;
        }
        // within-volume cumulative pose per shot (each event's jump persists for later shots)
        let mut shot_pose = vec![Pose::IDENTITY; n_shots];
        let mut cum = Pose::IDENTITY;
        for (k, sp) in shot_pose.iter_mut().enumerate() {
            for e in &evs {
                if e.shot == k {
                    for i in 0..3 {
                        cum.trans_mm[i] += e.jump_mm[i];
                        cum.rot_deg[i] += e.jump_deg[i];
                    }
                }
            }
            *sp = cum;
        }
        // geometric: resample each moved shot's slices from the volume's pre-motion signal
        if shot_pose.iter().any(|p| *p != Pose::IDENTITY) {
            for img in images.iter_mut() {
                let orig: Vec<f32> = (0..nvox).map(|v| img[v * ngrad + g]).collect();
                for (k, &pose) in shot_pose.iter().enumerate() {
                    if pose == Pose::IDENTITY {
                        continue;
                    }
                    let rs = resample_by_pose(&orig, dims, v2w, pose);
                    for &z in &schedule[k] {
                        for y in 0..ny {
                            for x in 0..nx {
                                img[at3(x, y, z) * ngrad + g] = rs[at3(x, y, z)];
                            }
                        }
                    }
                }
            }
        }
        // dropout: attenuate each event shot's slices per the law
        for e in &evs {
            let atten = law.attenuation(g, e.severity);
            for img in images.iter_mut() {
                for &z in &schedule[e.shot] {
                    for y in 0..ny {
                        for x in 0..nx {
                            img[at3(x, y, z) * ngrad + g] *= atten;
                        }
                    }
                }
            }
            gt.push(DroppedShot { volume: g, shot: e.shot, slices: schedule_full[e.shot].clone(), attenuation: atten });
        }
    }
    gt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_pose_is_the_identity_matrix() {
        let m = Pose::IDENTITY.to_matrix([10.0, -5.0, 3.0]);
        for i in 0..3 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - want).abs() < 1e-9, "{i},{j}={}", m[i][j]);
            }
        }
    }

    #[test]
    fn rotation_leaves_center_fixed_plus_translation() {
        let center = [4.0, 2.0, -1.0];
        let p = Pose { rot_deg: [0.0, 0.0, 30.0], trans_mm: [1.0, 0.0, 0.0] };
        let m = p.to_matrix(center);
        // apply to the center point: should land at center + translation
        let c = [
            m[0][0] * center[0] + m[0][1] * center[1] + m[0][2] * center[2] + m[0][3],
            m[1][0] * center[0] + m[1][1] * center[1] + m[1][2] * center[2] + m[1][3],
            m[2][0] * center[0] + m[2][1] * center[1] + m[2][2] * center[2] + m[2][3],
        ];
        assert!((c[0] - (center[0] + 1.0)).abs() < 1e-9);
        assert!((c[1] - center[1]).abs() < 1e-9);
        assert!((c[2] - center[2]).abs() < 1e-9);
    }

    #[test]
    fn linear_drift_ramps_to_amplitude() {
        // all 74 non-baseline volumes listed → smooth ramp 0 → amp (matches the 3 mm Fiberfox run)
        let n = 75;
        let volumes: Vec<usize> = (1..n).collect();
        let poses = resolve_poses(
            &MotionMode::Linear { trans_mm: [3.0, 3.0, 2.0], rot_deg: [3.0, 2.0, 2.0], volumes },
            n,
            0,
        );
        assert_eq!(poses[0].trans_mm, [0.0, 0.0, 0.0]);
        let last = poses[n - 1];
        assert!((last.trans_mm[0] - 3.0).abs() < 1e-9, "{:?}", last.trans_mm);
        assert!((last.rot_deg[2] - 2.0).abs() < 1e-9);
        // monotonic
        for v in 1..n {
            assert!(poses[v].trans_mm[0] >= poses[v - 1].trans_mm[0] - 1e-12);
        }
    }

    #[test]
    fn random_moves_only_listed_within_bounds_and_is_deterministic() {
        let mode = MotionMode::Random {
            trans_mm: [2.5, 2.5, 2.5],
            rot_deg: [2.0, 2.0, 2.0],
            volumes: vec![4, 8, 13],
        };
        let a = resolve_poses(&mode, 20, 42);
        let b = resolve_poses(&mode, 20, 42);
        for v in 0..20 {
            // determinism
            assert_eq!(a[v].trans_mm, b[v].trans_mm);
            if [4, 8, 13].contains(&v) {
                for i in 0..3 {
                    assert!(a[v].trans_mm[i].abs() <= 2.5 + 1e-9);
                    assert!(a[v].rot_deg[i].abs() <= 2.0 + 1e-9);
                }
            } else {
                assert_eq!(a[v].trans_mm, [0.0, 0.0, 0.0], "unlisted vol {v} moved");
                assert_eq!(a[v].rot_deg, [0.0, 0.0, 0.0]);
            }
        }
    }

    #[test]
    fn load_tsv_reads_trans_mm_and_rot_rad_to_deg() {
        let dir = std::env::temp_dir();
        let path = dir.join("sp_motion_test.tsv");
        std::fs::write(&path,
            "framewise_displacement\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n\
             n/a\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n\
             0.7\t-0.5\t0.25\t0.1\t0.0174533\t0.0\t0.0\n").unwrap();
        let poses = load_motion_tsv(&path).unwrap();
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0], Pose::IDENTITY);
        assert!((poses[1].trans_mm[0] - (-0.5)).abs() < 1e-9);
        assert!((poses[1].rot_deg[0] - 1.0).abs() < 1e-3); // 0.0174533 rad ≈ 1°
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_motion_translation_shifts_the_head() {
        let (n, ng) = (9usize, 1usize);
        let dims = [n, n, n];
        let id = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mut img = vec![0.0f32; n * n * n]; // one compartment, ngrad=1
        let at = |x: usize, y: usize, z: usize| x + n * (y + n * z);
        img[at(4, 4, 4)] = 1.0; // bright voxel at centre
        let mut images = vec![img];
        // +2 mm in x on an identity grid → head moves +2 voxels in x
        let poses = vec![Pose { trans_mm: [2.0, 0.0, 0.0], rot_deg: [0.0, 0.0, 0.0] }];
        apply_motion(&mut images, dims, ng, id, &poses);
        assert!(images[0][at(6, 4, 4)] > 0.8, "head should shift to x=6: {}", images[0][at(6, 4, 4)]);
        assert!(images[0][at(4, 4, 4)] < 0.2, "original position should empty out");
    }

    #[test]
    fn multiband_dropout_darkens_dwi_shot_not_b0() {
        let (nx, ny, nz) = (4, 4, 4);
        let dims = [nx, ny, nz];
        let (ngrad, nvox) = (2usize, nx * ny * nz);
        let id = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mut images = vec![vec![1.0f32; nvox * ngrad]]; // one compartment, uniform
        let law = DropoutLaw::Scaled { drive: vec![0.0, 1000.0], floor: 50.0 }; // vol 0 = b0, vol 1 = DWI
        let ev = MotionEvent { volume: 1, shot: 0, severity: 1.0, jump_mm: [0.0; 3], jump_deg: [0.0; 3] };
        let gt = apply_multiband_motion(&mut images, dims, ngrad, id, 2, false, &law, &[ev]);
        let at = |x: usize, y: usize, z: usize, g: usize| (x + nx * (y + ny * z)) * ngrad + g;
        // mb=2, nz=4 → shot 0 = slices {0, 2}: both drop in the DWI volume
        assert!(images[0][at(1, 1, 0, 1)] < 0.01, "DWI shot slice should drop");
        assert!(images[0][at(1, 1, 2, 1)] < 0.01, "MB-partner slice drops together");
        assert!((images[0][at(1, 1, 1, 1)] - 1.0).abs() < 0.01, "non-event slice kept");
        assert!((images[0][at(1, 1, 0, 0)] - 1.0).abs() < 1e-6, "b0 exempt from dropout");
        assert_eq!(gt.len(), 1);
        assert_eq!(gt[0].slices, vec![0, 2]);
    }

    #[test]
    fn scaled_law_reproduces_the_b_value_attenuation_and_uniform_does_not() {
        let law = DropoutLaw::Scaled { drive: vec![5.0, 1000.0, 2000.0], floor: 50.0 };
        // b0 row (below the floor) is exempt: attenuation 1.0.
        assert!((law.attenuation(0, 0.5) - 1.0).abs() < 1e-12);
        // b = 2000 is b_max, so the full severity applies.
        assert!((law.attenuation(2, 0.5) - 0.5).abs() < 1e-12);
        // b = 1000 is half of b_max, so half the severity.
        assert!((law.attenuation(1, 0.5) - 0.75).abs() < 1e-12);
        // An index past the drive list reads as 0, i.e. exempt (the old `bvals.get(g)` behavior).
        assert!((law.attenuation(7, 0.5) - 1.0).abs() < 1e-12);

        let u = DropoutLaw::Uniform;
        for g in 0..3 {
            assert!((u.attenuation(g, 0.5) - 0.5).abs() < 1e-12,
                    "Uniform must not vary with the volume index");
        }
    }

    /// TRXScan main's `multiband_slab_matches_the_full_volume_where_no_jump_crosses_the_edge`: pure
    /// dropout on a 2-slice slab of a 6-slice volume at mb 2 sees the full volume's schedule, so the
    /// same slices attenuate by the same factor, reported in full-FOV indices; the event's shot is
    /// chosen to reach into the slab, so the check is not vacuous.
    #[test]
    fn multiband_slab_matches_the_full_volume_where_no_jump_crosses_the_edge() {
        let (nx, ny, nz, ngrad) = (4usize, 3usize, 6usize, 2usize);
        let v2w = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mk = || vec![(0..nx * ny * nz * ngrad).map(|i| 1.0 + (i % 13) as f32).collect::<Vec<f32>>()];
        let law = DropoutLaw::Scaled { drive: vec![0.0, 2000.0], floor: 50.0 };
        let shot = slice_schedule(nz, 2, true).iter().position(|s| s.contains(&4)).unwrap();
        let events = [MotionEvent { volume: 1, shot, severity: 0.8, jump_mm: [0.0; 3], jump_deg: [0.0; 3] }];
        let mut full = mk();
        let gt_full = apply_multiband_motion(&mut full, [nx, ny, nz], ngrad, v2w, 2, true, &law, &events);
        let orig = mk();
        let slab_range = nx * ny * 4 * ngrad..nx * ny * 6 * ngrad;
        let mut slab = vec![orig[0][slab_range.clone()].to_vec()];
        let gt_slab = apply_multiband_motion_slab(&mut slab, [nx, ny, 2], ngrad, v2w, 2, true, &law, &events,
                                                  Some(&[4, 5]), Some(nz));
        assert_eq!(slab[0], full[0][slab_range.clone()].to_vec());
        assert_ne!(slab[0], orig[0][slab_range].to_vec(), "the event's shot reaches into the slab");
        assert_eq!((gt_full.len(), gt_slab.len()), (1, 1));
        assert_eq!(gt_full[0].slices, gt_slab[0].slices, "dropped slices are reported in full-FOV indices");
        assert!(gt_slab[0].slices.contains(&4));
        assert_eq!(gt_full[0].attenuation, gt_slab[0].attenuation);
        // without a slab the slab variant is the plain one
        let (mut a, mut b) = (mk(), mk());
        let ga = apply_multiband_motion(&mut a, [nx, ny, nz], ngrad, v2w, 2, true, &law, &events);
        let gb = apply_multiband_motion_slab(&mut b, [nx, ny, nz], ngrad, v2w, 2, true, &law, &events, None, None);
        assert_eq!((a, ga.len()), (b, gb.len()));
    }

    /// `ScaledTo` divides by its given normaliser, even one below the largest drive; the floor,
    /// a volume beyond `drive` and `max <= 0` attenuate nothing.
    #[test]
    fn scaled_to_divides_by_the_given_maximum() {
        let law = DropoutLaw::ScaledTo { drive: vec![0.0, 1005.0, 30.0], max: 1000.0, floor: 50.0 };
        assert_eq!(law.attenuation(1, 0.5).to_bits(), (1.0 - 0.5 * (1005.0f64 / 1000.0) as f32).to_bits());
        assert_eq!((law.attenuation(0, 0.5), law.attenuation(2, 0.5), law.attenuation(7, 0.5)), (1.0, 1.0, 1.0));
        let off = DropoutLaw::ScaledTo { drive: vec![1000.0], max: 0.0, floor: 50.0 };
        assert_eq!(off.attenuation(0, 0.9), 1.0);
    }

    // ---- P9: shot schedules from slice times, per-shot transforms ----

    /// An oblique, anisotropic grid, so the matrix path is exercised away from the identity.
    fn oblique_v2w() -> [[f64; 4]; 4] {
        let r = crate::mat::rotation_zyx_deg(4.0, -3.0, 7.0);
        let s = [1.5, 1.75, 3.0];
        [
            [r[0][0] * s[0], r[0][1] * s[1], r[0][2] * s[2], -12.0],
            [r[1][0] * s[0], r[1][1] * s[1], r[1][2] * s[2], 8.5],
            [r[2][0] * s[0], r[2][1] * s[1], r[2][2] * s[2], -20.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// A smooth field (so trilinear rounding is the only difference between two equal transforms).
    fn smooth(dims: [usize; 3], phase: f64) -> Vec<f32> {
        let [nx, ny, nz] = dims;
        let mut v = vec![0.0f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (a, b, c) = (x as f64 / nx as f64, y as f64 / ny as f64, z as f64 / nz as f64);
                    v[x + nx * (y + ny * z)] =
                        (1.0 + (3.0 * a + phase).sin() * (2.0 * b).cos() + 0.5 * (4.0 * c).sin()) as f32;
                }
            }
        }
        v
    }

    fn peak(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    }

    #[test]
    fn schedule_from_times_reproduces_slice_schedule() {
        for nz in [4usize, 6, 8, 9, 12, 20] {
            for mb in [1usize, 2, 3, 4] {
                if nz % mb != 0 {
                    continue;
                }
                for interleaved in [false, true] {
                    let want = slice_schedule(nz, mb, interleaved);
                    // the offsets a scanner would record for that schedule: shot k at 0.04 k
                    let mut offsets = vec![0.0f64; nz];
                    for (k, shot) in want.iter().enumerate() {
                        for &z in shot {
                            offsets[z] = 0.04 * k as f64;
                        }
                    }
                    assert_eq!(slice_schedule_from_times(&offsets), want, "nz {nz} mb {mb} interleaved {interleaved}");
                }
            }
        }
    }

    #[test]
    fn schedule_from_times_takes_any_order() {
        // descending single band
        assert_eq!(slice_schedule_from_times(&[0.3, 0.2, 0.1, 0.0]), vec![vec![3], vec![2], vec![1], vec![0]]);
        // odd-first interleaved (slices 1, 3, 5, then 0, 2, 4)
        let odd_first = [0.3, 0.0, 0.4, 0.1, 0.5, 0.2];
        assert_eq!(slice_schedule_from_times(&odd_first), vec![vec![1], vec![3], vec![5], vec![0], vec![2], vec![4]]);
        // a grouping slice_schedule cannot represent: {0, 1} together
        assert_eq!(slice_schedule_from_times(&[0.0, 0.0, 0.1, 0.1]), vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn from_matrix_inverts_to_matrix() {
        let c = [3.0, -7.5, 12.0];
        for p in [
            Pose { rot_deg: [2.0, -3.5, 5.0], trans_mm: [0.4, -1.2, 2.5] },
            Pose { rot_deg: [-12.0, 20.0, -30.0], trans_mm: [-5.0, 0.0, 1.0] },
            Pose::IDENTITY,
        ] {
            let q = Pose::from_matrix(p.to_matrix(c), c);
            for i in 0..3 {
                assert!((q.rot_deg[i] - p.rot_deg[i]).abs() < 1e-12, "{p:?} -> {q:?}");
                assert!((q.trans_mm[i] - p.trans_mm[i]).abs() < 1e-12, "{p:?} -> {q:?}");
            }
        }
    }

    #[test]
    fn trajectory_interpolates_and_clamps() {
        let a = Pose { rot_deg: [0.0, 0.0, 2.0], trans_mm: [0.0, 1.0, 0.0] };
        let b = Pose { rot_deg: [0.0, 0.0, 4.0], trans_mm: [1.0, 1.0, -2.0] };
        let times = [1.0, 3.0];
        assert_eq!(sample_trajectory(&times, &[a, b], 0.0), a);
        assert_eq!(sample_trajectory(&times, &[a, b], 1.0), a);
        assert_eq!(sample_trajectory(&times, &[a, b], 9.0), b);
        let m = sample_trajectory(&times, &[a, b], 2.5);
        assert!((m.rot_deg[2] - 3.5).abs() < 1e-12 && (m.trans_mm[0] - 0.75).abs() < 1e-12 && (m.trans_mm[2] + 1.5).abs() < 1e-12);
        // coincident nodes: the later holds from that time
        let c = Pose { trans_mm: [9.0, 0.0, 0.0], ..Pose::IDENTITY };
        assert_eq!(sample_trajectory(&[0.0, 1.0, 1.0, 2.0], &[a, b, c, c], 1.0), c);
        assert_eq!(sample_trajectory(&[], &[], 1.0), Pose::IDENTITY);
    }

    fn trace_file(name: &str, text: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("mrsim_acq_trace_{name}_{}.tsv", std::process::id()));
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn timed_trace_reads_a_time_column_with_load_motion_tsvs_poses() {
        let path = trace_file("col",
            "time\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n\
             -0.5\t0\t0\t0\t0\t0\t0\n\
             0.25\t-0.5\t0.25\tn/a\t0.0174533\t0\t0.06\n\
             1\t1\t2\t3\t0.1\t-0.2\t0.3\n");
        assert!(has_time_column(&path).unwrap());
        let tr = load_motion_trace(&path, None).unwrap();
        assert_eq!(tr.times, vec![-0.5, 0.25, 1.0]);
        // the same cells through the untimed reader: the same poses, bit for bit
        assert_eq!(tr.poses, load_motion_tsv(&path).unwrap());
        assert_eq!(tr.poses[1].trans_mm, [-0.5, 0.25, 0.0]);
        assert_eq!(tr.max_interval(), 0.75);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn timed_trace_takes_a_sampling_rate_and_start() {
        let path = trace_file("rate", "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n0\t0\t0\t0\t0\t0\n1\t0\t0\t0\t0\t0\n2\t0\t0\t0\t0\t0\n");
        assert!(!has_time_column(&path).unwrap());
        let tr = load_motion_trace(&path, Some(Sampling { frequency_hz: 4.0, start_s: 0.0 })).unwrap();
        assert_eq!(tr.times, vec![0.0, 0.25, 0.5]);
        let tr = load_motion_trace(&path, Some(Sampling { frequency_hz: 4.0, start_s: -1.0 })).unwrap();
        assert_eq!(tr.times, vec![-1.0, -0.75, -0.5]);
        assert_eq!(tr.poses[2].trans_mm, [2.0, 0.0, 0.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn timed_trace_refusals_name_their_condition() {
        let h = "time\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n";
        let rate = Some(Sampling { frequency_hz: 10.0, start_s: 0.0 });
        let cases: [(&str, String, Option<Sampling>, &str); 9] = [
            ("both", format!("{h}0\t0\t0\t0\t0\t0\t0\n1\t0\t0\t0\t0\t0\t0\n"), rate, "two clocks"),
            ("neither", "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n0\t0\t0\t0\t0\t0\n0\t0\t0\t0\t0\t0\n".into(), None, "not a timed trace"),
            ("flat", format!("{h}0\t0\t0\t0\t0\t0\t0\n0\t0\t0\t0\t0\t0\t0\n"), None, "does not increase"),
            ("back", format!("{h}1\t0\t0\t0\t0\t0\t0\n0.5\t0\t0\t0\t0\t0\t0\n"), None, "does not increase"),
            ("nan_t", format!("{h}0\t0\t0\t0\t0\t0\t0\nNaN\t0\t0\t0\t0\t0\t0\n"), None, "time = \"NaN\""),
            ("inf", format!("{h}0\t0\t0\t0\t0\t0\t0\n1\tinf\t0\t0\t0\t0\t0\n"), None, "trans_x = \"inf\""),
            ("short", format!("{h}0\t0\t0\t0\t0\t0\t0\n1\t0\t0\n"), None, "column trans_z is missing"),
            ("one", format!("{h}0\t0\t0\t0\t0\t0\t0\n"), None, "at least two samples"),
            ("nocol", "time\ttrans_x\n0\t0\n1\t0\n".into(), None, "column 'trans_y' missing"),
        ];
        for (name, text, s, want) in cases {
            let path = trace_file(name, &text);
            let e = load_motion_trace(&path, s).unwrap_err();
            assert!(e.contains(want), "{name}: {e}");
            let _ = std::fs::remove_file(&path);
        }
        let path = trace_file("hz", "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n0\t0\t0\t0\t0\t0\n0\t0\t0\t0\t0\t0\n");
        for hz in [0.0, -1.0, f64::NAN] {
            let e = load_motion_trace(&path, Some(Sampling { frequency_hz: hz, start_s: 0.0 })).unwrap_err();
            assert!(e.contains("SamplingFrequency"), "{e}");
        }
        let e = load_motion_trace(&path, Some(Sampling { frequency_hz: 1.0, start_s: f64::INFINITY })).unwrap_err();
        assert!(e.contains("StartTime"), "{e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_trace_is_its_samples_at_their_times_and_refuses_what_it_does_not_cover() {
        let path = trace_file("cover",
            "time\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n0\t0\t0\t0\t0\t0\t0\n0.04\t0.3\t0\t0\t0\t0\t0.01\n0.08\t0.5\t0\t0\t0\t0\t0\n");
        let tr = load_motion_trace(&path, None).unwrap();
        for (t, p) in tr.times.iter().zip(&tr.poses) {
            assert_eq!(tr.pose_at(*t), *p);
        }
        assert!(tr.check_covers(&[0.0, 0.02, 0.08], "run").is_ok());
        assert!(tr.check_covers(&[], "run").is_ok());
        for bad in [[-0.01, 0.05], [0.0, 0.09]] {
            let e = tr.check_covers(&bad, "overlay motion.trajectory x.tsv").unwrap_err();
            assert!(e.contains("overlay motion.trajectory x.tsv") && e.contains("spans 0 s to 0.08 s") && e.contains("never extrapolated"), "{e}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn slice_resample_by_matrix_matches_resample_by_pose() {
        let dims = [11usize, 9, 7];
        let v2w = oblique_v2w();
        let src = smooth(dims, 0.3);
        let c = fov_center(dims, v2w);
        let pose = Pose { rot_deg: [1.5, -2.0, 3.0], trans_mm: [0.8, -0.6, 1.1] };
        let full = resample_by_pose(&src, dims, v2w, pose);
        let slices = [5usize, 0, 3];
        let got = resample_slices_by_matrix(&src, dims, v2w, pose.to_matrix(c), &slices);
        let slab = dims[0] * dims[1];
        let tol = 1e-6 * peak(&src);
        let mut moved = false;
        for (k, &z) in slices.iter().enumerate() {
            for xy in 0..slab {
                let (a, b) = (got[k * slab + xy], full[z * slab + xy]);
                assert!((a - b).abs() <= tol, "slice {z} voxel {xy}: {a} vs {b}");
                moved |= (b - src[z * slab + xy]).abs() > 1e-3;
            }
        }
        assert!(moved, "the pose must move something, or the check is vacuous");
        // the identity copies exactly
        let same = resample_slices_by_matrix(&src, dims, v2w, IDENTITY_MATRIX, &slices);
        for (k, &z) in slices.iter().enumerate() {
            assert_eq!(&same[k * slab..(k + 1) * slab], &src[z * slab..(z + 1) * slab]);
        }
    }

    /// Cumulative jump matrices per shot, as a caller with identity volume poses builds them.
    fn jump_matrices(events: &[MotionEvent], ngrad: usize, n_shots: usize, c: Vec3) -> Vec<Vec<[[f64; 4]; 4]>> {
        (0..ngrad).map(|g| {
            let mut cum = Pose::IDENTITY;
            (0..n_shots).map(|k| {
                for e in events.iter().filter(|e| e.volume == g && e.shot == k) {
                    for i in 0..3 {
                        cum.trans_mm[i] += e.jump_mm[i];
                        cum.rot_deg[i] += e.jump_deg[i];
                    }
                }
                if cum == Pose::IDENTITY { IDENTITY_MATRIX } else { cum.to_matrix(c) }
            }).collect()
        }).collect()
    }

    #[test]
    fn shot_transforms_match_multiband_motion() {
        let dims = [10usize, 8, 6];
        let (ngrad, nvox) = (3usize, 10 * 8 * 6);
        let v2w = oblique_v2w();
        let c = fov_center(dims, v2w);
        let mk = || -> Vec<Vec<f32>> {
            (0..2).map(|comp| {
                let f = smooth(dims, comp as f64);
                (0..nvox * ngrad).map(|i| f[i / ngrad] * (1.0 + 0.1 * (i % ngrad) as f32)).collect()
            }).collect()
        };
        let law = DropoutLaw::Uniform;
        let events = [
            MotionEvent { volume: 0, shot: 1, severity: 0.3, jump_mm: [0.5, -0.2, 0.7], jump_deg: [0.0, 1.0, -2.0] },
            MotionEvent { volume: 2, shot: 0, severity: 0.0, jump_mm: [0.0, 0.0, 1.2], jump_deg: [1.5, 0.0, 0.0] },
            MotionEvent { volume: 2, shot: 2, severity: 0.6, jump_mm: [-0.4, 0.0, 0.0], jump_deg: [0.0, 0.0, 3.0] },
        ];
        let schedule = slice_schedule(dims[2], 2, true);
        let mut old = mk();
        apply_multiband_motion(&mut old, dims, ngrad, v2w, 2, true, &law, &events);
        let transforms = jump_matrices(&events, ngrad, schedule.len(), c);
        let attenuation: Vec<Vec<Vec<f32>>> = (0..ngrad).map(|g| {
            (0..schedule.len()).map(|k| {
                events.iter().filter(|e| e.volume == g && e.shot == k).map(|e| law.attenuation(g, e.severity)).collect()
            }).collect()
        }).collect();
        let mut new = mk();
        apply_shot_transforms(&mut new, dims, ngrad, v2w, &schedule, &transforms, &attenuation);
        let orig = mk();
        for comp in 0..2 {
            let tol = 1e-6 * peak(&orig[comp]);
            for i in 0..nvox * ngrad {
                assert!((old[comp][i] - new[comp][i]).abs() <= tol, "comp {comp} index {i}: {} vs {}", old[comp][i], new[comp][i]);
            }
            // volume 1 has no event: untouched, bit for bit
            for v in 0..nvox {
                assert_eq!(new[comp][v * ngrad + 1].to_bits(), orig[comp][v * ngrad + 1].to_bits());
            }
        }
        // and something did move
        assert!((0..nvox).any(|v| (new[0][v * ngrad + 2] - orig[0][v * ngrad + 2]).abs() > 1e-3));
    }

    /// A rotating jump on a slab, with the full stack's matrix, reproduces the full volume's slices
    /// away from the slab's edges (the slab's own centre would rotate about a different point).
    #[test]
    fn shot_transforms_on_a_slab_use_the_full_stack_matrix() {
        let (nx, ny, nz, ngrad) = (12usize, 10usize, 12usize, 1usize);
        let v2w = oblique_v2w();
        let full_dims = [nx, ny, nz];
        let c = fov_center(full_dims, v2w);
        let img = smooth(full_dims, 0.7);
        let jump = Pose { rot_deg: [0.0, 0.0, 4.0], trans_mm: [0.6, 0.0, 0.0] }.to_matrix(c);
        // two shots: slices 0..6 unmoved, 6..12 jumped
        let schedule: Vec<Vec<usize>> = vec![(0..6).collect(), (6..12).collect()];
        let mut full = vec![img.clone()];
        apply_shot_transforms(&mut full, full_dims, ngrad, v2w, &schedule, &[vec![IDENTITY_MATRIX, jump]], &[vec![vec![], vec![]]]);
        // the slab z = 4..12 (8 slices), its own v2w shifted by 4 slices, the schedule mapped to local indices
        let (z0, nzs) = (4usize, 8usize);
        let mut sv2w = v2w;
        for r in 0..3 {
            sv2w[r][3] += v2w[r][2] * z0 as f64;
        }
        let slab_dims = [nx, ny, nzs];
        let local: Vec<Vec<usize>> = schedule.iter().map(|s| s.iter().filter(|&&z| z >= z0).map(|&z| z - z0).collect()).collect();
        let mut slab = vec![img[nx * ny * z0..].to_vec()];
        apply_shot_transforms(&mut slab, slab_dims, ngrad, sv2w, &local, &[vec![IDENTITY_MATRIX, jump]], &[vec![vec![], vec![]]]);
        let tol = 1e-6 * peak(&img);
        let mut compared = 0;
        // jumped slices two or more slices inside the slab (the backward map reads neighbouring slices
        // on this oblique grid, and the slab holds the same data there)
        for z in 8..10 {
            for xy in 0..nx * ny {
                let (a, b) = (slab[0][(z - z0) * nx * ny + xy], full[0][z * nx * ny + xy]);
                assert!((a - b).abs() <= tol, "slice {z} voxel {xy}: slab {a} vs full {b}");
                compared += 1;
            }
        }
        assert!(compared > 0);
        assert!((0..nx * ny).any(|xy| (full[0][8 * nx * ny + xy] - img[8 * nx * ny + xy]).abs() > 1e-3), "the jump moves slice 8");
    }

    #[test]
    fn shot_factors_apply_in_order_and_identity_shots_stay() {
        let dims = [3usize, 2, 2];
        let mut images = vec![vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]];
        let orig = images.clone();
        let schedule = vec![vec![0], vec![1]];
        let (a, b) = (0.7f32, 0.3f32);
        apply_shot_transforms(&mut images, dims, 1, IDENTITY_MATRIX, &schedule, &[vec![IDENTITY_MATRIX; 2]], &[vec![vec![], vec![a, b]]]);
        assert_eq!(&images[0][..6], &orig[0][..6]);
        for i in 6..12 {
            assert_eq!(images[0][i].to_bits(), (orig[0][i] * a * b).to_bits());
        }
    }
}
