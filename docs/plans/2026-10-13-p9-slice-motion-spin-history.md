# P9: per-slice poses and spin history — implementation plan

**Goal:** implement the P9 addendum (`docs/specs/2026-10-13-p9-slice-motion-spin-history-design.md`)
across `mrsim-acq`, `aslscan` and TRXScan, with every existing output byte-identical. Three parts:

- **Part A:** per-shot poses for every 2D readout: single band, any slice order, the trajectory
  sampled at each slice's time, events from a file.
- **Part B:** spin history for the tissue of spin-echo series whose rows share one preparation.
- **Part C:** the same two in TRXScan.

**Architecture.** Everything shared goes into `mrsim-acq` as additions: three motion functions
and a new pure-std `spin_history` module. `aslscan` and TRXScan each add inputs, a dispatch to the
new path, and ground truth. No existing `mrsim-acq` function is edited, so the default path of
both consumers calls the code it calls today.

The order is set by two risks. The spin-history cost is an estimate, so the engine and its
measurement (Task 6) come before anything is built on it. TRXScan is pinned to a `mrsim-acq`
tag, so its work (Milestone C) follows the `mrsim-acq` API being settled by `aslscan`'s use of
it, and lands by pull request.

**Spec.** Parts A and B were Codex-reviewed (2026-10-13, 13 findings, applied). Part C, "Where it
lives" and this plan were Codex-reviewed on 2026-10-14 (16 findings, all valid, applied to both
documents; the record is at the end of this file). A first attempt on 2026-10-13 died on
exhausted credits before writing any finding.

**State at writing.** P8 is committed and untagged on branch `p8-kz-undersampling` in both
repositories (`mrsim-acq` `5b4dd60`, `aslscan` `3faa1be`). TRXScan `origin/main` is `689ad79`
(release 0.3.0), pinned to `mrsim-acq` tag `trxscan-resync`. The local `TRXScan` checkout's
`main` is behind it (`02ca507`), and the `TRXScan-resync` worktree's branch is merged.

**Base (user's choice, 2026-10-17).** Work happens in the worktrees `rust-trx/p9-wt/{mrsim-acq,
aslscan}` (made with WSL git, so LF line endings; side by side because `aslscan` finds
`../mrsim-acq`), branch `p9-slice-motion` in both. `mrsim-acq` starts at P8 (`5b4dd60`). `aslscan`
starts at the OSIPI terminology commit (`5dfab06`, branch `osipi-terminology`, PennLINC/aslscan#5
open against `p8-kz-undersampling`), because P9 edits the same files the rename touched. The
GRAPPA reference-scan branch (`grappa-series-kernel`, unpushed) is not in the base; it changes the
k-space stage, not motion, and is merged later. The main `mrsim-acq` and `aslscan` checkouts hold
the user's uncommitted spiral edits and are not touched.

## Global constraints

- Everything in the P1-P8 plans' global constraints still holds: the pure-std default build, no
  `cargo fmt`, no large data committed, every approximation named in the sidecar, rejections
  over silent defaults.
- **No P9 input, no changed byte**, in three places:
  1. **`mrsim-acq`**: `tests/golden_baseline.rs` and every bit-pinned test.
  2. **`aslscan`**: `tools/regress_identity.sh 5dfab06 5b4dd60`, both feature sets. Never two
     runs at once (`work/regress.lock`). The path decision lives in `protocol`, which 3D readouts
     share, so the gate keeps its P5-P7 3D cases and Task 0 adds a P8 one.
  3. **TRXScan**: `tools/run_resync_baseline.sh` and `tools/resync_python_baseline.py`, each
     writing a checksum file that is then `diff`ed against `tests/fixtures/resync_baseline` (the
     tools generate; they do not compare). Before every Python run, `maturin develop --release`
     from the worktree being tested, and a check that the imported `trxscan._core` is the one
     just built: the CLI build does not refresh the extension, and a stale one passes anything.
- **Gate cadence.** The test suites and the golden record at the end of every task. The
  `aslscan` regress at the end of Tasks 0, 3, 5, 8 and 10. The TRXScan baselines at the end of
  Tasks 0, 11, 12, 13 and on the final heads.
- **Legacy dispatch.** The new path runs only for a request P3 could not serve (the spec's
  "Which path runs" table). One function decides it, in `protocol`, and its result is in the
  sidecar.
- **Environment.** Everything builds and runs in WSL. Python through `micromamba run -n rust-trx`
  (TRXScan bindings, pytest) and `-n simasl` (phantom tools). Plots need `-n trxscan`, the only
  environment with matplotlib.
- **Commits and pushes.** One commit per task, with no co-author trailer and no Claude line in
  commits or pull requests (the user, 2026-10-16: "don't cosign"). Commits made with WSL git pass
  `-c user.name="Taylor Salo" -c user.email=tsalo90@gmail.com` (WSL git's identity differs).
  Nothing is pushed, tagged on a remote, or opened as a pull request without the user's go-ahead.

## File structure

| Repository | Path | Change |
|---|---|---|
| mrsim-acq | `src/motion.rs` | `slice_schedule_from_times`, `resample_slices_by_matrix`, `apply_shot_transforms`, `sample_trajectory`, `Pose::from_matrix` |
| mrsim-acq | `src/spin_history.rs` | New, pure std: the engine |
| mrsim-acq | `src/lib.rs` | `pub mod spin_history;` |
| aslscan | `src/protocol.rs` | `within_volume_sampling`, `events`, `spin_history`; the path decision; the lifted and added refusals |
| aslscan | `src/series/shots.rs` | New: schedule, shot times, shot transforms and attenuations, shared by `series.rs` and `series/p6.rs` |
| aslscan | `src/series.rs`, `src/series/p6.rs` | Dispatch to the new path; spin-history correction (`series.rs` only) |
| aslscan | `src/bids.rs`, `src/bin/aslscan.rs` | `desc-motionShots_gt.tsv`, `desc-spinHistory_gt.nii.gz`, sidecar keys, run summary |
| aslscan | `tests/end_to_end.rs`, `tests/fixtures/protocols/p9_*`, `tools/acceptance_p9.sh` | Tests, fixtures, acceptance |
| aslscan | `README.md` | The new keys and what they do and do not model |
| TRXScan | `Cargo.toml`, `.cargo/config.toml` | The pin; a local patch during development |
| TRXScan | `src/motion.rs` | Facade: slice orders, events from a file, shot transforms with the b-value law, spin-history gain |
| TRXScan | `src/bin/trxscan.rs` | `--slice-order`, `--motion-events`, `--spin-history`, `--t1`, `--tr`; the `--mb 1` gate |
| TRXScan | `python/src/lib.rs`, `python/trxscan/{motion,protocol,simulate}.py`, `python/tests/test_motion.py` | Bindings and their tests |
| TRXScan | `docs/FEATURES.md`, `README.md`, `CLAUDE.md` | What is modeled now |

---

## Milestone 0: baseline

### Task 0: branches, reference, review

- [x] `mrsim-acq` and `aslscan`: branch `p9-slice-motion` in `p9-wt/` from the bases above, the
      phantoms linked into `p9-wt/aslscan/work` as `grappa-wt` does. Commit the spec and this plan
      on the `mrsim-acq` branch.
- [ ] `aslscan`: add a P8 case to `tools/regress_identity.sh`: `p8_spiral_kz` with zero-fill
      reconstruction on `work/phantom-3t-hcp`, feature `kspace` (the POCS variant is too slow for
      a gate). Then `tools/regress_identity.sh 5dfab06 5b4dd60` exits 0 (no difference), and
      `--self-test` exits 0 too, which there means every executed case detected the injected
      perturbation. Record the case counts here.
- [ ] TRXScan: create a worktree `p9-wt/TRXScan` from `origin/main` on a new branch
      `p9-shot-motion`, with WSL git (the local `main` checkout is left as it is). Remove the stale `TRXScan-resync`
      worktree only if the user agrees.
- [ ] TRXScan: `maturin develop --release` in `rust-trx`; both baseline tools' checksum files
      `diff` clean against `tests/fixtures/resync_baseline` on the untouched branch. If they do
      not, stop: the baseline is the gate for everything in Milestone C.
- [x] Codex adversarial review of spec part C, "Where it lives", and this plan (2026-10-14).

---

## Milestone A: per-shot poses

### Task 1: `mrsim-acq` motion additions

All in `src/motion.rs`, pure std, nothing existing edited.

- `slice_schedule_from_times(offsets)`: group by `to_bits`, order groups by value, slices
  ascending within a group.
- `resample_slices_by_matrix(src, dims, v2w, m, slices)`: the backward map and trilinear sample
  of `resample_by_pose`, restricted to the given slices, taking the world-space matrix directly.
  Refactor nothing: `resample_by_pose` keeps its own loop. The matrix folds the rotation centre
  into the translation, so the arithmetic order differs and the two agree to rounding, not bit
  for bit.
- `apply_shot_transforms(images, dims, ngrad, v2w, schedule, transforms, attenuation)` with
  `attenuation: &[Vec<Vec<f32>>]` (`[volume][shot]`, one factor per event, in event order): per
  volume and compartment, copy the pre-motion volume once, resample each shot's slices by its
  matrix, then multiply by each factor in turn, as `apply_multiband_motion_slab` does. A shot
  with the identity matrix and no factors is left untouched bit for bit. The matrices are world
  space, built by the caller about the whole stack's field-of-view centre, so a slab and the full
  volume get the same transform.
- `sample_trajectory(node_times, poses, t)`: linear in the six parameters, clamped at both ends.
- `Pose::from_matrix(m, center)`: the inverse of `to_matrix` for the `Rz Ry Rx` convention, for
  the shots TSV.

Tests, from the spec: schedule equality with `slice_schedule` over a grid of `nz`, `mb` and both
orders; descending and odd-first orders; `resample_slices_by_matrix` equals the same slices of
`resample_by_pose` to `1e-6` of the peak, for mixed-axis rotations; `apply_shot_transforms` with
identity volume poses equals `apply_multiband_motion` to `1e-6` of the peak on the same events
(matrices from the cumulative jump poses), and bit for bit on the shots without events; a
rotating jump applied to a slab with the full stack's matrix equals the full volume's slices two
slices or more from the slab's edges (the existing slab test uses no rotation, so this is new);
`from_matrix(to_matrix(p))` returns `p` to `1e-12` for mixed-axis poses; the trajectory clamps.

- [ ] Write the functions and tests. `cargo test` under default, `kspace`, and `kspace,par`.
- [ ] The golden record and the bit-pinned tests pass unchanged.
- [ ] Commit: `feat(motion): shot schedules from slice times and per-shot transforms`.

### Task 2: `aslscan` protocol

- `[motion] within_volume_sampling`: `"volume"` (default) or `"slice"`; `"slice"` with `random`
  is refused.
- `[motion.within_volume] events`: a path, exclusive with the four random-draw keys. Validate
  the TSV as `check_motion_tsv` validates a trajectory: the ten columns present, finite numbers,
  `volume < n`, `shot < n_shots`, `attenuation` in `[0, 1]`, no repeated `(volume, shot)`.
- `MotionSpec` gains `sampling`, `events_file`, and `shot_path: bool`, the path decision, for 2D
  readouts only: single-band `within_volume`, `"slice"` sampling, an events file, or a multiband
  schedule `multiband_schedule` rejects. A 3D readout never takes the shot path: P5's segmented
  events keep theirs (`protocol.rs:1149`), and `"slice"` sampling or an events file with a 3D
  readout is refused.
- On the shot path: the multiband factor may be 1; `multiband_schedule`'s rejection is not
  raised; each distinct slice offset must occur exactly `mb` times at every `mb`, 1 included; a
  Look-Locker series is refused.
- Off the shot path nothing changes, including P3's refusals.

Tests: each key parses; each refusal names its values; a P3 multiband protocol and a segmented
3D protocol with within-volume events both resolve to `shot_path = false`; a single-band
`SliceTiming` with a repeated time is refused only on the shot path.

- [ ] Write it with the tests. Commit: `feat(protocol): per-shot motion inputs`.

### Task 3: `aslscan` series

New `src/series/shots.rs`, used by `series.rs` and `series/p6.rs`. Its input is what both callers
have, not the protocol rows: `p6.rs` simulates the schedule's raw volumes (`sched.raw_rows`, with
the clock `first_prep(v).start_s`), which under Hadamard are not the protocol's rows.

- `shot_plan(motion, starts: &[f64], signal_t: &[f64], offsets, sim_grid, seed) -> ShotPlan
  { schedule, times[v][s], transforms[v][s], attenuation[v][s][..], events }`, `v` a raw volume.
  `series.rs` passes `row_start` and the rows' `t`; `p6.rs` passes `first_prep(v).start_s` and
  the raw rows' `t`.
  - `times`: `starts[v] + signal_t[v] + offset(s)`.
  - `P(v, s)`: the volume's pose, or `sample_trajectory` over the nodes `T(v, 0)` under
    `"slice"`.
  - events: drawn by the existing `draw_events` with the schedule's shot count, or read from the
    file (radians to degrees, `severity = 1 - attenuation`), validated against the raw volume
    count.
  - `Q(v, s) = J . P` as matrices from `Pose::to_matrix` about `fov_center(sim_grid)`.
- On the shot path the images are moved by `apply_shot_transforms` in place of `apply_motion`
  followed by `apply_multiband_motion`. The ground truths are still moved by the per-volume
  poses with `apply_motion`, as now.
- `SeriesOutput` gains the plan's shot table for `bids`.

Tests: single-band events drop one slice each; `"slice"` sampling gives the first shot of each
volume the volume's pose and the last volume one pose throughout; a rotating volume pose with a
rotating jump gives the matrix product, and differs from the summed-parameter pose; an
integer-voxel in-plane step at shot `s` shifts exactly the slices of shots `>= s`; with events
only and identity volume poses, the shot path equals the P3 path to `1e-6` of the peak on a
multiband protocol (forced onto the shot path by a test hook); the same cases through `p6.rs` on
the multi-echo fixture and on the Hadamard fixture, where an event at raw volume `v` lands on that
raw volume's slices and times.

- [ ] Write it with the tests.
- [ ] Regress gate: identical.
- [ ] Commit: `feat(series): per-shot poses for 2D readouts`.

### Task 4: `aslscan` output

- `desc-motionShots_gt.tsv`: `volume`, `shot`, `slices`, `time`, six parameters from
  `Pose::from_matrix`, rotations in radians. Written on the shot path only.
- Sidecar `AslscanSimulation.Motion`: `Path: "shot"`, `WithinVolumeSampling` and `EventsFile`,
  written on the shot path only; nothing new on P3's path, whose sidecars the regress compares
  byte for byte. The `GroundTruth` list names the new file there. `.bidsignore` already covers it.
- The run summary line names the sampling and the event source.

Tests: the TSV round-trips; a shot-path run's `desc-motionEvents_gt.tsv` fed back as `events`
reproduces its images to `1e-6` of the peak (a P3-path run's is not expected to: it was
resampled twice); the P3 sidecar is byte-identical when no new key is set.

- [ ] Write it with the tests. Commit: `feat(bids): per-shot motion ground truth`.

### Task 5: Part A acceptance

- [ ] Fixture `tests/fixtures/protocols/p9_single_band`: the motivating protocol (single-PLD
      PCASL, 20 pairs, 20 slices of 5 mm ascending, TR 4.6 s, on `work/phantom-3t-z100`), with
      overlays for random events, for `"slice"` sampling of a linear drift, and for an events
      file.
- [ ] `tools/acceptance_p9.sh`, exiting nonzero on any failure. Criterion 1: single-band events
      write one slice per event. The validator reports no errors.
- [ ] README: the new keys.
- [ ] Regress gate: identical. Commit: `test: P9 part A acceptance`.

---

## Milestone B: spin history

### Task 6: the engine, and its cost

`mrsim-acq/src/spin_history.rs`, pure std, `par` used when the feature is on.

```rust
pub struct Source<'a> {            // the head frame
    pub dims: [usize; 3], pub v2w: [[f64; 4]; 4],
    pub t1: &'a [f32],             // seconds; 0 = no signal
    pub weight: &'a [f32],         // M0, or 1
    pub class: &'a [u8],           // NO_CLASS = not tracked
    pub n_classes: usize,
}
pub struct Slabs { pub dims: [usize; 3], pub v2w: [[f64; 4]; 4] }   // the simulation grid
pub enum Pulse { Inversion { t: f64, epsilon: f64 }, Saturation { t: f64 } }
pub struct SliceTerms { pub slice: usize, pub num: Vec<Vec<f32>>, pub den: Vec<Vec<f32>> }  // [class][cell]

impl SpinHistory {
    pub fn new(source: Source, subcells: usize, slabs: Slabs, pulses: Vec<Pulse>) -> Self;
    /// The fictitious volume: excite only.
    pub fn prime(&mut self, shots: &[(f64, &[usize])], transform: [[f64; 4]; 4]);
    /// One shot: accumulate against `mz_ref(t1)`, then excite. `weight` replaces the
    /// source's weight for this shot only (TRXScan's per-volume diffusion signal).
    pub fn excite(&mut self, t: f64, transform: [[f64; 4]; 4], slices: &[usize],
                  mz_ref: &dyn Fn(f64) -> f64, weight: Option<&[f32]>) -> Vec<SliceTerms>;
}
```

State per sub-cell: `mz: f32` and `t: f64`, stored only for voxels with a class. `Mz(T)` replays
recovery and the pulses in `(t, T]`. Overlap is the sub-cell's z interval about its moved centre
against `[z - 1/2, z + 1/2)` in slab voxel coordinates. The in-plane cell is the nearest to the
moved centre; a centre outside the field of view is excited and contributes to no cell.

Tests, at this level, on small uniform grids (spec, "Part B, engine"): static head on an aligned
grid, correction below `1e-6` of `den`, with and without pulses and saturation; static head on a
grid whose slabs cut voxels, below `1e-3`; in-plane motion, the same; a through-plane jump of a
whole number of sub-cells toward the next slab gives `(den + num) / den` of 0.9031 at `f = 0.1`,
`dt = 40 ms` and 0.9268 at `dt = 0.4 s` (`TR = 4.6 s`, `T1 = 1.33 s`); the opposite jump leaves
the skipped tenth at `0.9990` in the next volume and at the reference when that volume is
saturated; the jump's dip is equal with and without saturation; a slow drift gives a ratio with
no step larger than its neighbours' by more than `1e-3`; `T1 = 0` gives zero; a per-call weight
scales that shot's `num` and `den` and leaves the next shot on the source weight; two source
voxels of different `T1` and weight binned to one cell give the weight-sum of their corrections.

**The measurement.** A test marked `#[ignore]` builds the motivating size (source 197 x 233 x 100
with a third of the voxels in a class, 8 sub-cells, slabs 128 x 136 x 20, 800 excitations) and
times it under `par`.

- [x] Write the module and the tests.
- [x] Run the measurement. Record time and peak memory here. **If it is over 60 s or over 2 GiB,
      stop and report before Task 7**, with the two levers the spec names measured (fewer
      sub-cells; tracking only tissue that enters the stack).
- [x] The golden record passes unchanged. Commit: `feat: spin_history — box-slab piecewise engine`.

**Done 2026-10-17, with one design change.** The first test run showed the equal sub-cells with
mean-field mixing bias a static head by 0.57% of a slice's signal where slabs cut the voxels (the
spec had estimated about 0.1%): a sub-cell excited in part by one slab and in part by the next
mixes the two states at the second read. The engine now keeps each voxel's z extent as exact
pieces (split at slab edges, merged when equal; at most `MAX_PIECES = 8`, the closest thin pair
merged beyond), so there is no sub-cell count: `SpinHistory::new(source, slabs, pulses)`, and
`spin_history_z_subcells` (Task 7) is dropped. The spec was updated to match. Every engine test
passes, the static ones to `1e-6` on aligned and cut grids, and the closed forms (0.9031, the
presaturated dip, the slab after the jump, the skipped tenth at `0.9990`) to `1e-5`.

Measurement (`measure_the_motivating_series`, release, `par`, 28 threads): 2.8 s for the 800
excitations (a slow drift with a 0.5 and 0.3 degree rotation), 80 ms to build, peak RSS 545 MB
for the whole test process. Merges occur only at the ends of the slice stack under the drift (one
sliver per volume carried out of the stack); the merge picks those slivers.

### Task 7: `aslscan` protocol

`[motion] spin_history`. The refusals of
the spec's "What it accepts", each naming its condition: a 3D readout; a contrast other than
spin echo; any row that is not `control` or `label`; rows whose preparation differs (reuse the
`prep_key` comparison of `series.rs:841`, moved into `protocol`); more than one echo;
Look-Locker; Hadamard; compat; no other motion request. `spin_history = true` sets `shot_path`.

- [ ] Write it with a test per refusal and one accepting multiband with suppression and
      presaturation. Commit: `feat(protocol): spin_history inputs and refusals`.

### Task 8: `aslscan` series

In `series.rs`, after the tissue images are moved on the shot path:

- Build the `Source` from the phantom: `t1`, `weight = m0`, `class` from the tissue masks.
  `Slabs` is the simulation grid. `Pulse`s are every row's suppression pulses and presaturation
  on the series clock.
- `prime` with the first row's shot times minus its repetition time, at `Q(0, 0)`.
- For each volume and shot in time order: `excite` with
  `mz_ref = |t1| tissue_mz(1, t1, tr, t_v + offset(s), suppression)`; multiply `num` by the
  shot's physiological tissue factor and attenuation; add it to that volume's tissue class
  images at the shot's slices.
- Accumulate `num` before those factors into a per-volume simulation-grid image, summed over
  classes, and `block_mean_inplane` it for the ground truth.

`te_factor` exists only under compat, which is refused, so the tissue weight is `m0` alone.

Tests: no motion beyond identity events leaves the images within `1e-6` of the same run without
`spin_history`; an in-plane jump likewise; a through-plane jump from an events file dips the
event shot's slab by the engine's closed form within `5e-3` in a uniform-tissue phantom and
leaves the other slices' correction at zero; multiband runs; the linearity identity still holds
on the complex images when the three runs share the plan.

- [ ] Write it with the tests.
- [ ] Regress gate: identical. Commit: `feat(series): spin-history correction of the tissue`.

### Task 9: `aslscan` output

- `desc-spinHistory_gt.nii.gz` and its sidecar (units, zero means no effect, the model name).
- `AslscanSimulation.Motion.SpinHistory` as the spec lists it.
- The run summary reports the largest correction relative to the tissue signal.

- [ ] Write it with a test that the file exists only with the key set and equals, with
      physiology off and attenuation 1, the block-averaged difference of the captured tissue
      compartment images (the existing `capture` hook, before the acquisition) between runs with
      and without the key. The final images are not the comparison: the acquisition transforms
      the correction. Commit: `feat(bids): spin-history ground truth`.

### Task 10: Part B acceptance, measurements

- [ ] Extend `p9_single_band` with overlays for criteria 2, 3 and 4 (z drift with `"slice"`
      sampling; one event with a 0.5 mm through-plane jump and attenuation 1; the same with
      presaturation) and `tools/acceptance_p9.sh` with their checks.
- [ ] A figure for the user from the criterion 3 run: sagittal and coronal cuts of the event
      volume beside the static one, and the correction map (`work/`, not committed).
- [ ] Record run times and the largest correction per criterion here.
- [ ] README: spin history, its conditions, and the effect-size table.
- [ ] Regress gate: identical. The validator reports no errors on every run.
- [ ] Commit: `test: P9 part B acceptance`.

---

## Milestone C: TRXScan

All in the `p9-wt/TRXScan` worktree, against `origin/main` (`689ad79`). During development
`.cargo/config.toml` patches `mrsim-acq` to `../mrsim-acq` (not committed); the committed pin
moves in Task 15. Every Python step is preceded by `maturin develop --release` from this worktree
and a check of the imported extension's path.

### Task 11: the dependency

- [ ] Patch to the local `mrsim-acq` at the Task 10 head. Change no TRXScan source.
- [ ] Both baseline tools' checksum files `diff` clean against `tests/fixtures/resync_baseline`.
      `cargo test` under default, `cli`, `cli,par`, `kspace`; `pytest python/tests -m "not phantom"`.
- [ ] No commit: nothing changed. Record the result here.

### Task 12: per-shot events

Library (`src/motion.rs`), the one producer of shot plans for the CLI and for Python:

- `SliceOrder { Interleaved, Sequential, Descending, OddFirst }` and
  `shot_offsets(nz, mb, order, tr_s) -> Vec<f64>`: shot `s` of the order at `s x tr_s / n_shots`,
  every slice of a shot sharing it. `Interleaved` and `Sequential` reproduce `slice_schedule`'s
  shots exactly.
- `ShotPlan::new(offsets, n_volumes, tr_s, volume_poses: Option<&[Pose]>, events, full_grid)`:
  the schedule (`slice_schedule_from_times`), shot times `v x tr_s + offset`, and per shot both
  the geometry matrix and the history matrix about the full stack's centre. Without volume poses
  both are `J`. With them (the `--motion` / `Artifacts.motion` path, whose images the
  re-simulation has already put at the volume pose) the geometry matrix is `J` and the history
  matrix is `J . P`.
- `load_motion_events(path, n_volumes, n_shots)`: part A's format and validation, `attenuation`
  read as `1 - severity`; `write_motion_events(events)`: the same format, raw severities, full
  precision.
- `apply_shot_motion(images, dims, ngrad, v2w, plan, slice_z, nz_full, bvals, b_max)`:
  attenuations from `diffusion_dropout_law` applied to the raw severities, then
  `mrsim_acq::motion::apply_shot_transforms` with the plan's geometry matrices and the schedule
  mapped to the slab's local slices, global shot indices kept and empty local shots skipped.
  Returns the dropped-shot truth as today.

Command line (`src/bin/trxscan.rs`):

- The gate becomes "events requested", not `mb > 1`.
- With no new flag and `mb > 1` the existing `apply_multiband_motion` call runs unchanged.
  `--slice-order` other than `interleaved`, `--motion-events`, or `--mb 1` take
  `apply_shot_motion`, with `--tr` for the times when given (the geometry needs only the order).
- `--motion-events` conflicts with `--dropout-rate` (clap `conflicts_with`).
- Whenever events run, `<out>_desc-motionEvents.tsv` is written beside the dropped-slice file.

Python:

- New dataclass `ShotEvents` (`python/trxscan/motion.py`): `slice_order` or
  `use_protocol_slice_timing`, `rate` or `events`, and `spin_history` (Task 13). It goes in
  `Artifacts.shot_events`, separate from `Artifacts.motion`. `simulate.py` keeps choosing the
  signal stage on `Artifacts.motion` alone, so `ShotEvents` never selects the re-simulation path
  or its refusals. `Artifacts.dropout` with `mb > 1` and no `ShotEvents` runs today's call.
- Bind `slice_schedule_from_times`, `load_motion_events`, `write_motion_events` and
  `Compartments.apply_shot_transforms`. `simulate.py` builds the plan once per run on the full
  stack and hands each slab its share; `Simulation` gains the events table and `Dataset` writes
  it under `derivatives/trxscan`.

Tests: `--mb 1 --dropout-rate` drops single slices and writes them; each slice order's schedule;
a run's `desc-motionEvents.tsv` fed back reproduces it to `1e-6` of the peak, with the b-value
law applied once; a duplicate event is refused; `ShotEvents` on a synthetic fibre object (which
the re-simulation path refuses) runs; a chunked Python run with a rotating jump equals the
unchunked one two slices or more from every slab edge.

- [ ] Write it with the tests.
- [ ] Baselines `diff` clean (the baseline's `--mb 2 --dropout-rate 0.5` case runs the old call).
- [ ] Commit: `feat(motion): single-band events, slice orders and events from a file`.

### Task 13: spin history

Library: `spin_history_correction(comp, sig_grid, t1_s: [f64; 3], tr_s, plan) ->
Vec<f32>`. The source voxels are the signal-grid cells, every one tracked. The engine holds one
class per voxel and the three compartments overlap in every cell, so it runs once per
compartment with that compartment's `T1` everywhere. Per volume, each `excite` replaces the
weight with the compartment's signal in that volume divided by `1 - exp(-TR/T1_c)`, so `num` is
the signal correction, weighted by the signal that is actually there; it is added to the
compartment's image at the shot's slices after the geometry. The history uses the plan's history
matrices. `prime` uses the first volume's shot times minus `TR`. The returned array is the
correction summed over compartments, per volume, block-averaged from the signal grid to the
acquired grid by the oversampling factor.

Command line: `--spin-history` with clap `requires = ["t1", "tr"]`; refused without events that
jump or a `--motion` trace. Writes `<out>_desc-spinHistory.nii.gz`.

Python: `Protocol.t1_s` (three values, default `None`); `ShotEvents(spin_history=True)` needs it
and `RepetitionTime`. Refused, naming the reason, with `Protocol.head`, with `chunk=`, `slices=`
or `z_mm=`, and with an object that is itself a slab (`z_offset` or `nz_full` set).
`Simulation.spin_history` holds the correction; `Dataset` writes it under `derivatives/trxscan`.

Tests: no jump gives a correction below `1e-6` of the signal; in a uniform object with sequential
order, at the first event volume and an interior slab, one through-plane jump of a quarter slice
gives that shot's slab a relative correction equal to part B's closed form within `5e-3`, and none
in the volume's other slices; two runs that differ only in their b-values give the same relative
correction; two compartments of different `T1` in one cell give the signal-weighted correction,
not the mean of the two; each refusal names what is missing.

- [ ] Write it with the tests. Baselines `diff` clean.
- [ ] Commit: `feat(motion): spin-history correction`.

### Task 14: TRXScan documentation

- [ ] `docs/FEATURES.md`: spin history moves from "out of scope" to what is modeled (box slab,
      through-plane motion, tissue, signal-weighted) and what is not (profiles, cross-talk
      without motion, head compartments).
- [ ] `README.md` and `CLAUDE.md`: the new flags, `ShotEvents`, `Protocol.t1_s`, the events
      truth file; the `--mb 1` behaviour change for the release notes.
- [ ] Commit: `docs: per-shot motion and spin history`.

---

## Milestone D: review and landing

### Task 15: reviews, tags, the pull request

- [ ] Codex adversarial review of the implementation in the three repositories. Verify each
      finding against the source; fix the valid ones with tests; record the outcome here.
- [ ] Ordinary Codex code review as the final pass; same treatment.
- [ ] Final gates on the heads: `mrsim-acq` suites and golden record; `aslscan` suites, regress,
      `acceptance_p9.sh`; TRXScan suites and both baselines (`maturin develop` first, checksums
      `diff`ed).
- [ ] **With the user's go-ahead:** tag `p9-complete` in `mrsim-acq` and `aslscan` and push. P8
      is untagged beneath it; ask whether it gets its own tag first.
- [ ] **With the user's go-ahead:** move TRXScan's pin to `p9-complete`, re-run both baselines
      on the git dependency, push `p9-shot-motion`, and open a pull request to TRXScan `main`
      for the maintainer's review.
- [ ] Update the memory notes.

## Acceptance criteria coverage

| Criterion | Task |
|---|---|
| 1 single-band events | 5 |
| 2 slow z drift, correction below `1e-3` | 10 |
| 3 one jump, 0.903 in the event shot's slab | 6 (closed form), 8, 10 |
| 4 presaturation: same dip, clean next volume | 6, 10 |
| 5 no new key, no changed byte; validator clean | every gate; 5, 10 |
| 6 TRXScan baselines; single-band dropout | 11, 12 |
| 7 TRXScan spin-history correction, b-value independent | 13 |

## Open points this plan does not settle

- **TRXScan's Python surface** (`ShotEvents`, `Protocol.t1_s`) is a proposal; its maintainer
  decides it in the pull request.
- **The `--mb 1 --dropout-rate` behaviour change** in TRXScan is deliberate but visible: a
  single-band run that passed the flag was silently unaffected and will now drop slices.
- **TRXScan's drawn events never touch b=0 volumes** (`dropout_events` skips `b < 50`), so they
  give b=0 volumes no jump and no spin history. An events file can target them.
- **Cost.** Settled by Task 6: 2.8 s on the motivating size.

## Implementation record (2026-10-17)

Commits (all unpushed): mrsim-acq `102764e` (docs), `eb7b0ea` (Task 1), `99ad5d5` + `3465ef8`
(Task 6); aslscan `48187b3` (Task 0), `2acd4ea` (2), `d3c05bd` (3), `af219e9` (4), `e188546` (5),
`59489f4` (7), `a359bad` (8), `a99e721` (9), `a66402f` (10); TRXScan `c8594c8` (12), `2844c3f`
(13), `dbca4a3` (14), `4cc5423` (a `--t1` parsing fix the smoke test found).

- **Task 0.** Regress on the base (`5dfab06` / `5b4dd60`, with the new P8 case): 35 executed under
  `cli,kspace,par` and 33 under `cli` (`asl001_p5`, `p8_spiral_kz` need `kspace`), all identical.
  TRXScan baselines on the untouched branch: 594 CLI and 177 Python hashes, identical.
- **Task 2.** As the spec's table says, a multiband schedule the P3 path cannot represent takes the
  shot path whatever the motion request (volume motion alone included).
- **Task 3.** The regress gate after Task 3 was folded into Task 10's run.
- **Task 5.** Criterion 1 passes (61 events, one slice each), every run validates.
- **Task 6.** The pieces design change and the measurement are recorded under Task 6.
- **Task 10.** Criteria 2-4 pass: the slow drift's largest correction is 1.3e-4 of the tissue
  (138 930 merges, all at the stack's ends); one 0.5 mm jump gives slice 10's grey matter 0.9050 of
  the static against the closed form 0.9031 (723 voxels); with presaturation 0.9051 against its
  own closed form, and the next volume's correction is 4e-16. Regress at `a66402f` / `3465ef8`:
  35 + 33 identical; the self-test detected the perturbation (its first attempt, at Task 0, had
  snapshotted a half-edited tree and failed to build).
- **Task 11-13.** TRXScan baselines identical after each, every cargo feature set and pytest
  green. The Python `Artifacts.dropout` keeps its behaviour (multiband only); single-band dropout
  is `ShotEvents(rate=...)` there, and `--dropout-rate` at `--mb 1` on the command line.

## Codex adversarial review of the implementation (2026-10-17)

7 findings (`p9-wt/codex-impl-adversarial.md`), each checked against the source; 6 valid and
fixed with tests, 1 resolved by correcting the spec.

1. TRXScan read the spin-history weights from re-simulated (scanner-frame) images as if they were
   head-frame, and tracked voxels by the scanner-frame signal: the weights are now the images moved
   back by each volume's inverse pose, and the tracked set comes from them
   (`spin_history_weights_undo_the_volume_pose`).
2. A full voxel could merge away an edge it had just made while splitting the second: room for both
   edges is now made before either split (`a_full_voxel_keeps_the_edges_its_excitation_needs`, which
   fills the voxel with distinct slivers).
3. Merging averaged states from different times: both pieces are now replayed to the later time,
   pulses included, before averaging (`merging_propagates_both_pieces_to_the_later_time`).
4. Python accepted duplicate or out-of-range explicit events that its own events file could not
   replay: the bindings now check them as the file reader does.
5. The Python sidecar's `SliceTiming` could contradict the shot order `ShotEvents` used: it now
   states the shots' timing when `ShotEvents` sets the order and `Protocol.tr_s` is known.
6. A stack whose slice count is not a multiple of the multiband factor got a three-slice shot:
   `shot_offsets` now refuses it.
7. The legacy multiband path writes no events file: kept, because writing one would change the
   baseline's outputs; the spec now says the file comes with the per-shot path only.

## Final Codex code review (2026-10-17)

3 findings (`p9-wt/codex-impl-final.md`), all valid, all fixed with tests.

1. The room-making loop counted the edges it needed once, before merging, and a merge could remove
   one of them: a full voxel's `cut` now protects the edges it needs from the merges and recounts
   after each (`a_cut_keeps_its_existing_edges`).
2. TRXScan built shots from unchecked timing (`--tr 0` put every slice in one shot; explicit slice
   times could repeat at single band or exceed the repetition): `check_shot_timing` refuses a
   non-positive TR, offsets outside `[0, TR)` and shots that are not `mb` slices, in the CLI and
   both bindings (which take `mb`); `slice_order="protocol"` needs `Protocol.tr_s`.
3. Both events readers accepted a file without its `slices` column: it is now required (its value
   is still ignored).

**Final gates (2026-10-17), on mrsim-acq `3ad025c`, aslscan `c06b65d`, TRXScan `368173a`:**
mrsim-acq 138 / 165 / 165 tests (default, `kspace`, `kspace,par`) and the golden record; aslscan
93 + 225 unit and 59 integration tests, acceptance criteria 1-4 passing; regress 35 + 33 cases
identical (the self-test on `a66402f` detected the perturbation in every case); TRXScan 75 / 80 /
80 / 75 tests (default, `cli`, `cli,par`, `kspace`), 65 pytest, re-sync baselines identical (594
CLI and 177 Python hashes). Nothing is tagged or pushed: that, and the TRXScan pull request, wait
for the user.

## Codex adversarial review of Part C and this plan (2026-10-14)

16 findings (`aslscan/work/p9-codex-plan.md`), each checked against the source (TRXScan at
`origin/main` `689ad79`); all valid, all applied to the spec and to this plan.

1. TRXScan had no source for shot times or full transforms, and `J . P` would have re-applied the
   volume pose to re-simulated images: one shot-plan producer, explicit timing precedence,
   geometry `J` and history `J . P` on the re-simulation path (Task 12).
2. Any `Artifacts.motion` selects re-simulation in `simulate.py`: shot-only configuration moved to
   a separate `ShotEvents` (spec C2, Task 12).
3. A unit-weight gain averages saturations by geometry, not by signal, and diffusion changes the
   signal per volume: TRXScan is additive like `aslscan`, with a per-call weight in the engine
   ("Where it lives", Tasks 6 and 13).
4. One event does not give two volumes the same history: b-value independence is now tested with
   two runs differing only in b-values; order, volume and slab are specified (spec, Task 13).
5. TRXScan's dropped-slice file holds no jumps and a b-scaled attenuation: a new events truth file
   with raw severities (spec C2, Task 12).
6. `chunk` is not the only incomplete-stack run: `slices=`, `z_mm=` and slab objects are refused too.
7. `p6.rs` simulates raw volumes on its own clock: `shot_plan` takes starts and signal times from
   the caller, and Hadamard is tested (Task 3).
8. "Single-band within_volume" captured segmented 3D requests: the shot path is 2D only, the new
   keys are refused with 3D, and the regress keeps 3D cases plus a new P8 one (Tasks 0 and 2).
9. Matrix resampling and folded attenuation cannot be bit-identical to the pose path: tolerances,
   per-event attenuation in order, duplicates refused (spec, Task 1).
10. Replaying a P3-path run goes through a different resampling: replay is asserted for shot-path
    runs only (Task 4).
11. New sidecar keys on every run would break byte identity: written on the shot path only (Task 4).
12. The correction map is pre-acquisition: Task 9 compares it with captured compartment images,
    not final images.
13. The gain ground truth had no weighting, Python storage or writer, and head compartments had
    no `T1`: additive map, `Simulation.spin_history`, `Dataset` output, `Protocol.head` refused.
14. The existing slab test has no rotation and the centre came from the slab: full-stack matrices
    and a rotating-jump slab test (Tasks 1 and 12).
15. Python baselines could pass on a stale extension and the tools do not compare: rebuild, check
    the import, `diff` the checksum files (global constraints).
16. The regress self-test exits 0 on success: Task 0's wording fixed.
