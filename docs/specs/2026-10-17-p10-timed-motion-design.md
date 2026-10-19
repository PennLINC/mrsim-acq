# P10: motion traces sampled faster than the repetition

Design spec addendum, 2026-10-17. Extends `2026-10-13-p9-slice-motion-spin-history-design.md`
(P9). Where this document is silent, P9 stands. The user fixed one thing (2026-10-17): the input is
a motion file sampled faster than the repetition time. The user also asked whether `aslscan` can
simulate realistic motion artifacts for 3D readouts, and part C answers that with the same input.
Every other choice here was made by Claude while writing and has not been reviewed by the user. A
Codex adversarial review the same day returned 11 findings (3 high, 7 medium, 1 low). All were
checked against the source, all held, and this text is the result. "What the review changed"
lists them.

## Context

P9 gives every 2D shot its own pose, and spin history makes a shot excited after a through-plane
movement dark or bright. The PR figures for P9 showed that the inputs cannot ask for the motion
that produces the artifact seen in real data:

- **The trajectory has one pose per volume.** With `within_volume_sampling = "slice"`, P9 places
  each volume's pose at its first shot and interpolates linearly between volumes
  (`aslscan/src/series/shots.rs:80,95`). A move between two poses is spread over the whole
  repetition. In 2D PCASL the readout is the last 0.36 s of a 4.6 s repetition, so a nod given as
  the next volume's pose puts 8% of itself inside the readout. The remaining 92% falls in the
  labeling and the delay, and the next volume comes out rigidly displaced. This was measured, not
  estimated (`p9-wt/asl-fig/nod_sh-true`, `desc-motionShots_gt.tsv`: 0.23 mm of 3 mm by the last
  shot).
- **Events are jumps.** A continuous movement can be written as one event per shot, and the P9
  figures were made that way (`p9-wt/asl-fig/nod_events.tsv`). Nobody can be expected to turn a
  measured trace into per-shot jumps by hand, and jumps reset at every volume boundary.
- **A real trace looks like this.** In `ds006131` (`sub-20188`, `ses-1`, task `rat`, run 1,
  multiband 6, 72 slices in 12 shots, interleaved, TR 1.761 s), volumes 37 and 38 show adjacent
  slices alternating in intensity down the whole stack (0.52, 0.89, 0.54, 0.81 of the run's
  median in volume 37). That is a head moving several millimetres between the shots of one volume.

A trace sampled faster than the repetition puts the movement where it happened. P9's machinery
already turns a pose per shot into images and spin history, so most of P10 is input.

## Scope

In:

- **Part A**, a timed trace: the file format, its clock, sampling it at shot times, and its
  composition with events. Shared by both simulators.
- **Part B**, `aslscan` 2D: the trace drives P9's per-shot path.
- **Part C**, `aslscan` 3D: each readout segment takes the pose at its own time.
- **Part D**, TRXScan: the trace drives the volume poses of its re-simulation and the per-shot
  remainder.
- **Part E**, a small generator for synthetic traces (nods, steps, drifts).
- Their tests, ground truth and sidecar fields.

Out, with reasons under "Decisions deferred":

- motion during one shot's readout;
- motion during labeling and background suppression;
- spin history for gradient echo (the slow recovery seen in volumes 39-40 of the reference run);
- BIDS-Motion `_motion.tsv` files with a `_channels.tsv`;
- interpolation other than linear;
- TRXScan re-simulation per shot.

Nothing changes unless a timed trace is given. An untimed trace keeps its meaning.

---

# Part A: the timed trace

## The file

The trajectory TSV both simulators read today (`trans_x/y/z` in mm, `rot_x/y/z` in radians, one
row per volume; `mrsim_acq::motion::load_motion_tsv`) becomes timed in one of two ways:

| Form | Sample times |
|---|---|
| a `time` column | that column, in seconds |
| a JSON sidecar at the same path with `.json` for `.tsv`, holding `SamplingFrequency` (Hz) and optionally `StartTime` (s, default 0) | `StartTime + i / SamplingFrequency` for row `i` |

The sidecar form is the BIDS convention for physiological recordings, so a trace exported next to
a physio file needs no rewriting. Both forms in one file are refused: two clocks for one trace is
an error, not a choice. A sidecar without `SamplingFrequency` leaves the trace untimed. That keeps
fMRIPrep's `desc-confounds_timeseries.json`, which describes the columns and has no sampling rate,
meaning what it means today.

**Detection comes first.** A trace is timed when its header has a `time` column or its sidecar
has `SamplingFrequency`, and untimed otherwise. Only a timed trace goes through the new loader and
its checks.

An untimed trace is read exactly as each interface reads it today, permissive cells and row-count
rules included:
- `aslscan`: `check_motion_tsv` (`protocol.rs:1096`), then `load_motion_tsv`. The row count must
  equal the number of motion units (volumes, or raw volumes under Hadamard, `protocol.rs:2731`).
- TRXScan's command line: `load_motion_tsv`, which maps an unparseable cell to 0
  (`mrsim-acq/src/motion.rs:218`). Volumes beyond the trace get the identity
  (`TRXScan/src/compartments.rs:883`).
- TRXScan's Python layer: `Motion.from_confounds`. `poses_for` truncates a longer trace and
  refuses a shorter one (`motion.py:186-192`).

A timed trace is validated strictly:

- every cell present and finite (`n/a` is 0, as `aslscan` documents);
- every time finite and strictly increasing, and `SamplingFrequency` finite and positive;
- at least two samples;
- the trace covers every time the run queries it at: the reference times under one pose per
  volume, every shot or segment time under one pose per shot (see "Two samplings"). A run that
  outlasts its trace is refused, never extrapolated, because holding the last pose would invent
  stillness.

A timed trace's row count has nothing to do with the number of volumes.

## The clock

Times are seconds from the run's origin:

- **`aslscan`**: the clock of the acquisitions actually made, with its origin at the start of the
  first one. There is more than one such clock in the code:
  - the protocol rows' clock `row_start` (`protocol.rs:2917-2925`), which covers segmented 3D rows
    (`NumberShots` repetitions each);
  - the raw-acquisition clock of the P6 `Schedule`. Under Hadamard it advances once per encoding
    row and segment and emits `H` raw volumes per cycle, not the `H - 1` decoded rows
    (`schedule.rs:157-202`). The multi-echo and Look-Locker series use it too.

  P10 reads times from whichever clock the series path already uses for its acquisitions: P9's
  shot times (`series/shots.rs:72`, fed `first_prep(v).start_s` by `p6.rs:1190`), and P4's
  segment times for 3D (`series.rs:980-982`). It never builds a clock of its own. Every query time
  (reference, shot and segment) comes from that one list, and coverage, the reference poses, the
  shot poses and the truth all use the same list. For a single-delay series the origin is the
  first labeling's start.
- **TRXScan**: volume `v` starts at `v x TR`, as P9 part C's shot times already do
  (`TRXScan/src/motion.rs:331`).

**Priming.** P9 primes spin history with a fictitious previous volume at the first shot's pose
(`aslscan/src/series/shots.rs:152`, `TRXScan/src/motion.rs:148`). Trace samples before the first
real shot do not change that. A trace says where the head was; it does not say which slices were
excited before the run.

The written `desc-motionShots_gt.tsv` (aslscan) and the new TRXScan equivalent (part D) carry
each shot's time on this clock, so a user aligning a measured trace to the run can check the
alignment against the output.

## Sampling

A timed trace is a set of nodes `(time_i, pose_i)`. The pose at time `t` is
`mrsim_acq::motion::sample_trajectory(times, poses, t)`, the function P9 already uses: linear in
the six parameters between the neighbouring samples. P10 changes where the nodes come from, not
how they are read:

| Trace | Nodes |
|---|---|
| untimed (P9) | volume `v`'s row at the time of its first shot |
| timed (P10) | the file's samples at their own times |

For a timed trace, every query time is inside the trace (refused otherwise), so the clamping at the
ends of `sample_trajectory` is never reached. For an untimed trace under P9's `"slice"` sampling
the clamp is reached and stays: the last volume's later shots hold its pose, as in P9.

**Poses at the identity stay the identity** (found in Task 5). The series clock adds repetitions
in floating point, so a shot lands at, say, 58.80000000000001 s just past a still sample at 58.8 s.
Interpolated there, the pose is about 1e-13 mm instead of zero. That matters because the resampler
gives zero to any voxel whose sample point lies outside the grid (`mrsim_acq::motion::trilinear`).
A 1e-13 mm shift therefore blanked the bottom slice of that volume. A timed
trace's pose within `TRACE_POSE_EPSILON` (1e-9 mm and 1e-9 degrees in every parameter) of the
identity is the identity (`Trace::pose_at`). The edge behaviour itself is P3's and P9's, and
changing it would change their outputs. It is noted under "Decisions deferred".

Linear interpolation of Euler angles is not the geodesic between two rotations. Between two
samples of a trace that is sampled faster than the shots, the difference is second order in a
rotation that is already small, so it is ignored here as it was in P9.

**Sampling rate.** A trace sampled more slowly than the shots are spaced is interpolated linearly
between its samples, which is P9's behaviour between volumes. That is allowed and is not an
error. The largest sample interval and the smallest shot spacing are written to the sidecar, and
the run summary says so when the first is larger.

## Two samplings

Each simulator keeps a choice between one pose per volume and one per shot. For a timed trace:

- **one pose per volume** (`aslscan`'s `within_volume_sampling = "volume"`, TRXScan's
  `--motion-sampling volume`, the default for both): each volume has the trace's pose at its
  **reference time**, the time of its first shot. This is the before-P10 view of the same motion,
  and the comparison the P9 figures needed.
- **one pose per shot** (`"shot"`, see part B for the name): each shot has the trace's pose at its
  own time.

The reference time is the first shot so that, for an untimed trace, both samplings agree with P9
at the first shot of every volume. A mid-volume reference would halve the largest per-shot
remainder in TRXScan (part D). It was not chosen, because it would give an untimed trace two
meanings.

## Composition with events

A shot's transform is P9's `Q(v, s) = J(v, s) . P(v, s)`. `P` is now the trace's pose at the
shot's time and `J` is still the summed jumps of the volume's events. Events from a file or from
the random draw combine with a timed trace exactly as they combine with an untimed one. A trace
and events give the same transform when the trace's pose at each shot time equals the cumulative
jump. That is the link between this spec and the hand-written events behind the P9 figures, and
acceptance criterion 2 checks it.

## Where it lives

`mrsim-acq` gains one pure-std function and one type, both additions:

```rust
pub struct Trace { pub times: Option<Vec<f64>>, pub poses: Vec<Pose> }
pub struct Sampling { pub frequency_hz: f64, pub start_s: f64 }
pub fn load_motion_trace(path: &Path, sampling: Option<Sampling>) -> Result<Trace, String>
```

`load_motion_trace` is for timed traces only: callers detect timing first and keep
`load_motion_tsv` for untimed ones. It reads the six columns strictly, plus the `time` column when
there is one. A `sampling` argument gives the times of a file without one. It refuses a trace that
ends up with no times. It validates what is local to the file:
- finiteness;
- strictly increasing times;
- at least two samples;
- not both a `time` column and a `sampling`.

Coverage depends on the run, so the caller checks it. `load_motion_tsv` is not edited.

**The shared `MotionMode` enum is not changed.** `MotionMode::Trajectory { poses }`
(`mrsim-acq/src/motion.rs:38`) is built by TRXScan's Python binding (`python/src/lib.rs:1033`)
and matched in `aslscan`. Adding a field would break both mid-plan. Each consumer keeps a timed
trace in its own structure instead:
- `aslscan` in `MotionSpec`, as `trace: Option<Trace>`. `mode` holds `MotionMode::Off` as a
  placeholder. Every reader of `MotionSpec::mode` goes through one function,
  `volume_poses(spec, reference_times, seed)`, which consults `trace` first. This matters because
  `resolve_poses` on a sample list would take the samples as volumes. Part B lists the readers.
- TRXScan beside its volume poses.

`mrsim-acq` has no JSON parser and does not gain one. The caller reads the sidecar and passes
`sampling`:
- `aslscan` with its `serde_json`;
- TRXScan with `serde_json`, added to its `cli` feature (the command line has no JSON parser
  today), and with `json` in its Python layer.

A substring search was considered for TRXScan's command line and rejected. It can miss a valid
key: `"Sampling\u0046requency"` (a JSON escape for the F) decodes to `SamplingFrequency`. A miss would read a 100 Hz trace
as one pose per volume.

---

# Part B: `aslscan`, 2D

**Keys.** `[motion] mode = "trajectory"` and `trajectory = "<file>"` as today. A timed file is
detected from its contents (part A). `within_volume_sampling` gains the value `"shot"`, meaning a
pose per shot. In 2D it is the same as `"slice"`, which stays as a synonym so P9 overlays keep
working. The sidecar records the value as given.

**Path.** A timed trace with `"volume"` sampling stays on P3's per-volume path when nothing else
asks for the shot path: its poses are the trace at the volumes' reference times, and they are
applied as P3 applies any pose. Timed with `"shot"` or `"slice"` takes P9's shot path. `shot_plan`
(`series/shots.rs:65`) gets the nodes from the caller in place of building them from the first
shots.

**Where the times come from, and when they are checked.** Each series function already builds its
acquisition times: `series.rs` from `row_start`, `p6.rs` from its `Schedule`'s raw volumes
(`first_prep(v).start_s`, `p6.rs:1190`), and `ge3d.rs` from its own schedule. Each one builds the
list of query times first, at its start and before the signal stage. That way a coverage refusal
costs nothing and names the needed span. One helper (`shots::query_times`) is fed from each
clock. The protocol cannot do the check: under Hadamard the raw clock exists only after
`hadamard_spec` resolves (`protocol.rs:3102`), and `row_start` does not describe it.

**Hadamard and multi-echo.** Motion indexes raw volumes, `H` per Hadamard cycle
(`protocol.rs:2731`, used at `p6.rs:1187`). An untimed trace keeps that indexing, one row per raw
volume. A timed trace is sampled at the raw volumes' own times. Included `m0scan` preparations are
acquisitions on the same clock and are queried like any other.

**The readers of `mode`.** `MotionSpec::mode` is read by `resolve_poses` in `series.rs:1205`,
`p6.rs:1187` and `ge3d.rs:628`. Task 3 confirms there are no other readers by searching the
source. For a timed trace each of these reads goes through `volume_poses`, which returns the
reference poses.

**Spin history** uses the plan's transforms and needs no change. A timed trace is a motion trace,
so it satisfies the "needs motion to have a history of" refusal.

**Refusals added.**
- a timed trace that does not cover the run, naming the first and last time needed and the
  trace's span;
- a timed trace with `mode` other than `"trajectory"` (the key is read only there today);
- `"shot"` with `mode = "random"`, as `"slice"` is.

**Ground truth.**
- `desc-motion_gt.tsv` holds one row per volume, as now: the pose at each volume's reference
  time.
- `desc-motionShots_gt.tsv` (shot path) is unchanged in format.
- The sidecar's `Motion` block gains `Trajectory: { "Timed": true, "Samples", "Span": [first,
  last], "MaxSampleInterval", "MinShotSpacing", "Clock": "acquisition clock, s; 0 = the first
  acquisition's start" }`, only for a timed trace. An untimed run's sidecar stays byte-identical.
- The moved `delta_m` truth moves by the volume poses, as P9 decided.

---

# Part C: `aslscan`, 3D

**What 3D motion does today.** A segmented 3D readout (GRASE or stack of spirals, P5; the
gradient-echo EPI train, P7) acquires each volume over `NumberShots` repetitions, each with its
own labeling (`protocol.rs:2919-2925`). P5 part D draws random events per segment. The segments
after an event read their k-space lines from the volume's images moved by the cumulative jump
(three builders: `series.rs:1284-1328`; `series/p6.rs:1236-1290`, which also moves its
tissue-only images; `series/ge3d.rs:621-651`), and each event's segment is attenuated. Segments
at different poses give inconsistent k-space, and the reconstruction turns that into ghosting and
blurring along the segmented dimensions, not slice-wise banding. That is the real 3D artifact. What
is missing is a way to ask for it with a measured or designed movement: only random jumps reach
the segments.

**With a timed trace and `"shot"` sampling,** segment `s` of volume `v` has the trace's pose at
its time. The time of segment `s` is the time of its excitation on its acquisition clock:
`row_start[v] + s x TR_row + t_row`, the clock P4's physiology already uses for 3D segments
(`series.rs:980-982`). In the P6 series (multi-echo, Hadamard) it is the segment's preparation,
`sched.preps_of(v)[s].start_s + t` (`p6.rs:783`). For the gradient-echo EPI train it is the time
of the segment's first excitation on that path's schedule. Task 6 confirms each one from the
source before using it.

The volume's images are moved by its reference pose `P_v` as today (`apply_motion` on the
spin-echo 3D path, `Ge3dBuild::poses` on the train). Each segment then reads them moved further by
`J(v, s) . P(v, s) . P_v^-1`, its pose relative to the volume's. Segments that share a relative pose
share an image set, as now. With continuous motion every segment has its own set. In `p6.rs` the
tissue-only images get the same sets (`tissue_shot_sets`), so the tissue-only products stay
consistent with the signal.

**Kernels.** The three existing builders and their `resample_by_pose` calls (`series.rs:1321`,
`p6.rs:1273-1286`, `ge3d.rs:421`) stay as they are for every run that is not on the new path:
random events with `"volume"` sampling, untimed or timed. Matrix resampling agrees with
`resample_by_pose` only to rounding (`mrsim-acq/src/motion.rs:350`), so swapping kernels would
change bytes.

The new path (`"shot"` sampling) is a separate branch. It compares each segment's absolute pose
`J . P(t)` with the volume's `P_v` as poses, not as a matrix product. Where they are equal, it
reuses the base images, because `P_v . P_v^-1` computed in floating point need not be the exact
identity that skips interpolation (`motion.rs:361`). Otherwise it resamples by the relative matrix
with `resample_slices_by_matrix` over every slice.

**Cost.** One set of compartment images per distinct segment pose per volume, `NumberShots` sets
for a moving volume: 4-16 for the readouts in the test protocols. How long the sets live differs
by builder:
- `ge3d` counts them per volume in flight (`ge3d.rs:717`).
- `series.rs` keeps every volume's sets until the acquisition (`series.rs:1289`).
- `p6.rs` keeps every volume's sets and a tissue-only copy of each (`p6.rs:1241`).

So the estimate cannot be borrowed from `ge3d`. Each builder gets its own:
- `series.rs`: the sum over volumes of the distinct segment poses, times the compartment images
  of one volume;
- `p6.rs`: twice that;
- `ge3d`: its existing formula, with the pose count taken from the plan.

The estimate is computed from the query times and poses before any set is allocated.

**The limit.** `[images] max_memory_gib` is accepted today only for `epi3d` and compat's multi-TE
(`protocol.rs:3217-3226`). P10 also accepts it on the 3D `"shot"` path, with the same 4 GiB
default. The refusal applies only to the new path: a legacy 3D run with random events allocates
as today and is never refused by it.

**Ground truth.** `desc-motionShots_gt.tsv` is written by `bids.rs:1231` only from a 2D
`ShotPlan`, and every 3D builder returns `shots: None` (`ge3d.rs:1067`). P10 replaces the writer's
input with a readout-independent list, one row per (volume, shot):
- `volume` (the raw volume index under Hadamard);
- `shot` (the segment, in 3D);
- `slices` (`n/a` in 3D);
- `time`;
- the absolute pose `J . P(t)`, never the relative resampling transform.

The 2D path builds the list from its plan, so its file is byte-identical. The 3D path writes it
under `"shot"` sampling.

**Approximations, named in the sidecar.**
- One pose per segment: a segment's echo train (tens to hundreds of ms) is read at one pose.
- The segment images are resampled from images already moved by `P_v`, so they are interpolated
  twice, as P5's event segments already are.
- `within_volume_sampling = "slice"` stays refused in 3D because a 3D readout has no slices;
  `"shot"` is the 3D spelling.

**Unchanged.** Spin history stays 2D-only: a slab excitation has no slice history to track.
Events files stay 2D-only, and 3D keeps the random draw. A 3D run without `"shot"` sampling,
untimed or timed, uses today's builders and kernels. Its bytes match today's whenever its poses
do; an untimed trace's do, always.

---

# Part D: TRXScan

TRXScan has two motion layers since P9 (`TRXScan/src/bin/trxscan.rs:528-620`):

- `--motion <trace>` re-simulates every volume from streamlines moved by that volume's pose, so the
  angle between fibre and gradient changes. This is the "faithful" path, and it ignores SIFT2
  weights and κ.
- The P9 shot plan moves each shot's slices by its jumps, with `geometry = J` and
  `history = J . P_v` (`motion.rs:266-338`).

**A timed `--motion` trace** (either form; the sidecar is parsed with `serde_json`, added to the
`cli` feature).
- Timing is detected first (part A). An untimed trace takes today's code path exactly:
  `load_motion_tsv`, and identity for volumes beyond it.
- The per-volume poses handed to the re-simulation are the trace at each volume's reference time
  (`v x TR`, plus the first shot's offset, which is 0 for TRXScan's own orders).
- A timed trace needs `--tr`, and is refused without it, because the volumes have no times
  otherwise.
- New flag `--motion-sampling volume|shot`, default `volume`.
- With `shot`, the shot plan takes the trace's pose at each shot's time:
  - `history = J . P(t)`, the head's actual pose, for spin history;
  - `geometry = J . P(t) . P_v^-1`, what is left to move after the re-simulation put the volume at
    `P_v`.

  The per-shot path runs whenever `--motion-sampling shot` is given (the `shot_path` condition at
  `trxscan.rs:584` gains it). The slice order comes from `--slice-order`, as for events.

`ShotPlan::new` gains the per-shot poses as an optional argument. Without it, it builds today's
plan, so every P9 run is unchanged.

**What this approximates.** The per-shot remainder `P(t) . P_v^-1` is applied to the images, not
re-simulated, so the fibre-gradient angle follows the volume's reference pose and not the shot's.
For a nod of a few degrees within one repetition the remainder rotation is that large at most, and
the directional error it makes is the same order as the one `motion::apply_motion` makes and
TRXScan's guidance already names ("Two motion paths, different fidelity"). Re-simulating per shot
would multiply the signal-stage cost by the shot count. It is deferred.

**Python.**
- `Motion.from_confounds(path)` keeps its permissive reading for an untimed file. For a timed one
  (a `time` column, or a sidecar with `SamplingFrequency`, read with `json`) it reads strictly into
  a new field, `Motion.times` (seconds, `None` for untimed).
- `Motion.timed(times, trans_mm, rot, rot_units=...)` builds one from arrays.
- `Motion.resample` refuses a timed trace, because resampling it by volume count has no meaning.
- `Motion.poses_for(n_vol, *, seed=None, tr_s=None, offsets=None)` keyword-only additions; the
  existing positional call is unchanged. For a timed trace it returns the reference-time poses and
  refuses without `tr_s`.
- The three utilities that call `poses_for` (`matrices`, `framewise_displacement`, `plot`,
  `motion.py:209-233`) take the same `tr_s` and `offsets`. With them, a timed trace gives
  reference-time poses, so each utility stays per volume. Without them, `plot` draws the trace's
  samples against time (its natural picture), and `matrices` and `framewise_displacement` refuse,
  naming `tr_s`.
- A new field `Motion.sampling: Literal["volume", "shot"] = "volume"`. With `"shot"`, `simulate`
  builds the shot plan with the trace's per-shot poses. The slice order is
  `Artifacts.shot_events.slice_order` when there is one, otherwise `"interleaved"`, which records
  itself in the sidecar.
- Bindings:
  - `load_motion_trace(path, sampling_hz=None, start_s=0.0)` returns `(times, poses)`, with poses
    `(n, 6)` in mm and degrees.
  - The existing `apply_shot_motion` and `apply_shot_motion_spin_history` keep `volume_poses`
    (`(n_vol, 6)`, mm and degrees, `lib.rs:654`). It is still needed: the reference poses fill
    `ShotPlan.volume`, which spin history uses to undo the re-simulated pose (`motion.rs:134-139`).
  - They gain a keyword `shot_poses=None`: `(n_vol, n_shots, 6)`, mm and degrees, the absolute
    trace pose at each shot.
  - Both arrays are checked for finiteness and exact shapes. `shot_poses` without `volume_poses`
    is refused, because the remainder needs both.
  - Positional order and defaults are unchanged.

**Ground truth.** New `<out>_desc-motionShots.tsv` (CLI) and `Simulation.motion_shots` (Python),
whenever the per-shot path runs: one row per volume and shot, with `volume`, `shot`, `slices`,
`time` and the six parameters of `history`, rotations in radians. This is `aslscan`'s format.
Before P10 a TRXScan run had no per-shot pose truth; it is added because a timed trace makes it
the truth that matters.

**Unchanged.** An untimed `--motion` trace and every P9 flag combination produce identical
outputs: the re-sync baselines and P9's tests are the gate.

---

# Part E: a trace generator

A measured trace sampled faster than the repetition is rare, so a designed one has to be easy to
make. `aslscan/tools/motion_trace.py` is a stdlib-only script, so it runs in any environment. It
writes a timed trace (the `time` column form) from a short description:

```
python tools/motion_trace.py --rate 100 --duration 190 --out nod.tsv \
    --nod start=58.8,rise=0.4,hold=4.2,fall=0.4,trans=0:2:3,rot=0.06:0:0 \
    --drift trans=0:0:0.5
```

- `--nod` is a rise, a hold and a fall. The edges are half cosines by default, or linear with
  `shape=linear`. Its `trans` is mm and its `rot` radians, about the field-of-view centre.
- `--step` is a rise with no fall.
- `--drift` is a linear ramp over the whole run.
- Components add. The script prints each component's span so the user can line it up with the
  run's shot times.
- `--from <trace.tsv>` rewrites a sidecar-timed trace (`SamplingFrequency`, `StartTime`) into the
  `time` column form, for tools that do not read sidecars.

TRXScan reads the same format, and its README points at the script instead of copying it. In
Python, `Motion.timed` with numpy arrays does the same job.

---

# Testing and verification

- **mrsim-acq.**
  - `load_motion_trace` reads a `time` column, and a `sampling` with and without a start.
  - It refuses both at once, no times at all, non-increasing times, a non-finite time or cell, a
    non-positive frequency, and a single sample.
  - The golden record and every bit-pinned test pass unchanged.
- **Untimed compatibility, in each interface.** Permissive cells (an empty cell, `abc`, `inf`)
  and short and long traces behave exactly as today:
  - `aslscan`'s row-count refusal;
  - the TRXScan command line's zero cells and identity padding;
  - Python's zero cells, truncation and short-trace refusal.

  A clean baseline fixture is not enough to show this; these cases are tested explicitly.
- **Part A equivalence.**
  - A timed trace whose samples are each volume's untimed pose at the time of its first shot gives
    P9's untimed `"slice"` transforms bit for bit. To satisfy coverage it needs one more sample at
    the last shot of the last volume, holding the last pose: P9 holds there too, so the nodes stay
    the same numbers.
  - A timed trace that is linear between two shot times gives, at those shots, the transform of
    the cumulative jump that reaches the same pose, to `1e-12` in the matrix entries.
  - Coverage is checked over the query set of the chosen sampling:
    - a trace covering only the reference times is accepted under `"volume"` and refused under
      `"shot"`;
    - each refusal names the needed and the available span.
- **Part B.**
  - `"shot"` and `"slice"` give identical runs in 2D.
  - A timed trace with `"volume"` sampling gives P3's path and poses equal to the trace at the
    reference times.
  - **Hadamard, two cycles with an included `m0scan` row:** an untimed trace still has one row per
    raw volume. A timed trace is queried at every raw volume's own time, and a trace ending before
    the last raw acquisition is refused.
  - The multi-echo series samples at its raw volumes' starts.
  - No reader of `MotionSpec::mode` sees a timed trace's samples as volumes.
  - Spin history runs on a timed trace.
  - An untimed run's sidecar is byte-identical.
- **Part C**, each on the spin-echo path in `series.rs`, the P6 path (multi-echo and Hadamard
  GRASE) and the gradient-echo EPI train:
  - A timed trace constant within each volume, under `"shot"`, gives the same images as the same
    poses with `"volume"` sampling, bit for bit, because equal poses reuse the base images. This is
    checked with a non-identity constant rotation, on an oblique grid.
  - A trace moving between two segments of one volume gives that volume two image sets, and the
    segments' k-space lines come from the right set. In P6 the tissue-only sets match the signal
    sets.
  - The memory estimate counts every live set, and the refusal fires on a tight
    `max_memory_gib` on the new path only. A legacy 3D run with random events and the same small
    limit is not refused.
  - `desc-motionShots_gt.tsv` is written under `"shot"`, with absolute poses and `slices` `n/a`.
  - `"slice"` in 3D is refused; `"shot"` in 3D is accepted.
  - 3D regress cases are unchanged.
- **Part D.**
  - With `--motion-sampling volume` and a timed trace, the re-simulation receives the
    reference-time poses.
  - With `shot`, `geometry . P_v` equals `history` for every shot, to `1e-12`.
  - An untimed `--motion` run and every P9 flag combination reproduce the re-sync baselines and
    P9's checksums.
  - A timed trace without `--tr` is refused.
  - A sidecar-timed trace, including one with an escaped key, is read as timed.
  - Python's `Motion.from_confounds` and `Motion.timed` agree with the CLI's reader, and
    `Motion.resample` on a timed trace raises.
  - `matrices`, `framewise_displacement` and `plot` work per volume with `tr_s`. Without it,
    `plot` draws the samples and the other two refuse.
  - The bindings refuse `shot_poses` of the wrong shape and `shot_poses` without `volume_poses`.
- **Part E.**
  - The generator's output is read by `load_motion_trace` and covers the span it prints.
  - A nod's pose at its hold equals the requested amplitude.

## P10 acceptance criteria

1. With no timed trace, outputs are bit-identical to P9's on every gate: the `mrsim-acq` golden
   record, `aslscan`'s regress (P9's cases plus a P9 shot-path case added in Task 0), and
   TRXScan's re-sync baselines.
2. The P9 ASL nod run (`p9-wt/asl-fig`, multiband 2 interleaved, events file `nod_events.tsv`)
   repeated with a timed trace sampled at 1 kHz describing the same nod, `"shot"` sampling and
   spin history, gives the same images to `1e-5` of the peak. In particular the odd slices of
   volume 12 come out at 0.70-0.82 of the static signal and the even slices at 1.00.
3. The same nod as a timed trace at 100 Hz with half-cosine edges, rising through volume 12's
   readout and falling through volume 13's, run twice with spin history on:
   - With `"volume"` sampling, volume 12's reference pose (its first shot, before the rise) is the
     identity, so volume 12 equals the still run's to `1e-4` of the peak in every slice. Volume 13
     is rigidly displaced: the "before" panel.
   - With `"shot"` sampling, volume 12's odd slices sit below its even slices by at least 0.1 of
     the static signal.
   - Both runs validate with no errors.
4. A 3D GRASE run (`NumberShots` 4) with a timed trace that steps 2 mm through-plane between
   segments 2 and 3 of one volume and returns to baseline before the next volume's first segment.
   - The still run and the moved run differ only in that volume.
   - The difference is spread over the slab: no single partition holds more than half of its
     energy.
   - The sidecar names the one-pose-per-segment approximation.
5. TRXScan on the `trx-fig` multiband-4 grid (interleaved, `--tr 4.8`) with a timed nod trace
   inside one volume, returning to baseline before the next, and `--motion-sampling shot`:
   - `desc-motionShots.tsv`'s `time` column equals `v x 4.8 + offset`.
   - **Without spin history**, motion is localized:
     - the nod volume's slices acquired while the head is off baseline differ from the still run,
       with the largest difference above 1% of the peak;
     - its slices acquired before the nod starts match the still run to `1e-4` of the peak;
     - every other volume matches the still run to `1e-4` of the peak.

     All reference poses are the identity, so the re-simulation is the still run's.
   - **With spin history**, compared against the same motion without it, so the geometry is
     taken out:
     - the nod volume's odd slices sit below their even neighbours by at least 0.05;
     - the next volume echoes the comb with the opposite sign. Its even pass starts about 2.4 s
       after the nod volume's displaced odd pass, so the tissue now in its even slices has had
       half a repetition to recover. The tissue in its odd slices was skipped and has had one and
       a half;
     - the volumes after that are unchanged, because a whole volume of 90-degree excitations at
       rest erases the history.

     (Revised in Task 10. The draft compared against the still run, which mixes the geometry
     in, and expected the next volume mostly recovered. Both were wrong. The reference run in
     the context shows the same persistence into its volume 38.)

---

# What the review changed

Codex adversarial review, 2026-10-17, 11 findings, all valid:

- **High: the clock.** Coverage was to be checked in the protocol, on `row_start`. Hadamard runs
  on a separate raw-acquisition clock with `H` raw volumes per cycle (`schedule.rs:157-202`), and
  motion indexes raw volumes. Query times now come from each series path's own clock, built at
  the start of the series function. The draft's claim that an untimed raw volume uses its protocol
  row was wrong and is removed.
- **High: a third 3D builder.** `p6.rs:1236-1290` builds segment sets for multi-echo and Hadamard
  3D, tissue-only images included. Part C and Task 6 now cover it.
- **High: untimed compatibility.** The plan switched every reader to the strict loader. That would
  have refused files the TRXScan command line and Python accept today, and the draft misdescribed
  their row-count rules. Timing is now detected first, and untimed files keep each interface's
  rules.
- **Memory.** The `ge3d` estimate counts per volume in flight. The spin-echo builders keep every
  volume's sets, and P6 keeps tissue-only copies too. Each builder now gets its own estimate.
  `[images] max_memory_gib` is accepted on the new path, and the refusal applies only there.
- **Coverage and equivalence.** Coverage is over the chosen sampling's query set. The P9
  equivalence test adds one terminal sample.
- **Kernels.** Matrix resampling would have changed legacy 3D bytes. Legacy builders keep
  `resample_by_pose`, and the new path reuses base images where poses are equal.
- **3D truth.** The shots truth was 2D-only. It is now written from a readout-independent list.
- **Acceptance.** Criterion 4's step now returns before the next volume. Criterion 5 tests
  localization without spin history and bounds recovery with it. Criterion 3 is stated on volume
  12, whose reference pose is the identity.
- **Python.** The bindings keep `volume_poses` and gain `shot_poses` with stated shapes. The
  utilities that call `poses_for` take `tr_s`.
- **The shared enum.** `MotionMode` is `mrsim-acq`'s and is built by TRXScan's binding. The trace
  now lives in each consumer's own structure.
- **Low: the substring check** could miss an escaped key. TRXScan's command line now parses JSON.

The review also confirmed that `J . P(t) . P_v^-1` is the right relative transform on images
already at `P_v`, and that P9's priming is unaffected (stated in part A).

# Decisions deferred

- **Motion during one shot's readout.** An EPI readout of 30-60 ms, or a 3D echo train of
  hundreds, is read at one pose. Motion inside it blurs and ghosts within the shot, and needs the
  pose per k-space line in the forward model.
- **Motion during labeling and background suppression.** A head moving during PCASL labeling
  changes what is labeled, and one moving between suppression pulses changes what is nulled. The
  trace knows the pose then; the labeling and suppression models do not use poses.
- **Gradient-echo spin history.** The reference run's slow recovery over volumes 39-40 needs it
  (P9's deferral stands).
- **BIDS-Motion files** (`_motion.tsv` with no header and a `_channels.tsv`). They can be
  converted to the `time` column form; reading them directly can come when a user has them.
- **Geodesic interpolation.** It is second order for a trace sampled faster than the shots.
- **TRXScan per-shot re-simulation**, for the fibre-gradient angle at each shot's pose.
- **Poses that move the per-voxel maps and the fieldmap**, as in P3 and P9.
- **The resampler's edge.** `trilinear` returns zero for a sample point outside `[0, n - 1]`, so a
  head moved through the slice direction by any amount blanks the end slice it leaves. The tissue
  beyond the stack is not in the phantom, so zero is defensible for a real shift. But the value
  jumps from the slice's own to zero between 0 and 1e-13 mm, which is not. Zero-padding the
  missing corners would be continuous. It would also change every existing motion output at the
  stack ends, so it needs its own gate run and the user's decision.

# Risks

- **Alignment is the user's problem.** A measured trace has its own clock. P10 states the run's
  clock and writes every shot's time, and it refuses a trace that does not cover the run. A trace
  that covers the run but is offset by a second is accepted and is wrong.
- **3D memory.** Continuous motion makes every segment a separate image set, and the spin-echo
  builders keep all volumes' sets at once. The estimate refuses a run that will not fit; it does
  not make one fit. Keeping sets per volume in flight, as `ge3d` does, is the remedy if it bites.
- **The TRXScan remainder.** Applying the per-shot remainder to images is a fidelity step down
  from re-simulation, inside the path that is otherwise the faithful one. The sidecar and the docs
  have to say so plainly, or a user will read the result as fully re-simulated.
- **No oracle.** As in P9, the checks are closed forms and equivalences (to P9's events and to
  P9's untimed sampling), not an independent simulator.
