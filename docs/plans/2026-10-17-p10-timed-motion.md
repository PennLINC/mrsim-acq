# P10: motion traces sampled faster than the repetition — implementation plan

**Goal:** implement the P10 addendum (`docs/specs/2026-10-17-p10-timed-motion-design.md`) across
`mrsim-acq`, `aslscan` and TRXScan, with every existing output byte-identical, P9's included.

- **Part A:** a timed trace (a `time` column or a BIDS-physio-style sidecar), sampled at shot times.
- **Part B:** `aslscan` 2D on P9's shot path.
- **Part C:** `aslscan` 3D, a pose per readout segment.
- **Part D:** TRXScan: reference-time poses for the re-simulation, the per-shot remainder, and
  per-shot truth.
- **Part E:** a stdlib trace generator.

**Architecture.** `mrsim-acq` gains one loader and two types, all additions; the interpolation is
P9's `sample_trajectory`, unchanged. Each consumer builds its own shot times, as it does since P9,
and asks the trace for poses at them. The 3D work reuses P5's per-segment image sets and changes
only where their poses come from.

**Order.** The loader comes first, because every other task reads traces. The generator comes
next, because the acceptance runs need nod traces. `aslscan` 2D comes before 3D, because 3D reuses
2D's protocol work. TRXScan comes last, because it is pinned to a `mrsim-acq` tag and lands by pull
request.

**Spec state.** Written 2026-10-17 by Claude, not reviewed by the user. The same day, a Codex
adversarial review of the spec and this plan returned 11 findings. All held against the source and
were applied to both documents; the record is at the end of this file.

**State at writing.**
- P9 is tagged `p9-complete` in `mrsim-acq` (`ad1f081`) and `aslscan` (`f3c0593`; the branch has
  since moved to `7ded4f2` with a replaced figure).
- PennLINC/aslscan#6 is open against `osipi-terminology`.
- TRXScan's `p9-shot-motion` (`724c87c`) is pushed, with PennLINC/TRXScan#20 open against `main`
  and the pin at `p9-complete`.

**Base (proposed; the user chose P9's).** New worktrees `rust-trx/p10-wt/{mrsim-acq,aslscan,TRXScan}`,
made with WSL git, on branch `p10-timed-motion`, starting at the P9 heads: `mrsim-acq` `ad1f081`,
`aslscan` `7ded4f2`, TRXScan `724c87c`. The P10 branches stack on the open P9 PRs and are rebased if
those change in review.

## Global constraints

- Everything in the P1-P9 plans' global constraints still holds:
  - the pure-std default build;
  - no `cargo fmt`;
  - no large data committed;
  - every approximation named in the sidecar;
  - refusals over silent defaults.
- **No timed trace, no changed byte**, in four places:
  1. **`mrsim-acq`**: `tests/golden_baseline.rs` and every bit-pinned test.
  2. **`aslscan`**: `tools/regress_identity.sh f3c0593 ad1f081`, both feature sets, including the
     P9 shot-path case Task 0 adds. Never two runs at once (`work/regress.lock`).
  3. **TRXScan**: `tools/run_resync_baseline.sh` and `tools/resync_python_baseline.py`, diffed
     against `tests/fixtures/resync_baseline`. Before every Python run, `maturin develop
     --release` from the worktree being tested, and check that the imported `trxscan._core` is the
     one just built.
  4. **TRXScan P9 runs**: the checksum set Task 0 records (below), diffed after Tasks 8, 9 and 10.
     The re-sync baselines predate P9 and do not cover its flags.
- **Gate cadence.**
  - Every task: the test suites and the golden record.
  - `aslscan` regress: end of Tasks 0, 4, 6 and 7.
  - TRXScan baselines and the P9 checksum set: end of Tasks 0, 8, 9 and 10.
- **Environment.**
  - Everything builds and runs in WSL.
  - Python through `micromamba run -n rust-trx` (TRXScan bindings, pytest) and `-n simasl`
    (phantom tools).
  - Plots through `-n trxscan`, the only environment with matplotlib.
  - The generator is stdlib-only and runs under any of them.
- **Shell.**
  - Scripts start with `set -euo pipefail`.
  - Nothing with a `$` goes inside a PowerShell-quoted `wsl -e bash -lc "..."`; scripts go in the
    scratchpad and run with `wsl -e bash <script>`.
  - Every path a file-rewriting command writes to is absolute and inside the project.
- **Commits and pushes.**
  - One commit per task, authored Taylor Salo, with no co-author trailer and no Claude line, in
    commits and pull requests alike.
  - WSL git commits pass `-c user.name="Taylor Salo" -c user.email=tsalo90@gmail.com`.
  - Nothing is pushed, tagged on a remote, or opened as a pull request without the user's
    go-ahead.
  - TRXScan's local `.cargo/config.toml` patch is never committed.
- **`gh`** runs from a directory outside the WSL-made worktrees (it cannot read their `.git`
  files).

## File structure

| Repository | Path | Change |
|---|---|---|
| mrsim-acq | `src/motion.rs` | `Trace`, `Sampling`, `load_motion_trace` |
| aslscan | `src/protocol.rs` | timed-trace detection and sidecar read; `within_volume_sampling = "shot"`; coverage and the other new refusals |
| aslscan | `src/series/shots.rs` | `shot_plan` takes trace nodes; `reference_poses` |
| aslscan | `src/series.rs`, `src/series/p6.rs`, `src/series/ge3d.rs` | Query times and coverage at each series start; reference poses through `volume_poses`; a new 3D `"shot"` branch beside each of the three segment builders (P6's tissue-only sets included); per-builder memory estimates; the shots truth list |
| aslscan | `src/bids.rs`, `src/bin/aslscan.rs` | Sidecar `Trajectory` block and segment approximation; run summary |
| aslscan | `tools/motion_trace.py`, `tools/test_motion_trace.py` | The generator and its tests |
| aslscan | `tests/end_to_end.rs`, `tests/fixtures/protocols/p10_*`, `tools/acceptance_p10.sh`, `tools/acceptance_p10_check.py` | Tests, fixtures, acceptance |
| aslscan | `tools/regress_identity.sh`, `README.md` | The P9 shot-path case; the new keys |
| TRXScan | `Cargo.toml`, `.cargo/config.toml` | The pin; `serde_json` under `cli`; a local patch during development |
| TRXScan | `src/motion.rs`, `src/bin/trxscan.rs` | `ShotPlan::new` per-shot poses; `--motion-sampling`; timed `--motion`; `desc-motionShots.tsv` |
| TRXScan | `python/src/lib.rs`, `python/trxscan/{motion,simulate,bids}.py`, `python/tests/test_motion.py` | `Motion.times`, `Motion.timed`, `Motion.sampling`, `Simulation.motion_shots`; bindings |
| TRXScan | `docs/FEATURES.md`, `README.md`, `CLAUDE.md` | What is modeled now, and the per-shot remainder's fidelity |

---

## Milestone 0: baseline

### Task 0: worktrees, gates, review

- [ ] Worktrees and branches as under "Base", phantoms linked into `p10-wt/aslscan/work` as
      `p9-wt` does, and TRXScan's local patch pointed at `../mrsim-acq`. Commit the spec and this
      plan on the `mrsim-acq` branch.
- [ ] `aslscan`: add a P9 shot-path case to `tools/regress_identity.sh`, `p9_shot_spin`: the
      single-band acceptance protocol (`tests/fixtures/protocols/p9_single_band`) with an events
      file and spin history, on `work/phantom-3t-z100`. Then `regress_identity.sh f3c0593 ad1f081`
      exits 0, and `--self-test` exits 0 (every executed case detects the injected perturbation).
      Record the case counts here.
- [ ] TRXScan: the re-sync baselines `diff` clean on the untouched branch.
- [ ] TRXScan: record the P9 checksum set (`p10-wt/trx-p9-checksums.txt`, not committed): md5 of
      every NIfTI and TSV from the three `trx-fig` single-band runs, the three multiband-4 runs and
      the `nod_after` run, each rerun from its recorded command on the untouched branch. Rerun
      once and confirm it reproduces itself. If it does not, record which outputs vary, and why,
      before anything depends on them.
- [x] Codex adversarial review of the spec and this plan (2026-10-17, 11 findings, all applied). Record the
      findings and what was done with each at the end of this file.

---

## Milestone A: the trace

### Task 1: `mrsim-acq` loader

All in `src/motion.rs`, pure std, nothing existing edited.

- `pub struct Trace { times: Option<Vec<f64>>, poses: Vec<Pose> }` and
  `pub struct Sampling { frequency_hz: f64, start_s: f64 }`.
- `pub fn has_time_column(path) -> Result<bool, String>`: reads the header only. Callers use it,
  with their own sidecar read, to detect timing before choosing a loader.
- `load_motion_trace(path, sampling)`, for timed traces only:
  - reads the header and the six columns (rotations radians in the file, degrees in `Pose`) plus
    the `time` column;
  - is strict: a missing cell or a non-finite number is an error. `n/a` is 0, as `aslscan`'s
    `check_motion_tsv` documents.
  - Times come from the column or from `start_s + i / frequency_hz`.
  - It refuses neither and both, non-increasing times, a non-finite time, a non-positive
    frequency, and fewer than two samples.
- `load_motion_tsv` and `MotionMode` are not touched (spec part A).
- A doc comment saying that coverage is the caller's check and why.

Tests:
- a column trace, and a sampled trace with and without a start;
- each refusal, matched on its message;
- `has_time_column` on both kinds of header;
- `sample_trajectory` over a loaded timed trace returns the samples exactly at their own times.

The golden record passes.

### Task 2: the generator

`aslscan/tools/motion_trace.py`, stdlib only (`argparse`, `math`, `json`, `csv`):

- `--rate` (Hz), `--duration` (s), `--start` (s, default 0), `--out`.
- Components:
  - `--nod start=,rise=,hold=,fall=,trans=x:y:z,rot=x:y:z[,shape=cosine|linear]` (rise and fall
    are half cosines by default; `linear` is for criterion 2, whose samples must equal cumulative
    jumps at the shot times);
  - `--step start=,rise=,trans=,rot=`;
  - `--drift trans=,rot=` (linear over the duration).
  Each may repeat, and components add.
- `--from <trace.tsv>`: read a sidecar-timed trace (`SamplingFrequency`, `StartTime`) and write
  the `time` column form.
- Prints each component's span and the trace's span.

`aslscan/tools/test_motion_trace.py` (stdlib `unittest`):
- a nod's pose at its hold equals the amplitude;
- the rise is monotonic;
- components add;
- `--from` round-trips a sidecar trace;
- the output has strictly increasing times.

A cross-check in Task 3's tests reads a generated trace with `load_motion_trace`.

---

## Milestone B: `aslscan` 2D

### Task 3: protocol

- `MotionSpec` gains `trace: Option<Trace>`. `MotionMode` (`mrsim-acq`'s) is not changed. For a
  timed trace, `mode` holds `MotionMode::Off` as a placeholder and `mode_name` stays
  `"trajectory"` for the sidecar.
- New `volume_poses(spec, reference_times, seed) -> Vec<Pose>`: the trace at the reference times
  when `trace` is set, otherwise `resolve_poses(&spec.mode, n, seed)`. Search the source for every
  reader of `MotionSpec::mode` and of `resolve_poses`; the spec names `series.rs:1205`,
  `p6.rs:1187` and `ge3d.rs:628`. Route each through `volume_poses` and list them here.
- In `overlay_motion`'s `"trajectory"` arm, detect timing first:
  - `has_time_column`, and the sidecar at the trace's path with `.json`, read with `serde_json`.
    `SamplingFrequency` there gives a `Sampling` (with `StartTime`, default 0).
  - Untimed: today's code, `check_motion_tsv`, `load_motion_tsv` and the row-count check against
    the motion units, unchanged.
  - Timed: `load_motion_trace`, no row-count check.
- `within_volume_sampling` accepts `"shot"`. In 2D it is the same `WithinVolumeSampling::Slice`;
  the given spelling is kept for the sidecar.
- `"shot"` with `random` is refused, as `"slice"` is. In 3D, `"slice"` stays refused. `"shot"` in
  3D is implemented in Task 6. Until then it is refused with a message that names what is not yet
  supported.
- **Coverage is not checked here.** The protocol does not have every clock: Hadamard's raw clock
  exists only after `hadamard_spec` (`protocol.rs:3102`, `schedule.rs:157`). Task 4 checks it in
  each series function.
- The path decision (`protocol.rs:1278`) is unchanged for 2D: `"shot"` is `Slice`. A timed trace
  with `"volume"` sampling and nothing else stays on P3's path.

Tests (`protocol.rs`):
- a column-timed and a sidecar-timed trace parse, with times, and the sidecar may use an escaped
  key;
- a sidecar without `SamplingFrequency` leaves a trace untimed, and its row count is still checked;
- both forms at once are refused;
- untimed permissive cells (`n/a`) and the row-count refusal behave exactly as before;
- `"shot"` equals `"slice"` in 2D;
- `"shot"` with `random` is refused;
- a generated nod (Task 2) parses.

### Task 4: series, truth, sidecar

- `shots.rs`:
  - `shot_plan` takes `nodes: Option<(&[f64], &[Pose])>`. With `Some`, the trace's nodes replace
    the first-shot nodes at `shots.rs:80`. With `None` it is P9's code, unchanged.
  - New `query_times(starts, signal_t, offsets, sampling)`: the reference times under `"volume"`,
    every shot time under `"shot"`, from the caller's clock.
  - New `check_coverage(trace, times)`, whose message names the needed and the available span.
- Each series function builds its query times **at its start, before the signal stage**, and
  checks coverage there, so a refusal costs nothing:
  - `series.rs` from `row_start` and the rows' `t`;
  - `p6.rs` from its `Schedule`: raw volumes' `first_prep(v).start_s` (`p6.rs:1190`), with `H` raw
    volumes per Hadamard cycle and included `m0scan` preparations;
  - `ge3d.rs` from its own schedule (Task 6 fills in the 3D segment times).
- The per-volume `poses` come from `volume_poses` (Task 3), and `shot_plan` gets the trace's nodes.
- The `desc-motionShots_gt.tsv` writer (`bids.rs:1231`) takes a readout-independent list
  (`volume`, `shot`, `slices`, `time`, absolute pose). The 2D path fills it from its plan, so its
  bytes do not change. Task 6 fills it for 3D.
- `bids.rs`:
  - the `Motion.Trajectory` block (spec part B), only for a timed trace;
  - `ShotTransform`'s description for a timed trace says "the trace's pose at the shot's time";
  - an untimed run's sidecar is byte-identical.
- `bin/aslscan.rs`: the summary says "timed trace (N samples, S1-S2 s)". When the largest sample
  interval exceeds the smallest shot spacing, it says that the trace is interpolated between
  shots.
- `README.md`: the timed forms, the clock, `"shot"`, and the generator.

Tests:
- **Equivalence (spec part A):**
  - A timed trace with samples at the untimed first-shot times, plus one terminal sample holding
    the last pose at the last shot, gives P9's transforms bit for bit.
  - A trace linear between two shot times gives the cumulative-jump transform to `1e-12`.
- Coverage over the query set: a trace covering only the reference times is accepted under
  `"volume"` and refused under `"shot"`. A refusal comes before the signal stage (time it on the
  test protocol).
- `"volume"` sampling of a timed trace gives the reference poses and P3's path.
- **Hadamard, two cycles with an included `m0scan` row:**
  - the plan's times equal `first_prep(v).start_s + t + offset` for every raw volume;
  - an untimed trace still needs one row per raw volume;
  - a timed trace ending before the last raw acquisition is refused.
- The multi-echo series samples at its raw volumes' starts.
- Spin history runs on a timed trace.
- The sidecar block is present for a timed trace and absent otherwise.
- The 2D `desc-motionShots_gt.tsv` is byte-identical through the new list.

Regress at the end of this task.

### Task 5: 2D acceptance

`tools/acceptance_p10.sh` and `tools/acceptance_p10_check.py`, on the P9 ASL figure's inputs
(`p9-wt/asl-fig`: `asl.json`, `aslcontext.tsv`, phantom `phantom-3t-z100`) copied into
`tests/fixtures/protocols/p10_mb2/`. They must be small; check the sizes before committing.

- **Criterion 2:** generate the nod at 1 kHz that matches `nod_events.tsv`: a linear rise over
  volume 12's shots and a fall over volume 13's, at the shot times from P9's
  `desc-motionShots_gt.tsv`. Linear, not half-cosine, so the samples at the shot times equal the
  cumulative jumps. Run it with `"shot"` and spin history, and compare with the events run to
  `1e-5` of the peak. The check prints the per-slice ratios of volume 12.
- **Criterion 3:** the same nod at 100 Hz with half-cosine edges, rising through volume 12's
  readout and falling through volume 13's, spin history on, `"volume"` against `"shot"` sampling.
  - The check requires that the `"volume"` run's volume 12, whose reference pose is the identity,
    equals the still run's to `1e-4` of the peak.
  - It also requires that the `"shot"` run's odd slices in volume 12 are below its even slices by
    at least `0.1` of the static signal.
- Every run passes the BIDS validator.
- Record the outputs here.

---

## Milestone C: `aslscan` 3D

### Task 6: per-segment poses

- **Confirm the segment times** against the source, and write here which code gives each:
  - Spin-echo 3D (GRASE, stack of spirals) in `series.rs`: segment `s` of row `v` excites at
    `row_start[v] + s x TR_row + t_row` (`series.rs:980-982`).
  - The P6 series (multi-echo, Hadamard; 3D): `sched.preps_of(v)[s].start_s + t` (`p6.rs:783`).
  - Gradient-echo EPI train (`ge3d.rs`): the time of each segment's first excitation on that path's
    schedule.

  If any differs from the spec, stop and fix the spec first.
- `protocol.rs`: accept `"shot"` in 3D with `mode = "trajectory"` (timed or untimed). Untimed
  `"shot"` in 3D samples the P9 node trajectory at the segments' times, the 3D analogue of
  `"slice"`. Refuse `"shot"` with `random`. Accept `[images] max_memory_gib` on the 3D `"shot"`
  path (today refused outside `epi3d` and compat multi-TE, `protocol.rs:3224`), default 4 GiB.
- **Three builders, each with a new branch beside the existing one.** The existing code and its
  `resample_by_pose` calls are untouched for every run without `"shot"` sampling:
  - `series.rs:1284-1328` (spin-echo 3D);
  - `p6.rs:1236-1290` (its signal **and** tissue-only sets);
  - `ge3d.rs:631-651` (`shot_pose`; `Ge3dBuild::poses` holds the reference poses).

  Under `"shot"`, each segment's absolute pose is `J(v, s) . P(v, s)`, from the trace (or the
  untimed node trajectory) at the segment's time:
  - if it equals `P_v` as a pose, the segment uses the base images (no set);
  - otherwise the set is resampled by the matrix `J . P(t) . P_v^-1` with
    `resample_slices_by_matrix` over every slice.

  Distinct poses make sets as now. In `p6.rs`, each signal set has its tissue-only twin.
- **Memory.** One estimate per builder, computed from the query times and poses before any set is
  allocated:
  - `series.rs`: every volume's sets are live at once (`series.rs:1289`);
  - `p6.rs`: the same, twice, for the tissue-only copies (`p6.rs:1241`);
  - `ge3d`: its in-flight formula (`ge3d.rs:717`), with the pose count from the plan.

  Over the limit, the run is refused, naming the segment sets. The new path only.
- **Truth.** Fill the readout-independent shots list from Task 4 for the 3D `"shot"` path:
  - raw volume index;
  - segment;
  - `slices` `n/a`;
  - time;
  - absolute pose `J . P(t)`.

  `ge3d` returns it in place of `shots: None` on this path only.
- `bids.rs`: on the 3D per-segment path, the `Approximations` list gains "one pose per readout
  segment: a segment's echo train is read at one pose" and "segment images are resampled from the
  volume's moved images (two interpolations)".

Tests, on each of the three builders (GRASE through `series.rs`, multi-echo and Hadamard GRASE
through `p6.rs`, the EPI train through `ge3d`):
- a timed trace constant within each volume, under `"shot"`, equals `"volume"` sampling bit for
  bit, with a non-identity constant rotation on an oblique grid;
- a trace moving between segments 2 and 3 gives two image sets, and the lines of segments 3-4 come
  from the moved set. In `p6.rs` the tissue-only sets match;
- the memory estimate counts every live set. The refusal fires on a tight `max_memory_gib` on the
  new path, and a legacy random-event run with the same limit is not refused;
- `desc-motionShots_gt.tsv` is written with absolute poses;
- `"slice"` in 3D is refused.

### Task 7: 3D acceptance

- **Criterion 4**, added to `acceptance_p10.sh`: a GRASE run (`NumberShots` 4, the P5 regress
  protocol) with a timed trace that steps 2 mm through-plane between segments 2 and 3 of one
  volume and returns to baseline before the next volume's first segment. Otherwise the next
  volume's reference pose would stay displaced.
- The check requires that the still run and the moved run differ only in that volume. It also
  requires that the difference is spread over the slab: no single partition holds more than half
  of its energy.
- The sidecar names the approximation.
- The output validates.
- Regress at the end of this task (its 3D cases must not move).

---

## Milestone D: TRXScan

### Task 8: library and command line

On `p10-wt/TRXScan`, built against the local `mrsim-acq` through the patch.

- `src/motion.rs`:
  - `ShotPlan::new` gains `shot_poses: Option<&[Vec<Pose>]>` (`[volume][shot]`, the trace's pose
    at each shot time).
  - With it, `history = J . P(t)` and `geometry = J . P(t) . P_v^-1`. Without it, today's code
    path, unchanged.
  - Re-export `load_motion_trace`, `has_time_column`, `Trace` and `Sampling`.
- `Cargo.toml`: `serde_json` (optional), enabled by `cli`.
- `src/bin/trxscan.rs`:
  - `--motion` detects timing first: `has_time_column`, and the sidecar parsed with `serde_json`.
  - Untimed: today's `load_motion_tsv` call and identity padding (`compartments.rs:883`),
    unchanged.
  - Timed: `load_motion_trace`. It needs `--tr`, and is refused without it.
  - The re-simulation receives the reference poses (`v x TR + offset of the first shot`).
  - `--motion-sampling volume|shot` (default `volume`, `requires = "motion"`). With `shot`, the
    per-shot path runs (the `shot_path` condition at `trxscan.rs:584`) with
    `ShotPlan::new(.., Some(shot_poses))`.
  - `<out>_desc-motionShots.tsv` whenever the per-shot path runs: `volume`, `shot`, `slices`,
    `time`, and the six parameters of `history` (radians).
  - Coverage check, with the spans in the message.

Tests:
- `geometry . P_v == history` to `1e-12` for every shot;
- `shot_poses = None` gives today's plan bit for bit;
- **Untimed compatibility on the command line:**
  - a trace with an empty cell, `abc` and `inf` cells reads as today (zeros; `inf` as parsed);
  - a short trace pads with the identity;
  - a long trace ignores the extra rows;
- a sidecar-timed trace, including one with an escaped key, is read as timed;
- each refusal;
- the motion-shots TSV round-trips.

Gates: the re-sync baselines and the P9 checksum set.

**The `desc-motionShots.tsv` file is new on every per-shot run, P9's included.** A file the P9
checksum set did not have is an addition, not a changed byte; the set is checked for the files it
holds. Record the decision here.

### Task 9: Python

- `motion.py`:
  - `Motion.times: np.ndarray | None = None`;
  - `Motion.sampling: Literal["volume", "shot"] = "volume"`;
  - `Motion.from_confounds` keeps its permissive reading for untimed files and reads a `time`
    column or a `SamplingFrequency` sidecar (with `json`) strictly;
  - `Motion.timed(times, trans_mm, rot, rot_units=...)`;
  - `Motion.resample` raises on a timed trace;
  - `Motion.poses_for(n_vol, *, seed=None, tr_s=None, offsets=None)` gives reference poses for a
    timed trace and raises without `tr_s`. The existing call and its truncation and short-trace
    refusal are unchanged;
  - `matrices`, `framewise_displacement` and `plot` (`motion.py:209-233`) take `tr_s` and
    `offsets`. Without them, `plot` draws a timed trace's samples against time, and the other two
    raise, naming `tr_s`;
  - coverage is checked against the query times.
- `simulate.py`: with `sampling == "shot"`, build the shot plan with per-shot poses. The slice
  order is `Artifacts.shot_events.slice_order`, else `"interleaved"`, recorded in the sidecar.
  `Simulation.motion_shots` holds the shots table.
- `bids.py`: `Dataset` writes `desc-motionShots` under `derivatives/trxscan`.
- `python/src/lib.rs`:
  - `load_motion_trace(path, sampling_hz=None, start_s=0.0)` returns `(times, poses (n, 6))`.
  - `apply_shot_motion` and `apply_shot_motion_spin_history` keep `volume_poses` (`(n_vol, 6)`,
    `lib.rs:654`) and gain a keyword `shot_poses=None` (`(n_vol, n_shots, 6)`, mm and degrees,
    absolute).
  - Both are checked for finiteness and exact shapes, and `shot_poses` without `volume_poses` is
    refused.
  - Positional order and defaults are unchanged.

Tests (`test_motion.py`):
- `from_confounds` agrees with the CLI reader on both timed forms, and keeps its permissive untimed
  behaviour (zero cells, truncation, short-trace refusal);
- `timed` round-trips;
- `resample` raises;
- `poses_for` gives the reference poses, and the three utilities behave as specified with and
  without `tr_s`;
- the bindings refuse wrong `shot_poses` shapes and `shot_poses` without `volume_poses`;
- a `"shot"` simulation on a small phantom writes a shots table whose times equal the plan's;
- untimed `Motion` and every P9 path are unchanged (the Python baseline).

Gates: `maturin develop --release`, the import check, pytest, the Python baseline, and the P9
checksum set.

### Task 10: docs and acceptance

- `docs/FEATURES.md`, `README.md`, `CLAUDE.md`:
  - timed traces, `--motion-sampling`, and the motion-shots truth;
  - the generator, pointed at in `aslscan`;
  - plainly, that the per-shot remainder is resampled, not re-simulated, inside the otherwise
    faithful `--motion` path.
- **Criterion 5:** the `trx-fig` multiband-4 grid with a timed nod trace (generator, 100 Hz)
  inside one volume, returning to baseline before the next, with `--motion-sampling shot` and
  `--tr 4.8`. Two runs:
  - **Without spin history:**
    - the nod volume's slices acquired off baseline differ from the still run (largest difference
      above 1% of the peak);
    - its slices acquired before the nod match the still run to `1e-4` of the peak;
    - every other volume matches to `1e-4`.
  - **With spin history:**
    - the nod volume's odd slices sit below their even neighbours by at least `0.05` (mean ratios
      over the brain mask);
    - the next volume's odd/even difference is at most a quarter of the nod volume's. The check
      reports it and does not require it to vanish.
  - In both runs the `desc-motionShots.tsv` times equal `v x 4.8 + offset`.
- Gates: the re-sync baselines and the P9 checksum set.

---

## Milestone E: landing

### Task 11: final gates and review

- [ ] Every gate on the final heads:
  - `mrsim-acq` tests and the golden record;
  - `aslscan` tests, both feature sets, the regress, and acceptance criteria 1-4;
  - TRXScan's cargo matrix (default, `cli`, `cli,par`, `kspace`), pytest, both baselines, the P9
    checksum set, and criterion 5.
- [ ] Codex adversarial review of the implementation (read-only, per the `codex-cli-location`
      memory). Fix the findings that hold against the source; record each with what was done.
      Then an ordinary Codex code review as the final pass.
- [ ] On the user's go-ahead only:
  - tag `p10-complete` on `mrsim-acq` and `aslscan`;
  - move TRXScan's pin to it and rerun its gates without the patch;
  - push;
  - open the two PRs, stacked on PennLINC/aslscan#6 and PennLINC/TRXScan#20, each with a figure
    of the reference-run pattern reproduced from a timed trace (`"volume"` against `"shot"`
    sampling).

---

## Record

*(Task outcomes and case counts go here as the work proceeds.)*

### Implementation, 2026-10-18

**Base.** The user chose "start on the plan in a new worktree" without changing the proposed base.
Worktrees are `p10-wt/{mrsim-acq,aslscan,TRXScan}` on `p10-timed-motion`, from `ad1f081`,
`7ded4f2` and `724c87c`. The spec and plan moved there from `p9-wt` (`67a4a8c`). Edits are made in
`p10-scratch/` copies and tested there, then copied into the worktrees. The regress snapshots the
live trees when it starts, so the worktrees must never hold half-edited code.

**Task 0.**
- `regress_identity.sh` gains `p9_shot_spin` (the single-band events file with spin history) and
  `p9_slice_drift` (the slice-sampled drift with spin history): 37 + 35 cases.
- The TRXScan re-sync baselines on the untouched branch are identical (594 CLI, 177 Python).
- The P9 checksum set has 45 files from six runs (`still`, `after`, `sb_still`, `sb_after`,
  `nod_still`, `nod_after`), in `p10-wt/trx-p9-t0a`. It reproduces itself (`t0b`) and matches the
  P9 figure runs.

**Task 1** (`a8e0369`):
- `Trace` (times required, not optional: the loader is for timed traces only), `Sampling`,
  `has_time_column`, `load_motion_trace`;
- `Trace::pose_at`, `Trace::max_interval`, `Trace::check_covers` (the shared coverage message);
- tests: 4 new, 22 in `motion::`; all four mrsim-acq feature sets and the golden record pass.

**Task 2** (`e4fc580`): the generator and 8 stdlib tests.

**Tasks 3-4, one commit.** Task 3 alone would parse a timed trace that the series then ignored.
- `WithinVolumeSampling::Shot`, with `per_shot()` and `as_str()`.
- `MotionSpec.timed` (a `TimedTrajectory`), `MotionSpec::volume_poses` and
  `check_timed_coverage`.
- **The readers of `mode`:** `resolve_poses` in `series.rs`, `p6.rs` and `ge3d.rs` (now
  `volume_poses`), plus the "Off without events is no motion" check in `overlay_motion`, which
  would have dropped a timed trace held as `Off`. That one now checks `timed` too. There are no
  other readers.
- `shots::query_times`, `reference_times`, `min_shot_spacing`.
- **Coverage** is checked at the start of `simulate_legacy` (rows' clock), `p6::build` (raw
  volumes) and `ge3d` (raw volumes, first segment).
- **Sidecar:** the `Trajectory` block, plus `VolumePose` on P3's path. The shots table writer is
  unchanged for 2D; the readout-independent list comes with Task 6.
- Tests: protocol, shots, series (bit-for-bit equivalence under "volume", "slice"/"shot" and spin
  history; coverage over the query set, refused in under 0.5 s), Hadamard with two cycles and an
  m0scan, and an end-to-end sidecar test. Library 231, end to end 60 (`cli`) and 63
  (`cli,kspace,par`).

**Task 5.** Acceptance criteria 2 and 3 pass, and every run validates:
- criterion 2: the timed linear nod equals P9's events run **bit for bit** (difference 0); volume
  12's odd slices are at 0.69-0.82 and its even slices at 0.99-1.02;
- criterion 3: under "volume", volume 12 equals the still run (difference 0) and volume 13 is
  moved (0.968 of the peak); under "shot", the even slices sit 0.343 above the odd ones.

**Finding (Task 5): clock rounding against the resampler's edge.** The first acceptance run
failed: slice 0 of volume 12 sat at 0.16-0.24 in every timed run.
- **Cause:** the series clock puts that shot at 58.80000000000001 s, so the interpolated pose was
  about 1e-13 mm, not zero. `trilinear` zeroes any sample outside the grid, so the bottom slice was
  blanked.
- **Fix** (in `mrsim-acq`): `Trace::pose_at` returns the identity within `TRACE_POSE_EPSILON` =
  1e-9 mm and 1e-9 degrees. The spec's part A and "Decisions deferred" record it.
- **Not changed:** the kernel's edge itself, because changing it would change P3/P9 outputs.

**Task 0 regress.** `regress_identity.sh f3c0593 ad1f081` exits 0: 37 cases executed under
`cli,kspace,par` and 35 under `cli`, 2 skipped (they need `kspace`), all identical. The self-test
started on the committed tree (aslscan `a3c1890`, mrsim-acq `a984b88`).

**Task 6.**
- **Segment times**, confirmed from the source:
  - spin echo (`series.rs`): `row_start[v] + s x TR + t` (the protocol clock steps `NumberShots`
    repetitions per row);
  - P6: `sched.preps_of(v)[s].start_s + t`;
  - the gradient-echo train (`ge3d.rs`): the same, since the train starts at the row's signal time
    after each preparation.
- **One helper for all three builders:** `shots::segment_poses` returns each segment's pose relative
  to the volume's (`J . P(t) . P_v^-1`) and its absolute pose (`J . P(t)`). Where `P(t)` is `P_v`
  exactly, the relative pose is `J`, with no matrix arithmetic. So a trace held still within a
  volume is the volume-sampled run bit for bit, which all three tests check with a non-identity
  rotation.
- **Deviation from the plan:** the relative pose is decomposed with `Pose::from_matrix` and resampled
  by the builders' existing `resample_by_pose`, not by `resample_slices_by_matrix`. That keeps one
  set-building code path, at rounding-level cost. The legacy path still never enters the new branch.
- `distinct_moved` and `check_segment_memory` (all volumes' sets, P6's tissue-only twins counted).
- `Protocol.segment_images_gib` (`[images] max_memory_gib`, accepted on the 3D `"shot"` path,
  default 4).
- `SeriesOutput.segment_shots` (and `Ge3dBuild.segment_shots`); the writer emits the table with
  `slices` `n/a`.
- Sidecar keys `SegmentTransform`, two `Approximations` and a `GroundTruth` entry.
- Coverage queries every segment.
- Tests: three new (`series.rs` GRASE, P6 Hadamard GRASE with leakage twins, `ge3d` train).
  Library 234, end to end 60/63.

**Task 7.** Criterion 4 runs on `asl005` with its P5 overlay (four segments, 30 partitions,
`phantom-3t-z120`), not on `p5_grase`. On the crop phantom `p5_grase` has 2 partitions, so the
spread test was impossible: the larger one always holds at least half the energy (it measured
0.52). With a 2 mm step at segment 2 of volume 1:
- only volume 1 changes (0.271 of the peak);
- the largest partition holds 0.22 of the difference energy;
- the sidecar names the approximation;
- every run validates.

The end partition's share is the resampler's edge again (spec, "Decisions deferred"). Acceptance
criteria 2-4 all pass (`work/acc-t7b`).

**Task 8** (TRXScan `eb6cfe6`).
- `ShotPlan::with_shot_poses` (`new` calls it with `None`, so the seven existing call sites are
  unchanged): `history = J . P(t)`, `geometry = J . P(t) . P_v^-1`, or exactly `J` where `P(t)`
  is `P_v`.
- `ShotPlan::shot_times`, `ShotPlan::pose` and a `center` field; `write_motion_shots`.
- CLI:
  - `--motion-sampling volume|shot` (requires `--motion`);
  - `read_motion` detects timing first: `has_time_column`, and the sidecar parsed with
    `serde_json`, now under the `cli` feature;
  - an untimed trace keeps `load_motion_tsv`;
  - a timed trace or shot sampling needs `--tr`;
  - coverage is checked before the signal stage;
  - `<out>_desc-motionShots.tsv` on every per-shot run.
- Tests: one library test (geometry · P_v = history to 1e-12; held poses give `new` bit for bit;
  the table round-trips) and one binary test (untimed lenient cells and row count, both timed
  forms, an escaped key, `--tr`, coverage, two clocks, broken JSON, `requires`).
- Gates: the re-sync baselines are identical (594, 177). The P9 set is identical on its 45 files,
  with three files added (`*_desc-motionShots.tsv`).

**Task 9** (TRXScan `dd8e6f8`).
- `Motion.times`, `Motion.sampling`, `Motion.is_timed`, `Motion.timed(...)`.
  - **Deviation from the spec:** the spec named the constructor `Motion.timed`. To keep that name,
    the property became `is_timed`.
- `from_confounds`: timed through `_core.load_motion_trace`, untimed as before.
- `poses_for(..., tr_s=, offsets=)`, `shot_poses_for`, `resample` refusing a timed trace, and
  `matrices`, `framewise_displacement` and `plot` taking `tr_s`.
- `simulate`: the per-shot path runs for `sampling="shot"` without `ShotEvents` (in their default
  order) and fills `Simulation.motion_shots`. The CLI-style and `Dataset` writers write it.
- Bindings: `load_motion_trace`, `sample_poses`, `motion_shots_tsv`, and a `shot_poses` keyword on
  both shot-motion bindings (flat `(n * n_shots, 6)`; refused without `volume_poses` or in the
  wrong shape).
- Tests: three new; pytest 68 passed.

**Task 10.**
- `docs/FEATURES.md`, `README.md` and `CLAUDE.md` describe timed traces and `--motion-sampling`,
  and say plainly that the per-shot remainder is resampled, not re-simulated.
- **Criterion 5**, scratchpad `c5_runs.sh` and `c5_check.py`, on the `trx-fig` multiband-4 grid.
  A 100 Hz nod rises between volume 2's even and odd passes, holds through the odd pass, and is
  back by 14.37 s.
  - Shot times are exact.
  - Without spin history, every volume but 2 equals the still run exactly. So do volume 2's
    slices acquired before the nod, and its slices acquired off baseline differ (0.112 of the
    peak).
  - **Spin history, isolated** (with it against without):
    - volume 2's odd slices sit 0.118 below the even ones;
    - volume 3 echoes the comb with the opposite sign (even minus odd = -0.222): its even pass,
      2.4 s after volume 2's displaced odd pass, finds that tissue under-recovered;
    - volumes 0-1 and 4-5 are unchanged exactly. A volume of 90-degree excitations at rest erases
      the history.
- **The criterion as drafted failed and was wrong.**
  - It compared against the still run, which mixes the geometry in.
  - It expected the next volume mostly recovered, which the physics does not give. The reference
    run persists the same way into its volume 38.
  - The spec's criterion 5 is revised accordingly.
- **Checked on the way:** the image change matches the written correction map. The map is in the
  signal units before the acquisition (no scale, no T2 decay), so the ratio of image change to map
  is 31-36, slice after slice.

### Codex review of the spec and plan, 2026-10-17

Read-only `codex exec`, 11 findings. Each was checked against the source; all held, and all were
applied:

| # | Severity | Finding | Checked at | Done |
|---|---|---|---|---|
| 1 | high | Coverage on `row_start` misses Hadamard's raw clock (`H` raw volumes per cycle); the untimed raw-volume claim was wrong | `schedule.rs:157-202`, `protocol.rs:2731`, `p6.rs:1187` | Query times from each series clock, checked at each series start (Task 4); claim removed |
| 2 | high | A third 3D builder in `p6.rs`, with tissue-only sets, was omitted | `p6.rs:1236-1290` | Task 6 covers all three builders |
| 3 | high | The strict loader would refuse untimed files TRXScan accepts today; the row-count rules were misdescribed | `motion.rs:218`, `compartments.rs:883`, `motion.py:186-192` | Timing detected first; untimed paths unchanged; compatibility tests |
| 4 | medium | Memory estimate borrowed from `ge3d` undercounts; the limit key is refused outside `epi3d` | `ge3d.rs:717`, `series.rs:1289`, `p6.rs:1241`, `protocol.rs:3224` | Per-builder estimates; key accepted on the new path; refusal on the new path only |
| 5 | medium | Coverage through the last shot contradicts `"volume"` sampling and the P9 equivalence test | `shots.rs:71,93-95`, `motion.rs:103` | Coverage over the sampling's query set; terminal sample in the test |
| 6 | medium | Matrix resampling would change legacy 3D bytes; `P_v . P_v^-1` need not be exactly the identity | `series.rs:1321`, `ge3d.rs:421`, `motion.rs:350,361` | Legacy builders untouched; equal poses reuse base images |
| 7 | medium | The 3D shots truth was promised but not planned | `bids.rs:1231`, `ge3d.rs:1067` | Readout-independent shots list, absolute poses (Tasks 4, 6) |
| 8 | medium | Criteria 4-5 demanded localization the motion and history models do not give | `motion.rs:103`, `TRXScan/src/motion.rs:143-163` | Traces return to baseline; localization without history, bounded recovery with it |
| 9 | medium | Python's `poses` argument conflated volume and shot poses; the utilities break on timed traces | `lib.rs:654`, `motion.rs:134-139`, `motion.py:209-233` | `volume_poses` kept, `shot_poses` added with shapes; utilities take `tr_s` |
| 10 | medium | `MotionMode` is `mrsim-acq`'s and built by TRXScan's binding; changing it breaks Task 1's additive contract | `protocol.rs:18`, `motion.rs:38`, `lib.rs:1033` | Trace kept in `MotionSpec`; readers routed through `volume_poses` |
| 11 | low | The substring check can miss an escaped `SamplingFrequency` key | `trxscan.rs:543` | `serde_json` under TRXScan's `cli`; real parsing |

### Codex review of the implementation, 2026-10-18

Read-only `codex exec` over the three diffs, 9 findings. Each was checked against the source; all held, and all were fixed:

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | medium | Python slab/chunk runs: the re-simulation rotates about the slab's centre (`compartments.rs:879`), the per-shot remainder about the full stack's | `Motion(sampling="shot")` refused on anything but the whole stack; the underlying chunked-motion centre bug predates P10 and is filed as its own task |
| 2 | medium | Shot sampling without `ShotEvents` silently dropped `Artifacts.dropout` | Refused, naming `ShotEvents(rate=...)` |
| 3 | medium | Python reference poses ignored a nonzero first-shot offset (`slice_order="protocol"`) | The shot schedule is resolved before the re-simulation; its earliest offset places the reference poses and the untimed nodes |
| 4 | medium | CLI untimed shot sampling interpolated over the raw rows while the re-simulation pads with the identity | `MotionInput.per_volume`, exactly one pose per volume, for the sampling |
| 5 | medium | 3D relative pose decomposition (`Pose::from_matrix`) is wrong at 90 degrees of pitch | `segment_poses` checks that the decomposition reproduces the matrix (1e-9) and refuses otherwise |
| 6 | medium | The shot-motion bindings indexed `volume_poses` unchecked (panic on a wrong shape) | `poses_from_rows` in both; tests call the bindings themselves |
| 7 | medium | Empty traces, NaN queries and `tr <= 0` reached a panic or a degenerate answer | `Trace::check_covers` refuses short traces and non-finite queries; `sample_poses` refuses non-finite queries; positive `--tr` and `tr_s` |
| 8 | low | The loader did not check finiteness after generating times or converting rotations | Both checked |
| 9 | low | The generator could write a one-sample or non-finite trace | `start`, sample count and times checked; a sidecar's `StartTime` too |

The review confirmed the transform order, the `P(t) == P_v -> J` shortcut, the clocks on every path, and that every timed sampling path goes through `Trace::pose_at`.

**A process slip, recorded so it is not repeated.** A scratch-to-worktree push overwrote the committed spec and plan with the scratch copy's older versions. They were restored from `03fe1cf`, and the scratch copy now carries them.

### Ordinary Codex review (final pass), 2026-10-18

Three findings, all valid, all fixed:
- **Coverage compared times exactly.** Four volumes at TR 1.1 s put the last reference at 3.3000000000000003 s against a trace ending at 3.3 s, and the run was refused. `Trace::check_covers` now allows `1e-9 x max(1, |t|)` s at each end; the sampler holds the endpoint there.
- **TRXScan's shots table at 90 degrees of pitch.** `write_motion_shots` could write angles that did not reproduce the pose (a trajectory at (20, 45, 0) degrees with a (0, 45, 0) jump). It now checks the reconstruction (1e-9) and refuses. The aslscan 2D table (P9's writer) has the same latent edge and is left as it was.
- **The first-shot-offset test could not fail.** It now records the poses `_core.signal_moving` receives and checks them against `3 v + 0.2` s.

### After the tag: defaults, 2026-10-19

Two changes on `p10-timed-motion`, after `p10-complete`, at the user's request:
- **Per-shot is the default for a timed trace.** aslscan's `within_volume_sampling`, TRXScan's `--motion-sampling` and Python's `Motion.sampling` now default to shot sampling when the trace is timed, and to volume sampling (as before) when it is not. An explicit value still wins. Runs without a timed trace are unchanged; the timed-volume acceptance and test cases now name `"volume"`.
- **The whole-stack refusal is lifted.** It was finding 1's fix above. TRXScan's slab-centre fix (the re-simulation now rotates about the full stack's centre) removed the mismatch, so per-shot sampling runs on chunked and slab Python runs; a test checks a `chunk=4` run against the whole-stack run. The dropout refusal (finding 2) stays.
