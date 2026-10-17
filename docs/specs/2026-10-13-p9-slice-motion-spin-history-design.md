# P9: per-slice poses and spin history for 2D readouts

Design spec addendum, 2026-10-13. Extends `2026-09-24-p3-motion-bs-ir-design.md` (part C, motion).
Where this document is silent, P3 stands. The user fixed three things (2026-10-13): both parts
below are in, the slice profile is a box, and multiband is included. Every other choice here was
made by Claude while writing and has not been reviewed by the user. A Codex adversarial review the
same day returned 13 findings; all were checked against the source, all held at least in part, and
this text is the result. The largest changes it forced are listed under "What the review changed".
Part C (TRXScan) and "Where it lives" were added after that review. A second Codex review on
2026-10-14, of those sections and of the plan, returned 16 findings, all valid and applied: the
largest moved TRXScan from a gain to an additive correction, separated shot events from volume
motion in its Python layer, and replaced several bit-for-bit claims with tolerances.

## Context

P3's motion model gives one rigid pose per volume, applied to finished compartment images
(`mrsim_acq::motion::apply_motion`, `motion.rs:254`), plus within-volume shot events for multiband
readouts (`apply_multiband_motion`, `motion.rs:350`). Two things follow, and a 20-pair single-band
PCASL run with a 0 to 0.5 mm drift showed both: the moved volumes are clean copies of the static
one, repositioned; and a single-band protocol cannot ask for anything else, because
`[motion.within_volume]` is refused without `MultibandAccelerationFactor > 1`
(`aslscan/src/protocol.rs:1143`).

What a scanner does differently:

- **Each slice is excited at its own time, so each has its own pose.** A head that moves during a
  volume gives a stack of slices that do not fit together.
- **Tissue that moves through the slice direction is excited at the wrong time.** Tissue carried
  into a slab that has not fired yet is excited twice in one volume and is dark the second time;
  tissue carried out of one is skipped and is bright in the next volume. This is spin history. It
  is a longitudinal effect and needs no change to the acquisition physics.

The tissue's longitudinal state is today a closed form per row and slice that assumes the previous
excitation was one repetition earlier (`longitudinal::tissue_mz`, used at `series.rs:811-828`).
Part B corrects the tissue for the cases where that assumption fails.

## Scope

In: part A (per-shot poses for every 2D readout, single-band included, any slice order, and
events read from a file), part B (spin history for the tissue of spin-echo, single-echo 2D series
whose rows share one preparation), part C (the same two in TRXScan, as far as its diffusion model
allows), their tests, ground truth and sidecar fields. Out, with reasons
under "Decisions deferred": real slice profiles and slice gaps, spin history for gradient echo,
for series whose rows differ, for blood, for 3D readouts, for Look-Locker, multi-echo and Hadamard
series and for inversion-recovery contrast, per-shot trajectory files, a moving fieldmap, and
motion of the labeling plane.

Nothing changes unless a new key is set. Both parts are opt-in, and the existing code paths stay
for every protocol that does not opt in.

## Expected effect size

So that nobody expects more than the physics gives. The numbers below hold for 5 mm slices, a
4.6 s repetition, a 90 degree excitation, no suppression pulses, grey matter at `T1 = 1.33 s`
away from the ends of the slice stack. The steady state is `1 - exp(-4.6/1.33) = 0.9685 M0`.

| Motion | Tissue signal change in the affected slice |
|---|---|
| In-plane, any size | none |
| 0.5 mm through-plane, between volumes, sequential order | about 0.01%: the tenth of the slab that changed slabs was excited one repetition earlier, give or take a slice interval |
| 0.5 mm through-plane jump between two slices excited 40 ms apart | -9.7%: a tenth of the slab was excited 40 ms earlier and holds `0.030 M0` |
| The same jump when the neighbouring slab was excited 0.4 s earlier | -7.3%: that tenth holds `0.26 M0` |
| Tissue skipped for one volume | +3.1% on the skipped fraction (`0.9990 / 0.9685`) |

Presaturation at labeling start erases what earlier volumes left, so the last row vanishes with
it. It does not touch the two jump rows: tissue excited by one slice and carried into a later one
of the same volume has had only the interval between them to recover, presaturated or not.

A slow drift therefore stays visually clean under both parts. The effect is a dip of a few percent
in one or two slices at an abrupt through-plane movement, which is several times the perfusion
difference signal.

---

# Part A: per-shot poses

## The schedule

A shot is a set of slices excited together. `aslscan` derives the shots from its normalized slice
offsets (`protocol.rs:1764-1768`: `SliceTiming` minus its minimum, reversed for
`SliceEncodingDirection: k-`, so they are indexed by data slice and start at zero): slices with
bit-identical offsets form one shot, and shots are ordered by offset. This needs a new function in
`mrsim-acq`:

```rust
pub fn slice_schedule_from_times(offsets: &[f64]) -> Vec<Vec<usize>>
```

It represents any slice order (ascending, descending, even-first, odd-first) and any grouping.
`slice_schedule` (`motion.rs:274`) stays, unchanged, for TRXScan and for the existing path.

Two refusals are lifted, on the new path below:

- `[motion.within_volume]` with a multiband factor of 1 is accepted. A shot is then one slice.
- A multiband `SliceTiming` that `slice_schedule` cannot represent (P3's `multiband_schedule`
  check) is accepted.

One check is extended. P1 requires each distinct slice time to occur exactly `mb` times, but only
when `mb > 1` (`protocol.rs:1785`). On the new path it applies at `mb = 1` as well, so a
single-band `SliceTiming` with a repeated time is refused instead of silently forming a
two-slice shot.

For every schedule `slice_schedule` does represent, the two functions return the same shots in
the same order, so the events drawn by `draw_events` are the same.

## Sampling the trajectory within a volume

New overlay key `[motion] within_volume_sampling`:

| Value | Meaning |
|---|---|
| `"volume"` (default) | every shot of a volume has the volume's pose, as in P3 |
| `"slice"` | each shot has the pose of the trajectory at its own excitation time |

Shot `s` of volume `v` is excited at `T(v, s) = row_start[v] + t_v + offset(s)`, the clock the
physiology already uses (`series.rs:995`). `t_v` is zero for an `m0scan` row
(`protocol.rs:1773`), so those rows need no special case.

With `"slice"`, the per-volume poses of `linear` and `trajectory` become nodes of a continuous
trajectory. Volume `v`'s pose is placed at its first shot, `tau_v = T(v, 0) = row_start[v] + t_v`.
The pose at any other time is the linear interpolation of the six parameters between the
neighbouring nodes, held constant after the last one. The first shot of each volume therefore has
exactly the pose P3 gives the volume.

Interpolating the three angles separately is not the geodesic between two rotations. For the
rotations of a few degrees that the motion model is for, the difference is second order, and the
written per-shot poses are the truth either way.

`"slice"` with `mode = "random"` is refused: that mode's poses are impulses that return to
baseline, and interpolating between them would invent ramps the mode does not describe.

## Events from a file

P3 draws events at random (`draw_events`, `series.rs:365`): each shot has one with probability
`dropout_rate`, and each jump is uniform in `[-amp, amp]`. There is no way to ask for one event
at a chosen shot with a chosen signed jump, which is what a spin-history test needs and what a
user reproducing a run needs.

New key `[motion.within_volume] events = "events.tsv"`, exclusive with `dropout_rate`, `severity`,
`jump_mm` and `jump_deg`. The file has the columns `desc-motionEvents_gt.tsv` is written with
(`bids.rs:1160`): `volume`, `shot`, `slices`, `attenuation`, `jump_x/y/z` in mm, `jump_rx/ry/rz` in
radians. `slices` is ignored on input (the schedule determines it) and `attenuation` is
`1 - severity`, the input of the dropout law (for `DropoutLaw::Uniform` that is also the applied
attenuation; TRXScan's b-value law applies after reading, part C). Rows are validated like the
trajectory TSV: every column present, finite numbers, `volume` and `shot` in range, `attenuation`
in `[0, 1]`, and no two rows for the same volume and shot (a random draw never produces one).

A run's own events file reproduces it when that run was on the new path. A P3 run that combined
a volume pose with jumps was resampled twice; replayed, it goes through the new path's single
resampling and differs by interpolation error.

The new keys are for 2D readouts. With a 3D readout, `within_volume_sampling = "slice"` and
`events` are refused; P5's segmented-3D within-volume events keep their path unchanged.

## The per-shot pose

The transform of shot `s` of volume `v` is the composition

```
Q(v, s) = J(v, s) . P(v, s)
```

where `P(v, s)` is the trajectory's pose at `T(v, s)` (the volume's pose under `"volume"`
sampling) and `J(v, s)` is the pose whose six parameters are the sum of the jumps of the events at
shots `<= s` of volume `v`. Both are rigid transforms about the field-of-view centre, and `J` acts
on the result of `P`. This is what P3 does in two steps: `apply_motion` moves the volume by its
pose (`series.rs:1195`), and `apply_multiband_motion` resamples the moved image by the accumulated
jump (`motion.rs:407-437`). Adding the jump's parameters to the pose's would be a different
transform whenever both rotate. Jumps persist to the end of their volume and no further, as in P3.

## Applying it

P3's two steps interpolate twice. The new path resamples once: for each shot, the head-frame
simulation-grid image is resampled by the matrix `Q(v, s)` and the shot's slices are kept. New in
`mrsim-acq`:

```rust
pub fn apply_shot_transforms(
    images: &mut [Vec<f32>], dims: [usize; 3], ngrad: usize, v2w: [[f64; 4]; 4],
    schedule: &[Vec<usize>], transforms: &[Vec<[[f64; 4]; 4]>],   // [volume][shot], world space
    attenuation: &[Vec<f32>],                                     // [volume][shot]
)
pub fn resample_slices_by_matrix(src: &[f32], dims: [usize; 3], v2w: [[f64; 4]; 4],
                                 m: [[f64; 4]; 4], slices: &[usize]) -> Vec<f32>
```

`aslscan` builds the matrices with `Pose::to_matrix` about the field-of-view centre of the whole
slice stack, so a caller resampling a slab passes the same matrix as for the full volume, and
computes each shot's attenuation from its events with `DropoutLaw::Uniform`, applying the events
of one shot in order as P3 does. The slice-restricted resampler makes the cost per slice, not
per volume.

All of this is added. `apply_motion`, `apply_multiband_motion` and `apply_multiband_motion_slab`
are not edited, so TRXScan's results and the P0 bit-identity gate are untouched. The matrix form
folds the rotation centre into the translation, so it does not evaluate `resample_by_pose`'s
arithmetic in the same order: the new path agrees with the old one to rounding, not bit for bit,
and a shot with the identity transform and attenuation 1 is left untouched exactly. Where a
volume pose and a jump combine, the new path's images differ from P3's two-step ones by
interpolation error as well. That is why the new path is never chosen for a request P3 already
serves.

## Which path runs

| Request | Path |
|---|---|
| no new key; multiband events as in P3 | P3's, bit for bit |
| a 2D readout with single-band `[motion.within_volume]`, or `within_volume_sampling = "slice"`, or `events`, or a schedule only the new function represents, or `spin_history = true` | the new one |
| a 3D readout | P3/P5's; the new keys are refused |

The decision is made in `protocol`, which every readout shares, so the regression gate needs its
3D cases (P5-P8) to show the 3D dispatch did not move.

The multi-echo and Hadamard series (`series/p6.rs:1206-1215`) carry their own copy of the P3
motion block and get the same switch. Look-Locker series are refused on the new path: they read
several times per row, and which read a shot index means is not settled there.

## Ground truth

- `desc-motion_gt.tsv` and `desc-motionEvents_gt.tsv` are unchanged.
- The sidecar's `AslscanSimulation.Motion` gains `Path: "shot"`, `WithinVolumeSampling` and
  `EventsFile` only on the new path; on P3's path it is byte-identical.
- New, on the new path: `desc-motionShots_gt.tsv`, one row per volume and shot: `volume`, `shot`,
  `slices`, `time` (s, series clock), and the six parameters of `Q(v, s)` decomposed in
  `mrsim-acq`'s convention (`Rz Ry Rx` about the field-of-view centre), rotations in radians.
- The moved `delta_m` ground truth is still moved by the per-volume pose only. P3 kept the jumps
  out of it because they describe a corrupted acquisition; the same holds for the stagger, and the
  per-shot TSV carries what is needed to do otherwise.

---

# Part B: spin history

## The idea

The geometry stays where it is. The tissue images are computed as now, from the closed form, and
moved as in part A. Spin history is added to them as a correction: the image of the difference
between the longitudinal magnetization the tissue in each excited slab really has and the
magnetization the closed form assumed for it.

Formed that way, a head that does not move gets a correction of zero by construction, the
correction is measured with the same box that does the exciting, and nothing is divided by a
signal that background suppression may have nulled.

## What it accepts

New overlay keys under `[motion]`:

| Key | Meaning |
|---|---|
| `spin_history` | `false` (default) or `true` |

(The first draft had a second key, `spin_history_z_subcells`. The engine no longer subdivides
voxels; see "The model" and the plan's implementation record.)

`spin_history = true` needs a series the closed form describes exactly when nothing moves:

- a 2D readout with spin-echo contrast (a 90 degree excitation);
- only `control` and `label` rows, all with one preparation: the same repetition time, signal
  time, suppression pulse set and presaturation (`series.rs:841-846` already computes this as
  `uniform_prep`);
- a single echo, no Look-Locker, no Hadamard encoding, no `[compat] asldro = true`;
- some `[motion]` request besides the key itself.

Anything else is refused, naming which condition failed. Multiband, background suppression,
presaturation, every part A option and `M0Type: Separate` are accepted; the separate M0 scan is
its own acquisition and gets no correction.

The uniform-preparation rule is what makes the reference exact. `row_start` advances by each
row's repetition time (`protocol.rs:2693`), so with one preparation every slab is excited exactly
one repetition after its last excitation, which is the assumption inside `tissue_mz`. When rows
differ (several delays, an `m0scan` row among them) the true interval is not the row's own
repetition time and the closed form is already an approximation with no motion at all; a
correction relative to it would then be non-zero for a static head.

## The model

**State.** Every phantom voxel with tissue (`dseg > 0`) carries its z extent, fixed in the head,
as contiguous pieces. Each piece holds `mz`, its longitudinal magnetization just after its last
excitation, and `t`, the time of that excitation. An excitation that covers part of a voxel
splits a piece at the slab's edge; adjacent pieces in the same state merge. The history is
therefore exact along z: a static head on slabs that cut the voxels keeps two pieces in each cut
voxel, each excited by its own slab. A voxel holds at most eight pieces. Beyond that, the
neighbouring pair whose merging averages least (state difference weighted by the pair's reduced
length) merges into its length-weighted mean. In practice that happens to the voxels at the ends
of the slice stack under a drift, which collect one thin sliver per volume carried out of the
stack.

**Between excitations.** `Mz(T)` for a piece is found by replaying the series from `(mz, t)` to
`T`: recovery toward `M0` with the voxel's `T1`, every suppression pulse at `row_start[v] + p_k`
in `(t, T]` multiplying by `1 - 2 epsilon`, every presaturation at `row_start[v]` in `(t, T]`
setting it to zero. This is `longitudinal::timeline` on the series clock. The stored state is
never modified by a pulse, only by an excitation. A voxel with `T1 = 0` gives zero, the guard
`tissue_mz` applies.

**The slab.** A box: slice `z` excites everything whose scanner-frame slice coordinate falls in
`[z - 1/2, z + 1/2)` voxels, over the whole plane and not only the field of view. Slabs are
contiguous and as thick as the slice spacing. A voxel's z extent under a transform is the interval
of its own z length about its moved centre. Its tilt is ignored.

**An excitation.** For shot `s` of volume `v`, at time `T = T(v, s)` and transform `Q(v, s)`, and
for each of the shot's slices `z` and each piece inside slice `z`'s slab (after splitting at the
slab's edges):

1. *Correct.* The piece adds `l V (Mz(T) - Mz_ref)` to the correction image of its voxel's tissue
   class, at the simulation-grid cell of slice `z` nearest the voxel's moved centre. `l` is the
   piece's fraction of the voxel, `V` the voxel's volume over the simulation cell's, and `Mz_ref`
   is `tissue_mz(M0, T1, tr, t_v + offset(s), suppression)`, the value the closed form gives that
   voxel at this shot's time.
2. *Excite.* The piece's state becomes `mz = 0`, `t = T`.

The correction images are multiplied by whatever per-voxel and per-shot factors the tissue images
carry (the echo-time factor, the physiological tissue factor, the shot's dropout attenuation) and
added to the moved tissue images of that shot's slices.

**Start.** Before the first volume the engine runs one fictitious volume: the first row's shots,
one repetition early, all at the pose of the first volume's first shot, exciting only. Pieces it
does not reach start fully relaxed. A head that never moves then meets every later excitation
exactly as the closed form assumes.

**What is not touched.** The blood, arterial and extravascular-label compartments are computed as
before: labeled blood arrives after the previous excitation and is treated as fresh. The
`delta_m` ground truth is unchanged.

## Where it lives

The engine is a new pure-std module in `mrsim-acq`, `spin_history`, so that TRXScan can use it
(part C). It knows nothing of ASL. Its inputs are a source grid in the head frame with a `T1`, a
weight and a class per voxel, the slab geometry, the pulses on the series clock, and a reference
`Mz` per voxel and shot. With `M0 = 1` inside and the caller's weight outside, each excitation
returns two images per class and shot slice:

```
num = sum over excited pieces of  l V weight (Mz(T) - Mz_ref)
den = sum over excited pieces of  l V weight Mz_ref
```

The weight is fixed at construction and may be replaced for one excitation, because a diffusion
signal changes from volume to volume. Both consumers add `num`: `aslscan` with `weight = M0`
(the correction below), TRXScan with each compartment's signal over its reference magnetization
(part C). `den` is for tests and diagnostics; no consumer divides by it.

## What this gets right, and what it approximates

- **A static head is exact**, on aligned slabs and on slabs that cut the voxels: every excited
  piece has `Mz(T) = Mz_ref`, so the correction is zero.
- **Double excitation and skipping under through-plane translation** are exact for a box profile,
  as long as no voxel exceeds its eight pieces.
- **Merging when full.** The averaging falls on the thin slivers the stack's end voxels collect
  under a drift, which are outside the stack and are excited again only if the motion returns.
  The first draft's equal sub-cells, averaged when partly excited, biased a static head by about
  0.6% of a slice's signal wherever slabs cut voxels (the implementation's first test showed it);
  the pieces replaced them.
- **Rotation.** Ignoring a voxel's tilt misplaces the slab boundary inside the voxels it crosses
  by up to `(phantom in-plane voxel size) x sin(angle) / 2`, about 0.01 mm per degree on a 1 mm
  phantom. Where a rotation tilts a slab across many voxels the slab's shape is followed voxel by
  voxel; only each voxel's own edge is approximate.
- **Two kernels.** The tissue is moved by trilinear interpolation of 5 mm simulation slices and
  corrected with a box overlap binned to the nearest in-plane cell. A through-plane shift of a
  tenth of a slab reaches the image as a 10% blend of the neighbouring slice plus a correction
  for the tenth of the slab that was actually excited early. The correction is right for the
  tissue it describes; the image it is added to is P3's.
- **Slice timing still travels with the anatomy** in the moved tissue image, as P3 records. The
  correction is taken against the closed form at the shot's own time, so that approximation is
  neither removed nor counted twice.

## Ground truth and sidecar

- `desc-spinHistory_gt.nii.gz`: per volume, the correction summed over the tissue classes,
  box-averaged to the acquisition grid, in the units of the tissue signal before the echo-time,
  physiological and dropout factors. Zero means no effect. It is a difference and not a ratio,
  so it is defined where suppression nulls the tissue.
- `AslscanSimulation.Motion` gains, besides part A's keys, `SpinHistory`: `{ "Model":
  "box-slab-piecewise-correction", "SliceProfile": "box", "SlabThickness": <mm>, "MaxPieces": 8,
  "Merges": <count>, "Compartments": "tissue", "Reference": "longitudinal::tissue_mz" }`.

## Cost

Measured on 2026-10-17 (plan, Task 6) on the motivating size: a 197 x 233 x 100 source with a
third of the voxels tracked, slabs 128 x 136 x 20 of 5 mm, 800 excitations of a slow drift with a
small rotation. The engine took 2.8 s under `par` (28 threads) and the whole test process peaked at
545 MB. The budget was one minute.

---

# Part C: TRXScan

Added 2026-10-13 when the user asked for TRXScan to be in the plan, and revised 2026-10-14 after a
Codex review (16 findings across this part and the plan, all valid). Claude's choices, not
reviewed by the user or by TRXScan's maintainer. TRXScan facts are from `origin/main`.

## Where TRXScan stands

TRXScan `main` (`689ad79`, release 0.3.0) takes its acquisition stage and its motion module from
`mrsim-acq`, pinned by tag (`Cargo.toml`: `tag = "trxscan-resync"`), behind a thin facade
(`src/motion.rs`). It has the limitation part A removes, and more:

- within-volume events run only with a multiband factor above 1 (the CLI's `cli.mb > 1 &&
  cli.dropout_rate > 0.0`, Python's `protocol.mb > 1 and artifacts.dropout > 0.0` in
  `simulate.py`), and a single-band request is silently ignored;
- the slice order is hard-coded interleaved (the `true` argument of both calls);
- events are drawn by a fixed rule (`dropout_events`: diffusion-weighted volumes only, severity
  0.6 to 1, a jump of 1 to 3 mm with fixed axis ratios and fixed rotations);
- the dropped-slice truth (`volume`, `shot`, `slices`, `attenuation` to three decimals) holds no
  jumps, and its attenuation is the b-scaled one, so a run cannot be replayed from it;
- spin history is listed as out of scope (`docs/FEATURES.md`), and the diffusion signal has no
  `T1` or repetition time: saturation is folded into `tissue_s0`. The Python `Protocol` records
  `RepetitionTime` and `SliceTiming` in the sidecar, and builds its own slice timing, whose
  interleaving differs from the motion schedule's.
- In Python, any `Artifacts.motion` switches the signal stage to per-volume re-simulation from
  moved streamlines, with its own refusals (synthetic fibre objects, myelin, head compartments,
  truth peaks).

TRXScan's own guidance is that the Python package is the surface that grows, not the command
line. Part C follows it: every addition is a library function and a Python binding first, and the
command line gets only the flags listed.

## C1: the dependency

TRXScan moves its `mrsim-acq` pin to the tag P9 ends with. Parts A and B only add to
`mrsim-acq`, so TRXScan's baselines (`tests/fixtures/resync_baseline`) must be identical before
any TRXScan source changes.

## C2: per-shot events

**Shot-only configuration is not motion.** A new Python dataclass, `ShotEvents`, holds what acts
within a volume: the slice order or slice times, the drawn-event rate or an explicit event list,
and (C3) spin history. It sits in `Artifacts.shot_events`, beside `Artifacts.motion` and
independent of it, so asking for shot events never selects the re-simulation signal stage. The
existing `Artifacts.dropout` keeps its meaning and its legacy call.

**Timing.** A shot's time is the volume's start plus its offset. The volume start is
`v x RepetitionTime`. The offsets are, in order of precedence: the protocol's `SliceTiming` when
`ShotEvents` asks for it; otherwise `s x RepetitionTime / n_shots` in the chosen slice order. Only
spin history uses the times in seconds; part A's geometry needs only the order.

**Transforms.** The facade builds every shot's world-space matrix once, with the field-of-view
centre of the full slice stack, and keeps global shot indices, so a slab run (`chunk=`) applies
the same transform as the full run and its empty local shots stay in place. Under the
re-simulation path the images are already at the volume pose, so the geometry applies only the
jump `J`, while spin history (C3) uses the whole `J . P`.

**Events.** The command line gains `--mb 1` support for `--dropout-rate`, `--slice-order
interleaved|sequential|descending|odd-first` (default `interleaved`, today's value), and
`--motion-events TSV`. The event file is part A's format, with one rule both simulators share:
its `attenuation` column is `1 - severity`, the input of the dropout law, so a law that scales
with the b-value applies after reading. A new truth file, `<out>_desc-motionEvents.tsv`, written
whenever events run, is that format and replays the run. The dropped-slice file keeps its file
name and columns.

Bindings: `slice_schedule_from_times`, `load_motion_events`, and
`Compartments.apply_shot_transforms` (slab-local schedule, global shot indices).

## C3: spin history

A diffusion series is the case part B accepts: spin echo, 90 degrees, every volume with the same
preparation. What TRXScan lacks is the relaxation, so C3 adds it as input: `--spin-history` with
`--t1 WM,GM,CSF` and `--tr` (seconds) on the command line, and `ShotEvents(spin_history=True)`
with a new `Protocol.t1_s` and the protocol's `RepetitionTime` in Python.

**The correction is additive, as in `aslscan`, not a gain.** A gain per output cell would average
the gains of differently saturated source tissue by geometry where the signal is what weights
them, and the diffusion signal changes the weights from volume to volume. The engine therefore
takes a weight per call (see "Where it lives"): for compartment `c` in volume `v`, the weight of a
source cell is that compartment's signal there divided by its reference magnetization
`1 - exp(-TR/T1_c)`. `num` is then the signal correction itself, and it is added to the
compartment's image at the shot's slices after the geometry. The engine runs once per
compartment, every signal-grid cell tracked.

**Refusals.** Spin history is refused without `T1`s or a repetition time; without events that
jump or a motion trace; with `Protocol.head` (five compartments, three `T1`s); and with any run
that does not simulate the whole slice stack: `chunk=`, `slices=`, `z_mm=`, or an object that is
itself a slab (`z_offset` or `nz_full` set).

**Ground truth.** `desc-spinHistory`: per volume, the correction summed over the compartments,
block-averaged from the signal grid to the acquired grid, in signal units; zero means no effect.
`Simulation.spin_history` holds it in Python, and `Dataset` writes it under
`derivatives/trxscan`. The command line writes `<out>_desc-spinHistory.nii.gz`.

A source voxel there is as thick as a slice. The pieces keep the box overlap exact along z
regardless, but each piece has one in-plane position (its voxel's), so the correction's in-plane
detail is the signal grid's.

## Landing

TRXScan changes go to its `main` by pull request for the maintainer's review, as the re-sync
did. `docs/FEATURES.md` is updated to say what is now modeled and what is still not (slice
profiles, cross-talk without motion, the blood).

---

# Testing and verification

- **mrsim-acq** — `slice_schedule_from_times` equals `slice_schedule` on every schedule that one
  represents (a grid of `nz`, `mb`, both orders), and returns descending and odd-first orders
  correctly. `apply_shot_transforms` with identity volume poses equals `apply_multiband_motion`
  on the same events to `1e-6` of the peak, and leaves untouched shots bit for bit.
  `resample_slices_by_matrix` equals the same slices of `resample_by_pose` to `1e-6` of the peak,
  and a rotating jump resampled on a slab with the full stack's matrix equals the full volume's
  slices away from the slab's edges. The golden record and the bit-pinned tests pass unchanged.
- **Part A** — single-band events draw one slice per shot; a single-band `SliceTiming` with a
  repeated time is refused; with `"slice"` sampling the first shot of each volume has the
  volume's pose and the last volume's shots all have its pose; a shot transform with a rotating
  volume pose and a rotating jump equals the product of the two matrices and not the pose with
  summed parameters; an integer-voxel in-plane step between two shots shifts exactly the later
  shots' slices; a new-path run's `desc-motionEvents_gt.tsv` fed back as `events` reproduces its
  images to `1e-6` of the peak (the file holds rotations in radians and attenuations as `f32`, so
  the round trip is not exact); `desc-motionShots_gt.tsv` round-trips; the same cases run through
  the multi-echo and the Hadamard series; `"slice"` with `random`, a Look-Locker series on the new
  path, the new keys with a 3D readout, and duplicate events in a file are refused; the sidecar
  of a P3-path run is byte-identical.
- **Part B, engine** — a static head gives a correction below `1e-6` of the tissue signal, on
  aligned slabs and on slabs that cut the voxels, with and without presaturation. In-plane motion
  of any size gives the same. In uniform tissue, a through-plane jump of a fraction `f` of the
  slab toward the slab excited next scales that slab's tissue (reference plus correction, summed
  over the slab) by `1 - f (1 - (1 - exp(-dt/T1)) / R)`, `R` the reference: 0.9031 for `f = 0.1`,
  `dt = 40 ms` and `R = 1 - exp(-TR/T1)`, and the same form against the presaturated reference.
  The slab after it, at the same pose, holds a tenth last excited a repetition plus `dt` ago. The
  opposite jump leaves the skipped tenth at `0.9990 M0` in the next volume, and at the static value
  when that volume is presaturated. A slow through-plane drift gives a correction that varies
  smoothly, with no merges. Each refusal names its condition.
- **Part C** — after the pin moves and before any TRXScan source changes, the re-sync baselines
  are identical. With `--mb > 1`, `--slice-order interleaved` and no new flag, outputs are
  identical to the pinned release, in the CLI and in Python. `--mb 1 --dropout-rate` drops single
  slices and writes them to the dropped-slice file. A run's `desc-motionEvents.tsv` fed back
  through `--motion-events` reproduces it to `1e-6` of the peak. `ShotEvents` without
  `Artifacts.motion` keeps the mixture signal stage. A chunked run with a rotating jump equals
  the unchunked one away from slab edges. Each spin-history refusal names what is missing or
  unsupported. In a uniform-tissue object, sequential order, the first event volume and an
  interior slab, one through-plane jump of a quarter slice gives that shot's slab a relative
  correction equal to part B's closed form within `5e-3`, and none in the volume's other slices.
  Two runs differing only in their b-values give the same relative correction.
- **Regression** — with no new key, every existing test and `regress_identity.sh` pass unchanged.

## P9 acceptance criteria

1. The motivating protocol (single-band, 20 pairs, 20 slices, sequential) runs with
   `[motion.within_volume]` and writes one slice per event in `desc-motionEvents_gt.tsv`.
2. The same protocol with `mode = "linear"`, `trans_mm = [0, 0, 0.5]`,
   `within_volume_sampling = "slice"` and `spin_history = true` writes a correction below `1e-3`
   of the local tissue signal in every slice but the two at the ends of the stack: the drift is
   too slow to matter, which is the expected answer.
3. Adding an events file with one event at a mid-stack shot, attenuation 1 and a 0.5 mm
   through-plane jump in the direction the slices are acquired, gives in grey matter of that
   shot's slab a tissue signal 0.903 of the static one within `5e-3` (the tenth of it that the
   previous shot excited), and no correction in any other slice of the volume.
4. The same run with presaturation at labeling start shows the same dip in that volume and a
   correction below `1e-3` in the volume after it.
5. With no new key, outputs are bit-identical to the current ones; the output of every run
   validates with no errors.
6. TRXScan on the P9 `mrsim-acq` reproduces its re-sync baselines exactly, and a single-band
   TRXScan run with `--dropout-rate` writes dropped single slices.
7. A TRXScan run with spin history, `T1`s, a repetition time, sequential order and an events file
   holding one through-plane jump shows a negative correction in that shot's slab and none in the
   volume's other slices; repeated with every b-value changed, the correction relative to the
   signal is the same.

---

# What the review changed

- **The read.** The first draft evaluated whole phantom voxels at the shot time and let the
  resampler weight them. A voxel cut by a slab boundary would then have mixed tissue excited
  40 ms earlier into the next slice, halving that voxel's signal with no motion at all, and
  trilinear blending would have done the same across slices for any static offset. The
  correction formulation replaced it.
- **Scope.** Gradient echo (whose start state and row-to-row propagation need the existing
  `ge_propagated` path as the reference) and series with differing rows or `m0scan` rows (where
  the series clock and the closed form disagree without motion) were in the first draft and are
  now refused.
- **Presaturation** was credited with removing all spin history. It removes only what earlier
  volumes left.
- **Composition.** The first draft added jump parameters to pose parameters. P3 composes the two
  transforms, and so does this.
- **Events from a file** were added because the acceptance test could not be requested without
  them.
- **The cost fallback** to the closed form for volumes whose poses repeat was removed: history
  outlives the motion that made it.

# Decisions deferred

- **Real slice profiles and slice gaps.** A box as thick as the slice spacing overstates the
  overlap of a profile with a gap and understates the cross-talk of one with tails, which exists
  even without motion. Needs a profile and a `SliceThickness` separate from the spacing.
- **Gradient echo.** A flip angle other than 90 degrees keeps history over several repetitions.
  The engine's state handles it (`mz = (1 - w) Mz + w cos(a) Mz`), but the start state must be
  the preparation's fixed point and the reference must be `tissue_mz_ge` or, for differing rows,
  the `ge_propagated` sequence.
- **Series whose rows differ**, including multi-delay series and included `m0scan` rows. Needs a
  reference that follows the series clock, which would also change the static signal.
- **Spin history for blood.** Label that was in a slab at an earlier excitation and is still
  there has been saturated. Needs the bolus position per slice, which the per-parcel model has
  only in part.
- **Per-shot trajectory files**, one row per volume and shot, for real slice-resolution motion
  traces. The per-shot transform table is already the internal form; only the input is missing.
- **Look-Locker, multi-echo and Hadamard** under spin history, and Look-Locker on the part A
  path.
- **Poses that persist across volumes after a jump.** Jumps reset at the volume boundary, as in
  P3, so a jump is followed by a return that is itself a movement and leaves its own history.
- **Through-plane motion of the per-voxel maps and the fieldmap**, as in P3.

# Risks

**No oracle.** simasl has no spin-history model and neither does TRXScan, whose feature notes
list it as out of scope (`TRXScan/docs/FEATURES.md:42`). Part B is tested against closed forms
only. They cover the two mechanisms (double excitation, skipping) and the static limit in uniform
tissue under translation, and not rotation, heterogeneous tissue, or the ends of the stack.

**Two kernels for one shift.** The correction is exact for the tissue it describes and is added
to an image whose geometry is trilinear. A user measuring the dip in the final image will find it
close to the closed form and not equal. The correction map is written separately for that reason.

**Cost.** The estimate above is arithmetic, not a measurement.

**Narrow acceptance.** Single-delay control-label series with spin echo are the common 2D case
and the motivating one, but multi-delay and gradient-echo users will meet a refusal. The refusal
is the honest answer until the reference for those series is settled.
