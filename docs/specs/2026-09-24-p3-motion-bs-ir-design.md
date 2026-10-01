# P3: background suppression, inversion-recovery contrast, and ASL motion

Design spec addendum, 2026-09-24. Extends `2026-09-21-mrsim-acq-aslscan-design.md` (the P0/P1
spec, "the main spec" below). Where this document is silent, the main spec's contracts stand.

## Context

P1 delivered `aslscan` at tag `p1-complete`: BIDS protocol and phantom in, GKM and spin-echo
signal per phantom voxel, one `mrsim-acq` call per series, validator-clean BIDS out. Three things
the main spec deferred to P3 are what stand between it and any real dataset:

- **Background suppression.** Every one of the five `bids-examples` ASL datasets carries
  `BackgroundSuppression: true`, and P1 rejects all of them, correctly, because simulating without
  it would write a sidecar the data contradicts.
- **Inversion-recovery contrast.** simasl's third signal equation, deferred because it is an
  inversion preparation before an otherwise unchanged spin-echo readout, the same machinery as
  background suppression.
- **Motion.** simasl applies one rigid pose per volume by resampling the finished image;
  `mrsim-acq` carries TRXScan's per-volume and within-volume (multiband) motion model, unused by
  `aslscan` so far.

The main spec grouped these because they share one fact: each changes what the tissue and blood
compartments hold *before* the acquisition stage, and none changes the acquisition stage itself.
That is true here without exception: **P3 makes no change to `mrsim-acq`.**

The oracle situation differs from P1, and it is worth being exact about it. simasl v2.2.0 has
**no** background-suppression model (ASLDRO gained one in a later release), so part A is derived
from the Bloch longitudinal equation and tested against closed forms, not fixtures. Part B has
simasl's `mri_signal_filter.py:252-283` as its oracle. Part C has simasl's motion path as a
convention reference, with the numerical comparison itself belonging to P2.

## Scope

In: parts A, B, C below, their tests, and the BIDS fields they consume. Out, with reasons under
"Decisions deferred": pulse history along the delivery of the bolus (pulses are treated as
inverting the whole labeled bolus wherever it is), partial-bolus inversion by a pulse during
labeling, pulses between the slices of one readout, a label-inverting IR preparation, IR and
background suppression in one protocol, motion of the labeling plane relative to the vessels,
kinetics re-evaluated after through-plane motion, and any change to the acquisition physics.

---

# Part A: background suppression

## What is modeled

Background suppression is a train of inversion pulses applied during the post-labeling delay,
timed so that the static tissue's longitudinal magnetization is near zero at the readout while
the labeled blood, which is inverted by each pulse like everything else, keeps most of its
difference signal. It changes the longitudinal history of both compartments and nothing else: no
transverse effect, no readout effect. It is therefore a pre-acquisition scaling of the compartment
images, per voxel and per slice for tissue and per row for blood, and the acquisition stage is
untouched.

## Inputs

From the sidecar, all required when `BackgroundSuppression` is true:

| Field | Meaning |
|---|---|
| `BackgroundSuppressionNumberPulses` | `N`, the number of pulses after labeling start |
| `BackgroundSuppressionPulseTime` | `N` times in seconds from the **start of labeling** |

BIDS defines the times from labeling start, which is also the kinetic model's clock, so no
conversion is needed. A length mismatch between the two fields is an error naming both numbers.
BIDS also says that for multi-PLD series with differing pulse times "only the pulse time of the
first PLD should be defined"; `aslscan` then applies the first PLD's times to every row and
records that it did, and the overlay can supply `pulse_times_per_pld`, one array per distinct
PLD in ascending PLD order, to override.

From the overlay, `[background_suppression]`:

| Key | Default | Meaning |
|---|---|---|
| `inversion_efficiency` | 0.95 | `epsilon`, the fraction of longitudinal magnetization each pulse inverts |
| `presaturation` | false | A saturation pulse on the imaging region at labeling start |
| `pulse_times_per_pld` | absent | Per-PLD pulse times for multi-PLD series, see above |

Every applied default is recorded in the output sidecar, as in P1. `m0scan` rows never see the
pulses or the presaturation: a scanner acquires M0 without the preparation, an included M0 row
has no labeling clock to time pulses against (`rows.rs` gives it `t = tau = 0`), and a separate
M0 has no row at all. Both keep P1's `tissue_se`.

## Timing constraints

Per row, `t = 0` at labeling start and `t_read(z)` is the row's kinetic signal time plus the
slice's offset (main spec, protocol contract): `PLD + tau` for (P)CASL and `PLD` for PASL, plus
`SliceTiming[z] - min(SliceTiming)`. Three constraints make the independent-row model of the main
spec hold with pulses in it, and all are checked in `protocol`:

- **Every pulse precedes the first slice's excitation**: `p_k < min_z t_read(z)`. A pulse between
  two slices' excitations would act on the later slice's current image and on the earlier
  slice's *next* repetition, whose starting value below assumes uninterrupted recovery; modeling
  that needs longitudinal history carried across repetitions, which P3 does not do. The error
  names the pulse and the first readout time. All five `bids-examples` datasets satisfy this
  (`asl002`: last pulse 3.276 s, first readout 3.8 s; `asl004`: 1.604 s against 1.65 s).
- **Every readout fits in the repetition**: `t_read(z) <= TR`. Otherwise the recovery interval
  below is negative. This holds for P1 protocols too and is added as a general check.
- **Every pulse follows the bolus creation**: `p_k >= tau` for (P)CASL and
  `p_k >= BolusCutOffDelayTime[0]` for PASL. A pulse during labeling inverts only the part of the
  bolus already labeled (the fraction `p_k / tau` for (P)CASL); the rule is simple but it is
  deferred to P4 with the rest of the bolus-position modeling, and the error names the pulse and
  the bolus end.

Under the first constraint the pulse set is the same for every slice of a row; only `t_read(z)`
differs between slices.

## The tissue timeline

Two effects act on the tissue's longitudinal magnetization `Mz(t)`: free recovery toward `M0`
with the tissue T1, and the pulses.

```
recovery:   Mz(t) = M0 - (M0 - Mz(t_k)) * exp(-(t - t_k) / T1)     between events
pulse k:    Mz+ = (1 - 2 epsilon) * Mz-                              at t = p_k
readout:    tissue compartment = Mz(t_read(z))                       signed
```

The starting value is what the previous excitation left. The spin-echo readout saturates the
slab (90 degrees), so at labeling start the tissue has recovered for `TR - t_read(z)`:

```
Mz(0) = M0 * (1 - exp(-(TR - t_read(z)) / T1))     presaturation false
Mz(0) = 0                                          presaturation true
```

With `N = 0` and no presaturation this is exactly the P1 steady state,
`Mz(t_read) = M0 (1 - exp(-(TR - t_read)/T1) exp(-t_read/T1)) = M0 (1 - exp(-TR/T1))`, in exact
arithmetic. It is **not** bit-identical, because the two exponentials are evaluated separately.
`mrsignal::tissue_se` therefore stays as it is for the `N = 0`, no-presaturation case, and the
timeline is used only when there is at least one event. The P1 fixture test keeps passing
unchanged, and the timeline's `N = 0` result is asserted to agree with `tissue_se` to `1e-12`
relative in a test of its own.

The sign is kept. A tissue whose `Mz` is negative at readout produces transverse magnetization of
opposite phase, which the complex forward model represents as a negative real compartment value,
exactly as the blood compartment already does for `label` rows. The magnitude image loses the
sign; the phase image and the linearity identity keep it.

**Worked example.** `bids-examples` `asl002`: PCASL, `LabelingDuration` 1.8, `PostLabelingDelay`
2.0, `RepetitionTimePreparation` 4.5717, pulses at `[2.05, 3.276]`, `epsilon = 1`, **first slice**
(`t_read = 3.8`). For GM (T1 1.33 s): `Mz(0) = 0.440 M0`, `0.880` before the first pulse, `-0.880`
after, `0.252` before the second, `-0.252` after, `0.156 M0` at readout, against `0.968 M0`
unsuppressed: 84% suppression. WM (T1 0.83 s): `0.175` against `0.996`, 82%. CSF (T1 3.0 s):
`0.219` against `0.782`, 72%. The suppression is slice-dependent, because later slices recover
for longer after the last pulse: `asl002`'s last slice (`SliceTiming` 0.7315, `t_read = 4.5315`)
gives GM `0.499` against `0.968` (48%) and WM `0.656` against `0.996` (34%). A real scanner's
pulse timing is optimized for the middle of the readout; the first-slice numbers are what the
closed-form test below asserts, and the last-slice numbers are asserted alongside them so that
the slice dependence is pinned down and not just tolerated.

## The blood compartment

The GKM's `delta_m` is the control-minus-label difference of longitudinal magnetizations
(positive; `series` applies the row sign, `-1` for `label`), and inversion is linear, so a pulse
that inverts the labeled blood multiplies the difference by `(1 - 2 epsilon)`; the recovery
toward `M0` is the same in control and label and cancels, and the T1 decay continues unchanged.
Hence

```
delta_m_bs = delta_m_gkm * prod over all pulses of (1 - 2 epsilon)
```

For perfect pulses that is `(-1)^N`: an even count restores the sign, an odd count flips it, and
`control - label` then recovers `-delta_m`. That is physics, not a bug, and `aslscan` writes the
factor to the sidecar as `AslscanSimulation.BackgroundSuppressionLabelFactor` so a quantification
step can see it. The row-semantics table of the main spec is otherwise unchanged: `label` rows
carry `-delta_m_bs`, `deltam` rows `+delta_m_bs`.

**This is an approximation, and a large one, and it is named as such in the sidecar
(`AslscanSimulation.BackgroundSuppressionModel: "global-bolus"`).** It treats every pulse as
inverting the whole labeled bolus wherever the bolus is. Real pulses cover the imaging region
(sometimes wider), and blood still in transit below it is not inverted, so label that arrives
after a pulse has seen fewer inversions than label that arrived before it. The GKM keeps
delivering label until `ATT + tau` (`kinetic.rs`, the `dt < t <= dt + tau` branch), which for the
phantom's GM (`ATT` 0.8 s, `tau` 1.8 s) is 2.6 s, after `asl002`'s first pulse at 2.05 s. Under a
slab-confined model with two perfect pulses, the label delivered before 2.05 s is inverted twice
and the rest once, and weighting the delivery kernel (blood decay at the phantom's `T1b` 1.65 s
until arrival, tissue decay at 1.33 s after it) by the pulse history gives roughly `0.33` times
the unsuppressed difference instead of `+1` (corrected in the P4 addendum: `asl002` is PCASL, and
with PCASL's parcel weighting the value is `0.0821`; `0.33` came from weighting the parcels as
PASL ones). The global-bolus factor is therefore an upper bound
on the retained label, not an estimate of it. Doing better needs the pulse history along the
delivery, which is the bolus-position modeling P4 introduces with the vascular compartment; it
is the first item under "Decisions deferred", and the sidecar key is there so that P4's model
can be told apart from this one in any dataset either produces.

## Where it lives

A new `aslscan` module, `longitudinal`, owns the timeline:

```rust
pub struct Suppression { pub pulse_times: Vec<f64>, pub epsilon: f64, pub presaturation: bool }

/// Tissue Mz at `t_read` (s), signed, from the recovery-and-pulse timeline above.
pub fn tissue_mz(m0: f64, t1: f64, tr: f64, t_read: f64, s: &Suppression) -> f64
/// The blood factor: prod (1 - 2 epsilon) over the pulses.
pub fn label_factor(s: &Suppression) -> f64
```

`series` calls `tissue_mz` per phantom voxel and per acquired slice. P1's tissue-signal cache is
keyed on the row's `tr` alone; with suppression on it is keyed on the complete resolved
preparation, `(tr, t_read(z), pulse times, epsilon, presaturation)` as bit patterns, because
`t_read` differs per slice and per PLD (`asl004` has one TR and six PLDs) and the pulse set
differs per PLD under `pulse_times_per_pld`. `mrsignal::tissue_se` is untouched.

---

# Part B: inversion-recovery contrast

## The equation

simasl's IR branch (`mri_signal_filter.py:252-283`), with its transverse factor left to the
acquisition stage as in P1:

```
S_tissue = sin(fa) * M0 * (1 - (1 - cos(fa_inv)) exp(-TI/T1) - cos(fa_inv) exp(-TR/T1))
                        / (1 - cos(fa) cos(fa_inv) exp(-TR/T1))
S_blood  = sin(fa) * delta_m
```

with every division guarded to zero as simasl guards it (`np.divide(where=denominator != 0)`)
and the zero-T1 exponent rule of P1 (`exp(0) = 1`). This is a periodic steady state, not a
timeline: it assumes the same inversion-excitation cycle every TR, which is what simasl assumes.

`mrsignal` gains `Contrast::InversionRecovery` and

```rust
pub struct IrParams {
    pub inversion_time: f64,
    pub excitation_flip_deg: f64,
    pub inversion_flip_deg: f64,
}
pub fn tissue_ir(m0: f64, t1: f64, tr: f64, p: &IrParams) -> f64
pub fn blood_ir(delta_m: f64, p: &IrParams) -> f64
```

## Inputs

Two of the three parameters have standard BIDS fields, `InversionTime` (s) and `FlipAngle`
(degrees), and P1's sidecars already carry `FlipAngle` (90 in every 2D `bids-examples` dataset).
They follow the main spec's precedence, overlay > sidecar > default, with the source recorded;
the effective value is written to the standard key and a replaced input is kept under
`AslscanSimulation.InputValuesReplaced`, as the main spec requires of every standard key. The
inversion pulse angle has no BIDS field and lives in the overlay and the simulator block.

| Key (overlay `[signal]`) | Sidecar field | Default | Range |
|---|---|---|---|
| `acq_contrast` | none | `"se"` | `"se"` or `"ir"`; `"ge"` still rejected naming P5 |
| `inversion_time` | `InversionTime` | 1.0 s | `>= 0` |
| `excitation_flip_angle` | `FlipAngle` | 90 deg | -180..180 |
| `inversion_flip_angle` | none | 180 deg | -180..180 |

Defaults and ranges are simasl's (`user_parameter_input.py:156-164`); BIDS constrains `FlipAngle`
to `0..360`, and a simasl-legal negative angle is written as its positive equivalent with the
signed value kept in the simulator block. For `"se"` the excitation angle is not a free input:
the spin-echo equation assumes 90 degrees (main spec, signal model), so a sidecar or overlay
`FlipAngle` other than 90 with `"se"` is an error naming the assumption, where P1 silently copied
the field through. `InversionTime` present in a sidecar with `"se"` is likewise an error, since
it would be echoed while describing a preparation that was not simulated.

## What the blood term does not do

simasl adds `mag_enc` after the tissue steady state and applies only `sin(fa)` to it, so a
nonselective inversion pulse at `TI` does not invert the label in simasl's model. Faithfulness to
the oracle wins here, as it did for the spin-echo blood term in P1: `blood_ir` is `sin(fa) *
delta_m`. A label-inverting IR preparation is physically the same operator as one background
suppression pulse at `t_read - TI` and could be built from part A; it is listed under deferred
decisions, because doing it would make the IR fixtures disagree with simasl by construction.

## Interaction with background suppression

A protocol asking for `"ir"` with `BackgroundSuppression: true` is rejected in P3. The IR
equation is a steady state and the suppression model is a timeline; composing them means
choosing which one owns the inversion history, and that choice is not in the oracle. The error
names both.

---

# Part C: motion

## The model

One rigid pose per volume, applied to every compartment's simulation-grid image for that volume
before the acquisition call. The head moves; the scanner, the fieldmap, and the readout stay
fixed. This is `mrsim_acq::motion::apply_motion` (`motion.rs:254`), which is also TRXScan's
"fast" motion path, and it is what simasl does: resample the finished signal image by a
per-volume rigid transform and keep the field of view (`transform_resample_image_filter.py`,
`examples.py:364-380`).

Within-volume motion, where the head moves between the shots of a multiband readout, is
`mrsim_acq::motion::apply_multiband_motion` with `DropoutLaw::Uniform` (P0 change 7; the
b-value law has no meaning here). It needs `MultibandAccelerationFactor > 1` and a `SliceTiming`
that `motion::slice_schedule` can represent: simultaneously excited slices spaced `nz / mb`
apart, shots in sequential or even-then-odd (interleaved) order. P1's check that each distinct
slice time occurs `mb` times is necessary but not sufficient (`[0, 0, 1, 1]` with `mb = 2` passes
it and the schedule would group slices `{0, 2}` and `{1, 3}`), so `protocol` additionally derives
the shot grouping and order from the slice offsets, determines the `interleaved` argument from
them, and rejects any schedule the function cannot represent, naming the first mismatched slice.

## Conventions

The pose convention is `mrsim-acq`'s: `p' = R (p - c) + c + t`, `R = Rz Ry Rx` in degrees,
translation in mm, `c` the field-of-view centre (`fov_center`, `motion.rs:200`, hard-coded at
`motion.rs:231`). The centre is not a parameter and `mrsim-acq` is not changed, because a
rotation about any other origin `o` is the same rigid transform as the rotation about `c` with
the translation `t' = t + (R - I)(c - o)`; converting a pose is exact algebra, and it is P2's job
when comparing against simasl.

simasl has two rotation conventions, and P2 needs both. `AffineMatrixFilter` composes `Rz Ry Rx`
about `rotation_origin` (`affine_matrix_filter.py:165-175`), but the image-motion path
`TransformResampleImageFilter` calls `transform_resample_affine`, which composes **`Rx Ry Rz`**
(`utils/resampling.py:82-84`), about `rotation_origin` with a world-origin default. Neither a
centre change nor a per-axis sign fixes the order difference for a mixed-axis rotation; P2
converts by forming simasl's rotation matrix and decomposing it into `Rz Ry Rx` angles, and its
convention test must use a mixed-axis rotation, since single-axis ones cannot expose the
mismatch.

## Inputs

Overlay `[motion]`:

| Key | Meaning |
|---|---|
| `mode` | `"off"` (default), `"trajectory"`, `"random"`, `"linear"` |
| `trajectory` | Path to a TSV in the format `mrsim_acq::motion::load_motion_tsv` reads: one row per volume, `trans_x/y/z` in mm, `rot_x/y/z` in **radians** (the confounds convention; `motion.rs:149-172` converts to degrees) |
| `trans_mm`, `rot_deg` | Per-axis amplitudes for `random` (impulses that return to baseline) and `linear` (monotonic drift), as `MotionMode` defines them, in mm and degrees |
| `volumes` | The volumes `random`/`linear` affect; default all |
| `within_volume` | `{ dropout_rate, severity, jump_mm, jump_deg }` for multiband shot events; absent means none |

The series seed drives `random` and the within-volume events, salted differently from the
acquisition noise (`seed ^ 0x4D4F_5449_4F4E`, "MOTION") so that turning motion on does not change
the noise realization, and both salts are written to the sidecar.

## Where in the pipeline

`series` assembles the 4D compartment images exactly as in P1, then applies the poses, then
makes the one acquisition call. Motion acts after the class decomposition and after resampling to
the simulation grid, so a moved boundary voxel is a trilinear blend of neighbouring simulation
cells of the *same* compartment; relaxation properties travel with their compartment in `class`
mode. Two approximations follow from resampling finished images, and both are recorded in the
sidecar:

- **Slice timing travels with the anatomy.** P1 evaluates each simulation slab's kinetics and
  (with part A) its tissue timeline at the acquired slice's readout time. A through-plane
  translation of one slice hands destination slice `z` the values computed for the source slice's
  time, not `t_read(z)`. The error is the change in `delta_m` and `Mz` across one slice interval
  (tens of milliseconds, against PLDs of seconds), in both `class` and `voxel` mode. Evaluating
  the moved anatomy at the destination time would mean moving the phantom before the kinetics,
  a different pipeline; it is deferred, and the exact tests below use in-plane motion only.
- **Per-voxel maps do not move.** In `voxel` mode the T2 and T2' maps are fixed maps in the
  acquisition call and stay in scanner space while the compartments move. `voxel` mode with
  motion is accepted, recorded as an approximation, and the class-vs-voxel cross-check of P1 is
  extended with a moved case whose discrepancy is tracked, not asserted.

The fieldmap does not move either. A real head moving in a real B0 field changes the field, which
is a much larger change (a susceptibility model) and is out of scope; the sidecar says the
fieldmap was held in scanner space.

## Ground truth

The poses are written as `sub-XX_desc-motion_gt.tsv` (volume, six parameters, and the
within-volume events with their shots, jumps and attenuations). The moved `delta_m` ground truth
is **not** the P1 acquisition-grid ground truth resampled by the pose: motion interpolation and
box averaging do not commute, so that would disagree with the data at every boundary voxel.
Instead `series` forms the unsigned `delta_m` image on the simulation grid (the sum over the
blood compartments of what it already computes for the data path), moves it by the same
per-volume pose, and box-averages it to the acquisition grid exactly as the data are. It carries
neither the suppression factor (which is in the sidecar and would only obscure the answer key)
nor the within-volume jumps and dropout (which are in the TSV, and which describe a corrupted
acquisition rather than a different truth). The unmoved `delta_m` is written too, as
`desc-deltamStatic`, because both are useful and they are cheap.

---

# Testing and verification

New code, test-first, pure std where the P1 modules are.

- **longitudinal** — closed forms: `N = 0` equals `tissue_se` to `1e-12`; one perfect pulse an
  instant before readout gives `-Mz` of the unsuppressed value; two coincident perfect pulses
  cancel; `epsilon = 0` is the identity; presaturation zeroes `Mz(0)`; the `asl002` worked example
  reproduces GM `0.156 M0` and WM `0.175 M0` for the first slice and GM `0.499 M0` and WM
  `0.656 M0` for the last, all to `1e-3`; `label_factor` is `(-1)^N` at `epsilon = 1` and `0.81`
  for two pulses at `0.95`. In `protocol`: a pulse at or after the first readout, a readout after
  `TR`, and a pulse before the bolus end are each rejected naming the offending values.
- **mrsignal (IR)** — fixtures from simasl's IR branch with `exp(-TE/T2)` divided out, generated
  by extending `tools/gen_mrsignal_fixtures.py` with `acq_contrast = "ir"` cases over a grid of
  `TI`, `fa`, `fa_inv` including the zero-T1 guard; `1e-12` relative. A closed form: `fa_inv = 0`
  gives `sin(fa) M0 (1 - E) / (1 - cos(fa) E)` with `E = exp(-TR/T1)`, which equals P1's
  saturation recovery `M0 (1 - E)` only at `fa = 90`. `"ir"` with `BackgroundSuppression: true`,
  `"se"` with `FlipAngle != 90`, and `"se"` with `InversionTime` present are rejected naming the
  reason.
- **motion** — at the resampler boundary, an integer-voxel in-plane translation on the simulation
  grid shifts each compartment image by exactly that many voxels (interior bit-identical, since
  trilinear weights are 0 and 1), and the moved simulation-grid ground truth with it; through the
  acquisition, with `oversample 1`, no distortion, no noise, no partial Fourier and no
  acceleration, the acquired magnitude shifts by the same amount within a tolerance that the
  Fourier round trip sets (`1e-5` relative to the peak); a small rotation followed by its inverse
  returns a smooth field to itself within `1e-3` relative (trilinear resampling is not
  mass-conserving, so no conservation is asserted); the motion TSV round-trips with rotations
  written in radians; a `SliceTiming` the schedule cannot represent is rejected.
- **series** — the linearity identity of the main spec still holds with suppression on (it is a
  per-compartment scaling before the acquisition, so `I_C - I_L = I_B` on the complex images is
  unaffected), and with motion on when the three runs share the poses and events; the sidecar
  records every new default, both salts, the label factor, and the model name.
- **protocol** — `BackgroundSuppression: true` now parses, requiring the two pulse fields; the
  two 2D datasets among the five `bids-examples` ones (`asl002`, `asl004`) parse end to end, and
  the three 3D ones are still rejected naming P5.

## P3 acceptance criteria

1. `aslscan` simulates `bids-examples` `asl002` (PCASL, 2D EPI, two background-suppression
   pulses, `M0Type: Separate`) from its real sidecar, with an overlay supplying only the M0
   repetition time, and the output validates with no errors. The sidecar records
   `BackgroundSuppressionLabelFactor` `0.81` (two pulses at the default `epsilon = 0.95`) and the
   model name.
2. A second `asl002` run with the overlay also setting `inversion_efficiency = 1.0`, compared
   with a third run with suppression switched off in the overlay: the first slice's GM and WM
   tissue signal is suppressed by more than 80%, and the complex `control - label` difference
   changes by less than `1e-4` of its peak between the two runs (the label factor is `+1`, and
   the difference must not care about the tissue's suppression; the difference is about 1% of
   the tissue signal, so the float32 storage of the images puts the floor near `1e-5`).
   Magnitude subtraction does not inherit the linearity identity, so the comparison is on the
   complex images.
3. An `"ir"` protocol simulates against the simasl-derived fixtures at `1e-12`.
4. A `trajectory` motion run reproduces an integer-voxel in-plane translation exactly at the
   resampler and within tolerance through the acquisition, and a `random` motion run is
   reproducible from its seed and its written `desc-motion_gt.tsv`.
5. `mrsim-acq` is unchanged, so the P0 bit-identity gate does not need re-running; `cargo test`
   passes in the pure-std default build of both crates.

---

# Decisions deferred

- **Pulse history along the delivery.** The global-bolus approximation above overstates the
  retained label by a large factor for pulses that fall inside the delivery window. The fix needs
  the bolus's position at each pulse, which is what P4's vascular compartment introduces; until
  then the sidecar model name marks every dataset produced under the approximation.
- **Partial-bolus inversion** by a pulse during labeling: the `(p_k / tau)` fraction rule for
  (P)CASL and the pre-cutoff case for PASL. P4, together with the item above.
- **Pulses between slice excitations**, which need longitudinal history carried across
  repetitions. Not scheduled; no `bids-examples` dataset needs it.
- **A label-inverting IR preparation**, and IR composed with background suppression: after P2 has
  established the compat baseline, so that the deliberate departure from simasl is measured.
- **Motion of the labeling plane** relative to the feeding arteries, which modulates labeling
  efficiency per volume: P4, since it needs the vascular geometry P4 introduces.
- **Kinetics evaluated after through-plane motion**, moving relaxation maps in `voxel` mode, and
  a moving fieldmap: not scheduled; `class` mode with in-plane motion is the recommended
  configuration and the sidecar says so.
- **Suppression pulses that are spatially selective** (a slab narrower than the imaging region):
  not modeled; every pulse acts on every voxel.

# Risks

**The suppression model has no oracle, and its blood term is a named approximation.** The tissue
timeline is mitigated by the closed forms, by the hand-worked `asl002` example whose first-slice
suppression is what such schemes are built to achieve, and by the linearity identity, which is
indifferent to the model but catches wiring errors. The blood term is mitigated only by being
labeled: a quantifier that treats a P3 dataset like a real scanner's will find more label in it
than the scanner would have kept, by up to the factor discussed above, and the sidecar says so.
When ASLDRO's later background-suppression filter is available in the `simasl` environment, its
output is the natural fixture and part A should be re-checked against it.

**Sign conventions.** An odd pulse count flips `control - label`, and a negative tissue `Mz` puts
a pi phase on the tissue. Both are correct and both will surprise a downstream tool that assumes
positive magnitudes and positive differences. The sidecar factor and the signed ground truth are
the mitigation; a quantification step that ignores them is out of scope.

**Motion moves finished images.** The through-plane timing approximation and the fixed maps are
small effects at ASL resolution but they are approximations, and the tests are scoped to what the
model gets exactly right. P2's comparison against simasl, which makes the same approximation,
will not expose them; only the deferred move-then-evaluate pipeline would.
