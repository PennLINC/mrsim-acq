# P3: background suppression, IR contrast and motion — implementation plan

**Goal:** Implement the P3 addendum (`docs/specs/2026-09-24-p3-motion-bs-ir-design.md`) in
`aslscan`: a longitudinal timeline for background suppression, simasl's inversion-recovery
signal equation, and per-volume plus within-volume rigid motion through `mrsim-acq`'s motion
module, with the sidecar, ground truth and tests the addendum specifies.

**Architecture:** Same shape as P1. New pure-std module `longitudinal`; `mrsignal` grows the IR
branch with a simasl fixture; `protocol` grows the three input surfaces and their checks;
`series` applies the timeline, the IR equation, the poses and the shot events before the one
acquisition call; `bids` writes the new sidecar keys, the motion TSV and the static/moved ground
truth. **`mrsim-acq` is not modified** (addendum, part C).

**Spec:** the P3 addendum, which the Codex review of 2026-09-24 shaped; the main spec
(`2026-09-21-mrsim-acq-aslscan-design.md`) where the addendum is silent. P1 is complete at tag
`p1-complete` in `aslscan`.

## Global constraints

- Everything in the P1 plan's global constraints still holds (pure-std default build, seconds
  upstream, one acquisition call per series, physics on the phantom grid, no `cargo fmt`, no
  large data committed).
- **P1 outputs are unchanged when nothing new is enabled.** With `BackgroundSuppression: false`,
  `acq_contrast = "se"` and `[motion]` absent, every P1 test passes unchanged and the P1 code
  paths are the ones that run. Specifically `mrsignal::tissue_se` still produces the tissue
  signal (the timeline is used only with at least one event), and the P1 ground-truth path is
  what `desc-deltam` is when motion is off.
- **Every approximation is named in the sidecar**: `BackgroundSuppressionModel: "global-bolus"`,
  and the motion approximations (slice timing travels with the anatomy; per-voxel maps and the
  fieldmap stay in scanner space).
- **Rejections over silent defaults.** Every new constraint in the addendum is a `protocol`
  error naming the offending values.
- Commit per task with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; tag
  `p3-complete` at the end.

## File structure

| Path | Change |
|---|---|
| `src/longitudinal.rs` | New, pure std: `Suppression`, `tissue_mz`, `label_factor` |
| `src/mrsignal.rs` | `Contrast::InversionRecovery`, `IrParams`, `tissue_ir`, `blood_ir`; `parse_contrast` accepts `"ir"` |
| `tools/gen_mrsignal_fixtures.py` | `--ir` cases -> `tests/fixtures/mrsignal_ir.txt` |
| `src/rng.rs` | New, pure std: SplitMix64 for the within-volume events (mrsim-acq's is private) |
| `src/protocol.rs` | `[background_suppression]`, `[signal]` IR keys, `[motion]`; `Protocol::{suppression, ir, motion, mb_interleaved}`; the addendum's checks |
| `src/series.rs` | Per-slice tissue timeline, label factor, IR branch, poses and events, moved ground truth, `MOTION_SEED_SALT` |
| `src/bids.rs` | New sidecar keys, effective `InversionTime`/`FlipAngle`, `desc-motion_gt.tsv`, `desc-deltamStatic` |
| `tests/end_to_end.rs` | asl002-shaped suppression run, IR linearity, motion translation, reproducibility |
| `src/bin/aslscan.rs` | Report suppression / contrast / motion in the run summary |

---

### Task 1: `longitudinal`

Pure std. `Suppression { pulse_times: Vec<f64>, epsilon: f64, presaturation: bool }` with
`Suppression::new` sorting the pulse times. `tissue_mz(m0, t1, tr, t_read, s)` walks the
timeline of the addendum: `Mz(0) = m0 (1 - exp(-(tr - t_read)/t1))` or `0` with presaturation;
recovery to each pulse `p_k < t_read`, then `Mz *= 1 - 2 epsilon`; recovery to `t_read`. A zero
T1 returns 0, the same guard `tissue_se` applies (an infinite rate is the P1 convention for
"no signal", and a timeline through `exp(-x/0)` would give `m0`). `label_factor(s)` is
`prod (1 - 2 epsilon)` over all pulses.

Tests (closed forms from the addendum): `N = 0` equals `tissue_se` within `1e-12` relative;
`epsilon = 0` is the identity on the `N = 0` value; one perfect pulse an instant before readout
gives the negative of the `N = 0` value (to `1e-9`); two coincident perfect pulses cancel;
presaturation gives `m0 (1 - exp(-t_read/t1))` with no pulses; the asl002 example gives
first-slice GM `0.156`, WM `0.175`, CSF `0.219` and last-slice GM `0.499`, WM `0.656` (times
`m0`, to `1e-3`); `label_factor` is `(-1)^N` at `epsilon = 1` and `0.81` for two pulses at
`0.95`; a pulse at or after `t_read` is ignored by `tissue_mz`.

- [ ] Write `src/longitudinal.rs` with the tests, add `pub mod longitudinal;` to `lib.rs`.
- [ ] `cargo test` (default features) green in WSL.
- [ ] Commit: `feat: longitudinal — background-suppression tissue timeline and label factor`.

### Task 2: `mrsignal` inversion recovery

`Contrast::InversionRecovery` (the parameters live on `Protocol`, not the enum, so `Contrast`
stays `Copy + Eq`). `IrParams { inversion_time, excitation_flip_deg, inversion_flip_deg }`.
`tissue_ir(m0, t1, tr, p)` is the addendum's equation with simasl's guards: `exp(-x/t1)` is
`exp(0) = 1` when `t1 == 0` (both exponentials), the quotient is `0` when the denominator is `0`.
`blood_ir(delta_m, p) = sin(fa) delta_m`. `parse_contrast("ir")` now succeeds.

Fixtures: `tools/gen_mrsignal_fixtures.py` gains an IR section writing
`tests/fixtures/mrsignal_ir.txt` over `TI in {0.3, 1.0, 2.5}`, `fa in {90, 60, -30}`,
`fa_inv in {180, 120, 0}`, `TR in {2, 4}` on the same random voxel grids (with a zero-T1 and a
zero-T2 voxel per case), `exp(-TE/T2)` divided out as for spin echo. Test: every voxel to
`1e-12` relative. Closed form: `fa_inv = 0` gives `sin(fa) m0 (1 - E) / (1 - cos(fa) E)`, which
equals `tissue_se` only at `fa = 90`.

- [ ] Extend the generator; run it in the `simasl` env; commit the fixture.
- [ ] Implement and test.
- [ ] Commit: `feat: mrsignal — inversion-recovery contrast against the simasl oracle`.

### Task 3: `rng` and `protocol`

`src/rng.rs`: SplitMix64 with `unit()` and `signed(amp)`; two tests (determinism, range).

`protocol` additions, each with its error naming the values:

- **Background suppression.** When `BackgroundSuppression` is true:
  `BackgroundSuppressionNumberPulses` (non-negative integer) and
  `BackgroundSuppressionPulseTime` (array, that many finite non-negative entries) are
  required; length mismatch names both. Overlay `[background_suppression]
  inversion_efficiency` (default 0.95, in `[0, 1]`), `presaturation` (default false),
  `pulse_times_per_pld` (array of arrays, one per distinct non-m0 PLD in ascending order, each
  entry validated like the sidecar array). Per row the pulse set is the per-PLD array when
  given, else the sidecar array; m0scan rows get none. For a multi-PLD series without the
  per-PLD override, `first_pld_applied_to_all = true` is recorded. Checks per non-m0 row: every
  pulse `< row.t` (the first slice's readout), every pulse `>= row.tau`, and
  `row.t + max(slice_offsets) <= row.tr` (this last one for every row, suppression or not).
- **IR contrast.** `[signal] inversion_time`, `excitation_flip_angle`, `inversion_flip_angle`;
  sidecar `InversionTime`, `FlipAngle`. Precedence overlay > sidecar > default with `Source`.
  Rules: `"ir"` with suppression true is rejected naming both; `"se"` with a `FlipAngle` (sidecar
  or overlay) other than 90 is rejected naming the spin-echo assumption; `"se"` with
  `InversionTime` in the sidecar (or the overlay) is rejected. Ranges: TI `>= 0`, angles in
  `[-180, 180]`.
- **Multiband schedule.** With `mb > 1`, derive the shot grouping from the slice offsets: the
  slices sharing each distinct offset must be `{g, g + n_groups, ...}` for one `g`, and the
  groups ordered by offset must be `0, 1, 2, ...` (sequential) or `0, 2, 4, ..., 1, 3, ...`
  (interleaved). Any other pattern is rejected naming the first mismatched slice.
- **Motion.** `[motion] mode` (`off` default | `trajectory` | `random` | `linear`), `trajectory`
  (path, read relative to the overlay file's directory by `load`; `parse` reads it as given),
  `trans_mm`, `rot_deg` (3-vectors, finite, non-negative), `volumes` (indices `< n`, no repeats;
  default all), `within_volume` (`dropout_rate` and `severity` in `[0, 1]`, `jump_mm`,
  `jump_deg`; needs `mb > 1`). Trajectory pose count must equal the row count.

The resolved types on `Protocol`:

```rust
pub suppression: Option<SuppressionSpec>,  // epsilon, presaturation (with Source), per_row,
                                           // first_pld_pulses, first_pld_applied_to_all
pub ir: Option<IrSpec>,                    // IrParams plus a Source per parameter
pub motion: Option<MotionSpec>,            // mrsim_acq::motion::MotionMode, mode_name,
                                           // within: Option<WithinVolume>
pub mb_interleaved: bool,
```

Tests for each rule, plus: asl002 and asl004 now parse (suppression on, 2 pulses, factor 0.81
at the default), asl002 with `pulse_times_per_pld` of the wrong length is rejected, the multiband
`[0, 0, 1, 1]` counterexample is rejected while `[0, 0.05, 0, 0.05, 0, 0.05]` (sequential) and an
interleaved pattern are accepted with the right flag.

- [ ] Commit: `feat: protocol — background suppression, IR and motion inputs with the P3 checks`.

### Task 4: `series`

- **Tissue.** When `p.suppression` is `Some` and the row has at least one event (pulses or
  presaturation), the tissue image is built per slice with `r_sim.mean_slice`, from
  `tissue_mz(m0, t1, tr, row.t + offset[z], &Suppression)`, cached on
  `(tr bits, t_read bits, pulse-set index)` where pulse sets are deduplicated per protocol.
  Otherwise the P1 whole-volume `tissue_for(tr)` path runs unchanged. For IR the whole-volume
  path uses `tissue_ir` (steady state, no slice dependence). m0scan rows and the separate M0
  scan always use `tissue_se` at their own TR (the M0 scan is a plain readout; recorded).
- **Blood.** `sign * label_factor` for label/deltam rows under suppression; `blood_ir` under IR.
- **Motion.** After assembly: `resolve_poses(mode, n, seed ^ MOTION_SEED_SALT)` (trajectory
  poses pass through), `apply_motion(images, sim dims, n, sim v2w, poses)`. Within-volume:
  events drawn per (volume, shot) with the salted RNG at `dropout_rate`, then
  `apply_multiband_motion(images, dims, n, v2w, mb, interleaved, &DropoutLaw::Uniform, &events)`.
  `SeriesOutput` gains `poses: Vec<Pose>`, `dropped: Vec<DroppedShot>`, `motion_seed: Option<u64>`.
- **Ground truth.** `delta_m_static` is the P1 acquisition-grid value. With motion on,
  `delta_m` is the sim-grid unsigned `delta_m` (per label/deltam row, `r_sim.mean_slice(z, dm)`),
  moved by `apply_motion` with the same poses (no shot events, no attenuation), then
  block-averaged `o x o` in-plane to the acquisition grid. With motion off `delta_m` is the
  static one and `delta_m_static` is `None`.
- **Label factor and the sidecar data** on `SeriesOutput`: `label_factor: Option<f64>`.

Tests: the suppressed tissue image of a one-slice, one-label protocol equals `tissue_mz` of the
class mean (closed form on the homogeneous grid); an odd pulse count flips the sign of the GM
control-minus-label; IR at `fa = 90, fa_inv = 0, TI = 1` reproduces the spin-echo run's magnitude
to `1e-6` relative on a distortion-free, noise-free grid; a trajectory of one +1-voxel in-plane
translation on `oversample 1` shifts the interior of the acquired magnitude within `1e-5`
relative to peak and the moved ground truth exactly; `random` motion is reproducible from the
seed and `poses` are written; the motion salt leaves the noise realization unchanged (the
noise-difference of a moved run against its clean twin equals that of an unmoved run within
float noise for an unmoved volume).

- [ ] Commit: `feat: series — suppression timeline, IR branch, motion and moved ground truth`.

### Task 5: `bids` and the CLI

Sidecar: `AslscanSimulation.Resolved.AcqContrast` effective; `BackgroundSuppression` block
(`InversionEfficiency` resolved, `Presaturation` resolved, `LabelFactor`, `Model: "global-bolus"`,
`FirstPldPulseTimesAppliedToAll`, `PulseTimesPerRow`); `InversionRecovery` block (resolved
values with sources); `Motion` block (`Mode`, `Seed`, `WithinVolume`, `Approximations: [...]`);
`M0ScanContrast: "se"`. Standard keys: `InversionTime` and `FlipAngle` effective with originals
under `InputValuesReplaced` (only when IR). Files: `desc-motion_gt.tsv` (`volume`, six
parameters, then one line per dropped shot: `volume`, `shot`, `slices`, `attenuation`, jumps)
when motion is on; `desc-deltamStatic_gt.nii.gz` when motion is on. CLI prints the three new
switches. Validator: a suppression run, an IR run and a motion run validate with zero errors.

- [ ] Commit: `feat: bids — P3 sidecar keys, motion ground truth, static delta_m`.

### Task 6: end-to-end, acceptance, measurements

`tests/end_to_end.rs` (features `io,test-hooks`): linearity with suppression on (the three runs
share the pulses), linearity with motion on (shared trajectory), asl002's real sidecar with its
`SliceTiming` cut to the crop's slice count: label factor `0.81` at the default and, with
`inversion_efficiency = 1`, first-slice GM/WM suppression > 80% against a suppression-off run
and the complex `control - label` unchanged within `1e-6` relative.

Acceptance run (not a committed test; results recorded below): convert the full 3T phantom
cropped to 100 mm in z (`--crop 0:197 0:233 45:145`, 20 slices at 5 mm), run asl002's sidecar and
aslcontext with an overlay of `[m0] repetition_time`, validate; then the `epsilon = 1` and
suppression-off twins for criterion 2; an IR run; a `random` motion run and its reproduction.

- [ ] Commit: `test: P3 end-to-end runs and acceptance`; tag `p3-complete`.

## Acceptance criteria coverage

| Criterion | Where |
|---|---|
| 1. asl002 from its sidecar, validator-clean, factor 0.81 recorded | Task 6 acceptance run |
| 2. First-slice GM/WM > 80% suppressed, complex difference unchanged | Task 6 test + run |
| 3. IR against the simasl fixtures at 1e-12 | Task 2 |
| 4. Trajectory translation exact at the resampler and within tolerance through the acquisition; random reproducible | Task 4 + Task 6 |
| 5. mrsim-acq unchanged; `cargo test` default build green in both crates | every task |

## Measurements (2026-09-24, tag `p3-complete`)

Executed as six commits on `aslscan` (`80a99d3` … the Task 6 commit); `mrsim-acq` untouched.
Deviations from the plan text: the motion ground truth is two TSVs (`desc-motion_gt.tsv` with
the per-volume poses, rotations in radians, and `desc-motionEvents_gt.tsv` with the shot events)
rather than one mixed file; the IR fixture grid uses `TI in {0.3, 1.0, 1.8}` because simasl
requires `TR >= TE + TI`, a constraint `protocol` now enforces too; and criterion 2's tolerance
on the complex difference is `1e-4` of its peak, not `1e-6`, because the difference is about 1%
of the tissue signal and the float32 storage of the images puts the floor near `1e-5`
(measured `3.3e-5` on the crop, `4.1e-5` on the full phantom). The addendum was updated.

Tests: 38 (default build), 70 (`io`), 8 end-to-end (`io,test-hooks`); clippy clean with
`cli,test-hooks --tests`. The 24 IR fixture cases match simasl at `1e-12` relative and the
spin-echo fixture is byte-identical to P1's.

| Measurement | Value |
|---|---|
| Linearity residual / tolerance: P1, with suppression, with a shared pose | 0.138, 0.100, 0.152 |
| asl002 on the crop, first-slice GM control ratio (perfect pulses / off) | 0.1608 (closed form 0.1608) |
| asl002 on the crop, first-slice WM control ratio | 0.1757 (closed form 0.1757) |
| Complex `control - label` change, perfect pulses vs off (crop / full phantom) | 3.3e-5 / 4.1e-5 of peak |
| Default efficiency: deviation from 0.81 x the unsuppressed difference | 2.5e-5 / 3.3e-5 |
| Full phantom (z-cropped to 20 slices), slice-0 interior GM / WM suppression | 83.8% / 82.3% |
| Trajectory +2 voxel shift: acquired magnitude deviation (o = 1) | < 1e-5 of peak; ground truth bit-exact |
| Within-volume events, dropout 1.0 / severity 0.5: image vs half the still image | < 1e-6 of peak |
| Run time, asl002 real sidecar, 70 volumes, 106 x 126 x 20 simulation grid, release | 7.3 s (suppression), 7.9 s (IR), 9.0 s (random motion) |
| Random motion, same seed, twice | byte-identical magnitude NIfTI |
| bids-validator 3.0.2 on the suppression, IR and motion runs | 0 errors; warnings as P1 (NIFTI_UNIT/PIXDIM, recommended keys, TOO_FEW_AUTHORS) |

## Codex adversarial review (2026-09-24)

Nine findings on the tagged range, all verified and fixed; `p3-complete` moved onto the fix
commits. The blocker: the suppressed-tissue slice cache was keyed on `(TR, t_read, pulse set)`
but not the slice, so simultaneously excited multiband slices received the first slice's
anatomy (found independently while the review ran; the suppression ratio test now runs on a
multiband timing and fails with the old key). The defects: effective
`BackgroundSuppressionPulseTime`/`NumberPulses` are now written from the first PLD's resolved
pulses; the separate M0 sidecar's `FlipAngle` is the 90 degrees it is simulated at; the
trajectory TSV is validated before mrsim-acq's loader (which zeroes bad cells and accepts NaN);
a sidecar `FlipAngle` in BIDS' `[0, 360]` is normalised to simasl's signed range; the
readout-in-TR check covers included m0scan rows and the separate M0; duplicate `motion.volumes`
are refused (mrsim-acq's linear mode would halve the ramp); an overlay
`excitation_flip_angle = 90` is accepted for spin echo; and the series IR tests gained a real
inversion case and an exact `sin(fa)` check so a bypass of the IR tissue equation cannot pass.
Nothing numerical was found wrong in `longitudinal`, `mrsignal`, the motion application or the
ground truth.

Acceptance run recipe (gitignored): `tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t --out
work/phantom-3t-z100 --crop 0:197 0:233 45:145`, then `work/acceptance_p3.sh` (asl002's real
sidecar and aslcontext with `[m0] repetition_time = 8.0`; the `inversion_efficiency = 1.0` and
`BackgroundSuppression: false` twins; an IR overlay on the suppression-off sidecar; a `random`
motion overlay run twice) and `work/acc_check.py` for criterion 2.
