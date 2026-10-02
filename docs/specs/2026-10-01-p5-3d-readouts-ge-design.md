# P5: gradient-echo contrast, 3D GRASE and 3D stack-of-spirals readouts

Design spec addendum, 2026-10-01. Extends `2026-09-21-mrsim-acq-aslscan-design.md` (the main
spec) and follows the P2, P3 and P4 addenda (`2026-09-24-p2-asldro-compat-design.md`,
`2026-09-24-p3-motion-bs-ir-design.md`, `2026-10-01-p4-vascular-physio-design.md`). Where this
document is silent, those stand. Codex-reviewed twice before implementation (2026-10-01: 4
blockers and 18 majors; 2026-10-02: 2 blockers, 6 majors and a minor; all verified and applied;
both reviews are summarized at the end).

## Context

The roadmap gives P5 "3D readouts: GRASE, stack-of-spirals, gradient-echo contrast", and the main
spec explains why they travel together: "3D GRASE and stack-of-spirals readouts, and
gradient-echo contrast, all of which need a different echo-formation model in `mrsim-acq`"
(main spec, "Decisions deferred"). Every earlier sub-project left `mrsim-acq` unchanged; this one
cannot.

What exists today, verified against the source:

- **The acquisition stage is 2D spin-echo EPI, slice by slice.** `simulate_acquisition_oversampled`
  loops volumes, then slices, then coils (`kspace.rs:1383-1410`, `:1056-1058`); each slice is an
  independent 2D forward model and reconstruction. There is no kz, no slab, and no z Fourier term
  anywhere, and z is never oversampled (`kspace.rs:1357`, "slice count must match; z is never
  oversampled").
- **Timing is one value per phase-encode line, from a hard-coded single-shot EPI.**
  `line_times(epi: &SingleShotEpi)` (`kspace.rs:237-250`) samples each line's centre `kx` for
  `t` (from the echo), `trf` (from the RF) and `tread` (from the prep gradient). The `Readout`
  trait (`readout.rs:9-19`) is never used polymorphically; `SingleShotEpi` is built from the
  `Acquisition` at three sites (`kspace.rs:287`, `:484`, and the test oracle at `:1596`).
- **Spin-echo physics is explicit.** Relaxation is `exp(-trf/T2 - |t|/T2')`, symmetric in `t`
  about the echo (`kspace.rs:762`, per voxel at `:823`); the fieldmap phase is `2 pi fmap t`
  with `t` from the echo, and no term involves `trf` (`kspace.rs:588`, `:612`). `phase.rs:4-9`
  states the reason: a `2 pi fmap TE` term "would double-count B0 and impose gradient-echo
  physics on a spin-echo sequence."
- **aslscan refuses all three.** `MRAcquisitionType` other than `2D` is an error naming P5
  (`protocol.rs:862-868`); `acq_contrast = "ge"` is an error naming P5 (`mrsignal.rs:38-50`).
  The three 3D `bids-examples` datasets (`asl001`, `asl003`, `asl005`) are refused by test
  (`protocol.rs:2027-2032`).

What the three target datasets carry (pinned copies, `aslscan/tests/fixtures/protocols`,
bids-examples `7150efcf`):

| | `asl001` | `asl003` | `asl005` |
|---|---|---|---|
| Scanner, labeling | GE MR750, PCASL | Siemens Trio, PASL (FAIR, Q2TIPS) | Siemens Prisma, PCASL |
| `PulseSequenceType` | `spiral` | `3Dgrase` | `3Dgrase` |
| `NumberShots` | absent | 2 | 4 |
| `EchoTime` (s) | 0.010528 | 0.01192 | 0.01328 |
| `FlipAngle` | 111 | 180 | 130 |
| `AcquisitionVoxelSize` (mm) | 4, 4, 8 | 8, 4, 6 | 3.4, 3.4, 4 |
| `PhaseEncodingDirection` | absent | `j-` | absent |
| `EffectiveEchoSpacing`, `TotalReadoutTime` | absent, absent | 0.0005, absent | absent, absent |
| `DwellTime` | absent | 3.4e-6 | 3.2e-6 |
| `SliceTiming` | absent | absent | absent |

BIDS (schema 1.2.7, the one bids-validator 3.0.2 loads) settles three questions and leaves the
rest open:

- **`PostLabelingDelay` ends at the slab excitation in 3D**: "until the middle of the excitation
  pulse applied to the imaging slab (for 3D acquisition) or first slice (for 2D acquisition)"
  (`objects.metadata.PostLabelingDelay`). Every partition of a 3D shot therefore shares one
  kinetic time.
- **`SliceTiming` is required for 2D only** (`rules.sidecars.mri.SliceTimingASL`), and the
  specification's dependency table says it must not be defined for 3D (no validator rule
  enforces that).
- **`BackgroundSuppressionPulseTime` counts from the start of labeling**, as P3 assumed.
- **Nothing describes the trajectory or the echo train.** There is no `EchoTrainLength`, no
  `RefocusingFlipAngle`, no trajectory field outside MRS; `NumberShots` ("the number of RF
  excitations needed to reconstruct a slice or volume") is the only segmentation field, and
  `EffectiveEchoSpacing` and `TotalReadoutTime` are "effective" values defined for distortion
  correction, not physical durations.
- **`FlipAngle` is one number with no echo-train semantics.** In all three 3D examples it is
  plainly the refocusing angle (111, 180, 130 degrees: no 3D GRASE or FSE excitation is 180
  degrees); this spec reads it that way for the spin-echo-train readouts and says so.

simasl has no readout of any kind (no k-space, PSF or segmentation; it writes
`mr_acq_type = "3D"` for every image, `mri_signal_filter.py:186`), so it is an oracle for the
gradient-echo **signal equation** only (`mri_signal_filter.py:188-231`), not for any 3D effect.

## The one idea underneath

The 2D forward model already factors a slice's k-space into pieces that depend on the line in
different ways (`kspace.rs:497-531`): a static per-voxel amplitude and phase, an affine
off-resonance phase advanced by memoised rotors, a separable eddy term, and **one relaxation
scalar per compartment per line**. All three P5 readouts keep that structure and change only
what a "line" is and how its times are measured:

- **Gradient echo** changes the two functions of time a line carries: decay becomes
  `exp(-trf/T2 - trf/T2')` (monotonic from the RF), and the fieldmap phase becomes
  `2 pi fmap trf` (unrefocused), which is today's `2 pi fmap t` plus a static per-voxel
  `2 pi fmap TE`. A static per-voxel phase is exactly what the existing `phi0` holds, so the
  rotor and NUFFT paths survive unchanged.
- **A 3D echo train** (GRASE, and the stack of spirals) reads one kz partition per refocused
  spin echo. Each echo is a spin echo, so within an echo the 2D spin-echo physics holds with `t`
  measured from **that** echo's centre; across echoes the only change is a per-echo amplitude
  (T2 decay, with stimulated-echo contributions when the refocusing angle is below 180 degrees).
  That amplitude is a scalar per compartment per partition, the same kind of factor as today's
  per-line relaxation scalar. So the 3D k-space of compartment `c` is the z-DFT of the 2D
  per-slice k-spaces, each line weighted by its echo amplitude:

```
K(kx, ky, kz_p) = sum_c  W_c(p, ky) * sum_z exp(-i 2 pi (p - pc)(z - zc) / nz) * K2_{c,z}(kx, ky)
```

  where `K2_{c,z}` is the existing 2D forward of slice `z` of compartment `c` with the
  within-echo timing and **no relaxation**, and `W_c(p, ky)` carries everything that depends on
  which echo and which shot read line `(p, ky)`: the whole decay of that line (the echo
  amplitude and the within-echo T2 and T2' terms, as one exponent) and (part D) the per-shot
  physiological factors. The 2D forward runs once per slice and compartment, as now; the z-DFT
  and the weights are cheap. This is exact for the linear forward when the weights are spatial
  scalars: the static fieldmap, object phase and coil sensitivity stay inside `K2`, and a
  sampling mask that is the same for every partition (partial Fourier, GRAPPA) commutes with the
  z encoding. It is not exact for an echo-dependent eddy evolution, which is why the eddy drives
  are refused in 3D. In `voxel` mode the weights are not spatial scalars and the forward runs
  per echo ("The acquisition stage").
- **The stack of spirals** keeps the kz structure and replaces the in-plane Cartesian EPI block
  with a spiral, which the 2D machinery cannot express (integer `kspace_index`, one time per
  line, a Cartesian x-stage). It needs a non-Cartesian forward (a time-segmented type-2 NUFFT)
  and a non-Cartesian (least-squares) reconstruction. That is the one genuinely new numerical component, and it is
  part C.

The reconstruction mirrors the forward: an inverse z-DFT turns the 3D k-space into one 2D
k-space per partition, and each goes through today's 2D reconstruction (GRAPPA, window,
inverse FFT, Roemer combine) unchanged.

## Scope

In: gradient-echo contrast for 2D EPI (part A); the 3D echo-train model in `mrsim-acq` and
aslscan's 3D protocol and timing (part B, with the GRASE readout); the stack-of-spirals readout
(part C); segmentation, meaning per-shot factors and per-shot motion, for both 3D readouts
(part D). Out, with reasons under "Decisions deferred": 3D gradient-echo readouts, 3D
inversion recovery, slab profiles and kz aliasing, through-plane oversampling, out-of-plane
acceleration and kz partial Fourier, variable-density spirals and coil-coupled (SENSE) spiral
reconstruction, and longitudinal evolution during the echo train.

**Byte identity.** With no P5 input, every output is unchanged, now across two repositories:

- **TRXScan**: the consumer of `mrsim-acq` is TRXScan's `p0-mrsim-acq-extraction` branch
  (`130d62b` at this writing), not `main`, which still carries its own `kspace` and lacks the P0
  fixtures. The P0 gate (`docs/plans/2026-09-21-p0-mrsim-acq-extraction.md:20`), byte-identical
  `trxscan` output against `tests/fixtures/p0_baseline/checksums.txt` on that branch under both
  `--features cli` and `cli,kspace,par`, is re-run, with the branch's commit and the resolved
  `mrsim-acq` path recorded in the plan. `trxscan.rs:828` and aslscan's `protocol.rs:1688` build
  `Acquisition` with every field named, so the new field is a source change at both call sites
  (`echo: EchoFormation::Spin`); it moves no bits.
- **mrsim-acq**: every bit-pinned test passes unchanged, in particular
  `eddy_drive_some_reproduces_the_bvec_bval_product` (`EXPECTED_BITS_STD`, `EXPECTED_BITS_FFT`,
  `kspace.rs:2822-2849`) and `restructured_forward_matches_the_literal_sum` (`:2721-2779`,
  1e-10). Those pin eight coefficients and a tolerance, which is not a historical baseline, so
  P5 adds one: before any P5 change, the full complex `simulate_slice` output of every
  `restructured_forward_matches_the_literal_sum` case, under both feature sets, is written as
  golden `f32` bits to `tests/fixtures/p4_baseline/`, and a test compares against it bit for bit.
- **aslscan**: `tools/regress_identity.sh` compares against a baseline in which **both**
  repositories are at their pre-P5 revisions. Today it builds the base `aslscan` in a sibling
  worktree whose `path = "../mrsim-acq"` resolves to the live, modified `mrsim-acq`
  (`Cargo.toml:23`, `regress_identity.sh:16`), so a P5 change to the acquisition stage would be
  in both binaries and the comparison would pass vacuously. The script therefore takes two
  revisions (`aslscan` and `mrsim-acq`), checks both out as worktrees side by side in a scratch
  parent directory (so the relative path resolves to the pinned `mrsim-acq`), verifies with
  `cargo metadata` that the base build resolved that copy, and builds under both
  `cli,kspace,par` and `cli`. Its cases gain the P4 combinations (a `[physio]` case, a
  `[macrovascular]`-plus-crushing case, a bolus-position case), since P5 touches the physiology
  output. P2's benchmarks A-E pass unchanged.
- **Old outputs keep their bytes**: every new sidecar field and every new ground-truth column is
  written only when its feature is on. The P4 `desc-physio_gt.tsv` keeps its 2D schema and
  serialization exactly; the 3D schema (part D) is a separate branch of the writer.

The rule that makes this cheap: every new behavior is behind a new value of a new field whose
default is today's behavior, and the default path calls the existing code, not a generalized
copy of it. Where code is generalized (the line-timing table below), the default constructs
exactly the values the old code computed, in the same order, and a test pins the equality bit
for bit.

**Activation is resolved in one place and nothing is ignored** (the P4 rule, extended). A
readout is selected by `MRAcquisitionType` and the overlay's `[readout]` table; every P5 input
that belongs to a readout or contrast that is off is an error naming it. The specific cases are
listed per part.

---

# Part A: gradient-echo contrast (2D EPI)

## The acquisition physics

`Acquisition` gains `echo: EchoFormation { Spin, Gradient }`, default `Spin`. Under `Gradient`:

```
decay(line)  = exp(-trf/T2 - trf/T2')        (T2* = 1/(1/T2 + 1/T2') from the RF)
phase(voxel, line) = phi0 + 2 pi fmap (TE + t) + 2 pi ky_norm (y - c)
```

- **Decay**: the `|t|` in the `T2'` term becomes `trf` (both in ms), in the uniform per-line
  scalar (`kspace.rs:762`), the per-voxel map path (`:823`) and the literal oracle (`:1672`).
  `trf > 0` on every acquired line is already validated (`validate_acquisition_timing`,
  `:284-310`), so the decay never grows.
- **Phase**: `2 pi fmap TE` is a per-voxel constant, added to `phi0` where `phi0` is formed
  (`kspace.rs:585`), and only for the B0 part of `rate`: the replayed eddy shear that shares
  `rate` (`:589-594`) is a gradient effect that does not accrue before the readout, so the
  static term uses `fmap[i]`, not `rate[i]`, and is zero when `do_distortions` is false. The
  rotor path then needs nothing else, and the NUFFT path picks the term up through `phi0`
  (`:674`), so the NUFFT gate (`:645-698`) is unchanged.
- **`phase.rs:4-9`** is rewritten to say which term applies under which echo formation; the
  object phase model itself does not change.

Everything else is the same: line timing, distortion (`2 pi fmap t` is unchanged; the extra term
is static), partial Fourier, GRAPPA, ghosting, noise.

The literal oracle (`reference_coil_kspace`, `kspace.rs:1587-1760`) gains the same two changes,
written independently from the production expressions, and
`restructured_forward_matches_the_literal_sum` gains `Gradient` variants of its clean,
distortion-plus-relaxation and combined cases.

## The signal equation

`mrsignal` gains `Contrast::GradientEcho` with two forms of the tissue equation, both with the
transverse factor `exp(-TE/T2*)` divided out, as the main spec requires for every contrast
("Division of labor with the acquisition stage"); `a` is the excitation flip angle:

```
spoiled (the model):    tissue_ge = sin(a) * M0 (1 - E1) / (1 - cos(a) E1)                       E1 = exp(-TR/T1)
simasl (compat only):   tissue_ge = sin(a) * M0 (1 - E1) / (1 - cos(a) E1 - E2 (E1 - cos(a)))    E2 = exp(-TR/T2)
both:                   blood_ge  = sin(a) * delta_m
```

The simasl form (`mri_signal_filter.py:206-225`) keeps transverse coherence across repetitions
through `E2`. That is not small in general: at `TR = 4 s`, `a = 60` degrees, it moves `T1 = 3 s`
tissue by `4e-7` relative at `T2 = 0.3 s` and by 3.7 percent at `T2 = 2 s` (CSF-like; Codex's
numbers, checked). EPI and the 3D readouts spoil between repetitions, so the model is the spoiled
form, everywhere, with and without suppression, so that turning suppression on cannot change the
contrast model. The simasl form is used only under `[compat] asldro = true`, where matching the
oracle is the point. Both keep simasl's guards: the quotient is zero where its denominator is
zero (`np.divide(..., where=denominator != 0)`), and `T1 = 0` or `T2 = 0` gives a zero exponent
as `tissue_se` does (`mrsignal.rs:56-59`).

## Inputs

`[signal] acq_contrast = "ge"` (BIDS has no contrast field; the overlay is the only source, as
for `"ir"`). The flip angle resolves overlay `excitation_flip_angle` over sidecar `FlipAngle`
over a default of 90 degrees, with its `Source`, folded to a signed angle as IR already does
(`protocol.rs:1082-1090`). `InversionTime` and `inversion_flip_angle` are refused under `"ge"`
as under `"se"`. `"ge"` is 2D only in P5: with `MRAcquisitionType: 3D` it is an error naming
the deferred 3D gradient-echo readouts.

## Background suppression under a gradient-echo excitation

P3's tissue timeline starts from the saturation the spin-echo readout leaves:
`Mz(0) = M0 (1 - exp(-(TR - t_read)/T1))` (P3, "The tissue timeline"), which is the recovery
from `Mz = 0` at the excitation. A gradient-echo excitation of angle `a` leaves
`cos(a) Mz(t_read)` instead, so the starting value depends on the previous repetition's readout
value, and the timeline becomes a steady state. Every step of the timeline (recovery, pulse,
recovery) is affine in `Mz`, so the readout value is `Mz(t_read) = A Mz(0) + B` with `A` and `B`
obtained by running the existing timeline twice (from `Mz(0) = 0` for `B`, from `Mz(0) = 1` for
`A + B`), and the start of the next repetition is

```
Mz(0) = M0 (1 - Er) + cos(a) Er (A Mz(0) + B),      Er = exp(-(TR - t_read)/T1)
Mz(0) = (M0 (1 - Er) + cos(a) Er B) / (1 - cos(a) Er A)
```

the steady state of the spoiled sequence, the same model as the spoiled `tissue_ge`. The
denominator `1 - cos(a) Er A` is at least `1 - Er |A|`, and `|A| <= 1` (every step scales `Mz` by
at most 1 in magnitude), so the fixed point exists for every `TR - t_read > 0`. With `a = 90`
degrees this is P3's starting value, and the code
keeps P3's path for `a = 90` exactly (the fixed point is used only for `a != 90`), so the P3
outputs do not move by a bit. Presaturation still sets `Mz(0) = 0` and needs no fixed point.
The readout compartment is then `sin(a) Mz(t_read)`, signed, as in P3. The blood factor (P3's
label factor, or P4's bolus-position factors) is unchanged: the excitation does not touch the
label before it is read, and `sin(a)` multiplies it as in `blood_ge`.

Without suppression the gradient-echo tissue signal is the spoiled `tissue_ge` closed form, not
the timeline, as P3 kept `tissue_se`; a test asserts the timeline with no events agrees with the
spoiled closed form to `1e-12` relative (the same model, evaluated two ways).

**A series whose rows differ has no single fixed point.** A spin-echo excitation saturates the
slab, so each row is independent of the one before; a gradient-echo excitation below 90 degrees
does not, and both the fixed point above and the closed form assume the same preparation repeated
forever. Rows can differ (per-row PLD, `RepetitionTimePreparation`, pulse sets, `m0scan` rows with
their longer repetition time; `protocol.rs:1192-1225`), and then the steady state of one row is
wrong for the next: two alternating preparations at 30 degrees give transverse signals of
`-0.2731` and `-0.1109` from their isolated fixed points against `-0.2546` and `-0.0899` from the
actual alternation (second review). So under `"ge"` with `a != 90`:

- if every row of the series has the same preparation (repetition time, readout time, pulse set,
  presaturation), the per-row result is the fixed point (with suppression) or the closed form
  (without), as above;
- otherwise the longitudinal state is **propagated through the rows in acquisition order**,
  per slice (each slice has its own readout time): the first row starts from its own isolated
  fixed point (the scanner's dummy repetitions are taken to have used the first row's
  preparation), and each later row starts from `M0 (1 - Er) + cos(a) Er Mz_read`, with `Er` and
  `Mz_read` those of the row before, through the same affine timeline. The separate M0 scan is its own series and keeps its
  own fixed point.

The sidecar records which of the two applied. The test iterates the actual alternating sequence
by brute force (the second review's two-row example among the cases) and compares row by row to
`1e-12`.

## M0

`m0scan` rows and a separate M0 scan use the series' own gradient-echo equation and flip angle at
their repetition time. P3 kept the M0 scan spin-echo under `"ir"` because the M0 scan is not
inversion-prepared; under `"ge"` the M0 scan is the same excitation and readout without labeling,
so a spin-echo M0 would quantify against the wrong contrast. The M0 sidecar's `FlipAngle` and
`Contrast` (`bids.rs:422-432`, today forced to 90 and `"se"`) follow.

## compat

`[compat] asldro = true` accepts `"ge"` with the simasl form of `tissue_ge`: simasl implements
it, and P2's benchmark machinery (voxelwise against simasl with noise off) extends to a
gradient-echo case. P2 applies simasl's
transverse factor in compat (`series.rs:538-544`, `exp(-TE/T2)`, the `RelaxationAtEcho` sidecar
entry); under `"ge"` that factor is `exp(-TE/T2*)` with simasl's `T2* = 0` guard
(`mri_signal_filter.py:194-198`), computed from the phantom's T2\* map. `tools/compat_asldro.py`
loses its "gradient echo is P5" refusal (`:390`).

---

# Part B: the 3D echo-train model and the GRASE readout

## What is modeled

A 3D segmented spin-echo-train acquisition: per shot, one slab excitation (90 degrees) at
`t_exc`, then `ETL` refocusing pulses of angle `b` at echo spacing `ESP`, each refocused spin
echo reading one kz partition. In GRASE each echo reads an EPI block of `EPI` ky lines centred
on the spin echo. A volume takes `NumberShots` shots, each with its own labeling, delay and
excitation, at one shot per `RepetitionTimePreparation`.

The slab is ideal: it excites exactly the field of view in z, uniformly, and nothing outside it,
so there is no kz aliasing and no slab-edge attenuation (deferred). Partitions equal the
acquisition grid's z cells, and z is not oversampled (deferred).

## Echo amplitudes

The amplitude of echo `e` (1-based) for a compartment with relaxation times `T1`, `T2`, after a
90-degree excitation and refocusing angle `b` (CPMG phase), is computed by the extended phase
graph (EPG) with instantaneous pulses and relaxation between them: `A_e(T1, T2, ESP, b)`. EPG is
the standard way to get CPMG echo amplitudes with refocusing below 180 degrees, where stimulated
echoes contribute (the 130-degree `asl005` case); with `b = 180` it reduces to
`exp(-e ESP / T2)`. It is new code in `mrsim-acq` (`epg`, pure std).

`T1` enters only through stimulated echoes, so the 3D entry point takes a per-compartment `T1`
(`Uniform` or `Map`, like `t2`) when `b != 180`, and none otherwise. In `class` mode the amplitudes
are one scalar per compartment per echo. In `voxel` mode they are a map per echo.

`class` mode needs every relaxation input of a label to be uniform, and below 180 degrees that now
includes T1. The phantom's mode resolution (`phantom.rs:300`) checks T2 and T2' only, and a test
(`t1_perturbation_does_not_affect_the_mode`) pins that T1 does not matter, which is right for 2D.
For a 3D protocol with `b != 180`, `auto` resolution and an explicit `class` request also require
T1 constant per label (the same tolerance, foreground voxels only, background ignored as for
T2); otherwise `auto` picks `voxel` and an explicit `class` is an error naming the label. The 2D
rule is unchanged.

**The T1 map in `voxel` mode** follows the T2 maps' contract exactly: the phantom's T1 reaches
the simulation grid as an M0-weighted mean of the **rate** `1/T1` over the phantom voxels each
simulation cell overlaps, the same resampler and weights as the T2 and T2' rates
(`resample.rs:151-170`, `series.rs:449-450`), in ms at the hand-off; background cells take
`T1 = infinity` (their compartments are zero, as for T2'); it stays in scanner space under motion
(below); and the effective map is written to the ground truth as `desc-acqT1map_gt`, beside
P1's `acqT2map` and `acqT2primemap`. Averaging the rate and then running the EPG is not the same
as averaging the EPG signals of the mixed tissues: it is the same documented mixture
approximation the main spec makes for T2, extended to T1, and the sidecar names it. The
class-versus-voxel equality test therefore uses a phantom whose labels are homogeneous on the
simulation grid (no mixed cells), as P1's does; on a real phantom the boundary discrepancy is
measured and reported, not asserted.

**Amplitudes are carried as logarithms.** The decay of a line is the echo amplitude times the
within-echo terms, and the within-echo `t` is negative for the first half of a GRASE block, so
`exp(-t/T2)` grows. Multiplying separately evaluated factors fails for accepted inputs: `ESP = 10`
ms, `t = -4` ms, `T2 = 0.001` ms gives `0 * inf`, a NaN, where the physical decay is
`exp(-6000) = 0` (Codex's example). So the EPG returns `ln A_e` (with `-inf` for an exactly zero
amplitude), and every line weight is formed as a single `exp(ln A_e - t/T2 - |t|/T2')`, which is
`0` or finite for every positive relaxation time. A test drives `T2` down to `1e-3` ms and up to
infinity and asserts finite outputs.

The `T2'` term does not enter `A_e`: each echo is a spin echo, which refocuses static
inhomogeneity at its centre, so within an echo the `exp(-|t|/T2')` term of the 2D model applies
with `t` from that echo's centre. Blood compartments take blood T2 as in 2D. The blood's T1 for
the EPG is the arterial blood T1 the kinetic model already uses (`t1_arterial_blood`), recorded
with its source.

## Encoding order

Two segmentation axes, set in the overlay (below): `ky_segments` and `kz_segments`, with
`ky_segments * kz_segments = NumberShots`, `EPI = ny / ky_segments` and
`ETL = nz / kz_segments` (both must divide exactly).

- **kz**: shot `(sy, sz)` reads, at echo `e`, the partition `kz_order[sz + kz_segments (e - 1)]`,
  where `kz_order` lists partitions `centric` (centre first, then `+1, -1, +2, -2, ...`; the
  default, and what the Siemens product sequences do for ASL) or `linear` (low to high). The
  kz-centre partition is therefore read at echo `e_c` (1 for centric with one kz segment).
- **ky**: within an echo, the EPI block of shot segment `sy` reads lines `ky = sy + ky_segments j`
  (interleaved), `j = 0..EPI`, centred on the spin echo: acquisition index `j` is read at
  `t_j = (j - (EPI - 1)/2) t_line` from the echo centre, `t_line` being the actual line spacing.
  The traversal follows `SingleShotEpi`'s convention for the phase-encode sign
  (`readout.rs:46-58`): with `reverse_phase` false the block starts at its highest `ky` and
  descends (`j-`), with it true it ascends (`j`). Readout polarity alternates with the
  acquisition index `j`, not with `ky`: with two ky segments, one shot reads only even `ky` and
  the other only odd, and polarity by `ky % 2` (what the 2D ghost uses, `kspace.rs:846`) would
  give each shot a constant polarity.

So every acquired line `(p, ky)` has a shot, an echo, a time from its echo centre `t(ky)` (which
depends on `ky` only, since every echo of every shot uses the same block timing), a readout
polarity, a time from the excitation `trf = e ESP + t`, and a time from the last prep gradient
(`tread`, used only by the eddy model, which 3D refuses). The line-timing table carries all of
them, polarity included; the 3D ghost reads polarity from the table, and the 2D path keeps its
`ky % 2`.

**Shot chronology.** The shots of a volume run in one canonical order, `s = sy kz_segments + sz`
(ky segment outer), each one repetition time after the last. That order is the single source for
everything that depends on when a shot happened: part D's physiological factors and motion, the
ground truth, and the `shot` index in every output.

## Timing from BIDS

Two line spacings must be kept apart. The **actual** spacing `t_line` is the time between
successive lines of one EPI block; it sets the block duration, the within-echo decay and the
fit checks. The **effective** spacing is BIDS's: "the 'effective' sampling interval ... defined
based on the size of the reconstructed image in the phase direction", which for an interleaved
segmented readout is `t_line / ky_segments`; it sets the distortion (the phase accrued per
reconstructed line).

| Quantity | Rule |
|---|---|
| `t_exc` | `PLD + tau` for (P)CASL, `PLD` for PASL (BIDS: the slab excitation), the same for every partition |
| Effective spacing | `EffectiveEchoSpacing` if given; else `TotalReadoutTime / (ny - 1)` (BIDS's definition, used on the new 3D resolver only; 2D keeps `TotalReadoutTime / ny`, main spec); else none |
| `t_line` (actual) | overlay `line_spacing` if given; else effective spacing `* ky_segments`; else `nx_recv * DwellTime`, where `nx_recv` is overlay `readout_samples` or, absent, `nx` (source `"DwellTime"`, recorded as a lower bound: no ramps, no receiver oversampling); else an error naming all of them |
| `ESP` | from `EchoTime`, read as BIDS's echo time, the time from the excitation to the acquisition of the k-space centre: the centre line `ky_c = ny/2` of the kz-centre partition is read at `e_c ESP + t(ky_c)`, so `ESP = (EchoTime - t(ky_c)) / e_c`; overlay `echo_spacing` may set it instead, and then `EchoTime` must equal `e_c ESP + t(ky_c)` within 1 us |
| `PhaseEncodingDirection` | for GRASE, required: it sets the traversal and the distortion direction (main spec), and no default is defensible when the data might have either. It comes from overlay `[readout] phase_encoding_direction` over the sidecar (same values and validation as the sidecar's, `j` or `j-`, `protocol.rs:984-993`); the effective value is written as the standard key, with the input's under `InputValuesReplaced` when they differ. For spirals, refused from either source (part C) |

`t(ky_c)` is not zero in general: with two interleaved segments and `ny = 64`, the centre line is
`j = 16` of a 32-line block whose centre is `15.5`, half a line from the spin echo.

When both `EffectiveEchoSpacing` and `TotalReadoutTime` are given they must satisfy
`TotalReadoutTime = EffectiveEchoSpacing (ny - 1)` within 1 percent, or it is an error naming both;
when they agree, either gives the same `t_line`. When `t_line` comes from the overlay and an
effective spacing is also given, they must agree within 1 percent.

Checks, before anything is simulated, on the actual RF and sampling intervals:

- **The block fits between refocusing pulses**: the refocusing pulses sit midway between echoes,
  and each reserves `refocusing_time` (overlay, default 2.0 ms, the pulse and its crushers,
  recorded) centred on it. The block occupies `[t_0, t_{EPI-1}]` about the echo, so it needs
  `max(|t_0|, |t_{EPI-1}|) + t_line/2 + refocusing_time/2 <= ESP/2`. The error names `ESP`, the
  block's extent and the largest `t_line` or smallest block that would fit.
- **The first refocusing pulse follows the excitation**: `ESP/2 - refocusing_time/2 > 0`.
- **The train fits in the repetition**: the last sample of the last echo,
  `t_exc + ETL ESP + t_{EPI-1} + t_line/2`, is at most `TR`.
- **Every suppression pulse precedes the excitation**: P3's rule with `t_read = t_exc`.

These checks are not weakened to admit a dataset. A sidecar whose numbers do not describe a
feasible train under this model is refused with the numbers in the message, and the user supplies
the missing geometry (`line_spacing`, `ky_segments`, `kz_segments`, `echo_spacing`, the matrix)
in the overlay. The three `bids-examples` sidecars are such a case: under the BIDS reading,
`asl003` (`EffectiveEchoSpacing` 0.5 ms, two shots, 64 phase-encode lines) has a 32-line block of
1 ms lines against an `ESP` near 12 ms, and `asl005` resolved from its `DwellTime` has 16 lines
of 0.2 ms against 13 ms, which fits only through the dwell-time lower bound (Codex worked both).
The acceptance criteria therefore name full overlays, not minimal ones.

## Kinetics and tissue in 3D

Every partition shares `t_exc`, so the per-slice offsets of 2D are all zero in 3D: `aslscan`
represents a 3D protocol with `slice_offsets = [0; nz]` and every per-slice computation (P1
kinetics, P3's timeline, P4's label path) runs unchanged and returns the same value for every
slice. `SliceTiming` with `MRAcquisitionType: 3D` is refused (BIDS says it must not be defined),
as are `SliceEncodingDirection` and `MultibandAccelerationFactor > 1`.

The tissue's longitudinal state follows P3's spin-echo timeline with `t_read = t_exc` and the
90-degree excitation saturating the slab. **Approximation, named in the sidecar**: the echo train
is treated as leaving `Mz = 0` at the excitation and recovery is counted from `t_exc`; in reality
the train lasts `ETL ESP` (a few hundred ms) and refocusing below 180 degrees partially restores
`Mz`. The EPG computes that longitudinal state and a later revision can use it (deferred).

The same tissue and blood images feed every shot of a volume (each shot repeats labeling and
delay identically); part D adds what differs between shots.

## The acquisition stage

`mrsim-acq` gains `simulate_acquisition_3d`:

```rust
pub fn simulate_acquisition_3d(
    sim_dims: [usize; 3], acq_dims: [usize; 3], n_volumes: usize,
    images: &[Vec<f32>],               // per compartment, as today
    t2: &[T2Volume], t1: Option<&[T2Volume]>, fmap: &[f32], t_inhom: Option<&[T2Volume]>,
    acq: &Acquisition, train: &EchoTrain, readout: &Readout3d,
    line_weights: Option<&LineWeights>, // part D: per (volume, shot, compartment) scalars (physio x shot gain)
    shot_images: Option<&ShotImages>,   // part D: per-shot moved images
    phase: &PhaseModel, seed: u64,
) -> (Vec<f32>, Vec<f32>)
```

`EchoTrain { etl, esp_ms, refocusing_deg, kz_order, kz_segments }`, and
`Readout3d::Grase { ky_segments }` or `Readout3d::Spiral { .. }` (part C). Eddy drives, prep
drives and `noise_sigma` are absent: no consumer needs them in 3D, and each is refused rather
than half-supported.

**Forward**, per volume and coil:

1. For each slice `z` and compartment `c`, the existing 2D forward builds `K2_{c,z}` from a
   **line-timing table** (per ky: `t`, `trf`, `tread`, polarity) with relaxation off: the
   fieldmap phase `2 pi fmap t`, the object phase, the coil sensitivity, `signal_scale` and the
   `1/nvox` normalization (`kspace.rs:849`) are applied exactly as in 2D, and no decay. This
   needs `build_coil_kspace` to take a timing table and return per-compartment k-spaces before
   the compartment sum; see "Generalizing the 2D forward" below.
2. A z-DFT per compartment, `(kx, ky)` and partition `p`, with the same centred, asymmetric
   convention the in-plane axes use (`kspace.rs:497-503`, acquired band `[-nz/2, nz/2 - 1]`).
3. The weighted compartment sum `K(kx, ky, p) = sum_c W_c(p, ky) K_c(kx, ky, p)`, with
   `W_c = exp(ln A_{e(p)}(c) - t(ky)/T2_c - |t(ky)|/T2'_c)` (one exponent, see "Echo amplitudes")
   times part D's factors.
4. Spikes and k-space noise per acquired 3D sample, then the sampling mask.

In `voxel` mode (`Map` T2, T1 or T2') the decay is a per-voxel map and cannot leave the sum, so
step 1 runs once per echo `e` with the map path's per-voxel decay exponent extended by the
per-voxel `ln A_e(r)` (one exponent per voxel per line, `ln A_e(r) - t/T2(r) - |t|/T2'(r)`,
evaluated where the map path evaluates its decay today, `kspace.rs:813-823`), and steps 2-3 take
each partition from its own echo's slices. The cost is `ETL` times the class-mode forward, which
is accepted for P5's targets (ASL matrices are small: `asl005` is about 64 x 64 x 30, `ETL` at
most `nz`), and is measured and reported by the cross-check test.

**Precision.** The public entry point returns `f32` magnitude and phase, as the 2D one does
(`kspace.rs:1108`, `:1444`), so tests that need more than `f32` agreement (the PSF, the
segmentation ghost) use a crate-internal entry that returns the reconstructed complex `f64`
image before the cast. Tolerances against the public output are stated relative to `f32`
storage (`1e-6` of peak).

**Reconstruction**, per volume and coil: an inverse z-DFT (normalized by `1/nz`; the forward is
unnormalized), then each partition's 2D k-space through the existing per-slice reconstruction
(GRAPPA, window, inverse FFT, Roemer combine). The window is today's in-plane radial one; no kz
window is applied (deferred).

**Noise**. The per-sample k-space noise has today's SD, `sqrt(noise_variance / (nx ny))`
(`kspace.rs:891`), and the `1/nz` inverse z-DFT then gives an image noise SD of
`1/sqrt(nz)` times the 2D value at the same `noise_variance`. That is the physical 3D SNR
advantage (each sample carries the whole slab's signal at the same per-sample bandwidth), it is
stated in the sidecar, and a test measures it. The statement holds for a linear Cartesian
reconstruction (full sampling or zero-filled partial Fourier); GRAPPA's weights are calibrated on
the data and the spiral reconstruction has its own noise transfer, so for those the ratio is measured and
recorded, not asserted.

**Seeds**. The 3D path is new, so it does not need P0's per-slice seed: noise and spikes are keyed
on `(volume, partition, coil)` with a 3D salt (`seed ^ 0x3344_5245_4144`, "3DREAD"), written to
the sidecar.

**Coil sensitivities** are today's 2D ring (`kspace.rs:193-204`), the same for every partition.
A z-dependent sensitivity is deferred; the sidecar says the coils are uniform in z.

## Generalizing the 2D forward

The 3D path needs three things the 2D forward does not offer: an arbitrary per-line timing table,
per-compartment k-spaces before the compartment sum, and a static per-voxel weight per call. Each
is introduced so that the 2D path computes what it computes now:

- **Line timing**: `line_times` keeps its signature and its body; a new
  `LineTiming { t_ms, trf_ms, tread_ms }` is built from it for the 2D path, and the forward reads
  the table instead of calling `line_times` directly. The values, their order and every
  expression that consumes them are unchanged; a test asserts the table built for every case of
  `restructured_forward_matches_the_literal_sum` is bit-identical to `line_times`.
- **Per-compartment output**: a private variant of the y-sum accumulation keeps `ncomp`
  accumulators instead of one. The 2D path keeps the single accumulator it has; the two are not
  merged, so the 2D floating-point summation order is untouched.
- **The NUFFT gate** requires `t` affine per parity in `ky` (`kspace.rs:647-657`), and it is
  evaluated on the table as given, unchanged. An interleaved GRASE block is affine per
  segment; with two segments the segments are the parities and the gate can pass, with more it
  fails and the slice takes the rotor path, which handles any timing (memoised rotors fall back
  to the closed form on a mismatched step, `kspace.rs:719-732`). The parity term `delta` of the
  NUFFT path models a polarity offset by parity; with polarity carried by acquisition index
  (above) the gate must also require polarity to be a function of `ky % 2`, or decline. A
  per-residue NUFFT for more segments is deferred.

## Inputs

`MRAcquisitionType: 3D` with `PulseSequenceType` naming GRASE (case-insensitive match on
`grase`, which covers `3Dgrase`) selects GRASE; `spiral` selects part C; anything else is an
error listing the two, and the overlay's `[readout] type = "grase" | "spiral"` overrides a
missing or unrecognized `PulseSequenceType` (source recorded).

Overlay `[readout]`:

| Key | Default | Meaning |
|---|---|---|
| `type` | from `PulseSequenceType` | `"grase"` or `"spiral"` |
| `ky_segments` | `NumberShots` (all segmentation in ky) | GRASE only |
| `kz_segments` | 1 | partitions split across shots |
| `kz_order` | `"centric"` | or `"linear"` |
| `echo_spacing` | from `EchoTime` | ms |
| `refocusing_time` | 2.0 | ms, the refocusing pulse and crushers, centred midway between echoes |
| `refocusing_flip_angle` | sidecar `FlipAngle`, else 180 | degrees, in `(0, 180]` |
| `line_spacing` | see "Timing from BIDS" | ms, the actual EPI line spacing |
| `readout_samples` | `nx` | receiver samples per line, for the `DwellTime` fallback |
| `phase_encoding_direction` | sidecar `PhaseEncodingDirection` | `j` or `j-`; GRASE only |

The matrix comes from `[acquisition] matrix` (in-plane, as today) and the phantom extent along z;
GRASE needs `ny` divisible by `ky_segments` and `nz` by `kz_segments`, and the error names the
matrix override that would satisfy it.

For GRASE, `NumberShots` absent means 1 (spirals: part C). `ky_segments * kz_segments` must equal it. Every `[readout]` key is
refused with `MRAcquisitionType: 2D`, and the GRASE-only keys with `type = "spiral"`.

**FlipAngle**: for both 3D readouts the sidecar's `FlipAngle` is read as the refocusing angle and
the excitation is 90 degrees. An overlay `excitation_flip_angle` other than 90 is an error (the
readouts excite at 90 by construction), as is `acq_contrast` other than `"se"` (part A's `"ge"`
is 2D only; `"ir"` with 3D is deferred). The sidecar records the interpretation.

**Refused with 3D**: `ParallelReductionFactorOutOfPlane` other than 1, `PartialFourierDirection`
naming the slice axis, `SliceTiming`, `SliceEncodingDirection`, `MultibandAccelerationFactor > 1`,
`[compat] asldro = true` (simasl has no readout to compare), and the eddy keys of
`[acquisition]` (no prep gradient in ASL; the eddy model is TRXScan's). In-plane
`PartialFourier` and `ParallelReductionFactorInPlane` are accepted for GRASE: they act per
partition exactly as in 2D.

## Ground truth and outputs

- `AslscanSimulation.Readout`: the type, `NumberShots`, both segment counts, `ETL`, `ESP`,
  `t_line` and its source, the effective spacing, `kz_order`, `e_c`, `t(ky_c)`, the refocusing
  angle and its interpretation, `t_exc`, the `refocusing_time`, the shot order, the 3D salt, the
  noise statement, and the approximations (ideal slab, no z oversampling, `Mz = 0` after the
  train, uniform coils in z).
- The NIfTI time step (`pixdim[4]`, `bids.rs:335`) is the volume duration,
  `NumberShots * RepetitionTimePreparation`; the sidecar's `RepetitionTimePreparation` stays the
  per-shot value it was given, and `AslscanSimulation.Readout.VolumeDuration` states the
  relation.
- `AslscanSimulation.EchoAmplitudes`: in `class` mode, per compartment label, the `ETL` EPG
  amplitudes, and the resulting **kz modulation** (amplitude per partition in encoding order),
  from which the through-plane point-spread function follows; in `voxel` mode, the class values
  of the phantom's labels, recorded as indicative.
- Standard keys describe what was simulated (main spec, "Output contract"): `EffectiveEchoSpacing`
  and `TotalReadoutTime` are written as resolved for GRASE, with the input values under
  `InputValuesReplaced` where they differ; `NumberShots` is written; `FlipAngle` is the resolved
  refocusing angle; `SliceTiming` is never written for 3D.
- **The M0 scan** (`m0scan` rows and a separate M0) uses the same readout and train with its own
  resolved acquisition: no labeling, excitation at `t = 0` of its repetition, its own repetition
  time, and the train-in-TR check against that repetition time. Its sidecar is built from that
  resolution, not copied from the ASL series: the same `FlipAngle` (the refocusing angle; today's
  writer forces 90, `bids.rs:430`, which stays for 2D spin echo), `NumberShots`, the effective
  spacing, and a readout block without `t_exc`.
- In 3D, `desc-physio_gt.tsv` (P4) is written with a `shot` column in place of `slice`, one line
  per `(volume, shot)`, `time` each shot's excitation; the 2D file is unchanged byte for byte.

---

# Part C: the stack-of-spirals readout

## What is modeled

The echo train of part B with each echo reading one spiral-out interleaf in-plane at one kz
partition, starting at the spin echo. A shot is one interleaf over `ETL` partitions:
`NumberShots = interleaves * kz_segments`, in the canonical order `s = interleaf kz_segments + sz`.

Part C is a separate milestone of the plan, after parts A, B and D are complete: it is the one
part with new numerical machinery (the 2D NUFFT pair, time segmentation, gridding), and it opens
with a feasibility benchmark (one `asl001`-sized volume, timed, the interpolation `L` and the
gridding error reported) whose numbers are recorded before the rest of the part is built.

## Trajectory

An Archimedean constant-density spiral-out, interleaf `s` of `N` (`N` = `interleaves`), in cycles
per field of view, as a function of a normalized parameter `u` in `[0, 1]`:

```
k_s(u) = k_max u exp(i (2 pi n_turns u + 2 pi s / N)),     k_max = nx / 2,   n_turns = nx / (2 N)
```

so turns of one interleaf are `N` cycles/FOV apart radially and the `N` interleaves together are
`1` apart. Time enters through `u(tau)`: constant linear velocity, `u = sqrt(tau/T)`, outside a
centre region, and constant angular velocity, `u = tau / sqrt(tau_c T)`, inside `tau < tau_c`
(continuous at `tau_c`), because the pure square root has infinite speed at the centre. Samples
are at `tau_j = (j + 1/2) dwell`, `j = 0..n_samples`, `n_samples = floor(T / dwell)`. `nx = ny`
is required (a square in-plane matrix); a rectangular one is an error.

**Sampling bound, on the continuous trajectory.** A check on the chords between consecutive
samples is not enough: one interleaf of `nx = 64` sampled 32 times over 32 turns lands every
sample on the x-axis one cycle/FOV apart, passing a chord check with no two-dimensional coverage
(second review). The bound is on the **speed**, so the arc length between samples, and hence any
skipped winding, is bounded. The speed is `|dk/dtau| = k_max u'(tau) sqrt(1 + (2 pi n_turns u)^2)`.
In the centre region `u'` is constant and the speed increases with `tau`; in the outer region
it decreases toward `pi n_turns k_max / T`; the maximum is therefore at `tau_c` on the inside,

```
v_max(tau_c) = k_max sqrt(1/(tau_c T) + 4 pi^2 n_turns^2 / T^2)
```

which decreases monotonically in `tau_c`. The requirement `v_max dwell <= 1` cycle/FOV then has a
closed-form solution, the smallest admissible centre region,
`tau_c = 1 / (T (1/(k_max dwell)^2 - 4 pi^2 n_turns^2 / T^2))`, which exists if and only if
`dwell < T / (2 pi n_turns k_max)`; otherwise the error names that largest dwell time, and if the
solved `tau_c` exceeds `T` the error says the readout is too short for the dwell time. No search
is needed, and `tau_c` is recorded. The radial spacing between interleaves is `1` by construction,
so the bound plus the design gives two-dimensional Nyquist coverage. It is a sampling bound, not
a gradient or slew limit, and the sidecar says so.

Within an echo, the sample at `tau` has `t = tau` from the echo centre (spiral-out starts at the
echo) and `trf = e ESP + tau`, so the in-echo `T2'` decay is `exp(-tau/T2')`, monotonic in `tau`,
and the off-resonance phase is `2 pi fmap tau`: spirals turn off-resonance into blurring rather
than a shift.

## Forward and reconstruction

**Forward**: for slice `z`, compartment `c` and coil `q`, with the same coordinates, Fourier sign
and normalization as the Cartesian forward: positions `(xc, yc)` centred and half-cell registered
in acquired-voxel units (`kspace.rs:560-585`), frequencies `k_j = (kx_j, ky_j)` in cycles/FOV,
the **positive** exponent with each component divided by its matrix size, as the Cartesian
forward's `ky_norm = (ky - ys)/sny` on the simulation grid and its x-stage do
(`kspace.rs:603-612`, `:909-917`), and normalization `1/nvox` (`:849`):

```
K2_{c,z,q}(k_j) = (1/nvox) sum_r amp_q(r) exp(i phi0(r)) m_c(r) exp(i 2 pi f(r) tau_j)
                  exp(+i 2 pi (kx_j xc / nx + ky_j yc / ny))
```

(The first version of this equation had `exp(-i 2 pi k . r)` with `r` in voxels: wrong sign and
wrong scale by a factor `nx`, which a direct-sum test written from the same equation would not
catch. A test therefore evaluates the spiral forward at Cartesian frequency locations and compares
it with the Cartesian forward of the same object, a displaced complex point at oversampling 1 and
2, to `1e-12`.)

where `amp_q` is `signal_scale` times coil `q`'s sensitivity and `phi0` the object phase, both
exactly as the Cartesian forward forms them (`kspace.rs:566-587`). In `class` mode the decay is
part B's line weight per sample, `exp(ln A_e - tau_j/T2_c - tau_j/T2'_c)`, outside the sum. In
`voxel` mode it is inside: the per-voxel complex rate becomes
`rate(r) = i 2 pi f(r) - 1/T2(r) - 1/T2'(r)` with `ln A_e(r)` in the static amplitude, so T2, T2'
and the fieldmap all vary independently. Then part B's steps 2-4, with `p` and the interleaf in
place of `(p, ky)`.

The exact sum is `O(samples x voxels)` per slice. It is the **oracle**, run on small grids in
tests, and not a production path: at `64 x 64 x 30`, oversample 2, six compartments and 32,000
samples per partition it is about `1e11` operations per volume per coil (Codex's estimate), so a
spiral protocol in a build without the `kspace` feature is an error naming the feature.

Under `kspace` it is a **time-segmented type-2 NUFFT**: `exp(rate(r) tau)` is interpolated in
`tau` over `L` segments, `exp(rate tau) ~ sum_l b_l(tau) exp(rate tau_l)`, `L` NUFFTs per slice.
The interpolators are the least-squares ones (Sutton, Noll and Fessler 2003): `tau_l` uniform on
`[0, T]`, and for each `tau` the coefficients `b(tau)` minimizing the error over the tensor
Chebyshev grid (below) of complex rates `rate = -d + i 2 pi f` covering the slice's actual range, `d` in
`[d_min, d_max]` (the decay rates present, `>= 0`) and `f` in `[f_min, f_max]` (the fieldmap's
extent).

A residual of zero on the fitting grid proves nothing between its points (second review: two
nodes fitted exactly at decay rates 100 and 1100 s^-1 miss 600 s^-1 by `0.106`). The error is
therefore **certified for every rate in the rectangle**, not only the grid. A first-order
(Lipschitz) certificate is correct but useless: with `T = 4` ms it needs a grid spacing below
`1.8e-5` s^-1, more than `10^7` points across a 50 Hz fieldmap range and `10^15` with a decay
range (plan review). The certificate is instead high-order, because the error
`e(rate, tau) = exp(rate tau) - sum_l b_l(tau) exp(rate tau_l)` is an entire function of `rate`
whose `n`-th derivative is bounded, for `Re(rate) <= 0`, by `T^n (1 + B)`, `B = max_tau sum_l |b_l(tau)|`.

The fit and the certificate use a tensor **Chebyshev grid** of degree `m` on the rectangle
(`m + 1` Chebyshev points along each axis, decay `[d_min, d_max]` of length `R_d` and frequency
`2 pi [f_min, f_max]` of length `R_f`). For `P` the tensor Chebyshev interpolant of `e` on that
grid, `Lambda_m <= 1 + (2/pi) ln(m + 1)` its Lebesgue constant, and the one-dimensional Chebyshev
remainder `rem(R) = 2 (R T / 4)^(m+1) (1 + B) / (m + 1)!`, every rate in the rectangle satisfies

```
|e| <= |P e| + |e - P e| <= Lambda_m^2 max_grid |e| + rem(R_d) + Lambda_m rem(R_f)
```

with the maxima taken over every sample time of the trajectory, plus a floating-point model term
for rounding, `(Lambda_m^2 + 1)(1 + B)(4 eps (1 + theta) + 2 L eps)` with `theta = max |rate| T`:
forming and evaluating each exponential (an error growing with the phase), the uncertainty of the
measured grid residual (amplified by the Lebesgue factors), and evaluating the interpolant once
more. Rectangles with `theta > 1e4` rad (about 400 kHz over 4 ms) are refused. This is a model
bound, not interval arithmetic, and the tests compare it with exponentials whose phases are
reduced in double-double (amended twice on 2026-10-02: a first rounding term, `4 L eps (1 + B)`,
ignored the phase and was exceeded by `8.2e-15` against `3.8e-15` at a constant 10 kHz, the second
review's case). The least-squares fit is solved through a factored SVD
(one-sided Jacobi on the narrower side), never through an explicit pseudo-inverse: at these
condition numbers (`1e11` and more) applying one loses about `eps / sigma_min`, `4.5e-6` on the
grid, which stopped a 250 Hz range from certifying. `m` is the smallest degree making
the remainder terms below `5e-8`; at `T = 4` ms, a 50 Hz range (`R_f T = 1.26`) and decay to
`1100` s^-1 (`R_d T = 4.4`), `m = 20` suffices, so the grid has a few hundred points rather than
`10^15`. `L` starts at `ceil(T (f_max - f_min)) + 2` and doubles until the bound is below `1e-7`;
past `L = 64` it is an error naming the rate rectangle, not a silent loss of accuracy. In `class`
mode the decay is outside the segmentation and the rectangle is the frequency interval alone
(`rem(R_d)` drops, one Lambda factor drops). `L`, `m` and the certified bound are recorded. The
certification's cost (`(m+1)^2` rates times the sample count times `L`) is part of the
feasibility benchmark, separately for `class` and `voxel` mode.

**Reconstruction** (amended 2026-10-02 after the feasibility benchmark; the first version was
gridding alone): density-weighted least squares, per coil and partition, by a fixed linear
iteration. The image `x` on the `nx x ny` grid approximately minimizes
`sum_j w_j |(A x)_j - d_j|^2`, `A` the forward of an object on the acquired grid (the type-2 NUFFT
in the convention above, `1/(nx ny)`), `A^H` its adjoint (the type-1 NUFFT with the opposite sign,
deapodized), `w_j` the density weights. The solver is the **Chebyshev semi-iteration** on the
normal equations `A^H W A x = A^H W d` from `x = 0`, over the eigenvalue interval
`[lambda_hi / 100, lambda_hi]`, 80 iterations, `lambda_hi` 1.1 times the largest eigenvalue
estimated by power iteration from a pseudo-random start (fixed seed) run to convergence: every
coefficient depends only on the trajectory and the weights, so the reconstruction is a fixed
**linear** operator on the samples, which the linearity identity requires (conjugate gradients,
whose step sizes depend on the data, is not). It is a **regularized approximate inverse**, not the
least-squares solution: components with eigenvalues below `lambda_hi / 100` are only partly
recovered. Its residual polynomial is bounded by 1 on `[0, lambda_hi]`; power iteration estimates
the top of the spectrum from below and certifies nothing, so every reconstruction checks that its
normal-equation residual did not grow (it can only grow if an eigenvalue lies above `lambda_hi`)
and panics if it did. The supported domain, and what the tests assert: objects band-limited to
`|k| <= 0.625 k_max`, anywhere in the field of view, to `1e-2` of peak. The parameters were chosen
on a sweep (plan Measurements): `[lambda/30, lambda]` with 40 iterations met `1e-2` at the centre
but not at the edge (`1.5e-2`, the second review's case); `kappa = 300` meets it with more margin
but doubles the image noise, `kappa = 100` with 80 iterations meets it (worst `6.8e-3`) at 1.6
times the Cartesian noise.
The density weights are Pipe and Menon's in operator form, `w <- w / |A A^H w|` (10 iterations,
fixed, recorded), normalized so that a constant object with no off-resonance and no decay grids to
its Cartesian value at the image centre (the scale is common to every weight, so it does not move
the least-squares solution). Then the existing Roemer combine, which divides by the same
sensitivities the forward applied. The trajectory, its density weights, `lambda_hi` and the NUFFT
plans depend only on the protocol and are computed once per series. GRAPPA, partial Fourier,
ghosting and spikes have no spiral meaning in this model and are refused with `type = "spiral"`;
the in-plane window multiplies the samples before the reconstruction, by the same radial function
`KspaceWindow` defines (`kspace.rs:120-141`), so the image approximates the windowed object as
the Cartesian path's does.

Why not gridding (the benchmark, plan Measurements, Task 12): the designed interleaves are
exactly Nyquist-spaced radially, and at that spacing gridding reproduces a band-limited object to
4 percent of peak and a uniform one to 30-50 percent, whatever the density weights (kernel and
operator Pipe-Menon and the trajectory's exact annulus areas all give 1-6 percent on the
band-limited object). At oversampling 1 the data of an object on the acquired grid are exactly
`A x`, so the least squares recovers it as the iteration converges: the Chebyshev iteration
above reaches, on exactly band-limited objects (`|k| <= 0.625 k_max`), `1.4e-4` (centre),
`5.7e-3` (edge) and `6.8e-3` (corner) of peak at `32 x 32`, and `4.3e-5`, `2.7e-3` and `3.5e-3`
at `asl001`'s `64 x 64`; on the uniform object, under `1e-4` at the centre and `1.0e-3` to
`1.7e-3` over the FOV. At higher oversampling, what the
spiral samples of the sub-voxel structure outside the reconstruction's band is what it cannot
represent, as with the Cartesian path's truncation.

`nufft.rs` today is a 1D type-1 transform only (`nufft.rs:5`, one use site, `kspace.rs:661`). It
gains a 2D type-1 and type-2 pair on the same exponential-of-semicircle kernel, with the
direct-sum test pattern the 1D one already has (`matches_the_direct_sum`, `nufft.rs:141`).

**Noise** is per complex sample with SD `sqrt(noise_variance / (nx ny))`, as Cartesian; after
the density-weighted least squares the image noise is not that of a Cartesian acquisition with
the same `noise_variance`, and it is measured and recorded (the test reports the ratio), not
asserted equal.

## Inputs

Overlay `[readout]` for spirals:

| Key | Default | Meaning |
|---|---|---|
| `interleaves` | none: required | `N` |
| `spiral_readout_time` | none: required | `T`, ms |
| `dwell_time` | sidecar `DwellTime`, else required | s |
| `kz_segments`, `kz_order`, `echo_spacing`, `refocusing_time`, `refocusing_flip_angle` | as part B | |

The two required values have no defensible default (product spirals vary between vendors and
releases), and a missing one is an error naming it, as the main spec does for the PASL bolus
duration. Checks, on the actual intervals:

- **The spiral ends before the next refocusing pulse**: it starts at the echo, and the next pulse
  is centred `ESP/2` later and reserves `refocusing_time` about its centre, so
  `T + refocusing_time/2 <= ESP/2`. (Checking `T + refocusing_time <= ESP` would admit a spiral
  running through the pulse: `asl001`'s `ESP` near 10.5 ms with an 8 ms spiral, Codex's example.)
- **The train fits in the repetition**: `t_exc + ETL ESP + T <= TR`.
- The sampling bound above; `NumberShots`, when given, equals `interleaves * kz_segments`, and
  when absent it **is** `interleaves * kz_segments` (part B's "absent means 1" is GRASE's rule;
  for spirals the interleaves are shots by construction). `asl001` with eight interleaves is
  therefore an eight-shot acquisition with a 39.1 s volume duration, which the sidecar states.

`EchoTime` is the time to the start of the kz-centre echo's spiral (the k-space centre), so
`ESP = EchoTime / e_c`. `PhaseEncodingDirection`, `TotalReadoutTime` and `EffectiveEchoSpacing`
have no spiral meaning and are refused.

---

# Part D: segmentation (per-shot factors and per-shot motion)

A volume made of shots is sensitive to anything that differs between shots: that is the defining
artifact of segmented 3D ASL, and the model has two such things already.

## Per-shot physiological factors

P4's physiological factors are per row (label) and per slice readout time (tissue). In 3D they
are per shot: shot `s` of volume `v` (canonical order) labels over its own window and excites at
`row_start[v] + s TR + t_exc`. So `row_start` advances by `NumberShots * TR` per volume
(`protocol.rs:1480-1485`, today one `TR`), P4's `PhysioLine` is per `(volume, shot)`, and the
processes are generated through the last shot: the horizon is the last volume's last shot's
excitation plus one `TR`, not today's last `row_start` plus one `TR` (`series.rs:352`), which
for four-shot `asl005` would end 12.7 s early and clamp the drift (Codex).

The factors become `line_weights`, `W_c(p, ky) *= f_c(v, shot(p, ky))`, and they need the
compartments to separate what P4 scales differently. P4 puts the extravascular label into the
**tissue** compartment (P4 "Where the parts go"; `series.rs:769`), so a tissue compartment of a
label row holds `T - E`. In 2D both factors are applied to the images before the merge, so
`f_t T - f_l E` is formed correctly. A per-compartment line weight cannot form it: it would give
`f_t (T - E)`, leaving the extravascular label unmodulated (Codex's blocker). So in 3D, when
`[physio]` and part A are both on, the extravascular label gets its own compartment group,
`[3K..4K)` after P4's arterial group (or `[2K..3K)` without part B), with the tissue's relaxation
(T2, T2', T1 for the EPG), the label factor as its line weight, and the tissue group carrying
`T` only. The 2D layout is unchanged; the sidecar's `CompartmentOrder` names the group. A
negative control puts the extravascular label back into the tissue group under physio and
asserts the linearity identity fails.

The result is the shot-to-shot modulation of k-space and the ghosting along the segmented axis
that real segmented 3D ASL shows. Without `[physio]` there is no extra group, and without
`[physio]` or a dropout event `line_weights` is `None` and nothing changes.

## Per-shot motion

P3's `[motion] within_volume` (`dropout_rate`, `severity`, `jump_mm`, `jump_deg`) means multiband
shot events in 2D. In 3D a shot is a readout segment, and the same keys mean the same thing at the
shot level, in the canonical shot order: an event at shot `s` adds a pose jump that persists for
later shots of the volume and attenuates shot `s`'s lines by `1 - severity`
(`DropoutLaw::Uniform`). The two effects travel separately, as they do in 2D
(`motion.rs:394-409` attenuates independently of the geometry): `shot_images` carries, per
volume, the moved compartment images of each shot whose pose differs from the volume's, and the
forward runs once per distinct pose and keeps only that pose's shots' lines; the attenuation is a
**shot gain**, a per-`(volume, shot)` scalar applied to every compartment of that shot's lines,
composed multiplicatively with the physiological factors when both exist. `line_weights` is
therefore present whenever physio or a dropout event is, and an event with zero jumps still
attenuates (a test with `severity = 1`, zero jumps and physio off asserts the selected shot's
acquired lines are exactly zero). The cost is the
number of distinct poses per volume times the static forward. `MultibandAccelerationFactor` has
no role in 3D, so the 2D requirement `> 1` (`protocol.rs:801-804`) is replaced, for 3D, by
`NumberShots > 1`. The ground truth is P3's: `desc-motionEvents_gt.tsv` with `shot` meaning the
readout segment and an empty `slices` column.

Per-shot motion moves images, not the fieldmap, the T2, T2' or T1 maps, or the coils, as P3 states
for 2D.

---

# Division of labor

| Concern | Where |
|---|---|
| Gradient-echo decay and phase, `EchoFormation` | `mrsim-acq` `kspace` (and the oracle) |
| Echo-train timing, encoding order, the line-timing table | `mrsim-acq` `readout` |
| EPG echo amplitudes | `mrsim-acq` `epg` (new, pure std) |
| z-DFT, 3D noise and seeds, per-partition reconstruction | `mrsim-acq` `kspace` |
| Spiral trajectory, 2D NUFFT pair, time segmentation, density compensation, least-squares reconstruction | `mrsim-acq` `readout`, `nufft`, `tseg`, `spiral`, `grid_recon` |
| Gradient-echo signal equation, steady state under suppression | aslscan `mrsignal`, `longitudinal` |
| 3D timing from BIDS, `[readout]`, activation and refusals | aslscan `protocol` |
| Zero slice offsets, per-shot `row_start`, line weights, shot images | aslscan `series` |
| Sidecar blocks, standard keys, ground truth | aslscan `bids` |

`mrsim-acq` stays signal-model-free: it receives compartment images, relaxation times and
per-line scalars, and knows nothing of labeling, kinetics or physiology.

---

# Testing and verification

**mrsim-acq**

- **Bit identity**: the bit-pinned tests listed under "Scope" pass unchanged; the new golden
  baseline (full `simulate_slice` output of every oracle case, both feature sets, written before
  any P5 change) matches bit for bit; the line-timing table equals `line_times` bit for bit on
  every oracle case; the TRXScan P0 gate passes on the named branch under both feature sets.
- **Gradient echo**: the extended literal oracle at `1e-10` of peak, `Gradient` cases included.
  A static-phase test that does not assume the k-space centre is sampled at the echo: with a
  uniform fieldmap `f` and no decay, the `Gradient` and `Spin` k-spaces of the same object differ
  by exactly the factor `exp(i 2 pi f TE)` on every acquired sample (the distortion term is shared,
  whatever the line times), to `1e-12` relative on the internal `f64` k-space. Magnitudes at two
  `TE`s differ by `exp(-dTE/T2*)` on every line, to `1e-12`, on the same internal k-space.
- **3D wiring**: with relaxation off, no fieldmap and **no ghosting**, the 3D GRASE output equals
  the 2D output of the same object slice by slice to `f32` storage precision (the z-DFT round trip
  is the identity), for every combination of `ky_segments`, `kz_segments` and `kz_order`,
  multi-coil and GRAPPA included. Ghosting is excluded because segmented GRASE and 2D EPI have
  genuinely different polarity patterns (`++--...` against `+-+-...` for two ascending
  segments), which no z transform removes.
- **3D ghosting**: with a nonzero ghost offset, the GRASE k-space equals an independent direct sum
  that applies polarity by acquisition index within each segment, in both phase-encode
  directions, to `1e-10` of peak on the internal k-space; the same direct sum with polarity by
  `ky % 2` must differ by more than `1e-3` of peak (the polarity table is exercised).
- **Distortion direction**: a point off centre with a known uniform fieldmap moves the documented
  way for `j` and `j-` in GRASE, as P1's test does in 2D.
- **Through-plane PSF**: a one-partition object with uniform T2 reconstructs, on the internal
  `f64` image, to the analytic profile (the inverse z-DFT of the kz modulation in encoding order)
  to `1e-9` of peak; centric and linear give their different, separately computed widths.
- **EPG**: `b = 180` gives `exp(-e ESP/T2)` to `1e-12` relative; for `b = 130` the amplitudes
  agree with an independent isochromat Bloch simulation (rotation matrices, 2001 isochromats
  across one crusher cycle) to `1e-3`, including the first-echo pseudo-steady-state; extreme T2
  (`1e-3` ms to infinity) gives finite outputs.
- **Voxel vs class**: a `Map` that is constant per label reproduces `class` mode to `1e-6` of
  peak, as P1's cross-check does in 2D, and the run times of both are reported. A label with
  uniform T2 and T2' but varying T1 under `b = 130` resolves to `voxel` under `auto` and is an
  error under an explicit `class`.
- **3D noise**: with full sampling, at the same `noise_variance`, the image noise SD over a
  signal-free region is `1/sqrt(nz)` of the 2D value within 3 percent (64 x 64 x 16, 20 seeds);
  under GRAPPA the ratio is reported only.
- **Segmentation**: with relaxation off, no fieldmap and no ghosting, `NumberShots` does not
  change the image;
  per-shot line weights alternating `w1`, `w2` across two interleaved ky segments, on an object
  confined to the central half of the field of view in the phase-encode direction (so the ghost
  at FOV/2 does not overlap it), produce a ghost of relative amplitude `|w1 - w2| / (w1 + w2)` to
  `1e-9` on the internal `f64` image.
- **Spirals**: the 2D NUFFT pair against direct sums to `1e-10`; the adjoint identity
  `<A x, y> = <x, A^H y>` to `1e-10`; the time-segmented forward against the exact sum on a
  complete small trajectory (all samples, `32 x 32`, oversample 2) to `1e-6` of peak, with
  independent maps of the fieldmap (50 Hz range), T2 and T2', and with a nonconstant object phase,
  multi-coil sensitivities and `signal_scale`; the reconstruction of an object exactly
  band-limited to `|k| <= 0.625 k_max`, with no off-resonance, reproduces it to `1e-2` of peak at
  the centre, the edge and the corner of the field of view (declared before; gridding missed it at
  `4e-2`, and the spec was revisited, not the number; the edge and corner cases are the second
  implementation review's); a uniform object reconstructs to its Cartesian value at the image
  centre to `1e-3` (the first version said everywhere; gridding missed both, and the FOV error is
  reported); the shots select exactly their samples (a zeroed shot, a shot set with doubled images,
  on the acquired samples with unequal interleaf and kz-segment counts); the reconstruction is linear
  (the sum of two data sets' reconstructions equals the reconstruction of the sum to rounding);
  with uniform off-resonance the reconstruction equals the reconstruction of the exact-sum
  samples to `1e-6`; the sampling bound accepts the designed trajectory, rejects a
  dwell time one percent above its bound, and rejects the second review's counterexample (one
  interleaf, `nx = 64`, 32 samples over 32 turns, which a chord check passes); the spiral forward
  at Cartesian frequency locations equals the Cartesian forward (part C). The time-segmentation
  certificate is stress-tested on decay ranges as well as fieldmap ranges (the second review's
  `100` to `1100` s^-1 case among them), and the certified bound is compared with the actual error
  against the exact sum at rates between grid points, which must not exceed it, and with
  exponentials whose phases are reduced in double-double (a constant 10 kHz, a 950-1050 Hz range).

**aslscan**

- The simasl form of `tissue_ge` and `blood_ge` against simasl fixtures at `1e-12`, with simasl's
  transverse factor divided out by the fixture generator (P1's method); the spoiled form against
  simasl fixtures generated with `T2` small enough that `E2 < 1e-16` (where the two forms agree);
  the GE timeline: `a = 90` reproduces P3 bit for bit; `a != 90` equals the timeline iterated
  from `Mz(0) = 0` until successive starting values differ by less than `1e-15 M0` (the
  multiplier `cos(a) Er A` has magnitude below 1, so this terminates; the iteration count is
  reported), to `1e-12`; with no events it agrees with the spoiled closed form to `1e-12`.
- The extravascular-label group under 3D physio, with its negative control (part D).
- The linearity identity holds under `"ge"`, for GRASE and for spirals (complex images, the three
  runs sharing seeds and poses), with its negative controls.
- `protocol`: every activation and refusal above, each with the exact error; the `bids-examples`
  3D sidecars parse with the overlays below and the P5 refusals are gone.
- `compat` under `"ge"`: a P2-style voxelwise benchmark against simasl with noise off.
- Byte identity: `tools/regress_identity.sh`, with `aslscan` at `p4-complete` and `mrsim-acq` at
  its last pre-P5 commit, both feature sets, all cases (P4 combinations included) identical; a
  deliberate one-ulp perturbation of the acquisition stage, built once, must make it report a
  difference (so the gate is shown not to compare the new crate against itself).

# Acceptance criteria

1. With no P5 input, TRXScan, `mrsim-acq`'s pinned tests and aslscan's regression cases are
   byte-identical (Scope).
2. A 2D `"ge"` protocol simulates with the spoiled model, which matches its closed form (and
   simasl's equation where `E2` vanishes) at `1e-12`; under `[compat] asldro = true` a `"ge"`
   protocol matches simasl's own equation at `1e-12` and its compat benchmark passes.
3. `asl005` (PCASL, 3D GRASE, 4 shots, 130-degree refocusing, suppression) simulates from its real
   sidecar with a committed overlay that pins the phantom crop and the in-plane matrix (so that
   `ny` divides by the segment count; the documented phantom's extent gives `69`, which does not,
   Codex), the segmentation, `[m0] repetition_time`, and `[readout] phase_encoding_direction`
   (the sidecar lacks one and GRASE requires it); the train passes every timing check without
   any check relaxed, and the output validates with no errors. The overlay and every resolved
   value it implies are written into the plan before implementation.
4. A GRASE PASL case built from `asl003`. As given, `asl003` cannot parse under rules this spec
   does not change: four of its PLDs (0.3, 0.3, 0.6, 0.6 s) precede its 0.7 s bolus cutoff, which
   aslscan refuses for PASL (`protocol.rs:922`), and its suppression pulses at 0.15 and 0.2 s
   precede the bolus, which the global-bolus model refuses (`protocol.rs:1291`). The acceptance
   case is a committed derivative: the `asl003` sidecar with those rows removed (with
   `aslcontext.tsv` and every per-volume array cut to match), P4's bolus-position model with
   `pulse_region = "global"` (which accepts PASL pulses at any time, P4 part D), a pinned matrix,
   segmentation and line spacing that describe a feasible train; it simulates and validates.
   At 64 phase-encode lines the sidecar's `EffectiveEchoSpacing` (0.5 ms, so 1 ms actual lines at
   two interleaved segments) cannot give a feasible block, and an overlay `line_spacing` must
   agree with it; the feasible train the second review worked keeps it and shrinks the matrix
   instead: `ny = 20`, `nz = 30`, a ten-line block of 1 ms lines that with the 2 ms reserve fits
   `ESP` (12.42 ms), the train ending at 3.378 s within the 3.5 s repetition. Every field the
   derivative changes is listed in its `SOURCES.md` entry with the reason. Lifting the PASL
   readout-before-cutoff refusal is not P5 work.
5. `asl001` (PCASL, 3D spiral, `m0scan` and `deltam` rows) simulates with a committed overlay
   supplying `interleaves`, `spiral_readout_time`, `dwell_time`, a square in-plane matrix, and
   `[m0]` as needed, and validates. (Part C's milestone.)
6. The through-plane PSF, EPG, 3D-noise, segmentation and spiral tests pass, and the measured
   numbers (and the feasibility benchmark's) are recorded in the plan.

# Decisions deferred

- **3D gradient-echo readouts** (segmented 3D EPI, 3D GRE): a different train (no refocusing, so
  T2\* accumulates across the train). Not scheduled.
- **3D inversion recovery**: P3's IR composed with an echo train. Not scheduled.
- **Slab profile, kz aliasing and through-plane oversampling**: an ideal slab equal to the field of
  view today; a real slab's edge attenuation and wrap need z oversampling of the simulation grid,
  which `mrsim-acq` forbids (`kspace.rs:1357`).
- **Out-of-plane acceleration (CAIPI) and kz partial Fourier**: refused.
- **Longitudinal evolution during the echo train**, using the EPG's `Mz` instead of `Mz = 0` at the
  excitation.
- **A per-residue NUFFT** for interleaved GRASE timing with more than two ky segments, and the
  per-compartment NUFFT gate the main spec already defers.
- **Variable-density and spiral-in trajectories, gradient delays, and coil-coupled (SENSE) spiral
  reconstruction**.
- **z-dependent coil sensitivities**.
- **PASL readouts before the bolus cutoff** (`asl003`'s early delays): the refusal at
  `protocol.rs:922` is a P1 rule about the kinetics, not a readout question, and stays.

# Risks

**`mrsim-acq` changes, and TRXScan depends on it.** Mitigated by the rule under Scope (new
behavior only behind new non-default values, the default path calling the old code), by
re-running the P0 gate on the named TRXScan branch under both feature sets, by a golden baseline
of full forward outputs written before any change, and by an aslscan regression gate that pins
both repositories and is shown to detect a deliberate change. The one generalization on the
shared path, the line-timing table, is pinned bit for bit.

**There is no oracle for any 3D effect.** simasl has no readout. The mitigations are closed forms
(the PSF, `b = 180` EPG, the segmentation ghost), an independent Bloch simulation for the EPG, the
exact sum for the spiral NUFFT, and identities (3D equals 2D when nothing distinguishes partitions;
linearity). A vendor-reconstructed phantom scan would be the real check and is not available.

**BIDS underdetermines the readout.** The echo train, segmentation and spiral design are not in
any sidecar; this spec derives what it can (`ESP` from `EchoTime` read as the k-space-centre
time, the actual line spacing from the effective spacing or, as a recorded lower bound, the
dwell time), reads `FlipAngle` as the refocusing angle on evidence rather than definition, and
records every derivation and default. The review's arithmetic shows the consequence: the
`bids-examples` sidecars alone do not describe trains that fit, so every acceptance case carries
an explicit overlay, and a real user will usually need one too. A dataset whose `EchoTime` or
`FlipAngle` means something else will be simulated with the wrong train, and the sidecar is
where a user will see why.

**Part C may not be feasible at the stated accuracy and cost.** It is a separate milestone that
opens with a benchmark; if the time-segmented forward needs more than `L = 64` or the gridding
misses its declared tolerance, the spec is revisited before part C continues, and parts A, B and
D stand on their own. (It happened: gridding missed its tolerance, and the reconstruction became
the linear least squares of part C; see the review record.)

# Review record

Codex adversarial design review, 2026-10-01: 4 blockers, 18 majors. Every finding was checked
against the source and the fixtures; all were valid and are applied above. In summary:

- **Blockers.** The aslscan identity gate built both binaries against the live `mrsim-acq`
  (path dependency), so it could not see a P5 change: it now pins both repositories and is shown
  to detect a deliberate change. `asl003` cannot parse (PASL delays before the cutoff, pulses
  before the bolus): its acceptance case is a committed derivative. The GRASE fixtures fail the
  spec's own fit checks under the BIDS reading (`asl003`: 32 lines of 1 ms against `ESP` near
  12 ms): the checks stand and the acceptance cases carry full overlays. Per-compartment physio
  weights cannot reproduce P4's exchange split (the tissue compartment holds `T - E`): the
  extravascular label gets its own group in 3D under physio.
- **Majors.** The TRXScan gate names its branch, and the `Acquisition` literal sites are listed.
  The dwell-time fallback gives the actual, not effective, spacing; 3D uses BIDS's
  `TotalReadoutTime / (ny - 1)`. The line table carries polarity by acquisition index and the
  `j`/`j-` traversal. `EchoTime` is the k-space-centre time, so `ESP` subtracts `t(ky_c)`. The fit
  checks use the actual RF positions (midway between echoes) and the last sample, which also
  fixes a spiral check that admitted a readout through a pulse. The spiral sampling check runs on
  the discrete trajectory, centre included, and the trajectory gains a constant-angular-velocity
  centre. The gradient-echo model is spoiled everywhere and simasl's `E2` form is compat-only
  (`E2` is 3.7 percent at a CSF-like T2). Class mode below 180 degrees requires uniform T1.
  Decay is one exponent (separate factors gave `0 * inf`). Shots have one canonical order, and
  physiology is generated through the last shot. The 2D physio TSV keeps its bytes. The M0
  sidecar is built from its own resolution. The spiral forward is the complete per-coil
  operator, and voxel-mode T2 enters the segmented rate. Tests compare against actual sample
  timing and on internal `f64` outputs, a golden baseline replaces "bit-pinned" tests that
  compared two current runs, the gridding tolerance is declared now, and the steady-state test
  iterates to convergence. Part C is a separate milestone behind a feasibility benchmark, the
  direct spiral sum is an oracle only, and the NUFFT gate is evaluated on the table rather than
  excluded.

**Cost.** `voxel` mode multiplies the 3D forward by `ETL`, per-shot motion by the number of
distinct poses, and spirals by the segment count `L`. All three are measured by the tests and
reported, and `class` mode remains the recommended configuration.

Second Codex adversarial design review, 2026-10-02: 2 blockers, 6 majors, 1 minor, all checked
against the source and valid. It confirmed the first pass's fixes to the gradient-echo algebra,
the class-mode factorization, the RF-fit inequalities, the `1/sqrt(nz)` scope, the identity
strategy, the extravascular group and the shot chronology, and worked feasible trains for the
`asl003` derivative, `asl005` and `asl001`.

- **Blockers.** The spiral forward equation had the wrong Fourier sign and a factor-`nx` scale
  error against the Cartesian convention (`kspace.rs:603-612`): it is now written in the
  Cartesian convention, and a test compares the spiral forward at Cartesian frequencies with the
  Cartesian forward. The chord-based sampling check passed a trajectory that skipped whole turns,
  and its bisection assumed a monotonicity that does not hold: it is replaced by a speed bound on
  the continuous trajectory, whose maximum is monotone in the centre region's length and gives
  that length in closed form.
- **Majors.** Shot dropout had no carrier without physio: it is a shot gain in `line_weights`,
  composed with the physio factors. The 3D-equals-2D test cannot hold with ghosting (different
  polarity patterns): it runs without ghosting, and ghosting has its own direct-sum test. The T1
  map has a defined contract (M0-weighted rate mean, background, scanner frame, exported effective
  map, named mixture approximation). The gradient-echo fixed point assumed one repeated
  preparation: series whose rows differ propagate the state through the rows in order. `asl005`
  had no route for its phase-encode direction: `[readout] phase_encoding_direction`. The
  time-segmentation accuracy was checked only on its own fitting grid: the error is now certified
  over the whole rate rectangle by a derivative bound.
- **Minor.** The gradient-echo acceptance criterion is split into the spoiled model (closed form)
  and compat (simasl's equation).

The Codex review of the implementation plan (2026-10-02) found one defect in this spec: the
first-order time-segmentation certificate was correct but needed a grid spacing below `1.8e-5`
s^-1 (`10^15` points with a decay range). It is replaced by the high-order Chebyshev certificate
in part C, with the class and voxel cases separated in the feasibility benchmark.

Amendment after the feasibility benchmark (plan Task 12, 2026-10-02; decided by the user, not
yet Codex-reviewed). The time segmentation passed (`L = 6` at `asl001`, both modes; the bound held
on every stress case) and so did the cost (one `asl001` volume: 2.8 s class, 56 s voxel), but
gridding missed the declared tolerances (band-limited `4e-2`, uniform `0.3` to `0.5`) for a
reason no density weighting fixes: the designed trajectory is exactly Nyquist-spaced radially.
The reconstruction is now density-weighted least squares by a fixed Chebyshev semi-iteration
(linear in the data, as the linearity identity needs; conjugate gradients was measured first and
is accurate but data-dependent); the band-limited tolerance is unchanged, the uniform one is
stated at the image centre, and linearity is tested. The certificate gained a rounding term, and
the least-squares fit is required to be solved in factored form.

Codex adversarial implementation review of milestone C and of the amendment above, 2026-10-02:
3 majors, 4 minors, each checked against the code and found valid. The rounding term ignored the
phase (exceeded at a constant 10 kHz) and is now the model term above, with a phase limit and a
double-double oracle test. `lambda_hi` was claimed as an upper bound; power iteration only
estimates it, so the start is now pseudo-random, the iteration runs to convergence, the claim is
withdrawn and every reconstruction checks that its residual did not grow. The band-limited claim
held only for centred objects (`1.5e-2` at the edge with 40 iterations over `[lambda/30, lambda]`);
the reconstruction is now described as a regularized approximate inverse, its parameters chosen on
a sweep over objects at the centre, edge and corner (80 iterations, `kappa = 100`), and the test
covers all three. The shot test now asserts on the acquired samples. The aslscan minors (an
overridden `PulseSequenceType`, a non-string `PhaseEncodingDirection` with a spiral, the separate
M0's `DwellTime` provenance) are fixed in aslscan.
