# P2: ASLDRO compatibility mode and voxelwise benchmark — implementation plan

**Goal:** Implement the P2 addendum (`docs/specs/2026-09-24-p2-asldro-compat-design.md`): a
`[compat]` configuration of `aslscan` that pins its acquisition to what simasl can express, a
Python driver that runs both simulators on the same phantom and protocol and compares them
voxelwise, five benchmarks with recorded numbers, and the motion-convention conversion that P3
deferred to P2.

**Architecture:** Small, opt-in additions to `protocol`, `resample`, `series` and `bids`; a
driver in the `simasl` environment that owns every simasl-specific translation (parameter
names, the pose conversion, the archive layout) so that `aslscan` learns nothing about simasl
beyond `[compat]`. **`mrsim-acq` is not modified**: `do_relaxation`, `noise_variance` and the
same-size transform pair at `oversample = 1` already exist.

**Spec:** the P2 addendum; the main spec and the P3 addendum where it is silent. P3 is complete
at `aslscan` tag `p3-complete`. The Codex adversarial review of the addendum ran on 2026-10-01
(11 findings, all verified against the source and applied to the addendum and this plan): the
box weights needed a corner offset, not only a new affine; an IR series' m0scan row is IR in
simasl; the two noise references differ fivefold in variance; B's and D's in-plane interior
masks keep through-plane and post-motion mixing (27% and 31% of peak), replaced by a 3D pure
mask; compat must refuse unequal slice timings; simasl's kinetic overrides are
`parameter_override`, not series keys; the driver must require a positive identity source
affine; E runs on both phantoms; the crop needs a packed ground truth; C's seeds must differ
beyond bit 0; the ASL default SNR is 100.

## Global constraints

- Everything in the P1 and P3 plans' global constraints still holds.
- **P1 and P3 outputs are unchanged when `[compat]` is absent.** The default `grid_origin` is
  `corner`, `do_relaxation` stays on, `noise_variance` comes from the overlay, and every P1/P3
  test passes unchanged.
- **Compat overrides nothing silently.** Every pinned value is checked against what the
  overlay or sidecar set explicitly, and a conflict is an error naming the key, the requested
  value and the pinned value.
- **The driver never resamples either output.** It checks the two affines and dimensions agree
  and errors otherwise; a comparison after a resample would measure the resample.
- **Numbers, not adjectives, in the plan's Measurements section**, per benchmark, per phantom.
- Commit per task with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; tag
  `p2-complete` at the end.

## File structure

| Path | Change |
|---|---|
| `src/resample.rs` | `GridOrigin { Corner, VoxelCentre }`; `acquisition_grid` takes it |
| `src/protocol.rs` | `[compat]` overlay, `CompatSpec`, the pinned-value and refusal checks |
| `src/series.rs` | compat: relaxation off, per-voxel `exp(-TE/T2)`, SNR to noise variance, fieldmap refusal, `SeriesOutput` compat facts |
| `src/bids.rs` | `AslscanSimulation.Compat` block |
| `src/bin/aslscan.rs` | `--compat-asldro` |
| `tools/compat_asldro.py` | the driver: simasl replication, parameter translation, pose conversion, metrics, report |
| `tools/test_compat_asldro.py` | pure-numpy tests of the conversion and translation (run in the `simasl` env) |
| `tests/end_to_end.rs` | linearity under compat; an env-gated benchmark A on the crop |
| `work/compat/` | gitignored: converted phantoms, archives, reports |

---

### Task 1: `resample::GridOrigin` and `protocol` `[compat]`

`resample` (pure std): `pub enum GridOrigin { Corner, VoxelCentre }` and
`acquisition_grid(phantom, voxel_mm, matrix_override, origin)`. `Corner` is P1's rule
(`m[a][3] = o - 0.5 p + 0.5 v`); `VoxelCentre` is `m[a][3] = o`, acquisition voxel 0 centred on
phantom voxel 0, which is what simasl's `transform_resample_affine` produces
(`resampling_translate_affine` from the source origin). Dimensions are unchanged. The box
weights gain the corner offset: `axis_weights_offset(.., offset)`, `Resampler::with_offset`,
and `corner_offset(src, dst)` deriving it from the two grids (exactly 0.0 for P1's corner
grids, so `Resampler::new` is `with_offset(.., [0; 3])` bit for bit); a target cell before the
source's near edge is a partial or empty cell, as one past the far edge is. Tests: both origins
on a signed affine and their offsets; the 3 T phantom's dimensions `[197, 233, 189]` with voxel
`[197/64, 233/64, 189/12]` give `[64, 64, 12]` under the `ceil(extent / voxel - 1e-9)` rule
(each quotient is exactly representable); near-edge partial cells; offset 0 equals P1.

`protocol`: an overlay table `CompatOverlay` with `asldro: Option<bool>`,
`desired_snr: Option<f64>` and `grid_origin: Option<String>`; on the protocol,
`compat: Option<CompatSpec>` holding `desired_snr` and `grid_origin` (`Some` only when
`asldro = true`), plus `Protocol::grid_origin` for the non-compat case (default `Corner`;
`grid_origin` may be set without `asldro`). With `asldro = true`:

- Pinned, checked against the overlay's explicit `Option`s: `oversample = 1`,
  `partial_fourier = 1`, `n_coils = 1`, `ghost_offset = 0`, `n_spikes = 0`,
  `eddy_strength = eddy_quad = eddy_phase = 0`, `window = "none"`, `signal_scale = 1`,
  `noise_variance = 0`. A conflicting explicit value is an error naming the key, the value and
  the pinned value; an absent key takes the pinned value.
- Refused from the sidecar: `ParallelReductionFactorInPlane > 1`,
  `MultibandAccelerationFactor > 1`, `BackgroundSuppression: true`, `M0Type: Separate`
  (naming the m0scan row form), and a `SliceTiming` whose entries are not all equal; from the
  overlay: `[motion.within_volume]`. Each names simasl as the reason.
- `desired_snr` (finite, `>= 0`; 0 means none) is accepted only with `asldro = true`.
- `grid_origin` defaults to `voxel-centre` under compat and `corner` otherwise; the strings
  `"corner"` and `"voxel-centre"` are the only ones.

`Protocol::acquisition` sets `do_relaxation = false` and the pinned values when compat is on.
Tests for each rule, plus a compat protocol that parses cleanly with an otherwise empty overlay.

- [x] Commit: `feat: resample grid origin and the [compat] protocol table`.

### Task 2: `series` under compat

- `acquisition_grid(..., p.grid_origin)`, and both resamplers built with
  `corner_offset(phantom, acq_grid)` (the simulation grid shares the acquisition grid's corner).
- A phantom with a fieldmap is refused under compat, naming simasl.
- Under compat an included m0scan row takes the series' signal equation (IR for an IR series),
  with no label; outside compat it stays the P3 spin-echo readout.
- The per-voxel factor `te_factor(i) = exp(-TE / t2[i])` with `exp(0) = 1` at `t2 == 0`
  (simasl's `np.divide(where=t2 != 0)`), applied inside the tissue closures and the blood
  closure so tissue and blood of a voxel share it; `1.0` when compat is off, so the P1/P3 paths
  are untouched bit for bit (multiplying by a literal `1.0` is exact, but the closure is
  bypassed anyway when compat is off, to keep the code paths separate).
- `noise_variance = (signal_scale * mean(M0_acq over M0_acq != 0) / desired_snr)^2` from
  `r_acq.mean(&ph.m0)`, set on the `Acquisition` before the one call; `None` SNR leaves 0.
  `SeriesOutput` gains `compat: Option<CompatFacts>` carrying the noise variance, the M0
  reference mean and the grid origin.
- The separate M0 scan under compat: simasl has no such thing (its M0 is an `m0scan` row); a
  compat protocol with `M0Type: Separate` is refused in `protocol` (Task 1) naming the row form.

Tests: on the crop at the identity grid, a compat run's signed real image equals the closed
form `(tissue + blood) * exp(-TE/T2)` per voxel (`tissue_se`/`tissue_ir`, `delta_m`, the
zero-T2 guard) to float32 precision, for control, label and m0scan rows, SE and IR; the m0scan
row of a non-compat IR series is still spin echo; the voxel-centre grid at 2 mm puts a
single-voxel M0 point source where the affine says; the noise variance formula reproduces on
the crop; a fieldmap phantom is refused; the linearity identity holds under compat.

- [x] Commit: `feat: series — compat relaxation at the echo, SNR noise, grid origin`.

### Task 3: `bids` and the CLI

`AslscanSimulation.Compat`: `{ Asldro: true, Pinned: {...every pinned key and value...},
DesiredSnr, NoiseVariance, M0ReferenceMean, GridOrigin, T2BloodUnused: true,
RelaxationAtEcho: true, SliceTimingAllZero: true }`; `Resolved.T2Blood` gains
`"Used": false` under compat. `--compat-asldro` on the CLI sets `[compat] asldro = true` (an
overlay that also sets it is fine; one that sets it false while the flag is given is an
error). Validator run on a compat dataset from the crop.

- [x] Commit: `feat: bids — the Compat sidecar block; --compat-asldro`.

### Task 4: the driver

`tools/compat_asldro.py` (`simasl` env: numpy 1.19, nibabel 3.1, nilearn 0.6.2, asldro):

- `run_simasl(params, out_zip)`: runs `run_full_pipeline` itself with
  `asldro.examples.AcquireMriImageFilter` replaced by a recording subclass, and returns each
  ASL volume's complex image (from the recorded filters' outputs), the noise reference (the M0
  resample's output) and the archive's magnitude; `|complex| == magnitude` to float32 precision
  is asserted, so the recording is known to be the run.
- `translate(params) -> (asl_json, aslcontext, overlay_toml)` per the addendum's table, with
  `AcquisitionVoxelSize = extent / acq_matrix` computed from the ground truth's shape; an error
  unless the ground truth's linear affine is the positive identity; `lambda_blood_brain` and
  `t1_arterial_blood` resolved from the ground-truth JSON merged with
  `global_configuration.parameter_override`.
- `pure_mask(dseg, acq_affine, acq_dims, pose=None)`: the addendum's 3D purity rule (box
  footprint plus an 8-voxel cube about the pre-motion sample point, one foreground label),
  with a check that the phantom's maps are constant per label.
- `pose_to_mrsim(rot_deg_xyz, transl_mm, fov_centre, origin=(0, 0, 0)) -> (rot_deg_zyx, transl)`:
  `R = Rx Ry Rz` (simasl, `rot_x_mat` etc.), ZYX Euler extraction for `mrsim-acq`'s `Rz Ry Rx`,
  `t' = t + (R - I)(c - o)`, with the gimbal case (`|R[2][0]| = 1`) handled; the FOV centre from
  `aslscan`'s grid rule (`(n - 1)/2` voxels from voxel 0's centre). Writes the trajectory TSV
  in radians.
- `run_aslscan(inputs, phantom_dir, out_dir)`: the release binary with `--compat-asldro
  --t2-mode voxel` (the mode is numerically irrelevant under compat and halves the memory).
- `compare(a, b, dseg, erosion)`: max |a - b| / peak(b), RMS(a - b) / RMS(b), over all voxels
  and the interior mask (per-label in-plane erosion by `erosion` voxels), per volume and for
  `control - label`; `noise_stats(volumes)` for benchmark C.
- Subcommands `A`, `B`, `C`, `D`, `E`, `all`, each writing `work/compat/<bench>-<phantom>.json`
  and a Markdown table, exit status from the criteria.

`tools/test_compat_asldro.py`: ten random poses (and one at the gimbal case) map points through
`pose_to_mrsim` and `mrsim-acq`'s `Pose::to_matrix` (re-implemented in numpy) to the same place
as through simasl's transform, within `1e-12`, and each of D's wrong conversions does not; the
parameter translation of simasl's defaults yields `PostLabelingDelay = 1.8`, `LabelingDuration =
1.8`, `RepetitionTimePreparation = [10, 5, 5]`, `M0Type = "Included"`, `AcquisitionVoxelSize =
[3.078125, 3.640625, 15.75]`, `SliceTiming` of 12 zeros; a PASL case maps `signal_time` to the
PLD and `label_duration` to the cutoff; a `parameter_override` reaches `[kinetic]`; a flipped
source affine is refused.

- [x] Commit: `tools: the ASLDRO compat driver and its tests`.

### Task 5: the benchmarks and acceptance

Convert both phantoms (`work/phantom-3t`, `work/phantom-1.5t`, full resolution). Run `all` on
the 3 T phantom and `A` and `E` on the 1.5 T one. Record every number in Measurements. If a
criterion fails, the finding goes in Measurements with its cause; the criterion is not loosened
without a reason written next to it.

`tests/end_to_end.rs`: `linearity_holds_under_compat`; `compat_benchmark_a_on_the_crop`,
gated on `ASLSCAN_SIMASL_ENV` (the micromamba env name) and printing a loud skip otherwise,
which shells out to the driver's `crop` mode: it writes the crop window of the 3 T ground truth
as a packed ASLDRO ground truth (5D NIfTI with the shifted affine, and the JSON) under `work/`,
converts the same window with `hrgt_to_bids.py --crop`, and runs benchmark A on the pair.

- [x] Commit: `test: P2 benchmarks and acceptance`; tag `p2-complete`.

## Acceptance criteria coverage

| Criterion | Where |
|---|---|
| 1. A and E at 1e-5 of peak on both phantoms | Task 5 |
| 2. B pure mask at 1e-3; all-voxel numbers and ground truth recorded | Task 5 |
| 3. C statistics on both sides, cross-side reference ratio | Task 5 |
| 4. D passes converted, fails each wrong conversion | Task 4 tests + Task 5 |
| 5. Compat dataset validates; sidecar names every pinned value | Task 3 |
| 6. mrsim-acq unchanged; default builds green | every task |

## Measurements

Measured 2026-10-01 with `tools/compat_asldro.py` (reports under `aslscan/work/compat/`), release
build `cli,kspace,par`, `--t2-mode voxel`. "Max rel" is `max |aslscan - simasl| / peak(simasl)`
on the signed real part, per volume; `control - label` against its own peak.

**A and E (identity grid, criterion `1e-5`, `1e-4` for control - label): pass everywhere.**

| Run | m0scan | control | label | control - label |
|---|---|---|---|---|
| A, 3 T | 6.2e-8 | 6.2e-8 | 6.5e-8 | 1.8e-6 |
| A, 1.5 T | 7.0e-8 | 6.2e-8 | 6.3e-8 | 4.6e-6 |
| E (IR, m0scan IR), 3 T | 8.3e-8 | 9.9e-8 | 8.3e-8 | 1.1e-6 |
| E, 1.5 T | 1.1e-7 | 3.0e-8 | 5.7e-8 | 7.0e-6 |
| A, crop (cargo test) | 6.2e-8 | 6.2e-8 | 6.5e-8 | 1.8e-6 |
| E, crop (cargo test) | 8.3e-8 | 9.9e-8 | 8.3e-8 | 1.1e-6 |

The residual is float32 storage: the oversample-1 round trip, the compat relaxation factor and the
IR m0scan row reproduce simasl's pipeline. The P3 baseline the label-inverting IR would depart
from is therefore exact.

**B (`[64, 64, 12]`).** ICBM 3 T: pure mask **0 voxels** (no 17-voxel cube of one tissue exists;
half-width 6 has ~3 000 GM and ~3 500 WM voxels, 8 has none), so not gated. All-voxel max rel
0.73 / 0.71 / 0.71 (m0scan / control / label), RMS rel 0.16; control - label max 1.03, RMS 0.36.
Ground truth all-voxel max rel: perfusion 0.92, ATT 1.00 (the CSF sentinel), T1 0.79, T2 0.83,
M0 0.72. This is the measured cost of box averaging against a point-sampled spline at 15.75 mm
slices. Synthetic blocks (gated): pure mask 9 572 voxels (GM 5 117, WM 4 415, CSF 40); images
max rel **1.0e-4** on the mask (control - label 1.4e-4), pass at `1e-3`; with constant blocks
it was 7.7e-7, the difference being the M0 ramp (aslscan box-averages piecewise-constant phantom
voxels, simasl point-samples the interpolant). Ground truth on the mask: perfusion 4.0e-6,
T1 4.1e-6, T2 3.6e-6, M0 1.1e-4; ATT 4.7e-3, reported not gated (simasl's spline carries the
1000 s CSF sentinel past the 8-voxel reach: WM 1.2057 s for 1.2 s).

**C (3 T, `[64, 64, 12]`, SNR 50, seeds 2..16 even): pass.**

| Side | M0 ref mean | ref voxels | predicted var | var re | var im | rho re/im | adjacent rho x/y/z | Var(C-L)/Var(C) |
|---|---|---|---|---|---|---|---|---|
| aslscan | 56.360 | 13 282 | 1.271 | 1.270 | 1.270 | -0.0006 | +0.0007 / -0.0001 / -0.0002 | 1.999 |
| simasl | 25.513 | 29 492 | 0.2604 | 0.2600 | 0.2599 | -0.0003 | +0.0024 / +0.0000 / +0.0004 | 1.997 |

Cross-side variance ratio 0.2047 against a squared reference ratio of 0.2049: the fivefold
difference in noise at equal `desired_snr` is the M0 reference and nothing else. All 8
realizations distinct on both sides.

**D (identity grid, pose `(2, -3, 4)` deg / `(1.5, -2, 0.5)` mm).** Synthetic blocks with the M0
ramp (gated at `1e-4`, pure mask 1.79 M voxels): converted mixed pose 2.2e-5, single-axis
`rot_z = 5` 1.6e-5 (pass); wrong rotation order 1.0e-3, wrong rotation centre 1.8e-3 (fail, as
required). With constant blocks both wrong conversions had passed (2.1e-5): the ramp is what
makes D discriminate. All-voxel max rel (kernel cost at boundaries): 0.18 mixed, 0.10
single-axis. ICBM 3 T (not gated): pure mask 28 voxels (mixed), 0 (single-axis); on the 28,
9.7e-7; all-voxel max rel 0.34 mixed, 0.28 single-axis, 0.64 wrong order, 1.00 wrong centre.

**The motion approximation (D on `[64, 64, 12]`, reported).** Box average then trilinear on the
acquisition grid against simasl's single spline: ICBM 3 T all-voxel max rel 0.85, RMS 0.19;
synthetic all-voxel 0.69, on the 9 886-voxel pure mask 0.11 (trilinear reads a neighbouring
15.75 mm slice). This is the number the deferred move-then-evaluate pipeline would have to
beat.

**Compatibility of the non-compat paths.** P1/P3 outputs are byte-identical to `5812074` (every
NIfTI decompressed and every sidecar) for PASL cutoff on the full 3 T phantom, the crop PCASL
fixture, and asl002 with background suppression, random motion, noise and IR on the z-cropped
phantom. (`pcasl_single` and `pcasl_multipld` are refused identically by both builds: their
slice timing ends after their TR under the P3 readout-in-TR check, which predates P2.) A compat
crop dataset validates with no errors (three recommended-key warnings, as P1's). `mrsim-acq`
source is unchanged; `cargo test` is green in the default build of both crates and in
`aslscan --features cli,kspace,par,test-hooks` with `ASLSCAN_SIMASL_ENV=simasl`.
