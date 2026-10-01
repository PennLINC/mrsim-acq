# P2: the ASLDRO compatibility mode and the voxelwise benchmark

Design spec addendum, 2026-09-24. Extends `2026-09-21-mrsim-acq-aslscan-design.md` (the main
spec) and follows `2026-09-24-p3-motion-bs-ir-design.md` (P3). Where this document is silent,
those stand.

## Context

simasl (ASLDRO v2.2.0, frozen) has been the numerical oracle for `aslscan`'s pure functions
since P1: the kinetic model and the signal equations match it at `1e-9` and `1e-12`. Nothing
yet compares the two simulators' *outputs*. That comparison is what the main spec's roadmap
calls `--compat-asldro` and the voxelwise benchmark, and P3 left three decisions waiting on it:
whether a label-inverting IR preparation is worth the deliberate departure from simasl, how the
motion conventions convert, and how much the finished-image motion approximation costs.

The comparison is not a bit-identity gate. `aslscan` departs from simasl on purpose (the main
spec, "division of labor"): transverse relaxation belongs to the readout, the labeled blood is
its own compartment with its own T2, the acquisition is an EPI k-space model with distortion,
ringing and structured noise, and resampling is box averaging so that magnetization is
conserved. P2 builds a configuration of `aslscan` that switches every one of those off that
*can* be switched off, names the ones that cannot, and measures the residual per benchmark.
Where the residual is small the departure is validated; where it is large the number is the
finding.

## What simasl does

Read from `examples.py:37-420` and the filters it chains, so that the compat mode is built
against the code and not the paper.

Per ASL series, seeded with `np.random.seed(random_seed)`:

1. `GkmFilter` on the full-resolution ground truth gives `delta_m` per voxel; an
   `InvertImageFilter` negates it. This is what P1's `kinetic` reproduces.
2. `m0` is resampled to `acq_matrix` with no motion; it is the noise reference for every volume.
3. Per volume in `asl_context` (default `"m0scan control label"`), `AcquireMriImageFilter`:
   - `MriSignalFilter` at full resolution: the steady state of the series' `acq_contrast` at
     that volume's `repetition_time` (default `[10, 5, 5]` s), **m0scan rows included**: an
     IR series' m0scan row is an IR image (`examples.py:173`, `mri_signal_filter.py:252`). Plus
     `mag_enc` (the negated `delta_m`, label rows only; control and m0scan rows carry none),
     times `exp(-TE/T2)` with the *voxel's* T2, blood included.
   - `TransformResampleImageFilter`: one rigid pose (`rot_x/y/z` degrees, `transl_x/y/z` mm,
     rotation `Rx Ry Rz` about `rotation_origin`, world origin by default) composed with the
     resampling to `acq_matrix` in a single `nilearn.image.resample_img` call, interpolation
     `continuous` (a third-order spline, a point sample of the prefiltered interpolant at each
     target voxel centre, zero outside; nilearn 0.6.2). The target voxel size is the source
     shape over `acq_matrix` (assuming 1 mm source voxels), the target axes are positive
     whatever the source's signs (`utils/resampling.py:76, 107-127`), the nominal extent
     `n * voxel` equals the source's, and target voxel 0 is centred on source voxel 0's centre
     (so the outer boundaries do not coincide).
   - `AddComplexNoiseFilter`: a 3D FFT of the image, Gaussian noise of standard deviation
     `sqrt(N) * mean(|m0_acq| over nonzero voxels) / desired_snr` on the real and imaginary
     parts of every k-space sample, inverse FFT. The image-space noise is therefore white with
     per-component standard deviation `mean(|m0_acq|) / desired_snr` (the ASL default SNR is
     100, `validators/user_parameter_input.py:263`), the same amplitude for every volume.
     "Nonzero" is exact: every positive spline tail of the resampled `m0` counts, so the
     reference mean depends on the resampler as much as on the anatomy.
   - `PhaseMagnitudeFilter`: the ASL branch always keeps the magnitude
     (`examples.py:226-234`); `output_image_type = "complex"` exists for the structural series
     only.
4. The volumes are concatenated and written as BIDS with simasl's own (draft-era) sidecar keys.

A `ground_truth` series resamples every ground-truth quantity to `acq_matrix` with the same
spline and an optional pose. There is no readout model: no distortion, no line timing, no T2'
decay, no ringing beyond what the spline does, no coils, no acceleration.

## Scope

In: the compat configuration of `aslscan` (part A), the benchmark driver and its five
benchmarks (part B), and what they decide for P3's deferred items (part C). Out: porting
numpy's MT19937 and its Gaussian generator (noise is compared statistically), spline
resampling in `aslscan`, simasl's `structural` series, background suppression (simasl has none),
3D readouts (P5), and anything that changes `mrsim-acq`.

---

# Part A: the compat configuration

## Inputs

Overlay `[compat]`, and the CLI flag `--compat-asldro` which is the same as writing
`asldro = true`:

| Key | Default | Meaning |
|---|---|---|
| `asldro` | false | Pin the acquisition to what simasl can express (below) |
| `desired_snr` | absent | simasl's SNR; converted to the acquisition's noise variance, see below. 0 or absent means no noise |
| `grid_origin` | `"voxel-centre"` when `asldro` | `"corner"` (P1: the grids share their outer edge) or `"voxel-centre"` (simasl: acquisition voxel 0 is centred on phantom voxel 0) |

With `asldro = true`, `protocol` pins `oversample = 1`, `partial_fourier = 1`, `n_coils = 1`,
`ghost_offset = 0`, `n_spikes = 0`, `eddy_strength = eddy_quad = eddy_phase = 0`,
`window = "none"`, `signal_scale = 1`, `noise_variance = 0` (noise comes from `desired_snr`),
and `ParallelReductionFactorInPlane = 1`. An overlay or sidecar that sets any of these to
something else is an error naming the key and the pinned value: compat mode overrides nothing
silently. `MultibandAccelerationFactor > 1`, a fieldmap in the phantom, `[motion.within_volume]`,
`BackgroundSuppression: true` and `M0Type: Separate` (simasl's M0 is an `m0scan` row) are
likewise refused, because simasl cannot express them. So is a `SliceTiming` whose entries are
not all equal: the offsets would still give each slice its own kinetic time, and simasl has
none.

## What the mode changes in the signal stage

**Relaxation at the echo, per voxel.** simasl multiplies every voxel by `exp(-TE/T2_voxel)`
once. The acquisition stage's per-line weight `exp(-trf/T2 - |t|/T2')` is a k-space filter and
cannot reduce to that. Compat therefore sets `do_relaxation = false` on the acquisition and
applies `exp(-TE/T2)` in the signal stage, per phantom voxel, with simasl's zero-T2 guard
(`exp(0) = 1`), to the tissue **and** to the blood of that voxel. The blood compartment's own
T2 is not used; the sidecar records `T2Blood` as unused under compat. The compartment split
itself stays (it is what makes the linearity identity testable), and the class/voxel choice
becomes irrelevant to the numbers, since no per-compartment relaxation is applied.

**The m0scan row takes the series' contrast.** Outside compat an included `m0scan` row is a
plain spin-echo readout (P3). Under compat it takes the series' signal equation with no label,
as simasl's does, so an IR series' m0scan row is the IR steady state at its own TR.

**Readout with no effects.** `aslscan` always calls `simulate_acquisition_oversampled`. With
`oversample = 1` its simulation and acquisition grids coincide, so the forward transform and the
reconstruction are a same-size pair and inverses up to floating point, the property
`kspace.rs` documents for the legacy entry point ("no Gibbs ringing and no object phase") and
the one P1's homogeneous-grid test and P3's translation test already rely on. With
`do_distortions` off (no fieldmap is allowed), no relaxation, one coil, full sampling, no
window, and the CLI's phase model (global 0, no background modes, no preparation phase), the
acquired complex image is the real-valued, *signed* simulation-grid input: phase 0 where the
signal is positive and pi where it is negative (IR CSF is). Comparisons are therefore on the
signed real part. The `SliceTiming` the 2D contract requires is all zeros; the sidecar says so, and
`MRAcquisitionType` stays `"2D"` (simasl labels its output 3D; this is a metadata difference,
not a numerical one, and P5 owns 3D).

**The grid.** `grid_origin = "voxel-centre"` places acquisition voxel 0's centre on phantom
voxel 0's centre, as simasl's `transform_resample_affine` does, so the two outputs are on the
same world grid voxel for voxel. The dimensions come from `AcquisitionVoxelSize` as in P1; the
driver derives the voxel size from the phantom extent over `acq_matrix` so that the P1
`ceil(extent / voxel - 1e-9)` rule reproduces `acq_matrix` exactly. The affine alone is not
enough: the box weights assumed a shared corner, so `resample` gains a per-axis corner offset
(`0.5 p - 0.5 v` here, `(v - 1)/2` mm on benchmark B's grid, 7.4 mm through-plane), derived
from the two grids, used by both the phantom-to-simulation and the phantom-to-acquisition
resamplers, and exactly zero for P1's corner grids. A cell hanging past the phantom edge is a
partial cell, its overlap still divided by the full cell volume, which is zero extension, as
at the far edge in P1. `aslscan` still box-averages and simasl still point-samples a spline;
that difference is what benchmark B measures.

**Noise.** `desired_snr` converts to the acquisition's per-component image variance,

```
noise_variance = (signal_scale * mean(|M0_acq| over M0_acq != 0) / desired_snr)^2
```

with `signal_scale = 1` under compat and `M0_acq` the M0 ground truth on the acquisition grid
(`series` already has it). `mrsim-acq` defines `noise_variance` as exactly the per-component
image variance under full sampling with one coil (`kspace.rs:880-889`), and simasl's k-space
amplitude divided by the inverse FFT's `1/N` gives the same image-space quantity, so the
*formula* is the same on both sides. The *reference* is not: each side uses its own resampled
M0, and simasl's spline tails make its nonzero support more than twice as large and its mean
less than half (measured on the 3 T phantom at `[64, 64, 12]`: 29 851 voxels at mean 25.21
against 13 282 at 56.36), a variance ratio of about 0.20 at equal SNR. On the identity grid the
two references coincide. `aslscan` keeps its own reference rather than learning to imitate a
spline artifact; benchmark C checks each side against its own prediction and requires the
cross-side ratio to be the ratio of the reference means. The resolved variance and the
reference mean are written to the sidecar.

**Rows.** simasl's `asl_context` maps one to one: `m0scan` to an included M0 row at its own TR,
`control` and `label` to P1's rows. simasl's `signal_time` is `PLD + tau` for (P)CASL, which is
P1's `Row::t`, and `label_duration` and `label_efficiency` map to the sidecar. simasl's
`lambda_blood_brain` and `t1_arterial_blood` come from the ground truth's JSON merged with
`global_configuration.parameter_override` (`ground_truth_loader.py:187`); series-level keys of
those names are not overrides but a `FilterInputKeyError` (`basefilter.py:192`). The driver
resolves them that way and writes them to `[kinetic]`. The driver does this translation;
`aslscan` itself gains no simasl-named inputs beyond `[compat]`.

## What the mode cannot change

These are the residuals the benchmarks measure, listed so that a nonzero difference is read
against the right cause:

- **Resampling kernel**: a box average over the cell against a point sample of a third-order
  spline at its centre. Identical on the identity grid; elsewhere they agree only where the
  phantom is one tissue over the whole cell *and* the spline's reach around the centre. With
  15.75 mm slices, through-plane mixing alone puts about 27% of peak between them in voxels
  whose in-plane neighbourhood is pure (Codex probe, majority labels eroded `3 x 3 x 1`).
- **Motion interpolation**: trilinear on the simulation grid after resampling, against the
  spline applied to the full-resolution image in the same step as the resampling. Even on the
  identity grid the kernels differ, by up to 31% of peak at boundaries for benchmark D's pose.
- **Noise reference**: each side's own resampled M0 (above).
- **Precision**: `aslscan` stores float32 magnitude and phase; simasl computes in float64.
  The floor is about `1e-7` of the peak on the image and correspondingly larger on
  `control - label`.
- **Ground truth**: `aslscan`'s box-averaged maps against simasl's spline-resampled ones.

---

# Part B: the benchmark

## The driver

`aslscan/tools/compat_asldro.py`, run in the `simasl` environment, since it imports
`asldro`. It takes a benchmark name and a phantom, and for each case:

1. Builds simasl's `input_params` (the schema `validate_input_params` accepts) and runs
   `run_full_pipeline` itself, with `AcquireMriImageFilter` wrapped (in the `asldro.examples`
   namespace only) by a subclass that records each instance, so that each volume's **complex**
   output before `PhaseMagnitudeFilter` is read back from the filter objects of the real run
   (the pipeline writes magnitudes only). The archive's magnitude must equal the modulus of
   those complex volumes to `1e-6` relative (the archive is float32), so the recording is
   known to be the run. The comparisons below are on the signed real part (both sides' images
   are real up to noise; see part A) and on the complex noise.
2. Writes the equivalent `aslscan` inputs: an `asl.json` from the parameter table below, an
   `aslcontext.tsv` from `asl_context`, and an overlay with `[compat]`, `[kinetic]` and the
   pose trajectory when the case has motion, then runs the `aslscan` binary on the phantom
   converted by `tools/hrgt_to_bids.py` from the same ASLDRO ground truth.
3. Loads both outputs, checks the two affines agree (an error if not, never a resample),
   and computes per volume: the maximum absolute difference over the peak of simasl's volume,
   the RMS of the difference over the RMS of simasl's volume, both over all voxels and over a
   *pure* mask, and the same for `control - label`. An acquisition voxel is pure when every
   full-resolution phantom voxel in the union of its box footprint and the cube of half-width
   8 phantom voxels about its (pre-motion) sample point has the same foreground label. The
   footprint is what the box average reads; the cube is the spline's reach (the cubic's
   support of 2 voxels plus 6 for the prefiltered coefficients to settle, by `0.268^6 < 4e-4`
   of the step). The mask's size is reported and must be at least 50 voxels; it is computed in
   the driver from the phantom's `dseg` and the known geometry, never from either output.
4. Writes a JSON report and a Markdown table per benchmark under `work/compat/`, and exits
   nonzero when a criterion below fails.

The parameter translation, simasl on the left:

| simasl | `aslscan` |
|---|---|
| `label_type` | `ArterialSpinLabelingType` (upper-cased) |
| `label_duration` | `LabelingDuration` |
| `signal_time` | `PostLabelingDelay = signal_time - label_duration` ((P)CASL); PASL takes `BolusCutOffDelayTime = label_duration`, `PostLabelingDelay = signal_time` |
| `label_efficiency` | `LabelingEfficiency` |
| ground-truth JSON `lambda_blood_brain`, `t1_arterial_blood`, merged with `global_configuration.parameter_override` | `[kinetic]` overlay (the resolved values, overriding the phantom's `phantom.json`) |
| `asl_context` | `aslcontext.tsv`; `M0Type` `Included` when it has `m0scan`, else `Absent` |
| `echo_time[i]` | `EchoTime` (must be constant; multi-TE is P6) |
| `repetition_time[i]` | `RepetitionTimePreparation` array |
| `acq_matrix` | `AcquisitionVoxelSize = extent / acq_matrix` per axis, `[acquisition] matrix` in-plane |
| `acq_contrast`, `excitation_flip_angle`, `inversion_time`, `inversion_flip_angle` | `[signal]` overlay (P3) |
| `desired_snr`, `random_seed` | `[compat] desired_snr`, `seed` |
| `rot_x/y/z`, `transl_x/y/z` | a trajectory TSV after the conversion in part C |
| `output_image_type` | always complex on the `aslscan` side |

`MagneticFieldStrength` is the phantom's; `PhaseEncodingDirection` is `"j-"`;
`TotalReadoutTime` is `0.001` s (the readout is inert under compat but the P1 timing check still
runs); `SliceTiming` is `acq_matrix[2]` zeros; `BackgroundSuppression` is `false`.

The ground truth's linear affine must be the positive identity, not merely 1 mm: simasl's
target axes are positive whatever the source's, and `aslscan` keeps the source's signs, so a
flipped axis would put the two outputs on different grids. Both shipped phantoms qualify.

## The benchmarks

**A. Identity grid, no noise, no motion.** `acq_matrix` equal to the phantom dimensions, simasl's
default `asl_context`, TE, TR, PCASL timing. Both resamplers are the identity, so the only
differences left are precision and the round trip through k-space. Criterion: for every
volume, max |difference| `<= 1e-5` of simasl's peak; for `control - label`, `<= 1e-4` of its
own peak (the P3 floor argument: the difference is about 1% of the tissue signal).

**B. simasl's default acquisition matrix, no noise, no motion.** `acq_matrix = [64, 64, 12]` on
the 3 T phantom. This isolates box averaging against the spline. Criterion: on the pure mask,
max relative difference `<= 1e-3` (both kernels reproduce a region that is constant over
everything they read, and the phantom's maps are constant per tissue class, which the driver
checks); the all-voxel numbers are reported, not gated, and are the measured cost of the
resampling choice. The ground-truth maps are compared the same way. A failure on the pure mask
means a geometry error (an offset, an axis), not a kernel difference.

**C. Noise statistics.** Benchmark B's protocol with `desired_snr = 50` and 8 seeds on each
side, minus the noise-free run. The seeds are the even numbers `2, 4, ..., 16`: `mrsim-acq`
forces the low bit of its k-space generator's state (`kspace.rs:890`), so seeds differing in
bit 0 alone give the same realization; the driver checks that the 8 realizations differ.
Criteria on the noise field, per side, `n` being the count of independent samples (voxels
times realizations): mean within `3 sigma / sqrt(n)` of zero; per-component variance within 5%
of that side's predicted `(mean|M0_acq| / 50)^2` from its own reference; real and imaginary
components uncorrelated (`|rho| < 0.02`); `|adjacent-voxel correlation| < 0.02` along each axis
(white on both sides, since both add white noise in k-space); `Var(control - label) /
Var(control)` equal to 2 within 10%. Across the sides: the variance ratio equals the squared
ratio of the two reference means within 10%, so the amplitude difference is the reference and
nothing else; the two means and the ratio are reported. Nothing compares the two noise fields
sample by sample.

**D. Motion conventions.** The identity grid, no noise, one control volume with a mixed-axis
pose (`rot = (2, -3, 4)` degrees, `transl = (1.5, -2, 0.5)` mm) on the simasl side, converted for
`aslscan` by part C. The pure mask is taken about each voxel's pre-motion sample point
`M^-1 p` (trilinear reads 1 voxel around it, the spline 8). Criterion: max relative difference
on the pure mask `<= 1e-3`; the all-voxel numbers are reported (they are kernel differences, up
to 31% of peak). A single-axis rotation case must pass too. Two deliberately wrong conversions
must fail, each separately, so that each half of the conversion is known to matter: the
rotation order (simasl's angles passed straight through as `Rz Ry Rx` angles, translation
converted), and the rotation centre (the right angles, `t' = t` with no centre correction).
The conversion itself is also proved without images, by a pure-numpy test mapping points
through both transforms (part C).

**E. Inversion recovery.** Benchmark A's setup with `acq_contrast = "ir"`, `inversion_time = 1`,
`excitation_flip_angle = 60`, `inversion_flip_angle = 180`, TR 5 s for the ASL rows and simasl's
default 10 s for the m0scan row, which is an IR image on both sides (part A). Criterion as A,
on the signed real part (IR CSF is negative). This is the compat baseline P3 asked for: the
number a label-inverting preparation would then move away from.

The default phantom for every benchmark is `hrgt_icbm_2009a_nls_3t`; benchmarks A and E also run
on `hrgt_icbm_2009a_nls_1.5t` to cover the 1.5 T defaults.

---

# Part C: what P2 decides for P3

## The motion conversion

simasl moves the object by `p' = R_s (p - o) + o + t` with `R_s = Rx(a) Ry(b) Rz(c)` in degrees
(`utils/resampling.py:82-84`, the order the image path actually uses) about `o`, the world
origin by default. `mrsim-acq` moves it by `p' = R (p - c) + c + t'` with `R = Rz Ry Rx` about
the field-of-view centre `c`. The same rigid transform is

```
R  = R_s                       (as a matrix; decompose into Rz Ry Rx angles for the Pose)
t' = t + (R_s - I)(c - o)
```

The decomposition is the standard ZYX Euler extraction with the gimbal case handled at
`|R[2][0]| = 1`; the driver implements it in numpy, writes the resulting pose to the trajectory
TSV in radians (P3), and benchmark D proves it. The conversion lives in the driver, not in
`aslscan`: a user of `aslscan` describes motion in one convention, and that convention is
`mrsim-acq`'s.

## The IR question

Benchmark E establishes how closely `aslscan` reproduces simasl's IR output with the label left
uninverted. If P4 or a user then wants the physically expected inversion of the label, it is
built from P3's timeline as one pulse at `t_read - TI`, and the difference from benchmark E's
numbers is the documented departure. P2 does not build it.

## The motion approximation

Benchmark D is on the identity grid, where the finished-image approximation and simasl's
single-step resampling differ only by kernel. A variant of D on `acq_matrix = [64, 64, 12]`
measures the double-resampling cost (box average, then trilinear) against simasl's single
spline; it is reported, not gated, and is the number the deferred move-then-evaluate pipeline
would have to beat.

---

# Division of labor

`aslscan`: the `[compat]` overlay table and CLI flag in `protocol`, the pinned-value and
slice-timing checks, `grid_origin` in `resample::acquisition_grid` and the corner offset in the
box weights, the relaxation-at-the-echo factor, the m0scan row's contrast and the SNR
conversion in `series`, the sidecar block `AslscanSimulation.Compat` recording every pinned
value, the unused `T2Blood`, the resolved noise variance with its reference mean, and the grid
origin. `mrsim-acq`: no change; `do_relaxation`, `noise_variance` and the `oversample = 1`
round trip already exist. The driver and its report are Python in the `simasl` environment.

# Testing and verification

- **protocol**: `[compat] asldro = true` pins the listed values; each conflicting explicit key
  is rejected naming both; multiband, within-volume motion, suppression, `M0Type: Separate` and
  unequal slice timings are rejected under compat; `grid_origin` parses and defaults per the
  table.
- **resample**: `voxel-centre` places voxel 0's centre on the phantom's voxel 0 centre and
  reproduces `acq_matrix` from the derived voxel size for the 3 T phantom's dimensions and
  `[64, 64, 12]`; the offset weights give partial cells at the near edge and are P1's exactly
  at offset zero.
- **series**: a fieldmap is refused under compat; the relaxation factor equals `exp(-TE/T2)`
  per voxel with the zero-T2 guard, on tissue and blood alike, and `do_relaxation` is off; an
  included m0scan row of an IR compat series is the IR image (and stays spin echo outside
  compat); the SNR conversion reproduces the formula on the crop; the voxel-centre grid
  moves the data with the affine (a hand-placed point source lands where the affine says); the
  linearity identity holds under compat (it is a per-voxel scaling before the acquisition).
- **driver**: a pure-numpy test of the pose conversion that maps points through both
  transforms for ten random poses, including one at the gimbal case, and shows that each wrong
  conversion of benchmark D does not; a round trip of the parameter table, including a
  `parameter_override` of `lambda_blood_brain`.
- **end to end**: benchmark A on a crop of the 3 T ground truth, run from `cargo test` through
  the Python driver only when the `simasl` environment is present (skipped otherwise, loudly).
  The driver writes the crop as a packed ASLDRO ground truth (5D NIfTI and JSON) for simasl and
  converts the same crop with `hrgt_to_bids.py` for `aslscan`.

# Acceptance criteria

1. Benchmarks A and E pass at `1e-5` of peak on both phantoms; the numbers are recorded in the
   plan.
2. Benchmark B passes its pure-mask criterion on a mask of at least 50 voxels; its all-voxel
   numbers and the ground-truth comparison are recorded.
3. Benchmark C passes every statistic on both sides and the cross-side reference check.
4. Benchmark D passes for the mixed-axis and single-axis poses and fails for each of the two
   wrong conversions.
5. A compat dataset validates with no errors, and its sidecar names every pinned value.
6. `mrsim-acq` is unchanged; `cargo test` is green in the default build of both crates.

# Decisions deferred

- **Porting numpy's generator**, which would make benchmark C sample-exact. Not worth the
  maintenance; the statistics are the contract.
- **Spline resampling in `aslscan`**, which would make benchmark B tight at boundaries. Not
  planned: a point sample of an interpolant is not what a voxel measures, and B's all-voxel
  numbers are the documented cost of the difference.
- **Matching simasl's noise amplitude** by feeding its spline-resampled reference mean to
  `aslscan`. Benchmark C shows the amplitudes differ by the references alone; a user who wants
  simasl's number can set `noise_variance` without compat.
- **simasl's `structural` series** and its `ground_truth_modulate` / `image_override` hooks.
  The overlay's `[kinetic]` covers the parameter overrides the ASL series uses.
- **A `--compat-asldro` for simasl's BIDS sidecars as input**: simasl writes draft-era keys
  (`LabelingType`, `M0`), and `aslscan` reads current BIDS. The driver translates; `protocol`
  does not learn the old names.

# Risks

**The exact round trip is not exact.** `oversample = 1` composes a forward transform and a
reconstruction that are inverses in exact arithmetic; the float32 output and the FFT's rounding
set a floor near `1e-6` of peak. Criterion A's `1e-5` leaves room; if it fails, the cause is a
real difference (a stray relaxation weight, a half-voxel grid offset), which is the point.

**The M0 noise reference differs a great deal between the sides.** simasl's is the
spline-resampled `m0`, whose exactly-nonzero support includes every spline tail; `aslscan`'s is
the box-averaged M0 ground truth. At `[64, 64, 12]` the means are 25.21 and 56.36, so at equal
`desired_snr` simasl's noise variance is about a fifth of `aslscan`'s. This was a review
finding, not a design choice: "SNR" in simasl is partly a property of its resampler. Benchmark
C measures it, and the sidecar records the reference mean so a dataset says which SNR it
means.

**The pure mask may be small.** It needs one tissue over a 16-voxel cube about each sample
and, on B's grid, over 16 mm through-plane. If fewer than 50 voxels qualify the criterion is
not met and the finding is recorded with the mask size, rather than the mask relaxed.

**simasl's own rotation-order inconsistency.** `AffineMatrixFilter` and the image path disagree
(P3); P2 follows the image path, since that is what produces simasl's output. A future simasl
that fixes the filter would move benchmark D, and the conversion is a driver-side function to
update, not an `aslscan` change.
