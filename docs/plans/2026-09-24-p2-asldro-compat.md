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
at `aslscan` tag `p3-complete`. The Codex review of the P2 addendum could not run (the workspace
was out of credits); its prompt is saved and the review is owed before or during Task 1.

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
- Commit per task with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; tag
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
(`resampling_translate_affine` from the source origin). Dimensions are unchanged. Tests: both
origins on a signed affine; the 3 T phantom's dimensions `[197, 233, 189]` with voxel
`[197/64, 233/64, 189/12]` give `[64, 64, 12]` under the `ceil(extent / voxel - 1e-9)` rule
(each quotient is exactly representable).

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
  `MultibandAccelerationFactor > 1`, `BackgroundSuppression: true`; from the overlay:
  `[motion.within_volume]`. Each names simasl as the reason.
- `desired_snr` (finite, `>= 0`; 0 means none) is accepted only with `asldro = true`.
- `grid_origin` defaults to `voxel-centre` under compat and `corner` otherwise; the strings
  `"corner"` and `"voxel-centre"` are the only ones.

`Protocol::acquisition` sets `do_relaxation = false` and the pinned values when compat is on.
Tests for each rule, plus a compat protocol that parses cleanly with an otherwise empty overlay.

- [ ] Commit: `feat: resample grid origin and the [compat] protocol table`.

### Task 2: `series` under compat

- `acquisition_grid(..., p.grid_origin)`.
- A phantom with a fieldmap is refused under compat, naming simasl.
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

Tests: on the crop, a compat control run equals a non-compat run with `oversample = 1` and an
overlay `noise_variance = 0` in which the acquisition has `do_relaxation = false` and the input
was scaled by `exp(-TE/T2)` per voxel (built through `simulate_with` and a test hook that
exposes the factor, or by comparing against a hand-scaled phantom copy); the noise variance
formula reproduces on the crop; the linearity identity holds under compat.

- [ ] Commit: `feat: series — compat relaxation at the echo, SNR noise, grid origin`.

### Task 3: `bids` and the CLI

`AslscanSimulation.Compat`: `{ Asldro: true, Pinned: {...every pinned key and value...},
DesiredSnr, NoiseVariance, M0ReferenceMean, GridOrigin, T2BloodUnused: true,
RelaxationAtEcho: true, SliceTimingAllZero: true }`; `Resolved.T2Blood` gains
`"Used": false` under compat. `--compat-asldro` on the CLI sets `[compat] asldro = true` (an
overlay that also sets it is fine; one that sets it false while the flag is given is an
error). Validator run on a compat dataset from the crop.

- [ ] Commit: `feat: bids — the Compat sidecar block; --compat-asldro`.

### Task 4: the driver

`tools/compat_asldro.py` (`simasl` env: numpy 1.19, nibabel 3.1, nilearn 0.6.2, asldro):

- `simasl_asl_series(params, keep_complex=True)`: builds the filter chain of
  `examples.py:126-249` object for object (GkmFilter, InvertImageFilter, the M0
  `TransformResampleImageFilter`, one `AcquireMriImageFilter` per `asl_context` entry with the
  same inputs, `np.random.seed` first), returns the complex volumes, the resampled M0 and the
  affines. `simasl_archive(params, path)` runs `run_full_pipeline` and returns the archive's
  `asl/001_asl.nii.gz` magnitude; `assert_replication_faithful` checks
  `|complex| == magnitude` to `1e-10` relative.
- `translate(params) -> (asl_json, aslcontext, overlay_toml)` per the addendum's table, with
  `AcquisitionVoxelSize = extent / acq_matrix` computed from the ground truth's shape and the
  1 mm assumption checked against its affine (an error if the source voxels are not 1 mm, since
  simasl's `output_voxel_size = scale` assumes it).
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

`tools/test_compat_asldro.py`: ten random poses (and one at the gimbal case) round-trip
through `pose_to_mrsim` and re-composition to the same 4x4 within `1e-12`; the parameter
translation of simasl's defaults yields `PostLabelingDelay = 1.8`, `LabelingDuration = 1.8`,
`RepetitionTimePreparation = [10, 5, 5]`, `M0Type = "Included"`, `AcquisitionVoxelSize =
[3.078125, 3.640625, 15.75]`, `SliceTiming` of 12 zeros; a PASL case maps `signal_time` to the
PLD and `label_duration` to the cutoff.

- [ ] Commit: `tools: the ASLDRO compat driver and its tests`.

### Task 5: the benchmarks and acceptance

Convert both phantoms (`work/phantom-3t`, `work/phantom-1.5t`, full resolution). Run `all` on
the 3 T phantom and `A` on the 1.5 T one. Record every number in Measurements. If a criterion
fails, the finding goes in Measurements with its cause; the criterion is not loosened without a
reason written next to it.

`tests/end_to_end.rs`: `linearity_holds_under_compat`; `compat_benchmark_a_on_the_crop`,
gated on `ASLSCAN_SIMASL_ENV` (the micromamba env name) and printing a loud skip otherwise,
which shells out to the driver with the crop as the ground truth (the driver accepts a
converted-phantom directory plus the ASLDRO 5D file for simasl; the crop's source 5D is
regenerated by `hrgt_to_bids.py --crop` into `work/` for this).

- [ ] Commit: `test: P2 benchmarks and acceptance`; tag `p2-complete`.

## Acceptance criteria coverage

| Criterion | Where |
|---|---|
| 1. A and E at 1e-5 of peak on both phantoms | Task 5 |
| 2. B interior at 1e-3; boundary and ground truth recorded | Task 5 |
| 3. C statistics on both sides | Task 5 |
| 4. D passes converted, fails unconverted | Task 4 tests + Task 5 |
| 5. Compat dataset validates; sidecar names every pinned value | Task 3 |
| 6. mrsim-acq unchanged; default builds green | every task |

## Measurements

(filled in as the tasks complete)
