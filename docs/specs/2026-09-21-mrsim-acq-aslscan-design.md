# mrsim-acq extraction and aslscan core

Design spec, 2026-09-21.

Covers sub-projects P0 and P1 of the simasl-to-Rust roadmap. Later sub-projects get their own
specs.

## Context

Three repositories are involved.

- **simasl** is ASLDRO v2.2.0: a Python pipe-and-filter ASL digital reference object generator,
  about 6,800 non-test lines. Its pipeline is `GroundTruthLoader` to `GkmFilter` to
  `AcquireMriImageFilter` to `CombineTimeSeriesFilter` to `BidsOutputFilter`.
- **TRXScan** is a Rust diffusion-MRI simulator, about 11,300 lines. Its acquisition stage
  (`src/kspace.rs`, 2,619 lines) models EPI distortion, T2\* decay, eddy currents, Nyquist
  ghosting, partial Fourier, Gibbs ringing, spikes, multi-coil Roemer combination, GRAPPA, and
  k-space noise.
- **mrsim-acq** is this repository, created to hold the acquisition stage that both simulators
  share.

simasl's only k-space operation is `AddComplexNoiseFilter`, which runs a whole-volume `fftn`,
adds Gaussian noise, and inverts the transform. There is no readout model. Motion is one rigid
pose per volume applied by affine resampling. `signal_time` is a scalar per image series
(`src/asldro/validators/user_parameter_input.py:224`), so a multi-delay acquisition requires N
separate series rather than one 4D file. BIDS is written but never read.

The goal is an ASL simulator with TRXScan's acquisition fidelity. The chosen route is to extract
TRXScan's acquisition stage into a shared crate and write the ASL signal stage in Rust against it.
simasl is frozen and becomes the numerical oracle.

## Roadmap

Seven sub-projects. This spec covers P0 and P1.

| | Sub-project | Depends on |
|---|---|---|
| P0 | `mrsim-acq` extraction | — |
| P1 | `aslscan` core: BIDS in, GKM, k-space, BIDS out | P0 |
| P2 | `--compat-asldro` and the voxelwise benchmark | P1 |
| P3 | ASL motion: spin and saturation history, motion during labeling | P1 |
| P4 | Kinetic model extensions: macrovascular compartment, physiological noise | P1 |
| P5 | 3D readouts: GRASE, stack-of-spirals | P0 |
| P6 | Encoding variety: Hadamard, Look-Locker, velocity-selective, multi-TE | P1 |

P0 and P1 are specced together so that the extracted interface answers to a real second consumer
rather than to a guess about one.

## Repository layout

Sibling repositories under `~/Documents/rust-trx/`, matching how `odx-rs` and `rust/trx-rs` are
already wired in.

```
rust-trx/
  mrsim-acq/     shared acquisition stage        (new)
  aslscan/       ASL simulator                   (new)
  TRXScan/       diffusion simulator, now a consumer of mrsim-acq
  simasl/        Python oracle, frozen
  odx-rs/
  rust/trx-rs/
```

`TRXScan` and `aslscan` reference `mrsim-acq` as a path dependency.

---

# P0: extract the acquisition stage

## Modules moved

From `TRXScan/src/` into `mrsim-acq/src/`: `mat`, `orient`, `phase`, `readout`, `kspace`,
`nufft`, `noise`, `motion`, `config`, and the signal-model-independent half of `io`
(`load_volume`, `Grid`, `hires_grid`, `write_3d`, `write_4d`, `write_3d_i16`, and the complex
writer).

TRXScan keeps `scheme`, `raster`, `signal`, `compartments`, `sphere`, `mixture`,
`microstructure`, `truth`, `gnl`, `analytic`, `benchmark`, its streamline I/O, and its binaries.

Tests live as `#[cfg(test)] mod tests` at the bottom of each module, so they move with their
modules. That arrangement is preserved in `mrsim-acq`.

## Build conventions

`mrsim-acq` keeps TRXScan's build shape. The default build is pure std with every external
dependency optional, so `cargo test` runs offline with no system libraries. Feature flags are
`kspace`, `io`, `config`, and `par`, with the same meanings they have in TRXScan.

The source is hand-formatted at roughly 100 columns. `cargo fmt` is not run repo-wide, and no
`rustfmt.toml` is added.

## Interface changes

Seven changes, each driven by what the ASL consumer needs. Nothing else about the acquisition
stage changes.

### 1. Per-compartment T2 becomes a scalar or a map

`SliceInput.t2` is currently `&'a [f32]`, indexed by compartment at `kspace.rs:649` and
`kspace.rs:1479`. Each compartment decays at its own T2 across the readout. TRXScan supplies one
scalar per compartment. ASLDRO's phantom has per-voxel T2 maps.

```rust
pub enum T2Source<'a> {
    Uniform(f32),
    Map(&'a [f32]),   // on the simulation grid, layout x + snx*y
}

pub struct SliceInput<'a> {
    pub t2: &'a [T2Source<'a>],   // one entry per compartment
    ...
}
```

TRXScan's call sites wrap their existing scalars in `T2Source::Uniform`. The decay expression at
`kspace.rs:649` gains a match on the source rather than an unconditional index.

### 2. Diffusion gradients become a generic eddy drive

`SliceInput.bvec` and `SliceInput.bval` (`kspace.rs:269-272`) are read only by the eddy model, at
`kspace.rs:391-393`, `kspace.rs:484`, and `kspace.rs:663`. In every one of those places they
appear as the product `bvec * bval`, a gradient first moment.

```rust
pub struct SliceInput<'a> {
    /// Per-volume gradient first moment driving the eddy-current model.
    /// `None` disables eddy for this volume, reproducing the old `bval ~= 0` branch.
    pub eddy_drive: Option<[f64; 3]>,
    ...
}
```

TRXScan passes `Some([bvec[0]*bval, bvec[1]*bval, bvec[2]*bval])`, and `None` where `bval` is
within 1e-9 of zero. aslscan passes the crusher moment when vascular crushing is configured, and
`None` otherwise.

### 3. Readout timing loses the diffusion name

`SingleShotEpi::time_from_last_diffusion_gradient` becomes `time_from_prep_gradient`. The
computation is unchanged. It measures time from the last large preparation gradient, which is a
diffusion gradient in one consumer and a crusher or labeling gradient in the other.

### 4. The diffusion phase component becomes optional

`PhaseModel` (`phase.rs:132`) holds `BackgroundPhase`, `ShotPhase`, and `DiffusionPhase`
(`phase.rs:101`). The third becomes `prep: Option<PrepPhase>`, with the same fields and the same
evaluation. TRXScan constructs it; aslscan leaves it `None` until P4 introduces vascular crushing.

### 5. The acquisition entry point counts volumes, not gradients

```rust
pub fn simulate_acquisition_oversampled(
    sim_dims: [usize; 3],
    acq_dims: [usize; 3],
    n_volumes: usize,
    images: &[Vec<f32>],
    t2: &[T2Source],
    fmap: &[f32],
    t_inhom: Option<&[f32]>,
    acq: &Acquisition,
    eddy_drive: Option<&[[f64; 3]]>,   // one per volume
    phase: &PhaseModel,
    seed: u64,
    noise_sigma: Option<&[f32]>,
    eddy_trace: Option<&[[f64; 3]]>,
) -> (Vec<f32>, Vec<f32>)
```

`ngrad` becomes `n_volumes`, the `bvals`/`bvecs` pair becomes the single optional `eddy_drive`
slice, and `t_inhom` carries the per-voxel map from change 6. `simulate_acquisition_legacy` gets
the same treatment.

### 6. T2\* inhomogeneity becomes a scalar or a map

`Acquisition.t_inhom` is a global scalar. The readout decay at `kspace.rs:649` combines it with
the per-compartment T2 as `exp(-trf/t2[c] - |t|*1000/t_inhom)`. TRXScan's phantom has no T2\* map,
so a global constant is adequate there. ASLDRO's phantom has a per-voxel T2\* map, and dropping it
would discard susceptibility dropout, which is one of the artifacts ASL most needs to reproduce.

```rust
pub struct SliceInput<'a> {
    /// Per-voxel T2' inhomogeneity time (ms) on the simulation grid, overriding
    /// `Acquisition::t_inhom` where present.
    pub t_inhom: Option<&'a [f32]>,
    ...
}
```

`Acquisition.t_inhom` stays as the fallback, so TRXScan's call sites pass `None` and keep their
current behavior. aslscan derives the map from the phantom as
`1/T2' = 1/T2star - 1/T2`, per voxel, clamped at zero where the maps disagree.

### 7. The complex writer loses the DWI sidecar

`io::write_complex_dwi` and `SidecarInfo` become `io::write_complex_4d` plus a sidecar struct
carrying only the fields common to both modalities. Writing `.bval` and `.bvec` moves into
TRXScan. Writing `_aslcontext.tsv` lives in aslscan.

## P0 acceptance criteria

1. `cargo test` in `mrsim-acq` passes with the default pure-std build, offline.
2. `cargo test` in TRXScan passes unchanged, both with no features and with `--features cli`.
3. `trxscan` run on a fixed small fixture at a fixed seed produces bit-identical NIfTI output
   before and after the extraction.
4. A test asserts that `eddy_drive: Some(bvec * bval)` reproduces the pre-extraction eddy path.
5. A test asserts that `eddy_drive: None` disables eddy exactly as the `bval.abs() > 1e-9` guard
   at `kspace.rs:393` did.
6. `cargo clippy --all-targets` introduces no warnings beyond the roughly six that already exist.

Criterion 3 is the real gate. The extraction is correct only if the output does not move.

---

# P1: aslscan core

## Module map

```
protocol   asl.json + aslcontext.tsv (+ optional TOML overlay) -> Protocol
phantom    BIDS-derivatives maps -> Phantom
kinetic    Buxton general kinetic model -> delta_m
mrsignal   GE / SE / IR steady state -> two compartments
series     volume list from aslcontext, drives mrsim-acq once per volume
bids       part-mag / part-phase NIfTI, aslcontext.tsv, sidecar, ground-truth maps
bin/aslscan.rs   clap CLI
```

## Data flow

```
asl.json + aslcontext.tsv ----+
BIDS phantom maps -----------+
                              v
                     Protocol + Phantom
                              |
                       kinetic::gkm  ->  delta_m
                              |
                       mrsignal      ->  [tissue (T2_tissue), blood (T2_blood)]
                              |
             mrsim_acq::simulate_acquisition_oversampled
                              v
   sub-XX_part-mag_asl.nii.gz + part-phase + aslcontext.tsv + asl.json + ground truth
```

One pass per row of `aslcontext.tsv`.

## Protocol contract

`protocol` reads a BIDS ASL sidecar and an `_aslcontext.tsv`. Fields consumed:

`ArterialSpinLabelingType`, `LabelingDuration`, `PostLabelingDelay`, `BolusCutOffFlag`,
`BolusCutOffTechnique`, `BolusCutOffDelayTime`, `BackgroundSuppression`, `M0Type`,
`RepetitionTimePreparation`, `EchoTime`, `MagneticFieldStrength`, `AcquisitionVoxelSize`,
`SliceTiming`, `PhaseEncodingDirection`, `TotalReadoutTime`, `ParallelReductionFactorInPlane`,
`MultibandAccelerationFactor`.

`PostLabelingDelay` is a scalar or an array. When it is an array its length must equal the number
of non-`m0scan` rows in `aslcontext.tsv`, and a mismatch is an error naming both counts.
`aslcontext.tsv` supplies volume order from the values `m0scan`, `control`, `label`, `deltam`,
and `cbf`.

`M0Type` and `aslcontext.tsv` must agree. `Included` requires at least one `m0scan` row, and that
volume is written into the 4D ASL file. `Separate` forbids `m0scan` rows, and the M0 volume is
written as its own file. `Estimate` and `Absent` forbid `m0scan` rows and write no M0 volume.
Disagreement is an error naming both the `M0Type` value and the offending row.

`ArterialSpinLabelingType` of `PASL`, `CASL`, or `PCASL` selects the kinetic model. For `PASL`,
`BolusCutOffFlag` must be true and `BolusCutOffDelayTime` supplies the bolus duration, because
the kinetic model has no defined bolus length otherwise.

Parameters the acquisition stage needs that BIDS does not express come from an optional TOML
overlay: `ghost_offset`, `n_spikes`, `spike_amplitude`, `n_coils`, `t_inhom`, `window`,
`partial_fourier`, `pf_mode`, `eddy_*`, `noise_variance`, and the oversampling factor. The
overlay maps one-to-one onto `mrsim_acq::Acquisition`. This is also how `mrsim-acq` acquires the
file-driven protocol configuration that TRXScan's README lists as missing.

Precedence is overlay over sidecar over default, and the resolved protocol is written to the
output sidecar.

## Phantom contract

Separate NIfTI files, each with a JSON sidecar carrying `Units`:

| File | Quantity | Units |
|---|---|---|
| `perfusion.nii.gz` | perfusion rate | `ml/100g/min` |
| `att.nii.gz` | arterial transit time | `s` |
| `T1map.nii.gz` | tissue T1 | `s` |
| `T2map.nii.gz` | tissue T2 | `s` |
| `T2starmap.nii.gz` | tissue T2\* | `s` |
| `M0map.nii.gz` | equilibrium magnetization | arbitrary |
| `dseg.nii.gz` | tissue segmentation | label indices |

All maps must share a grid, and a mismatch is an error naming the offending file. Units are read
rather than assumed, and an unexpected unit string is an error rather than a silent conversion.

`tools/hrgt_to_bids.py`, run in the `simasl` micromamba environment, converts ASLDRO's packed 5D
`hrgt_icbm_2009a_nls_v3` ground truth into this layout. That provides a real phantom from the
first commit.

## Kinetic model

`kinetic` implements the Buxton general kinetic model as simasl implements it, so that the two
can be diffed. Perfusion rate is converted from `ml/100g/min` by dividing by 6000
(`src/asldro/filters/gkm_filter.py:105`).

Shared terms, with `f` the converted perfusion rate, `lambda` the blood-brain partition
coefficient, `T1t` tissue T1, `T1b` arterial blood T1, `dt` the transit time, `tau` the label
duration, `t` the signal time, and `alpha` the label efficiency:

```
M0b      = M0_tissue / lambda
1 / T1'  = 1 / T1t + f / lambda
```

Three delivery states, selected per voxel:

```
t <= dt                    ->  delta_m = 0
dt <  t <  dt + tau        ->  bolus arriving
t >= dt + tau              ->  bolus arrived
```

For PASL, with `k = 1/T1b - 1/T1'`:

```
q_arriving = exp(k*t) * (exp(-k*dt) - exp(-k*t)) / (k * (t - dt))
q_arrived  = exp(k*t) * (exp(-k*dt) - exp(-k*(dt + tau))) / (k * tau)

delta_m_arriving = 2 * M0b * f * (t - dt) * alpha * exp(-t / T1b) * q_arriving
delta_m_arrived  = 2 * M0b * f * tau      * alpha * exp(-t / T1b) * q_arrived
```

For CASL and PCASL:

```
q_arriving = 1 - exp(-(t - dt) / T1')
q_arrived  = 1 - exp(-tau / T1')

delta_m_arriving = 2 * M0b * f * T1' * alpha * exp(-dt / T1b) * q_arriving
delta_m_arrived  = 2 * M0b * f * T1' * alpha * exp(-dt / T1b)
                   * exp(-(t - tau - dt) / T1') * q_arrived
```

Every division guards its denominator and yields zero where the denominator is zero, matching
simasl's `np.divide(..., out=zeros, where=...)` pattern. The `t == dt` case in the PASL arriving
branch is the one that matters, and it yields zero rather than a NaN.

`signal_time` is `PostLabelingDelay + LabelingDuration`, taken per volume from the protocol.
Multi-delay is therefore a property of the volume list, not of separate runs. That is the change
that lifts simasl's one-PLD-per-series restriction.

## Signal model

The output of `kinetic` is labeled blood magnetization. It is carried as its own compartment
rather than folded into tissue M0.

```
compartments[0] = static tissue   T2 = T2map,      T2' = from T2starmap and T2map
compartments[1] = labeled blood   T2 = T2_blood,   T2' = same map as tissue
```

simasl instead adds `delta_m` to `M0` as `mag_enc` before relaxation
(`src/asldro/filters/acquire_mri_image_filter.py:46`), so labeled blood relaxes at tissue T2.
Separating the compartments corrects that, and `mrsim-acq` already decays each compartment
independently across the readout (`kspace.rs:649`). Compartment index 2 is where P4's arterial
component goes, and the tissue-blood T2 difference is what P6's multi-TE ASL measures.

`T2_blood` is a scalar, not a map, because the phantom has no blood T2 map. It comes from the
TOML overlay, defaulting by `MagneticFieldStrength` to 0.165 s at 3 T and 0.290 s at 1.5 T. The
blood compartment shares the tissue T2' map, since susceptibility dropout acts on both.

`mrsignal` applies the steady-state signal equation selected by the protocol's acquisition
contrast, one of gradient echo, spin echo, or inversion recovery, to each compartment. For
`control` volumes the blood compartment is zero. For `label` volumes it carries `-delta_m`. For
`m0scan` volumes it is zero and the tissue compartment uses the M0 repetition time.

Background suppression is out of scope for P1 and arrives with P3.

## Output contract

```
sub-XX[_ses-YY]_part-mag_asl.nii.gz
sub-XX[_ses-YY]_part-phase_asl.nii.gz
sub-XX[_ses-YY]_asl.json
sub-XX[_ses-YY]_aslcontext.tsv
sub-XX[_ses-YY]_m0scan.nii.gz + .json     when M0Type is Separate
```

Ground-truth maps resampled to the acquisition grid are written alongside, under a
`ground-truth/` subdirectory: `delta_m`, `perfusion`, `att`, and the tissue maps. They come from
the same arrays the signal stage used, so the answer key and the data cannot disagree. This
mirrors what `trxscan-microstructure` does for diffusion.

The output sidecar echoes the input protocol and adds what the simulator resolved, including the
overlay values and the random seed.

## P1 acceptance criteria

1. `aslscan` simulates a PCASL single-delay series from a BIDS protocol and a converted ASLDRO
   phantom, and writes a complete BIDS ASL output directory.
2. The same works for PASL with a bolus cutoff, and for a multi-delay PCASL series whose
   `PostLabelingDelay` is an array.
3. `bids-validator` reports no errors on the output directory.
4. The end-to-end subtraction property below holds.
5. `cargo test` passes in the pure-std default build.

---

# Testing and verification

## P0

The moved tests must pass unchanged. Output identity on a fixed seed is the gate, as stated in
the P0 acceptance criteria. The `eddy_drive` generalization gets the two targeted tests listed
there, because it is the only change that could alter numbers.

## P1

New code, written test-first.

- **kinetic** — hand-computed closed forms at the boundaries: `PLD < ATT`, `PLD > ATT + tau`,
  `tau` approaching zero, `f = 0`, and `t == dt` in the PASL arriving branch. Then checked-in
  fixtures generated from simasl's `GkmFilter` on a small array, diffed at 1e-9 relative.
  Both sides compute in f64, so the residual is rounding rather than model difference, and a
  larger gap means the port diverged.
  This follows the pattern of `TRXScan/tests/fixtures/force_moments.txt` and
  `TRXScan/tools/gen_force_fixtures.py`.
- **mrsignal** — gradient echo, spin echo, and inversion recovery closed forms, plus fixtures
  from simasl's `MriSignalFilter`.
- **protocol** — parse tests over real BIDS ASL sidecars. Cases: scalar `PostLabelingDelay`,
  array `PostLabelingDelay`, an array whose length disagrees with `aslcontext.tsv`, `PASL`
  without `BolusCutOffFlag`, and overlay precedence over sidecar.
- **phantom** — unit conversion, an unexpected unit string, a missing map, and a map on a
  different grid.
- **bids** — filename construction and `aslcontext.tsv` ordering.

Fixtures are generated with `micromamba run -n simasl`. That environment pins Python 3.8.20,
numpy 1.19.5, nibabel 3.1.1, nilearn 0.6.2, joblib 0.16.0, and jsonschema 3.2.0, and runs
simasl's own suite at 269 passed. nilearn 0.6.2 must be installed with `--no-deps` because it
declares the withdrawn `sklearn` shim, and joblib must stay below 1.0 because nilearn calls
`Memory(cachedir=None)`.

## The end-to-end property

With `snr = 0`, distortion off, no motion, and a single coil, the `control` minus `label`
difference read back from the written NIfTI must equal the `kinetic` stage's `delta_m`, resampled
to the acquisition grid and scaled by the readout's T2\* decay factor, at 1e-4 relative. The
looser bound reflects the f32 output and the resampling, not uncertainty about the physics.

This exercises every stage of the new pipeline in one assertion, and it fails on a sign error or
a control-label swap. Those are the errors a perfusion simulator most needs to exclude, and they
are invisible to per-stage unit tests that check each stage against its own convention.

## Gates that are not automated

TRXScan has no CI, and this spec does not add any. The `trxscan` output-identity check and the
`bids-validator` run are documented manual gates in the acceptance criteria above.

---

# Decisions deferred

These are settled as roadmap direction and specified in their own sub-projects, not here.

- Background suppression, spin and saturation history, and motion during the labeling period: P3.
- Macrovascular compartment, vascular crushing, and physiological noise: P4.
- 3D GRASE and stack-of-spirals readouts, which need a trajectory abstraction in `mrsim-acq`: P5.
- Hadamard time-encoding, Look-Locker, velocity-selective labeling, and multi-TE: P6.
- The `--compat-asldro` path and the voxelwise benchmark: P2. It requires porting numpy's
  MT19937 and its legacy Gaussian polar method to Rust, or running the diff at `snr = 0` and
  comparing noise statistically. P2 chooses between those.

# Risks

**The extraction changes TRXScan's numbers.** Mitigated by acceptance criterion 3, which compares
NIfTI output bit for bit at a fixed seed. If the comparison fails, the cause is one of the seven
interface changes, and each is small enough to bisect.

**The ASLDRO phantom does not carry what the two-compartment model needs.** It has one tissue
type per voxel and no blood volume fraction. P1 treats the blood compartment as a magnetization
map derived from `delta_m` rather than as a volume fraction, so this does not block P1. P4
revisits it when the arterial compartment arrives.

**BIDS ASL sidecars in the wild are incomplete.** Real datasets omit fields that the simulator
needs. The TOML overlay covers the acquisition parameters, and missing kinetic parameters are an
error that names the field rather than a silent default.
