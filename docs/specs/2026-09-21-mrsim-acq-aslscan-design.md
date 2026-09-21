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
| P3 | ASL motion, background suppression, inversion-recovery contrast | P1 |
| P4 | Kinetic model extensions: macrovascular compartment, physiological noise | P1 |
| P5 | 3D readouts: GRASE, stack-of-spirals, gradient-echo contrast | P0 |
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

Two of those modules are stubs, not working code, and moving them is not extraction.
`noise::add_complex_gaussian` is `todo!()` (`noise.rs:11`); the noise that actually runs lives
inside `kspace.rs` and is reached through `Acquisition::noise_variance` and the `noise_sigma`
argument. `config::load` is `todo!()` (`config.rs:15`) and the `SimConfig` struct is a commented
sketch. P0 moves both files as-is and implements neither. The TOML overlay that P1 needs is new
work in `aslscan`, described under the protocol contract, and it does not discharge the
file-driven configuration TRXScan's README lists as missing.

## Build conventions

`mrsim-acq` keeps TRXScan's build shape. The default build is pure std with every external
dependency optional, so `cargo test` runs offline with no system libraries. Feature flags are
`kspace`, `io`, `config`, and `par`, with the same meanings they have in TRXScan.

The source is hand-formatted at roughly 100 columns. `cargo fmt` is not run repo-wide, and no
`rustfmt.toml` is added.

## Interface changes

Eight changes, each driven by what the ASL consumer needs. Nothing else about the acquisition
stage changes.

### 1. Per-compartment T2 becomes a scalar or a map

`SliceInput.t2` is currently `&'a [f32]`, indexed by compartment at `kspace.rs:649` and
`kspace.rs:1479`. Each compartment decays at its own T2 across the readout. TRXScan supplies one
scalar per compartment. ASLDRO's phantom has per-voxel T2 maps.

Two types, not one. A single enum used at both levels leaves the z stride undefined and the
slicing responsibility unstated, so the dimensionality is carried in the type.

```rust
/// Per-compartment T2 for ONE slice. Goes in `SliceInput`.
pub enum T2Slice<'a> {
    Uniform(f32),
    /// `snx * sny`, layout `x + snx*y`. Same convention as `SliceInput::fmap`.
    Map(&'a [f32]),
}

/// Per-compartment T2 for the WHOLE volume. Goes to the entry point.
pub enum T2Volume<'a> {
    Uniform(f32),
    /// `snx * sny * nz`, layout `x + snx*(y + sny*z)`.
    Map(&'a [f32]),
}

pub struct SliceInput<'a> {
    pub t2: &'a [T2Slice<'a>],   // one entry per compartment
    ...
}
```

The entry point takes `&[T2Volume]` and cuts each z slice into a `T2Slice` itself, in the same
loop that copies compartment and fieldmap slices at `kspace.rs:1223`. `Uniform` passes through
unchanged. Without the split the caller would slice per compartment per z, which is the loop the
entry point exists to own.

TRXScan's call sites wrap their existing scalars in `T2Volume::Uniform`. The decay expression at
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

### 4. The diffusion phase component becomes optional, and gets its own drive

`PhaseModel` (`phase.rs:132`) holds `BackgroundPhase`, `ShotPhase`, and `DiffusionPhase`
(`phase.rs:101`). The third becomes `prep: Option<PrepPhase>`, with the same fields and the same
evaluation. TRXScan constructs it; aslscan leaves it `None` until P4 introduces vascular crushing.

`PrepPhase` needs a per-volume drive of its own, and `eddy_drive` cannot serve.
`DiffusionPhase::shot` forms `q_eff = c_q * sqrt(bval) * bvec_unit` (`phase.rs:110-117`), which is
nonlinear in `bval`, while `eddy_drive` is the linear moment `bvec * bval`. The two happen to be
interconvertible when `bvec` is a unit vector, but `phase.rs:110` divides by `|bvec|` precisely
because it does not assume that, and a spec that silently depends on unit-norm input is a spec
that breaks on the first non-normalized scheme file.

```rust
pub struct SliceInput<'a> {
    /// Per-volume preparation-gradient drive for the phase model: `(magnitude, direction)`.
    /// The direction is NOT required to be normalized. Diffusion passes `(bval, bvec)` exactly
    /// as it reads them. `None` disables the prep phase term for this volume.
    pub prep_drive: Option<(f64, [f64; 3])>,
    ...
}
```

`PrepPhase::shot` keeps the normalization and both early returns that `DiffusionPhase::shot` has
today: the `n < 1e-12` degenerate-direction guard and the non-positive-magnitude return
(`phase.rs:110-118`). The caller does not pre-normalize. That matters for more than tidiness: the
division by `n` happens inside the same expression it does now, so the floating-point operation
order is unchanged and P0's bit-identical criterion survives. Pre-normalizing at the call site
would be mathematically equivalent and would still move output bits.

A positive magnitude paired with a zero direction is therefore handled where it is handled today,
by the `n < 1e-12` guard, and yields a zero phase rather than an error.

`eddy_drive` and `prep_drive` stay separate because they are separate physics. Eddy currents
follow the gradient moment. Motion-induced phase follows the q-vector.

### 5. The acquisition entry point counts volumes, not gradients

```rust
pub fn simulate_acquisition_oversampled(
    sim_dims: [usize; 3],
    acq_dims: [usize; 3],
    n_volumes: usize,
    images: &[Vec<f32>],
    t2: &[T2Volume],
    fmap: &[f32],
    t_inhom: Option<&[f32]>,
    acq: &Acquisition,
    eddy_drive: Option<&[[f64; 3]]>,        // one per volume
    prep_drive: Option<&[(f64, [f64; 3])]>, // one per volume
    phase: &PhaseModel,
    seed: u64,
    // Per-voxel noise SD on the acquired grid. Length `nvox` applies one map to every volume,
    // which is the current behavior; length `nvox * n_volumes` gives each volume its own. ASL
    // needs the latter, because an M0 volume and a label volume differ in signal level by two
    // orders of magnitude.
    noise_sigma: Option<&[f32]>,
    eddy_trace: Option<&[[f64; 3]]>,
) -> (Vec<f32>, Vec<f32>)
```

`ngrad` becomes `n_volumes`, the `bvals`/`bvecs` pair becomes the single optional `eddy_drive`
slice, `prep_drive` carries the phase drive from change 4, and `t_inhom` carries the per-voxel map
from change 6. `noise_sigma` gains a per-volume length option: `kspace.rs:1204` indexes it as
`ns[vox]` only, so today one map serves the whole series. `simulate_acquisition_legacy` gets
the same treatment.

### 6. T2\* inhomogeneity becomes a scalar or a map

`Acquisition.t_inhom` is a global scalar. The readout decay at `kspace.rs:649` combines it with
the per-compartment T2 as `exp(-trf/t2[c] - |t|*1000/t_inhom)`. TRXScan's phantom has no T2\* map,
so a global constant is adequate there. ASLDRO's phantom has a per-voxel T2\* map, and dropping it
would discard susceptibility dropout, which is one of the artifacts ASL most needs to reproduce.

```rust
pub struct SliceInput<'a> {
    /// Per-voxel T2' inhomogeneity time (ms) on the simulation grid, overriding
    /// `Acquisition::t_inhom` where present. `f32::INFINITY` means no inhomogeneity decay.
    pub t_inhom: Option<&'a [f32]>,
    ...
}
```

`Acquisition.t_inhom` stays as the fallback, so TRXScan's call sites pass `None` and keep their
current behavior. aslscan derives the map from the phantom as `1/T2' = 1/T2star - 1/T2`, per
voxel, floored at zero where the maps disagree.

A floored rate of zero means no inhomogeneity decay, and it must not be stored as `T2' = 0`. The
decay term at `kspace.rs:649` divides by `t_inhom`, so a stored zero yields a division by zero and
a NaN at every line.

The map therefore stores the **time** in milliseconds, as `Acquisition::t_inhom` already does, and
represents a zero rate as `f32::INFINITY`. Dividing by infinity gives zero, which is exactly the
no-decay result, with no branch and no NaN.

Storing the reciprocal instead would be the obvious alternative and it is the wrong one. It would
turn `t.abs() * 1000.0 / t_inhom` into `t.abs() * 1000.0 * rate`, and IEEE rounding makes
`x / t` and `x * (1/t)` different in the last bit. That difference propagates through `exp` and
can move an output `f32`, which would break P0 acceptance criterion 3. Keeping the division keeps
TRXScan's arithmetic character for character when `t_inhom` is `None`.

### 7. Multiband dropout stops being b-value-scaled

`apply_multiband_motion` (`motion.rs:314`) attenuates a dropped shot by `1 - severity*(b/b_max)`
and exempts b0 volumes with a `bvals.get(g) < 50.0` test (`motion.rs:370`). That is a
diffusion-specific dropout law sitting in a module this spec moves wholesale, and it is the one
coupling that no other change touches.

```rust
pub enum DropoutLaw {
    /// `1 - severity * (drive / drive_max)`, with `drive < floor` exempt. Diffusion passes
    /// b-values; the floor reproduces the current `50.0` b0 test.
    Scaled { drive: Vec<f64>, floor: f64 },
    /// `1 - severity`, applied to every volume alike.
    Uniform,
}
```

TRXScan passes `Scaled` with its b-values and a floor of 50.0, reproducing current behavior
exactly. aslscan passes `Uniform`, because ASL spoiling does not scale with a diffusion weighting
it does not have. Whether ASL dropout should instead scale with the labeling state is a P3
question, not a P0 one.

### 8. The complex writer loses the DWI sidecar

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
mrsignal   post-excitation magnetization -> two compartments (spin echo only in P1)
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

BIDS does not carry the kinetic parameters the model requires. `LabelingEfficiency` is optional
in the standard and absent from most datasets, and nothing in BIDS expresses the blood-brain
partition coefficient or arterial blood T1. All three are mandatory GKM inputs
(`gkm_filter.py:109`). They come from the TOML overlay, with documented defaults: label
efficiency 0.85 for PCASL and 0.98 for PASL, lambda 0.9 ml/g, and arterial blood T1 by field
strength, 1.65 s at 3 T and 1.35 s at 1.5 T. The sidecar's `LabelingEfficiency` wins when present.

The same applies to the signal equation. BIDS has no field selecting gradient echo, spin echo, or
inversion recovery, so the overlay carries `acq_contrast`, defaulting to `"se"`, which is also
simasl's default. P1 accepts that value and rejects the other two, for the reasons in the signal
model section.

Excitation flip angle, inversion time, and inversion flip angle are deliberately **not** in the
P1 overlay. simasl needs them only for the gradient-echo and inversion-recovery branches
(`mri_signal_filter.py:190`, `mri_signal_filter.py:252`); its spin-echo branch assumes a 90
degree excitation and consumes none of them (`mri_signal_filter.py:235-248`). They arrive with the
contrasts that use them, in P3 and P5.

Parameters the acquisition stage needs that BIDS does not express come from the same optional TOML
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

Three delivery states, selected per voxel. The lower bound on the first is simasl's, not
decoration: `gkm_filter.py:156` writes the chained comparison `0 < signal_time <= transit_time`,
which with a scalar `signal_time` and an array `transit_time` degenerates to a scalar `False` when
`signal_time <= 0`. A faithful port reproduces that, and the Rust states it as an explicit guard
rather than inheriting a Python chained-comparison artifact.

```
0 <  t <= dt               ->  delta_m = 0
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
simasl's `np.divide(..., out=zeros, where=...)` pattern.

`t == dt` does not reach the PASL arriving branch. The arriving mask is strictly `dt < t`
(`gkm_filter.py:157`), so for positive `t` the equality falls in the not-arrived mask and the
answer is zero by masking, not by the quotient guard. The guard still matters, because simasl
evaluates `delta_m_arriving` over the whole array before masking, and an unguarded division would
produce a NaN in entries that are then discarded. The Rust computes per voxel and branches first,
so it must reproduce the masked result, not the intermediate.

Zero `t1_arterial_blood` is a separate case and is not covered by the division rule. simasl
replaces the entire arterial exponential with zero, not the quotient
(`gkm_filter.py:203`, `gkm_filter.py:250`), so `delta_m` is zero rather than the `exp(0) = 1` that
guarding only the division would give. The two branches also differ in their test: PASL uses
`t1_arterial_blood > 0` and CASL/PCASL uses `t1_arterial_blood != 0`. The port reproduces both,
including the difference.

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
(`src/asldro/filters/acquire_mri_image_filter.py:46`), so the label relaxes at tissue T2.
Separating the compartments lets the label carry its own T2, and `mrsim-acq` does decay each
compartment independently across the readout and sum them coherently before spatial encoding
(`kspace.rs:647`, `kspace.rs:677`). Compartment index 2 is where P4's arterial component goes.

The claim stops there. Assigning all of `delta_m` blood T2 is a modeling choice, not an
unambiguous correction. GKM `delta_m` is the total perfusion contrast under single-compartment
exchange, so at long post-labeling delay most of it is water that has already exchanged into
tissue and should relax at tissue T2, not blood T2. This spec's arrangement is therefore right at
short delay and wrong at long delay, where simasl's is the reverse. Neither is correct across the
range, and the fix is a real intravascular/extravascular split with an exchange rate, which is P4
work. What P1 buys is the compartment slot and an explicit knob, not better physics everywhere.

`T2_blood` is a scalar, not a map, because the phantom has no blood T2 map. It comes from the
TOML overlay, defaulting by `MagneticFieldStrength` to 0.165 s at 3 T and 0.290 s at 1.5 T. The
blood compartment shares the tissue T2' map, since susceptibility dropout acts on both.

### Division of labor with the acquisition stage

`mrsignal` produces **transverse magnetization immediately after excitation**, and applies no
transverse relaxation at all. Every `exp(-TE/T2)` and `exp(-TE/T2*)` factor belongs to the
acquisition stage, which applies `exp(-trf/T2 - |t|/T2')` per line with `trf = t_echo +
time_from_max_echo` (`readout.rs:64`, `kspace.rs:649`) — a term that already spans excitation to
each sampled line.

This is a departure from simasl, and it has to be. `mri_signal_filter.py:213` and
`mri_signal_filter.py:233` fold `exp(-TE/T2*)` or `exp(-TE/T2)` into the signal equation, because
simasl has no readout model and TE decay has nowhere else to live. Porting those equations
unchanged and then handing the result to `mrsim-acq` would apply transverse relaxation twice. The
fixture tests in the testing section therefore compare `mrsignal` against simasl's equation with
its transverse factor divided out, and that division is part of the fixture generator, not a
tolerance.

`mrsignal` also does not apply the steady-state longitudinal term to the blood compartment.
simasl computes the tissue steady state, then adds `mag_enc` to it, then applies flip angle and
transverse decay (`mri_signal_filter.py:204`). `delta_m` is already a magnetization difference
delivered by the kinetic model; running it through an `M0(1-exp(-TR/T1))` recovery term would
attribute to it a saturation history it does not have. The blood compartment gets the flip-angle
factor and nothing else.

### P1 supports spin-echo readouts only

The acquisition stage is spin-echo EPI by construction, not merely by naming. `phase.rs:4-9`
states that no phase term derives from the fieldmap, because static off-resonance is refocused at
the spin echo, and that adding `2*pi*fmap*TE` "would double-count B0 and impose gradient-echo
physics on a spin-echo sequence". The `exp(-|t|/T2')` term at `kspace.rs:649` is symmetric about
the echo for the same reason.

A gradient-echo readout refocuses neither. It needs monotonic `T2*` decay from excitation and an
unrefocused `2*pi*fmap*TE` phase term, which is a change to the acquisition physics rather than to
an interface. P1 therefore accepts `se` only, which is also simasl's default
(`user_parameter_input.py` sets `acq_contrast` to `"se"`). A protocol requesting `ge` or `ir` is
rejected with an error naming this restriction.

The two are deferred for different reasons, and it is worth not conflating them. Gradient echo
goes to P5 because it needs a different echo-formation model, which is the same work the 3D
readouts need. Inversion recovery does not: simasl's IR branch applies `exp(-TE/T2)`, the same
transverse factor as its spin echo (`mri_signal_filter.py:283`), and differs only in longitudinal
preparation. IR goes to P3 instead, alongside background suppression, because both are inversion
preparation applied before an otherwise unchanged readout.

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
4. A protocol requesting a gradient-echo or inversion-recovery contrast is rejected with an error
   naming the P1 spin-echo restriction, rather than simulated with spin-echo physics.
5. The blood-compartment linearity property below holds.
6. `cargo test` passes in the pure-std default build.

---

# Testing and verification

## P0

The moved tests must pass unchanged. Output identity on a fixed seed is the gate, as stated in
the P0 acceptance criteria. The `eddy_drive` generalization gets the two targeted tests listed
there. It is not the only change that could alter numbers, which is why criterion 3 is a
whole-output comparison rather than a spot check: the T2' sentinel (change 6) and the prep-drive
split (change 4) both touch arithmetic that runs on every line, and both are specified to keep
the existing operation order for exactly that reason.

## P1

New code, written test-first.

- **kinetic** — hand-computed closed forms at the boundaries: `PLD < ATT`, `PLD > ATT + tau`,
  `tau` approaching zero, `f = 0`, and `t == dt`, which falls in the not-arrived state and must
  return zero by masking rather than through the PASL quotient guard. Then checked-in
  fixtures generated from simasl's `GkmFilter` on a small array, diffed at 1e-9 relative.
  That threshold holds only if the oracle runs in f64, and it does not by default: simasl
  preserves the NIfTI storage dtype rather than promoting through `get_fdata`, so a float32
  phantom makes `perfusion_rate = image / 6000.0` float32 too (`gkm_filter.py:105`). The fixture
  generator therefore casts every input map to float64 before calling the filter, and records the
  dtype it used in the fixture header. Without that cast the achievable threshold is about 1e-7.
  This follows the pattern of `TRXScan/tests/fixtures/force_moments.txt` and
  `TRXScan/tools/gen_force_fixtures.py`.
- **mrsignal** — the spin-echo closed form, plus fixtures from simasl's `MriSignalFilter` with
  its transverse factor divided out, per the division of labor above. The generator asserts that
  `exp(-TE/T2)` is finite and non-zero for every fixture voxel before dividing; a small enough
  positive T2 underflows the exponential to zero and makes the division undefined. Exact
  `T2 == 0` is not the failure case, because simasl's guard yields `exp(0) = 1` there
  (`mri_signal_filter.py:170-173`). A test asserts that a
  protocol requesting `ge` or `ir` is rejected rather than silently simulated.
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

## The blood-compartment linearity property

The obvious property — that `control` minus `label` equals `delta_m` resampled and scaled by a
T2\* factor — is false, and it is worth saying why, because it is the version a reader will expect.

There is no single readout T2\* factor: relaxation varies per phase-encode line through both `trf`
and `|t|` (`kspace.rs:649`), so the operator is a k-space weighting, not a scalar. Partial Fourier
zeroes whole PE lines (`kspace.rs:336`) and the reconstruction window multiplies k-space before
inversion (`kspace.rs:934`). Oversampled acquisition is deliberate Fourier-band truncation that
creates Gibbs ringing (`kspace.rs:911`), so it is not image-space resampling. `signal_scale`
multiplies every compartment and defaults to 100 (`kspace.rs:211`). Above all the written image is
magnitude (`kspace.rs:1278`), and `|T| - |T - B|` is not `|B|`.

The property that is true is **linearity in the blood compartment**. Under the settings listed
below, the forward model is linear in the compartment images up to the magnitude/phase split, so
three runs that differ only in their compartment inputs satisfy the identity below in exact
arithmetic, and satisfy it to the stated tolerance in `f32`.

```
run C:  tissue = S_tissue,  blood = 0           ->  complex image  I_C
run L:  tissue = S_tissue,  blood = -delta_m    ->  complex image  I_L
run B:  tissue = 0,         blood = +delta_m    ->  complex image  I_B

assert:  I_C - I_L == I_B
```

The complex images are reconstructed from the written `part-mag` and `part-phase` files, so the
assertion runs on the output, not on an internal buffer.

Most acquisition effects cancel because all three runs share them. Per-compartment relaxation and
the coherent sum (`kspace.rs:647`, `kspace.rs:695`), Fourier encoding, partial Fourier, the
reconstruction window, ghosting, fixed distortion, and `signal_scale` are all linear in the
compartment images and may stay enabled. So may multiple coils: the Roemer denominator is built
from `coil_sensitivity` alone (`kspace.rs:967-977`), so it does not depend on the data.

Two effects are not linear and must be disabled.

**GRAPPA.** The weights are calibrated from each run's own ACS lines. `AᴴA` and `Aᴴb` accumulate
from the current k-space (`kspace.rs:1126-1142`) and are solved per run (`kspace.rs:1149`), so
runs C, L, and B apply three different operators. The fixed `1e-4` Tikhonov term (`kspace.rs:1147`)
means even a pure amplitude change moves the weights. The test requires `accel = 1`.

**Spikes.** A spike is not additive. The code finds the peak sample of the current run, scales it,
and overwrites the chosen k-space locations with that value (`kspace.rs:729-745`). A fixed seed
fixes which locations are hit, not what value lands there, and an overwrite destroys the
contribution it replaces. The test requires `n_spikes = 0`.

Required settings: `noise_variance = 0`, `noise_sigma = None`, `accel = 1`, `n_spikes = 0`, no
motion, a fixed seed, and identical acquisition parameters across the three runs.

Tolerance is `|I_C - I_L - I_B| <= atol + rtol * |I_B|` with `rtol = 1e-5` and `atol = 1e-6 *
max|I_B|`. The absolute floor is required because a pure relative bound is undefined wherever
`delta_m` is zero, which is every not-arrived voxel (`gkm_filter.py:163`).

The identity is not bit-exact and the tolerance is not slack. Run L sums tissue and blood before
the `f32` rounding at `kspace.rs:1278`, while C and B round separately, so floating-point
non-distributivity alone breaks equality. The magnitude/phase round trip adds to that: the writer
stores `sqrt(re^2+im^2)` and `atan2(im, re)` as `f32` (`kspace.rs:1278-1279`, `io.rs:353`), and
recovering `mag * exp(i*phase)` through `sin` and `cos` is not an exact inverse.

This still catches what the property was for. A sign error in the label mapping flips `I_C - I_L`
and fails. A control-label swap fails. A blood compartment wired to the wrong index fails. What it
does not check is whether `delta_m` itself is right, which is what the GKM fixtures are for.

## Gates that are not automated

TRXScan has no CI, and this spec does not add any. The `trxscan` output-identity check and the
`bids-validator` run are documented manual gates in the acceptance criteria above.

---

# Decisions deferred

These are settled as roadmap direction and specified in their own sub-projects, not here.

- Background suppression, inversion-recovery contrast, spin and saturation history, and motion
  during the labeling period: P3. Background suppression and IR share the inversion machinery.
- Macrovascular compartment, vascular crushing, and physiological noise: P4.
- 3D GRASE and stack-of-spirals readouts, and gradient-echo contrast, all of which need a
  different echo-formation model in `mrsim-acq`: P5.
- Hadamard time-encoding, Look-Locker, velocity-selective labeling, and multi-TE: P6.
- The `--compat-asldro` path and the voxelwise benchmark: P2. It requires porting numpy's
  MT19937 and its legacy Gaussian polar method to Rust, or running the diff with noise disabled
  on both sides and comparing noise statistically. Note that simasl is configured by `desired_snr`
  while `mrsim-acq` takes `noise_variance` (`kspace.rs:154`), so P2 also has to define the
  conversion between them and which one wins.

# Risks

**The extraction changes TRXScan's numbers.** Mitigated by acceptance criterion 3, which compares
NIfTI output bit for bit at a fixed seed. If the comparison fails, the cause is one of the eight
interface changes, and each is small enough to bisect.

**The ASLDRO phantom does not carry what the two-compartment model needs.** It has one tissue
type per voxel and no blood volume fraction. P1 treats the blood compartment as a magnetization
map derived from `delta_m` rather than as a volume fraction, so this does not block P1. P4
revisits it when the arterial compartment arrives.

**BIDS ASL sidecars in the wild are incomplete.** Real datasets omit fields that the simulator
needs. The TOML overlay covers the acquisition parameters, and missing kinetic parameters are an
error that names the field rather than a silent default.
