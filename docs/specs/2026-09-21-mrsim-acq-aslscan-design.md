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
| P5 | 3D readouts: GRASE, stack-of-spirals, gradient-echo contrast | P0, P1 |
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
`nufft`, `noise`, `motion`, `config`, `analytic`, and the signal-model-independent half of `io`
(`load_volume`, `hires_grid`, `header_for_grid`, `write_3d`, `write_4d`, `write_3d_i16`, and the
complex writer).

TRXScan keeps `scheme`, `raster`, `signal`, `compartments`, `sphere`, `mixture`,
`microstructure`, `truth`, `gnl`, `benchmark`, its streamline I/O, `write_benchmark`, and its
binaries.

Three items ride along because the moved modules do not compile without them. None is visible
from the module list.

- **`Vec3`** is defined at TRXScan's crate root (`lib.rs:28`) and used by `mat` and `motion`. It
  moves to `mrsim-acq`'s root, and TRXScan re-exports it (`pub use mrsim_acq::Vec3`) so its own
  modules keep compiling unchanged.
- **`Grid`** is defined in `raster` (`raster.rs:18`), which stays, but `io` takes it in every
  signature and `gnl` imports it. The struct and its affine helpers move to a new ungated
  `mrsim_acq::grid` module; `raster` re-exports it and keeps the DDA code. It cannot live in `io`,
  because `io` is feature-gated and `raster` is pure std.
- **`analytic`** is the oracle for a `kspace` test (`kspace.rs:1936`), so it moves with `kspace`.
  Nothing else in TRXScan uses it.

`mrsim-acq`'s `io` feature pulls `nifti`, `ndarray`, and `nalgebra`. The last is needed because
`header_for_grid` builds a `Matrix4` to set the qform and sform (`io.rs:247-254`) and moves with
the writers. TRXScan's `io` today also pulls
`trx-rs`, which builds HDF5 from source and needs cmake and network; that dependency stays with
the streamline loaders in TRXScan, so `aslscan` never pays for it.

TRXScan's `kspace`, `par`, and `io` features forward to the same-named `mrsim-acq` features. If
the forwarding is forgotten the build still succeeds and silently falls back from the NUFFT path
to the std-only one, which is slower and differs in the last bits, so the forwarding is checked by
acceptance criterion 3 being run under both feature sets.

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

TRXScan's call sites wrap their existing scalars in `T2Volume::Uniform`.

This is not a one-line change to the decay expression, and the first draft of this spec was wrong
to call it one. `kspace.rs:647-653` evaluates the decay **once per compartment per PE line**, into
a scalar `rel[c]`, outside the voxel loop. The whole forward model is organised around that fact:
the header comment at `kspace.rs:436` lists relaxation as the "per line" factor, and the NUFFT
path exists only because "the relaxation is an output-side scalar" (`kspace.rs:535-536`), applied
to each compartment's transformed row after the transform (`kspace.rs:681-687`). A per-voxel T2
is not a per-line scalar, so neither structure survives it. The same applies to the per-voxel
`t_inhom` map of change 6, and the two are handled together.

The decay factors as

```
exp(-trf/T2(r) - |t|/T2'(r))  with  trf = t_echo + t
```

and the rule is:

- **Every `t2` and every `t_inhom` entry `Uniform`.** The existing code runs untouched: `rel[c]`
  scalars, NUFFT path eligible. This is every TRXScan call, and it is also every aslscan call in
  `class` mode. It is what keeps criterion 3.
- **Any `Map` in either.** The slice takes the rotor path, and the per-compartment weight `w` at
  `kspace.rs:695-698` becomes `sum_c comp_c[i] * rel_c(i)` with `rel_c(i)` evaluated per voxel per
  line. `Uniform` compartments in a mixed slice still use their scalar. The NUFFT gate at
  `kspace.rs:540` gains this condition alongside `!do_eddy`.

Both terms of `exp(-trf/t2[c] - |t|*1000/t_inhom)` must be scalar for the factoring to work. A
uniform T2 beside a mapped T2' is still the rotor path.

The cost is one `exp` per mapped compartment per sim voxel per acquired line, the same order as
the rotor multiply already in that loop, plus the loss of the NUFFT speed-up. The size of that
speed-up is not recorded anywhere in TRXScan, and no one has measured it; the NUFFT accelerates
the y-stage of the sum only (`kspace.rs:536`), not the whole forward model, so it should be
measured before it is traded away.

**aslscan does not have to take the slow path, and by default it does not.** Both T2
representations are supported, and which one a run uses is a property of the phantom.

The ASLDRO phantom assigns one tissue type per voxel, and T2 is a property of that type. Its
`T2map.nii.gz` is therefore not a smooth field; it is a handful of scalars painted onto `dseg`.
Decomposing that tissue compartment into one compartment per segmentation label gives every
compartment a `Uniform` T2, restores the factoring, and approximates nothing, because the only
edges introduced are the ones the segmentation already has.

This is not the binning that was rejected. Binning a genuinely smooth map into quantiles puts bin
boundaries where the anatomy has none, and in a simulator whose subject is what k-space does to
edges that is a real artifact. Splitting a piecewise-constant map along its own discontinuities
adds nothing that was not there.

`phantom` therefore takes `--t2-mode`, defaulting to `auto`:

| Mode | Behavior |
|---|---|
| `auto` | Run the constancy test below. If it passes, decompose. Otherwise use a single tissue compartment with `Map`. |
| `class` | Force the decomposition. Error, naming the first offending label and voxel, if the test fails. |
| `voxel` | Force the single-compartment `Map` form, whatever the phantom looks like. |

The constancy test is exact bitwise equality of `T2` and the **derived** `T2'` within each
foreground `dseg` label. `T1`, `M0`, perfusion, and transit time are deliberately not tested:
`kinetic` and `mrsignal` run per phantom voxel before the decomposition (see grids and
resampling), so nothing downstream of them needs those to be uniform, and requiring it would
push a phantom with a smooth T1 map and tabulated T2 onto the slow path for no reason. Testing derived T2' rather than raw T2\* matters, because the derivation
collapses several raw cases to `INFINITY`: a label whose T2\* varies only among values that all
exceed its T2 is constant in T2' even though it is not constant in T2\*. Bitwise rather than
tolerant, because a converted phantom is painted from a table of constants and any spread in it
means the phantom is not what `class` mode assumes.

`dseg` label 0 is background. Its magnetization must be zero, `phantom` checks that it is, and it
gets no compartment. Foreground labels require strictly positive T1, T2, and T2\*; a zero there is
an error naming the label and voxel, not a silently infinite relaxation. `dseg` is mandatory, so
its absence is already an error.

`auto` on the converted ASLDRO phantom yields `class`, so the ordinary ASL run keeps the NUFFT
path and the extra compartments cost only the per-compartment work the stage already does. A
future run with a fitted T2 map from a scanner falls to `voxel` on its own, and pays the rotor
cost then, which is the case the cost actually belongs to.

### Compartment ordering

The compartment count is no longer fixed at two, so the order is pinned. For `K` foreground `dseg`
labels sorted ascending:

```
compartments[0 .. K)    tissue, one per foreground label, in sorted label order
compartments[K .. 2K)   labeled blood, one per foreground label, same order
compartments[2K]        arterial blood            (P4, not P1)
```

The blood is decomposed by label too, and that is forced rather than chosen. The blood
compartment's T2' is the T2' of the tissue the labeled water sits in, since susceptibility
dropout acts on both. In `voxel` mode that is one shared map. In `class` mode there is no single
map, only `K` uniform values, and a blood compartment can carry one value or a map. One value
would give WM-resident label GM's dropout; a map would take the whole slice off the fast path,
which is the thing `class` mode exists to keep. So label `i`'s blood is compartment `K + i`,
with `T2 = T2_blood` and label `i`'s uniform T2'. A label with zero perfusion everywhere, CSF in
the ASLDRO phantom, still gets its compartment; it is all zeros and costs one extra row pass,
and a fixed layout is worth more than the saving.

`voxel` mode is the `K = 1` case of this, with the two compartments carrying maps instead of
scalars. Everywhere this document says "the tissue compartment" it means `0..K`, and "the blood
compartment" means `K..2K`. The row semantics table and the linearity property read in those
terms: a row's tissue input is applied to every `0..K` and its blood input to every `K..2K`, and
the linearity property varies only `K..2K`. `mrsignal` is not per compartment at all; it runs per
phantom voxel and the decomposition masks its output.

The mode and the resolved label ordering are recorded in the output sidecar, because together they
determine what a per-compartment debug dump means.

**The two representations are cross-checked, and they are not equivalent.** `class` mode keeps
each tissue's decay separate, so a mixed voxel contributes a sum of exponentials. `voxel` mode
collapses that voxel to one exponential whose rate is the magnetization-weighted mean. Those agree
only where a simulation voxel is homogeneous, because in general

```
  sum_i w_i * exp(-t / T2_i)   !=   exp(-t * sum_i w_i / T2_i)
```

`class` is therefore the reference, and `voxel` is a documented approximation at partial-volume
boundaries, not an equal alternative. That is a second reason to prefer `class` where the phantom
allows it, beyond the NUFFT path.

Two tests follow from that, not one:

- **Equivalence on a homogeneous grid.** With the acquisition grid equal to the phantom grid and
  `o = 1`, every simulation voxel carries exactly one label, the weighted-rate collapse is exact,
  and the two modes must agree to `1e-5` relative. A disagreement here is a bug in the mapped path
  or in the decomposition, and nothing else would find it.
- **Characterization at boundaries.** On the ordinary oversampled grid, the two modes are compared
  and the maximum discrepancy is recorded as a tracked number rather than asserted to be zero. It
  quantifies the `voxel`-mode approximation, and a regression that moves it is worth seeing.

The first of these also measures both paths, which is where the missing speed-up number comes
from.

The exact alternative that keeps an FFT under a genuinely smooth map is a time-segmented NUFFT,
which approximates `exp(-t*rho(r))` by interpolating between a few static weightings. It is a
contained later optimisation and is listed under deferred decisions.

The test oracle `reference_coil_kspace` (`kspace.rs:1479`) gets the same per-voxel lookup. Two new
tests pin the map path: a `Map` filled with one constant reproduces `Uniform` of that constant to
`1e-12` in L2 relative norm over the k-space coefficients, not coefficient-wise relative, which is
unstable wherever a coefficient is near zero (not bit-exact either: the summation order differs),
and the
restructured forward with a varying map matches the literal sum at the existing 1e-10.

**Map contract.** Every `T2Slice::Map` value is strictly positive; `f32::INFINITY` is allowed and
means no decay. Zero is forbidden, for the reason set out under change 6: `-trf/0` is `-inf` when
`trf > 0`, which is harmless, but `NaN` when `trf == 0`, and `0 * NaN` poisons the Fourier sum.
The entry point rejects a non-positive or `NaN` map value before the volume loop. aslscan maps
background `T2 == 0` to `INFINITY`, as it does for `T2'`.

**`trf` must be positive on every acquired line.** `trf = t_echo + t` goes negative for early
lines whenever `t_echo` is shorter than the time from the first acquired line to the echo. TRXScan
never hits this at TE 88 ms. ASL will: TE 12 ms with a 64-line, 0.5 ms/line readout puts the
centre of the first line 15.75 ms before the echo, so `trf = -3.75 ms`. A negative `trf` is a
readout that starts before the excitation, and the code responds with signal *growth*,
`exp(+|trf|/T2)`. The entry point rejects an `Acquisition` whose earliest acquired line (after the
sampling mask) has `trf <= 0`, naming the `t_echo` the protocol must exceed. This is a new check
and it cannot fire on any currently valid TRXScan configuration.

The only remedies in this model are a longer `EchoTime` or a shorter `TotalReadoutTime`. Partial
Fourier is **not** one, although a scanner's is. `sampling_mask` drops the low-ky lines
(`kspace.rs:341-356`), the trajectory visits low ky *last* in both polarities (`readout.rs:44-56`),
and `line_times` (`kspace.rs:237`) times lines by index regardless of the mask, so partial Fourier
removes late lines and leaves the early ones exactly where they were. The 64-line example has the
same `trf` minimum at partial Fourier 1, 3/4, and 1/2. GRAPPA does not help either, for the same
reason: the readout shortens only through the effective `TotalReadoutTime` the sidecar reports,
never through `accel`. A line timing in which partial Fourier shortens the pre-echo readout is a
change to the readout model and is listed under deferred decisions.

### 2. Diffusion gradients become a generic eddy drive

`SliceInput.bvec` and `SliceInput.bval` (`kspace.rs:269-272`) are read only by the eddy model, at
`kspace.rs:391-393`, `kspace.rs:484`, and `kspace.rs:663`. In every one of those places they
appear as the product `bvec * bval`. That product is the drive the legacy eddy model takes, in
its own model units. It is not the physical gradient first moment, and this spec does not call it
one: `phase.rs:93-99` is explicit that TRXScan has no waveform timing and that its q-vector is an
effective one absorbing those constants.

```rust
pub struct SliceInput<'a> {
    /// Per-volume eddy drive in the legacy model's units, `bvec * bval` for diffusion.
    /// `None` disables eddy for this volume, reproducing the old `bval ~= 0` branch.
    pub eddy_drive: Option<[f64; 3]>,
    ...
}
```

TRXScan passes `Some([bvec[0]*bval, bvec[1]*bval, bvec[2]*bval])`, and `None` where `bval` is
within 1e-9 of zero. aslscan passes the crusher moment when vascular crushing is configured, and
`None` otherwise.

The `None` decision is TRXScan's, made on `bval`, and the library must not re-derive it from the
vector. The obvious library-side test, "the drive is the zero vector", is not the same predicate.
FSL-style scheme files routinely carry b0 rows as `bval = 5, bvec = [0, 0, 0]`. Today that row has
`do_eddy = true` with a zero gradient, which multiplies by identity rotors but still disables the
NUFFT path (`kspace.rs:540`). A zero-vector test would flip it to `None`, re-enable the NUFFT, and
move output bits under the `kspace` feature. So `Some([0.0; 3])` and `None` are distinct inputs
with distinct code paths, and `do_eddy` becomes `acq.eddy_strength != 0.0 && eddy_drive.is_some()`.

### 3. Readout timing loses the diffusion name

`Readout::time_from_last_diffusion_gradient` becomes `time_from_prep_gradient`. It is a method of
the `Readout` trait (`readout.rs:14`), not only of `SingleShotEpi`, so the rename is on the trait
and its one implementation. The computation is unchanged. It measures time from the last large preparation gradient, which is a
diffusion gradient in one consumer and a crusher or labeling gradient in the other.

### 4. The diffusion phase component becomes optional, and gets its own drive

`PhaseModel` (`phase.rs:132`) holds a global phase, a `BackgroundPhase`, and a `DiffusionPhase`
(`phase.rs:101`); a realized `ShotPhase` is not a field but is passed to `PhaseModel::at` per call
(`phase.rs:141`). The `DiffusionPhase` becomes `prep: Option<PrepPhase>`, with the same fields and
the same evaluation. TRXScan constructs it; aslscan leaves it `None` until P4 introduces vascular crushing.

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

The entry point takes `&[Option<_>]`, not `Option<&[_]>`, for both drives. A series can have a
prep gradient on some volumes and not others, which is exactly the b0-interleaved diffusion case,
and an outer `Option` can only say all or none. `eddy_drive` has the same shape for the reason
given under change 2: a zero vector does not encode "disabled", so the per-volume `Option` is the
only thing that reproduces the `bval.abs() > 1e-9` branch at `kspace.rs:393`.

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
    t_inhom: Option<&[T2Volume]>,             // one per compartment, see change 6
    acq: &Acquisition,
    eddy_drive: &[Option<[f64; 3]>],           // length n_volumes
    prep_drive: &[Option<(f64, [f64; 3])>],    // length n_volumes
    phase: &PhaseModel,
    seed: u64,
    noise_sigma: Option<&[f32]>,               // unchanged: one acquired-grid map, all volumes
    eddy_trace: Option<&[[f64; 3]]>,
) -> (Vec<f32>, Vec<f32>)
```

`ngrad` becomes `n_volumes`, the `bvals`/`bvecs` pair becomes the two drive slices, and `t_inhom`
carries the per-compartment T2' of change 6, each entry `Uniform` or a `Map` on the simulation
grid with the fieldmap's layout. Both
drive slices must have length `n_volumes`; a mismatch is rejected before the volume loop. So must
`eddy_trace` when present: it is indexed `tr[g]` inside the loop (`kspace.rs:1257`), and today
only TRXScan's CLI checks its length (`trxscan.rs:868-885`). An extracted public entry point
cannot rely on one particular caller having done that.
`do_eddy_phase` (`kspace.rs:394`) takes the same `eddy_drive.is_some()` guard as `do_eddy`.

`noise_sigma` is deliberately left alone. An earlier revision added a per-volume form on the
argument that an M0 volume and a label volume differ in signal by two orders of magnitude. They
do not: `delta_m` is about 1% of M0, but a *label volume* is static tissue minus `delta_m`, within
a few percent of a control volume and the same order as M0. More to the point, thermal noise SD
does not depend on signal level at all, so differing signal would not justify differing noise. P1
does not use `noise_sigma` (its noise is `Acquisition::noise_variance`), so the extension had no
consumer, and it is dropped.

`simulate_acquisition_legacy` gets the same treatment, with one trap. It receives a scaled
gradient, splits it into `bval = |g|` and `bvec = g/|g|`, and lets the forward model recombine
them (`kspace.rs:1332-1335`). `(g/|g|)*|g|` is not bit-equal to `g`. The legacy wrapper therefore
keeps the split and passes `Some([bvec[0]*bval, ...])`, never `Some(g)`, with `None` under the
same `bval.abs() > 1e-9` test the forward model applies today.

### 5a. One call simulates the whole series, and seeds are the caller's problem across calls

Every random stream in the acquisition stage is keyed on the volume index `g` within a call:
`slice_seed` (`kspace.rs:1240`), the image-space noise stream (`kspace.rs:1270`), and the shot
phase (`phase.rs:119`). That is correct for TRXScan, which makes one call per series. It is a trap
for a consumer that calls once per volume with `n_volumes = 1`: `g` is always 0, so every volume
draws the **same** k-space noise, and a control-label subtraction cancels it exactly. The output
looks plausible volume by volume and has infinite perfusion SNR.

The contract is therefore: one call per series, with all volumes in `images`. No interface change
is needed for that, only the statement. A consumer that must make a second call for the same
dataset, as aslscan does for a separate M0 scan, passes a different `seed` and documents how it
was derived.

### 6. T2\* inhomogeneity becomes a scalar or a map

`Acquisition.t_inhom` is a global scalar. The readout decay at `kspace.rs:649` combines it with
the per-compartment T2 as `exp(-trf/t2[c] - |t|*1000/t_inhom)`. TRXScan's phantom has no T2\* map,
so a global constant is adequate there. ASLDRO's phantom has a per-voxel T2\* map, and dropping it
would discard susceptibility dropout, which is one of the artifacts ASL most needs to reproduce.

```rust
pub struct SliceInput<'a> {
    /// Per-compartment T2' inhomogeneity time (ms). Same shape as `t2`, because T2' is a tissue
    /// property exactly as T2 is. `f32::INFINITY` means no inhomogeneity decay.
    pub t_inhom: Option<&'a [T2Slice<'a>]>,
    ...
}
```

`t_inhom` is per compartment, not one shared map. That mirrors `t2` deliberately: T2' is derived
from T2 and T2\*, both of which are tissue properties, so wherever T2 is uniform per compartment
T2' is too. A single shared map would make it impossible to say so, and the consequence is not
cosmetic — the NUFFT gate requires relaxation to be an output-side scalar for *every* term, so a
shared T2' map would take the fast path away even when every compartment's T2 is uniform. With
`t_inhom` per compartment, class mode passes `Uniform` for both and stays on the fast path, which
is the whole point of having class mode.

`Acquisition.t_inhom` stays as the scalar fallback, so TRXScan's call sites pass `None` and keep
their current behavior. The entry point validates each map as it does T2 maps: strictly positive
or `INFINITY`, never zero or `NaN`. aslscan derives T2' from the phantom as
`1/T2' = 1/T2star - 1/T2`, per voxel in `voxel` mode and per label in
`class` mode. The derivation is total, and the stored map contains only finite positive values or
`INFINITY` — never zero, never negative, never `NaN`.

| `dseg` | `T2`, `T2star` | Stored `T2'` |
|---|---|---|
| background (0) | anything finite | `INFINITY`; the compartments are zero there and the values are never read |
| foreground | `0 < T2star < T2` | `1 / (1/T2star - 1/T2)`, finite positive |
| foreground | `0 < T2 <= T2star` | `INFINITY` (no inhomogeneity decay) |
| foreground | either zero or negative | rejected at load, naming label and voxel |
| any | either `NaN` or infinite | rejected at load, naming the voxel |

The same rule applies to T1 and M0 in the foreground. Background voxels are expected to be zero
in a converted phantom and are not an error. `T2star > T2` is physically impossible but common in
noisy real maps, so it is tolerated as no inhomogeneity decay rather than rejected. The rows are
the same in `class` and `voxel` mode; `class` mode merely evaluates them once per label.

A floored rate of zero means no inhomogeneity decay, and it must not be stored as `T2' = 0`. The
decay term at `kspace.rs:649` divides by `t_inhom`. A stored zero gives `x/0`, which is `INFINITY`
for nonzero `x` — total signal loss rather than no decay, the opposite of what was meant — and
`NaN` at the echo line where `t == 0`. A `NaN` weight is not masked by a zero-valued background
compartment, because `0 * NaN` is `NaN`, so one bad voxel contaminates the whole Fourier sum at
`kspace.rs:677-703`.

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
   before and after the extraction, under **both** `--features cli` and
   `--features cli,kspace,par`. The two builds take different forward paths (twiddle tables and
   rotors versus `rustfft` and the NUFFT), and the NUFFT gate is one of the things this work
   edits, so one build does not vouch for the other.
4. A test asserts that `eddy_drive: Some(bvec * bval)` reproduces the pre-extraction eddy path.
5. A test asserts that `eddy_drive: None` disables eddy exactly as the `bval.abs() > 1e-9` guard
   at `kspace.rs:393` did, and that `Some([0.0; 3])` does not.
6. The two map-path tests under change 1 pass, and an `Acquisition` with `trf <= 0` on an
   acquired line is rejected.
7. `cargo clippy --all-targets` introduces no warnings beyond the roughly six that already exist.

Criterion 3 is the real gate. The extraction is correct only if the output does not move.

The baseline for criterion 3 is produced **first**, from TRXScan at commit `57858a5`, before any
file moves, and its checksums are recorded in the P0 plan. A baseline regenerated partway through
proves nothing. The fixture must exercise the paths at risk: at least one b0 row written the FSL
way (`bval > 0`, zero `bvec`), a nonzero `--eddy` run as well as an eddy-free one, multiple coils
with GRAPPA, and the multiband dropout path. TRXScan has no such fixture checked in (`tests/`
holds only `benchmark_cli.rs`), so building it is the first task of P0, not a by-product.

---

# P1: aslscan core

## Module map

```
protocol   asl.json + aslcontext.tsv (+ optional TOML overlay) -> Protocol
phantom    BIDS-derivatives maps -> Phantom
kinetic    Buxton general kinetic model -> delta_m
mrsignal   post-excitation magnetization, per phantom voxel (spin echo only in P1)
resample   phantom grid -> simulation grid and acquisition grid
series     volume list from aslcontext, assembles the 4D compartments, calls mrsim-acq once
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
                       mrsignal      ->  tissue, blood  (per phantom voxel)
                              |
                       resample      ->  simulation grid, split by label in class mode
                              |
             mrsim_acq::simulate_acquisition_oversampled
                              v
   sub-XX_part-mag_asl.nii.gz + part-phase + aslcontext.tsv + asl.json + ground truth
```

One volume per row of `aslcontext.tsv`, and one call to the acquisition stage for the whole
series. The rows are volumes of a single call, not separate calls, for the seeding reason given
under P0 change 5a. The static tissue signal depends on the row only through its resolved
repetition time, `M0 (1 - exp(-TR/T1))`, so it is computed once per distinct
`RepetitionTimePreparation` value and replicated into the 4D layout the entry point takes across
the rows that share it; with a scalar `RepetitionTimePreparation` that is once. At 128 x 128 x 40 simulation
voxels and 60 volumes that is about 160 MB per compartment, and `class` mode on the ASLDRO
phantom has six of them, so roughly 1 GB. That is accepted for P1; if it becomes a problem the
remedy is an entry-point variant that takes a per-volume closure instead of materialised 4D
arrays, which is an `mrsim-acq` change and not an aslscan one.

## Protocol contract

`protocol` reads a BIDS ASL sidecar and an `_aslcontext.tsv`. Fields consumed:

`ArterialSpinLabelingType`, `LabelingDuration`, `PostLabelingDelay`, `BolusCutOffFlag`,
`BolusCutOffTechnique`, `BolusCutOffDelayTime`, `BackgroundSuppression`, `M0Type`,
`RepetitionTimePreparation`, `EchoTime`, `MagneticFieldStrength`, `AcquisitionVoxelSize`,
`MRAcquisitionType`, `SliceTiming`, `SliceEncodingDirection`, `PhaseEncodingDirection`,
`TotalReadoutTime`, `ParallelReductionFactorInPlane`, `MultibandAccelerationFactor`.

`PostLabelingDelay`, `LabelingDuration`, and `RepetitionTimePreparation` are each a scalar or an
array. BIDS defines the array form as one value **per volume**, in acquisition order, `m0scan`
rows included, with `m0scan` entries of `PostLabelingDelay` and `LabelingDuration` set to zero.
The array length must therefore equal the number of rows in `aslcontext.tsv`, all of them, and a
mismatch is an error naming both counts. An earlier revision of this spec counted only
non-`m0scan` rows; that contradicts the standard, would reject valid datasets, and would have
aslscan write sidecars that `bids-validator` fails. A nonzero `m0scan` entry in either timing
array is an error. `aslcontext.tsv` supplies volume order from the values `m0scan`, `control`,
`label`, `deltam`, and `cbf`. A one-element array is an array, not a scalar, and is held to the
per-volume length rule; only a JSON number broadcasts.

Fields P1 does not model are read and refused rather than ignored: `LookLocker: true` (P6) and
`VascularCrushing: true` (P4) are errors naming the sub-project, as `BackgroundSuppression: true`
is (P3). `BackgroundSuppression` must be present, because BIDS requires it and the output sidecar
echoes the input; likewise `BolusCutOffTechnique` when `BolusCutOffFlag` is true, and `M0Estimate`
when `M0Type` is `Estimate`. `BolusCutOffDelayTime` accepts a number or a two-element array; more
entries are an error. Every timing and readout value must be finite and in range (positive
repetition times, echo time, readout time and field strength; non-negative delays and slice
times), and the overlay's values likewise, before anything is simulated.

The `RepetitionTimePreparation` array is how an `Included` M0 volume gets its own, usually longer,
repetition time: row `i` uses entry `i`. With a scalar, every row including `m0scan` uses the same
value, and that is simulated as written rather than second-guessed.

`SliceTiming` is consumed by the kinetic model, not merely echoed. `PostLabelingDelay` is defined
to the excitation of the *first slice* of a 2D acquisition, and every later slice is excited later
and sees a longer delay. That slice-dependent PLD is one of the defining features of 2D ASL data
and is cheap to get right, so the signal time for slice `z` is the row's `t` plus
`SliceTiming[z] - min(SliceTiming)`. The array length must equal the acquired slice count.

`MRAcquisitionType` is consumed rather than echoed, and P1 accepts `2D` only. `3D` is rejected
with an error naming P5, which is where the 3D readouts live; `mrsim-acq`'s readout is in-plane
EPI and has no through-plane encoding to offer a 3D protocol. `SliceTiming` is therefore required
in P1 rather than optional, since a 2D acquisition always has one, and an absent one is an error
rather than a silent no-offset.

`SliceEncodingDirection` is consumed too. Indexing `SliceTiming[z]` by the data's own z index
assumes the slice axis runs in the same direction as the data axis; BIDS uses a trailing `-` to
say it does not, and `protocol` reverses the mapping when it sees one. Ignoring the field would
put the slice timings on the wrong slices in exactly the datasets that bothered to record it.
`MultibandAccelerationFactor` is validated for consistency with `SliceTiming` when both are
present and otherwise only echoed; it has no model behind it until P3 brings motion.

`M0Type` and `aslcontext.tsv` must agree. `Included` requires at least one `m0scan` row, and that
volume is written into the 4D ASL file. `Separate` forbids `m0scan` rows, and the M0 volume is
written as its own file. `Estimate` and `Absent` forbid `m0scan` rows and write no M0 volume.
Disagreement is an error naming both the `M0Type` value and the offending row.

`ArterialSpinLabelingType` of `PASL`, `CASL`, or `PCASL` selects the kinetic model. For `PASL`,
`BolusCutOffFlag` must be true and `BolusCutOffDelayTime` supplies the bolus duration, because
the kinetic model has no defined bolus length otherwise. `BolusCutOffDelayTime` is a number, or
for Q2TIPS a two-element array of the first and last saturation pulse times, which BIDS requires
to satisfy `0 <= first <= last` and which `protocol` validates; the bolus duration
is the first element.

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
`partial_fourier`, `pf_mode`, `eddy_*`, `noise_variance`, `matrix`, `m0_repetition_time`, and the
oversampling factor. The overlay maps onto `mrsim_acq::Acquisition`, but not one-to-one, and the
gaps are where mistakes will land.

`protocol` checks the resolved timing before anything is simulated: the echo time must leave
every acquired line after the excitation (P0 change 1). A short-TE protocol with a long
`TotalReadoutTime` fails here, with the `EchoTime` it must exceed and the `TotalReadoutTime` that
would fit in the message, rather than deep inside the acquisition stage. Partial Fourier and
in-plane acceleration do not shorten the pre-echo readout in this model (change 1), so the
message does not suggest them.

**Units.** `Acquisition` is in milliseconds throughout: `t_line`, `t_echo`, `t_inhom`, `eddy_tau`
(`kspace.rs:147-175`), as is `readout.rs`. BIDS, the phantom, and simasl are in seconds, and so is
the kinetic model: `f = perfusion / 6000` is per second (`gkm_filter.py:105`) and is summed with
`1/T1t` (`gkm_filter.py:141-153`), so `kinetic` and `mrsignal` run in seconds, with the labeling
times, T1, and TR untouched. The conversion boundary is the hand-off to the acquisition stage:
`protocol` converts `EchoTime` and `TotalReadoutTime` to milliseconds for `Acquisition` and the
blood T2 (an overlay or default value, so it is `protocol`'s) through `Protocol::t2_blood_ms`, and
`phantom` converts T2, T2\*, and the derived T2' to milliseconds for the `T2Volume` inputs. Nothing downstream of that boundary converts again, and nothing upstream of it
is in milliseconds.

**Derived fields.** These have no direct counterpart and must be computed:

| BIDS / overlay | `Acquisition` | Rule |
|---|---|---|
| `EchoTime` | `t_echo` | seconds to milliseconds; scalar or all-equal array only, see below |
| `TotalReadoutTime` | `t_line` | `t_line = TotalReadoutTime * 1000 / ny`, the inverse of `trxscan.rs:968` |
| `ParallelReductionFactorInPlane` | `accel` | direct; `acs_lines` comes from the overlay |
| `PhaseEncodingDirection` | `reverse_phase` | `j-` is `false`, `j` is `true`, in the native (unreoriented) frame |
| oversampling factor | — | not an `Acquisition` field; it sets the simulation grid passed to the entry point |

The `PhaseEncodingDirection` row reads backwards and is not a typo. The forward readout starts at
the maximum k-space line (`readout.rs:48`), and `orient` pins the resulting label: the forward
scan is `j-` in the grid's own frame and becomes `j` only after reorientation to LAS
(`orient.rs:244-249`). Getting this wrong flips the distortion direction of every output, and
nothing else in P1 would notice, so a test distorts an off-centre point with a known fieldmap sign
and asserts which way it moves for each label.

`EchoTime` needs its own rule, because `Acquisition` holds one scalar `t_echo` for the whole call
(`kspace.rs:149`) and the series is now one call. BIDS permits an array, and simasl carries a
per-volume echo-time list (`user_parameter_input.py:243`, `examples.py:199`). P1 accepts a scalar,
or an array whose entries are all equal, which it collapses to that value after checking every
entry. An array with unequal entries is rejected with an error naming P6, which is where multi-TE
ASL lives. Making `t_echo` per-volume is the alternative and is a larger interface change than P0
should carry.

The `ny` in the `t_line` rule is the acquired matrix size along the phase-encode axis, matching
`grid.dims[1]` at `trxscan.rs:968`. It is `ny`, not `ny - 1`. The distinction changes every
distorted voxel, and the two conventions are both defensible, so this spec fixes the one TRXScan
already uses rather than inventing a second. This is also how `mrsim-acq` acquires the
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
| `fieldmap.nii.gz` | B0 off-resonance, optional | `Hz` |

All maps must share a grid, and a mismatch is an error naming the offending file. Units are read
rather than assumed, and an unexpected unit string is an error rather than a silent conversion.

The fieldmap is the one optional file, and it is listed because the acquisition entry point takes
a fieldmap unconditionally and EPI distortion is the first artifact this project exists to
reproduce. ASLDRO's ground truth has none. When it is absent, aslscan passes zeros and sets
`do_distortions = false`, and the output sidecar records that no distortion was simulated. Note
the phantom's T2\* map and a supplied fieldmap are independent inputs: nothing checks that the
intravoxel dephasing one implies is consistent with the field gradients of the other.

`tools/hrgt_to_bids.py`, run in the `simasl` micromamba environment, converts ASLDRO's packed 5D
ground truth into this layout. ASLDRO v2.2.0 ships two, `hrgt_icbm_2009a_nls_3t` and
`hrgt_icbm_2009a_nls_1.5t` (`src/asldro/data/filepaths.py`); an earlier revision of this spec
named a `_v3` that does not exist. Each is a 197 x 233 x 189 x 1 x 7 float64 NIfTI at 1 mm with
the quantities `perfusion_rate`, `transit_time`, `t1`, `t2`, `t2_star`, `m0`, and `seg_label`,
and a JSON carrying units, the segmentation names (`grey_matter` 1, `white_matter` 2, `csf` 3),
and a `parameters` block with `lambda_blood_brain`, `t1_arterial_blood`, and
`magnetic_field_strength`. That provides a real phantom from the first commit.

The converter writes the `parameters` block and the label names into a top-level `phantom.json`.
The phantom's `lambda_blood_brain` and `t1_arterial_blood` take precedence over the field-strength
defaults in the protocol contract and yield to the overlay, so the kinetic constants the oracle
was built with are the ones the port uses unless someone says otherwise. A phantom whose
`magnetic_field_strength` disagrees with the sidecar's `MagneticFieldStrength` is an error: the
relaxation values in the maps are field-specific, and simulating a 1.5 T phantom under a 3 T
protocol produces a dataset that is neither.

In these phantoms every map is exactly constant within each label (checked bitwise, 2026-09-23),
so `auto` resolves to `class` with `K = 3`. CSF carries `transit_time = 1000 s` as a
never-arrives sentinel with zero perfusion, which the kinetic model handles by the not-arrived
branch and which the ground-truth writer must not average into neighbouring voxels; see below.

## Grids and resampling

Three grids are in play, and the earlier revisions of this spec named only one of them.

- The **phantom grid** is whatever the maps are on. The ASLDRO ground truth is 1 mm isotropic.
- The **acquisition grid** is axis-aligned with the phantom and covers its field of view. Its
  voxel size is `AcquisitionVoxelSize`, and its matrix is the phantom extent divided by that size,
  rounded up, per axis. An overlay `matrix` overrides the in-plane size. The phase-encode axis
  named by `PhaseEncodingDirection` must be the second data axis, because `mrsim-acq` encodes
  phase along y; P1 rejects `i` and `k` rather than permuting axes, which is `orient`'s job and is
  wired up when a dataset needs it.
- The **simulation grid** is the acquisition grid refined in-plane by the integer oversampling
  factor `o`, with the same slices. `mrsim-acq` requires exactly this relationship
  (`kspace.rs:1212-1215`) and never oversamples z. `io::hires_grid` builds its affine.

The rule for getting from the first to the third is: **evaluate the physics on the phantom grid,
then average magnetization, not parameters.** `kinetic` and `mrsignal` run per phantom voxel. The
two compartment images are then resampled to the simulation grid by overlap-weighted box
averaging: each simulation voxel is the volume-weighted mean of the phantom voxels it overlaps,
with exact fractional weights at the boundaries. In-plane that is a partial-volume average;
through-plane it is an ideal rectangular slice profile over `slice thickness / phantom dz`
phantom slices.

**Class decomposition happens before resampling, never after.** In `class` mode, `kinetic` and
`mrsignal` run per phantom voxel as usual, and the result is split into one magnetization image
per foreground label on the phantom grid, each zero outside its label. Each of those images is
then box-averaged to the simulation grid independently. A boundary simulation voxel therefore
holds a fractional contribution in each of the labels it straddles, every one of which still
carries its own label's uniform T2 and T2'. Nothing is averaged that is not magnetization.

Averaging the T2 map first and then splitting would defeat the whole decomposition: a boundary
voxel would acquire an intermediate T2 belonging to no tissue, the constancy test would fail on
the resampled grid, and the run would silently fall back to `voxel` mode. The order is not an
implementation detail.

This order matters. Magnetization is what physically adds within a voxel, so averaging it gives
the correct partial-volume signal. Averaging perfusion, transit time, or T1 first and then
applying a nonlinear kinetic model gives a different and wrong answer at every tissue boundary,
which for ASL means most of the cortex.

Because `SliceTiming` makes `t` depend on the acquired slice, `kinetic` is evaluated once per
acquired slice, over the slab of phantom voxels that slice overlaps, at that slice's `t`. A
phantom voxel straddling two acquired slices contributes to each at that slice's timing.

In `voxel` mode the relaxation maps cannot follow the same rule, because a voxel has one decay
rate per compartment in `mrsim-acq`, fixed for the whole call. The T2 and T2' maps on the
simulation grid are the mean of the **rates** `1/T2` and `1/T2'` over the overlapped phantom
voxels, weighted by the phantom's equilibrium `M0`, inverted back to times, with a zero mean rate
(or a zero total weight) stored as `INFINITY`. The weight is `M0` and not the signal, on purpose.
The signal differs between the compartments and, for the blood compartment, between volumes:
`delta_m` is zero in every `control` row and follows the delivery state in every `label` row,
while the maps must be one value per voxel per compartment for the series. `M0` is static and
shared. For the tissue compartment this is the first-order-correct single-exponential stand-in
for a multi-exponential voxel, exact where the simulation voxel is homogeneous and slightly
overstating decay where it is not. For the blood compartment's T2' it is a further
approximation: the label in a mixed voxel sits preferentially in whichever tissue has received
it, and an `M0`-weighted rate does not know that. Both rate-derived maps are written under
`ground-truth/` as `desc-acq_T2map` and `desc-acq_T2primemap`, distinct from the plain
volume-weighted `T2map` that section describes, so the approximation is inspectable. `class` mode has no such step, which is the cross-check under change 1.
The fieldmap is volume-weighted averaged in both modes.

Ground-truth outputs on the acquisition grid use the same machinery: `delta_m` and the tissue
signal are box-averaged, and the parameter maps (`perfusion`, `att`, T1, T2) are plain
volume-weighted means, labelled as such in their sidecars. `att` is the exception: it is
averaged over the perfused phantom voxels only (`perfusion > 0`) and written as zero where there
are none, because the ASLDRO phantom marks CSF with a 1000 s sentinel that a plain mean would
smear into every voxel bordering a ventricle. `dseg` is resampled by majority vote.

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
decoration: `gkm_filter.py:157` writes the chained comparison `0 < signal_time <= transit_time`,
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
(`gkm_filter.py:158`), so for positive `t` the equality falls in the not-arrived mask and the
answer is zero by masking, not by the quotient guard. The guard still matters, because simasl
evaluates `delta_m_arriving` over the whole array before masking, and an unguarded division would
produce a NaN in entries that are then discarded. The Rust computes per voxel and branches first,
so it must reproduce the masked result, not the intermediate.

Zero `t1_arterial_blood` is a separate case and is not covered by the division rule. simasl
replaces the entire arterial exponential with zero, not the quotient
(`gkm_filter.py:205`, `gkm_filter.py:252`), so `delta_m` is zero rather than the `exp(0) = 1` that
guarding only the division would give. The two branches also differ in their test: PASL uses
`t1_arterial_blood > 0` and CASL/PCASL uses `t1_arterial_blood != 0`. The port reproduces both,
including the difference.

`tau` and `t` are defined per label type, because PASL and PCASL do not measure the same
durations.

| | `tau` (bolus duration) | `t` (signal time) |
|---|---|---|
| CASL, PCASL | `LabelingDuration` | `PostLabelingDelay + LabelingDuration` |
| PASL | `BolusCutOffDelayTime` (first element) | `PostLabelingDelay` |

The asymmetry is BIDS's, not an oversight. For CASL and PCASL, `PostLabelingDelay` runs from the
**end** of labeling, so the GKM's clock, which starts at labeling onset, needs the labeling
duration added. For PASL, BIDS defines `PostLabelingDelay` from the **middle of the labeling
pulse** to excitation, which is the inversion time TI and already the GKM's `t`. An earlier
revision added `BolusCutOffDelayTime` to it, which would have simulated every PASL dataset at
TI + TI1, typically 0.7 s late. A PASL protocol with `PostLabelingDelay <= BolusCutOffDelayTime`
is rejected: the cutoff pulse would fall after the readout.

For PASL, `LabelingDuration` is not defined by BIDS and is not used. The bolus is ended by the
cutoff pulse, so `BolusCutOffDelayTime` is the bolus duration. `BolusCutOffFlag` must be true, as
stated in the protocol contract; a PASL protocol without it is rejected rather than defaulted,
because there is no bolus duration to assume.

Both columns are per volume when the sidecar fields are arrays, and `t` further gains the
per-slice offset from `SliceTiming` described in the protocol contract.

This is where simasl's single `label_duration` parameter splits in two. simasl has one field
serving both roles (`gkm_filter.py:109`), which works because its input is not BIDS.
Multi-delay is therefore a property of the volume list, not of separate runs. That is the change
that lifts simasl's one-PLD-per-series restriction.

## Signal model

The output of `kinetic` is labeled blood magnetization. It is carried as its own compartment
rather than folded into tissue M0.

```
class mode, K foreground labels:
  compartments[0..K)   static tissue, one per label   T2, T2' uniform per label
  compartments[K..2K)  labeled blood, one per label   T2 = T2_blood, T2' of that label

voxel mode (K = 1):
  compartments[0]      static tissue                  T2 = T2map, T2' = derived map
  compartments[1]      labeled blood                  T2 = T2_blood, T2' = tissue map
```

simasl instead adds `delta_m` to `M0` as `mag_enc` before relaxation
(`src/asldro/filters/acquire_mri_image_filter.py:46`), so the label relaxes at tissue T2.
Separating the compartments lets the label carry its own T2, and `mrsim-acq` does decay each
compartment independently across the readout and sum them coherently before spatial encoding
(`kspace.rs:647`, `kspace.rs:677`). Compartment index `2K` is where P4's arterial component goes.

The claim stops there. Assigning all of `delta_m` blood T2 is a modeling choice, not an
unambiguous correction. GKM `delta_m` is the total perfusion contrast under single-compartment
exchange, so at long post-labeling delay most of it is water that has already exchanged into
tissue and should relax at tissue T2, not blood T2. This spec's arrangement is therefore right at
short delay and wrong at long delay, where simasl's is the reverse. Neither is correct across the
range, and the fix is a real intravascular/extravascular split with an exchange rate, which is P4
work. What P1 buys is the compartment slot and an explicit knob, not better physics everywhere.

`T2_blood` is a scalar, not a map, because the phantom has no blood T2 map. It comes from the
TOML overlay, defaulting by `MagneticFieldStrength` to 0.165 s at 3 T and 0.290 s at 1.5 T. The
blood compartments take the T2' of the tissue they sit in, per the compartment ordering above,
since susceptibility dropout acts on both.

### Division of labor with the acquisition stage

`mrsignal` produces **transverse magnetization immediately after excitation**, and applies no
transverse relaxation at all. Every `exp(-TE/T2)` and `exp(-TE/T2*)` factor belongs to the
acquisition stage, which applies `exp(-trf/T2 - |t|/T2')` per line with `trf = t_echo +
time_from_max_echo` (`readout.rs:64`, `kspace.rs:649`) — a term that already spans excitation to
each sampled line.

This is a departure from simasl, and it has to be. `mri_signal_filter.py:224` and
`mri_signal_filter.py:246` fold `exp(-TE/T2*)` or `exp(-TE/T2)` into the signal equation, because
simasl has no readout model and TE decay has nowhere else to live. Porting those equations
unchanged and then handing the result to `mrsim-acq` would apply transverse relaxation twice. The
fixture tests in the testing section therefore compare `mrsignal` against simasl's equation with
its transverse factor divided out, and that division is part of the fixture generator, not a
tolerance.

`mrsignal` also does not apply the steady-state longitudinal term to the blood compartment.
simasl computes the tissue steady state, then adds `mag_enc` to it, then applies flip angle and
transverse decay (`mri_signal_filter.py:222`, `:245`). `delta_m` is already a magnetization difference
delivered by the kinetic model; running it through an `M0(1-exp(-TR/T1))` recovery term would
attribute to it a saturation history it does not have. The blood compartment gets the flip-angle
factor and nothing else, and in P1 that factor is exactly 1: the only accepted contrast is spin
echo, whose excitation is 90 degrees. A configurable `sin(flip_angle)` arrives with the contrasts
that have one, in P3 and P5.

### Row semantics

Each `aslcontext.tsv` row maps to one volume and one pair of compartment inputs. This table is the
behavioral contract for the whole series.

| Row | tissue compartments `0..K` | blood compartments `K..2K` | Notes |
|---|---|---|---|
| `control` | tissue signal | 0 | |
| `label` | tissue signal | `-delta_m` at this row's PLD | |
| `m0scan` | tissue at this row's `RepetitionTimePreparation` | 0 | Only when `M0Type` is `Included` |
| `deltam` | 0 | `+delta_m` at this row's PLD | A pre-subtracted series |
| `cbf` | rejected in P1 | — | See below |

`control` minus `label` therefore recovers `+delta_m`, which is the sign convention the linearity
property depends on.

`deltam` rows carry no static tissue, so a `deltam` series is not the subtraction of two simulated
volumes. That is deliberate: subtracting two simulated volumes would carry twice the noise of a
real pre-subtracted reconstruction, and P1 has no model of what the scanner did before writing it.

`cbf` rows are rejected in P1 with an error. A CBF map is a quantified output rather than an
acquired volume, and producing one means choosing a quantification model, which this spec does not.

`M0Type == Separate` forbids an `m0scan` row, so no row generates the separate M0 file. It comes
from a second, one-volume call to the acquisition stage, with the same protocol, the blood
compartment zero, and the tissue compartment at the M0 repetition time. That time is the overlay's
`m0_repetition_time`, which is required when `M0Type` is `Separate`: the ASL sidecar's
`RepetitionTimePreparation` describes the ASL series, not the M0 scan, and BIDS puts the M0
scan's own value in the `m0scan.json` this simulator is about to write, not in anything it reads.

Execution is therefore one call for the series plus one for a separate M0. The second call is
volume 0 of its own call, so under the same seed it would draw exactly the k-space noise of the
series' volume 0 (P0 change 5a). It is seeded with `seed ^ 0x4D30_5343_414E` ("M0SCAN"), and both
seeds are written to the output sidecars.

`BackgroundSuppression: true` is rejected in P1 with an error naming P3. The field is read from
the sidecar, and accepting it while modeling nothing would write an output whose sidecar claims
suppression the data does not show.

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
sub-XX[_ses-YY]_part-mag_asl.nii.gz + .json
sub-XX[_ses-YY]_part-phase_asl.nii.gz + .json
sub-XX[_ses-YY]_aslcontext.tsv
sub-XX[_ses-YY]_m0scan.nii.gz + .json     when M0Type is Separate
```

Each part carries a complete sidecar (the phase part's with `Units: "rad"`), and there is no
inheritance-level `sub-XX_asl.json`. An earlier revision listed one; bids-validator 3.0.2 rejects
it as `SIDECAR_WITHOUT_DATAFILE`, since with `part-` entities no `_asl.nii.gz` exists, and it
also checks each part's required keys without merging a less specific sidecar in, so a
`Units`-only phase sidecar fails. Both were found on the first validator run (2026-09-23).

Ground-truth maps resampled to the acquisition grid are written alongside, under a
`ground-truth/` subdirectory: `delta_m`, `perfusion`, `att`, and the tissue maps. They come from
the same arrays the signal stage used, so the answer key and the data cannot disagree. This
mirrors what `trxscan-microstructure` does for diffusion.

The output sidecar echoes the input protocol and adds what the simulator resolved, including the
overlay values and the random seed. Standard keys describe what was **simulated**: where an
overlay overrides the sidecar's `LabelingEfficiency`, or the effective `PartialFourier`,
`ParallelReductionFactorInPlane`, `MultibandAccelerationFactor` or `TotalAcquiredPairs` differ
from the input, the standard key carries the effective value and the input's value is kept under
`AslscanSimulation.InputValuesReplaced`. A sidecar whose standard keys contradict the data would
mislead every consumer that does not know to look in the simulator's block.

## P1 acceptance criteria

1. `aslscan` simulates a PCASL single-delay series from a BIDS protocol and a converted ASLDRO
   phantom, and writes a complete BIDS ASL output directory.
2. The same works for PASL with a bolus cutoff, and for a multi-delay PCASL series whose
   `PostLabelingDelay` is an array.
3. `bids-validator` reports no errors on the output directory.
4. A protocol requesting a gradient-echo or inversion-recovery contrast is rejected with an error
   naming the P1 spin-echo restriction, rather than simulated with spin-echo physics.
5. The blood-compartment linearity property below holds.
6. With noise enabled, the noise in any two volumes of a series is uncorrelated, and so is the
   noise in a separate M0 scan and the series' first volume.
7. `cargo test` passes in the pure-std default build.

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
  array `PostLabelingDelay` with a zero `m0scan` entry, an array whose length disagrees with the
  total row count of `aslcontext.tsv`, a nonzero `m0scan` entry, array
  `RepetitionTimePreparation`, `PASL` without `BolusCutOffFlag`, PASL signal time equal to
  `PostLabelingDelay` with nothing added, a Q2TIPS two-element `BolusCutOffDelayTime`, an echo
  time too short for the readout, both `PhaseEncodingDirection` signs, and overlay precedence
  over sidecar.
- **phantom** — unit conversion, an unexpected unit string, a missing map, a map on a
  different grid, an absent fieldmap, a foreground zero in T2\*, a field-strength mismatch with
  the protocol, and the T2 modes: `auto` resolves to `class` on the converted ASLDRO phantom;
  one perturbed T2 voxel makes `auto` fall back to `voxel` and makes `class` fail naming that
  label and voxel; a perturbation in T2\* that leaves the derived T2' constant does neither; a
  perturbation in T1 does neither.
- **series / class mode** — the two cross-check tests under P0 change 1: mode equivalence on a
  homogeneous grid at `1e-5`, and the tracked boundary discrepancy on the oversampled grid.
- **resample** — a constant image stays constant; total magnetization is conserved to 1e-6
  relative when the grids tile the same extent; a two-tissue boundary voxel equals the
  volume-weighted mean of the two signals and not the signal of the mean parameters; slice `z`
  of a `SliceTiming` protocol matches a single-slice run at `t + SliceTiming[z]`.
- **series** — the noise in two volumes of one series is uncorrelated, and the separate M0
  volume's noise is uncorrelated with series volume 0. This is the test that catches the
  per-call seeding trap of P0 change 5a, which no other test would: every other property holds
  just as well when all volumes share one noise realization.
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
multiplies every compartment and defaults to 100 (`kspace.rs:212`). Above all the written image is
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

Tolerance is `|I_C - I_L - I_B| <= atol + rtol * |I_B|` with `rtol = 1e-5` and
`atol = 1e-6 * max(max|I_B|, max|I_C|)`. The absolute floor is required for two reasons. A pure
relative bound is undefined wherever `delta_m` is zero, which is every not-arrived voxel
(`gkm_filter.py:163`). And the floor must scale with the *tissue* signal, not only the blood:
`I_C` and `I_L` are each stored as `f32`, so their difference carries rounding error of a few ulps
of `|I_C|`, which for a 1% perfusion contrast is a few times `1e-5 |I_B|` and for a just-arriving
bolus is larger than `|I_B|` itself. A floor of `1e-6 max|I_C|` covers that and still fails a sign
or index error, whose residual is of order `2 |I_B|`, about a percent of `|I_C|`.

The identity is not bit-exact and the tolerance is not slack. Run L sums tissue and blood before
the `f64` to `f32` rounding in the coil combine at `kspace.rs:977`, while C and B round
separately, so floating-point
non-distributivity alone breaks equality. The magnitude/phase round trip adds to that: the writer
stores `sqrt(re^2+im^2)` and `atan2(im, re)` as `f32` (`kspace.rs:1278-1279`, `io.rs:353`), and
recovering `mag * exp(i*phase)` through `sin` and `cos` is not an exact inverse.

This still catches what the property was for. A sign error in the label mapping flips `I_C - I_L`
and fails. A control-label swap fails. A blood compartment wired to the wrong index **in some rows
but not others** fails, because the mis-wired rows relax at the wrong T2. A mis-wiring applied to
every row alike does not fail, and it is worth being exact about why: `I_L` and `I_B` then carry
the same mis-wired term, and the identity is linear in whichever compartment that term sits in. An
earlier revision claimed the consistent case fails; the first implementation's negative control
proved it does not (residual 0.11 of the tolerance). Consistent mis-wiring is what the
`class`/`voxel` cross-check and the ground-truth `delta_m` are for. What the identity does not
check either is whether `delta_m` itself is right, which is what the GKM fixtures are for.

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
- A time-segmented NUFFT that keeps the O(N log N) forward path under per-voxel T2 and T2' maps.
  Unscheduled; taken up only if `voxel` mode's measured run time demands it.
- Per-compartment NUFFT eligibility, so a `Uniform` compartment keeps the fast path alongside a
  `Map` one. The data structures at `kspace.rs:554-561` are already per compartment; only the gate
  at `kspace.rs:680` is not. Unscheduled, for the same reason.
- A line timing in which partial Fourier omits the *early* lines and shortens the pre-echo
  readout, as a scanner's does. Today `line_times` times lines by index and the mask drops the
  late, low-ky lines (P0 change 1), so a short-TE full-matrix ASL protocol cannot be simulated at
  its real `EchoTime`. Unscheduled; a change to the readout model, not to an interface.
- Phase encoding along a data axis other than the second, by permuting through `orient`.
  Unscheduled; taken up when a target dataset needs it.
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
needs. The TOML overlay covers the acquisition parameters. The three kinetic parameters with
literature-standard values (label efficiency, lambda, arterial blood T1) take the documented
defaults in the protocol contract, and every default that was applied is recorded in the output
sidecar so it is never silent. Parameters with no defensible default, namely the PASL bolus
duration and the separate M0 scan's repetition time, are an error that names the field.

**aslscan loses the fast forward path only under a voxelwise T2 map.** In `class` mode, which is
what `auto` selects for the ASLDRO phantom, every compartment carries a `Uniform` T2 and the NUFFT
path is eligible exactly as it is for TRXScan. In `voxel` mode every slice takes the rotor path.
ASL matrices are typically much smaller than diffusion ones and the rotor path is cubic in the
linear matrix dimension, so the voxelwise case is expected to be tolerable; the cross-check test
measures both rather than assuming either. If it is not tolerable, the remedy is the
time-segmented NUFFT under deferred decisions, not a T2 approximation in aslscan.

Two things narrow this further. `do_eddy` already disables the NUFFT gate on its own
(`kspace.rs:540`), so P4's vascular crushing gives up the fast path regardless of T2 mode. And the
NUFFT y-sum is already computed per compartment — `rows[c]`, with weight buffers sized
`2 * ncomp` (`kspace.rs:554-561`) — so only the *decision* at `kspace.rs:680` is all-or-nothing.
Making it per-compartment, so a `Uniform` blood compartment keeps the fast path beside a `Map`
tissue compartment, is a contained change. It is deferred rather than specified here, because P0
is a refactor and this would be the one place it changed behavior.
