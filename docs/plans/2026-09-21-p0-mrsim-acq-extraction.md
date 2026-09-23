# P0: mrsim-acq Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract TRXScan's acquisition stage into a new `mrsim-acq` crate that TRXScan consumes as a path dependency, applying eight interface changes, without moving a single output bit.

**Architecture:** Move first, change second. Tasks 1-6 relocate modules and prove the relocation is inert by diffing NIfTI checksums against a baseline captured before anything moves. Tasks 7-13 then apply the eight interface changes one at a time, re-running that same bit-identity gate after each. No task changes both location and behavior.

**Tech Stack:** Rust 2021, `cargo` workspaces avoided in favor of sibling path dependencies. Optional deps: `rustfft`/`nalgebra` (`kspace`), `nifti`/`ndarray`/`nalgebra` (`io`), `rayon` (`par`), `serde`/`toml` (`config`). Fixture generation uses Python in the `simasl` micromamba env (nibabel 3.1.1, numpy 1.19.5).

**Spec:** `/mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq/docs/specs/2026-09-21-mrsim-acq-aslscan-design.md` (committed at `a3629d8`)

## Global Constraints

- **Default build is pure std.** Every external dependency is optional and off by default. `cargo test` with no features must run offline with no system libraries.
- **Feature flags** are exactly `kspace`, `io`, `config`, `par`, with the same meanings as in TRXScan. TRXScan's `kspace`, `par`, and `io` forward to the same-named `mrsim-acq` features.
- **`mrsim-acq`'s `io` feature pulls `nifti`, `ndarray`, and `nalgebra` only.** It must never pull `trx-rs`, which builds HDF5 from source and needs cmake and network. Streamline I/O stays in TRXScan.
- **Source is hand-formatted at roughly 100 columns.** Do NOT run `cargo fmt` repo-wide. Do NOT add `rustfmt.toml`. TRXScan's source is hand-formatted and `cargo fmt` rewrites ~2300 lines.
- **Tests live as `#[cfg(test)] mod tests` at the bottom of each module.** They move with their modules. `tests/` holds fixture data and integration entry points only.
- **Bit-identity is the gate.** After every task from Task 6 onward, `trxscan` output must be byte-identical to the Task 1 baseline under **both** `--features cli` and `--features cli,kspace,par`.
- **P0 changes no behavior** beyond the eight interface changes in the spec. Any other behavioral difference is a bug in the extraction.
- **`cargo clippy --all-targets`** must introduce no warnings beyond the ~6 that already exist. There is no CI and no `deny(warnings)`.
- **`SliceInput` is a struct literal, so each interface-change task breaks the previous task's tests.** Tasks 8-11 each add or change a field, and every `SliceInput { .. }` in the test modules must be updated in the same task that changes the struct. This is expected churn, not a sign something is wrong; the compiler names every site.

---

## File Structure

**New crate `mrsim-acq/` (currently holds only `docs/`):**

| Path | Responsibility |
|---|---|
| `Cargo.toml` | Crate manifest, four optional features, no default features |
| `src/lib.rs` | Module declarations, `pub type Vec3 = [f64; 3]` |
| `src/mat.rs` | std-only 3x3 / vector helpers (moved) |
| `src/orient.rs` | Voxel-axis reorientation to LAS (moved) |
| `src/grid.rs` | **New module.** `Grid` struct + affine helpers, lifted out of `raster.rs:18` |
| `src/analytic.rs` | Analytic Fourier oracles; moves because a `kspace` test uses it |
| `src/phase.rs` | Object phase model; `DiffusionPhase` becomes `PrepPhase` (change 4) |
| `src/readout.rs` | EPI trajectory and timing; trait method renamed (change 3) |
| `src/noise.rs` | `todo!()` stub, moved as-is |
| `src/motion.rs` | Poses, multiband schedule, dropout; gains `DropoutLaw` (change 7) |
| `src/config.rs` | `todo!()` stub, moved as-is |
| `src/kspace.rs` | Forward model and reconstruction; changes 1, 2, 5, 6 land here |
| `src/nufft.rs` | Type-1 NUFFT behind `kspace` |
| `src/io.rs` | `load_volume`, `hires_grid`, `header_for_grid`, `write_3d`, `write_4d`, `write_3d_i16`, `write_complex_4d` (change 8) |

**Modified in `TRXScan/`:**

| Path | Change |
|---|---|
| `Cargo.toml` | Add `mrsim-acq` path dep; features forward |
| `src/lib.rs` | Delete moved `pub mod` lines; `pub use mrsim_acq::Vec3` |
| `src/raster.rs` | `Grid` definition removed, re-exported from `mrsim_acq::grid` |
| `src/io.rs` | Keeps streamline loaders, `write_benchmark`, `write_dwi`; bval/bvec writing |
| `src/bin/trxscan.rs` | Call-site updates for every interface change |
| `src/bin/trxscan_microstructure.rs`, `trxscan_benchmark.rs`, `trxscan_gnl.rs` | Import path updates; `trxscan_benchmark.rs` also follows `produce_slice`'s signature |
| `src/benchmark.rs` | Stays, but builds `SliceInput` and `DiffusionPhase` literals (`:85-95`, `:234`, `:265`), so Tasks 8, 9 and 10 each touch it |
| `tests/fixtures/p0_baseline/` | **New.** Baseline inputs + checksums (Task 1) |
| `tools/gen_p0_fixture.py` | **New.** Fixture generator (Task 1) |

---

### Task 1: Capture the bit-identity baseline

Nothing moves in this task. Its only output is a fixture and a set of checksums taken from TRXScan exactly as it stands. A baseline regenerated partway through the extraction proves nothing.

`57858a5` is current `HEAD` of branch `gnl`, so no checkout is needed — but verify that before generating, because the whole gate depends on it.

The fixture must exercise the paths the eight changes put at risk: an FSL-style b0 row (`bval > 0`, zero `bvec`) for change 2, a nonzero `--eddy` run and an eddy-free run for changes 2 and 5, multiple coils with GRAPPA for the reconstruction path, multiband dropout for change 7, and **both entry points**: `--oversample 2` (the default, `simulate_acquisition_oversampled`) and `--oversample 1` (`simulate_acquisition_legacy`, which changes 1, 2, 4 and 5 also edit). Two things about the binary shape this. The oversampled path *requires* `--sim-wm/--sim-gm/--sim-csf/--sim-mask/--sim-fmap` on a grid exactly `o` times finer in-plane (`src/bin/trxscan.rs:372-405`) and errors without them, so the fixture carries two grids. And `acs_lines` is hard-coded to 24 (`src/bin/trxscan.rs:854`), so the PE axis must be long enough for lines to exist outside the ACS band or GRAPPA never synthesizes anything.

**Files:**
- Create: `TRXScan/tools/gen_p0_fixture.py`
- Create: `TRXScan/tests/fixtures/p0_baseline/` (inputs + `checksums.txt`)
- Create: `TRXScan/tools/run_p0_baseline.sh`

**Interfaces:**
- Consumes: nothing
- Produces: `TRXScan/tests/fixtures/p0_baseline/checksums.txt`, the file every later task diffs against. Format: one `sha256sum`-style line per output file, for each of the four runs described below.

- [ ] **Step 1: Verify the baseline commit is HEAD**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
test "$(git rev-parse HEAD)" = "$(git rev-parse 57858a5)" && echo "OK: HEAD is the baseline commit" || echo "STOP: HEAD is not 57858a5"
git status --short
```

Expected: `OK: HEAD is the baseline commit` and a clean working tree. If HEAD is not `57858a5`, stop and ask — the spec pins this commit and a baseline from a different one is worthless.

- [ ] **Step 2: Write the fixture generator**

Create `TRXScan/tools/gen_p0_fixture.py`. It writes a deliberately tiny phantom so the O(N^3) direct DFT stays fast.

```python
"""Generate the P0 bit-identity fixture on TWO grids.

trxscan's production path is `--oversample 2` (the default), which reads the object from a
finer SIMULATION grid via `--sim-wm/--sim-gm/--sim-csf/--sim-mask/--sim-fmap` and reads the
ACQUISITION grid via `--wm/--gm/--csf/--mask` only for the output matrix
(`src/bin/trxscan.rs:372-405`). Without the `--sim-*` files the binary exits with an error,
and with `--oversample 1` it takes the legacy entry point instead, so a fixture on one grid
cannot gate `simulate_acquisition_oversampled` at all. This writes both grids, plus an FSL
scheme with a b0 row written the FSL way (bval > 0, zero bvec), and a few streamlines.

The phase-encode axis (y) is 32 lines, not 8. `acs_lines` is hard-coded to 24
(`src/bin/trxscan.rs:854`); on an 8-line axis every line is inside the ACS band and GRAPPA
calibrates but synthesizes nothing. On 32 lines it synthesizes lines 1, 3 and 29.

Run inside the simasl micromamba env for nibabel/numpy.
"""
import os
import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, TckFile

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "p0_baseline")
O = 2                                   # in-plane oversampling, trxscan's default
ACQ_DIMS = (8, 32, 4)
SIM_DIMS = (ACQ_DIMS[0] * O, ACQ_DIMS[1] * O, ACQ_DIMS[2])
ACQ_AFFINE = np.diag([2.0, 2.0, 3.0, 1.0])
# Sim cells 2X and 2X+1 tile acquired voxel X, whose centre is at 2X mm, so the sim cell
# centres sit at 2X-0.5 and 2X+0.5 mm. z is never oversampled.
SIM_AFFINE = np.array(
    [[1.0, 0.0, 0.0, -0.5], [0.0, 1.0, 0.0, -0.5], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def write_volume(name, data, affine):
    img = nib.Nifti1Image(np.ascontiguousarray(data, dtype=np.float32), affine)
    img.to_filename(os.path.join(OUT, name))


def block_mean(v):
    """Simulation grid -> acquisition grid: mean over each O x O in-plane block."""
    x, y, z = v.shape
    return v.reshape(x // O, O, y // O, O, z).mean(axis=(1, 3))


def fieldmap(dims, affine):
    """A smooth, nonzero off-resonance field (Hz), evaluated in WORLD mm so the two grids
    describe the same field rather than two differently-stretched copies of one pattern."""
    ijk = np.indices(dims).reshape(3, -1).astype(float)
    xyz = affine[:3, :3] @ ijk + affine[:3, 3:4]
    f = 12.0 * np.sin(xyz[0] / 6.0) + 7.0 * np.cos(xyz[1] / 5.0)
    return f.reshape(dims)


def main():
    os.makedirs(OUT, exist_ok=True)

    # Tissue fractions on the SIM grid. The block starts at sim cell 5, halfway through
    # acquired voxel 2, so the acquisition grid carries a genuine partial-volume edge and
    # the rim of the FOV stays empty.
    wm = np.zeros(SIM_DIMS, dtype=np.float32)
    gm = np.zeros(SIM_DIMS, dtype=np.float32)
    csf = np.zeros(SIM_DIMS, dtype=np.float32)
    wm[5:12, 8:56, 1:3] = 0.6
    gm[5:12, 8:56, 1:3] = 0.3
    csf[5:12, 8:56, 1:3] = 0.1
    mask = (wm + gm + csf > 0).astype(np.float32)

    for name, v in [("wm", wm), ("gm", gm), ("csf", csf), ("mask", mask)]:
        write_volume(f"sim_{name}.nii.gz", v, SIM_AFFINE)
        acq = block_mean(v)
        if name == "mask":
            acq = (acq > 0).astype(np.float32)
        write_volume(f"{name}.nii.gz", acq, ACQ_AFFINE)

    write_volume("sim_fmap.nii.gz", fieldmap(SIM_DIMS, SIM_AFFINE), SIM_AFFINE)
    write_volume("fmap.nii.gz", fieldmap(ACQ_DIMS, ACQ_AFFINE), ACQ_AFFINE)

    # Scheme: one FSL-style b0 row (bval>0, zero bvec) plus three DWI rows.
    # The b0 row is the change-2 regression case and must not be bval=0.
    bvals = [5.0, 1000.0, 1000.0, 1000.0]
    bvecs = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
    ]
    with open(os.path.join(OUT, "scheme.bval"), "w") as f:
        f.write(" ".join(f"{b:g}" for b in bvals) + "\n")
    with open(os.path.join(OUT, "scheme.bvec"), "w") as f:
        for axis in range(3):
            f.write(" ".join(f"{v[axis]:.16g}" for v in bvecs) + "\n")

    # Streamlines in RAS mm, crossing the live block (x 4.5-11.5, y 7.5-55.5, slices 1-2)
    # along two directions so the rasteriser sees more than one orientation per voxel.
    lines = []
    for y in np.linspace(12.0, 50.0, 6):
        lines.append(np.array([[4.0, y, 4.5], [12.0, y, 4.5]], dtype=np.float32))
    for x in np.linspace(5.5, 10.5, 4):
        lines.append(np.array([[x, 6.0, 4.5], [x, 56.0, 4.5]], dtype=np.float32))
    tractogram = Tractogram(lines, affine_to_rasmm=np.eye(4))
    TckFile(tractogram).save(os.path.join(OUT, "streamlines.tck"))

    print("wrote fixture to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the fixture**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
micromamba run -n simasl python tools/gen_p0_fixture.py
ls tests/fixtures/p0_baseline/
```

Expected: `wm gm csf mask fmap` and `sim_wm sim_gm sim_csf sim_mask sim_fmap` (all `.nii.gz`), `scheme.bval scheme.bvec streamlines.tck`. Check the grids: `sim_*` must be `16 x 64 x 4` and the rest `8 x 32 x 4`.

- [ ] **Step 4: Write the baseline runner**

Create `TRXScan/tools/run_p0_baseline.sh`. It runs five configurations under two feature sets and writes checksums. The five configurations between them cover every at-risk path. `--fmap` is passed always: the legacy configuration requires it and the oversampled ones ignore it (`src/bin/trxscan.rs:399-402`).

```bash
#!/usr/bin/env bash
# Run the P0 fixture under both feature sets and emit checksums.
# Usage: tools/run_p0_baseline.sh <output-checksum-file>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$REPO/tests/fixtures/p0_baseline"
OUTFILE="${1:?usage: run_p0_baseline.sh <output-checksum-file>}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

COMMON=(--wm "$FIX/wm.nii.gz" --gm "$FIX/gm.nii.gz" --csf "$FIX/csf.nii.gz"
        --mask "$FIX/mask.nii.gz" --fmap "$FIX/fmap.nii.gz"
        --sim-wm "$FIX/sim_wm.nii.gz" --sim-gm "$FIX/sim_gm.nii.gz"
        --sim-csf "$FIX/sim_csf.nii.gz" --sim-mask "$FIX/sim_mask.nii.gz"
        --sim-fmap "$FIX/sim_fmap.nii.gz"
        --streamlines "$FIX/streamlines.tck"
        --bval "$FIX/scheme.bval" --bvec "$FIX/scheme.bvec"
        --seed 20260921)

run_config() {  # $1 = features, $2 = config name, rest = extra flags
  local features="$1" name="$2"; shift 2
  local out="$WORK/${name}_$(echo "$features" | tr ',' '_')"
  cargo run --quiet --release --features "$features" --bin trxscan -- \
    "${COMMON[@]}" "$@" -o "$out" >/dev/null
  for f in "$out"*; do
    [ -f "$f" ] || continue
    # Basename only. The directory is a fresh mktemp every run, so including it would make
    # every manifest line unique per invocation and Step 7 could never pass. The basename
    # already carries the configuration and feature-set names.
    printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(basename "$f")"
  done
}

: > "$OUTFILE"
for FEATURES in "cli" "cli,kspace,par"; do
  # 1. plain: oversampled production path (o=2), no eddy, single coil, no accel, no MB
  run_config "$FEATURES" plain                          >> "$OUTFILE"
  # 2. eddy: nonzero linear and quadratic eddy + eddy phase
  run_config "$FEATURES" eddy    --eddy 0.03 --eddy-quad 0.01 --eddy-phase 0.02 >> "$OUTFILE"
  # 3. parallel: multiple coils with GRAPPA
  run_config "$FEATURES" parallel --coils 4 --accel 2   >> "$OUTFILE"
  # 4. multiband: within-volume motion dropout
  run_config "$FEATURES" mb      --mb 2 --dropout-rate 0.5 >> "$OUTFILE"
  # 5. legacy: simulate_acquisition_legacy (o=1), also touched by changes 1, 2, 4, 5.
  #    Eddy must be ON here: the legacy wrapper's own trap, (g/|g|)*|g| != g, is only
  #    reachable through the eddy model, and with eddy off (kspace.rs:393-394) the gate
  #    could not tell Some(g) from Some(bvec*bval).
  run_config "$FEATURES" legacy  --oversample 1 --eddy 0.03 --eddy-quad 0.01 --eddy-phase 0.02 >> "$OUTFILE"
done

sort -k2 -o "$OUTFILE" "$OUTFILE"
wc -l "$OUTFILE"
```

- [ ] **Step 5: Confirm the flag spellings before trusting the script**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
cargo run --quiet --release --features cli --bin trxscan -- --help
```

Expected: every flag used in `run_p0_baseline.sh` appears. `--eddy-quad`, `--eddy-phase`, `--dropout-rate`, `--mb`, `--accel`, `--coils`, `--seed`, `--oversample`, and the five `--sim-*` flags are the ones to check; clap derives kebab-case from the field names at `src/bin/trxscan.rs:155-200`. Fix the script if any differ. The first build compiles the I/O stack from source (trx-rs, itk-transforms-rs, hdf5-metno-src) and needs network and cmake; expect it to take a while.

- [ ] **Step 6: Generate the baseline**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
chmod +x tools/run_p0_baseline.sh
tools/run_p0_baseline.sh tests/fixtures/p0_baseline/checksums.txt
cat tests/fixtures/p0_baseline/checksums.txt
```

Expected: 10 groups of output files (5 configs x 2 feature sets), each with at least a `part-mag` and `part-phase` NIfTI. Non-empty checksums for every line. The `legacy` runs print a `WARNING: --oversample 1 uses the legacy path` line on stderr; that is expected. The `mb` runs also write `_desc-dropout_slices.tsv`, which is checksummed with the rest.

- [ ] **Step 7: Prove the baseline is reproducible**

A baseline that does not reproduce against itself cannot detect anything. Run it twice and diff.

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
tools/run_p0_baseline.sh /tmp/p0_recheck.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_recheck.txt && echo "REPRODUCIBLE"
```

Expected: `REPRODUCIBLE`. If it fails, the simulator has an unseeded source of nondeterminism and that must be found before the extraction starts — it would otherwise produce false failures in every later task.

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
git switch -c p0-mrsim-acq-extraction
git add tools/gen_p0_fixture.py tools/run_p0_baseline.sh tests/fixtures/p0_baseline/
git commit -m "test: add P0 bit-identity baseline fixture and checksums

Captured at 57858a5 before any files move. Five configurations (plain,
eddy, multi-coil GRAPPA, multiband dropout, legacy o=1) under both
--features cli and --features cli,kspace,par, on a two-grid fixture so
the default oversampled entry point is the one under test. The scheme's b0 row is written the FSL way,
bval=5 with a zero bvec, because that is the row interface change 2 puts
at risk.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Create the crate and move the pure-std core

Moves `mat`, `orient`, `analytic`, `Vec3`, and the `Grid` struct. Nothing here is feature-gated, so this task ends with both crates building and testing on the default pure-std build.

`Grid` needs care. It is defined at `raster.rs:18` with two **private** helpers, `linear_and_origin` (`raster.rs:34`) and `world_to_voxel` (`raster.rs:43`), used only by `intersect_segment` (`raster.rs:51`). `raster` stays in TRXScan. Rust does not allow inherent impls on a foreign type, so `intersect_segment` cannot stay a method on a `Grid` that lives in another crate. The split is: `Grid` moves as a **plain data struct** (`dims`, `voxel_to_world`), and all three geometry functions move into an extension trait in `raster`. Call sites keep method syntax and only gain a `use`.

**Files:**
- Create: `mrsim-acq/Cargo.toml`, `mrsim-acq/src/lib.rs`, `mrsim-acq/src/grid.rs`
- Move: `TRXScan/src/mat.rs` → `mrsim-acq/src/mat.rs`
- Move: `TRXScan/src/orient.rs` → `mrsim-acq/src/orient.rs`
- Move: `TRXScan/src/analytic.rs` → `mrsim-acq/src/analytic.rs`
- Modify: `TRXScan/Cargo.toml`, `TRXScan/src/lib.rs`, `TRXScan/src/raster.rs`

**Interfaces:**
- Consumes: Task 1's baseline (not exercised until Task 6)
- Produces: `mrsim_acq::Vec3` (`= [f64; 3]`), `mrsim_acq::grid::Grid { dims: [usize; 3], voxel_to_world: [[f64; 4]; 4] }`, `mrsim_acq::mat`, `mrsim_acq::orient`, `mrsim_acq::analytic`. TRXScan re-exports `Vec3` at its root and `Grid` from `raster`, so every existing `use crate::Vec3` and `use crate::raster::Grid` in TRXScan keeps compiling. TRXScan gains `raster::GridRaster`, a trait providing `intersect_segment(&self, a: Vec3, b: Vec3) -> Vec<SegmentHit>`.

- [ ] **Step 1: Write the crate manifest**

Create `mrsim-acq/Cargo.toml`. Note `io` does **not** list `trx-rs`.

```toml
[package]
name = "mrsim-acq"
version = "0.0.0"
edition = "2021"
description = "Shared MRI acquisition stage: k-space forward model, EPI readout, motion, complex NIfTI output"
license = "MIT OR Apache-2.0"
publish = false

# Every external dependency is optional and off by default, so the default build is
# pure std and `cargo test` runs offline with no system libraries.
[features]
default = []
# NIfTI volume read + scalar/complex 4D write. Deliberately no trx-rs: streamline
# I/O stays in TRXScan so this crate never pulls HDF5.
io = ["dep:nifti", "dep:ndarray", "dep:nalgebra"]
# k-space FFT backend (rustfft); the default k-space path is std-only.
kspace = ["dep:rustfft", "dep:ndarray", "dep:nalgebra", "dep:rand", "dep:rand_distr"]
# config parsing (TOML); currently a stub.
config = ["dep:serde", "dep:toml"]
# parallelism
par = ["dep:rayon"]

[dependencies]
nifti = { version = "0.17", features = ["ndarray_volumes"], optional = true }
ndarray = { version = "0.16", optional = true }
nalgebra = { version = "0.33", optional = true }
rustfft = { version = "6", optional = true }
rand = { version = "0.8", optional = true }
rand_distr = { version = "0.4", optional = true }
rayon = { version = "1", optional = true }
serde = { version = "1", features = ["derive"], optional = true }
toml = { version = "0.8", optional = true }
```

Check the `toml` version against `TRXScan/Cargo.toml` and match it exactly; mismatched versions across the two crates are a needless source of build surprise.

- [ ] **Step 2: Write the crate root**

Create `mrsim-acq/src/lib.rs`. Only the modules this task moves are declared; later tasks add the rest.

```rust
//! # mrsim-acq
//!
//! The MRI acquisition stage shared by TRXScan (diffusion) and aslscan (ASL): a per-slice
//! k-space forward model with EPI distortion, relaxation, eddy currents, ghosting, partial
//! Fourier, ringing, spikes, multi-coil combination, GRAPPA and noise, plus the motion model
//! that cuts across it and the complex NIfTI writer at the end of it.
//!
//! Extracted from TRXScan so the same acquisition physics serves both signal models. The
//! signal stage — what fills the compartment images — belongs to the consumer.
//!
//! ## Build shape
//! Every external dependency is optional; the default build is pure std, so `cargo test`
//! exercises the core offline. Features: `io`, `kspace`, `config`, `par`.

/// A direction or point in 3D. The pure-std core uses a plain `[f64; 3]`; the feature-gated
/// I/O and k-space paths convert to `nalgebra::Vector3<f64>` where they need heavier algebra.
pub type Vec3 = [f64; 3];

/// std-only 3×3 / vector helpers, keeping the pure-math core dependency-free and testable.
pub mod mat;
/// std-only voxel-axis reorientation to the FSL/dcm2niix (radiological LAS) convention.
pub mod orient;
/// The acquisition voxel grid: dimensions plus a voxel→world affine.
pub mod grid;
/// Analytic Fourier references used as test oracles.
pub mod analytic;
```

- [ ] **Step 3: Move the three modules verbatim**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx
git -C TRXScan mv src/mat.rs ../mrsim-acq/src/mat.rs 2>/dev/null || {
  cp TRXScan/src/mat.rs mrsim-acq/src/mat.rs && git -C TRXScan rm -q src/mat.rs
}
cp TRXScan/src/orient.rs mrsim-acq/src/orient.rs && git -C TRXScan rm -q src/orient.rs
cp TRXScan/src/analytic.rs mrsim-acq/src/analytic.rs && git -C TRXScan rm -q src/analytic.rs
```

`git mv` across repository boundaries does not work; the `cp` + `git rm` fallback is the expected path. Do not edit the file contents in this step beyond what Step 4 requires.

- [ ] **Step 4: Fix the moved files' crate-root references**

In `mrsim-acq/src/mat.rs`, `orient.rs`, and `analytic.rs`, every `use crate::Vec3` already resolves, because `Vec3` is at the new crate's root too. Fix anything that referenced a module that did *not* move:

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
grep -n "use crate::" src/mat.rs src/orient.rs src/analytic.rs
```

Expected: only `crate::Vec3`, `crate::mat`, and `crate::grid`. Anything else names a module still in TRXScan and must be resolved before continuing — report it rather than inventing a fix.

- [ ] **Step 5: Write the grid module**

Create `mrsim-acq/src/grid.rs`, taking the struct from `TRXScan/src/raster.rs:16-23` exactly:

```rust
//! The acquisition voxel grid.
//!
//! A plain data struct on purpose. Rasterisation geometry lives with the rasteriser in the
//! consumer crate, because a foreign type cannot take inherent impls; see `raster::GridRaster`
//! in TRXScan.

/// The acquisition voxel grid: dimensions + voxel→world (RAS mm) affine (may be oblique).
#[derive(Debug, Clone)]
pub struct Grid {
    pub dims: [usize; 3],
    /// voxel→world 4×4, row-major.
    pub voxel_to_world: [[f64; 4]; 4],
}
```

- [ ] **Step 6: Convert raster's Grid methods into an extension trait**

In `TRXScan/src/raster.rs`, delete the `Grid` struct definition and replace the `impl Grid { ... }` block (`raster.rs:32` through the end of `intersect_segment`) with a trait. Keep the bodies of all three functions **character for character** — this is a move, not a rewrite, and the DDA's floating-point behavior is what Task 6 checks.

```rust
pub use mrsim_acq::grid::Grid;

/// Rasterisation geometry for [`Grid`]. `Grid` lives in `mrsim-acq`, so these cannot be
/// inherent methods; the trait keeps call sites in method syntax.
pub trait GridRaster {
    fn linear_and_origin(&self) -> (mat::Mat3, Vec3);
    fn world_to_voxel(&self, p: Vec3) -> Option<Vec3>;
    fn intersect_segment(&self, a: Vec3, b: Vec3) -> Vec<SegmentHit>;
}

impl GridRaster for Grid {
    // Paste the bodies of raster.rs:34-42 (linear_and_origin), :43-50 (world_to_voxel) and
    // :51-onward (intersect_segment) here unchanged. Do not retype them: the DDA's
    // floating-point expression order is what Task 6's gate checks.
}
```

- [ ] **Step 7: Wire TRXScan to the new crate**

In `TRXScan/Cargo.toml`, under `[dependencies]`:

```toml
mrsim-acq = { path = "../mrsim-acq" }
```

In `TRXScan/src/lib.rs`, delete the `pub mod mat;`, `pub mod orient;`, and `pub mod analytic;` lines and the `pub type Vec3` definition, and replace them with re-exports so existing `use crate::Vec3` paths keep working:

```rust
pub use mrsim_acq::{mat, orient, analytic, Vec3};
```

- [ ] **Step 8: Add the trait import where rasterisation is called**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
grep -rn "intersect_segment" src/ --include=*.rs
```

Add `use crate::raster::GridRaster;` to each file that appears, other than `raster.rs` itself.

- [ ] **Step 9: Build and test both crates**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test
```

Expected: `mrsim-acq` passes with the tests that rode along inside `mat.rs`, `orient.rs`, and `analytic.rs`. TRXScan passes its remaining ~110 unit tests with no failures and no change in count beyond the ones that moved.

- [ ] **Step 10: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat: create crate, move mat/orient/analytic/Vec3/Grid

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
git add -A && git commit -m "refactor: depend on mrsim-acq for mat/orient/analytic/Vec3/Grid

Grid becomes a plain data struct in mrsim-acq. Its three geometry
functions move into raster::GridRaster, because a foreign type cannot
take inherent impls. The bodies are unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Move phase, readout, noise, motion, config

Five modules, all pure std except `config`. No interface changes yet — `DiffusionPhase` keeps its name until Task 9, and `time_from_last_diffusion_gradient` until Task 8.

**Files:**
- Move: `TRXScan/src/{phase,readout,noise,motion,config}.rs` → `mrsim-acq/src/`
- Modify: `mrsim-acq/src/lib.rs`, `TRXScan/src/lib.rs`

**Interfaces:**
- Consumes: `mrsim_acq::{Vec3, mat}` from Task 2
- Produces: `mrsim_acq::phase::{BackgroundPhase, ShotPhase, DiffusionPhase, PhaseModel}`, `mrsim_acq::readout::{Readout, SingleShotEpi}`, `mrsim_acq::motion::{MotionMode, Pose, MotionEvent, DroppedShot, resolve_poses, load_motion_tsv, fov_center, resample_by_pose, apply_motion, slice_schedule, apply_multiband_motion}`, `mrsim_acq::noise`, `mrsim_acq::config`. All signatures unchanged from TRXScan.

- [ ] **Step 1: Move the files**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx
for m in phase readout noise motion config; do
  cp "TRXScan/src/$m.rs" "mrsim-acq/src/$m.rs" && git -C TRXScan rm -q "src/$m.rs"
done
```

- [ ] **Step 2: Declare them in the new crate root**

Add to `mrsim-acq/src/lib.rs`, after the `analytic` declaration:

```rust
/// Object phase model.
pub mod phase;
/// EPI readout trajectory and line timing.
pub mod readout;
/// Noise models. `add_complex_gaussian` is a `todo!()` stub; the noise that runs lives in
/// `kspace` and is reached through `Acquisition::noise_variance`.
pub mod noise;
/// Rigid poses, multiband slice schedules, within-volume dropout.
pub mod motion;
/// Simulation config (feature `config`). `load` is a `todo!()` stub.
#[cfg(feature = "config")]
pub mod config;
```

- [ ] **Step 3: Remove them from TRXScan and re-export**

In `TRXScan/src/lib.rs`, delete the five `pub mod` lines and add to the existing re-export:

```rust
pub use mrsim_acq::{mat, orient, analytic, phase, readout, noise, motion, Vec3};
#[cfg(feature = "config")]
pub use mrsim_acq::config;
```

- [ ] **Step 4: Resolve dangling references**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo build 2>&1 | head -40
```

Expected failures name `crate::kspace` (from `phase.rs`'s `Rng` comment reference or `motion`'s use of grid helpers) or `crate::raster::Grid`. For `Grid`, change to `crate::grid::Grid`.

One failure surfaces on the **TRXScan** side instead: `gnl.rs:433` calls `crate::motion::trilinear`, which is `pub(crate)` (`motion.rs:179`). Through the re-export that is now a foreign crate's private item. Make it `pub` in `mrsim-acq/src/motion.rs` with a one-line doc comment; `gnl` stays in TRXScan and its call site does not change. Also expect `motion.rs:250`'s doc link to `crate::compartments::generate_compartments_moving` to become a broken intra-doc link (rustdoc warning, not a build error); reword it to plain text naming TRXScan. For anything naming `kspace`, note it and leave it broken **only** if Task 4 will supply it; if `phase` or `motion` genuinely needs a symbol from a module that is not moving, stop and report rather than duplicating code.

- [ ] **Step 5: Build and test both crates**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test
```

Expected: both pass. TRXScan's test count drops by the number that moved; `mrsim-acq`'s rises by the same number.

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat: move phase, readout, noise, motion, config

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
git add -A && git commit -m "refactor: source phase/readout/noise/motion/config from mrsim-acq

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Move kspace and nufft

The largest move: `kspace.rs` is 2619 lines and carries the bulk of the crate's tests. Still a pure move — no signature changes.

**Files:**
- Move: `TRXScan/src/kspace.rs` → `mrsim-acq/src/kspace.rs`
- Move: `TRXScan/src/nufft.rs` → `mrsim-acq/src/nufft.rs`
- Modify: `mrsim-acq/src/lib.rs`, `TRXScan/src/lib.rs`

**Interfaces:**
- Consumes: `mrsim_acq::{phase, readout, analytic, Vec3}` from Tasks 2-3
- Produces: `mrsim_acq::kspace::{C, Rng, PartialFourierMode, KspaceWindow, Acquisition, SliceInput, sampling_mask, box_hires, step_hires, simulate_slice, simulate_slice_kspace, phase_slice, simulate_acquisition_oversampled, simulate_acquisition_legacy}`, `mrsim_acq::nufft::Nufft1`. All signatures unchanged; the entry point still takes `ngrad`, `bvals`, `bvecs`.

- [ ] **Step 1: Move the files**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx
for m in kspace nufft; do
  cp "TRXScan/src/$m.rs" "mrsim-acq/src/$m.rs" && git -C TRXScan rm -q "src/$m.rs"
done
```

- [ ] **Step 2: Declare and re-export**

Add to `mrsim-acq/src/lib.rs`:

```rust
/// Per-slice k-space forward model and reconstruction.
pub mod kspace;
/// Type-1 NUFFT (feature `kspace`): the fieldmap y-sum as one gridded FFT.
pub mod nufft;
```

Delete the corresponding lines from `TRXScan/src/lib.rs` and add `kspace, nufft` to its `pub use mrsim_acq::{...}` list.

- [ ] **Step 3: Build and fix import paths**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo build --features kspace 2>&1 | head -40
```

`kspace.rs:1936` uses `crate::analytic::truncated_step_profile`, which resolves because `analytic` moved in Task 2. Fix any `crate::raster::Grid` to `crate::grid::Grid`.

- [ ] **Step 4: Test under both build shapes**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test
cargo test --features kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
cargo test
```

Expected: all pass. The `kspace` feature build exercises the `rustfft` and NUFFT paths, which the default build does not, and both must be green before Task 6 compares them.

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat: move kspace and nufft

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
git add -A && git commit -m "refactor: source kspace and nufft from mrsim-acq

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Split io and forward the features

`io.rs` splits by responsibility. The volume readers, grid helper, NIfTI header builder and array writers move; the streamline loaders, `write_benchmark`, `write_dwi`, `write_scalar_maps` and `subsample_streamlines` stay, because they depend on `trx-rs` or on TRXScan's own types. `write_complex_dwi` and `SidecarInfo` **also stay, until Task 13**: the function takes `scheme: &GradientScheme` and calls the private `write_bval_bvec` (`io.rs:444`, `:450`), both TRXScan-only, so moving it verbatim would make `mrsim-acq` depend on TRXScan, a cycle. Task 13 creates `write_complex_4d` in `mrsim-acq` and deletes `write_complex_dwi` from TRXScan.

**Files:**
- Create: `mrsim-acq/src/io.rs`
- Modify: `TRXScan/src/io.rs`, `TRXScan/Cargo.toml`, `mrsim-acq/src/lib.rs`, `TRXScan/src/lib.rs`

**Interfaces:**
- Consumes: `mrsim_acq::grid::Grid`
- Produces: `mrsim_acq::io::{load_volume, hires_grid, write_3d, write_4d, write_3d_i16}`, all behind the `io` feature, with signatures unchanged. `header_for_grid`, `affine_from_header`, and `quatern_to_mat44` move too and stay private to the module.

- [ ] **Step 1: Create the moved half**

Create `mrsim-acq/src/io.rs` containing, copied verbatim from `TRXScan/src/io.rs`: the `nifti`/`ndarray`/`nalgebra` imports, `affine_from_header` (`:164`) and `quatern_to_mat44` (`:184`) — private, but `load_volume` calls them at `:212` and does not compile without them — `load_volume` (`:208`), `hires_grid` (`:264`), `header_for_grid` (`:242`), `write_4d` (`:353`), `write_3d` (`:364`), `write_3d_i16` (`:374`), and the `#[cfg(test)] mod tests` cases that exercise only those. Not `SidecarInfo` or `write_complex_dwi`, for the reason above.

The manifest needs one thing TRXScan's does not say: `nifti = { version = "0.17", features = ["ndarray_volumes", "nalgebra_affine"] }`. `header_for_grid` calls `NiftiHeader::set_qform`/`set_sform(&Matrix4, ..)`, which nifti gates behind `nalgebra_affine`. TRXScan never enables it explicitly; `trx-rs` does, and feature unification hides that. Standalone, `mrsim-acq --features io` fails to compile without it, while TRXScan's build of the same file succeeds.

Head it with:

```rust
//! NIfTI volume read and array write. Streamline I/O stays in the consumer crate, which is why
//! this feature pulls `nifti`/`ndarray`/`nalgebra` and never `trx-rs`: `aslscan` must not pay
//! for an HDF5 build it has no use for.
```

- [ ] **Step 2: Trim TRXScan's io**

Delete the moved items from `TRXScan/src/io.rs` and add at the top:

```rust
pub use mrsim_acq::io::{load_volume, hires_grid, write_3d, write_3d_i16, write_4d};
```

`write_complex_dwi` and `SidecarInfo` remain defined in this file and call the re-exported `write_4d`.

`load_tissue` (`io.rs:227`) stays and calls the re-exported `load_volume`.

- [ ] **Step 3: Declare it and forward the features**

Add to `mrsim-acq/src/lib.rs`:

```rust
/// NIfTI volume read + scalar/complex 4D write (feature `io`).
#[cfg(feature = "io")]
pub mod io;
```

In `TRXScan/Cargo.toml`, make the three shared features forward. Forgetting this compiles fine and silently drops to the std-only path, which is why criterion 3 runs under both feature sets:

```toml
io = ["dep:trx-rs", "dep:nifti", "dep:ndarray", "dep:nalgebra", "mrsim-acq/io"]
kspace = ["dep:rustfft", "dep:ndarray", "dep:nalgebra", "dep:rand", "dep:rand_distr", "mrsim-acq/kspace"]
config = ["dep:serde", "dep:toml", "mrsim-acq/config"]
par = ["dep:rayon", "mrsim-acq/par"]
```

`config` forwards too, and it is the one that fails loudly rather than silently: Task 3 re-exports `mrsim_acq::config` under TRXScan's `config` feature, and `mrsim-acq` gates the module on its own `config` feature, so without the forward `cargo build --features config` in TRXScan has nothing to re-export.

- [ ] **Step 4: Verify the forwarding actually took**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
cargo tree -e features --features cli,kspace,par -i mrsim-acq 2>/dev/null
```

(The inverted form. `-p mrsim-acq --features ...` is refused because features can only be named for workspace members, and the forward tree shows only `mrsim-acq feature "default"` because the forwarded features hang off `trxscan feature "io"` etc., not off the package node.)

Expected: `mrsim-acq feature "io"`, `"kspace"`, and `"par"` nodes, each with a `trxscan feature` child. If they are absent the forwarding did not take, and the NUFFT path is silently off. Also `cargo build --features config` in TRXScan, which must compile.

- [ ] **Step 5: Confirm mrsim-acq never pulls HDF5**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo tree --features io 2>/dev/null | grep -iE "hdf5|trx-rs|itk" && echo "STOP: HDF5 leaked in" || echo "OK: no HDF5"
```

Expected: `OK: no HDF5`. This is the property that keeps `aslscan`'s build cheap, and it is easier to keep than to recover.

- [ ] **Step 6: Build and test**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test && cargo test --features io,kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test && cargo test --features cli
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat: move the volume-io half of io

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
git add -A && git commit -m "refactor: source volume io from mrsim-acq; forward features

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Bit-identity checkpoint — the move must be inert

No code changes. This task exists because everything after it is easier to debug once the move alone is known to be clean. If the checksums differ here, the cause is in Tasks 2-5 and nowhere else.

**Files:**
- Create: `TRXScan/tests/fixtures/p0_baseline/checksums_after_move.txt` (temporary, not committed)

**Interfaces:**
- Consumes: `TRXScan/tests/fixtures/p0_baseline/checksums.txt` from Task 1
- Produces: proof that Tasks 2-5 moved no bits. No new symbols.

- [ ] **Step 1: Re-run the baseline script against the moved code**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
tools/run_p0_baseline.sh /tmp/p0_after_move.txt
```

- [ ] **Step 2: Diff against the baseline**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_after_move.txt \
  && echo "PASS: move is bit-identical" \
  || echo "FAIL: the move changed output"
```

Expected: `PASS: move is bit-identical`.

If it fails, the likely causes in order: (a) the `GridRaster` trait bodies were retyped rather than pasted, changing floating-point expression order; (b) a feature failed to forward, so one run took the std-only path where the baseline took the NUFFT path — check `cargo tree` as in Task 5 Step 4; (c) a moved module picked up a different `mat` helper. Bisect by reverting to each task's commit and re-running.

- [ ] **Step 3: Run the full test suites under every build shape**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test && cargo test --features io,kspace,par,config
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test && cargo test --features cli && cargo test --features cli,kspace,par
```

Expected: all green. This is acceptance criteria 1 and 2 for the move.

- [ ] **Step 4: Commit the checkpoint marker**

Nothing to commit in source. Tag the point instead, so later bisects have an anchor:

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
git tag p0-move-complete
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git tag p0-move-complete
```

---

### Task 7: Change 3 — rename the readout timing method

Smallest of the eight. `time_from_last_diffusion_gradient` is a method of the `Readout` **trait** (`readout.rs:14`), not only of `SingleShotEpi`, so the rename lands on the trait and its one implementation. The computation does not change.

**Files:**
- Modify: `mrsim-acq/src/readout.rs:14` (trait), `mrsim-acq/src/readout.rs:61` (impl)
- Modify: `mrsim-acq/src/kspace.rs:246` (only caller)

**Interfaces:**
- Consumes: `mrsim_acq::readout::Readout`
- Produces: `Readout::time_from_prep_gradient(&self, tick: usize) -> f64`. `time_from_last_diffusion_gradient` no longer exists.

- [ ] **Step 1: Find every reference**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx
grep -rn "time_from_last_diffusion_gradient" mrsim-acq/src TRXScan/src
```

Expected: the trait declaration, the `SingleShotEpi` impl, and the call at `kspace.rs:246`.

- [ ] **Step 2: Rename, and update the doc comment to match**

```rust
    /// Time from the last large preparation gradient to the sample at `tick`. That gradient is
    /// a diffusion gradient in one consumer and a crusher or labeling gradient in the other.
    fn time_from_prep_gradient(&self, tick: usize) -> f64 {
        tick as f64 * self.dt() + self.dt() / 2.0
    }
```

- [ ] **Step 3: Test and verify bit-identity**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test && cargo test --features io,kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test --features cli
tools/run_p0_baseline.sh /tmp/p0_t7.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_t7.txt && echo "PASS: bit-identical"
```

Expected: tests pass, `PASS: bit-identical`. A rename that moves bits means something other than the name changed.

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "refactor!: Readout::time_from_last_diffusion_gradient -> time_from_prep_gradient

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Change 2 — eddy drive replaces bvec/bval

`SliceInput.bvec` and `bval` are read only by the eddy model, at `kspace.rs:391-393`, `:484`, and `:663`, and in all three places as the product `bvec * bval`.

The subtle part is the disabled predicate. Today `do_eddy` tests `bval.abs() > 1e-9`. FSL scheme files routinely write b0 rows as `bval = 5, bvec = [0,0,0]`, which today takes `do_eddy = true` with a zero gradient: identity rotors, but the NUFFT path is still disabled at `kspace.rs:540`. If the library re-derived "disabled" from a zero drive vector, that row would flip to `None`, re-enable the NUFFT, and move bits under the `kspace` feature. So `Some([0.0; 3])` and `None` must remain distinct inputs — the decision is the caller's.

**Files:**
- Modify: `mrsim-acq/src/kspace.rs` — `SliceInput` (`:269-272`), `simulate_slice` (`:391-394`, `:484`, `:663`), `simulate_acquisition_oversampled` (`:1237`), `simulate_acquisition_legacy` (`:1332-1350`)
- Modify: `TRXScan/src/bin/trxscan.rs` — call site
- Modify: `TRXScan/src/benchmark.rs` — `produce_slice` takes `bvec: [f64; 3], bval: f64` (`:61-62`) and forwards them into a `SliceInput` literal (`:85-95`); it becomes `eddy_drive: Option<[f64; 3]>`. `src/bin/trxscan_benchmark.rs` calls it. This file is easy to miss because it is a library module, not a binary, and the file-structure table above lists only binaries.

**Interfaces:**
- Consumes: `mrsim_acq::kspace::SliceInput`
- Produces: `SliceInput { eddy_drive: Option<[f64; 3]>, .. }` replacing `bvec: [f64;3]` and `bval: f64`. `simulate_acquisition_oversampled` gains `eddy_drive: &[Option<[f64; 3]>]` of length `n_volumes` **alongside** `bvals`/`bvecs`, which Task 9 removes.

- [ ] **Step 1: Write the failing tests for criteria 4 and 5**

Add to `mrsim-acq/src/kspace.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn eddy_drive_some_reproduces_the_bvec_bval_product() {
    // Criterion 4: the drive is exactly what the old code formed internally.
    let (nx, ny) = (16, 16);
    let comps = vec![box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
    let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
    let fmap = vec![0.0f32; nx * ny];
    let acq = Acquisition { eddy_strength: 0.05, ..Default::default() };
    let (bval, bvec) = (1000.0f64, [0.6f64, 0.8, 0.0]);

    let inp = SliceInput {
        compartments: &comp_refs, t2: &[100.0], fmap: &fmap, phase0: None,
        sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
        eddy_drive: Some([bvec[0] * bval, bvec[1] * bval, bvec[2] * bval]),
        slice_seed: 7, eddy_lin: None,
    };
    let got = simulate_slice_kspace(&inp, &acq);
    // EXPECTED_BITS is this exact slice's first eight k-space coefficients as produced by the
    // pre-change bvec/bval path at tag `p0-move-complete` (captured in Step 2, before the
    // struct changes). Bit equality, not a tolerance: the drive is the same product the old
    // code formed internally, so the arithmetic must be the same. A nonzero check would pass
    // with eddy ignored entirely.
    const EXPECTED_BITS: [(u64, u64); 8] = [/* paste from Step 2 */];
    for (i, (re, im)) in got.iter().take(8).enumerate() {
        assert_eq!((re.to_bits(), im.to_bits()), EXPECTED_BITS[i], "coefficient {i} moved");
    }
}

#[test]
fn eddy_drive_none_disables_but_zero_vector_does_not() {
    // Criterion 5: None and Some([0,0,0]) are DIFFERENT inputs. The FSL b0 row
    // (bval=5, bvec=[0,0,0]) is Some([0,0,0]) and must keep eddy "on" with a zero gradient,
    // because that is what disables the NUFFT path today.
    let (nx, ny) = (16, 16);
    let comps = vec![box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
    let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
    let fmap = vec![3.0f32; nx * ny];
    let acq = Acquisition { eddy_strength: 0.05, ..Default::default() };

    let make = |drive| SliceInput {
        compartments: &comp_refs, t2: &[100.0], fmap: &fmap, phase0: None,
        sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
        eddy_drive: drive, slice_seed: 7, eddy_lin: None,
    };
    let none = simulate_slice(&make(None), &acq);
    let zero = simulate_slice(&make(Some([0.0; 3])), &acq);

    // Identical numerically: a zero gradient multiplies by identity rotors.
    for ((a, b), (c, d)) in none.iter().zip(zero.iter()) {
        assert!((a - c).abs() < 1e-6 && (b - d).abs() < 1e-6,
                "zero-vector drive should match None numerically on this slice");
    }
    // But they must not be collapsed into the same input. The eligibility decision lives in
    // `eddy_enabled` (Step 4) so it can be pinned here: `is_some()`, never "is the vector zero".
    assert!(eddy_enabled(&acq, Some([0.0; 3])), "a zero-vector drive must keep eddy enabled");
    assert!(!eddy_enabled(&acq, None), "None must disable eddy");
    assert!(!eddy_enabled(&Acquisition::default(), Some([1.0; 3])), "eddy_strength 0 disables");
}
```

- [ ] **Step 2: Capture the reference bits, then run the tests to confirm they fail**

Before touching the struct, add a temporary test that builds the same slice with the *old* fields, `bvec: [0.6, 0.8, 0.0], bval: 1000.0`, and prints `(re.to_bits(), im.to_bits())` for the first eight entries of `simulate_slice_kspace(&inp, &acq)`. Run it **twice**, `cargo test print_eddy_reference -- --nocapture` and the same with `--features kspace`, and keep both sets as `EXPECTED_BITS_STD` and `EXPECTED_BITS_FFT`, selected in the test by `cfg!(feature = "kspace")`. The two differ in the last bits: the `kspace` feature swaps the x-stage to rustfft, and the std twiddle-table sum and rustfft do not agree bit for bit, before or after this change. One reference would pass under one build and fail under the other. Delete the temporary test afterwards. The tree is still at `p0-move-complete` plus Task 7's rename, so these are the pre-change numbers.

Two more `SliceInput` sites exist than the module-level grep suggests, and the compiler will name them: the oversampled-path test in `kspace.rs` that calls `simulate_acquisition_oversampled` directly, and TRXScan's integration test `tests/kspace_alignment.rs`.

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test eddy_drive 2>&1 | tail -20
```

Expected: compile failure — `SliceInput` has no field `eddy_drive`, and `eddy_enabled` is undefined.

- [ ] **Step 3: Change the struct**

In `mrsim-acq/src/kspace.rs`, replace the `bvec`/`bval` fields:

```rust
    /// Per-volume eddy drive in the legacy model's units, `bvec * bval` for diffusion.
    /// `None` disables eddy for this volume, reproducing the old `bval ~= 0` branch.
    /// `Some([0.0; 3])` does NOT: an FSL b0 row (`bval = 5`, zero `bvec`) is `Some`, keeps
    /// `do_eddy` true with identity rotors, and keeps the NUFFT path disabled.
    pub eddy_drive: Option<[f64; 3]>,
```

- [ ] **Step 4: Change the three read sites**

At `kspace.rs:391-394`:

```rust
    let gradient = inp.eddy_drive.unwrap_or([0.0; 3]);
    let do_eddy = eddy_enabled(acq, inp.eddy_drive);
    let do_eddy_phase = acq.eddy_phase != 0.0 && inp.eddy_drive.is_some();
```

with the decision factored out above `build_coil_kspace` so the test can pin it:

```rust
/// The eddy eligibility decision, in one place. `None` disables; `Some` enables even for a
/// zero vector (the FSL b0 row, which must keep the NUFFT path disabled as it is today); a zero
/// `eddy_strength` disables regardless.
pub(crate) fn eddy_enabled(acq: &Acquisition, drive: Option<[f64; 3]>) -> bool {
    acq.eddy_strength != 0.0 && drive.is_some()
}
```

The test oracle `reference_coil_kspace` (`kspace.rs:1420-1423`) has the same three lines and gets the same replacement. `kspace.rs:484` and `:663` already read the local `gradient`, so they need no edit. Confirm with `grep -n "inp.bval\|inp.bvec" mrsim-acq/src/kspace.rs` returning nothing.

- [ ] **Step 5: Change the entry point and the legacy wrapper**

In `simulate_acquisition_oversampled`, **add** `eddy_drive: &[Option<[f64; 3]>]` next to the existing `bvals: &[f64], bvecs: &[[f64; 3]]` parameters, validate its length before the volume loop, and pass `eddy_drive[g]` into `SliceInput`. `bvals` and `bvecs` cannot go yet: `kspace.rs:1238` still feeds them to `phase.diffusion.shot(bvals[g], bvecs[g], ..)`, and rebuilding them from the product would change that call's operation order. Task 9 Step 5 deletes them when `prep_drive` replaces that call. Through Task 8, TRXScan's call site passes all three.

```rust
    assert_eq!(eddy_drive.len(), n_volumes,
               "eddy_drive has {} entries for {} volumes", eddy_drive.len(), n_volumes);
```

`simulate_acquisition_legacy` has a trap. It receives a scaled gradient, splits it into `bval = |g|` and `bvec = g/|g|`, and lets the forward model recombine them (`kspace.rs:1332-1335`). `(g/|g|) * |g|` is **not** bit-equal to `g`. Keep the split and pass the recombined product, never the original vector:

```rust
    let bval = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let drive = if bval.abs() > 1e-9 {
        let bvec = [g[0] / bval, g[1] / bval, g[2] / bval];
        Some([bvec[0] * bval, bvec[1] * bval, bvec[2] * bval])
    } else {
        None
    };
```

- [ ] **Step 6: Update TRXScan's call site**

In `TRXScan/src/bin/trxscan.rs`, build the drive slice from the scheme:

```rust
    let eddy_drive: Vec<Option<[f64; 3]>> = scheme.bvals.iter().zip(scheme.bvecs.iter())
        .map(|(&b, v)| if b.abs() > 1e-9 { Some([v[0] * b, v[1] * b, v[2] * b]) } else { None })
        .collect();
```

- [ ] **Step 7: Run the tests, then the bit-identity gate**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test && cargo test --features io,kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test --features cli
tools/run_p0_baseline.sh /tmp/p0_t8.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_t8.txt && echo "PASS: bit-identical"
```

Expected: `PASS: bit-identical`. The fixture's FSL b0 row is what makes this gate meaningful — if the predicate was collapsed to a zero-vector test, the `cli,kspace,par` runs will differ while the `cli` runs match, because only the former has a NUFFT path to re-enable.

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat!: replace SliceInput bvec/bval with eddy_drive

Some([0,0,0]) and None stay distinct inputs. FSL b0 rows are bval=5 with
a zero bvec; collapsing them to None would re-enable the NUFFT path and
move bits.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Change 4 — PrepPhase and its own drive

`DiffusionPhase` becomes `PrepPhase`, and `PhaseModel.diffusion` becomes `prep: Option<PrepPhase>`. It needs a drive of its own: `DiffusionPhase::shot` forms `q_eff = c_q * sqrt(bval) * bvec_unit` (`phase.rs:110-117`), nonlinear in `bval`, while `eddy_drive` is the linear product. They are interconvertible only when `bvec` is unit-norm, and `phase.rs:110` divides by `|bvec|` precisely because it does not assume that.

The caller must **not** pre-normalize. The division by `n` has to stay inside the same expression it is in now, or the floating-point operation order changes and bit-identity fails even though the math is equivalent.

**Files:**
- Modify: `mrsim-acq/src/phase.rs:101-132`
- Modify: `mrsim-acq/src/kspace.rs` — `SliceInput`, `:1238`
- Modify: `TRXScan/src/bin/trxscan.rs`
- Modify: `TRXScan/src/benchmark.rs` — `produce_slice` gains `prep_drive`, and its two tests build `DiffusionPhase { .. }.shot(0.0, [0.0; 3], ..)` (`:234-242`, `:265-272`), which follow the rename and the new parameter names; their `PhaseModel` literals use `..PhaseModel::none()` (`:240`) and need nothing. `bin/trxscan_benchmark.rs` constructs `PhaseModel` and may name the old field. The compiler lists every site.

**Interfaces:**
- Consumes: `mrsim_acq::phase::{PhaseModel, ShotPhase}`
- Produces: `mrsim_acq::phase::PrepPhase` (fields unchanged from `DiffusionPhase`: `c_q`, `sigma_dx`, `sigma_rot`), `PrepPhase::shot(&self, magnitude: f64, direction: [f64; 3], volume: usize, slice_group: usize, seed: u64) -> ShotPhase`, `PhaseModel { prep: Option<PrepPhase>, .. }`, `SliceInput { prep_drive: Option<(f64, [f64; 3])>, .. }`, and `simulate_acquisition_oversampled`'s `prep_drive: &[Option<(f64, [f64; 3])>]`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn prep_shot_keeps_its_guards_and_does_not_require_unit_input() {
    let p = PrepPhase { c_q: 1.0, sigma_dx: 0.0, sigma_rot: 0.0 };
    // Non-unit direction: shot normalizes internally.
    let a = p.shot(1000.0, [0.0, 0.0, 3.0], 0, 0, 1);
    let b = p.shot(1000.0, [0.0, 0.0, 1.0], 0, 0, 1);
    for i in 0..3 {
        assert!((a.q_eff[i] - b.q_eff[i]).abs() < 1e-12,
                "a non-unit direction must give the same q_eff as its unit form");
    }
    // Degenerate direction: the n < 1e-12 guard yields zero phase, not NaN.
    let z = p.shot(1000.0, [0.0, 0.0, 0.0], 0, 0, 1);
    assert_eq!(z.q_eff, [0.0; 3], "zero direction must yield zero q_eff");
    // Non-positive magnitude: early return.
    let m = p.shot(0.0, [0.0, 0.0, 1.0], 0, 0, 1);
    assert_eq!(m.q_eff, [0.0; 3], "zero magnitude must yield zero q_eff");
}
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test prep_shot 2>&1 | tail -10
```

Expected: `cannot find type PrepPhase`.

- [ ] **Step 3: Rename the type and its method parameters**

In `mrsim-acq/src/phase.rs`, rename `DiffusionPhase` to `PrepPhase` and change `shot`'s first two parameters from `bval: f64, bvec: [f64; 3]` to `magnitude: f64, direction: [f64; 3]`. **Keep the body byte-for-byte**, including `let n = (...).sqrt();`, the `if bval <= 0.0 || n < 1e-12` early return (now reading `magnitude`), and `let s = self.c_q * magnitude.sqrt() / n;`.

- [ ] **Step 4: Make the phase model's component optional**

```rust
pub struct PhaseModel {
    pub global: f64,
    pub background: BackgroundPhase,
    /// Preparation-gradient phase. `None` for a sequence with no large prep gradient.
    pub prep: Option<PrepPhase>,
}
```

`ShotPhase` is not a field and never was (`phase.rs:132-137`); a realized shot is passed to `PhaseModel::at` per call (`phase.rs:141`).

- [ ] **Step 5: Thread the drive through k-space**

Add to `SliceInput`:

```rust
    /// Per-volume preparation-gradient drive: `(magnitude, direction)`. The direction is NOT
    /// required to be normalized — `PrepPhase::shot` normalizes, and moving that division to
    /// the caller changes floating-point operation order and breaks bit-identity.
    pub prep_drive: Option<(f64, [f64; 3])>,
```

Replace the call at `kspace.rs:1238`:

```rust
            let shot = match (&phase.prep, prep_drive[g]) {
                (Some(p), Some((mag, dir))) => p.shot(mag, dir, g, z, seed),
                _ => ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] },
            };
```

Add `prep_drive: &[Option<(f64, [f64; 3])>]` to the entry point with the same length assertion as `eddy_drive`, and **delete the `bvals`/`bvecs` parameters**: the call just replaced was their last reader. Confirm with `grep -n "bvals\|bvecs" mrsim-acq/src/kspace.rs`, which should return only the legacy wrapper's local names. TRXScan's call site drops the two arguments in the same step.

- [ ] **Step 6: Update TRXScan's call site**

```rust
    let prep_drive: Vec<Option<(f64, [f64; 3])>> = scheme.bvals.iter().zip(scheme.bvecs.iter())
        .map(|(&b, &v)| Some((b, v)))
        .collect();
```

Pass `Some(...)` for every volume, including b0 rows: `PrepPhase::shot`'s own `magnitude <= 0.0` guard is what disables them today, and moving that decision to the caller would change which branch runs.

- [ ] **Step 7: Test and gate**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test && cargo test --features io,kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test --features cli
tools/run_p0_baseline.sh /tmp/p0_t9.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_t9.txt && echo "PASS: bit-identical"
```

Expected: `PASS: bit-identical`. A failure here almost certainly means the caller normalized.

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat!: DiffusionPhase -> PrepPhase with its own per-volume drive

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Changes 1 and 6 — per-compartment T2 and T2' as scalar or map

The structural change, and the one the spec warns is not a one-line edit. `kspace.rs:647-653` evaluates relaxation **once per compartment per PE line** into a scalar `rel[c]`, outside the voxel loop. The header comment at `kspace.rs:436` lists relaxation as the "per line" factor, and the NUFFT path exists only because "the relaxation is an output-side scalar" (`kspace.rs:535-536`). A per-voxel T2 is not a per-line scalar, so neither structure survives it. T2 and T2' change together because both terms of `exp(-trf/T2 - |t|*1000/T2')` must be scalar for the factoring to hold.

**Files:**
- Modify: `mrsim-acq/src/kspace.rs` — new types, `SliceInput` (`:255-280`), `simulate_slice` (`:647-653`, `:695-698`), NUFFT gate (`:540`), `reference_coil_kspace` (`:1479`), entry point (`:1188`, `:1223`)
- Modify: `TRXScan/src/bin/trxscan.rs`, `trxscan_benchmark.rs` — wrap scalars in `T2Volume::Uniform`
- Modify: `TRXScan/src/benchmark.rs` — `produce_slice(.., t2: &[f32], ..)` (`:52`) becomes `t2: &[T2Slice]` with `t_inhom: None` in its `SliceInput` literal

**Interfaces:**
- Consumes: everything from Tasks 8-9
- Produces: `mrsim_acq::kspace::{T2Slice, T2Volume}`, `SliceInput { t2: &[T2Slice], t_inhom: Option<&[T2Slice]>, .. }`, and `simulate_acquisition_oversampled(.., t2: &[T2Volume], .., t_inhom: Option<&[T2Volume]>, ..)`. Both `t2` and `t_inhom` have one entry per compartment.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn constant_map_reproduces_uniform() {
    // Change 1's first map test. L2 relative over the coefficients, not coefficient-wise:
    // a near-zero coefficient makes a relative bound meaningless.
    let (nx, ny) = (16, 16);
    let comps = vec![box_hires(nx, ny, 4.0, 12.0, 4.0, 12.0)];
    let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
    let fmap = vec![0.0f32; nx * ny];
    let acq = Acquisition::default();
    let constant = 100.0f32;
    let map = vec![constant; nx * ny];

    let make = |t2: &[T2Slice]| SliceInput {
        compartments: &comp_refs, t2, fmap: &fmap, phase0: None, t_inhom: None,
        sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
        eddy_drive: None, prep_drive: None, slice_seed: 3, eddy_lin: None,
    };
    let u = simulate_slice_kspace(&make(&[T2Slice::Uniform(constant)]), &acq);
    let m = simulate_slice_kspace(&make(&[T2Slice::Map(&map)]), &acq);

    let (mut num, mut den) = (0.0f64, 0.0f64);
    for ((ur, ui), (mr, mi)) in u.iter().zip(m.iter()) {
        num += (ur - mr).powi(2) + (ui - mi).powi(2);
        den += ur.powi(2) + ui.powi(2);
    }
    let l2_rel = (num / den.max(1e-300)).sqrt();
    assert!(l2_rel < 1e-12, "constant map vs uniform: L2 relative {l2_rel:e}");
}

#[test]
fn varying_t2_map_matches_the_literal_sum() {
    // Change 1's second map test: the restructured forward against the reference oracle.
    let (nx, ny) = (12, 12);
    let comps = vec![box_hires(nx, ny, 3.0, 9.0, 3.0, 9.0)];
    let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
    let fmap = vec![0.0f32; nx * ny];
    let map: Vec<f32> = (0..nx * ny).map(|i| 60.0 + (i % 7) as f32 * 10.0).collect();
    let acq = Acquisition::default();
    let inp = SliceInput {
        compartments: &comp_refs, t2: &[T2Slice::Map(&map)], fmap: &fmap, phase0: None,
        t_inhom: None, sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
        eddy_drive: None, prep_drive: None, slice_seed: 5, eddy_lin: None,
    };
    let got = simulate_slice_kspace(&inp, &acq);
    // (inp, acq, coil, ncoils) -> Vec<C>, per kspace.rs:1403; single coil here.
    let want = reference_coil_kspace(&inp, &acq, 0, 1);
    for ((gr, gi), w) in got.iter().zip(want.iter()) {
        assert!((gr - w.re).abs() < 1e-10 && (gi - w.im).abs() < 1e-10,
                "restructured forward diverged from the literal sum");
    }
}

#[test]
fn negative_trf_on_an_acquired_line_is_rejected() {
    // A readout that starts before the excitation. TRXScan never hits this at TE 88 ms;
    // ASL will, at TE near 12 ms with a 64-line full-Fourier readout.
    let acq = Acquisition { t_echo: 2.0, t_line: 1.0, partial_fourier: 1.0, ..Default::default() };
    let err = validate_acquisition_timing(&acq, 64, 64).unwrap_err();
    assert!(err.contains("t_echo"), "the error must name the minimum feasible t_echo: {err}");
}

#[test]
fn zero_in_a_t2_map_is_rejected() {
    // -trf/0 is -inf when trf > 0 (harmless) but NaN when trf == 0, and 0 * NaN poisons
    // the Fourier sum. INFINITY is allowed and means no decay.
    assert!(validate_t2_map(&[100.0, 0.0, 50.0]).is_err());
    assert!(validate_t2_map(&[100.0, f32::INFINITY, 50.0]).is_ok());
    assert!(validate_t2_map(&[100.0, f32::NAN, 50.0]).is_err());
}
```

- [ ] **Step 2: Run them to confirm they fail**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test --features kspace 2>&1 | tail -20
```

Expected: `cannot find type T2Slice`, `cannot find function validate_t2_map`.

- [ ] **Step 3: Add the two types and the validators**

```rust
/// Per-compartment T2 (or T2') for ONE slice. Goes in [`SliceInput`].
#[derive(Debug, Clone, Copy)]
pub enum T2Slice<'a> {
    Uniform(f32),
    /// `snx * sny`, layout `x + snx*y`. Same convention as [`SliceInput::fmap`].
    Map(&'a [f32]),
}

/// Per-compartment T2 (or T2') for the WHOLE volume. Goes to the entry point, which cuts
/// each z slice itself.
#[derive(Debug, Clone, Copy)]
pub enum T2Volume<'a> {
    Uniform(f32),
    /// `snx * sny * nz`, layout `x + snx*(y + sny*z)`.
    Map(&'a [f32]),
}

/// Every map value must be strictly positive; `f32::INFINITY` is allowed and means no decay.
pub fn validate_t2_map(m: &[f32]) -> Result<(), String> {
    for (i, &v) in m.iter().enumerate() {
        if v.is_nan() || v <= 0.0 {
            return Err(format!("T2 map value {v} at index {i} is not strictly positive"));
        }
    }
    Ok(())
}

/// `trf = t_echo + t` must be positive on every acquired line. A negative `trf` is a readout
/// that begins before the excitation, and the forward model answers it with signal growth.
pub fn validate_acquisition_timing(acq: &Acquisition, nx: usize, ny: usize) -> Result<(), String> {
    // SingleShotEpi has no constructor; build the literal exactly as `simulate_slice` does at
    // kspace.rs:383-389, or the timing this validates is not the timing that runs.
    let epi = SingleShotEpi {
        kx_max: nx,
        ky_max: ny,
        t_line: acq.t_line,
        t_echo: acq.t_echo,
        reverse_phase: acq.reverse_phase,
    };
    let (t_ms, trf_ms, _) = line_times(&epi);
    let mask = sampling_mask(nx, ny, acq);
    let mut worst = f64::INFINITY;
    for ky in 0..ny {
        if mask[ky * nx] && trf_ms[ky] < worst { worst = trf_ms[ky]; }
    }
    if worst <= 0.0 {
        let need = acq.t_echo - worst;
        return Err(format!(
            "trf = {worst:.3} ms <= 0 on an acquired line: the readout starts before the \
             excitation. Raise t_echo above {need:.3} ms or shorten t_line. Partial Fourier \
             does not help: it drops low-ky lines, which this trajectory reads last."));
    }
    let _ = t_ms;
    Ok(())
}
```

- [ ] **Step 4: Restructure the relaxation**

Replace `kspace.rs:647-653` with a uniformity test hoisted above the line loop, and a per-voxel branch inside it:

```rust
    // Both terms of exp(-trf/T2 - |t|/T2') must be scalar for the per-line factoring — and
    // therefore the NUFFT path — to survive. One Map anywhere takes the whole slice to the
    // rotor path; Uniform compartments in a mixed slice still use their scalar.
    let uniform = t2.iter().all(|s| matches!(s, T2Slice::Uniform(_)))
        && inp.t_inhom.map_or(true, |ti| ti.iter().all(|s| matches!(s, T2Slice::Uniform(_))));
```

When `uniform`, keep the existing `rel[c]` loop, reading both scalars out of their enums. The per-compartment `t_inhom` **must** be read here too, not only in the map branch: `class` mode in aslscan passes `Uniform` T2' per label and expects it honoured on the fast path, and nothing in P0's gate would notice if it were silently ignored, because TRXScan always passes `None`.

```rust
        for (c, w) in rel.iter_mut().enumerate() {
            *w = if acq.do_relaxation {
                let T2Slice::Uniform(t2c) = t2[c] else { unreachable!("uniform branch") };
                let tic = match inp.t_inhom.map(|ti| ti[c]) {
                    None => acq.t_inhom,
                    Some(T2Slice::Uniform(v)) => v as f64,
                    Some(T2Slice::Map(_)) => unreachable!("uniform branch"),
                };
                (-trf / t2c as f64 - t.abs() * 1000.0 / tic).exp()
            } else {
                1.0
            };
        }
```

With `t_inhom: None` the expression is `-trf / t2c as f64 - t.abs() * 1000.0 / acq.t_inhom`, the same operations in the same order as today, which is what keeps the gate. Add a fifth test to Step 1, `uniform_t_inhom_overrides_the_acquisition_scalar`: the same slice with `t_inhom: Some(&[T2Slice::Uniform(20.0)])` and `acq.t_inhom = 50.0` must differ from the `None` run, and must equal the run with `t_inhom: None, acq.t_inhom = 20.0` bit for bit.

When not uniform, leave `rel` unused and evaluate inside the voxel loop at `kspace.rs:695-698`:

```rust
                for (c, comp) in compartments.iter().enumerate() {
                    let r = if acq.do_relaxation {
                        let t2c = match t2[c] {
                            T2Slice::Uniform(v) => v as f64,
                            T2Slice::Map(m) => m[i] as f64,
                        };
                        let tic = match inp.t_inhom.map(|ti| ti[c]) {
                            Some(T2Slice::Uniform(v)) => v as f64,
                            Some(T2Slice::Map(m)) => m[i] as f64,
                            None => acq.t_inhom,
                        };
                        (-trf / t2c - t.abs() * 1000.0 / tic).exp()
                    } else { 1.0 };
                    w += r * comp[i] as f64;
                }
```

The `uniform` branch must keep the existing expression character for character. Any retyping risks changing floating-point order, and Task 6's gate will catch it but only after you have spent the time.

- [ ] **Step 5: Gate the NUFFT path**

At `kspace.rs:540`:

```rust
    let nufft_rows: Option<Vec<(Vec<f64>, Vec<f64>)>> = (!do_eddy && ny >= 3 && uniform).then(|| {
```

- [ ] **Step 6: Give the reference oracle the same lookup**

`reference_coil_kspace` at `kspace.rs:1479` has the same `t2[c]` index. Apply the same `match`. The oracle must model what the forward model does, or the second test above compares two different things and passes for the wrong reason.

- [ ] **Step 7: Slice the volumes at the entry point**

In `simulate_acquisition_oversampled`, take `t2: &[T2Volume]` and `t_inhom: Option<&[T2Volume]>`, validate every `Map` with `validate_t2_map` and the timing with `validate_acquisition_timing` (with the acquired matrix) before the volume loop, and cut each z slice in the loop that already copies compartment and fieldmap slices at `kspace.rs:1223`:

```rust
            let t2_slice: Vec<T2Slice> = t2.iter().map(|v| match v {
                T2Volume::Uniform(s) => T2Slice::Uniform(*s),
                T2Volume::Map(m) => T2Slice::Map(&m[z * snx * sny..(z + 1) * snx * sny]),
            }).collect();
```

- [ ] **Step 8: Update TRXScan's call sites**

Wrap the existing scalars: `t2: &[T2Volume::Uniform(t2_wm), T2Volume::Uniform(t2_gm), ...]`, and pass `t_inhom: None` so `Acquisition::t_inhom` stays the source.

- [ ] **Step 9: Update the earlier tasks' test literals**

The two tests added in Task 8 and the one in Task 9 construct `SliceInput` with `t2: &[100.0]`. That field is now `&[T2Slice]`. Update them to `t2: &[T2Slice::Uniform(100.0)]` and add `t_inhom: None`. The compiler lists every site.

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test --features kspace 2>&1 | grep -c "^error" || echo "no errors"
```

- [ ] **Step 10: Test and gate**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test && cargo test --features io,kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test --features cli
tools/run_p0_baseline.sh /tmp/p0_t10.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_t10.txt && echo "PASS: bit-identical"
```

Expected: `PASS: bit-identical` under both feature sets. TRXScan passes only `Uniform`, so it must take the same branch it always did, including the NUFFT path under `kspace`.

- [ ] **Step 11: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat!: per-compartment T2 and T2' as scalar or map

One Map anywhere takes the slice to the rotor path and disables the
NUFFT gate, because the factoring needs both terms of
exp(-trf/T2 - |t|/T2') to be per-line scalars. Uniform compartments in a
mixed slice keep their scalar. Adds the map contract (strictly positive
or INFINITY) and the trf > 0 check, which no valid TRXScan
configuration can trip but a short-TE ASL protocol will.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Change 5 — finalize the entry point

Renames `ngrad` to `n_volumes` and adds the validation an extracted public entry point cannot delegate to one particular caller. `eddy_trace` is indexed `tr[g]` inside the loop (`kspace.rs:1257`) and is checked today only by TRXScan's CLI (`trxscan.rs:868-885`).

`noise_sigma` is deliberately left alone: one acquired-grid map for all volumes.

**Files:**
- Modify: `mrsim-acq/src/kspace.rs` — `simulate_acquisition_oversampled`, `simulate_acquisition_legacy`

**Interfaces:**
- Consumes: Tasks 8-10
- Produces: the final entry-point signature from the spec, with `n_volumes`, both drive slices, `t2: &[T2Volume]`, `t_inhom: Option<&[T2Volume]>`, `noise_sigma: Option<&[f32]>`, `eddy_trace: Option<&[[f64; 3]]>`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn per_volume_slice_lengths_are_validated_before_the_loop() {
    let dims = [4usize, 4, 1];
    let images = vec![vec![0.0f32; 16 * 2]];
    let t2 = [T2Volume::Uniform(100.0)];
    let fmap = vec![0.0f32; 16];
    let acq = Acquisition::default();
    let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
    // The panic MESSAGE is asserted, not the panic. An out-of-bounds `tr[g]` (kspace.rs:1257)
    // already panics today, so `is_err()` would pass before any validation exists.
    fn message(r: std::thread::Result<(Vec<f32>, Vec<f32>)>) -> String {
        let e = r.expect_err("must panic");
        e.downcast_ref::<String>().cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    }
    // Two volumes, but only one drive entry.
    let r = std::panic::catch_unwind(|| {
        simulate_acquisition_oversampled(
            dims, dims, 2, &images, &t2, &fmap, None, &acq,
            &[None], &[None, None], &phase, 1, None, None)
    });
    assert!(message(r).contains("eddy_drive has 1 entries for 2 volumes"));

    let r = std::panic::catch_unwind(|| {
        simulate_acquisition_oversampled(
            dims, dims, 2, &images, &t2, &fmap, None, &acq,
            &[None, None], &[None, None], &phase, 1, None, Some(&[[0.0; 3]]))
    });
    assert!(message(r).contains("eddy_trace has 1 entries for 2 volumes"));
}
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test per_volume_slice_lengths 2>&1 | tail -10
```

Expected: the second assertion fails. The call does panic, but with `index out of bounds` from `tr[g]` at `kspace.rs:1257`, not with the validation message, which does not exist yet.

- [ ] **Step 3: Rename and validate**

```rust
    assert_eq!(eddy_drive.len(), n_volumes,
               "eddy_drive has {} entries for {n_volumes} volumes", eddy_drive.len());
    assert_eq!(prep_drive.len(), n_volumes,
               "prep_drive has {} entries for {n_volumes} volumes", prep_drive.len());
    if let Some(tr) = eddy_trace {
        assert_eq!(tr.len(), n_volumes,
                   "eddy_trace has {} entries for {n_volumes} volumes", tr.len());
    }
```

Rename `ngrad` to `n_volumes` throughout both entry points. `do_eddy_phase` (`kspace.rs:394`) takes the same `eddy_drive.is_some()` guard as `do_eddy`.

- [ ] **Step 4: Document the one-call-per-series contract (change 5a)**

No code changes — TRXScan already makes one call per series — but the contract must be written down, because `aslscan` will read this doc comment and the alternative is a silent bug. Add above `simulate_acquisition_oversampled`:

```rust
/// Simulate a whole series in ONE call.
///
/// Every random stream here is keyed on the volume index `g` **within this call**:
/// `slice_seed` (used by k-space noise and spikes), the image-space noise stream, and the
/// per-shot phase realization. Calling this once per volume therefore gives every volume the
/// same `g = 0` and the same noise, which for ASL would make control minus label cancel the
/// noise exactly and produce impossibly clean perfusion maps.
///
/// A caller that must make more than one call — an ASL series plus a separate M0 scan, say —
/// passes a different `seed` to each, because `seed` is the only thing distinguishing them.
```

- [ ] **Step 5: Test and gate**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test --features io,kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test --features cli
tools/run_p0_baseline.sh /tmp/p0_t11.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_t11.txt && echo "PASS: bit-identical"
```

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat!: entry point counts volumes and validates its per-volume slices

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Change 7 — multiband dropout stops being b-value-scaled

`apply_multiband_motion` (`motion.rs:314`) attenuates a dropped shot by `1 - severity*(b/b_max)` and exempts b0 with a `bvals.get(g) < 50.0` test (`motion.rs:370`). That is a diffusion-specific dropout law sitting in a crate that is no longer diffusion-specific.

**Files:**
- Modify: `mrsim-acq/src/motion.rs:290-380`
- Modify: `TRXScan/src/bin/trxscan.rs`

**Interfaces:**
- Consumes: `mrsim_acq::motion`
- Produces: `mrsim_acq::motion::DropoutLaw` with variants `Scaled { drive: Vec<f64>, floor: f64 }` and `Uniform`. `apply_multiband_motion` takes `law: &DropoutLaw` in place of `bvals: &[f64], b_max: f64`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn scaled_law_reproduces_the_b_value_attenuation_and_uniform_does_not() {
    let law = DropoutLaw::Scaled { drive: vec![5.0, 1000.0, 2000.0], floor: 50.0 };
    // b0 row (below the floor) is exempt: attenuation 1.0.
    assert!((law.attenuation(0, 0.5) - 1.0).abs() < 1e-12);
    // b = 2000 is b_max, so the full severity applies.
    assert!((law.attenuation(2, 0.5) - 0.5).abs() < 1e-12);
    // b = 1000 is half of b_max, so half the severity.
    assert!((law.attenuation(1, 0.5) - 0.75).abs() < 1e-12);

    let u = DropoutLaw::Uniform;
    for g in 0..3 {
        assert!((u.attenuation(g, 0.5) - 0.5).abs() < 1e-12,
                "Uniform must not vary with the volume index");
    }
}
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test scaled_law 2>&1 | tail -10
```

Expected: `cannot find type DropoutLaw`.

- [ ] **Step 3: Add the enum**

```rust
/// How a dropped shot's signal is attenuated. Diffusion scales with b-value and exempts b0;
/// a sequence without a diffusion weighting has nothing to scale with.
#[derive(Debug, Clone)]
pub enum DropoutLaw {
    /// `1 - severity * (drive / drive_max)`, with `drive < floor` exempt. Diffusion passes
    /// b-values and a floor of 50.0, reproducing the previous behavior exactly.
    Scaled { drive: Vec<f64>, floor: f64 },
    /// `1 - severity`, applied to every volume alike.
    Uniform,
}

impl DropoutLaw {
    pub fn attenuation(&self, volume: usize, severity: f32) -> f32 {
        match self {
            DropoutLaw::Uniform => 1.0 - severity,
            DropoutLaw::Scaled { drive, floor } => {
                let d = drive.get(volume).copied().unwrap_or(0.0);
                let d_max = drive.iter().cloned().fold(0.0f64, f64::max);
                if d < *floor || d_max <= 0.0 { 1.0 } else { 1.0 - severity * (d / d_max) as f32 }
            }
        }
    }
}
```

- [ ] **Step 4: Use it in apply_multiband_motion**

Replace the `bvals`/`b_max` parameters with `law: &DropoutLaw` and the inline computation at `motion.rs:370-375` with `law.attenuation(g, e.severity)`.

- [ ] **Step 5: Update TRXScan's call site**

```rust
    let law = mrsim_acq::motion::DropoutLaw::Scaled {
        drive: scheme.bvals.clone(),
        floor: 50.0,
    };
```

- [ ] **Step 6: Test and gate**

The fixture's multiband configuration is what exercises this.

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo test && cargo test --features io,kspace,par
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo test --features cli
tools/run_p0_baseline.sh /tmp/p0_t12.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_t12.txt && echo "PASS: bit-identical"
```

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat!: DropoutLaw replaces b-value-scaled multiband dropout

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Change 8 — split the complex writer, then run the full gate

`write_complex_dwi` becomes `write_complex_4d` in `mrsim-acq`, and `SidecarInfo` moves with it now (Task 5 left both in TRXScan because `write_complex_dwi` names `GradientScheme`). `.bval`/`.bvec` writing stays in TRXScan as `write_dwi_scheme`, and `write_complex_dwi` is deleted. Ends with every acceptance criterion run in one pass.

**Files:**
- Modify: `mrsim-acq/src/io.rs`
- Modify: `TRXScan/src/io.rs`, `TRXScan/src/bin/trxscan.rs`

**Interfaces:**
- Consumes: Tasks 5, 10-12
- Produces: `mrsim_acq::io::{write_complex_4d, SidecarInfo}`, where `SidecarInfo` holds only modality-independent fields. `TRXScan::io::write_dwi_scheme` writes `.bval`/`.bvec`.

- [ ] **Step 1: Split the function**

`SidecarInfo` (`io.rs:422-435`) needs no change: `phase_encoding_direction`, `total_readout_time`, `echo_time`, `partial_fourier`, `accel`, `mb`, and `b0_field_source` are all modality-independent already. Two things in `write_complex_dwi` are not.

First, it takes `scheme: &GradientScheme` and calls `write_bval_bvec` (`io.rs:450`). Drop both.

Second, every output filename hardcodes `_dwi`: `_part-mag_dwi.nii.gz`, `_part-phase_dwi.nii.gz`, `_part-mag_dwi.json`, `_part-phase_dwi.json`. Parameterize it, or `aslscan` cannot write `_asl` files through this function.

```rust
/// Write a complex 4D volume as BIDS `part-mag` / `part-phase` NIfTI plus sidecars.
/// `suffix` is the BIDS modality suffix without the leading underscore: `"dwi"`, `"asl"`.
pub fn write_complex_4d(
    out_prefix: &str,
    suffix: &str,
    dims: [usize; 3],
    n_volumes: usize,
    mag: &[f32],
    phase: &[f32],
    grid: &Grid,
    info: &SidecarInfo,
) -> R<()> {
    // PathBuf, as io.rs:447 has it: `write_4d` takes `&Path` (io.rs:353) and `&String` does
    // not coerce to it.
    let p = |tail: &str| PathBuf::from(format!("{out_prefix}{tail}"));
    write_4d(&p(&format!("_part-mag_{suffix}.nii.gz")), dims, n_volumes, mag, grid)?;
    write_4d(&p(&format!("_part-phase_{suffix}.nii.gz")), dims, n_volumes, phase, grid)?;
    // the sidecar-writing tail of the old body follows here unchanged, with `_dwi.json`
    // replaced by `_{suffix}.json`
    Ok(())
}
```

- [ ] **Step 2: Add the scheme writer to TRXScan**

In `TRXScan/src/io.rs`:

```rust
/// Write the FSL scheme files beside a complex 4D DWI. Diffusion-specific, so it stays here.
pub fn write_dwi_scheme(out_prefix: &str, scheme: &GradientScheme) -> R<()> {
    // The `write_bval_bvec` call lifted out of write_complex_dwi (io.rs:450), with the same
    // `_dwi.bval` / `_dwi.bvec` names it wrote before.
    // PathBuf, as io.rs:447 has it: `write_4d` takes `&Path` (io.rs:353) and `&String` does
    // not coerce to it.
    let p = |tail: &str| PathBuf::from(format!("{out_prefix}{tail}"));
    write_bval_bvec(&p("_dwi.bval"), &p("_dwi.bvec"), scheme)
}
```

- [ ] **Step 3: Update the binary**

`trxscan.rs` now calls `write_complex_4d` then `write_dwi_scheme`.

- [ ] **Step 4: Verify the written files are unchanged**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
cargo test --features cli
tools/run_p0_baseline.sh /tmp/p0_t13.txt
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_t13.txt && echo "PASS: bit-identical"
```

The baseline checksums cover every file the run writes, `.bval` and `.bvec` included, so a split that drops or reorders them fails here.

- [ ] **Step 5: Run every acceptance criterion**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
cargo test                                    # criterion 1: pure std, offline
cargo test --features io,kspace,par,config

cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
cargo test                                    # criterion 2: no features
cargo test --features cli                     # criterion 2: with cli
cargo test --features cli,kspace,par

tools/run_p0_baseline.sh /tmp/p0_final.txt    # criterion 3, both feature sets
diff tests/fixtures/p0_baseline/checksums.txt /tmp/p0_final.txt && echo "CRITERION 3: PASS"

cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq && cargo clippy --all-targets 2>&1 | grep -c "^warning"
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan && cargo clippy --all-targets 2>&1 | grep -c "^warning"
```

Expected: every test green; `CRITERION 3: PASS`; clippy warning count in TRXScan no higher than the ~6 that pre-existed. Criteria 4, 5 and 6 are the tests added in Tasks 8 and 10 and are covered by the suites above.

- [ ] **Step 6: Commit and tag**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq
git add -A && git commit -m "feat!: write_complex_4d replaces write_complex_dwi

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git tag p0-complete
cd /mnt/c/Users/tsalo/Documents/rust-trx/TRXScan
git add -A && git commit -m "refactor: write the FSL scheme from TRXScan, not the shared writer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git tag p0-complete
```

---

## Acceptance Criteria Coverage

| Spec criterion | Where it is checked |
|---|---|
| 1. `cargo test` in `mrsim-acq`, pure std, offline | Tasks 2-5 Step "test", Task 13 Step 5 |
| 2. `cargo test` in TRXScan, no features and `--features cli` | Tasks 2-5, Task 13 Step 5 |
| 3. Bit-identical `trxscan` output, both feature sets | Task 1 (baseline), Task 6, and every task from 7 on |
| 4. `eddy_drive: Some(bvec * bval)` reproduces the old eddy path | Task 8 Step 1, first test |
| 5. `None` disables as `bval.abs() > 1e-9` did; `Some([0;3])` does not | Task 8 Step 1, second test |
| 6. Two map-path tests; `trf <= 0` rejected | Task 10 Step 1, five tests |
| 7. No new clippy warnings | Task 13 Step 5 |

## What P0 does not do

Per the spec, P0 moves `noise.rs` and `config.rs` as `todo!()` stubs and implements neither. The TOML overlay P1 needs is new work in `aslscan`. Per-compartment NUFFT eligibility, so a `Uniform` compartment keeps the fast path beside a `Map` one, is a deferred optimization: the data structures at `kspace.rs:554-561` are already per compartment, but only the gate at `kspace.rs:680` is not, and changing it would make P0 something other than a refactor.
