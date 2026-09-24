# P1: aslscan Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new `aslscan` crate that reads a BIDS ASL protocol and a BIDS-derivatives phantom, runs the Buxton general kinetic model and the spin-echo signal equation per phantom voxel, resamples magnetization to the simulation grid, drives `mrsim-acq` once for the whole series, and writes a complete, validator-clean BIDS ASL dataset with ground truth.

**Architecture:** Signal stage in `aslscan`, acquisition stage in `mrsim-acq` (P0). Modules are built inside-out — the two pure functions with a Python oracle first (`kinetic`, `mrsignal`), then the two loaders (`protocol`, `phantom`), then the grid machinery (`resample`), then the orchestration and writer (`series`, `bids`, CLI) — so every module's tests exist before the next one consumes it. simasl is the numerical oracle for the pure functions; the spec's linearity property and noise-decorrelation test are the oracles for the orchestration, because nothing else can check that.

**Tech Stack:** Rust 2021. `mrsim-acq` (path dep, feature `io`). `serde`/`serde_json` (BIDS JSON), `toml` (overlay), `clap` (CLI), all behind features so the default build stays pure std. Fixture generation and the phantom converter run in the `simasl` micromamba env (Python 3.8, numpy 1.19.5, nibabel 3.1.1). BIDS validation with `deno run -A jsr:@bids/validator` (3.0.2; `deno` is at `~/.nvm/versions/node/v22.22.2/bin`).

**Spec:** `/mnt/c/Users/tsalo/Documents/rust-trx/mrsim-acq/docs/specs/2026-09-21-mrsim-acq-aslscan-design.md`, sections "P1: aslscan core", "Testing and verification / P1", and "The blood-compartment linearity property". P0 is complete at tag `p0-complete` in `mrsim-acq` and TRXScan.

## Global Constraints

- **Default build is pure std.** `cargo test` with no features runs `kinetic`, `mrsignal`, `resample`, and `bids` naming tests offline with no external crates. `protocol`, `phantom`, `series` and the binary need `io` (serde, serde_json, toml, mrsim-acq/io); the binary needs `cli`.
- **Seconds upstream, milliseconds downstream.** `kinetic` and `mrsignal` run in seconds; the only conversions are in `protocol` (`EchoTime`, `TotalReadoutTime` -> `Acquisition`) and `phantom` (T2, T2\*, T2', blood T2 -> `T2Volume`). Grep for `* 1000` and `/ 1000` after each task; they belong in those two files only.
- **One acquisition call per series** (spec P0 change 5a). The separate M0 scan is the only second call and uses `seed ^ 0x4D30_5343_414E`.
- **Evaluate physics on the phantom grid, then average magnetization.** Never average perfusion, ATT, or T1 before the kinetic model. Class decomposition happens on the phantom grid before resampling.
- **Faithful port, not a better one.** `kinetic` reproduces simasl's guards (`0 < t <= dt`, the two different zero-`T1b` tests, `np.divide(where=)` zeros), and the fixture diff at `1e-9` relative is the referee. Deviations from simasl (transverse relaxation removed from `mrsignal`, no steady-state term on the blood compartment, separate blood compartment) are the spec's and are the only ones.
- **Every default that is applied is written to the output sidecar.** Silent defaults are the failure mode for BIDS-in-the-wild datasets.
- **Hand-formatted at ~100 columns**, matching the two sibling crates. No `cargo fmt`, no `rustfmt.toml`.
- **Large generated data is never committed.** The converted ASLDRO phantom (197 x 233 x 189 x 7 float64, ~1 GB of NIfTI) lives under `aslscan/work/` (gitignored). Tests use a checked-in 24 x 24 x 6 crop of it (`tests/fixtures/phantom-crop/`, ~60 KB gzipped), produced by the same converter with `--crop`.

---

## File Structure

**New crate `aslscan/` (sibling of `mrsim-acq/`, `TRXScan/`, `simasl/`):**

| Path | Responsibility |
|---|---|
| `Cargo.toml` | Features `io`, `cli`; `mrsim-acq` path dep |
| `src/lib.rs` | Module declarations, `Seconds`/`Millis` newtypes are NOT used (plain `f64`, unit in the name) |
| `src/kinetic.rs` | Buxton GKM per voxel, PASL and (P)CASL branches, simasl's guards |
| `src/mrsignal.rs` | Spin-echo post-excitation magnetization; `ge`/`ir` rejection |
| `src/protocol.rs` | `asl.json` + `aslcontext.tsv` + TOML overlay -> `Protocol`; `Protocol::acquisition()` |
| `src/phantom.rs` | BIDS-derivatives maps + `phantom.json` -> `Phantom`; T2 modes; T2' derivation |
| `src/resample.rs` | Separable box-overlap weights; phantom -> simulation and acquisition grids |
| `src/series.rs` | Row semantics, class split, 4D assembly, the one call, the separate M0 |
| `src/bids.rs` | Output names, `aslcontext.tsv`, sidecars, `dataset_description.json`, `.bidsignore`, ground truth |
| `src/bin/aslscan.rs` | clap CLI |
| `tools/hrgt_to_bids.py` | ASLDRO packed 5D -> BIDS-derivatives phantom (+ `--crop`) |
| `tools/gen_gkm_fixtures.py` | simasl `GkmFilter` oracle -> `tests/fixtures/gkm.txt` |
| `tools/gen_mrsignal_fixtures.py` | simasl `MriSignalFilter` oracle, TE factor divided out -> `tests/fixtures/mrsignal.txt` |
| `tools/fetch_protocol_fixtures.sh` | Real sidecars from `bids-examples` -> `tests/fixtures/protocols/` |
| `tests/fixtures/phantom-crop/` | Cropped ASLDRO 3T phantom in the phantom contract's layout |
| `tests/fixtures/protocols/` | Real and hand-written `asl.json` / `aslcontext.tsv` / overlay cases |
| `tests/end_to_end.rs` | The runs of acceptance criteria 1, 2, 5, 6 (feature `io`) |
| `work/` | gitignored: full converted phantoms, run outputs |

**Modified:** nothing in `mrsim-acq` or TRXScan. If a task finds it needs an `mrsim-acq` change, that is a finding to report, not a thing to do inline.

---

### Task 1: Crate scaffold, phantom converter, checked-in crop

Everything later needs a phantom to load and a crate to live in. The converter is written first because the spec's `class`-mode claims rest on properties of the real phantom (bitwise per-label constancy, the CSF ATT sentinel), and the crop that tests use must be cut from the real thing, not synthesized.

**Files:**
- Create: `aslscan/Cargo.toml`, `aslscan/src/lib.rs`, `aslscan/.gitignore`, `aslscan/tools/hrgt_to_bids.py`
- Create: `aslscan/tests/fixtures/phantom-crop/` (converter output)

**Interfaces:**
- Produces: the phantom layout of the spec's phantom contract. Per map `<name>.nii.gz` + `<name>.json` with `{"Units": ...}`; `dseg.json` additionally carries `{"LabelMap": {"1": "grey_matter", ...}}`; `phantom.json` carries `{"LambdaBloodBrain": 0.9, "T1ArterialBlood": 1.65, "MagneticFieldStrength": 3, "Source": "hrgt_icbm_2009a_nls_3t", "Converter": "hrgt_to_bids.py"}`.

- [ ] **Step 1: Manifest and crate root**

```toml
[package]
name = "aslscan"
version = "0.0.0"
edition = "2021"
description = "ASL digital reference object simulator on the mrsim-acq acquisition stage"
license = "MIT OR Apache-2.0"
publish = false

[features]
default = []
# BIDS JSON/TSV/TOML in, NIfTI in and out (via mrsim-acq/io).
io = ["dep:serde", "dep:serde_json", "dep:toml", "mrsim-acq/io"]
# The binary. Implies io.
cli = ["dep:clap", "io"]
# Forwarded to the acquisition stage.
kspace = ["mrsim-acq/kspace"]
par = ["mrsim-acq/par"]

[dependencies]
mrsim-acq = { path = "../mrsim-acq" }
serde = { version = "1", features = ["derive"], optional = true }
serde_json = { version = "1", optional = true }
toml = { version = "0.8", optional = true }
clap = { version = "4", features = ["derive"], optional = true }

[[bin]]
name = "aslscan"
required-features = ["cli"]
```

`src/lib.rs` declares `kinetic`, `mrsignal`, `resample`, `bids` unconditionally and `protocol`, `phantom`, `series` under `#[cfg(feature = "io")]`. Only `kinetic` exists after this task; add the others as their tasks land.

`.gitignore`: `target/`, `Cargo.lock`, `work/`.

- [ ] **Step 2: Write the converter**

`tools/hrgt_to_bids.py`:

```python
"""Convert an ASLDRO packed ground truth (5D NIfTI + JSON) into the aslscan phantom layout.

    micromamba run -n simasl python tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t --out work/phantom-3t
    micromamba run -n simasl python tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t \
        --out tests/fixtures/phantom-crop --crop 88:112 104:128 90:96

One NIfTI per quantity, each with a JSON sidecar carrying Units; dseg.json carries the label
names; phantom.json carries the kinetic constants the oracle was built with. Everything is
written float32 except dseg (int16). The 5D file's 4th axis is a singleton and is dropped.
"""
```

Quantity -> file mapping: `perfusion_rate` -> `perfusion` (`ml/100g/min`), `transit_time` -> `att` (`s`), `t1` -> `T1map` (`s`), `t2` -> `T2map` (`s`), `t2_star` -> `T2starmap` (`s`), `m0` -> `M0map` (`arbitrary`), `seg_label` -> `dseg` (label indices). Units come from the source JSON's `units` list, not from this table; the table is what the converter *expects*, and a mismatch is an error. The affine is the source affine, cropped affines get the origin shifted by the crop start times the voxel size.

The crop `88:112 104:128 90:96` was chosen on 2026-09-23 to contain all three labels: verify with the script's own summary line, which must print voxel counts for labels 1, 2 and 3, each nonzero.

- [ ] **Step 3: Run it both ways and check the crop**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/aslscan
micromamba run -n simasl python tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t --out work/phantom-3t
micromamba run -n simasl python tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t \
    --out tests/fixtures/phantom-crop --crop 88:112 104:128 90:96
du -sh tests/fixtures/phantom-crop work/phantom-3t
```

Expected: the crop prints nonzero counts for labels 1, 2, 3 and is well under 100 KB. If a label is missing, move the crop window; the plan's window is a guess checked at run time, not a fact.

- [ ] **Step 4: Confirm the spec's per-label constancy claim on the full phantom**

```bash
micromamba run -n simasl python - <<'EOF'
import nibabel as nib, numpy as np
d = "work/phantom-3t/"
seg = nib.load(d + "dseg.nii.gz").get_fdata()
for m in ["T1map", "T2map", "T2starmap", "M0map", "perfusion", "att"]:
    v = nib.load(d + m + ".nii.gz").get_fdata()
    print(m, [len(np.unique(v[seg == L])) for L in (1, 2, 3)])
EOF
```

Expected: every list is `[1, 1, 1]`. This is the fact `class` mode depends on.

- [ ] **Step 5: Build and commit**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/aslscan && cargo build && cargo build --features io
git init -q && git add -A && git commit -q -m "feat: aslscan crate scaffold, ASLDRO phantom converter, test crop

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: kinetic

The Buxton GKM as simasl computes it, per voxel. Test-first against hand-computed closed forms, then against simasl fixtures at `1e-9` relative.

**Files:**
- Create: `src/kinetic.rs`, `tools/gen_gkm_fixtures.py`, `tests/fixtures/gkm.txt`

**Interfaces:**
- Produces:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LabelType { Pasl, Casl, Pcasl }

/// Per-series kinetic constants. Seconds throughout.
#[derive(Debug, Clone, Copy)]
pub struct Kinetic {
    pub label_type: LabelType,
    /// Bolus duration `tau` (s): LabelingDuration for (P)CASL, BolusCutOffDelayTime[0] for PASL.
    pub tau: f64,
    pub alpha: f64,
    pub lambda: f64,
    pub t1b: f64,
}

/// `delta_m` for one voxel at signal time `t` (s). `f` in ml/100g/min as the phantom stores it;
/// the /6000 happens here, as in gkm_filter.py:105. Returns exactly what simasl's masked array
/// holds for this voxel, including its zeros.
pub fn delta_m(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64) -> f64
```

- [ ] **Step 1: Write the closed-form tests**

In `src/kinetic.rs`'s test module, with `K_PCASL = Kinetic { Pcasl, tau 1.8, alpha 0.85, lambda 0.9, t1b 1.65 }` and GM-like inputs `f = 60, dt = 0.8, t1t = 1.33, m0 = 74.622`:

- `not_arrived_is_zero`: `t = 0.5` (< dt) gives exactly `0.0`; so does `t = dt` (the `<=`); so does `t = 0.0` and `t = -1.0` (the `0 < t` half of simasl's chained comparison makes the not-arrived mask false, but the arriving and arrived masks are false too, so the result is the initial zero).
- `pcasl_arrived_matches_closed_form`: `t = 3.6` (PLD 1.8 + tau 1.8, the simasl default): compute `T1' = 1/(1/1.33 + (60/6000)/0.9)`, `M0b = 74.622/0.9`, and the arrived expression by hand in the test with `f64` arithmetic; assert equality to `1e-12` relative. This is the same arithmetic, so it also pins the operation order.
- `pcasl_arriving_at_midpoint`: `t = dt + tau/2`: `q = 1 - exp(-(t-dt)/T1')`.
- `pasl_arriving_and_arrived`: PASL with `tau = 0.7` (a bolus cutoff), `t = 1.2` (arriving) and `t = 2.0` (arrived), against the `k`-form expressions written out.
- `zero_perfusion_is_zero`: `f = 0` at every `t`.
- `zero_t1b_is_zero_in_both_branches`: `t1b = 0`, arrived `t`: PASL and PCASL both give `0.0` exactly (the exponential is replaced, not the quotient).
- `pasl_k_zero_uses_the_quotient_guard`: choose `t1b` so that `1/t1b == 1/T1'` exactly (set `t1b = T1'` computed the same way); `k = 0` makes `denominator = 0`, simasl's `np.divide(where=)` yields `q = 0`, so `delta_m = 0` — not the `0/0` limit a textbook would take. This is a fidelity test, and the comment says so.

- [ ] **Step 2: Write the fixture generator**

`tools/gen_gkm_fixtures.py`, modeled on `TRXScan/tools/gen_force_fixtures.py`:

```python
"""simasl GkmFilter oracle -> tests/fixtures/gkm.txt (plain text, pure-std parsing).

Every input is cast to float64 before the filter runs: simasl keeps the NIfTI dtype and a
float32 phantom makes perfusion_rate float32 (gkm_filter.py:105), which caps agreement at ~1e-7.
The header records the dtype so a regenerated fixture cannot silently be float32.
"""
import numpy as np
from asldro.containers.image import NumpyImageContainer
from asldro.filters.gkm_filter import GkmFilter

SEED = 20260923
CASES = [  # (label_type, label_duration, signal_time, label_efficiency, lambda, t1b)
    ("pcasl", 1.8, 3.6, 0.85, 0.9, 1.65), ("pcasl", 1.8, 2.0, 0.85, 0.9, 1.65),
    ("pcasl", 1.8, 1.0, 0.85, 0.9, 1.65), ("casl", 1.5, 3.0, 0.68, 0.9, 1.65),
    ("pasl", 0.7, 1.8, 0.98, 0.9, 1.65), ("pasl", 0.7, 1.2, 0.98, 0.9, 1.65),
    ("pasl", 0.7, 0.9, 0.98, 0.9, 1.65), ("pcasl", 1.8, 3.6, 0.85, 0.9, 0.0),
    ("pasl", 0.7, 1.8, 0.98, 0.9, 0.0), ("pcasl", 1.8, 3.6, 0.85, 0.0, 1.65),
]
```

Per case: a `(6, 6, 2)` array per input, drawn with the seeded RNG: `perfusion_rate` uniform 0..90 with a quarter of the voxels set to exactly 0, `transit_time` uniform 0.3..2.5 with a few voxels set to exactly `signal_time` (the `t == dt` edge) and one to 1000 (the CSF sentinel), `t1_tissue` uniform 0.8..3.0 with one exact 0, `m0` uniform 50..90. Run `GkmFilter` with `NumpyImageContainer(image=arr.astype(np.float64))` per image input and the scalars as plain floats; write, per case, the header line, the scalar line, then `perfusion_rate`, `transit_time`, `t1_tissue`, `m0`, `delta_m` as space-separated `repr(float)` values. The header is:

```
# simasl GkmFilter fixtures -- seed 20260923, 10 cases, inputs float64
# generated by tools/gen_gkm_fixtures.py; do not edit by hand
```

- [ ] **Step 3: Generate, then write the fixture test**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/aslscan
micromamba run -n simasl python tools/gen_gkm_fixtures.py && head -3 tests/fixtures/gkm.txt
```

The test parses the file with `split_whitespace` and `parse::<f64>` (no serde), reads the case scalars into a `Kinetic`, and asserts `|got - want| <= 1e-9 * max(|want|, 1e-300) + 1e-300` for every voxel. Where `want == 0` exactly the Rust must also be exactly `0.0`, because those zeros come from masks, not arithmetic. The test fails, not skips, if the fixture is absent: it is committed.

- [ ] **Step 4: Implement**

Per voxel, branch first, then compute only the branch's expression:

```rust
    let f = f_ml_100g_min / 6000.0;
    let m0b = if k.lambda != 0.0 { m0 / k.lambda } else { 0.0 };
    let flow_over_lambda = if k.lambda != 0.0 { f / k.lambda } else { 0.0 };
    let one_over_t1t = if t1t != 0.0 { 1.0 / t1t } else { 0.0 };
    let denom = one_over_t1t + flow_over_lambda;
    let t1p = if denom != 0.0 { 1.0 / denom } else { 0.0 };
    // simasl: not_arrived = 0 < t <= dt; arriving = dt < t < dt + tau; arrived = t >= dt + tau.
    // The array starts at zero and only the arriving/arrived masks write into it.
    let arriving = dt < t && t < dt + k.tau;
    let arrived = t >= dt + k.tau;
    if !(arriving || arrived) { return 0.0; }
```

then the PASL and (P)CASL expressions exactly as the spec writes them, with `div0(n, d) = if d != 0.0 { n / d } else { 0.0 }` for every `np.divide(where=)` site and the two different `t1b` tests (`> 0.0` for PASL, `!= 0.0` for CASL/PCASL) reproduced verbatim with a comment naming `gkm_filter.py:205` and `:252`.

Note `t1p` can be `0.0` (zero T1 tissue with zero flow); then `q_ss` uses `div0(.., t1p) = 0` so `exp(-0) = 1` and `q = 0`, which is what simasl's `np.divide(where=t1_prime != 0)` gives. Do not "fix" this.

- [ ] **Step 5: Test and commit**

```bash
cargo test kinetic 2>&1 | tail -5
git add -A && git commit -q -m "feat: kinetic — Buxton GKM as simasl computes it, fixture-diffed at 1e-9

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: mrsignal

The spin-echo steady state without its transverse factor, plus the `ge`/`ir` rejection.

**Files:**
- Create: `src/mrsignal.rs`, `tools/gen_mrsignal_fixtures.py`, `tests/fixtures/mrsignal.txt`

**Interfaces:**
- Produces:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Contrast { SpinEcho }

/// Parse the overlay's `acq_contrast`. Only "se" is accepted in P1; "ge" and "ir" are rejected
/// with an error naming the P1 spin-echo restriction and the sub-project each waits for.
pub fn parse_contrast(s: &str) -> Result<Contrast, String>

/// Transverse magnetization immediately after the 90-degree excitation of a spin-echo sequence
/// with repetition time `tr` (s): `m0 * (1 - exp(-tr/t1))`, with simasl's zero-T1 guard
/// (`np.divide(where=t1 != 0)` makes the exponent 0, so the signal is 0). NO transverse
/// relaxation: that is the acquisition stage's (spec: division of labor).
pub fn tissue_se(m0: f64, t1: f64, tr: f64) -> f64

/// The blood compartment: `delta_m` times the flip-angle factor, which for spin echo is 1.
pub fn blood_se(delta_m: f64) -> f64
```

- [ ] **Step 1: Tests**

- `se_closed_form`: `tissue_se(74.622, 1.33, 4.0) == 74.622 * (1.0 - (-4.0f64 / 1.33).exp())` to `1e-15` relative.
- `zero_t1_gives_zero_signal_not_m0`: `tissue_se(74.622, 0.0, 4.0) == 0.0` (exp(0) = 1, so `1 - 1`), matching `mri_signal_filter.py:170-173`.
- `ge_and_ir_are_rejected_naming_the_restriction`: `parse_contrast("ge")` and `("ir")` are `Err` whose text contains `"spin-echo"` and, respectively, `"P5"` and `"P3"`. `"SE"` (case) is accepted, as simasl lower-cases.
- `blood_is_passthrough_in_p1`: `blood_se(x) == x`.
- The fixture test, below.

- [ ] **Step 2: Fixture generator**

`tools/gen_mrsignal_fixtures.py`: seeded `(6, 6, 2)` inputs `t1` (0.5..4.0, one exact 0), `t2` (0.03..0.3), `t2_star` (0.02..0.2), `m0` (40..100), `mag_enc` = 0, `acq_contrast = "se"`, `echo_time` in {0.010, 0.030}, `repetition_time` in {2.0, 4.0, 10.0}. Run `MriSignalFilter`; then **divide the output by `exp(-echo_time / t2)`** voxel-wise (with `t2 == 0` giving factor 1, simasl's guard), asserting first that the factor is finite and nonzero everywhere (`t2 >= 0.03` guarantees it; the assertion is there for the day someone lowers the range). Write inputs and the divided output as in Task 2. Header records `inputs float64, transverse factor exp(-TE/T2) divided out`.

- [ ] **Step 3: Generate, test, commit**

```bash
micromamba run -n simasl python tools/gen_mrsignal_fixtures.py
cargo test mrsignal 2>&1 | tail -5
git add -A && git commit -q -m "feat: mrsignal — spin-echo excitation magnetization, TE factor left to the acquisition

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

The fixture agreement is `1e-12` relative here: the only arithmetic is one `exp` and one subtraction, and the division-out is done in float64 by the generator.

---

### Task 4: protocol

`asl.json` + `_aslcontext.tsv` + optional TOML overlay -> `Protocol`, and `Protocol -> Acquisition`. Feature `io`.

**Files:**
- Create: `src/protocol.rs`, `tools/fetch_protocol_fixtures.sh`, `tests/fixtures/protocols/*`

**Interfaces:**
- Produces:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowKind { M0scan, Control, Label, Deltam }   // `cbf` is rejected at parse time

#[derive(Debug, Clone)]
pub struct Row {
    pub kind: RowKind,
    /// Signal time `t` for the kinetic model (s), before the per-slice offset: PLD + tau for
    /// (P)CASL, PLD for PASL. Zero for m0scan rows.
    pub t: f64,
    /// Bolus duration for this row (s). Zero for m0scan rows.
    pub tau: f64,
    /// Repetition time this row's tissue signal is evaluated at (s).
    pub tr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum M0Type { Included, Separate, Estimate, Absent }

/// Where a resolved value came from, for the output sidecar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Source { Sidecar, Phantom, Overlay, Default }

#[derive(Debug, Clone)]
pub struct Protocol {
    pub label_type: LabelType,
    pub rows: Vec<Row>,
    pub m0_type: M0Type,
    /// Per acquired slice, in data z order after SliceEncodingDirection, minus its minimum (s).
    pub slice_offsets: Vec<f64>,
    pub field_strength: f64,
    pub voxel_size_mm: [f64; 3],
    pub matrix_override: Option<[usize; 2]>,
    pub reverse_phase: bool,
    pub t_echo_ms: f64,
    pub total_readout_time_s: f64,
    pub accel: usize,
    pub mb: usize,
    pub alpha: (f64, Source), pub lambda: (f64, Source), pub t1b: (f64, Source),
    pub t2_blood_s: (f64, Source),
    pub contrast: Contrast,
    pub oversample: usize,
    pub m0_repetition_time_s: Option<f64>,
    pub seed: u64,
    pub overlay_acq: OverlayAcq,        // the pass-through Acquisition knobs
    pub input_sidecar: serde_json::Value, // echoed verbatim into the output sidecar
}

pub fn load(asl_json: &Path, aslcontext_tsv: &Path, overlay: Option<&Path>,
            phantom_params: Option<&PhantomParams>) -> Result<Protocol, String>

/// The kinetic constants for a row (alpha, lambda, t1b, tau, label type).
impl Protocol { pub fn kinetic(&self, row: &Row) -> Kinetic }

/// `mrsim_acq::kspace::Acquisition` for an acquired matrix `[nx, ny]`, with the trf > 0 check
/// run here so the failure names EchoTime and TotalReadoutTime, not t_echo and t_line.
impl Protocol { pub fn acquisition(&self, nx: usize, ny: usize) -> Result<Acquisition, String> }
```

`PhantomParams` is `phantom.json`'s block (Task 5 defines the struct; Task 4 defines it here as a plain `{lambda, t1b, field_strength}` and Task 5 reuses it).

**Precedence** (spec): overlay > sidecar > phantom > default, where "sidecar" applies only to `LabelingEfficiency`. `t2_blood` and the acquisition knobs have no sidecar or phantom source. `Source` records which won.

**Overlay schema** (TOML), every key optional:

```toml
seed = 20260923
[kinetic]
label_efficiency = 0.85
lambda_blood_brain = 0.9
t1_arterial_blood = 1.65
[signal]
acq_contrast = "se"
t2_blood = 0.165            # s
[acquisition]
oversample = 2
matrix = [64, 64]           # in-plane override
acs_lines = 24
ghost_offset = 0.0
n_spikes = 0
spike_amplitude = 1.0
n_coils = 1
t_inhom = 50.0              # ms, the fallback the T2' maps override
window = "none"             # none | hann:<alpha> | fermi:<radius>,<width>
partial_fourier = 1.0
pf_mode = "fiberfox"        # fiberfox | contiguous
eddy_strength = 0.0
eddy_quad = 0.0
eddy_phase = 0.0
eddy_tau = 70.0
noise_variance = 0.0
signal_scale = 100.0
[m0]
repetition_time = 8.0       # s, required when M0Type == "Separate"
```

- [ ] **Step 1: Fetch real sidecars**

`tools/fetch_protocol_fixtures.sh` pulls, from `https://raw.githubusercontent.com/bids-standard/bids-examples/master/`, the `perf/*_asl.json` and `*_aslcontext.tsv` of `asl001` (PCASL, `M0Type: Separate`), `asl002` (PCASL, `Included`), `asl003` (PCASL, multi-PLD? check), `asl004` (PASL), `asl005` (Look-Locker, used only to assert rejection). Record the commit SHA fetched in `tests/fixtures/protocols/SOURCES.md`. The exact subject paths are looked up at fetch time (`ls` the example via the GitHub API or the repo checkout); do not hard-code them in this plan.

Hand-written cases beside them: `pld_array_with_m0_zero`, `pld_array_wrong_length`, `pld_array_nonzero_m0scan`, `pasl_no_cutoff_flag`, `pasl_q2tips`, `te_too_short`, `pe_j`, `pe_j_minus`, `pe_i`, `mracq_3d`, `no_slice_timing`, `slice_encoding_reversed`, `bg_suppression_true`, `cbf_row`, `overlay_precedence` (+ its `.toml`), `m0_separate_no_tr`.

- [ ] **Step 2: Tests (feature `io`)**

One test per case above, plus:

- `pasl_signal_time_is_the_pld_itself`: asl004-style PASL with `PostLabelingDelay 1.8`, `BolusCutOffDelayTime 0.7`: every label row has `t == 1.8` and `tau == 0.7`; nothing added.
- `pcasl_signal_time_adds_the_labeling_duration`: `PLD 1.8, LabelingDuration 1.8` -> `t == 3.6`.
- `array_timing_expands_per_row`: `PostLabelingDelay [0, 1.0, 1.0, 2.0, 2.0]` with rows `m0scan control label control label`: rows carry 0, 1, 1, 2, 2 (+tau), and `RepetitionTimePreparation [8, 4, 4, 4, 4]` gives the m0scan row `tr 8.0`.
- `acquisition_derivation`: `EchoTime 0.012`, `TotalReadoutTime 0.032`, matrix `[64, 64]` -> `t_echo 12.0`, `t_line 0.5`; with `partial_fourier 1.0` this is the spec's negative-`trf` example and `acquisition()` errs naming `EchoTime` and `TotalReadoutTime`; with `TotalReadoutTime 0.016` it passes.
- `pe_direction_sign`: `"j-"` -> `reverse_phase == false`, `"j"` -> `true`, `"i"` -> error naming the y-axis restriction.
- `slice_encoding_direction_reverses_offsets`: `SliceTiming [0, 0.05, 0.10]` with `"k-"` gives `slice_offsets [0.10, 0.05, 0]`.
- `defaults_are_recorded`: with no overlay and no `LabelingEfficiency`, `alpha == (0.85, Default)` for PCASL and `(0.98, Default)` for PASL; with a phantom params block, `t1b == (1.65, Phantom)`; with the overlay, `(x, Overlay)`.

- [ ] **Step 3: Implement**

`serde_json::Value` for the sidecar (the field set is open-ended and is echoed back), typed extraction helpers `num(&v, "EchoTime")`, `num_or_array`, `string`, `bool`. The TSV is two-column-free: one `volume_type` header line then one token per line; anything else is an error. The `Acquisition` builder maps the overlay fields one-to-one and computes `t_line = total_readout_time_s * 1000.0 / ny as f64`, then calls `mrsim_acq::kspace::validate_acquisition_timing(&acq, nx, ny)` and rewrites its error into BIDS terms.

- [ ] **Step 4: Test and commit**

```bash
cargo test --features io protocol 2>&1 | tail -5
git add -A && git commit -q -m "feat: protocol — BIDS ASL sidecar, aslcontext, TOML overlay -> Protocol and Acquisition

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: phantom

**Files:**
- Create: `src/phantom.rs`

**Interfaces:**
- Produces:

```rust
pub struct Phantom {
    pub grid: mrsim_acq::grid::Grid,          // dims + affine of the phantom maps
    pub perfusion: Vec<f32>,                  // ml/100g/min
    pub att: Vec<f32>, pub t1: Vec<f32>,      // s
    pub t2: Vec<f32>, pub t2star: Vec<f32>,   // s (as loaded; ms happen in `relaxation()`)
    pub m0: Vec<f32>,
    pub dseg: Vec<i32>,
    pub fieldmap: Option<Vec<f32>>,           // Hz
    pub labels: Vec<(i32, String)>,           // foreground labels, ascending
    pub params: Option<PhantomParams>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum T2Mode { Auto, Class, Voxel }

/// Resolved relaxation for the acquisition stage, in MILLISECONDS.
pub enum Relaxation {
    /// One (T2, T2') per foreground label, in `labels` order.
    Class { t2_ms: Vec<f32>, t2p_ms: Vec<f32> },
    /// Per-voxel maps on the PHANTOM grid (resample turns them into simulation-grid maps).
    Voxel { t2_ms: Vec<f32>, t2p_ms: Vec<f32> },
}

pub fn load(dir: &Path) -> Result<Phantom, String>
impl Phantom {
    /// The spec's T2' derivation table, per voxel: INFINITY in background, error on a foreground
    /// zero/negative, INFINITY where T2* >= T2, else 1/(1/T2* - 1/T2). Milliseconds.
    pub fn t2prime_ms(&self) -> Result<Vec<f32>, String>;
    /// The constancy test (bitwise T2 and derived T2' per label) and mode resolution.
    pub fn relaxation(&self, mode: T2Mode) -> Result<(Relaxation, T2Mode), String>;
}
```

- [ ] **Step 1: Tests (feature `io`, on `tests/fixtures/phantom-crop`)**

- `loads_the_crop_with_three_labels`: dims `[24, 24, 6]`, labels `[(1, grey_matter), (2, white_matter), (3, csf)]`, params `Some(lambda 0.9, t1b 1.65, B0 3)`, fieldmap `None`.
- `unit_strings_are_checked`: copy the crop to a temp dir, edit `att.json` to `"Units": "ms"` -> error naming `att` and `ms`.
- `missing_map_is_an_error`: delete `T2starmap.nii.gz` -> error naming it.
- `grid_mismatch_is_an_error`: replace `M0map.nii.gz` with a `[24, 24, 5]` volume -> error naming `M0map`.
- `t2prime_derivation`: on the crop, every background voxel is `INFINITY`; every GM voxel equals `1/(1/0.066 - 1/0.08) * 1000` ms; WM `1/(1/0.053 - 1/0.11)`; CSF `1/(1/0.2 - 1/0.3)`. Then perturb: set one WM voxel's T2\* to `0.12` (> T2 0.11) -> that voxel is `INFINITY`; set one to `0.0` -> error naming label 2 and the voxel index.
- `auto_resolves_to_class_on_the_crop`: `relaxation(Auto)` -> `(Class{..}, Class)` with `t2_ms == [80, 110, 300]`.
- `one_perturbed_t2_voxel_falls_back_to_voxel_and_fails_class`: set one GM T2 voxel to `0.0801` -> `Auto` gives `Voxel`, `Class` errs naming label 1 and the voxel.
- `t2star_perturbation_that_keeps_t2prime_constant_is_still_class`: set one CSF T2\* voxel from `0.2` to `0.35` (both `>= T2 = 0.3`... no: `0.2 < 0.3`). Use WM: `0.053 -> 0.12`, both give... `0.053` gives finite T2', `0.12 >= 0.11` gives INFINITY, so that is NOT constant. Pick a label where both values exceed T2: none in the crop. So this test constructs its own tiny phantom with `T2 = 0.1`, `T2* in {0.15, 0.2}` in one label -> derived T2' is INFINITY throughout -> `Class`. Write that phantom in the test with the writer from `mrsim_acq::io` and JSON sidecars by hand.
- `t1_perturbation_does_not_affect_mode`: set one GM T1 voxel to `1.5` -> still `Class`.
- `field_strength_mismatch_is_checked_by_the_caller`: `params.field_strength` is exposed; the check lives in `series` (Task 7) and is tested there.

- [ ] **Step 2: Implement**

`load` reads each map with `mrsim_acq::io::load_volume`, checks `dims` and the affine (to `1e-6`) against the first map, reads each `.json` with `serde_json` and checks `Units` against the contract table, reads `dseg.json`'s `LabelMap`, and `phantom.json` if present. Foreground validation per the spec's table: T1, T2, T2\*, M0 strictly positive where `dseg > 0`, all maps finite everywhere; background M0 must be zero. `t2prime_ms` is the table, `relaxation` is the bitwise constancy test over T2 and derived T2'.

- [ ] **Step 3: Test and commit**

```bash
cargo test --features io phantom 2>&1 | tail -5
git add -A && git commit -q -m "feat: phantom — BIDS-derivatives maps, T2' derivation, class/voxel resolution

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: resample

Separable overlap-weighted box averaging between axis-aligned grids sharing an origin corner. Pure std.

**Files:**
- Create: `src/resample.rs`

**Interfaces:**
- Produces:

```rust
/// Per-axis overlap weights: for each target cell, the source cells it overlaps and the
/// fraction of the target cell each covers. Both grids start at the same corner; the target
/// may extend past the source (padding), in which case the uncovered fraction is simply
/// absent and `weights[j]` sums to less than 1 -- a partially-covered edge voxel is a partial
/// voxel, not a rescaled one.
pub struct AxisWeights { pub per_target: Vec<Vec<(usize, f64)>> }
pub fn axis_weights(n_src: usize, d_src: f64, n_dst: usize, d_dst: f64) -> AxisWeights

pub struct Resampler { pub x: AxisWeights, pub y: AxisWeights, pub z: AxisWeights,
                       pub src_dims: [usize; 3], pub dst_dims: [usize; 3] }
impl Resampler {
    pub fn new(src_dims: [usize; 3], src_vox: [f64; 3], dst_dims: [usize; 3], dst_vox: [f64; 3]) -> Self;
    /// Volume-weighted mean (sum of w*v over the overlapped source cells).
    pub fn mean(&self, src: &[f32]) -> Vec<f32>;
    /// Mean of `src` restricted to voxels where `mask` is true; 0 where none. (`att` over perfused.)
    pub fn masked_mean(&self, src: &[f32], mask: &[bool]) -> Vec<f32>;
    /// Weighted mean of RATES 1/src with weights `w`, inverted back; INFINITY where the weighted
    /// rate (or total weight) is zero. Times in, times out. (`voxel`-mode T2/T2' maps.)
    pub fn rate_mean(&self, src_time: &[f32], w: &[f32]) -> Vec<f32>;
    /// Majority vote over overlapped labels, ties to the lower label. (`dseg`.)
    pub fn majority(&self, src: &[i32]) -> Vec<i32>;
    /// The source z-cells (with weights) that target slice `z` overlaps. (Per-slice timing.)
    pub fn z_slab(&self, z: usize) -> &[(usize, f64)];
}

/// The acquisition grid for a phantom FOV: `ceil(extent / voxel)` per axis, corner-aligned,
/// with the affine's origin shifted so voxel centres are at corner + (j + 0.5) * voxel.
pub fn acquisition_grid(phantom: &Grid, voxel_mm: [f64; 3], matrix_override: Option<[usize; 2]>) -> Grid
```

The phantom grid is required to be axis-aligned (its affine's off-diagonal linear part must be zero to `1e-9`); an oblique phantom is an error naming the offending entry, because the spec's box average is defined on axis-aligned grids and silently rotating it would be wrong.

- [ ] **Step 1: Tests**

- `weights_tile_exactly_when_grids_nest`: `axis_weights(8, 1.0, 4, 2.0)`: target j = {(2j, 0.5), (2j+1, 0.5)}.
- `weights_handle_non_integer_ratios`: `axis_weights(3, 1.0, 2, 1.5)`: target 0 = {(0, 2/3), (1, 1/3)}, target 1 = {(1, 1/3), (2, 2/3)}, each summing to 1 to `1e-12`.
- `padding_gives_partial_last_voxel`: `axis_weights(5, 1.0, 3, 2.0)`: target 2 = {(4, 0.5)} only.
- `constant_stays_constant`: a constant 3D image resamples to the constant wherever coverage is full.
- `mass_is_conserved`: `sum(dst * dst_voxel_volume) == sum(src * src_voxel_volume)` to `1e-6` relative when the grids tile the same extent.
- `boundary_voxel_is_the_weighted_mean_of_signals_not_of_parameters`: two source cells with `m0 = 1`, T1 `1.0` and `3.0`, TR 4: the target cell's `mean(tissue_se per cell)` is `0.5*(1-e^-4) + 0.5*(1-e^-4/3)`, and it differs from `tissue_se(1, 2.0, 4)` by more than `1e-3` — the test asserts both numbers, and the difference.
- `rate_mean_is_infinity_for_zero_weight_and_matches_hand_value`.
- `majority_breaks_ties_low`.
- `acquisition_grid_covers_the_fov`: phantom `[24, 24, 6]` at 1 mm, voxel `[3, 3, 3]` -> dims `[8, 8, 2]`; voxel `[3.5, 3.5, 3]` -> `[7, 7, 2]` (ceil); centre of dst voxel 0 is at `corner + 1.75`.

- [ ] **Step 2: Implement, test, commit**

```bash
cargo test resample 2>&1 | tail -5
git add -A && git commit -q -m "feat: resample — separable box-overlap averaging between corner-aligned grids

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: series

The orchestration: rows -> compartment volumes on the simulation grid -> one `simulate_acquisition_oversampled` call (+ one for a separate M0). Feature `io`.

**Files:**
- Create: `src/series.rs`

**Interfaces:**
- Produces:

```rust
pub struct SeriesOutput {
    pub acq_grid: Grid, pub sim_grid: Grid,
    pub n_volumes: usize,
    pub mag: Vec<f32>, pub phase: Vec<f32>,          // voxel-major interleaved, acquisition grid
    pub m0: Option<(Vec<f32>, Vec<f32>)>,            // the separate scan, when M0Type == Separate
    pub mode: T2Mode, pub labels: Vec<(i32, String)>,
    pub ground_truth: GroundTruth,                   // acquisition-grid maps, see bids
    pub seeds: (u64, Option<u64>),
}

pub fn simulate(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel) -> Result<SeriesOutput, String>
```

Inside, per the spec:

1. Field-strength check (`phantom.json` vs sidecar), `BackgroundSuppression: true` rejection naming P3, `cbf` rows already rejected by `protocol`.
2. `acq_grid = acquisition_grid(..)`, `sim_grid = hires_grid(&acq_grid, p.oversample)`, `Resampler` phantom -> sim and phantom -> acq.
3. `Acquisition` from `p.acquisition(nx, ny)`; `do_distortions = fieldmap.is_some()`; the fieldmap resampled (mean) to the sim grid, zeros if absent.
4. Relaxation: `phantom.relaxation(mode)`; in `Class`, compartments are `2K`: `T2Volume::Uniform(t2_ms[i])` for tissue `i`, `Uniform(t2_blood_ms)` for blood `K+i`, `t_inhom` `Uniform(t2p_ms[i])` for both. In `Voxel`, `K = 1`, tissue T2 map and shared T2' map via `rate_mean` with `m0` weights, blood T2 uniform.
5. Tissue signal per distinct `tr` among rows: `tissue_se` per phantom voxel, split by label (class) or whole (voxel), resampled once per distinct `tr` and cached.
6. Per row: `t = row.t`; for each acquired slice `z`, `kinetic::delta_m` over the phantom slab `z_slab(z)` at `t + slice_offsets[z]`, signed per the row table (`label` -> `-delta_m`, `deltam` -> `+delta_m`, else 0), split by label, box-averaged into that acquired slice's sim-grid rows. The phantom-grid `delta_m` at the row's own `t` (first slice) is also kept for the ground truth.
7. Assemble `images: Vec<Vec<f32>>` (one per compartment, `nvox_sim * n_volumes`, voxel-major) and call the entry point once with `eddy_drive = vec![None; n]`, `prep_drive = vec![None; n]`, `PhaseModel { prep: None, .. }`, `noise_sigma: None`, `eddy_trace: None`.
8. If `M0Type::Separate`: one more call with `n_volumes = 1`, tissue at `m0_repetition_time`, blood zero, `seed ^ 0x4D30_5343_414E`.

- [ ] **Step 1: Tests (feature `io`, on the crop, tiny protocol from `tests/fixtures/protocols/`)**

- `class_and_voxel_agree_on_a_homogeneous_grid`: protocol with `AcquisitionVoxelSize [1, 1, 1]`, `oversample 1`, no noise: `simulate(.., Class)` and `simulate(.., Voxel)` agree to `1e-5` relative on every nonzero voxel of `mag`.
- `class_vs_voxel_boundary_discrepancy_is_tracked`: `AcquisitionVoxelSize [3, 3, 3]`, `oversample 2`: compute the max relative discrepancy and assert it is below `0.05`, printing it. This is the characterization number the spec asks to track; the `0.05` is a tripwire, not a claim.
- `noise_is_uncorrelated_across_volumes_and_with_the_separate_m0`: `noise_variance = 1.0`, tissue and blood zero (an empty phantom crop: set `m0` to zero everywhere in a copy), rows `control control control`: the pairwise Pearson correlation of volumes' `re`-parts is `< 0.1` in magnitude for each pair, and the separate M0 versus volume 0 likewise. Volumes are recovered as `mag * cos(phase)`.
- `series_rejects_a_field_strength_mismatch_and_background_suppression`.
- `label_rows_carry_negative_delta_m`: with noise off and `signal_scale 1`, one control and one label row: `|I_C| - |I_L| > 0` in perfused GM voxels (a coarse sign check; the exact property is Task 8's).

- [ ] **Step 2: Implement, test, commit**

```bash
cargo test --features io series 2>&1 | tail -5
git add -A && git commit -q -m "feat: series — row semantics, class split, one acquisition call, separate M0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: bids output and the CLI

**Files:**
- Create: `src/bids.rs`, `src/bin/aslscan.rs`

**Interfaces:**
- `bids::names(sub, ses) -> Names` (`sub-XX[_ses-YY]_...` stems; the `perf/` directory).
- `bids::write_dataset(out_root, sub, ses, &Protocol, &SeriesOutput) -> Result<(), String>` writes `dataset_description.json` (`"BIDSVersion": "1.10.0"`, `"DatasetType": "raw"`, `"GeneratedBy": [{"Name": "aslscan", "Version": ..}]`), `.bidsignore` (`ground-truth/`), `sub-XX/perf/sub-XX_part-mag_asl.nii.gz` + `part-phase` (through `mrsim_acq::io::write_complex_4d(.., "asl", ..)`), `sub-XX_asl.json` (the input sidecar's fields echoed, then `"AslscanSimulation": {..}` with every resolved value and its `Source`, the seed(s), the T2 mode, the label order, `Oversample`, the acquisition knobs, `FieldmapPresent`), `sub-XX_aslcontext.tsv`, and when `Separate`, `sub-XX_m0scan.nii.gz` + `.json` (`RepetitionTimePreparation` from the overlay). Ground truth under `sub-XX/perf/ground-truth/`: `deltam` (4D, one per row), `perfusion`, `att` (masked mean), `T1map`, `T2map`, `M0map` (means), `dseg` (majority), `desc-acq_T2map` / `desc-acq_T2primemap` in `voxel` mode. Each with a JSON carrying `Units` and `"Resampling": "volume-weighted mean" | "masked mean over perfused voxels" | "majority vote"`.

  The two BIDS part sidecars written by `write_complex_4d` carry `"Manufacturer": "TRXScan"` today (P0 kept that byte-exact). `bids` overwrites both part sidecars after the call with the full ASL sidecar content plus `"ImageComparison"`, so the shared writer's DWI-flavoured stub is never what ships. That is a documented workaround; making `Manufacturer` a `SidecarInfo` field is a one-line `mrsim-acq` change to propose after P1, not to make here.

- CLI: `aslscan --asl-json X --aslcontext Y --phantom DIR [--overlay T.toml] [--t2-mode auto|class|voxel] [--sub SUB] [--ses SES] --out ROOT [--seed N]`, printing the resolved protocol summary and timings.

- [ ] **Step 1: Tests**

- `names_follow_bids` (pure std): `sub-01`, `ses-02` -> `sub-01/ses-02/perf/sub-01_ses-02_part-mag_asl.nii.gz`; no session -> `sub-01/perf/...`.
- `aslcontext_is_written_in_row_order` (pure std, string output).
- `sidecar_records_every_default` (feature `io`): run the PCASL crop protocol with no overlay and read back `AslscanSimulation.LabelingEfficiency.Source == "Default"` etc.

- [ ] **Step 2: Implement, build the binary, run it on the crop**

```bash
cargo test --features io bids 2>&1 | tail -5
cargo run --features cli --bin aslscan -- --asl-json tests/fixtures/protocols/pcasl_single/asl.json \
  --aslcontext tests/fixtures/protocols/pcasl_single/aslcontext.tsv --phantom tests/fixtures/phantom-crop \
  --sub 01 --out work/run-crop
find work/run-crop -type f | sort
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -q -m "feat: bids writer and aslscan CLI

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end runs, the linearity property, validation, acceptance

**Files:**
- Create: `tests/end_to_end.rs` (feature `io`), `tests/fixtures/protocols/{pcasl_single,pasl_cutoff,pcasl_multipld}/`

- [ ] **Step 1: The three acceptance runs on the full phantom**

With `work/phantom-3t` (Task 1) and `AcquisitionVoxelSize [3.5, 3.5, 5]`, `oversample 2`, `matrix [64, 64]`... the phantom FOV is 197 x 233 mm, so `[57, 67]` at 3.5 mm; use the override `[64, 68]`. Each run writes to `work/run-<name>/`:

```bash
for p in pcasl_single pasl_cutoff pcasl_multipld; do
  cargo run --release --features cli,kspace,par --bin aslscan -- \
    --asl-json tests/fixtures/protocols/$p/asl.json --aslcontext tests/fixtures/protocols/$p/aslcontext.tsv \
    --phantom work/phantom-3t --overlay tests/fixtures/protocols/$p/overlay.toml --sub 01 --out work/run-$p
done
```

Record wall time per run here; the spec asked for the `voxel`-mode cost to be measured, so run `pcasl_single` once more with `--t2-mode voxel` and record that too.

**Measured 2026-09-23** (release, `cli,kspace,par`, WSL on the full 3T phantom, acquisition 64 x 68 x 38, simulation 128 x 136 x 38, `class` mode with 6 compartments): `pcasl_single` (20 volumes) 4.0 s wall, 0.7 GB RSS; `pasl_cutoff` (10) 2.7 s; `pcasl_multipld` (16) 2.9 s. `voxel` mode on `pcasl_single`: 4.2 s. The rotor path is not the bottleneck at ASL matrix sizes; the spec's fallback to a time-segmented NUFFT is not needed for P1. The tracked `class`-vs-`voxel` boundary discrepancy on the crop at 3 mm / `oversample 2` is 4.9e-3 of peak.

- [ ] **Step 2: bids-validator on each**

```bash
export PATH=$HOME/.nvm/versions/node/v22.22.2/bin:$PATH
for p in pcasl_single pasl_cutoff pcasl_multipld; do deno run -A jsr:@bids/validator work/run-$p --ignoreWarnings; done
```

Expected: no errors. Any *error* is a defect in `bids` to fix before continuing. Criterion 3.

Two errors the first run produced, both fixed in `bids`: bids-validator 3.0.2 checks required keys on each `part-` file **without merging the inheritance-level `_asl.json` in**, so a `part-phase` sidecar carrying only `Units` fails `SIDECAR_KEY_REQUIRED` — both part sidecars are now written complete, with `_asl.json` kept as well; and `.bidsignore` needs `**/ground-truth` with no trailing slash (the `**/ground-truth/`, `**/ground-truth/**` and rooted forms all left the directory flagged). A third error appeared once the part sidecars were complete: the inheritance-level `_asl.json` is `SIDECAR_WITHOUT_DATAFILE`, because with `part-` entities there is no `_asl.nii.gz`; it is no longer written and the spec's output contract was corrected. The remaining warnings are `NIFTI_UNIT`/`NIFTI_PIXDIM` from the shared writer's header (`header_for_grid` leaves `pixdim[4]` and the time unit unset), which is an `mrsim-acq` change to propose rather than make here, plus recommended-key warnings.

- [ ] **Step 3: The linearity property (criterion 5)**

`tests/end_to_end.rs::blood_compartment_linearity` on the crop, `AcquisitionVoxelSize [2, 2, 3]`, `oversample 2`, `accel 1`, `n_spikes 0`, `noise_variance 0`, `signal_scale 100`, one seed: three `series::simulate` calls whose row lists are `control`, `label`, and a `deltam` row — that is exactly runs C, L, B. Reconstruct `re + i im` from `mag`/`phase` of each, assert `|I_C - I_L - I_B| <= 1e-6 * max(max|I_B|, max|I_C|) + 1e-5 * |I_B|` at every voxel. Then the three negative controls the spec lists, each as its own test: a flipped sign in the label row, a control/label swap, and the blood compartment wired to index 0 — each must FAIL the identity by a margin of at least `1e-3 * max|I_C|` somewhere, so the tolerance is proven not to be slack. Implement the negatives by exposing a `series::simulate_with(.., RowOverride)` test hook rather than by copy-pasting the pipeline.

- [ ] **Step 4: Noise decorrelation on the real run (criterion 6)**

Already a unit test in Task 7; here, additionally, on `work/run-pcasl_single` re-run with `noise_variance 4.0`, compute the control-minus-label difference volume's variance and assert it is about twice one volume's background variance (ratio in `[1.6, 2.4]`), which is what independent noise gives and what a shared realization would make zero.

- [ ] **Step 5: Full acceptance pass**

```bash
cd /mnt/c/Users/tsalo/Documents/rust-trx/aslscan
cargo test                                   # criterion 7: pure std
cargo test --features io,kspace,par
cargo clippy --all-targets --features cli,kspace,par 2>&1 | grep -c "^warning"
```

Expected: all green, and clippy clean (new crate, no inherited warnings: zero is the bar).

- [ ] **Step 6: Commit and tag**

```bash
git add -A && git commit -q -m "test: end-to-end runs, blood-compartment linearity, BIDS validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git tag p1-complete
```

---

## Acceptance Criteria Coverage

| Spec criterion | Where it is checked |
|---|---|
| 1. PCASL single-delay from BIDS + converted phantom, complete output directory | Task 9 Step 1 (`pcasl_single`) |
| 2. PASL with bolus cutoff; multi-delay PCASL with array PLD | Task 9 Step 1 (`pasl_cutoff`, `pcasl_multipld`) |
| 3. `bids-validator` reports no errors | Task 9 Step 2 |
| 4. `ge`/`ir` rejected naming the restriction | Task 3 test; Task 4 `overlay` case |
| 5. Blood-compartment linearity | Task 9 Step 3, with three negative controls |
| 6. Noise uncorrelated across volumes and with the separate M0 | Task 7 unit test; Task 9 Step 4 |
| 7. `cargo test` passes in the pure-std default build | Every task; Task 9 Step 5 |

## Codex adversarial review (2026-09-24)

Fifteen findings on `p1-complete`, all accepted and fixed in the follow-up commit: output sidecar
standard keys now carry the simulated values (originals under `InputValuesReplaced`); one-element
timing arrays are arrays; empty `EchoTime` arrays, negative or non-finite timing/overlay values,
`LookLocker: true`, `VascularCrushing: true`, an absent `BackgroundSuppression`, a missing
`BolusCutOffTechnique`/`M0Estimate`, more than two cutoff times, and multiband groups of the wrong
size are all rejected; non-finite fieldmaps and non-integral or out-of-int16 labels are rejected;
the M0 sidecar carries `SliceEncodingDirection` and the readout keys; the converter refuses
negative crop bounds; `simulate_with`/`RowOverride` are behind `cfg(test)` or the `test-hooks`
feature; the homogeneous-grid test is per-voxel relative with an absolute floor; blood T2's
conversion moved to `protocol`. Nothing numerical was found wrong in `kinetic`, `mrsignal`,
`resample` or `series`.

## What P1 does not do

Background suppression, IR and GE contrasts, motion, vascular crushing, the macrovascular compartment, 3D readouts, multi-TE, Hadamard/Look-Locker, `--compat-asldro`: all later sub-projects per the spec. Phase encoding must be the second data axis. The phantom must be axis-aligned. `Manufacturer` in the part sidecars is overwritten after the fact rather than parameterized in `mrsim-acq`.
