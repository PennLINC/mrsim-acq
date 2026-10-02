# P5: gradient-echo contrast, 3D GRASE and 3D stack-of-spirals readouts — implementation plan

**Goal:** Implement the P5 addendum (`docs/specs/2026-10-01-p5-3d-readouts-ge-design.md`):
gradient-echo contrast for 2D EPI (part A), the 3D echo-train model with the GRASE readout
(part B), segmentation (part D), and, as a separate milestone, the stack-of-spirals readout
(part C), across `mrsim-acq` and `aslscan`, with TRXScan's and aslscan's existing outputs
byte-identical.

**Architecture:** Baselines first, because this is the first sub-project since P0 that changes
`mrsim-acq`: a golden record of the full forward outputs and a regression gate that pins both
repositories, each shown to detect a change before any change is made. Then `mrsim-acq` from the
inside out: the line-timing table (a refactor that must not move a bit), gradient-echo echo
formation, the EPG, the echo-train readout and its timing, and the 3D entry point. Then
`aslscan`: gradient echo end to end, the 3D protocol, the 3D series, the outputs, and the GRASE
acceptance. Then milestone C: the 2D NUFFT pair, the spiral trajectory and a feasibility
benchmark that decides whether the rest of part C proceeds, the time-segmented forward and
gridding, and the `asl001` acceptance.

**Spec:** the P5 addendum, Codex-reviewed twice (2026-10-01, 2026-10-02) before this plan; the
main spec and the P2-P4 addenda where it is silent. P4 is complete at `aslscan` tag `p4-complete`
(`1df98a2`); `mrsim-acq`'s source last changed at `054bfdf`, and its docs head is `b463dd1`;
TRXScan's consumer branch is `p0-mrsim-acq-extraction` at `130d62b`.

## Global constraints

- Everything in the P1-P4 plans' global constraints still holds (pure-std default build of both
  crates, seconds upstream and milliseconds from the acquisition boundary on, one acquisition
  call per series, no `cargo fmt`, no large data committed, every approximation named in the
  sidecar, rejections over silent defaults).
- **No P5 input, no changed byte**, in three places (addendum, "Scope"):
  1. TRXScan, in WSL, inside a worktree of branch `p0-mrsim-acq-extraction` beside `mrsim-acq`
     (so its `../mrsim-acq` path resolves to the working copy under test, and TRXScan's own
     `main` checkout is not disturbed; the script calls bare `cargo run`, so the working
     directory selects the crate):
     `git -C TRXScan worktree add ../TRXScan-p0 p0-mrsim-acq-extraction` once, then
     `cd TRXScan-p0 && cargo metadata --format-version 1` (the resolved `mrsim-acq` manifest must
     be the working copy under test), then
     `tools/run_p0_baseline.sh /abs/out` and
     `diff /abs/out/checksums.txt tests/fixtures/p0_baseline/checksums.txt` (5 configurations x 2
     feature sets, about 90 s). The script only writes checksums; the `diff` is the gate.
  2. `mrsim-acq`: the bit-pinned tests unchanged, and the golden baseline of Task 0.
  3. `aslscan`: `tools/regress_identity.sh` against `aslscan` `p4-complete` with `mrsim-acq`
     `054bfdf`, both feature sets, P4 combinations included.
  Each is run at the end of every `mrsim-acq` task, not only at the end.
- **New behavior only behind new non-default values; the default path calls the existing code.**
  Where code is generalized, the default constructs the same values in the same order, and a test
  pins the equality bit for bit.
- **Nothing is ignored**: every P5 input belonging to a readout or contrast that is off is an
  error naming it (the addendum lists them per part).
- **Commits**: `mrsim-acq` tasks commit in `mrsim-acq`; consumer source changes commit in the
  consumer (TRXScan's branch, `aslscan`), each with `Co-Authored-By: Claude Opus 5.5
  <noreply@anthropic.com>`. Commit from Git Bash, not through nested `wsl -e bash -lc` quoting
  (apostrophes break it). Tag `p5-grase-complete` in both crates after Task 10's reviews, and
  `p5-complete` after milestone C.
- **Test commands** (the end-to-end suite is gated on `io` and `test-hooks`,
  `aslscan/tests/end_to_end.rs:5`, so a bare `cargo test` runs none of it):
  `mrsim-acq`: `cargo test` and `cargo test --features kspace,par`;
  `aslscan`: `cargo test --features io,test-hooks` and
  `cargo test --release --features cli,kspace,par,test-hooks`. Each task's checklist names the
  new tests, and the run's output is checked to contain them (a filtered `cargo test <name>`
  that reports `0 passed` is a failure).
- **Bits differ between the default and `kspace` builds** (the rustfft x-stage), so every golden
  or bit-pinned reference has one value per `cfg!(feature = "kspace")`.
- WSL can become unresponsive (`HCS_E_CONNECTION_TIMEOUT`); Windows-native cargo works for both
  crates with `CARGO_TARGET_DIR=target-win` (already excluded in `aslscan`; add the exclude in
  `mrsim-acq`). The TRXScan gate needs WSL (cmake).

## File structure

| Path | Change |
|---|---|
| `mrsim-acq/tests/fixtures/p4_baseline/` | New: golden `f32` bits of every oracle case, both feature sets (Task 0) |
| `mrsim-acq/src/kspace.rs` | `LineTiming` (with polarity) consumed by the forward; `EchoFormation`; the private per-compartment accumulator; `simulate_acquisition_3d` and its internal `f64` entry; 3D noise, spikes and seeds; the per-partition reconstruction loop |
| `mrsim-acq/src/readout.rs` | `EchoTrain`, `Readout3d::{Grase, Spiral}`, encoding order, shot chronology, line tables, train timing checks; the spiral trajectory and sampling bound (milestone C) |
| `mrsim-acq/src/epg.rs` | New, pure std: CPMG extended phase graph, `ln A_e` |
| `mrsim-acq/src/phase.rs` | Header doc: which fieldmap term applies under which echo formation |
| `mrsim-acq/src/nufft.rs` | 2D type-1 and type-2 on the existing kernel (milestone C) |
| `mrsim-acq/src/tseg.rs` | New (`kspace` feature): least-squares time segmentation with the certified bound (milestone C) |
| `mrsim-acq/src/grid_recon.rs` | New (`kspace` feature): Pipe-Menon density compensation and gridding (milestone C) |
| `TRXScan/src/bin/trxscan.rs` (branch `p0-mrsim-acq-extraction`) | `echo: EchoFormation::Spin` in the `Acquisition` literal (`:828`) |
| `aslscan/tools/regress_identity.sh` | Two revisions (aslscan, mrsim-acq), side-by-side worktrees, `cargo metadata` check, both feature sets, P4 cases, self-test |
| `aslscan/src/mrsignal.rs` | `Contrast::GradientEcho`; spoiled and simasl `tissue_ge`; `blood_ge` |
| `aslscan/src/longitudinal.rs` | The gradient-echo fixed point and the chronological propagation |
| `aslscan/src/protocol.rs` | `"ge"` inputs; `MRAcquisitionType: 3D`; `[readout]`; the 3D timing resolver and checks; refusals; per-shot `row_start`; the `echo` field in `acquisition()` |
| `aslscan/src/phantom.rs` | Class-mode eligibility including T1 for 3D below 180 degrees |
| `aslscan/src/series.rs` | GE wiring and M0; zero slice offsets in 3D; T1 maps; the extravascular group; shot gains, physio per shot and its horizon; per-shot motion; the 3D call |
| `aslscan/src/bids.rs` | GE keys; `Readout` and `EchoAmplitudes` blocks; 3D standard keys; M0 sidecar from its own resolution; 3D physio TSV schema; `pixdim[4]`; `desc-acqT1map_gt` |
| `aslscan/tools/compat_asldro.py`, `test_compat_asldro.py` | Gradient echo under compat |
| `aslscan/tests/fixtures/protocols/{p5_ge,asl005_p5,asl003_p5,asl001_p5}` | Overlays and the `asl003` derivative |
| `aslscan/tests/end_to_end.rs` | Linearity under GE, GRASE and spirals; the new negative controls |

---

## Milestone AB: gradient echo, the echo-train model, GRASE, segmentation

### Task 0: baselines and the gate, before any change

Nothing in this task changes simulation code. It exists so that every later task has something
immutable to be compared with.

- **Golden forward outputs** (`mrsim-acq`): a test-support function that runs every case of
  `restructured_forward_matches_the_literal_sum` (`kspace.rs:2721-2779`) and
  `varying_t2_map_matches_the_literal_sum` (`:2926`) through `simulate_slice` and writes the full
  complex `f32` output as little-endian bits, one file per case per feature set, to
  `tests/fixtures/p4_baseline/` (generated by an `#[ignore]` test with an environment-variable
  guard, run once in each build, committed). A non-ignored test reads them back and compares bit
  for bit. Also the full `simulate_acquisition_oversampled` output of one small multi-volume,
  multi-coil, GRAPPA, noise-on configuration.
- **`regress_identity.sh` rework** (`aslscan`), addendum "Scope":
  - Arguments `<aslscan-rev> <mrsim-acq-rev>`. Worktrees of both, side by side under
    `../p5-base/` (so `aslscan`'s `path = "../mrsim-acq"` resolves to the pinned copy), each
    verified clean and at its commit (the checks the script already makes for one worktree).
  - `cargo metadata --format-version 1` on the base: the `mrsim-acq` package's manifest path must
    be under `../p5-base/mrsim-acq`, and on the new build under the live `../mrsim-acq`; either
    wrong is an error.
  - Both `cli,kspace,par` and `cli` builds, separate target directories per side and feature set.
  - New cases: the `p4_all` fixture (every P4 part, on the crop), a `[physio]`-only case, a
    `[macrovascular]` with crushing case, a bolus-position suppression case.
  - `--self-test`: builds a third layout, `../p5-selftest/{aslscan,mrsim-acq}`, holding copies of
    the **live** `aslscan` source and of the live `mrsim-acq` with a patch applied that scales
    `signal_scale` by `1 + 1e-4` inside the forward (large enough to survive reconstruction and
    the `f32` cast; never committed; no hook in production code). The metadata check for this
    layout expects `../p5-selftest/mrsim-acq`. The self-test passes only if every case's base and
    patched runs succeed **and** at least one decompressed NIfTI differs in content; a failed
    build or run is a self-test failure, not a detected difference.
- **Run all three gates now**, at the pre-P5 heads, and record the outputs: TRXScan
  `run_p0_baseline.sh` diffs clean; the golden test passes; `regress_identity.sh p4-complete
  054bfdf` reports every case identical in both builds; `--self-test` reports a difference.

- [ ] Golden baseline files and test in `mrsim-acq`; commit `test: golden forward outputs before P5`.
- [ ] `regress_identity.sh` rework in `aslscan`; commit `tools: regress_identity pins both repositories`.
- [ ] All three gates run and recorded in Measurements.

### Task 1: `mrsim-acq` — the line-timing table

`LineTiming { t_ms, trf_ms, tread_ms, polarity }`, one entry per `ky`. `polarity` is `+1`/`-1`
for the readout direction of that line. For the 2D path a constructor
`LineTiming::from_epi(&SingleShotEpi)` calls `line_times` unchanged and sets
`polarity[ky] = if ky % 2 == 1 { -1 } else { +1 }`, which is what the ghost uses today
(`kspace.rs:846`). `build_coil_kspace` and `validate_acquisition_timing` read the table instead of
calling `line_times`; the ghost reads `polarity` instead of `ky % 2`. Every expression that
consumes a time is unchanged, character for character.

Tests: for every oracle case, `LineTiming::from_epi` equals `line_times` element-wise
`to_bits`, and the polarity equals `ky % 2`; the golden baseline, the bit-pinned tests and the
literal-sum tests pass unchanged; the three gates are clean.

- [ ] Implement and test; the three gates clean.
- [ ] Commit (`mrsim-acq`): `refactor: kspace reads a line-timing table`.

### Task 2: `mrsim-acq` — gradient-echo echo formation

`Acquisition.echo: EchoFormation { Spin, Gradient }`, `Default` is `Spin`. Under `Gradient`
(addendum part A):

- decay `exp(-trf/T2 - trf/T2')`: the `|t|` term becomes `trf` in the uniform scalar
  (`kspace.rs:762`), the map path (`:823`) and the literal oracle (`:1672`);
- static phase: `phi0 += TAU * fmap[i] * t_echo_s` where `phi0` is formed (`:585`), only when
  `do_distortions`, using `fmap`, never `rate` (the replayed eddy shear does not accrue before the
  readout); the oracle gains the same term written independently.

Under `Spin` the code path is the existing one: the `echo` test is a branch around the two new
expressions, not a rewrite of the old ones. `phase.rs:4-9` is rewritten to state both cases.

Consumers: the `Acquisition` literals at TRXScan `src/bin/trxscan.rs:828` (branch
`p0-mrsim-acq-extraction`) and `aslscan` `src/protocol.rs:1688` gain `echo: EchoFormation::Spin`.
Commit each in its own repository.

Tests:
- `restructured_forward_matches_the_literal_sum` gains `Gradient` variants of its clean,
  distortion-plus-relaxation and combined cases, `1e-10` of peak.
- On the internal `f64` k-space (an existing crate-internal accessor or a new `pub(crate)` one):
  with a uniform fieldmap `f`, no decay, the `Gradient` k-space equals the `Spin` k-space times
  `exp(i 2 pi f TE)` on every acquired sample, `1e-12` relative; with decay and no fieldmap, the
  ratio of two `TE`s' k-spaces is `exp(-dTE/T2*)` on every line, `1e-12`.
- `eddy_lin` with `Gradient`: the static term does not include the shear (a case with
  `eddy_lin` and zero fieldmap equals `Spin` exactly in phase).
- The NUFFT and rotor paths agree under `Gradient` (`constant_map_reproduces_uniform`'s pattern).
- The three gates clean (the `Spin` default must not move a bit).

- [ ] Implement and test in `mrsim-acq`; commit `feat: gradient-echo echo formation`.
- [ ] TRXScan branch: `echo: EchoFormation::Spin`; P0 gate clean; commit
  `Name the echo formation in the mrsim-acq Acquisition literal`.
- [ ] `aslscan`: the same field in `protocol.rs`; commit `chore: name the echo formation`.

### Task 3: `mrsim-acq` — `epg`

Pure std. `epg_cpmg(n_echoes, esp_ms, refocusing_deg, t1_ms, t2_ms) -> Vec<f64>` returning
`ln A_e` for `e = 1..n_echoes` (with `-inf` for an exactly zero amplitude): 90-degree excitation
about x, refocusing about y (CPMG), instantaneous pulses, relaxation of `F+`, `F-` and `Z` states
over `ESP/2` either side of each pulse, the echo amplitude the `F0` state at each echo.
`T1 = infinity` and `T2 = infinity` are allowed (no relaxation of that kind).

Tests:
- `b = 180`: the amplitude, not its logarithm, to `1e-12` relative:
  `|expm1(ln A_e - (-e ESP / T2))| <= 1e-12`, any T1 (a log-space relative tolerance would allow
  several times that amplitude error at `|ln A| ~ 6`); exact zeros, `T2 = infinity` and
  underflowing amplitudes checked separately.
- An independent isochromat simulation in the test module (rotation matrices, 2001 isochromats
  uniformly dephased over one crusher cycle, the same timings): `b = 130` and `b = 111`, T1/T2 of
  GM, WM, CSF and blood, every echo of a 30-echo train, within `1e-3` absolute of the
  normalized amplitude, the first echoes included (the pseudo-steady-state onset).
- Extremes: `T2 = 1e-3` ms gives finite `ln A_e` (very negative) and no NaN; `T2 = infinity`
  and `b = 180` gives `0`.

- [ ] Implement and test; commit `feat: extended phase graph for CPMG echo trains`.

### Task 4: `mrsim-acq` — the echo-train readout

In `readout.rs`:

- `EchoTrain { etl, esp_ms, refocusing_deg, kz_order: KzOrder, kz_segments, refocusing_time_ms }`.
- `Readout3d::Grase { ky_segments, t_line_ms, reverse_phase }` (spirals in Task 12).
- `kz_order`: `Centric` (centre, `+1`, `-1`, `+2`, ...) or `Linear`; the partition read by shot
  `(sy, sz)` at echo `e` is `kz_order[sz + kz_segments (e - 1)]`; `e_c` the echo of the centre.
- The canonical shot index `s = sy kz_segments + sz` and its inverse.
- `grase_lines(&EchoTrain, &Grase, nx, ny, nz) -> Grase3dTable`: per `(p, ky)` the shot, echo,
  `t`, `trf = e ESP + t`, polarity by acquisition index `j`, and the traversal by
  `reverse_phase` (`readout.rs:46-58`'s convention). Two timings are kept apart: the
  **within-echo** `LineTiming` for the relaxation-free 2D forward, which carries `t`, `tread` and
  polarity and depends on `ky` only (the constructor asserts exactly these three are the same in
  every partition and shot), with its `trf` field set to `t` and never read on the 3D path (the
  forward runs with relaxation off); and the full per-`(p, ky)` `trf`, held only in
  `Grase3dTable`, which the weights and the timing checks use.
- `check_grase_timing(...) -> Result<(), String>` with the addendum's four checks on actual
  intervals (block between refocusing pulses, first pulse after the excitation, last sample
  within `TR`, the suppression rule is aslscan's), each error naming the numbers and the fix.
- `esp_from_echo_time(te_ms, &table) -> f64`: `(TE - t(ky_c)) / e_c`.

Tests: every `(p, ky)` read exactly once, for every combination of `ky_segments` in `{1, 2, 4}`,
`kz_segments` in `{1, 2}`, both orders and both traversals; `t(ky)`, `tread(ky)` and polarity are
independent of partition and shot for each `ky`, while `trf` differs by `ESP` between
consecutive echoes; polarity alternates in `j` within each block; `e_c = 1` for centric with one kz
segment; the `asl003`-derivative numbers (`ny = 20`, two segments, 1 ms lines,
`TE = 11.92` ms) give `t(ky_c) = -0.5` ms and `ESP = 12.42` ms, and the block check passes with
the 2 ms reserve; the `asl003` sidecar at `ny = 64` fails it with a message naming `ESP`, the
block's extent and the fix; the `asl005` numbers (`64 x 64 x 30`, four segments, 0.2048 ms lines)
give `ESP = 13.3824` ms and a train ending at `4.2031104` s (addendum, second review).

- [ ] Implement and test; commit `feat: GRASE echo-train readout and its timing`.

### Task 5: `mrsim-acq` — `simulate_acquisition_3d` (Cartesian)

The addendum's "The acquisition stage", "Generalizing the 2D forward" and part D's carriers.

- A private variant of `build_coil_kspace` that returns per-compartment k-spaces before the sum
  and applies no relaxation (`rel = 1`, `T2'` term off), with the 2D accumulator untouched. It
  takes the `LineTiming` (time, polarity) of Task 4. In `voxel` mode it takes, per echo, a
  per-voxel log-amplitude map added inside the map path's single decay exponent
  (`ln A_e(r) - t/T2(r) - |t|/T2'(r)`).
- The z-DFT per compartment, the centred asymmetric convention; the weighted compartment sum with
  `W_c(p, ky) = exp(ln A_{e(p)}(c) - t/T2_c - |t|/T2'_c)` times `line_weights` (physio x shot
  gains, `LineWeights` indexed `(volume, shot, compartment)`); `voxel` mode runs the forward per
  echo and takes each partition from its echo's slices.
- `shot_images`: per volume, the moved images of each shot with a distinct pose; the forward runs
  once per distinct pose and keeps that pose's shots' lines.
- Spikes and noise per acquired 3D sample, seeded on `(volume, partition, coil)` with
  `seed ^ 0x3344_5245_4144`; SD `sqrt(noise_variance / (nx ny))`.
- Reconstruction: inverse z-DFT `/ nz`, then each partition through the existing per-slice path
  (GRAPPA, window, inverse FFT, Roemer).
- `pub(crate) fn simulate_acquisition_3d_complex` returning the `f64` complex image; the public
  entry casts to `f32` magnitude and phase as the 2D one does. And a `pub(crate)` observation
  point returning the weighted 3D k-space per coil **after** the shot weighting and sampling mask
  and **before** spikes, noise and reconstruction, for the tests that assert on acquired samples
  (ghosting, dropout).
- Refused inputs: eddy drives, prep drives, `noise_sigma`, `eddy_*` nonzero, and a `T1` absent
  when `refocusing_deg != 180`.

Tests (internal `f64` entry unless stated):
- 3D wiring: relaxation off, no fieldmap, no ghosting: equals the 2D output slice by slice to
  `f32` precision on the public output, every segmentation and order, multi-coil and GRAPPA.
- Ghosting, on the k-space observation point: equals an independent direct sum with polarity by
  acquisition index, both traversals, `1e-10` of peak; the same sum with polarity by `ky % 2`
  differs by more than `1e-3`.
- Distortion direction: a point off centre under a uniform fieldmap moves the documented way for
  `j` and `j-`.
- PSF: a one-partition object, uniform T2, reconstructs to the inverse z-DFT of the kz
  modulation in encoding order, `1e-9` of peak; centric and linear widths computed separately and
  both checked.
- Noise: full sampling, signal-free region, SD ratio to 2D `1/sqrt(nz)` within 3 percent
  (64 x 64 x 16, 20 seeds); GRAPPA ratio printed.
- Segmentation: relaxation off, no fieldmap, no ghosting, `NumberShots` in `{1, 2, 4}` gives the
  same image to `1e-12`; alternating `w1`, `w2` over two interleaved segments on an object in the
  central half of the phase-encode field of view gives a ghost of `|w1 - w2| / (w1 + w2)` to
  `1e-9`; `severity = 1` with zero jumps, noise and spikes off, zeroes the selected shot's lines
  at the k-space observation point exactly and leaves the other shots' lines bit-identical to a
  run without the event.
- Extremes: `T2 = 1e-3` ms and `T2 = infinity` give finite outputs (the one-exponent rule).
- Voxel vs class: a constant-per-label map on a homogeneous phantom equals class mode to `1e-6`
  of peak; both run times printed.
- The three gates clean (the 2D path must be untouched).

- [ ] Implement and test; commit `feat: 3D echo-train acquisition (Cartesian)`.

### Task 6: `aslscan` — gradient echo end to end

- `mrsignal`: `Contrast::GradientEcho`; `tissue_ge_spoiled`, `tissue_ge_simasl` (compat), and
  `blood_ge`, with simasl's guards (addendum part A).
- `longitudinal`: the gradient-echo fixed point via two timeline runs (`A`, `B`) for `a != 90`;
  `a = 90` keeps P3's function call exactly; presaturation needs no fixed point. The
  chronological propagation for series whose rows differ in preparation (TR, readout time per
  slice, pulse set, presaturation), first row from its isolated fixed point, per slice.
- `protocol`: `acq_contrast = "ge"`; flip angle overlay over sidecar over 90 with its `Source`;
  `InversionTime` and `inversion_flip_angle` refused under `"ge"`; `"ge"` with
  `MRAcquisitionType: 3D` refused naming the deferred 3D gradient-echo readouts; `echo:
  EchoFormation::Gradient` in `acquisition()`; whether the rows share one preparation, resolved
  once and recorded.
- `series`: the tissue and blood under `"ge"` (closed form, fixed point, or propagation); `m0scan`
  rows and the separate M0 under the series' gradient-echo equation and flip angle.
- `bids`: `FlipAngle` the resolved excitation angle; the M0 sidecar's `FlipAngle` and `Contrast`
  from the series under `"ge"` (2D spin echo keeps forcing 90 and `"se"`); `Resolved.AcqContrast`;
  which steady-state rule applied.
- compat: `"ge"` accepted with the simasl form; the compat transverse factor `exp(-TE/T2*)` with
  simasl's `T2* = 0` guard (`series.rs:538-544`); `compat_asldro.py` loses its refusal (`:390`)
  and gains a GE case in its benchmark; `test_compat_asldro.py` updated (`:152-156`).

Tests:
- `tissue_ge_simasl` and `blood_ge` against simasl fixtures at `1e-12` (generator divides out
  `exp(-TE/T2*)`, P1's method), flip angles 30, 60, 90; `tissue_ge_spoiled` against simasl at
  `T2` small enough that `E2 < 1e-16`, `1e-12`.
- Fixed point: `a = 90` equals P3 bit for bit on every P3 suppression test case; `a != 90` equals
  the timeline iterated from zero until successive starts differ by less than `1e-15 M0`
  (iteration count printed), `1e-12`; no events equals the spoiled closed form, `1e-12`.
- Propagation: the second review's two-row alternation (`TR = 4` s, `T1 = 3` s, 30 degrees,
  readouts at 2 and 3 s, perfect pulses at 1.9 s and at 1 and 2.9 s) against a brute-force
  time-stepped simulation, row by row, `1e-12`; a uniform series equals the fixed point.
- `protocol`: every `"ge"` acceptance and refusal with its exact error.
- End to end: the linearity identity under `"ge"` with suppression, its negative controls;
  `bids-validator` clean on a `p5_ge` fixture.
- compat: the GE benchmark passes against simasl.
- `regress_identity.sh` clean.

- [ ] Implement and test; commit `feat: gradient-echo contrast for 2D EPI`.

### Task 7: `aslscan` — the 3D protocol

**Two phases.** `parse` (`protocol.rs:827`) receives the sidecar, context, overlay and
`PhantomParams`, not the acquisition grid, which `series` computes (`series.rs:405-410`). So
`parse` does everything that does not need the grid (the keys, their sources, the refusals, the
sidecar-level consistency checks) and stores a `ReadoutSpec`; a new
`protocol::resolve_readout(&Protocol, acq_dims: [usize; 3]) -> Result<ReadoutResolution, String>`
does what does (`EPI`, `ETL`, divisibility, `t(ky_c)`, `ESP`, `t_line` from the dwell time,
every timing check, the separate M0's train) and returns the resolved values the sidecar
records. `series::simulate` calls it right after the acquisition grid is computed and before any
simulation; the CLI calls it too, so a run that would fail does so before any output directory
is written; tests call it with explicit dimensions. `[readout] type = "spiral"` parses in this
task and is refused by `resolve_readout` naming milestone C until Task 14.

The acceptance fixtures (`asl005_p5`, `asl003_p5`: the overlays, the `asl003` derivative, the
phantom crops; their contents as listed under Task 10) are created **in this task**, so that its
tests can use them; Task 10 runs them end to end.

- `MRAcquisitionType: 3D` accepted; `PulseSequenceType` match (`grase`, `spiral`) with
  `[readout] type` overriding; anything else an error listing both.
- `[readout]` table (addendum part B "Inputs"): `type`, `ky_segments`, `kz_segments`,
  `kz_order`, `echo_spacing`, `refocusing_time`, `refocusing_flip_angle`, `line_spacing`,
  `readout_samples`, `phase_encoding_direction`; every key refused in 2D, GRASE-only keys refused
  for spirals.
- The 3D timing resolver: `t_exc`; effective spacing (`EffectiveEchoSpacing`, else
  `TotalReadoutTime / (ny - 1)`); `t_line` (overlay, else effective x `ky_segments`, else
  `nx_recv * DwellTime` recorded as a lower bound, else an error naming all); the two 1-percent
  consistency checks; `ESP` from `EchoTime` through `t(ky_c)` (Task 4's function), or the overlay
  with the 1 us check; `PhaseEncodingDirection` overlay over sidecar, required for GRASE; the
  `ky`/`kz` divisibility with the matrix override that would satisfy it in the error.
- Checks: Task 4's timing checks; every suppression pulse before `t_exc`; the separate M0's train
  within its own repetition time.
- `FlipAngle` read as the refocusing angle; `excitation_flip_angle` other than 90 refused;
  `acq_contrast` other than `"se"` refused with 3D.
- Refusals with 3D: `ParallelReductionFactorOutOfPlane != 1`, `PartialFourierDirection` naming
  the slice axis, `SliceTiming`, `SliceEncodingDirection`, `MultibandAccelerationFactor > 1`,
  `[compat] asldro = true`, the `[acquisition]` eddy keys.
- `slice_offsets = [0; nz]` for 3D (set after the acquisition grid is known, in `series`, from a
  protocol flag); `row_start` advancing `NumberShots * TR` per volume; the M0 repetition-time check
  against the train instead of the slice offsets.
- `within_volume` motion in 3D requires `NumberShots > 1` instead of `MultibandAccelerationFactor
  > 1`.
- `phantom`: for a 3D protocol with refocusing below 180 degrees, class eligibility also requires
  T1 constant per label (same tolerance, foreground only); `auto` falls back to `voxel`; explicit
  `class` errors naming the label. 2D unchanged (`t1_perturbation_does_not_affect_the_mode`
  keeps passing).

Tests: every acceptance and refusal with its exact error, in `parse` or `resolve_readout` as
assigned above; the three `bids-examples` 3D fixtures no longer fail naming P5 and instead fail
(as given) with specific messages: `asl003` on the PASL cutoff (`parse`), `asl005` on the missing
phase-encode direction (`parse`), `asl001` with `resolve_readout`'s milestone-C refusal (its
spiral-parameter diagnostics are Task 14's); with their overlays, `asl005` and the `asl003`
derivative parse, and `resolve_readout` with the Task 10 dimensions gives the numbers in Task 4's
tests; the T1 eligibility rule both ways.

- [ ] Fixtures (overlays, the `asl003` derivative with its `SOURCES.md` entry, crop commands
  recorded in `tools/` notes); commit `test: P5 GRASE acceptance fixtures`.
- [ ] Implement and test; commit `feat: protocol — 3D readouts, timing and refusals`.

### Task 8: `aslscan` — the 3D series

- Zero slice offsets: every per-slice computation (P1 kinetics, P3 timeline with
  `t_read = t_exc`, P4 label path) runs unchanged and agrees across slices; the
  `slice_offsets.len() != nz` check (`series.rs:412-416`) applies to 2D only.
- T1 maps in `voxel` mode: `r_sim.rate_mean(t1_ms, &ph.m0)` with background infinity, exported
  as `desc-acqT1map_gt`; `Uniform` T1 per label in class mode; the blood's T1 the kinetic
  model's `t1_arterial_blood`.
- Physio per shot: `PhysioLine` per `(volume, shot)`, excitation at
  `row_start[v] + s TR + t_exc`, label window per shot; the process horizon through the last
  shot's excitation plus one `TR` (`series.rs:352`).
- The extravascular-label group when 3D, `[physio]` and part A are all on: `[3K..4K)` after the
  arterial group (or `[2K..3K)` without part B), tissue relaxation, label factor; the tissue group
  `T` only; `CompartmentOrder` names it. `RowOverride::ExtravascularIntoTissue` (test hook) puts it
  back for the negative control.
- Shot gains from `within_volume` dropout events, composed with physio into `LineWeights`;
  per-shot poses into `shot_images` (events in canonical shot order, jumps persisting).
- The call: `simulate_acquisition_3d` with the train and table from the protocol; the separate M0
  with its own resolution (no labeling, excitation at its `t = 0`, its own TR).

Tests: 3D kinetics equal 2D kinetics at zero slice offset for every row type; the extravascular
group's linearity identity under physio and its negative control (fails by more than `1e2` of
tolerance); the physio horizon covers the last shot (a four-shot series's last factor is not
clamped: the drift at that time is not the grid's last value); shot gains and poses reach the
call in canonical order; separate-M0 timing.

- [ ] Implement and test; commit `feat: series — 3D echo-train readouts`.

### Task 9: `aslscan` — 3D outputs

- `AslscanSimulation.Readout` (addendum list, including `VolumeDuration`) and `EchoAmplitudes`
  (per label `ETL` amplitudes and the kz modulation in encoding order).
- Standard keys: `EffectiveEchoSpacing`, `TotalReadoutTime`, `PhaseEncodingDirection` as resolved
  for GRASE with `InputValuesReplaced`; `NumberShots`; `FlipAngle` the refocusing angle;
  `SliceTiming` never for 3D.
- NIfTI `pixdim[4]` the volume duration for 3D (`bids.rs:335`); 2D unchanged.
- The M0 sidecar from its own resolution (addendum): the refocusing `FlipAngle`, `NumberShots`,
  the effective spacing, a readout block without `t_exc`.
- `desc-physio_gt.tsv` 3D schema (`shot` in place of `slice`) on its own branch of the writer;
  the 2D writer byte-identical.

- `desc-motionEvents_gt.tsv` in 3D: one row per event, `shot` the canonical shot index, the
  attenuation, the jumps, and an empty `slices` column (the 2D writer zips events with dropped
  slice groups, `bids.rs:522-524`; the 3D one has no slice groups and must not drop events).

Tests: the 3D motion-events TSV (row count equals the events drawn, canonical shot indices,
attenuations, empty `slices`); the sidecar blocks and keys on a GRASE run; the 2D physio TSV byte-identical to
`p4-complete`'s on the `p4_all` fixture (also covered by the gate); `bids-validator` clean on a
small GRASE fixture.

- [ ] Implement and test; commit `feat: bids — 3D readout blocks and keys`.

### Task 10: GRASE acceptance, gates, reviews

The acceptance overlays (addendum criteria 3 and 4). The resolved values below are what Task 7's
parse must produce; a difference is investigated, not absorbed.

- **`asl005_p5`** (`tests/fixtures/protocols/asl005_p5/overlay.toml`), on a phantom cropped with
  `tools/hrgt_to_bids.py --crop 0:197 8:225 40:160` (`work/phantom-3t-z120`, 120 mm in z):
  `[acquisition] matrix = [64, 64]`, `[readout] ky_segments = 4, kz_segments = 1,
  phase_encoding_direction = "j-"`, `[m0] repetition_time = 6.0`. Resolves to `64 x 64 x 30`,
  `EPI = 16`, `ETL = 30`, `t_line = 0.2048` ms (from `DwellTime`, lower bound, recorded),
  `ESP = 13.3824` ms, refocusing 130 degrees, the train ending at `4.2031104` s within 4.95 s, the
  last suppression pulse 95 ms before the excitation (second review's numbers).
- **`asl003_p5`**: the derivative sidecar and context (the four PLDs below the 0.7 s cutoff
  removed, every per-volume array cut to match, `SOURCES.md` listing each change), on a phantom
  cropped to `192 x 80 x 180` mm (`--crop 2:194 76:156 5:185`), overlay
  `[acquisition] matrix = [24, 20]`, `[readout] ky_segments = 2`,
  `[background_suppression] model = "bolus-position", pulse_region = "global"`,
  `[m0] repetition_time = 5.0` (its `M0Type` is `Separate`). Resolves to `24 x 20 x 30`, `EPI = 10`,
  `t_line = 1` ms (from `EffectiveEchoSpacing` 0.5 ms x 2), `ESP = 12.42` ms, the train ending
  at 3.378 s within 3.5 s.
- Both simulate and validate (`deno run -A jsr:@bids/validator`, no errors).
- `tests/end_to_end.rs`: the linearity identity on a small GRASE protocol with every P4 part,
  physio and a shot dropout event on; its negative controls.
- Gates: TRXScan `run_p0_baseline.sh` clean; the golden baseline; `regress_identity.sh
  p4-complete 054bfdf` clean in both builds and its self-test detecting; P2's benchmarks A-E
  unchanged; `cargo test` and clippy (by warning kind against the base) in both crates.
- Record every number in Measurements.
- Codex adversarial review of the implementation, fixes, then the ordinary Codex review as the
  final pass; tag `p5-grase-complete` in `aslscan` and `mrsim-acq`.

- The class-versus-voxel discrepancy on the real (mixed-cell) `asl005_p5` phantom, with
  refocusing at 130 degrees, measured and recorded, not asserted (spec, "The T1 map in voxel
  mode").
- [ ] Acceptance runs (fixtures from Task 7), gates, Measurements.
- [ ] Codex reviews and fixes; tag `p5-grase-complete`.

---

## Milestone C: the stack of spirals

Begins only after `p5-grase-complete`. Task 12 ends in a decision.

**Preflight, before Task 11** (a short worked note in Measurements, no code): for the
`asl001_p5` geometry (`T = 4` ms, dwell 4 us, 1000 samples per interleaf, 8 interleaves, 20
partitions, `64 x 64` at oversample 2) and the phantom's fieldmap and decay ranges, compute the
certificate's degree `m` (spec, part C: the smallest `m` with the Chebyshev remainders below
`5e-8`), the grid size `(m+1)^2` (voxel) and `m+1` (class), the certification cost
`grid x samples x L` for `L` up to 64, and the forward cost `L` NUFFTs per slice per compartment
per coil. Proceed to Task 11 only if both are within an order of magnitude of the ten-minute
budget; otherwise revisit part C of the spec first. (The plan review found the spec's earlier
first-order certificate needed `10^15` grid points; the Chebyshev certificate replaced it, and
this preflight is where its cost is checked before code depends on it.)

### Task 11: `mrsim-acq` — the 2D NUFFT pair

`nufft.rs` (feature `kspace`): `Nufft2d::type1` (non-uniform to grid) and `type2` (grid to
non-uniform) on the existing exponential-of-semicircle kernel, `sigma = 2`, tolerance argument as
the 1D one; positions in the Cartesian convention of the addendum (`kx xc / nx`, positive
exponent).

Tests: each against a direct sum to `1e-10` at random positions and at grid positions (where it
reduces to the DFT); the adjoint identity `<A x, y> = <x, A^H y>` to `1e-10`; the 1D tests
unchanged.

- [ ] Implement and test; commit `feat: 2D NUFFT pair`.

### Task 12: `mrsim-acq` — the spiral trajectory, and the feasibility decision

- `Readout3d::Spiral { interleaves, readout_ms, dwell_ms }` and `spiral_trajectory`: the
  Archimedean spiral in `u`, `u(tau)` constant-angular-velocity inside `tau_c` and
  constant-linear-velocity outside, samples at `(j + 1/2) dwell`.
- The sampling bound: `tau_c = 1 / (T (1/(k_max dwell)^2 - 4 pi^2 n_turns^2 / T^2))`, feasible iff
  `dwell < T / (2 pi n_turns k_max)`, `tau_c <= T`; errors naming the limiting dwell or the
  readout length.
- `check_spiral_timing`: `T + refocusing_time/2 <= ESP/2`; train in `TR`; `NumberShots`
  absent means `interleaves * kz_segments`.
- The spiral forward by **exact sum** (oracle), in the addendum's convention, per coil with
  `amp_q`, `phi0`, `1/nvox`.

Tests: the speed bound holds on a fine continuous sampling of the trajectory (maximum speed x
dwell at most 1 + `1e-12`); the second review's 32-sample counterexample is rejected; a dwell
one percent above the bound is rejected; the exact spiral forward at Cartesian frequency
locations equals the Cartesian forward of a displaced complex point at oversampling 1 and 2 to
`1e-12`; the `asl001` overlay numbers (`64 x 64 x 20`, 8 interleaves, `T = 4` ms, dwell 4 us) give
`ESP = 10.528` ms, `T + 1 = 5 <= 5.264` ms, the train ending at 3.68956 s.

**Feasibility benchmark** (`#[ignore]`, run once, numbers recorded): one `asl001`-sized volume
(`64 x 64 x 20`, oversample 2, six compartments, one coil) with a 50 Hz fieldmap range and a
decay range from the phantom, timed for a prototype of Task 13's time-segmented forward (its
`L`, `m`, certification time including setup and the least-squares fit, and forward time) and
Pipe-Menon gridding (its error on a band-limited object), **separately in `class` mode** (the
frequency interval only) **and in `voxel` mode** (the decay-frequency rectangle). **Decision**,
recorded here, per mode: proceed for a mode if `L <= 64` certifies at `1e-7`, the gridding error
is within the declared `1e-2` of peak, and one volume, certification and setup included, takes
under ten minutes in WSL. If only `class` passes, milestone C proceeds with spiral `voxel` mode
refused (an error naming the measured cost) and the spec amended to say so; if neither passes,
stop, revisit part C, and leave milestone AB as P5's result.

- [ ] Implement and test; commit `feat: spiral trajectory and sampling bound`.
- [ ] Feasibility benchmark run; decision recorded in Measurements.

### Task 13: `mrsim-acq` — time segmentation, gridding, the spiral path

- `tseg.rs`: least-squares interpolators fitted on the tensor Chebyshev grid of degree `m` over
  the rate rectangle (the frequency interval alone in `class` mode), `m` from the remainder bound,
  `L` doubling from its start value, the certified bound
  `Lambda_m^2 max_grid |e| + rem(R_d) + Lambda_m rem(R_f)` (spec, part C), `L > 64` an error;
  `L`, `m`, `B` and the bound recorded in the returned plan.
- The spiral forward: `class` mode with decay as per-sample line weights outside the sum and only
  the fieldmap in the segmentation; `voxel` mode with the full complex rate
  `i 2 pi f - 1/T2 - 1/T2'` in the segmentation and `ln A_e(r)` in the static amplitude.
- `grid_recon.rs`: Pipe-Menon density compensation (10 iterations), normalized so a constant
  object with no off-resonance reconstructs to its Cartesian value; type-1 adjoint, deapodized;
  per coil, then Roemer.
- `simulate_acquisition_3d` dispatches `Readout3d::Spiral`; a non-`kspace` build refuses spirals;
  GRAPPA, partial Fourier, ghosting and spikes refused with spirals; the window as a radial
  sample weight.

Tests: the time-segmented forward against the exact sum on a complete `32 x 32`, oversample 2
trajectory with independent fieldmap, T2 and T2' maps, nonconstant object phase, two coils and a
`signal_scale`, `1e-6` of peak; the certified bound never exceeded by the actual error at rates
between grid points, on fieldmap and decay stress cases including `100` to `1100` s^-1; gridding
of a band-limited object to `1e-2` of peak; a uniform object to its Cartesian value to `1e-3`;
uniform off-resonance equals gridding of exact-sum samples to `1e-6`; noise ratio printed.

- [ ] Implement and test; commit `feat: stack-of-spirals forward and gridding`.

### Task 14: `aslscan` — spirals end to end, `asl001`

- `protocol`: `type = "spiral"` with `interleaves` and `spiral_readout_time` required, `dwell_time`
  from the overlay or `DwellTime`; `PhaseEncodingDirection`, `TotalReadoutTime`,
  `EffectiveEchoSpacing` refused (from either source); `NumberShots` rule; square matrix required.
- `series` and `bids`: the spiral readout block (trajectory parameters, `tau_c`, `L`, `m`, the
  certified bound, the density-compensation iterations).
- **`asl001_p5`**: on a phantom cropped to 160 mm in z (`--crop 0:197 0:233 15:175`), overlay
  `[acquisition] matrix = [64, 64]`, `[readout] interleaves = 8, spiral_readout_time = 4.0,
  dwell_time = 4e-6`. Resolves to `64 x 64 x 20`, eight shots, a 39.1 s volume duration,
  refocusing 111 degrees; simulates and validates.
- End to end: the linearity identity on a small spiral protocol; `regress_identity.sh` clean.

- [ ] Implement and test; commit `feat: spiral protocols and asl001 acceptance`.
- [ ] Codex adversarial review of milestone C, fixes, ordinary Codex final review; tag
  `p5-complete` in both crates.

## Acceptance criteria coverage

| Criterion | Where |
|---|---|
| 1. No P5 input, no changed byte (TRXScan, `mrsim-acq`, `aslscan`) | Task 0 (baselines, gate), every `mrsim-acq` task, Task 10 |
| 2. GE spoiled model and compat against simasl | Task 2, Task 6 |
| 3. `asl005` GRASE simulates and validates | Tasks 4, 5, 7-10 |
| 4. `asl003` derivative simulates and validates | Tasks 4, 7-10 |
| 5. `asl001` spiral simulates and validates | Tasks 11-14 |
| 6. PSF, EPG, 3D noise, segmentation, spiral tests; numbers recorded | Tasks 3, 5, 12, 13; Measurements |

## Measurements

### Milestone AB (2026-10-02)

- **Gates at the pre-P5 heads** (aslscan `p4-complete`, mrsim-acq `054bfdf`, TRXScan `130d62b`):
  TRXScan P0 diff clean (62 checksums, 7 min 22 s); `regress_identity.sh` 38 runs identical (19
  cases x 2 feature sets); `--self-test` detected the 1e-4 perturbation in all 38; golden record
  written (44 outputs per build, identical across debug/release and with or without `par`).
- **After each mrsim-acq task**: TRXScan P0 clean after Tasks 1, 2 and 5; golden record unchanged
  throughout; `regress_identity.sh` 38 identical after Tasks 1, 2 and 6 (the Task 6 run also
  covers Task 5's mrsim-acq).
- **Task 3**: the EPG and an independent 2001-isochromat Bloch simulation agree within `1e-11`
  (asserted at `1e-9`) for 111, 130 and 160 degrees, GM/WM/CSF/blood, 30 echoes.
- **Task 5**: 3D noise SD `0.2517` of the 2D value at `nz = 16` (want `0.25`); class `1.7 ms`,
  voxel `4.3 ms` on the 12 x 12 x 6 cross-check; GRASE under 62.5 Hz moves exactly one voxel, in
  the 2D EPI's direction. Deviation from the plan: each compartment's k-space is its own call of
  the 2D forward rather than a private accumulator (same result, 2D path untouched).
- **Task 6**: simasl's gradient-echo form matches its fixtures at `1e-12`, the spoiled form on
  every voxel with `E2 < 1e-16`; the row propagation reproduces the review's alternation
  (`-0.254639`, `-0.089888`); linearity under `"ge"` `0.118` of the tolerance; compat benchmark G
  passes (crop and full 3 T: `5e-8` per volume, `4.5e-6` control - label); `p5_ge` validates.
- **Tasks 7-9**: the acceptance fixtures resolve to the reviewed numbers; a 3D series's compartment
  images equal its 2D twin's at zero slice offset bit for bit; linearity under GRASE with physio and
  exchange `0.113`, its extravascular-into-tissue control `2312x`; `p5_grase` validates.
- **Task 10**: `asl005` (64 x 64 x 30, 4 shots, ESP 13.3824 ms, 0.2048 ms lines from DwellTime,
  130 degrees) simulates in 8.2 s and validates; the `asl003` derivative (24 x 20 x 30, 2 shots,
  ESP 12.42 ms, 1 ms lines) in 1.6 s and validates; linearity with every P4 part, physio and a
  shot event `0.157`. Class vs voxel on `asl005` (mixed cells, 130 degrees; measured, not
  asserted): max `9.7e-2` of peak over all volumes and `0.31%` median over brain; control - label
  max `0.114` of its peak, median `6.3e-5`; voxel mode 80 s against class 6.6 s. P2 benchmarks
  A-E on 3 T unchanged from P4's log (G added), the synthetic-block gates and A, E on 1.5 T pass.
- **Reviews**: the Codex adversarial review of milestone AB died for lack of credits (2026-10-02,
  no findings produced); the user chose to continue with milestone C and tag later. Open: that
  review, the ordinary final review, and the `p5-grase-complete` tag.

### Milestone C preflight (2026-10-02, before Task 11)

For the `asl001_p5` geometry: `64 x 64 x 20` at oversample 2 (`128 x 128` simulation slices), 8
interleaves of `floor(4 ms / 4 us) = 1000` samples (8000 per partition), 20 partitions, one coil,
six compartments in class mode. The ASLDRO phantoms carry no fieldmap, so the benchmark uses a
synthetic one of 50 Hz range, as Task 12 says.

- **Certificate degree** (spec, part C): `R_f T = 2 pi 50 x 0.004 = 1.257`; the remainder
  `2 (R T / 4)^(m+1) (1 + B)/(m+1)!` is `5.3e-9 (1 + B)` at `m = 7` and `9.2e-11 (1 + B)` at `m = 8`,
  so `m = 8` for `B` up to about 500. The decay range in voxel mode (`1/T2 + 1/T2'` up to about
  70 s^-1, `R_d T = 0.28`) needs less. Grid: 9 rates (class), 81 (voxel).
- **Certification cost**: grid x 1000 sample times x `L` (start `ceil(T x 50 Hz) + 2 = 3`, at most
  64): under `5e6` complex exponentials even at `L = 64`, plus one small least-squares solve per
  sample time. Negligible.
- **Forward cost**: one type-2 NUFFT on the `256 x 256` fine grid (FFT about `5e6` flops) plus
  spreading `8000 x 8^2` (kernel width 8 for `1e-7`): about `6e6` flops; per volume
  `20 slices x 6 compartments x L` NUFFTs, `7e8 L` flops, a few seconds for `L <= 8`; voxel mode
  `x 20` echoes, about a minute.
- **Reconstruction**: one type-1 NUFFT per partition per coil, and the Pipe-Menon density weights
  once per series (10 iterations of a type-1/type-2 pair): about `2e8` flops. Negligible.
- **Verdict**: both the certification and the forward are well within an order of magnitude of
  the ten-minute budget (they are seconds to a minute), in both modes. Proceed to Task 11.

### Task 12: the feasibility benchmark and decision (2026-10-02)

Built: the spiral trajectory and sampling bound, the exact-sum oracle (pinned against the
Cartesian forward at Cartesian frequencies, oversample 1 and 2, to `1e-12`), and, as the
benchmark's prototype, Task 13's pieces themselves: `tseg.rs` (the certified least-squares
segmentation), the time-segmented forward (`spiral.rs`, `SegmentedForward`) and `grid_recon.rs`
(operator-form Pipe-Menon, 10 iterations, gridding). All in WSL, release.

- **Segmentation**: the least-squares fit is solved by a factored one-sided Jacobi SVD (an explicit
  pseudo-inverse loses `4.5e-6` to cancellation at these condition numbers, `1e11` and more, and
  failed to certify a 250 Hz range at all). The certified bound includes `4 L eps (1 + B)` for
  rounding. Stress cases at `T = 4` ms (bound / actual error on a 24 x 38 off-grid sampling of the
  rectangle): 50 Hz, `L 6 m 7 B 3.1`, `2.6e-8 / 2.9e-9`; 250 Hz, `L 12 m 15 B 49`,
  `6.6e-9 / 1.3e-12`; decay `100..1100 s^-1`, `L 8 m 12 B 17`, `1.8e-8 / 5.0e-9`; the same with
  `+-50` Hz, `L 12 m 12 B 136`, `4.4e-8 / 1.1e-11`. The actual error never exceeded the bound. A
  4000 Hz range at `T = 20` ms is refused naming the rectangle.
- **Forward**: against the exact sum on a complete `32 x 32`, oversample 2 trajectory with
  independent fieldmap (50 Hz), T2, T2' and `ln A` maps, nonconstant phase, two coils and
  `signal_scale` 3: `1.2e-9` of peak (voxel), `6.0e-10` (class); declared `1e-6`.
- **One `asl001` volume** (`64 x 64 x 20`, oversample 2, 8 interleaves of 1000 samples, six
  compartments, one coil, 50 Hz, decay `0.019..0.058 /ms`): class `L 6 m 7`, certification 4 ms,
  forward 2.8 s; voxel `L 6 m 7`, certification 3 ms, forward 56 s (20 echoes x 20 slices x 6
  compartments at 23 ms); density weights 83 ms, gridding 20 partitions 79 ms. Both modes are far
  inside ten minutes.
- **Gridding accuracy fails the declared tolerances**, in both modes (the reconstruction is
  mode-independent). Band-limited object (Gaussian blobs, spectrum below `1e-13` at `k_max`, at
  oversample 1 so the data are exact): `4.1e-2` of peak (`32 x 32`, 4 interleaves), `4.3e-2`
  (`asl001`'s `64 x 64`, 8 interleaves); declared `1e-2`. Uniform object: `0.34` and `0.50` over
  the FOV (`0.25` over the central half); declared `1e-3`. The density weights are not the cause:
  kernel Pipe-Menon, operator Pipe-Menon and the trajectory's exact annulus areas all give 1-6%,
  and more iterations or an oversampled density grid do not help. The cause is the design's
  sampling, radially exactly Nyquist (interleaves 1 cycle/FOV apart), at which gridding's
  quadrature is a few percent; the uniform object fills the FOV, so its spectrum has the box's
  Dirichlet tails outside the sampled disc.
- **What does meet them**: density-weighted least squares by conjugate gradients on the same NUFFT
  pair, per coil (not SENSE), from zero. Band-limited: `4.6e-3` (5 iterations), `2.7e-3` (10),
  `4.1e-4` (20) at `32 x 32`; `8.9e-3`, `1.8e-3`, `4.9e-4` at `64 x 64`. Uniform: `8.4e-2`, `1.3e-2`,
  `2.7e-3` at `32 x 32`; `0.12`, `1.6e-2`, `8.2e-3` at `64 x 64` (still short of `1e-3` at 20). Cost
  about 11 ms per iteration per partition per coil at `64 x 64`: 20 iterations x 20 partitions,
  4 s per coil per volume.
- **Decision (per the plan's rule)**: `L <= 64` certifies at `1e-7` and the volume time passes in
  both modes, but the gridding error does not, in either mode, so neither mode passes. The plan
  says: stop, revisit part C. The user chose (2026-10-02) to revise part C: per-coil
  density-weighted least squares, the band-limited `1e-2` kept, the uniform `1e-3` restated at
  the image centre.
- **The revision** (spec part C, "Reconstruction", amended; not yet Codex-reviewed): conjugate
  gradients is accurate but its step sizes depend on the data, so it is not linear, and the
  linearity identity (`I_C - I_L - I_B`) needs a linear reconstruction. The solver is therefore
  the Chebyshev semi-iteration on the density-weighted normal equations over
  `[lambda_hi / 30, lambda_hi]`, 40 iterations, `lambda_hi` 1.1 times 50 power iterations: fixed
  coefficients, linear, residual polynomial at most 1. Measured against the declared tolerances:
  band-limited `3.4e-4` (`32 x 32`) and `3.1e-4` (`64 x 64`) of peak; uniform `5.2e-5` and
  `7.0e-7` at the centre, `2.2e-3` and `5.5e-3` over the FOV; linearity to `1e-12`; noise SD
  ratio to Cartesian `1.01` (2D), `0.98` (3D). One `asl001` volume's reconstruction: setup 0.49 s
  once per series, 20 partitions 6.2 s per coil. (Chebyshev at `kappa = 10`/`100` and 20
  iterations measured too: `kappa = 100` needs the 40.)

### Task 13 (2026-10-02)

`kspace3d` dispatches `Readout3d::Spiral` (a non-`kspace` build panics naming the feature):
per-slice certified segmentations (class: the slice's fieldmap interval; voxel: its decay x
frequency rectangle), the z-DFT, per-sample echo-amplitude and in-echo decay weights (class),
per-echo `ln A` maps (voxel), per-shot line weights and shot sets, noise per (volume, partition,
coil) on the 3D path's seeds, the inverse z-DFT and the least squares per partition, Roemer.
`spiral_segmentation` reports every slice's `L`, `m`, `B` and bound (or the error naming the
rectangle) for the caller to check and record. GRAPPA, partial Fourier, ghosting and spikes
panic with spirals. The decay-mode resolution moved out of the GRASE `plan()` verbatim
(`resolve_modes`), shared by both paths. Tests: 3D equals the 2D spiral pipeline per slice when
nothing distinguishes partitions (`1e-9`); class and voxel modes agree to `4.2e-12` at 130
degrees; linear in the images (`1e-6`), line weights scale their shots; noise seeded and
measured; the refusals and the uncertifiable-range report.

### Task 14 (2026-10-02, aslscan a5ccff2)

- `[readout] type = "spiral"`: `interleaves` and `spiral_readout_time` required (errors naming
  them), `dwell_time` from the overlay over `DwellTime`, `NumberShots` absent meaning
  `interleaves x kz_segments` and checked when given, a square matrix, `PhaseEncodingDirection`,
  `TotalReadoutTime`, `EffectiveEchoSpacing`, GRAPPA, partial Fourier, ghosting and spikes refused,
  a build without `kspace` refused naming the feature. The series certifies every slice's
  segmentation before the acquisition and records it; the sidecar's `Readout` block carries the
  trajectory (`tau_c`, samples, turns), the segmentation (per mode: max `L`, `m`, `B`, bound, and
  per slice) and the reconstruction's fixed parameters; `DwellTime` and `NumberShots` are the
  standard keys written (no phase-encode keys), on the M0 sidecar too.
- **`asl001_p5`** (crop `0:197 0:233 15:175`, `work/phantom-3t-asl001`): resolves to
  `64 x 64 x 20`, 8 shots, ESP `10.528` ms, 1000 samples per interleaf, `tau_c = 0.0116` ms,
  refocusing 111 degrees (sidecar), volume duration `8 x 4.886 = 39.088` s; the phantom has no
  fieldmap, so the segmentation is `L = 2` (exact); simulates in 7.5 s (WSL, release) and
  validates with no errors (warnings only: authors, recommended keys).
- End to end: linearity under spirals with every P4 part, physiology and a shot event `0.163` of
  the tolerance, its negative control failing; the sidecar test; the non-`kspace` refusal.
- Gates on the final state of Tasks 11-14 (aslscan a5ccff2, mrsim-acq 10c40df): TRXScan P0 clean;
  `regress_identity.sh` 38 identical.

### Codex adversarial implementation reviews (2026-10-02)

Two runs: milestone AB at its commits (mrsim-acq `054bfdf..f13d157`, aslscan
`p4-complete..9b29d48`) and milestone C with the part C amendment. Every finding was checked
against the code; all were valid and are fixed (mrsim-acq 247f96e, aslscan 10ee904).

- **AB** (2 major, 3 minor): a zero arterial blood T1 panicked the 3D EPG (refused in
  `resolve_readout`); voxel mode never wrote `desc-acqT1map_gt` (written now, tested); 3D spike
  streams ignored the coil (keyed on it now, tested); an accepted overlay line spacing published
  the sidecar's effective spacing (the simulated one now); the GE compat sidecar named `T2`
  instead of `T2*`.
- **C** (3 major, 4 minor): the certificate's rounding term ignored the phase (now
  `(Lambda^2 + 1)(1 + B)(4 eps (1 + theta) + 2 L eps)`, phases past `1e4` rad refused, tested
  against a double-double oracle: at a constant 10 kHz the bound is `9.0e-13` against an error of
  `8.2e-15`); `lambda_hi` was claimed as an upper bound (now estimated from a pseudo-random start
  to convergence, the claim withdrawn, each reconstruction checking its residual did not grow);
  the band-limited accuracy held only for centred objects. A sweep over exactly band-limited
  objects (`|k| <= 0.625 k_max`) at the centre, edge and corner:

  | `kappa`, iterations | `32 x 32` worst | `64 x 64` worst | noise / Cartesian (`32 x 32`) |
  |---|---|---|---|
  | 30, 40 | `1.5e-2` (edge) | `6.6e-3` | about 1.0 |
  | 100, 80 | `6.8e-3` (corner) | `3.5e-3` | 1.60 |
  | 300, 80 | `4.2e-3` | `2.5e-3` | 2.03 |

  The reconstruction is now 80 iterations over `[lambda/100, lambda]` (the noise cost of
  `kappa = 300` judged not worth its margin), described as a regularized approximate inverse; the
  test covers all three positions at both sizes. The shot test asserts on acquired samples; an
  overridden `PulseSequenceType`, a non-string `PhaseEncodingDirection` with a spiral and the
  separate M0's replaced `DwellTime` are handled. `asl001` now simulates in 22.3 s (80 iterations)
  and validates.

## Codex review of this plan (2026-10-02)

Nine findings (1 blocker, 4 major, 4 minor), all verified and applied. The blocker was in the
spec: the time-segmentation certificate's first-order bound needed `10^15` rate-grid points; the
spec now uses a Chebyshev certificate (a few hundred points), and milestone C opens with an
analytical cost preflight and decides `class` and `voxel` mode separately. Majors: the Task 0
self-test now has its own layout and expected manifest path and a perturbation that survives the
`f32` cast, and passes only on a content difference after successful runs; Task 7 is split into
`parse` and a geometry-aware `resolve_readout` (the parser never sees the acquisition grid), the
acceptance fixtures move into Task 7, and spiral diagnostics move to Task 14; the within-echo
`LineTiming` is partition-independent but `trf` is not, so the two timings are kept apart; the
test commands are explicit (the end-to-end suite needs `io,test-hooks`). Minors: a k-space
observation point for exact acquired-line assertions; the EPG tolerance is on amplitudes, not
logarithms; the TRXScan gate runs in a worktree of its branch and the `diff` is the gate; the 3D
motion-events TSV and the mixed-cell class/voxel discrepancy have tasks. The review verified the
worked numbers (`asl003` derivative 12.42 ms and 3.3776 s, `asl005` 13.3824 ms and 4.2031104 s,
`asl001` 10.528 ms, 3.68956 s and `tau_c ~ 0.0116` ms), the crops against the phantom, and the
existing overlay keys.
