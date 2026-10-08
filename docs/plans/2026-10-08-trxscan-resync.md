# TRXScan Re-sync with mrsim-acq: Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TRXScan `main` (currently `acb7506`) consume mrsim-acq again, without moving one TRXScan output
bit relative to `main` and without moving one aslscan output bit relative to `p7-complete`.

**Why.** The P0 extraction (`docs/plans/2026-09-21-p0-mrsim-acq-extraction.md`) moved TRXScan's acquisition
stage into mrsim-acq from a TRXScan snapshot, `05c76bf` (2026-09-23). Its TRXScan branch,
`p0-mrsim-acq-extraction` (`12a9d09`), was never merged. Since then, TRXScan `main` changed the moved modules:
- `ef4abe9` (2026-09-24) squashes the GNL work but leaves part of it out. The snapshot's last GNL commit,
  `57858a5`, has eddy replay, the smooth background phase and the legacy path; the squash does not.
- `02ca507` (2026-09-28) adds the Python bindings.

A trial merge of the old branch into `main` conflicts in 19 files (modify/delete on `kspace.rs`, `motion.rs`
and `phase.rs`). The old branch would also bring back the work the squash left out.

**Architecture.** Redo the TRXScan side on a fresh branch from `main`. Port `main`'s acquisition changes into
mrsim-acq as additive, default-off capabilities. Keep TRXScan's public API (CLI, Python bindings) on a thin
TRXScan-side facade over mrsim-acq. Order of work:
1. Capture `main`'s own output as the baseline.
2. Grow mrsim-acq additively, gated on the aslscan regress run and the TRXScan P0 gate.
3. Re-point TRXScan, gated on the new baseline.
4. Only then remove the pieces `main` dropped.

No task changes both TRXScan's code location and its behavior.

**Inputs.** The change inventory (2026-10-08, from `git diff 05c76bf origin/main` on the moved modules) is
summarized in "What main changed" below. The old branch's re-pointing commits, `4aa20fd..12a9d09`, are the
reference for the mechanical TRXScan-side work.

## Decisions already made

- **The pieces `main` dropped are removed from mrsim-acq** (user, 2026-10-08):
  - `SliceInput.eddy_lin` with the `eddy_trace` entry argument (eddy replay);
  - `BackgroundPhase.smooth` (the smooth background-phase modes);
  - `simulate_acquisition_legacy` and its test;
  - `Nufft1::n`.

  Neither consumer uses them. They stay in git history and on the old branch. The TRXScan-side features
  dropped with them are simply not re-extracted: `--eddy-trace`, `--phase-bg-scale`, `--phase-smooth`,
  `--s0-map`/`--t2-map`, `--csf-scale-map`, and the MESE/MEGRE scripts. The removal comes last (Task 7);
  see "Why removals last".
- **A fresh branch from `main`, not a merge of the old branch.** Merging would bring back the work `main` left
  out, and the pre-squash GNL history.
- **mrsim-acq keeps its general API** (the P0 interface changes: `T2Volume`, `eddy_drive`/`prep_drive`,
  `PrepPhase`, `DropoutLaw`, the echo formation). `main`'s diffusion-shaped API becomes a TRXScan facade:
  `SimulationInput` with `bvals`/`bvecs`/`t2`, `simulate_acquisition`, `simulate_acquisition_complex`,
  `Acquisition::hbcd`, `dropout_events`, and `apply_multiband_motion(_slab)` taking `bvals, b_max`.
- **mrsim-acq's `Acquisition::default()` keeps `FiberfoxCompatible`.** aslscan, `kspace3d`, the spiral path
  and mrsim-acq's tests rely on it. It is the only field whose default differs from `main`'s
  (`main` defaults `pf_mode` to `Scanner`; `echo` is mrsim-acq's own field). `Default` cannot be overridden
  on a foreign type, so TRXScan gets `acq::default_acquisition()`, and no TRXScan code may call
  `Acquisition::default()` (Task 6 greps for it).
- **mrsim-acq as a git dependency.** TRXScan pins sibling crates as git dependencies (`trx-rs` by rev,
  `odx-rs` by tag), and its CI, wheel and release workflows (`ci.yml`, `python.yml`, `release.yml`;
  maturin-action builds in Docker) cannot see a `../mrsim-acq` path. mrsim-acq is public.
  - TRXScan depends on `mrsim-acq = { git = "https://github.com/PennLINC/mrsim-acq", tag = "..." }`.
  - Local gates use a `[patch."https://github.com/PennLINC/mrsim-acq"]` path override, kept out of the
    committed tree (in the gate script, or an uncommitted `.cargo/config.toml`).
  - A tagged mrsim-acq must be pushed before the TRXScan PR (Task 8).
- **One deliberate behavior difference.** mrsim-acq's entry point refuses (panics) when a readout starts
  before its excitation (`trf ≤ 0` on an acquired line; P0, `3102f75`). `main` has no such check and
  simulates those timings. The re-pointed TRXScan inherits the refusal. The Python facade validates first
  and raises a Python `ValueError`, rather than letting a Rust panic reach the binding. No baseline case
  may sit in that regime.

## Why removals last

The frozen TRXScan P0 worktree (`../TRXScan-p0`, `12a9d09`) is the only diffusion-shaped check on mrsim-acq
until TRXScan is re-pointed. It calls `simulate_acquisition_legacy` and passes `eddy_trace`/`eddy_lin`, so it
stops compiling the moment those are removed.
- Tasks 2-5 are additive. Nothing the frozen tree uses changes signature, and it matches `PartialFourierMode`
  only by constructing it. So its 62 checksums keep guarding Tasks 2-5.
- The removals (Task 7) come after Task 6 re-points TRXScan `main`, whose own baseline then takes over. The
  P0 gate is retired at Task 7.

## What main changed (inventory)

Categories: A = API shape only; B = new capability beside existing paths; C = could change output bits for an
existing configuration; D = absent on `main`.

| Change on `main` | Cat | Port to |
|---|---|---|
| `PartialFourierMode::Scanner` (late start, skips the first lines), `pf_skipped_lines`, the Scanner branch in `sampling_mask`, and the eddy clock (`tread`) shifted by the skipped lines | B (C only in Scanner mode) | mrsim-acq, default-off. The `tread` shift must go into `LineTiming::for_acquisition`, so `kspace3d`, spiral and GRASE agree. mrsim-acq's `sampling_mask` tests only `== Contiguous` and falls through to the Fiberfox rule, so Scanner needs its own branch |
| `#[default] Scanner` on `Acquisition` and the CLI (`--pf-mode scanner`) | C | TRXScan facade only (`default_acquisition`) |
| `Acquisition::hbcd(ny)` preset | B | TRXScan facade, as a free function (the type is foreign) |
| `EpiTiming`, `epi_timing`, `epi_trajectory` | B | mrsim-acq, beside `LineTiming` |
| `SliceCapture`, `SliceRecon`, `simulate_slice_full` (`simulate_slice` becomes a wrapper) | B | mrsim-acq |
| `simulate_acquisition_complex`, `AcquisitionOptions` (`slice_z`/`nz_full`, `kspace_slices`, `capture`, `t_echo_per_volume`, `progress`), `KspaceCapture`, `AcquisitionOutput` | B (C only for slabs) | mrsim-acq, over its general entry point |
| `SimulationInput` and `simulate_acquisition` (replacing the positional entry point) | A | TRXScan facade |
| Noise-map RNG keyed on the global voxel (identical when not a slab) | A | mrsim-acq, with the options |
| `motion::apply_multiband_motion_slab`; `apply_multiband_motion` delegates to it. Both take `bvals, b_max` on `main` | B (C only for slabs) | mrsim-acq, written with `DropoutLaw`; TRXScan facade wrappers with `main`'s signatures |
| `motion::dropout_events` and `dropout_seed` (hard-codes `b < 50`) | A (moved from the bin) | TRXScan facade (diffusion-specific) |
| `eddy_lin`, `BackgroundPhase.smooth`, `simulate_acquisition_legacy`, `Nufft1::n` | D | Removed from mrsim-acq, last (Task 7) |
| `io::write_gre_fieldmap`; `hires_grid` delegating to a new inherent `Grid::hires` (used by Python and `io.rs`) | B / A | TRXScan. `Grid` is foreign there, so use `mrsim_acq::io::hires_grid` (identical arithmetic) or the old branch's `GridRaster` trait. Python builds without `io`, so it needs the trait or a std-only helper |

`readout.rs`, `noise.rs`, `mat.rs`, `orient.rs`, `analytic.rs` and `config.rs` are byte-identical between
the snapshot and `main`. `kspace::Rng`, which `main`'s new `gre.rs` uses, exists in mrsim-acq unchanged. The
three newest `main` commits (`0b917e6`, `be68c24`, `acb7506`) touch no moved module.

## Global constraints

- **Identity gates.**
  - aslscan, every task: `tools/regress_identity.sh p7-complete p7-complete`, all cases identical (67 at P7).
  - TRXScan P0 (frozen worktree, 62 checksums), every mrsim-acq task through Task 5. Retired at Task 7.
  - TRXScan `main`, from Task 6: outputs byte-identical to the Task 1 baseline under both `--features cli`
    and `--features cli,kspace,par`; the Python hashes identical; and `main`'s CI steps passing locally
    (`cargo test` with default features, `cargo test --features cli --all-targets`, `maturin develop` and
    `pytest` in `python/`).
- **The old P0 checksums do not describe `main`.** `main`'s CLI default partial Fourier moved from
  `contiguous` to `scanner`, so `main` itself differs from `05c76bf`. The Task 1 baseline is the reference
  for TRXScan. The P0 checksums only guard mrsim-acq's own history (above).
- Hand-formatted source, about 100 columns. No repo-wide `cargo fmt`. Tests live in module `mod tests`.
- Clippy: no new warnings in any crate.
- Git: `-c core.autocrlf=false` for worktree add, commit and merge. Run gates from WSL-made detached
  worktrees.
- Nothing is pushed, and no PR is opened, without the user's go-ahead. TRXScan `main` is Matt Cieslak's
  branch: the re-sync lands there by PR, for his review.

---

## Task 0: Branches, worktrees and the starting gate

- [ ] TRXScan: create a worktree `../TRXScan-resync` on a new branch `resync-mrsim-acq` from `origin/main`
  (`acb7506`), made from WSL so both gits can read it.
- [ ] mrsim-acq: work on `main`, as in P1-P7. Tag the starting point `pre-resync` (local), so Task 6 can
  bisect mrsim-acq if its gate fails.
- [ ] Run the TRXScan P0 gate once against the current mrsim-acq (expect 62 of 62).
- [ ] Ask the user which micromamba environment has maturin and the bindings' Python dependencies (`numpy`,
  `scipy`, `nibabel`, `pooch`, `dipy`, `trx-python`, `pytest`), as `python.yml` installs them.
- [ ] Record the heads and the environment in this plan's Measurements.

## Task 1: Baseline of TRXScan main

- [ ] Bring the P0 fixture tooling onto `resync-mrsim-acq`: `tools/gen_p0_fixture.py`,
  `tools/run_p0_baseline.sh` and `tests/fixtures/p0_baseline/` inputs (content from `05c76bf`; no source
  change). Every flag the script uses exists on `main`'s CLI (verified). Rename its `legacy` case
  (`--oversample 1`): `main` has no legacy path, so it runs the oversampled path at o = 1.
- [ ] Extend the run matrix so it covers what `main` added or changed. Each case is one CLI invocation, under
  both feature sets:
  - partial Fourier `scanner` (the new default), `contiguous` and `fiberfox`;
  - the `hbcd` preset;
  - multiband with motion and dropout;
  - a noise map;
  - `--gre-out` (the GRE fieldmap) and GNL ground truth;
  - `--eddy-phase`.

  No case may have `trf ≤ 0` (see the behavior difference above).
- [ ] Python baseline: build with `maturin develop --release` in `python/`, then a script that hashes the
  arrays from:
  - `simulate`;
  - `simulate_acquisition_complex` with capture, and with `slice_z`/`nz_full` and `t_echo_per_volume`;
  - `epi_timing`/`epi_trajectory`;
  - `apply_multiband_motion_slab`;
  - `acquisition_from_dict` with an empty dict (which pins the default), and `Acquisition::hbcd`.

  Also run `pytest` (it must pass on `main` before anything changes).
- [ ] Run against unmodified `main`. Commit `tests/fixtures/resync_baseline/` (the CLI checksums and the
  Python hashes) on `resync-mrsim-acq`.
- [ ] Self-test: perturb one constant in `main`'s `kspace.rs` locally, confirm both baselines detect it, and
  revert.

## Task 2: mrsim-acq — Scanner partial Fourier

- [ ] Add `PartialFourierMode::Scanner` and `pf_skipped_lines`, and give Scanner its own branch in
  `sampling_mask`, ported from `main` as-is. The default stays `FiberfoxCompatible`.
- [ ] Shift the eddy clock by the skipped lines in `LineTiming::for_acquisition`, so the 2D, GRASE, spiral
  and gradient-echo 3D paths all see it. Fix `validate_acquisition_timing`'s message, which says partial
  Fourier lines are read last; under Scanner they are read first.
- [ ] Tests:
  - `main`'s Scanner test (the first lines skipped, the centre reached sooner, Contiguous timing unchanged);
  - the mask and timing equal `main`'s on a set of `(ny, pf)`;
  - the Scanner mask differs from Fiberfox's at `pf < 1` (the fall-through trap);
  - each mode is unchanged at `pf = 1`;
  - a GRASE run in Scanner mode reads the shifted clock.
- [ ] Gates: mrsim-acq tests (both feature sets), clippy, aslscan regress, TRXScan P0. Commit:
  `feat: scanner-style partial Fourier (late start)`.

## Task 3: mrsim-acq — EPI timing and the slice capture

- [ ] Add `EpiTiming`, `epi_timing` and `epi_trajectory`, consistent with `LineTiming` (the test asserts
  agreement).
- [ ] Add `SliceCapture`, `SliceRecon` and `simulate_slice_full`, split at `reconstruct_coils`.
  `simulate_slice` becomes a wrapper; its arithmetic is unchanged.
- [ ] Port `main`'s tests: `capturing_kspace_changes_no_arithmetic`,
  `captured_reconstructed_kspace_inverts_to_combined`,
  `acquired_kspace_is_pre_grappa_and_mask_is_the_sampling_mask`, and the timing-trajectory agreement test.
- [ ] Gates as in Task 2. Commit: `feat: EPI timing and per-slice k-space capture`.

## Task 4: mrsim-acq — complex output and acquisition options

- [ ] Add `AcquisitionOptions`, `KspaceCapture`, `AcquisitionOutput` and `simulate_acquisition_complex`
  over mrsim-acq's general input. That input is a struct holding today's positional arguments (`T2Volume`,
  `t_inhom`, `eddy_drive`, `prep_drive`, `noise_sigma`, and `eddy_trace` until Task 7).
  `simulate_acquisition_oversampled` keeps its signature as a wrapper, so aslscan and the frozen P0 tree do
  not change.
- [ ] Slab support (`slice_z`/`nz_full`): the global `z` feeds the shot, the phase slice, the slice seed,
  `SliceInput.nz` and the noise-map key. Without a slab it is identical.
- [ ] `t_echo_per_volume`: clone `acq` per volume and run `validate_acquisition_timing` per volume (on `main`
  it runs once).
- [ ] Port `main`'s tests:
  - `simulate_acquisition_is_bit_identical_to_the_per_slice_loop`;
  - `a_single_slice_run_is_bit_identical_to_its_slice_of_the_full_run`;
  - `per_volume_te_equals_separate_runs`.

  Plus one new test: magnitude and phase from the complex output equal `simulate_acquisition_oversampled`'s
  bit for bit.
- [ ] Gates as in Task 2. Commit: `feat: complex output, k-space capture and slab options on the 2D entry point`.

## Task 5: mrsim-acq — slab multiband motion

- [ ] Add `apply_multiband_motion_slab`, written against `DropoutLaw` (the P0 change). `apply_multiband_motion`
  delegates to it with no slab, and keeps its signature. Keep `dropout_events`/`dropout_seed` out of
  mrsim-acq (they hard-code diffusion's `b < 50`).
- [ ] Port `multiband_slab_matches_the_full_volume_where_no_jump_crosses_the_edge`.
- [ ] Gates as in Task 2. Commit: `feat: multiband motion on a slab of slices`.
- [ ] Push mrsim-acq and tag it (`resync-api`) **with the user's go-ahead**, so TRXScan can pin it. Until
  then, Task 6 builds against the local path override.

## Task 6: TRXScan — re-point the moved modules (on `resync-mrsim-acq`)

- [ ] Add mrsim-acq as a git dependency at the Task 5 tag, with the local `[patch]` override for the gates.
  Forward features as `main` needs them:
  - TRXScan's `kspace` forwards to `mrsim-acq/kspace`, and `par` to `mrsim-acq/par`.
  - `io` forwards to `mrsim-acq/io` but never pulls it into the Python crate, which builds without `io`.
- [ ] Delete the moved modules and re-export from `lib.rs`. Replace `Grid::hires` with a std-only path that
  works without `io`: the old branch's `GridRaster` trait, or a TRXScan helper with `hires_grid`'s arithmetic.
- [ ] Port the old branch's re-pointing commits (`4aa20fd..12a9d09`) by hand where `main` rewrote the files:
  `trxscan.rs`, `compartments.rs`, `gnl.rs`, `raster.rs`, `benchmark.rs`, and the new `gre.rs`. Their
  content is the guide; do not cherry-pick blindly.
- [ ] The TRXScan facade (`src/acq.rs`, re-exported under the `kspace`/`motion` paths the bindings use, so
  they change least):
  - `default_acquisition()` (Scanner) and `hbcd_acquisition(ny)`. Replace every `Acquisition::default()` and
    `Acquisition::hbcd` in TRXScan: the CLI, the Python bindings (`acquisition_from_dict`'s base), the
    benchmark bins and the tests. A test asserts `default_acquisition()` equals `main`'s default field by
    field;
  - `SimulationInput` and `simulate_acquisition` / `simulate_acquisition_complex`, translating `bvals`/`bvecs`
    to `eddy_drive`/`prep_drive`, `t2` to `T2Volume::Uniform`, and `DiffusionPhase` to `PrepPhase`;
  - `apply_multiband_motion` and `apply_multiband_motion_slab` with `main`'s `bvals, b_max` signatures,
    building the diffusion `DropoutLaw`; plus `dropout_events` and `dropout_seed`;
  - a timing pre-check in the Python entry points that raises `ValueError`;
  - `acquisition_from_dict` keeps `..base` with `base = default_acquisition()`.
- [ ] Gates: the TRXScan gates (Global constraints), clippy, aslscan regress and TRXScan P0, plus
  `grep -rn "Acquisition::default()\|Acquisition::hbcd" src python/src` finding nothing.
- [ ] Commits: one per mechanical step, as P0 did.

## Task 7: mrsim-acq — remove what main dropped

- [ ] Remove `SliceInput.eddy_lin`, the `eddy_trace` argument (from `simulate_acquisition_oversampled` and
  the Task 4 input struct), `BackgroundPhase.smooth`, `simulate_acquisition_legacy` (and its test) and
  `Nufft1::n`.
- [ ] Update the callers: aslscan (its `simulate_acquisition_oversampled` calls pass `eddy_trace: None`) and
  the TRXScan facade.
- [ ] Signed zero: `poly + 0.0` becomes `poly`, which differs only when `poly` is `-0.0`. The aslscan and
  TRXScan gates decide whether that reaches any output; `main` never had the `+ 0.0`, so TRXScan can only
  get closer to its baseline. If aslscan moves, keep the `+ 0.0` there and say why in a comment.
- [ ] Retire the TRXScan P0 gate: the frozen tree no longer compiles. Record that in Measurements.
- [ ] Gates: aslscan regress and the TRXScan gates. Commits in each crate: `refactor: drop eddy replay, the
  smooth background phase and the legacy path, as TRXScan main did`.

## Task 8: Reviews, docs and landing

- [ ] READMEs: mrsim-acq gains the new capabilities (Scanner partial Fourier, capture, complex output,
  slabs). TRXScan's README and CLAUDE.md say where the acquisition stage lives and how to build against a
  local mrsim-acq (the `[patch]` override).
- [ ] Codex adversarial review of the range in both crates (credits permitting; otherwise a Claude one),
  verified and fixed; then an ordinary review.
- [ ] Final gates on the final heads.
- [ ] With the user's go-ahead:
  - push mrsim-acq, with a tag for the final API, and repin TRXScan to it;
  - push `resync-mrsim-acq` and open a PR to TRXScan `main` for Matt's review; GitHub CI (`ci.yml`,
    `python.yml`) must pass on the PR;
  - after it merges, ask whether to delete the old remote branch `p0-mrsim-acq-extraction`.

## Acceptance criteria

- TRXScan `main` + this branch: every CLI case and every Python hash in the Task 1 baseline identical, under
  both feature sets; `pytest` and `main`'s CI steps pass, locally and on the PR.
- aslscan regress identical against `p7-complete`. mrsim-acq tests pass under both feature sets. No new
  clippy warnings.
- mrsim-acq carries no TRXScan-specific code (no `b < 50`, no `hbcd`), and `main`'s new capabilities are
  available to aslscan.
- The one deliberate difference (refusing `trf ≤ 0`) is documented in TRXScan's README and the PR.

## Risks

- **TRXScan `main` keeps moving.** Rebase `resync-mrsim-acq` before Task 6's gate. Re-run Task 1 if a new
  commit touches the moved modules (check with `git diff --stat <baseline>..origin/main -- src/kspace.rs
  src/motion.rs src/phase.rs src/nufft.rs src/io.rs`) or the Python bindings.
- **No TRXScan baseline check between Task 1 and Task 6.** The P0 gate covers the snapshot-era paths, and
  the ported bit-identity tests cover the new ones. A Task 6 failure is bisected with the `pre-resync` tag.
- **Signed zero** (Task 7) and the **Scanner eddy clock in 3D** (Task 2) are the two places a port could move
  bits quietly. The gates and the named tests cover them.
- **Whether aslscan should adopt Scanner partial Fourier** is a behavior change for aslscan, out of scope
  here. It would be its own phase.

## Measurements

**Task 0 (2026-10-08).**
- TRXScan worktree `../TRXScan-resync` on `resync-mrsim-acq` from `origin/main` `acb7506`. Its upstream
  is unset, so a stray push cannot reach `main`. mrsim-acq tagged `pre-resync` at `d839703`.
- Python environment: the micromamba env `rust-trx`, created for this work (user request):
  - python 3.12, maturin 1.15, pytest, numpy, scipy, nibabel, pooch, dipy and patchelf from conda-forge;
  - trx-python from pip.
- TRXScan P0 gate at the start: 62 of 62.
- `main`'s CI matrix before any change:
  - `cargo test`: 134;
  - `--features cli --all-targets`: 141;
  - `--features cli,par --all-targets`: 141;
  - `--features kspace`: 137 passed, 2 ignored;
  - `pytest scripts`: 73 passed, 1 skipped;
  - `pytest` in `python/`: 48 passed, 1 skipped, 4 deselected;
  - clippy: 57 warnings.

**Task 1 (TRXScan `1a5414d`).**
- `tools/run_resync_baseline.sh` covers the P0 fixture through the CLI under both feature sets: the P0
  cases (with "legacy" renamed `o1`), the three `--pf-mode`s, a noise map, `--gre-out` and `--gnl`. That
  is 162 checksums, identical across two runs.
- `tools/resync_python_baseline.py` gives 158 hashes, identical across two runs.
- Self-tests:
  - `hbcd`'s `ghost_offset` 0.015 → 0.016 changes 44 CLI checksums (every magnitude and phase image) and
    the `hbcd` hash.
  - The default `t_inhom` + 1 ms changes 34 Python hashes (every `simulate` case).
  - Both were restored, and the restored build reproduces the baselines.

**Tasks 2-5 (mrsim-acq `2304c06`, `be1dc2c`, `7ffb1d2`, `df447f6`).**
- TRXScan P0 62 of 62 after each.
- aslscan regress at `2304c06`: 67 of 67 identical.
- mrsim-acq tests: 121 default and 146 under `io,kspace,par` at `df447f6`. Clippy is at baseline.

**Task 6 (TRXScan `bf64888`, `3379956`, `507aaf3`, `8efbd91`, `d929d79`).**
- Every step: CLI 162 of 162 and Python 158 of 158 identical.
- At the end:
  - `cargo test`: 68 / 75 (`cli`) / 75 (`cli,par`) / 68 (`kspace`). The moved modules' tests now run in
    mrsim-acq.
  - pytest 48 passed; `pytest scripts` 73 passed.
  - Clippy: 35 warnings. Those in the touched files are code carried over verbatim.

**Task 7 (mrsim-acq `5561ecf`, aslscan `8291945`, TRXScan `ef56b0d`).**
- TRXScan baselines identical.
- The golden record's six "everything" files were re-recorded, because their eddy shear is gone. Every
  other golden file is bit-identical, and the six eddy-trace files were removed.
- The TRXScan P0 gate is retired here: the frozen tree calls the removed legacy path.

**aslscan regress gates.**
- `2304c06` (Task 2): 67 of 67.
- `df447f6` (Tasks 3-5): 67 of 67. The snapshot read mrsim-acq as `+local`, with CRLF-only differences
  from a Windows-git checkout; see Traps.
- aslscan `8291945` + mrsim-acq `b45a719` (Task 7): 67 of 67.
- Final, aslscan `8291945` + mrsim-acq `68785c4` (after the review's fixes): 67 of 67.

**Task 8: the Claude adversarial review of the implementation (2026-10-08; Codex out of credits).**
- Eight findings, each verified against the source. Fixed in mrsim-acq `f796d43` and TRXScan `de8cf2c`,
  `7328e0e` and `22f038e`:
  1. The Python timing pre-check that Task 6 required was missing; the acquisition panicked instead.
     Now `ValueError`, per volume with `te_per_volume`, with a test. It also showed the reach of the
     refusal: Python's default protocol (TE 90 ms, 1 ms per line) is refused from about 180 lines.
  2. The facade's `b_max` refusal was a second behaviour difference, reachable from Python (a nominal
     `b_max` with jittered b-values). Fixed exactly: mrsim-acq gained `DropoutLaw::ScaledTo` with an
     explicit normaliser, and the facade passes the caller's `b_max`. No refusal is left.
  3. The aslscan gates after Task 2 were unrecorded (recorded above).
  4. The literal-sum oracle had lost Scanner coverage. It now applies Scanner's shifted eddy clock and
     has two Scanner + eddy cases.
  5. One comparison in the per-slice-loop test was a tautology. Its doc now says so.
  6. Facade asserts on the lengths of `bvals`/`bvecs`, where `main` accepted longer arrays. Replaced by
     reading the first `ngrad`, as `main` did.
  7. Stale `Acquisition::hbcd`/`default()` references in TRXScan's docs and Python comments; the
     dependency and README items are left for landing.
  8. Baseline coverage. The baselines were regenerated from `main` in a temporary worktree and extended:
     the `trxscan-benchmark` binary, a multi-slice, non-sorted, multi-coil capture, and the
     jittered-`b_max` dropout. That is CLI 594 and Python 177 entries, all identical on this branch.
- Left as is: mrsim-acq's own `t2.len() == ncomp` assert, which comes from P0.

**Deviations from the plan.**
- **Scanner partial Fourier is refused on GRASE** (Task 2), instead of GRASE reading the shifted clock.
  The skip is defined by the 2D EPI train's line order, which a GRASE echo block does not follow. GRASE
  also never reads the eddy clock. The 3D gradient-echo and spiral paths already refuse partial Fourier.
- **mrsim-acq is a path dependency during the work** (Task 6). The commits are unpushed, so a git
  dependency cannot resolve yet. It is pinned to a tag at landing (Task 8).
- **The timing refusal reached `main`'s own tests** (Task 6).
  - `main`'s Python clean b0 runs with relaxation on at the protocol's TE. Two tests use TE 0, so their
    clean b0 started its readout before the excitation, which mrsim-acq refuses.
  - The user chose to keep the refusal. The two tests take the default TE instead (TRXScan `507aaf3`).
    Their acquisitions have relaxation and distortion off, so TE has no effect on what they check.
  - A gate on relaxation alone was written and reverted. It would not have saved these tests.
- **The motion facade's `b_max`** (Task 6; superseded by the review's fix 2, `ScaledTo`). mrsim-acq's `DropoutLaw::Scaled` normalises by the largest
  drive, while `main` divides by the caller's `b_max`.
  - `diffusion_dropout_law` appends `b_max` after the volumes' b-values (zero-padded to `ngrad`), which is
    exact whenever `b_max` is at least the largest b-value. Every caller satisfies this: the scheme's
    maximum, or 1000 for a b0-only Python scheme.
  - `b_max <= 0` attenuates nothing, as on `main`.
  - A smaller `b_max` is refused. A test pins the law against `main`'s formula bit for bit.
- **Two `main` `SliceInput`/`BackgroundPhase` literals** named mrsim-acq's `eddy_lin` and `smooth` from
  Task 6 until Task 7 removed them (`benchmark.rs`).

**Traps.**
- **The regress harness's base worktree.**
  - The P7 cleanup ran `git worktree prune`, which unregistered `../p5-base/aslscan` but left the
    directory, so the harness refused it.
  - Removing the directory let the harness recreate it. Its targets in `../p5-base/target-*` survive.
- **CRLF from Windows git.** A `git checkout --` from Windows git rewrote `kspace.rs` with CRLF. WSL git,
  which the regress harness uses, saw that as a local change ("`+local`" in the Tasks 3-5 gate's
  snapshot). Restore files with WSL git.

## Claude adversarial review of this plan (2026-10-08)

Run while Codex was out of credits. It found 4 majors and 6 minors, each verified against the source, and all
were applied above.

**Majors:**

- **R1. The removals broke the only diffusion-shaped gate.** The plan removed `eddy_lin`, `smooth` and the
  legacy path first, then said to re-run the P0 gate. But the frozen P0 tree (`12a9d09`) calls
  `simulate_acquisition_legacy` and passes `eddy_trace`/`eddy_lin`, so it would not compile, and Tasks 2-5
  would have run with no TRXScan check at all. Fixed: the additive tasks come first, under the P0 gate; the
  removals come after the re-point (Task 7), when `main`'s baseline guards TRXScan.
- **R2. `Acquisition::default()` would have changed TRXScan's defaults silently.** After the re-point,
  `Acquisition` is mrsim-acq's type, whose default is Fiberfox where `main`'s is Scanner. Python's
  `acquisition_from_dict` and the CLI start from `Acquisition::default()`, and `Default` cannot be overridden
  on a foreign type. Fixed: `default_acquisition()` in TRXScan, every call replaced, a grep in the gate, and
  a field-by-field test.
- **R3. A path dependency would break TRXScan's CI, wheels and release.** `main` pins `trx-rs`/`odx-rs` as git
  dependencies. `python.yml`/`release.yml` check out only TRXScan, and maturin-action builds in Docker. Fixed:
  mrsim-acq (public) as a pinned git dependency, a local `[patch]` for the gates, a tagged push before the PR,
  and CI passing on the PR.
- **R4. The facade missed the motion API.** The bindings call `apply_multiband_motion_slab(..., &bvals,
  b_max, ...)` with `main`'s signature, while mrsim-acq takes a `DropoutLaw`. Fixed: facade wrappers for both
  motion functions.

**Minors:**

- **r1.** The squash was described as cut "at an earlier point than the snapshot". `57858a5` (the snapshot's
  last GNL commit) has the features, so the squash left them out. Corrected.
- **r2.** mrsim-acq's P0 timing check panics where `main` simulates (`trf ≤ 0`). It is now listed as the one
  deliberate difference, with a Python `ValueError` and no baseline case in that regime.
- **r3.** `main`'s `pytest` suite (`python/tests/`) and CI steps were not in the gates. Added.
- **r4.** The P0 script's `legacy` case runs the oversampled path on `main`. Renamed and kept.
- **r5.** mrsim-acq's `sampling_mask` falls through to the Fiberfox rule for anything not `Contiguous`. A
  test now asserts that the Scanner mask differs.
- **r6.** Python builds without `io`, so `mrsim_acq::io::hires_grid` is unavailable there. `Grid::hires`'s
  replacement must be std-only.
