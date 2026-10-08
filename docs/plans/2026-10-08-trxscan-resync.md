# TRXScan Re-sync with mrsim-acq: Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TRXScan `main` (currently `acb7506`) consume mrsim-acq again, without moving one TRXScan output
bit relative to `main` and without moving one aslscan output bit relative to `p7-complete`.

**Why.** The P0 extraction (`docs/plans/2026-09-21-p0-mrsim-acq-extraction.md`) moved TRXScan's acquisition
stage into mrsim-acq from a TRXScan snapshot, `05c76bf` (2026-09-23). Its TRXScan branch,
`p0-mrsim-acq-extraction` (`12a9d09`), was never merged. Since then, TRXScan `main` changed the moved modules:
- `ef4abe9` (2026-09-24) squashes the GNL work at an earlier point than the snapshot.
- `02ca507` (2026-09-28) adds the Python bindings.

A trial merge of the old branch into `main` conflicts in 19 files (modify/delete on `kspace.rs`, `motion.rs`
and `phase.rs`). The old branch also carries snapshot-only GNL work that `main` does not have.

**Architecture.** Redo the TRXScan side on a fresh branch from `main`. Port `main`'s acquisition changes into
mrsim-acq as additive, default-off capabilities. Keep TRXScan's public API (CLI, Python bindings) on a thin
TRXScan-side facade over mrsim-acq. Order of work:
1. Capture `main`'s own output as the baseline.
2. Grow mrsim-acq, gated on the aslscan regress run.
3. Re-point TRXScan, gated on the new baseline.

No task changes both TRXScan's code location and its behavior.

**Inputs.** The change inventory (2026-10-08, from `git diff 05c76bf origin/main` on the moved modules) is
summarized in "What main changed" below. The old branch's re-pointing commits, `4aa20fd..12a9d09`, are the
reference for the mechanical TRXScan-side work.

## Decisions already made

- **Snapshot-only pieces are removed from mrsim-acq to match `main`** (user, 2026-10-08):
  - `SliceInput.eddy_lin` with the `eddy_trace` entry argument (eddy replay);
  - `BackgroundPhase.smooth` (the smooth background-phase modes);
  - `simulate_acquisition_legacy` and its test;
  - `Nufft1::n`.

  Neither consumer uses them. They stay in git history and on the old branch. TRXScan-side snapshot-only
  features are simply not re-extracted: `--eddy-trace`, `--phase-bg-scale`, `--phase-smooth`,
  `--s0-map`/`--t2-map`, `--csf-scale-map`, and the MESE/MEGRE scripts.
- **A fresh branch from `main`, not a merge of the old branch.** Merging would resurrect the snapshot-only
  work and the pre-squash GNL history.
- **mrsim-acq keeps its general API** (the P0 interface changes: `T2Volume`, `eddy_drive`/`prep_drive`,
  `PrepPhase`, `DropoutLaw`, the echo formation). `main`'s diffusion-shaped API becomes a TRXScan facade:
  `SimulationInput` with `bvals`/`bvecs`/`t2`, `simulate_acquisition`, `simulate_acquisition_complex`,
  `Acquisition::hbcd` and `dropout_events`.
- **mrsim-acq's `Acquisition::default()` keeps `FiberfoxCompatible`.** aslscan, `kspace3d`, the spiral path
  and mrsim-acq's tests rely on it. TRXScan's facade sets `Scanner` where `main` defaults to it.

## What main changed (inventory)

Categories: A = API shape only; B = new capability beside existing paths; C = could change output bits for an
existing configuration; D = absent on `main`.

| Change on `main` | Cat | Port to |
|---|---|---|
| `PartialFourierMode::Scanner` (late start, skips the first lines), `pf_skipped_lines`, the Scanner branch in `sampling_mask`, and the eddy clock (`tread`) shifted by the skipped lines | B (C only in Scanner mode) | mrsim-acq, default-off. The `tread` shift must go into `LineTiming::for_acquisition`, so `kspace3d`, spiral and GRASE agree |
| `#[default] Scanner` on `Acquisition` and the CLI (`--pf-mode scanner`) | C | TRXScan facade only |
| `Acquisition::hbcd(ny)` preset | B | TRXScan facade, as a free function (the type is foreign) |
| `EpiTiming`, `epi_timing`, `epi_trajectory` | B | mrsim-acq, beside `LineTiming` |
| `SliceCapture`, `SliceRecon`, `simulate_slice_full` (`simulate_slice` becomes a wrapper) | B | mrsim-acq |
| `simulate_acquisition_complex`, `AcquisitionOptions` (`slice_z`/`nz_full`, `kspace_slices`, `capture`, `t_echo_per_volume`, `progress`), `KspaceCapture`, `AcquisitionOutput` | B (C only for slabs) | mrsim-acq, over its general entry point |
| `SimulationInput` and `simulate_acquisition` (replacing the positional entry point) | A | TRXScan facade |
| Noise-map RNG keyed on the global voxel (identical when not a slab) | A | mrsim-acq, with the options |
| `motion::apply_multiband_motion_slab`; `apply_multiband_motion` delegates to it | B (C only for slabs) | mrsim-acq, written with `DropoutLaw` |
| `motion::dropout_events` and `dropout_seed` (hard-codes `b < 50`) | A (moved from the bin) | TRXScan facade (diffusion-specific) |
| `eddy_lin`, `BackgroundPhase.smooth`, `simulate_acquisition_legacy`, `Nufft1::n` | D | Removed from mrsim-acq (decision above) |
| `io::write_gre_fieldmap`; `hires_grid` delegating to a new inherent `Grid::hires` | B / A | TRXScan. `Grid` is foreign there, so use `mrsim_acq::io::hires_grid` (identical arithmetic) or the old branch's `GridRaster` trait |

`readout.rs`, `noise.rs`, `mat.rs`, `orient.rs`, `analytic.rs` and `config.rs` are byte-identical between
the snapshot and `main`. The three newest `main` commits (`0b917e6`, `be68c24`, `acb7506`) touch no moved
module.

## Global constraints

- **Two identity gates, every task.**
  - aslscan: `tools/regress_identity.sh p7-complete p7-complete`, all cases identical (67 at P7).
  - TRXScan: from Task 7, its outputs byte-identical to the Task 1 baseline of `main`, under both
    `--features cli` and `--features cli,kspace,par`.
- **The old P0 checksums no longer describe TRXScan.** `main`'s CLI default moved from `contiguous` to
  `scanner` partial Fourier, so `main` itself differs from `05c76bf`. The Task 1 baseline replaces them.
  The TRXScan P0 gate (frozen worktree at `12a9d09`) stays a check on mrsim-acq's history only. Re-run it
  after Task 2 to confirm the removals.
- Hand-formatted source, about 100 columns. No repo-wide `cargo fmt`. Tests live in module `mod tests`.
- Clippy: no new warnings in either crate.
- Git: `-c core.autocrlf=false` for worktree add, commit and merge. Run gates from WSL-made detached
  worktrees.
- Nothing is pushed, and no PR is opened, without the user's go-ahead. TRXScan `main` is Matt Cieslak's
  branch: the re-sync lands there by PR, for his review.

---

## Task 0: Branches and worktrees

- [ ] TRXScan: create a worktree `../TRXScan-resync` on a new branch `resync-mrsim-acq` from `origin/main`
  (`acb7506`), made from WSL so both gits can read it.
- [ ] mrsim-acq: work on `main`, as in P1-P7.
- [ ] Record the heads in this plan's Measurements.

## Task 1: Baseline of TRXScan main

- [ ] Bring the P0 fixture tooling onto `resync-mrsim-acq`: `tools/gen_p0_fixture.py`,
  `tools/run_p0_baseline.sh` and `tests/fixtures/p0_baseline/` inputs (content from `05c76bf`; no source
  change).
- [ ] Extend the run matrix so it covers what `main` added or changed. Each case is one CLI invocation, under
  both feature sets:
  - partial Fourier `scanner` (the new default), `contiguous` and `fiberfox`;
  - multiband with motion and dropout;
  - a noise map;
  - `--gre-out` (the GRE fieldmap) and GNL ground truth;
  - `--eddy-phase`.
- [ ] Add a Python-bindings baseline: a script that builds the extension (maturin) and hashes the arrays from
  `simulate`, `simulate_acquisition_complex` with capture, `epi_timing`/`epi_trajectory` and
  `apply_multiband_motion_slab`. **Needs a micromamba environment with maturin and the bindings' Python
  deps: ask the user which one.**
- [ ] Run against unmodified `main`. Commit `tests/fixtures/resync_baseline/checksums.txt` and the Python
  hashes on `resync-mrsim-acq`.
- [ ] Self-test: perturb one constant in `main`'s `kspace.rs` locally, confirm both baselines detect it, and
  revert.

## Task 2: mrsim-acq — the removals (decision above)

- [ ] Remove `SliceInput.eddy_lin`, the `eddy_trace` argument of `simulate_acquisition_oversampled`,
  `BackgroundPhase.smooth`, `simulate_acquisition_legacy` (and its test) and `Nufft1::n`.
- [ ] Update aslscan's call sites for the dropped `eddy_trace` argument (about 11, all passing `None`).
- [ ] Signed zero: `poly + 0.0` becomes `poly`, which differs only when `poly` is `-0.0`. The aslscan regress
  gate decides whether that reaches any output. If it does, keep the `+ 0.0` and say why in a comment.
- [ ] Gates: mrsim-acq tests (both feature sets), clippy, aslscan regress, TRXScan P0 (frozen worktree).
- [ ] Commit in each crate: `refactor: drop the snapshot-only eddy replay, smooth background phase and legacy
  path`.

## Task 3: mrsim-acq — Scanner partial Fourier

- [ ] Add `PartialFourierMode::Scanner` and `pf_skipped_lines`, and add the Scanner branch to
  `sampling_mask`, ported from `main` as-is. The default stays `FiberfoxCompatible`.
- [ ] Shift the eddy clock by the skipped lines in `LineTiming::for_acquisition`, so the 2D, GRASE, spiral
  and gradient-echo 3D paths all see it. Fix `validate_acquisition_timing`'s message, which says partial
  Fourier lines are read last; under Scanner they are read first.
- [ ] Tests: `main`'s Scanner test (the first lines skipped, the centre reached sooner, Contiguous timing
  unchanged). The mask and timing equal `main`'s on a set of `(ny, pf)`. Each mode is unchanged with
  `pf = 1`. A GRASE run in Scanner mode reads the shifted clock.
- [ ] Gates as in Task 2. Commit: `feat: scanner-style partial Fourier (late start)`.

## Task 4: mrsim-acq — EPI timing and the slice capture

- [ ] Add `EpiTiming`, `epi_timing` and `epi_trajectory`, consistent with `LineTiming` (the test asserts
  agreement).
- [ ] Add `SliceCapture`, `SliceRecon` and `simulate_slice_full`, split at `reconstruct_coils`.
  `simulate_slice` becomes a wrapper; its arithmetic is unchanged.
- [ ] Port `main`'s tests: `capturing_kspace_changes_no_arithmetic`,
  `captured_reconstructed_kspace_inverts_to_combined`,
  `acquired_kspace_is_pre_grappa_and_mask_is_the_sampling_mask`, and the timing-trajectory agreement test.
- [ ] Gates. Commit: `feat: EPI timing and per-slice k-space capture`.

## Task 5: mrsim-acq — complex output and acquisition options

- [ ] Add `AcquisitionOptions`, `KspaceCapture`, `AcquisitionOutput` and `simulate_acquisition_complex`
  over mrsim-acq's general input. That input is a struct holding today's positional arguments (`T2Volume`,
  `t_inhom`, `eddy_drive`, `prep_drive`, `noise_sigma`). `simulate_acquisition_oversampled` stays as a
  wrapper, so aslscan does not change.
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
- [ ] Gates. Commit: `feat: complex output, k-space capture and slab options on the 2D entry point`.

## Task 6: mrsim-acq — slab multiband motion

- [ ] Add `apply_multiband_motion_slab`, written against `DropoutLaw` (the P0 change). `apply_multiband_motion`
  delegates to it with no slab. Keep `dropout_events`/`dropout_seed` out of mrsim-acq (they hard-code
  diffusion's `b < 50`).
- [ ] Port `multiband_slab_matches_the_full_volume_where_no_jump_crosses_the_edge`.
- [ ] Gates. Commit: `feat: multiband motion on a slab of slices`.

## Task 7: TRXScan — re-point the moved modules (on `resync-mrsim-acq`)

- [ ] Add mrsim-acq as a path dependency with forwarded features (`kspace`, `io`, `par`), as on the old
  branch.
- [ ] Delete the moved modules. Re-export from `lib.rs`. Replace `Grid::hires` with
  `mrsim_acq::io::hires_grid`, or the old branch's `GridRaster` trait.
- [ ] Port the old branch's re-pointing commits (`4aa20fd..12a9d09`) by hand where `main` rewrote the files:
  `trxscan.rs`, `compartments.rs`, `gnl.rs`, `raster.rs`, `benchmark.rs`, and the new `gre.rs` (it uses
  `kspace::Rng`). Their content is the guide; do not cherry-pick blindly.
- [ ] The TRXScan facade (`src/acq.rs`, or kept under the `kspace`/`motion` module names so the Python
  bindings change least):
  - `SimulationInput` and `simulate_acquisition` / `simulate_acquisition_complex`, translating `bvals`/`bvecs`
    to `eddy_drive`/`prep_drive`, `t2` to `T2Volume::Uniform`, and `DiffusionPhase` to `PrepPhase`;
  - an `hbcd_acquisition(ny)` free function, plus the `Scanner` default where `main` relies on it;
  - `dropout_events` and `dropout_seed`;
  - Python's `acquisition_from_dict` gains `..base`, because mrsim-acq's `Acquisition` has the `echo` field.
- [ ] Gates: TRXScan output identical to the Task 1 baseline (both feature sets), the Python hashes identical,
  `cargo test` in TRXScan (both feature sets), clippy, and the aslscan regress run.
- [ ] Commits: one per mechanical step, as P0 did.

## Task 8: Reviews, docs and landing

- [ ] READMEs: mrsim-acq gains the new capabilities (Scanner partial Fourier, capture, complex output,
  slabs). TRXScan's README and CLAUDE.md say where the acquisition stage lives.
- [ ] Codex adversarial review of the range in both crates, verified and fixed; then an ordinary Codex review.
- [ ] Final gates on the final heads.
- [ ] With the user's go-ahead:
  - push mrsim-acq;
  - push `resync-mrsim-acq` and open a PR to TRXScan `main` for Matt's review;
  - after it merges, ask whether to delete the old remote branch `p0-mrsim-acq-extraction`.

## Acceptance criteria

- TRXScan `main` + this branch: every CLI case and every Python hash in the Task 1 baseline identical, under
  both feature sets.
- aslscan regress identical against `p7-complete`. mrsim-acq and TRXScan tests pass under both feature sets.
  No new clippy warnings.
- mrsim-acq carries no TRXScan-specific code (no `b < 50`, no `hbcd`), and `main`'s new capabilities are
  available to aslscan.

## Risks

- **TRXScan `main` keeps moving.** Rebase `resync-mrsim-acq` before Task 7's gate. Re-run Task 1 if a new
  commit touches the moved modules (check with `git diff --stat <baseline>..origin/main -- src/kspace.rs
  src/motion.rs src/phase.rs src/nufft.rs src/io.rs`).
- **The Python baseline** depends on an environment with maturin; without one, the bindings are checked only
  by compiling.
- **Signed zero** (Task 2) and the **Scanner eddy clock in 3D** (Task 3) are the two places a port could move
  bits quietly. The gates and the named tests cover them.
- **Whether aslscan should adopt Scanner partial Fourier** is a behavior change for aslscan, out of scope
  here. It would be its own phase.

## Measurements

(filled in during implementation)
