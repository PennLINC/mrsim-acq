# In-plane GRAPPA: a per-series reference scan, no reference band in the volumes, relative regularization

2026-10-16. Branch `grappa-series-kernel` (mrsim-acq from 5b4dd60, aslscan from 3faa1be). Written with
the code; the user chose the two fixes in part 2 after seeing the part 1 measurements.

## The problem

The aslprep-zoo comparison (`aslscan/work/zoo_compare`) found that, under
`ParallelReductionFactorInPlane = 2` (2D EPI and GRASE), aslscan's temporal noise did not behave like
thermal noise:

- the in-brain residual's lag-1 correlation along phase encode was about -0.4 (real data: +0.35 to +0.43);
- the in-brain tSD grew like `noise_variance^0.3` instead of `^0.5`;
- the tSD maps showed bands at FOV/2.

The comparison guessed the cause was the kernel being refitted on every (volume, slice)'s own noisy ACS
lines, which makes the kernel error vary from volume to volume.

## 1. What the measurements showed

Fitting the kernel once per series (on a reference scan) was implemented first and measured on two zoo
protocols, `morys` (2D EPI, 8 coils, 24 ACS lines, 7/8 partial Fourier) and `kashyap_lores` (3D GRASE),
at the calibrated noise variance and at 4 and 16 times it
(`aslscan/work/grappa/{run_exp.sh,analyze.py}` in the worktree):

| | PE lag-1, in-brain | PE lag-1, background (complex controls) | tSD exponent |
|---|---|---|---|
| morys, per-volume kernel / per-series kernel | -0.357 / -0.357 | -0.35 / -0.35 | 0.194 / 0.193 |
| kashyap_lores, per-volume / per-series | -0.440 / -0.440 | -0.44 / -0.44 | 0.345 / 0.345 |

Nothing changed, and the anticorrelation is as strong in the background's thermal noise, so it is not a
kernel-error-times-signal term. A toy series (32 x 32, 4 coils, R = 2; the tests'
`grappa_series_tests`) showed the same, and isolated two causes:

1. **The kernel was regularized by the noise.** The Tikhonov term was an absolute `1e-4` on the
   diagonal of `AᴴA`. At aslscan's signal scale `AᴴA` is many orders larger, so the fit was in effect
   unregularized, and the reference band's own noise did the regularizing: the larger the noise, the
   smaller the kernel's weak directions and the lower the g-factor. In the toy, `var/V` fell from 283
   to 9 as `V` rose over seven decades. Hence `V^0.3`.
2. **The reference band was kept in every volume.** Its 24 central lines carry the noise unamplified,
   and the synthesized lines carry it amplified by the kernel, so the noise power rises away from the
   centre of k-space. That is high-pass noise, a negative lag-1 correlation along phase encode. In the
   toy, synthesizing the centre too (no reference band in the volume) took the correlation from -0.23
   to +0.07.

Fitting per volume against per series changed the toy's temporal variance by 2-5% in every regime.

## 2. The design

**A reference scan per series.** The kernel of each slice is fitted once, on a reference scan, and
applied to every volume: the series' reconstruction is one linear map. The reference scan is the
calibration object read by the series' own readout without the acceleration (every PE line; partial
Fourier as the series has it), without spikes, without the per-volume drives (no eddy current, no
preparation phase, no physiology or shot weights), with thermal noise on its own streams (the seed
salted by `SEED_SALT_GRAPPA_CAL`). The kernel is fitted on its central `acs_lines`.

- API: `kspace::GrappaCalibration { images, seed }` (one simulation-grid volume per compartment), passed
  as `AcquisitionOptions::grappa_calibration` (2D) or the new last argument of
  `kspace3d::simulate_acquisition_3d_kz` (GRASE). `None` means the call's first volume and its seed.
  The 3D gradient-echo path always uses its first volume (`volume(0)`, without shot images).
- Per-volume echo times (`t_echo_per_volume`): one reference scan per distinct echo time, read at it,
  which is what separate runs at each echo time do (it keeps `per_volume_te_equals_separate_runs`).
  Multi-echo series (`simulate_acquisition_echoes`, the GE echoes) fit per echo.
- A lone slice (`simulate_slice`, `simulate_slice_full`) is its own series: its kernel comes from a
  reference scan of its own object, its noise on the slice seed salted by `SEED_SALT_GRAPPA_CAL`.
- aslscan: the reference object is the series' first control or label volume, else its first m0scan,
  else its first deltam (P8's rule, now `protocol::calibration_volume`), captured before motion; a
  calibration volume with no signal is an error naming the row. The separate M0 is its own series,
  with its own reference scan (mrsim-acq's default). The P6 and 3D-GE paths use the default (the first
  volume of each call).

**No reference band in the volumes.** `sampling_mask` under acceleration is every R-th line aligned on
the centre, nothing else. The kernel synthesizes every other line, the centre of k-space included; the
lines within `d` of the first or `R - d` of the last, whose sources fall off the matrix, stay zero (as
before). This is what a scanner's EPI does: the reference lines are a separate scan.

**Relative regularization.** `λ = GRAPPA_LAMBDA * mean diag(AᴴA)`, `GRAPPA_LAMBDA = 1e-3`, so the
kernel does not depend on the signal scale, and the reference scan's noise no longer regularizes it. In
the toy, `var/V` is flat to 2% over `V` = 1e-2 to 1e3 at 1e-3; 1e-2 lowers the g-factor further (8.7
against 12.0) without changing the scaling. 1e-3 is the smaller change to the image. A reference scan
with no signal is refused (an assertion in the fit).

## 3. Results

Same protocols, same noise levels, after both fixes:

| | PE lag-1, in-brain | tSD exponent | complex control variance exponent | in-brain tSD at x1 |
|---|---|---|---|---|
| morys, before / after | -0.357 / +0.125 | 0.194 / 0.499 | 0.404 / 0.999 | 41.1 / 9.31 |
| kashyap_lores, before / after | -0.440 / +0.027 | 0.345 / 0.498 | 0.784 / 0.995 | 6.10 / 1.44 |

The temporal noise is now the thermal noise through one linear map. The in-brain tSD at the zoo's
calibrated noise level fell about 4.3 times (the old kernel's g-factor at low noise was large), so the
zoo comparison's noise calibration (`noise-pass2.json`) must be redone. The in-brain over background
variance ratio (about 3 for morys, 2.5 for kashyap) is the g-factor's spatial pattern (ring coils: the
centre aliases onto the edge) and the background's lower signal.

## 4. Tests

- `kspace::grappa_series_tests::the_series_reconstruction_is_linear`: noise-free, volume `a + b` is
  volume `a` plus volume `b` and `2a` is twice `a` (explicit and default reference); per-volume kernels
  break it.
- `kspace::grappa_series_tests::the_temporal_noise_is_thermal`: `var/V` constant to 5% over `V` = 1 to
  1000 at signal scale 1e4, and |lag-1 PE correlation| < 0.1. Fails with `GRAPPA_LAMBDA = 1e-12` (`var/V`
  373 to 54).
- `complex_tests::simulate_acquisition_is_bit_identical_to_the_per_slice_loop`: the literal oracle now
  fits each slice's kernel from the reference scan itself.
- `kspace3d::tests::with_nothing_to_distinguish_partitions_3d_is_2d` still holds under R = 2 (the GRASE
  reference scan is a plan of the unaccelerated acquisition).
- The golden record (`tests/fixtures/p4_baseline/{default,kspace}`) was regenerated: exactly the 11
  accelerated cases changed (`*-grappa-coils`, `*-grappa3`, `*-everything`, `series-uniform`,
  `series-mapped`); the other 30 are bit-identical.
- aslscan `tools/regress_identity.sh 3faa1be 5b4dd60` (no case uses in-plane GRAPPA): see the hand-over.

## 5. TRXScan

TRXScan uses this path (`simulate_acquisition_complex`, the CLI's `--accel`, the Python `accel` and
`acs_lines`; its HBCD-like tests use R = 2). It pins mrsim-acq at tag `trxscan-resync`, so nothing changes
until it moves the pin. When it does:

- every GRAPPA output changes (and its R = 2 baselines);
- `AcquisitionOptions` gained a field, and `python/src/lib.rs` builds it with every field named, so it
  needs `..Default::default()`;
- the captured `mask`/`acquired` k-space no longer holds the reference band;
- `python/trxscan/kspace.py`'s ISMRMRD export flags lines within `acs_lines/2` of the centre as
  calibration lines; under this model they are imaging lines only, and the reference scan is not
  exported.

## 6. Rebase onto P10 (2026-10-19)

Both branches were rebased onto `main` after P10 merged (mrsim-acq 7468e79, aslscan 79d79ac), with no
conflicts. On the rebased build the part 3 measurements reproduce: PE lag-1 +0.129 / +0.030, tSD exponent
0.499 / 0.498, in-brain tSD at x1 9.31 / 1.44, against 41.1 / 6.10 on `main`. The aslscan PR carries the
figures. All four feature sets pass here; aslscan's `tools/regress_identity.sh 79d79ac 7468e79` is
identical on every case (the P8 spiral case, whose phantom is not in the worktree, was run by hand on the
same two binaries). GRASE at R = 2 now also simulates the reference scan: `kashyap_lores` 28 s -> 41 s.

## 7. Codex review of the PRs (2026-10-19)

Read-only `codex exec` over both diffs: six medium findings, no high or low. Each was checked against
the source.

| # | Finding | Verdict | Change |
|---|---|---|---|
| 1 | The solver's pivot floor (`1e-18` on the squared magnitude) is absolute, so with relative regularization a weak reference (a small signal scale, a faint deltam) zeroes the kernel | Valid | Both sides of the normal equations are divided by the mean diagonal before the solve, so the system is O(1) and the Tikhonov term is `GRAPPA_LAMBDA` itself. Test: the kernel at `1e-16` of the signal scale equals the full-scale one (it fails on the old solve) |
| 2 | The 2D reference carries the calibration volume's physiological tissue and label factors | Accurate, intended | A scanner's reference scan is acquired under physiology too, and a per-slice scale does not change a scale-free kernel. The aslscan comment that said "without physiology" is corrected |
| 3 | P6 (multi-echo, Hadamard, Look-Locker) took mrsim-acq's default, the first volume after motion and within-volume dropout, so a full dropout there would ruin that slice's kernel for the whole series (per-volume fitting had confined it to one volume), and an m0scan first won over a later control or label | Valid | P6 `build` captures the calibration raw volume (`calibration_volume` over the raw rows) before motion, refuses an empty one, and passes it per echo: `simulate_acquisition_echoes` takes a per-echo `GrappaCalibration`, 2D goes through `acquire_2d` and 3D through `simulate_acquisition_3d_kz` |
| 4 | The 3D gradient-echo reference was `volume(0)`: moved, with its shot gains | Valid | `simulate_acquisition_3d_ge` takes a `reference: Option<&GeVolume>`. aslscan builds it with `ge3d::reference_inputs`: the calibration raw volume unmoved, no shot images, gains of one, physiology kept as in 2D. Test: it is the still series' volume bit for bit |
| 5 | The fit treated lines that partial Fourier left out of the reference as measured zeros, and the apply synthesized a line next to the partial-Fourier edge from a missing source | Valid (pre-existing for the fit; the apply's reach grew with the reference band out of the volumes) | `fit` takes the reference's acquired lines and fits only triples it read entirely; `apply` takes the volume's acquired lines and leaves a line zero unless both its sources were acquired. Test: junk in the left-out lines does not change the kernel; a line with a missing source stays zero. The golden record's three `*-everything` cases (the only ones with partial Fourier and GRAPPA) change, in both builds |
| 6 | The library's empty-reference assertion runs after noise, so a pure-noise reference passes | Rejected as framed | A slice outside the object legitimately holds noise only, and a kernel fitted to it is what a scanner fits. The real defect beside it: with no noise (aslscan's default) such a slice panicked. An empty reference now gives a zero kernel (tested); aslscan refuses an empty calibration volume in every path (2D, GRASE, P6, 3D GE) |

After these changes, on the zoo protocols: `morys` (7/8 partial Fourier) PE lag-1 +0.166, tSD exponent
0.4995, in-brain tSD at x1 9.14 (was +0.129, 0.499, 9.31 in section 6: the line beside the
partial-Fourier edge now stays zero); `kashyap_lores` unchanged (+0.030, 0.498, 1.44). aslscan's
`tools/regress_identity.sh 79d79ac 7468e79` is identical on all 72 cases, the P8 spiral case included
(its phantom linked into the worktree), so the P6 and 3D GE paths' new entry points change nothing
without acceleration. All four mrsim-acq feature sets and both aslscan feature sets pass.
