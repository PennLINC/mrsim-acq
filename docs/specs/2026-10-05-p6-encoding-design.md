# P6: Hadamard time-encoded labeling, Look-Locker readouts and multi-TE ASL

Design spec addendum, 2026-10-05. Extends `2026-09-21-mrsim-acq-aslscan-design.md` (the main
spec) and follows the P2-P5 addenda (`2026-09-24-p2-asldro-compat-design.md`,
`2026-09-24-p3-motion-bs-ir-design.md`, `2026-10-01-p4-vascular-physio-design.md`,
`2026-10-01-p5-3d-readouts-ge-design.md`). Where this document is silent, those stand. Not yet
reviewed.

## Context

The roadmap gives P6 "Encoding variety: Hadamard, Look-Locker, velocity-selective, multi-TE"
(main spec, sub-project table), depending only on P1. Three of the four are in scope here;
**velocity-selective labeling is deferred** (user decision, 2026-10-05): BIDS defines no VSASL
type (`ArterialSpinLabelingType` is `CASL`, `PCASL` or `PASL` in the released specification
1.11.1, the validator's schema 1.2.7 and the specification's `master`), no BEP or open issue
proposes one, and the ASL-BIDS paper (Clement et al., Sci Data 2022) lists velocity-selective and
time-encoded ASL among the approaches a future release may add. A VSASL dataset would therefore
either fail the validator or be misdescribed as PASL. See "Decisions deferred".

What exists today, verified against the source (`aslscan` at `715bcb0`, `mrsim-acq` at
`b340133`):

- **A row is one labeling and one readout.** `Row { kind, t, tau, tr }` (`rows.rs:8-38`) carries
  one kinetic time, one bolus duration and one repetition time; `RowKind` is
  `{M0scan, Control, Label, Deltam}`; the clock `row_start` advances by one `tr` per row (by
  `NumberShots x tr` for a segmented 3D volume, `protocol.rs:1898-1909`). The blood image of a
  row is one signed delta-M (`blood_sign`, `series.rs:187-203`: `-1` label, `+1` deltam).
- **The only multi-bolus primitive is P4's sub-bolus.** `delta_m_sub(k, .., t, a, b)`
  (`kinetic.rs:128-137`) is the delta-M of the parcels labeled during `[a, b]` of `[0, tau]`,
  with a partition identity tested to `1e-12` (`kinetic.rs:556-591`); its intravascular twin is
  `delta_m_iv_sub` (`:174`). `bolus::subbolus_factors` (`bolus.rs:61-72`) produces factors only
  from background-suppression pulses. Nothing produces a 0/1 encoding pattern.
- **One readout per labeling, one excitation per slice per row.** The tissue timeline
  (`longitudinal.rs:54-74`, `tissue_mz_ge` `:83-100`, `tissue_mz_ge_sequence` `:117-140`) is
  affine in the starting `Mz`, with events that scale `Mz` (inversions, `1 - 2 eps`) and one
  readout at `t_read`; blood under gradient echo is `sin(a) dm` with no depletion by earlier
  readouts (`mrsignal.rs:106`). Pulses must precede the first readout (`protocol.rs:1694-1701`).
- **One echo time per series.** `echo_time_s` is a scalar; an unequal `EchoTime` array is an
  error naming P6 (`protocol.rs:1208-1213`); `Acquisition.t_echo` is one value per acquisition
  call (`kspace.rs:148`). `LookLocker: true` is an error naming P6 (`protocol.rs:1023-1025`). The
  compat translator refuses a varying `echo_time` (`tools/compat_asldro.py:357-358`).
- **In `mrsim-acq`, `t_echo` enters only `trf`.** It is read by `LineTiming::for_acquisition`
  (`trf = t_echo + t`, `kspace.rs:251-296`), by the gradient-echo static phase
  `2 pi fmap TE` (`kspace.rs:636-641`) and by the timing check (`:331-350`); the line times `t`,
  the NUFFT affinity check and the rotors do not depend on it. Noise and spikes are keyed on
  `(volume, slice)` within one call plus the call's `seed` (`slice_seed`, `kspace.rs:1485-1489`);
  the prepared-shot phase uses the same `seed` (`phase.rs:121-138`).

What BIDS settles (schema 1.2.7, `rules/sidecars/asl.yaml`, `rules/checks/asl.yaml`):

- **Look-Locker**: `LookLocker` (boolean, optional) true makes `FlipAngle` required
  (`MRIFlipAngleLookLockerTrue`), here and in `*_m0scan.json`; `FlipAngle` may be an array, one
  value per volume, whose length must equal both the NIfTI `dim[4]` and the `aslcontext.tsv` rows
  (an error). `PostLabelingDelay` is an array "for multi-PLD and Look-Locker", one value per
  volume in acquisition order, `0` for `m0scan`, length-checked as an error. Each Look-Locker
  readout is its own volume and its own `aslcontext.tsv` row. `RepetitionTimeExcitation` is not an
  ASL key.
- **Multi-TE**: the `echo` entity is allowed on `_asl`, `_m0scan` and `_noRF` files (since
  specification PR #1884); `aslcontext.tsv` takes no `echo` entity, so one `sub-X_aslcontext.tsv`
  is inherited by every `echo-N` series. Each series has a scalar `EchoTime`.
- **Hadamard / time-encoded**: no representation. The raw encoded volumes are neither `control`
  nor `label`. What validates is the **decoded** form: one `deltam` volume per sub-bolus, with
  `PostLabelingDelay` and `LabelingDuration` arrays giving each sub-bolus's effective delay and
  duration (both length-checked as errors). `aslcontext.tsv`'s `volume_type` enum is `control`,
  `label`, `m0scan`, `deltam`, `cbf`, `noRF`.

simasl (ASLDRO v2.2.0) implements none of the three: one Buxton evaluation per series
(`examples.py:125-150`), `pasl`/`casl`/`pcasl` only (`filters/gkm_filter.py:95-97`), no
Look-Locker or encoding. It does carry a per-volume `echo_time` list
(`validators/user_parameter_input.py:243-249`, applied at `examples.py:199`) with a
single-compartment `exp(-TE/T2)`; that is the one P6 case it can check (part C, "compat").

## The one idea underneath

P1-P5 identify a volume with **one labeling followed by one readout**. All three P6 features
break that identity in a different direction, and each is a re-indexing rather than new physics:

- **Hadamard** puts several labelings (sub-boli, each on or off) in **one** volume. Kinetics are
  linear in the labeling, so an encoded volume's blood image is a 0/1-weighted sum of P4
  sub-bolus delta-Ms, which already exist (`delta_m_sub`), and decoding is a fixed linear
  combination of reconstructed volumes.
- **Look-Locker** puts several readouts after **one** labeling. Each readout is an event on the
  existing affine `Mz` timeline (it scales `Mz` by `cos(a)` exactly as an inversion scales it by
  `1 - 2 eps`), and the label it reads has been depleted by the readouts before it, which splits
  the kinetic integral by arrival window.
- **Multi-TE** reads several echoes after **one** excitation. The compartment images are the same
  for every echo; only the acquisition's `t_echo` changes, and the blood and tissue compartments
  already decay with their own T2 (and P4's exchange already splits delta-M into intravascular
  and extravascular parts, which is the two-compartment multi-TE model of Gregori et al.).

So the new structure is a **labeling cycle** (`aslscan`): one labeling (possibly encoded) and one
or more readouts (Look-Locker), each readout producing one volume per echo (multi-TE). Every
existing protocol is a cycle with one unencoded labeling, one readout and one echo, and keeps its
bytes.

## Scope

In: Hadamard time-encoded (P)CASL for 2D and 3D readouts (part A); Look-Locker readouts for 2D EPI
under gradient-echo excitation (part B); multi-TE for 2D EPI, spin echo and gradient echo, with the
`echo-N` layout (part C); the combination Hadamard x multi-TE. Out, with reasons under "Decisions
deferred": velocity-selective labeling; Look-Locker with 3D readouts; multi-TE with 3D readouts;
Look-Locker x Hadamard and Look-Locker x multi-TE; vessel-encoded labeling; Walsh-ordered or
non-Sylvester encoding matrices; the arterial (QUASAR) compartment under Look-Locker.

**Byte identity.** With no P6 input, every output is unchanged, in the three places P5 established:

- **TRXScan** (`p0-mrsim-acq-extraction`, now published at `12a9d09`): the P0 gate
  (`tools/run_p0_baseline.sh`, diff against `tests/fixtures/p0_baseline/checksums.txt`, both
  feature sets). P6 adds **no field** to `Acquisition` or `SliceInput` and changes no existing
  signature, so TRXScan needs no source change: both structs are built with every field named at
  `trxscan.rs:828` and in TRXScan's benchmark and test literals, which a new field would break.
- **mrsim-acq**: the golden record (`tests/golden_baseline.rs`, `tests/fixtures/p4_baseline/`) and
  every bit-pinned test. The one new entry point (part C) calls the existing per-volume code with a
  per-echo `Acquisition` clone; with one echo equal to the series `t_echo` and the echo index 0 it
  computes exactly what `simulate_acquisition_oversampled` computes, and a test pins that bit for
  bit.
- **aslscan**: `tools/regress_identity.sh` against `p5-complete` with `mrsim-acq` at its
  `p5-complete`, both feature sets, all 19 cases. The default path builds one-cycle rows exactly as
  today; a test pins that the cycle structure of every existing fixture is "one labeling, one
  readout, one echo" and that `simulate` takes the existing code path for it.

**Activation and refusals** (the P4 rule): Hadamard by an overlay `[hadamard]` table, Look-Locker
by `LookLocker: true`, multi-TE by more than one `--asl-json`. Every P6 input belonging to a feature
that is off is an error naming it, and every refused combination is an error naming both features.

---

# Part A: Hadamard time-encoded labeling

## What is modeled

A (P)CASL labeling of total duration `tau_tot = sum_j tau_j` divided into `N` consecutive
sub-boli, `j = 1..N` in time order (sub-bolus 1 is labeled first, so it is the oldest at the
readout), followed by a delay `PLD` to the readout. An **encoding cycle** is `N + 1` acquisitions;
acquisition `i` labels sub-bolus `j` when `h_ij = -1` and controls it when `h_ij = +1`, where `h`
is the Sylvester Hadamard matrix of order `N + 1` (a power of two: 4, 8, 16, 32) with its all-ones
first column removed (Dai et al., MRM 2013; Teeuwisse et al., MRM 2014). The overlay gives `N + 1`.

**Kinetics.** Sub-bolus `j` occupies `[a_j, b_j]` of `[0, tau_tot]` (`a_1 = 0`,
`b_j = a_j + tau_j`). The labeled parcels of acquisition `i` are the union of the sub-boli it
labels, so its blood delta-M is

```
dM_i(t) = sum_j w_ij * dM_sub(t; a_j, b_j),     w_ij = (1 - h_ij) / 2  in {0, 1}
```

with `dM_sub` P4's `delta_m_sub` (and, under exchange, `delta_m_iv_sub` for the intravascular
part), evaluated at the readout time `t = tau_tot + PLD` plus the slice offset. This is exact for
the linear kinetic model; the partition identity (`kinetic.rs:556-591`) already guarantees that the
all-labeled acquisition equals the unencoded bolus of duration `tau_tot`. The tissue of every
encoded acquisition is the control tissue (the labeling RF saturates nothing in the imaged slices,
the P1 assumption). The raw acquisition's image is `tissue - dM_i` (the label sign convention).

**PASL is refused** with Hadamard: time encoding needs a labeling that can be switched on and off
during the bolus (continuous or pseudo-continuous labeling). CASL and PCASL are allowed.

**Decoding.** The columns of `h` sum to zero and are orthogonal (`sum_i h_ij h_ik = (N+1) delta_jk`),
so with `S_i = C - sum_j w_ij dM_j`,

```
D_j = (2 / (N+1)) * sum_i h_ij * S_i = dM_j
```

exactly, for any common `C`. aslscan decodes the **reconstructed complex images** (before the
magnitude), per cycle, voxel by voxel, in `f64`: linear and exact when noise is off. The noise of a
decoded image is that of one raw image times `2 / sqrt(N+1)` (measured and recorded, not asserted:
the reconstruction's own noise transfer is unchanged).

## Inputs

The input describes the dataset that will be written, the **decoded** form, plus the encoding:

- `aslcontext.tsv`: `deltam` rows, `N` per cycle, in sub-bolus order `j = 1..N`, any number of
  cycles, and `m0scan` rows anywhere. `control` and `label` rows are refused with Hadamard (the raw
  encoded volumes are generated, not listed).
- `LabelingDuration`: an array, one per row, `tau_j` for the `deltam` rows (the same per cycle),
  `0` for `m0scan`; a scalar means equal sub-boli.
- `PostLabelingDelay`: an array, one per row, the **effective** delay of each sub-bolus,
  `PLD_j = PLD + sum_{k>j} tau_k` (the time from the end of sub-bolus `j` to the readout, BIDS's
  definition applied per sub-bolus). aslscan recovers `PLD = PLD_N` and checks every `PLD_j`
  against the formula to `1e-6` s, naming the first mismatch.
- `RepetitionTimePreparation`: the raw acquisition's repetition (one per raw acquisition, so a
  scalar or an array over the `deltam` rows constant within a cycle).
- Overlay `[hadamard]`: `order = N + 1` (required; 4, 8, 16 or 32), and nothing else. `N` must
  equal the number of `deltam` rows per cycle; the `deltam` count must be a multiple of `N`.

Each cycle is `N + 1` raw acquisitions, each one repetition, in matrix-row order; the clock
`row_start` advances by one `tr` per raw acquisition. A raw acquisition's readout window is checked
as today (`t + max slice offset <= tr`, `t = tau_tot + PLD`).

## The series

Rows become **raw acquisitions**: a new `RowKind::Encoded { cycle, i }` with `t = tau_tot + PLD`,
`tau = tau_tot`, the cycle's `tr`, and the encoding row `w_i`. The blood image of an encoded row is
`-sum_j w_ij dM_sub(.; a_j, b_j)`, through the same per-voxel path as a label row's (P4's label
path when exchange, bolus-position suppression, physiology or the arterial compartment is on, with
the arterial term also summed over the labeled sub-boli; the P1 path otherwise). `m0scan` rows are
unchanged and are carried into both the raw and the decoded series at their positions.

Everything per volume applies to the **raw** acquisitions, which are what the scanner acquires:
noise and spikes, motion poses and shot events, physiological factors (each raw acquisition has
its own labeling window `[row_start, row_start + tau_tot]`), background suppression (pulses timed
from the start of labeling, so they fall after `tau_tot` under the global model; the bolus-position
model applies its factors per sub-bolus parcel as P4 does). The one acquisition call simulates all
raw volumes.

Decoding follows the call: per cycle, `D_j = (2/(N+1)) sum_i h_ij S_i` on the complex images
(`simulate_acquisition_oversampled` returns magnitude and phase in `f32`; the series converts back
to complex, as `complex_from` already does, and decodes in `f64`). The decoded volumes, in
sub-bolus order, with the `m0scan` volumes at their input positions, are the main output.

**Ground truth**: `desc-deltam_gt` holds, per decoded volume, the true `dM_j` (the sum of its
parcels, at its own timing), and the P4 ground truths likewise per sub-bolus. The raw series'
ground truth (per raw acquisition, the encoded sum) is written beside the raw data.

## Outputs

- **Main dataset** (validates): the decoded series, `aslcontext.tsv` as input, the input
  `PostLabelingDelay` and `LabelingDuration` arrays, `TotalAcquiredPairs` the `deltam` count (the
  existing rule), and `AslscanSimulation.Hadamard`: the order, the matrix (rows of `+1/-1`), the
  sub-bolus boundaries `[a_j, b_j]`, `PLD`, the raw volume count, the decoding rule, and the noise
  scaling `2/sqrt(N+1)`.
- **Raw series** under `sourcedata/sub-X/[ses-Y/]perf/`: the `N + 1` raw volumes per cycle (plus
  the `m0scan` volumes), `part-mag`/`part-phase`, a sidecar carrying the matrix, the raw volume
  order and each raw volume's labeled sub-boli, and a `sourcedata` `aslcontext`-like TSV with a
  column `encoding_row` (`sourcedata` is not validated; the file names follow BIDS so that tools
  find them). The raw ground truth sits beside it.

## Refusals

PASL; `control`/`label` rows; `order` not a power of two in `[4, 32]`; inconsistent effective PLDs;
`LabelingDuration` varying across cycles; `[hadamard]` with `LookLocker: true` (deferred); compat
(simasl has no encoding).

---

# Part B: Look-Locker readouts (2D EPI)

## What is modeled

After one labeling, `M` readouts at times `t_1 < ... < t_M` (from the start of labeling for
(P)CASL, the kinetic convention), each a low-flip gradient-echo excitation of every slice in turn
(`a_n`, the slice's own excitation at `t_n + slice offset`), each producing one volume. The
labeling cycle repeats every `TR`. This is the ITS-FAIR / QUASAR family (Gunther et al., MRM 2001;
Petersen et al., MRM 2006) without the QUASAR crushed/uncrushed arterial separation, which is
deferred.

**Tissue.** For one slice, the cycle is a sequence of events on `Mz`: background-suppression
inversions (`1 - 2 eps`, before the first readout as today) and the readouts (`cos(a_n)` each),
with T1 recovery between them, and the transverse signal of readout `n` is `sin(a_n) Mz(t_n^-)`.
The whole cycle is affine in the starting `Mz`, so its steady state is the same fixed-point solve
as P5's `tissue_mz_ge` with the event list extended; at `M = 1` it **is** `tissue_mz_ge`, and at
`M = 1`, `a = 90` it is P3's `tissue_mz` (both pinned by test). Cycles with different preparations
(control and label differ only in blood; suppression may differ per PLD set) propagate state cycle
to cycle as P5's `tissue_mz_ge_sequence` does.

**Blood.** Label magnetization that has arrived in the imaged slice is depleted by every readout
of that slice after its arrival; label still in transit is not (the labeling plane and the feeding
arteries are outside the imaged slices, the P1 geometry). The delta-M read at readout `n` therefore
splits by arrival window:

```
dM_n = sum_{k=0}^{n-1} [ prod_{m=k+1}^{n-1} cos(a_m) ] * dM_arr(t_n; t_k, t_{k+1})      (t_0 = 0)
```

where `dM_arr(t; u1, u2)` is the delta-M at time `t` of the label that **arrived** during
`[u1, u2)`. For the (P)CASL Buxton model the arterial input is `2 alpha M0b f exp(-delta/T1b)` on
`[delta, delta + tau]` and the residue is `exp(-(t - u)/T1')`, so

```
dM_arr(t; u1, u2) = 2 alpha M0b f exp(-delta/T1b) T1' (exp(-(t - u_hi)/T1') - exp(-(t - u_lo)/T1'))
u_lo = max(u1, delta),  u_hi = min(u2, delta + tau, t),  zero when u_hi <= u_lo
```

and for PASL the input is `2 alpha M0b f exp(-u/T1b)` on `[delta, delta + tau]`, which integrates in
closed form the same way. The windows partition `[0, t_n)`, so with every `cos(a_m) = 1` the sum
is today's `delta_m(t_n)` (a partition identity, pinned to `1e-12` like P4's). The read signal is
`sin(a_n) dM_n`. These functions are new in `kinetic.rs` (`delta_m_arrival`), with the GKM's
existing guards; exchange, the arterial compartment, crushing and bolus-position suppression are
refused under Look-Locker (their arrival-window forms are deferred), physiology and motion are
allowed (per readout volume, on the cycle's clock).

**Acquisition.** Each readout is one volume of the existing single acquisition call, gradient-echo
echo formation (P5 part A), `t_echo` the series `EchoTime`. The flip enters only the signal stage
(`sin(a_n)` and the `cos(a_n)` events), so `mrsim-acq` is unchanged for part B.

## Inputs

- `LookLocker: true`; `acq_contrast = "ge"` (the overlay's `[signal]`; spin echo with Look-Locker
  is refused: a Look-Locker readout is a low-flip gradient echo); 2D only.
- `FlipAngle`: required (the BIDS rule); a scalar or an array, one per volume (in `(0, 90]` for
  the readouts; `m0scan` entries are their own excitation's flip).
- `PostLabelingDelay`: an array, one per volume, the readout's delay (BIDS: "for multi-PLD and
  Look-Locker"). For (P)CASL `t_n = tau + PLD_n`; for PASL `t_n = PLD_n` (the existing rules).
- **Cycles from the arrays.** A cycle is a maximal run of consecutive non-`m0scan` rows of the
  same `volume_type` with strictly increasing `PostLabelingDelay` and equal `RepetitionTimePreparation`
  and `LabelingDuration`; a new cycle starts when the type changes or the delay does not increase.
  An optional overlay `[look_locker] readouts_per_cycle = M` makes the grouping explicit and is
  checked against the arrays (an error naming the first row that disagrees).
- Timing: each readout's slices must finish before the next readout begins
  (`t_n + max slice offset + readout duration <= t_{n+1}`, the readout duration being
  `TotalReadoutTime` plus the half echo time before the centre), and the last before `TR`.

The clock: `row_start` advances by one `TR` per **cycle**; a readout volume's time is
`row_start + t_n` (physiology and motion use it).

## Outputs

The volumes are written as they are acquired (one per readout per cycle), with the input arrays,
`LookLocker: true`, `FlipAngle` as resolved (an array when the input was), and
`AslscanSimulation.LookLocker`: the cycles (start row, readout count, readout times, flips), the
tissue model ("affine timeline with readout events, steady state per slice"), the blood model
(the arrival-window sum), and the refused P4 parts. The `m0scan` sidecar of a separate M0 gets
`FlipAngle` (required by BIDS under Look-Locker). `pixdim[4]` stays the first non-`m0scan` row's
`tr` (BIDS's ASL timing is in the sidecar arrays); the sidecar says so.

**Ground truth**: `desc-deltam_gt` per readout volume is the **undepleted** `delta_m(t_n)` (the
perfusion signal, as today), and a new `desc-deltamRead_gt` the depleted `sin(a_n) dM_n` the
readout saw; the tissue `Mz(t_n^-)` per readout is recorded in the TSV `desc-lookLocker_gt.tsv`
per cycle (one row per readout: cycle, readout, time, flip, mean tissue `Mz` per label).

## Refusals

3D; spin echo; exchange, the arterial compartment, crushing, bolus-position suppression; Hadamard;
multi-TE; compat (simasl has no Look-Locker). A `FlipAngle` array without `LookLocker: true`
stays an error, as today (`opt_num`, `protocol.rs:626-631`: "FlipAngle must be a number"):
per-volume flips are Look-Locker only.

---

# Part C: multi-TE (2D EPI, `echo-N`)

## What is modeled

Each excitation is followed by `E` echoes at `TE_1 < ... < TE_E`, each read by its own EPI block,
each producing its own image of the same volume. The compartment images are the same for every
echo (one excitation, one magnetization state); what differs is the transverse decay, which the
acquisition already applies per compartment: `exp(-trf/T2_c - |t|/T2'_c)` for spin echo,
`exp(-trf/T2_c - trf/T2'_c)` for gradient echo, with `trf = TE_e + t` (`kspace.rs:813-830`).
Because blood is its own compartment with `T2_blood` (and the arterial compartment with
`T2_arterial`), and P4's exchange already splits the label's delta-M into intravascular (blood
T2) and extravascular (tissue T2) parts by the exchange time, the multi-TE delta-M is the
two-compartment model

```
dM(t, TE) = dM_iv(t) exp(-TE/T2b) + dM_ev(t) exp(-TE/T2t)
```

of Gregori et al. (JMRI 2013) and Ohene et al. (MRM 2021), with no new kinetics. Without exchange
(`exchange_time` absent) all label stays intravascular, which the sidecar states.

**Echo formation.** Spin echo: every echo is a refocused spin echo (a CPMG train, T2' refocused at
each echo centre). Gradient echo: one excitation, echoes along the free induction decay, T2' from
the RF and the static `2 pi fmap TE_e` phase per echo (P5 part A). The model assumes perfect
refocusing for spin-echo trains (no stimulated echoes; the 2D CPMG amplitudes at 180 degrees are
`exp(-TE/T2)` exactly, P5's EPG).

## The acquisition (`mrsim-acq`)

One new public entry point, no change to any existing struct or signature:

```rust
pub fn simulate_acquisition_echoes(
    sim_dims, acq_dims, n_volumes, images, t2, fmap, t_inhom, acq: &Acquisition,
    echo_times_ms: &[f64], eddy_drive, prep_drive, phase, seed, noise_sigma, eddy_trace,
) -> Vec<(Vec<f32>, Vec<f32>)>        // per echo: magnitude, phase
```

For echo `e` it runs the existing per-volume path with `Acquisition { t_echo: echo_times_ms[e],
..acq.clone() }`. Noise and spikes get a per-echo stream: the `slice_seed` of echo `e` is today's
`slice_seed` XORed with `echo_salt(e)`, `echo_salt(0) = 0`; the prepared-shot phase, which belongs
to the excitation, keeps the unsalted `seed`, so every echo of a volume sees the same shot phase,
while k-space noise is independent per echo (separate receiver samplings). With one echo equal to
`acq.t_echo` it is bit-identical to `simulate_acquisition_oversampled` (pinned). The timing check
runs per echo. The per-echo clone is the minimal change the code review of the interface
identified (`t_echo` enters only `trf` and the GE static phase; the line times, NUFFT path and
rotors are unchanged), and a per-volume `LineTiming` alternative would leave the GE static phase at
the series `t_echo`.

The echoes of one volume are acquired by independent calls of the 2D forward with the same
images; nothing models the echo train's own k-space trajectory (e.g. the k-space centre of a
later echo's EPI block) beyond each echo's own `t_echo`, which the sidecar states.

## Inputs (`aslscan`)

- **The CLI takes `--asl-json` once per echo**, in echo order, each with a scalar `EchoTime`
  (BIDS's `echo-N` layout: one sidecar per echo, one shared `aslcontext.tsv`). Every other key must
  agree across the sidecars (compared numerically, the first difference named); `EchoTime` must
  strictly increase. One `--asl-json` is today's single-echo protocol, unchanged.
- An `EchoTime` array in any one sidecar keeps today's rule (equal entries collapse; unequal is an
  error), now naming the `echo-N` layout as the way to give several echoes.
- Timing: for gradient echo, consecutive EPI blocks must not overlap
  (`TE_{e+1} - TE_e >= TotalReadoutTime`); for spin echo, a refocusing pulse sits between them
  (`TE_{e+1} - TE_e >= TotalReadoutTime + refocusing_time`, P5's `refocusing_time`, default 2 ms);
  the last echo's block must end within the slice's share of the repetition (the existing readout
  window check, at the last echo).

## The series and outputs

The series builds the compartment images once and makes **one** call of
`simulate_acquisition_echoes`; the separate M0 likewise (its own seed, as today, with per-echo
salts). Ground truth is echo-independent and written once (`desc-deltamIntravascular_gt` is the
quantity multi-TE separates). The dataset has one series per echo:
`sub-X_echo-<e>_part-{mag,phase}_asl.nii.gz` with its own sidecar (scalar `EchoTime`, the input
sidecar of that echo as the base), and one `sub-X_aslcontext.tsv`; a separate M0 likewise gets
`echo-<e>` files. `AslscanSimulation.MultiEcho`: the echo times, the echo formation, the noise
streams, the two-compartment statement, and whether exchange is on.

**Hadamard x multi-TE** (Mahroo et al., Front Neurosci 2021): allowed; each echo's raw series is
decoded separately, and the decoded dataset has one `echo-N` series per echo.

## compat

Multi-TE under `[compat] asldro = true` is allowed: simasl applies one `exp(-TE/T2)` (or `T2*`) per
voxel to every volume and accepts a per-volume `echo_time`, so echo `e` of an aslscan compat run
equals a single-echo compat run with `EchoTime = TE_e`, and equals simasl's run with a constant
`echo_time = TE_e`. The compat translator (`tools/compat_asldro.py`) gains the echo loop; benchmark
H compares each echo voxelwise, as A-G do.

---

# Division of labor

| Piece | Where |
|---|---|
| `simulate_acquisition_echoes`, per-echo noise salt | `mrsim-acq` `kspace` |
| Labeling cycles, `RowKind::Encoded`, Look-Locker cycle grouping, multi-sidecar parsing | aslscan `rows`, `protocol` |
| Hadamard matrix, encoding weights, decoding | aslscan `hadamard` (new, pure std) |
| `delta_m_arrival` (PASL and (P)CASL), partition identity | aslscan `kinetic` |
| Look-Locker tissue timeline (readout events, fixed point, propagation) | aslscan `longitudinal` |
| Per-cycle assembly, decoding after the call, the echo call | aslscan `series` |
| Decoded main dataset, `sourcedata` raw series, `echo-N` series, Look-Locker ground truth | aslscan `bids` |
| Repeated `--asl-json` | aslscan `bin/aslscan` |
| Multi-TE compat, benchmark H | aslscan `tools/compat_asldro.py` |

# Testing and verification

**mrsim-acq**

- `simulate_acquisition_echoes` with one echo at `acq.t_echo` equals
  `simulate_acquisition_oversampled` bit for bit (both feature sets, noise on); with two echoes,
  echo 0 equals the single-echo call bit for bit and echo 1 equals a single-echo call at `TE_1`
  except for the noise stream (noise off: bit for bit); the echoes' noise is independent (sample
  correlation within `3/sqrt(n)` of 0) and the shot phase is shared (prepared phase on, noise off:
  echo images differ only by the decay); the golden record and every bit-pinned test unchanged.

**aslscan**

- **Hadamard**: the matrix is orthogonal with zero column sums (orders 4-32); the encoded blood of
  the all-labeled row equals the unencoded bolus of `tau_tot` to `1e-12` (the P4 partition
  identity); **decoded volume `j` of a noiseless series equals a single `deltam` run with
  `LabelingDuration = tau_j`, `PostLabelingDelay = PLD_j`** to the linearity tolerance (shift
  invariance of the (P)CASL kinetics; without suppression or physiology, whose timing is not
  shift-invariant), and the decoded ground truth equals that run's ground truth; with suppression
  and physiology on, decoding is still exact against the encoded ground truth (decode the raw
  ground truth, compare); the decoded noise SD over the raw is recorded (want about
  `2/sqrt(N+1)`); the input checks (PLD formula, order, PASL, `control` rows).
- **Look-Locker**: the arrival-window partition identity to `1e-12` (both label types, arrival
  before, during and after the readouts); `M = 1` equals P5's gradient-echo series bit for bit
  (tissue and blood); the tissue steady state equals brute-force iteration of the cycle until
  successive starting values differ by `1e-15 M0`; with all flips at `1e-6` degrees the readout
  volumes equal multi-PLD single-readout rows (sign and timing); the cycle grouping from the arrays
  and its overlay check; the linearity identity per readout (control cycle - label cycle - deltam
  cycle), with its negative control.
- **Multi-TE**: one echo is today's output bit for bit (regress); echo `e`'s images equal a
  single-echo run at `TE_e` with noise off; the ratio of decoded delta-M between two echoes matches
  `exp(-dTE/T2b)` with exchange off and the two-compartment formula with exchange on (per voxel,
  noise off, uniform compartments); compat benchmark H; the sidecar agreement checks.
- The validator passes on every P6 output dataset (the decoded Hadamard dataset, Look-Locker,
  multi-TE `echo-N`).
- Byte identity: `regress_identity.sh` against `p5-complete`, TRXScan P0, the golden record.

# Acceptance criteria

1. No P6 input, no changed byte (TRXScan, `mrsim-acq`, `aslscan`).
2. A Hadamard-8 PCASL protocol (7 sub-boli, 2 cycles, `m0scan`) on the crop and on the 3 T phantom
   simulates, decodes to the single-sub-bolus runs, and its main dataset validates; the raw series
   is written under `sourcedata`.
3. A QUASAR-like Look-Locker protocol (PASL, 2D, 12 readouts 0.3 s apart, flips 35 degrees,
   control/label cycles) simulates, matches the single-readout reduction, and validates.
4. A three-echo PCASL protocol (`TE` 13, 32, 51 ms gradient echo; and a spin-echo variant) with
   exchange simulates, matches the two-compartment formula, its `echo-N` dataset validates, and
   compat benchmark H passes.
5. Hadamard x multi-TE simulates and its decoded `echo-N` dataset validates.
6. The tests above pass and the measured numbers (decoded noise, Look-Locker steady states,
   multi-TE ratios, run times) are recorded in the plan.

# Decisions deferred

- **Velocity-selective ASL**: no BIDS type; revisit when BIDS adds one (the model, Wong et al.
  MRM 2006 and the consensus Qin et al. MRM 2022, is a saturation or inversion bolus with
  efficiency about 0.5 for VSS and a bolus defined by the vascular crushing module, and would reuse
  the PASL machinery).
- **Look-Locker with 3D readouts**: needs 3D gradient-echo readouts, which P5 deferred.
- **Multi-TE with 3D readouts**: the 3D echo-train model reads one partition per echo; a multi-echo
  3D readout is a different train.
- **Look-Locker x Hadamard, Look-Locker x multi-TE**: each is a product of the two cycle
  structures; no target dataset motivates them yet.
- **The QUASAR arterial compartment**, crushing, exchange and bolus-position suppression under
  Look-Locker: their arrival-window forms.
- **Vessel-encoded labeling, Walsh-ordered or non-Sylvester matrices, T1-adjusted variable
  sub-bolus designs with per-cycle variation.**
- **A multi-echo EPI's own k-space**: echoes are independent 2D readouts at their own `TE`.

# Risks

**BIDS underdetermines all three.** Hadamard has no representation, so the main dataset is the
decoded form and the encoding lives in `sourcedata` and `AslscanSimulation`; Look-Locker's cycles
are inferred from the delay arrays (with an explicit overlay check); multi-TE uses a layout
(`echo-N` inheriting one `aslcontext.tsv`) whose inheritance this design assumes from the
specification's rules and the validator must confirm on the first output.

**The Look-Locker blood model is a model.** Depletion is applied to arrived label only and
in-transit label is untouched; partial arrival into a slice during a readout train, inflow of
unsaturated arterial blood into the slice between readouts, and slice-profile effects on the flip
are not modeled. The sidecar says so, and the arrival-window form reduces exactly to the existing
model when the flips vanish.

**Decoding noise and motion.** Decoding mixes `N + 1` raw volumes; motion between them produces
decoding artefacts, which is realistic, and the ground truth is per sub-bolus at the unmoved
positions (as P3's static truth). Physiological factors per raw acquisition likewise leak between
sub-boli after decoding, as on a scanner.

# Review record

None yet.
