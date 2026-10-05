# P6: Hadamard time-encoded labeling, Look-Locker readouts and multi-TE ASL

Design spec addendum, 2026-10-05. Extends `2026-09-21-mrsim-acq-aslscan-design.md` (the main
spec) and follows the P2-P5 addenda (`2026-09-24-p2-asldro-compat-design.md`,
`2026-09-24-p3-motion-bs-ir-design.md`, `2026-10-01-p4-vascular-physio-design.md`,
`2026-10-01-p5-3d-readouts-ge-design.md`). Where this document is silent, those stand.
Codex-reviewed once (2026-10-05: 1 blocker, 13 majors, 1 minor; all verified and applied; the
review is summarized at the end).

## Context

The roadmap gives P6 "Encoding variety: Hadamard, Look-Locker, velocity-selective, multi-TE"
(main spec, sub-project table), depending only on P1. Three of the four are in scope here;
**velocity-selective labeling is deferred** (user decision, 2026-10-05): BIDS defines no VSASL
type (`ArterialSpinLabelingType` is `CASL`, `PCASL` or `PASL` in the released specification
1.11.1, the validator's schema 1.2.7 and the specification's `master`), no BEP or open issue
proposes one, and the ASL-BIDS paper (Clement et al., Sci Data 2022) lists velocity-selective and
time-encoded ASL among the approaches a future release may add. See "Decisions deferred".

What exists today, verified against the source (`aslscan` and `mrsim-acq` at `p5-complete`:
`715bcb0`, `b340133`):

- **A row is one labeling and one readout.** `Row { kind, t, tau, tr }` (`rows.rs:8-38`) carries
  one kinetic time (`t`, the excitation time from the start of labeling), one bolus duration and
  one repetition time; `RowKind` is `{M0scan, Control, Label, Deltam}`. The clock `row_start`
  advances by one `tr` per row, by `NumberShots x tr` for a segmented 3D volume, because each shot
  has its own labeling (`protocol.rs:1898-1909`); 3D physiology runs per shot at
  `row_start + shot x tr` (`series.rs:813-823`). The blood image of a row is one signed delta-M
  (`blood_sign`, `series.rs:187-203`: `-1` label, `+1` deltam).
- **The only multi-bolus primitive is P4's sub-bolus.** `delta_m_sub(k, .., t, a, b)`
  (`kinetic.rs:128-137`) is the delta-M of the parcels labeled during `[a, b]` of `[0, tau]` (for
  (P)CASL a bolus of length `b - a` started `a` later); `delta_m_iv_sub` (`:174-181`) is its
  intravascular part; the arterial term can be selected by its labeling coordinate
  (`:213-222`). A partition identity is tested to `1e-12` (`:556-591`).
  `bolus::subbolus_factors` produces factors only from suppression pulses.
- **Per-row inputs are keyed to rows.** Background suppression assigns one pulse set per distinct
  input `PostLabelingDelay` (`pulse_times_per_pld`, `protocol.rs:1610-1638`); a
  `VascularCrushingVENC` array is expanded per row (`:1794`).
- **One readout per labeling.** The tissue timeline (`longitudinal.rs:54-74`; `tissue_mz_ge`
  `:83-100`; `tissue_mz_ge_sequence` `:117-140`, which carries state between rows whose
  preparations differ, including `m0scan` rows, `series.rs:682-709`) is affine in the starting
  `Mz` with events that scale `Mz` (inversions, `1 - 2 eps`) and one readout at `t_read`; blood
  under gradient echo is `sin(a) dm` with no depletion (`mrsignal.rs:106`). Physiological label
  factors are sampled at the labeling time (PASL) or averaged over the labeling window ((P)CASL)
  (`series.rs:378-393`); motion poses are indexed by volume (`resolve_poses` takes a volume count,
  `series.rs:1017-1020`).
- **One echo time per series.** `echo_time_s` is a scalar; an unequal `EchoTime` array and
  `LookLocker: true` are errors naming P6 (`protocol.rs:1208-1213`, `:1023-1025`); a `FlipAngle`
  array is an error ("must be a number", `opt_num`, `:626-631`). Under compat the echo-time decay
  `exp(-TE/T2)` (`T2*` under gradient echo) is applied **while building the compartment images**
  (`te_factor`, `series.rs:615-634`) and the acquisition's relaxation is off
  (`protocol.rs:2149-2151`). The 2D `[readout]` table is refused (`protocol.rs:1018-1019`).
- **In `mrsim-acq`, `t_echo` enters `trf` and the gradient-echo static phase.**
  `LineTiming::for_acquisition` gives `trf = t_echo + t` (`kspace.rs:251-296`); gradient echo adds
  `2 pi fmap TE` to the static phase (`:636-641`); the line times `t`, the NUFFT affinity check and
  the rotors do not depend on it. Within one call: k-space noise and spikes use `slice_seed`
  (keyed on `(volume, slice)` and the call `seed`, `:1485-1489`, spikes `:940`, noise `:958`);
  image-space noise (`noise_sigma`) uses the call `seed` directly (`:1512-1521`); the prepared-shot
  phase uses the call `seed` (`:1480-1482`). The public function returns magnitude and phase in
  `f32` (`:1524-1525`); aslscan rebuilds complex values (`complex_from`, `series.rs:1250-1251`).

What BIDS settles (schema 1.2.7):

- **Look-Locker**: `LookLocker` (boolean, optional) true makes `FlipAngle` required **for that
  file** (`MRIFlipAngleLookLockerTrue` applies to a file whose resolved sidecar has
  `LookLocker: true`); `FlipAngle`, `PostLabelingDelay` and `LabelingDuration` arrays have one value
  per volume and error-level length checks against both `dim[4]` and the `aslcontext.tsv` rows.
- **Multi-TE**: `rules.files.raw.perf` allows the `echo` entity on `_asl`, `_m0scan` and `_noRF` but
  not on `aslcontext.tsv`, and `meta.associations.aslcontext` with the validator's subset-entity
  inheritance (validator 3.0.2, `files/inheritance.ts:41-47`) associates one echo-less
  `aslcontext.tsv` with every `echo-N` series. An `EchoTime` array's length check is a warning.
- **Hadamard / time-encoded**: no representation. `volume_type` is `control`, `label`, `m0scan`,
  `deltam`, `cbf`, `noRF`; the raw encoded volumes are none of these. `TotalAcquiredPairs` is "the
  number of acquired control-label pairs" and is checked against the control and label counts only
  when control rows exist.

simasl (ASLDRO v2.2.0) implements none of the three, but carries a per-volume `echo_time`
(`validators/user_parameter_input.py:243-249`, applied at `examples.py:199`) with a
single-compartment `exp(-TE/T2)`: the one P6 case it can check (part C).

## The one idea underneath

P1-P5 identify a volume with **one labeling (a preparation) followed by one readout**. P6 breaks
that identity in three directions, each a re-indexing of existing physics:

- **Hadamard** puts several sub-boli, each on or off, in **one preparation**. Kinetics are linear
  in the labeling, so an encoded preparation's blood image is a 0/1-weighted sum of P4 sub-bolus
  delta-Ms, and decoding is a fixed linear combination of reconstructed volumes.
- **Look-Locker** puts several readouts after **one preparation**. Each readout is an event on the
  affine `Mz` timeline (a `cos(a)` scaling, like an inversion's `1 - 2 eps`), and the label it reads
  has been depleted by the readouts after its arrival.
- **Multi-TE** reads several echoes after **one excitation**. The longitudinal state is the same for
  every echo; the transverse decay differs per compartment, which the acquisition already applies.

So aslscan gains an explicit **schedule** with three indices: a **preparation** (one labeling, with
its suppression and crushing; P5's shots are preparations), a **raw volume** (one reconstructed
image per readout, made of one or more preparations in 3D), and an **output volume** (what is
written: the raw volume, or a decoded combination). Every existing protocol is the identity
schedule, and its code path is unchanged (below).

## Scope

In: Hadamard time-encoded (P)CASL for 2D and 3D readouts (part A); Look-Locker readouts for 2D EPI
under gradient-echo excitation (part B); multi-TE for 2D EPI, spin echo and gradient echo, with the
`echo-N` layout (part C); Hadamard x multi-TE. Out, with reasons under "Decisions deferred":
velocity-selective labeling; Look-Locker and multi-TE with 3D readouts; Look-Locker x Hadamard and
Look-Locker x multi-TE; vessel-encoded labeling; non-Sylvester matrices; the arterial (QUASAR)
compartment and the other P4 parts under Look-Locker.

**Byte identity.** With no P6 input, every output is unchanged:

- **TRXScan** (`p0-mrsim-acq-extraction`, `12a9d09`): the P0 gate. P6 adds **no field** to
  `Acquisition` or `SliceInput` and changes no existing signature (TRXScan builds both with every
  field named, `trxscan.rs:828`, `benchmark.rs:84`), so TRXScan needs no source change; the gate
  checks the bytes.
- **mrsim-acq**: the golden record and every bit-pinned test; the new entry point (part C) is a
  separate function whose one-echo case is pinned bit for bit against
  `simulate_acquisition_oversampled`.
- **aslscan**: `tools/regress_identity.sh` against **`p5-complete`** for both repositories, both
  feature sets. Its 19 cases are all P1-P4 (`regress_identity.sh:125-144`), so P6 first adds P5
  cases to it, run against `p5-complete` before any P6 change: gradient echo at 90 and 35 degrees
  with mixed preparations and an included M0 (state propagation), gradient-echo separate M0,
  GRASE (`asl005_p5`), spiral (`asl001_p5`), and a segmented GRASE case with physiology and a
  shot event.
- **Explicit legacy dispatch**: when no P6 input is present the series takes today's code path,
  not a generalized schedule evaluated with identity indices (generic timeline arithmetic is known
  to differ in the last bits from the closed forms, `longitudinal.rs:51-53`). Likewise
  Look-Locker with one readout dispatches to P5's functions, and multi-TE with one echo to the
  existing acquisition call.

**Activation and refusals** (the P4 rule): Hadamard by an overlay `[hadamard]` table, Look-Locker
by `LookLocker: true`, multi-TE by more than one `--asl-json`. Every P6 input belonging to a feature
that is off is an error naming it; every refused combination is an error naming both.

---

# Part A: Hadamard time-encoded labeling

## What is modeled

A (P)CASL labeling of total duration `tau_tot = sum_j tau_j` divided into `N` consecutive
sub-boli, `j = 1..N` in time order (sub-bolus 1 is labeled first), followed by a delay `PLD` to the
excitation. An **encoding cycle** is `H = N + 1` encoded preparations; preparation `i` labels
sub-bolus `j` when `h_ij = -1` and controls it when `h_ij = +1`, `h` the Sylvester Hadamard matrix
of order `H` (4, 8, 16 or 32) with its all-ones first column removed (Dai et al., MRM 2013;
Teeuwisse et al., MRM 2014). Encoding row 1 labels nothing; every other row labels `H/2` sub-boli.

**Kinetics.** Sub-bolus `j` occupies `[a_j, b_j]` of `[0, tau_tot]`. Preparation `i`'s blood
delta-M is

```
dM_i(t) = sum_j w_ij * dM_sub(t; a_j, b_j),     w_ij = (1 - h_ij) / 2  in {0, 1}
```

with `dM_sub` P4's `delta_m_sub` (`delta_m_iv_sub` for the intravascular part under exchange, and
the arterial term restricted to the labeled parcels), at `t = tau_tot + PLD` plus the slice offset.
This is exact for the linear kinetic model. The raw image is `tissue - dM_i` (label subtracts).
**PASL is refused**: time encoding needs a labeling switched during the bolus.

**Decoding.** With `S_i = C_i - sum_j w_ij dM_ij` (tissue `C_i` and sub-bolus signals `dM_ij` as
acquired in preparation `i`), the columns of `h` sum to zero and are orthogonal, so

```
D_j = (2 / H) * sum_i h_ij * S_i
    = dM_j                                       when C_i = C and dM_ij = dM_j for every i
    = (2/H) sum_i h_ij C_i  -  (2/H) sum_i h_ij sum_k w_ik dM_ik     in general
```

The general form is what a scanner gets: tissue that differs between encoded preparations
(gradient-echo transients after an M0 or a differently prepared row, physiological tissue
factors) leaks into every decoded sub-bolus, and preparation-dependent label factors
(physiology) mix sub-boli. aslscan keeps that physics; it does not force a common `C`. The sidecar
reports, per cycle, the largest tissue difference across its encoded preparations
(`TissueLeakage`, relative to the tissue mean), so a user sees when decoding is not clean.

aslscan decodes the reconstructed complex images per cycle in `f64`, from the acquisition's `f32`
magnitude and phase. Decoding is a fixed linear map, so it is exact **when the acquisition is the
same linear operator for every preparation of the cycle**: no GRAPPA (its weights are calibrated
per volume, `kspace.rs:1335-1360`), no spikes (placed per volume at its own peak, `:926-943`), the
same pose, and up to the `f32` rounding of the tissue (an absolute error of order `1e-7` of the
tissue magnitude, which the tests' tolerance scales with). GRAPPA, spikes and motion are allowed and
decode into realistic artefacts; the sidecar says which were on.

The decoded noise SD is the raw SD times `2/sqrt(H)` (measured, not asserted).

## Inputs

The input describes the **decoded** dataset to be written, plus the encoding:

- `aslcontext.tsv`: `deltam` rows, `N` per cycle, in sub-bolus order, and `m0scan` rows **only
  between cycles** (before the first, between two, after the last). `control` and `label` rows are
  refused with Hadamard.
- `LabelingDuration`: per row, `tau_j` for the `deltam` rows (identical in every cycle), `0` for
  `m0scan`; a scalar means equal sub-boli.
- `PostLabelingDelay`: per row, the effective delay of each sub-bolus,
  `PLD_j = PLD + sum_{k>j} tau_k` (sub-bolus end to excitation, BIDS's definition per sub-bolus);
  aslscan takes `PLD = PLD_N` and checks every `PLD_j` to `1e-6` s, naming the first mismatch.
- `RepetitionTimePreparation`: the preparation repetition, constant within a cycle.
- **Per-preparation inputs are per cycle.** Background suppression: the pulse set is resolved
  once per cycle; under P3's `pulse_times_per_pld` the cycle's `deltam` rows must all map to the
  same set (they carry `N` different effective PLDs, so an overlay listing sets per PLD must list
  the same set for all of them, else an error naming the cycle). `VascularCrushingVENC`: a scalar,
  or an array constant within each cycle (broadcast to the cycle's preparations).
- Overlay `[hadamard]`: `order = H` (required; 4, 8, 16 or 32). `N = H - 1` must equal the number
  of `deltam` rows per cycle.

## The schedule

Each cycle is `H` **raw volumes** in matrix-row order; each raw volume is one preparation in 2D and
`NumberShots` preparations in 3D (every shot of a raw volume repeats that raw volume's encoding row;
P5's shot order, per-shot physiology and motion are unchanged), each one `tr`. So a cycle is
`H x NumberShots` preparations and the clock advances by `NumberShots x tr` per raw volume. `m0scan`
rows sit between cycles as one raw volume each (`NumberShots` preparations in 3D, as today).

aslscan publishes the maps: decoded row -> (cycle, sub-bolus); raw volume -> (cycle, encoding row,
first preparation, preparation count); `m0scan` row -> its raw volume.

Rows become raw volumes of a new `RowKind::Encoded { cycle, row }`, `t = tau_tot + PLD`,
`tau = tau_tot`, with the encoding weights `w_i`. The blood image of an encoded raw volume is
`-sum_j w_ij dM_sub(.; a_j, b_j)` through the same per-voxel path as a label row (P4's label path
when exchange, bolus-position suppression, physiology or the arterial compartment is on, P1's
otherwise). Everything per preparation or per volume applies to the raw volumes: noise and spikes,
poses and shot events, physiological factors (each preparation's labeling window is its own
`[start, start + tau_tot]`), suppression (pulses from the start of labeling; after `tau_tot` under
the global model; per parcel under P4's bolus-position model). One acquisition call simulates all
raw volumes; decoding follows it.

## Ground truth

Two truths, named apart:

- **Ideal sub-bolus truth** (main dataset, `desc-deltam_gt`, per decoded volume): `dM_j` from the
  kinetics alone, at the sub-bolus's own timing, without physiological factors, unmoved (static,
  since a decoded volume has no single pose; the sidecar says so). The P4 truths
  (`deltamIntravascular`, `deltamArterial`, `deltamSuppressed`) likewise per sub-bolus.
- **Raw acquisition truth** (`sourcedata`, per raw volume): the encoded `sum_j w_ij dM_sub` with the
  preparation's physiological and suppression factors, moved with the raw volume's pose, as P3/P4
  write `desc-deltam_gt` today.

## Outputs

- **Main dataset** (validates): the decoded series (`deltam` in sub-bolus order per cycle, `m0scan`
  as raw volumes between cycles), the input arrays, and `AslscanSimulation.Hadamard`: the order, the
  matrix, `[a_j, b_j]`, `PLD`, the maps above, the preparation, raw and decoded counts, the decoding
  rule, `2/sqrt(H)`, `TissueLeakage` per cycle, and which non-exact features were on (GRASE/spikes/
  motion/physiology/transients).
- **`TotalAcquiredPairs`**: BIDS defines it as acquired control-label pairs, which a Hadamard
  acquisition does not have. aslscan writes the **number of encoding cycles** (each cycle gives one
  measurement of every sub-bolus, the role a control-label pair plays for one PLD) and states the
  convention in `AslscanSimulation.Hadamard.TotalAcquiredPairsConvention`; the validator does not
  check it for `deltam`-only contexts, which this design does not take as an endorsement.
- **Raw series** under `sourcedata/sub-X/[ses-Y/]perf/`: the raw volumes, `part-mag`/`part-phase`,
  a sidecar with the matrix and the maps, a TSV with one row per raw volume (`cycle`,
  `encoding_row`, `labeled_subboli`, `first_preparation`), and the raw truth.

## Refusals

PASL; `control`/`label` rows; `m0scan` inside a cycle; `order` not in {4, 8, 16, 32}; inconsistent
effective PLDs; `LabelingDuration` varying across cycles; suppression or VENC varying within a cycle;
Look-Locker; compat.

---

# Part B: Look-Locker readouts (2D EPI)

## What is modeled

After one preparation, `M` readouts at `t_1 < ... < t_M` (excitation times from the start of
labeling, the existing kinetic convention), each a low-flip gradient-echo excitation `a_n` of every
slice in turn (slice `z` at `t_n + offset_z`), each producing one raw volume. The cycle repeats every
`TR`. This is the ITS-FAIR / QUASAR family (Gunther et al., MRM 2001; Petersen et al., MRM 2006)
without QUASAR's crushed/uncrushed arterial separation (deferred).

**Tissue.** Per slice, the cycle is a list of events on `Mz`: suppression inversions (`1 - 2 eps`,
before the first readout as today) and the slice's own readouts (`cos(a_n)` each, at
`t_n + offset_z`), with T1 recovery between them; readout `n`'s transverse signal is
`sin(a_n) Mz(t_n + offset_z, before the pulse)`. The cycle is affine in the starting `Mz`, so its
steady state is P5's fixed point with the event list extended, solved per slice, and cycles with
different preparations carry state forward as `tissue_mz_ge_sequence` does. One readout
(`M = 1`) dispatches to P5's `tissue_mz_ge` and `tissue_mz_ge_sequence` themselves.

**Blood.** The readout depletes the difference magnetization of label that has **arrived** in the
imaged slice (the excitation scales `Mz` of both control and label by `cos(a)`, so their difference
by the same factor); label in transit is outside the imaged slices (the P1 geometry) and is not
depleted. The delta-M read at readout `n` splits by arrival window:

```
dM_n = sum_{k=0}^{n-1} [ prod_{m=k+1}^{n-1} cos(a_m) ] * dM_arr(t_n; t_k, t_{k+1})      (t_0 = 0)
```

with `dM_arr(t; u1, u2)` the delta-M at `t` of the label that arrived during `[u1, u2)`, residue
`exp(-(t - u)/T1')` after arrival (the existing GKM's `T1'`, `1/T1' = 1/T1t + f/lambda`):

- (P)CASL, input `2 alpha M0b f exp(-delta/T1b)` on `[delta, delta + tau]`:
  `dM_arr = 2 alpha M0b f exp(-delta/T1b) T1' (exp(-(t - u_hi)/T1') - exp(-(t - u_lo)/T1'))`,
  `u_lo = max(u1, delta)`, `u_hi = min(u2, delta + tau, t)`, zero when `u_hi <= u_lo`.
- PASL, input `2 alpha M0b f exp(-u/T1b)` on `[delta, delta + tau]`, `q = 1/T1b - 1/T1'`:
  `dM_arr = 2 alpha M0b f exp(-t/T1') (exp(-q u_lo) - exp(-q u_hi)) / q`, keeping the existing
  GKM's guard at `q = 0` (it returns zero there, `kinetic.rs:92-105`, not the continuous limit).

The windows partition `[0, t_n)`, so with every `cos(a_m) = 1` the sum is `delta_m(t_n)` (pinned to
`1e-12`). The read signal is `sin(a_n) dM_n`. New in `kinetic.rs` as `delta_m_arrival`. Exchange, the
arterial compartment, crushing and bolus-position suppression are refused under Look-Locker.

**Physiology and motion.** One labeling factor per cycle (sampled or window-averaged at the cycle's
labeling, as today), shared by all its readouts; the tissue factor of readout `n` at
`cycle start + t_n + offset_z`. Motion keeps today's convention: poses indexed by volume, so each
readout volume has its own pose (the sidecar says the pose index is the volume, not a time).

**Acquisition.** Each readout is one volume of the existing single call, gradient-echo echo
formation, `t_echo` the series `EchoTime`; the flip enters only the signal stage, so `mrsim-acq` is
unchanged for part B.

## Inputs

- `LookLocker: true`; overlay `[signal] acq_contrast = "ge"` (spin echo with Look-Locker is refused);
  2D only.
- `FlipAngle`: required; scalar or one per volume, `(0, 90]` for readouts; `m0scan` entries are
  that volume's own excitation.
- `PostLabelingDelay`: one per volume. (P)CASL `t_n = tau + PLD_n`, PASL `t_n = PLD_n`.
- **Cycles from the arrays**: a cycle is a maximal run of consecutive non-`m0scan` rows of the same
  `volume_type` with strictly increasing `PostLabelingDelay` and equal `RepetitionTimePreparation`
  and `LabelingDuration`; optional overlay `[look_locker] readouts_per_cycle = M` is checked against
  that grouping. Suppression is resolved once per cycle (all readouts of a cycle must map to the same
  pulse set), and pulses must precede the cycle's first excitation.
- **Separate M0**: its excitation flip is the overlay `[m0] flip_angle` (degrees), required when the
  ASL `FlipAngle` is an array; when it is a scalar the M0 takes that scalar. The M0's signal and its
  sidecar `FlipAngle` use the same resolved value. The M0 sidecar is not a Look-Locker file (no
  `LookLocker: true`).
- **Timing on explicit intervals**: for every slice excitation at `e = t_n + offset_z`, the sampled
  readout is `[e + TE - h, e + TE + h]` (`h` half the EPI block, the sample times
  `TE + time_from_max_echo`, `readout.rs:60-67`); it must end before the next slice's excitation in
  the same readout and the last slice's before the next readout's first excitation, the last
  readout's before `TR`, and every sample must follow its excitation (the existing check).

The clock advances one `TR` per cycle; readout volume times are `cycle start + t_n`.

## Outputs

The raw volumes as acquired, the input arrays, `LookLocker: true`, `FlipAngle` as resolved, and
`AslscanSimulation.LookLocker`: the cycles (first row, readout count, times, flips), the tissue and
blood models, the refused P4 parts. `pixdim[4]` stays the first non-`m0scan` row's `tr`, a storage
convention (BIDS's ASL timing is in the arrays); the sidecar records the actual readout schedule.

**Ground truth**: `desc-deltam_gt` per readout volume is the undepleted `delta_m(t_n)` (moved, as
today); `desc-deltamRead_gt` the depleted `sin(a_n) dM_n`; `desc-lookLocker_gt.tsv` one row per
readout (cycle, readout, time, flip, mean tissue `Mz` before the pulse per label).

## Refusals

3D; spin echo; exchange, arterial compartment, crushing, bolus-position suppression; Hadamard;
multi-TE; compat; a `FlipAngle` array without `LookLocker: true` (as today); suppression differing
within a cycle.

---

# Part C: multi-TE (2D EPI, `echo-N`)

## What is modeled

Each excitation is followed by `E` echoes at `TE_1 < ... < TE_E`, each read by its own EPI block and
producing its own image of the same volume. The longitudinal state is common to the echoes; the
transverse decay is applied per compartment by the acquisition. Echo `e` of compartment `c` at line
`t` carries

```
spin echo (CPMG):  exp(-(TE_e + t)/T2_c - |t|/T2'_c)
gradient echo:     exp(-(TE_e + t)(1/T2_c + 1/T2'_c)) * exp(i 2 pi fmap TE_e)
```

(`kspace.rs:813-830`, `:636-641`). The `TE_e` part is a common factor of every line, so between two
echoes the image of a compartment changes by exactly `exp(-dTE/T2_c)` (spin echo) or
`exp(-dTE (1/T2_c + 1/T2'_c)) exp(i 2 pi fmap dTE)` (gradient echo); blood has `T2_blood` and inherits
its label's `T2'` (`series.rs:495-497`); the arterial compartment has `T2_arterial`. P4's exchange
puts the intravascular delta-M in the blood compartment and the extravascular delta-M in the tissue
compartment (`series.rs:487-497`, `:951-954`, `:990-992`), so the spin-echo delta-M follows

```
dM(t, TE) = dM_iv(t) exp(-TE/T2b) + dM_ev(t) exp(-TE/T2t)  [+ arterial term]
```

(Gregori et al., JMRI 2013; Ohene et al., MRM 2021), and gradient echo multiplies each term by its
`exp(-TE/T2')` and the fieldmap phase. P4's kinetics use the single residue `T1'` after arrival;
they are not a two-compartment T1 exchange model (stated). Without `exchange_time` all label is
intravascular. Spin-echo trains assume perfect refocusing (at 180 degrees the CPMG amplitudes are
`exp(-TE/T2)`, P5's EPG).

## The acquisition (`mrsim-acq`)

One new public entry point; no existing struct or signature changes:

```rust
pub fn simulate_acquisition_echoes(
    sim_dims, acq_dims, n_volumes, images_per_echo: &[&[Vec<f32>]], t2, fmap, t_inhom,
    acq: &Acquisition, echo_times_ms: &[f64], eddy_drive, prep_drive, phase, seed,
    noise_sigma, eddy_trace,
) -> Vec<(Vec<f32>, Vec<f32>)>        // per echo: magnitude, phase
```

`images_per_echo` gives each echo its compartment images (the same slice for every echo, except
under compat, below). Echo `e` runs the existing per-volume path with
`Acquisition { t_echo: echo_times_ms[e], ..acq.clone() }`. Internally the call's `seed` is split
into an **excitation seed** (the prepared-shot phase, unsalted: every echo of a volume sees the same
shot phase) and a **receiver seed** salted per echo (`echo_salt(0) = 0`), used for k-space noise and
spikes (through `slice_seed`) and for image-space noise (`noise_sigma`): independent noise per echo.
With one echo at `acq.t_echo` it is bit-identical to `simulate_acquisition_oversampled` (pinned).
Echoes of a volume are independent 2D readouts at their own `TE`; the echo train's own k-space
trajectory is not modeled (stated).

## Inputs (`aslscan`)

- **`--asl-json` once per echo**, in echo order, each with a scalar `EchoTime`, strictly increasing;
  every other key must agree across the sidecars (compared numerically; the first difference named).
  One `aslcontext.tsv`. One `--asl-json` is today's protocol, unchanged.
- An `EchoTime` array in any sidecar keeps today's rule, naming the `echo-N` layout.
- Overlay `[multi_te] refocusing_time` (ms, default 2; spin echo only, refused otherwise): the 2D home
  for the refocusing reserve (the 2D `[readout]` table stays refused).
- **Timing on explicit intervals**, per slice: echo `e`'s sampled block is
  `[exc + TE_e - h, exc + TE_e + h]`. Gradient echo: blocks must not overlap. Spin echo: the first
  refocusing pulse is centred at `TE_1/2` and reserves `refocusing_time` about its centre; it must
  clear the excitation and end before the first block; each later pulse is centred at
  `(TE_e + TE_{e+1})/2` and must fit between the two blocks. The last block must end before the next
  slice's excitation, the last slice's before `TR`; the separate M0 likewise at its own `TR`.

## The series and outputs

Outside compat the series builds the compartment images once and passes the same images for every
echo. **Under compat** the echo-time decay belongs to the signal stage (`te_factor`), so the series
builds one image set per echo, applying `exp(-TE_e/T2)` (or `T2*`) before resampling as today, and
passes them per echo; with one echo this is today's path. The separate M0 likewise (its own seed,
per-echo salts). Ground truth is echo-independent and written once. Output: one series per echo,
`sub-X_echo-<e>_part-{mag,phase}_asl.nii.gz` with its own sidecar (scalar `EchoTime`, that echo's
input sidecar as base), one `sub-X_aslcontext.tsv`, a separate M0 per echo, and
`AslscanSimulation.MultiEcho`: echo times, echo formation, the noise streams, the decay statement,
whether exchange is on.

**Hadamard x multi-TE** (Mahroo et al., Front Neurosci 2021): allowed; each echo's raw series is
decoded separately; the decoded dataset has one `echo-N` series per echo.

## compat

Multi-TE under `[compat] asldro = true` is allowed: echo `e` equals, with noise off, a single-echo
compat run at `EchoTime = TE_e` and simasl's run with a constant `echo_time = TE_e` (later echoes
use their own noise streams, so equality is noise-off only). The translator gains the echo loop;
benchmark H compares each echo voxelwise.

---

# Division of labor

| Piece | Where |
|---|---|
| `simulate_acquisition_echoes`, excitation and receiver seeds | `mrsim-acq` `kspace` |
| The schedule (preparations, raw volumes, output volumes), `RowKind::Encoded`, Look-Locker cycles, multi-sidecar parsing, per-cycle suppression and VENC | aslscan `rows`, `protocol` |
| Hadamard matrix, weights, decoding, maps | aslscan `hadamard` (new, pure std) |
| `delta_m_arrival` | aslscan `kinetic` |
| Look-Locker timeline | aslscan `longitudinal` |
| Assembly, decoding, per-echo compat images, legacy dispatch | aslscan `series` |
| Decoded dataset, `sourcedata`, `echo-N`, Look-Locker truth | aslscan `bids` |
| Repeated `--asl-json` | aslscan `bin/aslscan` |
| P5 regress cases; compat benchmark H | aslscan `tools` |

# Testing and verification

**mrsim-acq**

- `simulate_acquisition_echoes`: one echo at `acq.t_echo` equals `simulate_acquisition_oversampled`
  bit for bit (both feature sets, k-space noise, image-space noise and a non-`None` `prep_drive` with
  a prepared phase model); with two echoes, echo 0 equals the single-echo call bit for bit; echo 1
  equals a single-echo call at `TE_1` with noise off; with noise on, echo 1's k-space and image-space
  noise are each uncorrelated with echo 0's (sample correlation within `3/sqrt(n)`); with a prepared
  phase on, noise off, no fieldmap and spin echo, echo 1 equals echo 0 times the per-compartment
  decay (single-compartment images), which fails if the shot phase were salted.
- The golden record and every bit-pinned test unchanged.

**aslscan**

- **Byte identity**: the P5 regress cases (above) pass against `p5-complete` before and after P6.
- **Hadamard**: orthogonality and zero column sums (orders 4-32); the partition identity with a
  synthetic all-ones weight vector (outside the matrix) to `1e-12`; every actual encoding row's blood
  against an independent weighted sum of `delta_m_sub`; **decoded volume `j` of a noiseless series
  without suppression, physiology, GRAPPA, spikes, motion or GE transients equals a single `deltam`
  run with `tau_j`, `PLD_j`** (shift invariance of (P)CASL kinetics) to the linearity tolerance with a
  tissue-scaled absolute term; the same with gradient echo after enough cycles for the transient to
  vanish; a **transient negative case** (an included long-`TR` M0 before a low-flip encoding cycle)
  where decoding leaks tissue and `TissueLeakage` reports it; **with physiology on**, the decoded
  image equals the general formula `(2/H) sum_i h_ij [C_i - sum_k w_ik dM_ik]` assembled from
  independent single-row runs scaled by the recorded per-preparation factors (not by decoding the
  truth); the 3D schedule (`H x NumberShots` preparations, clock, maps); the input checks.
- **Look-Locker**: the arrival-window partition identity to `1e-12` (both label types; arrival before,
  during, after the readouts; the PASL `q = 0` guard); `M = 1` equals P5's gradient-echo series bit for
  bit (legacy dispatch); the steady state equals brute-force cycle iteration until successive starts
  differ by `1e-15 M0`; with flips at `1e-6` degrees the readout volumes equal multi-PLD single-readout
  rows; cycle grouping and its overlay check; per-cycle suppression; the separate-M0 flip rule; the
  timing intervals (accept and reject cases); the linearity identity per readout (control cycle -
  label cycle - deltam cycle) and its negative control.
- **Multi-TE**: one echo bit-identical (regress); echo `e` equals a single-echo run at `TE_e` (noise
  off); spin echo, uniform compartments, no fieldmap: the IV/EV images scale between echoes by
  `exp(-dTE/T2b)` / `exp(-dTE/T2t)` exactly, and the delta-M follows the two-compartment formula;
  gradient echo: the same with `T2'` and the fieldmap phase; the arterial term when enabled; compat
  benchmark H (noise off); sidecar agreement; the spin-echo refocusing-interval checks.
- The validator passes on every P6 output (the decoded Hadamard dataset, Look-Locker, `echo-N`).

# Acceptance criteria

1. No P6 input, no changed byte (TRXScan, `mrsim-acq`, `aslscan` including the P5 cases).
2. A Hadamard-8 PCASL protocol (7 sub-boli, 2 cycles, `m0scan` before the first cycle) on the crop and
   on the 3 T phantom simulates, decodes to the single-sub-bolus runs under the exactness conditions,
   validates, and writes the raw series under `sourcedata`; a GRASE variant simulates with
   `H x NumberShots` preparations.
3. A QUASAR-like Look-Locker protocol (PASL, 2D, 12 readouts 0.3 s apart, 35 degrees) simulates,
   reduces to single readouts at vanishing flip, and validates.
4. A three-echo PCASL protocol with exchange (gradient echo `TE` 13, 32, 51 ms and a spin-echo
   variant meeting the refocusing intervals) simulates, matches the decay formulas, its `echo-N`
   dataset validates, and compat benchmark H passes.
5. Hadamard x multi-TE simulates and its decoded `echo-N` dataset validates.
6. The tests above pass and the measured numbers (decoded noise, `TissueLeakage`, Look-Locker steady
   states, multi-TE ratios, run times) are recorded in the plan.

# Decisions deferred

- **Velocity-selective ASL**: no BIDS type; revisit when BIDS adds one (Wong et al., MRM 2006; the
  consensus Qin et al., MRM 2022: a saturation or inversion bolus, efficiency about 0.5 for VSS, the
  bolus defined by the vascular crushing module; it would reuse the PASL machinery).
- **Look-Locker and multi-TE with 3D readouts**: need 3D gradient-echo or multi-echo 3D trains.
- **Look-Locker x Hadamard, Look-Locker x multi-TE.**
- **The QUASAR arterial compartment**, crushing, exchange and bolus-position suppression under
  Look-Locker (their arrival-window forms).
- **Vessel-encoded labeling, Walsh-ordered or non-Sylvester matrices, variable designs across
  cycles, dummy cycles to settle gradient-echo transients.**
- **A multi-echo EPI's own k-space**, and a two-compartment T1 exchange model.

# Risks

**BIDS underdetermines all three.** Hadamard has no representation: the main dataset is the decoded
form under stated conventions (`TotalAcquiredPairs`, static truth), with the encoding in
`sourcedata` and `AslscanSimulation`. Look-Locker cycles are inferred from the arrays (with an
overlay check). The `echo-N` inheritance of one `aslcontext.tsv` is supported by the schema and the
validator's inheritance code; the first output confirms it.

**Decoding is exact only under conditions the scanner rarely meets**, and aslscan simulates the
rest (transients, physiology, motion, GRAPPA, spikes) as artefacts rather than idealizing them; the
sidecar reports which were on and the tissue leakage, and the tests separate the exact case from the
realistic one.

**The Look-Locker blood model is a model**: depletion of arrived label only; in-transit label, inflow
of fresh blood into the slice between readouts and slice-profile effects are not modeled; it reduces
exactly to the existing model when the flips vanish.

# Review record

Codex adversarial design review, 2026-10-05: 1 blocker, 13 majors, 1 minor, each checked against the
source (and the cached schema) and found valid; all applied.

- **Blocker.** Multi-TE built the compartment images once, but under compat the echo-time decay is
  applied while building them (`series.rs:615-634`), so every echo would carry the first echo's
  decay: the entry point now takes images per echo and compat builds one set per echo.
- **Majors.** The multi-TE formula omitted `T2'` and the fieldmap phase under gradient echo (now per
  echo formation, with tests in the conditions where it is exact). Salting only `slice_seed` left the
  image-space noise channel identical across echoes (`kspace.rs:1512-1521`): the call's seed is now
  split into an unsalted excitation seed and a salted receiver seed, tested with a non-`None`
  `prep_drive`. The timing checks admitted overlapping blocks and had no 2D home for the refocusing
  reserve: they are now explicit intervals, with `[multi_te] refocusing_time`. Hadamard's clock
  ignored P5's per-shot labelings: a raw volume is `NumberShots` preparations. Suppression (keyed by
  PLD) and VENC (per row) had no mapping onto cycles: both are resolved per cycle and must be
  constant within it. `m0scan` "anywhere" had no raw order: only between cycles, with published maps.
  Gradient-echo transients break the common tissue decoding needs: the physics is kept, the exact
  test is restricted, a transient negative case and a `TissueLeakage` report added. Exact decoding
  was over-promised under GRAPPA, spikes and `f32` output: conditions and a tissue-scaled tolerance
  stated. The physiology ground-truth test decoded the truth twice: an independent reference from
  single-row runs replaces it, and ideal vs raw truth are named apart. Look-Locker physiology is one
  labeling factor per cycle with tissue factors per readout, and motion stays volume-indexed. A
  separate M0 under a variable-flip Look-Locker had no flip: `[m0] flip_angle`. The regress gate had
  no P5 case: P5 cases are added and legacy dispatch is explicit. `TotalAcquiredPairs` for Hadamard is
  not a BIDS rule: a stated convention (encoding cycles).
- **Minor.** The Sylvester matrix has no all-labeled row: the partition test uses a synthetic weight
  vector.
