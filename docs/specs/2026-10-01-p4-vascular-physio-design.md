# P4: vascular compartments, bolus position, vascular crushing and physiological noise

Design spec addendum, 2026-10-01. Extends `2026-09-21-mrsim-acq-aslscan-design.md` (the main
spec) and follows `2026-09-24-p3-motion-bs-ir-design.md` (P3) and
`2026-09-24-p2-asldro-compat-design.md` (P2). Where this document is silent, those stand.
Codex-reviewed before implementation (2026-10-01, 2 blockers and 10 majors, all verified and
applied; the review is summarized at the end).

## Context

The roadmap gives P4 "kinetic model extensions: macrovascular compartment, physiological noise".
The earlier documents have since handed it more items, each because it needed a model of where
the labeled blood is:

- **The intravascular/extravascular split** (main spec, "Compartments"). P1 gives all of
  `delta_m` the blood's T2, right at short delay and wrong at long delay, where most of the label
  has exchanged into tissue: "the fix is a real intravascular/extravascular split with an
  exchange rate, which is P4 work."
- **Vascular crushing** (main spec, protocol contract): `VascularCrushing: true` is refused,
  naming P4.
- **Pulse history along the delivery** (P3, deferred item 1): the global-bolus factor inverts the
  whole bolus at every pulse, an upper bound on the retained label.
- **Partial-bolus inversion** (P3, deferred item 2): a pulse during labeling, refused today.
- **Motion of the labeling plane** (P3, deferred item 5).
- **A label-inverting IR preparation** (P3 deferred, P2 "the IR question").

The last two stay deferred (part F), with reasons.

## The one idea underneath

Every one of these is a statement about a **parcel** of labeled blood. For (P)CASL a parcel is
indexed by its labeling time `l` in `[0, tau]`: it reaches the tissue of voxel `x` at
`l + ATT(x)` and decays with `T1b` until then. For PASL the whole bolus is labeled at `t = 0`,
and a parcel is indexed by its arrival time `t'` in `[ATT, ATT + tau]`, decaying with `T1b`
until `t'`. After arrival a parcel resident for `s` seconds decays with `T1'`
(`1/T1' = 1/T1 + f/lambda`). `kinetic::delta_m` is the closed form of the integral of this kernel
over the parcels that have arrived, with simasl's guards (`kinetic.rs`, `gkm_filter.py:157-270`);
the guards are part of the model, not decoration (the PASL branch returns zero for
`T1b <= 0`, `kinetic.rs:85`), and every P4 formula keeps them.

P4 gives each parcel what P1 does not track: a **factor** (the product of `(1 - 2 epsilon)` for
each suppression pulse whose region it was inside at the pulse, part D, and its row's
physiological label factor, part E), a **residence split** between capillary and tissue
(part A), and a **macrovascular passage** before arrival (part B), where crushing acts (part C).

The factor is piecewise constant in `l`: it changes only where a pulse falls. The GKM is linear
in the bolus, so the bolus is cut into **sub-boluses** at those points, each an ordinary GKM
bolus, and the result is the factor-weighted sum. For (P)CASL, the parcels labeled in `[a, b]`
give

```
delta_m_sub(t; a, b) = delta_m(t - a; ATT, tau = b - a)         (the exp(-ATT/T1b) transit decay is the same for every parcel)
```

and for PASL, the parcels arriving in `[ATT + a, ATT + b]` give

```
delta_m_sub(t; a, b) = delta_m(t; ATT' = ATT + a, tau = b - a)   (no shift: exp(-t/T1b) already counts from t = 0)
```

These are exact identities, checked numerically before this was written (sum over partitions vs
the whole bolus, both labeling types, every delivery state and the mask edges: worst relative
difference `8e-16`). In floating point they hold to rounding, not bit for bit: at a cut, the
shifted argument `t - a - ATT` and the sub-bolus duration need not round alike, and near a mask
edge the relative difference reaches `1e-8` of a value that is itself `1e-8` of peak (Codex's
example). The contract is therefore **absolute**: `|sum - delta_m| <= 1e-12 * peak(delta_m)` over
the row, plus the exact-zero cases (not arrived, guards) staying exact zeros. Byte identity with
P1-P3 does not rest on it: cuts are made only strictly inside `(0, tau)`, and a bolus with no cut
calls `delta_m` with its original arguments, so the P1 path is the P1 code path.

## Scope

In: the residence split (A), the arterial compartment and its phantom inputs (B), vascular
crushing (C), the bolus-position suppression model and partial-bolus inversion (D),
physiological noise (E). Out, with reasons under part F and "Decisions deferred": labeling-plane
motion, the label-inverting IR preparation, bolus dispersion, global pulses during (P)CASL
labeling, a vessel-tree phantom, spatially structured physiological noise, and anything that
changes `mrsim-acq`.

**Every part is opt-in, and P1, P2 and P3 outputs are byte-identical when none is enabled.** The
P3 rule stands; the check is P2's (the pre-P4 binary in a sibling worktree, every NIfTI
decompressed and every sidecar compared), plus P2's benchmarks A-E passing unchanged. One input
changes on purpose (implementation review): a sidecar carrying `VascularCrushingVENC` with
`VascularCrushing` false or absent parsed before P4, its VENC ignored, and is refused now, since
"nothing is ignored" applies to it as to any inactive input. No accepted input changes its
output.

**Activation is resolved in one place and nothing is ignored.** A part is on when its inputs are
present (listed per part), and every P4 input that belongs to a part that is off is an error
naming the part, not a silent no-op: `t2_arterial` without part B, `[vascular_crushing]` without
`VascularCrushing: true`, `model`, `pulse_region` or `slab_entry_time` without suppression, a
lone `abv` or `aatt` map. The phantom's optional maps reach `protocol` through `PhantomParams`,
which gains `has_abv` and `has_aatt` (set by `phantom::load`), so activation is decided before
the compat check. Under `[compat] asldro = true` every part is refused, naming simasl.

---

# Part A: the intravascular/extravascular split

## The model

After arrival, label sits in the capillary and leaves it by exchange: the probability that a
parcel resident for `s` seconds has **not** yet exchanged is `exp(-s/tau_ex)` (single pass,
irreversible). The split is applied to the GKM's own parcel kernel, parcel by parcel:

```
dM_iv(t) = integral over arrived parcels of  kernel_GKM(parcel, t) * exp(-s/tau_ex)
dM_ev(t) = delta_m(t) - dM_iv(t)
```

Because `exp(-s/T1') exp(-s/tau_ex) = exp(-s/T1'')` with `1/T1'' = 1/T1' + 1/tau_ex`, the
intravascular part **is the GKM with `T1'` replaced by `T1''`**, and nothing else changed: the
same closed form, the same delivery masks, the same guards. `kinetic` gains
`delta_m_with_t1p(k, f, ATT, t1p, m0, t)`, the body of `delta_m` from the `t1p` line on, which
`delta_m` itself then calls (so `delta_m` is unchanged bit for bit), and `dM_iv` calls it with
`1/T1'' = div0(1, t1p) + 1/tau_ex` under the same `div0` guards. The mathematics is the same;
the PASL evaluation is not: its branch forms `exp(kk t)` and `exp(-kk dt)` separately
(`kinetic.rs:79`), and with `T1''` short `kk = 1/T1b - 1/T1''` is large and negative, so at
`tau_ex = 1e-6` those overflow and the product is `0 * (inf - inf)`, a NaN (plan review). The
intravascular PASL path therefore uses an algebraically equal form with the exponents combined
(`exp(-kk (dt - t))` and `exp(-kk (dt + tau - t))`, with `expm1` for the difference) and the
guards applied before any exponential; `delta_m` keeps its own form, untouched. The (P)CASL
branch has no such cancellation and is reused as is.

Properties, each a test:

- **Partition-invariant**: linear in parcels, so sub-boluses add (the reviewed design's
  per-sub-bolus fraction did not, by 0.2%; this one cannot fail that way). A pulse of zero
  efficiency, which only creates a cut, changes neither part.
- **Bounded**: `0 <= exp(-s/tau_ex) <= 1` per parcel, so `0 <= dM_iv <= delta_m` wherever
  `delta_m >= 0`, and both are zero before arrival. There is no fraction and no 0/0.
- **Limits**, stated at a residence: `tau_ex -> infinity` gives `dM_iv = delta_m` (P1: all
  blood T2); `tau_ex -> 0` gives `dM_iv -> 0` except for label that has just arrived, which
  correctly has not exchanged yet (at `t = ATT + 1e-7` with `tau_ex = 1e-6` the intravascular
  share is about 0.95; the test is at `t = ATT + 1` s, where it is about `1.5e-6`, below `2e-6`).

What this model leaves out: intravascular and exchanged water relax with the same `T1'` here,
where a full exchange model would give the capillary `T1b`. The GKM makes the same choice
(instant exchange), and keeping the GKM total exact is the point: the `delta_m` ground truth, the
P2 compatibility baseline and the meaning of perfusion stay P1's. A model whose own total
replaces the GKM's is a kinetic change, deferred.

## Where the parts go

The intravascular part stays in the blood compartments `K..2K` with `T2_blood`. The
extravascular part is added to the **tissue** compartment of its label (`0..K`): it has the
tissue's T2 and T2' exactly, so adding it there is exact under the acquisition's linearity and
costs no compartment. Signs follow the row semantics: `label` rows `-iv` in blood and `-ev` in
tissue, `deltam` rows `+iv` and `+ev`, `control` and `m0scan` rows neither. Under suppression
(P3) the tissue compartment holds the per-slice timeline plus the extravascular label; under
motion both move with the compartment, as today.

The linearity identity `I_C - I_L = I_B` holds with the split on. Its negative controls need two
additions: the `BloodIntoTissue0` control is run with a `tau_ex` that leaves a measurable
intravascular part (the test uses `tau_ex = 10` s), and a new `ExtravascularIntoBlood` override
(the `label` row's extravascular part sent to the blood compartment) must break it, since the
two compartments' T2 differ.

## Inputs

`[kinetic] exchange_time` (s, finite, positive) turns part A on. There is no default: published
values for brain water exchange span a range wide enough that any one would be a claim this
simulator cannot back, and the sidecar records the value given.

---

# Part B: the arterial (macrovascular) compartment

## The model

Label passing through the arteries of voxel `x`, on its way to tissue anywhere, is signal in `x`.
With arterial blood volume fraction `aBV(x)` and arterial transit time `aATT(x)` (labeling to
arrival in the voxel's arteries), and plug flow through the voxel fast against `tau` (the
arterial blood in the voxel is the parcel arriving now), the arterial difference magnetization is
the macrovascular term of Chappell et al. (MRM 2010) without dispersion:

```
(P)CASL: dM_art(t) = 2 alpha M0b aBV exp(-aATT/T1b) g(l = t - aATT)   for aATT <= t < aATT + tau
PASL:    dM_art(t) = 2 alpha M0b aBV exp(-t/T1b)    g(t' = t)         for aATT <= t < aATT + tau
```

zero otherwise, `M0b = M0 / lambda` per voxel with the GKM's `lambda` and `T1b` guards (`T1b`
tested `!= 0` for (P)CASL and `> 0` for PASL, as `kinetic.rs` does), and `g` the parcel's factor
(1 without parts D and E). The arterial component is independent of the voxel's own tissue
kinetics: a passing artery may feed tissue elsewhere, so no ordering between `aATT(x)` and
`ATT(x)` is assumed or checked.

## Compartments

The arterial compartment is decomposed by label, for the reason the main spec gives for the
blood: its T2' is that of the tissue it sits in, and a single compartment carrying a T2' map
would take every slice off the NUFFT path. The layout becomes

```
compartments[0 .. K)     tissue, one per label (plus the extravascular label, part A)
compartments[K .. 2K)    labeled blood, intravascular
compartments[2K .. 3K)   arterial blood, one per label        (only when part B is on)
```

with `T2 = T2_arterial` and the label's T2'. The main spec's single index `2K` was a placeholder
and is replaced by the range. In `voxel` mode it is one more compartment, with the tissue T2'
map. `T2_arterial` comes from `[signal] t2_arterial` (s) and defaults to the resolved
`T2_blood`, recorded as such: the phantom has no arterial T2, and this simulator does not invent
a field-dependent value for it.

## Phantom inputs

Two optional phantom maps, `abv.nii.gz` (`Units: "fraction"`, in `[0, 1]`) and `aatt.nii.gz`
(`Units: "s"`, non-negative), on the phantom grid, or per-label values in the overlay:

```toml
[macrovascular]
arterial_blood_volume = { grey_matter = 0.02, white_matter = 0.008, csf = 0.0 }
arterial_transit_time = { grey_matter = 0.55, white_matter = 0.8, csf = 0.0 }
```

(values illustrative, not defaults). Keys are `dseg.json` label names; every foreground label
needs a value; unknown names are errors. Part B is on when `[macrovascular]` is present or both
maps are. A quantity given both as a map and in the overlay is an error rather than a precedence
rule, and one map without the other (with no overlay value for the missing quantity) is an
error. The ASLDRO phantoms carry neither, so the converter is unchanged and part B stays off on
them until a user supplies values.

---

# Part C: vascular crushing

## The model

Crusher gradients dephase moving spins by a phase proportional to their velocity along the
gradient. **The simulator assumes** that `VascularCrushingVENC` is the velocity that acquires a
phase of `pi`. BIDS defines the field only as the crusher strength in cm/s (DICOM's crusher flow
limit), so this convention is a declared assumption, recorded in the sidecar, and a user
matching a specific sequence calibrates `VENC` to it.

Within a voxel the arteries run in all directions (isotropic) and carry a laminar (Poiseuille)
profile, whose volume-weighted speed is uniform on `[0, v_max]`. The projected velocity of a
speed `v` at a random direction is uniform on `[-v, v]`, which averages the phase factor to
`sin(pi v/VENC)/(pi v/VENC)`; averaging that over speeds gives the surviving fraction of the
arterial signal:

```
c = Si(pi r) / (pi r),     r = v_max / VENC,     Si(x) = integral from 0 to x of sin(u)/u du
c = 1 for VENC = 0 (crushing off for that volume)
```

real (the isotropic average leaves no net phase), between about 0.59 at `r = 1` and 0 as
`r -> infinity`, monotone over the range a crusher is used in. `Si` is evaluated by its power
series for `x <= 2` and, beyond, as `pi/2 + Im E1(i x)` with `E1` from its complex continued
fraction (modified Lentz), which is accurate for every larger `x` with no asymptotic switch;
tested against `scipy.special.sici` at `1e-12`.

Capillary and tissue water are assumed unaffected: their velocities are orders of magnitude
below `v_max`. That is a model assumption, recorded, not a computed bound; for a `VENC` below
`0.1` cm/s the assumption is not defensible and the input is refused.

## Inputs

`VascularCrushing: true` with part B on turns part C on. `VascularCrushingVENC` is then required:
a number for every volume, or an array of one per volume, `0` meaning crushing off for that
volume (BIDS), so a QUASAR-style alternation is one series. `[vascular_crushing]
arterial_velocity` gives `v_max` per label (cm/s, keys as in part B). `VascularCrushing: true`
**without** part B is refused, naming `[macrovascular]`, unless `[vascular_crushing]
no_arterial_compartment = true` says the user accepts that the crushers act on nothing modeled;
that flag is recorded and the run is then P1-P3's, plus the echo of the standard fields.

## What the acquisition stage is not asked to do

The main spec reserved `PrepPhase` and the crusher moment as `eddy_drive` for this. Neither is
used: `aslscan` keeps passing `None` for both. Crusher-induced eddy currents and the bulk-motion
phase the crushers would imprint are **unmodeled**, and the sidecar says so. (The reviewed draft
argued them negligible from a b-value ratio; that argument does not hold, since the preparation
phase scales with the square root of b, and `VENC` alone does not determine the crusher's
b-value or waveform.) A consequence worth having: with both drives `None`, `do_eddy` stays false
(`kspace.rs:471-473`), so crushing keeps the NUFFT path, and the main spec's remark that
crushing gives it up no longer applies. `mrsim-acq` is unchanged.

---

# Part D: the bolus-position suppression model

## The model

`[background_suppression] model = "bolus-position"` (default `"global-bolus"`, P3, unchanged).
A pulse at `p` multiplies a parcel's factor by `(1 - 2 epsilon)` when the parcel is inside the
pulse's region at `p`. The region is set by `pulse_region` (required with this model):

| `pulse_region` | A (P)CASL parcel `l` is inside from | A PASL parcel `t'` is inside from |
|---|---|---|
| `"global"` | `l` (its labeling) | `0` (its labeling) |
| `"slab"`, `slab_entry_time = d` | `l + d` | `t' - (ATT - d)` |
| `"slab"`, `slab_entry_time = "arrival"` | `l + ATT` | `t'` |

PASL's entry is the arrival less the in-slab travel `ATT - d`, the same for every parcel under
plug flow. For the arterial compartment the same rule holds with `aATT` in place of `ATT`. `d`
must not exceed the `ATT` (or `aATT`) of any voxel whose label it applies to, else that label
would enter the slab after reaching the voxel; the error names the voxel.

Each pulse whose entry-time cut falls strictly inside the bolus splits it there: at most `N + 1`
sub-boluses per voxel and row, each one GKM evaluation (and one more for part A's intravascular
part). The arterial compartment needs no partition: at any `t` it holds one parcel, whose factor
is evaluated directly. The tissue's own timeline is P3's, unchanged: tissue is always inside the
region.

## Reductions, and a correction to P3

**`"global"` with every pulse after the bolus is created is P3's global-bolus model**: every
parcel is inside from its labeling, so every parcel sees every pulse and the factor is P3's
`label_factor`. The test is equality to `1e-12` on every row.

**P3's slab-confined estimate was wrong.** P3 worked `asl002`'s GM by hand under "pulses act on
delivered label only" (here `"slab"` with `"arrival"`) and got about `0.33` of the unsuppressed
difference. `asl002` is PCASL. Its first pulse, at 2.05 s, cuts the labeling time at
`2.05 - 0.8 = 1.25` s; the second, at 3.276 s, falls after all delivery. So parcels `l <= 1.25`
are inverted twice (factor `+1`) and the rest once (`-1`). For PCASL the transit decay is the
same for every parcel, and at the readout the parcel weights are proportional to `exp(l/T1')`,
which gives

```
F = 2 (exp(1.25/T1') - 1) / (exp(1.8/T1') - 1) - 1 = 0.0821,     T1' = (1/1.33 + 0.01/0.9)^-1
```

The `0.33` came from weighting parcels as PASL ones (`exp(l (1/T1' - 1/T1b))` gives 0.328).
The conclusion P3 drew stands, and is stronger: the global-bolus factor of `+1` is an upper bound
far above the slab-confined value. P3's text carries a one-line correction pointing here. The
test is `0.0821045` from the closed form above and from an independent parcel quadrature, to
`1e-6`.

## Partial-bolus inversion

P3 refuses a pulse before `tau` (or before the PASL cutoff). Under bolus-position:

- `"slab"`: allowed. The pulse acts on parcels already in the slab and never on blood upstream of
  it, so the labeling of later parcels is undisturbed; the fraction affected comes out of the
  entry times rather than P3's assumed `p / tau`.
- `"global"`, PASL: allowed. Every delivered parcel was labeled at `0`, and control and label see
  the pulse alike, so the difference is multiplied by `(1 - 2 epsilon)`.
- `"global"`, (P)CASL: still refused. The pulse also inverts blood not yet labeled, whose
  longitudinal magnetization then recovers until its labeling, so later parcels carry a reduced,
  labeling-time-dependent difference: that needs the inflowing blood's history, deferred, and
  the error says so.

P3's other constraint stays: every pulse precedes the first slice's excitation. IR with
suppression stays refused under either model (part F).

---

# Part E: physiological noise

## The model

Three processes on the series' clock, all drawn from their own stream so that turning part E on
leaves the acquisition noise and the motion draws unchanged (the noise is added in k-space
independently of the signal, so with `ParallelReductionFactorInPlane = 1` the reconstructed
noise is identical too; with GRAPPA only the draws are, since the reconstruction weights depend
on the calibration data):

- **Cardiac and respiratory phase.** Initial phase uniform on `[0, 2 pi)`; successive periods
  drawn independently from a normal of mean `1/f` and coefficient of variation `cv`, truncated
  at `±3 sd`, with `cv <= 0.3` required so every period is positive; the phase advances by
  `2 pi` per period, linearly within it. Defaults `f_c = 1.0` Hz, `cv_c = 0.1`, `f_r = 0.25` Hz,
  `cv_r = 0.2`.
- **Drift.** An Ornstein-Uhlenbeck process of unit variance and time constant `tau_d` (default
  30 s), started from its stationary distribution, sampled on a fixed grid of `0.05` s from the
  series start (exact discretization `x_{k+1} = x_k exp(-dt/tau_d) + sqrt(1 - exp(-2 dt/tau_d)) z`)
  and linearly interpolated between grid points.

The clock: row `v` starts at `T(v) = sum of RepetitionTimePreparation over rows before v`
(BIDS: each row's preparation block follows the previous one's); its labeling starts at `T(v)`;
its slice `z` is read at `T(v) + t + offset(z)`.

The streams: `SplitMix64(seed ^ PHYSIO_SEED_SALT)` with `PHYSIO_SEED_SALT = 0x5048_5953_494F`
("PHYSIO"), from which three sub-streams are seeded by XOR with `1`, `2` and `3` (cardiac,
respiratory, drift); within each, draws are taken in time order, normals by Box-Muller (both
values of each pair used, in order), which `aslscan::rng` gains for this (today it has uniforms
only). Normative, so a dataset can be regenerated from its sidecar.

Two factors, each `1 + a_c sin(phi_c) + a_r sin(phi_r) + a_d x`:

- **Tissue**, evaluated at each slice's readout time and multiplying that row's static tissue
  magnetization in that slice (the signal-proportional fluctuation that dominates ASL noise, the
  `lambda` term of Krüger and Glover).
- **Label**, averaged over the labeling window `[T(v), T(v) + tau]` for (P)CASL (the average of
  `sin` over a piecewise-linear phase and of the piecewise-linear drift are both exact), or at
  `T(v)` for PASL, multiplying the row's whole label: intravascular, extravascular and arterial.

A separate M0 scan (`M0Type: Separate`) is not a row of the series and has no time on its
clock, so neither factor applies to it; it keeps its P1 path, and the sidecar says so.

Both are global (one value per time, the same in every voxel); spatially structured
physiological noise is a different model, deferred. The six amplitudes (`tissue_cardiac`,
`tissue_respiratory`, `tissue_drift`, `label_cardiac`, `label_respiratory`, `label_drift`)
default to zero, and a `[physio]` table whose six are all zero is an error, as a motion mode with
zero amplitudes is (P3). Part E is on when `[physio]` is present.

## Ground truth

`desc-physio_gt.tsv`, one line per volume and slice: the readout time, both phases at it, the
drift at it, the tissue factor; and per volume the labeling window, the window averages of
`sin phi_c`, `sin phi_r` and the drift, and the label factor. Everything needed to reproduce the
factors, and the true phases a correction method can be scored against.

---

# Part F: what stays deferred, and why

**Labeling-plane motion** was given to P4 "since it needs the vascular geometry P4 introduces".
P4 does not introduce a geometry: the arterial compartment is voxelwise (`aBV`, `aATT`), with no
vessel positions at the labeling plane, and that is the model a phantom can be given. The
efficiency change with head motion depends on the angle and off-resonance of the feeding arteries
at the plane, which a voxelwise model cannot express. Deferred to a vessel-tree phantom,
unscheduled.

**The label-inverting IR preparation.** The reviewed draft made it a pulse at `t_read - TI` in
the bolus-position model. That is not a consistent model: `mrsignal::tissue_ir` is a periodic
steady state for an arbitrary excitation angle, while the suppression timeline
(`longitudinal.rs:44`) assumes a 90-degree excitation leaves zero, so the tissue side cannot be
the same event; a per-slice pulse at `t_read(z) - TI` is a different preparation for every slice,
not one global pulse; and `TI` may exceed `t_read` or fall inside (P)CASL labeling, where the
inflowing-blood history (above) is needed. It needs one longitudinal recurrence for tissue
under IR and suppression together, with an excitation residual, a defined initial state and
event ordering. Deferred, together with IR composed with suppression; P2's benchmark E remains
the baseline it will be measured against.

---

# Outputs

**Sidecar**, under `AslscanSimulation`, each block only when its part is on (so P1-P3 sidecars
are unchanged):

- `Exchange`: `ExchangeTime`, `Model: "single pass, irreversible, applied to the GKM kernel per
  parcel"`.
- `Macrovascular`: per-label `ArterialBloodVolume` and `ArterialTransitTime` or the map names,
  `T2Arterial` with its source, `Model: "plug flow, no dispersion (Chappell 2010)"`.
- `VascularCrushing`: per-volume `VENC`, per-label `ArterialVelocity`, per-volume and per-label
  survival `c`, `Model: "isotropic laminar"`, `VencConvention: "phase pi at VENC (assumed)"`,
  `Unmodeled: ["crusher eddy currents", "crusher bulk-motion phase", "capillary flow"]`, or
  `NoArterialCompartment: true`. The standard `VascularCrushing` and `VascularCrushingVENC` are
  echoed as given.
- `BackgroundSuppression.Model: "bolus-position"`, `PulseRegion`, `SlabEntryTime`, and
  `BackgroundSuppressionModel` likewise; `LabelFactor` is then absent (it is no longer one number
  per row) and `desc-deltamSuppressed_gt` carries it.
- `Physio`: every amplitude, frequency, `cv`, `tau_d`, the grid step, the seed and salt.

**Ground truth.** The signal is formed in a fixed order, and each truth is named by the stages it
includes:

```
kinetics -> exchange split (A) -> pulse factors (D) -> physiological label factor (E)
         -> crushing (C, arterial only) -> excitation (signal equation) -> T2 at readout
         -> pose and shot events (P3) -> box average to the acquisition grid
```

| Output | Stages | Under motion |
|---|---|---|
| `desc-deltam_gt` | kinetics (unchanged) | moved, with `desc-deltamStatic_gt` (P3) |
| `desc-deltamIntravascular_gt` | kinetics, exchange split | moved, like `desc-deltam_gt` |
| `desc-deltamSuppressed_gt` | kinetics, pulse factors (tissue label, both parts) | moved |
| `desc-deltamArterial_gt` | kinetics (the arterial term with `g = 1`) | moved |
| `desc-aBV_gt`, `desc-aATT_gt` | phantom maps (`aATT` as a mean over voxels with `aBV > 0`) | static |
| `desc-physio_gt.tsv` | part E's factors and phases | not spatial |

"Moved" is P3's convention: the simulation-grid quantity moved by the per-volume poses (no shot
events) and then block-averaged, so a truth and the data are in the same frame and ratios of
truths are meaningful. The effective suppression factor is the ratio of
`desc-deltamSuppressed_gt` to `desc-deltam_gt`, undefined where the latter is zero; no ratio map
is written, so no sentinel has to be chosen.

---

# Division of labor

`aslscan`:
- `kinetic`: `delta_m_with_t1p` (with `delta_m` calling it), `delta_m_sub`, `delta_m_iv`,
  `arterial_dm`;
- a new pure-std module `bolus`: entry times, the per-voxel partition, the parcel factors;
- a new pure-std module `physio`: the phase processes, the OU drift, the window averages;
- `crushing` (pure std, or inside `mrsignal`): `Si` and the survival `c`;
- `protocol`: `[kinetic] exchange_time`, `[macrovascular]`, `[signal] t2_arterial`,
  `VascularCrushing*` and `[vascular_crushing]`, `[background_suppression] model /
  pulse_region / slab_entry_time`, `[physio]`; the activation and refusal rules;
- `phantom`: the optional `abv` and `aatt` maps and the `PhantomParams` flags;
- `series`: the compartment layout, the per-voxel split and factors, the physiological factors;
- `bids`: the blocks and ground truth above.

`mrsim-acq`: no change. The compartment count is already arbitrary, `eddy_drive` and
`prep_drive` stay `None`, and every new compartment carries a uniform T2 in `class` mode.

# Testing and verification

- **kinetic**: sub-boluses over random partitions of `[0, tau]` against `delta_m` within
  `1e-12 * peak` (both labeling types, every delivery state, cuts at representable neighbours of
  the mask edges), exact zeros kept; `delta_m` unchanged bit for bit after the refactor (the
  `gkm.txt` fixture diff and a direct comparison of the old and new bodies on a grid);
  `delta_m_iv`: partition invariance, a zero-efficiency pulse changing nothing, `0 <= iv <= dm`,
  the limits at the stated residences, the guards for `lambda = 0` and `T1b <= 0`;
  `arterial_dm` against its closed forms and zero outside its window.
- **bolus**: `"global"` with post-labeling pulses equals `label_factor` for every row
  (`1e-12`); `"slab"` with `"arrival"` on `asl002`'s GM gives `0.0821045` by the closed form and
  by quadrature (`1e-6`); PASL and arterial entry times; the partial-bolus rules, including the
  (P)CASL `"global"` refusal; a pulse exactly at a sub-bolus edge.
- **crushing**: `Si` against tabulated values (`1e-12`), `c(0) = 1`, `c(1) = 0.58949`, monotone
  decrease on `[0, 3]`; `VENC = 0` on a volume leaves its arterial signal untouched; the
  `VENC < 0.1` refusal; crushing without part B refused, and with `no_arterial_compartment`
  byte-identical to the run without crushing.
- **physio**: period statistics over 10 000 draws (mean and cv within 2%), the exact window
  averages against quadrature (`1e-10`), the OU lag correlation at `tau_d`; the stream
  convention reproduces a fixed reference sequence; acquisition noise bit-identical with part E
  on and off at `ParallelReductionFactorInPlane = 1`; the ground-truth TSV reproduces the
  applied factors.
- **series** (crop, homogeneous grid, no noise, no GRAPPA, no spikes): a run with part A on and
  `T2_blood = T2_tissue` equals the P1 run to float32; the arterial compartment's image equals
  `arterial_dm` per voxel, box-averaged, in compat-like settings; a matched pair of one-row runs
  differing only in `VENC` differs by `sum over labels of (c_i - 1) A_i`, `A_i` the label's
  arterial image, as complex images; the linearity identity holds with every part on, its runs
  being one-row series so they share the physiological factors and poses, and fails for the
  `ExtravascularIntoBlood` override and (at `tau_ex = 10` s) for `BloodIntoTissue0`; every
  activation route and every refusal.
- **end to end**: P2's byte-identity regression with P4 off; P2's benchmarks A-E; a validator run
  on a dataset with every part on.

# Acceptance criteria

1. With no P4 input, every P1-P3 output is byte-identical to the pre-P4 build, and P2's
   benchmarks A-E pass unchanged.
2. The sub-bolus identity, the intravascular part, the arterial term and the crushing survival
   pass their tests at the tolerances above.
3. `"global"` reproduces P3's factor exactly; the `"arrival"` slab gives `asl002`'s GM factor as
   `0.0821045` by closed form and quadrature.
4. The matched `VENC` pair isolates the arterial signal per label to float32 precision.
5. Physiological noise reproduces its statistics and reference sequence, leaves the acquisition
   noise unchanged, and its ground truth reproduces the factors applied.
6. Every part is refused under compat and every inactive input is refused; a dataset with every
   part on validates with no errors.
7. `mrsim-acq` is unchanged; `cargo test` is green in the default build of both crates.

# Decisions deferred

- **Bolus dispersion** (a gamma or Gaussian arrival kernel). The partition is exact for plug
  flow; dispersion makes it a convolution. Unscheduled.
- **Global pulses during (P)CASL labeling**: needs the inflowing blood's longitudinal history.
- **The label-inverting IR preparation, and IR with suppression**: part F.
- **Labeling-plane motion** and **a vessel-tree phantom**: part F.
- **Spatially structured physiological noise** and cardiac-gated arterial volume.
- **The exchange model's own total** (capillary `T1b` until exchange) in place of the GKM's: a
  kinetic change, not a split.
- **Per-compartment NUFFT eligibility** (main spec): still unneeded, since every P4 compartment is
  uniform in `class` mode.

# Risks

**The arterial compartment's parameters have no source in the shipped phantoms.** Part B is
inert until a user supplies `aBV` and `aATT`, and there are no defaults to hide that.

**The crusher model rests on a declared convention.** The phase-at-`VENC` assumption and the
isotropic laminar distribution are both named in the sidecar; a dataset compared against a real
sequence needs `VENC` calibrated, which this simulator cannot do for it.

**The bolus-position model is still a model.** Plug flow and one in-slab travel time per voxel
stand in for a vascular tree. Its exact reduction to P3's factor and its exact `"arrival"` value
bound the useful range from both sides.

**Physiological noise aliases.** At ASL repetition times the cardiac process is sampled far below
its Nyquist rate. That is the real situation, and the ground truth carries the true phases.

# Review record

Codex adversarial review, 2026-10-01, of the first draft. Applied: the exchange split was a
per-sub-bolus fraction of the GKM total, which depends on where the bolus is cut (0.2% in the
review's example), and is now a per-parcel survival on the GKM kernel; the fraction's 0/0 at
arrival and its limit tests went with it; the sub-bolus identity's tolerance is absolute and the
P1 path no longer depends on it; P3's `0.33` slab estimate was a PASL weighting of a PCASL
protocol, corrected to `0.0821`; the IR label inversion had no consistent tissue history and is
deferred; the crusher survival is the isotropic laminar `Si(pi r)/(pi r)` (the draft's sinc
assumed a different distribution) and the phase-at-`VENC` convention is declared rather than
attributed to BIDS; crusher eddy and preparation phase are unmodeled rather than argued
negligible; the `aATT <= ATT` check contradicted the model and is gone; crushing without the
arterial compartment, lone maps and inactive inputs are refused; the physiological processes'
initial states, grid, streams and salt are specified; the test conditions for linearity, the
`VENC` pair and noise identity are stated; each ground truth's stages and motion frame are
defined.
