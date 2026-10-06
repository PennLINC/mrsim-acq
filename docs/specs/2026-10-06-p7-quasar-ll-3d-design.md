# P7: QUASAR under Look-Locker, Look-Locker combinations, and 3D gradient-echo trains

Design spec addendum, 2026-10-06. Extends `2026-09-21-mrsim-acq-aslscan-design.md` (the main
spec) and follows the P2-P6 addenda (`2026-09-24-p2-asldro-compat-design.md`,
`2026-09-24-p3-motion-bs-ir-design.md`, `2026-10-01-p4-vascular-physio-design.md`,
`2026-10-01-p5-3d-readouts-ge-design.md`, `2026-10-05-p6-encoding-design.md`). Where this
document is silent, those stand. Reviewed once (Claude, 2026-10-06: 4 majors and 9 minors, all
applied; the record is at the end). The Codex review is pending.

## Context

P6 deferred four items that compose existing pieces rather than adding new physics (P6,
"Decisions deferred"). The user chose three of them for P7 (2026-10-06):

- **Part A:** the QUASAR arterial compartment, crushing, exchange and bolus-position suppression
  under Look-Locker readouts.
- **Part B:** Look-Locker × multi-TE and Look-Locker × Hadamard.
- **Part C:** Look-Locker and multi-TE with 3D readouts. These need a 3D gradient-echo readout,
  which is new in `mrsim-acq`.

Velocity-selective labeling, vessel encoding and multi-echo 3D spin-echo trains stay deferred.

What exists today, checked against the source (`aslscan` and `mrsim-acq` at `p6-complete`:
`1df23de`, `c20cc7a`):

- **Look-Locker is refused with every P4 part.** `look_locker_spec` (`protocol.rs:1235`) refuses
  `LookLocker: true` with:
  - 3D;
  - `[hadamard]`;
  - multi-TE;
  - compat;
  - `exchange_time` (P4 part A);
  - the arterial compartment (part B);
  - `VascularCrushing` (part C);
  - the bolus-position suppression model (part D).
- **The Look-Locker blood term is one arrival-window sum.**
  - `delta_m_arrival` (`kinetic.rs:238`) is the GKM delta-M at `t` of the label that arrived in
    `[u1, u2)`, with relaxation `T1'` after arrival.
  - `delta_m_read` (`kinetic.rs:265`) sums it over the windows between excitations, weighting
    each window by the `cos(a)` of every later readout.
  - The full bolus is the only labeling it takes. There is no sub-bolus or intravascular form.
  - Its PASL branch forms `exp(kk t)` and `exp(-kk u)` separately (`kinetic.rs:248`). This is
    the form `delta_m_iv` had to avoid for short `T1''` (`pasl_stable`, `kinetic.rs:189`).
- **The pieces P7 composes exist as single-readout functions.**
  - The intravascular part `delta_m_iv` (`kinetic.rs:156`) is the GKM with `T1'` replaced by
    `T1'' = (1/T1' + 1/tau_ex)^-1`.
  - The arterial term `arterial_dm` (`kinetic.rs:213`) holds one parcel at `t` (`t - aATT`), with
    no depletion state.
  - The crushing survival `crushing::survival(v_max, venc)` (`crushing.rs:81`) is a per-volume
    scalar.
  - The bolus-position factors split a bolus at pulse entry-time cuts into sub-boli, evaluated by
    `delta_m_sub`'s shifts (P4 part D).
- **The Look-Locker tissue timeline takes any event list.**
  - `LlCycle { tr, s, t_read, flip_deg }` (`longitudinal.rs:150`) applies each readout as a
    `cos(a)` event.
  - `tissue_mz_ll_sequence` carries state between cycles.
  - Every map is affine with an offset proportional to `M0`, so tissue `Mz` before every readout
    is `M0(r)` times a function of `T1(r)` and the schedule.
- **3D is spin echo only.**
  - `acq_contrast "ge"` with `MRAcquisitionType 3D` is refused (`protocol.rs:1821`), and so is
    multi-TE with 3D (`protocol.rs:1674`).
  - `simulate_acquisition_3d` (`kspace3d.rs:447`) takes an `EchoTrain` (one 90-degree excitation
    per shot, then refocusing pulses) and a `Readout3d` (`Grase` or `Spiral`, `readout.rs:103`).
  - Its decay per compartment is a line weight `exp(ln_a - t/T2 - |t|/T2')`, where `ln_a` is
    EPG's per echo (`Mode::Class`), or per-voxel `ln_a` maps (`Mode::Voxel`,
    `kspace3d.rs:286-330`).
  - Physiology and shot gains enter as `LineWeights`, one scalar per `(volume, shot, compartment)`
    (`kspace3d.rs:42`). Per-shot images enter as `ShotSet`s.
- **BIDS fixes the 3D post-labeling delay at the slab excitation** (P5 addendum, "Inputs"):
  "until the middle of the excitation". It defines no meaning for a train of excitations.
- **The 3D path takes every volume's images in one call.** Compartment images are laid out
  `vox * n_volumes + g` (`kspace3d.rs:201`, `:250`).
- **The 3D separate M0 is a spin-echo readout** at its own TR (`series/p6.rs:1325`). P5 made the
  2D gradient-echo M0 use the series' own excitation (P5 addendum, "M0").
- **P6's Hadamard aslcontext lists the decoded volumes, not the raw ones.**
  - `hadamard_spec` refuses `control` and `label` rows (`protocol.rs:1112-1116`). The context has
    `order - 1` `deltam` rows per encoding cycle, and `m0scan` rows only between cycles.
  - Each `deltam` row's delay must be its sub-bolus's effective delay: the PLD plus the later
    sub-boli's durations (`protocol.rs:1156-1165`).
  - The raw volumes go to `sourcedata/`. `TotalAcquiredPairs` is the number of encoding cycles
    (`bids.rs:597`).
- **P6's depletion rests on P1's geometry.** Label in transit is outside the imaged slices and is
  not depleted (P6 part B).

**Terms.** A **Look-Locker cycle** is one preparation and its readouts. An **encoding cycle** is
P6's pass through the Hadamard matrix: `H` raw volumes, each of one preparation in 2D and
`NumberShots` preparations in 3D. "Cycle" alone means a Look-Locker cycle.

## The one idea underneath

P6 made each readout an event on the `Mz` timeline. Each label window between readouts is
depleted by the `cos(a)` of every readout after it. P7 applies that rule in three places:

- **Part A** applies it to each component of the blood signal. Each is a sum over arrival
  windows:
  - the intravascular and extravascular parts (`T1''` and `T1'` residues);
  - each bolus-position sub-bolus, with its own factor;
  - each Hadamard sub-bolus.

  In 2D the arterial blood is the exception. It is read as it passes, so it is not depleted.
- **Part B** has no new physics. An echo does not touch `Mz`, so multi-TE reads every
  Look-Locker image E times. Hadamard decoding is linear, so it runs per readout index.
- **Part C** treats each excitation of a 3D gradient-echo train as an event on the same timeline.
  The train's excitations then see different longitudinal states. Each kz partition is read by
  one excitation, so the train becomes a weight per partition. This is the gradient-echo
  counterpart of GRASE's per-echo EPG amplitudes. A Look-Locker cycle in 3D is a series of such
  sub-trains, one per readout time. A 3D excitation covers the slab, so label is depleted from
  its **entry into the slab**, not from its arrival in the voxel. This shifts P6's arrival
  windows by a per-voxel constant.

## Scope

**In scope:**

- Part A, for 2D EPI Look-Locker. The arterial compartment, crushing (a per-volume VENC
  alternation, so a QUASAR series is one series), exchange, and bolus-position suppression. Any
  subset of them can be used together.
- Part B. Look-Locker × multi-TE, Look-Locker × Hadamard, and all three together.
- Part C. A 3D gradient-echo stack-of-EPI readout in `mrsim-acq`, with one or more echoes per
  excitation. In `aslscan` it supports:
  - single-readout 3D gradient-echo ASL;
  - 3D Look-Locker, segmented across cycles;
  - multi-TE 3D gradient echo;
  - their combinations with Hadamard and the Part A components.

**Out of scope** (reasons under "Decisions deferred"):

- multi-echo GRASE and spiral trains;
- 3D gradient-echo spiral;
- in 2D, depletion of label crossing other imaged slices before it reaches its own (3D models
  depletion from slab entry, Part C);
- balanced (SSFP) gradient-echo trains;
- compat for any P7 feature;
- velocity-selective and vessel-encoded labeling.

**Byte identity.** Without P7 inputs, every output stays the same:

- **TRXScan** (`p0-mrsim-acq-extraction`, `12a9d09`): the P0 gate. P7 adds no field to
  `Acquisition`, `SliceInput`, `EchoTrain`, `Readout3d` or `LineWeights`, and changes no
  signature. The new readout has its own types and its own entry point.
- **mrsim-acq**: the golden record and every bit-pinned test, including the 3D GRASE and spiral
  pins.
  - Internals of `kspace3d` may be shared with the new path, such as the z-kernel, the
    per-partition noise and the coil reconstruction. They must stay bit-identical for the
    existing calls, which the pins check.
- **aslscan**: `tools/regress_identity.sh` against `p6-complete` for both repositories and both
  feature sets.
  - Its 49 cases are P1-P5. P7 first adds P6 cases, from the existing fixtures:
    - `p6_hadamard`, `p6_hadamard_grase`, `p6_hadamard_multite`;
    - `p6_multite`, `p6_multite_se`;
    - `p6_ll`, and a multi-readout Look-Locker case with an included M0.
  - These run against `p6-complete` before any P7 change.
- **Explicit legacy dispatch**, as in P6.
  - A Look-Locker series without a Part A or B input takes P6's code path. P6's `delta_m_read` is
    unchanged. The new arrival forms are new functions whose reduction to it is tested, not
    substituted for it.
  - A 3D spin-echo series takes `simulate_acquisition_3d` as today.

**Activation and refusals** follow the P4 rule. Each P7 combination becomes allowed where P6
refused it. An input for a feature that is off is still an error naming it. Every combination
that is still refused is an error naming the reason and this addendum's part.

---

# Part A: QUASAR and the P4 parts under Look-Locker

## What is modeled

A Look-Locker cycle has one preparation followed by readouts at excitation times
`e_1 < ... < e_M` (per slice, in seconds from the start of labeling), with flips `a_1 ... a_M`.
P6's blood term for readout `n` is

```
dM_read(e_n) = sum_{w=0}^{n-1} D_{w,n} dM_arr(e_n; [e_w, e_{w+1}))
D_{w,n} = prod_{w < m < n} cos(a_m)
```

with `e_0 = 0`. `D` is the depletion of the label that arrived in window `w`. Each P4 part
changes what is summed or adds a term. The changes compose because each is linear in the label.

**Exchange (P4 part A).**

- The intravascular label is the parcels that have not yet exchanged. This is the GKM with
  residue `T1''`. Its arrival-window form is `delta_m_arrival` with `T1'` replaced by `T1''`:
  `dM_arr_iv`.
- The extravascular label in each window is `dM_arr - dM_arr_iv`.
- Both are in the voxel, so both are depleted by every later readout with the same `D_{w,n}`.
- The intravascular part goes to the blood compartment and the extravascular part to the tissue
  compartment, as in P4.
- The PASL branch combines its exponents as `pasl_stable` does:
  `exp(kk (t - u_hi)) expm1(kk (u_hi - u_lo))`. This keeps it finite when a short `tau_ex` makes
  `kk` large and negative. Each window's intravascular part is clamped to `[0, dM_arr]`, as
  P4's PASL clamp does for the whole bolus.
- The (P)CASL form is used as derived. It is a difference of two residue exponentials with no
  overflow.
- As in P4, exchange is a split of the GKM's single residue. It is not a two-compartment `T1`
  exchange model, and the sidecar says so.

**Bolus-position suppression (P4 part D).**

- The pulses of a Look-Locker cycle all precede its first readout (P6, part B). Each pulse whose
  entry-time cut falls inside the bolus splits the bolus into sub-boli `[a_i, b_i]`, each with its
  factor `F_i`, as in P4.
- The read is `sum_i F_i sum_w D_{w,n} dM_arr_sub(e_n; [e_w, e_{w+1}); a_i, b_i)`.
- `dM_arr_sub` is `delta_m_arrival` with P4's sub-bolus shifts:
  - (P)CASL: a bolus of length `b - a`, read at `t - a`, with its arrival window shifted by `-a`;
  - PASL: arrival delay `dt + a`.
- The arrival windows and the sub-bolus cuts are independent partitions, one in arrival time and
  one in labeling time. Each term is one closed-form evaluation, so the cost per voxel and readout
  is `(N + 1) n` evaluations, doubled with exchange.
- `"global"` with every pulse after the bolus forms reduces to the global factor times P6's read.
  This reduction is tested.

**The arterial compartment (P4 part B).**

- The arterial blood at `e_n` is the parcel `e_n - aATT`, which is passing through. QUASAR's model
  treats it as fresh: blood excited by an earlier readout has left the voxel's arteries before
  the next one. This is P4's model, where the compartment holds one parcel at any `t`.
- Readout `n` reads `sin(a_n) arterial_dm(e_n)`, with no depletion, times the bolus-position
  factor of that parcel. As in P4, that factor is evaluated directly with no partition.
- This is the QUASAR assumption (Petersen et al., MRM 2006), stated in the sidecar. It holds in
  2D when blood crosses a slice much faster than the readout spacing. At 10-50 cm/s a 5 mm slice
  is crossed in about 10-50 ms, against readout spacings of hundreds of ms.
- In 2D, label could cross other imaged slices' arteries, and be excited there, before it reaches
  its own voxel. That is **not modeled**: in 2D, label is depleted from its arrival in the voxel
  onward, as in P6. The sidecar records the assumption. In 3D it is modeled (Part C, "Depletion
  from slab entry").

**Crushing (P4 part C).**

- `VascularCrushingVENC` is one value per volume, `0` meaning uncrushed. A Look-Locker row is a
  readout, so each readout has its own VENC. QUASAR's alternation is crushed and uncrushed
  cycles, which is VENC constant within a cycle and alternating between cycles.
- The arterial term of readout `n` is multiplied by `survival(v_max, VENC_n)`. As in P4,
  capillary and tissue label are unaffected.
- A VENC that varies within a cycle is allowed. Each readout has its own bipolar gradients, so a
  per-readout value is physical. A test covers it.

Each P4 part keeps its own refusals: the slab entry time against `ATT` and `aATT`, the VENC
floor, and the `"global"` (P)CASL partial-bolus refusal.

## Inputs

There are no new keys. The P4 inputs are read under `LookLocker: true` as they are without it.
`look_locker_spec` loses four refusals: exchange, arterial, crushing, and bolus-position.

## Ground truth and outputs

- P6's `desc-deltamRead_gt` gains its components, each the readout's value before `sin(a)`, with
  every factor applied:
  - `desc-deltamReadIV_gt` and `desc-deltamReadEV_gt`, written with exchange on;
  - `desc-arterialRead_gt`, written with the arterial compartment on. It includes the survival
    and the bolus-position factor.
- `desc-deltamRead_gt` stays the sum of the label read: intravascular plus extravascular. The
  arterial term is separate, as P4 keeps it.
- `desc-lookLocker_gt.tsv` gains one column per component.
- The sidecar's `AslscanSimulation.LookLocker` records:
  - the parts in force;
  - the fresh-arterial assumption;
  - that in 2D, depletion before arrival in the voxel is not modeled.

---

# Part B: Look-Locker × multi-TE and Look-Locker × Hadamard

## Look-Locker × multi-TE

- Each readout is one excitation followed by `E` echoes. The echoes are transverse, so the
  longitudinal timeline, the depletion and the ground truth are the same for every echo. The
  truth is written once, as in P6 part C.
- The series builds its images once and calls `simulate_acquisition_echoes`. P6 already
  supports this outside compat. Compat stays refused for Look-Locker.
- **Timing.** P6's `check_excitation_timing` already runs on excitation groups. Under Look-Locker
  every readout of every cycle is an excitation group occurrence. The check runs at each of
  them:
  - every echo's block must end before the same slice group's next readout;
  - a slice's readout train must not overlap the next group's.

  The test is a protocol where the third echo of readout `n` overlaps readout `n+1`. That
  protocol must be refused, naming both.
- **Output.** `echo-N` series as in P6, one `aslcontext.tsv`, and one Look-Locker truth.

## Look-Locker × Hadamard

**What is modeled.**

- Each Look-Locker cycle has one encoded preparation, with one encoding row's weights `h_j`,
  followed by `M` readouts. Its blood at readout `n`
  is the 0/1-weighted sum of the sub-bolus reads,
  `sum_j h_j dM_read_sub(e_n; sub-bolus j)`. Each sub-bolus is depleted by the readouts after its
  arrival, as in Part A, and its arrival-window form is `dM_arr_sub`.
- Decoding is linear and runs **per readout index**. Output volume `(j, n)` is
  `(2/H) sum_r h_{r,j} S_{r,n}`, summed over the `H` encoding rows `r` of one encoding cycle,
  where `S_{r,n}` is readout `n` after encoding row `r`'s preparation.

**The schedule** (P6's three indices):

- An encoding cycle is `H` Look-Locker cycles, one per encoding row, in matrix-row order. Each
  Look-Locker cycle is one preparation in 2D (`NumberShots` in 3D, Part C) and gives `M` raw
  volumes, one per readout. An encoding cycle therefore has `H M` raw volumes.
- The output volumes are `(sub-bolus j, readout n)`: `(H - 1) M` per encoding cycle, readout-major
  (all sub-boli of readout 1, then readout 2), so each readout's block has P6's sub-bolus order.
- `TotalAcquiredPairs` is the number of encoding cycles, as in P6 (`bids.rs:597`), in 2D and 3D
  alike.

**Tissue.**

- The encoding rows of one matrix pass share their tissue timeline only in the steady state. The
  Look-Locker readouts make the cycle the repeating unit.
- P6's transient residual `L_j` is per readout here. A negative-case test with an included M0
  shows it nonzero, as P6 did.

**Inputs.**

- `[hadamard]` with `LookLocker: true`. As in P6, the aslcontext lists the **decoded** volumes:
  `deltam` rows only, `(H - 1) M` per encoding cycle in the output order above. `m0scan` rows go
  only between encoding cycles. `control` and `label` rows stay refused.
- Row `(j, n)`'s `PostLabelingDelay` must be sub-bolus `j`'s effective delay at readout `n`: `PLD_n`
  plus the durations of the sub-boli after `j`. P6's check (`protocol.rs:1156-1165`) is applied per
  readout. `PLD_n`, the last sub-bolus's delay, must strictly increase with `n`.
- `[look_locker] readouts_per_cycle` gives `M`. It must be one value for the whole series.
  Different counts would give readout indices different meanings across the encoding rows, so a
  per-cycle list is refused.
- The raw volumes, `H M` per encoding cycle, are written to `sourcedata/` as in P6. The
  raw-volume table gains a `readout` column.

**Ground truth.**

- The decoded truth per `(j, n)` is the sub-bolus read of the noiseless physics, with every
  factor applied.
- The test compares it with the decoded noiseless images, using P6's tolerance and its sign-flip
  and column-swap negative cases.

**All three together** is allowed. Each echo is decoded per readout index. The test checks one
echo against a single-echo run, with noise off.

---

# Part C: 3D gradient-echo trains

## What is modeled (`mrsim-acq`)

**A segmented 3D gradient-echo stack-of-EPI readout** ("3D EPI"):

- Per shot there is a train of small-flip excitations, `exc_spacing_ms` apart. Each excitation is
  followed by one EPI block per echo.
- The block is GRASE's: `ny / ky_segments` interleaved lines centred on its echo
  (`grase_block`). It reads one kz partition.
- The partitions are split across `kz_segments` shots in `kz_order`, as in `EchoTrain`. The
  canonical shot index is GRASE's `sy kz_segments + sz`.
- Transverse magnetization is assumed fully spoiled between excitations. This is the FLASH model
  P5's gradient echo uses: no stimulated or SSFP echo pathways. It is stated in the sidecar.
- The decay of echo `e` at line time `t`, measured from the echo centre, is P6 part C's 2D
  gradient-echo factor:

  ```
  exp(-(TE_e + t)(1/T2_c + 1/T2'_c)) * exp(i 2 pi fmap (TE_e + t))
  ```

  It is counted from each partition's own excitation, so the line weight has no EPG term.
- Uniform relaxation inputs make the decay a line weight. Otherwise it is applied per voxel inside
  the forward, as GRASE's `Mode` does.

**The longitudinal state is the caller's**, as a weight per excitation, supplied with each
volume's images:

```rust
/// One volume's input, built on demand by the caller.
pub struct GeVolume {
    /// Per compartment, one volume on the simulation grid (`x + snx*(y + sny*z)`).
    pub images: Vec<Vec<f32>>,
    /// w[(p * ky_segments + sy) * n_compartments + c]: the scalar multiplying compartment c's
    /// object for the excitation that reads partition p, in-plane segment sy.
    pub weights: Vec<f64>,
    /// P5's per-shot images, for per-shot motion; None when the shots share `images`.
    pub shot_images: Option<Vec<ShotSet>>,
}
```

- The compartment count is fixed for the whole call. Each compartment has one `T2`, `T2'` entry.
  A compartment that a volume does not use carries zero weights and may carry an empty image,
  which the forward skips.

- Each `(partition, in-plane segment)` is read by exactly one excitation per volume. So a weight
  per excitation is a weight per k-space plane, and that carries every longitudinal effect of the
  train:
  - the approach to the train's steady state;
  - the label's depletion;
  - the blood's kinetics within the train;
  - P5's per-shot physiology and gains.
- A weight that varies across kz blurs through-plane, as the real train does. This is the effect
  a 3D gradient-echo readout adds.
- `mrsim-acq` has no `T1` and no flip angle on this path. Keeping the physics in the caller
  mirrors P4: `mrsim-acq` does not know about labels.

**Entry point:**

```rust
pub struct ExcitationTrain {
    pub kz_segments: usize,
    pub kz_order: KzOrder,
    pub exc_spacing_ms: f64,
    pub echo_times_ms: Vec<f64>,   // from each excitation, strictly increasing
}
pub struct Ge3dReadout { pub ky_segments: usize, pub t_line_ms: f64, pub reverse_phase: bool }

pub fn simulate_acquisition_3d_ge(
    sim_dims, acq_dims, n_volumes, t2: &[T2Volume], fmap: &[f32], t_inhom: Option<&[T2Volume]>,
    acq: &Acquisition, train: &ExcitationTrain, readout: &Ge3dReadout,
    volume: &(dyn Fn(usize) -> GeVolume + Sync), phase: &PhaseModel, seed: u64,
) -> Vec<(Vec<f32>, Vec<f32>)>      // per echo: magnitude, phase; layout as simulate_acquisition_3d
```

- **Volumes are built on demand.** `volume(g)` is called once per volume, from the worker that
  simulates it, and its result is dropped when that volume is done. The compartment count is
  `t2.len()`, and every `GeVolume` must match it (asserted).
  - Memory then scales with the volumes in flight, not with the series. A 3D series with
    interpolated blood (below) needs this: at all volumes at once, a 128 × 128 × 32 simulation
    grid with 4 shots, 8 points, 3 families and 100 volumes would be about 19 GiB.
  - The noise and spike seeds are keyed on the volume index, as on the 3D spin-echo path, so the
    on-demand order changes no noise.
  - The existing `simulate_acquisition_3d` keeps its all-volumes interface.

- Noise and spikes are per partition with `SEED_SALT_3D`, as on the 3D spin-echo path. Echo `e`
  salts its receiver seed with `echo_salt(e)` (P6). The prepared-shot phase uses the unsalted
  excitation seed, so every echo sees the same shot phase.
- `check_ge3d_timing(train, block, t_exc_ms, tr_ms)` checks:
  - every echo block lies between its excitation and the next one;
  - consecutive echo blocks do not overlap;
  - the last shot's train ends before `TR`.
- Shot motion uses P5's `ShotSet`, keyed by the canonical shot.
- **Refused, as on the 3D spin-echo path:** partial Fourier, the eddy-current model, and
  multiband. 3D has no multiband today (`protocol.rs:1552`). Each refusal names the readout.

## The weights (`aslscan`)

**Excitation times.**

- Shot `s` of a volume starts its train at the preparation's readout time `t`, the BIDS
  `PostLabelingDelay` end. **P7 reads BIDS's "the excitation" as the first excitation of the
  train.** The sidecar records the time of the excitation that reads the kz centre, and the
  effective PLD it implies.
- Excitation `j` of the shot is at `t + j exc_spacing`. Every excitation is an event in an
  `LlCycle`: P6's timeline with one entry per excitation.

**Depletion from slab entry.**

- A 3D excitation covers the slab, and the slab contains the arteries that feed it. P1's geometry,
  where label in transit is outside the imaged region, does not hold. Label inside the slab is
  excited by every excitation from its **slab entry** onward, both before and after it reaches
  the voxel.
- **The slab entry time `d`** is the time from labeling to entry into the slab, under P4 part D's
  plug-flow rule:
  - a (P)CASL parcel `l` enters at `l + d`;
  - a PASL parcel arriving at `t'` enters at `t' - (ATT - d)`.

  For the arterial compartment the same rule holds with `aATT` in place of `ATT`.
- **Tissue label.** Let `Delta = ATT(r) - d`. A parcel arriving at `u` entered the slab at
  `u - Delta`. Excitation `m` depletes it when `u - Delta <= e_m < t`, that is when
  `u <= e_m + Delta` and `m < n`. The depleting set therefore changes only at the shifted times
  `e_m + Delta`, and the read keeps P6's form with every window shifted:

  ```
  dM_read(e_n) = sum_{w=0}^{n-1} D_{w,n} dM_arr(e_n; [e_w + Delta, e_{w+1} + Delta))
  ```

  - `D_{w,n}` is unchanged.
  - The first window starts at `0`, and every window is cut at `t`, as `delta_m_arrival` does.
  - `Delta = 0` is P6's read. It is tested to `1e-12`, and with `"arrival"` (below) the code
    calls P6's function.
  - The cost is P6's. The shift applies to every Part A component (intravascular, extravascular,
    each sub-bolus, each Hadamard sub-bolus), since `Delta` does not depend on the parcel.
- **Arterial label.** Let `Delta_a = aATT(r) - d`. The parcel read at `e_n` entered the slab at
  `e_n - Delta_a`. Its read is multiplied by the `cos(a_m)` of every excitation `m` with
  `e_n - Delta_a <= e_m < e_n`. `d = aATT` gives the fresh term of Part A.
- **The excitation list** is every excitation of the cycle (every sub-train under Look-Locker), or
  of the train for a single-readout series. Label from the previous preparation is taken to have
  left, as in P6.
- **Input.** `d` comes from exactly one source:
  - `[background_suppression] slab_entry_time`, when the bolus-position model with
    `pulse_region = "slab"` gives it; or
  - `[readout] slab_entry_time`, required under `"epi3d"` otherwise.

  Giving both is an error naming both (P4's rule for a quantity with two sources). The value is
  seconds, or `"arrival"`, which means entry at arrival: no depletion before the voxel, recorded
  as such.
- **Checks.** `d` must not exceed `ATT` or, with the arterial compartment, `aATT` at any voxel
  whose label it applies to. The error names the voxel, as in P4 part D.
- **The sidecar** records `d`, its source, and the per-label mean `Delta` and `Delta_a`.

**Tissue is separable.**

- Tissue `Mz` before excitation `j` is `M0(r) m_j(T1(r))`, from the affine timeline (Context).
  Its signal is `sin(a) M0(r) m_j(T1)`.
- **Grouping is by phantom voxel, before resampling.** `T1` is defined per phantom voxel, and
  resampling to the simulation grid mixes voxels.
  - Each tissue compartment is split into groups of equal `T1` at the phantom voxels.
  - Each group's `M0` is masked at the phantom voxels and then resampled, giving one compartment
    image. Its weights are `sin(a) m_j(T1)`. The split is exact.
  - In class `T2` mode a group is a (label, `T1`) pair. In voxel `T2` mode the single tissue
    compartment splits by `T1` alone.
  - A label-wise `T1` gives one group per label.
- The overlay `[readout] max_t1_groups` (default 16) refuses beyond its limit, naming the count.
  A smooth `T1` map refuses rather than being approximated: the simulator does not choose an
  approximation silently.

**Blood is not separable.**

- The read at excitation `j` is the Part A sum at `e_j`, with depletion from slab entry. It
  depends on each voxel's `ATT`, CBF and `aATT` through the whole train, so it is not
  `image × scalar`.
- It is expanded on piecewise-linear "hat" nodes in excitation index, within each shot:
  `B(r, j) ≈ sum_k phi_k(j) B(r, j_k)`.
  - Each node is one compartment image (`B` at the node's excitation time), sharing the blood's
    `T2`.
  - Its weights are `phi_k(j)`, zero outside the shot.
- Nodes start at the train's first and last excitations and the kz-centre excitation, and are
  bisected until the a-posteriori error is below tolerance.
  - The error is computed exactly, at every voxel and every excitation of the train:
    `|B(r, j) - interpolant| <= node_tolerance × max_{r, j} |B(r, j)|`. The reference is the
    largest value over the whole train, which is nonzero whenever any label is read. A train
    during which no label is read needs no nodes.
  - `[readout] node_tolerance` defaults to `1e-4`.
  - Nodes at every excitation make the expansion exact, so the bisection always terminates.
- **One node count per series.** `mrsim-acq` takes a fixed compartment count, so aslscan first
  runs the node selection for every (volume, shot). This pass costs kinetic evaluations only, no
  images. The series then uses the largest count `K` for every family, padding with zero-weight,
  empty-image compartments.
- The only refusal is the image memory. It is estimated before building, as the images of the
  volumes in flight (one volume times the worker threads). It is checked against the limit,
  which is now overlay `[images] max_memory_gib`, default 4.
  - The old key `[multi_te] max_image_memory_gib` stays an accepted alias. Giving both is an
    error.
  - When the old key is given, the sidecar records it under the old name exactly as today
    (`bids.rs:127`), so no P6 output byte moves.
- The sidecar records `K`, the node count per shot, and the achieved error.
- The arterial term and each exchange part are expanded the same way, each as its own compartment
  family.

**A single-readout 3D gradient-echo series** (no Look-Locker) is this with one train per
preparation.

- The preparation-to-preparation state uses P5's two rules, with the train's end state in place
  of P5's single excitation:
  - the fixed point when every preparation is the same;
  - propagation otherwise.
- `kz_segments = nz` (one excitation per shot, `nz > 1`) is an ordinary protocol: each shot's
  train is one excitation. Only `nz = 1` is refused, since a 3D readout of one partition is a 2D
  one; the error says to use `MRAcquisitionType 2D`.

## M0

- **The separate M0 scan** under `"epi3d"` is the same train, without labeling or suppression,
  repeated at its own repetition time. Its tissue weights come from that train's steady state,
  through the same timeline. This replaces the spin-echo M0 the 3D path uses today
  (`series/p6.rs:1325`), as P5 did for 2D: a spin-echo M0 would quantify against the wrong
  contrast.
- The M0 sidecar's `FlipAngle` is the train's excitation, and its `Contrast` is gradient echo.
- **Included `m0scan` rows** in a segmented series form their own group of `NumberShots` cycles,
  one per shot. Each is one train at the start of its repetition, as P6's `m0scan` cycle is in
  2D. They sit only between groups.

## 3D Look-Locker

- A Look-Locker cycle has `M` readouts. In 3D each readout is a **sub-train**: the excitations of
  one shot's partitions, starting at `PostLabelingDelay_n`.
- The volume of readout `n` is assembled over `S = NumberShots` consecutive cycles. Cycle `c`
  reads shot `c mod S` in every one of its sub-trains. This is P5's segmentation, where each shot
  has its own labeling, applied to every readout index.
- **The schedule.**
  - The preparations are the cycles.
  - Raw volume `(group, n)` has `S` preparations.
  - A group of `S` cycles produces `M` raw volumes, and the clock advances by `S × tr` per group,
    as P5's segmented clock does.
  - The aslcontext lists one row per raw volume: `M` rows per group, with `PostLabelingDelay`
    strictly increasing within the group.
  - The `S` cycles of a group must share their preparation. They read different shots of the same
    volumes, so cycles of different types are refused.
- **Depletion.** Every excitation of every sub-train is an event on the cycle's timeline. Label
  in the slab is depleted by every excitation after its slab entry ("Depletion from slab entry"),
  including those of earlier sub-trains.
  - With `n_exc` excitations per sub-train this is a strong effect. At 10 degrees and 16
    excitations, `cos^16 = 0.78`.
  - It is what the real sequence does. The sidecar records the cumulative depletion at each
    readout, for a reader comparing against a model that ignores it.
- **Timing.** A sub-train must end before the next one starts (`check_ge3d_timing` per
  sub-train). The last sub-train must end before the cycle's `tr`.
- **Motion.** Per-shot poses are taken at each sub-train's first excitation, as P5's per-shot
  poses are taken at the shot.

## Multi-TE in 3D

- Multi-TE 3D is `ExcitationTrain.echo_times_ms` with more than one echo, using P6's
  `--asl-json`-per-echo input and `echo-N` output.
- The weights are echo-independent, so `aslscan` computes them once.
- Echo blocks must fit between excitations. The check above enforces it, and it is what limits
  the echo count at a given `exc_spacing`.
- Multi-TE with 3D GRASE or spiral stays refused, naming this part and "Decisions deferred".

## Inputs (`aslscan`)

- `[signal] acq_contrast = "ge"` with `MRAcquisitionType 3D` is no longer refused.
- `[readout] type = "epi3d"` overrides `PulseSequenceType`, as `"grase"` and `"spiral"` do.
  - Existing keys reused: `ky_segments`, `kz_segments`, `kz_order`, `line_spacing`,
    `phase_encoding_direction`, `readout_samples`.
  - New keys: `excitation_spacing` (ms, required), `slab_entry_time` (s or `"arrival"`; see
    "Depletion from slab entry"), `max_t1_groups`, `node_tolerance`.
  - GRASE and spiral keys (`echo_spacing`, `refocusing_time`, `refocusing_flip_angle`,
    `interleaves`, `spiral_readout_time`, `dwell_time`, `radial_oversampling`) are refused under
    `"epi3d"`, each by name.
- The flip angle is P5's resolution (overlay over sidecar over default), applied to every
  excitation.
  - Under `"epi3d"` the sidecar's `FlipAngle` is the **excitation** angle. Under `"grase"` it is
    the refocusing angle (`protocol.rs:1815-1820`). `refocusing_flip_angle` is refused under
    `"epi3d"` (above), and the output sidecar records which meaning applied.
  - A 90-degree default is refused for `"epi3d"`, as for Look-Locker: it would saturate the
    train.
  - A per-volume `FlipAngle` array gives each volume's train its own flip. A variable flip within
    a train is deferred.
- `NumberShots` must equal `ky_segments × kz_segments`, as for GRASE.

## Ground truth and outputs

- The truth is the object at the kz-centre excitation:
  - the tissue `sin(a) Mz`;
  - the Part A blood components, before `sin(a)`.
- Under Look-Locker, the `desc-lookLocker_gt.tsv` table lists every excitation's time and the
  per-label mean weights, so the through-plane blurring is reconstructible.
- The sidecar records:
  - the train;
  - the excitation-time convention;
  - the spoiling assumption;
  - the slab entry time and its source;
  - the `FlipAngle` meaning;
  - the tissue groups;
  - the blood nodes and their achieved error.

---

# Division of labor

| Piece | Where |
|---|---|
| `ExcitationTrain`, `Ge3dReadout`, `GeVolume`, `simulate_acquisition_3d_ge` (volumes on demand), `check_ge3d_timing` | `mrsim-acq` `readout`, `kspace3d` |
| `delta_m_arrival_iv`, `delta_m_arrival_sub` (and its intravascular form), the depletion sum over components with the slab-entry shift `Delta`, the arterial slab-entry depletion | aslscan `kinetic` |
| Excitation events in `LlCycle`, the tissue groups (phantom grid), the 3D M0 train | aslscan `longitudinal` |
| Lifted refusals, `"epi3d"` inputs, `slab_entry_time`, the 3D Look-Locker schedule, the Look-Locker Hadamard context, per-readout VENC, `[images] max_memory_gib` | aslscan `protocol`, `schedule` |
| Hat-node expansion with its series-wide count and a-posteriori check, the per-volume builder, per-readout decoding | aslscan `series` (new `series/ge3d.rs`) |
| Component truths, the train sidecar | aslscan `bids` |
| P6 regress cases | aslscan `tools` |

# Testing and verification

**mrsim-acq**

- **An independent reference.** With one shot, no noise, uniform relaxation and a weight that is
  constant across partitions, `simulate_acquisition_3d_ge` must equal a direct forward: the
  per-partition 2D EPI forward of the z-DFT of the object, with the 2D gradient-echo decay. This
  holds to the `f32` bound under both feature sets.
- **Through-plane blur from varying weights.** With weights varying across `p`, the result must
  equal the inverse z-DFT of `w_p × (the z-DFT of the object)`, computed separately.
- **Echoes.** Echo 0 of a multi-echo call equals the one-echo call bit for bit. Each later echo,
  with noise off, equals a one-echo call at its `TE`. The noise of echoes 0 and 1 is uncorrelated
  (within `3/sqrt(n)`), and the shot phase is shared, as P6 tested in 2D.
- **Timing.** `check_ge3d_timing` must refuse each of:
  - a block crossing the next excitation;
  - overlapping echoes;
  - a train past `TR`.
- **Volumes on demand.** With noise and spikes on, the result equals the same call with
  `volume(g)` returning precomputed images, under one thread and under many, bit for bit. A
  counter shows `volume` is called once per volume. A `GeVolume` with the wrong compartment
  count panics.
- **Refusals.** Partial Fourier, the eddy model and multiband are refused.
- **Byte identity.** The golden record, the GRASE and spiral pins, and the P0 gate are
  unchanged.

**aslscan**

- **Byte identity.** The P6 regress cases pass against `p6-complete` before and after P7.
- **Arrival forms.**
  - `delta_m_arrival_iv` with `tau_ex → ∞` equals `delta_m_arrival`.
  - Summed over a window partition, it equals `delta_m_iv` at `1e-12`. This must also hold for
    PASL at a `tau_ex` where the naive form overflows; the test asserts that the naive form does
    overflow there.
  - `delta_m_arrival_sub` over a sub-bolus partition sums to `delta_m_arrival`.
  - Over both partitions it sums to `delta_m`.
- **Part A against an independent parcel reference.** A brute-force quadrature over labeling
  parcels, where each parcel:
  - arrives at `l + ATT`;
  - exchanges with probability `1 - exp(-s/tau_ex)`;
  - is inverted by the pulses whose entry cut it passed;
  - is multiplied by `cos(a)` at each readout after arrival.

  The test compares the readouts of a QUASAR-like protocol to `1e-6` relative, for PASL and
  PCASL, with every Part A component on. The arterial term is checked as fresh: no `cos(a)`, and
  survival per readout.
- **Reductions.**
  - With every Part A input off, the new Look-Locker path equals P6's bit for bit. The legacy
    dispatch is checked by the regress cases.
  - `"global"` suppression equals P6's read times the global factor.
- **Look-Locker × multi-TE.** With noise off, echo `e` equals a single-echo Look-Locker run at
  `TE_e`. The overlap protocol is refused.
- **Look-Locker × Hadamard.**
  - Decoded `(j, n)` of a noiseless series with no transient equals the independent sub-bolus
    read, within P6's tolerance.
  - Sign-flip and column-swap negatives.
  - The transient case gives a nonzero `L_{j,n}`.
  - Context checks: a `label` row is refused; a row whose delay is not `PLD_n` plus the later
    sub-boli is refused, naming `(j, n)`; `PLD_n` not increasing is refused. The
    bids-validator accepts the written context.
  - `TotalAcquiredPairs` equals the number of encoding cycles, in 2D and in 3D with
    `NumberShots > 1`.
- **Slab entry.**
  - The shifted read with `Delta = 0` equals P6's `delta_m_read` to `1e-12`.
  - For PCASL and PASL, with and without exchange and bolus-position, the shifted read equals the
    parcel reference with depletion from slab entry, to `1e-6` relative.
  - The arterial read with `d = aATT` equals Part A's fresh term exactly. With `d < aATT` it is
    multiplied by `cos(a)` of each excitation in `[e_n - Delta_a, e_n)`, which a hand count
    checks.
  - `d` above a voxel's `ATT` or `aATT` is refused, naming the voxel. Giving both sources is
    refused.
- **3D tissue groups.** Weights equal an independent per-voxel brute-force timeline at `1e-12`
  (the `brute_ll_with` reference of `longitudinal.rs`, not the production `LlCycle` path). The
  groups are formed on the phantom grid: a two-label phantom with two `T1` values in one label
  gives three groups, and their resampled images sum to the resampled `M0`. A smooth `T1` map is
  refused with its count.
- **3D blood nodes.**
  - The achieved error is at most `node_tolerance`, checked against an every-excitation
    evaluation.
  - Refining to nodes at every excitation gives the per-excitation reference images exactly.
  - A synthetic voxel with arrival inside the train forces more than three nodes.
  - A train with label arriving only after the kz-centre excitation still meets the tolerance:
    this is the case the old kz-centre reference got wrong.
  - Volumes with fewer nodes than `K` carry zero-weight padding, and the result equals the
    unpadded per-volume evaluation.
- **3D Look-Locker.** A noiseless, motionless series must equal an independent assembly. Each
  readout's volume is built partition by partition from the forward of the object at that
  partition's excitation time. Tissue comes from the brute-force timeline, and blood from the
  parcel reference with depletion from slab entry: neither shares code with the weight builder.
  Two further checks:
  - a cycle-type mismatch inside a shot group is refused;
  - the cumulative depletion recorded in the sidecar equals the product of `cos(a)` over the
    excitations.
- **3D M0.** The separate M0 equals the train's steady state from the brute-force timeline, and
  its sidecar has the train's `FlipAngle` and gradient-echo contrast. An included `m0scan` group
  in a segmented series is checked the same way.
- **Memory key.** A P6 multi-TE fixture with `[multi_te] max_image_memory_gib` writes a
  byte-identical sidecar. Giving both the old and new keys is refused.
- **`nz = 1`** under `"epi3d"` is refused. `kz_segments = nz` runs.

# Acceptance criteria

- Every test above passes under `cli` and `cli,kspace,par`. Clippy is at its P6 baseline.
- `regress_identity.sh p6-complete p6-complete` reports every case identical, including the new
  P6 cases. The TRXScan P0 gate reports 62 of 62 checksums identical.
- The bids-validator passes these outputs with no new errors:
  - a QUASAR series (PASL, Look-Locker, alternating VENC);
  - a Look-Locker Hadamard series;
  - a 3D Look-Locker series;
  - a multi-TE 3D series.
- Both READMEs describe the new readout and the combinations.

# Decisions deferred

- **Depletion before arrival in 2D.** Label crossing other imaged slices before reaching its own
  is excited there. Modeling it needs the vascular path through the slices, which P4 does not
  have. 3D models depletion from slab entry with a single `d` (Part C).
- **Multi-echo GRASE and spiral trains.** A multi-echo spin-echo train reads each partition at
  several echo times inside one CPMG train. It needs its own echo-ordering and EPG design.
- **3D gradient-echo spiral.**
- **Variable flip angles within a train**, and the balanced (SSFP) gradient echo, which needs
  transverse coherence.
- **Smooth `T1` maps under `"epi3d"`.** They are refused rather than approximated. A low-rank or
  node expansion in `T1` would admit them.
- **Compat** for every P7 feature: simasl models none of them.
- From P6, still deferred: velocity-selective ASL, vessel encoding, non-Sylvester and
  Walsh-ordered matrices, dummy cycles, and a multi-echo EPI's own k-space.

# Risks

- **The excitation-time convention for a 3D train.** BIDS says "the excitation". Reading it as the
  first excitation is a choice that someone comparing with a scanner's PLD definition may
  dispute. Recording the kz-centre time and the effective PLD makes the choice visible.
  A reviewer should check this convention against the vendor 3D EPI ASL literature.
- **Node expansion cost.** A short `ATT` arriving inside a long train needs many nodes. Building
  volumes on demand bounds memory by the volumes in flight, but compute grows with `K`: each node
  is one more image through the forward. The fallback is smaller shots (more `kz_segments`).
- **Fresh arterial blood in 2D** is QUASAR's assumption. It holds when blood crosses a slice much
  faster than the readout spacing (about 10-50 ms against hundreds of ms for 5 mm slices). It
  fails for slow flow, thick slices, or a short readout spacing. The sidecar states it.
- **One slab entry time.** Plug flow with one `d` for every parcel and voxel is P4's
  simplification. Real arteries enter the slab at different times. The sidecar records `d` and
  the mean `Delta` per label, so the size of the depletion is visible.
- **The PASL intravascular arrival form** is where the overflow trap lies. Its test asserts that
  the naive form overflows at the tested `tau_ex`, so the stable form is shown to be needed.
- **The 3D path's internals shared with GRASE.** Moving helpers can move bits. The pins are the
  check, and moved code keeps its arithmetic order.

# Review record

**Claude adversarial review, 2026-10-06.** Run while Codex was out of credits. It found 4 majors
and 9 minors, each verified against the source at `p6-complete`, and all were applied.

**Majors:**

- **M1. The Look-Locker × Hadamard context contradicted P6.** The draft had the context list raw
  `label` rows. P6 lists decoded `deltam` rows, with effective delays, and refuses `label`
  (`protocol.rs:1112-1116`, `:1156-1165`). "Cycle" was used for both the Look-Locker cycle and
  the encoding cycle, which made `TotalAcquiredPairs` wrong in 3D.
  - Fixed: the context lists the decoded `(j, n)` volumes, with per-readout effective delays.
  - The raw volumes go to `sourcedata/`, and `TotalAcquiredPairs` is the number of encoding
    cycles.
  - The two terms are now defined separately.
- **M2. Depletion before arrival in the voxel is large in 3D.** P6's model assumes label in
  transit is outside the imaged region. A 3D slab contains its feeding arteries and is excited
  every 30-50 ms, and bolus-position `"slab"` already declares in-slab label.
  - User decision: option (a), depletion from slab entry.
  - Tissue label uses P6's windows shifted by `Delta = ATT - d`. Arterial label is depleted by
    every excitation in `[e_n - (aATT - d), e_n)`.
  - `d` comes from bolus-position's `slab_entry_time` or a new `[readout] slab_entry_time`.
  - 2D keeps arrival in the voxel, with the slice-transit estimate stated.
- **M3. The blood interpolation needed every volume's images at once.** That is the 3D path's
  layout, and it came to about 19 GiB for an ordinary protocol.
  - Fixed: `simulate_acquisition_3d_ge` builds volumes on demand (`GeVolume` from a closure).
  - The noise is keyed on the volume index and unchanged.
  - The memory limit applies to the volumes in flight.
- **M4. The 3D gradient-echo separate M0 was undefined.** Today it would be the 3D path's
  spin-echo M0. Fixed: the M0 is the same train at its own repetition time, with a
  gradient-echo sidecar. Included `m0scan` rows form their own shot group.

**Minors:**

- **m1.** The blood tolerance's reference was the kz-centre value, which can be zero. It is now
  the largest value over the train.
- **m2.** The node count varied while the compartment count was fixed. There is now one count per
  series, with zero-weight padding.
- **m3.** `T1` groups are now formed on the phantom grid before resampling, for both `T2` modes.
- **m4.** The degenerate-case sentence conflated `kz_segments = nz` with `nz = 1`. Only `nz = 1`
  is refused.
- **m5.** `FlipAngle` means the excitation under `"epi3d"` and the refocusing angle under
  `"grase"`. This is now stated and recorded.
- **m6.** The new readout now refuses partial Fourier, the eddy model and multiband.
- **m7.** The memory-key rename could have changed P6 sidecar bytes. The old key is now recorded
  as today, and giving both keys is refused.
- **m8.** The 3D test references now use the brute-force timeline and the parcel reference, not
  the weight builder.
- **m9.** The 2D fresh-arterial and slice-transit assumption is now quantified in Part A and
  Risks.
