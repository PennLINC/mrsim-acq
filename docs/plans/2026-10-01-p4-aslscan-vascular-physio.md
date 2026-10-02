# P4: vascular compartments, bolus position, crushing and physiological noise — implementation plan

**Goal:** Implement the P4 addendum (`docs/specs/2026-10-01-p4-vascular-physio-design.md`) in
`aslscan`: the intravascular/extravascular split (part A), the arterial compartment (B),
vascular crushing (C), the bolus-position suppression model with partial-bolus inversion (D)
and physiological noise (E), each opt-in, with the sidecar blocks, ground truth and tests the
addendum specifies.

**Architecture:** The pure-std physics first, each against closed forms: `kinetic` grows the
sub-bolus and intravascular evaluations without changing `delta_m`'s bits; new modules
`crushing`, `physio` and `bolus`; `rng` gains a normal generator. Then the input surface
(`protocol`, `phantom`) with the activation and refusal rules; then `series`, which wires the
parts into the compartment images before the one acquisition call; then `bids` and the CLI;
then the end-to-end acceptance. **`mrsim-acq` is not modified** (addendum, "Division of
labor").

**Spec:** the P4 addendum, Codex-reviewed on 2026-10-01 before this plan; the main spec, P2 and
P3 where it is silent. P2 is complete at `aslscan` tag `p2-complete`.

## Global constraints

- Everything in the P1, P2 and P3 plans' global constraints still holds (pure-std default build,
  seconds upstream, one acquisition call per series, physics on the phantom grid, no `cargo fmt`,
  no large data committed, every approximation named in the sidecar, rejections over silent
  defaults).
- **P1, P2 and P3 outputs are byte-identical when no P4 input is given.** Concretely: `delta_m`
  keeps its bits (Task 1 proves it); a bolus with no cut strictly inside `(0, tau)` calls
  `delta_m` with its original arguments; no P4 code path runs when its part is off; no sidecar
  key or ground-truth file appears when its part is off. Task 8 checks the whole binary against
  the `p2-complete` build.
- **Nothing is ignored.** Every P4 input that belongs to a part that is off is a `protocol` (or,
  where only the phantom can tell, `series`) error naming the part.
- **Every part is refused under `[compat] asldro = true`**, naming simasl.
- Commit per task with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; tag
  `p4-complete` at the end, after the implementation review's fixes.

## File structure

| Path | Change |
|---|---|
| `src/kinetic.rs` | `delta_m_with_t1p` (the body of `delta_m` from `t1p` on; `delta_m` calls it), `delta_m_sub`, `delta_m_iv`, `arterial_dm` |
| `src/crushing.rs` | New, pure std: `si(x)`, `survival(v_max, venc)` |
| `src/rng.rs` | `SplitMix64::normal` (Box-Muller, both values of each pair, in order) |
| `src/physio.rs` | New, pure std: `PhaseProcess`, `OuDrift`, `PhysioParams`, the factors and window averages |
| `src/bolus.rs` | New, pure std: `Region`, entry offsets, `cuts`, `subbolus_factors`, `arterial_factor` |
| `src/lib.rs` | `pub mod crushing; pub mod physio; pub mod bolus;` |
| `src/phantom.rs` | Optional `abv` / `aatt` maps; `PhantomParams::{has_abv, has_aatt}` |
| `src/protocol.rs` | `[kinetic] exchange_time`, `[macrovascular]`, `[signal] t2_arterial`, `VascularCrushing*`, `[vascular_crushing]`, `[background_suppression] model / pulse_region / slab_entry_time`, `[physio]`; activation, refusals, compat; `Protocol::row_start` |
| `src/series.rs` | `3K` layout, parts A-E in the compartment images, `RowOverride::ExtravascularIntoBlood`, new ground truth, `PHYSIO_SEED_SALT` |
| `src/bids.rs` | `Exchange`, `Macrovascular`, `VascularCrushing`, `BackgroundSuppression` (bolus-position), `Physio` blocks; new ground-truth files and `desc-physio_gt.tsv` |
| `src/bin/aslscan.rs` | Report the P4 parts in the run summary |
| `tests/end_to_end.rs` | Linearity with every part on, the new negative controls, the matched `VENC` pair, physio noise independence |
| `tools/regress_identity.sh` | New: byte-identity of a build against a base revision (the P2 check, made a tool) |
| `tests/fixtures/protocols/p4_*` | Small P4 protocols for the end-to-end runs and the validator |

---

### Task 1: `kinetic` — sub-boluses, the intravascular part, the arterial term

`delta_m_with_t1p(k, f_ml_100g_min, dt, t1p, m0, t)`: the body of `delta_m` from the
delivery-state masks on, taking `t1p` (and computing `m0b`, `f`, `flow_over_lambda` exactly as
today, since the masks and both branches use them). `delta_m` computes `t1p` as today and calls
it. Nothing else in `delta_m` moves, so its arithmetic is the same sequence of operations.

`delta_m_sub(k, f, dt, t1t, m0, t, a, b)`: (P)CASL `delta_m(k with tau = b - a, f, dt, t1t, m0,
t - a)`, PASL `delta_m(k with tau = b - a, f, dt + a, t1t, m0, t)`; and when `a == 0 && b ==
k.tau` it calls `delta_m(k, f, dt, t1t, m0, t)` with the original arguments (the uncut path).

`delta_m_iv(k, f, dt, t1t, m0, t, tau_ex)` and the matching `delta_m_iv_sub`: the GKM with
`t1p'' = div0(1, div0(1, t1p) + 1/tau_ex)`, `t1p` computed with `delta_m`'s guards. (P)CASL
reuses `delta_m_with_t1p`. PASL does not: its branch forms `exp(kk t)` and `exp(-kk dt)`
separately (`kinetic.rs:79`), which overflow to `0 * (inf - inf)` once `kk = 1/T1b - 1/T1''` is
large and negative (at `tau_ex = 1e-6` the exponents are about `±8e5`). The intravascular PASL
evaluator is its own function with the exponents combined: arriving
`num = expm1(kk (t - dt))`, arrived `num = exp(kk (t - dt)) - exp(kk (t - dt - tau))` (each
exponent non-positive when `kk < 0`, and the positive-`kk` case keeping `delta_m`'s form, which
is stable there), with the masks and `T1b`, `lambda` guards checked before any exponential.
`delta_m`'s own PASL form is untouched.

`arterial_dm(k, abv, aatt, m0, t) -> (f64, Option<f64>)`: the arterial term with `g = 1` and the
parcel coordinate it holds (`Some(t - aatt)` for (P)CASL, `Some(t - aatt)` as the PASL `a`), or
`(0, None)` outside `aatt <= t < aatt + tau`; `M0b = m0 / lambda` with the GKM's `lambda` guard,
`T1b` tested `!= 0` for (P)CASL and `> 0` for PASL.

Tests:
- `delta_m` bit-identical to the pre-refactor body: a copy of today's function in the test
  module, compared with `to_bits` on a grid of `t` (0 to 6 s, 1 ms), both labeling types, the GM,
  WM and CSF constants, and the degenerate `lambda = 0`, `T1b = 0`, `T1b < 0` cases; and the
  `gkm.txt` fixture test unchanged.
- Sub-boluses over 200 random partitions of `[0, tau]` (2 to 6 cuts, plus cuts at the
  representable neighbours of `t - ATT`, `t - ATT - tau` and `0`), both labeling types, `t` over
  every delivery state: `|sum - delta_m| <= 1e-12 * peak`; where `delta_m` is an exact zero, the
  sum is an exact zero; the uncut call is `to_bits`-equal to `delta_m`.
- `delta_m_iv`: the same partition invariance; `0 <= iv <= dm`; `tau_ex = 1e9` gives `iv` within
  `1e-8` relative of `dm`; at `tau_ex = 1e-6` the share is `0.95 ± 0.01` at `t = ATT + 1e-7` and
  below `2e-6` at `t = ATT + 1`, for **both** labeling types and both delivery branches, every
  value finite (the NaN the shared PASL form would give is the regression this pins); the
  stable PASL form against `delta_m_with_t1p` where the latter is finite (moderate `tau_ex`,
  `1e-12` relative) and at partition edges; zero before arrival; the guards (`lambda = 0`,
  `T1b <= 0`) give the same zeros `delta_m` gives.
- `arterial_dm`: the two closed forms at three times in the window, zero at `aatt - eps` and at
  `aatt + tau`, nonzero at `aatt`; the parcel coordinate.

- [x] Write the functions and tests in `src/kinetic.rs`.
- [x] `cargo test` (default features) green in WSL.
- [x] Commit: `feat: kinetic — sub-boluses, the intravascular part and the arterial term`.

### Task 2: `crushing`, `rng::normal`, `physio`

`crushing` (pure std): `si(x)` for every finite `x` (odd, so `|x|`): the power series
`sum (-1)^n x^(2n+1) / ((2n+1)(2n+1)!)` for `|x| <= 2`, summed until a term is below `1e-17` of
the sum (no cancellation worth the name at that range); for `|x| > 2`,
`Si(x) = pi/2 + Im E1(i x)` with `E1(i x)` from its complex continued fraction evaluated by the
modified Lentz method to a relative change below `1e-16` (Numerical Recipes' `cisi`), which
converges faster as `x` grows, so there is no upper limit and no asymptotic switch.
`survival(v_max, venc)`: `1` for `venc == 0`, else `si(pi r) / (pi r)` with `r = v_max / venc`,
`1` at `r == 0`; `r` is finite because `protocol` bounds both (Task 4), and an infinite `r`
would be a bug, asserted.

Tests: `si` against values from `scipy.special.sici` (`Si(0.5) = 0.493107418043067`, `Si(1) = 0.946083070367183`,
`Si(pi) = 1.851937051982466`, `Si(4) = 1.758203138949053`, `Si(10) = 1.658347594218874`,
`Si(20) = 1.548241701043440`) to `1e-12`, plus `Si(1e3)` and `Si(1e6)` against `pi/2 - cos(x)/x`
to the asymptotic's own error, odd symmetry, both methods at `x = 2 ± 1e-9` within `1e-12` of
each other; `survival(·, 0) = 1`; `survival(v, v) = 0.5894898722360835` (`Si(pi)/pi`); monotone
decrease on `r` in `[0, 3]` sampled at 3001 points.

`rng`: a separate `Normal { rng: SplitMix64, cached: Option<f64> }` with `Normal::new(seed)`
and `next(&mut self) -> f64` by Box-Muller from two `unit()` draws (the first mapped to `(0, 1]`
to avoid `ln 0`), returning the cosine value and caching the sine value for the next call.
`SplitMix64` itself stays the public one-field tuple struct, so its existing constructor sites
(`series.rs:165`, the within-volume events) and its uniform stream are untouched. Tests: mean
and variance of 1e6 draws within `3e-3`; the cache order (the second call returns the cached
value without a draw); `SplitMix64`'s existing test unchanged; `cargo check --features io`
in this task so a feature-gated caller cannot break unseen.

`physio` (pure std):
- `PhaseProcess::new(f, cv, seed, horizon)`: initial phase `2 pi unit()`, periods
  `1/f + (cv/f) z`, `z` a normal resampled until `|z| <= 3`, generated until the horizon (the
  series duration plus `1/f`); `phase(t)` linear within each period; `mean_sin(t0, t1)`, the
  exact average of `sin phase` over `[t0, t1]` from the piecewise-linear phase (a sum of
  `(cos a - cos b) / (b - a)` terms weighted by segment length).
- `OuDrift::new(tau_d, seed, horizon)`: grid step `0.05` s, `x_0` standard normal, the exact
  discretization; `value(t)` linear interpolation; `mean(t0, t1)` exact for the interpolant.
- `PhysioParams` (amplitudes, frequencies, `cv`, `tau_d`) and
  `Physio::new(params, series_seed, horizon)`, which **alone** applies the salt:
  `pub const PHYSIO_SEED_SALT: u64 = 0x5048_5953_494F` lives here (pure std), the three streams
  are seeded from `(series_seed ^ PHYSIO_SEED_SALT) ^ 1`, `^ 2`, `^ 3` (cardiac, respiratory,
  drift), and callers pass the series seed unsalted; `tissue_factor(t)`,
  `label_factor_window(t0, t1)`, `label_factor_at(t)`.

Tests: period mean and cv over 10 000 periods within 2% of the requested; every period positive
at `cv = 0.3`; `mean_sin` against 100 000-point midpoint quadrature to `1e-10` (windows inside one
period, spanning several, starting on a boundary); OU lag correlation at `tau_d` within 0.05 of
`exp(-1)` over a long horizon, unit variance within 5%; `OuDrift::mean` exact for a linear
interpolant (hand-built two-point case); a fixed series seed reproduces a recorded reference
sequence of the first five periods and drift values (the normative stream test), and the same
reference is checked again through `series` in Task 5, so a correct unit and a wrongly salted
caller cannot coexist.

- [x] Write the three modules and their tests; add the modules to `lib.rs`.
- [x] `cargo test` (default features) green.
- [x] Commit: `feat: crushing, physio and a normal generator`.

### Task 3: `bolus`

Pure std, on top of `longitudinal::Suppression` (the pulse times and `epsilon`).

```rust
pub enum Region { Global, Slab(f64), Arrival }
/// The offset `delta` such that parcel `a` (sub-bolus coordinate in [0, tau]) is inside the
/// region from `a + delta` (PCASL: a = l; PASL: a = t' - ATT). `None`: inside from the start
/// (PASL global).
pub fn entry_offset(label_type, region, att: f64) -> Option<f64>
/// The cuts strictly inside (0, tau), ascending and deduplicated: `p_k - delta`.
pub fn cuts(pulses: &[f64], tau: f64, delta: Option<f64>) -> Vec<f64>
/// Per sub-bolus between consecutive cuts (with 0 and tau), the product of (1 - 2 epsilon) over
/// the pulses whose cut is at or beyond the sub-bolus's upper end (they act on all of it).
pub fn subbolus_factors(pulses: &[f64], epsilon: f64, tau: f64, delta: Option<f64>) -> Vec<(f64, f64, f64)>
/// The arterial compartment's parcel at `a`: the product over pulses with entry <= p.
pub fn arterial_factor(pulses: &[f64], epsilon: f64, a: f64, delta: Option<f64>) -> f64
```

`delta`: (P)CASL `Global` 0, `Slab(d)` `d`, `Arrival` `att`; PASL `Global` `None`, `Slab(d)`
`d`, `Arrival` `att` (PASL entry `t' - (ATT - d) = a + d`, and `t' = a + ATT` for arrival). For
the arterial compartment the caller passes `aatt`. A cut at or below 0 means the pulse acts on
no parcel; at or above `tau`, on every parcel; both give no cut.

Tests: `Global` with every pulse after `tau` gives one sub-bolus whose factor equals
`longitudinal::label_factor` (`1e-12`) for P3's asl002 pulses and three random sets; the asl002
GM `Arrival` case: sub-boluses `[0, 1.25]` factor `+1` and `[1.25, 1.8]` factor `-1` at
`epsilon = 1`, and the weighted sum through `kinetic::delta_m_sub` at the first slice's readout
over the unsuppressed `delta_m` equals `0.0821045` (`1e-6`), and an independent parcel sum over
the GKM kernel (`exp(l/T1')` weights) agrees (`1e-6`): the sum is split **at the cut** (1.25) and
each smooth interval integrated by a 20 000-point midpoint rule. A uniform midpoint rule across
the cut is off by `1.3e-5` (the discontinuity is not a cell edge), which must not be "fixed" by
loosening the tolerance; a pulse exactly at a cut
boundary; PASL `Global` never cuts; `Slab(d)` cuts at `p - d`; the arterial factor at parcels on
both sides of a cut; zero-efficiency pulses give factor 1 but still cut (part A's invariance test
in Task 6 uses this).

- [x] Write `src/bolus.rs` and tests; add to `lib.rs`.
- [x] `cargo test` green.
- [x] Commit: `feat: bolus — parcel entry, sub-bolus partition and factors`.

### Task 4: `phantom` and `protocol` — inputs, activation, refusals

`phantom`: optional `abv.nii.gz` (`Units: "fraction"`, every value finite in `[0, 1]`, zero on
background) and `aatt.nii.gz` (`Units: "s"`, finite and non-negative), on the phantom grid, into
`Phantom::{abv, aatt}: Option<Vec<f32>>`; `PhantomParams` gains `has_abv: bool, has_aatt: bool`
(and `phantom::load` returns `Some(params)` when either map exists even without
`phantom.json`, with the kinetic fields `None`, which `protocol` already treats as absent).

`protocol`, overlay additions (all `Option`, `deny_unknown_fields`):
- `[kinetic] exchange_time` (positive finite);
- `[signal] t2_arterial` (positive finite);
- `[macrovascular] arterial_blood_volume`, `arterial_transit_time`: `BTreeMap<String, f64>`
  (values in `[0, 1]` and non-negative finite);
- `[vascular_crushing] arterial_velocity: BTreeMap<String, f64>` (non-negative finite),
  `no_arterial_compartment: Option<bool>`;
- `[background_suppression] model` (`"global-bolus"` | `"bolus-position"`), `pulse_region`
  (`"global"` | `"slab"`), `slab_entry_time` (number or `"arrival"`, via an untagged enum);
- `[physio]` with the six amplitudes (finite), `cardiac_frequency`, `cardiac_cv`,
  `respiratory_frequency`, `respiratory_cv` (frequency positive, `0 <= cv <= 0.3`), `drift_time`.

Sidecar: `VascularCrushing: true` accepted (P1 refused it); `VascularCrushingVENC` a number or a
per-row array (length rule as the other per-volume arrays), each finite and `0` or `>= 0.1`
(cm/s).

`Protocol` gains `exchange: Option<f64>`, `macrovascular: Option<MacroSpec>`, `crushing:
Option<CrushSpec>` (per-row VENC, per-label velocity) or the `no_arterial_compartment` marker,
`suppression.model` and the region, `physio: Option<PhysioParams>`, and `row_start: Vec<f64>`
(cumulative `RepetitionTimePreparation`). `MacroSpec` resolves the two quantities
**independently**, each `Source::Map` or `Source::Table(BTreeMap)`, so all four valid
combinations (both maps, both tables, `abv` map with an `aatt` table, `aatt` map with an `abv`
table) are representable; plus `t2_arterial: (f64, Source)`, defaulting to the **resolved**
`T2Blood` value with source `"T2Blood"` (so an overlay `t2_blood` carries through).

Numeric checks, before any object is built: `exchange_time`, `t2_arterial`, `drift_time` finite
and positive; numeric `slab_entry_time` finite and non-negative; table values as above;
frequencies positive, `cv` in `[0, 0.3]`, amplitudes finite with `|cardiac| + |respiratory| < 1` per
factor, `arterial_velocity` in `[0, 1000]` cm/s, `exchange_time >= 1e-6` s, frequencies in
`[0.01, 10]` Hz (implementation-review additions; spec, parts A,
C and E).

Activation and refusal, in `parse`, in this order (the combination matrix; every row not listed
as valid is an error naming the part):
1. Part B is on iff `[macrovascular]` is present or `has_abv || has_aatt`. On, each of `aBV` and
   `aATT` must have exactly one source: neither is an error naming the missing quantity, both is
   an error naming the duplicate. An empty `[macrovascular]` with no maps is an error.
2. `t2_arterial` without part B: error.
3. Crushing. `VascularCrushing: true` requires `VascularCrushingVENC`. With part B it requires
   `[vascular_crushing] arterial_velocity` and refuses `no_arterial_compartment = true` as
   contradictory. Without part B it is an error naming `[macrovascular]` unless
   `no_arterial_compartment = true`, and then `arterial_velocity` is refused (it would act on
   nothing). `VascularCrushing` false or absent with `VascularCrushingVENC` or any
   `[vascular_crushing]` key: error.
4. Suppression model. `model`, `pulse_region`, `slab_entry_time` without suppression: error.
   `model = "global-bolus"` (or absent) with `pulse_region` or `slab_entry_time`: error (they
   would have no effect). `"bolus-position"` requires `pulse_region`; `slab_entry_time` only with
   `"slab"`, and `"slab"` requires it.
5. Partial-bolus rules: with `"bolus-position"`, P3's `p >= tau` check is replaced by: `"slab"`
   any `p >= 0`; `"global"` PASL any `p >= 0`; `"global"` (P)CASL `p >= tau`, the error naming the
   inflowing-blood history. With `"global-bolus"` P3's check is unchanged.
6. IR with suppression stays refused under both models.
7. `[physio]` with all six amplitudes zero: error.
8. Compat: any of parts A-E on, or `no_arterial_compartment`, is refused naming simasl (after the
   P2 checks).

Label-name validation (every foreground label has a value, no unknown names, and, only when a
name-keyed table is in use, no two foreground labels sharing a name, which `phantom::load`
allows today, `phantom.rs:146`, including a collision with a generated `label-N`) needs the
phantom's labels and lives in `series` (Task 5), before any simulation; the error names the
conflicting numeric labels.

Mechanical migrations in this task: the explicit `PhantomParams` literals (`protocol.rs:1585`,
`phantom.rs:345`) gain the two flags; no assertion changes.

Tests: each activation route, including all four `aBV`/`aATT` source combinations and the
missing/duplicate/empty cases; `t2_arterial` inheritance with and without an overlay `t2_blood`,
and its recorded source; each refusal in the matrix with the named part in the message; the
numeric checks; the overlay types (untagged `slab_entry_time`, the maps); the VENC array length
and value rules; compat refusals for every part; P1-P3 protocols parse to the same `Protocol`
fields as before (the existing protocol tests unchanged, plus a field-by-field comparison on the
asl002 and pasl fixtures with no P4 input).

- [x] Write the phantom and protocol changes and tests.
- [x] `cargo test --features io,test-hooks` green.
- [x] Commit: `feat: protocol and phantom — P4 inputs, activation and refusals`.

### Task 5: `series`

All per phantom voxel, per acquired slice, per row, in the existing loops:

- **Layout.** `K` tissue, `K` blood, then (part B) `K` arterial with `T2Volume::Uniform(t2_arterial)`
  and the label's T2' (`class`), or one arterial compartment with the tissue T2' map (`voxel`).
  `n_compartments` and the sidecar's `CompartmentOrder` follow.
- **Label names** for parts B and C validated against `ph.labels`; per-voxel `aBV`, `aATT` and
  `v_max` resolved from the maps or the per-label tables.
- **The delivered label** per voxel at `t = row.t + offset(z)`: with part D on, `bolus::cuts`
  and `subbolus_factors` for the voxel's `ATT` (`delta` per region), the total
  `sum f_j delta_m_sub_j` and, with part A, `sum f_j delta_m_iv_sub_j`; with D off, `delta_m` and
  `delta_m_iv` uncut. The `slab_entry_time <= ATT` (and `<= aATT`) check runs here, naming the
  voxel. Without part D the P3 global-bolus factor applies as today. **Multiplication order:** when
  the partition has a single sub-bolus (no cut), its factor is folded into `sign` exactly as P3
  does (`sign = blood_sign * factor`, then `blood_signal(sign * dm)` per voxel), so `"global"`
  with post-labeling pulses runs P3's arithmetic and gives P3's bits; only a partition with cuts
  sums `f_j * delta_m_sub_j` first.
- **Part A.** Blood compartment `K + c` gets `blood_signal(sign * iv)`; tissue compartment `c`
  gets `blood_signal(sign * (dm - iv))` added to its static tissue (the tissue path, cached per
  `TR` or per suppressed slice, is copied before the addition so the caches stay clean).
- **Part B.** Arterial compartment `2K + c` gets `blood_signal(sign * g * arterial_dm)`, `g` from
  `bolus::arterial_factor` (part D) and the physio label factor (part E), times the crushing
  survival of row `v` and label `c` (part C). In `voxel` mode the single arterial compartment is
  the sum over labels, and each label's survival is applied **before** the sum, per phantom
  voxel, so distinct per-label velocities stay distinct.
- **Part E.** `Physio::new(params, p.seed, horizon)` (unsalted: `physio` owns the salt, Task 2);
  the tissue factor multiplies each row's static tissue slab per slice (before part A's
  addition, and after the cache lookup, so two rows sharing a cache key but not a time get their
  own factors); the label factor (window average for (P)CASL, instant for PASL) multiplies the
  row's label in all three places.
- **The separate M0 scan** keeps its existing path: a plain steady-state readout with no label,
  so parts A-D do not apply, and part E does not modulate it either (it is not a row of the
  series' clock); the sidecar says so.
- **Compat-like relaxation** is unchanged: under compat no part is on.
- **`RowOverride::ExtravascularIntoBlood`**: the `label` row's extravascular part goes to blood
  compartment `K + c` instead of tissue `c`.
- **Ground truth**: `deltam_iv` (kinetics, part A), `deltam_suppressed` (kinetics and part D
  factors, both parts of the tissue label), `deltam_arterial` (`g = 1`), all moved under motion as
  `delta_m` is (P3) by the same poses, then block-averaged; `abv` (mean) and `aatt` (mean over
  `aBV > 0`) on the acquisition grid; the physio factors and phases per volume and slice for the
  TSV.
- `SeriesOutput` gains the per-row crushing survivals, the physio record, and the new ground
  truth as `Option`s (`None` when the part is off).

Tests (crop, homogeneous grid, `oversample = 1`, no GRAPPA, no spikes, and no noise unless the
test says otherwise). Image comparisons that regroup float32 compartment sums are bounded, not
equalities: `|a - b| <= 1e-6 * peak + 1e-6 * |b|`; the kinetic identities underneath are tested
tightly in f64 in Tasks 1 and 3.
- part A with `T2_blood = T2_tissue` (overlay `t2_blood` set to GM's T2, a GM-only check) matches
  the part-A-off run within that bound; with `tau_ex = 1e9` it matches in every voxel within the
  bound plus the `1e-9` relative the finite `tau_ex` leaves;
- part B: the arterial compartment image, read before the acquisition through a `test-hooks`
  accessor `simulate_compartments` added for it, equals `blood_signal(sign * arterial_dm)` per
  phantom voxel, box-averaged (`1e-6` relative), for (P)CASL and PASL, in and out of the window;
- part C: a matched pair of one-row runs (`VENC` 0 and 4 cm/s), with the readout inside every
  label's arterial window (the test sets `aATT` per label and the PLD so that
  `aATT <= t < aATT + tau`) and **distinct positive** velocities per label, differs by
  `sum_c (c_c - 1) A_c` as complex images, where `A_c` is the acquired image of a third run with
  only label `c`'s arterial compartment nonzero (same relaxation, factors, pose); the test asserts
  every `A_c` and the predicted difference are nonzero, so it cannot pass on empty windows; run
  in both `class` and `voxel` mode (the latter checks survival-before-merge);
- part D: `Global` with post-labeling pulses at `epsilon = 0.95` and `0.7` equals the
  `global-bolus` run **bit for bit** (same arithmetic, per the multiplication-order rule); a
  zero-efficiency `Slab` pulse inside the bolus (a cut, factor 1) matches the run without it
  within the bound in both the blood and the tissue compartment images (part A on), via the
  compartment accessor; suppression with part A on a multiband protocol, checking that slices
  sharing a readout time keep their own anatomy (P3's cache-key test, extended);
- part E: tissue scaled by the factor per slice, including two rows that share a cache key but not
  a start time; the reference stream of Task 2 reproduced through `simulate` (the salt has one
  owner); noise **on** (`noise_variance > 0`, asserting a nonzero residual variance) with E on and
  off at `ParallelReductionFactorInPlane = 1`: the residuals `noisy - clean` agree within the
  float32 rounding of the signal (`1e-5 * peak`), and the acquisition seed and configuration are
  identical (the bitwise noise claim is about the draws, which the reconstructed float32 images
  cannot show, and is not made about the images); the record reproduces the applied factors;
- motion with parts A, B and D on: each new moved truth equals the static truth moved by
  `mrsim_acq::motion::resample_by_pose` with the same pose and block-averaged (`1e-6`), and shot
  events do not enter it;
- `class` and `voxel` mode agree on a homogeneous grid with parts A and B on (the P1 cross-check,
  extended);
- the separate M0 scan is unchanged by every part (bit for bit against the part-off run);
- with every part off, every existing series test passes unchanged.

- [x] Implement and test.
- [x] `cargo test --features io,test-hooks` green.
- [x] Commit: `feat: series — the P4 compartments, factors and ground truth`.

### Task 6: `bids` and the CLI

The sidecar blocks of the addendum ("Outputs"), each only when its part is on;
`VascularCrushing` and `VascularCrushingVENC` echoed as given; `BackgroundSuppressionModel` and
`BackgroundSuppression.Model` set to `"bolus-position"` with `PulseRegion` and `SlabEntryTime`,
and `LabelFactor` absent under it; `CompartmentOrder` updated for part B. Ground-truth files:
`desc-deltamIntravascular_gt`, `desc-deltamSuppressed_gt`, `desc-deltamArterial_gt` (4D, units
as `desc-deltam_gt`, with JSON naming the stages per the addendum's table), `desc-aBV_gt`,
`desc-aATT_gt` (3D), `desc-physio_gt.tsv` with its columns. The CLI summary line names the P4
parts on.

Tests: an end-to-end write on the crop with every part on, checking each block's keys and values
against the `Protocol` and `SeriesOutput`, each ground-truth file's presence and shape; a P3
protocol's sidecar unchanged byte for byte with P4 code (the regression in Task 7 covers it at
scale); validator run on the every-part dataset.

- [x] Implement and test; validator clean (errors none).
- [x] Commit: `feat: bids — P4 sidecar blocks and ground truth`.

### Task 7: end to end, regression, acceptance

- `tools/regress_identity.sh <base-rev>`: the P2 check as a tool (a sibling worktree at the base,
  both release builds, the P1 fixtures on the full phantom, the crop, and the asl002 variants on
  the z-cropped phantom: suppression with its separate M0 scan, random motion, noise, IR;
  decompressed NIfTI and sidecar byte comparison; exit nonzero on a difference). Run against
  `p2-complete` (722ba19), **the pre-P4 head**: P2 was implemented after P3 (`p3-complete` is
  older), so this build contains all of P1-P3 and is the right baseline. (The plan review read
  `p2-complete` as pre-P3; it is not.) Additionally, a fresh suppression/IR/motion case on the
  crop, so the P3 paths are covered at a second geometry.
- `cargo test` in the default build of both crates, `mrsim-acq` included (`cargo test` there,
  69 tests at `742158d`), and `git -C mrsim-acq diff --stat <pre-P4> -- src` empty.
- P2's benchmarks: `compat_asldro.py all --phantom 3t` and `A`, `E` on 1.5 T, all passing with
  unchanged numbers.
- `tests/end_to_end.rs`: linearity with every part on (one-row series, so they share the physio
  factors and poses); the `ExtravascularIntoBlood` control fails it, and `BloodIntoTissue0` at
  `tau_ex = 10` s fails it; the matched `VENC` pair; physio leaving the noise unchanged.
- `tests/fixtures/protocols/p4_all`: a PCASL multi-PLD protocol with two slab suppression pulses
  (one during labeling), crushing alternating `0`/`4` cm/s, physio, exchange, and per-label
  arterial values chosen so the shortest PLD's readout falls inside every label's arterial window
  (the linearity and `VENC` tests assert a nonzero arterial image, so parts B and C are
  exercised); run on the crop, validated.
- Record every number in Measurements.

- [x] Commit: `test: P4 end to end and acceptance`.
- [ ] Codex adversarial review of the implementation (skipped for now: the workspace was out of
  credits; an internal adversarial review stood in and its findings are fixed); ordinary Codex
  review as the final pass; tag `p4-complete`.

## Acceptance criteria coverage

| Criterion | Where |
|---|---|
| 1. P1-P3 byte-identical with no P4 input; P2 A-E unchanged | Task 1 (`delta_m` bits), Task 7 (regression, benchmarks) |
| 2. Sub-bolus identity, intravascular part, arterial term, crushing survival | Tasks 1, 2 |
| 3. `"global"` equals P3; `"arrival"` gives `0.0821045` | Task 3, Task 5 |
| 4. Matched `VENC` pair isolates the arterial signal per label | Task 5, Task 7 |
| 5. Physio statistics, reference sequence, noise unchanged, ground truth | Task 2, Task 5, Task 7 |
| 6. Compat and inactive inputs refused; every-part dataset validates | Task 4, Task 6 |
| 7. `mrsim-acq` unchanged; default builds green | every task |

## Codex review of this plan (2026-10-01)

Thirteen findings (2 blockers, 9 major, 2 minor); twelve verified and applied, one rejected in
part. Applied: the PASL intravascular evaluator overflowed to NaN at short `tau_ex` and now has a
stable combined-exponent form (the spec says so too); the physio salt was applied twice and now
has one owner, checked through `series`; the 0.0821045 quadrature must split at the cut (a uniform
midpoint rule misses by `1.3e-5`); the noise-identity test was vacuous under "no noise" and
claimed an unobservable bitwise image equality; `aBV` and `aATT` sources resolve independently,
with the arterial T2 inheriting the resolved blood T2; a full combination matrix and numeric
checks for the inputs; tests for the moved truths, class/voxel, voxel-mode survival, cache keys
versus times, multiband with part A, and the separate M0 scan; the `VENC` and all-parts tests
must have arterial signal in their windows; image comparisons that regroup float32 sums are
bounded, and `"global"` keeps P3's multiplication order so its equality is bitwise; the normal
generator is a separate type so `SplitMix64`'s constructor sites do not move; duplicate label
names are refused when name-keyed tables are used; `Si` uses a continued fraction for every
`x > 2`. Rejected in part: the review read the regression baseline `p2-complete` as pre-P3; it is
the pre-P4 head and contains P3 (`p3-complete` is its ancestor), so it stays, with a second P3
geometry added and the `mrsim-acq` test command made explicit.

## Measurements

Measured 2026-10-01 at `aslscan` `9247127`. Unit and integration tests were run natively on
Windows (`CARGO_TARGET_DIR=target-win`) while WSL was unresponsive, and in WSL for the release
builds, the regression and the benchmarks; both builds pass the same suites (128 unit, 16
end-to-end tests, clippy clean).

**Task 1, `kinetic`.** `delta_m` is bit-identical to its pre-refactor body on 216 000+ points
(three label types, four tissue settings, the `lambda` and `T1b` guards). Sub-boluses over random
partitions, with cuts at the representable neighbours of the delivery edges, match the whole bolus
to `1.9e-15` of peak (criterion `1e-12`). The intravascular part: partition-invariant and bounded
by `delta_m`; the share is `0.95` at `ATT + 1e-7` s and below `2e-6` at `ATT + 1` s for
`tau_ex = 1e-6`, finite in both PASL branches where the shared GKM form gives NaN.

**Task 2.** `Si` matches `scipy.special.sici` to `1e-12` at six points and the asymptotic at
`1e6`; `c(1) = 0.5894898722360835`; survival strictly decreasing on `r` in `[0, 3]`. Period mean
and CV within 2% over 10 000 periods (the CV is 0.987 of the requested, the truncation at
`±3 sd`); `mean_sin` exact against Simpson to `1e-10`; the OU lag-`tau_d` correlation within 0.05
of `e^-1`. The reference stream is pinned and reproduced through `simulate`.

**Task 3, `bolus`.** `"global"` after labeling gives P3's `label_factor` bit for bit at
`epsilon` 1, 0.95 and 0.7. `asl002`'s GM with slab entry at arrival: `0.0821045` by the sub-bolus
sum, by the closed form and by a parcel quadrature split at the cut.

**Task 5, `series`** (crop, homogeneous grid). A matched `VENC` pair (0 and 4 cm/s, velocities
10 / 6 / 3 cm/s) differs by `sum_c (c_c - 1) A_c` to `3.0e-7` of peak in both `class` and
`voxel` mode. `"global"` bolus-position equals the global-bolus run bit for bit. The split
conserves each label's total; the arterial image is the closed form per voxel for PCASL and PASL,
in and out of the window. The separate M0 scan is bit-identical with every part on.

**Task 7.** Linearity with every part on: `0.18` of the tolerance. Negative controls: the
extravascular part routed to the blood breaks it by `54x`; the intravascular part routed to
tissue 0 (`tau_ex = 10` s) and a control/label swap on the P4 path by more than `1e2`.
`tools/regress_identity.sh p2-complete`: all 14 cases byte-identical (`pasl_cutoff` on the full
phantom; the crop PCASL fixture in class and voxel mode; asl002 with suppression and its separate
M0, motion, noise and IR on the z-cropped phantom; the crop at 2 x 2 x 3 mm with suppression, IR,
motion, and PASL with suppression and with motion; asl004 in class and voxel mode on a 97-slice
crop, its readout shortened to 0.025 s because as published it reads before the excitation in
this model, the main spec's deferred line timing). P2's benchmarks rerun with the P4 build: every
number as recorded in the P2 plan (A `6.2e-8`, E `9.9e-8` on 3 T; B, C, D, D-grid unchanged).
`tests/fixtures/protocols/p4_all` (every part on, a slab pulse during labeling) on the crop
validates with no errors (the three recommended-key warnings of every aslscan dataset).

**Reviews.** The Codex adversarial review of the implementation died at its start ("workspace is
out of credits") and was skipped at the user's direction. An internal adversarial review stood in;
it confirmed the PASL stable algebra, the sub-bolus coordinates, compartment placement, slab
indexing, the activation matrix and byte identity, and found one real defect (the drift window
mean merged most windows into one trapezoid through grid-index rounding) and minor ones (the swap
control on the P4 path, unbounded physiological amplitudes, three tests weaker than their names,
regression coverage); all fixed in `9247127`, the VENC refusal kept and recorded in the spec.
