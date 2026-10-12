# P8: partial Fourier and acceleration along kz for the stack of spirals

Design spec addendum, 2026-10-12. Extends `2026-10-01-p5-3d-readouts-ge-design.md` (part C, the
stack of spirals). Where this document is silent, P5 stands. Written with the implementation, at
the user's request to "implement kz partial Fourier and R=2" (2026-10-12); the design choices below
were made by Claude during that work and have not been reviewed by the user or by Codex.

## Context

The target is the accelerated 3D RARE stack-of-spirals pCASL of Vidorreta et al. (2017, PLoS ONE
12(8):e0183762), as the Siemens Prisma protocol `SPIRAL_V20_HCP` prescribes it:

- 64 x 64 at 3.75 mm, 34 slices with 5.9% slice oversampling (36 partitions), centric;
- two spiral interleaves, each its own shot (two-shot mode);
- slice partial Fourier 6/8 (5/8 single-shot), reconstructed with POCS;
- acceleration 2 along kz: from `kz = 0` outward in steps of two;
- a "GRAPPA-like" 4 x 1 kernel, calibrated on fully sampled readouts played in two preparation
  repetitions, applied after gridding.

Before P8, `aslscan` refused `ParallelReductionFactorOutOfPlane != 1` and a slice
`PartialFourierDirection` for every 3D readout ("deferred"), and the spiral train read every
partition. On this protocol that gave a 36-echo train where the scanner's has about 14.

## What is modeled (`mrsim-acq`)

**Sampling.** `readout::KzSampling { partial_fourier, accel }` is a field of `Readout3d::Spiral`.
With `kz = p - nz/2`:

- partial Fourier keeps the `round(partial_fourier nz)` highest partitions (the centre, the
  positive side, part of the negative side); it must keep the centre (`> 0.5`);
- acceleration keeps, of those, the partitions with `kz` a multiple of `accel`.

The train reads the kept partitions in `kz_order` (centric or linear) restricted to them. They must
divide into `kz_segments`, and `EchoTrain::etl` counts them. `KzSampling::FULL` is the P5 train,
bit for bit (the golden record is unchanged).

**Coils.** Acceleration along kz needs sensitivities that differ along z, which the single ring of
`kspace::coil_sensitivity` does not have. The coils form `coil_rows` rings along z, each of
`n_coils / coil_rows` elements. A coil's sensitivity is its element's in-plane profile times a z
factor: a Gaussian about its ring's position (rings evenly spaced over the slab, width
`COIL_ROW_SIGMA = 0.6` ring spacings) on the in-plane profile's 0.15 floor. One ring gives the
factor 1 exactly. The combine is P5's (Roemer, known sensitivities), with the z factor included.

**Kernel.** Per missing partition of the partial-Fourier band and per coil: a combination of every
coil's samples at the (up to) four nearest acquired partitions, two on each side, sample by sample.
The kernel has no in-plane extent, so it commutes with any linear in-plane reconstruction and is
applied to the spiral samples directly, before the inverse z-DFT. There is one weight set per offset
within the acceleration period and per set of sources present (the band's edges have fewer). Each is
fitted by least squares with a relative Tikhonov term (`KZ_KERNEL_LAMBDA = 1e-8` of the normal
matrix's mean diagonal).

**Calibration.** `kspace3d::KzOptions` carries the calibration: compartment images (one volume),
read by the same train without the acceleration (its own, longer echo train and echo amplitudes),
with the series' noise level on its own seeded stream. The kernel is fitted once per call and
applied to every volume, so the reconstruction stays linear in each volume's samples.

**Partial Fourier reconstruction.** Zero filling by default: the partitions below the band stay
zero. This is
what the 2D path does in-plane, and it keeps the reconstruction linear.

POCS is the option (`KzOptions::partial_fourier_recon`, added 2026-10-12 at the user's request). Per
coil: each partition of the band through the in-plane least squares (it commutes with the z-DFT),
then along z per voxel, the phase from the Hann-windowed symmetric centre of the band, and
`POCS_ITERATIONS = 10` rounds of imposing that phase at the current magnitude and restoring the
acquired partitions; then the Roemer combine. It is homogeneous under a positive scale and not
additive. Because it acts per coil, the rings of an element cannot be combined before the least
squares, so its cost grows with the coil count.

**Noise.** Per acquired sample, as before; unread partitions get none. The image noise under
partial Fourier and the kernel's noise amplification follow from that and are not asserted.

## What is modeled (`aslscan`)

- `[readout] kz_partial_fourier` and `kz_acceleration`, over the sidecar's `PartialFourier` (only
  with a slice `PartialFourierDirection`) and `ParallelReductionFactorOutOfPlane`. Spiral only.
- `[acquisition] coil_rows` (default 1), which must divide `n_coils` and be at least the
  acceleration. Spiral only.
- The calibration reads the clean compartment images (before any motion) of the series' first control
  or label volume; a series without one uses its first `m0scan`, then its first `deltam`. It is
  seeded by the series seed. A separate M0 is reconstructed with the same kernel. A calibration volume
  with no signal is an error.
- The calibration train must also fit in that volume's repetition time.
- The sidecar records the sampling, the coil model, the kernel and the calibration under
  `AslscanSimulation.Readout.KzSampling`, and states `ParallelReductionFactorOutOfPlane`,
  `PartialFourier` and `PartialFourierDirection` as simulated.

## Decisions

1. **Zero-filled partial Fourier by default, POCS as an option.** POCS is nonlinear in the data and breaks the
   linearity identity, so `aslscan` refuses `deltam` rows with it (a `deltam` volume is simulated as the
   image of the difference object). Zero filling changes the through-plane point spread function
   relative to the paper.
2. **A separate calibration acquisition, not autocalibration.** This is the paper's scheme, and it
   keeps the kernel independent of the volume it is applied to.
3. **The calibration object is a series volume.** The paper does not say what its preparation
   repetitions hold; the first labeled or control volume has the series' contrast.
4. **Spiral only.** GRASE and the 3D EPI train still refuse both.
5. **Hadamard is refused** with any kz undersampling or more than one coil ring (the P6 series path
   was not extended).
6. **Coil z profiles are a model choice.** The width was not tuned to a real array.

## Measurements

`kspace3d` tests, relaxation off, no noise, an object rough along z, 16 x 16 x 12, against the same
band without the acceleration on the same coils (max error over voxels, of the image peak). This is
the kernel's error alone: under partial Fourier the reference is zero-filled as well.

| Sampling | Coil rings x elements | Error |
|---|---|---|
| R = 2 | 4 x 1 | 0.9% |
| R = 2 | 4 x 2 | 0.8% |
| R = 2, partial Fourier 6/8 | 4 x 1 | 3.2% |
| R = 3 | 6 x 1 | 2.0% |

On a smooth object (only `kz = 0, +-1`) the error was below 0.1%, which is why the test object is
rough. Lower regularization reduced the full-band error further (0.3% at 1e-12) but not the
partial-Fourier case, whose error sits at the band's edge.

The acceptance protocol (`aslscan/tests/fixtures/protocols/p8_spiral_kz`), 29 volumes on the 3 T
phantom cropped to 135 mm: 13 of 36 partitions read, 13 echoes of 10 ms.

## Decisions deferred

- Homodyne reconstruction along kz; `deltam` rows under POCS (they would need two reconstructions and
  a subtraction).
- 2D (CAIPI) and in-plane acceleration of a spiral.
- kz undersampling for GRASE, the 3D EPI train and Hadamard series.
- The echo spacing of this sequence: the paper's readout durations imply about 16.5 ms at an
  effective TE of 10.3 ms, which the P5 spiral timing (`EchoTime = CentreEcho x ESP`, each spiral
  starting at its echo) cannot express.
- Gradient and slew limits on the spiral (P5 bounds only the k-space speed per dwell time).

## Review record

Codex adversarial review of the implementation, 2026-10-12: one major and five minors, each checked
against the source.

| Finding | Outcome |
|---|---|
| Major: the calibration images were copied after per-volume motion, so a moved first volume changed the kernel | Fixed: captured before the motion block (`aslscan/src/series.rs`) |
| Minor: `InputValuesReplaced.PartialFourier` recorded the in-plane value, not the input | Fixed: the in-plane entry is dropped before the kz one (`aslscan/src/bids.rs`) |
| Minor: a sidecar `PartialFourierDirection: "k"` without `PartialFourier` failed even with an overlay fraction | Fixed: the overlay's fraction stands in |
| Minor: a non-integer sidecar `ParallelReductionFactorOutOfPlane` fails even with an overlay value | Kept: malformed metadata is reported |
| Minor: a non-string `PartialFourierDirection` was read as absent | Fixed: refused for 3D readouts |
| Minor: `band_start` refused bands that hold the centre (`nz = 4`, 0.51) | Fixed: a fraction above one half always holds it; the check is a debug assertion |
| Minor: the 3.2% partial-Fourier figure was described as against the fully sampled volume | Fixed: the wording says what the reference is |

After that review:

- coils in different rings share their in-plane forward, and an element's rings are combined before
  the least squares (one reconstruction per in-plane element). The acceptance protocol went from
  11.4 min to 3.0 min with the same output;
- the `KzSampling` sidecar block is written only when the sampling or the coil rings differ from
  P5's, so a fully sampled spiral's sidecar is unchanged.

Codex final review, 2026-10-12: two findings, both applied.

| Finding | Outcome |
|---|---|
| At a very small `signal_scale` the kernel's normal equations fell under the solver's pivot floor and the missing partitions stayed empty | Fixed: the equations are normalized to a unit mean diagonal before the solve; a test runs at `signal_scale = 1e-12` |
| A series whose first non-M0 volume has no signal (a `deltam` before arrival) panicked in the fit | Fixed: the calibration volume is the first control or label, else the first `m0scan`, else the first `deltam`; an empty one is an `aslscan` error naming the row |

## POCS measurements

`kspace3d` test, a real positive object rough along z, 16 x 16 x 12, relaxation off, no noise, 6/8
along kz, against full sampling (max error over voxels, of the image peak):

| Sampling | Zero filling | POCS |
|---|---|---|
| 6/8, one coil | 19.7% | 2.2% |
| 6/8 and R = 2, four coil rings | 18.8% | 4.1% |

Additivity residual (reconstruction of a sum against the sum of reconstructions): 1e-8 of peak for
zero filling (the `f32` rounding of the summed images), 2.3e-3 for POCS.

Codex review of the POCS option, 2026-10-12: one finding, applied. A fraction that rounds to every
partition (0.99 of 12) ran the linear path while the sidecar recorded POCS; `aslscan` now refuses
POCS when no partition is left out.
