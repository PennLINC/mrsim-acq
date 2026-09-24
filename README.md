# mrsim-acq

mrsim-acq is a Rust library that simulates how an MRI scanner acquires an image. You give it
images of the signal inside an object. It returns the images a scanner would reconstruct from
that object, with the imperfections a real echo-planar (EPI) acquisition introduces.

mrsim-acq is not a program you run directly. It is the shared acquisition step of two
simulators:

- [aslscan](https://github.com/PennLINC/aslscan) simulates arterial spin labeling (ASL) data.
- [TRXScan](https://github.com/PennLINC/TRXScan) simulates diffusion MRI data.

Each simulator computes the MR signal for its own kind of scan and then calls mrsim-acq to turn
that signal into scanner-like images. If you want to simulate ASL or diffusion data, use one of
those tools. This library is useful if you are writing a new simulator that needs a realistic
acquisition stage.

The code was extracted from TRXScan without changing its numerical results. It is at an early
stage (version 0.0.0) and its interface may change.

## What it models

Simulation is done slice by slice, by computing the k-space data the scanner would record and
reconstructing an image from it. The following effects are included:

- **EPI distortion.** Given a B0 fieldmap, the image is displaced along the phase-encoding
  direction, as in real EPI data.
- **Signal decay during the readout.** Each line of k-space is weighted by T2 and T2' decay at
  the time it is acquired. Several tissue compartments, each with its own T2 and T2', can be
  combined in one voxel.
- **Gibbs ringing.** The object is simulated on a finer grid than the scan and truncated in
  k-space, so ringing appears as it does in real data.
- **Partial Fourier** acquisition and k-space **windowing** (Hann, Tukey, or Fermi filters).
- **Multiple receive coils**, combined into one image, and **GRAPPA** parallel imaging.
- **Artifacts:** Nyquist ghosting, eddy currents, and k-space spikes.
- **Noise**, added in k-space or, optionally, in image space with a spatially varying level.
- **Head motion** (rigid, per volume) and **multiband** slice timing, including motion-related
  signal dropout.
- **Output** as BIDS `part-mag` and `part-phase` NIfTI files with JSON sidecars.

The readout model is 2D single-shot spin-echo EPI. Gradient-echo and 3D readouts are not modeled.

## Installation

mrsim-acq is used as a dependency of another Rust project. It is not published on crates.io.

1. Install the Rust toolchain with [rustup](https://rustup.rs/).
2. Clone this repository next to the project that uses it:

   ```bash
   git clone https://github.com/PennLINC/mrsim-acq.git
   ```

3. Add it to that project's `Cargo.toml`:

   ```toml
   [dependencies]
   mrsim-acq = { path = "../mrsim-acq", features = ["io", "kspace", "par"] }
   ```

### Optional features

By default, the library has no external dependencies. The following features add functions or
speed:

| Feature | Adds |
|---|---|
| `io` | Reading NIfTI images and writing NIfTI output with BIDS sidecars. |
| `kspace` | FFT-based transforms. These give the same results as the default code, much faster. Recommended. |
| `par` | Multi-threading across volumes. |
| `config` | Placeholder for configuration-file support; not yet functional. |

## Usage

The main entry point is `kspace::simulate_acquisition_oversampled`. It simulates a whole series
of volumes in one call. Its inputs are:

- **Grid sizes.** The scan matrix (for example 64 x 64 x 30), and a simulation grid that is an
  integer multiple of the scan matrix in-plane (for example 128 x 128 x 30). Slices are not
  oversampled.
- **Compartment images.** One or more signal images on the simulation grid, one value per voxel
  per volume. Each compartment has its own T2 and T2', given as either a single value or a map.
- **A fieldmap** in Hz on the simulation grid. Pass zeros if distortion is not wanted.
- **Acquisition settings** in a `kspace::Acquisition` value: echo time, time per phase-encoding
  line, noise level, number of coils, partial Fourier fraction, and so on. Times are in
  milliseconds. `Acquisition::default()` provides a starting point.
- **A random seed.**

It returns the magnitude and phase images on the scan grid.

```rust
use mrsim_acq::kspace::Acquisition;

let acq = Acquisition {
    t_echo: 12.0,          // echo time, ms
    t_line: 0.3,           // time per phase-encoding line, ms
    noise_variance: 1e-4,  // 0 disables noise
    n_coils: 8,
    seed: 42,
    ..Acquisition::default()
};
```

**Simulate all volumes of a series in one call.** The random noise for each volume is determined
by the seed and the volume's position within the call. Simulating volumes one call at a time
would give every volume identical noise. If a second call is unavoidable, such as for a separate
M0 or reference scan, pass it a different seed.

For a complete example, see `src/series.rs` in aslscan. It builds the compartment images, calls
the entry point, and writes the result. To browse the API documentation locally, run:

```bash
cargo doc --open --features io,kspace,par
```

## Testing

```bash
cargo test --features io,kspace,par
```

Running `cargo test` without features tests the dependency-free code. The tests compare the
k-space model against analytic Fourier results and check that the FFT and direct-sum versions of
each transform agree.

## Documentation

The design of mrsim-acq and aslscan, including how the library was separated from TRXScan, is
described in [docs/specs/](docs/specs/). Implementation plans are in [docs/plans/](docs/plans/).

## License

MIT or Apache-2.0, at your option.
