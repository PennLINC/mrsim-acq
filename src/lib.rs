//! # mrsim-acq
//!
//! The MRI acquisition stage shared by TRXScan (diffusion) and aslscan (ASL): a per-slice
//! k-space forward model with EPI distortion, relaxation, eddy currents, ghosting, partial
//! Fourier, ringing, spikes, multi-coil combination, GRAPPA and noise, plus the motion model
//! that cuts across it and the complex NIfTI writer at the end of it.
//!
//! Extracted from TRXScan so the same acquisition physics serves both signal models. The
//! signal stage — what fills the compartment images — belongs to the consumer.
//!
//! ## Build shape
//! Every external dependency is optional; the default build is pure std, so `cargo test`
//! exercises the core offline. Features: `io`, `kspace`, `config`, `par`.

/// A direction or point in 3D. The pure-std core uses a plain `[f64; 3]`; the feature-gated
/// I/O and k-space paths convert to `nalgebra::Vector3<f64>` where they need heavier algebra.
pub type Vec3 = [f64; 3];

/// std-only 3×3 / vector helpers, keeping the pure-math core dependency-free and testable.
pub mod mat;
/// std-only voxel-axis reorientation to the FSL/dcm2niix (radiological LAS) convention.
pub mod orient;
/// The acquisition voxel grid: dimensions plus a voxel→world affine.
pub mod grid;
/// Analytic Fourier references used as test oracles.
pub mod analytic;

/// Object phase model.
pub mod phase;
/// EPI readout trajectory and line timing.
pub mod readout;
/// CPMG echo amplitudes by the extended phase graph (P5, the 3D echo trains).
pub mod epg;
/// Noise models. `add_complex_gaussian` is a `todo!()` stub; the noise that runs lives in
/// `kspace` and is reached through `Acquisition::noise_variance`.
pub mod noise;
/// Rigid poses, multiband slice schedules, within-volume dropout.
pub mod motion;
/// Per-slice k-space forward model and reconstruction.
pub mod kspace;
/// The 3D echo-train acquisition (P5): kz encoding by echo, per-partition reconstruction.
pub mod kspace3d;
/// The stack-of-spirals forward (P5 part C): the exact-sum oracle.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod spiral;
/// Type-1 NUFFT (feature `kspace`): the fieldmap y-sum of the forward model as one gridded FFT.
pub mod nufft;
/// Simulation config (feature `config`). `load` is a `todo!()` stub.
#[cfg(feature = "config")]
pub mod config;
/// NIfTI volume read + scalar/complex 4D write (feature `io`).
#[cfg(feature = "io")]
pub mod io;
