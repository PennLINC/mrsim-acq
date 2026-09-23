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
