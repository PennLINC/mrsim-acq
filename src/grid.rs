//! The acquisition voxel grid.
//!
//! A plain data struct on purpose. Rasterisation geometry lives with the rasteriser in the
//! consumer crate, because a foreign type cannot take inherent impls; see `raster::GridRaster`
//! in TRXScan.

/// The acquisition voxel grid: dimensions + voxel→world (RAS mm) affine (may be oblique).
#[derive(Debug, Clone)]
pub struct Grid {
    pub dims: [usize; 3],
    /// voxel→world 4×4, row-major.
    pub voxel_to_world: [[f64; 4]; 4],
}
