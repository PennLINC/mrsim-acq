//! NIfTI volume read and array write. Streamline I/O stays in the consumer crate, which is why
//! this feature pulls `nifti`/`ndarray`/`nalgebra` and never `trx-rs`: `aslscan` must not pay
//! for an HDF5 build it has no use for.
//!
//! - **Volumes in**: `nifti` 0.17 → f32 volume + voxel→world affine (sform, else qform
//!   quaternion, mirroring `TRXViz/trxviz-core/src/data/nifti_data.rs`).
//! - **Volumes out**: NIfTI-1 3D / 4D with the acquisition affine, and the complex
//!   `part-mag`/`part-phase` pair with its BIDS sidecars (`write_complex_4d`).

use crate::grid::Grid;

use nalgebra::Matrix4;
use ndarray::{Array4, Ix3};
use nifti::writer::WriterOptions;
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions, XForm};
use std::error::Error;
use std::path::{Path, PathBuf};

type R<T> = Result<T, Box<dyn Error>>;

/// voxel→world affine from a NIfTI header: sform if active, else qform quaternion.
fn affine_from_header(h: &NiftiHeader) -> [[f64; 4]; 4] {
    if h.sform_code > 0 {
        let (sx, sy, sz) = (h.srow_x, h.srow_y, h.srow_z);
        [
            [sx[0] as f64, sx[1] as f64, sx[2] as f64, sx[3] as f64],
            [sy[0] as f64, sy[1] as f64, sy[2] as f64, sy[3] as f64],
            [sz[0] as f64, sz[1] as f64, sz[2] as f64, sz[3] as f64],
            [0.0, 0.0, 0.0, 1.0],
        ]
    } else {
        let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        quatern_to_mat44(
            h.quatern_b as f64, h.quatern_c as f64, h.quatern_d as f64,
            h.quatern_x as f64, h.quatern_y as f64, h.quatern_z as f64,
            h.pixdim[1] as f64, h.pixdim[2] as f64, h.pixdim[3] as f64, qfac,
        )
    }
}

/// NIfTI-1 `quatern_to_mat44` (voxel→world from the qform quaternion + pixdims).
fn quatern_to_mat44(
    b: f64, c: f64, d: f64, qx: f64, qy: f64, qz: f64, dx: f64, dy: f64, dz: f64, qfac: f64,
) -> [[f64; 4]; 4] {
    let mut a = 1.0 - (b * b + c * c + d * d);
    let (mut b, mut c, mut d) = (b, c, d);
    if a < 1e-7 {
        let n = (b * b + c * c + d * d).sqrt();
        b /= n;
        c /= n;
        d /= n;
        a = 0.0;
    } else {
        a = a.sqrt();
    }
    let dz = dz * qfac;
    [
        [(a * a + b * b - c * c - d * d) * dx, 2.0 * (b * c - a * d) * dy, 2.0 * (b * d + a * c) * dz, qx],
        [2.0 * (b * c + a * d) * dx, (a * a + c * c - b * b - d * d) * dy, 2.0 * (c * d - a * b) * dz, qy],
        [2.0 * (b * d - a * c) * dx, 2.0 * (c * d + a * b) * dy, (a * a + d * d - c * c - b * b) * dz, qz],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Read a 3D NIfTI scalar volume as a flat `x + nx*(y + ny*z)` f32 buffer + its grid.
pub fn load_volume(path: &Path) -> R<(Vec<f32>, Grid)> {
    let obj = ReaderOptions::new().read_file(path)?;
    let header = obj.header().clone();
    let dims = [header.dim[1] as usize, header.dim[2] as usize, header.dim[3] as usize];
    let aff = affine_from_header(&header);
    let arr = obj.into_volume().into_ndarray::<f32>()?.into_dimensionality::<Ix3>()?;
    let [nx, ny, nz] = dims;
    let mut flat = vec![0.0f32; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                flat[x + nx * (y + ny * z)] = arr[[x, y, z]];
            }
        }
    }
    Ok((flat, Grid { dims, voxel_to_world: aff }))
}

fn header_for_grid(v: [[f64; 4]; 4]) -> NiftiHeader {
    let mut h = NiftiHeader::default();
    let col_norm = |c: usize| (v[0][c].powi(2) + v[1][c].powi(2) + v[2][c].powi(2)).sqrt() as f32;
    h.pixdim = [1.0, col_norm(0), col_norm(1), col_norm(2), 0.0, 0.0, 0.0, 0.0];
    h.xyzt_units = 2; // mm
    let m = Matrix4::<f64>::new(
        v[0][0], v[0][1], v[0][2], v[0][3],
        v[1][0], v[1][1], v[1][2], v[1][3],
        v[2][0], v[2][1], v[2][2], v[2][3],
        v[3][0], v[3][1], v[3][2], v[3][3],
    );
    h.set_qform(&m, XForm::ScannerAnat);
    h.set_sform(&m, XForm::ScannerAnat);
    h
}

/// Write a 4D `[x,y,z,g]` (layout `(x+nx*(y+ny*z))*ngrad+g`) as NIfTI-1 with the given affine.
/// The simulation grid's [`Grid`]: in-plane refined by `o`, same FOV, slice direction untouched.
///
/// The origin shifts by half the difference between the coarse and fine voxel sizes on each
/// refined axis, matching the half-cell registration the forward transform uses and the geometry
/// `scripts/prepare_acquisition_grid.py` writes. Identity at `o = 1`.
pub fn hires_grid(g: &Grid, o: usize) -> Grid {
    assert!(o > 0, "oversampling factor must be positive");
    let f = o as f64;
    let mut m = g.voxel_to_world;
    for r in 0..3 {
        for c in 0..2 {
            m[r][c] /= f;
        }
    }
    for r in 0..3 {
        m[r][3] = g.voxel_to_world[r][3]
            - 0.5 * g.voxel_to_world[r][0] * (1.0 - 1.0 / f)
            - 0.5 * g.voxel_to_world[r][1] * (1.0 - 1.0 / f);
    }
    Grid { dims: [g.dims[0] * o, g.dims[1] * o, g.dims[2]], voxel_to_world: m }
}

pub fn write_4d(path: &Path, dims: [usize; 3], ngrad: usize, data: &[f32], grid: &Grid) -> R<()> {
    let [nx, ny, nz] = dims;
    let arr = Array4::from_shape_fn((nx, ny, nz, ngrad), |(x, y, z, g)| {
        data[(x + nx * (y + ny * z)) * ngrad + g]
    });
    let hdr = header_for_grid(grid.voxel_to_world);
    WriterOptions::new(path).reference_header(&hdr).write_nifti(&arr)?;
    Ok(())
}

/// Write one 3D scalar volume (layout `x + nx*(y + ny*z)`) as NIfTI-1 with the given affine.
pub fn write_3d(path: &Path, dims: [usize; 3], data: &[f32], grid: &Grid) -> R<()> {
    let [nx, ny, nz] = dims;
    let arr =
        ndarray::Array3::from_shape_fn((nx, ny, nz), |(x, y, z)| data[x + nx * (y + ny * z)]);
    let hdr = header_for_grid(grid.voxel_to_world);
    WriterOptions::new(path).reference_header(&hdr).write_nifti(&arr)?;
    Ok(())
}

/// Write one 3D volume as int16 (Siemens-style phase images are stored as integers 0..4095).
pub fn write_3d_i16(path: &Path, dims: [usize; 3], data: &[i16], grid: &Grid) -> R<()> {
    let [nx, ny, nz] = dims;
    let arr =
        ndarray::Array3::from_shape_fn((nx, ny, nz), |(x, y, z)| data[x + nx * (y + ny * z)]);
    let hdr = header_for_grid(grid.voxel_to_world);
    WriterOptions::new(path).reference_header(&hdr).write_nifti(&arr)?;
    Ok(())
}

/// Acquisition facts the BIDS JSON sidecars record. A plain struct (not
/// `kspace::Acquisition`) so `io` stays usable without the `kspace` feature. Every field is
/// modality-independent; anything diffusion- or ASL-specific belongs beside the consumer's own
/// writer (TRXScan's `write_dwi_scheme`, aslscan's `_aslcontext.tsv`).
pub struct SidecarInfo {
    /// BIDS PhaseEncodingDirection ("j", "j-", …) already resolved for the *written* voxel frame
    /// (see `orient` + the caller): a +off-resonance field displaces signal toward +this-axis.
    pub phase_encoding_direction: String,
    pub total_readout_time: f64, // s
    pub echo_time: f64,          // s
    pub partial_fourier: f64,
    pub accel: usize,
    pub mb: usize,
    /// Modern BIDS B0 linkage: when a fieldmap is written for this DWI, its
    /// `B0FieldIdentifier` label is recorded here so the DWI carries the matching
    /// `B0FieldSource` (the replacement for the deprecated `IntendedFor`).
    pub b0_field_source: Option<String>,
}

/// Write a **complex** 4D series as BIDS `part-mag` / `part-phase` NIfTIs (phase in radians)
/// plus JSON sidecars — the layout `dwidenoise`/`dwidenoise2` consume for complex denoising.
/// `out_prefix` is a BIDS stem, e.g. `.../sub-01_ses-V02_dir-AP_run-01`; `suffix` is the BIDS
/// modality suffix without the leading underscore: `"dwi"`, `"asl"`. Modality-specific
/// companions (`.bval`/`.bvec`, `_aslcontext.tsv`) are the consumer's to write.
#[allow(clippy::too_many_arguments)]
pub fn write_complex_4d(
    out_prefix: &str,
    suffix: &str,
    dims: [usize; 3],
    n_volumes: usize,
    mag: &[f32],
    phase: &[f32],
    grid: &Grid,
    info: &SidecarInfo,
) -> R<()> {
    let p = |s: &str| PathBuf::from(format!("{out_prefix}{s}"));
    write_4d(&p(&format!("_part-mag_{suffix}.nii.gz")), dims, n_volumes, mag, grid)?;
    write_4d(&p(&format!("_part-phase_{suffix}.nii.gz")), dims, n_volumes, phase, grid)?;
    // PED is resolved by the caller for the written frame (native grid, or reoriented to LAS with
    // --fsl-orientation). EffectiveEchoSpacing is per the PE axis the code names, not a fixed axis.
    let ped = info.phase_encoding_direction.as_str();
    let pe_axis = match ped.as_bytes().first() { Some(b'i') => 0, Some(b'k') => 2, _ => 1 };
    let ees = info.total_readout_time / dims[pe_axis].saturating_sub(1).max(1) as f64;
    let b0src = match &info.b0_field_source {
        Some(id) => format!(",\n  \"B0FieldSource\": \"{id}\""),
        None => String::new(),
    };
    let common = format!(
        "  \"Manufacturer\": \"TRXScan\",\n  \"PhaseEncodingDirection\": \"{ped}\",\n  \
         \"TotalReadoutTime\": {:.6},\n  \"EffectiveEchoSpacing\": {:.8},\n  \
         \"EchoTime\": {:.4},\n  \"PartialFourier\": {},\n  \
         \"ParallelReductionFactorInPlane\": {},\n  \"MultibandAccelerationFactor\": {}{b0src}",
        info.total_readout_time, ees, info.echo_time, info.partial_fourier, info.accel, info.mb,
    );
    std::fs::write(p(&format!("_part-mag_{suffix}.json")),
        format!("{{\n{common},\n  \"ImageComparison\": \"magnitude\"\n}}\n"))?;
    std::fs::write(p(&format!("_part-phase_{suffix}.json")),
        format!("{{\n{common},\n  \"ImageComparison\": \"phase\",\n  \"Units\": \"rad\"\n}}\n"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hires_affine_preserves_the_fov() {
        // The hires volume must cover the same FOV as the nominal one, or the two references
        // cannot be compared voxel-for-voxel after block reduction.
        let g = Grid {
            dims: [8, 8, 2],
            voxel_to_world: [
                [1.7, 0.0, 0.0, -10.0],
                [0.0, 1.7, 0.0, -12.0],
                [0.0, 0.0, 1.7, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let o = 4;
        let h = hires_grid(&g, o);
        assert_eq!(h.dims, [32, 32, 2]);
        for ax in 0..2 {
            let nom = g.dims[ax] as f64 * g.voxel_to_world[ax][ax];
            let hi = h.dims[ax] as f64 * h.voxel_to_world[ax][ax];
            assert!((nom - hi).abs() < 1e-9, "axis {ax}: FOV {nom} vs {hi}");
        }
        assert_eq!(h.voxel_to_world[2][2], g.voxel_to_world[2][2], "z must not be refined");
    }

    #[test]
    fn hires_grid_is_identity_at_o_equals_one() {
        let g = Grid {
            dims: [5, 7, 3],
            voxel_to_world: [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 2.0],
                [0.0, 0.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let h = hires_grid(&g, 1);
        assert_eq!(h.dims, g.dims);
        for r in 0..4 {
            for c in 0..4 {
                assert!((h.voxel_to_world[r][c] - g.voxel_to_world[r][c]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn hires_corners_coincide_with_the_nominal_grid() {
        // Same check the prep script's geometry satisfies: the outer FOV corners must land in the
        // same world position, so the two grids describe one physical volume.
        let g = Grid {
            dims: [6, 9, 2],
            voxel_to_world: [
                [1.7, 0.0, 0.0, -5.0],
                [0.0, 1.7, 0.0, -7.0],
                [0.0, 0.0, 1.7, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        for &o in &[2usize, 3, 4, 8] {
            let h = hires_grid(&g, o);
            // corner of voxel (0,0): left edge = origin - half a voxel along each in-plane axis
            for r in 0..3 {
                let nom = g.voxel_to_world[r][3]
                    - 0.5 * g.voxel_to_world[r][0]
                    - 0.5 * g.voxel_to_world[r][1];
                let hi = h.voxel_to_world[r][3]
                    - 0.5 * h.voxel_to_world[r][0]
                    - 0.5 * h.voxel_to_world[r][1];
                assert!((nom - hi).abs() < 1e-9, "o={o} row {r}: {nom} vs {hi}");
            }
        }
    }
}
