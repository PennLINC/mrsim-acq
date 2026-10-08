//! The golden record of the forward model's full outputs, written before P5 changed anything
//! (P5 plan, Task 0). The bit-pinned unit tests pin eight coefficients and a tolerance; this pins
//! every output sample, so a change to the 2D spin-echo path that moves any bit is caught.
//!
//! Regenerate only on purpose, and only at a revision whose outputs are the reference:
//!
//!     P5_WRITE_GOLDEN=1 cargo test --test golden_baseline -- --ignored
//!     P5_WRITE_GOLDEN=1 cargo test --features kspace,par --test golden_baseline -- --ignored
//!
//! The default and `kspace` builds differ in their bits (the rustfft x-stage), so each has its
//! own directory.

use std::f64::consts::TAU;
use std::path::PathBuf;

use mrsim_acq::kspace::{
    simulate_acquisition_oversampled, simulate_slice, Acquisition, PartialFourierMode, Rng, SliceInput, T2Slice,
    T2Volume,
};
use mrsim_acq::phase::PhaseModel;

fn dir() -> PathBuf {
    let build = if cfg!(feature = "kspace") { "kspace" } else { "default" };
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/p4_baseline").join(build)
}

/// Every output of the reference configurations, by name, as `f32` bit patterns.
fn outputs() -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    // The cases of `restructured_forward_matches_the_literal_sum` (kspace.rs), through the whole
    // per-slice pipeline (all coils, reconstruction, combine), plus T2 and T2' maps.
    let mut rng = Rng(0xC0FFEE);
    for &(nx, ny, o) in &[(17usize, 70usize, 2usize), (16, 40, 1), (8, 9, 3)] {
        let (snx, sny) = (nx * o, ny * o);
        let n = snx * sny;
        let mut comps: Vec<Vec<f32>> = Vec::new();
        for c in 0..3 {
            comps.push((0..n).map(|i| {
                let (x, y) = ((i % snx) as f64 / snx as f64, (i / snx) as f64 / sny as f64);
                let blob = (-(((x - 0.5).powi(2) + (y - 0.45).powi(2)) / (0.03 * (c + 1) as f64))).exp();
                (blob + 0.05 * rng.unit()) as f32
            }).collect());
        }
        let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
        let fmap: Vec<f32> = (0..n).map(|i| {
            let (x, y) = ((i % snx) as f64 / snx as f64, (i / snx) as f64 / sny as f64);
            (300.0 * ((3.0 * x).sin() * (2.0 * y + 1.0).cos()) + 20.0 * rng.unit()) as f32
        }).collect();
        let phase0: Vec<f64> = (0..n).map(|_| TAU * rng.unit()).collect();
        let t2_map: Vec<f32> = (0..n).map(|i| (60.0 + 80.0 * ((i % snx) as f64 / snx as f64)) as f32).collect();
        let ti_map: Vec<f32> = (0..n).map(|i| (30.0 + 40.0 * ((i / snx) as f64 / sny as f64)) as f32).collect();
        let full = Acquisition { do_distortions: true, do_relaxation: true, signal_scale: 100.0, ..Acquisition::default() };
        let uniform = [T2Slice::Uniform(70.0f32), T2Slice::Uniform(100.0), T2Slice::Uniform(2000.0)];
        let mapped = [T2Slice::Map(&t2_map), T2Slice::Uniform(100.0), T2Slice::Map(&t2_map)];
        let ti_mapped = [T2Slice::Map(&ti_map), T2Slice::Uniform(45.0), T2Slice::Map(&ti_map)];
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, Acquisition, Option<[f64; 3]>, bool)> = vec![
            ("clean", Acquisition { do_distortions: false, do_relaxation: false, ..full.clone() }, None, false),
            ("distortion+relaxation", full.clone(), None, false),
            ("reverse", Acquisition { reverse_phase: true, ..full.clone() }, None, false),
            ("ghost", Acquisition { ghost_offset: 0.015, ..full.clone() }, None, false),
            ("eddy-poly", Acquisition { eddy_strength: 3.0, eddy_quad: 0.4, eddy_tau: 70.0, ..full.clone() },
                Some([0.3, -0.8, 0.5]), false),
            ("eddy-phase", Acquisition { eddy_phase: 0.2, ..full.clone() }, Some([0.6, 0.6, 0.5]), false),
            ("pf-fiberfox", Acquisition { partial_fourier: 0.75, ..full.clone() }, None, false),
            ("pf-contiguous-reverse", Acquisition { partial_fourier: 0.75, pf_mode: PartialFourierMode::Contiguous,
                reverse_phase: true, ..full.clone() }, None, false),
            ("grappa-coils", Acquisition { accel: 2, acs_lines: 6, n_coils: 4, ..full.clone() }, None, false),
            ("grappa3", Acquisition { accel: 3, acs_lines: 8, n_coils: 4, ..full.clone() }, None, false),
            ("everything", Acquisition { ghost_offset: 0.02, eddy_strength: 2.0, eddy_quad: 0.3, eddy_phase: 0.1,
                partial_fourier: 0.8, accel: 2, acs_lines: 8, n_coils: 3, ..full.clone() },
                Some([0.5, 0.5, 0.7]), false),
            ("maps", full.clone(), None, true),
            ("maps-coils-noise", Acquisition { n_coils: 3, noise_variance: 4.0, ..full.clone() },
                None, true),
        ];
        for (name, acq, eddy_drive, maps) in cases {
            let inp = SliceInput {
                compartments: &comp_refs, t2: if maps { &mapped } else { &uniform },
                t_inhom: if maps { Some(&ti_mapped) } else { None }, fmap: &fmap, phase0: Some(&phase0),
                sim: [snx, sny], acq_matrix: [nx, ny], z: 3, nz: 9, eddy_drive, prep_drive: None, slice_seed: 7,
            };
            let img = simulate_slice(&inp, &acq);
            let bits = img.iter().flat_map(|(re, im)| [re.to_bits(), im.to_bits()]).collect();
            out.push((format!("slice-{nx}x{ny}o{o}-{name}"), bits));
        }
    }

    // One whole series through the production entry point: several volumes and slices, coils,
    // GRAPPA, k-space noise, the object phase model, uniform and mapped relaxation.
    let (nx, ny, nz, o, nv) = (16usize, 16usize, 3usize, 2usize, 4usize);
    let (snx, sny) = (nx * o, ny * o);
    let nsim = snx * sny * nz;
    let mut rng = Rng(0xBA5E);
    let images: Vec<Vec<f32>> = (0..2).map(|c| {
        (0..nsim * nv).map(|i| {
            let v = i / nsim;
            let vox = i % nsim;
            let (x, y) = ((vox % snx) as f64 / snx as f64, ((vox / snx) % sny) as f64 / sny as f64);
            let blob = (-(((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.05)).exp();
            ((1.0 + 0.1 * c as f64 + 0.01 * v as f64) * blob + 0.02 * rng.unit()) as f32
        }).collect::<Vec<f32>>()
    }).map(|by_volume| {
        // the entry point's layout is voxel-major with the volume innermost
        let mut v = vec![0.0f32; nsim * nv];
        for g in 0..nv {
            for vox in 0..nsim {
                v[vox * nv + g] = by_volume[g * nsim + vox];
            }
        }
        v
    }).collect();
    let fmap: Vec<f32> = (0..nsim).map(|i| (40.0 * ((i % snx) as f64 / snx as f64 - 0.5)) as f32).collect();
    let t2map: Vec<f32> = (0..nsim).map(|i| (50.0 + (i % 7) as f64 * 10.0) as f32).collect();
    let acq = Acquisition { n_coils: 3, accel: 2, acs_lines: 6, noise_variance: 2.0, signal_scale: 100.0, t_echo: 30.0,
        t_line: 0.5, ..Acquisition::default() };
    for (name, t2) in [("uniform", vec![T2Volume::Uniform(80.0), T2Volume::Uniform(160.0)]),
                       ("mapped", vec![T2Volume::Map(&t2map), T2Volume::Uniform(160.0)])] {
        let (mag, ph) = simulate_acquisition_oversampled(
            [snx, sny, nz], [nx, ny, nz], nv, &images, &t2, &fmap, None, &acq, &vec![None; nv], &vec![None; nv],
            &PhaseModel::hbcd_like(), 42, None);
        let bits = mag.iter().chain(&ph).map(|x| x.to_bits()).collect();
        out.push((format!("series-{name}"), bits));
    }
    out
}

fn path(name: &str) -> PathBuf {
    dir().join(format!("{name}.bin"))
}

#[test]
#[ignore = "writes the golden record; run only on purpose (see the module doc)"]
fn write_golden_outputs() {
    if std::env::var("P5_WRITE_GOLDEN").as_deref() != Ok("1") {
        panic!("set P5_WRITE_GOLDEN=1 to overwrite the golden record");
    }
    std::fs::create_dir_all(dir()).unwrap();
    for (name, bits) in outputs() {
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        std::fs::write(path(&name), bytes).unwrap();
    }
}

#[test]
fn outputs_match_the_golden_record_bit_for_bit() {
    let all = outputs();
    assert!(all.len() >= 40, "{} cases", all.len());
    for (name, bits) in all {
        let bytes = std::fs::read(path(&name)).unwrap_or_else(|e| panic!("{name}: {e} (no golden record for this build)"));
        let want: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        assert_eq!(want.len(), bits.len(), "{name}: length");
        if let Some(i) = (0..bits.len()).find(|&i| bits[i] != want[i]) {
            panic!("{name}: first difference at {i}: {:e} vs golden {:e}", f32::from_bits(bits[i]), f32::from_bits(want[i]));
        }
    }
}
