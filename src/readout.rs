//! Readout schemes: `tick → (kx, ky)` k-space trajectory and per-tick timing.
//!
//! [`SingleShotEpi`] is a faithful port of `Sequences/mitkSingleShotEpi.h`.
//! Fast/Conventional spin-echo (`mitkFastSpinEcho.h`, `mitkConventionalSpinEcho.h`) can follow the
//! same trait later. The `AcquisitionType` interface is purely in-plane — slice timing for
//! multiband lives in `motion`/`kspace`.

/// In-plane readout trajectory + timing. `tick` runs `0..kx_max*ky_max` in acquisition order.
pub trait Readout {
    fn kspace_index(&self, tick: usize) -> (usize, usize);
    /// ms from the maximum-echo (k-space centre) time.
    fn time_from_max_echo(&self, tick: usize) -> f64;
    /// ms from the last large preparation gradient to the sample at `tick` (drives eddy-current
    /// decay). That gradient is a diffusion gradient in one consumer and a crusher or labeling
    /// gradient in the other.
    fn time_from_prep_gradient(&self, tick: usize) -> f64;
    /// ms since the RF pulse (used for T2/T2* relaxation).
    fn time_from_rf(&self, tick: usize) -> f64;
}

/// Single-shot EPI: one echo, max intensity at k-space centre, zig-zag ("snake") trajectory.
#[derive(Debug, Clone, Copy)]
pub struct SingleShotEpi {
    pub kx_max: usize,
    pub ky_max: usize,
    /// time to read one PE line (ms)
    pub t_line: f64,
    /// echo time (ms)
    pub t_echo: f64,
    pub reverse_phase: bool,
}

impl SingleShotEpi {
    /// time to read one k-space sample (ms)
    #[inline]
    pub fn dt(&self) -> f64 {
        self.t_line / self.kx_max as f64
    }
    #[inline]
    fn half_read_time(&self) -> f64 {
        (self.kx_max * self.ky_max) as f64 * self.dt() / 2.0
    }
}

impl Readout for SingleShotEpi {
    fn kspace_index(&self, tick: usize) -> (usize, usize) {
        let mut x = tick % self.kx_max;
        let mut y = tick / self.kx_max;
        if !self.reverse_phase {
            y = self.ky_max - 1 - y; // start at the maximum k-space line
            if y % 2 == 1 {
                x = self.kx_max - x - 1; // reverse frequency-encode direction on odd lines
            }
        } else if y % 2 == 1 {
            x = self.kx_max - x - 1;
        }
        (x, y)
    }

    fn time_from_max_echo(&self, tick: usize) -> f64 {
        self.dt() * (tick as f64 + 0.5) - self.half_read_time()
    }
    fn time_from_prep_gradient(&self, tick: usize) -> f64 {
        tick as f64 * self.dt() + self.dt() / 2.0
    }
    fn time_from_rf(&self, tick: usize) -> f64 {
        self.t_echo + self.time_from_max_echo(tick)
    }
}

// ---- 3D echo trains (P5 addendum, part B) ----

/// The order in which kz partitions are read along the echo train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KzOrder {
    /// The kz centre first, then `+1, -1, +2, -2, ...` (what the product sequences do for ASL).
    Centric,
    /// Low to high.
    Linear,
}

/// A segmented spin-echo train: per shot one 90-degree excitation, then `etl` refocusing pulses of
/// `refocusing_deg` at echo spacing `esp_ms`, each echo reading one kz partition. The partitions
/// are split across `kz_segments` shots. `refocusing_time_ms` is the time each refocusing pulse
/// and its crushers occupy, centred midway between echoes.
#[derive(Debug, Clone, PartialEq)]
pub struct EchoTrain {
    pub etl: usize,
    pub esp_ms: f64,
    pub refocusing_deg: f64,
    pub kz_order: KzOrder,
    pub kz_segments: usize,
    pub refocusing_time_ms: f64,
}

/// The in-plane readout of each echo. GRASE: an EPI block of `ny / ky_segments` lines, interleaved
/// (segment `sy` reads `ky = sy + ky_segments j`), centred on the spin echo, at the actual line
/// spacing `t_line_ms`; `reverse_phase` as [`SingleShotEpi`]'s (false: the block starts at its
/// highest `ky` and descends). Spiral: one spiral-out interleaf of `interleaves` per echo, starting
/// at the spin echo, `readout_ms` long, sampled every `dwell_ms` ([`spiral_trajectory`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Readout3d {
    Grase { ky_segments: usize, t_line_ms: f64, reverse_phase: bool },
    Spiral { interleaves: usize, readout_ms: f64, dwell_ms: f64 },
}

/// Partitions in the order the train reads them: `kz_order(nz, order)[n]` is the `n`-th read.
/// The kz centre is index `nz / 2`, the centred convention's `k = 0` (acquired band
/// `[-nz/2, nz/2 - 1]`, as in-plane).
pub fn kz_order(nz: usize, order: KzOrder) -> Vec<usize> {
    match order {
        KzOrder::Linear => (0..nz).collect(),
        KzOrder::Centric => {
            let c = nz / 2;
            let mut out = vec![c];
            let mut d = 1;
            while out.len() < nz {
                if c + d < nz {
                    out.push(c + d);
                }
                if d <= c && out.len() < nz {
                    out.push(c - d);
                }
                d += 1;
            }
            out
        }
    }
}

/// The within-echo timing of a GRASE block, per `ky`: time from the echo centre (ms), readout
/// polarity (`+1`/`-1`, alternating with the acquisition index within the block, not with `ky`)
/// and the ky segment. The same for every echo of every shot.
#[derive(Debug, Clone, PartialEq)]
pub struct GraseBlock {
    pub t_ms: Vec<f64>,
    pub polarity: Vec<i8>,
    pub segment: Vec<usize>,
    /// Lines per block.
    pub epi: usize,
    /// The actual line spacing (ms).
    pub t_line_ms: f64,
}

/// The block timing of [`Readout3d::Grase`] on `ny` phase-encode lines.
pub fn grase_block(ny: usize, ky_segments: usize, t_line_ms: f64, reverse_phase: bool) -> Result<GraseBlock, String> {
    if ky_segments == 0 || !ny.is_multiple_of(ky_segments) {
        return Err(format!("{ny} phase-encode lines do not divide into {ky_segments} ky segments"));
    }
    let epi = ny / ky_segments;
    let (mut t_ms, mut polarity, mut segment) = (vec![0.0; ny], vec![0i8; ny], vec![0usize; ny]);
    for ky in 0..ny {
        let (sy, j) = (ky % ky_segments, ky / ky_segments);
        // acquisition index within the block: descending from the top line unless reversed
        let a = if reverse_phase { j } else { epi - 1 - j };
        t_ms[ky] = (a as f64 - (epi as f64 - 1.0) / 2.0) * t_line_ms;
        polarity[ky] = if a % 2 == 1 { -1 } else { 1 };
        segment[ky] = sy;
    }
    Ok(GraseBlock { t_ms, polarity, segment, epi, t_line_ms })
}

/// One acquired line of a 3D GRASE acquisition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraseLine {
    /// Canonical shot index `sy kz_segments + sz`.
    pub shot: usize,
    /// Echo number, 1-based.
    pub echo: usize,
    /// Time from the echo centre and from the excitation (ms).
    pub t_ms: f64,
    pub trf_ms: f64,
    pub polarity: i8,
}

/// Every acquired line of a 3D GRASE acquisition, indexed `p * ny + ky`.
#[derive(Debug, Clone, PartialEq)]
pub struct Grase3dTable {
    pub ny: usize,
    pub nz: usize,
    pub lines: Vec<GraseLine>,
    pub block: GraseBlock,
    /// The echo that reads the kz centre.
    pub e_c: usize,
    pub n_shots: usize,
}

impl Grase3dTable {
    pub fn line(&self, p: usize, ky: usize) -> &GraseLine {
        &self.lines[p * self.ny + ky]
    }

    /// The within-echo [`crate::kspace::LineTiming`] the relaxation-free 2D forward reads: `t`
    /// and polarity per `ky`, the same for every partition and shot (asserted). Its `trf_ms` and
    /// `tread_ms` are set to `t` and are not read on the 3D path, which runs the forward with
    /// relaxation off and refuses the eddy model; the full `trf` per `(p, ky)` is in `lines`.
    pub fn within_echo_timing(&self) -> crate::kspace::LineTiming {
        for p in 0..self.nz {
            for ky in 0..self.ny {
                let l = self.line(p, ky);
                assert!(l.t_ms == self.block.t_ms[ky] && l.polarity == self.block.polarity[ky],
                        "within-echo timing differs between partitions at ({p}, {ky})");
            }
        }
        crate::kspace::LineTiming {
            t_ms: self.block.t_ms.clone(),
            trf_ms: self.block.t_ms.clone(),
            tread_ms: self.block.t_ms.clone(),
            polarity: self.block.polarity.clone(),
        }
    }
}

/// The echo that reads the kz centre, for a train on `nz` partitions.
pub fn centre_echo(train: &EchoTrain, nz: usize) -> Result<usize, String> {
    let order = kz_order(nz, train.kz_order);
    let n = order.iter().position(|&p| p == nz / 2).expect("the centre is a partition");
    if train.kz_segments == 0 || !nz.is_multiple_of(train.kz_segments) {
        return Err(format!("{nz} partitions do not divide into {} kz segments", train.kz_segments));
    }
    Ok(n / train.kz_segments + 1)
}

/// `ESP` from BIDS's `EchoTime`, the time from the excitation to the k-space centre: the centre
/// line `ny / 2` of the kz-centre partition is read at `e_c ESP + t(ky_c)`.
pub fn esp_from_echo_time(te_ms: f64, e_c: usize, block: &GraseBlock) -> f64 {
    (te_ms - block.t_ms[block.t_ms.len() / 2]) / e_c as f64
}

/// The full line table of a GRASE acquisition on `ny x nz`.
pub fn grase_lines(train: &EchoTrain, readout: &Readout3d, ny: usize, nz: usize) -> Result<Grase3dTable, String> {
    let Readout3d::Grase { ky_segments, t_line_ms, reverse_phase } = *readout else {
        return Err("a spiral readout has no GRASE line table (spiral_lines)".to_string());
    };
    let block = grase_block(ny, ky_segments, t_line_ms, reverse_phase)?;
    if train.kz_segments == 0 || !nz.is_multiple_of(train.kz_segments) {
        return Err(format!("{nz} partitions do not divide into {} kz segments", train.kz_segments));
    }
    if train.etl != nz / train.kz_segments {
        return Err(format!("echo train length {} is not {nz} partitions / {} kz segments", train.etl, train.kz_segments));
    }
    let order = kz_order(nz, train.kz_order);
    let mut lines = vec![GraseLine { shot: 0, echo: 0, t_ms: 0.0, trf_ms: 0.0, polarity: 0 }; ny * nz];
    let mut seen = vec![false; ny * nz];
    for sy in 0..ky_segments {
        for sz in 0..train.kz_segments {
            let shot = sy * train.kz_segments + sz;
            for e in 1..=train.etl {
                let p = order[sz + train.kz_segments * (e - 1)];
                for j in 0..block.epi {
                    let ky = sy + ky_segments * j;
                    let i = p * ny + ky;
                    assert!(!seen[i], "line ({p}, {ky}) read twice");
                    seen[i] = true;
                    let t = block.t_ms[ky];
                    lines[i] = GraseLine { shot, echo: e, t_ms: t, trf_ms: e as f64 * train.esp_ms + t, polarity: block.polarity[ky] };
                }
            }
        }
    }
    assert!(seen.iter().all(|&s| s), "a line was never read");
    let e_c = centre_echo(train, nz)?;
    Ok(Grase3dTable { ny, nz, lines, block, e_c, n_shots: ky_segments * train.kz_segments })
}

/// The timing checks of a GRASE train on the actual RF and sampling intervals (P5 addendum,
/// "Timing from BIDS"): the block between refocusing pulses, the first pulse after the
/// excitation, the last sample within the repetition. Times in ms; `t_exc_ms` from the start of
/// the repetition (labeling start) to the excitation.
pub fn check_grase_timing(train: &EchoTrain, table: &Grase3dTable, t_exc_ms: f64, tr_ms: f64) -> Result<(), String> {
    let t_line = table.block.t_line_ms;
    let extent = table.block.t_ms.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    let half = train.esp_ms / 2.0;
    let need = extent + t_line / 2.0 + train.refocusing_time_ms / 2.0;
    if need > half + 1e-9 {
        return Err(format!(
            "the {}-line EPI block reaches {:.4} ms from its echo (lines of {:.4} ms), and with half the {:.3} ms \
             refocusing time needs {:.4} ms, more than half the echo spacing ({:.4} ms of {:.4} ms). Fewer lines \
             per block (more ky segments, a smaller matrix), shorter lines, or a longer echo spacing would fit: \
             the echo spacing must be at least {:.4} ms",
            table.block.epi, extent + t_line / 2.0, t_line, train.refocusing_time_ms, need, half, train.esp_ms, 2.0 * need));
    }
    if half - train.refocusing_time_ms / 2.0 <= 0.0 {
        return Err(format!("the first refocusing pulse ({:.3} ms long, centred {:.4} ms after the excitation) \
                            overlaps the excitation", train.refocusing_time_ms, half));
    }
    let last = t_exc_ms + train.etl as f64 * train.esp_ms + extent + t_line / 2.0;
    if last > tr_ms + 1e-9 {
        return Err(format!("the echo train ends at {:.4} ms (excitation at {:.4} ms, {} echoes of {:.4} ms), after \
                            the {:.4} ms repetition", last, t_exc_ms, train.etl, train.esp_ms, tr_ms));
    }
    Ok(())
}

// ---- the stack of spirals (P5 addendum, part C) ----

/// An Archimedean constant-density spiral-out design on a square `nx x nx` matrix, in cycles/FOV:
/// interleaf `s` of `N` is `k_s(u) = k_max u exp(i (2 pi n_turns u + 2 pi s / N))`, `k_max = nx/2`,
/// `n_turns = nx / (2N)`, with `u(tau)` constant angular velocity inside `tau_c` and constant
/// linear velocity (`u = sqrt(tau / T)`) outside. `tau_c` is the smallest centre region keeping the
/// speed times the dwell time at most one cycle/FOV everywhere on the continuous trajectory (a
/// sampling bound, not a gradient or slew limit).
#[derive(Debug, Clone, PartialEq)]
pub struct SpiralTrajectory {
    pub interleaves: usize,
    pub readout_ms: f64,
    pub dwell_ms: f64,
    pub k_max: f64,
    pub n_turns: f64,
    pub tau_c_ms: f64,
    /// Sample times from the echo (ms), `(j + 1/2) dwell`, `j = 0..floor(T / dwell)`.
    pub tau_ms: Vec<f64>,
    /// `k[s * n_samples + j] = [kx, ky]`, cycles/FOV, the Cartesian forward's `kx - nx/2`.
    pub k: Vec<[f64; 2]>,
}

impl SpiralTrajectory {
    pub fn n_samples(&self) -> usize {
        self.tau_ms.len()
    }

    pub fn interleaf(&self, s: usize) -> &[[f64; 2]] {
        let n = self.n_samples();
        &self.k[s * n..(s + 1) * n]
    }

    /// The normalized parameter `u(tau)`.
    pub fn u(&self, tau_ms: f64) -> f64 {
        let (t, tc) = (self.readout_ms, self.tau_c_ms);
        if tau_ms < tc { tau_ms / (tc * t).sqrt() } else { (tau_ms / t).sqrt() }
    }

    /// The continuous position of interleaf `s` at `tau_ms`.
    pub fn at(&self, s: usize, tau_ms: f64) -> [f64; 2] {
        let u = self.u(tau_ms);
        let th = std::f64::consts::TAU * (self.n_turns * u + s as f64 / self.interleaves as f64);
        [self.k_max * u * th.cos(), self.k_max * u * th.sin()]
    }

    /// The speed `|dk/dtau|` (cycles/FOV per ms) at `tau_ms > 0`, the same for every interleaf.
    pub fn speed(&self, tau_ms: f64) -> f64 {
        let (t, tc) = (self.readout_ms, self.tau_c_ms);
        let du = if tau_ms < tc { 1.0 / (tc * t).sqrt() } else { 0.5 / (tau_ms * t).sqrt() };
        let w = std::f64::consts::TAU * self.n_turns * self.u(tau_ms);
        self.k_max * du * (1.0 + w * w).sqrt()
    }
}

/// The smallest constant-angular-velocity centre region (ms) of the spiral of `interleaves` on an
/// `nx` matrix: the speed is largest at `tau_c` (inside), `v = k_max sqrt(1/(tau_c T) +
/// 4 pi^2 n_turns^2 / T^2)`, and `v dwell = 1` solves to
/// `tau_c = 1 / (T (1/(k_max dwell)^2 - 4 pi^2 n_turns^2 / T^2))`.
pub fn spiral_tau_c(nx: usize, interleaves: usize, readout_ms: f64, dwell_ms: f64) -> Result<f64, String> {
    let pos = |v: f64| v.is_finite() && v > 0.0;
    if interleaves == 0 || !pos(readout_ms) || !pos(dwell_ms) {
        return Err(format!("a spiral needs interleaves > 0, a positive readout time and a positive dwell time \
                            (got {interleaves}, {readout_ms} ms, {dwell_ms} ms)"));
    }
    let k_max = nx as f64 / 2.0;
    let n_turns = nx as f64 / (2.0 * interleaves as f64);
    let t = readout_ms;
    let pi = std::f64::consts::PI;
    let dwell_max = t / (2.0 * pi * n_turns * k_max);
    let den = 1.0 / (k_max * dwell_ms).powi(2) - 4.0 * pi * pi * n_turns * n_turns / (t * t);
    if dwell_ms >= dwell_max || den <= 0.0 {
        return Err(format!(
            "a {nx} x {nx} spiral of {interleaves} interleaves ({n_turns} turns each) over {t} ms cannot be sampled \
             every {dwell_ms} ms: its outer turns alone move more than one cycle/FOV per sample. The dwell time must be \
             below {dwell_max:.6e} ms (or more interleaves, or a longer readout)"));
    }
    let tau_c = 1.0 / (t * den);
    if tau_c > t {
        return Err(format!(
            "a {nx} x {nx} spiral of {interleaves} interleaves sampled every {dwell_ms} ms needs a constant-angular-velocity \
             centre of {tau_c:.6} ms to keep within one cycle/FOV per sample, longer than the {t} ms readout: the readout \
             is too short for the dwell time"));
    }
    Ok(tau_c)
}

/// The sampled spiral of [`Readout3d::Spiral`] on an `nx x ny` matrix (`nx = ny` required).
pub fn spiral_trajectory(nx: usize, ny: usize, interleaves: usize, readout_ms: f64, dwell_ms: f64)
                         -> Result<SpiralTrajectory, String> {
    if nx != ny {
        return Err(format!("a spiral readout needs a square in-plane matrix, not {nx} x {ny}"));
    }
    let tau_c_ms = spiral_tau_c(nx, interleaves, readout_ms, dwell_ms)?;
    // floor(T / dwell), robust to T / dwell landing a rounding error below an integer
    let n = (readout_ms / dwell_ms * (1.0 + 1e-12)).floor() as usize;
    let mut tr = SpiralTrajectory {
        interleaves, readout_ms, dwell_ms, k_max: nx as f64 / 2.0, n_turns: nx as f64 / (2.0 * interleaves as f64),
        tau_c_ms, tau_ms: (0..n).map(|j| (j as f64 + 0.5) * dwell_ms).collect(), k: Vec::with_capacity(n * interleaves),
    };
    for s in 0..interleaves {
        for j in 0..n {
            let p = tr.at(s, tr.tau_ms[j]);
            tr.k.push(p);
        }
    }
    Ok(tr)
}

/// The partition chronology of a stack of spirals on `nx x ny x nz`: each echo reads one interleaf
/// at one partition; shot `s kz_segments + sz` is interleaf `s` over kz segment `sz`.
#[derive(Debug, Clone, PartialEq)]
pub struct Spiral3dTable {
    pub nz: usize,
    pub traj: SpiralTrajectory,
    /// Per partition: the echo (1-based) that reads it and its kz segment.
    pub echo: Vec<usize>,
    pub kz_segment: Vec<usize>,
    pub kz_segments: usize,
    pub esp_ms: f64,
    /// The echo that reads the kz centre.
    pub e_c: usize,
    pub n_shots: usize,
}

impl Spiral3dTable {
    /// The canonical shot reading interleaf `s` of partition `p`.
    pub fn shot(&self, p: usize, s: usize) -> usize {
        s * self.kz_segments + self.kz_segment[p]
    }

    /// Time from the excitation (ms) of sample `j` of partition `p`.
    pub fn trf_ms(&self, p: usize, j: usize) -> f64 {
        self.echo[p] as f64 * self.esp_ms + self.traj.tau_ms[j]
    }
}

/// The table of a [`Readout3d::Spiral`] train.
pub fn spiral_lines(train: &EchoTrain, readout: &Readout3d, nx: usize, ny: usize, nz: usize) -> Result<Spiral3dTable, String> {
    let Readout3d::Spiral { interleaves, readout_ms, dwell_ms } = *readout else {
        return Err("a GRASE readout has no spiral table (grase_lines)".to_string());
    };
    let traj = spiral_trajectory(nx, ny, interleaves, readout_ms, dwell_ms)?;
    if train.kz_segments == 0 || !nz.is_multiple_of(train.kz_segments) {
        return Err(format!("{nz} partitions do not divide into {} kz segments", train.kz_segments));
    }
    if train.etl != nz / train.kz_segments {
        return Err(format!("echo train length {} is not {nz} partitions / {} kz segments", train.etl, train.kz_segments));
    }
    let order = kz_order(nz, train.kz_order);
    let (mut echo, mut kz_segment) = (vec![0; nz], vec![0; nz]);
    for (n, &p) in order.iter().enumerate() {
        echo[p] = n / train.kz_segments + 1;
        kz_segment[p] = n % train.kz_segments;
    }
    let e_c = centre_echo(train, nz)?;
    Ok(Spiral3dTable { nz, traj, echo, kz_segment, kz_segments: train.kz_segments, esp_ms: train.esp_ms, e_c,
                       n_shots: interleaves * train.kz_segments })
}

/// `ESP` from BIDS's `EchoTime` for a spiral: the kz-centre echo's spiral starts at its echo, so
/// the k-space centre is read at `e_c ESP`.
pub fn spiral_esp_from_echo_time(te_ms: f64, e_c: usize) -> f64 {
    te_ms / e_c as f64
}

/// The timing checks of a spiral train on the actual intervals (P5 addendum, part C, "Inputs"):
/// each spiral, starting at its echo, ends before the next refocusing pulse's reserved interval
/// (`T + refocusing_time/2 <= ESP/2`); the first pulse clears the excitation; the train ends
/// within the repetition. Returns the train's end (ms from the start of the repetition).
pub fn check_spiral_timing(train: &EchoTrain, table: &Spiral3dTable, t_exc_ms: f64, tr_ms: f64) -> Result<f64, String> {
    let t = table.traj.readout_ms;
    let half = train.esp_ms / 2.0;
    let need = t + train.refocusing_time_ms / 2.0;
    if need > half + 1e-9 {
        return Err(format!(
            "the {t} ms spiral with half the {:.3} ms refocusing time needs {need:.4} ms after its echo, more than half \
             the echo spacing ({half:.4} ms of {:.4} ms): it would run into the next refocusing pulse. A shorter spiral \
             (more interleaves) or a longer echo spacing would fit: the echo spacing must be at least {:.4} ms",
            train.refocusing_time_ms, train.esp_ms, 2.0 * need));
    }
    if half - train.refocusing_time_ms / 2.0 <= 0.0 {
        return Err(format!("the first refocusing pulse ({:.3} ms long, centred {:.4} ms after the excitation) \
                            overlaps the excitation", train.refocusing_time_ms, half));
    }
    let last = t_exc_ms + train.etl as f64 * train.esp_ms + t;
    if last > tr_ms + 1e-9 {
        return Err(format!("the echo train ends at {last:.4} ms (excitation at {t_exc_ms:.4} ms, {} echoes of {:.4} ms, \
                            a {t} ms spiral), after the {tr_ms:.4} ms repetition", train.etl, train.esp_ms));
    }
    Ok(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_is_a_permutation() {
        // every (kx,ky) cell is visited exactly once, both phase polarities.
        for reverse in [false, true] {
            let epi = SingleShotEpi { kx_max: 6, ky_max: 5, t_line: 1.0, t_echo: 90.0, reverse_phase: reverse };
            let mut seen = vec![false; epi.kx_max * epi.ky_max];
            for tick in 0..epi.kx_max * epi.ky_max {
                let (x, y) = epi.kspace_index(tick);
                assert!(x < epi.kx_max && y < epi.ky_max);
                let flat = y * epi.kx_max + x;
                assert!(!seen[flat], "cell ({x},{y}) visited twice");
                seen[flat] = true;
            }
            assert!(seen.iter().all(|&v| v), "trajectory missed a cell");
        }
    }

    #[test]
    fn echo_time_is_monotonic() {
        let epi = SingleShotEpi { kx_max: 8, ky_max: 8, t_line: 1.0, t_echo: 90.0, reverse_phase: false };
        let n = epi.kx_max * epi.ky_max;
        for tick in 1..n {
            assert!(epi.time_from_max_echo(tick) > epi.time_from_max_echo(tick - 1));
        }
    }

    fn train(nz: usize, kz_segments: usize, order: KzOrder, esp_ms: f64) -> EchoTrain {
        EchoTrain { etl: nz / kz_segments, esp_ms, refocusing_deg: 180.0, kz_order: order, kz_segments, refocusing_time_ms: 2.0 }
    }

    #[test]
    fn kz_orders_are_permutations_with_the_centre_where_stated() {
        for nz in [1usize, 2, 7, 16, 30] {
            for order in [KzOrder::Centric, KzOrder::Linear] {
                let o = kz_order(nz, order);
                let mut s = o.clone();
                s.sort();
                assert_eq!(s, (0..nz).collect::<Vec<_>>(), "{nz} {order:?}");
            }
            assert_eq!(kz_order(nz, KzOrder::Centric)[0], nz / 2);
        }
        assert_eq!(kz_order(6, KzOrder::Centric), vec![3, 4, 2, 5, 1, 0]);
    }

    #[test]
    fn every_line_is_read_once_with_ky_only_within_echo_timing() {
        let (ny, nz) = (16usize, 8usize);
        for ky_segments in [1usize, 2, 4] {
            for kz_segments in [1usize, 2] {
                for order in [KzOrder::Centric, KzOrder::Linear] {
                    for reverse_phase in [false, true] {
                        let tr = train(nz, kz_segments, order, 20.0);
                        let ro = Readout3d::Grase { ky_segments, t_line_ms: 0.5, reverse_phase };
                        let tab = grase_lines(&tr, &ro, ny, nz).unwrap();
                        assert_eq!(tab.n_shots, ky_segments * kz_segments);
                        // within-echo time, polarity independent of partition and shot (asserted
                        // inside); trf differs by ESP between consecutive echoes
                        let lt = tab.within_echo_timing();
                        for p in 0..nz {
                            for ky in 0..ny {
                                let l = tab.line(p, ky);
                                assert_eq!(l.t_ms, lt.t_ms[ky]);
                                assert!((l.trf_ms - (l.echo as f64 * 20.0 + l.t_ms)).abs() < 1e-12);
                                assert_eq!(l.shot, (ky % ky_segments) * kz_segments + (l.shot % kz_segments));
                            }
                        }
                        // polarity alternates with the acquisition index inside each block
                        let blk = &tab.block;
                        let mut by_time: Vec<usize> = (0..ny).filter(|ky| blk.segment[*ky] == 0).collect();
                        by_time.sort_by(|a, b| blk.t_ms[*a].partial_cmp(&blk.t_ms[*b]).unwrap());
                        for w in by_time.windows(2) {
                            assert_eq!(blk.polarity[w[0]], -blk.polarity[w[1]]);
                            assert!((blk.t_ms[w[1]] - blk.t_ms[w[0]] - 0.5).abs() < 1e-12);
                        }
                        // the traversal: j- descends from the block's top line, j ascends
                        let first = by_time[0];
                        assert_eq!(first, if reverse_phase { 0 } else { ny - ky_segments });
                    }
                }
            }
        }
        assert_eq!(centre_echo(&train(nz, 1, KzOrder::Centric, 20.0), nz).unwrap(), 1);
        assert_eq!(centre_echo(&train(nz, 1, KzOrder::Linear, 20.0), nz).unwrap(), nz / 2 + 1);
    }

    #[test]
    fn worked_numbers_for_the_acceptance_cases() {
        // the asl003 derivative: 20 lines, two segments, 1 ms lines, j- (reverse_phase false),
        // TE 11.92 ms: t(ky_c) = -0.5 ms, ESP = 12.42 ms, and the block fits the 2 ms reserve
        let blk = grase_block(20, 2, 1.0, false).unwrap();
        assert!((blk.t_ms[10] + 0.5).abs() < 1e-12, "{}", blk.t_ms[10]);
        let esp = esp_from_echo_time(11.92, 1, &blk);
        assert!((esp - 12.42).abs() < 1e-12, "{esp}");
        let tr = EchoTrain { etl: 30, esp_ms: esp, refocusing_deg: 180.0, kz_order: KzOrder::Centric, kz_segments: 1,
                             refocusing_time_ms: 2.0 };
        let tab = grase_lines(&tr, &Readout3d::Grase { ky_segments: 2, t_line_ms: 1.0, reverse_phase: false }, 20, 30).unwrap();
        // the latest retained excitation of the derivative is at 3.0 s; the train ends at 3.3776 s
        check_grase_timing(&tr, &tab, 3000.0, 3500.0).unwrap();
        let end = 3000.0 + 30.0 * esp + 4.5 + 0.5;
        assert!((end - 3377.6).abs() < 1e-9, "{end}");
        // the asl003 sidecar as given (64 lines): 32 lines of 1 ms cannot fit near 12 ms
        let blk64 = grase_block(64, 2, 1.0, false).unwrap();
        let esp64 = esp_from_echo_time(11.92, 1, &blk64);
        let tr64 = EchoTrain { etl: 30, esp_ms: esp64, ..tr.clone() };
        let tab64 = grase_lines(&tr64, &Readout3d::Grase { ky_segments: 2, t_line_ms: 1.0, reverse_phase: false }, 64, 30).unwrap();
        let e = check_grase_timing(&tr64, &tab64, 3000.0, 3500.0).unwrap_err();
        assert!(e.contains("echo spacing must be at least") && e.contains("ky segments"), "{e}");
        // asl005: 64 x 64 x 30, four segments, 0.2048 ms lines, TE 13.28 ms, excitation at 3.8 s
        let blk5 = grase_block(64, 4, 0.2048, false).unwrap();
        assert!((blk5.t_ms[32] + 0.1024).abs() < 1e-12);
        let esp5 = esp_from_echo_time(13.28, 1, &blk5);
        assert!((esp5 - 13.3824).abs() < 1e-12, "{esp5}");
        let tr5 = EchoTrain { etl: 30, esp_ms: esp5, refocusing_deg: 130.0, ..tr.clone() };
        let tab5 = grase_lines(&tr5, &Readout3d::Grase { ky_segments: 4, t_line_ms: 0.2048, reverse_phase: false }, 64, 30).unwrap();
        check_grase_timing(&tr5, &tab5, 3800.0, 4950.0).unwrap();
        let end5 = 3800.0 + 30.0 * esp5 + 7.5 * 0.2048 + 0.1024;
        assert!((end5 - 4203.1104).abs() < 1e-9, "{end5}");
        // a train past the repetition, and a refocusing pulse on the excitation, are refused
        assert!(check_grase_timing(&tr5, &tab5, 4600.0, 4950.0).unwrap_err().contains("after"));
        let tiny = EchoTrain { esp_ms: 1.9, refocusing_time_ms: 2.0, ..tr5.clone() };
        let one = grase_lines(&EchoTrain { etl: 1, kz_segments: 1, ..tiny.clone() },
                              &Readout3d::Grase { ky_segments: 1, t_line_ms: 0.0001, reverse_phase: false }, 1, 1).unwrap();
        assert!(check_grase_timing(&EchoTrain { etl: 1, ..tiny }, &one, 0.0, 1000.0).is_err());
        // indivisible matrices are refused with the counts
        assert!(grase_block(69, 4, 0.2, false).unwrap_err().contains("69"));
    }

    #[test]
    fn spiral_speed_bound_holds_on_the_continuous_trajectory() {
        // the asl001 design and a spread of others: max speed x dwell <= 1 on a fine sampling of
        // tau, and the bound is tight (reached just inside tau_c)
        for &(nx, n, t, frac) in &[(64usize, 8usize, 4.0f64, 0.8f64), (32, 1, 10.0, 0.5), (32, 4, 2.0, 0.95), (128, 16, 6.0, 0.3)] {
            let dmax = t / (std::f64::consts::TAU * (nx as f64 / (2.0 * n as f64)) * (nx as f64 / 2.0));
            let dwell = frac * dmax;
            let tr = match spiral_trajectory(nx, nx, n, t, dwell) {
                Ok(tr) => tr,
                Err(e) => { assert!(e.contains("too short"), "{e}"); continue; }
            };
            let steps = 200_000;
            let mut vmax = 0.0f64;
            for i in 1..=steps {
                let tau = t * i as f64 / steps as f64;
                vmax = vmax.max(tr.speed(tau));
            }
            vmax = vmax.max(tr.speed(tr.tau_c_ms * (1.0 - 1e-12)));
            assert!(vmax * dwell <= 1.0 + 1e-12, "nx {nx} N {n}: {}", vmax * dwell);
            assert!(vmax * dwell >= 1.0 - 1e-6, "the bound is not tight: {}", vmax * dwell);
            // the speed matches a finite difference of the position, inside and outside tau_c
            for &tau in &[0.3 * tr.tau_c_ms, 0.5 * (tr.tau_c_ms + t), 0.99 * t] {
                let h = 1e-7;
                let (a, b) = (tr.at(1, tau - h), tr.at(1, tau + h));
                let fd = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() / (2.0 * h);
                assert!((fd - tr.speed(tau)).abs() <= 1e-5 * fd, "{fd} vs {}", tr.speed(tau));
            }
            // continuous at tau_c; consecutive samples within one cycle/FOV; interleaves are rotations
            let (a, b) = (tr.at(0, tr.tau_c_ms * (1.0 - 1e-13)), tr.at(0, tr.tau_c_ms));
            assert!((a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 1e-9);
            for s in 0..n {
                let il = tr.interleaf(s);
                for w in il.windows(2) {
                    assert!(((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt() <= 1.0 + 1e-12);
                }
                let th = std::f64::consts::TAU * s as f64 / n as f64;
                let p0 = tr.interleaf(0)[7];
                let rot = [p0[0] * th.cos() - p0[1] * th.sin(), p0[0] * th.sin() + p0[1] * th.cos()];
                assert!((rot[0] - il[7][0]).abs() + (rot[1] - il[7][1]).abs() < 1e-12);
            }
            // the last sample stays inside k_max
            let last = tr.interleaf(0)[tr.n_samples() - 1];
            assert!((last[0].hypot(last[1])) <= tr.k_max);
        }
    }

    #[test]
    fn spiral_sampling_bound_rejects_undersampled_designs() {
        // the second review's counterexample: one interleaf, nx = 64, 32 samples over 32 turns,
        // every sample on the x-axis one cycle/FOV apart, which a chord check passes
        let e = spiral_trajectory(64, 64, 1, 3.2, 0.1).unwrap_err();
        assert!(e.contains("dwell time must be below"), "{e}");
        // a dwell one percent above the bound
        let dmax = 4.0 / (std::f64::consts::TAU * 4.0 * 32.0);
        assert!(spiral_trajectory(64, 64, 8, 4.0, 1.01 * dmax).unwrap_err().contains("dwell time must be below"));
        // a dwell just below the bound needs a centre region longer than the readout
        assert!(spiral_trajectory(64, 64, 8, 4.0, 0.9999 * dmax).unwrap_err().contains("too short"));
        // rectangular matrices and nonsense inputs
        assert!(spiral_trajectory(64, 48, 8, 4.0, 0.004).unwrap_err().contains("square"));
        assert!(spiral_trajectory(64, 64, 0, 4.0, 0.004).is_err());
        assert!(spiral_trajectory(64, 64, 8, 4.0, 0.0).is_err());
    }

    #[test]
    fn spiral_asl001_numbers() {
        // asl001 with the committed overlay: 64 x 64 x 20, 8 interleaves, T = 4 ms, dwell 4 us,
        // TE 10.528 ms, excitation at LD + PLD = 3.475 s, TR 4.886 s, refocusing time 2 ms
        let ro = Readout3d::Spiral { interleaves: 8, readout_ms: 4.0, dwell_ms: 0.004 };
        let train = EchoTrain { etl: 20, esp_ms: 0.0, refocusing_deg: 111.0, kz_order: KzOrder::Centric, kz_segments: 1,
                                refocusing_time_ms: 2.0 };
        let tab = spiral_lines(&train, &ro, 64, 64, 20).unwrap();
        assert_eq!(tab.traj.n_samples(), 1000);
        assert_eq!((tab.e_c, tab.n_shots), (1, 8));
        let esp = spiral_esp_from_echo_time(10.528, tab.e_c);
        assert!((esp - 10.528).abs() < 1e-12);
        let train = EchoTrain { esp_ms: esp, ..train };
        let tab = spiral_lines(&train, &ro, 64, 64, 20).unwrap();
        // T + refocusing/2 = 5 <= ESP/2 = 5.264
        let end = check_spiral_timing(&train, &tab, 3475.0, 4886.0).unwrap();
        assert!((end - 3689.56).abs() < 1e-9, "{end}");
        // tau_c from the closed form
        let pi = std::f64::consts::PI;
        let want = 1.0 / (4.0 * (1.0 / (32.0f64 * 0.004).powi(2) - 4.0 * pi * pi * 16.0 / 16.0));
        assert!((tab.traj.tau_c_ms - want).abs() < 1e-15, "{}", tab.traj.tau_c_ms);
        // every partition read once, the centre first; shots s kz_segments + sz
        assert_eq!(tab.echo[10], 1);
        let mut echoes = tab.echo.clone();
        echoes.sort();
        assert_eq!(echoes, (1..=20).collect::<Vec<_>>());
        assert_eq!(tab.shot(3, 5), 5);
        assert!((tab.trf_ms(10, 0) - (10.528 + 0.002)).abs() < 1e-12);
        // Codex's example: an 8 ms spiral at this ESP would run through the next pulse
        let ro8 = Readout3d::Spiral { interleaves: 4, readout_ms: 8.0, dwell_ms: 0.004 };
        let tab8 = spiral_lines(&train, &ro8, 64, 64, 20).unwrap();
        assert!(check_spiral_timing(&train, &tab8, 3475.0, 4886.0).unwrap_err().contains("next refocusing pulse"));
        // two kz segments: shots interleave x segment; past the repetition is refused
        let t2 = EchoTrain { etl: 10, kz_segments: 2, ..train.clone() };
        let tab2 = spiral_lines(&t2, &ro, 64, 64, 20).unwrap();
        assert_eq!(tab2.n_shots, 16);
        assert_eq!(tab2.shot(10, 3), 3 * 2 + tab2.kz_segment[10]);
        assert!(check_spiral_timing(&train, &tab, 4700.0, 4886.0).unwrap_err().contains("after"));
        // the GRASE and spiral tables refuse each other's readouts
        assert!(grase_lines(&train, &ro, 64, 20).is_err());
        assert!(spiral_lines(&train, &Readout3d::Grase { ky_segments: 1, t_line_ms: 0.5, reverse_phase: false }, 64, 64, 20).is_err());
    }
}
