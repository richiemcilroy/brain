//! Synthetic network construction for the Rust-only benchmarks.
//!
//! Mirrors `brain/connectivity.py::Synapses.__init__` semantics: fixed fan-out
//! `k_out` per neuron, Dale's principle (excitatory neurons own positive
//! weights, inhibitory neurons negative), |N(1, w_noise)| weight magnitude
//! scaling, and integer delays drawn from [delay_min, delay_max].
//!
//! The RNG differs from MLX's, so this graph is NOT the same graph the Python
//! path would build for a given seed. That is deliberate and harmless here:
//! these runs measure throughput, not learned behaviour. The correctness
//! comparison uses the Python-exported graph instead (see `io.rs`).

use crate::rng::Rng;
use crate::sim::{Params, Synapses};

pub struct BuildCfg {
    pub n: u32,
    pub k_out: u32,
    pub excitatory_frac: f32,
    pub w_exc: f32,
    pub w_inh: f32,
    pub w_noise: f32,
    pub delay_min: u32,
    pub delay_max: u32,
    pub seed: u64,
    /// Fraction of neurons that receive tonic drive and are therefore awake.
    pub driven_frac: f32,
    pub drive_low: f32,
    pub drive_high: f32,
}

impl BuildCfg {
    pub fn new(n: u32, k_out: u32, seed: u64) -> Self {
        BuildCfg {
            n,
            k_out,
            excitatory_frac: 0.8,
            w_exc: 0.08,
            w_inh: 0.32,
            w_noise: 0.5,
            delay_min: 1,
            delay_max: 4,
            seed,
            driven_frac: 0.01,
            drive_low: 1.2,
            drive_high: 1.6,
        }
    }
}

impl Synapses {
    pub fn build(cfg: &BuildCfg) -> Synapses {
        let n = cfg.n as usize;
        let k = cfg.k_out as usize;
        let mut rng = Rng::new(cfg.seed);
        let n_exc = (n as f64 * cfg.excitatory_frac as f64).round() as usize;

        let mut targets = vec![0u32; n * k];
        let mut weights = vec![0f32; n * k];
        let mut delays = vec![0u32; n * k];

        for pre in 0..n {
            let exc = pre < n_exc;
            for j in 0..k {
                let slot = pre * k + j;
                targets[slot] = rng.below(cfg.n as u64) as u32;
                // |N(0,1)| magnitude scaling, matching base * (1 + w_noise*mag)
                let mag = rng.normal().abs() as f32;
                let base = if exc { cfg.w_exc } else { cfg.w_inh };
                let w = base * (1.0 + cfg.w_noise * mag);
                weights[slot] = if exc { w } else { -w };
                let span = cfg.delay_max - cfg.delay_min + 1;
                delays[slot] = cfg.delay_min + rng.below(span as u64) as u32;
            }
        }
        Synapses { targets, weights, delays }
    }
}

/// Tonic drive applied to a random `driven_frac` subset of neurons.
///
/// This is the configuration that makes the sparsity claim testable: only the
/// driven neurons are awake, everything else sits at rest and (with
/// `noise_std == 0`) provably does nothing.
pub fn build_sparse_drive(cfg: &BuildCfg, p: &Params, seed: u64) -> Vec<(u32, f32)> {
    let mut rng = Rng::new(seed ^ 0xA5A5_1234);
    let n_driven = ((cfg.driven_frac as f64) * cfg.n as f64).round() as usize;
    let mut v: Vec<(u32, f32)> = Vec::with_capacity(n_driven);
    for _ in 0..n_driven {
        let u = rng.below(cfg.n as u64) as u32;
        let amp = cfg.drive_low + (cfg.drive_high - cfg.drive_low) * rng.f64() as f32;
        let _ = p;
        v.push((u, amp));
    }
    v.sort_by_key(|e| e.0);
    v.dedup_by_key(|e| e.0);
    v
}

/// Dense drive vector, matching the repo's `forced_activity` protocol shape
/// (a value for every neuron, mostly within U(drive_low, drive_high)).
pub fn build_dense_drive(cfg: &BuildCfg, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed ^ 0x5A5A_9876);
    (0..cfg.n)
        .map(|_| cfg.drive_low + (cfg.drive_high - cfg.drive_low) * rng.f64() as f32)
        .collect()
}

/// Dense drive in which only a  subset receives current; the rest
/// are exactly zero. This is the shape the Python benchmark uses: a full-length
/// vector so the interface is uniform, but genuinely sparse input.
pub fn build_sparse_within_dense(cfg: &BuildCfg, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed ^ 0x1234_ABCD);
    let mut v = vec![0.0f32; cfg.n as usize];
    let n_driven = ((cfg.driven_frac as f64) * cfg.n as f64).round() as usize;
    for _ in 0..n_driven {
        let u = rng.below(cfg.n as u64) as usize;
        v[u] = cfg.drive_low + (cfg.drive_high - cfg.drive_low) * rng.f64() as f32;
    }
    v
}
