//! Binary interchange format for the Python <-> Rust correctness comparison.
//!
//! The point of this format is to make the comparison fair: the Rust core must
//! not re-draw connectivity with its own RNG. Python exports the exact targets,
//! delays, weights, initial state and drive it used, and Rust loads them, so any
//! divergence in the spike output is attributable to the dynamics and not to a
//! different random graph.
//!
//! Layout (all little-endian, header fields 4 bytes each):
//!
//! ```text
//! magic         8 bytes  b"RBRAIN01"
//! n             u32
//! k             u32
//! max_delay     u32
//! steps         u32
//! dend_mode     u32   (0 dcaap, 1 linear, 2 none)
//! has_drive     u32
//! 13 x f32      tau_soma, tau_dend, tau_adapt, e_leak, e_dend, v_reset,
//!               v_thresh, adapt_base, adapt_inc, dend_scale, dend_gain,
//!               refractory_ms, noise_std
//! dt            f32
//! arrays        targets u32[n*k], delays u32[n*k], weights f32[n*k],
//!               v_soma f32[n], v_dend f32[n], adapt f32[n], refrac f32[n],
//!               [drive f32[n]],
//!               exp_neuron_spikes u32[n], exp_step_spikes u32[steps]
//! ```

use crate::sim::{Params, Sim, Synapses, DEND_DCAAP, DEND_LINEAR, DEND_NONE};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"RBRAIN01";

pub struct Fixture {
    pub params: Params,
    pub syn: Synapses,
    pub v_soma0: Vec<f32>,
    pub v_dend0: Vec<f32>,
    pub adapt0: Vec<f32>,
    pub refrac0: Vec<f32>,
    pub drive: Option<Vec<f32>>,
    pub steps: u32,
    pub exp_neuron_spikes: Vec<u32>,
    pub exp_step_spikes: Vec<u32>,
}

fn read_u32(r: &mut impl Read, count: usize) -> std::io::Result<Vec<u32>> {
    let mut buf = vec![0u8; count * 4];
    r.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_f32(r: &mut impl Read, count: usize) -> std::io::Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    r.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_f32_one(r: &mut impl Read) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_u32_one(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn load_fixture(path: &Path) -> std::io::Result<Fixture> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad magic {magic:?}, expected {MAGIC:?}"),
        ));
    }

    let n = read_u32_one(&mut r)?;
    let k = read_u32_one(&mut r)?;
    let max_delay = read_u32_one(&mut r)?;
    let steps = read_u32_one(&mut r)?;
    let dend_mode_raw = read_u32_one(&mut r)?;
    let has_drive = read_u32_one(&mut r)?;

    let tau_soma = read_f32_one(&mut r)?;
    let tau_dend = read_f32_one(&mut r)?;
    let tau_adapt = read_f32_one(&mut r)?;
    let e_leak = read_f32_one(&mut r)?;
    let e_dend = read_f32_one(&mut r)?;
    let v_reset = read_f32_one(&mut r)?;
    let v_thresh = read_f32_one(&mut r)?;
    let adapt_base = read_f32_one(&mut r)?;
    let adapt_inc = read_f32_one(&mut r)?;
    let dend_scale = read_f32_one(&mut r)?;
    let dend_gain = read_f32_one(&mut r)?;
    let refractory_ms = read_f32_one(&mut r)?;
    let noise_std = read_f32_one(&mut r)?;
    let dt = read_f32_one(&mut r)?;

    let dend_mode = match dend_mode_raw {
        x if x == DEND_DCAAP => DEND_DCAAP,
        x if x == DEND_LINEAR => DEND_LINEAR,
        x if x == DEND_NONE => DEND_NONE,
        _ => DEND_DCAAP,
    };

    let mut params = Params::new(n, k);
    params.max_delay = max_delay;
    params.dend_mode = dend_mode;
    params.tau_soma = tau_soma;
    params.tau_dend = tau_dend;
    params.tau_adapt = tau_adapt;
    params.e_leak = e_leak;
    params.e_dend = e_dend;
    params.v_reset = v_reset;
    params.v_thresh = v_thresh;
    params.adapt_base = adapt_base;
    params.adapt_inc = adapt_inc;
    params.dend_scale = dend_scale;
    params.dend_gain = dend_gain;
    params.refractory_ms = refractory_ms;
    params.noise_std = noise_std;
    params.dt = dt;

    let ns = (n * k) as usize;
    let nn = n as usize;

    let syn = Synapses {
        targets: read_u32(&mut r, ns)?,
        delays: read_u32(&mut r, ns)?,
        weights: read_f32(&mut r, ns)?,
    };
    let v_soma0 = read_f32(&mut r, nn)?;
    let v_dend0 = read_f32(&mut r, nn)?;
    let adapt0 = read_f32(&mut r, nn)?;
    let refrac0 = read_f32(&mut r, nn)?;
    let drive = if has_drive != 0 {
        Some(read_f32(&mut r, nn)?)
    } else {
        None
    };
    let exp_neuron_spikes = read_u32(&mut r, nn)?;
    let exp_step_spikes = read_u32(&mut r, steps as usize)?;

    for (i, t) in syn.targets.iter().enumerate() {
        if *t >= n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("target {t} at slot {i} out of range for n={n}"),
            ));
        }
    }
    for (i, d) in syn.delays.iter().enumerate() {
        if *d < 1 || *d > max_delay {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "delay {d} at slot {i} outside [1, {max_delay}]. The Python \
                     exporter must declare max_delay as the largest delay it \
                     actually produced; see syn_max_delay() in \
                     python/export_fixture.py."
                ),
            ));
        }
    }

    Ok(Fixture {
        params,
        syn,
        v_soma0,
        v_dend0,
        adapt0,
        refrac0,
        drive,
        steps,
        exp_neuron_spikes,
        exp_step_spikes,
    })
}

pub fn write_fixture(
    path: &Path,
    params: &Params,
    syn: &Synapses,
    v_soma0: &[f32],
    v_dend0: &[f32],
    adapt0: &[f32],
    refrac0: &[f32],
    drive: Option<&[f32]>,
    exp_neuron_spikes: &[u32],
    exp_step_spikes: &[u32],
) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    w.write_all(&params.n.to_le_bytes())?;
    w.write_all(&params.k_out.to_le_bytes())?;
    w.write_all(&params.max_delay.to_le_bytes())?;
    w.write_all(&(exp_step_spikes.len() as u32).to_le_bytes())?;
    w.write_all(&params.dend_mode.to_le_bytes())?;
    w.write_all(&(drive.is_some() as u32).to_le_bytes())?;
    for v in [
        params.tau_soma, params.tau_dend, params.tau_adapt, params.e_leak, params.e_dend,
        params.v_reset, params.v_thresh, params.adapt_base, params.adapt_inc, params.dend_scale,
        params.dend_gain, params.refractory_ms, params.noise_std, params.dt,
    ] {
        w.write_all(&v.to_le_bytes())?;
    }
    for v in &syn.targets {
        w.write_all(&v.to_le_bytes())?;
    }
    for v in &syn.delays {
        w.write_all(&v.to_le_bytes())?;
    }
    for v in &syn.weights {
        w.write_all(&v.to_le_bytes())?;
    }
    for arr in [v_soma0, v_dend0, adapt0, refrac0] {
        for v in arr {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    if let Some(d) = drive {
        for v in d {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    for v in exp_neuron_spikes {
        w.write_all(&v.to_le_bytes())?;
    }
    for v in exp_step_spikes {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()
}

/// Build a `Sim` whose state matches the fixture exactly, then run it.
///
/// `zero_weights` is the negative control: it destroys the delivery path while
/// leaving everything else identical. If the comparison still reported a match
/// with zeroed weights, the comparison would have no power to detect a broken
/// delivery path, and an exact match would be vacuous.
pub fn run_fixture_ctl(fx: &Fixture, mode: crate::sim::Mode, steps: u32, zero_weights: bool) -> Sim {
    let mut sim = Sim::new(fx.params.clone(), fx.syn_copy_zeroed(zero_weights));
    sim.v_soma.copy_from_slice(&fx.v_soma0);
    sim.v_dend.copy_from_slice(&fx.v_dend0);
    sim.adapt.copy_from_slice(&fx.adapt0);
    sim.refrac.copy_from_slice(&fx.refrac0);
    let drive = match &fx.drive {
        Some(d) => crate::sim::Drive::Dense(d.clone()),
        None => crate::sim::Drive::None,
    };
    let cap = ((sim.p.n as f32 * 0.1) as usize).max(1);
    for _ in 0..steps {
        sim.step(&drive, mode, false, Some(cap));
    }
    sim
}

/// Run the fixture normally.
pub fn run_fixture(fx: &Fixture, mode: crate::sim::Mode, steps: u32) -> Sim {
    run_fixture_ctl(fx, mode, steps, false)
}

impl Fixture {
    pub fn syn_copy(&self) -> Synapses {
        self.syn_copy_zeroed(false)
    }

    pub fn syn_copy_zeroed(&self, zero: bool) -> Synapses {
        Synapses {
            targets: self.syn.targets.clone(),
            weights: if zero {
                vec![0.0; self.syn.weights.len()]
            } else {
                self.syn.weights.clone()
            },
            delays: self.syn.delays.clone(),
        }
    }
}
