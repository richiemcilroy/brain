//! Event-driven spiking core: CLI entry point.
//!
//! Subcommands:
//!   sanity            biophysical invariants (refractory, fixed point, kinetics)
//!   bench-fraction    wall-clock vs active fraction (the headline measurement)
//!   bench-scale       throughput and RAM at large N
//!   compare           correctness vs a Python-exported fixture
//!   export-fixture    (debug) inspect a fixture header

mod build;
mod io;
mod rng;
mod sim;

use build::{build_sparse_drive, build_sparse_within_dense, BuildCfg};
use sim::{Drive, Mode, Params, Sim, Synapses};
use std::time::Instant;

fn rss_bytes() -> u64 {
    // getrusage(RUSAGE_SELF).ru_maxrss is bytes on macOS, kilobytes on Linux.
    #[cfg(target_os = "macos")]
    unsafe {
        let mut ru: libc_rusage = std::mem::zeroed();
        if getrusage(0, &mut ru) == 0 {
            return ru.ru_maxrss as u64;
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

#[repr(C)]
#[allow(non_camel_case_types)]
struct libc_rusage {
    ru_utime_sec: i64,
    ru_utime_usec: i64,
    ru_stime_sec: i64,
    ru_stime_usec: i64,
    ru_maxrss: i64,
    ru_ixrss: i64,
    ru_idrss: i64,
    ru_isrss: i64,
    ru_minflt: i64,
    ru_majflt: i64,
    ru_nswap: i64,
    ru_inblock: i64,
    ru_oublock: i64,
    ru_msgsnd: i64,
    ru_msgrcv: i64,
    ru_nsignals: i64,
    ru_nvcsw: i64,
    ru_nivcsw: i64,
}

extern "C" {
    fn getrusage(who: i32, usage: *mut libc_rusage) -> i32;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "sanity" => sanity(),
        "bench-fraction" => bench_fraction(),
        "bench-scale" => bench_scale(),
        "bench-fixture" => bench_fixture(&args),
        "compare" => compare(&args),
        "help" | _ => {
            eprintln!(
                "usage: rust_core <sanity|bench-fraction|bench-scale|bench-fixture FIXTURE|compare FIXTURE>"
            );
            std::process::exit(if cmd == "help" { 0 } else { 2 });
        }
    }
}

/// Biophysical and structural invariants that must hold for the core to be a
/// faithful port of the Python dynamics.
fn sanity() {
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        if ok {
            pass += 1;
            println!("  PASS  {name}  {detail}");
        } else {
            fail += 1;
            println!("  FAIL  {name}  {detail}");
        }
    };

    // --- 1. refractory: no interspike interval below refractory_ms ---------
    {
        let mut p = Params::new(4, 4);
        p.noise_std = 0.0;
        let syn = Synapses {
            targets: vec![0; 16],
            weights: vec![0.0; 16],
            delays: vec![1; 16],
        };
        let mut s = Sim::new(p, syn);
        let drive = Drive::Dense(vec![2.0; 4]);
        let mut times: Vec<Vec<u32>> = vec![Vec::new(); 4];
        for t in 0..400u32 {
            s.step(&drive, Mode::EventDriven, false, Some(8));
            let idx = s.spike_idx_snapshot();
            for u in idx {
                times[u as usize].push(t);
            }
        }
        let mut min_gap = u32::MAX;
        let mut nseq = 0;
        for t in times.iter() {
            for w in t.windows(2) {
                min_gap = min_gap.min(w[1] - w[0]);
                nseq += 1;
            }
        }
        check(
            "refractory_min_isi",
            min_gap >= 2 && nseq > 0,
            format!("min inter-spike interval = {min_gap} ms over {nseq} intervals, refractory_ms = 2.0"),
        );
    }

    // --- 2. rest state is a fixed point (the event-driven premise) --------
    {
        let mut p = Params::new(1000, 4);
        p.noise_std = 0.0;
        let cfg = BuildCfg::new(1000, 4, 7);
        let syn = Synapses::build(&cfg);
        let mut s = Sim::new(p, syn);
        let drive = Drive::None;
        let mut fired = 0u64;
        let mut updates = 0u64;
        for _ in 0..200 {
            let st = s.step(&drive, Mode::EventDriven, false, None);
            fired += st.spikes;
            updates += st.neuron_updates;
        }
        let vs_max = s.v_soma.iter().cloned().fold(f32::MIN, f32::max);
        check(
            "rest_fixed_point_undriven",
            fired == 0 && updates == 0 && vs_max == 0.0,
            format!("undriven: {fired} spikes, {updates} neuron-updates, max v_soma = {vs_max} (all must be 0)"),
        );
    }

    // --- 3. refractory gates a suprathreshold soma ------------------------
    {
        let mut p = Params::new(1, 1);
        p.noise_std = 0.0;
        let syn = Synapses { targets: vec![0], weights: vec![0.0], delays: vec![1] };
        let mut s = Sim::new(p, syn);
        s.v_soma[0] = 5.0;
        s.refrac[0] = 2.0;
        s.request_reseed();
        let st = s.step(&Drive::None, Mode::EventDriven, false, Some(4));
        let blocked_ok = st.spikes == 0;
        s.refrac[0] = 0.0;
        s.v_soma[0] = 5.0;
        s.request_reseed();
        let st2 = s.step(&Drive::None, Mode::EventDriven, false, Some(4));
        check(
            "refractory_gates_suprathreshold",
            blocked_ok && st2.spikes == 1,
            format!("blocked={} spikes, released={} spikes (expect 0 then 1)", st.spikes, st2.spikes),
        );
    }

    // --- 4. dCaAP is non-monotonic, peaks at dend_scale -------------------
    {
        let p = Params::new(1, 1);
        let mut peak_x = 0f32;
        let mut peak_y = f32::MIN;
        let mut vals = Vec::new();
        let mut x = 0.0f32;
        while x <= 4.0 {
            let y = sim::dend_activation(&p, x);
            vals.push((x, y));
            if y > peak_y {
                peak_y = y;
                peak_x = x;
            }
            x += 0.01;
        }
        let tail_decreasing = vals
            .windows(2)
            .filter(|w| w[0].0 >= peak_x)
            .all(|w| w[1].1 < w[0].1);
        let far = sim::dend_activation(&p, 4.0 * p.dend_scale);
        check(
            "dcaap_non_monotonic",
            (peak_x - 1.0).abs() < 0.02 && (peak_y - 1.0).abs() < 1e-3 && tail_decreasing
                && far < 0.25,
            format!("peak at v_dend={peak_x:.2} value={peak_y:.4}; phi(4*s)={far:.4}; tail strictly decreasing={tail_decreasing}"),
        );
    }

    // --- 5. delay ring delivers at exactly t + delay ----------------------
    {
        let mut p = Params::new(8, 1);
        p.max_delay = 4;
        p.noise_std = 0.0;
        let mut syn = Synapses { targets: vec![3], weights: vec![0.5], delays: vec![2] };
        syn.targets[0] = 3;
        syn.delays[0] = 2;
        let mut s = Sim::new(p, syn);
        // Deliver via a single pre-neuron spike scheduled by hand.
        s.force_schedule(0, 0);
        let mut arrivals_at = Vec::new();
        for t in 0..7u64 {
            let v = s.probe_current(3, t);
            if v != 0.0 {
                arrivals_at.push((t, v));
            }
        }
        check(
            "delay_ring_t_plus_delay",
            arrivals_at.len() == 1 && arrivals_at[0].0 == 2 && (arrivals_at[0].1 - 0.5).abs() < 1e-6,
            format!("arrivals={arrivals_at:?} (expect exactly [(2, 0.5)])"),
        );
    }

    // --- 6. soma leak follows tau_soma, and sleeping loses nothing ---------
    {
        let mut p = Params::new(1, 1);
        p.noise_std = 0.0;
        p.dend_mode = sim::DEND_NONE;
        let syn = Synapses { targets: vec![0], weights: vec![0.0], delays: vec![1] };
        let mut s = Sim::new(p.clone(), syn);
        s.v_soma[0] = 1.0;
        s.request_reseed();
        let steps = 20u32;
        for _ in 0..steps {
            s.step(&Drive::None, Mode::EventDriven, false, Some(1));
        }
        let tau = 20.0f32;
        let expected = (1.0 - 1.0 / tau).powi(steps as i32);
        let got = s.v_soma[0];
        let rel = ((got - expected) / expected).abs();
        check(
            "soma_leak_tau",
            rel < 1e-5,
            format!("v after {steps} ms = {got:.8}, Euler prediction = {expected:.8}, rel err {rel:.2e}"),
        );
        // The neuron decays toward rest but is NOT at rest, so it must still be
        // awake. Only an exact fixed point may sleep, precisely because sleeping
        // requires no catch-up arithmetic.
        check(
            "non_rest_neuron_stays_awake",
            !s.asleep_snapshot()[0],
            format!("v_soma = {got:.6} (not exactly e_leak), so the neuron must not be asleep"),
        );
        // Drive it to exact rest and confirm it then sleeps.
        s.v_soma[0] = p.e_leak;
        s.adapt[0] = 0.0;
        s.refrac[0] = 0.0;
        s.v_dend[0] = p.e_dend;
        s.request_reseed();
        s.step(&Drive::None, Mode::EventDriven, false, None);
        check(
            "exact_rest_neuron_sleeps",
            s.asleep_snapshot()[0] && s.v_soma[0] == p.e_leak,
            format!("at exact rest after one undriven step: asleep = {}, v_soma = {}", s.asleep_snapshot()[0], s.v_soma[0]),
        );
    }

    // --- 7. event-driven vs dense give identical spikes at noise_std = 0 ---
    {
        let mut p = Params::new(3000, 8);
        p.noise_std = 0.0;
        let cfg = {
            let mut c = BuildCfg::new(3000, 8, 11);
            c.driven_frac = 0.05;
            c
        };
        let syn = Synapses::build(&cfg);
        let drive_sparse = build_sparse_drive(&cfg, &p, 11);
        let drive_dense: Vec<f32> = {
            let mut v = vec![0.0f32; 3000];
            for &(u, a) in drive_sparse.iter() {
                v[u as usize] = a;
            }
            v
        };

        let mut a = Sim::new(p.clone(), syn_copy(&syn));
        let mut b = Sim::new(p.clone(), syn_copy(&syn));
        let da = Drive::Sparse(drive_sparse.clone());
        let db = Drive::Dense(drive_dense.clone());
        let mut sa = Vec::new();
        let mut sb = Vec::new();
        let mut ua = 0u64;
        let mut ub = 0u64;
        for _ in 0..300 {
            let x = a.step(&da, Mode::EventDriven, false, None);
            let y = b.step(&db, Mode::Dense, false, None);
            sa.push(x.spikes);
            sb.push(y.spikes);
            ua += x.neuron_updates;
            ub += y.neuron_updates;
        }
        let mismatch = sa.iter().zip(sb.iter()).filter(|(x, y)| x != y).count();
        check(
            "event_driven_matches_dense_exactly",
            mismatch == 0,
            format!("{mismatch}/300 steps differ; updates event-driven={ua} dense={ub} ({:.2}% of dense)",
                    ua as f64 / ub as f64 * 100.0),
        );
    }

    // --- 7b. event-driven == dense with a DENSE partially-nonzero drive -----
    // Regression test for a real divergence: with a `Drive::Dense` vector where
    // half the entries are nonzero AND recurrent weights are strong enough to
    // matter, event-driven fired ~2.4x fewer spikes than dense while still
    // integrating 99.9% of neurons. The dense path was verified correct first
    // (it matched the Python simulator on all 60 measured steps), so the defect
    // was in the sleep/wake path, not the dynamics.
    {
        let mut p = Params::new(4000, 64);
        p.noise_std = 0.0;
        let mut cfg = BuildCfg::new(4000, 64, 5);
        cfg.w_exc = 0.40;
        cfg.w_inh = 1.60;
        cfg.driven_frac = 0.5;
        cfg.drive_low = 5.0;
        cfg.drive_high = 5.0;
        let syn = Synapses::build(&cfg);
        let drive = Drive::Dense(build_sparse_within_dense(&cfg, 5));

        let mut ev = Sim::new(p.clone(), syn_copy(&syn));
        let mut dn = Sim::new(p.clone(), syn_copy(&syn));
        ev.request_reseed();
        dn.request_reseed();
        let mut first_diff = None;
        let mut ev_tot = 0u64;
        let mut dn_tot = 0u64;
        for step in 0..40u32 {
            let a = ev.step(&drive, Mode::EventDriven, false, None);
            let b = dn.step(&drive, Mode::Dense, false, None);
            ev_tot += a.spikes;
            dn_tot += b.spikes;
            if a.spikes != b.spikes && first_diff.is_none() {
                first_diff = Some((step, a.spikes, b.spikes));
            }
        }
        // Report WHERE state diverges, not just that spikes differ.
        if first_diff.is_some() {
            let mut maxdv = 0.0f32;
            let mut arg = 0usize;
            for i in 0..ev.v_soma.len() {
                let d = (ev.v_soma[i] - dn.v_soma[i]).abs();
                if d > maxdv { maxdv = d; arg = i; }
            }
            println!("        state divergence: max |v_soma diff| = {maxdv:.6e} at index {arg}");
            println!("          event  v_soma[{arg}] = {:.8} adapt = {:.8} refrac = {:.8}",
                     ev.v_soma[arg], ev.adapt[arg], ev.refrac[arg]);
            println!("          dense  v_soma[{arg}] = {:.8} adapt = {:.8} refrac = {:.8}",
                     dn.v_soma[arg], dn.adapt[arg], dn.refrac[arg]);
            println!("          event awake={} asleep={} | dense awake={} asleep={}",
                     ev.asleep_snapshot().iter().filter(|x| !**x).count(),
                     ev.n_asleep(),
                     dn.asleep_snapshot().iter().filter(|x| !**x).count(),
                     dn.n_asleep());
        }
        check(
            "event_driven_matches_dense_dense_drive",
            first_diff.is_none(),
            match first_diff {
                None => format!("40/40 steps identical ({ev_tot} spikes both)"),
                Some((st, a, b)) => format!(
                    "DIVERGED at step {st}: event={a} dense={b}; totals event={ev_tot} dense={dn_tot}"
                ),
            },
        );
    }

    // --- 8. nonzero noise => event-driven must NOT skip resting neurons ----
    {
        let mut p = Params::new(500, 4);
        p.noise_std = 0.05;
        let cfg = BuildCfg::new(500, 4, 3);
        let syn = Synapses::build(&cfg);
        let mut s = Sim::new(p, syn);
        s.v_soma[0] = 0.0;
        let mut updates = 0u64;
        for _ in 0..50 {
            updates += s.step(&Drive::None, Mode::EventDriven, false, None).neuron_updates;
        }
        check(
            "noise_forces_dense_integration",
            updates > 0,
            format!("with noise_std>0 an undriven network still performed {updates} neuron-updates (must be >0: noise wakes every neuron)"),
        );
    }

    println!("\n{pass} passed, {fail} failed");
    if fail > 0 {
        std::process::exit(1);
    }
}


/// Matched-configuration benchmark against a Python-exported fixture.
///
/// Runs THREE configurations on the SAME network and initial conditions:
///
///   dense           every neuron integrated every step, true-sparse delivery.
///                   Same algorithm as Python, Rust implementation.
///   python_shaped   every neuron integrated every step, PLUS the padded
///                   `capacity * k_out` gather Python performs. Same algorithm
///                   AND same work shape as Python; the gap to `dense` is what
///                   the padding defect costs.
///   event_driven    only neurons that can change are integrated, true-sparse
///                   delivery. The algorithmic change under test.
///
/// Reading the three together separates two effects that are easy to conflate:
/// rust-dense vs python isolates IMPLEMENTATION speed; rust-event-driven vs
/// rust-dense isolates the ALGORITHMIC gain from skipping resting neurons.
///
/// JSON on stdout so the Python driver can consume it.
fn bench_fixture(args: &[String]) {
    let path = match args.get(2) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: rust_core bench-fixture FIXTURE [--steps N] [--warm N]");
            std::process::exit(2);
        }
    };
    let mut steps_override: Option<u32> = None;
    let mut warm: u32 = 20;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                steps_override = args.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            "--warm" => {
                warm = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(20);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let fx = match io::load_fixture(std::path::Path::new(&path)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };
    let steps = steps_override.unwrap_or(fx.steps);
    let n = fx.params.n as usize;
    let k = fx.params.k_out as u64;
    let capacity = ((n as f32 * 0.10) as usize).max(1);

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"n_neurons\": {n},\n"));
    out.push_str(&format!("  \"k_out\": {k},\n"));
    out.push_str(&format!("  \"n_synapses\": {},\n", n as u64 * k));
    out.push_str(&format!("  \"steps\": {steps},\n"));
    out.push_str(&format!("  \"warm\": {warm},\n"));
    out.push_str(&format!("  \"padded_capacity\": {capacity},\n"));

    // Each configuration gets a FRESH sim from the fixture's initial state, so
    // none of them inherits state from another.
    let mut sections: Vec<String> = Vec::new();
    // `spike_cap` is set to Python's buffer capacity so that ALL configurations
    // reproduce Python's truncation semantics and can be compared with it. The
    // `event_driven_untruncated` row shows what the core would do without that
    // artifact: it delivers every spike, which is strictly more information.
    let spike_cap = Some(capacity);
    for (name, mode, padded, cap) in [
        ("event_driven", Mode::EventDriven, false, spike_cap),
        ("dense", Mode::Dense, false, spike_cap),
        ("python_shaped", Mode::Dense, true, spike_cap),
        ("event_driven_untruncated", Mode::EventDriven, false, None),
    ] {
        let mut sim = io::run_fixture_ctl(&fx, mode, 0, false);
        sim.request_reseed();
        let drive = match &fx.drive {
            Some(d) => Drive::Dense(d.clone()),
            None => Drive::None,
        };
        for _ in 0..warm {
            sim.step(&drive, mode, padded, cap);
        }
        sim.total_spikes = 0;
        sim.total_neuron_updates = 0;
        sim.total_synops_active = 0;
        sim.total_synops_executed = 0;
        sim.step_spikes.clear();
        // The warmup exists to clear transients; its spikes must not be counted
        // as if they belonged to the measured window.
        for c in sim.spike_count.iter_mut() {
            *c = 0;
        }
        let mut per_step: Vec<f64> = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            let t0 = Instant::now();
            sim.step(&drive, mode, padded, cap);
            per_step.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let wall: f64 = per_step.iter().sum::<f64>() / 1e3;
        let mut sorted = per_step.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let mean = per_step.iter().sum::<f64>() / per_step.len() as f64;
        let p10 = sorted[(0.10 * (sorted.len() - 1) as f64) as usize];
        let p90 = sorted[(0.90 * (sorted.len() - 1) as f64) as usize];
        // MINIMUM is the load-robust statistic. This machine is shared and the
        // measured load average is often >100 on 16 cores, so the median mostly
        // reflects contention by other processes. The fastest observed step is
        // the closest available estimate of the uncontended cost, and is the
        // statistic used for the BENCH.md tables.
        let min_ms = sorted[0];
        let spikes = sim.total_spikes;
        let nspike = spikes as f64 / steps as f64;

        // Trajectory agreement with Python for THIS configuration.
        let cmp_len = (steps as usize).min(fx.exp_step_spikes.len()).min(sim.step_spikes.len());
        let matches = (0..cmp_len)
            .filter(|&j| fx.exp_step_spikes[j] == sim.step_spikes[j])
            .count();

        let series = sim
            .step_spikes
            .iter()
            .take(12)
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        sections.push(format!(
            "  \"{name}\": {{\n    \"first12\": [{series}],\n    \"ms_per_step_min\": {min_ms:.6},\n    \"ms_per_step_median\": {median:.6},\n    \"ms_per_step_mean\": {mean:.6},\n    \"ms_per_step_p10\": {p10:.6},\n    \"ms_per_step_p90\": {p90:.6},\n    \"wall_s\": {wall:.6},\n    \"steps\": {steps},\n    \"spikes_total\": {spikes},\n    \"spikes_per_step\": {nspike:.4},\n    \"active_frac\": {:.8},\n    \"bio_ms_per_wall_s\": {:.3},\n    \"synops_active_per_s\": {:.6e},\n    \"synops_executed_per_s\": {:.6e},\n    \"neuron_updates_per_step\": {:.2},\n    \"state_bytes\": {},\n    \"connectivity_bytes\": {},\n    \"trajectory_steps_matching_python\": {matches},\n    \"trajectory_steps_compared\": {cmp_len}\n  }}",
            nspike / n as f64,
            if min_ms > 0.0 { 1000.0 / min_ms } else { 0.0 },
            sim.total_synops_active as f64 / wall,
            sim.total_synops_executed as f64 / wall,
            sim.total_neuron_updates as f64 / steps as f64,
            sim.state_bytes(),
            sim.syn.bytes(),
        ));
        let _ = p10;
    }
    out.push_str(&sections.join(",\n"));
    out.push_str(&format!(",\n  \"rss_bytes\": {}\n}}\n", rss_bytes()));
    print!("{out}");
}

#[allow(dead_code)]
fn syn_copy(s: &Synapses) -> Synapses {
    Synapses {
        targets: s.targets.clone(),
        weights: s.weights.clone(),
        delays: s.delays.clone(),
    }
}

fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Headline measurement: does wall-clock cost track ACTIVITY or track N?
///
/// Both engines are run at the same N and k, at three or more driven fractions
/// spanning ~0.1% to ~10% of neurons receiving input, plus an undriven control.
/// Reported: neuron-updates/step (the algorithmic claim), ms/step (wall clock),
/// bio-ms per wall-second, and the row count the delivery loop touches.
fn bench_fraction() {
    println!("# Event-driven vs dense cost vs active fraction");
    println!("# noise_std=0 (the exactness condition for skipping resting neurons)");
    println!();
    println!(
        "{:>10} {:>8} {:>10} {:>12} {:>12} {:>12} {:>13} {:>18}",
        "n", "k_out", "driven%", "active%", "awake/step", "ms/step", "bio-ms/wall-s", "synops_active/s"
    );

    for &n in &[100_000u32, 1_000_000u32] {
        let k = 64u32;
        for &frac in &[0.0f32, 0.001, 0.01, 0.05, 0.10] {
            let mut p = Params::new(n, k);
            p.noise_std = 0.0;
            let mut cfg = BuildCfg::new(n, k, 12345);
            cfg.driven_frac = frac;
            let syn = Synapses::build(&cfg);
            let mut sim = Sim::new(p.clone(), syn);
            let drive = Drive::Sparse(build_sparse_drive(&cfg, &p, 12345));

            let warm = 20;
            let steps = 200u32;
            for _ in 0..warm {
                sim.step(&drive, Mode::EventDriven, false, Some((n / 10) as usize));
            }
            sim.total_spikes = 0;
            sim.total_neuron_updates = 0;
            sim.total_synops_active = 0;
            let t0 = Instant::now();
            for _ in 0..steps {
                sim.step(&drive, Mode::EventDriven, false, Some((n / 10) as usize));
            }
            let dt = t0.elapsed().as_secs_f64();
            let ms = dt / steps as f64 * 1e3;
            let spk = sim.total_spikes as f64 / steps as f64;
            let awake = sim.total_neuron_updates as f64 / steps as f64;
            println!(
                "{:>10} {:>8} {:>10.2} {:>12.5} {:>12.1} {:>12.4} {:>13.1} {:>18.3e}",
                n,
                k,
                frac * 100.0,
                spk / n as f64 * 100.0,
                awake,
                ms,
                if ms > 0.0 { 1000.0 / ms } else { 0.0 },
                sim.total_synops_active as f64 / dt
            );
        }
        println!();
    }
}

/// Throughput and memory at scale, including a 1M-neuron configuration.
fn bench_scale() {
    println!("# Rust core throughput at scale (event-driven, noise_std=0)");
    println!();
    println!(
        "{:>12} {:>8} {:>14} {:>10} {:>12} {:>14} {:>12} {:>10}",
        "n", "k_out", "synapses", "driven%", "active%", "ms/step", "bio-ms/wall-s", "RSS GiB"
    );

    for &(n, k, frac) in &[
        (1_000_000u32, 64u32, 0.001f32),
        (1_000_000, 64, 0.01),
        (4_000_000, 64, 0.01),
        (4_000_000, 256, 0.01),
    ] {
        let mut p = Params::new(n, k);
        p.noise_std = 0.0;
        let mut cfg = BuildCfg::new(n, k, 999);
        cfg.driven_frac = frac;
        let t_build = Instant::now();
        let syn = Synapses::build(&cfg);
        let syn_bytes = syn.bytes();
        let build_s = t_build.elapsed().as_secs_f64();
        let mut sim = Sim::new(p.clone(), syn);
        let drive = Drive::Sparse(build_sparse_drive(&cfg, &p, 999));

        let warm = 10;
        let steps = 60u32;
        for _ in 0..warm {
            sim.step(&drive, Mode::EventDriven, false, Some((n / 10) as usize));
        }
        sim.total_spikes = 0;
        sim.total_neuron_updates = 0;
        sim.total_synops_active = 0;
        let t0 = Instant::now();
        for _ in 0..steps {
            sim.step(&drive, Mode::EventDriven, false, Some((n / 10) as usize));
        }
        let dt = t0.elapsed().as_secs_f64();
        let ms = dt / steps as f64 * 1e3;
        let spk = sim.total_spikes as f64 / steps as f64;
        let rss = rss_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{:>12} {:>8} {:>14} {:>10.2} {:>12.5} {:>14.4} {:>12.1} {:>10.3}",
            n,
            k,
            n as u64 * k as u64,
            frac * 100.0,
            spk / n as f64 * 100.0,
            ms,
            if ms > 0.0 { 1000.0 / ms } else { 0.0 },
            rss
        );
        eprintln!(
            "    [n={n} k={k}] build {build_s:.2}s, connectivity {:.2} GiB, state {:.2} GiB",
            syn_bytes as f64 / (1024.0f64.powi(3)),
            sim.state_bytes() as f64 / (1024.0f64.powi(3))
        );
    }
}

/// Correctness comparison against a fixture exported by the Python simulator.
///
/// Both engines start from bit-identical connectivity, state and drive, with
/// `noise_std = 0` so the comparison is meaningful. Three things are reported:
///   1. agreement of the spike trajectory step by step;
///   2. agreement of per-neuron firing statistics;
///   3. a NEGATIVE CONTROL with all weights zeroed, which must FAIL to match.
/// Without (3) an exact match would be uninformative: it could simply mean the
/// comparison is insensitive to the delivery path.
fn compare(args: &[String]) {
    let path = match args.get(2) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: rust_core compare FIXTURE.bin");
            std::process::exit(2);
        }
    };
    let fx = match io::load_fixture(std::path::Path::new(&path)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };
    let steps = fx.steps;
    let rust = io::run_fixture(&fx, Mode::EventDriven, steps);

    let n = fx.params.n as usize;
    let rust_counts: Vec<u32> = rust.spike_count.clone();
    let py_counts = &fx.exp_neuron_spikes;

    let rust_total: u64 = rust_counts.iter().map(|&c| c as u64).sum();
    let py_total: u64 = py_counts.iter().map(|&c| c as u64).sum();
    let rust_active = rust_counts.iter().filter(|&&c| c > 0).count();
    let py_active = py_counts.iter().filter(|&&c| c > 0).count();

    println!("# Correctness comparison vs Python simulator");
    println!("fixture: {path}");
    println!("n = {n}, k_out = {}, steps = {steps}, noise_std = {}",
             fx.params.k_out, fx.params.noise_std);
    println!();
    println!("{:<30} {:>14} {:>14}", "quantity", "python", "rust");
    println!("{:<30} {:>14} {:>14}", "total spikes", py_total, rust_total);
    println!("{:<30} {:>14} {:>14}", "neurons that spiked", py_active, rust_active);
    println!("{:<30} {:>14.4} {:>14.4}", "mean spikes/neuron",
             py_total as f64 / n as f64, rust_total as f64 / n as f64);
    println!();

    // --- per-step trajectory ------------------------------------------------
    let py_steps = &fx.exp_step_spikes;
    let ru_steps = &rust.step_spikes;
    let cmp_len = steps.min(ru_steps.len() as u32) as usize;
    let step_matches = (0..cmp_len).filter(|&i| py_steps[i] == ru_steps[i]).count();
    println!("per-step spike-count agreement   = {step_matches}/{cmp_len} steps");
    let mut first_div = None;
    for i in 0..cmp_len {
        if py_steps[i] != ru_steps[i] {
            first_div = Some(i);
            break;
        }
    }
    match first_div {
        None => println!("first divergence                 = none (all {cmp_len} steps identical)"),
        Some(i) => println!(
            "first divergence                 = step {i} (python {}, rust {})",
            py_steps[i], ru_steps[i]
        ),
    }

    // --- per-neuron statistics ---------------------------------------------
    let mp = py_total as f64 / n as f64;
    let mr = rust_total as f64 / n as f64;
    let (mut cov, mut vp, mut vr) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let a = py_counts[i] as f64;
        let b = rust_counts[i] as f64;
        cov += (a - mp) * (b - mr);
        vp += (a - mp) * (a - mp);
        vr += (b - mr) * (b - mr);
    }
    let corr = if vp > 0.0 && vr > 0.0 { cov / (vp.sqrt() * vr.sqrt()) } else { f64::NAN };
    println!("per-neuron spike-count Pearson r = {corr:.4}");

    let agree: usize = (0..n)
        .filter(|&i| (py_counts[i] > 0) == (rust_counts[i] > 0))
        .count();
    println!("silent/active agreement          = {agree}/{n} ({:.4})", agree as f64 / n as f64);

    let rel = if py_total > 0 {
        (rust_total as f64 - py_total as f64).abs() / py_total as f64
    } else {
        0.0
    };
    println!("relative total-spike divergence  = {rel:.6}");

    // --- negative control ---------------------------------------------------
    let ctl = io::run_fixture_ctl(&fx, Mode::EventDriven, steps, true);
    let ctl_total: u64 = ctl.spike_count.iter().map(|&c| c as u64).sum();
    let ctl_steps_match = (0..cmp_len)
        .filter(|&i| py_steps[i] == ctl.step_spikes[i])
        .count();
    let ctl_rel = if py_total > 0 {
        (ctl_total as f64 - py_total as f64).abs() / py_total as f64
    } else {
        0.0
    };
    println!();
    println!("Negative control (all weights zeroed, delivery path destroyed):");
    println!("  total spikes                   = {ctl_total} (python {py_total}, divergence {ctl_rel:.4})");
    println!("  per-step agreement             = {ctl_steps_match}/{cmp_len} steps");
    let has_power = ctl_total != rust_total && ctl_rel > 0.05;
    println!("  comparison has power           = {has_power} (must be true, else the match above is vacuous)");

    // --- determinism --------------------------------------------------------
    println!();
    println!("Determinism (same fixture, second run in this process):");
    let again = io::run_fixture(&fx, Mode::EventDriven, steps);
    println!("  identical per-step spike counts = {}", again.step_spikes == rust.step_spikes);
    println!("  identical per-neuron counts     = {}", again.spike_count == rust.spike_count);
    println!("  bitwise identical final v_soma  = {}",
             again.v_soma.iter().zip(rust.v_soma.iter()).all(|(a, b)| a == b));

    println!();
    println!("VERDICT:");
    let trajectory_ok = first_div.is_none();
    let stats_ok = corr > 0.999 && agree == n;
    println!("  trajectory-identical to Python  = {trajectory_ok}");
    println!("  per-neuron statistics identical = {stats_ok}");
    println!("  control demonstrates power      = {has_power}");
    if !has_power {
        println!("  WARNING: the control did not diverge from the real run, so an");
        println!("  exact match here does NOT demonstrate that delivery is correct.");
    }
}
