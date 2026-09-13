//! Event-driven spiking core.
//!
//! Semantics are a port of `brain/neurons.py::NeuronState.update` and
//! `brain/connectivity.py::DelayBuffer`. The integration ORDER is not arbitrary;
//! it reproduces the Python update exactly:
//!
//!   1. dendrite Euler step
//!   2. dendritic activation phi(v_dend)          (dCaAP: z*exp(1-z))
//!   3. adaptation decay, refractory countdown
//!   4. soma Euler step (drive + dend_drive - adapt)
//!   5. threshold test against refractory block
//!   6. reset v / bump adapt / set refractory
//!
//! Because `refrac` decays BEFORE the block test in step 5, a neuron that spiked
//! at t has refrac = refractory_ms, then 1.0 at t+1 (blocked), then 0.0 at t+2
//! (free). Minimum inter-spike interval is therefore exactly `refractory_ms`,
//! matching the Python test-suite.
//!
//! State layout is struct-of-arrays, one contiguous `Vec` per variable.
//!
//! NOT implemented here (out of scope for this prototype, and stated so in
//! BENCH.md): k-WTA inhibition, three-factor plasticity, homeostasis, and the
//! readout/decoder stack. This core is the dynamics + communication path only.

pub const DEND_DCAAP: u32 = 0;
pub const DEND_LINEAR: u32 = 1;
pub const DEND_NONE: u32 = 2;

#[derive(Clone, Debug)]
pub struct Params {
    pub n: u32,
    pub k_out: u32,
    pub max_delay: u32,
    pub tau_soma: f32,
    pub tau_dend: f32,
    pub tau_adapt: f32,
    pub e_leak: f32,
    pub e_dend: f32,
    pub v_reset: f32,
    pub v_thresh: f32,
    pub adapt_base: f32,
    pub adapt_inc: f32,
    pub dend_scale: f32,
    pub dend_gain: f32,
    pub refractory_ms: f32,
    pub noise_std: f32,
    pub dt: f32,
    pub dend_mode: u32,
}

impl Params {
    pub fn new(n: u32, k_out: u32) -> Self {
        Params {
            n,
            k_out,
            max_delay: 4,
            tau_soma: 20.0,
            tau_dend: 10.0,
            tau_adapt: 100.0,
            e_leak: 0.0,
            e_dend: 0.0,
            v_reset: 0.0,
            v_thresh: 1.0,
            adapt_base: 0.0,
            adapt_inc: 0.05,
            dend_scale: 1.0,
            dend_gain: 1.0,
            refractory_ms: 2.0,
            noise_std: 0.0,
            dt: 1.0,
            dend_mode: DEND_DCAAP,
        }
    }
}

/// `phi(v_dend)`, the dendritic nonlinearity.
///
/// dCaAP: `u/s * exp(1 - u/s)` with `u = max(v_dend - e_dend, 0)`. This is the
/// tuned, NON-MONOTONIC response (Gidon et al. 2020, Science 367:83): it peaks
/// at value 1.0 when `v_dend - e_dend == dend_scale` and decays beyond.
#[inline(always)]
pub fn dend_activation(p: &Params, v_dend: f32) -> f32 {
    if p.dend_mode == DEND_NONE {
        return 0.0;
    }
    let u = if v_dend > p.e_dend { v_dend - p.e_dend } else { 0.0 };
    if p.dend_mode == DEND_LINEAR {
        return u / p.dend_scale;
    }
    let z = u / p.dend_scale;
    z * (1.0 - z).exp()
}

/// Fixed fan-out connectivity, same layout as Python's `Synapses` (every
/// pre-neuron owns exactly `k_out` contiguous slots).
pub struct Synapses {
    pub targets: Vec<u32>,
    pub weights: Vec<f32>,
    pub delays: Vec<u32>,
}

impl Synapses {
    pub fn bytes(&self) -> u64 {
        (self.targets.len() * 4 + self.weights.len() * 4 + self.delays.len() * 4) as u64
    }
}

pub enum Drive {
    None,
    /// `(neuron_index, amplitude)`, sorted by index. Only these neurons receive
    /// current, so the active set is genuinely a subset of the population.
    Sparse(Vec<(u32, f32)>),
    /// One amplitude per neuron (possibly mostly zero), as the repo's
    /// `forced_activity` protocol uses.
    Dense(Vec<f32>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Integrate every neuron every step: structurally identical work to the
    /// Python path. Isolates implementation speed from algorithmic gain.
    Dense,
    /// Integrate only neurons that can change state. Exact when a resting neuron
    /// provably unable to fire without new input (see `Sim::can_sleep`).
    EventDriven,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub spikes: u64,
    pub neuron_updates: u64,
    pub synops_active: u64,
    pub synops_executed: u64,
    pub overflow: u64,
    pub awake: u64,
}

pub struct Sim {
    pub p: Params,
    pub syn: Synapses,
    // --- neuron state, struct of arrays ---
    pub v_soma: Vec<f32>,
    pub v_dend: Vec<f32>,
    pub adapt: Vec<f32>,
    pub refrac: Vec<f32>,
    pub spike_count: Vec<u32>,
    pub v_pre_reset: Vec<f32>,
    // --- delay ring ---
    flat: Vec<f32>,
    depth: usize,
    slot_touched: Vec<Vec<u32>>,
    seen: Vec<u64>,
    // --- active set bookkeeping ---
    awake: Vec<u32>,
    in_awake: Vec<u32>,
    awake_epoch: u32,
    carried: Vec<u32>,
    /// Neurons proven unable to fire without new input. While asleep they cost
    /// nothing and are advanced in closed form on wake (see `fast_forward`).
    asleep: Vec<bool>,
    sleep_since: Vec<u64>,
    idx_len: usize,
    needs_reseed: bool,
    // --- per-step scratch (no allocation in the hot loop) ---
    pub i_soma_buf: Vec<f32>,
    arrivals: Vec<u32>,
    prev_arrivals: Vec<u32>,
    /// Neurons that slept and are being woken this step, so their state can be
    /// fast-forwarded before they are integrated.
    pending_wake: Vec<u32>,
    spike_idx: Vec<u32>,
    last_spikes: Vec<u32>,
    pub step_spikes: Vec<u32>,
    // --- counters ---
    pub time: u64,
    pub total_spikes: u64,
    pub total_neuron_updates: u64,
    pub total_synops_active: u64,
    pub total_synops_executed: u64,
    pub total_overflow: u64,
    pub peak_awake: u64,
    pub total_sleep_transitions: u64,
}

impl Sim {
    pub fn new(p: Params, syn: Synapses) -> Self {
        let n = p.n as usize;
        let depth = (p.max_delay + 1) as usize;
        Sim {
            v_soma: vec![p.e_leak; n],
            v_dend: vec![p.e_dend; n],
            adapt: vec![p.adapt_base; n],
            refrac: vec![0.0; n],
            spike_count: vec![0; n],
            v_pre_reset: vec![p.e_leak; n],
            flat: vec![0.0; depth * n],
            depth,
            slot_touched: (0..depth).map(|_| Vec::new()).collect(),
            seen: vec![u64::MAX; n],
            awake: Vec::new(),
            in_awake: vec![0; n],
            awake_epoch: 1,
            carried: Vec::new(),
            asleep: vec![false; n],
            sleep_since: vec![0; n],
            idx_len: 0,
            needs_reseed: true,
            i_soma_buf: vec![0.0; n],
            arrivals: Vec::new(),
            pending_wake: Vec::new(),
            prev_arrivals: Vec::new(),
            spike_idx: Vec::new(),
            last_spikes: Vec::new(),
            step_spikes: Vec::new(),
            time: 0,
            total_spikes: 0,
            total_neuron_updates: 0,
            total_synops_active: 0,
            total_synops_executed: 0,
            total_overflow: 0,
            peak_awake: 0,
            total_sleep_transitions: 0,
            p,
            syn,
        }
    }

    /// Snapshot of the sleep flags, for tests and diagnostics.
    pub fn asleep_snapshot(&self) -> Vec<bool> {
        self.asleep.clone()
    }

    /// Number of neurons currently skipping integration.
    pub fn n_asleep(&self) -> u64 {
        self.asleep.iter().filter(|&&a| a).count() as u64
    }

    pub fn n_synapses(&self) -> u64 {
        self.p.n as u64 * self.p.k_out as u64
    }

    /// Bytes held by neuron state, ring buffers and index structures.
    /// Excludes connectivity, which is reported separately.
    pub fn state_bytes(&self) -> u64 {
        (self.v_soma.len() * 4      // v_soma
            + self.v_dend.len() * 4            // v_dend
            + self.adapt.len() * 4             // adapt
            + self.refrac.len() * 4            // refrac
            + self.spike_count.len() * 4       // spike_count
            + self.v_pre_reset.len() * 4       // v_pre_reset
            + self.flat.len() * 4              // delay ring
            + self.i_soma_buf.len() * 4        // current buffer
            + self.in_awake.len() * 4          // active-set membership
            + self.seen.len() * 8              // dedup tokens
            + (self.slot_touched.len() * 32)) as u64 // Vec headers
    }

    /// Spike indices produced by the most recent `step`, in ascending order.
    ///
    /// `step` clears its working buffer, so consumers that need the spike list
    /// (tests, correctness comparison, readouts) read this copy instead.
    pub fn spike_idx_snapshot(&self) -> Vec<u32> {
        self.last_spikes.clone()
    }

    /// Schedule one pre-neuron's fan-out by hand. Exists for the unit checks
    /// that verify the delay ring in isolation from the dynamics.
    pub fn force_schedule(&mut self, t: u64, pre: u32) {
        self.deliver(t, pre);
    }

    /// Read the ring buffer value scheduled to reach `target` at absolute time
    /// `t`, without clearing it. Test-only introspection.
    pub fn probe_current(&self, target: usize, t: u64) -> f32 {
        let n = self.p.n as usize;
        let slot = (t as usize) % self.depth;
        self.flat[slot * n + target]
    }

    /// True when the neuron PROVABLY cannot fire again until new input arrives.
    ///
    /// With no arriving current the soma recurrence is
    ///
    /// ```text
    /// w(t+1) = a*w(t) + (dt/tau)*D(t),   w = v_soma - e_leak,  a = 1 - dt/tau
    /// D(t)   = dend_gain*phi(v_dend(t)) - adapt(t)
    /// ```
    ///
    /// Unrolling, `w(t) = a^t*w(0) + (dt/tau)*sum_{s<t} a^(t-1-s)*D(s)`. If
    /// `D(t) <= 0` for all `t >= 0` the whole sum is non-positive, giving
    ///
    /// ```text
    /// sup_t w(t) <= max(w(0), 0)   hence   sup_t v_soma <= e_leak + max(v_soma - e_leak, 0)
    /// ```
    ///
    /// If that supremum is below `v_thresh`, the neuron cannot fire again until
    /// input arrives, so integrating it is pure waste.
    ///
    /// Two conditions pin `D(t) <= 0`:
    ///
    /// * `v_dend == e_dend` exactly, so `phi(v_dend) == 0`. This holds in this
    ///   substrate because `brain/simulator.py` always passes `i_dend = 0` and
    ///   `v_dend` is initialised to `e_dend`, so it never moves. It is CHECKED
    ///   rather than assumed, so the moment a caller introduces dendritic current
    ///   the shortcut disables itself instead of silently corrupting results.
    ///   (The dendritic path is this substrate's most distinctive feature, so
    ///   this is a real constraint on the prototype, not a formality.)
    /// * `adapt >= 0`. With `adapt_base = 0` and `adapt_inc > 0`, adaptation is
    ///   non-negative throughout, so `-adapt(t) <= 0`. Also checked.
    ///
    /// `refrac > 0` is deliberately NOT required: a refractory neuron is even
    /// safer, and requiring `refrac == 0` would exclude exactly the neurons that
    /// dominate sparse activity.
    ///
    /// This is a proof, not a tolerance. The remaining approximation is the
    /// floating-point rounding of the closed-form fast-forward, which is exact in
    /// real arithmetic.
    ///
    /// Sleeping is the ONLY thing that makes event-driven cheaper than dense, and
    /// it is why the criterion must not be "exactly at rest": `adapt` decays as
    /// `a_adapt^t` and never reaches exactly zero, so a criterion requiring exact
    /// rest makes every neuron that has ever spiked awake forever. Measured, that
    /// grew the awake set from 55k to 880k neurons over 120 steps and erased the
    /// advantage above ~0.15% activity.
    #[inline(always)]
    fn can_sleep(p: &Params, v_soma: f32, v_dend: f32, adapt: f32) -> bool {
        v_dend == p.e_dend
            && adapt >= 0.0
            && p.e_leak + (v_soma - p.e_leak).max(0.0) < p.v_thresh
    }

    /// Catch a sleeping neuron up to `now` with an EXACT cheap loop.
    ///
    /// A closed form (`a.powf(t)`) is mathematically right but not bit-exact with
    /// repeated Euler steps: `powf` is computed as `exp(t*ln a)` and rounds
    /// differently from t sequential multiplies. Measured, that ~1 ULP difference
    /// was enough to make event-driven diverge from the dense path in a chaotic
    /// spiking network (~0.02% of spikes at 20% activity, growing with activity),
    /// so a closed form is not acceptable here even though the algebra is correct.
    ///
    /// This loop instead applies the same arithmetic the integrator would, in the
    /// same order, but only the terms that are non-zero for a neuron that provably
    /// cannot fire:
    ///
    ///   * no dendritic activation (`phi == 0`, guaranteed by `can_sleep`)
    ///   * no arriving current and no drive (guaranteed by the caller)
    ///   * no threshold test and no branch, because the neuron cannot cross it
    ///
    /// so it is a few flops per skipped millisecond instead of a full `integrate_one`
    /// (which pays an `exp` for the dendritic activation and a compare/branch pair
    /// every step). Bit-exactness is the point: the state after catching up is
    /// identical to what step-by-step integration would have produced, so
    /// event-driven and dense emit identical spike trains.
    #[inline]
    pub fn relax(&self, u: usize, t: u64) -> (f32, f32, f32, f32) {
        let p = &self.p;
        let dt = p.dt;
        let (mut v_soma, mut v_dend, mut adapt, mut refrac) =
            (self.v_soma[u], self.v_dend[u], self.adapt[u], self.refrac[u]);
        let a_soma = dt / p.tau_soma;
        let a_dend = dt / p.tau_dend;
        let a_adapt = dt / p.tau_adapt;
        for _ in 0..t {
            // Same order as `integrate_one` with i_soma = 0 and drive = 0.
            v_dend += a_dend * (-(v_dend - p.e_dend) + 0.0);
            adapt *= 1.0 - a_adapt;
            refrac = (refrac - dt).max(0.0);
            v_soma += a_soma * (-(v_soma - p.e_leak) + 0.0 + 0.0 - adapt);
        }
        (v_soma, v_dend, adapt, refrac)
    }

    /// Bring every sleeping neuron's state exactly up to the current time.
    ///
    /// Sleeping neurons are not integrated step by step, so their stored fields
    /// hold the values from when they fell asleep. Spike output and delivered
    /// current are unaffected, because a neuron is fast-forwarded before it is
    /// integrated again. But a consumer that reads the continuous state directly
    /// (a rate readout over `v_soma`, an analysis plot, a decoder) must call this
    /// first or it will read a stale value.
    ///
    /// O(N) with no per-step cost, so the hot loop stays activity-scaled while
    /// exact continuous state remains available on demand. It does not perturb
    /// the spike trajectory: it writes the values the step-by-step integrator
    /// would have produced.
    pub fn materialize_state(&mut self) {
        let t = self.time;
        for u in 0..self.p.n as usize {
            if self.asleep[u] {
                let n = t.saturating_sub(self.sleep_since[u]).saturating_sub(1);
                if n > 0 {
                    let (vs, vd, ad, rf) = self.relax(u, n);
                    self.v_soma[u] = vs;
                    self.v_dend[u] = vd;
                    self.adapt[u] = ad;
                    self.refrac[u] = rf;
                }
                self.sleep_since[u] = t;
            }
        }
    }



    /// Mark the active set as needing a full rescan. Call after installing state
    /// from outside (fixture load, manual state edit, `reset`).
    pub fn request_reseed(&mut self) {
        self.needs_reseed = true;
    }

    /// Advance one millisecond of biological time.
    ///
    /// Pipeline: deliver -> wake -> integrate -> sleep -> deliver.
    ///
    /// `padded_gather` reproduces the Python implementation's delivery shape: a
    /// dense `capacity * k_out` rectangle whose padding rows re-deliver neuron
    /// 0's synapses. It exists to measure that defect in a controlled way, so the
    /// algorithmic effect can be separated from the language effect.
    /// `spike_cap` reproduces Python's spike-buffer truncation: `Brain._compact`
    /// keeps only the first `capacity` spikes in ascending neuron-index order and
    /// counts the rest as `overflow`, so those spikes are NEVER DELIVERED. This is
    /// a lossy artifact of the Python implementation, not a modelling choice, and
    /// it matters at high activity: at 20% driven the Python path discarded 43% of
    /// the synaptic events it counted. Passing `None` means "deliver everything",
    /// which is what an event-driven core naturally does, and is the reason the
    /// two engines diverge above ~0.1% activity unless truncation is enabled.
    pub fn step(
        &mut self,
        drive: &Drive,
        mode: Mode,
        padded_gather: bool,
        spike_cap: Option<usize>,
    ) -> StepStats {
        let k = self.p.k_out as usize;
        let t = self.time;

        // ---- 1. current arriving now --------------------------------------
        self.read_and_clear();

        // ---- 2. wake: decide who must be integrated ------------------------
        // Event-driven skipping is only valid when every unvisited neuron
        // provably cannot change. That fails as soon as noise is injected into
        // every neuron. The test is on the CONFIGURATION, never on how many
        // neurons happen to be awake.
        let event_mode = mode == Mode::EventDriven && self.p.noise_std == 0.0;
        let dense = !event_mode;
        let mut wake = std::mem::take(&mut self.awake);
        wake.clear();
        let mut fast_fwd = std::mem::take(&mut self.pending_wake);
        fast_fwd.clear();

        if dense {
            for u in 0..self.p.n {
                wake.push(u);
            }
        } else {
            self.awake_epoch = self.awake_epoch.wrapping_add(1);
            let ep = self.awake_epoch;

            macro_rules! enqueue {
                ($u:expr) => {{
                    let u: u32 = $u;
                    if self.in_awake[u as usize] != ep {
                        self.in_awake[u as usize] = ep;
                        if self.asleep[u as usize] {
                            self.asleep[u as usize] = false;
                            fast_fwd.push(u);
                        }
                        wake.push(u);
                    }
                }};
            }

            let cur = std::mem::take(&mut self.arrivals);
            for &u in cur.iter() {
                enqueue!(u);
            }
            self.arrivals = cur;

            // A neuron holding state that could still make it fire must keep
            // being integrated, or it would freeze mid-trajectory.
            let carried = std::mem::take(&mut self.carried);
            for &u in carried.iter() {
                enqueue!(u);
            }
            self.carried = carried;

            match drive {
                Drive::None => {}
                Drive::Sparse(list) => {
                    for &(u, _) in list.iter() {
                        enqueue!(u);
                    }
                }
                Drive::Dense(d) => {
                    for (u, &v) in d.iter().enumerate() {
                        if v != 0.0 {
                            enqueue!(u as u32);
                        }
                    }
                }
            }

            // Robustness: after installing state from outside (fixture load,
            // manual edit, reset) a non-rest neuron might not be in any of the
            // lists above. Sweep once so it cannot be silently frozen.
            if self.needs_reseed {
                let p2 = self.p.clone();
                for u in 0..self.p.n {
                    let uu = u as usize;
                    if Self::can_sleep(&p2, self.v_soma[uu], self.v_dend[uu], self.adapt[uu]) {
                        self.asleep[uu] = true;
                        self.sleep_since[uu] = t;
                    } else {
                        enqueue!(u as u32);
                    }
                }
                self.needs_reseed = false;
            }
        }

        // Dense-iteration fallback. When almost every neuron is awake, the active
        // set buys no skipping but still costs indirect indexed access, plus a
        // 1M-entry `wake` vector. Iterating 0..n sequentially is then strictly
        // cheaper. Measured at n=1M, k=64: event-driven integrated 99.998% of
        // neurons at 0.5 driven and took 97.5 ms/step versus 68.9 ms/step for the
        // dense loop -- i.e. the sparse machinery was pure overhead once there was
        // nothing to skip. This fallback keeps the semantics identical (every
        // neuron is integrated either way) and removes that overhead.
        //
        // The threshold is on the WORK, not on the model: it activates only when
        // the awake set is >= 90% of the population.
        let wake_nearly_full = event_mode && wake.len() * 10 >= self.p.n as usize * 9;
        let idx: Vec<u32> = if wake_nearly_full {
            // Falling back to dense iteration means EVERY neuron is about to be
            // integrated, so every sleeping neuron must first be brought exactly
            // up to date. A sleeping neuron that is neither driven nor receiving
            // current is never enqueued, so it would otherwise be integrated from
            // stale state -- which measured as event-driven diverging from dense
            // by 0.03% of spikes at 20% activity, with the first divergence two
            // steps in. Relaxing them here makes the fallback step EXACTLY the
            // dense step.
            for u in 0..self.p.n {
                let uu = u as usize;
                if self.asleep[uu] {
                    let n = t.saturating_sub(self.sleep_since[uu]).saturating_sub(1);
                    if n > 0 {
                        let (vs, vd, ad, rf) = self.relax(uu, n);
                        self.v_soma[uu] = vs;
                        self.v_dend[uu] = vd;
                        self.adapt[uu] = ad;
                        self.refrac[uu] = rf;
                    }
                    self.asleep[uu] = false;
                }
            }
            let mut v = wake;
            v.clear();
            v.extend(0..self.p.n);
            v
        } else {
            wake
        };
        self.idx_len = idx.len();

        // Advance every neuron that slept, in closed form, to the end of the
        // previous step. `sleep_since` is the step whose POST-integration state is
        // stored, and we are about to integrate step `t`, so the target state is
        // post-step-(t-1): that is `t - sleep_since - 1` further steps.
        if !fast_fwd.is_empty() {
            for &u in fast_fwd.iter() {
                let n = t.saturating_sub(self.sleep_since[u as usize]).saturating_sub(1);
                if n > 0 {
                    let (vs, vd, ad, rf) = self.relax(u as usize, n);
                    self.v_soma[u as usize] = vs;
                    self.v_dend[u as usize] = vd;
                    self.adapt[u as usize] = ad;
                    self.refrac[u as usize] = rf;
                }
            }
            self.pending_wake = fast_fwd;
        }

        // ---- 3. integrate ---------------------------------------------------
        let mut spikes: Vec<u32> = Vec::new();
        let mut still: Vec<u32> = Vec::new();
        {
            let vs = &mut self.v_soma;
            let vd = &mut self.v_dend;
            let ad = &mut self.adapt;
            let rf = &mut self.refrac;
            let vpr = &mut self.v_pre_reset;
            let sc = &mut self.spike_count;
            let ibuf = &self.i_soma_buf;
            let mut noise = NoiseStream::new(t, self.p.noise_std);
            let p = self.p.clone();
            for &u in idx.iter() {
                let uu = u as usize;
                if integrate_one(&p, uu, ibuf[uu], drive.at(uu), vs, vd, ad, rf, vpr, &mut noise) {
                    sc[uu] += 1;
                    spikes.push(u);
                }
                if event_mode {
                    let driven_now = drive.at(uu) != 0.0;
                    // Sleep only when all three hold:
                    //   * the neuron provably cannot fire without new input;
                    //   * it received no drive this step, so nothing will wake it
                    //     next step -- otherwise the sleep/wake round trip is paid
                    //     EVERY step (measured: 3x slower than dense at 10% driven);
                    //   * the sparse path is actually in use. When the dense
                    //     fallback is engaged the whole population is integrated
                    //     anyway, so sleeping would only add bookkeeping.
                    if !wake_nearly_full
                        && !driven_now
                        && Self::can_sleep(&p, vs[uu], vd[uu], ad[uu])
                    {
                        self.asleep[uu] = true;
                        self.sleep_since[uu] = t;
                        self.total_sleep_transitions += 1;
                    } else {
                        // Carry it into the next step's active set. A neuron that is
                        // neither driven nor asleep must stay in the set, or it
                        // freezes mid-trajectory.
                        still.push(u);
                    }
                }
            }
        }
        self.total_neuron_updates += idx.len() as u64;
        if idx.len() as u64 > self.peak_awake {
            self.peak_awake = idx.len() as u64;
        }
        self.awake = idx;
        self.carried = still;

        let n_spk = spikes.len();
        // Python keeps the first `capacity` spikes in ascending index order
        // (`keep = mask & (pos < capacity)`), so truncation is a head-slice of
        // the sorted spike list. Delivery itself is order-independent (it is a
        // scatter-add), so sorting is only required when truncation will actually
        // discard something. Sorting unconditionally cost more than the entire
        // dense integration loop at 1M neurons (~50 ms/step for 1M u32), which
        // made event-driven look slower than dense at high activity for reasons
        // that had nothing to do with the event-driven design.
        let overflow = match spike_cap {
            Some(cap) => {
                let excess = n_spk.saturating_sub(cap);
                if excess > 0 {
                    spikes.sort_unstable();
                    spikes.truncate(cap);
                }
                self.total_overflow += excess as u64;
                excess
            }
            None => 0,
        };

        // ---- 4. deliver ------------------------------------------------------
        self.spike_idx = spikes;
        if padded_gather {
            let cap = spike_cap.unwrap_or(self.spike_idx.len());
            self.schedule_padded(t, cap);
        } else {
            self.schedule(t);
        }

        // ---- 5. bookkeeping --------------------------------------------------
        let delivered = self.spike_idx.len();
        let synops_active = (delivered as u64) * (k as u64);
        let synops_executed = if padded_gather {
            let cap = spike_cap.unwrap_or(delivered);
            (cap as u64) * (k as u64)
        } else {
            synops_active
        };
        self.total_spikes += n_spk as u64;
        self.total_synops_active += synops_active;
        self.total_synops_executed += synops_executed;
        self.time += 1;
        self.step_spikes.push(n_spk as u32); // pre-truncation, matching Python's StepStats.spikes
        self.last_spikes.clear();
        self.last_spikes.extend_from_slice(&self.spike_idx);
        self.spike_idx.clear();

        StepStats {
            spikes: delivered as u64,
            neuron_updates: self.idx_len as u64,
            synops_active,
            synops_executed,
            overflow: overflow as u64,
            awake: self.idx_len as u64,
        }
    }

    /// Deliver current for time `t`, clearing only slots actually used.
    ///
    /// Two separate concerns, both O(arrivals) rather than O(N):
    ///   * the PREVIOUS step's entries must be zeroed, or a neuron that had
    ///     current at t-1 and none at t would keep integrating stale drive;
    ///   * only the ring slot for time t is examined, so a step with no arriving
    ///     current touches nothing at all.
    fn read_and_clear(&mut self) {
        let n = self.p.n as usize;

        // 1. expire last step's current
        for &u in self.prev_arrivals.iter() {
            self.i_soma_buf[u as usize] = 0.0;
        }

        // 2. consume this step's ring slot
        let slot = (self.time as usize) % self.depth;
        let base = slot * n;
        let mut cur = std::mem::take(&mut self.arrivals);
        cur.clear();
        for i in 0..self.slot_touched[slot].len() {
            let u = self.slot_touched[slot][i] as usize;
            let v = self.flat[base + u];
            if v != 0.0 {
                self.i_soma_buf[u] = v;
                cur.push(u as u32);
            }
            self.flat[base + u] = 0.0;
        }
        self.slot_touched[slot].clear();
        self.prev_arrivals.clear();
        self.prev_arrivals.extend_from_slice(&cur);
        self.arrivals = cur;
    }

    #[inline]
    fn deliver(&mut self, t: u64, pre: u32) {
        let n = self.p.n as usize;
        let k = self.p.k_out as usize;
        let base = pre as usize * k;
        for j in 0..k {
            let post = self.syn.targets[base + j] as usize;
            let arrival = t + self.syn.delays[base + j] as u64;
            let slot = (arrival as usize) % self.depth;
            self.flat[slot * n + post] += self.syn.weights[base + j];
            if self.seen[post] != arrival {
                self.seen[post] = arrival;
                self.slot_touched[slot].push(post as u32);
            }
        }
    }

    /// True-sparse delivery: touches exactly `spikes * k_out` rows.
    fn schedule(&mut self, t: u64) {
        let spikes = std::mem::take(&mut self.spike_idx);
        for &pre in spikes.iter() {
            self.deliver(t, pre);
        }
        self.spike_idx = spikes;
    }

    /// Python-shaped delivery: iterate exactly `capacity` rows. Rows at or beyond
    /// the spike count are padding and re-deliver neuron 0's synapses, which is
    /// the documented cause of the substrate's contaminated-activity artefact.
    fn schedule_padded(&mut self, t: u64, capacity: usize) {
        let spikes = std::mem::take(&mut self.spike_idx);
        for r in 0..capacity {
            let pre = if r < spikes.len() { spikes[r] } else { 0 };
            self.deliver(t, pre);
        }
        self.spike_idx = spikes;
    }

    pub fn reset(&mut self) {
        let p = &self.p;
        self.v_soma.fill(p.e_leak);
        self.v_dend.fill(p.e_dend);
        self.adapt.fill(p.adapt_base);
        self.refrac.fill(0.0);
        self.spike_count.fill(0);
        self.flat.fill(0.0);
        for s in self.slot_touched.iter_mut() {
            s.clear();
        }
        self.seen.fill(u64::MAX);
        self.carried.clear();
        self.awake.clear();
        self.asleep.fill(false);
        self.sleep_since.fill(0);
        self.total_sleep_transitions = 0;
        self.i_soma_buf.fill(0.0);
        self.arrivals.clear();
        self.prev_arrivals.clear();
        self.last_spikes.clear();
        self.step_spikes.clear();
        self.time = 0;
        self.total_spikes = 0;
        self.total_neuron_updates = 0;
        self.total_synops_active = 0;
        self.total_synops_executed = 0;
        self.total_overflow = 0;
        self.peak_awake = 0;
        self.needs_reseed = true;
    }

    pub fn total_spikes_in_counters(&self) -> u64 {
        self.spike_count.iter().map(|&c| c as u64).sum()
    }

    pub fn mean_rate_hz(&self, steps: u64) -> f64 {
        let tot = self.total_spikes_in_counters();
        let dur_s = steps as f64 * self.p.dt as f64 / 1000.0;
        tot as f64 / self.p.n as f64 / dur_s
    }

    pub fn active_neurons(&self) -> u64 {
        self.spike_count.iter().filter(|&&c| c > 0).count() as u64
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn integrate_one(
    p: &Params,
    u: usize,
    i_soma: f32,
    drive_v: f32,
    vs: &mut [f32],
    vd: &mut [f32],
    ad: &mut [f32],
    rf: &mut [f32],
    vpr: &mut [f32],
    noise: &mut NoiseStream,
) -> bool {
    let dt = p.dt;

    // 1. dendrite (no external dendritic current in this core)
    vd[u] += (dt / p.tau_dend) * (-(vd[u] - p.e_dend));
    // 2. dendritic activation
    let dend_drive = p.dend_gain * dend_activation(p, vd[u]);
    // 3. adaptation decay + refractory countdown
    ad[u] *= 1.0 - dt / p.tau_adapt;
    rf[u] = (rf[u] - dt).max(0.0);
    // 4. soma
    let mut drive = i_soma + drive_v;
    drive += noise.next(u);
    vs[u] += (dt / p.tau_soma) * (-(vs[u] - p.e_leak) + drive + dend_drive - ad[u]);
    // 5. threshold test against the refractory block
    let blocked = rf[u] > 0.0;
    let fired = vs[u] >= p.v_thresh && !blocked;
    vpr[u] = vs[u];
    // 6. reset
    if fired {
        vs[u] = p.v_reset;
        ad[u] += p.adapt_inc;
        rf[u] = p.refractory_ms;
    }
    fired
}

/// Counter-based per-(step, neuron) noise: deterministic and independent of the
/// order neurons are visited, so results do not depend on active-set size.
pub struct NoiseStream {
    scale: f32,
    t: u64,
}

impl NoiseStream {
    #[inline]
    pub fn new(t: u64, scale: f32) -> Self {
        NoiseStream { scale, t }
    }
    #[inline]
    pub fn next(&mut self, u: usize) -> f32 {
        if self.scale == 0.0 {
            return 0.0;
        }
        let mut z = (u as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.t.wrapping_mul(0xD1B5_4A32_D192_ED03));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u1 = ((z >> 11) as f64 + 1.0) * (1.0 / 9_007_199_254_740_992.0);
        let z2 = z.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let u2 = ((z2 >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
        let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (g as f32) * self.scale
    }
}

impl Drive {
    pub fn at(&self, u: usize) -> f32 {
        match self {
            Drive::None => 0.0,
            Drive::Dense(d) => d[u],
            Drive::Sparse(list) => match list.binary_search_by_key(&(u as u32), |e| e.0) {
                Ok(i) => list[i].1,
                Err(_) => 0.0,
            },
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            Drive::None => true,
            Drive::Dense(_) => false,
            Drive::Sparse(l) => l.is_empty(),
        }
    }
}
