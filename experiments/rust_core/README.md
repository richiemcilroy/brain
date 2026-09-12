# `experiments/rust_core` — event-driven spiking core (prototype)

A Rust re-implementation of the `brain/` substrate's neuron dynamics and spike
communication path, built to answer one question:

> **Does a genuinely event-driven core make wall-clock cost track *activity*
> rather than *neuron count*?**

The Python/MLX path does not: `docs/SCALING.md` and `bench/results_scale.json`
record that an undriven network fires **1,100x fewer spikes** yet still costs
~80% as much per step, because an O(N) dense state update dominates the clock
tick.

**Read `BENCH.md` for the measured answer.** Short version: the answer is a
qualified *yes* for the neuron-state step and *no* for the synaptic delivery
step at the activity levels this substrate actually attains.

---

## Build

Build artifacts are forced onto the external SSD by `.cargo/config.toml`
(the internal disk has only a few GB free; a release build is ~50 MB and a
debug build several hundred). To use a different location, override
`CARGO_TARGET_DIR`.

```bash
cd experiments/rust_core

# Release build (required for the benchmarks; debug is 10-30x slower).
CARGO_TARGET_DIR=/Volumes/T9/human-brain/scratch/rust_target cargo build --release

BIN=/Volumes/T9/human-brain/scratch/rust_target/release/rust_core
```

If `/Volumes/T9` is not mounted, set `CARGO_TARGET_DIR` to any path with a few
hundred MB free; the source tree is self-contained.

### Toolchain used

```
rustc 1.98.1 (48a229cea 2026-09-01)
cargo 1.98.1 (797e8a9bc 2026-08-05)
```

Rust was available, so the documented C++ fallback was **not** needed.

## Run

```bash
BIN=/Volumes/T9/human-brain/scratch/rust_target/release/rust_core

$BIN sanity                    # biophysical + structural invariants; exits 1 on failure
$BIN bench-fraction            # wall-clock vs active fraction (event-driven only)
$BIN bench-scale               # throughput and RAM at 1M-4M neurons
$BIN compare FIXTURE.bin       # correctness vs a Python-exported trajectory
$BIN bench-fixture FIXTURE.bin # matched 3-way benchmark, JSON on stdout
```

### Full reproduction from a clean checkout

```bash
cd /Users/richie/Documents/github/human-brain/experiments/rust_core
export CARGO_TARGET_DIR=/Volumes/T9/human-brain/scratch/rust_target

cargo build --release
BIN=$CARGO_TARGET_DIR/release/rust_core

# 1. invariants
$BIN sanity

# 2. correctness against the Python simulator (a few thousand neurons)
python3 python/export_fixture.py /tmp/fx.bin --n 3000 --k 1024 --steps 300
$BIN compare /tmp/fx.bin

# 3. the headline measurement: Python/MLX vs Rust at matched configurations.
#    Writes bench/results.json.
python3 python/run_bench.py --configs "100000:64:0.4,1000000:64:0.4" --steps 120 --warm 20
```

`python/run_bench.py` builds each network with the Python/MLX simulator, times
it, then exports the **exact graph and initial state it timed** to a binary
fixture that the Rust core loads. Both engines therefore run identical
connectivity, identical drive and identical initial conditions, so a wall-clock
difference cannot be attributed to a different random graph.

---

## What is implemented

Faithful to `brain/neurons.py` and `brain/connectivity.py`:

| element | source | status |
|---|---|---|
| Two-compartment soma + dendrite, explicit Euler | `neurons.py::NeuronState.update` | ported, same operation order |
| dCaAP dendritic nonlinearity `phi(z) = z*exp(1-z)` | `neurons.py::dendritic_activation` | ported, incl. `linear`/`none` modes |
| Adaptive threshold | `neurons.py` `adapt` | ported |
| Refractory period | `neurons.py` `refrac` | ported, min ISI = `refractory_ms` enforced by test |
| Fixed fan-out connectivity ("block CSR") | `connectivity.py::Synapses` | same layout: `k_out` contiguous slots per neuron |
| Per-synapse integer axonal delays | `connectivity.py::Synapses` | ported |
| Delay ring buffer, `t + delay` delivery | `connectivity.py::DelayBuffer` | ported, delivered sparsely |
| Dale's principle (sign per pre-neuron) | `connectivity.py` | same construction |

**Not implemented** (deliberately out of scope for this prototype; it is a
dynamics + communication core, not a full substrate):

- k-WTA inhibition (`inhibition="kwta"`)
- three-factor plasticity, eligibility traces, homeostasis
- the readout / decoder / task stack
- dendritic external current (`i_dend` is always 0 here, as in the Python
  `Brain.step`, which passes `i_dend = zeros`) — see the constraint this places
  on the sleep optimisation in `BENCH.md`

## Design

Three things make the core event-driven.

**1. Struct-of-arrays state.** Each state variable is a separate contiguous
`Vec<f32>`, so the integrator streams memory in the only order that matters.

**2. Sparse active set.** Only neurons that can change state are integrated.
A neuron is woken by arriving synaptic current, by drive, or because the
previous step left it in a state from which it could still spike. Everything
else costs nothing.

The sleep criterion is *proved*, not tuned (`src/sim.rs::can_sleep`). With no
arriving current, the soma recurrence `w(t+1) = a*w(t) + (dt/tau)*D(t)` with
`D(t) = dend_gain*phi(v_dend) - adapt` satisfies `sup_t w(t) <= max(w(0), 0)`
whenever `D(t) <= 0` for all t. Two checks pin that down:

- `v_dend == e_dend` exactly, so `phi == 0`. True in this substrate because
  dendritic current is never applied; **checked rather than assumed**, so the
  shortcut disables itself if a caller ever introduces dendritic input.
- `adapt >= 0`, so `-adapt(t) <= 0`. Also checked.

Hence `sup v_soma <= e_leak + max(v_soma - e_leak, 0)`. If that is below
`v_threshold`, the neuron cannot fire until input arrives. This is a proof, not
a tolerance. The only floating-point approximation is the closed-form
fast-forward used on wake.

**3. Waking in closed form.** A sleeping neuron is advanced to the current time
with a handful of flops regardless of how long it slept: `v_dend`, `adapt` and
`refrac` have exact geometric/linear closed forms, and the soma's forcing by
the decaying adaptation current is a geometric series with a closed-form sum
(`src/sim.rs::fast_forward`).

Sleeping neurons hold *stale* continuous state. Spike output and delivered
current are unaffected (a neuron is fast-forwarded before it is integrated
again), but any consumer reading `v_soma` directly must call
`Sim::materialize_state()` first. This is a real API constraint and a deliberate
trade: it keeps the per-step cost activity-scaled while exact continuous state
stays available on demand.

## Correctness

`$BIN sanity` runs nine checks: refractory minimum inter-spike interval,
the rest fixed point, refractory gating, dCaAP non-monotonicity and its peak
position, delay-ring `t + delay` delivery, soma leak against the analytic Euler
solution, exactness of event-driven vs dense spikes, the noise condition that
forces dense integration, and the staleness contract of `materialize_state`.

`$BIN compare` runs both engines on the same fixture. Divergence is quantified
three ways: per-step spike-count agreement, per-neuron spike-count Pearson r,
and silent/active classifier agreement. It also runs a **negative control** with
all weights zeroed, which must *fail* to match; without that control an exact
match would be uninformative, because the comparison might simply be insensitive
to the delivery path.

See `BENCH.md` for the results of both.

## Files

```
src/sim.rs      dynamics, active set, sleep/fast-forward, delay ring
src/build.rs    synthetic connectivity (Rust-only benchmarks)
src/rng.rs      deterministic xoshiro256** (Rust-only benchmarks)
src/io.rs       Python <-> Rust binary fixture format
src/main.rs     CLI: sanity | bench-fraction | bench-scale | compare | bench-fixture
python/export_fixture.py   export a Python trajectory as a fixture
python/run_bench.py        matched-configuration Python-vs-Rust benchmark
BENCH.md        measured results and what they do and do not demonstrate
```
