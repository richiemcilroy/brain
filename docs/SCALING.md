# Scaling: what this machine can simulate, and how far that is from a brain

> Source of truth for every number below is `bench/results_scale.json`, written by
> `python3 bench/bench_scale.py` on 2026-09-11T19:46:08Z on this host:
> Apple M4 Max, 16 CPU cores / 40-core GPU, 128 GB unified memory
> (`total_ram_gb = 137.44`, macOS reports this as 128 GiB-class unified memory),
> MLX 0.31.0 / Metal, CPython 3.14.3, NumPy 2.4.1.
> Load average at start: `[10.78, 17.23, 26.68]` — **the host was not idle**
> (another local build process was running); timings are medians and the
> run-to-run spread is reported honestly below.

Claims in this document are tagged: **[measured]** = from the run above,
**[arithmetic]** = a calculation whose inputs are measured or cited,
**[literature]** = a published figure with its citation, and nothing else is allowed
to be a number.

---

## 1. The benchmark, and the trap it avoids

The simulator is clock-driven for state and spike-driven for communication: every
neuron integrates once per millisecond, but synaptic work is only performed for
neurons that spiked. That is the design that makes sparse activity matter, and it
means a benchmark's headline number depends entirely on how much the network happens
to be firing.

**MLX is lazy.** Nothing is computed until the graph is evaluated, so a naive
`t0 = perf_counter(); brain.step(); t1 = perf_counter()` times graph *construction*,
not simulation. `bench_scale.py` therefore calls `force_eval()` — `be.eval()` on all
live state arrays followed by a device synchronise — **inside every timed step**,
during warmup and during calibration, and never reads a value it has not forced.

Two independent guards run in every sweep:

| Guard | Purpose | Result on this run |
|---|---|---|
| `lazy_eval_guard` | Times the identical op chain with and without `mx.eval`; asserts the evaluated version is strictly slower. If forcing had been a no-op, the ratio would be ~1. | dispatch-only **0.00298 ms/step** vs evalled **0.18872 ms/step** = **63.3x** [measured] |
| `ms_per_step_above_plausible_floor` | Rejects any per-step time below 0.05 ms as a lazy-eval artefact | min measured 1.057 ms/step [measured] |

The 63x gap is the exact failure mode that produced a bogus 0.02 ms/step figure in an
earlier version of this project. It is now a hard check, not a comment.

A second artefact guard is `spike_accounting_consistent`: the per-step spike readout is
summed and compared against the device-side `spike_count` accumulator. At 4,000,000
neurons the absolute error was **0.0** over 26,205,080 spikes (fp32 accumulation, so the
tolerance is relative, `max(10, 1e-3 * total)`) [measured].

Determinism is checked by running the same seed twice: spike trajectories are
**identical**, and the continuous soma state differs by at most
**1.19e-07** at 20,000 neurons — one fp32 ULP, because `scatter_add` accumulates with
atomics rather than in a fixed order. The discrete output that a readout consumes is
exactly reproducible [measured].

Protocol: heterogeneous tonic drive `U(1.0, 2.0)` per neuron, plasticity disabled
(`plas.eligibility="off"`, `plasticity.enabled=False`), 20 warmup steps, then
calibrated to ~1.5 s of timed work per repetition (3 repetitions, median over all
timed steps), with `force_eval` after every step. Each configuration is measured twice:
once driven (**forced_activity**) and once with no drive at all (**spontaneous**).
Every configuration passes all five checks.

**These are upper bounds, not emergent dynamics.** The driven protocol forces a known
activity level, so it measures how fast the substrate can move data while active — not
what an emergent network would cost. The `spontaneous` companion runs are the honest
comparison, and they are damning for the sparsity story: at 1,000,000 / k=64 the forced
network fires at **3.567% active and takes 2.646 ms/step**, while the undriven network
fires **1,100x less (0.0032% active) and takes 3.178 ms/step** [measured]. Across all
eleven configurations the spontaneous run ranges 0.0008%-5.81% active against
3.567%-36.132% forced, and the per-step ratio stays between **0.64x and 1.20x** — i.e.
cost is essentially independent of activity. `mean_active_frac` is therefore reported
next to every throughput figure in this document, and no claim here of the form
"sparse spiking is cheap" is made: on this substrate, it currently is not.

---

## 2. Measured results

`RAM` is `brain.memory_bytes()` (the substrate's own accounting), in GiB;
`MLX active` is `mx.get_active_memory()`. `events/s` counts **delivered** synaptic
events, i.e. `min(spikes, buffer_capacity) * k_out` per step; `rows/s` is what the
implementation actually gathers every step, `capacity * k_out`. `pad%` is the fraction
of that gather that is buffer padding rather than real spikes. `active%` is the achieved
mean active fraction — **it belongs next to every throughput figure**, and the `!` marks
a configuration where spikes overflowed the buffer and were counted but never delivered.

| n_neurons | k_out | synapses | RAM GiB | MLX active GB | ms/step | p90 | rep spread | bio-ms/wall-s | RT factor | active% | spikes/step | delivered events/s | rows gathered/s | pad% | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 64 | 0.64M | 0.007 | 0.01 | 1.057 | 1.298 | 1.36 | **946.0** | **0.946** | 3.786 | 378.6 | 22.9M | 60.5M | 62.1 | 0 |
| 10,000 | 256 | 2.56M | 0.029 | 0.03 | 1.256 | 1.423 | 1.03 | 796.3 | 0.796 | 4.962 | 496.2 | 101.1M | 203.8M | 50.4 | 0 |
| 10,000 | 1024 | 10.24M | 0.115 | 0.12 | 1.449 | 1.622 | 1.02 | 690.0 | 0.690 | 36.132 | 3613.2 | 706.5M | 706.5M | (overflow) | **3,180,367 !** |
| 100,000 | 64 | 6.4M | 0.075 | 0.08 | 1.631 | 1.767 | 1.05 | 612.9 | 0.613 | 3.598 | 3597.7 | 141.1M | 392.3M | 64.0 | 0 |
| 100,000 | 256 | 25.6M | 0.289 | 0.31 | 1.170 | 1.287 | 1.00 | 854.4 | 0.854 | 3.777 | 3776.6 | 826.0M | 2187.2M | 62.2 | 0 |
| 100,000 | 1024 | 102.4M | 1.148 | 1.23 | 2.204 | 2.336 | 1.03 | 453.8 | 0.454 | 4.086 | 4085.6 | 1898.4M | 4646.6M | 59.1 | 0 |
| 1,000,000 | 64 | 64.0M | 0.749 | 0.81 | 2.646 | 2.805 | 1.01 | 378.0 | 0.378 | 3.567 | 35,670 | 862.9M | 2419.1M | 64.3 | 0 |
| 1,000,000 | 256 | 256.0M | 2.895 | 3.12 | 6.187 | 6.385 | 1.00 | 161.6 | 0.162 | 3.654 | 36,537 | 1511.9M | 4137.9M | 63.5 | 0 |
| 1,000,000 | 1024 | **1024.0M** | **11.478** | 12.33 | 19.805 | 20.170 | 1.01 | 50.5 | 0.0505 | 3.976 | 39,761 | **2055.8M** | 5170.4M | 60.2 | 0 |
| 4,000,000 | 64 | 256.0M | 2.995 | 3.25 | 11.737 | 12.098 | 1.01 | 85.2 | 0.0852 | 3.658 | 146,300 | 797.8M | 2181.2M | 63.4 | 0 |
| 4,000,000 | 256 | **1024.0M** | **11.578** | 12.47 | 29.877 | 30.446 | 1.01 | 33.5 | 0.0335 | 3.874 | 154,971 | 1327.9M | 3427.4M | 61.3 | 0 |
| 4,000,000 | 1024 | 4096.0M | — | — | — | — | — | — | — | — | — | — | — | — | skipped |

`4,000,000 x 1024` is **skipped**, not failed: its estimated construction peak is
**98.56 GB** against a 48 GB benchmark budget (the estimate is analytic, from the
12 bytes/synapse storage format below). Raising `--budget-gb` is the documented way to
attempt it.

Headline from the JSON:

- 11 configurations measured, 1 skipped.
- **max neurons measured: 4,000,000**; **max synapses measured: 1,024,000,000**
  (twice: 1M x 1024 and 4M x 256, both ~12.4 GB).
- **max RAM measured: 12.432 GB (11.578 GiB)** from `memory_bytes()`; MLX peak for that
  configuration was 28.74 GB, i.e. peak during construction is ~2.3x steady state.
- **best biological throughput: 946.0 ms of biological time per wall-clock second**
  (realtime factor 0.946, i.e. 1.06x slower than realtime) at 10,000 neurons.
- **max delivered synaptic events: 2.056e9 / second** at 1M x 1024.

**What that ceiling is, biologically.** Using Blue Brain's reconstructed rat somatosensory
column as the density reference (31,000 neurons / 37M synapses / 0.29 mm^3, Markram et al.
2015 [literature]), the measured 4,000,000-neuron network corresponds to **~129 cortical
columns, i.e. ~37 mm^3 (0.037 cm^3) of cortex** [arithmetic]. That is a large patch of tissue
by slice-electrophysiology standards and a vanishingly small fraction of a brain: about
1/21,500 of the human 86.1e9-neuron count — 0.0037% of a brain's neurons and 0.037 cm^3.
It is enough for population-level computations; it is not a tissue model, an
organ model, or a brain.

Measured synapse storage is exactly **12 bytes per synapse** — three int32/fp32 arrays
(`targets`, `weights`, `delays`) — confirmed by the per-array breakdown at 1.024e9
synapses: 4,096,000,000 bytes each [measured].

### What sets the cost: the padded buffer, not the activity

The substrate promises spike-driven cost, but the current implementation gathers a
dense `capacity x k_out` rectangle every step, where `capacity = spike_capacity_frac * n`
(default 0.10 * n). Three measurements show this directly:

1. **Forced vs spontaneous is nearly free.** At 10,000 / k=64 the undriven network fires
   0.320% of neurons and takes **1.15 ms/step**, versus **1.06 ms/step** when driven at
   3.786% — a factor of 1.09 [measured]. A 12x difference in activity changes cost by
   ~9%.
2. **Capacity sensitivity at 1M / 256**: scaling the spike buffer only, at identical
   activity — `capacity_frac=0.05` -> **4.679 ms/step** (21,816,843 spikes overflowed),
   `0.10` -> **6.272 ms/step**, `0.20` -> **10.284 ms/step** [measured]. Cost tracks the
   padded buffer size; sparsity is not where the time goes.
3. **Padded fraction**: 50-64% of every gather is padding (table column `pad%`).

### Known substrate defect that contaminates "emergent activity" claims

The zero-input control (noise_std=0, no drive, `reset()` so v=0 exactly) still emits
spikes:

| configuration | capacity | spikes/step | expected |
|---|---:|---:|---:|
| 100,000 / 64 | 10,000 | 32.0 | 0 |
| 1,000,000 / 64 | 100,000 | 32.0 | 0 |
| 1,000,000 / 1024 | 100,000 | 511.5 | 0 |
| 4,000,000 / 256 | 400,000 | 128.0 | 0 |

A silent network must emit zero spikes. The signature — constant in `n`, tracking
`k_out/2` of one fan-out — is a phantom drive: `Brain._compact()`
(`brain/simulator.py`) fills unused spike-buffer slots with index 0, and `Brain.step()`
then gathers `targets`/`weights`/`delays` over the *entire* padded buffer, so every
padding slot re-delivers neuron 0's real outgoing synapses. The in-code comment claims
padding is "garbage multiplied by zero weights"; the measurements disprove that.
**Consequence: activity in this substrate sits on a floor of order `k_out/2` spikes per
step**, so spike-rate and event-count figures above are contaminated by that floor.
This is a defect report, not a fix — `brain/` is outside this task's ownership.

### Honesty about the timing environment

All figures are medians over 1200 timed steps (or 135 for 4M x 256), and the rep-to-rep
spread is 1.00-1.36x, but the host was shared: an independent earlier full run on the
same machine reported e.g. 4M x 256 at 32.13 ms/step where this run reports 29.88 ms
[measured; the earlier run's JSON was superseded by this one]. Per-configuration comparisons should stay within a single JSON.
The headline throughput is also bounded by the measured memory system: a streaming
triad probe (`c=(a+b)/2` over 2^26 fp32 elements x 20 reps) reaches
**257.5 GB/s** on this machine [measured]. That is a best case: random-access gathers
over the synapse tables are strictly worse.

---

## 3. The gap to a human brain

### 3.1 Neuron count

| | Value | Source |
|---|---|---|
| Human brain neurons | **86.1e9** | Azevedo et al. 2009, *J Comp Neurol* 513:532 [literature] |
| Largest measured here | **4.0e6** | this run [measured] |
| Gap | 86.1e9 / 4.0e6 = 2.15e4 | **[arithmetic] = 10^4.33x** |

### 3.2 Synapse count

| | Value | Source |
|---|---|---|
| Human cortical synapses | **~1e14 to 1e15** | standard estimate, task-assigned range [literature] |
| Largest measured here | **1.024e9** | twice, at 1M x 1024 and 4M x 256 [measured] |
| Gap at 1e14 | 9.77e4 | **[arithmetic] = 10^4.99x** |
| Gap at 1e15 | 9.77e5 | **[arithmetic] = 10^5.99x** |

### 3.3 Memory

The substrate stores synapses as three dense fp32/int32 arrays: `targets`, `weights`,
`delays` = **12 bytes per synapse**, measured exactly at 1.024e9 synapses
(3 x 4,096,000,000 bytes) [measured]. The largest measured configuration therefore
consumes **12.432 GB (11.578 GiB)**.

| | Value | Source |
|---|---|---|
| Human synapses in this format | 1e14 x 12 B = **1.2 PB**; 1e15 x 12 B = **12 PB** | [arithmetic] |
| Measured largest | 12.432 GB = 1.2432e10 B | [measured] |
| Gap | 9.65e4 to 9.65e5 | **[arithmetic] = 10^4.98x to 10^5.98x** |

Even the *biological* storage requirement alone is at least 10^5 times this machine's
memory. A more direct statement of the local ceiling: the OS reports a recommended
working set of **115.45 GB** [measured]; if essentially all of it were devoted to
synapse arrays at 12 B/synapse, that is **9.62e9 synapses ≈ 10^9.98**, i.e. still a
factor of **~1.0e5 (10^5.02)** short of 1e15. Spending that memory on neurons instead of
fan-out would buy ~1.5e8 neurons at k=64 or ~3.8e7 at k=256 — still **10^2.76 to 10^3.36
short of 86.1e9**. (All figures in this paragraph are [arithmetic] on the measured
12 B/synapse and measured working-set limit, not measurements.)

### 3.4 Time

Measured per-synapse cost at the largest measured configuration: 19.805 ms of wall clock
per 1 ms of biological time for 1.024e9 synapses, i.e. **1.93e-8 ms per synapse per
step** [measured].

Linear extrapolation to human synapse counts — which is *optimistic*, because it assumes
perfect scaling, zero communication cost, and no bandwidth wall:

| Synapses | Wall clock per 1 s biological | Realtime factor | Wall clock per 1 ms biological |
|---|---:|---:|---:|
| 1.024e9 (measured) | 19.8 s | 0.0505 | 19.8 ms |
| 1e14 | **1.93e6 s ≈ 22.4 days** | 5.2e-7 | 1934 s ≈ 32 min |
| 1e15 | **1.93e7 s ≈ 224 days** | 5.2e-8 | 19,340 s ≈ 5.4 h |

[arithmetic from measured]

Sanity check against real hardware: the K computer's 1.86e9-neuron / 11.1e12-synapse run
took 2481.66 s per biological second [literature]. This laptop's extrapolation to
11.1e12 synapses is ~2.1e5 s per biological second — **~87x slower than the K computer's
actual result**, which is the expected direction: K used 82,944 processors and 1.07 PB of
RAM, and one M4 Max is not 82,944 processors. Convergent evidence, not a defect in the
extrapolation.

### 3.5 Bandwidth and energy

- **Bandwidth wall.** A human-scale network with ~1e15 synaptic events/s would need to
  move 1e15 x 12 B = 1.2e16 B/s of synapse data even in the most optimistic
  representation, against a *measured* streaming ceiling of **257.5 GB/s** on this
  machine [measured] — a shortfall of **4.7e4 (10^4.67x)**, before accounting for the
  fact that gathers are random-access and slower than streaming [arithmetic].
- **Energy proxy.** The substrate's own accounting (see `brain/simulator.py`) notes that
  a neuromorphic SynOp costs ~23.6 pJ on Loihi versus ~1-100 fJ for a biological
  synapse, and that ~59% of the brain's energy budget goes to synaptic signalling
  (Attwell & Laughlin 2001) [literature]. At 23.6 pJ per operation, a human-scale
  event rate of 1e15 events/s would be **~2.4e4 W = 24 kW**, i.e. ~10^3 times the
  brain's ~20 W [literature + arithmetic]. "Spiking is cheap" is not a free claim, and
  this repository does not make it.

---

## 4. What large-scale brain simulation has actually cost (literature anchors)

| System | Scale | Speed | Notes |
|---|---|---|---|
| Blue Brain rat somatosensory cortical column, Markram et al. 2015, *Cell* 163:456 | **31,000 neurons, 37M synapses, 0.29 mm^3** | — | ~1 mm^3 of cortex is ~1e5 neurons; the reconstruction and simulation campaign required leadership-class supercomputer resources [literature] |
| K computer, Kunkel et al. 2014, *Front Neuroinform* 8:78; Jordan et al. 2018, *Front Neuroinform* 12:2 | **1.86e9 neurons, 11.1e12 synapses**, 1.07 PB RAM, 82,944 processors | **2481.66 s per 1 s biological** (realtime factor 4.0e-4) | multicompartment neurons with plastic synapses; the largest brain simulation of its kind [literature] |
| Whole-human-scale on GPU cluster (DTB), 2023 | **86e9 neurons, 47.8e12 synapses**, 3503 nodes / 14,012 GPUs | **65-119 s per 1 s biological** (realtime factor ~0.008-0.015) | static connectivity, no plasticity at that scale; human-scale only as a frozen HPC artifact [literature] |
| Human Brain Project | ~**EUR 607M**, 10 years, ~150 institutions | — | the consortium's own conclusion: a full replica of the brain is *"neither achievable nor does it seem of clear practical use"* (HBP white paper, 2023) [literature] |
| **This laptop, this run** | **4e6 neurons, 1.024e9 synapses, 12.4 GB** | **33.5-946 bio-ms per wall-s** (RT 0.034-0.946) | two-compartment dCaAP point neurons, no plasticity at scale; the fidelity gap cuts both ways [measured] |

The fidelity comparison is stated explicitly because it would otherwise be a
dishonest comparison: the K and DTB runs simulated richer neuron models with synaptic
plasticity. The 4e6-neuron figure here is a *population-dynamics-scale* model, not an
equivalent of those runs, and the honest local claim is about mechanism-and-algorithm
experiments, not about fidelity.

---

## 5. Plain statement: a full human-brain replica is out of reach here

**A full human-brain replica on one laptop is arithmetically out of reach by many orders
of magnitude.** On the measured numbers in this document the shortfalls are:

- **10^4.33x** in neurons,
- **10^4.99x to 10^5.99x** in synapses,
- **10^4.98x to 10^5.98x** in memory for this storage format,
- **10^6.3x to 10^7.3x** in wall-clock time, assuming implausibly perfect scaling,
- **10^4.67x** in memory bandwidth even at the measured streaming ceiling.

The Human Brain Project spent roughly EUR 607M over ten years and concluded that a full
replica was neither achievable nor of clear practical use. Blue Brain's *single*
cortical column — 31,000 neurons, 0.29 mm^3 — was already supercomputer-scale. Any
proposal that a 128 GB laptop should succeed where those programmes concluded otherwise
is a proposal to repeat a programme that has already run and already reported.

**What this platform therefore is:** a mechanism-and-algorithm testbed. Its value is
that it runs on a laptop, in seconds, for a specific class of questions: does a
dendritic nonlinearity change what a population can compute; does a local plasticity
rule improve retention under task sequences; how do sparsity and energy proxies trade
against a dense baseline. Those questions are answerable *now* at 1e6-4e6 neurons
(measured), and their answers do not require 86.1e9 neurons. The scaling roadmap below
says what each further order of magnitude would buy — and the honest position is that
several of them would buy very little science for a great deal of engineering.

---

## 6. Roadmap: what each order of magnitude would require, and what it would answer

Costs are anchored on the measured storage format (12 B/synapse) and the measured
per-synapse time (1.93e-8 ms/step at 1.024e9 synapses, single GPU). All forward
projections are [arithmetic]; the "measured now" rows are [measured].

| Scale (neurons / synapses) | Storage at 12 B/syn | Wall clock per bio-second | What it requires | What question it would actually answer |
|---|---|---|---|---|
| **1e4 - 1e6 / 1e6 - 1e8** | 0.01 - 1.2 GB | 1.1 - 6 s | *measured now* | Mechanism ablations (dCaAP vs linear dendrite), local plasticity rules, small continual-learning benchmarks. Cheap enough to run many seeds. |
| **1e7 / 1e9** | 12.4 GB | 30 s | *measured now at 1.024e9 synapses* — this is the practical laptop ceiling | Population-scale motifs, multi-region interactions, whether continual-learning results survive scale-up; the largest regime where a full sweep is a coffee break, not a day. |
| **1e8 / 1e10** | ~120 GB | ~3 min | More unified memory than this laptop (256-512 GB workstation or 2-4 GPUs with state partitioning); **or** a storage-format change: int8 weights + int16 targets + 8-bit delays -> 4-5 B/syn, cutting the footprint ~2.5x; and the padded gather must become a genuine scatter, or cost stays capacity-bound | ~1 cm^3 of cortex by Blue Brain density (3,200 columns). Enough for realistic laminar/long-range structure with local learning **enabled at scale**, which is the first configuration where the paradigm's own claim (local plasticity competing with backprop) can be tested at a size that is not toy. |
| **1e9 / 1e11** | ~1.2 TB | ~32 min | Multi-node: 8+ accelerators with 100+ GB each and a fast interconnect for spike exchange; event-driven scatter as a hard requirement; compressed synapse format (<=4 B/syn) or the memory bus is spent just streaming the table | ~10 cm^3 of cortex. Mesoscale circuits, whole-region dynamics, and graph-structure effects (are the results still just re-scaled small-network behaviour?). This is where the engineering cost starts to exceed the scientific return for a local project. |
| **1e10 / 1e12** | ~12 TB | ~5.4 h | A real cluster: tens of accelerators, 1-2 TB RAM aggregate, partitioned connectivity with distributed routing, sub-linear communication; sparsity assumptions must hold *or* the event rate alone exceeds the interconnect | Whole-structure dynamics at reduced fidelity. Whether large-scale state phenomena (global oscillations, sleep-wake-like regimes) require the full structure or emerge at 1e9. |
| **1e11 / 1e13** | ~120 TB | ~2.2 days | HPC scale — 1,000+ accelerators, 100+ GB/s bisection bandwidth, and a research group to keep it alive | Mostly about simulation software, not neuroscience. Two orders of magnitude that buy fidelity, not new mechanisms. |
| **8.61e10 / 1e14 - 1e15 (human brain)** | **1.2 - 12 PB** | **22 - 224 days** (linear extrapolation, single GPU, no communication cost) | What the K computer and DTB actually used: 82,944 processors / 1.07 PB, or 14,012 GPUs — and those runs either sacrificed plasticity or accepted ~65-2500x slower than realtime | Nothing that can be asked locally; HBP's conclusion after ~EUR 607M is the relevant prior. |

Note the pattern in the table, which is the honest strategic point: the first two rows
cost nothing, the middle rows cost hardware, and the last rows cost *programmes*. The
scientific return per order of magnitude is highest at the small end — which is why the
platform is framed as a mechanism testbed.

---

## 7. What would falsify this analysis

This section exists so the claims above are falsifiable rather than rhetorical. The
"out of reach" conclusion breaks if any of the following is demonstrated:

1. **Storage collapse.** A synapse representation at <=0.1 B/synapse *at rest* with
   O(1) random access (compressed/sparse formats, or in-memory neuromorphic weights).
   That is a 120x reduction versus the measured 12 B/syn — it would move the 1e15-synapse
   footprint from ~10^5x this machine's memory to ~10^3x. Not demonstrated here.
2. **Event-driven cost.** Fixing the padded-gather defect so per-step cost is
   proportional to spikes rather than to `capacity`, and measuring the resulting
   throughput at 1-5% activity. **Discriminating test**: at 10,000 / k=64 the measured
   cost is 1.057 ms/step with 3.786% active [measured], and the spontaneous network at
   0.320% active costs 1.15 ms/step [measured] — i.e. a 12x activity difference changes
   cost by ~9%. A genuinely event-driven gather would remove the padding: at 3.786% active
   with a 10% buffer it gathers **~2.6x more rows than there are spikes** (`pad%` = 62.1),
   so the realistic per-step gain from fixing this is ~2-3x, rising to ~31x only if activity
   fell to the spontaneous level (0.320%). If cost still does not track the spike count after
   that fix, the sparsity premise fails on this hardware.
3. **Effective-synapse argument.** Evidence that cortical computation depends on a small
   effective subset of synapses (say 1e6-1e8 functionally dominant weights), so that a
   laptop-scale model can be *functionally* faithful while structurally 10^5 short.
   This is the strongest possible falsifier and cannot be settled by simulation of the
   brain; it can only be attacked by showing that a laptop-scale model matches a
   larger-scale model on the capability metrics that matter.
4. **Capability saturation.** If the repo's continual-learning and energy-proxy
   comparisons show the same result at 1e6 and 1e7 neurons (a measured scaling exponent
   near zero), then the neuron-count gap stops being an argument about capability and
   becomes one about literal replica-building only.
5. **Neuroscience says the scale is the wrong target.** Independent of hardware: if the
   mechanisms that matter are local (dendritic nonlinearity + local plasticity +
   sparse coding), then the 86.1e9 figure is a fact about brains, not a requirement for
   an AI paradigm. The current experiments in this repo are small steps toward testing
   that; they do not yet establish it.

What would *not* falsify this analysis, and should not be presented as progress:
extrapolating throughput from a smaller network, quoting the 0.02 ms/step lazy-eval
artefact, or reporting sparse-network efficiency without the achieved active fraction
next to it. Every throughput figure in this document states its active fraction and its
measured buffer padding for exactly that reason.

---

## 8. Reproducing

```bash
cd /Users/richie/Documents/github/human-brain
python3 bench/bench_scale.py                 # full sweep, writes bench/results_scale.json
python3 bench/bench_scale.py --quick         # two configurations
python3 bench/bench_scale.py --sweep "4000000:256"   # single configuration
python3 bench/bench_scale.py --budget-gb 110         # attempt 4M x 1024 (~98.6 GB peak)
```

Outputs: a printed table (ms/step, bio-ms/wall-s, realtime factor, active fraction,
spikes/step, delivered events/s, gathered rows/s, overflow, RAM) and
`bench/results_scale.json` containing protocol, host/load average, substrate file hashes,
the lazy-eval and determinism guards, per-configuration medians and rep spreads,
zero-input controls, capacity sensitivity, and machine-generated findings.

Units: `RAM` in the JSON's principal figures is `memory_bytes()/2^30` (GiB); the
`headline` block reports decimal GB, hence 11.578 GiB / 12.432 GB for the same
configuration. Realtime factor = biological seconds per wall-clock second.
