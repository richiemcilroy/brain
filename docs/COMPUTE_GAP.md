# How much compute would a full human-brain simulation need?

Every number here is one of three kinds, labelled: **[measured]** on this
machine by this repo, **[literature]** with its source named, or **[arithmetic]**
on those two. Script: `experiments/brain_compute_gap.py`; raw output in
`experiments/results/brain_compute_gap.json`.

**Short answer: ~52,000 machines' worth of memory and ~300,000 machines' worth of
throughput, on the substrate we actually built.** Both gaps are ~10^5. That is
not a "bigger cluster" gap.

## The inputs

| quantity | value | source |
|---|---|---|
| neurons | 86.1e9 | [literature] Azevedo et al. 2009, *J Comp Neurol* 513:532 |
| synapses | 6.03e14 | [arithmetic] 86.1e9 x 7e3 synapses/neuron [literature] |
| bytes per synapse | **12 exactly** | **[measured]** — verified across four configs, 12.04-12.14 B |
| peak synaptic events/s | **2.02e9** | **[measured]** — 1e6 neurons / 1.024e9 synapses, 5.61% active, 28.42 ms/step |
| RAM at that config | 11.48 GiB | [measured] |
| average firing rate | 1 Hz | [literature] working figure; **cost scales linearly with it** |

The 12 bytes/synapse is the number that decides this. It is not an estimate: we
measured it at every scale we ran.

## Three constraints, and they bind in this order

### 1. Memory — binds first, and does not care how many chips you have

    6.03e14 synapses x 12 bytes = 7.23e15 bytes = 7.2 PB
    = 52,623 machines at 128 GB each

This is the wall. Adding compute does not help: every synapse needs its weight
resident, and there is no compression scheme in this design.

For reference, the 2023 human-scale GPU run (DTB, 14,012 GPUs) used **47.8e12
synapses — about 2x fewer than the biological estimate** — and that is already
**574 TB** of weights at our 12 bytes/synapse. Their model is a simplification,
and the simplification is exactly in the expensive dimension.

### 2. Throughput — for real time

    needed at 1 Hz:  6.03e14 events/s
    we measure:      2.02e9 events/s
    deficit:         2.98e5x  = 298,235 of our machines

    -> 1 second of brain time = 3.5 days of compute on one machine
    -> 1 hour of compute       = 12.1 ms of brain time

Note the throughput gap (2.98e5) is **larger** than the memory gap (5.3e4), by
about 6x. So in a real datacentre the two would be hit at roughly similar
budgets; neither is free.

The 1 Hz assumption is doing real work here: at 0.1 Hz the deficit is 29,824x,
at 10 Hz it is 2,982,351x. **Any headline should quote the rate it assumed.**

### 3. Energy — the one biology wins outright

Our measured machine: 2.02e9 events/s on roughly 60 W, so ~29,690 pJ per
synaptic event.

| comparison | energy/event | ratio |
|---|---|---|
| this laptop | 29,690 pJ | 1x |
| Loihi 2 (neuromorphic) | 23.6 pJ | **1,258x** better |
| biological synapse | 1-100 fJ | **3e5x - 3e7x** better |

A human brain runs the entire job on ~20 W. This is the figure that makes the
whole framing look wrong: the brain is not doing an expensive computation
efficiently, it is doing a *different* computation.

*Caveat, and it matters:* the energy figures are operation-count proxies scaled
by package power, not measured synapse energy. The substrate's own docs already
say this (`docs/DECISIONS.md`): a Loihi SynOp is 23.6 pJ while a biological
synapse is 1-100 fJ, so "spiking is cheap" is **not** something we have earned
the right to assert.

## Sanity check against hardware that actually ran at this scale

| system | scale | speed | what it left out |
|---|---|---|---|
| K computer (2014), 82,944 procs | 1.86e9 neurons, 1.1e13 syn | 2,482 s per biological second | — |
| DTB (2023), 14,012 GPUs | 86e9 neurons, 47.8e12 syn | 0.008-0.015x realtime | **static connectivity, no plasticity, 2x fewer synapses** |
| **this laptop (2026)** | **4e6 neurons, 1.024e9 syn** | **0.025x realtime** | no plasticity at scale |

One genuinely encouraging fact hides in here: **per neuron, a 2026 laptop is
about 29,000x more efficient than the 2014 K computer** (6.2e-9 vs 2.2e-13
realtime-per-neuron). Hardware has improved enormously. It has not improved by
10^5 *and* simultaneously fixed the memory wall, and the biological estimate also
grew over the same period.

## What this actually means

The gap is ~10^5 in both memory and throughput, and it is the same gap for every
spiking simulator built this way. The conclusion is not "wait for hardware":

> Any approach that stores a 12-byte weight per synapse and touches every synapse
> per event is dead at this scale **by construction**. The ~10^5 gap is a
> property of the design, not of the machine.

That framing is the useful output. It says the escape route, if one exists, is
not a faster matrix multiply. It is one of:

1. **Do not store every synapse.** Sparse or procedural connectivity; this repo
   already stores 256-1024 outgoing synapses per neuron rather than the full
   matrix.
2. **Do not touch every synapse per event.** Event-driven delivery, which is
   what the Rust prototype in `experiments/rust_core/` tests — and its measured
   answer is a qualified *yes* for the neuron-state step and *no* for synaptic
   delivery at the activity levels this substrate attains (8.5x faster at zero
   activity, but ~10x **slower** at 1% activity, because the constant factor
   dominates).
3. **Do not simulate at 1 Hz.** Cost is linear in rate; cortex is sparse and
   asynchronous in a way our synchronous clock tick is not.
4. **Do not simulate a brain.** Model the function you actually need.

Option 4 is what the project pivoted to, and the falsification in
`docs/THREEFACTOR.md` is the evidence that we checked rather than assumed.

## What this does NOT say

- It does not say the brain is uncomputable or that human-scale simulation is
  impossible in principle — only that *this* design, stored and executed this
  way, is 10^5 away, and that no plausible amount of hardware closes that by
  itself.
- It does not estimate *fidelity*. A simulation can hit 86.1e9 neurons and still
  be wrong about the neuron model. The gap here is a lower bound on cost for the
  cheapest useful model, not a claim about what would suffice for a mind.
- The 7,000 synapses/neuron and 1 Hz rate are literature working figures with
  real spread; the 10^5 order of magnitude is robust to both, since cost is
  linear in each.
- "Plasticity at scale" is where all of these systems cut corners, including
  ours. The DTB run is static; ours is forward-only at 4e6 neurons.
