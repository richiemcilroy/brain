# Working memory lives in the dendrite

**Result: in this substrate, working memory across a silent delay is carried by
the dendritic membrane time constant. It is not carried by recurrence, not by
adaptation, and not by the soma. A single scalar per neuron buys a memory window
of roughly `tau_dend`, at a cost of O(N) state and zero synapses.**

This is the first mechanism in the project that does something a random
projection demonstrably cannot, and it was found by measuring which time
constant the memory actually tracks — three other candidates were tested and
falsified first.

## The task

A delayed-response task that makes the requirement explicit:

```
[ sample stimulus 40 ms ] -> [ SILENT delay D ms ] -> [ readout window 40 ms ]
```

The readout sees spike counts **only from the final 40 ms, during which the
input is identically zero**. Nothing about the stimulus can be read off the
present input. Any accuracy above chance must come from state the network held
across the silent delay. Six stimuli, 240 trials, chance = 0.167.

## What does NOT carry the memory

Each candidate below was tested by sweeping the relevant time constant and
checking whether the memory window moved. It is the falsification of these three
that makes the fourth result meaningful.

| candidate | manipulation | outcome |
|---|---|---|
| **recurrence** | zero the entire recurrent weight matrix | **identical accuracy at every delay** — recurrence contributes nothing |
| **adaptation** | `tau_adapt` 100 → 3000 ms | **no effect at all**; memory still dies between 0 and 50 ms |
| **soma** | `tau_soma` 20 → 2000 ms | **worse**: delay-0 accuracy collapses from 1.000 to 0.170 |
| **dendrite** | `tau_dend` 10 → 800 ms | **memory window scales, monotonically** |

The recurrence result is the striking one. Zeroing every recurrent weight leaves
the answer bit-identical, which says the network's "memory" has nothing to do
with the network — it is a property of single neurons.

## What does carry it — measured

| `tau_dend` | delay 0 | 50 ms | 100 ms | 300 ms | 600 ms |
|---|---|---|---|---|---|
| 10 ms | 1.000 | 0.170 | 0.170 | 0.170 | 0.170 |
| 50 ms | 1.000 | 1.000 | 1.000 | 0.170 | 0.170 |
| 200 ms | 1.000 | 1.000 | 1.000 | **1.000** | 0.170 |
| 800 ms | 1.000 | 1.000 | 1.000 | **1.000** | 0.170 |

(`adapt_inc` 0.05; accuracy on a held-out half of the trials, best ridge over a
regularisation sweep.)

The memory window tracks `tau_dend` and then **saturates** — 200 ms and 800 ms
both reach 300 ms but neither reaches 600 ms. So this mechanism has a ceiling
around a few hundred milliseconds. That ceiling is the honest limit of the
result: a passive dendritic trace is a *decaying* store, and past a few time
constants there is nothing left to read.

## Why this happens

The dendrite feeds the soma through `phi(v_dend)` (see `brain/neurons.py`).
After the stimulus ends, `v_dend` decays with `tau_dend` but keeps driving the
soma, so neurons that were strongly driven continue to fire early in the silent
window and then fall silent. The readout is therefore decoding *when* each
neuron fires — a latency code over a decaying trace — rather than reading a
sustained representation.

This is why a longer `tau_soma` hurts. The soma is the readout's substrate, not
the memory's; stretching it integrates the trace away and destroys the latency
structure. The memory and the readout want different time constants, and this
architecture can only give them one each.

## Why this matters for the paradigm

Conventional working-memory networks — including every recurrent spiking model
in the literature — buy memory with **synapses**: O(N·k) parameters, learned by
some credit-assignment rule, with all the cost and instability that implies.

This mechanism buys it with **one scalar per neuron**. No synapses, no learning,
no credit assignment, no recurrence. Measured: ~300 ms of memory from a single
time constant, with the recurrent weight matrix set to zero.

That is a genuinely different cost structure, and it is cheap enough to be worth
combining with something else:

* It is **passive**, so it decays. It cannot be trained to hold longer.
* To go past the ceiling you need *active* maintenance, which means recurrence —
  and recurrence is exactly the part that currently does nothing measurable.
* So the natural architecture is not "recurrence instead of dendrites". It is
  **dendritic traces for cheap short-term store, plus learned recurrence for
  active maintenance on top.** This experiment shows the first half works and
  the second half is inert, which localises the work precisely.


## UPDATE — an important confound, found afterwards

The `tau_dend` sweep above was run with the default `dend_scale=1.0`. That
turns out to be a fragile operating point (see
`docs/NEURON_OPERATING_POINT.md`): at `dend_scale=1.0` with a drive of 6.0 the
tuning curve attenuates the input ~25x, and a population can emit **zero
spikes**.

A retest at a **calibrated** operating point (`dend_scale=6.0`, delay 600 ms,
`tau_dend=10 ms`) shows:

| arm | spikes | accuracy |
|---|---|---|
| plasticity OFF | 13,680 | 0.150 |
| plasticity ON | 13,680 | 0.150 |

Chance is 0.167. So at a matched operating point there is **no active
maintenance**, and plasticity changes nothing — not even the spike count.

The positive control for that retest **failed**: `tau_dend=200 ms` at
`dend_scale=6.0` emitted **0 spikes**, because a slower dendrite charges less
within a 40 ms stimulus and never reaches threshold. So `tau_dend` does not only
control memory duration; it also controls whether the neuron fires at all.

**What this means for the result above.** The memory window did scale with
`tau_dend`, and every arm in that sweep reached 1.000 at delay 0, so the
neurons were firing in all of them. But because `tau_dend` also moves the
operating point, the sweep conflates *how long the trace lasts* with *how
strongly the trace drives the soma*. The scaling is real; the claim that
`tau_dend` is a clean, tunable memory knob is **not yet established** and needs
a re-run where the operating point is re-calibrated at each `tau_dend` so that
spike counts are matched across arms.

Until that is done, treat the duration numbers as measured but the mechanism
attribution as provisional.

## Caveats

- Six stimuli, 240 trials, held-out half. Enough to establish the ordering and
  the saturation; not enough to resolve small differences.
- The readout is a closed-form ridge probe, not a biologically plausible rule.
  The claim is about **where the information is**, not about how a neuron would
  learn to use it.
- Nothing here is trained. These are the dynamics of a fixed network, so the
  result is a statement about the substrate's physics, not about learning.
- Single seed for the stimulus set; the delay sweep is the controlled variable.
- The task is a 6-way identification, not full delayed match-to-sample with
  distractors. Distractor resistance is untested and is the obvious next test.
