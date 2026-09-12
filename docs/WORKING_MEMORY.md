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


## UPDATE — the confound, and its resolution

The `tau_dend` sweep above was run at the default `dend_scale=1.0`, which turns
out to be a fragile operating point (see `docs/NEURON_OPERATING_POINT.md`): at
`dend_scale=1.0` with a drive of 6.0 the tuning curve attenuates the input ~25x
and a population can emit **zero spikes**. A retest at a calibrated operating
point at delay 600 ms found no memory and no effect of plasticity (13,680 spikes
in both arms, accuracy 0.150 against a chance of 0.167).

Worse, `tau_dend` changes **both** how long the trace lasts **and** the operating
point: a slower dendrite charges less within a fixed 40 ms stimulus. At
`dend_scale=6.0`, `tau_dend=200 ms` emitted **0 spikes**, because the dendrite
never charged far enough to reach the tuning peak.

So the original sweep conflated *trace duration* with *drive strength*. It is
fixed by measuring, for each `tau_dend`, the dendritic potential the stimulus
actually reaches, and setting `dend_scale` to exactly that value so every arm
sits on the peak (`z = 1`) by construction. With the operating point matched:

| `tau_dend` | calibrated `dend_scale` | d=0 | 50 ms | 100 ms | 300 ms | 600 ms |
|---|---|---|---|---|---|---|
| 10 ms | 5.91 | 1.000 | 0.160 | 0.260 | 0.180 | 0.100 |
| 50 ms | 3.33 | 1.000 | **1.000** | 0.260 | 0.180 | 0.100 |
| 200 ms | 1.09 | 1.000 | **1.000** | **1.000** | 0.220 | 0.100 |

(100 trials per cell, held-out half, chance 0.167.)

The memory window now tracks `tau_dend` **monotonically and with the confound
removed**: 10 ms bridges nothing past the stimulus, 50 ms bridges 50 ms, 200 ms
bridges 100 ms. The original conclusion survives its own control.

The ceiling still holds — no arm reaches 300 ms, and 600 ms is at or below
chance everywhere. A passive decaying trace has a hard limit, and the limit is
of order `tau_dend`, i.e. a few hundred milliseconds at most.

**Superseded in part.** A later feature-count-matched retest across four
different `tau_dend` values (50/100/200/400 ms) and a random-tau control found
that the *ordering above does not survive matching feature counts*: at 1200
features, multiscale and single-tau are identical. See `docs/MULTISCALE.md` for
the matched result and its limits. The memory-window finding in this document is
reported from a 6-way task and is not the same measurement.

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


## Multiscale retest (feature-count matched)

**Verdict: the original claim is REFUTED.** The recorded result that a
multiscale `tau_dend` set scores **0.388** against **0.287** for a single
`tau_dend` on 8-way time-since-event decoding compared 1200 features against
300. At matched feature count, on the documented delay set, the difference is
**exactly zero: 1.0000 vs 1.0000** at 1200 features (paired delta +0.0000, 95%
CI [+0.0000, +0.0000]). The +0.101 was the feature count.

Script: `experiments/multiscale_fair.py` (runs from the repo root, exits 0).
Artifacts: `experiments/results/multiscale_fair.json`. Wall time 734.9 s.

### The matched comparison (documented delays, 8 classes, chance 0.125)

Every arm gets the same feature count and therefore the same readout parameter
count (`features x 8`). Three seeds; mean test accuracy, and the paired
multiscale-minus-single-best difference with its 95% CI across seeds.

| features | multiscale | single (best) | random tau | no memory | linear (same ladder) | raw input | delta ms-best | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.9389 | 0.9222 | 0.7778 | 0.1250 | 0.8333 | 0.1333 | +0.0167 | [-0.0733, +0.1066] |
| 16 | **0.9944** | 0.9611 | 0.9806 | 0.1250 | 0.9778 | 0.1389 | +0.0333 | [-0.0327, +0.0993] |
| 32 | **1.0000** | 0.9861 | 1.0000 | 0.1250 | 1.0000 | 0.1361 | +0.0139 | [-0.0005, +0.0283] |
| 64 | **1.0000** | 0.9972 | 0.9972 | 0.1250 | 1.0000 | 0.1222 | +0.0028 | [-0.0027, +0.0082] |
| 128 | 0.9944 | **0.9972** | 1.0000 | 0.1250 | 0.9972 | 0.1361 | -0.0028 | [-0.0172, +0.0116] |
| 300 | 1.0000 | 1.0000 | 1.0000 | 0.1250 | 1.0000 | 0.1667 | +0.0000 | [+0.0000, +0.0000] |
| 1200 | **1.0000** | **1.0000** | 1.0000 | 0.1250 | 1.0000 | 0.1222 | +0.0000 | [+0.0000, +0.0000] |

`single_best` picks, at each feature count, the single-tau arm (50/100/200/400 ms)
with the best **validation** accuracy; `single_400` wins at every width. The
single-tau control therefore gets its best shot, and multiscale still has no
advantage at any matched width: every paired CI includes zero.

### What this does and does not overturn

* **Refuted:** "multiscale beats a single `tau_dend` on 8-way time-since-event
  decoding", as stated. At the width the claim was made (1200), the two arms are
  identical, and at every other matched width the difference is inside noise.
* **Refuted as an explanation of that gap:** the +0.101 is reproducible as a
  *feature-count* effect. Comparing a 4x-wider multiscale arm against a
  narrow single-tau arm recovers a spurious gain (+0.0778 for 8 features vs 32;
  +0.0389 for 16 vs 64), while the matched comparison at the same widths gives
  +0.0139 and +0.0028. Both arms gain roughly equally from extra features
  (`single_best_gain_from_width_alone` 0.0639 for 8->32, 0.0361 for 16->64),
  which is exactly why the confounded comparison flattered multiscale.
* **Not established, and previously overstated:** at 8 features multiscale
  (0.9389) is above a random spread of time constants (0.7778) by +0.1611, but
  the paired 95% CI is **[-0.0159, +0.3381] - it includes zero**. With 3 seeds
  this is a *direction*, not a bounded effect. It should not be published as a
  measured sample-efficiency win; the earlier reading of that number as "real"
  is corrected here.
* **The pre-gap input carries nothing:** the raw-input control (same probe, same
  population, stimulus-window counts, class-independent by construction) sits at
  0.1222-0.1667 - chance - at every width, and `no_memory` (1 ms dendrite) is at
  exactly 0.1250 everywhere. Class information exists only in the post-gap
  state, so the decoding above is genuinely reading memory.
* **The tuning curve is not load-bearing on this task:** a *linear* dendritic
  transfer on the same ladder reaches 0.8333 / 0.9778 / 1.0000 at 8 / 16 / 1200
  features, i.e. essentially the same place. At these delays a slowly decaying
  state is readable whether the transfer is tuned or monotonic.

### A real effect exists, but on a harder task

Where the documented delays saturate a single tau, the identical comparison was
repeated with the gap set stretched 3x (0/75/150/300/525/825/1200/1650 ms), at
64 and 300 matched features. **Exploratory, not pre-registered**, so it is a
scope probe rather than the verdict:

| features | multiscale | single (best, 400 ms) | random tau | delta ms-best | 95% CI | delta ms-random | 95% CI |
|---|---|---|---|---|---|---|---|
| 64 | 0.9917 | 0.7472 | 0.9083 | **+0.2444** | [+0.2390, +0.2499] | **+0.0833** | [+0.0173, +0.1493] |
| 300 | 1.0000 | 0.7500 | 0.9944 | **+0.2500** | [+0.2500, +0.2500] | +0.0056 | [-0.0053, +0.0164] |

When a single time constant genuinely cannot span the task, the multiscale set
beats the best single tau by ~0.25 with a CI that excludes zero, and at 64
features it also beats a random spread. That is a real mechanism effect - but it
belongs to a 1.65 s delay range, not to the documented 550 ms one, and the
claim being retested was about the documented task.

### Controls that make this trustworthy

* **Zero-spike trap reproduced and asserted against.** At `dend_scale=1.0`,
  `dend_gain=1.0`, drive 6.0 a 200 ms population emits **0 spikes in 400 ms** -
  the documented silent failure. The script asserts that this trap reproduces
  and that its own operating point spikes (455 spikes / 400 ms). A third check
  is reported honestly: at this experiment's gain of 2.5 the unit-scale point
  emits 431 spikes, so the trap is gain-dependent, and calibration (not gain
  alone) is what guarantees the peak.
* **Every arm asserts non-zero measured spikes** before any accuracy is trusted.
  Arms expected to carry memory must emit window spikes; the memoryless control
  must instead emit *stimulus* spikes, so "no memory" cannot be confused with a
  dead population. Both checks abort with `SystemExit` (verified by injecting
  zeros into each path).
* **Determinism.** Same seed gives bit-identical features (MLX laziness would
  break this, so it is asserted). Two independent full runs produced identical
  per-seed accuracies in the primary and extended sweeps.
* **One fixed population per arm.** Per-neuron `tau_dend` and stimulus amplitude
  are drawn once per arm and reused across every trial and class, as in a real
  recording; only the substrate's own somatic noise differs between trials.

### Limits

* Three seeds. Enough to refute a +0.101 claim and to resolve the ~0.25
  extended-range effect; not enough to bound the low-feature direction above.
* The readout is a closed-form ridge probe, not a biologically plausible rule.
  The claim is about **where information is**, not how a neuron learns to use it.
* The primary sweep runs at `k_out=0` (no recurrent synapses) because recurrence
  was already measured inert here. An auxiliary arm with recurrence present
  (`k_out=32`, 32 features) scores 0.9389 vs 1.0000 without, so recurrence does
  not rescue the claim either.
* The task saturates: from 32 features every memory-bearing arm is at or near
  1.000, so above that the design cannot rank arms. The 1200-feature "identical"
  result is a ceiling tie, which is why the extended-range probe was added.
* See `docs/MULTISCALE.md` for the companion write-up; this section supersedes
  its statistical reading, since that document correctly flagged that paired
  confidence intervals were unavailable from the partial run it was based on.
  They are available here and are reported above.

### A degeneracy control that cuts the other way (reported because it is real)

The `no_jitter` auxiliary arm (per-neuron stimulus amplitude jitter removed;
1200 matched features, 3 seeds) is a trap that this document should not walk
into. There, multiscale scores 1.0000 and the best single tau (400 ms) collapses
to 0.2778, a difference of +0.7222 (95% CI [+0.7078, +0.7366]) that excludes
zero. **That difference is not a multiscale memory advantage and must not be
quoted as one.** With a single time constant and no per-neuron heterogeneity,
every neuron in the population is identical and receives an identical drive, so
the whole population is a rank-2 code - the measured window-count matrix has
rank 2 (of a possible 24) with only 5 distinct rows
across the 24 trial x class rows, versus rank 7 / 8 distinct
rows with jitter on. A degenerate rank-2 code cannot be linearly decoded, which
is why the single-tau arm falls to chance. `random_tau` scores exactly the same
as multiscale (1.0000 vs 1.0000, delta 0.0000) in that condition, which is the
tell: the effect is "the neurons are not identical", not "the time constants are
laddered".

Two consequences, both worth keeping:

* The *pre-registered* comparison runs with jitter on, precisely so that every
  arm is a non-degenerate population and the single-tau arm is not
  handicapped by an artefact of the simulation. The primary numbers and the
  extended-range numbers above are from that condition.
* Per-neuron heterogeneity is doing real work in this substrate, and this is the
  clearest evidence of it so far: identical neurons give a rank-2 representation
  that discards the memory, while a heterogeneous population makes the same
  memory linearly readable. That is a statement about the *code*, not about the
  time constants.
