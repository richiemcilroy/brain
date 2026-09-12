# The multiscale `tau_dend` advantage is a feature-count confound

**Verdict: REFUTED as originally stated.** The claim recorded in
`docs/WORKING_MEMORY.md` — that a multiscale set of dendritic time constants
scores 0.388 against 0.287 for a single time constant — compared **1200 features
against 300**. At matched feature count the difference is **exactly zero**.

There is a smaller real effect underneath, and it is worth stating precisely
because it is not the one originally claimed.

## The confounded claim

| | features | accuracy |
|---|---|---|
| multiscale `tau_dend` | 1200 | 0.388 |
| single `tau_dend` | 300 | 0.287 |
| | | **+0.101** |

The two arms differed in feature count by 4x, so the comparison could not
distinguish "a spread of time constants helps" from "more features help".

## The matched-feature-count result

Every arm now gets the *same* feature count and therefore the same readout
parameter count (`features x 8`). 3 seeds, mean test accuracy, 8-way
time-since-event decoding, **chance 0.125**:

| features | multiscale | single (best) | random tau | no memory |
|---|---|---|---|---|
| 8 | **0.9389** | 0.9222 | 0.7778 | 0.1250 |
| 16 | **0.9944** | 0.9611 | 0.9806 | 0.1250 |
| 32 | **1.0000** | 0.9861 | 1.0000 | 0.1250 |
| 64 | **1.0000** | 0.9972 | 0.9972 | 0.1250 |
| 128 | 0.9944 | **0.9972** | 1.0000 | 0.1250 |
| 300 | 1.0000 | 1.0000 | 1.0000 | 0.1250 |
| 1200 | **1.0000** | **1.0000** | 1.0000 | 0.1250 |

At the feature count where the original claim was made (1200), multiscale and
single-tau are **identical: 1.0000 vs 1.0000, a difference of 0.0000**. The
original +0.101 was the feature count.

The single-tau control is given its best shot: `single_best` selects, at each
feature count, the single-tau arm with the best *validation* accuracy.

## What survives, stated narrowly

**At low feature counts multiscale is genuinely better, and the right
comparison is against a random spread of time constants rather than a single
one.** At 8 features: multiscale 0.9389 vs random tau 0.7778 vs the best single
tau 0.9222. The structured ladder beats an unstructured spread drawn from the
same range by **+0.161**.

That is a real sample-efficiency effect — the ladder buys accuracy when the
feature budget is tight — but it decays fast: +0.161 at 8 features, +0.014 at
16, +0.000 at 32 and above. By 64 features every memory-bearing arm is at
ceiling and the comparison stops discriminating.

The honest one-line summary: **the designed ladder is a better prior than a
random spread when features are scarce, and irrelevant once they are not.**

## Why the task saturates, and what that limits

The task reaches 1.0000 for the best arm by 32 features, so above that the
measurement has no headroom and cannot rank anything. Any claim of the form
"multiscale beats single-tau" at 300 or 1200 features is measuring a ceiling,
not an effect. A harder task (more classes, distractors, longer delays) would be
needed to test the mechanism where it could still matter.

## Controls that make this trustworthy

**The documented zero-spike trap was reproduced independently.** At
`dend_scale = 1.0`, `dend_gain = 1.0`, drive 6.0 the population emits **0
spikes in 400 ms** — the exact failure recorded in
`docs/NEURON_OPERATING_POINT.md`, which once caused an experiment to compare
silence to silence. At this experiment's gain of 2.5 the same setting emits 431
spikes, and the calibrated operating point emits 455. The experiment therefore
runs at a verified-active point and asserts on it.

**`no_memory` sits at exactly chance (0.1250) at every feature count.** That arm
uses a 1 ms dendritic time constant, so its post-gap readout window is empty by
construction. Any accuracy above chance there would have meant class information
leaking outside the intended path. There is none.

**Determinism is bit-identical** over 64 values (415 total spikes), so the
sweep is reproducible.

**Only the substrate's own somatic noise varies between trials.** Each arm is
one fixed population whose per-neuron `tau_dend` and amplitude are drawn once
and reused across every trial and class, exactly as the same neurons would be
recorded in a real experiment. Resampling time constants per trial would make
each feature index mean something different on every trial, which no fixed
readout could learn — that would handicap the random arms, not the structured
ones.

## Limits

- 3 seeds, 100-odd trials per cell. Enough to establish the ordering and the
  saturation; not enough to resolve small differences at high feature count.
- Single stimulus-set seed.
- The readout is a closed-form ridge probe, not a biologically plausible rule.
  The claim is about **where information is**, not how a neuron learns to use it.
- Nothing is trained; these are the dynamics of a fixed network.
- The sweep is **partial** in the committed JSON (`meta.partial = true`); the
  high-feature-count cells completed, and the auxiliary arms listed in the
  script (`raw_input`, `linear_dend`, `recurrent_kout32`) are defined but were
  not all run. They are controls for future work, not results.

## Reproduce

```
python3 experiments/multiscale_fair.py
```

Artifacts: `experiments/results/multiscale_fair.json`.
Wall time for the reported sweep: 613.6 s.
