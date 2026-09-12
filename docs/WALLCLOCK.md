# The FLOP model and the wall clock disagree, and the wall clock wins

**Result: in a gated-memory block the scan is the wall-clock bottleneck, not the
MLP.** The primitive that owns 72.7% of the multiply-accumulates (the MLP) takes
less time than the primitive that owns 27.3% of them (the scan). At `d=128`,
`ctx=512`, `B=16` the memory block is **1.17x faster** than the attention block
in wall clock while having **1.8x fewer MACs**.

Measured by `experiments/wallclock_split.py`; raw numbers in
`experiments/results/wallclock_split.json`. Re-run: `python3 experiments/wallclock_split.py`.

## Correction to an earlier version of this document

The first version of this file reported the memory arm's MAC advantage as
**2.9x** and the wall-clock advantage as 1.19x. **The 2.9x was wrong.** The
attention row was charged `8d² + 4Td`, which is this repo's *FLOP* formula — it
already doubles the count — while the memory and MLP rows were charged true 1x
MACs. That mixed units and overstated attention by exactly 2x. Corrected, the
advantage is **1.82x** MACs (attention computing the full masked `T x T`
matrix) or **1.46x** if the kernel skips masked work. Caught by an independent
review; the corrected formula is in the script with a comment saying why.

The *ordering* inside the memory block — scan dominates, MLP does not — is
unaffected by that error and is the part that matters.

It also reported LayerNorm at 26% of the block and implied components sum past
100% because they "overlap on the GPU". That is wrong too: each component is
timed in its own serial call, and every call pays a fixed dispatch cost. A fused
kernel over a 4 MB tensor cannot need 0.48 ms of GPU time. The dispatch floor is
now measured and reported alongside the numbers; component times are **not
additive**.

## The measurement

M4 Max, MLX 0.31.0, `d=128`, `B=16`, `T=512`, min-of-30, warmup discarded,
`mx.eval` inside every timed region (MLX is lazy — an unevaluated graph times at
dispatch speed and produces meaningless numbers).

| component | MACs/token | min ms | share of block wall | share of block MACs |
|---|---|---|---|---|
| dispatch floor (trivial op) | — | 0.158 | — | — |
| LayerNorm | 0 | 0.476 | 38.7% (mostly dispatch) | 0.0% |
| `GatedMemory(banks=1)` | 49,152 | 0.979 | 79.5% | 27.3% |
| MLP `d->4d->d` (+ its LN) | 131,072 | 0.595 | 48.3% | **72.7%** |
| **`Block(mem)` total** | **180,224** | **1.231** | 100% | 100% |
| `Block(attn)` total | 327,680 | 1.440 | 117.0% | 181.8% |

Every component row includes the dispatch floor, so subtract roughly 0.16 ms
from each before comparing them. Even after subtracting it the scan is the
largest single term in the block. Run-to-run variation at this size is roughly
5%, so treat the pattern as the result and the third digit as noise.

## Why this matters

Every efficiency argument in this project before now was made in MACs or FLOPs,
including the retracted 26% claim, whose own post-mortem lists "FLOPs presented
as if wall-clock" among its seven sins. This document made the same class of
error in its first version and was corrected by review. The lesson is the one
already written in `docs/EFFICIENCY.md`: measure, do not count.

The mechanism is execution character:

- the memory arm's context term is an **elementwise recurrence** — many small
  kernels, dispatch-bound at this size;
- the attention arm's context term is a **dense matmul**, which runs near peak.

At `d=128`, `T=512` the memory arm's arithmetic advantage (1.8x) does **not**
translate into a wall-clock advantage anywhere near that size (1.17x), because
the scan is dispatch-bound rather than compute-bound. This is consistent with
`docs/EFFICIENCY.md` §9, where the wall-clock ratio improves with context
(0.46x at T=512 → 3.15x at T=4096 → 16.26x at T=32768 at `d=512`): the advantage
appears only once the attention matmul is large enough to dominate the scan's
per-op inefficiency.

**Corollary:** at this scale, wall clock measures kernel count, not arithmetic.
Any efficiency claim at `d=128` must say so.

## Two optimisations killed by this measurement

Both were tried because the MAC model said they should win. Neither does.

**1. k-WTA top-k sparse MLP: fewer MACs, 2.3x slower.**

| arm | MACs/token | min ms | vs dense |
|---|---|---|---|
| dense MLP `h=512` | 131,072 | 0.385 | 1.00x |
| dense MLP `h=320` (MAC-matched) | 81,920 | 0.309 | 0.80x |
| **k-WTA `h=512, k=256`** | **98,304** | **0.901** | **2.34x** |
| **k-WTA `h=512, k=64`** | **73,728** | **0.882** | **2.29x** |

Every sparse variant is slower than the dense MLP it was meant to replace,
including at 44% fewer MACs. The sort/threshold/mask machinery costs more than
the dense arithmetic it saves. **Unstructured sparsity does not cash out on
dense hardware**, which is a direct answer to the suggestion that the
substrate's biological sparsity should be imported into the language model.

**2. Log-space `O(T)` cumsum scan: 1.4-2.1x faster, numerically unsafe.**

Replacing the `O(T log T)` Hillis-Steele scan with a per-chunk log-space cumsum
is faster (1.42x at T=512, 1.66x at T=2048, 1.93x at T=8192 at chunk 64; up to
2.09x at chunk 128) and is `O(T)` rather than `O(T log T)`. It is also **wrong
at the operating point**:

| gate bias | exact ref max abs | relative error |
|---|---|---|
| 0.0 | 4.41 | **2.0e-01** |
| 4.0 | 16.7 | 4.9e-02 |
| 9.0 | 17.8 | 1.6e-04 |
| 13.0 | 23.1 | 4.4e-06 |

The configured initial decays are 0.90/0.99/0.999/0.9999, i.e. biases 2.2-9.2,
so the scan would run exactly in its worst regime. The safe version of this idea
is a different formulation: weight by `exp(cum_t − cum_s)` so every weight is at
most one and nothing is divided by a small number. Recorded here so it is not
rediscovered.

By contrast `mx.compile` on the **existing** scan is exact (max relative error
1.3e-07 against `sequential_scan`) and gives 1.34x at T=512, 1.19x at T=8192.
That is a real, safe win and it is free.

## What this does not say

- It is a **forward-pass** measurement. Backward is not included.
- It is one machine (M4 Max / Metal). The direction (elementwise suffers,
  matmul does not) is general; the numbers are not.
- It does not overturn the confirmed quality result in
  `docs/MATCHED_COMPARISON.md`. That result is about bits-per-character at
  matched training, not about speed.
- At this scale a wall-clock comparison is a comparison of **kernel counts**,
  so it should not be presented as an efficiency result in its own right.
  Independent review made exactly this point and it is accepted.
