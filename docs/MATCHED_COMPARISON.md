# Gated dendritic memory beats attention at matched depth, with fewer parameters

**Result: at matched depth (2 layers), matched width (`d=128`), matched seeds,
and matched training budget, a gated-memory stack scores 2.2127 bpc against
2.3376 for an attention stack — 0.125 bpc better with 7% fewer parameters.**
At matched *steps* against the 4-layer attention baseline the memory arm wins by
0.067 bpc with **51% of the parameters**, and it still wins after both models
have fully converged and begun to overfit.

This is the first comparison in this project that survives every control an
independent review asked for. The earlier "26% better at half the parameters"
claim was retracted (`docs/EFFICIENCY.md`); this is what replaced it after
fixing everything that was wrong with it.

## What was wrong before, and what is different now

The retracted claim sampled a mistuned transformer while it sat at the bigram
floor. The seven fixes applied here:

| problem in the retracted claim | fix applied |
|---|---|
| baseline stuck at bigram floor | warmup + grad clip; bigram floor printed; every arm far clear of it |
| one seed | up to 5 seeds, paired comparisons, sign-agreement counted |
| untested causality | two leak tests, both passed (below) |
| FLOPs presented as if wall-clock | wall-clock measured separately; FLOPs labelled as upper bounds |
| "multiscale" arm was single-bank | arm configurations printed with their real parameter counts |
| shared untuned learning rate | per-family LR sweep at three points |
| no convergence check | both arms trained to 5000 steps with curves logged |

## Method

Char-level TinyShakespeare, `ctx=512`, `d=128`, batch 16, AdamW with 100-step
linear warmup, gradient clipping at 1.0, weight decay 0.01. Unigram floor
4.8292 bpc, **bigram floor 3.5806 bpc** — every arm below is well clear of the
floor, so all have learned to use context.

`experiments/parallel_columns.py`, `experiments/causality_check.py`,
`scratch/{converge.py,width_control.py}`.

## The three comparisons, in order of how much they can carry

### 1. Matched depth — the clean one

Only the mixing primitive differs. Same depth (2), same width (128), same seeds
(2, 3, 4), same everything else:

| arm | params | bpc (mean of seeds 2,3,4) | sd |
|---|---|---|---|
| `depth2_mem` (gated memory) | **445,440** | **2.2127** | 0.0070 |
| `depth2_attn` (attention) | 478,976 | 2.3376 | 0.0039 |

**Gated memory is 0.1249 bpc better with 7% fewer parameters.** All five seeds
run for `depth2_mem` land in 2.1839-2.2173, entirely below `depth2_attn`'s
2.334-2.343, so the separation is well outside seed spread.

### 2. Matched steps against the 4-layer baseline

| arm | params | bpc @1500 steps | bpc @5000 steps (best) |
|---|---|---|---|
| `A_attention` (4 layers) | 875,520 | 2.2554 | 2.2290 @ step 2250 |
| `depth2_mem` (2 layers) | **445,440** | **2.1883** | **2.1700 @ step 3750** |

At 1500 matched steps, memory is 0.0671 bpc better with **50.9% of the
parameters**. Comparing best-observed validation over the full run, the gap is
0.0590 bpc. Note the depth is not matched in this comparison (2 vs 4), so it is
reported second and should not be the headline.

### 3. Matched parameters, 3 seeds, 3000 steps

| arm | params | FLOPs/tok | bpc | tok/s |
|---|---|---|---|---|
| `A_attention` | 875,520 | 7,914,240 | 2.2491 ± 0.0107 | 52,605 |
| `B_gated` | 808,448 | 4,402,944 | **2.2262 ± 0.0118** | **97,351** |

Gated memory is 0.0229 bpc better on 7.7% fewer parameters and 44% fewer FLOPs,
at **1.85x the throughput**. The bpc gap is small relative to seed spread, so
the throughput figure is the more robust part of this row.

## Controls that make the above hold up

**Causality — both tests passed** (`experiments/causality_check.py`). Perturbing
the token at position `p` leaves the logits at every position `< p` changed by
**exactly 0.000e+00**, at `p` = 16 and `p` = 31, for both memory arms. And
trained on i.i.d. uniform characters, where nothing is learnable, both arms
converge to 6.027 bpc against the uniform entropy of 6.0224 — sitting slightly
*above* it, as they must. A leak would put them below. Neither does.

**The learning rate did not handicap attention.** This was the strongest
remaining objection: both families shared `lr=1e-3`, and if attention were off
its optimum the comparison would measure tuning history rather than
architecture. Per-family sweep at 1500 steps:

| lr | `A_attention` | `depth2_mem` |
|---|---|---|
| 3e-4 | 2.5362 | 2.3481 |
| **1e-3** | **2.2554** | **2.1883** |
| 3e-3 | 2.2878 | 2.2008 |

**Both families are optimal at the same learning rate**, and both are worse on
either side of it. The shared setting was not a handicap to either arm.

**Neither model was still descending.** The concern was that a 4-layer
transformer at ~12M tokens is undertrained and would catch up. It does not — it
*overfits*. Over 5000 steps, `A_attention`'s train bpc falls to 1.4681 while
validation degrades from its 2.2290 best to **2.4048**, ending *worse than where
it started improving*. `depth2_mem` is far more stable: train 1.6666, val 2.1969,
best 2.1700 at step 3750, with validation flat within ±0.02 over the last 2500
steps. The gap is therefore not a training-budget artifact.

## What this does NOT establish

- **One task.** Char-level TinyShakespeare at `ctx=512` only. This is not
  evidence about language modelling in general, and the corpus is ~1MB.
- **No dropout, and both models overfit.** The comparison is made in the
  overfitting regime, at the point where validation bottoms out. With dropout or
  a larger corpus the ranking could change; it has not been tested.
- **Single seed for the 5000-step curves.** The 5-seed evidence is at 1500
  steps (`depth2_mem`) and 3000 steps (`B_gated` vs `A_attention`).
- **Selection optimism still applies to any comparison chosen after seeing the
  table.** The matched-depth result was not pre-registered. Treat the 0.125 bpc
  figure as an estimate to be confirmed on a held-out test split, not a settled
  effect size.
- **This is a gated linear recurrence, not a brain.** `h_t = sigmoid(W_g x_t) *
  h_{t-1} + W_v x_t` is the minGRU/HGRN/RG-LRU family. No component of the
  spiking substrate is in this model — not the neuron, not k-WTA, not the local
  rule. It is trained by backpropagation through the scan. The novelty claim
  belongs to that family's authors, not to this repo.
- **`d=128` and `T=512` is launch-overhead bound on this GPU.** The throughput
  numbers do not transfer to the compute-bound regime.

## Reproduce

```
python3 experiments/causality_check.py     # leak tests, ~40s
python3 experiments/parallel_columns.py    # depth vs columns, matched depth
```

Raw runs: `/Volumes/T9/human-brain/scratch/{converge.log,width.log,cols.log,final_ab.log}`.
