# Correction: the gate decay was never fitted, and it decides the result

**Status: SUPERSEDES the headline numbers in `docs/HYBRID.md`.** Everything in
this file was measured after the bug below was found. The earlier numbers are
retained there as the record of what was claimed, with a pointer here.

## The bug

`llm_hybrid.py` claimed the gated trace's time constant was "initialised from
the MEASURED recency profile" of the attention it replaced. It was not.

```python
try:
    _, attn_w = inner(x, None, None, return_weights=True)
except Exception:
    return 0.99, None        # <-- every layer, every run
```

MLX's `Attention.__call__` does not accept `return_weights`, so that call raised
for **every** layer and **every** run, and the decay was silently the literal
0.99. The evidence is in the committed result files, where every arm of every
layer carries `recency_profile_first8: null` and `fit_cdf_l1: null`. A fit that
succeeded would have populated both. Nothing was measured; a constant was
reported as a measurement.

This is a textbook example of the failure mode the repo has hit before: a
`try/except` around a call that always fails, returning a plausible default, so
the pipeline runs and the number looks real. The `except` branch was written as a
convenience for the case "this model doesn't support attention weights" and
silently became the only branch ever taken.

## The fix, and what the real profile looks like

The profile is now computed from the layer's own `q_proj`/`k_proj` weights: build
q and k, score, causal softmax, histogram the mean attention weight by **past**
token distance, with GQA query heads routed to their kv head.

One further bug surfaced in the replacement and is worth recording, because it is
the kind that produces a *plausible* answer rather than a crash: the distance
convention was inverted (`col - row` instead of `row - col`), so every positive
distance indexed a **future** token — which the causal mask had set to exactly
zero. The profile then had all its mass at distance 0 and the fit returned the
bottom of the grid. Wrong, but not obviously wrong.

Measured profiles, layers 0/4/8/12/15 of Llama-3.2-1B:

| layer | best single-exponential decay | CDF-L1 residual |
|---|---|---|
| 0 | 0.99999 | 16.8 |
| 4 | 0.99999 | 29.6 |
| 8 | 0.99999 | 22.3 |
| 12 | 0.99999 | 44.1 |
| 15 | 0.99999 | 39.2 |

**A well-specified fit has an interior optimum and a small residual.** This one
pins at the top of the grid with a large residual at every layer: the model class
is wrong. Real attention recency is *not one exponential*. The profiles have a
recency spike on top of a broad background — attention sinks plus a diffuse
long-range component — which no single geometric kernel reproduces.

So the honest statement is: **the decay cannot be fitted from the profile at all.**
It has to be swept.

## The sweep decides the result

Frozen, no training, fixed 3000-token window, no sampling, every arm scaled to
the same output rms. The decay is chosen on a held-out **SELECT** split (60-80%
of the corpus) and reported on **VAL** (the last 10%), so the reported optimum is
not an artefact of the split it is reported on.

Layer 8 of 16, `unsloth/Llama-3.2-1B`, teacher **20.3756**, `zero` (attention
deleted) **24.1044**:

| decay | horizon | `transfer` | `random_matched` |
|---|---|---|---|
| 0.30 | 1 | 23.8166 | 27.0981 |
| 0.40 | 2 | 23.5644 | 27.2323 |
| 0.50 | 2 | 23.3080 | 27.3618 |
| 0.60 | 2 | 23.0960 | 27.5185 |
| 0.70 | 3 | **22.8744** | 27.7046 |
| 0.80 | 5 | 22.7791 | 27.9145 |
| 0.90 | 10 | 23.0019 | 28.1822 |
| 0.999 | 1000 | 23.9974 | 25.6703 |
| **0.99 (the committed default)** | 100 | **24.3495** | 27.8538 |
| 1.00 | ∞ | 23.9168 | 25.2481 |

Reading it:

- **The committed constant 0.99 is near the worst point on transfer's own curve.**
  At 0.99 the module scores 24.3495 — *worse than deleting attention* (24.1044).
  At 0.70-0.80 it scores 22.78-22.87. The single unvalidated scalar moved the
  module from a clear win to a loss.
- **The optimum is interior for `transfer`** (0.70 on select, 0.80 on val — flat
  between them) but at the **edge** for `random_matched` (0.999 select, best val
  1.0). The two arms want opposite time constants, which is direct evidence the
  transplanted weights encode real temporal structure that random weights do not.
  A random matrix prefers "never forget" — an unweighted running mean — because
  it has no temporal information to preserve.
- **`transfer` beats the best `random_matched` setting by 2.37 ppl** at their
  respective optima (22.8744 vs 25.2481), against a total zero-gap of 3.7288.
  That is **33.0% recovery vs 0.0% for the control** (the control recovers
  nothing at any decay: its best is worse than deleting attention).

## What this changes about the claims

**Strengthened.** The weight transplant is a real effect, and a stronger one than
previously reported: 33% recovery instead of 10%, and the control is worse than
deletion at *every* point on the grid, not just at one setting. The
"weights carry temporal structure" claim now has a mechanism, not just a margin.

**Weakened.** The committed headline was reported at a decay setting that is
near-worst, and the `random_output_matched` figure of 74.7101 in `HYBRID.md` came
from a different normalisation; the two controls are not comparable and the
earlier 3.2x-worse claim should be read only within its own table.

**Unchanged.** This is still not "better than a transformer": 33% recovery on one
layer of 16, still strictly worse than the teacher, no generation-quality
evidence, no end-to-end speedup. The layer-dependence in `HYBRID.md` Result 2 was
measured at the wrong decay throughout and needs re-running before it means
anything.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
HF_HOME=~/zbrain/hf DECAY_ARMS=transfer,random_matched,banks_multi \
  DECAYS=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  ~/zbrain/venv/bin/python experiments/hybrid_decay.py
```

Writes `experiments/results/hybrid_decay.json`, which records the SELECT and VAL
perplexity for every (arm, decay) pair.
