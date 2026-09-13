# Path A: the fair kernel benchmark, and what actually survives

This supersedes `docs/RETRACTION_FUSED_BASELINE.md` with a clean measurement.

## The mistakes this had to fix

Three, all of which had produced clean-looking numbers:

1. **Wrong baseline.** Attention was hand-rolled (matmul -> `triu` mask -> softmax
   -> matmul) instead of MLX's fused `mx.fast.scaled_dot_product_attention`.
2. **Measured under load.** Load average hit 77 during earlier runs. This project
   already knows that produces junk (`docs/WALLCLOCK.md`).
3. **No sanity check.** A row where the hand-rolled arm beat the fused kernel was
   reported rather than flagged as impossible.

## What the fixed benchmark does

`experiments/fair_bench.py`:

- **Verifies before timing.** Both attention arms share their projections and
  differ *only* in the attention op, so they must compute the same function.
  Measured discrepancy: **1.6e-7 relative**. The chunked scan is checked against
  the sequential recurrence: **4.8e-7 absolute**. The run **aborts** on
  disagreement.

  This check immediately earned its keep. It caught that
  `mx.fast.scaled_dot_product_attention(mask=None)` is **unmasked** -- it attends
  to the future -- while the naive arm was causal. Those are different functions
  and timing them would have been meaningless. The fused arm now uses
  `mask="causal"`.

- **Records load per row**, so a contaminated run is identifiable afterwards.
- **Flags impossible rows**: if the naive arm beats the fused one by >15%, the
  row is marked, because a fused kernel cannot be slower than an unfused
  implementation of the same computation.

## Results (B=4, min-of-15, load 5.6-7.1, all 15 rows clean, 0 flagged)

| d | T | naive | **fused** | memory | mem/fused | winner |
|---|---|---|---|---|---|---|
| 512 | 128 | 0.388 | 0.349 | 0.483 | 1.384 | attention |
| 512 | 512 | 1.093 | 0.701 | 0.984 | 1.403 | attention |
| 512 | 1024 | 3.389 | 1.333 | 1.750 | 1.312 | attention |
| 512 | 2048 | 12.600 | 3.204 | 3.553 | 1.109 | attention |
| 512 | 4096 | 42.705 | 10.158 | 8.459 | 0.833 | **memory** |
| 2048 | 128 | 2.110 | 1.661 | 1.745 | 1.051 | attention |
| 2048 | 512 | 8.510 | 6.088 | 6.703 | 1.101 | attention |
| 2048 | 1024 | 22.901 | 12.242 | 13.831 | 1.130 | attention |
| 2048 | 2048 | 60.838 | 29.990 | 27.839 | 0.928 | **memory** |
| 2048 | 4096 | 205.444 | 84.721 | 56.581 | 0.668 | **memory** |
| 4096 | 128 | 8.013 | 7.914 | 6.637 | 0.839 | **memory** |
| 4096 | 512 | 38.196 | 31.644 | 31.852 | 1.007 | tie |
| 4096 | 1024 | 74.281 | 65.970 | 47.438 | 0.719 | **memory** |
| 4096 | 2048 | 186.511 | 135.826 | 104.292 | 0.768 | **memory** |
| 4096 | 4096 | 579.276 | 347.762 | 249.055 | 0.716 | **memory** |

Fused kernel median speedup over the hand-rolled arm: **1.56x**.
Memory faster than fused attention at **7/15** points; faster than the naive
baseline at 14/15.

## The real finding: context length decides the winner

| T | memory faster at |
|---|---|
| 128 | 1/3 widths |
| 512 | 0/3 |
| 1024 | 1/3 |
| 2048 | 2/3 |
| **4096** | **3/3** |

Memory wins **everywhere at T=4096**, loses almost everywhere at T<=1024, and is
mixed in between. That is the physically sensible direction: attention pays
`O(T^2 d)` while the memory pays `O(T d^2)` in its projections plus a linear
scan, so attention must eventually lose as `T` grows. The advantage is a
**long-context** effect, not a universal one.

## Does the crossover match theory?

The cost model is:

```
attention:  MACs/token ~ 4*d^2 (projections) + 2*T*d (scores + weighted sum)
memory:     MACs/token ~ 3*d^2 (v, gate, o) + scan over T*d
```

Attention has a `T^2` term and memory does not, so the ratio `mem/fused` should
fall as `(a*d^2 + b*T*d) / (4*d^2 + 2*T*d)`. Fitting those two free parameters
per width (least squares, 5 points each):

| d | a | b | model crossover | RMSE | measured crossover |
|---|---|---|---|---|---|
| 512 | 7.12 | 1.28 | T ~ 2217 | 0.135 | between 2048 and 4096 |
| 2048 | 4.89 | 0.36 | T ~ 1117 | 0.086 | between 1024 and 2048 |
| 4096 | 3.56 | 0.67 | negative (no crossover) | 0.085 | between 512 and 1024 |

**The model fits the measured ratios well** (RMSE 0.085-0.135 against ratios that
span 0.67-1.40), and it correctly predicts a crossover in the right region for
d=512 and d=2048. So the observed behaviour is consistent with the
`T^2`-vs-`d^2` argument, with the important correction that the constant `a` is
**above 3** (7.1, 4.9, 3.6) -- memory's per-token constant cost is higher than
the raw MAC count suggests, and it falls as width grows, i.e. the memory block
amortises better at wide `d`.

The d=4096 fit is the outlier: it prefers a model with no crossover at all, and
indeed memory already wins at T=128 there while losing (barely, 1.007) at T=512.
That single non-monotonic point is the weakest part of the dataset.

## Honest summary

**What survives.** The memory block is faster than a *fused* attention baseline at
long context: **7 of 15 points, and 3 of 3 at T=4096**. This is the first
efficiency measurement in this repository taken against a fair baseline, with
implementations verified equivalent in-run, on a machine at load < 8.

**What does not survive.** Any claim that the memory block is faster in general.
It is **slower at 8 of 15 points**, including at *every* width tested for T=512.

**What is not claimed.** No end-to-end training or inference speedup on a real
model. This measures isolated sublayers at B=4. A real transformer has LayerNorms,
residual adds, an MLP, and 16-32 such layers, and none of that is in this
measurement.

## The corrected efficiency claim

> Replacing a causal-attention sublayer with a gated multi-timescale memory
> sublayer is **slower at short context and faster at long context**. Against
> MLX's fused attention kernel at B=4, the memory block wins at every width
> tested at T=4096 (by 1.20x, 1.50x and 1.40x at d=512/2048/4096) and loses at
> every width at T=512 (by 1.40x, 1.10x and 1.01x). The crossover is consistent
> with attention's `T^2` term dominating beyond roughly `T ~ 1000-2200`
> depending on width.

That is a narrower claim than "our architecture is more efficient", and it is the
one the measurements support.
