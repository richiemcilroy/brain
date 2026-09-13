# Retraction: the "memory is faster at every width" result used a slow baseline

**Status: RETRACTED.** `experiments/width_crossover.py` reported that the gated
memory block was faster than attention at *every* width tested, including
realistic `d=2048` and `d=4096`, with the crossover moving *down* as width grew
(the opposite of the predicted `T ~ d` scaling). That result does not survive a
fair baseline.

## What was wrong

The benchmark built attention by hand:

```
scores  = q @ k^T
mask    = mx.triu(mx.full((T, T), -1e30))   # materialised EVERY call
weights = softmax(scores + mask)
out     = weights @ v
```

MLX ships **`mx.fast.scaled_dot_product_attention`**, a fused kernel that applies
causal masking internally and never materialises the `T x T` matrix. Comparing
our block against the hand-rolled version measured *our block against a slow
implementation of attention*, not against attention.

This is the third instance of the same failure class in this project, after the
FLOP/MAC unit error (`docs/WALLCLOCK.md`) and the bigram-floor baseline
(`docs/EFFICIENCY.md`): a comparison that looked clean because the baseline was
quietly handicapped.

## The fair re-measurement

`experiments/width_crossover_fair.py` times three arms at the same width and
batch: the hand-rolled attention, the fused-kernel attention, and our memory
block. `B=4`, min-of-9, `mx.eval` inside the timing loop.

| d | T | naive attn | **fused attn** | memory | mem/fused | verdict |
|---|---|---|---|---|---|---|
| 512 | 128 | 4.00 ms | 2.07 ms | 4.87 ms | 2.36 | attention faster |
| 512 | 512 | 6.76 | 2.96 | 5.36 | 1.81 | attention faster |
| 512 | 1024 | 6.32 | 3.41 | 9.41 | 2.76 | attention faster |
| 512 | 2048 | 63.19 | 19.93 | 17.37 | 0.87 | memory faster |
| 512 | 4096 | 174.47 | 60.46 | 12.83 | 0.21 | memory faster |
| 2048 | 128 | 5.84 | 4.02 | 4.87 | 1.21 | attention faster |
| 2048 | 512 | 36.84 | 20.99 | 11.66 | 0.56 | memory faster |
| 2048 | 1024 | 94.84 | 46.08 | 71.60 | 1.55 | attention faster |
| 2048 | 2048 | 288.58 | 133.07 | 46.70 | 0.35 | memory faster |
| 2048 | 4096 | 1193.82 | 206.76 | 207.87 | 1.01 | tie |
| 4096 | 128 | 21.86 | 22.66 | 19.14 | 0.85 | memory faster |
| 4096 | 512 | 63.69 | 84.54 | 57.34 | 0.68 | memory faster |
| 4096 | 1024 | 264.89 | 188.99 | 175.23 | 0.93 | memory faster |
| 4096 | 2048 | 959.28 | 272.00 | 388.64 | 1.43 | attention faster |
| 4096 | 4096 | 1986.36 | 899.49 | 355.33 | 0.40 | memory faster |

**Summary: memory is faster than the hand-rolled baseline at 13/15 points, but
faster than the FUSED kernel at only 8/15.** The fused kernel is a median
**2.06x** faster than the hand-rolled one -- precisely the margin that flattered
the earlier result.

## What survives

**A defect found in this file's own harness, then fixed.** The first version of
`width_crossover_fair.py` passed `mask=None` to
`mx.fast.scaled_dot_product_attention`. **In MLX, `mask=None` is UNMASKED** -- it
attends to future tokens -- while the hand-rolled arm and the memory block are
both causal by construction. So the "fair" baseline was silently solving an
*easier* problem than the thing it was compared against, and every ratio in the
table below was computed across two different functions. Measured directly
(prefix-perturbation: add 7.0 to the last token, watch position 0):

| arm | leak into position 0 |
|---|---|
| `sdpa(mask=None)` | **0.1029** |
| `sdpa(mask="causal")` | **0.0000** |

This is the fourth instance of the same failure class in this project, after the
FLOP/MAC unit error, the bigram-floor baseline, and the hand-rolled baseline --
and it is the most instructive, because it was introduced *by the retraction
itself*, in the course of fixing the previous instance. The harness now calls
`verify_causality()` before timing anything and aborts if either arm is
non-causal or if the two arms disagree (`rel` must be < 1e-3; it currently
measures exactly `0.000e+00`).

**Re-measured after the fix** (same grid, corrected causal baseline): memory is
faster than the hand-rolled arm at **13/15** points and faster than the *fused
causal* arm at **9/15** (was 8/15). The median fused speedup over hand-rolled
falls from 2.06x to **1.31x**. **The retraction's direction is unchanged: the
claim still does not survive**, and the honest position is still
"slower at short context, faster at long context only". The correction moved one
point across the line, not the conclusion.

**Does not survive:** "memory is faster than attention at every width." It is
faster at some widths and context lengths and slower at others, with no clean
monotonic pattern.

**The honest position:** the advantage over a *fused* attention is
context-dependent and not established. It appears at large `T` (T >= 2048 at
d=512, T=2048 at d=2048, T=4096 at d=4096), where attention's `T^2` term must
eventually dominate, and is absent or reversed in the mid-range. That is
consistent with the cost model's *direction*, but the measured points are not
clean enough to state a crossover.

**A caveat on the measurements themselves.** Several rows are internally
implausible and must be read as noise, not signal:

- at `d=4096, T=512` the "naive" arm (63.69 ms) is FASTER than the fused arm
  (84.54 ms), which cannot be right for a fused kernel;
- the memory block swings from 388.64 ms to 355.33 ms between adjacent rows at
  `d=4096`.

The machine was running other MLX jobs concurrently, and this project has already
recorded that wall-clock numbers taken under load are junk (`docs/WALLCLOCK.md`).
**These numbers need a clean re-run on an idle machine before any magnitude is
quoted.** The retraction itself does not depend on that re-run -- the fused
baseline is the correct comparator regardless of the exact ratios -- but the
*numbers* should not be cited until the measurement is cleaned up.

## Why this matters beyond one number

The correct baseline for an efficiency claim is the **fastest available
implementation** of the thing you are comparing against, not a straightforward
one. Every kernel-level result in this repository was measured against
hand-written attention; none of them was measured against
`mx.fast.scaled_dot_product_attention`. Until that is done, **no efficiency claim
in this repo should be treated as established.**

The single highest-value next experiment is therefore not a new architecture. It
is to re-run the existing comparisons with `mx.fast.scaled_dot_product_attention`
as the attention baseline, on an idle machine, and to report the result even if
it is uniformly negative.
