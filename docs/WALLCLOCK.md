# The FLOP model and the wall clock disagree, and the wall clock wins

**Result: in a gated-memory block, the primitive that owns 72.7% of the
multiply-accumulates (the MLP) owns 43.3% of the wall clock, and the primitive
that owns 27.3% of the MACs (the scan) owns 77.2%.** At `d=128`, `ctx=512`,
`B=16`, the memory arm has a **2.9x MAC advantage** over the attention arm but
only a **1.19x wall-clock advantage**.

Measured by `experiments/wallclock_split.py`; raw numbers in
`experiments/results/wallclock_split.json`. Re-run: `python3 experiments/wallclock_split.py`.

## The measurement

M4 Max, MLX 0.31.0, `d=128`, `B=16`, `T=512`, min-of-30, warmup discarded,
`mx.eval` inside every timed region (MLX is lazy — an unevaluated graph times at
dispatch speed and produces meaningless numbers).

| component | MACs/token | min ms | share of block wall | share of block MACs |
|---|---|---|---|---|
| LayerNorm | 0 | 0.327 | 26.4% | 0.0% |
| `GatedMemory(banks=1)` | 49,152 | 0.955 | **77.2%** | 27.3% |
| MLP `d->4d->d` (+ its LN) | 131,072 | 0.536 | **43.3%** | **72.7%** |
| **`Block(mem)` total** | **180,224** | **1.238** | 100% | 100% |
| `Block(attn)` total | 524,288 | 1.470 | 118.8% | 290.9% |

Shares sum above 100% because the components overlap on the GPU; the per-row
share is that component's time divided by the whole memory block's time.
Run-to-run variation at this size is roughly 5%, so treat the pattern as the
result and the third digit as noise. A second independent run of the same code
gave 90.4%/39.1% with the same ordering and the same conclusion.

## Why this matters

Every efficiency argument in this project before now was made in MACs or FLOPs,
including the retracted 26% claim, whose own post-mortem lists "FLOPs presented
as if wall-clock" as one of its seven sins. This is the same error at a smaller
scale, and it has the same fix: measure, do not count.

The mechanism is the execution character of the two ops:

- the memory arm's context term is an **elementwise recurrence**, bandwidth-bound;
- the attention arm's context term is a **dense matmul**, which runs near peak.

A bandwidth-bound op with 3x fewer FLOPs can be slower than a matmul with 3x
more. Here it is not slower, but the 2.9x MAC advantage collapses to 1.19x in
wall clock, so **most of the memory arm's arithmetic advantage is not
realisable on this hardware at this context length.**

This is consistent with, and explains, the earlier finding in
`docs/EFFICIENCY.md` §9 that the wall-clock ratio improves with context
(0.46x at T=512 → 3.15x at T=4096 → 16.26x at T=32768 at `d=512`): at long
context the attention matmul grows and the scan does not, so the wall-clock
advantage appears only where the arithmetic advantage is large enough to
overcome the scan's poor per-FLOP efficiency.

## Two optimisations killed by this measurement

Both were tried because the MAC model said they should win. Neither does.

**1. k-WTA top-k sparse MLP: fewer MACs, 2.3x slower.**

| arm | MACs/token | min ms | vs dense |
|---|---|---|---|
| dense MLP `h=512` | 131,072 | 0.385 | 1.00x |
| dense MLP `h=320` (FLOP-matched) | 81,920 | 0.309 | 0.80x |
| **k-WTA `h=512, k=256`** | **98,304** | **0.901** | **2.34x** |
| **k-WTA `h=512, k=64`** | **73,728** | **0.882** | **2.29x** |

Every sparse variant is slower than the dense MLP it was meant to replace,
including at 44% fewer MACs. The gather/sort/threshold machinery costs more
than the dense arithmetic it saves. **Unstructured sparsity does not cash out on
dense hardware**, which is a direct answer to the recurring suggestion that the
substrate's biological sparsity should be imported into the language model.

**2. Log-space `O(T)` cumsum scan: 1.4-2.1x faster, numerically unsafe.**

Replacing the `O(T log T)` Hillis-Steele scan with a per-chunk log-space
cumsum is genuinely faster (1.42x at T=512, 1.66x at T=2048, 1.93x at T=8192
at chunk 64; up to 2.09x at chunk 128) and is `O(T)` rather than `O(T log T)`.
It is also **wrong at the operating point**:

| gate bias | exact ref max abs | relative error |
|---|---|---|
| 0.0 | 4.41 | **2.0e-01** |
| 4.0 | 16.7 | 4.9e-02 |
| 9.0 | 17.8 | 1.6e-04 |
| 13.0 | 23.1 | 4.4e-06 |

The configured initial decays are 0.90/0.99/0.999/0.9999, i.e. biases 2.2-9.2,
so the scan would run exactly in its worst regime. Reported here so nobody
rediscovers it: the speedup is real, the numerics are not, and the fix must be
demonstrated against the exact reference at bias 0-13 rather than asserted.

By contrast `mx.compile` on the **existing** scan is exact (max relative error
1.3e-07 against `sequential_scan`) and gives 1.34x at T=512, 1.19x at T=8192.
That is a real, safe win and it is free.

## What this does not say

- It is a **forward-pass** measurement. Backward is not included, so the ratios
  are a property of the architecture's forward path, not of training.
- It is one machine (M4 Max / Metal). The specific numbers will differ on
  hardware with a different matmul-to-bandwidth ratio, though the direction
  (elementwise suffers, matmul does not) is general.
- It does not overturn the confirmed quality result in
  `docs/MATCHED_COMPARISON.md`. The memory arm is still better *and* smaller; it
  is simply not as much faster in wall clock as its FLOP count implies.
