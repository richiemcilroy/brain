# Retraction: the claimed efficiency win over a transformer was a training-budget artifact

**Status: RETRACTED.** The claim recorded earlier in this session — that a gated
multiscale memory arm beat a GPT-2-style attention baseline by 26% on
bits-per-character with 2x fewer parameters and 30% fewer FLOPs — is **wrong**.
There is no such win. This document records the claim, the measurement that
killed it, and what (little) survives.

## 1. The claim that was made

Char-level language modelling on TinyShakespeare, MLX, `d=128`, `ctx=512`,
500 steps, batch 16, one seed:

| arm | params | FLOPs/tok | bpc |
|---|---|---|---|
| A attention (GPT-2 style) | 875,520 | 7,914,240 | 3.666 |
| F gated memory | 808,448 | 5,554,944 | **2.709** |

Read as: same task, half the parameters, 30% fewer FLOPs, 26% better loss.

## 2. Why it was wrong

**The attention baseline was at the bigram floor, and the bigram floor is a
one-line calculation.**

A Laplace-smoothed 65x65 bigram count table, fit on the train split, scored on
the validation split:

| model | bpc | nats |
|---|---|---|
| unigram | 4.8292 | 3.3474 |
| **bigram** | **3.5806** | **2.4819** |
| our attention baseline | 3.6664 | 2.5414 |

The "language model" scored **0.0858 bpc worse than a bigram**. Its context path
was contributing nothing, so the comparison was *gated memory versus a broken
transformer*, not memory versus attention.

**Diagnosis.** No warmup, no gradient clipping, AdamW at `lr=3e-3`, MLX's
default uniform init. Adam's normalised update moves every Q and K weight by
~3e-3 from the first step; the attention logits saturate and the softmax
collapses onto a single position, degenerating the layer to a per-token MLP.
The gated arm survived the same learning rate because its gate bias is
initialised to a useful ~10-character decay, so it starts with locality for
free and never has to discover context.

**The architecture was never broken.** Verified directly:

- perturbing the input at position `t-1` changes the output at position `t`, so
  context does flow (max |delta| 0.53 at the perturbed position, decaying with
  distance);
- at init, layer-0 attention entropy is 1.87 nats. (The true causal maximum at
  `T=16` is the mean of `ln(t+1)` = 1.92 nats, **not** `ln 16` = 2.77 — a
  correction to an earlier note in this session. 1.87 of 1.92 means the heads
  are essentially uniform at init, which is the expected behaviour at
  `mean|qk|` = 0.27, not "collapsed".)

## 3. The measurement that killed it

Attention trained with warmup (100-step linear), gradient clipping at 1.0, and
weight decay 0.01, `lr=1e-3`, validation bpc every 200 steps:

| step | val bpc | vs bigram floor |
|---|---|---|
| 200 | 3.5632 | +0.0173 |
| 400 | 3.1705 | +0.4101 |
| 600 | 2.7111 | +0.8695 |
| 800 | 2.5125 | +1.0681 |
| 1000 | 2.4028 | +1.1778 |
| 1200 | 2.3117 | +1.2689 |
| 1400 | 2.3035 | +1.2771 |
| 1600 | 2.2377 | +1.3429 |
| 1800 | 2.2427 | +1.3379 |
| **2000** | **2.2354** | **+1.3452** |

The attention arm sits at the bigram floor for roughly 200 steps and then breaks
away. **The 26% "win" was obtained by stopping the comparison at 500 steps,
which is inside that plateau.** The plateau is the transformer's induction-head
formation phase; a gated recurrence clears the bigram floor immediately because
its locality prior does that work for free. The gated arm's early lead is the
absence of a known transformer training event, and says nothing about the
mechanism.

At 2000 steps the properly trained attention arm reaches **2.2354 bpc**, which
is better than the gated arm's quoted 2.7086 at 500 steps. The comparison as
first reported was measuring the baseline's warmup, not the architecture.

## 4. Two further errors in the same claim

**The "multiscale" arm was not multiscale.** The configuration labelled as
multiscale used `banks=1` — a single decay, 808,448 parameters. At `banks=4` the
parameter count is 1,399,808, so that arm was never a matched-parameter
comparison against the 875,520-parameter attention baseline either.

**The FLOP advantage does not survive at realistic width.** Per-token cost,
attention `= 8d^2 + 2*T*d + 16d^2` (causal), gated memory
`= 4*B*d^2 + 3*B*d*log2(64) + 16d^2`:

| config | attention | memory (1 bank) | ratio | memory (4 banks) | ratio |
|---|---|---|---|---|---|
| d=512, T=512 | 7,077,888 | 5,776,384 | 1.23x | 10,522,624 | **0.67x** |
| d=1024, T=512 | 26,214,400 | 23,087,104 | 1.14x | 42,016,768 | **0.62x** |
| d=1024, T=8192 | 41,943,040 | 23,087,104 | 1.82x | 42,016,768 | 1.00x |
| d=1024, T=32768 | 92,274,688 | 23,087,104 | 4.00x | 42,016,768 | 2.20x |

At short context and realistic width the advantage is 1.14x-1.23x for one bank,
and **negative for four banks**. Most attention cost at short `T` is still the
projections, not the scores. The advantage is a long-context property.

Two corrections were applied to an earlier version of this table: the MLP term
is `16d^2` (the prose said `8d^2` while the code used `16d^2`), and attention is
charged `2*T*d` for a causal mask rather than `4*T*d`, since a kernel that skips
masked blocks evaluates about half the score matrix. Both corrections reduce the
reported advantage.

## 5. What survives, stated narrowly

One sentence, and it is not ours:

> A diagonal gated linear recurrence has context cost flat in `T`, so past a few
> thousand tokens at `d` of 512 to 1024 it is a 1.8x-4x FLOP advantage over
> dense causal attention.

That is the value proposition of S4, H3, Mamba, RWKV and Griffin, with **worse
constants than theirs**, because they ship fused kernels that keep state in SRAM
and this does not. The gated recurrence here
(`h_t = sigmoid(W_g x_t) * h_{t-1} + W_v x_t`) is minGRU without the convex
combination, HGRN without the lower-bound trick, and RG-LRU without input
normalisation. It is a known mechanism, correctly identified as such in the
`llm_efficiency.py` docstring, and the measurement adds nothing to what that
family already claims.

**And no component of the spiking substrate is in it.** Not the neuron, not
k-WTA, not the local rule, not the dendritic trace. The LM arms are trained by
backpropagation through the scan. The link to `docs/WORKING_MEMORY.md` is the
word "trace".

## 6. A note on the parallel scan, which is real

The 4x wall-clock penalty originally reported for the memory arm was an
implementation artifact: a naive Python loop over `T`. Replacing it with a
chunked parallel scan (sequential only over `T/C` chunks, Hillis-Steele within a
chunk) gives:

- correctness: 4.8e-07 max absolute error against the naive recurrence at
  `T` in {1, 2, 7, 33, 64, 100, 129, 512, 1000};
- `B=16, D=128`: `T=512` 6.22ms -> 1.00ms (6.2x); `T=4096` 57.92ms -> 8.17ms
  (7.1x);
- at `ctx=512, steps=20`: gated 251,345 tok/s vs attention 250,445 tok/s.

That is a real 6-7x speedup over my own first implementation. It is **not** an
efficiency win over attention: at `d=128, T=512, B=16` both arms are
launch-overhead bound on this GPU and parity establishes nothing about the
compute-bound regime. The scan is a corrected building block, not a result.

## 7. What this cost, and the lesson

The original comparison was reported after a single 500-step run at one seed
with a shared, untuned learning rate, and the gap was large enough to be
believed without a floor check. **A bigram count table is five lines of NumPy
and would have caught it in the first minute.** Any language-model comparison
must print the unigram and bigram floors next to every arm, because an arm at
the floor has not learned to use context at all and cannot serve as a baseline
for anything. `experiments/llm_tuned.py` now computes and prints both floors on
every run and labels any arm at or above the floor.

The second lesson is that a matched-parameter claim needs the parameter counts
checked against the arm actually built, not the arm intended. The "multiscale"
arm was single-bank.

## 8. Reproduce

```
python3 experiments/cost_model.py         # the FLOP table and crossovers
python3 experiments/llm_tuned.py A_attention C_gated_banks4   # floors printed
```

Artifacts: `experiments/llm_efficiency.py`, `experiments/llm_tuned.py`,
`experiments/cost_model.py`, `experiments/llm_context_scaling.py`.
Raw runs: `/Volumes/T9/human-brain/scratch/{curve.log,curve_ab.log}`.

---

## 9. Addendum: the FLOP advantage IS realisable, and here is the measured crossover

Section 5 said the FLOP advantage might not be realisable in wall clock,
because attention's context term is a dense matmul near peak FLOPs while the
scan is elementwise, bandwidth-bound, and stores `O(log C)` levels. **Measured:
it is realisable, and the crossover is between `T=2048` and `T=4096`.**

Pure forward pass, `d=512`, 8 heads, warmup discarded, `mx.eval` called every
iteration (`experiments/long_context_bench.py`):

| T | attn us/tok | mem1 us/tok | A/M1 | A/M4 |
|---|---|---|---|---|
| 512 | 3.114 | 6.803 | **0.46x** | 0.37x |
| 1024 | 2.569 | 3.029 | 0.85x | 0.29x |
| 2048 | 2.549 | 2.788 | 0.91x | 0.26x |
| 4096 | 8.134 | 2.584 | **3.15x** | 0.70x |
| 8192 | 12.487 | 3.403 | 3.67x | 1.21x |
| 16384 | 23.864 | 6.139 | 3.89x | 0.97x |
| 32768 | 44.897 | 2.762 | **16.26x** | 3.61x |

Two things are worth noting. First, the memory arm's per-token cost is
**roughly flat** (2.6-6.8 us/tok) while attention's grows by 14x over the same
range — that is the `O(T)` versus `O(T^2)` signature showing up in wall clock,
not just in a cost model. Second, single-bank memory crosses over at `T~4096`
and reaches 16x at `T=32768`; four banks only cross at `T~8192` and reach 3.6x,
because the bank dimension multiplies the projection cost that dominates at
short `T`.

**What this does and does not establish.** It establishes that on this hardware
a diagonal gated recurrence is materially faster than dense causal attention
past a few thousand tokens, which is the regime where long-context inference
actually lives. It does **not** establish novelty: this is the measurement
Mamba, RWKV and Griffin already publish, and their fused kernels start from a
better constant than this scan does. The value of this table is that it is ours,
it was measured rather than assumed, and it survived a falsification attempt
that killed the original claim.

**The honest summary of the whole efficiency episode:**

- the "26% better loss at half the parameters" claim: **retracted, artifact**
- the "30% fewer FLOPs at short context": **wrong at realistic width** (1.14x
  at `d=1024, T=512`, and *worse* than attention with 4 banks)
- the "4x slower wall clock": **implementation artifact**, fixed by the scan
- the long-context wall-clock advantage: **real, measured, 3.1x at T=4096 and
  16.3x at T=32768, and not novel**
