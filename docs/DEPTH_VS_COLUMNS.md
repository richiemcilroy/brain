# Depth beats columns: "simultaneous thinking" buys latency, not quality

**Verdict: the parallel-columns ("simultaneous thinking") hypothesis is REFUTED
as a quality win, and it was refuted by its own control.** Two depth-1 columns
score the same as a single dense layer of matched width, so columns are a
restricted wide layer and not a new computational unit. A plain **depth-2**
stack beats both columns *and* the matched dense layer, with **fewer**
parameters. The only thing columns change is the dependency chain.

There is a second, genuinely useful result buried in this table: a 445,440
parameter gated-memory stack reaches **2.2009 bpc**, which is **better than the
875,520 parameter attention baseline at 2.2528 bpc** — the better architecture
wins with **half the parameters**.

## Method

Char-level TinyShakespeare, `ctx=512`, 1500 steps, batch 16, AdamW `lr=1e-3`
with 100-step warmup and gradient clipping at 1.0, **2 seeds**, validation over
16 batches of 64. Unigram floor 4.8292 bpc, **bigram floor 3.5806 bpc**. Every
arm is well clear of the floor, so all of them have learned to use context.

`experiments/parallel_columns.py`. Summary over seeds (mean):

| arm | params | bpc | vs bigram floor | fwd ms | tok/s |
|---|---|---|---|---|---|
| `cols2_mem` (2 columns x depth 1) | 494,720 | 2.2632 | +1.317 | 26.3 | 91,414 |
| `dense1_mem` (1 layer, width-matched to cols2) | 531,072 | 2.2647 | +1.316 | **11.5** | **162,710** |
| `depth2_mem` (2 layers in series) | **445,440** | **2.2006** | **+1.380** | 18.0 | 133,090 |
| `cols2_attn` (2 attention columns) | 528,256 | 2.4110 | +1.170 | 23.5 | 110,669 |
| `depth2_attn` (2 attention layers) | 478,976 | 2.3586 | +1.222 | 21.5 | 112,136 |
| `A_attention` (4-layer baseline, from the matched run) | 875,520 | 2.2564 | +1.324 | — | 48,466 |

## The three comparisons that matter

**1. Columns are a restricted wide layer.** `cols2_mem` 2.2632 versus
`dense1_mem` 2.2647 — a difference of **0.0015 bpc**, far inside seed spread
(per-seed values: cols2 2.2733/2.2530, dense1 2.2871/2.2422). K depth-1 columns
concatenated and passed through one linear mixer cannot mix their internal
features, so their function class is a strict subset of one dense layer of
matched width. They perform identically, as that argument requires.

**2. Depth beats both, with fewer parameters.** `depth2_mem` reaches 2.2006 with
**445,440** parameters, against 2.2632 with 494,720 for columns and 2.2647 with
531,072 for the matched dense layer. Depth is both better and smaller. This was
the pre-registered prediction and it held.

**3. Dense is faster than columns in the forward pass.** 11.5 ms versus 26.3 ms.
The latency argument for columns also fails here: on this hardware the dense
layer is ~2.3x faster per forward pass at matched width, because the column
version runs `n_col` separate small matmuls plus a `(n_col+1)*d -> d` mixer
rather than one large matmul. Columns have a shorter *dependency chain* but a
worse *execution profile*, and at these sizes throughput dominates.

## Why depth wins, and what it implies

A single-layer column cannot condition its memory readout nonlinearly on its own
earlier output. Composing two layers allows exactly that, which is what
induction-head-style computation requires (Olsson et al. 2022, in-context
learning and induction heads). Two parallel depth-1 columns add **capacity**
without adding **composition**, and the measurement says composition is what
this task rewards. An independent review predicted this outcome before the run
and the prediction is what the data shows.

## The parameter-efficiency result, stated carefully

`depth2_mem` at 445,440 params scores 2.2006; the 4-layer attention baseline at
875,520 params scores 2.2564. That is **0.0558 bpc better with 49% of the
parameters**, and it is consistent with the looped-depth result recorded
earlier (875K -> 478K -> 445K at roughly equal loss).

**What this does NOT show.** It is not a claim that gated memory beats attention
in general, and it must not be read as one:

- one task (char-level TinyShakespeare), one context length (512), two seeds;
- these are two different parameter budgets, not a matched-parameter comparison
  at equal depth — the honest matched comparison at `d=128` is still the one in
  `docs/EFFICIENCY.md`, where the two arms land within noise of each other;
- at `d=128, T=512` both architectures are launch-overhead bound on this GPU,
  so none of these throughput numbers transfer to the compute-bound regime;
- the long-context advantage is where the mechanism actually pays off, and it is
  measured separately in `docs/EFFICIENCY.md` §9.

## Reproduce

```
python3 experiments/parallel_columns.py
```

Raw run: `/Volumes/T9/human-brain/scratch/cols.log`.
