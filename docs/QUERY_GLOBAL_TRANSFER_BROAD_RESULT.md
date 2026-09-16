# Balanced attention transfer lowered output error but worsened model quality

**Measured 16 September 2026 after the [protocol and evaluator](QUERY_GLOBAL_TRANSFER_BROAD_PROTOCOL.md)
were pushed as `4097150`:** the warm-started layer-8 feature map reduced
teacher-attention output MSE on both Shakespeare and WikiText validation,
and on every one of eight fresh quality-window prefixes. Yet cached
next-token perplexity was slightly **worse than the old map on both fresh
texts**. Both maps remained behind unchanged fused-attention Llama.
The predeclared rule therefore **rejects the broad map** for a balanced
stage-2 or multi-layer quality attempt. More attention-output fitting
alone did not produce a better 1B model in this test.

## Fixed run and checkpoint

The runner used the offline bf16 `unsloth/Llama-3.2-1B` teacher, the
previously selected 163,840-scalar Shakespeare map as an exact warm
start, the same q/k/v/o/RoPE, exact 64-token local window and fixed
global gain. Only `delta_q` and `delta_k` were trainable; the runner
asserted that scope. It captured genuine layer-8 inputs and full teacher
attention outputs for 128 disjoint Shakespeare train windows in the
first half and 256 WikiText-2 raw train windows, all 512 tokens. The
seed-31 schedule visited all 384 windows with 300 updates per text,
**196,608 distinct training positions** and **307,200 token exposures**.
The WikiText test split did not enter training or checkpoint choice.

The lowest equally weighted Shakespeare 55/58% and WikiText-validation
20/40/60/80% output-MSE score selected the **step-600** checkpoint:

| attention-output selection MSE | old map, step 0 | broad map, step 600 | reduction |
|---|---:|---:|---:|
| Shakespeare mean | 0.00055822 | 0.00052222 | 6.45% |
| WikiText validation mean | 0.00069331 | 0.00062717 | 9.54% |
| balanced score | 0.00062576 | 0.00057470 | 8.16% |

This selected score was never a downstream next-token or serving
measurement. Teacher activation/target capture took **36.58 s**;
tokenizing the texts took **3.14 s**; gradient updates took **10.86 s**
within an **11.68 s** loop, including **0.68 s** selection. Loading
the model took **0.60 s**. Capture and training-stage Metal peaks were
**4.424 GB** and **5.395 GB** in the shared process. The new conversion
work adds roughly 52 seconds before any serving use and does not change
the attention execution path or cache shape relative to the old map.

## Predeclared fresh quality

The evaluator used three untouched 3,000-token Shakespeare windows at
**64/66/68%** and five untouched WikiText-2 raw test windows at
**2/22/42/62/82%**. They were fixed and checked for overlap before
the primary run. Each arm scored **8,997 Shakespeare** and **14,995
WikiText** next-token predictions from persistent caches, with float32
loss from bf16 logits. Short whole/split/decode top-1 parity and bounded
cache counts passed for teacher and both maps.

| one layer-8 attention arm | Shakespeare PPL | WikiText-2 raw test PPL |
|---|---:|---:|
| unchanged fused-attention teacher | **29.3664** | **10.4894** |
| published Shakespeare-only map | 30.0651 | 10.9122 |
| newly selected balanced map | 30.1029 | 10.9179 |

The broad-map-minus-old-map aggregate NLL was **+0.001256** on
Shakespeare and **+0.000522** on WikiText. It was worse on **six of
eight** paired windows. Its gap to unchanged teacher was **+0.024770
NLL** on Shakespeare and **+0.040036** on WikiText, beyond the
predeclared +0.01 aggregate limit on both texts. All five WikiText
windows exceeded teacher +0.03 NLL. The query layer's 2,999-input-token
logical cache was **92.525 MB** versus **98.271 MB** for teacher, the
same one-layer reduction as the old map; this is neither a deployed
resident-memory nor a compute-cost result.

The surrogate moved in the opposite direction from model quality.
Fresh attention-output MSE fell from **0.00052833 to 0.00049856** on
the three Shakespeare prefixes, and from **0.00074986 to
0.00067058** on the five WikiText prefixes. The broad map improved
output MSE on **8/8** prefixes, while worsening next-token NLL on
**6/8** complete windows. This shows that the selected attention
output metric is misaligned with the downstream quality decision in
this bounded conversion; it does not prove that all attention-output
training is ineffective.

## Decision and evidence

The [predeclared rule](QUERY_GLOBAL_TRANSFER_BROAD_PROTOCOL.md) first
required both-text selection MSE improvement, both-text fresh aggregate
quality at least as good as the old map, and no window above old +0.02
NLL. The first and third conditions passed; **both-text quality
failed**. The stricter unchanged-teacher conditions also failed.
The selected broad map is therefore **not retained** for stage-2
adjustment, and no multi-layer conversion is promoted. The old map is
still the better measured starting point. A direct, balanced
next-token objective with a matched full-attention control is a
candidate next quality experiment; any gain must survive *new*
holdouts and complete-model serving/training measurements. The
[single-layer timing result](QUERY_GLOBAL_FAST_DECODE.md) already
found no reliable whole-model speed gain.

The raw training run, source/protocol/data hashes, window IDs, full
selection curve, timing and selected-weight SHA are in
`experiments/results/query_global_transfer_broad.json` and
`query_global_transfer_broad_weights.npz`. The raw fresh-window NLL,
attention-output MSE, paired differences, cache parity/counts,
logical-cache bytes and exact input hashes are in
`experiments/results/query_global_transfer_broad_holdout.json`.
The two-window smoke files were diagnostics and are excluded from the
published primary result. The protocol commit time precedes both
primary training and evaluation start times; their recorded source,
protocol, model and selected-weight hashes match.

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python experiments/train_query_global_transfer_broad.py
~/zbrain/venv/bin/python experiments/evaluate_query_global_transfer_broad.py
```
