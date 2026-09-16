# Bounded query-addressable state on a pretrained 1B Llama layer

**Measured 16 September 2026:** a fixed local-plus-query-state layer and a
small attention-transfer run recover more of the quality lost by a 64-token
local window. The transfer improvement persists on five WikiText-2 raw test
windows that supplied **no training data**. The converted model remains
worse than the unchanged teacher, and its one-layer complete-model benchmark
does **not** show a speed gain. This is a useful conversion mechanism,
not a cheaper AI model yet.

## Architecture and prior art

`experiments/query_global_attention.py` replaces layer 8 of the offline
bf16 `unsloth/Llama-3.2-1B` MLX checkpoint. It keeps the teacher's
q/k/v/o projections and RoPE. Each query attends exactly to the current
token and up to 63 predecessors with fused softmax attention. Older keys
leave that local cache and update two float32 states per one of eight KV
heads:

```text
S_t = sum_{i <= t-64} phi(k_i) v_i^T
z_t = sum_{i <= t-64} phi(k_i)
global(q_t) = phi(q_t)^T S_t / max(phi(q_t)^T z_t, epsilon)
output = (1 - alpha) local(q_t) + alpha global(q_t)
```

The fixed positive map is
`phi(x) = softmax(2x) || softmax(-2x)` after RoPE, using identity
head dimensions, with 128 features. The fixed `alpha=0.449684877` came
from a six-position layer-output fit on a 60% selection window and is zero
until at least one older key exists. The implementation mixes *separately
normalized* local and global outputs; it is not the exact LoLCATs mixing
formula.

[LoLCATs](https://arxiv.org/abs/2410.10254) already introduced a
64-token local softmax window plus bounded feature-map global attention,
attention-output transfer, and LoRA adjustment. Our untrained identity map
and Apple MLX implementation are a **prior-art-informed baseline**, not a
claim that local-plus-linear attention is new. LoLCATs also reports
Llama-3.2-1B results after much more training data. A claim of an
improvement over that method would require a matched implementation,
quality and hardware comparison.

The cache holds only 63 older local keys/values plus the two states.
At 128 features and 64 value dimensions, `S` and `z` use **266,240
bytes** at batch 1 across eight KV heads, independent of context length.
No new model weights are required by the fixed baseline. The first-stage
trainer in `experiments/query_global_trainable.py` adds zero-initialized,
per-head query and KV residual matrices: **163,840 trainable scalars**
at this 1B configuration. At step zero it is exactly the fixed map.

## Teacher-output probe and cached model quality

On six sampled layer-8 queries from genuine teacher activations, a single
gain for each map was fitted on the 60% window and applied unchanged to
two untouched 75%/85% windows. The selected identity feature map with
temperature 2 recovered **43.4%** of the local-window attention-output
residual MSE on selection, **44.9%/40.1%** on those two windows. The
selection-fitted transplanted gated trace recovered **24.9%/25.4%/21.5%**
respectively. This is a sampled attention-output diagnostic, not a
perplexity or time result
(`experiments/results/query_global_probe.json`).

The untrained full model was then scored with persistent caches over four
disjoint 3,000-token Shakespeare windows at 96–99%. Each arm scored
2,999 next-token predictions per window in 128-token model chunks,
using float32 loss from bf16 logits:

| arm | aggregate PPL on 11,996 tokens |
|---|---:|
| unchanged Llama teacher | **20.7592** |
| exact local-only layer | 22.1861 |
| local + fixed query state | **21.9800** |

Query state beat local-only on **4/4** windows, but remained **5.9%**
worse in perplexity than the teacher
(`experiments/results/query_global_quality.json`).

## One-layer attention transfer

`experiments/train_query_global_transfer.py` used **32 disjoint
512-token training windows** from the first 50% of TinyShakespeare and
two disjoint selection windows at 55%/58%. It captured genuine teacher
layer inputs and full softmax outputs, then trained only `delta_q` and
`delta_k` for **200 steps** of AdamW at learning rate 0.01. The
pretrained projections were frozen; an assertion checked that only those
two feature-map arrays were trainable. The best checkpoint was chosen by
selection attention-output MSE, never by the downstream quality windows.

Selection output MSE fell from **0.00100655** at the fixed-map start to
**0.00055822** at step 200, a **44.5%** reduction. Fresh 512-token
attention-output checks at 70%/72%/74% recovered **47.0%/49.6%/50.4%**
relative to the untrained query map. Full-model cached quality on the
three new 3,000-token windows was:

| arm | aggregate PPL on 8,997 tokens |
|---|---:|
| unchanged teacher | **24.5753** |
| local-only | 25.7117 |
| fixed untrained query state | 25.5201 |
| transfer-trained query state | **25.0549** |

The trained map beat the untrained map on **3/3** windows, but its
perplexity remained **1.95%** above the teacher. On the earlier iterative
96–99% windows, the trained arm scored **21.4604**, versus **20.7592**
for the teacher. These are one corpus and one layer. The result files
are `experiments/results/query_global_transfer.json`,
`query_global_transfer_weights.npz`, and
`query_global_trained_quality.json`.

Teacher activation/target capture took **2.81 seconds**, the measured
training loop **3.56 seconds**, and loading **0.55 seconds**. The raw
artifact records validation time, peak Metal allocation, all source and
corpus hashes, the fixed batch schedule and the selected weight-file
hash. One-minute host load was **24.15–24.54**, so these timings are
records under contention, not a universal conversion-cost estimate.
Only **16,384 unique training tokens** and **102,400 token exposures**
were used; this does not reproduce LoLCATs' much larger data protocol.

## Independent text domain

Five 3,000-token windows at 10/30/50/70/90% of the
[Salesforce WikiText-2 raw test](https://huggingface.co/datasets/Salesforce/wikitext)
were fixed before this quality read. The frozen test zip came from the
[ggml-org CI mirror](https://huggingface.co/datasets/ggml-org/ci);
the artifact records its exact revision and SHA-256. No WikiText text
was used in feature-map training or checkpoint selection.

| arm | aggregate PPL on 14,995 WikiText tokens |
|---|---:|
| unchanged teacher | **11.2492** |
| local-only | 11.8751 |
| fixed untrained query state | 11.8230 |
| Shakespeare-transfer-trained query state | **11.6476** |

The trained map improved over the untrained map on **5/5** windows.
Layer-output MSE improved **37.4–43.6%** per window. It still scored
**3.54%** worse in perplexity than the unchanged teacher. This supports
transfer beyond the training text, while one Wikipedia-derived test split
cannot establish general language-model quality
(`experiments/results/query_global_wikitext_quality.json`).

No WikiText text or archive is committed here. The linked dataset card
states its Creative Commons Attribution-ShareAlike and GFDL licensing;
only the evaluation source, hashes and aggregate metrics are published.

## Generation, serving time and memory

With the explicit `make_query_global_cache(model)`, the official
`mlx_lm.generate_step` path and a manual greedy loop generated the same
four token IDs for the trained map. A real 8,192-token prefix served
whole or in two 4,096-token calls, then followed by one decoded token,
preserved top-1; the maximum one-pass/decode logit difference was
**0.129** in bf16. The custom cache retained 63 old KV positions
and summarized 8,130 older keys at offset 8,193. The default
`model.make_cache()` still produces an incompatible plain KV slot for
this opt-in wrapper
(`experiments/results/verify_trained_query_generation.json`).

The paired serving-style one-layer benchmark used last-token vocabulary
prefill, 32 teacher-forced single-token decodes, batch 1 and five
rotated-order repeats. For the **untrained** map with periodic
evaluation every eight 64-token blocks, its 8,192-token median prefill
time was **1.022×** the teacher and decode speed **0.987×**.
The unsynchronized current-source control gave **1.015×** prefill time
and **0.968×** decode speed. Different host loads (**11.48** and
**22.77**) make the small timing differences inconclusive; neither
run demonstrates a material faster complete model. The trained map's
serving speed has not yet been measured.

At 8,192 prefix plus 32 decoded tokens, the active logical cache was
**269.484 MB** for the teacher and **253.037 MB** for query state:
**16.447 MB** saved by one converted layer. In the same-process
benchmark, unsynchronized long-prefix lazy scans raised transient
Metal peak from **3.764 GB** teacher to **5.893 GB** query state.
Periodic evaluation cut query peak to **3.747 GB** without changing
cached predictions, at a possible synchronization-time cost. Teacher
and candidate coexist in that process, so these peaks do not prove
deployment resident memory or dollar cost
(`experiments/results/query_global_benchmark_nosync_b1.json`,
`query_global_benchmark_sync8_b1.json`).

## Decision gate

The one-layer transfer is reproducible and generalizes across the two
texts tested, but it has not met the unchanged teacher's held-out
quality. The next step is a **matched** rank-8 LoRA adjustment on
query-state and full-attention controls, followed by fresh next-token
and retrieval tests. A quality-preserving multi-layer conversion must
then beat a strong complete-model serving baseline and count teacher
forwards, LoRA, peak/resident memory and conversion time. Only then
could this work support a lower-compute claim. The current mechanism
is not a claimed major unlock.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python -m pytest -q tests/test_query_global_attention.py
~/zbrain/venv/bin/python experiments/query_global_probe.py
~/zbrain/venv/bin/python experiments/query_global_quality.py
~/zbrain/venv/bin/python experiments/verify_query_generation.py
~/zbrain/venv/bin/python experiments/train_query_global_transfer.py --smoke
~/zbrain/venv/bin/python experiments/train_query_global_transfer.py
~/zbrain/venv/bin/python experiments/query_global_trained_quality.py
~/zbrain/venv/bin/python experiments/verify_trained_query_generation.py
curl -fL --silent --show-error \
  https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip \
  -o /tmp/human_brain_wikitext2_raw.zip
~/zbrain/venv/bin/python experiments/query_global_wikitext_quality.py
~/zbrain/venv/bin/python experiments/benchmark_query_global.py \
  --sync-blocks 0 --repeats 5 \
  --output experiments/results/query_global_benchmark_nosync_b1.json
~/zbrain/venv/bin/python experiments/benchmark_query_global.py \
  --sync-blocks 8 --repeats 5 \
  --output experiments/results/query_global_benchmark_sync8_b1.json
```

The scripts require the pinned offline model snapshot and the vendored
TinyShakespeare corpus. WikiText archive and member hashes are checked
before use; an interrupted raw result stays marked `running`.
