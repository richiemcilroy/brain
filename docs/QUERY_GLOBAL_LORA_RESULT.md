# Matched rank-8 adjustment did not close the teacher gap

**Measured 16 September 2026, after the predeclared protocol in
`QUERY_GLOBAL_LORA_PROTOCOL.md` was pushed:** a 300-step Shakespeare
next-token adjustment overfit in all three matched LoRA arms. The
query-state arm retained a gain over local-only LoRA, but its selected
checkpoint was worse than its *unadjusted stage-1* starting point on
all nine fresh windows. The unchanged 1B Llama was best on both texts.
This fixed small-data recipe does not provide a faster, cheaper model.

## What was trained and selected

All arms trained only rank-8 q/k/v/o LoRA, **106,496 scalars each**, at
the same zero-B, seeded-A initialization, LR 0.0003, 32 disjoint
512-token training windows, seed-19 300-step draw and two Shakespeare
selection windows at 55%/58%. The original projection hashes were
identical before and after. A split at frozen layer 8 matched full
teacher logits **exactly** before training. The query arm carried its
previously selected **163,840 frozen feature-map scalars** and the same
constant-size state. There was no WikiText training or checkpoint
selection.

Selection next-token NLL was lowest at **step 50 in every arm**:

| arm | step-0 NLL | selected step-50 NLL | final step-300 NLL |
|---|---:|---:|---:|
| full attention LoRA | 3.07085 | **3.05841** | 3.52873 |
| exact local-only LoRA | 3.11894 | **3.09926** | 3.58079 |
| query-state LoRA | 3.09305 | **3.06856** | 3.52015 |

Training loss kept falling while selection loss rose after step 50.
The selected files and the complete per-arm curve, gradient norms,
schedule and source hashes are in
`experiments/results/query_global_lora_transfer.json` and the three
`query_global_lora_*_weights.npz` files. The two-step smoke run was
diagnostic and is excluded from this verdict.

## First read of the fixed next-token windows

The quality script and percentages were committed before the primary
training run. Each 3,000-token window scored 2,999 predictions with
persistent caches and float32 loss from bf16 logits; short whole,
split and decode cache parity passed for all five arms.

| arm | Shakespeare 80/82/84/86%, 11,996 tokens | WikiText-2 raw test 15/35/55/75/95%, 14,995 tokens |
|---|---:|---:|
| unchanged Llama teacher | **21.5442 PPL** | **9.7111 PPL** |
| frozen stage-1 query map, no LoRA | 22.1745 | 10.0243 |
| full-attention LoRA | 22.5141 | 10.0438 |
| exact local-only LoRA | 24.0219 | 10.8131 |
| query-state LoRA | 22.8402 | 10.2657 |

Query LoRA was worse than the teacher, full-attention LoRA and its own
stage-1 starting point on **9/9** paired windows, but better than
local-only LoRA on **9/9**. Full-attention LoRA was also worse than the
unchanged teacher on all nine windows. The cross-domain result shows
that this protocol overfit generally, not merely that the converted
attention failed to adapt. It does **not** establish that no LoRA
recipe can preserve quality. The full window records and cache bytes
are in `experiments/results/query_global_lora_holdout.json`.

## Compute implication and next gate

The stage-2 trainer captured **17,408 unique prefix tokens** through
the frozen first eight layers in **1.49 s** and exposed **153,600
training tokens per arm**. Measured adjustment-loop time was
**73.48 s** for full-attention LoRA and **78.33 s** for query-state
LoRA; query state was about **6.6% slower** on this 512-token setting.
Selection time was about 1.5 s per arm, peak Metal allocation across
co-resident arms was **4.59 GB**, and host load moved from 22.52 to
8.29. These are same-host observations under variable contention, not
stable FLOP, deployment memory or dollar-cost estimates. The
attention-output stage-1 teacher capture and training must also count
toward conversion cost. The earlier serving benchmark of that selected
stage-1 map found no material complete-model speed gain.

Before any multi-layer speed claim, use a matched lower-LR and larger,
more diverse train/selection protocol with true fresh held-out text
and a full-attention control. If the converted model still misses
quality, changing more layers would amplify the risk rather than
demonstrate an efficiency unlock. A query-state model that meets
quality must then beat complete-model prefill/decode and training
baselines after all conversion work is counted.
