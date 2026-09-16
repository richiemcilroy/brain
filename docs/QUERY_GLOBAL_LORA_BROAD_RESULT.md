# Larger-data LoRA improved WikiText, but failed the two-text quality gate

**Measured 16 September 2026 after the protocol and evaluator were pushed
as `006c5cc`:** the matched one-layer 1B query-state LoRA checkpoint
improved fresh WikiText-2 raw test perplexity from the frozen stage-1
map's **8.882** to **8.468**, beating the unchanged Llama teacher's
**8.598** on all five windows. It worsened fresh Shakespeare
perplexity to **17.189** versus the teacher's **16.633**. The
full-attention LoRA control also worsened Shakespeare, so the
predeclared recipe-control rule fails. Query state remains worse than
full-attention LoRA on both texts and its matched training loop was
about **9% slower**. This is useful adaptation evidence, not a
quality-preserving or cheaper model.

## Fixed training and checkpoint choice

The run used the frozen selected stage-1 feature map, original
pretrained tensors and three rank-8 q/k/v/o LoRA arms, each with
**106,496 trainable scalars**. Identical seeded A arrays and zero B
arrays passed the initial guards; the original q/k/v/o hashes matched
after training. Ninety-six disjoint Shakespeare and 256 WikiText raw
train windows provided **180,224 distinct training-window input
positions** for frozen teacher-prefix capture. The seed-23 fixed
500-step schedule visited **137,728 distinct input positions** and
made **256,000 token exposures per arm**, with 125 Shakespeare and
375 WikiText steps. LR was 0.0001, AdamW weight decay zero, gradient
clip 1.0. No WikiText test or fresh Shakespeare holdout entered
training or checkpoint choice.

Every arm selected its lowest score on two Shakespeare 55/58% windows
and four separate WikiText **validation** 20/40/60/80% windows;
the score was half the mean NLL from each text. The baseline and
selected scores were:

| arm | step-0 selection score | selected step | selected score |
|---|---:|---:|---:|
| full-attention LoRA | 2.71568 | 500 | **2.63047** |
| local-only LoRA | 2.76566 | 400 | 2.65442 |
| query-state LoRA | 2.74105 | 500 | 2.64594 |

Scores fell much longer than in the earlier 16K-token recipe, which
had selected step 50 in every arm. The full and query checkpoints
made their lowest selection score at the final read. That validation
improvement did not imply two-text held-out parity.

## Fresh cached next-token quality

The [predeclared protocol](QUERY_GLOBAL_LORA_BROAD_PROTOCOL.md) fixed
Shakespeare **77/88/93%** and WikiText raw **test 5/25/45/65/85%**
windows before primary training. The evaluator asserted these
3,000-token windows were disjoint from previously inspected windows,
including earlier hybrid work at Shakespeare 90%. Every arm scored
8,997 Shakespeare and 14,995 WikiText predictions with persistent
caches, float32 loss from bf16 logits, and passing whole/split/decode
cache parity.

| arm | Shakespeare PPL | WikiText-2 raw test PPL |
|---|---:|---:|
| unchanged Llama teacher | **16.633** | 8.598 |
| selected stage-1 query map, no LoRA | 17.068 | 8.882 |
| full-attention LoRA | 16.909 | **8.260** |
| exact local-only LoRA | 17.473 | 8.532 |
| query-state LoRA | 17.189 | 8.468 |

Query state beat local-only LoRA on **8/8** paired windows. It beat
its stage-1 starting point and the unchanged teacher on **5/5
WikiText** windows, but lost to both on **3/3 Shakespeare** windows.
Full-attention LoRA beat the query arm on all eight windows. The
mean query-minus-teacher NLL was **+0.03292** on Shakespeare and
**−0.01522** on WikiText; query-minus-full was **+0.01643** and
**+0.02490**. The worst Shakespeare query-minus-teacher window was
**+0.03826 NLL**, beyond the predeclared +0.03 ceiling.

The protocol first checks whether full-attention LoRA stays within
teacher +0.01 NLL on both texts. It failed on Shakespeare by
**+0.01649 NLL**, while improving WikiText by **−0.04012**. The
query arm also failed its teacher/full aggregate limits. The script's
decision is therefore **no multi-layer quality promotion** and an
**inconclusive query-specific failure**, because the same recipe
degraded the full-attention control on Shakespeare. This could be a
tradeoff from the 3:1 WikiText/Shakespeare step mix, but this run
does not isolate that cause.

## Compute and remaining gate

The frozen first-eight-layer teacher-prefix capture took **16.73 s**;
read/tokenization took **3.06 s**. Matched adjustment-loop times were
**126.70 s full**, **128.10 s local**, and **138.04 s query** at
512 tokens. Selection took about 7.8–7.9 s per arm. The broad run's
training-stage co-resident Metal peak was **5.277 GB**; host load
moved from 4.79 to 6.38. The earlier attention-transfer stage added
**2.81 s teacher capture**, **3.56 s feature-map training** and
selection work before this LoRA stage. Query-state training was
**8.95% slower** than matched full attention even before counting
that conversion work. At 2,999 scored input tokens, logical active
model cache was **92.525 MB** query versus **98.271 MB** teacher;
that is a one-layer cache result, not deployed memory or energy.

The complete run and selected checkpoint hashes are in
`experiments/results/query_global_lora_broad.json` and the three
`query_global_lora_broad_*_weights.npz` files. The fresh window
records, paired differences, cache bytes, source/model/data hashes
and exact protocol hash are in
`experiments/results/query_global_lora_broad_holdout.json`. The
two-step smoke files are diagnostic and excluded. The unchanged
teacher still wins on Shakespeare, the query arm trails full
attention on both texts, and the [single-layer complete-model
serving benchmark](QUERY_GLOBAL_FAST_DECODE.md) showed no reliable
speed gain. A matched, more balanced two-text schedule and broader
attention-transfer map are plausible next quality tests; each needs
new untouched windows and a full-attention control before another
promotion decision.
