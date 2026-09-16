# Predeclared larger-data LoRA gate for one converted 1B layer

**Frozen before the primary run:** the 16K-unique-token Shakespeare
stage-2 recipe overfit all three matched arms. This gate tests whether
more diverse data and a lower learning rate preserve full-attention
quality while letting the selected query-state layer adapt. It does
not assume a serving speed win: [the exact one-token benchmark](QUERY_GLOBAL_FAST_DECODE.md)
already found none for one converted layer.

The offline bf16 Llama-3.2-1B checkpoint and the selected stage-1
163,840-scalar feature map are fixed. Only layer 8 changes. The three
rank-8 q/k/v/o LoRA arms remain `full_lora`, `local_lora`, and
`query_lora`, each with **106,496 trainable scalars**, identical
seeded A arrays and zero B arrays. The stage-1 map is frozen in the
query arm. Pretrained q/k/v/o tensors are shared frozen references;
the runner verifies their hashes and trainable scope. [LoLCATs](https://arxiv.org/abs/2410.10254)
already studies local-plus-linear Llama conversions with attention
transfer and LoRA. This protocol tests a bounded replication-informed
engineering question, not a new attention form.

The [Salesforce WikiText-2 raw dataset](https://huggingface.co/datasets/Salesforce/wikitext)
has distinct train, validation and test splits in its [dataset script](https://huggingface.co/datasets/Salesforce/wikitext/blob/f5562967961a45407fa15044c5535a607200983f/wikitext.py).
We read a [pinned
ggml-org CI archive](https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip)
with SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.
Its raw train, validation and test member hashes are specified in
`experiments/query_global_broad_data.py` and verified before use.
No text or archive is committed.

Training uses **96 disjoint 512-input-plus-label Shakespeare windows**
spread through the first 50% of that corpus and **256 disjoint WikiText
train windows** spread through the full train split. This exposes
**180,224 distinct input positions to teacher-prefix capture** across
training windows. The fixed schedule actually visits **269 distinct
windows, or 137,728 input positions** for gradient training, about
8.4× the previous stage-2 run's 16,384. The schedule is fixed at 500
single-window steps per arm: **125 Shakespeare, 375 WikiText train**,
shuffled with seed 23. All arms see the same chosen window at each
step; their update order rotates. AdamW
uses constant LR **0.0001**, zero weight decay and gradient clip 1.0.
All arms receive 256,000 input-token exposures and the same schedule.
The frozen first eight layers are captured once for all arms, checked
against full teacher logits, and counted in conversion time.

At step 0, step 1 and every 50 steps, each arm is read on two fixed
512-token Shakespeare selection windows at **55/58%** and four
WikiText **validation** windows at **20/40/60/80%**. The checkpoint
score is exactly half the mean Shakespeare NLL plus half the mean
WikiText-validation NLL. Lowest score selects that arm's checkpoint;
no test split or fresh Shakespeare window enters selection. Training,
selection and teacher-prefix capture times, Metal peak,
source/checkpoint/data hashes and host load are recorded. The final
evaluator records active cache bytes. An
interrupted run remains marked `running`.

After selection, the predeclared fresh cached next-token gate uses
3,000-token windows at **Shakespeare 77/88/93%** and **WikiText raw
test 5/25/45/65/85%**. These percentages are disjoint from each other
and from the windows previously inspected across this project, including
the hybrid work at Shakespeare 90%. The evaluator conservatively excludes
an 8,192-token span starting there when checking overlap. The
WikiText test *split* has appeared in earlier experiments, so these
are new windows within a previously used split, not a new dataset.
The evaluator scores unchanged teacher, unadjusted selected stage-1
query map and all three LoRA arms with identical persistent-cache
float32 next-token loss from bf16 logits, short whole/split/decode
parity and window-level paired differences.

Decision rule, set before outcome:

- If full-attention LoRA exceeds teacher aggregate NLL by more than
  **0.01** on either text, classify this training recipe as generally
  overfit; a query-specific failure is inconclusive.
- A query arm is ready for a multi-layer quality attempt only if its
  aggregate NLL is at most **teacher + 0.01** and **full-attention
  LoRA + 0.01** on *both* texts, and no individual fresh window exceeds
  teacher NLL by more than **0.03**.
- Report query versus local-only on every window, and compare measured
  full/query training-loop time at 512 tokens. No training-efficiency
  claim follows from analytic FLOPs or a lower selection loss alone.

Even a quality pass would only authorize a multi-layer *experiment*.
A practical cost claim still needs quality at longer contexts and
stronger tasks, complete-model prefill/decode and peak-memory gains,
multi-layer training compute, and both conversion stages counted.

```sh
cd /Users/richie/Documents/github/human-brain
curl -fLsS \
  https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip \
  -o /tmp/human_brain_wikitext2_raw.zip
~/zbrain/venv/bin/python experiments/train_query_global_lora_broad.py --smoke
~/zbrain/venv/bin/python experiments/train_query_global_lora_broad.py
~/zbrain/venv/bin/python experiments/evaluate_query_global_lora_broad.py
```
