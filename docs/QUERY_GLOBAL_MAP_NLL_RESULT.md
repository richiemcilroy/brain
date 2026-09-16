# Direct prediction-loss map training made no meaningful quality gain

**Measured 16 September 2026 after the [protocol and evaluator](QUERY_GLOBAL_MAP_NLL_PROTOCOL.md)
were pushed as `17c2fbf`:** four hundred balanced updates differentiated
Llama's true next-token loss through the existing layer-8 query feature
map. The lowest validation score was reached after **one update**;
every later scheduled read was worse. On the fixed three-text holdout,
that selected map was essentially tied with the old map on WikiText,
slightly worse on Shakespeare and the independent literary book, and
behind unchanged fused-attention Llama on all three. The predeclared
rule **rejects the direct-loss map** for another LoRA or multi-layer
quality attempt. Changing this map's objective alone did not close
the pretrained-model gap in this run.

## Fixed adjustment and selection

The runner exactly warm-started the earlier Shakespeare-selected
`delta_q`/`delta_k` map, converted only layer 8, and froze all original
pretrained tensors. A zero-logit-difference check proved that capturing
the first eight frozen layers once gave the same teacher computation as
a whole forward. Only the **163,840** map scalars received nonzero
gradients; the original q/k/v/o projection hashes remained unchanged.
The seed-37 schedule visited every one of 128 disjoint Shakespeare
first-half and 192 WikiText-2 raw train windows, for **163,840 distinct
input positions**, **204,800 token exposures**, and exactly 200 updates
per text. Each window had 512 model inputs plus labels. AdamW used
constant LR 0.0001, zero weight decay and gradient clip 1.0.

Checkpoint choice used only three Shakespeare **55/58/61%** and six
WikiText raw *validation* **10/20/40/60/80/90%** windows, with equal
weight for the mean loss from each text. The fixed scores were:

| one-layer arm | Shakespeare selection NLL | WikiText-valid selection NLL | balanced score |
|---|---:|---:|---:|
| unchanged teacher, reference only | **3.17791** | **2.49950** | **2.83871** |
| old map, step 0 | 3.20107 | 2.52456 | 2.86281 |
| selected direct-loss map, step 1 | 3.20071 | 2.52399 | 2.86235 |

The selected balanced-score gain was only **0.00046 NLL**. Reads at
steps 50, 100, 150, 200, 250, 300, 350 and 400 were all worse than
step 1. The result is a nearly unchanged early checkpoint, not a
sustained model improvement. Prefix capture took **14.84 s**;
gradient updates took **74.68 s** within an **81.64 s** loop.
Teacher/map selection reads took **8.28 s**; text read/tokenization
**3.17 s** and model load **0.60 s**. The capture and training-stage
Metal peaks were **3.524 GB** and **5.188 GB** in the shared process.
These conversion costs precede serving and cannot support cheaper
training by themselves. The full-attention teacher here is frozen,
so this run does not provide a matched *trainable* full-attention
control for attributing an overfit recipe to the query architecture.

## New prediction-quality windows

The [predeclared gate](QUERY_GLOBAL_MAP_NLL_PROTOCOL.md) scored
3,000-token Shakespeare **52/69/76%**, WikiText-2 raw test
**8/18/38/58/78/98%**, and pinned Project Gutenberg *Alice's
Adventures in Wonderland* book-body **10/45/80%** windows. The book
was outside adjustment and selection, though it may have appeared
in Llama's original pretraining. Shakespeare 69/76% windows are
disjoint from, but close to, previously inspected prose. The evaluator
checked old-window overlap, recorded exact input hashes, scored
8,997/17,994/8,997 next-token predictions respectively with
persistent caches and float32 loss from bf16 logits, and passed
short whole/split/decode top-1 and bounded-state parity for all arms.

| one layer-8 attention arm | Shakespeare PPL | WikiText-2 raw test PPL | literary book PPL |
|---|---:|---:|---:|
| unchanged fused-attention teacher | **22.0359** | **9.5546** | **7.1751** |
| old selected query map | 22.5784 | 9.8794 | 7.5444 |
| selected direct-loss query map | 22.5791 | 9.8792 | 7.5462 |

Direct-minus-old aggregate NLL was **+0.000029** Shakespeare,
**−0.000020** WikiText and **+0.000240** literary text. These tiny
directions are not a practical quality gain. Direct-minus-teacher
NLL was **+0.024352**, **+0.033409** and **+0.050422** respectively,
failing teacher +0.01 on every text. No individual window exceeded
old +0.02, but the strict predeclared **all-three-text aggregate**
condition failed on Shakespeare and the book. Fresh attention-output
MSE was essentially unchanged from the old map on all three texts.
At 2,999 scored inputs, the logical one-layer query-cache reduction
remained **92.525 MB versus 98.271 MB** teacher, the same as the old
map. This is not a measured whole-model speed or deployed-memory gain.

## Decision and evidence

The direct map reduced both validation-domain losses by a few ten-thousandths,
but selected at step 1 and failed to improve fresh aggregate quality
on all three texts. The evaluator's `direct_map_retained_for_matched_lora_quality_attempt`
and `direct_map_teacher_quality_gate_pass` flags are **false**.
This result rejects the *specific* LR, data, capacity and direct-NLL
recipe; it does not establish that prediction-level adjustment or
bounded attention cannot work. Together with the [output-MSE
failure](QUERY_GLOBAL_TRANSFER_BROAD_RESULT.md), it points to the
fixed feature-state and local/global mixing rule as the next
architecture question, before paying for more map-only training.
A quality-preserving multi-layer conversion and complete-model
prefill/decode improvement are still required for the user's
compute-cost goal.

The primary run, exact schedule/window/data/source/protocol/model
hashes, selection curve, frozen-projection check, timing and selected
weight hash are in `experiments/results/query_global_map_nll.json`
and `query_global_map_nll_weights.npz`. The fresh per-window NLL,
attention-output MSE, cache parity/counts, logical-cache bytes,
literary raw/body hashes and all input hashes are in
`experiments/results/query_global_map_nll_holdout.json`. The four-step
smoke files are diagnostic and excluded. The published protocol commit
time preceded primary training and evaluation starts, and the current
source, protocol and selected-weight hashes match both raw records.

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python experiments/train_query_global_map_nll.py
~/zbrain/venv/bin/python experiments/evaluate_query_global_map_nll.py
```
