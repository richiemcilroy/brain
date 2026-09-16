# Predeclared prediction-loss adjustment of one 1B query map

**Freeze this protocol, data helper, trainer and evaluator before the primary
run.** A balanced teacher-attention [output-transfer run](QUERY_GLOBAL_TRANSFER_BROAD_RESULT.md)
lowered output MSE on every fresh prefix but slightly worsened full-model
next-token quality on both measured texts. This gate isolates the training
objective: keep the **older, better quality map** as a warm start, then
differentiate real next-token loss through its feature-map parameters.
It does not change the bounded-attention architecture, and it does not
presume that validation loss or lower arithmetic cost means cheaper serving.
[LoLCATs](https://arxiv.org/abs/2410.10254) already studies related
local-plus-linear Llama conversion and adjustment; this is a bounded
prediction-objective engineering test, not a new attention mechanism.

The pinned offline bf16 `unsloth/Llama-3.2-1B` checkpoint, original
q/k/v/o/RoPE, exact 64-token local window, 128-feature positive global
state and gain `0.449684877` are fixed. Only layer 8 is converted.
The selected `delta_q`/`delta_k` arrays from
`experiments/results/query_global_transfer_weights.npz` start unchanged.
The runner freezes all other model weights, asserts that exactly those
**163,840** feature-map scalars are trainable, checks the original
projection hashes after training, and proves that the frozen first-eight-layer
prefix shortcut reproduces whole-teacher logits exactly. It captures that
prefix once per train/selection window, then computes full suffix logits
and next-token cross-entropy from each arm's genuine attention output.
No LoRA is trained in this stage. The unchanged fused-attention model is
a *quality reference*, while a matched trainable full-attention control
would still be required for a later stage-2 or multi-layer comparison.

Training uses **128 disjoint 512-input-plus-label Shakespeare windows**
from the first half of that corpus and **192 disjoint WikiText-2 raw train
windows** from its full train split. The [Salesforce dataset
script](https://huggingface.co/datasets/Salesforce/wikitext/blob/f5562967961a45407fa15044c5535a607200983f/wikitext.py)
provides separate raw train, validation and test splits. We reuse the
[pinned ggml-org CI archive](https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip),
SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`,
and verify split member hashes. The schedule seeded **37** visits every
train window at least once, then draws extra windows, shuffling exactly
**200 Shakespeare and 200 WikiText updates**. Thus the gradient schedule
sees **163,840 distinct input positions** and **204,800 input-token
exposures**. AdamW uses constant LR **0.0001**, zero weight decay and
gradient clip 1.0. First-eight-layer prefix capture includes the nine
selection windows, for **168,448 input positions**. Capture, gradient
updates, validation, overall loop, Metal peak, host load, exact
window/schedule/data/model/source/checkpoint hashes and initial map
gradient norms are recorded. An interrupted run remains `running`.

At step 0, step 1 and every 50 steps, the checkpoint score is half the
mean Shakespeare next-token NLL on fixed **55/58/61%** windows plus half
the mean WikiText raw *validation* NLL on **10/20/40/60/80/90%** windows.
The lowest score selects the map, including step 0. Unchanged teacher
validation NLL is recorded for context but cannot select a checkpoint.
No WikiText test, later Shakespeare quality window or literary text
enters training or selection.

The predeclared cached next-token gate scores fresh **3,000-token
Shakespeare 52/69/76%** windows, **WikiText-2 raw test
8/18/38/58/78/98%** windows, and three windows at **10/45/80%** of a
pinned literary book body. The latter is the plain text of [Project
Gutenberg eBook #11, *Alice's Adventures in Wonderland*](https://www.gutenberg.org/ebooks/11).
Its [official plain-text file](https://www.gutenberg.org/cache/epub/11/pg11.txt)
has raw SHA-256
`01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`;
normalizing line endings and removing the Gutenberg START/END wrapper
gives book-body SHA-256
`4e04ea77acf3b0215cae2089c977bc07526fc834597a1d463598d895354ba41d`
and **36,923 model tokens**. The book is outside *adjustment* data; it
may have appeared in Llama's original pretraining. No text is committed.
The helper rejects changed raw or body bytes; the official download URL
can change, so the pinned SHA is the exact-run identity. Fresh windows
are disjoint from previously inspected positions. Shakespeare 69/76%
windows sit close to older inspected windows, limiting claims of
independent prose samples. The evaluator checks conservative 8,192-token
older spans at Shakespeare 55/58/60/90% and records every input-window
hash, short whole/split/decode parity, one-layer cache bytes and
attention-output MSE on each new 512-token prefix as a diagnostic.

**Decision rule, set before outcome:** retain the direct-NLL map for a
matched full/local/query LoRA quality experiment only if its selected
NLL improves on **both** selection texts, fresh aggregate next-token
NLL is no worse than the old map on **all three** holdout texts, and no
fresh window is worse than old +**0.02 NLL**. A separate unchanged-teacher
quality gate requires aggregate NLL at most **teacher +0.01** on each
text and no individual window beyond **teacher +0.03**, in addition to
retention. Even a pass only authorizes the next matched quality test.
Complete-model prefill/decode, stronger tasks and longer contexts,
multi-layer quality, conversion costs and matched training compute
must still support a practical efficiency claim. Prior [exact decode
work](QUERY_GLOBAL_FAST_DECODE.md) sped one layer but found no reliable
whole-model gain.

```sh
cd /Users/richie/Documents/github/human-brain
curl -fLsS \
  https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip \
  -o /tmp/human_brain_wikitext2_raw.zip
curl -fLsS https://www.gutenberg.org/cache/epub/11/pg11.txt \
  -o /tmp/human_brain_alice_pg11.txt
~/zbrain/venv/bin/python experiments/train_query_global_map_nll.py --smoke
~/zbrain/venv/bin/python experiments/train_query_global_map_nll.py
~/zbrain/venv/bin/python experiments/evaluate_query_global_map_nll.py
```
