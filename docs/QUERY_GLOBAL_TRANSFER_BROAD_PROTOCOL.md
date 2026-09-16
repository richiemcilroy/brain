# Predeclared balanced attention-transfer gate for one 1B layer

**Freeze this protocol, trainer and evaluator before the primary run.** The
published layer-8 feature map was selected after learning from only 32
Shakespeare windows, or 16,384 distinct input positions. It improved a
second text without training on it, but missed the unchanged Llama teacher's
next-token quality on both texts. The later [broader LoRA run](QUERY_GLOBAL_LORA_BROAD_RESULT.md)
learned WikiText at a cost to Shakespeare; even its full-attention control
degraded there. This stage isolates the **attention map** from LoRA and tests
balanced transfer before any more converted layers. [LoLCATs](https://arxiv.org/abs/2410.10254)
already studies local-plus-linear Llama conversion with attention transfer
and LoRA. This is an engineering test of that prior-art-informed route.

The offline bf16 `unsloth/Llama-3.2-1B` checkpoint, its original
q/k/v/o/RoPE, layer 8, exact 64-token local window, 128-feature positive
global state and gain `0.449684877` are fixed. The trainer **warm-starts**
the previously selected 163,840-scalar `delta_q`/`delta_k` map from
`experiments/results/query_global_transfer_weights.npz`; it does not
modify that artifact. Only those two feature arrays are trainable. Original
pretrained tensors and all other layers are frozen. The target is the
unchanged teacher's full causal attention output on genuine layer-8 inputs.
Loss is mean squared output error **after the first 64 positions** of each
512-token window. This is a surrogate for downstream model quality, which
the separate holdout gate measures.

Training uses **128 disjoint 512-token Shakespeare windows** evenly spread
through the first half of that corpus and **256 disjoint 512-token windows**
from WikiText-2 raw *train*. The [Salesforce dataset script](https://huggingface.co/datasets/Salesforce/wikitext/blob/f5562967961a45407fa15044c5535a607200983f/wikitext.py)
defines separate train, validation and test splits. We read the same [pinned
ggml-org CI archive](https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip)
as the broad LoRA run, SHA-256
`ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`,
and verify member hashes before tokenizing. No corpus text or archive is
committed. The training windows expose **196,608 distinct input positions**,
12 times the earlier stage-1 run; teacher capture also includes six
selection windows, for **199,680 distinct positions** total.

The fixed seed-31 schedule visits every training window at least once,
then draws 172 additional Shakespeare and 44 additional WikiText windows.
It shuffles the combined list, preserving exactly **300 single-window steps
per text**, 600 steps and **307,200 input-token exposures**. Each step uses
AdamW at constant LR **0.002**, zero weight decay and gradient clip 1.0.
At step 0, step 1 and every 50 steps, checkpoint selection reads
Shakespeare **55/58%** windows and WikiText raw *validation*
**20/40/60/80%** windows. The exact score is half the mean Shakespeare
attention-output MSE plus half the mean WikiText-valid MSE. The lowest
score selects the checkpoint, including the warm-start step 0. Neither
WikiText test nor a fresh Shakespeare quality window enters training or
selection. Source/protocol/model/checkpoint/data/window/schedule hashes,
teacher-capture, gradient-update, selection and overall loop time, Metal
peak and host load are recorded.
An interrupted run remains marked `running`.

After selection, the fixed fresh cached next-token gate uses **3,000-token
Shakespeare 64/66/68%** windows and **WikiText-2 raw test 2/22/42/62/82%**
windows. They are disjoint from one another and from previously inspected
windows in this project. The evaluator checks conservative 8,192-token
spans at older Shakespeare 55/58/60/90% positions; the 90% position
includes prior hybrid work. These are new windows within texts and a test
split already inspected in earlier runs, not a new dataset. Each arm uses
the same cached bf16 checkpoint and float32 next-token loss. The evaluator
compares unchanged fused-attention teacher, the old selected map and the
new selected map; it also records 512-token attention-output MSE on the
fresh prefixes and short whole/split/decode cache parity. Neither
attention MSE nor validation score alone establishes model quality.

**Decision rule, set before outcome:** retain the broad map for a balanced
stage-2 quality attempt only if its selected attention-output MSE improves
on **both** selection texts, its fresh aggregate next-token NLL is no worse
than the old map on **both** holdout texts, and no fresh window is worse
than the old map by more than **0.02 NLL**. A multi-layer quality attempt
requires those conditions **and** new-map aggregate NLL at most
**teacher + 0.01** on both texts, with no individual fresh window beyond
**teacher + 0.03**. A stage-1 pass is an experiment gate, not a useful
compute saving. Stronger tasks and longer contexts, a matched
full-attention stage-2 control, multi-layer quality, complete-model
prefill/decode speed and all conversion/training costs remain required
before a cost claim. The [exact single-layer decode benchmark](QUERY_GLOBAL_FAST_DECODE.md)
already showed no complete-model serving gain.

```sh
cd /Users/richie/Documents/github/human-brain
curl -fLsS \
  https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip \
  -o /tmp/human_brain_wikitext2_raw.zip
~/zbrain/venv/bin/python experiments/train_query_global_transfer_broad.py --smoke
~/zbrain/venv/bin/python experiments/train_query_global_transfer_broad.py
~/zbrain/venv/bin/python experiments/evaluate_query_global_transfer_broad.py
```
