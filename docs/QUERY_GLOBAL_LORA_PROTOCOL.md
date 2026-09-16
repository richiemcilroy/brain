# Predeclared one-layer LoRA quality control

**Frozen before the primary stage-2 run:** the goal is to test whether a
small next-token adjustment closes the quality gap left by the selected
query-addressable state. This is a matched control, not a claim that
local-plus-linear attention is new; [LoLCATs](https://arxiv.org/abs/2410.10254)
already uses local/global attention, attention-output transfer and LoRA.

`experiments/train_query_global_lora.py` uses the same offline bf16
`unsloth/Llama-3.2-1B` model and only changes layer 8. Three arms receive
rank-8 LoRA on all four original q/k/v/o projections at scale 2
(alpha 16), **106,496 trainable scalars each**:

| arm | attention at layer 8 | additional frozen conversion weights |
|---|---|---:|
| `full_lora` | unchanged full softmax | 0 |
| `local_lora` | exact 64-token local softmax only | 0 |
| `query_lora` | exact local plus selected bounded query state | 163,840 stage-1 map scalars |

All pretrained tensors are shared frozen references. The LoRA A arrays
have identical seeded initialization across arms, B arrays start at zero,
and the trainer asserts that only the eight q/k/v/o LoRA arrays are
trainable in each arm, with the same parameter count. The
`query_lora` map comes from the 200-step attention-output checkpoint in
`docs/QUERY_GLOBAL.md` and stays frozen during this stage. Stage-1 teacher
capture and training still count as conversion cost even if this stage
works.

Primary schedule: 32 disjoint 512-token training windows in the first
50% of TinyShakespeare, one input plus shifted next-token label per
position, 300 single-window steps per arm with a shared seed-19 draw,
AdamW at constant 0.0003 learning rate, zero weight decay, gradient-norm
clip 1.0 and one selection read every 50 steps. The full and converted
arms use the same two fixed 512-token Shakespeare selection windows at
55% and 58%; each arm's checkpoint is the lowest mean next-token NLL
there. We will count all teacher prefix forwards, training and selection
time, peak Metal allocation and model/cache bytes. The first eight frozen
layers are captured once and their outputs are checked against full-model
teacher logits before optimizing the last eight layers. An interrupted
result remains marked `running`.

`experiments/evaluate_query_global_lora.py` fixes the following disjoint
3,000-token **next-token quality** windows before any primary stage-2
outcome:

| text | fixed percentages | scored predictions |
|---|---|---:|
| TinyShakespeare, unused by stage-2 selection | 80, 82, 84, 86 | 11,996 |
| WikiText-2 raw test, no training or selection from this text | 15, 35, 55, 75, 95 | 14,995 |

The evaluation loads only checkpoints whose SHA-256 matches the complete
primary training artifact. It scores the unchanged teacher, selected
stage-1 query map, all three LoRA arms with the same persistent-cache
float32 next-token loss from bf16 logits, and short whole/split/decode
cache parity. We will compare query LoRA with its stage-1 starting point,
local-only LoRA, unchanged teacher and full-attention LoRA, window by
window and in aggregate. A lower selection NLL alone is not a pass.

A faster-model claim would also require quality on stronger tasks and a
quality-preserving **multi-layer** conversion; complete-model prefill,
decode and training compute would then need to beat full-attention
baselines after counting both conversion stages. One converted layer can
only save a small fraction of total model cache and attention work.
