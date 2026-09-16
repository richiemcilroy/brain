# A 64-token local-attention hybrid on the pretrained 1B model

**Measured 16 September 2026:** a signed-gain extension selected **−0.1** for
the transplanted recurrent trace. On four untouched 3,000-token windows scored
with a persistent cache, this arm beats local-only in **4/4** windows, but its
aggregate perplexity remains **22.01 versus 20.76** for the unchanged 1B
teacher. It has **no verified complete-model speed advantage**. At batch 1
and 8,192 context tokens, converting one layer saves about **16.7 MB of active
KV state** while retaining the other 15 full-attention layers.

## Prototype and controls

`experiments/local_window_hybrid.py` wraps layer 8 of the same bf16
`unsloth/Llama-3.2-1B` checkpoint used in `docs/OSS_BASELINE.md`. It keeps the
original q/k/v/o projections and rotary positions. Its local path attends
exactly to the current token and at most 63 preceding tokens, using chunked
fused attention with an explicit window mask. The optional long-range path is
the repository's value/output transplant and gated trace at decay 0.7. The
fixed `memory_gain=0` arm skips that path and is the necessary local-only
control. A signed gain scales the trace without changing its weights; the
quality-selected setting is −0.1 and the earlier positive control is 0.05.
All other pretrained layers and weights remain untouched.

The replaced layer's custom cache holds at most 63 old keys/values and a
persistent recurrent state, with an absolute offset for RoPE. Cached calls
must use `make_hybrid_cache(model)`. The official `mlx_lm.generate_step` path
accepts that cache through its `prompt_cache` argument; the default
`model.make_cache()` still creates an incompatible plain KV slot for this
prototype. The implementation adds about 12.6 million fp32 carrier scalars
(roughly 50 MB) beside the original attention weights. One layer's bounded KV
does not imply lower resident model memory.

## Quality and generation

An attention-output probe on genuine teacher layer-8 activations found that,
at query position 2,047, the median head placed **79.8%/80.4%** of its
softmax mass before the latest 64 tokens in the 60%/90% windows. The fixed
trace was *negatively* aligned with the missing full-attention output: its
best scalar gain was **−0.366/−0.358**, recovering **26.4%/25.9%** of that
output residual's MSE on those same windows. This is an in-window
least-squares diagnostic, not a model-quality or speed result
(`experiments/results/attention_mass_probe.json`).

After that probe, the exploratory extension fixed a symmetric 13-gain grid
`[-1, -.5, -.25, -.1, -.05, -.01, 0, .01, .05, .1, .25, .5, 1]`.
It chose the lowest NLL on 2,999 next-token predictions at 60% of the
vendored corpus. The 90% window was already inspected; a disjoint 95% window
was its first fresh holdout. These sweeps reset context every 512 tokens and
train nothing.

| arm | 60% selection PPL | 90% comparison PPL | fresh 95% PPL |
|---|---:|---:|---:|
| unchanged teacher | **22.6700** | **20.3756** | **18.8588** |
| local-only, gain 0 | 23.5954 | 21.3778 | 19.8922 |
| selected signed gain −0.1 | **23.5326** | **21.1509** | 19.8393 |
| positive control +0.05 | 23.5954 | 21.3778 | 19.9454 |

The selected gain is best on the 60% selection NLL, not on the evaluation
windows. On fresh 95% it narrows the local-only gap slightly but remains
**5.2% worse in perplexity than the teacher**. All thirteen gains and their
NLLs are in `experiments/results/local_hybrid_quality.json`.

A second check held the gains fixed and evaluated four *previously untouched*,
disjoint 3,000-token windows at 96–99% with a persistent cache. Each arm
scored 2,999 predictions per window in 128-token chunks, using float32
log-softmax of bf16 logits:

| arm | aggregate PPL on 11,996 scored tokens | versus teacher |
|---|---:|---:|
| unchanged teacher | **20.7592** | — |
| local-only | 22.1932 | +6.91% |
| selected signed gain −0.1 | **22.0124** | +6.04% |
| positive control +0.05 | 22.2864 | +7.36% |

The signed gain beat local-only on **all four** windows, with per-window
NLL differences from **−0.00649 to −0.01147**. All four are adjacent
Shakespeare windows on one checkpoint, so they do not establish wider model
quality. Cache offsets and the one-layer bounded KV state were checked in
`experiments/results/local_hybrid_cached_quality.json`.

The 64-token local function matches a dense fused sliding-window reference
in tests, including the 63/64-token cache boundary. On the real 1B checkpoint,
teacher, local-only, positive hybrid and signed hybrid all preserved the top
prediction for an 8,192-token prefix supplied whole or split at 4,096, then
followed by one decoded token. The maximum one-pass-versus-decoded logit
difference was **0.125** in bf16; cache offsets matched. With a 129-token
prompt, 64-token prefill chunks and four greedy outputs, every arm's
`mlx_lm.generate_step` token IDs matched a fresh-cache manual greedy loop.
Different arms may generate different outputs. The raw IDs, probabilities
and offsets are in `experiments/results/verify_generation.json`.

## Complete-model speed and memory

The interleaved benchmark uses a serving-style last-token vocabulary head,
teacher-forced single-token decode and evaluated calls inside each timer.
Each **seven-repeat** run rotates the teacher, local-only and hybrid arm
order. Ratios below are medians of within-repeat pairs; prefill below 1 is
faster, decode above 1 is faster.

| fixed gain | context | hybrid prefill time / teacher | hybrid decode speed / teacher | local-only decode speed / teacher |
|---:|---:|---:|---:|---:|
| −0.1 selected | 512 | 1.017 | 0.957 | 0.998 |
| −0.1 selected | 2,048 | 0.973 | 1.002 | 0.980 |
| −0.1 selected | 8,192 | **1.022** | **0.969** | 1.018 |
| +0.05 control | 512 | 1.029 | 0.980 | 1.003 |
| +0.05 control | 2,048 | 1.027 | 0.964 | 1.003 |
| +0.05 control | 8,192 | **1.074** | **0.942** | 0.989 |

The quality-selected gain has **no complete-model speed win** at 8,192:
prefill and decode were both slower in the measured median. A few other
ratios cross 1, but the signed 2,048 local-only decode ratios span
**0.769–1.044** within one run, and the two runs began at one-minute
host loads **3.21** and **7.05**. These are one machine's short,
nonisolated measurements, not a scaling or cost verdict. The current-source
raw readings, order and load are in
`experiments/results/local_hybrid_benchmark_signed_b1.json` and
`local_hybrid_benchmark_positive_b1.json`.

At 8,192 prefix plus 32 decoded tokens, active logical cache was **269.48 MB**
for the teacher, **252.77 MB** for local-only, and **252.78 MB** for the hybrid.
The saving is about **16.7 MB** for one replaced layer. The benchmark holds
teacher and candidate modules in one process, so its Metal allocator peak is
not a deployment-memory comparison. In the signed run, measured transient
Metal peak at 8,192 rose from **3.81 GB** for teacher/local-only to
**4.26 GB** for the hybrid. A separate-process implementation and
quality-matched multi-layer conversion are needed before claiming lower total
memory or cost.

## Next falsifier

The fixed gated trace is not query addressed: its update depends on input,
but a later query cannot select an earlier key. The signed gain confirms a
small, consistent improvement over local-only on this corpus, yet still leaves
a quality gap to the teacher.

[LoLCATs](https://arxiv.org/abs/2410.10254) is direct prior art for the
obvious next step: it already combines a 64-token exact local window with
query-addressed linear global state, trains attention-output matching and
then uses LoRA to recover model quality. A plain local-plus-linear
implementation here would be a baseline, not a novel faster transformer.
The next candidate needs to beat that prior/control on **quality-matched**
conversion cost and complete-model serving cost. It must account for teacher
forwards and training, cached generation, longer-context retrieval, and
multi-layer quality. A one-layer cache saving alone is not the goal.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python -m pytest -q tests/test_local_window_hybrid.py
~/zbrain/venv/bin/python experiments/attention_mass_probe.py
~/zbrain/venv/bin/python experiments/local_hybrid_quality.py
~/zbrain/venv/bin/python experiments/local_hybrid_cached_quality.py
~/zbrain/venv/bin/python experiments/verify_generation.py
~/zbrain/venv/bin/python experiments/benchmark_local_hybrid.py \
  --contexts 512,2048,8192 --repeats 7 --memory-gain -0.1 \
  --output experiments/results/local_hybrid_benchmark_signed_b1.json
~/zbrain/venv/bin/python experiments/benchmark_local_hybrid.py \
  --contexts 512,2048,8192 --repeats 7 --memory-gain 0.05 \
  --output experiments/results/local_hybrid_benchmark_positive_b1.json
```

The offline benchmark requires the local snapshot and the vendored corpus;
shared-machine throughput can vary between runs.
