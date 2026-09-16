# An OSS-model baseline before an efficiency claim

**Measured outcome, 16 September 2026:** replacing one of Llama-3.2-1B's 16
attention layers with the repository's untrained gated trace does **not** make
the complete model materially faster. Held-out perplexity gets worse, from
20.3756 to 23.4074. The replacement saves the KV cache of one layer, but its
current MLX implementation uses *more* peak Metal memory at an 8,192-token
prefix. There is no training-cost result in this experiment.

This is the first run here that times the complete pretrained model's cached
prefill and decode path, rather than an isolated mixing sublayer. The corrected
fused-kernel sublayer result in `docs/FAIR_BENCH.md` remains valid within its
scope, but it is not a full-model speed result.

## Exactly what was compared

- **Teacher:** local snapshot `9535bd9b1d1dea6acafbdc4813b728796aeb28da`
  of `unsloth/Llama-3.2-1B`, bf16, on this M4 Max/128 GB machine with MLX
  0.32.2 and mlx-lm 0.31.3. The baseline uses the model's fused causal
  attention and standard KV caches.
- **Candidate:** the same frozen checkpoint, with layer 8's attention replaced
  by `h_t = sigmoid(W_g x_t) h_(t-1) + W_v x_t`, read out through transplanted
  `W_o`. `W_v`/`W_o` come from that layer's attention; decay is explicitly 0.7;
  the otherwise untrained gate is MLX-seeded at 0. The model cache holds one
  recurrent state vector at that layer. No other layer or weight is changed.
- **Quality:** the same 2,999 next-token predictions on a fixed 3,000-token
  validation window of the vendored TinyShakespeare corpus. The teacher and
  candidate use the same tokenizer and `perplexity()` function.
- **Serving workload:** batch 1 or 4, cached prefill at 128–8,192 tokens,
  followed by 32 or 64 teacher-forced single-token decode steps. Every call
  forces MLX evaluation inside the timer. Prefill projects only the *last*
  hidden state through the vocabulary head, as generation does. Future tokens
  are fixed from the same corpus, so sampling time cannot confound the arms.
- **Timing design:** the speed verdict uses one loaded checkpoint, alternates
  teacher/candidate order within each pair, excludes same-shape warmups, and
  records every repetition and load average. Separate-process runs are used to
  inspect memory without retaining both modules in one process.

The source and raw paired repetitions are in
`experiments/streaming_memory.py`, `experiments/oss_model_benchmark.py`,
`experiments/paired_model_benchmark.py`,
`experiments/results/oss_benchmark_paired_b1.json`, and
`experiments/results/oss_benchmark_paired_b4.json`. The rerun artifacts mark
completion and record the benchmark source hashes and content-addressed local
weight blob ID. Active logical KV bytes and allocated KV tensor capacity are
reported separately; the first pass of this harness mistakenly labelled the
256-token capacity reserve as active cache.

## Functional guard

The previous carrier ignored the model cache and restarted its trace from zero
on every call. That made token-by-token generation a different function from
one-pass evaluation. `streaming_memory.py` stores the trace and uses the same
transplanted weights. Two standalone split-prefix tests cover one and four
memory banks; model-level checks compare a cached 128-token call to
`[64, 32, 32]` chunks and a 129-token call to `128 + 1`. In both arms and both
batch sizes, cache offsets, top predictions and logits agree within 0.129
absolute logit units (bf16). The test aborts if the top prediction changes or
the maximum logit difference exceeds 0.5.

## Quality and speed

| arm | held-out NLL, nats/token | held-out perplexity |
|---|---:|---:|
| teacher | 3.01434 | **20.3756** |
| one-layer trace | 3.15305 | **23.4074** |

The candidate loses **0.13871 nats/token** and raises perplexity by **14.9%**.
It is not a quality-preserving conversion.

The table gives the median *within-pair* ratio, with every underlying pair in
the JSON. `prefill time` above 1 means the candidate is slower; `decode speed`
above 1 means it is faster. Small differences should be read as ties because
the raw repetitions still move with shared-machine load.

| batch | prefix tokens | prefill time, candidate/teacher | decode speed, candidate/teacher | logical cache saved |
|---:|---:|---:|---:|---:|
| 1 | 128 | 1.046 | 0.987 | 0.39 MB |
| 1 | 512 | 1.026 | 0.985 | 1.17 MB |
| 1 | 2,048 | 1.002 | 0.993 | 4.32 MB |
| 1 | 4,096 | 0.988 | 0.994 | 8.51 MB |
| 1 | 8,192 | 1.026 | 1.001 | 16.90 MB |
| 4 | 512 | 1.052 | 0.972 | 4.42 MB |
| 4 | 2,048 | 1.007 | 0.970 | 17.01 MB |
| 4 | 8,192 | 0.994 | 1.002 | 67.34 MB |

One replaced layer cannot materially reduce the other 15 layers' attention
work. The paired measurements show no complete-model throughput advantage from
this untrained conversion, even at 8,192 tokens.

At batch 1 and an 8,192-token prefix plus 32 decode tokens, separate-process
runs measured **active logical cache** of **269.48 MB** for the teacher and
**252.65 MB** for the candidate, a **16.83 MB** saving. Their allocated KV
tensor capacities were **276.82 MB** and **259.53 MB**. Peak Metal allocation
was **3.764 GB** versus **4.294 GB**,
respectively; final active Metal memory was **2.749 GB** versus **2.761 GB**.
The current conversion therefore does not reduce whole-process memory at that
point. These are allocator observations for this implementation, not a hardware
energy or dollar-cost measurement. Raw memory runs are in
`experiments/results/oss_memory_teacher_b1.json` and
`experiments/results/oss_memory_transfer_b1.json`.

## The next gate

The untrained pure trace fails both quality and end-to-end speed criteria. A
controlled [64-token local-attention hybrid](LOCAL_HYBRID.md) was tested next:
its local-only control improved quality relative to the pure trace, but still
lost to the teacher, and an untrained long-range trace gave no measured benefit
at its small gains. Its bounded cache has no verified complete-model speed or
total-memory advantage. The next design must make its long-range state content
addressable and train the replacement to match frozen-teacher outputs before
next-token adjustment. Retrieval/copying tests are needed alongside held-out
perplexity: an average language-model score can hide a lost recall capability.

Only after a *quality-matched*, multi-layer conversion beats the complete
teacher on prefill or decode and memory can this repo claim cheaper inference.
Cheaper **training** needs a separate from-scratch test of time and compute to
the same held-out loss, counting the teacher/conversion cost when pretrained
weights are reused. A paired [small-model time-to-quality result](TRAINING_PARETO.md)
has passed its five-seed gate, but it does not establish cheaper OSS-model
training. Neither claim follows from isolated sublayer timings.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python -m pytest -q tests/test_streaming_memory.py
~/zbrain/venv/bin/python experiments/paired_model_benchmark.py \
  --contexts 128,512,2048,4096,8192 --decode-tokens 64 --repeats 5
~/zbrain/venv/bin/python experiments/paired_model_benchmark.py \
  --batch 4 --contexts 512,2048,8192 --decode-tokens 32 --repeats 3
~/zbrain/venv/bin/python experiments/oss_model_benchmark.py \
  --arm teacher --contexts 8192 --decode-tokens 32 --repeats 3 \
  --output experiments/results/oss_memory_teacher_b1.json
~/zbrain/venv/bin/python experiments/oss_model_benchmark.py \
  --arm transfer --contexts 8192 --decode-tokens 32 --repeats 3 \
  --output experiments/results/oss_memory_transfer_b1.json
```

The scripts require the exact local snapshot and vendored corpus. They set
Hugging Face offline mode and refuse to silently download another checkpoint.
