# Where the bounded query state does and does not save time

**Measured 16 September 2026 on an Apple M4 Max, batch 1, bf16
Llama-3.2-1B:** a one-layer query-state conversion is slower than the
unchanged fused attention through 16K tokens, but its *layer-only*
prefill crosses over near 32K. The current converted model still misses
teacher quality, and a one-layer crossover cannot deliver a substantial
complete-model speedup.

`experiments/profile_query_global_layer.py` captures genuine teacher
layer-8 inputs from the first eight frozen layers. Five rotated repeats
at 512/2,048/8,192 and three at 16,384/32,768 time q/k/v/o plus
attention, with a persistent cache and 32 genuine teacher-derived
decode activations. A teacher complete-model last-token-head prefill and
decode call is included in the paired order. All modules coexist in
one process, and host load varied.

| context | trained query-layer prefill time / fused teacher layer | trained query-layer decode speed / teacher layer | fused layer's share of teacher complete prefill | teacher layer cache / query layer cache after 32 decodes |
|---:|---:|---:|---:|---:|
| 512 | 5.197× | 0.523× | 1.61% | 1.114 / 0.395 MB |
| 2,048 | 5.779× | 0.583× | 1.60% | 4.260 / 0.395 MB |
| 8,192 | 3.332× | 0.690× | 2.03% | 16.843 / 0.395 MB |
| 16,384 | 1.892× | 0.815× | 2.73% | 33.620 / 0.395 MB |
| 32,768 | **0.836×** | 1.253×, noisy | 3.55% | 67.174 / 0.395 MB |

Ratios are the median of within-repeat pairs, not a ratio of separate
median times. At 8K, fused layer prefill took a median **36.9 ms** and
trained query prefill **123.7 ms**. At 32K, fused layer prefill took
**678.5 ms** and query prefill **582.1 ms**. The decode-speed readings
at 32K were **0.975×/2.685×/1.253×**, so the median should not be
treated as a stable decode win. The full teacher prefill was about
**18.3 seconds** at 32K; the attention of *one* layer contributed
about 3.55% of that. Even an instantaneous replacement of only that
layer would have an approximate Amdahl ceiling of **1.037×** complete
prefill speed. The measured layer saving is around 0.1 seconds,
roughly 0.5% of that complete prefill time, before any whole-model
overhead. This fraction is a diagnostic from separately timed calls,
not a precise component trace.

The state has fixed-size cache, but the existing MLX path pays for
many 64-token block scans, feature/value outer products, state reads
and lazy-graph evaluation. Llama's full attention calls MLX's fused
scaled-dot-product kernel. At short and medium contexts, fewer
arithmetic operations in the state formula do **not** translate to
less wall-clock time. The earlier paired [complete-model trained-map
benchmark](QUERY_GLOBAL.md#generation-serving-time-and-memory) found
roughly tied decode and no material prefill gain at 8K, consistent
with this layer share.

Larger blocks were a paired engineering diagnostic on the same 8K
layer input. With evaluation every 512 tokens in all arms,
`chunk=256, inference_sync_blocks=2` had median within-repeat prefill
time **0.719×** the 64/8 default (about **28% less**), and median
transient Metal peak **1.009×**. The maximum bf16 layer-output
difference was **0.0078125**. Decode used a one-token block regardless
of the configured chunk and showed no improvement in this run. Host
load began above **30**, so the paired trend needs complete-model
verification (`experiments/results/query_global_chunk_profile_b1.json`).
The next implementation gate is a paired cached-model test of that
configuration, followed by a custom Metal scan only if
its correctness and memory behavior can be demonstrated. [MLX
documents custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html),
including the need for a separate VJP if one is used for training.

The raw records are
`experiments/results/query_global_layer_profile_b1.json` and
`query_global_layer_profile_long_b1.json`; each stores arm order,
per-repeat times, cache bytes, source/model/corpus hashes, host load,
activation-capture time and Metal peak. They do not establish
long-context perplexity, retrieval quality, multi-layer conversion,
deployment resident memory or reduced dollars per token.

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python experiments/profile_query_global_layer.py \
  --contexts 512,2048,8192 --decode-tokens 32 --repeats 5 \
  --output experiments/results/query_global_layer_profile_b1.json
~/zbrain/venv/bin/python experiments/profile_query_global_layer.py \
  --contexts 16384,32768 --decode-tokens 32 --repeats 3 \
  --output experiments/results/query_global_layer_profile_long_b1.json
~/zbrain/venv/bin/python experiments/profile_query_global_chunks.py
```
