# Exact one-token decode helped one layer, not the 1B model

**Measured 16 September 2026 on an Apple M4 Max, batch 1, bf16
Llama-3.2-1B:** an opt-in one-token query-state path reproduced the
selected 1B conversion's layer outputs and bounded state exactly, and
raised that layer's decode throughput by a median **1.170× at 8K** and
**1.182× at 32K** over the general path. The paired complete model did
not retain the gain: fast/general decode throughput was **1.005× at
8K** and **0.962× at 32K**. The teacher was still competitive. This
is an exact layer optimization and a negative complete-model serving
result, not a cheaper or quality-matched model.

`experiments/query_global_fast_decode.py` specializes only evaluation
calls with one token and no caller-supplied mask. A current query reads
the previously accumulated older-key state and fused exact local
attention over at most 64 keys. After output, one evicted local key
enters the fixed-size global feature state for the next query. The
general routine builds prefix sums, pads and gathers state for every
query in a multi-token block; a single-token update needs none of
those operations. Multi-token prefill, training and masked calls use
the unchanged general path. The selected stage-1 map, local/global
gain, pretrained q/k/v/o weights and 256/2 prefill configuration are
identical between the two query arms. This is an engineering
specialization of the previously measured local-plus-linear baseline;
it makes no new-attention claim.

The behavioral tests cover a two-batch fixed map with and without
global state, a bf16 selected-style trainable map, exact cache tensors
and counts, and caller-mask fallback. The official `mlx_lm.generate_step`
four-token greedy sequence and a fresh-cache manual loop agreed between
general and fast arms, with IDs **[323, 584, 527, 539]**. An 8,192-token
whole/split/decode prefix passed the existing top-1/logit cache parity
and retained **8,130** older keys after one decode. The complete-model
short prefill and next-token logits were **bit identical**, with 66
older keys in both caches. The raw checks are in
`experiments/results/verify_fast_decode_generation.json` and
`query_global_fast_decode_benchmark_b1.json`.

The layer profile used genuine teacher layer-8 inputs, 32 genuine
teacher-derived one-token decode activations, the frozen selected map,
and fused teacher attention in the same paired process. Every one of
the 32 general/fast layer outputs and the final feature state differed
by **zero** at both 8,192 and 32,768 tokens. Both caches remained
**395,264 bytes**; their older-key counts after decode were 8,161 and
32,737 respectively.

| context | fast/general layer decode throughput | fast/fused-teacher layer decode throughput | fast/general complete-model decode throughput | fast/fused-teacher complete-model decode throughput |
|---:|---:|---:|---:|---:|
| 8,192 | **1.170×** (5 pairs) | 0.814× | **1.005×** (5 pairs) | 1.015× |
| 32,768 | **1.182×** (3 pairs) | 1.359× | **0.962×** (3 pairs) | 0.966× |

Each cell is the median of ratios within paired repeats, with arm
order rotated. At 32K, complete-model fast/general decode readings
were **1.817×, 0.962×, 0.951×**; the large first reading conflicts
with the other two. Fast/general complete-model prefill ratios were
**1.102×, 0.895×, 1.060×**, even though both modules run the same
multi-token prefill code. Those spreads and host load near 22 at the
start preclude a stable speed claim. At 8K, all five complete-model
fast/general decode readings were between **0.990× and 1.020×**. The
32K query cache saved **66.779 MB** of logical active cache relative
to the complete teacher (**1,008.011 MB** versus **1,074.790 MB**),
but fast and general query caches were equal. At 8K the corresponding
complete caches were **253.037 MB** and **269.484 MB**. Allocator
peaks are co-resident MLX-process observations, not deployed resident
memory.

The layer profile is
`experiments/results/query_global_fast_decode_profile_b1.json`; the
complete-model pairs are
`experiments/results/query_global_fast_decode_benchmark_b1.json`.
Both retain source/model/corpus/checkpoint hashes, arm order, per-repeat
timings, memory and host load. The complete benchmark uses the same
last-token-head prefill and persistent-cache teacher-forced decode
harness as the earlier trained-map serving work. The selected
conversion still misses teacher perplexity on fresh Shakespeare and
WikiText-2 windows; these timing records do not establish matched
quality, long-context retrieval, multi-layer training efficiency, or
lower dollars per token.

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python -m pytest -q tests/test_query_global_fast_decode.py
~/zbrain/venv/bin/python experiments/verify_fast_decode_generation.py
~/zbrain/venv/bin/python experiments/profile_query_global_fast_decode.py
~/zbrain/venv/bin/python experiments/benchmark_query_global_fast_decode.py
```
