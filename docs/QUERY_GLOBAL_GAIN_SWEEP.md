# Fixed gain calibration did not close the 1B quality gap

**Exploratory diagnostic measured 17 September 2026:** with the original
selected layer-8 feature map and pretrained Llama weights unchanged, the
existing local/global output-mixing gain **0.449684877** was the best of
six tested fixed gains for both aggregate next-token loss and
teacher-attention output MSE on Shakespeare, WikiText-2 raw test and a
literary book. Exact local-only was worse, and stronger global mixing
degraded quality sharply. This makes a simple scalar-gain correction an
unlikely explanation for the remaining teacher gap.

The [source](../experiments/sweep_query_global_gain.py) was pushed as
`86dbed1` before the run. It reused **two 3,000-token windows per text**
from the already published [direct-map holdout](QUERY_GLOBAL_MAP_NLL_RESULT.md):
Shakespeare 52/76%, WikiText raw test 8/78%, and pinned Project
Gutenberg book-body 10/80%. These are **previously inspected windows**.
No scalar was trained or selected for deployment, and the sweep is
not a fresh quality gate. Each arm shared the same bf16
`unsloth/Llama-3.2-1B`, original q/k/v/o/RoPE, exact 64-token local
window and selected `delta_q`/`delta_k` map. Gains tested were
**0, 0.2, 0.449684877, 0.65, 0.85 and 1.0**. A gain of zero is exact
local-only; one uses only the feature-global output when older keys
exist. All arms passed short whole/split/decode top-1 and bounded-cache
parity. The teacher and candidates scored identical persistent-cache
next-token windows with float32 loss from bf16 logits; 512-token
teacher-output prefixes supplied the MSE diagnostic.

| fixed global gain | Shakespeare NLL minus current gain | WikiText NLL minus current gain | literary NLL minus current gain |
|---:|---:|---:|---:|
| 0, local-only | +0.03126 | +0.02199 | +0.03437 |
| 0.2 | +0.00745 | +0.00227 | +0.00567 |
| **0.449684877, current** | **0** | **0** | **0** |
| 0.65 | +0.01756 | +0.02697 | +0.03017 |
| 0.85 | +0.06997 | +0.09969 | +0.10663 |
| 1.0, global-only after 64 | +0.15023 | +0.21161 | +0.20942 |

The current gain also had the lowest mean attention-output MSE among
these six on **each** text: **0.00055827** Shakespeare,
**0.00066425** WikiText and **0.00058812** literary text. Its
aggregate teacher NLL gaps were still **+0.02447**, **+0.03140** and
**+0.04795**. One WikiText window slightly favored gain 0.2, but the
two-window aggregate and the other two texts favored the existing
gain. More scalar values or fresh windows could shift the optimum;
this diagnostic does not prove a global optimum. It shows that
the coarse current blend is already near the best of the tested
scalar settings on these reused windows.

The [LoLCATs paper](https://arxiv.org/html/2410.10254v3) combines
windowed softmax and under-window linear contributions with a **joint
normalizer** and learns head-specific window factors. Our implementation
mixes two **separately normalized** outputs with one constant gain.
This sweep tests scalar calibration only; it does **not** test that
joint formula, query-dependent local/global mass, head-specific
mixing or a different feature map. Any such implementation would need
new untouched quality windows, cache/generation parity and a measured
complete-model speed and training-cost comparison. Computing an
explicit local normalizer could also add MLX kernel work, so a
quality gain alone would not meet the compute goal.

The raw six-gain records are in
`experiments/results/query_global_gain_sweep.json`: exact reused
window token hashes, per-arm NLL/PPL, attention-output MSE, logical
cache bytes, parity/counts, source/model/checkpoint/corpus hashes,
wall readings and a **3.218 GB** co-resident Metal peak. Its status
is `complete`, the source/checkpoint hashes match, and the source
commit time precedes the measurement start. Wall readings were not
paired or designed as a serving benchmark.

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python experiments/sweep_query_global_gain.py
```
