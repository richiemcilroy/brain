# A paired time-to-quality result at small scale

**Measured 16 September 2026:** on a two-layer, width-128 character model,
the input-only gated recurrence reached the same held-out **2.4
bits-per-character** target faster than fused causal attention in all five
paired seeds. The median within-seed attention/recurrent ratio was **3.95× in
synchronized training seconds** and **4.29× in estimated train FLOPs**. This is
a positive small-model training result. It does not establish that converting
or training a modern open-source LLM is cheaper.

## Comparison and decision rule

`experiments/train_time_to_quality.py` compares `A_attn` with `B_match` from
`experiments/content_gate.py`. The attention arm calls MLX's fused causal
scaled-dot-product kernel. `B_match` has an input-only gate
`g_t = sigmoid((W_g + P)x_t)` and parallel gated scan; the extra trainable
input projection `P` makes it parameter matched to the earlier state-dependent
gate. It cannot address an earlier key with a later query. The arms have
478,976 and 478,464 parameters, a **0.107%** spread. This is a from-scratch
comparison; no pretrained Llama weights or teacher forward pass are included.

Both arms saw each seed's identical ordered batch positions: TinyShakespeare,
character vocabulary 65, context 512, batch 16, AdamW learning rate 0.001 with
100-step warmup, clip 1, weight decay 0.01. Validation used all 217 disjoint
512-token windows of the fixed held-out split, 111,104 scored tokens. The first
full-validation checkpoint at or below 2.4 bpc, checked every 100 steps, is
the observed upper bound on the quality crossing; an arm that missed by 1,500
steps would be right censored. The 2.4 threshold and a **2×** median gate in
both time and analytic FLOPs, with at least 4/5 paired wins, were fixed in the
script before this run. The threshold sits below the measured bigram floor of
3.5806 bpc. Prior same-step loss results informed the choice of target; this
is not an independent preregistered benchmark.

## Raw crossings

| seed | attention first pass | recurrence first pass | attention train seconds | recurrence train seconds | paired time ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,100 steps | 600 steps | 16.35 | 6.75 | 2.42× |
| 1 | 1,200 | 500 | 27.15 | 6.88 | 3.95× |
| 2 | 1,300 | 500 | 27.97 | 6.29 | 4.44× |
| 3 | 1,300 | 500 | 25.89 | 6.19 | 4.19× |
| 4 | 1,200 | 600 | 23.59 | 6.74 | 3.50× |

The median paired **setup-plus-training-and-validation** wall-time ratio is
**3.01×**. The analytic training cost to the detected threshold is a median
**4.29×** lower for recurrence: about **39.1 trillion** estimated FLOPs for
attention and **9.1 trillion** for recurrence at the median seed. This estimate
uses the source's forward-plus-backward per-token cost model; it excludes exact
kernel work, optimizer operations, hardware energy and conversion cost. It is
not a FLOP counter or a dollar-cost measurement.

All ten trained arms beat the bigram floor and reached the quality target. At
1,500 steps, attention scored 2.314–2.345 bpc and recurrence 2.191–2.218 bpc.
The parallel scan agreed with its sequential reference to `4.77e-7` absolute
error. The output records first-step compilation cost separately and includes
it in synchronized training time. The one-minute host load ranged 5.04–11.03
on 16 logical CPUs, below the declared `>24` contention flag; this is a coarse
host check, not proof of GPU isolation. Median peak Metal allocation was
**0.844 GB attention versus 0.872 GB recurrence**, so peak memory did not
improve.

The complete per-seed schedule hashes, 160 validation checkpoints, crossing
brackets, timing, memory and load observations are in
`experiments/results/train_time_to_quality.json`. The two-step
`train_time_to_quality_smoke.json` is labelled diagnostic and has no efficiency
verdict. A separate raw-artifact check recomputed the corpus and schedule
hashes, every first crossing and the five paired ratios; all matched.

A fresh [source-hashed replication](../experiments/results/train_time_to_quality_r2.json)
repeated **all ten first-passing steps exactly** and all ten final bpc scores
within 0.00032. Its median paired training-time ratio was **4.13×** and its
analytic FLOP ratio **4.29×**, again with 5/5 paired wins; recorded load was
3.87–6.70. Wall-time ratios varied between the two runs, so the stable finding
is the crossing-step and estimated-compute advantage at this small scale,
alongside a multi-fold measured time advantage in both recorded runs. The
replication artifact records hashes of the training harness and its imported
model and cost-code sources.

## What needs to happen next

This width-128 character result gives a reason to test a wider, deeper model
on multiple tokenized corpora at the **same held-out quality**. The repository's
one-layer Llama conversion has already failed that quality condition, so the
small training win cannot be applied to it. The next OSS-model test must train
a quality-preserving replacement, account for any teacher or conversion cost,
and measure complete-model prefill, decode, resident memory and time to the
same held-out loss. A content-addressing or retrieval task is also needed;
bpc on Shakespeare alone does not test that capability.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
~/zbrain/venv/bin/python experiments/train_time_to_quality.py --smoke
~/zbrain/venv/bin/python experiments/train_time_to_quality.py
~/zbrain/venv/bin/python experiments/train_time_to_quality.py \
  --out experiments/results/train_time_to_quality_r2.json
```

The complete run on this host took just under five minutes in the recorded
conditions. Environment overrides in `content_gate.py` that change width,
depth or context are rejected by this fixed protocol.
