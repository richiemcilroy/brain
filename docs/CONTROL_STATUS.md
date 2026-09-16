# Control status: what the current artifacts establish

`experiments/analyze_controls.py` audits the existing JSON with the Python
standard library. It reads the raw rows, pairs arms by seed, keeps learning-rate
banks separate, and writes nothing. Run:

```sh
python3 -B experiments/analyze_controls.py
python3 -B experiments/analyze_controls.py --json
```

The files audited here are earlier local results. This note describes their
evidence and completion status; it does not promote a partial result to a
completed experiment.

## The single-layer LoRA control

`experiments/results/hybrid_lora_control.json` tests a rank-64 recurrent branch
(`T`) against two rank-64 static adapters (`L1` on the existing attention output,
`L2` on the layer input) and a random-weight recurrent branch (`R`). Each arm
trains 262,144 parameters at layer 8 of `unsloth/Llama-3.2-1B`; attention remains
in the teacher. The recurrent arms also carry 12,584,960 frozen carrier
parameters that the static adapters do not. The artifact's initialization guard
records exact teacher logits for every arm, and a separate gradient check finds
initial `up` gradient norms within 1.039× across arms.

The declared learning-rate metadata contains only `1e-5`, but the raw file
contains two banks. They must not be pooled or selected by their validation
scores after the fact.

| Learning rate | Coverage | Float32 gain vs teacher, T / L1 / L2 | Paired `L1 − T` float32 ppl | Paired t 95% interval |
|---|---:|---:|---:|---:|
| `1e-5` | 20/20 arm-seed rows; five seeds | +0.5466 / +0.6648 / +0.7220 | **−0.1182**; all five seeds favor L1 | [−0.1496, −0.0868] |
| `1e-4` | 12/20 rows; seeds 3–4 missing for every arm | +0.9201 / +0.9004 / +1.0194 | +0.0197; mixed signs | [−0.1663, +0.2057] |

Here `L1 − T < 0` means the static adapter has lower perplexity. At `1e-5`,
the exhaustive paired-bootstrap 95% interval is [−0.1368, −0.0966], and the
repo's original bf16 metric agrees in direction (`L1 − T = −0.0952`, paired t
interval [−0.1500, −0.0404]). The float32 per-seed differences are
`[−0.1462, −0.1035, −0.1266, −0.1324, −0.0823]`.

The five-seed result establishes that this single-layer validation gain does
not require the **added** recurrent memory: a static adapter of the intact
attention output does better with the same trainable parameter count. It does
not establish that the recurrent features contain no information, or that
attention can be removed. The comparison is one model, one layer, one split,
and one fully sampled learning rate. With five concordant signs, an exact
two-sided sign test is `p = 0.0625`; the interval estimates describe this pilot
and should not be generalized to larger models or workloads.

## The state-dependent gate comparison

`experiments/results/content_gate.json` is a partial TinyShakespeare run.
Attention (`A_attn`), the plain input-only gate (`B_plain`), and its
parameter/MAC-matched input-only control (`B_match`) each have five seeds;
the state-dependent gate (`C_statedep`) has only seeds 0–2; the multiscale gate
(`D_multiscale`) has no trained row. Its `timing` and `recency` fields are empty.

At the three paired seeds, `C_statedep − B_match` is
`[+0.001196, +0.024059, +0.011859]` bpc, mean **+0.012371**: every observed
seed favors the input-only gate. The paired t 95% interval is
**[−0.016047, +0.040789]**, which includes zero. Its exhaustive bootstrap
interval [ +0.003506, +0.021415 ] excludes zero simply because all three
observed differences have the same sign; with three seeds the exact two-sided
sign-test probability is 0.25. The preregistered five-seed hypothesis verdict
is therefore unavailable. The matching device itself has `B_match − B_plain =
−0.006962` bpc over five seeds, paired t interval [−0.021147, +0.007223].

The attention arm scores 2.332845 bpc against the bigram floor of 3.580577,
so the learning harness clears its stated floor guard. `B_match` and
`C_statedep` have equal recorded parameters and analytic per-token MACs.
Attention uses 1.789× C's analytic per-token MACs and is a floor/harness control,
not a compute-matched baseline. The shared machine's load average was 74.64
when this run started. Individual C/B training-wall ratios span 1.81× to
6.14×, but there is no dedicated timing result; a stable wall-clock speed claim
is unavailable.

## The recency verdict must be discarded

`experiments/results/content_gate_recency.json` is a smoke run with one training
step, one seed, and two gate-bias decays. The attention arm scores 6.095670 bpc,
worse than the 3.580577 bigram floor. Two nonconstant points force Pearson's
`r` to be either +1 or −1, so these coefficients do not show a robust trend.
The artifact predates the source's trained-gate fit fields; it correlates the
**initial** decay's fit with the one-step bpc.

The saved “SUPPORTED” verdict also reverses the fit-error sign. `cdf_l1` is a
fit **error**, so lower is better; lower bpc is also better. Thus:

| `corr(cdf_l1, bpc)` | Meaning |
|---:|---|
| Positive | Better kernel fit goes with **better** bpc; fit helps. |
| Negative | Better kernel fit goes with **worse** bpc; fit hurts. |

For the input-only arm, decay 0.9 lowers fit error from 32.028 to 24.104 and
lowers bpc from 6.12675 to 6.07889: its `r = +1` points toward fit **helping**.
For the state-dependent arm the same fit improvement raises bpc from 6.27736 to
6.28497: its `r = −1` points toward fit **hurting**. The saved verdict says the
opposite. The source has since been corrected to use this sign convention and
to reject one-step/floor-failing smoke runs; the existing raw JSON remains an
invalid historical artifact. A fuller trained sweep is still required.

These controls do not yet establish a faster transformer, lower training cost,
or an efficient conversion of an open-source model. A useful next test would
preserve a held-out evaluation protocol while comparing an actual attention
replacement against the untouched OSS teacher and a static adapter, then
measure quality, training cost, and real prefill/decode throughput under the
same hardware and token workload.
