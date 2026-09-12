# Depth beats columns: "simultaneous thinking" buys latency, not quality

**Verdict: the parallel-columns ("simultaneous thinking") hypothesis is REFUTED
as a quality win, and it was refuted by its own control.** Two depth-1 columns
score the same as a single dense layer of matched width, so columns are a
restricted wide layer and not a new computational unit. A plain **depth-2**
stack beats both columns *and* the matched dense layer, with **fewer**
parameters. The only thing columns change is the dependency chain.

There is a second, genuinely useful result buried in this table: a 445,440
parameter gated-memory stack reaches **2.2009 bpc**, which is **better than the
875,520 parameter attention baseline at 2.2528 bpc** — the better architecture
wins with **half the parameters**.

## Method

Char-level TinyShakespeare, `ctx=512`, 1500 steps, batch 16, AdamW `lr=1e-3`
with 100-step warmup and gradient clipping at 1.0, **2 seeds**, validation over
16 batches of 64. Unigram floor 4.8292 bpc, **bigram floor 3.5806 bpc**. Every
arm is well clear of the floor, so all of them have learned to use context.

`experiments/parallel_columns.py`. Summary over seeds (mean):

| arm | params | bpc | vs bigram floor | fwd ms | tok/s |
|---|---|---|---|---|---|
| `cols2_mem` (2 columns x depth 1) | 494,720 | 2.2632 | +1.317 | 26.3 | 91,414 |
| `dense1_mem` (1 layer, d=192, BLOCK-matched not width-matched) | 531,072 | 2.2647 | +1.316 | **11.5** | **162,710** |
| `depth2_mem` (2 layers in series) | **445,440** | **2.2006** | **+1.380** | 18.0 | 133,090 |
| `cols2_attn` (2 attention columns) | 528,256 | 2.4110 | +1.170 | 23.5 | 110,669 |
| `depth2_attn` (2 attention layers) | 478,976 | 2.3586 | +1.222 | 21.5 | 112,136 |
| `A_attention` (4-layer baseline, from the matched run) | 875,520 | 2.2564 | +1.324 | — | 48,466 |

## The three comparisons that matter

**1. Columns buy no quality over a dense layer of similar size — but the
control has a defect, stated plainly.** `cols2_mem` 2.2632 (494,720 params)
versus `dense1_mem` 2.2647 (531,072 params): a difference of **0.0015 bpc**,
with the paired difference disagreeing in sign between the two seeds
(-0.014 at seed 0, +0.011 at seed 1). So there is no detectable quality
difference.

**Correction to an earlier version of this document.** I originally described
`dense1_mem` as *width-matched*. It is not. Two columns of `d=128` give 256
effective features; `dense1_mem` was built at `d=192`. The two are matched on
**block parameters** (within ~1%), not on width. That distinction matters,
because the theoretical argument for why columns should equal a dense layer
(that K depth-1 columns through a single linear mixer cannot mix their internal
features, making their function class a strict subset of a denser layer) only
applies to a **superset** control at width 256, which was not run. The
theoretical argument therefore remains an argument; the *empirical* finding is
the narrower one, that columns do not beat a comparable dense layer here.

The correct one-variable control is residual `d=128` with memory width 256 and
the MLP hidden width chosen to land near 495K total parameters. That control has
not been run and is the honest next experiment.

**2. Depth beats both, with fewer parameters — this is the result worth
leading with.** `depth2_mem` reaches 2.2006 with **445,440** parameters, against
2.2632 with 494,720 for columns and 2.2647 with 531,072 for the comparable dense
layer. It is the only comparison here where the winner has **fewer** parameters
across every seed, and it was pre-registered.

Paired per-seed differences (negative = `depth2_mem` better), both seeds agreeing
in sign:

| comparison | seed 0 | seed 1 | sign agrees |
|---|---|---|---|
| `depth2_mem` - `cols2_mem` | -0.056 | -0.069 | yes |
| `depth2_mem` - `dense1_mem` | -0.070 | -0.058 | yes |
| `cols2_mem` - `dense1_mem` | -0.014 | +0.011 | **no** |

Extra seeds for `depth2_mem` reinforce it: seeds 2/3/4 give 2.2172 / 2.2028 /
2.2181, so all five seeds land in 2.1839-2.2173.

**3. Dense measures faster than columns — but this is an implementation
artifact, not an architectural fact.** 11.5 ms versus 26.3 ms in the forward
pass. Two parallel columns *cannot architecturally* be slower than two
sequential layers of the same size, so the correct reading is that the column
version executes `n_col` separate small matmuls plus a `(n_col+1)*d -> d` mixer
as separate kernels, while the dense version is one matmul. At batch 16 x 512
tokens and `d=128` every arm here is launch-overhead bound, so these
milliseconds count kernel launches, not FLOPs.

I am therefore **withdrawing the claim that columns fail on latency.** The
honest statement is: in this unfused Python implementation, columns measure
slower. Testing the latency question properly requires fusing the columns into
one batched matmul and re-timing with device synchronisation, warmup, and a
batch large enough to saturate the device. That has not been done, so the
original "or even latency" conclusion is not supported.

## Why depth wins, and what it implies

A single-layer column cannot condition its memory readout nonlinearly on its own
earlier output. Composing two layers allows exactly that, which is what
induction-head-style computation requires (Olsson et al. 2022, in-context
learning and induction heads). Two parallel depth-1 columns add **capacity**
without adding **composition**, and the measurement says composition is what
this task rewards. An independent review predicted this outcome before the run
and the prediction is what the data shows.

## The parameter-efficiency result, stated carefully

`depth2_mem` at 445,440 params scores 2.2006; the 4-layer attention baseline at
875,520 params scores 2.2564. That is a **0.0558 bpc better score with 49% of
the parameters**.

**This differs in depth AND in mixing primitive, so it is not a clean
architectural comparison, and it is made at a training budget that may favour
the recurrent arm.** 12.3M tokens is about 12 passes over the corpus, roughly
15% of the nanoGPT Shakespeare reference budget; a 4-layer transformer at that
many tokens is plausibly still descending, and gated-memory layers carry a
recency prior that char-level text rewards, so they converge in fewer steps.
Parameter efficiency is also a **curve, not a point** — one size per family is
not a Pareto claim. The defensible statement is the budget-qualified one: *at
this training budget and these shared hyperparameters*, a 2-layer gated-memory
model with about half the parameters matches or beats a 4-layer attention
baseline. Verifying it requires attention trained to 5000 steps with the curve
logged, and at least three sizes per family trained past the crossover.

The cleaner comparison is same-depth: `depth2_mem` (445,440) versus
`depth2_attn` (478,976), where only the primitive changes. That gap appeared to
be 0.158 bpc, which is **large enough to be suspicious** — a difference that
big from swapping a mixing primitive is more often an undertrained baseline than
a real architectural win. It needs >=5 seeds, a per-family learning-rate sweep,
and the causality checks before it is stated.

**What this does NOT show.** It is not a claim that gated memory beats attention
in general, and it must not be read as one:

- one task (char-level TinyShakespeare), one context length (512), two seeds;
- these are two different parameter budgets, not a matched-parameter comparison
  at equal depth — the honest matched comparison at `d=128` is still the one in
  `docs/EFFICIENCY.md`, where the two arms land within noise of each other;
- at `d=128, T=512` both architectures are launch-overhead bound on this GPU,
  so none of these throughput numbers transfer to the compute-bound regime;
- the long-context advantage is where the mechanism actually pays off, and it is
  measured separately in `docs/EFFICIENCY.md` §9.

## Causality verified

Both memory arms pass two independent checks, which is what makes the numbers
above worth anything at all (`experiments/causality_check.py`).

**Perturbation test.** In eval mode, changing the token at position `p` must
leave the logits at every position `< p` unchanged. Measured max |delta| before
`p`:

| arm | perturbed position | max abs delta before p | delta at p | verdict |
|---|---|---|---|---|
| `depth2_mem` | 16 | **0.000e+00** | 2.924e-01 | causal |
| `depth2_mem` | 31 | **0.000e+00** | 2.669e-01 | causal |
| `cols2_mem` | 16 | **0.000e+00** | 5.434e-01 | causal |
| `cols2_mem` | 31 | **0.000e+00** | 6.235e-01 | causal |

Not "small" — **exactly zero**. No backward information flow.

**Random-data floor.** Trained on i.i.d. uniform characters there is nothing to
learn, so the loss must converge to `log2(65) = 6.0224` bpc and cannot go below
it without reading the answer from a forbidden position:

| arm | bpc on random data | vs uniform | verdict |
|---|---|---|---|
| `depth2_mem` | 6.0272 | +0.0048 | no leak |
| `cols2_mem` | 6.0267 | +0.0043 | no leak |

Both sit slightly *above* the uniform entropy, as they must.

## Statistical caveats, stated

- **Two seeds per arm for most cells is not enough to quantify a null.** At n=2
  the 95% t-multiplier on the standard error is 12.7, versus 4.3 at n=3 and 2.8
  at n=5. The `cols2_mem` vs `dense1_mem` tie can only be stated as "no
  detectable difference", not as an equivalence, until more seeds exist. Five
  seeds is the target for any comparison that gets stated.
- **Seeds are paired, and there is a shared effect.** Seed 1 beats seed 0 in
  every arm that has two seeds. Under independent noise that happens about 6% of
  the time, so something shared (data order or the sampled validation windows)
  is doing work. Differences should therefore be reported **paired**, with the
  count of seeds where the sign agrees, not as independent means.
- **Sampled validation oversamples.** 16 batches of 64 at `ctx=512` is 1,024
  windows drawn from an ~111K-character split, so the split is covered about
  five times over. Deterministic non-overlapping windows (about 217 of them)
  would be the correct evaluation.
- **Selection across ~15 arms adds optimism.** Picking the best-looking arm from
  a table inflates it by roughly 1.7 standard errors, which at a two-seed SE of
  ~0.015 bpc is 0.02-0.03 bpc. This applies to comparisons chosen *after* seeing
  the table; pre-registered comparisons (`depth2_mem` > columns, which was
  stated before running) are not exposed to it.
- **Hyperparameters were tuned on the memory family across earlier rounds**, so
  the attention arms may be off their optimum. A three-point learning-rate sweep
  per family is the minimum to address this.
- **Train bpc is not reported.** At ~12 passes with no dropout, overfitting and
  underfitting cannot be distinguished from validation alone.

## Reproduce

```
python3 experiments/parallel_columns.py
```

Raw run: `/Volumes/T9/human-brain/scratch/cols.log`.
