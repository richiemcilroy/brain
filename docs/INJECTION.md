# Does our module improve a pretrained LLM?

**Yes, measurably — but most of the gain is not from our weights, and the honest
number for what the transplant specifically buys is small.**

Teacher: `unsloth/Llama-3.2-1B` (1,235,814,400 params, 16 layers)
Eval: token-level perplexity, 3000 held-out TinyShakespeare tokens, deterministic
(no sampling), so these are exact numbers. Train = first 50% of the corpus,
select = 60-80%, val = last 10%, disjoint by construction and asserted in code.

## The setup, and why it is a fair test

Every earlier experiment in this repo **replaced** an attention sublayer. That is
destructive: it can only measure how much damage the swap does, and its ceiling
is the teacher. This one keeps attention untouched and **adds** our gated memory
alongside it:

```
x = x + attn(ln1(x)) + up(down(mem(ln1(x))))
```

`up` is initialised to zero, so at step 0 the model is **bit-for-bit the
pretrained teacher** — asserted, not assumed, and a run now aborts if it is not
exact. Any perplexity below the teacher is therefore a real improvement over an
unmodified pretrained model, not a recovered fraction of self-inflicted damage.

Only the low-rank projection is trained (262k parameters, 0.02% of the model).
Everything else — attention, MLP, embeddings — stays frozen.

## The result

| arm | val ppl | vs teacher | what it is |
|---|---|---|---|
| **teacher (unmodified)** | **20.3756** | — | baseline |
| **`transfer`** | **19.8922** | **−0.4834** | memory with W_v/W_o transplanted from the attention in the same layer |
| `summary` | 19.9454 | −0.4302 | same module, no transplant — a learned exponential summary of the past |
| `random` | 19.9987 | −0.3769 | same module, random W_v/W_o at matched weight scale |

**All three arms beat the unmodified model.** The ordering is exactly what the
mechanism predicts — transplant < summary < random — but the spread between them
is much smaller than the spread between "any branch at all" and "no branch".

## A reproducibility defect in the headline number, now fixed

An independent worker re-ran the committed command and got **19.8393** for the
`transfer` arm where this repo records **19.8922** -- a 0.053 ppl spread on a
result whose claimed transplant-specific effect is 0.1065 ppl. The teacher
matched exactly in both runs, so this was not the data, the tokeniser, or the
evaluation window.

The cause: `Injection.__init__` draws the branch's `down` projection from
`mx.random.normal`, and **the harness never seeded MLX**. It seeded numpy
(`np.random.default_rng(seed)`), which controls the batch ORDER only. MLX's
global RNG is seeded implicitly and differs run to run, so the branch's
initialisation -- the thing being trained -- was different every time. Repeated
in this repo, unseeded MLX draws give sums 10.26 / 14.74 / 14.67 where seeded
draws agree exactly.

This is the fifth measurement defect found in this project and it is the most
uncomfortable, because unlike the others it does not change the DIRECTION of a
claim, only the precision of a number that was being quoted to four decimals.
The margin was 0.1065 ppl and the run-to-run spread is 0.053 ppl: **the
transplant-specific margin is roughly half the noise floor it was measured
against.** The honest statement is that the 1B injection result is
directionally reproducible but its specific margin is not resolvable at one
seed.

Fixed: `main()` now calls `mx.random.seed(arm_seed * 1000 + arm_i)` before each
arm is constructed, so the init is a deterministic function of the arm and seed.
The committed 19.8922 predates this fix and should be treated as one draw from a
distribution whose spread is ~0.05 ppl, not as an exact value.

## What this does and does not establish

**Established:**

- Adding a 262k-parameter trained branch driven by our recurrent primitive
  **improves a 1.24B pretrained transformer on held-out perplexity**, from
  20.3756 to 19.8922 (−2.37% relative). This is the first result in this project
  that improves on a pretrained model rather than recovering damage it caused.
- The transplant beats a **rigorously scale-matched random control** under
  identical training: 19.8922 vs 19.9987, a margin of **0.1065 ppl**. So
  attention's own weights do carry information that random weights do not.

**The honest decomposition — and this is the number that matters:**

Of the 0.4834 total improvement, the matched random control already achieves
0.3769. **78% of the gain is "a trained low-rank branch helps", and only 22%
(0.1065 ppl) is attributable to the transplanted weights specifically.**

Anyone quoting "our brain method beats Llama-3.2-1B by 0.48 ppl" without the
control would be overstating the contribution by a factor of ~4.5.

**Not established:**

- **Single seed, single layer (8 of 16), single corpus.** No error bars. The
  0.1065 transplant margin in particular has not been shown to exceed seed
  variance, and should not be quoted as a stable effect until it is.
- **Perplexity is not capability.** A −2.37% in-distribution perplexity change
  says nothing about generation quality, reasoning, or any downstream task.
- **Two earlier bugs in this exact experiment were found and fixed**, and both
  were the kind that produce plausible numbers rather than crashes:
  (1) a dtype promotion that silently ran the model in float32 and moved
  perplexity by 0.23 on its own (`docs/DTYPE_BUG.md`); (2) `module.unfreeze()`
  recursively unfreezing the wrapped attention, which trained the whole layer,
  drove ppl to 1889, and corrupted the model for later arms. Both now fail
  loudly. A third issue — a full-rank 4.19M-parameter projection memorising the
  training set (train loss fell while val ppl went 20.4 → 69.1) — is why the
  projection is low-rank.
- **The comparison has not yet been independently verified.** An adversarial
  verification job is in flight.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
HF_HOME=~/zbrain/hf INJ_ARMS=transfer,random,summary INJ_STEPS=400 INJ_LR=1e-4 \
  INJ_RANK=64 INJ_DECAY=0.7 ~/zbrain/venv/bin/python experiments/hybrid_inject.py
```

Writes `experiments/results/hybrid_inject.json`. The run aborts immediately if
any arm fails to reproduce the teacher exactly at initialisation.
