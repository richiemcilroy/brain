# The dtype promotion bug: a mathematical no-op that moved perplexity by 0.23

**Status: found and fixed. It invalidates any earlier comparison between an
arm that ran in float32 and a teacher that ran in bfloat16.**

## The bug

`unsloth/Llama-3.2-1B` runs in **bfloat16**. Our module's `nn.Linear` layers are
created in **float32** by default. So in

```python
x = x + mem(x_norm)
```

`mem(x)` returns float32, `x` is bfloat16, and MLX promotes the sum to float32 —
**the entire residual stream then runs in float32 for every subsequent layer**.

The demonstration that makes this unambiguous: add the module multiplied by
**exactly zero**, which is mathematically the identity function.

| arm | model output dtype | max abs logit change vs teacher | val ppl |
|---|---|---|---|
| teacher, untouched | bfloat16 | — | 27.5975 |
| `x + 0.0 * mem(x)` | **float32** | 1.94e-01 | **27.8236** |
| `x + (0.0 * mem(x)).astype(bf16)` | bfloat16 | **0.0** | 27.5975 |

A no-op changed perplexity by 0.23 — larger than several of the effects this
project has been reporting — and the logits differed by up to 0.194. Nothing
about the module's content caused this; the precision of the whole network
changed.

The fix is one cast: bring the branch back to the residual's own dtype before
adding it.

## Why it took a while to find

Three hypotheses were tested and eliminated before the real one:

1. **The attention module mutates its cache.** Eliminated by reading
   `mlx_lm.models.llama.Attention.__call__` — it is stateless when
   `cache is None`, which is how the block calls it.
2. **The memory module emits NaN/Inf.** Eliminated by direct measurement: no
   NaN, no Inf, max abs 2.32, rms 0.46.
3. **`perplexity` is not reproducible.** Eliminated by calling it four times on
   an untouched model: identical to 15 significant figures.

What actually localised it was a differential test — the same wrapper with and
without the multiply-by-zero branch — which showed the *presence* of the branch,
not its value, was what mattered. From there, printing `logits.dtype` gave the
answer immediately.

The general lesson, which is why this file exists: **a numerical comparison
across two code paths must assert up front that both paths produce the same
output on a no-op input.** The identity check in `hybrid_inject.py` now does
exactly that, and it is **fatal** rather than a warning:

```python
if not exact:
    raise RuntimeError("identity check FAILED ... this is fatal rather than a warning")
```

The earlier `hybrid_parallel.py` run *did* print a MISMATCH at its identity
check and continued anyway. That was the correct signal, ignored because it was
wired to a print rather than an exception. Its `mem_transfer` cascade of
worsening perplexity as gain rose is therefore suspect: part of that slope may
be precision drift rather than the module's content.

## What this invalidates

- **`hybrid_parallel.py` results** (`results/hybrid_parallel.json`): the
  identity check reported MISMATCH and the run continued. Its numbers are not
  trustworthy as a pure measure of the memory branch. Re-run before quoting.
- **Any `hybrid_decay.py` / `hybrid_finetune.py` number produced before the
  cast.** These compared arms that each installed a float32 module, so arms were
  compared against *each other* under the same promotion — the internal
  comparisons are more robust than the teacher-relative ones, but a
  teacher-relative claim ("still worse than the teacher by X ppl") inherited the
  promotion on one side only and is overstated by up to ~0.23 ppl.
- **`docs/HYBRID_DECAY.md`'s recovery percentages** shift slightly for the same
  reason. The *direction* of every comparison is unchanged — the arms were all
  float32 — but the exact margins need re-measuring with the fix in place.

## What it does not invalidate

The layer-8 decay sweep's central finding survives independently of the dtype
issue: at a fixed decay, transfer's curve is **non-monotone with an interior
optimum** while the random control's is not, and the committed constant 0.99
sits near transfer's worst point. Those are comparisons between arms under the
same promotion.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
HF_HOME=~/zbrain/hf INJ_ARMS=transfer,random,summary INJ_STEPS=300 INJ_LR=3e-3 \
  ~/zbrain/venv/bin/python experiments/hybrid_inject.py
```

The run raises immediately if any arm fails to reproduce the teacher exactly at
initialisation, so a recurrence of this bug fails loudly instead of producing
plausible numbers.
