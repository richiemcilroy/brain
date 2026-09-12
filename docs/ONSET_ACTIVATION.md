# The substrate's band-pass nonlinearity does not transfer to language modelling

**Verdict: NEGATIVE, as pre-registered.** Replacing the MLP's GELU with the
substrate's measured onset-detector transfer function
(`phi(z) = z*exp(1-z)`, band-pass, peak at `z=1`) makes language modelling
**worse**, not better: **-0.1012 bpc** for a single bank and **-0.3018 bpc** for
a four-bank multiscale version, at exactly matched parameter count and 3 seeds.
The eight-bank version is numerically unstable and produced `nan` in 2 of 3
seeds.

This was predicted before the run, and the prediction is what happened. It is
reported because it closes a direction that looked promising on biological
grounds.

## Why this was worth testing

`docs/NEURON_OPERATING_POINT.md` established, with real spikes, that this
substrate's neuron is a **transient onset detector**: its transfer function peaks
at a preferred drive (`z = 1`) and decays on *both* sides, firing once at ms
12-14 of a 15 ms window and then ignoring a persistent drive.

That is a genuinely distinctive property. **Every activation in mainstream
sequence models is monotonic** — ReLU, GELU, SiLU, and the SiLU-based gates in
Mamba, H3 and RWKV all increase with input magnitude forever. A monotonic
activation can express "this input is big enough" but cannot express "this input
is the right *size*". A band-pass unit is suppressed by both too-weak and
too-strong input, which is a different function class.

So the question was whether the substrate's most distinctive nonlinearity,
transplanted into a working sequence model, buys anything.

## Method

Char-level TinyShakespeare, `ctx=512`, `d=128`, 2 gated-memory layers, 1500
steps, batch 16, AdamW `lr=1e-3` with 100-step warmup and grad clip 1.0, 3 seeds,
deterministic validation over all 217 non-overlapping 512-token windows.
Unigram floor 4.8292 bpc, **bigram floor 3.5806 bpc** — all arms clear it.

**Every arm has exactly 445,440 parameters.** The banked arms split the same
hidden width into `B` narrower banks, so capacity is held constant and only the
nonlinearity changes.

| arm | activation | val bpc | vs gelu | params |
|---|---|---|---|---|
| `gelu` | standard monotonic | **2.2533** | — | 445,440 |
| `onset` | `phi(z)`, one scale | 2.3545 | **-0.1012** | 445,440 |
| `onset_bank` | 4 scales, 4x narrower each | 2.5552 | **-0.3018** | 445,440 |
| `onset_bank8` | 8 scales | `nan` (2/3 seeds) | — | 445,440 |

The band across the four bank scales was 0.35x to 2.83x (preferred input
magnitudes 2.83 down to 0.35), a factor of 8 in tuning.

## Interpretation

**The monotonicity is worth more than the selectivity.** A plausible reading:
GELU's monotonicity gives the network a scale-free "how much" signal that is
easy to optimise, and the network can already implement any needed
magnitude-preference implicitly by learning the *weights* into a monotonic unit
— `W1` can scale its own input. A hard-coded band-pass `phi` therefore removes
an easy function class and adds a constraint, rather than adding capability. The
banked version makes this worse, because each bank is narrower and so the same
total width has less representational freedom per scale.

**The failure mode is informative about the substrate too.** The
`onset_bank8` `nan` is a numerical instability of the same transfer function
that the spiking substrate uses: with a very small scale the pre-activation
`z * scale` can be large, and `z*exp(1-z)` underflows toward zero, while the
gradient through `exp` at large negative argument vanishes. The brain does not
have this problem because its "activation" is a *threshold on an integral*, not
a static function evaluated at unbounded pre-activations. **Porting the tuning
curve as a pointwise nonlinearity loses the mechanism that makes it stable.**

## What this does and does not show

- It does **not** show the onset detector is useless in its native context. The
  substrate uses it as a threshold on a charging dendrite with bounded dynamics,
  and `docs/NEURON_OPERATING_POINT.md` shows it produces real, reproducible
  spikes and a measurable working memory there. The negative result is about
  transplanting the *curve* into a different computational setting.
- It **does** show that "biologically distinctive" is not evidence of
  "architecturally better", which is the same lesson this project has now
  learned from the dendrite confound, the multiscale confound, and the affinity
  confound.
- One task, one corpus, one context length. The conclusion is about this setting.

## Reproduce

```
python3 experiments/onset_activation.py
```

Artifact: `/Volumes/T9/human-brain/scratch/onset_activation.json`.
