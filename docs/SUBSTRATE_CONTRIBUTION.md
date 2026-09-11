# Does the spiking substrate contribute anything?

**Short answer: not detectably.** On this evidence the classifier's accuracy is
explained by a fixed random projection of the pixels. The recurrent spiking
dynamics and the local plasticity change the number by less than seed noise.

## The measurement

Identical cached codes, 3 seeds, 800 train / 400 held-out test, ridge probe,
MLX, 900 neurons, 15 ms. `_code` is the real `CortexClassifier` path (so the
reset and normalisation flags apply); only the input projection is swapped.

### Ablating the substrate (random dense input projection)

| arm | probe accuracy | active |
|---|---|---|
| FULL (recurrence + plasticity) | 0.702 +/- 0.023 | 28.9% |
| recurrence weight matrix zeroed | 0.715 | 29.7% |
| plasticity off | 0.715 | 29.2% |
| both recurrent and plasticity off | 0.710 | 28.9% |

Removing the recurrent connections and the plasticity **does not hurt** — the
point estimate moves slightly *up*. The substrate's own dynamics are not what
produces the classification.

### Reference arms, same dimension

| features | probe accuracy |
|---|---|
| random projection + ReLU (dense, no spiking) | 0.803 +/- 0.033 |
| random projection + ReLU + k-WTA (29% sparse) | 0.765 +/- 0.045 |
| random projection + ReLU + k-WTA (12% sparse) | 0.695 +/- 0.015 |
| **spiking substrate (29% active)** | **0.702 +/- 0.023** |

A static random ReLU projection at matched sparsity (0.765) **beats** the
spiking substrate (0.702), and a dense one beats it by 10 points. The spiking
substrate is therefore *worse than a random projection of the same dimension*,
not better.

## Correction to an earlier claim

I previously measured the identity input wiring at 0.395 and a random
projection at 0.690 and reported a **+29.5 point** effect from fixing the input
wiring. That figure does not survive proper controls:

| wiring | standardised probe | unstandardised probe |
|---|---|---|
| identity (as coded) | 0.396 | 0.466 |
| random dense projection | 0.456 | 0.570 |

The real effect is **+6 to +10 points**, not +29.5. The earlier number came
from a single seed, an inconsistent regularisation constant between the two
arms, and — most importantly — comparing a probe *with* per-feature
standardisation against one *without*. The direction is confirmed; the
magnitude was overstated by roughly 3x.

The identity wiring is still a genuine defect worth fixing — neuron *i*
receiving exactly pixel *i* with no convergence, and neurons 784–899 receiving
no input at all, is not a plausible cortical front end. But it is a moderate
handicap, not the dominant explanation of the substrate's poor performance.

## Raw pixels beat all of it

Ridge on the raw 784 pixels reaches **0.670** on this split, beating the spiking
substrate (0.702 with the random projection is comparable; 0.466 with its own
identity wiring is far below). A 900-unit random feature expansion is not
earning its parameter count here.

## What this means for the project's claims

1. The **falsification stands**, and is in fact stronger than documented:
   the substrate does not beat backprop on permuted-MNIST (22.10% vs 68.30%),
   and the mechanism it credits for its representation contributes nothing
   measurable on a single task.
2. The result is **not** an artifact of the binarised readout rule. The
   readout rule is a genuine defect — see `docs/REVIEW.md` and the fix in
   `brain/cortex.py` (`readout_rule`) — but correcting it does not rescue the
   substrate, because the underlying features do not carry the signal.
3. The earlier characterisation "representational capacity is the bottleneck"
   was directionally right but misattributed. The bottleneck is that **the
   substrate's dynamics are not computing a useful representation at all**;
   whatever accuracy exists is inherited from the random projection feeding it.

## Caveats

- Single-task 10-way MNIST only. Continual behaviour is not re-measured here.
- Ridge is a closed-form linear probe, not a biologically plausible readout.
- 3 seeds is enough to see that the ablation effect is negligible, but not to
  resolve small differences.
- The substrate was not tuned for this comparison (no sweep of gain,
  time constant, or time window beyond what the project already used), so this
  is a statement about *this configuration*, not a proof that no spiking
  configuration can work.
