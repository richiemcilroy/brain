# Does error-modulated three-factor plasticity shape a hidden layer?

**Verdict: NO, on this substrate, and the missing mechanism is identified.**
Making the afferent projection plastic under an error-derived third factor made
performance **significantly worse** than the identical model with its hidden
learning rate set to zero. The pre-registered criterion was that a plastic arm
must beat its own frozen ablation by more than twice the seed standard
deviation; neither arm came close. This is a **successful falsification of the
central claim in `docs/PARADIGM.md` section 3.3** — credit assignment through a
broadcast third factor, applied to a hidden layer of this substrate, does not
shape that layer's representation.

| arm | what it is | accuracy (mean ± sd, 3 seeds) | per-seed | vs frozen | verdict |
|---|---|---|---|---|---|
| 1 `readout_only` | current model: frozen input, readout learns | **79.00% ± 2.08** | 81.3 / 77.3 / 78.3 | — | current baseline |
| 2 `frozen_hidden` | **control**: identical plasticity code path, `hidden_lr = 0` | **74.89% ± 2.14** | 77.3 / 74.0 / 73.3 | — | **the only valid test** |
| 3 `scalar` | hidden plasticity, broadcast scalar third factor | **68.11% ± 1.02** | 67.0 / 69.0 / 68.3 | **−6.78 pts** | **FAIL** (threshold 3.36) |
| 4 `dfa` | hidden plasticity, direct feedback alignment | **64.44% ± 2.34** | 64.7 / 62.0 / 66.7 | **−10.44 pts** | **FAIL** (threshold 4.49) |
| 5 `static_relu` | fixed random ReLU + k-WTA at matched sparsity | **80.22% ± 3.36** | 83.3 / 76.7 / 80.7 | — | absolute reference |
| 6 `feedback_true_weights` | diagnostic: DFA with the readout's own weights | **60.00% ± 7.22** | 64.3 / 64.0 / 51.7 | −14.89 pts | diagnostic only |

Single-task 10-way MNIST, 600 train / 300 held-out, 900 neurons, 3 seeds,
`hidden_lr = 0.02`. Raw data: `experiments/results/threefactor_hidden.json`.
Run: `python3 experiments/threefactor_hidden.py`.

Arm 5 is matched to the **substrate's own measured active fraction** on the same
seed (14.0%, 14.2%, 14.4% → `k_wta` 126, 128, 130). That is a stronger control
than the 29% figure quoted in `docs/SUBSTRATE_CONTRIBUTION.md` (0.765): at
*matching* sparsity a non-spiking random ReLU projection reaches **80.22%**
against the substrate's 74.89% frozen and 68.11% plastic.

## What was tested, and why it is the test that mattered

`docs/PARADIGM.md` §3.3 claims that eligibility × broadcast neuromodulator can
shape a hidden layer with no backward pass and no weight transport. Reading the
code before this experiment showed the claim had never been exercised:

* `brain/cortex.py` ran three-factor plasticity at a **constant `M = 1`** — pure
  unsupervised Hebbian learning. No error signal reached the afferent projection.
* The only error-driven learning was the readout, a single linear layer trained
  by the delta rule. **A single linear layer trained by the delta rule is exact
  gradient descent on that layer.**
* The published permuted-MNIST comparison (22.10% vs 68.30%, two-layer MLP) was
  therefore not a test of the paradigm. It compared a one-linear-layer model
  against a two-layer network.

So this experiment switched the mechanism on and measured it. Two third factors
were tested, both derived from the error:

1. **Scalar** — `+1` if the predicted class is correct, `−1` otherwise,
   broadcast identically to every hidden synapse.
2. **DFA (direct feedback alignment)** — the per-class readout error passed
   through a **fixed random matrix**. This is a **known method** (Lillicrap et
   al. 2016, *Nat Commun* 7:13276), deliberately included because it uses no
   weight transport and no backward pass through the network's own weights.
   **It is not novel and is not claimed as novel.**

Arm 6 substitutes the readout's own weights for the random matrix. It is **not**
presented as a paradigm candidate: it uses weight transport, so it is not local,
and it is **not the true gradient** — the gradient w.r.t. the projection also
contains the derivative of the code through a spiking non-monotonic
nonlinearity, which is zero almost everywhere. It is the no-randomness ablation
of DFA, nothing more.

The pre-registered criterion (fixed before running, see the module docstring):
`mean(plastic) − mean(frozen_hidden) > 2·sd`, `sd = sqrt((var_p + var_f)/2)`,
`ddof = 1`. Honest prediction recorded before the run: scalar probably fails,
DFA probably works on one task and then forgets. The measurement disagreed with
the second half of that prediction — DFA does not work even on one task.

**Because no plastic arm passed, the permuted-MNIST continual-learning stage was
deliberately not run.** A model that cannot beat its own frozen ablation on a
single task has no continual-learning claim to test, and running it would have
reported forgetting in a model that had nothing to forget. The JSON records
`continual.ran = false` with the reason and the measured deltas.

## What the mechanism actually is (measured, not asserted)

### 1. Every plastic arm moved the representation *downhill*

The `frozen_hidden` arm is the identical model, the identical code path, the
identical eligibility accumulation — only the weight write is skipped
(`hidden_lr = 0`). It reached 74.89%. Turning the write on dropped the same
model to 68.11% (scalar) and 64.44% (DFA). **The learning rule was not weak; it
was harmful.** The one thing that would have confirmed the paradigm — plastic
beating frozen — is exactly what did not happen.

### 2. The eligibility matrix is rank 1, so a third factor has almost nothing to select

`dw_ij = lr · M_j · e_ij` can only do something useful if `e` contains
per-neuron input structure that `M_j` can select. It does not. For a static
input held across the window, the eligibility from
`brain/plasticity.py`'s rule reduces to

```
e_ij = a_plus · x_i · c_j  −  a_minus · k_j
```

whose columns are all one fixed input vector `x` times a per-column scalar, minus
a per-column constant. That is rank 1. Measured on the real eligibility matrix
(3 samples): top singular value holds **99.3%, 99.3%, 99.3%** of the energy;
`rank_at_99pct_energy = 1`. The consequence: scaling by any `M_j` can only make
one neuron's copy of the *same* input direction stronger or weaker. It cannot
select a *different* input pattern for a different neuron, because the matrix
does not contain different patterns to select. **This is the proximate cause of
the negative result, and it grows no better with more neurons or a bigger
hidden layer — it is a property of driving a static input through a fixed
projection, not a small-network artefact.**

Two sub-findings that a reader should not have to rediscover:

* **The LTD term is load-bearing.** The first draft of this experiment used only
  `e = Σ_t x_i · s_j` — a purely correlational eligibility. Because both factors
  are non-negative, so is `e`, and since `dw` is then `M_j` times a
  non-negative number it has **one sign for a whole neuron's weight row**. That
  is not a receptive field; it is a gain knob. Measured: `frac(e < 0) = 0.000`,
  and 96.7% of entries exactly zero. The library's `− a_minus · y_post` term is
  what makes the eligibility signed and contrastive, and it was restored before
  any result was recorded. The sign-locked version measured *below* its frozen
  ablation too. Both versions are reported here because the diagnosis depends on
  the difference.
* **The substrate is an onset detector, and it goes silent under weight drift.**
  `docs/NEURON_OPERATING_POINT.md` documents that these units fire while the
  dendritic potential ramps *through* the peak of a non-monotonic tuning curve.
  Any change to a neuron's input scale moves it off that peak. Measured at
  `hidden_lr = 1.0`: activity fell from 129 to 8.6 spikes/sample and accuracy
  from 68% to 12% — silence, not learning. The homeostatic per-neuron
  rescaling in `HiddenThreeFactor.apply` (column L2 norm preserved, applied
  identically to every arm) bounds the *drift* — with it, `mean|W_aff|/σ₀`
  stays ≤ 0.87 across every arm at every rate tested, against 1.04–1.43 at
  `hidden_lr = 1.0` without it (both still inside the hard 4σ clip, which is
  what stops either variant diverging outright) — but it does **not** by itself
  restore activity at a large rate: measured at `hidden_lr = 1.0` *with* the
  rescaling, activity was still 12–23 spikes/sample against a frozen 129 and
  accuracy 13.5–19.5% against a frozen 68.0%. It is the activity-based rate
  selection below, not the rescaling, that keeps the population inside its
  operating range.

### 3. Learning rate does not rescue it — the whole sweep fails

`hidden_lr` was chosen by an **activity-only rule fixed before any accuracy was
read**: "the largest rate whose post-training spike rate stays within 20% of the
frozen baseline in every plastic arm". Spikes/sample relative to frozen:

| `hidden_lr` | `scalar` | `dfa` | `feedback_true_weights` |
|---|---|---|---|
| 0.005 | 1.03 | 1.05 | 0.91 |
| 0.01 | 1.04 | 1.08 | 0.83 |
| **0.02 (used)** | **1.06** | **1.07** | 0.71 |
| 0.05 | 1.14 | 0.84 | 0.48 |

Accuracy at the same rates (1 seed, same budget; frozen = 77.3%):

| `hidden_lr` | `scalar` | `dfa` |
|---|---|---|
| 0.005 | 72.7% | 76.3% |
| **0.02** | **68.1%** | **64.4%** |
| 0.05 | 24.0% | 45.0% |
| 0.2 | 15.7% | 16.7% |

**No value in the sweep passes the criterion.** The best single cell anywhere in
the sweep — DFA at `0.005`, 76.3% — is still 1.0 point *below* the frozen
ablation at the same budget. The verdict does not rest on the choice of
`hidden_lr`.

### 4. A perfect feedback direction still fails

Arm 6's third factor *is* the readout direction, so its alignment with the
  reference direction is 1.000 by construction (verified in the JSON, as a check
  on the diagnostic). It scored **60.00%** — the worst of the plastic arms, below
  both the frozen ablation and the plain `scalar` rule. This is worth
  stating plainly: **alignment of the third factor with the readout's error
direction is not sufficient for learning here.** It is also a warning about the
DFA literature's alignment metric, which in this substrate is close to
uninformative because the bottleneck is the rank of `e`, not the quality of `M`.

## What this does and does not establish

**Establishes:**

* Error-modulated three-factor plasticity on this substrate's afferent
  projection **does not** improve a single-task 10-way MNIST classification, and
  significantly *degrades* it relative to its own frozen ablation. The claim in
  `docs/PARADIGM.md` §3.3 is **falsified as stated**, with a pre-registered
  criterion and a control that isolates the learning rule as the only variable.
* The proximate mechanism is measurable: the eligibility matrix is **rank 1**, so
  no third factor — including a perfectly aligned one — has per-neuron input
  structure to select.
* At matched sparsity (14%), the substrate is beaten by a **non-spiking** random
  ReLU projection of the same width and the same readout (80.22% vs 74.89%
  frozen, 68.11% plastic). Whatever accuracy the model has comes from the random
  projection and the linear readout, not from plasticity or spiking dynamics.

**Does not establish:**

* That three-factor plasticity is useless in general, or that DFA cannot work.
  Both are established methods and both are reported to work in other settings.
  This is a negative result about **this substrate, this interface, and this
  eligibility construction**.
* That continual learning fails. It was not run, on purpose. The finding is one
  step earlier in the chain: the hidden layer never learned a single task well
  enough for its retention to be worth measuring.
* That the spiking substrate cannot beat a random projection on *any* task.
  `docs/NEURON_OPERATING_POINT.md` argues its natural regime is transient input
  and a latency code; every measurement in this repo, including this one, feeds
  it static images and reads spike counts. **The substrate has never been tested
  in the regime its own mechanism is built for.**

## Limits

* **One task family, one dataset.** Single-task 10-way MNIST. No other dataset,
  no regression, no sequence task.
* **Small scale.** 900 neurons, 784→900 projection, 600 training samples. The
  rank-1 argument is scale-free (it is about the shape of the eligibility), but
  the *accuracy* numbers are not evidence about larger networks.
* **Three seeds.** Enough to resolve a 6.8-point effect against a 2.1-point sd,
  not enough to characterise a distribution. Per-seed values are reported so the
  spread is visible rather than summarised away.
* **`hidden_lr` is a new hyperparameter with no precedent in the repo.** It was
  set by a pre-declared activity rule and the full sweep is reported, but a
  reader who insists on a different value should read the sweep table, where
  every value fails.
* **The readout was not retuned.** `readout_lr = 0.05` with NLMS, the
  `CortexConfig` default, identical in every arm. That was a deliberate
  constraint from the brief: a comparison in which the readout's rate is chosen
  per arm is not a comparison of the hidden rule.
* **Arm 6 is not a gradient.** It is named `feedback_true_weights` rather than
  "oracle" precisely so it is not read as an upper bound on what is learnable.
* **The DFA feedback matrix is a fixed ±1 sign matrix**, not a Gaussian draw. A
  Gaussian `B` was not tested; given the rank-1 finding, no choice of `B` can
  change the conclusion, but the choice should be stated.
* **Ridge, and the 0.765/0.702/0.803 numbers in `docs/SUBSTRATE_CONTRIBUTION.md`,
  use a different readout and split** (900 neurons, 800/400, ridge probe). They
  are quoted as absolute reference points, not as arms of this experiment. The
  arm-5 control is the one measured here under this readout.

## Why the negative result is the useful outcome

A positive result here would have been easy to fake and hard to believe: the
readout alone already reaches 79%, so any hidden-layer effect would have had to
be separated from a strong gradient-trained layer, and the continual-learning
claim that motivated the whole paradigm had never been tested. What the
experiment produced instead is a **specific, mechanical, falsifiable reason** why
the mechanism cannot work as constructed:

> a broadcast third factor can only rescale a rank-1 eligibility, so it cannot
> tell two input weights of the same neuron to move in opposite directions.

That is checkable in one line (SVD of `e`) and it predicts the failure of any
variant that keeps the eligibility construction fixed — which is why arm 6, with
a perfect feedback direction, also failed. It also says what would have to
change: **the eligibility must acquire per-neuron input structure before any
third factor can shape it.** A static input through a fixed projection cannot
supply that, because all of a neuron's input weights are driven by the same
input vector at the same time.

Per the acceptance criteria, the permuted-MNIST stage is therefore reported as
not run, the falsification is stated as such, and the criterion was honoured
rather than renegotiated after seeing the numbers.
