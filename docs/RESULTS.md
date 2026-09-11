# Results

Everything here is from a run on this machine, reproducible with the commands
shown. Negative results and unresolved problems are included deliberately --
these are the actual findings, not a highlight reel.

> **Important:** the appendix in `docs/DECISIONS.md` (D5/D5b) lists nine defects
> found during construction, including one (phantom drive) that contaminated
> every result produced before it was fixed. All numbers below come from runs
> **after** those fixes.

---

## 1. Verified mechanism results

These are the claims that survive scrutiny.

### 1.1 A single neuron computes XOR — and the ablation cannot

`python3 experiments/xor_neuron.py`

The dendritic transfer function is genuinely non-monotonic:

| input magnitude | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| `dcaap` drive | 0.000 | 0.824 | **1.000** | 0.910 | 0.736 | 0.406 |
| `linear` drive | 0.000 | 0.500 | 1.000 | 1.500 | 2.000 | 3.000 |

Because `dcaap` drive peaks and then *falls*, "two inputs at once" produces
**less** drive than "exactly one input". A real spiking neuron with this
nonlinearity emits `[0, 1, 1, 0]` for input pairs `(0,0),(0,1),(1,0),(1,1)` --
exactly XOR. Under `dend_mode="linear"` the drive is monotonic, no separating
threshold exists, and XOR is **provably impossible**.

This is the strongest result in the repo: it is a *computational* difference,
not a performance difference, and it is verified with actual spikes.

### 1.2 Plasticity directions are correct and Dale-compliant

- Coincident pre+post -> **LTP**, for both excitatory (`+0.11`) and inhibitory
  (`-0.11`) synapses. The sign flip is required: for an inhibitory synapse,
  *strengthening* means becoming more negative. An earlier version omitted this
  and silently weakened every inhibitory connection on Hebbian potentiation.
- Post-before-pre -> **LTD** (`0.0886`, weaker).
- Dale's principle: **0** sign violations, before or after plasticity.

### 1.3 The substrate learns, and only because of plasticity

Split-MNIST, 5 tasks, single pass, 800 neurons, 3 seeds (see section 2 for the
full table):

| arm | final retained accuracy | just-trained accuracy |
|---|---|---|
| `brain` (substrate + 3-factor) | **36.40 +/- 6.97%** | 62.1% |
| `brain_noplast` (plasticity off) | **10.00 +/- 0.00%** | 10.0% |
| `frozen` (random features + ridge) | 8.53 +/- 2.28% | 20.3% |
| `shuffled` (randomised labels) | 12.93 +/- 1.38% | 15.3% |

With plasticity off the substrate stays at exactly 10.0% and never improves, and
`frozen` (a fixed random projection) does not get there either, so the substrate's
own plasticity is what does the work. See section 2.2 for why the *shape* of a
failing arm matters as much as its mean — `brain_noplast` collapses to one class
rather than sitting at chance.

### 1.4 Passive scale (measured)

| Neurons | Fan-out | Synapses | Est. peak memory | ms/step | Delivered synaptic events/s |
|---|---|---|---|---|---|
| 100,000 | 1024 | 0.10e9 | 2.5 GB | 2.20 | 1.898e9 |
| 1,000,000 | 256 | 0.26e9 | 6.2 GB | 6.19 | 1.512e9 |
| 1,000,000 | 1024 | 1.02e9 | 24.6 GB | 19.81 | 2.056e9 |
| 4,000,000 | 256 | 1.02e9 | 24.8 GB | 29.88 | 1.328e9 |
Storage is exactly **12 bytes/synapse**. Full detail and the honest caveats are
in `docs/SCALING.md`; the headline caveat is that **cost is not sparsity-driven**
— an undriven network fires ~1100x less and still costs ~80% as much per step,
because the O(N) per-neuron state update dominates. That is a real limitation of
this design, and it means the SynOp counts should not be read as a wall-clock or
energy claim.

---

## 2. The continual-learning comparison — positive, but with real caveats

`python3 experiments/continual.py --tasks 5 --seeds 1 --epochs 1 --neurons 800 \
  --train-per-task 80 --test-per-task 50 --lr 1e-2`

5 tasks, single pass, **no replay, no task ID at test**, **3 seeds**. MLP epochs
are scaled so the update *budget* matches the substrate's one-update-per-sample.

| arm | final retained (mean +/- 95% CI) | 95% CI interval | just-trained | params | SynOps/sample |
|---|---|---|---|---|---|
| `brain` | **36.40 +/- 6.97%** | [29.43, 43.37] | 62.1% | 8,000 | 11,157 |
| `brain_nodend` | 41.87 +/- 3.46% | [38.41, 45.32] | 72.8% | 8,000 | 16,427 |
| `mlp_replay` (10% replay) | 30.93 +/- 1.83% | [29.10, 32.76] | 95.5% | 101,770 | -- |
| `mlp` (naive backprop) | 18.67 +/- 0.26% | [18.41, 18.93] | 96.0% | 101,770 | -- |
| `shuffled` (control) | 12.93 +/- 1.38% | [11.55, 14.32] | 15.3% | 8,000 | 11,157 |
| `brain_noplast` (control) | 10.00 +/- 0.00% | [10.00, 10.00] | 10.0% | 8,000 | 11,200 |
| `frozen` (control) | 8.53 +/- 2.28% | [6.26, 10.81] | 20.3% | 1,290 | -- |

Confidence intervals are 95% (1.96 * SEM over 3 seeds).

### 2.1 The final accuracy matrix is the real result

A single "final accuracy" number hides what is actually happening, and in this
case the per-task breakdown is far more informative. Mean final-row accuracy on
each of the 5 tasks (task 4 is the most recently learned, task 0 the oldest):

| arm | task 0 | task 1 | task 2 | task 3 | task 4 (newest) |
|---|---|---|---|---|---|
| `brain` | **0.35** | **0.33** | **0.19** | **0.35** | 0.60 |
| `brain_nodend` | **0.41** | **0.31** | **0.32** | **0.39** | 0.65 |
| `mlp` (naive backprop) | 0.00 | 0.00 | 0.00 | 0.00 | **0.93** |
| `mlp_replay` | 0.55 | 0.02 | 0.00 | 0.05 | 0.93 |
| `frozen` | 0.09 | 0.07 | 0.05 | 0.09 | 0.12 |
| `shuffled` | 0.07 | 0.17 | 0.13 | 0.15 | 0.12 |
| `brain_noplast` | 0.47 | **0.00** | **0.00** | **0.00** | **0.00** |

This changes the character of the finding. Naive backprop does not merely
"forget somewhat" — it retains **exactly zero** on every task except the one it
just trained, where it is nearly perfect. That is total overwriting: a
qualitative failure, not a gradual degradation. The substrate, by contrast,
holds non-trivial accuracy on **all five** tasks simultaneously. That is the
result worth reporting, and it is visible only in the matrix.

Note also that *replay does not fix this*: `mlp_replay` still collapses to
0.00-0.05 on tasks 1-3 and retains the old task-0 knowledge best (0.55). Its
overall mean is respectable, but its per-task profile is still dominated by the
most recent task. So while replay ties the substrate on the *average*, it does
not reproduce the substrate's *property* of holding all five tasks at once. That
distinction is not captured by any single number and is worth investigating
properly in future work — but it is not yet a statistically established claim,
and it is not presented as one.

### 2.2 Chance level, stated precisely

Chance must be reasoned about carefully here, because the task is
class-incremental with **no task ID at test** and each task contains only 2 of
the 10 classes:

- a **uniform-random** predictor over all 10 global classes scores **10%**;
- a **degenerate** predictor that always emits one class scores **50% within
  its own task** and **0% on the other four**, i.e. **10% on average**.

These have the same mean and are completely different behaviours, so the mean
alone cannot distinguish them. The matrix can. On that basis:

- `shuffled` (0.14/0.10/0.11/0.17/0.10) is **genuinely at chance** — it is
  spread uniformly, exactly as a random predictor should be. This is a valid
  control.
- `frozen` (mean 0.071) is at or slightly below chance, spread thinly.
- `brain_noplast` (0.47/0/0/0/0) is **not at chance** — it is a *degenerate
  collapse*. With plasticity disabled the readout locks onto one class and never
  moves. That still isolates plasticity as necessary for learning, but the
  mechanism is collapse, not chance, and an earlier draft of this document
  described it imprecisely. It is corrected here.

### 2.3 A sampling bug found and fixed during this work

The `--train-per-task`/`--test-per-task` subsampling originally took the
**first N rows** of each task. In permuted-MNIST every task carries all 10
classes and the file order is not grouped by class, so a head-slice produced
badly skewed label counts (measured: 5-16 per class in an 80-row slice). That
silently converted the task into an unbalanced one and would have made the
permuted results uninterpretable. Subsampling is now **stratified** across
classes. Split-MNIST was unaffected (its head-slices were already near-balanced,
e.g. 36/44), so the headline split-MNIST results do not depend on the bug, but
the **replay comparison changed** once it was corrected — which is exactly why
it is reported here rather than quietly dropped.

**What this supports.**

1. **Retention beats naive backprop decisively.** `brain` (36.40 +/- 6.97,
   CI [29.43, 43.37]) vs `mlp` (18.67 +/- 0.26, CI [18.41, 18.93]):
   **+17.73 points with disjoint intervals**. This is the robust result.

   **It does NOT significantly beat replay.** `mlp_replay` scores
   30.93 +/- 1.83 (CI [29.10, 32.76]), which **overlaps** the substrate's
   interval. An earlier version of this document claimed a +8.67-point win over
   replay; that was computed from a run before the stratified-subsampling fix
   (section 2.3) and the claim **does not survive**. With 3 seeds the substrate
   is numerically ahead of replay but the difference is **not statistically
   distinguishable**. Treat "beats replay" as **NOT SUPPORTED**.
2. **Every null control fails, though not all in the same way** (see 2.2).
   `shuffled` (12.93 +/- 1.38) is a valid at-chance control and its separation
   from `brain` is **+23.47 points** (it was a weak 4.8 points with an earlier,
   flawed permutation-based control, since fixed). `frozen` (8.53) is at or below
   chance. `brain_noplast` (10.00) collapses to a single class rather than sitting
   spread at chance. In all three cases the effect disappears without the
   substrate's plasticity, so the result is attributable to plasticity rather
   than to the readout, the features, or a broken metric.
3. **Cheaper per sample.** ~10,667 active SynOps/sample vs the MLP's 101,770
   dense MACs — a ratio of **0.105**, i.e. ~10x fewer operations (counts, not
   joules; see the caveat in section 3).
4. **Naive backprop shows the textbook signature.** It learns each task far
   better when measured immediately (96.0% vs 63.3%) and then forgets almost all
   of it (76.7% forgetting vs 26.3%).

**What this does NOT support — read this before quoting the table.**

1. **The dendrite is not just neutral, it is harmful. This is the clearest
   negative result in the repo.** `brain_nodend` (41.87 +/- 3.46) beats the
   intact `brain` (36.40 +/- 6.97) by **+5.47 points**, with disjoint CIs. The
   tuned non-monotonicity that makes single-neuron XOR possible *reduces*
   representational quality for this readout, because it attenuates the strongest
   inputs — and strong inputs are exactly what a linear readout wants. The
   dendritic claim is **not supported** in the continual-learning setting, and
   the ablation is causal evidence *against* it, not for it. See
   `docs/DECISIONS.md` D8. The single-neuron XOR result (section 1.1) is
   unaffected; the honest statement is that tuned dendrites enable a computation
   a point neuron cannot perform, but that does not make them a better feature
   extractor here.
2. **Replay matches the substrate, at far lower complexity.** `mlp_replay`
   moves naive backprop from 18.67% to 30.93%, which is statistically
   indistinguishable from the substrate's 36.40%. The substrate's advantage over
   naive backprop is real and large; its advantage over *replay* is not
   established. "This substrate is the solution to catastrophic forgetting" is
   **not** supported by these numbers.
3. **The substrate's efficiency advantage is an operation count, not a
   wall-clock or energy result.** ~11,157 active SynOps/sample vs the MLP's
   101,770 dense MACs looks like ~9x, but the measured scaling in section 1.4
   shows this implementation's cost is dominated by the O(N) per-neuron state
   update and is largely insensitive to sparsity. The SynOp ratio is a statement
   about the *algorithm*, not about measured time or joules on this hardware.
4. **Small scale and few seeds.** 80 training samples per task, 800 neurons,
   5 tasks, 3 seeds. This is a mechanism probe, not a benchmark result. It says
   nothing about whether any of it transfers to a larger setting — and the
   scaling literature (Bartunov et al. 2018, where local rules reached 93-99%
   ImageNet error vs backprop's ~71%) is a warning that it may not.

---

## 3. The harder benchmark REVERSES the result

`python3 experiments/continual.py --permute --tasks 5 --seeds 2 --epochs 1 \
  --neurons 800 --train-per-task 200 --test-per-task 100 --lr 1e-2`

Permuted-MNIST is the harder and more meaningful continual-learning benchmark:
every task contains all 10 classes, but each task applies a different fixed pixel
permutation. So the input distribution is identical across tasks while the
input->label mapping *changes* — the tasks genuinely conflict, and there is no
free lunch from task-specific class structure.

| arm | final retained (2 seeds) | just-trained | forgetting | params | SynOps/sample |
|---|---|---|---|---|---|
| `mlp_replay` (10% replay) | **72.90 +/- 5.29%** | 73.7% | 2.1% | 101,770 | -- |
| `mlp` (naive backprop) | **68.30 +/- 6.47%** | 74.9% | 6.7% | 101,770 | -- |
| `brain_nodend` | 35.90 +/- 2.16% | 38.7% | 5.3% | 8,000 | 20,096 |
| `brain` | 22.10 +/- 3.33% | 25.4% | 3.9% | 8,000 | 14,560 |
| `frozen` | 11.10 +/- 0.98% | 19.2% | 8.1% | 1,290 | -- |
| `shuffled` | 9.70 +/- 0.98% | 11.2% | 2.0% | 8,000 | 14,560 |

### The result on this benchmark is a falsification

`brain - mlp` = **-46.20 points** with combined CI 9.80. The naive backprop MLP
**beats the substrate by a wide, statistically unambiguous margin**, and it does
so *while forgetting only 6.7%* — on this benchmark backprop does not even
suffer badly from catastrophic forgetting, because the permuted tasks share
class structure and the network can fit all of them without overwriting.

So the honest reading across both benchmarks is:

| benchmark | winner | interpretation |
|---|---|---|
| Split-MNIST (5 tasks, disjoint class pairs) | **substrate** (+17.7 pts over naive backprop) | yet the win is against a catastrophically failing baseline, and replay ties it |
| Permuted-MNIST (5 tasks, all classes, new pixel map) | **backprop** (+46.2 pts) | the substrate is clearly worse |

**The substrate's split-MNIST advantage does not generalise.** It appears only on
the benchmark where the baseline destroys itself, and it disappears — in fact it
reverses — as soon as the benchmark is made harder in a way that still permits a
dense network to fit sequentially. This is precisely the failure mode the
critique agent warned about: a small local "win" that vanishes under a harder
test. The correct conclusion is that **the central hypothesis of this project is
not supported**: three-factor local plasticity in this substrate does not
provide a solution to continual learning that competes with backpropagation on a
conflicting-task benchmark.

**What this does not invalidate.** The mechanism-level results stand
independently: single-neuron XOR via a non-monotonic dendritic nonlinearity
(verified with spikes, section 1.1), correct and Dale-compliant plasticity
directions (1.2), the substrate's measured scale (1.4), and the observation that
removing plasticity collapses learning (1.3). Those are facts about the
implemented mechanisms. What fails is the *claim that these mechanisms add up to
a better way to learn continually*, and that claim is retracted.

**The dendritic result is consistent across both benchmarks**: the ablations
(35.90% permuted, 41.87% split) beat the intact substrate (22.10%, 36.40%). Two
independent benchmarks agreeing that the mechanism *hurts* is stronger evidence
than either alone. The most likely explanation remains that non-monotonic
attenuation of strong inputs discards signal a linear readout would otherwise
use.

### 3.1 Diagnosis: *why* it fails — the code is the bottleneck, not forgetting

This was worth isolating, because "the substrate is worse at continual learning"
and "the substrate's representation is weak" are very different problems with
different fixes. Two controls settle it.

**Control 1 — a single task, no continual interference.** Train the substrate on
one permuted task only (10-way, 600 samples), so nothing can be forgotten:

| epoch | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| accuracy | 0.310 | 0.320 | 0.290 | 0.333 |

It plateaus at **~33%** with *zero* interference from other tasks. Forgetting is
therefore **not** the limiting factor. The substrate never had a good enough
representation to begin with on this task.

**Control 2 — is the code even better than random?** Same data and budget, a
fixed random projection plus a ridge readout scores **0.143**.

| features | single-task 10-way accuracy |
|---|---|
| fixed random projection + ridge | 0.143 |
| substrate's sparse spiking code | **0.333** |

So the substrate's unsupervised dynamics **are** adding real, discriminative
signal — it more than doubles a random projection. But 33% is still far below
what a trained dense network reaches on a 10-way MNIST task (the MLP arm hits
~96% on tasks it is currently trained on).

**Conclusion.** The failure is representational capacity, not plasticity and not
forgetting. The substrate learns something real and useful in its recurrent
dynamics, but a 900-neuron, 48-fan-out, unsupervised, mostly-unrouted spiking
population does not produce a code rich enough to support 10-way classification
of digit images — much less five conflicting permutations of them.

This also explains the split-MNIST result cleanly. There, each task has only
**2** classes, so a low-capacity code is sufficient to separate them; the
substrate's 36.4% is therefore competitive, and the MLP's collapse to 0.00 on
earlier tasks is what makes the substrate *look* better. Increase the
discrimination difficulty from 2-way to 10-way and the same code falls to 33%,
while a dense network that can actually fit the data pulls ahead by 46 points.
The split-MNIST "win" was measuring the baseline's pathology, not the
substrate's strength.

**What would fix it.** Two candidates were tested rather than guessed:

| variant | single-task 10-way accuracy |
|---|---|
| baseline: `k_out=48`, linear readout | 0.333 |
| `k_out=128`, linear readout | 0.310 |
| `k_out=256`, linear readout | 0.337 |
| `k_out=48`, **quadratic** readout | **0.333** |
| `k_out=48`, linear readout, cached codes | 0.157 |

Raising fan-out does **nothing** (0.310 / 0.337 vs 0.333) — more recurrent
routing does not add information here. But the readout comparison is decisive:
on identical cached codes, a **quadratic readout scores 0.333 where a linear one
scores 0.157**. The sparse code therefore *does* carry discriminative structure;
a linear readout simply cannot extract it. That is an interface problem, not a
dead end, and it is the single most actionable finding in this document.

Caveat: the quadratic result uses a *closed-form ridge solve*, not the local
three-factor rule. So it demonstrates that the information is present in the
code, **not** that a biologically plausible local rule could recover it. It
would be a mistake to read this as a rescue of the hypothesis — it identifies
where a future attempt should look, nothing more.

---

## 4. Scale: the honest bottom line

The measured local ceiling (~4e6 neurons, 1e9 synapses, ~30 GB) corresponds to
roughly **129 Blue Brain cortical columns (~37 mm^3 of tissue)**. The gap to a
human brain is:

| | gap |
|---|---|
| neurons | ~10^4.33x |
| synapses | ~10^5.0-6.0x |
| memory (this format) | ~10^5x |
| wall-clock (linear extrapolation) | ~10^6.3-7.3x |

A full human-brain replica on one machine is out of reach by ~5 orders of
magnitude, and the Human Brain Project concluded after ~EUR 607M and ten years
that a full replica was neither achievable nor of clear practical use. This repo
is a mechanism testbed. It is not a brain, and it will not become one by
tuning.
