# human-brain — a sparse, dendritic, locally-plastic cortical substrate

A working, locally-runnable brain-inspired simulation substrate, and a
falsifiable test of whether its mechanisms solve a problem that current AI
structurally cannot: **learning continually, in a single pass, without replay.**

> **Outcome, stated up front: the headline continual-learning result is a
> FAILURE, and the mechanism it was meant to test was never actually running.**
> The substrate beats naive backprop on Split-MNIST (+17.7 points) but **loses
> by 46 points to backprop on the harder Permuted-MNIST**, and a plain replay
> buffer matches it on the easier one.
>
> The important correction, established after that result: **no error signal
> ever reached a hidden layer.** `brain/cortex.py` drives the substrate with a
> *constant* neuromodulator (`M = 1`, i.e. unsupervised Hebbian), and the only
> error-driven layer is a single linear readout updated by the delta rule —
> which is exact gradient descent on that layer. So the model was a frozen
> projection + a fixed nonlinearity + one linear layer, and the comparison
> measured whether *that* can do 10-way MNIST. It cannot. `docs/RESULTS.md` §3.1
> says the same thing from the other side: on a single task with **zero**
> interference it plateaus at ~33%, so the limit is **representational
> capacity, not forgetting**.
>
> The correct label for the paradigm claim (credit assignment via a broadcast
> scalar, no backward pass) is therefore **UNTESTED, not falsified**. What *is*
> falsified is that this configuration's self-organising dynamics contribute
> anything measurable (`docs/SUBSTRATE_CONTRIBUTION.md`: 0.702 → 0.710, inside
> seed noise, and a static random ReLU projection scores 0.765).
>
> The mechanism-level results (dendritic XOR, plasticity correctness,
> determinism, measured scale) all hold and are verified. See
> `docs/PARADIGM.md` §3.3 for the correction and `EXIT.md` for the full ledger
> of what is established, retracted, and still untested.

Everything here runs on one Apple M4 Max with MLX/Metal. No cluster, no cloud.

---

## Read this first: what this is *not*

This is **not** a simulated human brain, and it never will be on a laptop. The
gap is not a matter of optimisation:

| | Human brain | This machine (measured) | Gap |
|---|---|---|---|
| Neurons | ~86.1e9 | ~1e6 | ~1e5x |
| Synapses | ~1e14–1e15 | ~1e9 | ~1e5–1e6x |

The Human Brain Project spent ~EUR 607M over ten years and concluded that a full
replica was *neither achievable nor of clear practical use*. Blue Brain's
**single cortical column** — 31,000 neurons, 0.29 mm³ — was already
supercomputer-scale. Repeating that race locally would be a worse version of an
experiment that has already been run.

So this repo does something narrower and more useful: it builds the brain's
*mechanisms* at the largest scale this machine supports, and puts them under
test against matched baselines. Mechanisms transfer across five orders of
magnitude. Parameter counts do not.

Full reasoning, including what is genuinely novel vs. already-known, and the
exact conditions that would falsify the claim: **[`docs/PARADIGM.md`](docs/PARADIGM.md)**.

---

## Quickstart

```bash
cd human-brain

# 0. fetch MNIST for the continual-learning experiments (~11 MB, once).
#    Step 4 needs it; steps 1-3 do not. See data/README.md for the checksum.
curl -sL -o data/mnist.npz \
  https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

# 1. prove the core biological claim: a single neuron computes XOR,
#    and the linear-dendrite ablation cannot.
python3 experiments/xor_neuron.py

# 2. run the validation suite
python3 -m pytest tests/ -q

# 3. measure the scale ceiling on this machine
python3 bench/bench_scale.py

# 4. the falsification experiment: single-pass continual learning, no replay
python3 experiments/continual.py --tasks 5 --seeds 3 --neurons 800 \
  --train-per-task 80 --test-per-task 50 --lr 1e-2

# 5. the benchmark that falsified the hypothesis (harder: new pixel map per task)
python3 experiments/continual.py --permute --tasks 5 --seeds 2 --neurons 800 \
  --train-per-task 200 --test-per-task 100 --lr 1e-2
```

Requires `numpy` and `mlx`. If MLX is unavailable the whole stack falls back to
NumPy automatically (`BRAIN_BACKEND=numpy`) — slower, same results.

---

## Architecture

```
brain/
  backend.py       MLX/Metal with a NumPy fallback; the scatter/gather primitive
  neurons.py       two-compartment adaptive neurons (soma + nonlinear dendrite)
  connectivity.py  fixed fan-out sparse synapses, Dale's principle, delay ring buffer
  plasticity.py    three-factor local plasticity + homeostatic scaling
  simulator.py     event-driven simulator, exact spike compaction, SynOp accounting
  tasks.py         Split-MNIST / permuted-MNIST task suites + rate encoder
  baselines.py     backprop MLP, replay MLP, frozen-features readout
  cortex.py        spiking classifier whose plasticity does the learning
  metrics.py       BWT / forgetting / final accuracy, SynOp comparison
```

### The four mechanisms

1. **Two-compartment neurons.** A soma plus a distal dendrite whose activation is
   *non-monotonic* (`dend_mode="dcaap"`), modelling tuned dendritic Ca²⁺ action
   potentials (Gidon et al. 2020, *Science* 367:83). Adaptive thresholds give
   temporal memory (Bellec et al. 2020, *Nat Commun* 11:3625). The dendrite is
   **ablatable with one flag**, because whether it is causal is the question.
2. **Sparse event-driven coding.** k-WTA inhibition holds ~1–5% of neurons
   active; synaptic work is performed only for neurons that spike, so cost
   scales with *spikes*, not synapses.
3. **Three-factor local plasticity.** Each synapse keeps a local eligibility
   trace and the weight change is `eligibility × broadcast scalar`. No backward
   pass, no weight transport; the only non-local quantity is one scalar.
4. **Homeostatic regulation.** Synaptic scaling plus adaptive thresholds. This
   is not optional: two-factor Hebbian rules diverge in recurrent networks.

### Verified core behaviour

Measured, not asserted:

- **Single-neuron XOR.** The dcaap dendritic drive is non-monotonic —
  `[0.0, 0.82, 1.00, 0.91, 0.74, 0.41]` across input magnitudes
  `[0, 0.5, 1, 1.5, 2, 3]` — so "two inputs at once" produces *less* drive than
  "exactly one". A real spiking neuron then emits `[0,1,1,0]`, exactly XOR.
  With `dend_mode="linear"` the drive is monotonic and XOR is **provably
  impossible** at any threshold.
- **Plasticity direction.** Coincident pre+post → **+0.0100** (LTP);
  post-before-pre → **−0.0114** (LTD).
- **Dale's principle.** Zero sign violations, before or after plasticity.
- **Exact spike compaction.** The cumsum-based index extraction (MLX has no
  `nonzero`) matches `np.flatnonzero` exactly and in order.
- **Determinism.** Identical seeds give bit-identical spike counts and weights.

### Measured scale (M4 Max, 40-core GPU, 128 GB)

Full simulator, real spiking dynamics at a biologically plausible rate
(~39-56 Hz, 4-6% active), zero dropped spikes:

| Neurons | Fan-out | Synapses | ms/step | Biological time per wall-second |
|---|---|---|---|---|
| 1e5 | 256 | 0.03e9 | 1.31 | **0.77x real-time** |
| 1e6 | 256 | 0.26e9 | 8.38 | **0.12x real-time** |
| 1e6 | 1024 | 1.02e9 | 28.42 | **0.04x real-time** |
| 2e6 | 256 | 0.51e9 | 18.97 | 0.05x real-time |
| 4e6 | 256 | 1.02e9 | 40.24 | **0.02x real-time** |

Two honest caveats, both of which cost me an earlier overclaim:

1. **These are full-simulator numbers, and they are ~5x slower than a
   synaptic-delivery-only microbenchmark** (which suggested ~0.66x real-time at
   1e6). The microbenchmark excluded per-neuron state updates and was not a
   simulator measurement. The table above is the number that reproduces.
2. **The simulator is neuron-state bound, not synapse bound.** At 1e6 neurons
   the ~10 element-wise O(N) operations per timestep (membrane, dendrite with an
   `exp`, adaptation, refractory, thresholding, compaction cumsum) dominate over
   the ~1e7 synaptic events. Removing membrane noise changed nothing (8.45 vs
   8.45 ms/step), which confirms it. Consequence: sparsity buys less than the
   SynOp counts suggest, because the O(N) state update is paid regardless of how
   few neurons spike. This is a real limit of the current design and is stated
   rather than hidden.

MLX is ~**28x** faster than NumPy on the identical sparse delivery workload.
See `docs/SCALING.md` for the sweep and `bench/lead_independent_scale.json`.

---

## The falsification experiment

Class-incremental learning, single pass, **no replay, no task ID at test**.

| Arm | Model | Question it answers |
|---|---|---|
| `brain` | dendrites + k-WTA + three-factor | the proposal |
| `brain_nodend` | dendrites ablated | is the dendrite causal? |
| `brain_noplast` | plasticity off | is anything being learned? |
| `mlp` | backprop, parameter-matched | the real competitor |
| `mlp_replay` | backprop + 10% replay | the **strong** baseline |
| `frozen` | frozen features + ridge | is plasticity causal at all? |
| `shuffled` | randomised labels | is the metric meaningful? |

Results: see `experiments/results/`. **A negative result counts.** If backprop
matches or beats the substrate, that is the finding, and it will be reported as
such — consistent with Bartunov et al. 2018 (NeurIPS), where local learning
rules reached 93–99% top-1 error on ImageNet against ~71% for backprop.

### The headline result is a FALSIFICATION — please read this before the tables

The substrate wins on Split-MNIST but **loses badly** on the harder
Permuted-MNIST, and the harder benchmark is the one that matters.

| benchmark | `brain` | `mlp` (naive backprop) | verdict |
|---|---|---|---|
| Split-MNIST (disjoint class pairs) | **36.40 ± 6.97%** | 18.67 ± 0.26% | substrate wins by +17.7 pts |
| Permuted-MNIST (all classes, new pixel map each task) | 22.10 ± 3.33% | **68.30 ± 6.47%** | **backprop wins by +46.2 pts** |

On Split-MNIST the substrate beats naive backprop — but that baseline has
collapsed (it retains **0.00** on every task except the most recent), and a
plain 10% replay buffer matches the substrate (30.93 ± 1.83%, intervals
overlapping). On Permuted-MNIST, where tasks genuinely conflict and backprop
does *not* collapse, **backprop beats the substrate by 46 points**.

**Conclusion: the substrate's advantage does not generalise, and the project's
central hypothesis — that three-factor local plasticity enables competitive
continual learning — is NOT supported.** This is reported as the primary finding
rather than buried, because the split-MNIST number in isolation would be
misleading.

What *does* hold, verified independently of that claim:

- single-neuron **XOR** from a non-monotonic dendritic nonlinearity, with real
  spikes, and provably impossible under the linear ablation;
- correct **LTP/LTD directions** (+0.11 / −0.09) and zero Dale's-principle
  violations;
- **determinism** given a seed; **114 passing tests** including exact spike
  compaction verified against `np.flatnonzero`;
- measured scale of **4M neurons / 1.02e9 synapses / ~12 bytes per synapse** on
  one laptop;
- removing plasticity destroys learning, so the substrate's plasticity is doing
  real work.

Two further results went *against* the hypothesis and are reported as such:

1. **The dendritic nonlinearity hurts.** The ablation scores 41.87% (split) and
   35.90% (permuted) vs the intact 36.40% and 22.10%. Two independent benchmarks
   agree the mechanism is a *worse* feature extractor, because non-monotonic
   attenuation discards exactly the strong inputs a linear readout wants.
   Single-neuron XOR still works; it just doesn't make a better code.
2. **Replay matches the substrate** on split-MNIST, at far lower complexity.

**Why it fails, diagnosed rather than guessed.** On a single permuted task with
*no* interference from other tasks, the substrate plateaus at ~33% accuracy
(10-way), so forgetting is not the limiting factor — the representation is. It
does beat a fixed random projection (14.3%), so the recurrent dynamics add real
signal, but a linear readout cannot extract it: on identical cached codes, a
**quadratic readout scores 33.3% where a linear one scores 15.7%**. Raising
fan-out to 256 changes nothing (31.0-33.7%). The bottleneck is the code→readout
interface, and that is where a future attempt should look. See `docs/RESULTS.md`
§3.1.

Full analysis and every defect found along the way: `docs/RESULTS.md` and
`docs/DECISIONS.md`.

Full analysis, caveats and the defects found along the way:
`docs/RESULTS.md` and `docs/DECISIONS.md`.

### Energy accounting is honest

We report **active synaptic operations (SynOps)** — an *operation count*, not
joules. This distinction matters: a biological synapse costs ~1e-16 J, while a
SynOp on Loihi costs ~23.6 pJ. "Spiking is cheap" is not a free claim, and this
repo does not make it.

---

## Repo layout

- `brain/` — the substrate (library)
- `experiments/` — falsification experiments and their JSON results
- `bench/` — scale benchmarks
- `tests/` — independent validation suite
- `docs/PARADIGM.md` — the paradigm, novelty claim, and falsification criteria
- `docs/DECISIONS.md` — decision record, including bugs found and claims overridden
- `docs/SCALING.md` — measured scaling and the gap to 86e9 neurons
- `docs/EFFICIENCY.md` — **retraction**: the claimed win over a transformer on
  efficiency was a training-budget artifact (the baseline was at the bigram
  floor). Read this before believing any language-model number in this repo.
- `docs/WORKING_MEMORY.md` — working memory is a dendritic trace, not recurrence
- `docs/NEURON_OPERATING_POINT.md` — the dCaAP neuron is an onset detector, with
  a fragile operating point that silently produced zero spikes
- `docs/SUBSTRATE_CONTRIBUTION.md` — ablation showing self-organisation contributes ~nothing
- `docs/REVIEW.md` — independent adversarial adjudication of every published claim
- `docs/READOUT.md` — the readout rule, not the code, is the bottleneck
- `docs/AFFERENTS.md` — projection polarity, not convergence, drives the input-path gain
- `docs/TRACE_LOCUS.md` — whether the dendritic trace is per-neuron or per-synapse
- `docs/THREEFACTOR.md` — the thesis experiment (error signal reaching a hidden layer)
- `EXIT.md` — consolidated honest status: what is established, what is retracted

## References

Key sources are listed in [`docs/PARADIGM.md`](docs/PARADIGM.md) §9.
