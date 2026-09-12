# A New Way To Create AI: The Cortical Substrate as a Learning Harness

> **Status: TESTED AND NOT SUPPORTED.** The experiment in §6 was run on two
> benchmarks. The paradigm survived the easier one and was **falsified on the
> harder one**: on Permuted-MNIST, backprop beats the substrate by 46 points.
> The dendritic mechanism specifically made results *worse* on both benchmarks.
> See `docs/RESULTS.md` §3 for the numbers and §7 below for how the stated
> falsification conditions were met. The document is retained unedited in its
> reasoning so that the *proposal* can be compared against the *outcome*.

## 1. The honest starting point

The original brief was "fully simulate the human brain locally, then scale it
up." That specific goal is arithmetically impossible and it is important to say
so plainly rather than quietly redefine it.

| | Human brain | This laptop (measured) | Gap |
|---|---|---|---|
| Neurons | ~86.1e9 (Azevedo et al. 2009) | 4.0e6 | ~10^4.33x |
| Synapses | ~1e14-1e15 | 1.024e9 | ~10^5.0-6.0x |
| Synapse storage | ~1 PB | 11.6 GiB (12 B/synapse) | ~10^5x |

Measured figures come from `bench/results_scale.json`; see `docs/SCALING.md`.

The Human Brain Project spent roughly EUR 607M over ten years and concluded that
a full replica was neither achievable nor of clear practical use. Blue Brain's
*single cortical column* — 31,000 neurons, 37M synapses, 0.29 mm^3 (Markram et
al. 2015, Cell 163:456) — was already supercomputer-scale. Repeating the replica
race on a laptop would be a strictly worse version of a programme that has
already been run and already reported its conclusion.

So the useful question is not "how do we simulate a brain". It is:

> **Which of the brain's computational mechanisms solve problems that current
> AI structurally cannot solve — and can we build those mechanisms as a
> substrate that runs locally?**

That question is answerable. It is also where the genuinely new work is.

## 2. What is structurally broken about LLMs

These three deficits are not about parameter count and will not be fixed by more
scale. They are architectural.

1. **Weights are frozen at inference.** An LLM cannot learn from a single
   experience and keep it. Fine-tuning on new data causes catastrophic
   forgetting. Every deployed LLM is therefore a snapshot, not a learner.
2. **Compute is dense and input-independent.** Every token costs roughly the
   same FLOPs whether the question is trivial or hard. The brain runs on ~20 W
   with ~1-5% of neurons active at any moment, and pays per *spike*, not per
   parameter.
3. **Credit assignment is global.** Backpropagation requires storing
   activations and performing a full backward pass with exact weight transport
   — a mechanism with no known biological instantiation.

## 3. The three mechanisms we take from the brain

Each mechanism maps onto exactly one of the deficits above.

### 3.1 Dendritic nonlinearity (→ representational efficiency)

Human L2/3 pyramidal dendrites emit Ca2+ action potentials that are *tuned*:
they peak for a preferred input magnitude and **attenuate** for stronger input
(Gidon et al. 2020, Science 367:83). A non-monotonic subunit makes a single
neuron capable of linearly non-separable (XOR-class) computation that a point
neuron cannot perform at any width of a single layer.

This is implemented here as an explicit, ablatable nonlinearity
(`dend_mode="dcaap"` vs `"linear"`). It is the causal hypothesis under test —
not decoration.

### 3.2 Sparse, event-driven coding (→ compute efficiency)

k-WTA inhibition gives ~1-5% active populations. Because communication is
spike-driven, cost scales with *spikes*, not synapses. On this machine that
measured out at roughly 10^9 synapses simulated at a fraction of biological
real-time (see `docs/SCALING.md`), and the operation count — active synaptic
operations, SynOps — is reported honestly rather than asserted.

### 3.3 Three-factor local plasticity (→ continual learning)

Each synapse keeps a local eligibility trace from its own pre/post activity, and
the weight change is eligibility x a broadcast scalar neuromodulator
(Frémaux & Gerstner 2015; Gerstner et al. 2018). There is no backward pass and
no weight transport. The only non-local quantity is **one scalar**, broadcast
identically to every synapse.

This is the mechanism that could let a system learn from experience in a single
pass without replay — the thing an LLM cannot do.

> **STATUS CORRECTION (must read): mechanism 3.3 was never applied to a hidden
> layer, so the continual-learning comparison did not test it.**
>
> Two facts from this repo's own code and results:
>
> 1. `brain/cortex.py` line 183 drives the substrate with `neuromod=1.0`, a
>    *constant*. That is pure unsupervised Hebbian learning. No error signal
>    reaches the afferent projection, so the eligibility trace above is never
>    modulated by anything informative.
> 2. The only error-driven layer is the readout, and it is a single linear
>    layer updated as `W += lr * outer(c, target - W.T @ c)`. That is **exact
>    gradient descent on one linear layer**, not a local approximation to it.
>
> So the model under test was structurally: *frozen input projection -> fixed
> spiking nonlinearity -> one linear layer trained by gradient descent*. The
> comparison against a two-layer backprop MLP therefore measured whether **one
> linear layer on untrained random features** can do 10-way MNIST. It cannot,
> and the answer was 22% versus 68%.
>
> `docs/RESULTS.md` §3.1 already said the same thing from the other side:
> trained on a *single* task with *zero* interference the substrate plateaus at
> **~33%**, and a static random ReLU projection at matched sparsity beats its
> probe (0.765 vs 0.702). The limit is **representational capacity**, not
> forgetting.
>
> **The correct label for the continual-learning result is UNTESTED, not
> falsified.** "Local plasticity with a broadcast scalar beats backprop at
> retaining tasks" has not been contradicted by this project; it has not been
> run. What *is* falsified is the weaker claim that this configuration's
> self-organising dynamics contribute anything measurable
> (`docs/SUBSTRATE_CONTRIBUTION.md`).
>
> The thesis experiment therefore is: make the afferent projection plastic
> under the three-factor rule with an **error-derived** third factor, and
> compare it against the identical model with the hidden learning rate set to
> zero. Success is pre-registered as beating that frozen ablation by more than
> twice the seed standard deviation. See `experiments/threefactor_hidden.py`
> and `docs/THREEFACTOR.md`.

## 4. The paradigm

> **A sparse, dendritically-expanded, locally-plastic cortical substrate used as
> a learning harness: the substrate supplies the learning *dynamics*, and a
> conventional model supplies the representational *priors*.** The substrate is
> the harness; the model is mounted in it.

This inverts the usual framing. Instead of asking a huge dense network to become
more brain-like, we let a small plastic substrate do the one thing dense frozen
networks cannot — adapt online, permanently — while a conventional model
contributes what it is already good at: broad semantic generalisation.

Concretely, two composable pieces:

1. **The substrate (built and running here).** Two-compartment adaptive neurons,
   fixed fan-out sparse connectivity with axonal delays, three-factor
   plasticity, homeostatic regulation. It learns a task sequence in one pass.
2. **The harness loop (next).** A frozen LLM proposes representations or
   predictions; the substrate consumes them as input and consolidates them into
   sparse, plastic, permanent circuits. Consolidation is the substrate's job;
   generalisation is the LLM's.

## 5. What is and is not new

Intellectual honesty requires separating these, because two of the component
ideas have known anti-results.

**Not new, and must not be claimed as new:**

- *Predictive coding as a local alternative to backprop.* Whittington & Bogacz
  (2017) proved that predictive coding with local Hebbian updates **converges to
  backprop** under stated conditions. "Local" does not by itself mean "new
  paradigm", and any claim resting on that equivalence is dead on arrival.
- *Three-factor learning rules.* Established (Frémaux & Gerstner 2015) and
  already implemented in neuromorphic hardware.
- *Dendritic computation.* Established experimentally (Gidon et al. 2020);
  "active dendrites" have appeared in prior SNN work. Iyer et al. 2022
  (Front Neurorobot 16:846219) is specifically criticised for confounding
  dendritic structure with activation sparsity by comparing a k-WTA spiking
  model against a dense ReLU MLP. That criticism is correct, and the ablation
  design in §6 is built to avoid it.

**The defensible claim of novelty** is narrower and therefore stronger: the
*specific combination* — non-monotonic dendritic subunits + enforced sparse
k-WTA coding + three-factor local plasticity carrying credit through a broadcast
scalar — applied to **single-pass continual learning without replay**, evaluated
against a **parameter- and SynOp-matched** backprop baseline, with the dendritic
mechanism isolated by ablation. The novelty is in the composition and in the
measurement, not in any single ingredient.

## 6. The falsifiable experiment

Class-incremental learning over a task sequence, single pass, no replay,
**no task identity at test time**. Every arm sees the same data in the same
order with the same budget.

| Arm | Model | Purpose |
|---|---|---|
| B | **Brain substrate** — dendrites + k-WTA + three-factor | the proposal |
| C | **B with dendrites ablated** (`dend_mode="linear"`) | is the dendrite causal? |
| D | **B with plasticity disabled** | is learning happening at all? |
| A | **Backprop MLP**, parameter-matched | the real competitor |
| E | **Backprop + 10% replay** | the *strong* baseline (Shen 2019) |
| F | **Frozen random features + linear readout** | is plasticity causal, or are fixed features enough? |
| G | **Shuffled labels** | sanity: must NOT learn |

Metrics: final mean accuracy, backward transfer (BWT), forgetting, and active
SynOps per sample. Multiple seeds, mean +/- CI.

**Falsification conditions** (any one of these kills the claim):

- **A >= B** within confidence intervals — backprop is no worse at continual
  learning here, so the substrate adds nothing.
- **C ≈ B** — the dendritic nonlinearity is not causal; the result is just
  sparsity, which is a well-known effect.
- **F ≈ B** — plasticity is not causal; a fixed random projection suffices.
- **E dominates B** — replay fixes forgetting more cheaply, so the
  no-replay property is not worth the cost.
- **G learns** — there is a bug, or the metric is meaningless.

A **negative result is a real result** and will be reported as one. If the
substrate does not beat the matched baselines, the correct conclusion is that
single-pass local learning does not solve continual forgetting at this scale,
which is itself worth knowing and is consistent with Bartunov et al. 2018
(NeurIPS), where local rules reached 93-99% top-1 error on ImageNet against
~71% for backprop.

## 7. Scale-up path (and why scale is not the hard part)

The cortex is remarkably uniform (Mountcastle's column hypothesis), so scaling
means replicating one canonical circuit, not designing a bespoke brain. The
scaling axis is therefore engineering, not science:

| Scale | Neurons | Where | What question it answers |
|---|---|---|---|
| Now | 1e5-4e6 | this laptop | do the mechanisms learn without forgetting? |
| | 1e8 | a few GPUs | does the effect survive realistic recurrent dynamics? |
| | 1e9 | small cluster | does a canonical circuit scale, or does it need architecture? |
| | 1e11 | datacentre | whole-cortex approximations |

The measured local ceiling corresponds to roughly **129 Blue Brain cortical
columns (~37 mm^3)** of tissue. That is a plausible slice of cortical tissue,
and it is 5 orders of magnitude short of a brain.

The binding constraint is memory bandwidth for irregular synaptic access, not
FLOPs — which is why the measured numbers here matter more than a parameter
count. The honest framing is that local scale buys a **mechanism testbed**, and
mechanisms are what transfer across five orders of magnitude. Parameter counts
do not.

## 8. Why this is a better bet than the replica race

- It tests a *mechanism* rather than reproducing a *specimen*.
- It is falsifiable, with explicit controls and a pre-registered negative
  outcome.
- It targets the three deficits that scaling LLMs provably does not fix.
- Every claim is tied to a measurement on commodity hardware.

## 9. References

- Azevedo et al. 2009, *J Comp Neurol* 513:532 — 86.1 +/- 8.1e9 neurons.
- Markram et al. 2015, *Cell* 163:456 — Blue Brain cortical column, 31k neurons.
- Gidon et al. 2020, *Science* 367:83 — tuned dendritic Ca2+ action potentials.
- Bellec et al. 2020, *Nat Commun* 11:3625 — adaptive neurons give LSTM-like capacity.
- Frémaux & Gerstner 2015, *Front Neural Circuits* 9:85 — three-factor learning.
- Whittington & Bogacz 2017, *PLoS Comput Biol* 13:e1005381 — predictive coding ≈ backprop.
- Bartunov et al. 2018, *NeurIPS* — local learning rules do not scale to ImageNet.
- Iyer et al. 2022, *Front Neurorobot* 16:846219 — dendrite/sparsity confound.
- Shen et al. 2019 — replay closes most of the continual-learning gap.
- Attwell & Laughlin 2001, *J Cereb Blood Flow Metab* 21:1133 — energy budget.
- Kunkel et al. 2014, *Front Neuroinform* 8:78 — K computer scale simulation.
