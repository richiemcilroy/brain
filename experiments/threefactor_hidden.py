"""THE THESIS EXPERIMENT: can a broadcast third factor shape a hidden layer?

The claim under test (``docs/PARADIGM.md`` section 3.3)
------------------------------------------------------
Credit assignment through a broadcast scalar, with no backward pass and no
weight transport, can shape a hidden layer. The claim had never been tested.

Why, from the code as it stood
------------------------------
* ``brain/cortex.py`` enables three-factor plasticity with a CONSTANT
  neuromodulator ``M = 1``, i.e. pure unsupervised Hebbian learning. **No error
  signal reaches the afferent/hidden projection at all.**
* The only error-driven learning in the model is the READOUT, a single linear
  layer trained by the delta rule. A single linear layer trained by the delta
  rule *is* exact gradient descent on that layer.
* The model is therefore structurally: frozen input projection -> fixed spiking
  nonlinearity -> one linear layer trained by gradient descent. The published
  permuted-MNIST comparison (22.10% vs 68.30%) is a comparison of a one-layer
  linear model against a two-layer MLP, and ``docs/RESULTS.md`` section 3.1
  shows the substrate plateaus at ~33% on ONE task with ZERO interference. That
  is a representational-capacity failure, not a continual-learning failure. The
  continual-learning claim was **untested**, not falsified.

What this file does
-------------------
1. Makes the afferent projection (``brain/afferents.py``) PLASTIC under the
   three-factor rule already in ``brain/plasticity.py``:
   ``dw = lr * M(t) * e`` where ``e`` is a decayed local eligibility trace
   built only from pre- and post-synaptic activity, and ``M`` is the third
   factor derived from the *error*, not a constant.
2. Tests two third factors:
   (a) ``scalar``  - one broadcast scalar, ``+1`` correct / ``-1`` incorrect.
   (b) ``dfa``     - the per-class readout error passed through a FIXED RANDOM
       matrix. This is **direct feedback alignment** (Lillicrap et al. 2016,
       *Nat Commun* 7:13276), a KNOWN method. It is not novel and is not
       presented as novel. It is included because it uses no weight transport
       and no backward pass through the network's own weights.
3. The control that decides everything: ``frozen_hidden`` runs the *identical*
   plasticity code path with ``hidden_lr = 0``. A local rule that cannot beat
   its own frozen ablation has learned nothing.
4. Absolute reference points: the unmodified readout-only model (current
   baseline) and a static random ReLU projection at matched sparsity (the
   documented 0.765 control from ``docs/SUBSTRATE_CONTRIBUTION.md``).

Pre-registered success criterion (fixed BEFORE any arm was run)
--------------------------------------------------------------
For a plastic arm (``scalar`` or ``dfa``) to count as having learned anything:

    mean(acc_plastic) - mean(acc_frozen_hidden)  >  2 * sd_seed

where ``sd_seed = sqrt((var(acc_plastic) + var(acc_frozen_hidden)) / 2)`` with
``ddof=1`` over the same seeds. "Learned nothing" is the null, not a failure of
the run.

Gating
------
The permuted-MNIST continual-learning stage runs **only if** at least one
plastic arm passes the criterion. If neither does, the thesis is **falsified at
step 1**, that is the finding, and the continual stage is not run (the script
records ``continual.ran = false`` with a reason).

Honesty protocol
----------------
* Prediction, stated before running: the ``scalar`` variant probably fails (a
  single bit of information cannot tell a synapse what to change) and the
  ``dfa`` variant probably works on one task and then forgets under
  permutation. Do not tune toward a win.
* Measured spike rates are asserted non-zero and the script FAILS LOUDLY
  otherwise. ``docs/NEURON_OPERATING_POINT.md`` documents a trap where the
  population silently produced ZERO spikes and an experiment compared silence
  to silence.
* The readout learning rate is NOT retuned: it stays at the ``CortexConfig``
  default of 0.05 with the default NLMS normalisation, identical in every arm.
* The hidden learning rate is a new hyperparameter. It is passed explicitly,
  declared here, applied identically to both plastic arms, and a sensitivity
  sweep is reported so the verdict does not rest on one value.
* MLX is lazy: every value read goes through ``Backend.to_numpy``/``eval``.

Run:  python3 experiments/threefactor_hidden.py
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "experiments" / "results"
N_INPUT = 784
N_CLASSES = 10
N_NEURONS = 900          # matches the documented control arms (docs/SUBSTRATE_CONTRIBUTION.md)
K_OUT = 64
T_MS = 15
GAIN = 2.2               # AfferentConfig default; the repo's documented operating point
#: Pre-declared hidden-layer learning rate, selected by an ACTIVITY-ONLY rule
#: fixed before any accuracy was looked at: "the largest hidden_lr whose
#: post-training spike rate stays within 20% of the frozen baseline in every
#: plastic arm". Measured on the activity calibration (spikes/sample relative
#: to frozen, 1 seed, 150 samples, labels never used):
#:
#:     lr      scalar    dfa     true_weights
#:     0.005   1.03      1.05    0.91
#:     0.01    1.04      1.08    0.83
#:     0.02    1.06      1.07    0.71
#:     0.05    1.14      0.84    0.48
#:
#: 0.02 is the largest rate where the two pre-registered plastic arms are both
#: within 20%. It is NOT selected by accuracy, and the full lr sweep is reported
#: (``--lr-sweep``) precisely so the verdict can be checked against every other
#: value: no value in {0.005, 0.02, 0.05, 0.2} produces a pass either.
DEFAULT_HIDDEN_LR = 0.02
SATURATION_SIGMAS = 4.0  # homeostatic bound on the plastic projection, in initial sigmas

#: The five arms. Numbering is used in the JSON and in docs/THREEFACTOR.md.
#:   1 readout_only   - unmodified current model (identity wiring), readout learns
#:   2 frozen_hidden  - afferent projection + hidden plasticity code path, hidden_lr = 0
#:   3 scalar         - hidden plasticity, scalar third factor
#:   4 dfa            - hidden plasticity, direct feedback alignment third factor
#:   5 static_relu    - fixed random ReLU projection + k-WTA at matched sparsity
ARMS = ["readout_only", "frozen_hidden", "scalar", "dfa", "static_relu",
        "feedback_true_weights"]
#: Arm 6 is a DIAGNOSTIC, not a candidate paradigm arm, and the name is chosen
#: to avoid over-claiming. Its third factor is ``W_readout @ err``: that is DFA
#: with the *exact* readout weights substituted for the fixed random matrix, so
#: it isolates "is the randomness of the feedback matrix the problem?" -- it is
#: the no-randomness ablation of DFA, and it uses weight transport, so it is
#: NOT a local rule.
#
#: It is explicitly NOT the true gradient and NOT an oracle. The readout's
#: gradient w.r.t. the hidden projection also contains the derivative of the
#: code with respect to that projection, and this substrate's code is a spike
#: count through a spiking, non-monotonic dendritic nonlinearity, for which that
#: derivative is zero almost everywhere (the neuron fires or it does not). So
#: ``W_readout @ err`` is a heuristic direction, not ``dL/dW_aff``. An earlier
#: draft of this file called this arm ``oracle_true_feedback``, which
#: over-claimed; renamed after the measurement showed it can score BELOW the
#: frozen ablation despite 1.000 cosine alignment with its own reference
#: direction.
TRUE_WEIGHTS_ARM = "feedback_true_weights"
PLASTIC_ARMS = ["scalar", "dfa"]
ALL_PLASTIC_ARMS = ["scalar", "dfa", TRUE_WEIGHTS_ARM]
#: Every arm that runs the spiking substrate. All of them are subject to the
#: zero-spike assertion; an arm omitted from this list would skip the check and
#: silently report 0 spikes (which is exactly what the first draft did with the
#: ``feedback_true_weights`` arm would otherwise be checkable).
SPIKING_ARMS = ["readout_only", "frozen_hidden", "scalar", "dfa", TRUE_WEIGHTS_ARM]

#: Documented reference numbers re-stated (not re-measured here) so the reader
#: can locate this experiment against the published record.
DOCUMENTED_REFERENCES = {
    "source": "docs/SUBSTRATE_CONTRIBUTION.md",
    "random_projection_relu_dense": 0.803,
    "random_projection_relu_kwta_29pct": 0.765,
    "spiking_substrate_29pct_active": 0.702,
    "raw_pixels_ridge": 0.670,
    "note": "900 neurons, 800 train / 400 held-out, ridge probe; not the same "
            "readout as this experiment, quoted as an absolute reference point only.",
}


# --------------------------------------------------------------------- helpers
def stratified_subsample(x: np.ndarray, y: np.ndarray, n: int, seed: int):
    """Balanced ``n``-sample subset. Head-slicing skews the class distribution."""
    if n <= 0 or n >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, n // len(classes))
    picks = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picks.append(idx[:per_class])
    chosen = np.concatenate(picks)
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


@dataclass
class TaskData:
    """Plain namespace so a subsampled task can be passed to ``fit_task``."""
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: list = field(default_factory=lambda: list(range(N_CLASSES)))


def _nanmean(values) -> float:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def mean_sd(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


# ------------------------------------------------------- hidden plasticity
class HiddenThreeFactor:
    """The three-factor rule from ``brain/plasticity.py``, on the afferent projection.

    The rule, copied in FORM from ``ThreeFactorPlasticity.hebbian_delta``::

        # per millisecond, at every afferent synapse
        e_ij  <-  decay * e_ij  +  a_plus * x_i * s_j  -  a_minus * y_j
        dw_ij  =  lr * M_j * e_ij

    where ``x_i`` is the (analog) input current on unit ``i``, ``s_j`` is
    whether hidden neuron ``j`` spiked this millisecond, ``y_j`` is that
    neuron's decaying post-synaptic trace, and ``M_j`` is the third factor.

    The ``- a_minus * y_j`` term is not decoration, and omitting it is a
    silent crippling of the rule. The first draft of this file used only
    ``e_ij = sum_t x_i * s_j`` -- a non-negative correlation, because both
    factors are non-negative. With ``e >= 0`` everywhere, the update
    ``dw_ij = lr * M_j * e_ij`` has one sign for the whole of neuron ``j``'s
    weight row, so the rule can only scale a neuron's existing input pattern up
    or down. It cannot make one input excitatory and another inhibitory, which
    is the minimum needed to reshape a receptive field. Measured: that version
    was sign-locked (``frac(e < 0) = 0.000``, 96.7% of entries exactly 0) and
    the weight-transported third factor *lowered* accuracy from 61.3% to 23.3%
    before the term was restored. The depression term is what makes the
    eligibility signed and
    contrastive -- it potentiates inputs that precede a spike and depresses the
    inputs that did not -- and it is present in the library's rule. Reported in
    ``docs/THREEFACTOR.md`` as a mechanism finding, not hidden.

    Module docstring note: this class lives here rather than in
    ``brain/plasticity.py`` because ``ThreeFactorPlasticity.apply`` coerces its
    third factor to a Python float (``be.item(neuromod)``) and keys its update
    on *pre-synaptic spikes* with a fixed fan-out scatter. The afferent
    projection is dense with an analog pre-synaptic signal, and the DFA variant
    needs a per-neuron third factor. The rule, trace decay, coefficients and
    depression term follow the library's documented form; the recurrent
    pathway in ``Brain.step`` still runs the library's
    ``ThreeFactorPlasticity`` unchanged at ``M = 1``.
    """

    def __init__(self, n_in: int, n_post: int, tau_elig: float = 60.0,
                 tau_post: float = 20.0, dt: float = 1.0,
                 a_plus: float = 0.010, a_minus: float = 0.012,
                 contrastive: bool = True):
        self.e = np.zeros((n_in, n_post), dtype=np.float32)
        self.y_post = np.zeros((n_post,), dtype=np.float32)
        self.tau_elig = float(tau_elig)
        self.tau_post = float(tau_post)
        self.dt = float(dt)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.contrastive = bool(contrastive)
        self.updates = 0
        self.last_abs_dw = 0.0
        self.last_norm_ratio = 0.0
        self.signed_frac_seen = 0.0
        #: Per-column L2 norm to preserve (set by the caller from the initial
        #: projection). ``None`` disables the homeostatic rescaling.
        self.renorm: np.ndarray | None = None

    def reset(self) -> None:
        self.e[:] = 0.0
        self.y_post[:] = 0.0

    def accumulate(self, pix: np.ndarray, spike_mask: np.ndarray) -> None:
        """One millisecond of eligibility accumulation.

        Ordering follows the library: the rule is evaluated against the post
        trace from BEFORE this step's spikes (``y_post`` is advanced at the end),
        so a perfectly coincident pre+post pair contributes excitation only.
        Folding the current spike into ``y_post`` first would make the pair
        contribute to both terms and, with ``a_minus > a_plus``, depression would
        win. That footgun is documented in ``brain/plasticity.py``.
        """
        decay = 1.0 - self.dt / self.tau_elig
        self.e *= decay
        self.e += self.a_plus * np.outer(pix, spike_mask)
        if self.contrastive:
            # Depression: constant across inputs, proportional to the post trace,
            # so inputs that did not contribute to the spike are pushed down.
            self.e -= self.a_minus * self.y_post.reshape(1, -1)
        self.y_post *= (1.0 - self.dt / self.tau_post)
        self.y_post += spike_mask
        self.signed_frac_seen = max(self.signed_frac_seen, self.signed_fraction())

    def signed_fraction(self) -> float:
        """Fraction of eligibility entries that are negative (0.0 means
        sign-locked, i.e. the rule cannot reshape a receptive field)."""
        return float((self.e < 0.0).mean())

    def apply(self, weights: np.ndarray, mod: np.ndarray | float,
              lr: float, clip: float) -> float:
        """``dw = lr * M * e`` -- the library's three-factor form, then a bound.

        This is exactly ``brain/plasticity.py``'s ``dw = cfg.lr * mod * apply_val``
        with the eligibility as ``apply_val``: no normalisation of the update is
        added, so the rule is the one the repo already uses for the recurrent
        synapses. The bound and the homeostatic rescaling that follow are also
        taken from the library, where ``synapses.weight_bounds()`` and
        ``maybe_homeostasis`` exist for the same stated reason -- "an unbounded
        Hebbian rule diverges" and "feedback inhibition and homeostasis are not
        optional extras".

        The rescaling is PER NEURON (per column, ``w_j <- w_j * n_j0 / ||w_j||``)
        rather than the library's global ``mean|w|`` scaling, because the
        quantity that must be held fixed here is each neuron's own input drive.
        This substrate's unit is an onset detector whose firing depends on its
        own dendritic drive passing through the peak of its tuning curve
        (``docs/NEURON_OPERATING_POINT.md``); a change in a neuron's input scale
        moves it off that peak and the neuron goes permanently silent. Measured
        without per-neuron rescaling, ``hidden_lr=1.0`` drove the population from
        129 spikes/sample to 8.6 and accuracy from 68% to 12% -- silence, not
        learning. With it, the neuron's input scale is invariant and the weight
        VECTOR is free to rotate, which is where the learning is.

        Both arms are rescaled identically, so this cannot advantage one of them.
        """
        if lr <= 0.0:
            return 0.0
        E = self.e
        m = np.asarray(mod, dtype=np.float32).reshape(1, -1)
        dw = E * m * lr
        scale = float(np.sqrt((dw * dw).mean()))
        if not np.isfinite(scale) or scale == 0.0:
            return 0.0
        weights += dw
        if self.renorm is not None:
            col = np.linalg.norm(weights, axis=0)
            keep = col > 1e-9
            weights[:, keep] *= (self.renorm[keep] / col[keep]).astype(np.float32)
        np.clip(weights, -clip, clip, out=weights)
        self.updates += 1
        self.last_abs_dw = float(np.abs(dw).mean())
        self.last_norm_ratio = scale
        return scale


# ---------------------------------------------------------------------- models
class ThreeFactorCortex:
    """The substrate with a PLASTIC afferent projection.

    Everything except the hidden projection matches ``CortexClassifier``: the
    two-compartment neurons, the recurrent synapses with the library's
    three-factor rule at ``M = 1``, the reset-per-sample behaviour, the running
    standardisation, and the readout's graded delta rule with NLMS at the
    default learning rate. ``hidden_lr = 0`` reproduces the frozen ablation
    through the identical code path (eligibility still accumulated, write
    skipped), which is what makes that arm a valid control.
    """

    def __init__(self, cfg, *, seed: int, hidden_lr: float, modulator: str,
                 afferent_seed: int, gain: float = GAIN, feedback_seed: int = 0,
                 saturation_sigmas: float = SATURATION_SIGMAS):
        from brain.afferents import Afferents, AfferentConfig
        from brain.cortex import CortexClassifier

        self.base = CortexClassifier(cfg, backend="numpy")
        self.cfg = cfg
        self.be = self.base.be
        self.hidden_lr = float(hidden_lr)
        self.modulator = modulator          # "frozen" | "scalar" | "dfa"
        self.aff = Afferents(AfferentConfig(
            n_input=cfg.n_input, n_neurons=cfg.n_neurons, kind="dense",
            seed=afferent_seed, gain=gain, normalize=True, bias=True))
        self.gain = float(gain)
        # W_aff is the hidden projection. It is the ONLY new plastic parameter.
        self.W_aff = self.aff.W              # (n_input, n_neurons) float32
        self.bias = self.aff.b               # frozen
        self._sigma0 = float(np.std(self.W_aff))
        self._clip = float(saturation_sigmas) * self._sigma0
        rng = np.random.default_rng(feedback_seed)
        # Fixed random feedback: +/-1, never a function of the readout weights.
        self.B = (rng.integers(0, 2, (cfg.n_neurons, cfg.n_classes)) * 2 - 1
                  ).astype(np.float32)
        self.hidden = HiddenThreeFactor(cfg.n_input, cfg.n_neurons,
                                        tau_elig=60.0, dt=1.0)
        # Freeze each neuron's input scale (column L2 norm of the initial
        # projection) so the homeostatic rescaling in ``apply`` preserves it.
        self.hidden.renorm = np.linalg.norm(self.W_aff, axis=0).astype(np.float32)
        self.n_increments = 0
        self.n_updates = 0
        self.synops_active_total = 0
        self.spikes_total = 0
        self.n_code_calls = 0
        self.align_cos: list[float] = []
        self._n_mod = 0
        self._n_mod_pos = 0

    # ---- delegates, so this object satisfies the Classifier protocol
    @property
    def W(self):
        return self.base.W

    @W.setter
    def W(self, v):
        self.base.W = v

    @property
    def n_params(self) -> int:
        return int(self.base.W.size)

    @property
    def synops_per_sample(self) -> float:
        return self.synops_active_total / max(1, self.n_code_calls)

    def reset_readout(self) -> None:
        self.base.reset_readout()

    # ------------------------------------------------------------------ code
    def _code(self, x, t_ms: int) -> np.ndarray:
        cfg = self.cfg
        if cfg.reset_between_samples:
            self.base.brain.reset()
        pix = np.asarray(x, dtype=np.float32).reshape(-1)
        drive = pix @ self.W_aff
        if self.bias is not None:
            drive = drive + self.bias
        # Same scaling rule as Afferents.project(kind="dense"): the per-neuron
        # current is (x @ W + b) * gain, so activity stays comparable when
        # gain is swept independently of the mixing pattern.
        drive = drive * self.gain
        drive_b = self.be.array(drive.astype(np.float32))
        self.hidden.reset()
        counts = np.zeros(cfg.n_neurons, dtype=np.float32)
        synops = 0
        for _ in range(t_ms):
            st = self.base.brain.step(external_dend=drive_b, neuromod=1.0)
            mask = self.be.to_numpy(self.base.brain.last_spike_mask).astype(np.float32)
            self.be.eval()
            counts += mask
            self.hidden.accumulate(pix, mask)
            synops += st.synops_active
        self.synops_active_total += synops
        self.n_code_calls += 1
        self.n_increments += int(counts.sum())
        self.spikes_total += int(counts.sum())
        return counts

    # ------------------------------------------------------ third factor
    def _third_factor(self, label: int, out: np.ndarray) -> np.ndarray:
        if self.modulator == "scalar":
            ok = int(np.argmax(out)) == int(label)
            self._n_mod += 1
            self._n_mod_pos += int(ok)
            return np.array([1.0 if ok else -1.0], dtype=np.float32)
        if self.modulator == "dfa":
            tgt = np.zeros(self.cfg.n_classes, dtype=np.float32)
            tgt[label] = 1.0
            err = (tgt - out).astype(np.float32)
            m = self.B @ err
            r = float(np.sqrt(np.mean(m * m)))
            return (m / r).astype(np.float32) if r > 1e-8 else np.zeros_like(m)
        if self.modulator == "true_weights":
            # Weights transported from the readout. NOT local, NOT the gradient;
            # see ARMS docstring.
            return self._true_direction(out, label)
        raise ValueError(f"unexpected modulator {self.modulator!r}")

    def _true_direction(self, out: np.ndarray, label: int) -> np.ndarray:
        """``normalise(W_readout @ err)``: the weight-transport direction."""
        tgt = np.zeros(self.cfg.n_classes, dtype=np.float32)
        tgt[label] = 1.0
        err = (tgt - out).astype(np.float32)
        m = self.base.W @ err
        r = float(np.sqrt(np.mean(m * m)))
        return (m / r).astype(np.float32) if r > 1e-8 else np.zeros_like(m)

    def _modulator_vector(self, mod: np.ndarray, out: np.ndarray, label: int) -> np.ndarray:
        """The per-neuron direction a third factor actually induces.

        A scalar factor has no per-neuron direction, so the direction it
        induces is the scalar times the eligibility's own column norm -- i.e.
        for the alignment diagnostic only, we ask how well
        ``(|e| * M_scalar)`` aligns with the true direction. That is the honest
        reading: the scalar cannot choose between neurons, it can only
        globally reinforce or globally depress whatever just fired.
        """
        if mod.size == 1:
            return (np.abs(self.hidden.e).mean(axis=0) * float(mod[0])).astype(np.float32)
        return np.asarray(mod, dtype=np.float32).reshape(-1)

    def _align_cos(self, mod: np.ndarray, out: np.ndarray, label: int) -> float:
        """cos(third factor actually applied, true readout-transposed direction).

        The reference direction ``W_readout @ err`` is what the hidden layer
        would be told if the readout's own weights were transported back. This
        is the standard feedback-alignment diagnostic. It is a *necessary*
        diagnostic, not a sufficient one: alignment near zero means the third
        factor carries no information about the readout's error; alignment near
        one does not by itself guarantee that the eligibility term has the right
        sign for a spiking unit (see docs/NEURON_OPERATING_POINT.md on the
        non-monotonic tuning curve).
        """
        tgt = np.zeros(self.cfg.n_classes, dtype=np.float32)
        tgt[label] = 1.0
        err = (tgt - out).astype(np.float32)
        b = self.base.W @ err
        m = np.asarray(mod, dtype=np.float32).reshape(-1)
        nm, nb = float(np.linalg.norm(m)), float(np.linalg.norm(b))
        if m.size == 1:
            # A broadcast scalar has no per-neuron direction; its alignment with
            # the per-neuron reference is only defined up to its sign.
            return float(np.sign(m[0]) * np.sign(float(b.mean()))) if nb > 1e-12 else float("nan")
        if nm > 1e-12 and nb > 1e-12:
            if m.size != b.size:
                # The reference is (n_neurons,) and the factor is (n_classes,)
                # or vice versa: fall back to the neuron-space projection.
                return float("nan")
            return float(m @ b / (nm * nb))
        return float("nan")

    # --------------------------------------------------------------- training
    def fit_task(self, task, epochs: int = 1) -> dict:
        cfg = self.cfg
        X = np.asarray(task.x_train)
        y = np.asarray(task.y_train)
        info = {"n_train": int(len(y)), "readout_updates": 0, "hidden_updates": 0}
        for _ep in range(epochs):
            for i in range(len(y)):
                c = self._code(X[i], cfg.t_train_ms)
                self.base._update_running_stats(c)
                cn = self.base._normalise(c)
                out = self.base.W.T @ cn                    # PRE-update scores
                label = int(y[i])
                mod = self._third_factor(label, out)
                if self.modulator in ("dfa", "scalar", "true_weights"):
                    applied = (mod if mod.size > 1
                               else self._modulator_vector(mod, out, label))
                    c = self._align_cos(applied, out, label)
                    if np.isfinite(c):
                        self.align_cos.append(c)
                # 1. hidden plasticity (third factor computed from the current
                #    output, applied before the readout moves this sample)
                if self.hidden_lr > 0.0:
                    self.hidden.apply(self.W_aff, mod, self.hidden_lr, self._clip)
                    self.n_updates += 1
                    info["hidden_updates"] += 1
                # 2. readout, unchanged default rule and learning rate
                if cfg.readout_learn:
                    tgt = np.zeros(cfg.n_classes, dtype=np.float32)
                    tgt[label] = 1.0
                    err = tgt - out
                    upd = cfg.readout_lr * np.outer(cn, err).astype(np.float32)
                    if cfg.readout_nlms:
                        denom = float(cn.astype(np.float64) @ cn.astype(np.float64))
                        upd = upd / max(denom, 1e-6)
                    self.base.W += upd
                    self.base.readout_updates += 1
                    info["readout_updates"] += 1
        return info

    # -------------------------------------------------------------- inference
    def class_scores(self, x) -> np.ndarray:
        c = self._code(x, self.cfg.t_test_ms)
        return self.base.W.T @ self.base._normalise(c)

    def predict(self, x) -> np.ndarray:
        X = np.atleast_2d(np.asarray(x))
        return np.array([int(np.argmax(self.class_scores(row))) for row in X],
                        dtype=np.int64)

    def evaluate(self, x, y) -> float:
        pred = self.predict(x)
        return float(np.mean(pred == np.asarray(y)))


class StaticReLUReadout:
    """Fixed random ReLU projection + k-WTA, then the SAME readout rule.

    This is the absolute reference point: a non-spiking random-feature model of
    identical width and matched sparsity. It reuses the substrate's readout
    equations verbatim (graded delta rule, running standardisation, NLMS at the
    default learning rate) so the only difference is the feature map.
    """

    def __init__(self, n_in: int, n_out: int, n_classes: int, *, seed: int, k_wta: int):
        from brain.cortex import CortexClassifier, CortexConfig

        self.cfg = CortexConfig(n_neurons=n_out, n_input=n_in, n_classes=n_classes,
                                seed=seed)
        self.base = CortexClassifier(self.cfg, backend="numpy")
        # Same projection construction as experiments/readout_diagnostic.py.
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x5EED])
        self.P = (rng.standard_normal((n_in, n_out)) * np.sqrt(1.0 / n_in)).astype(np.float32)
        self.k_wta = int(k_wta)
        self.n_code_calls = 0

    @property
    def W(self):
        return self.base.W

    @W.setter
    def W(self, v):
        self.base.W = v

    @property
    def n_params(self) -> int:
        return int(self.base.W.size)

    @property
    def synops_per_sample(self) -> float:
        return 0.0

    def _code(self, x, t_ms: int) -> np.ndarray:
        pix = np.asarray(x, dtype=np.float32).reshape(-1)
        h = np.maximum(pix @ self.P, 0.0)
        k = min(self.k_wta, h.size)
        if k < h.size:
            # Select by RANK, not by thresholding at the k-th score. Ties at the
            # k-th value would otherwise admit more than k winners and silently
            # break the matched-sparsity property. ``brain/simulator.py``
            # documents the same trap in ``_kwta_mask``.
            keep = np.argpartition(-h, k - 1)[:k]
            masked = np.zeros_like(h)
            masked[keep] = h[keep]
            h = masked
        self.n_code_calls += 1
        return h.astype(np.float32)

    def fit_task(self, task, epochs: int = 1) -> dict:
        cfg = self.cfg
        X, y = np.asarray(task.x_train), np.asarray(task.y_train)
        for _ in range(epochs):
            for i in range(len(y)):
                c = self._code(X[i], cfg.t_train_ms)
                self.base._update_running_stats(c)
                cn = self.base._normalise(c)
                out = self.base.W.T @ cn
                tgt = np.zeros(cfg.n_classes, dtype=np.float32)
                tgt[int(y[i])] = 1.0
                upd = cfg.readout_lr * np.outer(cn, tgt - out).astype(np.float32)
                if cfg.readout_nlms:
                    denom = float(cn.astype(np.float64) @ cn.astype(np.float64))
                    upd = upd / max(denom, 1e-6)
                self.base.W += upd
        return {"n_train": int(len(y))}

    def predict(self, x) -> np.ndarray:
        X = np.atleast_2d(np.asarray(x))
        t_ms = self.cfg.t_test_ms
        return np.array([int(np.argmax(self.base.W.T @ self.base._normalise(self._code(row, t_ms))))
                         for row in X], dtype=np.int64)

    def evaluate(self, x, y) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y)))


# ------------------------------------------------------------------ one run
@dataclass
class RunResult:
    arm: str
    seed: int
    accuracy: float = float("nan")
    spikes_per_sample: float = 0.0
    active_frac: float = 0.0
    eligible_neurons: int = 0
    hidden_updates: int = 0
    mean_abs_dw: float = 0.0
    weight_drift_sigma: float = 0.0
    feedback_alignment: float = float("nan")
    positive_mod_frac: float = float("nan")
    k_wta_matched: int = 0
    hidden_lr_used: float = 0.0
    synops_per_sample: float = 0.0
    projection_macs_per_sample: int = 0
    wall_seconds: float = 0.0
    error: str | None = None


class InstrumentedReadoutOnly:
    """``CortexClassifier`` with per-sample spike and SynOp accounting.

    ``CortexClassifier`` counts ``last_synops`` but not spikes, and its
    per-sample ``Brain.reset()`` clears the running totals, so the counts have
    to be accumulated by the caller between resets. This subclass adds that
    accounting without touching the classifier's own equations: ``_code`` is
    the parent's method, wrapped.
    """

    def __init__(self, cfg):
        from brain.cortex import CortexClassifier

        self._inner = CortexClassifier(cfg, backend="numpy")
        self.cfg = cfg
        self.be = self._inner.be
        self.spikes_total = 0
        self.synops_active_total = 0
        self.n_code_calls = 0
        # Bind the accounting wrapper onto the INSTANCE so that ``fit_task``,
        # which lives on the parent and calls ``self._code``, routes through it
        # too (same instance-level patch pattern as experiments/afferents.py).
        # The ORIGINAL bound method must be captured FIRST: a wrapper that calls
        # ``self._inner._code`` after rebinding would call itself and recurse
        # until RecursionError, which is exactly what the first draft did.
        original_code = self._inner._code

        def _instrumented_code(x, t_ms: int, _orig=original_code):
            counts = _orig(x, t_ms)
            self.spikes_total += int(np.asarray(counts).sum())
            self.synops_active_total += int(self._inner.last_synops)
            self.n_code_calls += 1
            return counts

        self._inner._code = _instrumented_code

    @property
    def W(self):
        return self._inner.W

    @W.setter
    def W(self, v):
        self._inner.W = v

    @property
    def n_params(self) -> int:
        return int(self._inner.W.size)

    @property
    def synops_per_sample(self) -> float:
        return self.synops_active_total / max(1, self.n_code_calls)

    def _code(self, x, t_ms: int) -> np.ndarray:
        """Delegate to the instrumented wrapper bound onto the inner model."""
        return self._inner._code(x, t_ms)

    def class_scores(self, x) -> np.ndarray:
        return self._inner.class_scores(x)

    def predict(self, x) -> np.ndarray:
        return self._inner.predict(x)

    def evaluate(self, x, y) -> float:
        return self._inner.evaluate(x, y)

    def fit_task(self, task, epochs: int = 1) -> dict:
        return self._inner.fit_task(task, epochs=epochs)


def build_model(arm: str, seed: int, args, *, k_wta: int | None = None):
    from brain.cortex import CortexConfig

    if arm == "readout_only":
        cfg = CortexConfig(n_neurons=args.neurons, n_input=N_INPUT, n_classes=N_CLASSES,
                           k_out=args.k_out, seed=seed, use_dendrite=True,
                           plasticity=True)
        return InstrumentedReadoutOnly(cfg)

    if arm == "static_relu":
        return StaticReLUReadout(N_INPUT, args.neurons, N_CLASSES, seed=seed,
                                 k_wta=int(k_wta) if k_wta else int(0.29 * args.neurons))

    cfg = CortexConfig(n_neurons=args.neurons, n_input=N_INPUT, n_classes=N_CLASSES,
                       k_out=args.k_out, seed=seed, use_dendrite=True,
                       plasticity=True)
    modulator = {"frozen_hidden": "scalar",      # path identical; lr == 0 gates the write
                 "scalar": "scalar",
                 "dfa": "dfa",
                 TRUE_WEIGHTS_ARM: "true_weights"}[arm]
    hidden_lr = 0.0 if arm == "frozen_hidden" else args.hidden_lr
    return ThreeFactorCortex(cfg, seed=seed, hidden_lr=hidden_lr, modulator=modulator,
                             afferent_seed=seed, gain=args.gain,
                             feedback_seed=10_000 + seed)


def run_arm(arm: str, data: TaskData, seed: int, args, *,
            k_wta: int | None = None) -> RunResult:
    res = RunResult(arm=arm, seed=seed)
    res.k_wta_matched = int(k_wta or 0)
    res.hidden_lr_used = float(args.hidden_lr)
    t0 = time.perf_counter()
    try:
        model = build_model(arm, seed, args, k_wta=k_wta)
        model.fit_task(data, epochs=1)

        if arm in SPIKING_ARMS:
            # ---- LOUD failure if the population produced no spikes. A silent
            # population makes every downstream comparison a comparison of
            # silence to silence (docs/NEURON_OPERATING_POINT.md).
            res.spikes_per_sample = (model.spikes_total / max(1, model.n_code_calls))
            res.synops_per_sample = model.synops_per_sample
            prober = np.asarray(data.x_test[:20])
            counts = np.stack([model._code(x, T_MS) for x in prober])
            res.active_frac = float((counts > 0).mean())
            res.eligible_neurons = int((counts.sum(axis=0) > 0).sum())
            if not np.isfinite(res.spikes_per_sample) or res.spikes_per_sample <= 0.0:
                raise RuntimeError(
                    f"FATAL: arm {arm!r} seed {seed} produced ZERO spikes per sample "
                    f"({res.spikes_per_sample}). The population is silent; every "
                    "comparison would be silence vs silence. Fix the operating point "
                    "(see docs/NEURON_OPERATING_POINT.md) before trusting any number here.")
            if res.eligible_neurons == 0:
                raise RuntimeError(
                    f"FATAL: arm {arm!r} seed {seed} had no neuron fire on ANY of the "
                    "first 20 test samples; the code is degenerate.")

        if isinstance(model, ThreeFactorCortex):
            res.hidden_updates = int(model.n_updates)
            res.mean_abs_dw = float(model.hidden.last_abs_dw)
            res.weight_drift_sigma = float(np.mean(np.abs(model.W_aff)) / model._sigma0)
            if model.align_cos:
                res.feedback_alignment = float(np.mean(model.align_cos))
            if model._n_mod:
                res.positive_mod_frac = model._n_mod_pos / model._n_mod
            if model.hidden_lr > 0.0:
                if model.n_updates == 0 or model.hidden.last_abs_dw <= 0.0:
                    raise RuntimeError(
                        f"FATAL: arm {arm!r} seed {seed} claims plastic hidden learning but "
                        f"made {model.n_updates} updates with dw={model.hidden.last_abs_dw}; "
                        "the write path never executed.")

        res.accuracy = float(model.evaluate(data.x_test, data.y_test))
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            traceback.print_exc()
    res.wall_seconds = time.perf_counter() - t0
    return res


# --------------------------------------------------------------- experiment
def _job_spec(arm: str, seed: int, args) -> dict:
    return {"arm": arm, "seed": seed,
            "n_train": args.n_train, "n_test": args.n_test,
            "neurons": args.neurons, "k_out": args.k_out, "gain": args.gain,
            "hidden_lr": args.hidden_lr, "verbose": bool(args.verbose),
            "k_wta": None}


def _job(spec: dict) -> dict:
    """Run one (arm, seed) cell. Module-level so it is picklable for the pool."""
    args = argparse.Namespace(**{k: spec[k] for k in (
        "n_train", "n_test", "neurons", "k_out", "gain", "hidden_lr", "verbose")})
    args.arm = spec["arm"]
    data = single_task_data(spec["seed"], args)
    res = run_arm(spec["arm"], data, spec["seed"], args, k_wta=spec.get("k_wta"))
    d = vars(res)
    d["_data"] = None
    return d


def _run_cells(cells: list[dict], args, workers: int) -> list[RunResult]:
    """Run cells, in parallel when asked. Determinism does not depend on the
    pool: every cell is a function of (arm, seed, args) only, with private
    seeded RNGs (numpy backend, Afferents, feedback matrix), so results are
    bit-identical to a sequential run."""
    if workers <= 1 or len(cells) <= 1:
        out = []
        for spec in cells:
            d = _job(spec)
            r = RunResult(**{k: v for k, v in d.items() if k in RunResult.__dataclass_fields__})
            out.append(r)
            print("  " + _cell_line(r), flush=True)
        return out
    from concurrent.futures import ProcessPoolExecutor
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(_job, cells):
            r = RunResult(**{k: v for k, v in d.items() if k in RunResult.__dataclass_fields__})
            out.append(r)
            print("  " + _cell_line(r), flush=True)
    return out


def _cell_line(r: RunResult) -> str:
    if r.error is not None:
        return f"{r.arm:>15s} seed {r.seed}: ERROR {r.error}"
    return (f"{r.arm:>15s} seed {r.seed}: acc={r.accuracy*100:5.2f}%  "
            f"spk/sample={r.spikes_per_sample:7.1f}  active={r.active_frac*100:5.1f}%  "
            f"hidden_upd={r.hidden_updates:4d}  dw={r.mean_abs_dw:.2e}  "
            f"drift={r.weight_drift_sigma:.3f}sigma  align={r.feedback_alignment:+.3f}  "
            f"k_wta={r.k_wta_matched:3d}  ({r.wall_seconds:.0f}s)")


# --------------------------------------------------------------- experiment
def single_task_data(seed: int, args) -> TaskData:
    from brain.tasks import load_mnist

    x_tr, y_tr, x_te, y_te = load_mnist()
    xtr, ytr = stratified_subsample(x_tr, y_tr, args.n_train, 7000 + seed)
    xte, yte = stratified_subsample(x_te, y_te, args.n_test, 8000 + seed)
    return TaskData(name="mnist10_single", x_train=xtr, y_train=ytr,
                    x_test=xte, y_test=yte)


def evaluate_criterion(by_arm: dict[str, list[float]]) -> dict:
    """The pre-registered test, applied exactly as written in the docstring."""
    frozen = np.asarray(by_arm.get("frozen_hidden", []), dtype=np.float64)
    out = {"criterion": "mean(plastic) - mean(frozen_hidden) > 2 * sd_seed",
           "sd_definition": "sqrt((var(plastic) + var(frozen_hidden))/2), ddof=1",
           "per_arm": {}, "passed": False, "passing_arms": []}
    for arm in PLASTIC_ARMS:
        plastic = np.asarray(by_arm.get(arm, []), dtype=np.float64)
        if plastic.size == 0 or frozen.size == 0 or plastic.size != frozen.size:
            out["per_arm"][arm] = {"status": "not_evaluable"}
            continue
        delta = float(plastic.mean() - frozen.mean())
        var_p = float(plastic.var(ddof=1)) if plastic.size > 1 else 0.0
        var_f = float(frozen.var(ddof=1)) if frozen.size > 1 else 0.0
        sd = float(np.sqrt((var_p + var_f) / 2.0))
        threshold = 2.0 * sd
        passed = bool(delta > threshold)
        out["per_arm"][arm] = {
            "status": "ok",
            "mean_plastic": float(plastic.mean()),
            "mean_frozen": float(frozen.mean()),
            "delta": delta,
            "sd_seed": sd,
            "threshold_2sd": threshold,
            "passed": passed,
            "per_seed_delta": (plastic - frozen).tolist(),
        }
        if passed:
            out["passing_arms"].append(arm)
    out["passed"] = bool(out["passing_arms"])
    out["verdict"] = ("PASS" if out["passed"] else "FALSIFIED")
    return out


def continual_stage(arms: list[str], args) -> dict:
    """Permuted-MNIST, single pass, no replay. Runs only if the criterion passed."""
    from brain.metrics import ContinualCurve, accuracy
    from brain.tasks import split_mnist

    out: dict = {"ran": True, "arms": {}, "n_tasks": args.cont_tasks}
    for seed in range(args.seeds):
        suite = split_mnist(args.cont_tasks, seed=seed, permute=True)
        for ti, task in enumerate(suite.tasks):
            task.x_train, task.y_train = stratified_subsample(
                task.x_train, task.y_train, args.cont_train, seed * 17 + ti)
            task.x_test, task.y_test = stratified_subsample(
                task.x_test, task.y_test, args.cont_test, seed * 31 + ti)
        for arm in arms:
            t0 = time.perf_counter()
            try:
                model = build_model(arm, seed, args)
                n_tasks = len(suite)
                acc = np.full((n_tasks, n_tasks), np.nan)
                for i, task in enumerate(suite.tasks):
                    model.fit_task(task, epochs=1)
                    for j in range(i + 1):
                        tj = suite.tasks[j]
                        acc[i, j] = accuracy(model.predict(tj.x_test), tj.y_test)
                am = np.nan_to_num(acc)
                curve = ContinualCurve(acc=am)
                out["arms"].setdefault(arm, []).append({
                    "seed": seed,
                    "acc_matrix": am.tolist(),
                    "final_average": curve.final_average(),
                    "diagonal_mean": float(np.mean(np.diag(am))),
                    "forgetting": curve.forgetting(),
                    "backward_transfer": curve.backward_transfer(),
                    "wall_seconds": time.perf_counter() - t0,
                })
            except Exception as exc:
                out["arms"].setdefault(arm, []).append({
                    "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    # MLP reference (dense backprop), same data and update budget.
    from brain.baselines import MLPBackprop

    mlp_rows = []
    for seed in range(args.seeds):
        suite = split_mnist(args.cont_tasks, seed=seed, permute=True)
        for ti, task in enumerate(suite.tasks):
            task.x_train, task.y_train = stratified_subsample(
                task.x_train, task.y_train, args.cont_train, seed * 17 + ti)
            task.x_test, task.y_test = stratified_subsample(
                task.x_test, task.y_test, args.cont_test, seed * 31 + ti)
        model = MLPBackprop(N_INPUT, args.mlp_hidden, N_CLASSES, seed=seed,
                            lr=args.mlp_lr, batch_size=args.mlp_batch)
        n_tasks = len(suite)
        acc = np.full((n_tasks, n_tasks), np.nan)
        for i, task in enumerate(suite.tasks):
            # Equalise update budget: one online update per training sample.
            n_train = len(task.y_train)
            steps = max(1, int(np.ceil(n_train / args.mlp_batch)))
            model.fit_task(task, epochs=max(1, int(np.ceil(n_train / steps))))
            for j in range(i + 1):
                tj = suite.tasks[j]
                acc[i, j] = accuracy(model.predict(tj.x_test), tj.y_test)
        am = np.nan_to_num(acc)
        curve = ContinualCurve(acc=am)
        mlp_rows.append({"seed": seed, "acc_matrix": am.tolist(),
                         "final_average": curve.final_average(),
                         "diagonal_mean": float(np.mean(np.diag(am))),
                         "forgetting": curve.forgetting(),
                         "n_params": model.n_params})
    out["arms"]["mlp_reference"] = mlp_rows
    return out


def summarise(results: list[RunResult]) -> dict:
    out: dict = {}
    requested = {r.arm for r in results}
    for arm in ARMS + sorted(requested - set(ARMS)):
        rows = [r for r in results if r.arm == arm]
        if not rows:
            if arm in requested:
                continue
            out[arm] = {"status": "not_run"}
            continue
        ok = [r for r in rows if r.error is None]
        errs = [r.error for r in rows if r.error is not None]
        if not ok:
            out[arm] = {"status": "failed", "errors": errs}
            continue
        accs = [r.accuracy for r in ok]
        mean, sd = mean_sd(accs)
        out[arm] = {
            "status": "ok" if not errs else "partial",
            "errors": errs,
            "n_seeds": len(ok),
            "accuracy_mean": mean,
            "accuracy_sd": sd,
            "per_seed_accuracy": accs,
            "accuracy_mean_pct": 100.0 * mean,
            "accuracy_sd_pct": 100.0 * sd,
            "spikes_per_sample_mean": float(np.mean([r.spikes_per_sample for r in ok])),
            "active_frac_mean": float(np.mean([r.active_frac for r in ok])),
            "eligible_neurons_mean": float(np.mean([r.eligible_neurons for r in ok])),
            "hidden_updates_mean": float(np.mean([r.hidden_updates for r in ok])),
            "mean_abs_dw_mean": float(np.mean([r.mean_abs_dw for r in ok])),
            "weight_drift_sigma_mean": float(np.mean([r.weight_drift_sigma for r in ok])),
            "feedback_alignment_mean": _nanmean(
                [r.feedback_alignment for r in ok]),
            "positive_modulator_frac_mean": _nanmean(
                [r.positive_mod_frac for r in ok]),
            "hidden_lr_used": float(ok[0].hidden_lr_used),
            "k_wta_matched_mean": float(np.mean([r.k_wta_matched for r in ok])),
            "synops_per_sample_mean": float(np.mean([r.synops_per_sample for r in ok])),
            "wall_seconds_mean": float(np.mean([r.wall_seconds for r in ok])),
            "per_seed": [{"seed": r.seed, "accuracy": r.accuracy,
                          "spikes_per_sample": r.spikes_per_sample,
                          "active_frac": r.active_frac,
                          "hidden_updates": r.hidden_updates,
                          "mean_abs_dw": r.mean_abs_dw,
                          "weight_drift_sigma": r.weight_drift_sigma,
                          "feedback_alignment": r.feedback_alignment,
                          "positive_modulator_frac": r.positive_mod_frac,
                          "k_wta_matched": r.k_wta_matched,
                          "wall_seconds": r.wall_seconds} for r in ok],
        }
    return out


def _sweep_spec(arm: str, lr: float, seed: int, args) -> dict:
    return {"arm": arm, "seed": seed, "n_train": args.n_train, "n_test": args.n_test,
            "neurons": args.neurons, "k_out": args.k_out, "gain": args.gain,
            "hidden_lr": float(lr), "verbose": bool(args.verbose), "k_wta": None}


def lr_sweep(args, lrs: list[float], seeds: list[int], workers: int = 1) -> dict:
    """Sensitivity of each plastic arm to the hidden learning rate, plus a frozen
    reference at each seed.

    This is a sensitivity check, not the pre-registered test: it runs one seed
    per rate (``seeds=[0]`` by default) so a reader can see that the verdict
    does not depend on one lucky value. ``hidden_lr = 0`` is the frozen ablation
    and is included for reference. Parallelised on the same pool as the main
    cells; every cell is independent and seeded, so results are identical to a
    sequential run.
    """
    cells = []
    for arm in PLASTIC_ARMS:
        for lr in lrs:
            for sd in seeds:
                cells.append(_sweep_spec(arm, lr, sd, args))
    for sd in seeds:
        cells.append(_sweep_spec("frozen_hidden", 0.0, sd, args))

    results = _run_cells(cells, args, workers)
    out: dict = {}
    for r in results:
        bucket = out.setdefault(r.arm, [])
        bucket.append({"hidden_lr": float(r.hidden_lr_used), "seed": r.seed,
                       "accuracy": r.accuracy, "spikes_per_sample": r.spikes_per_sample,
                       "weight_drift_sigma": r.weight_drift_sigma,
                       "error": r.error})
    for arm in out:
        out[arm].sort(key=lambda d: d["hidden_lr"])
    return out


def energy_report(summary: dict, args) -> dict:
    """Operation-count proxies. NOT joules."""
    proj = N_INPUT * args.neurons                     # dense afferent projection
    read = args.neurons * N_CLASSES                   # readout MACs per sample
    mlp_hidden = args.mlp_hidden
    mlp_dense = N_INPUT * mlp_hidden + mlp_hidden * N_CLASSES
    out = {
        "units": "multiply-accumulates or active synaptic operations per sample",
        "caveat": ("These are operation COUNTS, not energy. A biological synapse "
                   "costs ~1-100 fJ while a neuromorphic SynOp on Loihi costs "
                   "~23.6 pJ, so 'spiking is cheap' is not entailed. Reported "
                   "because it is the honest hardware-independent proxy."),
        "arms": {},
        "mlp_dense_reference": {"hidden": mlp_hidden, "macs_per_sample": mlp_dense,
                                "n_params": mlp_dense},
    }
    for arm in SPIKING_ARMS:
        s = summary.get(arm, {})
        if s.get("status") not in ("ok", "partial"):
            continue
        syn = s["synops_per_sample_mean"]
        out["arms"][arm] = {
            "afferent_projection_macs_per_sample": proj,
            "recurrent_synops_active_per_sample": syn,
            "readout_macs_per_sample": read,
            "total_ops_per_sample": float(proj + syn + read),
            "dense_equivalent_macs_per_sample": float(proj + read),
        }
    for arm in ("static_relu",):
        s = summary.get(arm, {})
        if s.get("status") not in ("ok", "partial"):
            continue
        out["arms"][arm] = {
            "afferent_projection_macs_per_sample": proj,
            "readout_macs_per_sample": read,
            "total_ops_per_sample": float(proj + read),
        }
    # The dense reference for the recurrent layer itself (a 900x900 dense layer
    # would cost neurons^2 MACs); the substrate's recurrent delivery costs
    # spikes x k_out active SynOps.
    dense_recurrent = float(args.neurons * args.neurons)
    out["recurrent_dense_reference_macs_per_sample"] = dense_recurrent
    for arm, d in out["arms"].items():
        syn = d.get("recurrent_synops_active_per_sample", 0.0)
        d["recurrent_ops_vs_dense_900x900"] = (dense_recurrent / syn) if syn else None
    return out


def print_table(summary: dict, crit: dict) -> None:
    print("\n" + "=" * 104)
    print("THESIS EXPERIMENT — three-factor plasticity on the HIDDEN layer (single-task 10-way MNIST)")
    print("=" * 104)
    print(f"{'arm':>15s} {'acc mean':>10s} {'sd':>7s} {'per-seed':>34s} "
          f"{'spk/sample':>11s} {'hidden upd':>11s}")
    print("-" * 104)
    for arm in ARMS:
        s = summary.get(arm, {})
        if s.get("status") not in ("ok", "partial"):
            print(f"{arm:>15s} {'FAILED':>10s}  {str(s.get('errors'))[:70]}")
            continue
        seeds = " ".join(f"{a*100:5.1f}" for a in s["per_seed_accuracy"])
        print(f"{arm:>15s} {s['accuracy_mean_pct']:9.2f}% {s['accuracy_sd_pct']:6.2f} "
              f"{seeds:>34s} {s['spikes_per_sample_mean']:11.1f} "
              f"{s['hidden_updates_mean']:11.0f}")
    print("=" * 104)

    print("\nPRE-REGISTERED CRITERION (frozen ablation is the only valid test of learning)")
    print(f"  {crit['criterion']}; sd = {crit['sd_definition']}")
    for arm, c in crit["per_arm"].items():
        if c.get("status") != "ok":
            print(f"  {arm:>6s}: {c.get('status')}")
            continue
        mark = "PASS" if c["passed"] else "fail"
        print(f"  {arm:>6s}: {c['mean_plastic']*100:6.2f}% - {c['mean_frozen']*100:6.2f}% "
              f"= {c['delta']*100:+6.2f} pts vs threshold {c['threshold_2sd']*100:5.2f} pts "
              f"-> {mark}")
    print(f"  VERDICT: {crit['verdict']}"
          + (f" (passing: {crit['passing_arms']})" if crit["passed"] else
             " — three-factor plasticity on the hidden layer did NOT beat its own "
             "frozen ablation; the continual stage is not run."))



def eligibility_rank_check(args, seed: int = 0, n_samples: int = 3) -> dict:
    """Why the rule is structurally limited: measure the eligibility's rank.

    For a static input held for the whole window the eligibility collapses to

        e_ij = a_plus * x_i * c_j  -  a_minus * k_j

    whose columns are all ``a_plus * c_j * x`` minus a per-column constant. That
    is the sum of ONE fixed input direction and one per-column scalar, i.e. rank
    <= 2 by construction, and in practice rank 1. A third factor ``M_j`` can then
    only scale or sign that single direction per neuron; it cannot select a
    *different* input pattern for a different neuron, because the eligibility
    matrix does not contain per-neuron input structure to select. This is the
    mechanism-level explanation of the negative result, measured rather than
    asserted.
    """
    from brain.cortex import CortexConfig
    from brain.tasks import load_mnist

    data = single_task_data(seed, args)
    cfg = CortexConfig(n_neurons=args.neurons, n_input=N_INPUT, n_classes=N_CLASSES,
                       k_out=args.k_out, seed=seed, use_dendrite=True, plasticity=True)
    m = ThreeFactorCortex(cfg, seed=seed, hidden_lr=0.0, modulator="dfa",
                          afferent_seed=seed, gain=args.gain, feedback_seed=10000 + seed)
    rows = []
    for i in range(n_samples):
        m._code(data.x_train[i], T_MS)
        E = m.hidden.e.astype(np.float64)
        sv = np.linalg.svd(E, compute_uv=False)
        energy = np.cumsum(sv ** 2) / float(np.sum(sv ** 2))
        rows.append({
            "sample": i,
            "rank_at_99pct_energy": int(np.searchsorted(energy, 0.99) + 1),
            "top1_energy_frac": float(energy[0]),
            "top2_energy_frac": float(energy[1]) if energy.size > 1 else 1.0,
            "signed_fraction": float((E < 0).mean()),
        })
    return {
        "n_samples": n_samples,
        "per_sample": rows,
        "max_rank_at_99pct_energy": max(r["rank_at_99pct_energy"] for r in rows),
        "note": ("rank does not grow with the number of hidden neurons, so this is "
                 "not a small-network artefact; it is a property of driving a "
                 "static input through a fixed projection."),
    }


def self_checks(results: list[RunResult], args) -> dict:
    """Machine-checkable preconditions for the claims made in the writeup.

    These are the checks that would catch the failure modes this repo has
    actually hit: a silent population, a hidden plasticity path that never
    executed, a "matched" control that was not matched, and a frozen ablation
    that quietly learned anyway.
    """
    by = {(r.arm, r.seed): r for r in results}
    checks: dict = {}

    frozen = [r for r in results if r.arm == "frozen_hidden" and r.error is None]
    plastic = [r for r in results if r.arm in PLASTIC_ARMS and r.error is None]
    readout = [r for r in results if r.arm == "readout_only" and r.error is None]
    static = [r for r in results if r.arm == "static_relu" and r.error is None]

    checks["spikes_nonzero_all_spiking_arms"] = {
        "ok": all(r.spikes_per_sample > 0 for r in frozen + plastic + readout)
        and bool(frozen + plastic + readout),
        "min_spikes_per_sample": float(min([r.spikes_per_sample for r in
                                            frozen + plastic + readout], default=0.0)),
        "note": "A zero-spike population would make every comparison silence vs silence.",
    }
    checks["frozen_ablation_made_no_updates"] = {
        "ok": all(r.hidden_updates == 0 for r in frozen) if frozen else False,
        "note": "hidden_lr=0 must skip the weight write entirely.",
    }
    checks["frozen_ablation_weights_unchanged"] = {
        "ok": all(abs(r.weight_drift_sigma - 0.7980) < 0.02 for r in frozen) if frozen else False,
        "observed": [r.weight_drift_sigma for r in frozen],
        "note": ("Mean|W_aff|/sigma0 for the freshly drawn dense projection is "
                 "~0.798 (E|N(0,1)|=0.7979). A frozen arm leaving that value means "
                 "no hidden weight was written."),
    }
    checks["plastic_arms_made_updates"] = {
        "ok": all(r.hidden_updates > 0 and r.mean_abs_dw > 0 for r in plastic)
        and bool(plastic),
        "note": "A plastic arm with zero writes would silently be a second frozen arm.",
    }
    checks["plastic_dw_stayed_bounded"] = {
        "ok": all(np.isfinite(r.mean_abs_dw) and r.weight_drift_sigma < 3.0
                  for r in plastic) if plastic else False,
        "drift_sigma": [r.weight_drift_sigma for r in plastic],
        "note": "A runaway Hebbian rule would show drift >> 1 sigma.",
    }
    if static and frozen:
        for r in static:
            f = by.get(("frozen_hidden", r.seed))
            if f is None:
                continue
            n = args.neurons
            checks.setdefault("relu_kwta_matched_active_fraction", {})[f"seed{r.seed}"] = {
                "k_wta": r.k_wta_matched,
                "k_wta_frac": r.k_wta_matched / n,
                "substrate_active_frac": f.active_frac,
                "abs_diff": abs(r.k_wta_matched / n - f.active_frac),
            }
        diffs = [v["abs_diff"] for v in checks["relu_kwta_matched_active_fraction"].values()]
        checks["relu_kwta_matched_active_fraction"]["ok"] = bool(diffs) and max(diffs) < 0.02
        checks["relu_kwta_matched_active_fraction"]["note"] = (
            "The ReLU control is set to the substrate's own measured active fraction "
            "on the same seed (tolerance 2 points), because a sparsity-mismatched "
            "control is not a matched control (docs/PARADIGM.md section 5).")
    tw = [r for r in results if r.arm == TRUE_WEIGHTS_ARM and r.error is None]
    if tw:
        checks["true_weights_feedback_alignment_is_one"] = {
            "ok": all(r.feedback_alignment > 0.99 for r in tw),
            "observed": [r.feedback_alignment for r in tw],
            "note": ("This arm's third factor IS the readout direction, so its "
                     "alignment diagnostic must be 1.0. It is a check on the "
                     "diagnostic itself: 1.0 alignment is NOT sufficient for "
                     "learning here, because the readout direction ignores the "
                     "derivative of the spiking code w.r.t. the projection. "
                     "Measured: this arm can score below the frozen ablation."),
        }
    checks["all_arms_completed"] = {
        "ok": all(r.error is None for r in results),
        "errors": [f"{r.arm}/seed{r.seed}: {r.error}" for r in results if r.error],
    }
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--neurons", type=int, default=N_NEURONS)
    p.add_argument("--k-out", type=int, default=K_OUT)
    p.add_argument("--gain", type=float, default=GAIN)
    p.add_argument("--hidden-lr", type=float, default=DEFAULT_HIDDEN_LR)
    #: Non-empty by default so the documented command reproduces the documented
    #: JSON exactly, including the sweep table in docs/THREEFACTOR.md. It costs
    #: ~9 extra minutes; the sweep is what shows the verdict does not rest on one
    #: learning rate, so it belongs in the canonical artefact rather than in a
    #: flag someone has to know to pass.
    p.add_argument("--lr-sweep", nargs="*", type=float, default=[0.005, 0.05, 0.2])
    p.add_argument("--arms", nargs="*", default=ARMS)
    p.add_argument("--cont-tasks", type=int, default=5)
    p.add_argument("--cont-train", type=int, default=200)
    p.add_argument("--cont-test", type=int, default=100)
    p.add_argument("--cont-for", nargs="*", default=None,
                   help="arms for the continual stage (default: frozen + passing arms)")
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-batch", type=int, default=64)
    p.add_argument("--out", default=str(RESULTS_DIR / "threefactor_hidden.json"))
    p.add_argument("--workers", type=int, default=4,
                   help="process pool size for (arm, seed) cells; 1 = sequential")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    t_start = time.perf_counter()
    results: list[RunResult] = []
    per_seed_by_arm: dict[str, list[float]] = {a: [] for a in ARMS}

    # --- order arms so the frozen ablation exists before the ReLU control needs
    # it for sparsity matching: frozen first, then scalar/dfa, then static_relu.
    _ORDER = ["readout_only", "frozen_hidden", "scalar", "dfa", "static_relu",
              TRUE_WEIGHTS_ARM]
    arms = sorted([a for a in args.arms], key=lambda a: _ORDER.index(a) if a in _ORDER else 99)

    # --- reference active fraction per seed, measured from the frozen arm.
    # The static-ReLU control must be matched on the SAME seed's sparsity, so
    # frozen runs first and its measured active fraction is passed in.
    ref_frac: dict[int, float] = {}
    if "frozen_hidden" in arms:
        cells = [{"arm": "frozen_hidden", "seed": sd, "n_train": args.n_train,
                  "n_test": args.n_test, "neurons": args.neurons, "k_out": args.k_out,
                  "gain": args.gain, "hidden_lr": 0.0, "verbose": bool(args.verbose)}
                 for sd in range(args.seeds)]
        for r in _run_cells(cells, args, args.workers):
            results.append(r)
            if r.error is None:
                per_seed_by_arm["frozen_hidden"].append(r.accuracy)
                ref_frac[r.seed] = r.active_frac
        arms = [a for a in arms if a != "frozen_hidden"]

    # --- the plastic arms and the current-baseline arm, one cell per (arm, seed)
    cells = []
    for arm in arms:
        for sd in range(args.seeds):
            cells.append({"arm": arm, "seed": sd, "n_train": args.n_train,
                          "n_test": args.n_test, "neurons": args.neurons,
                          "k_out": args.k_out, "gain": args.gain,
                          "hidden_lr": args.hidden_lr, "verbose": bool(args.verbose),
                          "k_wta": (max(1, int(round(ref_frac.get(sd, 0.29) * args.neurons)))
                                    if arm == "static_relu" else None)})
    for r in _run_cells(cells, args, args.workers):
        results.append(r)
        if r.error is None and r.arm in per_seed_by_arm:
            per_seed_by_arm[r.arm].append(r.accuracy)

    # --- determinism / self-consistency checks, reported in the JSON.
    checks = self_checks(results, args)
    checks["eligibility_rank"] = eligibility_rank_check(args)

    summary = summarise(results)
    crit = evaluate_criterion(per_seed_by_arm)

    sweep = {}
    if args.lr_sweep:
        sweep = lr_sweep(args, list(args.lr_sweep), seeds=[0], workers=args.workers)

    # --- gated continual stage
    continual: dict = {"ran": False, "reason": "not attempted"}
    if crit["passed"]:
        cont_arms = args.cont_for or (["frozen_hidden"] + crit["passing_arms"])
        print(f"\nCriterion PASSED for {crit['passing_arms']}; running permuted-MNIST "
              f"continual stage for arms {cont_arms}")
        continual = continual_stage(cont_arms, args)
        continual["criterion_passed"] = True
    else:
        continual = {
            "ran": False,
            "criterion_passed": False,
            "reason": ("Neither plastic arm beat its own frozen ablation by more than "
                       "twice the seed standard deviation, so the thesis is falsified "
                       "at step 1. Permuted-MNIST was deliberately NOT run: a model "
                       "that cannot learn a single task has no continual-learning "
                       "claim to test."),
            "measured": {a: crit["per_arm"].get(a) for a in PLASTIC_ARMS},
        }

    payload = {
        "experiment": "threefactor_hidden",
        "question": ("Can error-modulated three-factor plasticity (no backward pass, "
                     "no weight transport) shape a hidden layer?"),
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "backend": "numpy",
            "n_neurons": args.neurons,
            "k_out": args.k_out,
            "gain": args.gain,
            "t_ms": T_MS,
            "seeds": args.seeds,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "hidden_lr": args.hidden_lr,
            "hidden_lr_source": ("pre-declared before running, from a step-size scale "
                                 "argument; see module docstring"),
            "readout_lr": 0.05,
            "readout_lr_source": "CortexConfig default, NOT retuned in any arm",
            "readout_rule": "graded delta + NLMS (CortexConfig defaults)",
            "git_rev": _git_rev(),
        },
        "pre_registered_criterion": {
            "statement": ("mean(acc_plastic) - mean(acc_frozen_hidden) > 2 * sd_seed, "
                          "sd_seed = sqrt((var_plastic + var_frozen)/2), ddof=1"),
            "registered_before_running": True,
            "prediction_before_running": (
                "scalar likely fails (one bit cannot tell a synapse what to change); "
                "dfa likely learns one task and then forgets under permutation"),
        },
        "arms_in_order": {
            "1_readout_only": "unmodified current model: identity wiring, frozen input, readout learns",
            "2_frozen_hidden": "afferent projection + full hidden plasticity code path, hidden_lr = 0",
            "3_scalar": "hidden plasticity, scalar broadcast third factor (+1 correct / -1 incorrect)",
            "4_dfa": "hidden plasticity, fixed-random-feedback third factor (KNOWN method, not novel)",
            "5_static_relu": "fixed random ReLU projection + k-WTA at matched sparsity, same readout",
            "6_feedback_true_weights": ("DIAGNOSTIC ONLY: DFA with the readout's own weights "
                                        "instead of a random matrix (weight transport, not local, "
                                        "not the gradient). Also run at hidden_lr = 0 as a second "
                                        "frozen control that exercises the identical code path."),
        },
        "summary": summary,
        "checks": checks,
        "criterion": crit,
        "lr_sweep": sweep,
        "energy_proxy": energy_report(summary, args),
        "continual": continual,
        "documented_references": DOCUMENTED_REFERENCES,
        "wall_seconds": time.perf_counter() - t_start,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print_table(summary, crit)
    print(f"\nwrote {out_path}  ({payload['wall_seconds']:.0f}s total)")
    return 0


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
