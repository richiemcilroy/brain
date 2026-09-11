"""A spiking classifier whose learning is local end-to-end.

What does the learning
----------------------
Two mechanisms, neither of which is backpropagation:

1. **Substrate self-organisation.** The recurrent spiking network runs with
   three-factor plasticity enabled and a constant neuromodulator (M = 1), i.e.
   pure unsupervised Hebbian learning. Its sparse code is the representation.
2. **Error-modulated readout.** A weight matrix ``W`` maps the sparse code to
   class scores and is trained by::

       dW_ij = lr * M_j * x_i        with     M_j = target_j - output_j

   That is the *same* three-factor form used everywhere else in this repo -
   a local eligibility term ``x_i`` multiplied by a broadcast third factor -
   where the third factor is a per-class error signal instead of a global
   reward. It is the delta rule re-read as neuromodulated plasticity. There is
   no backward pass through the network and no weight transport.

Note on fidelity. A deliberate deviation from a teacher-forced label-neuron
design is worth recording, because it was settled by measurement: label neurons
wired with the standard fixed fan-out received a median of **2** synapses from
the driven input population (out of 784 possible), which is far too sparse to
ever drive them at test time when the teacher current is removed. The
error-modulated readout above learns robustly where that design did not.

Honest limitation
-----------------
The readout is linear in the sparse code. A reviewer should therefore treat the
``frozen`` arm (fixed random projection + ridge readout) as the control that
matters: if a random projection matches this, the substrate's self-organisation
is not contributing and the claim collapses. That arm exists precisely to test
this, and ``brain_noplast`` isolates the contribution of mechanism 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import Backend, get_backend
from .neurons import NeuronConfig
from .plasticity import PlasticityConfig
from .simulator import Brain, SimConfig


@dataclass
class CortexConfig:
    n_neurons: int = 600
    n_input: int = 784
    n_classes: int = 10
    k_out: int = 48
    t_train_ms: int = 15
    t_test_ms: int = 15
    gain: float = 2.2
    dend_scale: float = 1.0
    dend_gain: float = 2.5
    readout_lr: float = 0.05
    readout_epochs: int = 15
    w_init: float = 0.0
    batch: int = 1
    seed: int = 0
    use_dendrite: bool = True
    plasticity: bool = True
    #: Whether the READOUT learns. Deliberately separate from ``plasticity``:
    #: an earlier version gated the readout update on the same flag, so the
    #: ``brain_noplast`` arm removed *all* learning, not substrate plasticity.
    #: That made it score exactly chance with zero variance, which was written
    #: up as "plasticity is causal" when it actually tested nothing about the
    #: substrate. Use ``plasticity=False, readout_learn=True`` to isolate
    #: substrate plasticity properly.
    readout_learn: bool = True
    #: Local readout rule. ``"binary"`` is the ORIGINAL behaviour
    #: (``err = target - (out > 0)``), which binarises the output so a class
    #: already scoring positive receives exactly zero error and stops learning.
    #: ``"delta"`` is the graded rule the docstring always claimed to implement.
    readout_rule: str = "delta"
    #: Reset the network between samples. Without this the adaptation state
    #: (tau ~100 ms) carries across ~7 samples, so every code is partly a
    #: function of the *previous* image - a leak that contaminated every
    #: measurement in the repo.
    reset_between_samples: bool = True
    inhibition: str = "none"
    k_wta: int = 0
    spike_capacity_frac: float = 0.30
    normalize_code: bool = True

    def __post_init__(self) -> None:
        if self.n_input > self.n_neurons:
            raise ValueError("n_input cannot exceed n_neurons")
        if self.n_classes > self.n_neurons:
            raise ValueError("n_classes cannot exceed n_neurons")
        if self.readout_rule not in ("binary", "delta"):
            raise ValueError(
                f"readout_rule must be 'binary'|'delta', got {self.readout_rule!r}")


class CortexClassifier:
    """Spiking substrate + locally-learned readout."""

    def __init__(self, cfg: CortexConfig, backend: Backend | str | None = None):
        self.cfg = cfg
        be = backend if isinstance(backend, Backend) else get_backend(backend, seed=cfg.seed)
        self.be = be
        n = cfg.n_neurons

        neu = NeuronConfig(
            n=n,
            dend_mode="dcaap" if cfg.use_dendrite else "linear",
            dend_scale=cfg.dend_scale,
            dend_gain=cfg.dend_gain,
            noise_std=0.02,
        )
        plas = PlasticityConfig(
            enabled=cfg.plasticity,
            eligibility="off",
            homeostasis=True,
        )
        self.brain = Brain(
            SimConfig(
                n_neurons=n, k_out=cfg.k_out, seed=cfg.seed,
                spike_capacity_frac=cfg.spike_capacity_frac,
                inhibition=cfg.inhibition, k_wta=cfg.k_wta,
                neu=neu, plas=plas,
            ),
            be,
        )

        rng = np.random.default_rng(cfg.seed)
        self.W = (rng.standard_normal((n, cfg.n_classes)).astype(np.float32)
                  * cfg.w_init if cfg.w_init else
                  np.zeros((n, cfg.n_classes), dtype=np.float32))
        self._n_seen = 0
        self._mean = np.zeros(n, dtype=np.float32)
        self._m2 = np.zeros(n, dtype=np.float32)
        self.last_synops = 0.0
        self.readout_updates = 0

    # ------------------------------------------------------------ substrate
    def _code(self, x: np.ndarray, t_ms: int) -> np.ndarray:
        """Sparse population code: spike counts over ``t_ms`` milliseconds.

        Current routing, which matters for the dendritic claim: the feedforward
        image is injected into the **dendritic** compartment, while recurrent
        synaptic input is delivered to the **soma** (see ``Brain.step``, which
        feeds the delay buffer into ``i_soma``). This division is what puts the
        nonlinearity in the input pathway, so ablating it is a real ablation.

        An earlier version injected the image into the soma, which left
        ``v_dend`` identically zero and made ``dend_mode`` a no-op; the
        dendrite-ablated and intact arms then produced bit-identical results.
        That is a bug, not a finding, and it is why the routing is stated here.
        """
        be, cfg = self.be, self.cfg
        # Adaptation (tau ~100 ms) outlives a 15 ms sample, so without a reset
        # each code is partly a function of the PREVIOUS image. Every
        # measurement in this repo predating this fix carries that leak.
        if cfg.reset_between_samples:
            self.brain.reset()
        pix = np.asarray(x, dtype=np.float32).reshape(-1)
        drive = np.zeros(cfg.n_neurons, dtype=np.float32)
        drive[: cfg.n_input] = pix * cfg.gain
        drive_b = be.array(drive)
        counts = np.zeros(cfg.n_neurons, dtype=np.float32)
        synops = 0
        for _ in range(t_ms):
            st = self.brain.step(external_dend=drive_b, neuromod=1.0)
            counts += be.to_numpy(self.brain.last_spike_mask).astype(np.float32)
            synops += st.synops_active
        self.last_synops = float(synops)
        return counts

    def _normalise(self, c: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize_code:
            return c
        sd = np.sqrt(self._m2 / max(1, self._n_seen - 1)) if self._n_seen > 1 else 1.0
        return (c - self._mean) / (sd + 1e-6)

    def _update_running_stats(self, c: np.ndarray) -> None:
        self._n_seen += 1
        if self._n_seen == 1:
            self._mean = c.copy()
            self._m2 = np.zeros_like(c)
            return
        d = c - self._mean
        self._mean += d / self._n_seen
        self._m2 += d * (c - self._mean)

    # ------------------------------------------------------------- training
    def fit_task(self, task, epochs: int = 1) -> dict:
        """Train on one task in a single sequential pass (no replay)."""
        cfg = self.cfg
        X = np.asarray(task.x_train)
        y = np.asarray(task.y_train)
        info = {"n_train": int(len(y)), "updates": 0}
        for _ep in range(epochs):
            for i in range(len(y)):
                c = self._code(X[i], cfg.t_train_ms)
                self._update_running_stats(c)
                cn = self._normalise(c)
                if cfg.readout_learn:
                    out = self.W.T @ cn
                    target = np.zeros(cfg.n_classes, dtype=np.float32)
                    target[int(y[i])] = 1.0
                    if cfg.readout_rule == "delta":
                        # Graded error: the correction shrinks as the output
                        # approaches the target. This is the rule the docstring
                        # has always described.
                        err = target - out
                    else:
                        # ORIGINAL, DEFECTIVE. Kept only for ablation. Binarising
                        # the output means a class that already scores positive
                        # gets zero error and stops learning entirely.
                        err = target - (out > 0.0).astype(np.float32)
                    self.W += cfg.readout_lr * np.outer(cn, err).astype(np.float32)
                    self.readout_updates += 1
                    info["updates"] += 1
        return info

    # ------------------------------------------------------------ inference
    def class_scores(self, x: np.ndarray) -> np.ndarray:
        c = self._code(x, self.cfg.t_test_ms)
        cn = self._normalise(c)
        return self.W.T @ cn

    def predict(self, x: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(x))
        return np.array([int(np.argmax(self.class_scores(row))) for row in X],
                        dtype=np.int64)

    def evaluate(self, x, y) -> float:
        pred = self.predict(x)
        return float(np.mean(pred == np.asarray(y)))

    # ------------------------------------------------------------- metadata
    @property
    def n_params(self) -> int:
        """Trainable scalars in the readout (what learning actually adjusts)."""
        return int(self.W.size)

    @property
    def n_substrate_synapses(self) -> int:
        return int(self.brain.n_synapses)

    @property
    def synops_per_sample(self) -> float:
        return float(self.last_synops)

    def reset_readout(self) -> None:
        self.W[:] = 0.0
        self._n_seen = 0
        self._mean[:] = 0.0
        self._m2[:] = 0.0
        self.readout_updates = 0
