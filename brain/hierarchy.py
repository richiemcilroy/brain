"""A two-layer cortical hierarchy trained by local rules only.

The question this module exists to answer
----------------------------------------
The single-layer substrate in :mod:`brain.cortex` plateaus well below a dense
backpropagation network on 10-way MNIST, and the diagnosis on file was that its
*representation* is the bottleneck, not forgetting. One observation kept the
door open: on identical cached codes, a **quadratic** ridge readout was reported
to reach roughly twice the accuracy of a **linear** one, which suggested the
code carries discriminative structure that a linear decoder cannot reach.

That motivates the hypothesis tested here:

    A second layer of the *same* dendritic spiking neurons, trained by the
    *same* local rule, can extract nonlinear structure that a linear readout
    misses. Dendritic nonlinearity substitutes for explicit quadratic feature
    expansion, with no backpropagation and no kernel.

This module implements the substrate for that test. It reports numbers; it does
not decide what they mean. The verdict lives in ``docs/HIERARCHY.md``.

Architecture (and exactly where the label enters)
-------------------------------------------------
``layer 1`` is **unsupervised**. It receives the 784-pixel image in its
*dendritic* compartment (the routing :mod:`brain.cortex` documents as what makes
the dendritic ablation a real ablation) and runs three-factor plasticity with a
constant neuromodulator (M = 1), i.e. pure Hebbian self-organisation. It never
sees a label.

``layer 2`` is driven by layer 1's **actual spikes from the same time window** -
one spike mask per millisecond, projected through a fixed random relay - and
also self-organises with the same local rule. The **label enters only through
the readout** on the final layer, trained by the delta rule exposed by
:mod:`brain.readout`. Layer 1 is never trained with the label in the main arm;
that is what keeps this a hierarchy rather than a supervised stack.

Relay topology (a design choice worth stating plainly)
-----------------------------------------------------
``relay="fanin"`` (default) gives each layer-2 neuron a fixed random set of
``relay_k`` layer-1 afferents with row-normalised weights, so layer 2 genuinely
*recombines* layer-1 features. This is the biologically standard construction:
cortical neurons integrate thousands of presynaptic partners, not one.

``relay="diagonal"`` is the literal reading of "input population size = previous
layer's neuron count": layer-2 neuron *j* receives layer-1 neuron *j*. That is a
hard-wired identity channel, and it makes layer 2 a thresholded echo of layer 1.
It is implemented because it is the naive construction, and its measured failure
is part of the result rather than something to hide.

The relay weights are **fixed random**, never learned. That matters for
interpretation: relay + dendritic nonlinearity is structurally a random-features
expansion, and the honest control for it is a random-features readout at matched
width. One of those is run in ``experiments/hierarchy.py``.

Drive scaling (why the relay gain is calibrated, not guessed)
------------------------------------------------------------
A spiking layer emits a sparse mask: at ``k_out=48`` roughly 0.2 of neurons fire
per millisecond, so a relay into 900 neurons delivers *very* few spikes per
target per millisecond. Below a threshold layer 2 is silent, and a silent layer 2
would produce a bogus "depth does not help" verdict. Because the cliff is sharp
(layer 2 emits no spikes at all up to about 6x row-sum gain and a full code by
about 12x), the gain is **calibrated on training data only** - homeostatic rate
matching: pick the gain whose layer-2 mean active fraction best matches layer
1's. Tuning this on the test set would be leakage, and the sweep is reported as
a sensitivity curve rather than used for selection.

Determinism
-----------
Every stochastic choice comes from a seeded generator: one :class:`Backend` per
layer (seeded ``seed + 1000 * layer``) and one host-side numpy generator for the
relay (``seed * 7919 + layer``). Repeating a run with the same seed reproduces
the same codes and the same accuracy. MLX builds a lazy graph, so every value
this module reads is forced through ``Backend.to_numpy`` before use - reading an
unevaluated array is how this project already produced one bogus throughput
figure.

Cost accounting
---------------
Two counts are reported and they are *different quantities*:

* ``n_params`` - every scalar a learning rule can change: plastic substrate
  synapses (``n_neurons * k_out`` per plastic layer) plus the readout weights.
* ``synops_per_sample`` - **active** synaptic operations actually executed,
  i.e. ``spikes x fan-out`` summed over layers, which is the quantity to set
  against a dense baseline's MACs. It excludes the padded fan-out the
  implementation gathers but discards.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # normal package import
    from .backend import Backend, has_mlx
    from .neurons import NeuronConfig
    from .plasticity import PlasticityConfig
    from .simulator import Brain, SimConfig
except ImportError:  # pragma: no cover - direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from brain.backend import Backend, has_mlx
    from brain.neurons import NeuronConfig
    from brain.plasticity import PlasticityConfig
    from brain.simulator import Brain, SimConfig

__all__ = ["StackConfig", "CorticalStack", "READOUT_AVAILABLE"]

# brain/readout.py is written in parallel with this module. Import it if it is
# there; otherwise fall back to a local implementation of the same contract so
# that this experiment still runs. brain/readout.py is never modified here.
try:  # pragma: no cover - depends on a sibling agent's timing
    from .readout import Readout, ReadoutConfig

    READOUT_AVAILABLE = True
except Exception:  # pragma: no cover
    Readout = None  # type: ignore[assignment]
    ReadoutConfig = None  # type: ignore[assignment]
    READOUT_AVAILABLE = False
# =====================================================================
# configuration
# =====================================================================
@dataclass
class StackConfig:
    """Configuration for :class:`CorticalStack`.

    The first thirteen fields are the frozen interface this module was built
    to. Fields after ``seed`` are additions needed to run the experiment
    honestly (relay topology, drive calibration, cost accounting); every one of
    them has a default, so the frozen interface is preserved exactly - the
    documented constructions keep working unchanged.
    """

    # ---- frozen interface -------------------------------------------------
    n_layers: int = 2
    n_input: int = 784
    n_classes: int = 10
    layer_neurons: int = 900
    k_out: int = 48
    t_ms: int = 15
    gain: float = 2.2
    dend_gain: float = 2.5
    use_dendrite: bool = True
    plasticity: bool = True
    readout_kind: str = "local_delta"
    readout_lr: float = 1e-2
    readout_epochs: int = 1
    seed: int = 0

    # ---- extensions (all defaulted; see the module docstring) -------------
    #: ``"fanin"`` = fixed random fan-in of ``relay_k`` layer-1 neurons per
    #: layer-2 neuron (cortical integration). ``"diagonal"`` = layer-2 neuron
    #: ``j`` receives layer-1 neuron ``j`` (the naive construction).
    relay: str = "fanin"
    relay_k: int = 64
    #: Inter-layer drive scale. ``None`` -> calibrated on training data by
    #: homeostatic rate matching (see the module docstring).
    relay_gain: float | None = None
    #: Log-spaced because the response is multiplicative: a spiking layer emits
    #: a sparse mask, and the relay must amplify ~0.2 spikes/neuron/ms into a
    #: threshold-crossing drive. Measured on MNIST, layer 2 with dendrites is
    #: silent below gain ~4, peaks near ~16-32, and is *vetoed* above ~128 (the
    #: non-monotonic dendritic activation attenuates strong inputs). A linear
    #: grid in [0.25, 6] straddles the silent region only.
    relay_gain_grid: tuple = (2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0,
                              64.0, 96.0, 128.0)
    relay_calib_samples: int = 16
    #: Deprecated alias retained for interface compatibility; see
    #: ``rescale_features``. The readout is always handed frozen-standardised
    #: features so that its own drifting normaliser cannot inflate lambda_max.
    standardize: bool = True
    #: Sampled pairwise products for the quadratic readout kinds. 0 makes those
    #: kinds degenerate to their linear counterparts (the contract warns about
    #: this), so the diagnostics set it explicitly.
    quad_dim: int = 2048
    #: Per-layer overrides. ``None`` -> use ``plasticity`` / ``use_dendrite``
    #: for every layer. A tuple is indexed by layer, so the frozen-layer-1
    #: control and the layer-2 dendrite ablation are expressible without
    #: duplicating the whole configuration.
    layer_plasticity: tuple | None = None
    layer_use_dendrite: tuple | None = None
    #: Reset neuron state between samples. Default ``False`` = carry over, which
    #: is what :class:`brain.cortex.CortexClassifier` does and what makes the
    #: dynamics comparable to the published single-layer number. It is kept as a
    #: switch because it is a *real* confound with a large measured effect
    #: (resetting drops layer 1's active fraction from ~0.016 to ~0.005 and
    #: silences a dendritic layer 2 at every gain in the grid, since the
    #: dendritic potential needs ~10 ms to charge). Reported as an ablation
    #: rather than silently chosen. Note that with carry-over the code for a
    #: sample depends on the samples preceding it, so the experiment resets
    #: state at every phase boundary (train / validation / test) to keep each
    #: phase reproducible from its own sample order.
    reset_between_samples: bool = False
    #: Apply a **frozen** per-feature z-score (fitted on training codes only)
    #: before the local rule runs. This is an anti-divergence measure, not a
    #: tuning knob: with a drifting normaliser lambda_max(X^T X) measures ~1.6e4,
    #: so the frozen default lr=1e-2 diverges and the readout reports chance.
    #: See :meth:`CorticalStack._standardise` for the full argument.
    rescale_features: bool = True
    inhibition: str = "kwta"
    k_wta: int = 0          # 0 -> use k_out
    spike_capacity_frac: float = 0.30
    noise_std: float = 0.02
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_input > self.layer_neurons:
            raise ValueError(
                f"n_input ({self.n_input}) cannot exceed layer_neurons "
                f"({self.layer_neurons}): layer 1 injects the input one element per "
                f"neuron, so a larger input would be silently truncated"
            )
        if self.relay not in ("fanin", "diagonal"):
            raise ValueError("relay must be fanin|diagonal")
        if self.relay_k < 1:
            raise ValueError("relay_k must be >= 1")
        if self.t_ms < 1:
            raise ValueError("t_ms must be >= 1")
        if self.readout_epochs < 1:
            raise ValueError("readout_epochs must be >= 1")
        if self.layer_neurons % 2:
            raise ValueError(
                "layer_neurons must be even: the simulator splits the population "
                "into excitatory and inhibitory halves"
            )
        for name in ("layer_plasticity", "layer_use_dendrite"):
            over = getattr(self, name)
            if over is not None and len(over) != self.n_layers:
                raise ValueError(
                    f"{name} must have one entry per layer ({self.n_layers}); "
                    f"got {len(over)}"
                )


# =====================================================================
# fallback readout
# =====================================================================
@dataclass
class _FallbackReadoutConfig:
    """Mirror of ``brain.readout.ReadoutConfig``'s contract.

    Used only when ``brain/readout.py`` is not importable. The field names and
    order match the frozen interface so that passing keywords behaves
    identically for both implementations.
    """

    kind: str = "local_delta"
    n_features: int = 1
    n_classes: int = 10
    lr: float = 1e-2
    ridge: float = 1.0
    epochs: int = 1
    normalize: bool = True
    quad_dim: int = 0
    quad_seed: int = 0
    w_init: float = 0.0
    shuffle: bool = True
    seed: int = 0


class _FallbackReadout:
    """Local readout implementing the frozen ``Readout`` contract.

    Three kinds, all of which receive features from the substrate:

    ``local_delta``
        The delta rule already used by :meth:`brain.cortex.CortexClassifier.fit_task`:
        ``dW_ij = lr * (target_j - sign(out_j)) * x_i``, one sequential update
        per sample, no backward pass and no weight transport. This is the main
        arm's rule and the one the hypothesis is about.
    ``ridge``
        Closed-form ridge solve, an *oracle* reference (not local, not a
        claim): what a linear decoder can reach from the same code.
    ``quadratic_ridge``
        Closed-form ridge on ``[x, x^2]``, the explicit quadratic expansion the
        hypothesis says dendritic depth should substitute for.

    Normalisation uses running statistics accumulated from ``partial_fit``
    inputs only. Nothing here ever sees test data, so the oracle arms cannot
    leak into the deployment path either.
    """

    def __init__(self, cfg: _FallbackReadoutConfig, backend: Any = None):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self.n_params = 0
        self.updates = 0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        cfg = self.cfg
        if cfg.w_init:
            self.W = (self._rng.standard_normal((cfg.n_features, cfg.n_classes))
                      .astype(np.float32) * cfg.w_init)
        else:
            self.W = np.zeros((cfg.n_features, cfg.n_classes), dtype=np.float32)
        self._n_seen = 0
        self._mean = np.zeros(cfg.n_features, dtype=np.float32)
        self._m2 = np.zeros(cfg.n_features, dtype=np.float32)
        self._xtx = np.zeros((cfg.n_features, cfg.n_features), dtype=np.float64)
        self._xty = np.zeros((cfg.n_features, cfg.n_classes), dtype=np.float64)
        self._bias = np.zeros(cfg.n_classes, dtype=np.float64)
        self._solved: np.ndarray | None = None
        self._quad_idx: np.ndarray | None = None
        self.n_params = 0
        self.updates = 0

    # -------------------------------------------------------------- features
    def _as_features(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if X.shape[1] != self.cfg.n_features:
            raise ValueError(
                f"expected {self.cfg.n_features} features, got {X.shape[1]}"
            )
        return X

    def _quadratic(self, Z: np.ndarray, *, fit: bool) -> np.ndarray:
        """``[z, z^2]`` on a fixed subset of columns chosen once from train data.

        Squaring all 900 columns would give 1800 features, which is affordable,
        but the subset is selected by variance so the oracle can be narrowed
        without a code change. ``quad_dim=0`` means "all columns".
        """
        cfg = self.cfg
        if self._quad_idx is None:
            k = cfg.n_features if cfg.quad_dim <= 0 else min(cfg.quad_dim, cfg.n_features)
            if fit:
                var = Z.var(axis=0)
                self._quad_idx = np.argsort(-var)[:k]
            else:  # pragma: no cover - predict is always preceded by a fit
                self._quad_idx = np.arange(k)
        return np.hstack([Z, Z[:, self._quad_idx] ** 2]).astype(np.float32)

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return X
        sd = np.sqrt(self._m2 / max(1, self._n_seen - 1)) if self._n_seen > 1 else 1.0
        return (X - self._mean) / (sd + 1e-6)

    def _observe(self, X: np.ndarray) -> None:
        """Welford pass so normalisation statistics come from train data only."""
        for row in X:
            self._n_seen += 1
            if self._n_seen == 1:
                self._mean = row.copy()
                self._m2 = np.zeros_like(row)
                continue
            d = row - self._mean
            self._mean += d / self._n_seen
            self._m2 += d * (row - self._mean)

    @staticmethod
    def _onehot(Y: np.ndarray, n_classes: int) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] == n_classes:
            return Y.astype(np.float32)
        out = np.zeros((len(Y), n_classes), dtype=np.float32)
        out[np.arange(len(Y)), Y.astype(np.int64)] = 1.0
        return out

    # --------------------------------------------------------------- fitting
    def partial_fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        cfg = self.cfg
        X = self._as_features(X)
        T = self._onehot(Y, cfg.n_classes)
        self._observe(X)
        Z = self._normalise(X)

        if cfg.kind == "local_delta":
            W = self.W
            idx = np.arange(len(Z))
            for _ in range(max(1, cfg.epochs)):
                if cfg.shuffle:
                    self._rng.shuffle(idx)
                for i in idx:
                    err = T[i] - (W.T @ Z[i] > 0.0).astype(np.float32)
                    W += cfg.lr * np.outer(Z[i], err).astype(np.float32)
                    self.updates += 1
            self.W = W
            self.n_params = int(self.W.size)
            return {"updates": int(self.updates), "n_samples": int(len(Z)),
                    "kind": cfg.kind}

        # closed-form kinds: accumulate normal equations
        if cfg.kind == "quadratic_ridge":
            Z = self._quadratic(Z, fit=True)
        Zb = np.hstack([Z, np.ones((len(Z), 1), dtype=np.float32)])
        self._xtx += Zb.astype(np.float64).T @ Zb.astype(np.float64)
        self._xty += Zb.astype(np.float64).T @ T.astype(np.float64)
        self.n_params = int((Zb.shape[1]) * cfg.n_classes)
        return {"updates": 0, "n_samples": int(len(Z)), "kind": cfg.kind}

    def _solve(self) -> None:
        if self._solved is not None:
            return
        n = self._xtx.shape[0]
        A = self._xtx + self.cfg.ridge * np.eye(n)
        self._solved = np.linalg.solve(A, self._xty)
        self.updates = 1

    # ------------------------------------------------------------- inference
    def _design(self, X: np.ndarray) -> np.ndarray:
        X = self._as_features(X)
        Z = self._normalise(X)
        if self.cfg.kind == "quadratic_ridge":
            Z = self._quadratic(Z, fit=False)
        if self.cfg.kind == "local_delta":
            return Z
        return np.hstack([Z, np.ones((len(Z), 1), dtype=np.float32)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        D = self._design(X)
        if self.cfg.kind == "local_delta":
            return np.argmax(D @ self.W, axis=1).astype(np.int64)
        self._solve()
        return np.argmax(D.astype(np.float64) @ self._solved, axis=1).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1)
        return float(np.mean(self.predict(X) == y))


def _make_readout(cfg: StackConfig, n_features: int, seed: int,
                  kind: str | None = None) -> Any:
    """Build the readout, preferring ``brain.readout`` and falling back locally."""
    kind = kind or cfg.readout_kind
    if READOUT_AVAILABLE:
        return Readout(ReadoutConfig(
            kind=kind, n_features=int(n_features), n_classes=int(cfg.n_classes),
            lr=cfg.readout_lr, ridge=1e-2, epochs=cfg.readout_epochs,
            normalize=False, quad_dim=cfg.quad_dim, quad_seed=seed,
            w_init=0.0, shuffle=True, seed=seed,
        ))
    return _FallbackReadout(_FallbackReadoutConfig(
        kind=kind, n_features=int(n_features), n_classes=int(cfg.n_classes),
        lr=cfg.readout_lr, ridge=1.0, epochs=cfg.readout_epochs,
        normalize=False, quad_dim=cfg.quad_dim, quad_seed=seed,
        w_init=0.0, shuffle=True, seed=seed,
    ))


class CorticalStack:
    """Stacked spiking layers with a locally-trained readout on the top layer.

    Layer 1 is unsupervised; the label enters only at the readout (see the
    module docstring). ``fit_task`` performs a single sequential pass with no
    replay of earlier tasks, which is what makes the split/permuted benchmarks
    a measurement of interference rather than of a replay heuristic.
    """

    def __init__(self, cfg: StackConfig, backend: Any = None):
        self.cfg = cfg
        be_name = backend if isinstance(backend, str) else (
            getattr(backend, "name", None) or cfg.backend)
        self.be = backend if isinstance(backend, Backend) else (
            Backend(be_name, seed=cfg.seed) if be_name else
            Backend("auto", seed=cfg.seed))
        self.be_name = self.be.name

        self.brains: list[Brain] = []
        for layer in range(cfg.n_layers):
            self.brains.append(self._make_brain(layer))

        self._relay: list[np.ndarray] = [
            self._make_relay(layer) for layer in range(1, cfg.n_layers)
        ]
        self._relay_scale: list[float | None] = [
            cfg.relay_gain for _ in range(max(0, cfg.n_layers - 1))
        ]
        self._calibrated = False

        self.readout = _make_readout(cfg, cfg.layer_neurons, cfg.seed + 77)
        self._last_layer_synops = 0.0
        self._last_layer_spikes = 0.0
        self._last_layer_active = 0.0
        #: Active SynOps for the most recent ``code()`` call, summed over layers.
        self.last_synops_per_sample = 0.0
        #: Frozen z-score statistics (identity until ``fit_task`` measures them).
        self._norm_mean = np.zeros(cfg.layer_neurons, dtype=np.float32)
        self._norm_sd = np.ones(cfg.layer_neurons, dtype=np.float32)
        self._norm_dead = np.zeros(cfg.layer_neurons, dtype=bool)
        #: Spike-overflow counts per layer (must stay 0; see SimConfig).
        self.layer_nudges: dict[int, int] = {}
        self.layer_spikes: dict[int, float] = {}
        self.layer_active_frac: dict[int, float] = {}

    # ------------------------------------------------------------- construction
    def _layer_flag(self, name: str, layer: int, default: bool) -> bool:
        """Resolve a per-layer boolean, honouring an override tuple."""
        over = getattr(self.cfg, name)
        if over is None:
            return bool(default)
        return bool(over[layer])

    def plastic_layers(self) -> list[bool]:
        return [self._layer_flag("layer_plasticity", i, self.cfg.plasticity)
                for i in range(self.cfg.n_layers)]

    def _make_brain(self, layer: int) -> Brain:
        cfg = self.cfg
        n = cfg.layer_neurons
        be = Backend(self.be_name, seed=cfg.seed + 1000 * layer)
        k_wta = cfg.k_wta if cfg.k_wta > 0 else cfg.k_out
        if k_wta > n:
            raise ValueError(f"k_wta ({k_wta}) cannot exceed layer_neurons ({n})")
        br = Brain(
            SimConfig(
                n_neurons=n, k_out=cfg.k_out, seed=cfg.seed + 1000 * layer,
                spike_capacity_frac=cfg.spike_capacity_frac,
                inhibition=cfg.inhibition, k_wta=max(k_wta, 1),
                neu=NeuronConfig(
                    n=n,
                    dend_mode="dcaap" if self._layer_flag(
                        "layer_use_dendrite", layer, cfg.use_dendrite
                    ) else "linear",
                    dend_scale=1.0, dend_gain=cfg.dend_gain,
                    noise_std=cfg.noise_std,
                    refractory_ms=2.0,
                ),
                plas=PlasticityConfig(
                    enabled=self._layer_flag("layer_plasticity", layer, cfg.plasticity),
                    eligibility="off", homeostasis=True,
                ),
            ),
            be,
        )
        return br

    def _make_relay(self, layer: int) -> np.ndarray:
        """Fixed random relay feeding ``layer`` from ``layer - 1``.

        ``fanin``: each target neuron samples ``relay_k`` distinct source
        neurons; rows are then normalised. ``diagonal``: identity coupling, the
        naive construction that makes the target a thresholded echo.
        """
        cfg = self.cfg
        n = cfg.layer_neurons
        rng = np.random.default_rng(cfg.seed * 7919 + layer)
        if cfg.relay == "diagonal":
            # identity: target j <- source j, rescaled so a spike drives the
            # target as strongly as a max-brightness pixel drives layer 1.
            return (np.eye(n, dtype=np.float32) * cfg.gain)

        k = min(cfg.relay_k, n)
        R = np.zeros((n, n), dtype=np.float32)
        for j in range(n):
            idx = rng.choice(n, size=k, replace=False)
            R[j, idx] = rng.standard_normal(k).astype(np.float32)
        # normalise rows so the drive scale does not drift with relay_k; the
        # sign structure (inhibitory/excitatory afferents) is preserved.
        norm = np.linalg.norm(R, axis=1, keepdims=True)
        R = R / np.maximum(norm, 1e-6)
        return R.astype(np.float32)

    # ------------------------------------------------------------------ spikes
    def _run_layer1(self, x: np.ndarray) -> np.ndarray:
        """Run layer 1 for ``t_ms``; return the spike-count code."""
        cfg, be = self.cfg, self.be
        n = cfg.layer_neurons
        brain = self.brains[0]
        pix = np.asarray(x, dtype=np.float32).reshape(-1)
        if pix.size > cfg.n_input:
            raise ValueError(
                f"layer 1 takes {cfg.n_input} inputs, got {pix.size}"
            )
        drive = np.zeros(n, dtype=np.float32)
        drive[: pix.size] = pix * cfg.gain
        drive_b = be.array(drive)

        counts = np.zeros(n, dtype=np.float32)
        synops = 0.0
        nudges = 0
        masks: list[np.ndarray] = []
        for _ in range(cfg.t_ms):
            st = brain.step(external_dend=drive_b, neuromod=1.0)
            mask = be.to_numpy(brain.last_spike_mask).astype(np.float32)
            counts += mask
            synops += float(st.synops_active)
            nudges += int(st.overflow)
            masks.append(mask)
        self._l1_masks = masks
        self._last_layer_synops = synops
        self._last_layer_spikes = float(counts.sum())
        self._last_layer_active = float(counts.sum()) / n / max(1, cfg.t_ms)
        return counts

    def _run_hidden_layer(self, layer: int, masks_prev: list[np.ndarray],
                          scale: float) -> tuple[np.ndarray, list[np.ndarray]]:
        """Drive ``layer`` from ``layer-1``'s per-millisecond spike masks.

        The masks are the *actual* spikes emitted during the same time window,
        not a re-encoded frozen array, so the stack is spiking end to end.
        """
        cfg, be = self.cfg, self.be
        n = cfg.layer_neurons
        brain = self.brains[layer]
        R = self._relay[layer - 1]

        counts = np.zeros(n, dtype=np.float32)
        synops = 0.0
        nudges = 0
        out_masks: list[np.ndarray] = []
        for mask in masks_prev:
            # (n,) @ (n, n) -> each target neuron's weighted afferent sum
            drive = (mask @ R) * scale
            st = brain.step(external_dend=be.array(drive.astype(np.float32)),
                            neuromod=1.0)
            m = be.to_numpy(brain.last_spike_mask).astype(np.float32)
            counts += m
            synops += float(st.synops_active)
            nudges += int(st.overflow)
            out_masks.append(m)
        self._last_layer_synops = synops
        self._last_layer_spikes = float(counts.sum())
        self._last_layer_active = float(counts.sum()) / n / max(1, cfg.t_ms)
        self.layer_nudges[layer] = self.layer_nudges.get(layer, 0) + nudges
        return counts, out_masks

    # -------------------------------------------------------------- forward pass
    def code(self, x: np.ndarray) -> np.ndarray:
        """Final-layer sparse spike-count code for one sample.

        Applies the calibrated relay gains when available. Before calibration
        the configured/default gains are used, so a caller that never calls
        :meth:`calibrate_relay` still gets a deterministic forward pass.
        """
        cfg = self.cfg
        if cfg.reset_between_samples:
            for brain in self.brains:
                brain.reset()
        counts = self._run_layer1(x)
        total_synops = self._last_layer_synops
        self.layer_spikes = {0: self._last_layer_spikes}
        self.layer_active_frac = {0: self._last_layer_active}
        if cfg.n_layers == 1:
            self.last_synops_per_sample = float(total_synops)
            return counts
        masks = self._l1_masks
        for layer in range(1, cfg.n_layers):
            scale = self._relay_scale[layer - 1]
            if scale is None:
                scale = 1.0
            counts, masks = self._run_hidden_layer(layer, masks, float(scale))
            total_synops += self._last_layer_synops
            self.layer_spikes[layer] = self._last_layer_spikes
            self.layer_active_frac[layer] = self._last_layer_active
        self.last_synops_per_sample = float(total_synops)
        return counts

    def codes(self, X: np.ndarray) -> np.ndarray:
        """Codes for a batch of samples, shape ``(len(X), layer_neurons)``."""
        return np.stack([self.code(row) for row in np.atleast_2d(np.asarray(X))])

    # ------------------------------------------------------------- calibration
    def calibrate_relay(self, x_train: np.ndarray, *, verbose: bool = False) -> dict:
        """Choose inter-layer gains by homeostatic rate matching.

        For each hidden layer, sweep candidate scales, measure that layer's mean
        active fraction on **training** data, and pick the scale whose activity
        best matches the layer below's. This is done by measurement rather than
        by guess because the response is sharply nonlinear: with dendrites
        enabled layer 2 emits *nothing* below gain ~4 and is actively suppressed
        above ~128 (the dCaAP-style activation attenuates strong input), so a
        badly-scaled relay produces a silent layer and a bogus "depth does not
        help" verdict.

        Only training data is touched, and every layer's state is reset between
        samples (and again afterwards) so calibration cannot leak
        sample-to-sample state into the measured codes.

        Returns the full sweep: the sensitivity curve is itself reportable
        evidence, not a hidden implementation detail.
        """
        cfg = self.cfg
        if cfg.n_layers < 2:
            return {"sweep": [], "chosen": [], "reference_active_frac": None}
        Xs = np.atleast_2d(np.asarray(x_train))[: max(1, cfg.relay_calib_samples)]

        def reset_all() -> None:
            for brain in self.brains:
                brain.reset()

        # ---- pass 1: reference activity of layer 1, and its spike masks
        #
        # Resetting happens BETWEEN SAMPLES only, never between milliseconds.
        # That distinction is load-bearing: the dendritic potential integrates
        # over tau_dend (~10 ms), so a single millisecond of drive from a reset
        # state is far too small to cross threshold. Resetting per-ms made every
        # candidate gain look silent and would have produced a bogus "a second
        # layer cannot be driven" conclusion.
        refs: list[float] = []
        masks_by_sample: list[list[np.ndarray]] = []
        for x in Xs:
            if cfg.reset_between_samples:
                reset_all()
            self._run_layer1(x)
            refs.append(self._last_layer_active)
            masks_by_sample.append([m.copy() for m in self._l1_masks])
        ref = float(np.mean(refs))

        sweep: list[dict] = []
        for layer in range(1, cfg.n_layers):
            best: tuple[float, float, float] | None = None
            for g in cfg.relay_gain_grid:
                fracs = []
                for masks in masks_by_sample:
                    reset_all()
                    self._run_hidden_layer(layer, masks, float(g))
                    fracs.append(self._last_layer_active)
                mean_frac = float(np.mean(fracs))
                err = abs(mean_frac - ref)
                sweep.append({
                    "layer": layer, "gain": float(g), "active_frac": mean_frac,
                    "abs_err": err, "reference_layer": layer - 1,
                    "reference_active_frac": ref,
                })
                if best is None or err < best[0]:
                    best = (err, float(g), mean_frac)
            assert best is not None
            self._relay_scale[layer - 1] = best[1]
            if verbose:  # pragma: no cover - diagnostic
                print(f"  layer {layer}: relay_gain={best[1]:g} "
                      f"(active_frac {best[2]:.5f} vs layer {layer-1} {ref:.5f})")
            if layer + 1 < cfg.n_layers:
                # calibrate the next layer against the drive it will really see
                next_masks: list[list[np.ndarray]] = []
                for masks in masks_by_sample:
                    reset_all()
                    _, out = self._run_hidden_layer(layer, masks, best[1])
                    next_masks.append([m.copy() for m in out])
                masks_by_sample = next_masks
                ref = best[2]

        # Calibration ran the substrate (and possibly adapted its synapses as a
        # side effect if plasticity is on). Reset every layer so training starts
        # from the same state it would have without calibration, and clear the
        # diagnostic counters the sweep inflated.
        reset_all()
        self.layer_nudges = {}
        self._calibrated = True
        return {
            "sweep": sweep,
            "chosen": [None if v is None else float(v) for v in self._relay_scale],
            "reference_active_frac": ref,
            "n_calib_samples": int(len(Xs)),
        }

    # ------------------------------------------------------- feature standardisation
    def _standardise(self, codes: np.ndarray) -> np.ndarray:
        """Fit a *frozen* per-feature z-score on these (training) codes.

        Why this exists, and why it is not a tuning knob
        ------------------------------------------------
        The delta rule's per-class weights are the least-squares solution and are
        stable only while ``lr * lambda_max(X^T X) < 2``. Measured on this
        substrate's codes with a *drifting* normaliser (the readout's
        ``normalize=True`` path), ``lambda_max = 1.6e4``, so the frozen default
        ``lr = 1e-2`` diverges: the readout's reported mean loss settles around
        1e22 and accuracy lands at chance. That is an instability of the step
        size, not a fact about the substrate's code, and reporting it as "the
        local rule cannot decode this" would be a measurement artefact.

        Standardising each feature to unit variance makes ``lambda_max``
        O(n_features), which puts ``lr = 1e-2`` inside the stability bound. The
        statistics are computed once from the training codes and then **frozen**
        (the readout is constructed with ``normalize=False`` and sees already-
        standardised inputs). Freezing matters: a running normaliser is
        estimated from very few samples early in a pass, so its first few
        samples get divided by a near-zero standard deviation and amplified into
        the very large-magnitude rows that dominate ``lambda_max``.

        Dead features (never active on the training batch) are zeroed rather
        than divided by ~0, which is what previously produced a 1e8 scale
        factor when layer 2 was silent.
        """
        cfg = self.cfg
        if not cfg.rescale_features:
            self._norm_mean = np.zeros(codes.shape[1], dtype=np.float32)
            self._norm_sd = np.ones(codes.shape[1], dtype=np.float32)
            self._norm_dead = np.zeros(codes.shape[1], dtype=bool)
            return codes.astype(np.float32)
        C = codes.astype(np.float64)
        mean = C.mean(axis=0)
        sd = C.std(axis=0)
        # A feature that never fires on the training batch carries no
        # information; zeroing it is exact, whereas dividing by sd ~ 0 injects
        # numerical noise at arbitrary magnitude.
        dead = sd <= 1e-8
        sd = np.where(dead, 1.0, sd)
        self._norm_mean = mean.astype(np.float32)
        self._norm_sd = sd.astype(np.float32)
        self._norm_dead = dead
        return self._apply_standardise(codes)

    def _apply_standardise(self, codes: np.ndarray) -> np.ndarray:
        z = ((codes.astype(np.float32) - self._norm_mean) / self._norm_sd)
        z[:, self._norm_dead] = 0.0
        return z.astype(np.float32)

    # ----------------------------------------------------------------- training
    def fit_task(self, task: Any, epochs: int = 1) -> dict:
        """Single sequential pass over one task. The label enters at the readout only.

        Substrate plasticity runs with ``neuromod=1`` (unsupervised Hebbian
        self-organisation, exactly as in :class:`brain.cortex.CortexClassifier`).
        Layer 1 never sees the label. After the pass, the readout is trained on
        the codes with the local rule.
        """
        cfg = self.cfg
        X = np.asarray(getattr(task, "x_train", task))
        y = np.asarray(getattr(task, "y_train", None)) if hasattr(task, "y_train") else None
        if y is None:
            raise ValueError("fit_task requires a task with x_train and y_train")

        if cfg.n_layers > 1 and not self._calibrated:
            self.calibrate_relay(X)

        codes = np.empty((len(X), cfg.layer_neurons), dtype=np.float32)
        t0 = time.perf_counter()
        synops = 0.0
        for i in range(len(X)):
            for _ep in range(epochs):
                codes[i] = self.code(X[i])
                synops += self._last_layer_synops

        codes_s = self._standardise(codes)
        info = self.readout.partial_fit(codes_s, y)
        self.last_fit_synops = float(synops)
        self.last_fit_seconds = time.perf_counter() - t0
        self._train_codes = codes
        return {
            "n_train": int(len(X)),
            "n_features": int(cfg.layer_neurons),
            "epochs": int(epochs),
            "synops": float(synops),
            "wall_seconds": float(self.last_fit_seconds),
            "readout": dict(info),
        }

    # ---------------------------------------------------------------- inference
    def predict(self, x: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(x))
        codes = self.codes(X)
        return self.readout.predict(self._apply_standardise(codes))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1)
        return float(np.mean(self.predict(x) == y))

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Alias for :meth:`evaluate`, matching the readout's method name."""
        return self.evaluate(x, y)

    # ----------------------------------------------------------------- metadata
    @property
    def n_params(self) -> int:
        """Total trainable scalars across all layers and the readout.

        Counts plastic substrate synapses (each is a scalar the three-factor
        rule can change) plus the readout's parameters. A frozen layer
        contributes nothing, because nothing in it can be modified by learning.
        """
        per_layer = self.cfg.layer_neurons * self.cfg.k_out
        n_plastic = sum(1 for p in self.plastic_layers() if p)
        return int(per_layer * n_plastic + int(self.readout.n_params))

    @property
    def n_substrate_synapses(self) -> int:
        """Structural synapses in the simulated stack (plastic or not)."""
        return int(sum(b.n_synapses for b in self.brains))

    @property
    def synops_per_sample(self) -> float:
        """Active SynOps per sample for the last forward pass.

        Active means ``spikes x fan-out``: the algorithmic cost, which is the
        quantity comparable to a dense baseline's MAC count. The padded
        rectangular gather this implementation actually executes is larger and
        is reported separately rather than conflated with this number.
        """
        return float(self.last_synops_per_sample)

    @property
    def readout_params(self) -> int:
        return int(self.readout.n_params)

    def reset_state(self) -> None:
        """Reset every layer's neuron state (not the learned weights).

        Called at phase boundaries so that each block of codes is reproducible
        from that block's own sample order. With ``reset_between_samples=False``
        the substrate carries state within a block, matching
        :class:`brain.cortex.CortexClassifier`; without this, a test block's
        codes would depend on where the previous training pass happened to stop.
        """
        for brain in self.brains:
            brain.reset()

    def reset_readout(self) -> None:
        self.readout.reset()
        self._norm_mean = np.zeros(self.cfg.layer_neurons, dtype=np.float32)
        self._norm_sd = np.ones(self.cfg.layer_neurons, dtype=np.float32)
        self._norm_dead = np.zeros(self.cfg.layer_neurons, dtype=bool)
