"""Readouts for the cortical sparse code, with an explicit accounting of
*which* part of the pipeline is being blamed when accuracy is low.

Why this module exists
----------------------
The project's continual-learning result (22.10% permuted / 36.40% split for the
substrate against 68.30% for a naive backprop MLP) was recorded as a
falsification of the central hypothesis. A confound was then noticed in the
learning rule used by :meth:`brain.cortex.CortexClassifier.fit_task`::

    err = target - (out > 0.0).astype(np.float32)

That rule **binarises the output**. Once a class's score is positive its error
is exactly zero, so the readout stops learning that class; it can never apply a
graded correction and it never builds a margin. The docstring called it a delta
rule; it is not one. A second, independent observation muddied the picture
further: on identical cached codes a closed-form *quadratic* ridge readout
reached 0.333 on a single 10-way task where a *linear* one reached 0.157. That
comparison may have conflated readout **capacity** with readout **learning
rule**.

Three hypotheses can explain a low number, and they have completely different
fixes:

* ``H-code``   - the substrate's representation is simply weak; every readout
  fails on it.
* ``H-rule``   - the code is fine; the local learning rule is defective, so a
  corrected rule should approach the capacity ceiling.
* ``H-linear`` - the code is nonlinearly decodable but not linearly decodable,
  so a quadratic feature space should beat a linear one under the *same* rule.

This module implements every arm needed to separate them behind one frozen
interface, so that a single experiment can hold the inputs bit-identical and
vary only the readout. It does not decide which hypothesis wins; it makes the
comparison possible.

The rules, and how plausible each one is
---------------------------------------
``local_perceptron``
    The rule currently in ``cortex.py``. ``err = target - 1[out > 0]``. A
    per-class error signal is broadcast to every incoming synapse of that
    class, so the update ``dW_ij = lr * x_i * err_j`` needs only quantities
    available at the synapse (pre-synaptic activity) and at the post-synaptic
    cell (its own error) - biologically the cheapest of the local rules. It is
    also *defective by construction*: the error saturates at 0 for a positive
    class and the rule has no margin, so a class that barely wins stops
    improving. Kept deliberately, as the arm under test.

``local_delta``
    ``err = target - out`` with ``out`` linear. The LMS / Widrow-Hoff delta
    rule, i.e. what ``cortex.py``'s docstring always claimed. Graded: the
    correction shrinks as the output approaches the target, so it *does* build
    a margin and it does not stop learning. Biologically the same three-factor
    form as above (a per-class error broadcast to the synapses of that class);
    the difference is only the nonlinearity in the error term, i.e. whether the
    post-synaptic cell reports a graded prediction error or a thresholded one.
    Graded error signalling is what dopaminergic prediction-error coding looks
    like in the striatum (Schultz 1998), so this is not a biologically absurd
    rule - but it needs a mechanism for an unclamped graded error, which is a
    real requirement, not a detail.

``local_softmax``
    ``p = softmax(out)``, ``err = target - p``. Cross-entropy gradient. Same
    local form: the only non-local-ish quantity is the normalising denominator
    shared by all classes in one output group, which is a single scalar per
    timestep (# per-class normaliser), so the update is local modulo a
    broadcast. This is the rule a spiking classifier usually wants, because it
    makes the class scores compete; it is also the rule whose *gradient* is
    well-conditioned for one-hot targets.

``ridge``
    Closed-form solution of ``min_W ||XW - Y||^2 + ridge*||W||^2`` via the
    normal equations. **Not biologically plausible.** No known synapse
    implements a global matrix inverse; it needs all samples retained
    simultaneously and a device-agnostic solver. It is here as a *capacity*
    probe: it tells us what a linear readout could achieve if the learning
    problem were solved perfectly, which bounds what any linear local rule
    could ever reach.

``ridge_quad``
    Same closed-form solve, but on an explicit quadratic expansion - squared
    terms plus sampled pairwise products. Also **not plausible**. It separates
    "the information is not in the code" from "the information is in the code
    but not linearly decodable". Read ``ridge_quad`` vs ``ridge`` as a statement
    about the *feature space*, not about learning.

``local_quad``
    The local delta rule applied to the same quadratic expansion. This is the
    arm that answers the question that actually matters: can a *plausible* rule
    recover the quadratic gain, or is the gain only available to a device that
    solves a global least-squares problem? Quadratic terms are why a neuron has
    dendrites: a proximal-plus-distal coincidence detector computes products of
    input streams, and Ca2+ plateaus sum those products. So a quadratic feature
    space is not a free lunch invented for this benchmark - it is the
    `dend_mode="dcaap"` claim restated as a linear-algebra statement.

Frozen interface
----------------
The public surface below (``ReadoutConfig`` field names/defaults and the
``Readout`` methods) is a **contract**: other experiments are written against
it and it must not change. Extra helpers (``feature_dim``, ``expand``,
``diagnostics``, ``self_check``) are additive and safe to ignore.

Parameter accounting
--------------------
The readout deliberately has **no bias column**, exactly like
``CortexClassifier`` (``W.T @ cn`` adds no offset). That keeps ``n_params``
identical across rules at a given feature dimension, so a difference between
two arms cannot be an artifact of one of them owning more parameters. The
cost of that choice is stated in ``docs/READOUT.md``: with centred features it
is second order, and it applies equally to every arm.

Numerical policy
----------------
All linear algebra runs on the host in ``float64`` (Gram accumulation in
``float32`` for memory), regardless of backend, because the readout is a
statistician bolted onto the substrate and its arithmetic is not part of the
biological claim. If an MLX backend is supplied, inputs are routed through
``Backend.to_numpy``, which calls ``mx.eval`` first - nothing is ever read out
of a lazy MLX graph as if it were a value.

Run ``python3 brain/readout.py`` for the self-checks (hand-computed cases for
every rule, including the potentiation/depression directions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # normal package import
    from .backend import Backend, get_backend
except ImportError:  # pragma: no cover - direct execution: `python3 brain/readout.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from brain.backend import Backend, get_backend

__all__ = ["ReadoutConfig", "Readout", "KINDS", "self_check"]

#: Local (biologically plausible) rules - online, one sample at a time.
#: ``local_quad`` is in here because it is the delta rule run on the quadratic
#: feature space: same local update, richer input. It is the arm that tests
#: whether a plausible rule can reach what a closed-form solve reaches.
LOCAL_KINDS: tuple[str, ...] = (
    "local_perceptron",
    "local_delta",
    "local_softmax",
    "local_quad",
)

#: Rules that expand the code with squared terms and sampled pairwise products.
QUAD_KINDS: tuple[str, ...] = ("ridge_quad", "local_quad")

#: Closed-form (NOT biologically plausible) rules. These solve a global least
#: squares problem and exist as capacity probes, never as candidate mechanisms.
CLOSED_FORM_KINDS: tuple[str, ...] = ("ridge", "ridge_quad")

#: The six kinds the experiment may request. Order matches the frozen contract.
KINDS: tuple[str, ...] = (
    "local_perceptron",
    "local_delta",
    "local_softmax",
    "ridge",
    "ridge_quad",
    "local_quad",
)

#: A local rule that consumes quadratic features behaves like ``local_delta``.
_LOCAL_ERROR_OF: dict[str, str] = {"local_quad": "local_delta"}


@dataclass
class ReadoutConfig:
    """Configuration for :class:`Readout`.

    Attributes:
        kind: One of ``"local_perceptron"`` (the current, binarised cortex
            rule), ``"local_delta"`` (graded error), ``"local_softmax"``
            (cross-entropy), ``"ridge"`` (closed-form linear), ``"ridge_quad"``
            (closed-form quadratic), ``"local_quad"`` (local delta on quadratic
            features).
        n_features: Width of the incoming code ``X`` (the **raw** width, before
            any quadratic expansion).
        n_classes: Number of class scores.
        lr: Learning rate for the local rules. A delta rule on unit-variance
            features is only stable while ``lr * lambda_max(X^T X) < 2``, so this
            is a real hyper-parameter and not a formality - an unstable ``lr``
            will look exactly like a bad learning rule. Callers should sweep it.
        ridge: L2 penalty for the closed-form rules. Also used as the
            pseudo-inverse floor when the Gram is singular, which it will be
            whenever ``n_samples < n_features``.
        epochs: Passes over each ``partial_fit`` batch for local rules. Ignored
            by the closed-form rules, which see the batch once (twice would
            change nothing for an exact solve).
        normalize: If True, standardise the code with a **running** per-feature
            mean/variance updated as samples arrive (the behaviour of
            ``CortexClassifier``), which keeps drifting during training and is
            then frozen at test time. Set False to use the codes as given. The
            drifting normaliser is a candidate confound in its own right; a
            fixed (non-drifting) z-score can be tested *without* touching this
            interface by pre-transforming ``X`` and passing ``normalize=False``.
        quad_dim: Number of sampled pairwise products added by the quadratic
            kinds. Squares of all ``n_features`` inputs are always added; the
            full pairwise expansion would be ``n_features*(n_features-1)/2``
            wide (320k at 800 features), which is far too large for a dense
            Gram, so pairs are subsampled at fixed, seeded indices.
            ``quad_dim=0`` is honoured but makes the quadratic kinds degenerate
            to their linear counterparts - the returned stats report
            ``feature_dim`` so this cannot pass silently.
        quad_seed: Seed for the pairwise-product subsample. Fixed so that the
            *same* expanded space is used by every arm that shares it.
        w_init: If non-zero, weights start at ``N(0, w_init)`` (seeded by
            ``seed``), matching ``CortexClassifier``. Zero means zero init.
        shuffle: Per-epoch shuffling for the local rules. With ``epochs=1`` this
            only changes the order within the single pass.
        seed: Seed for shuffling and for ``w_init``.
    """

    kind: str = "local_delta"
    n_features: int = 900
    n_classes: int = 10
    lr: float = 1e-2
    ridge: float = 1e-2
    epochs: int = 1
    normalize: bool = True
    quad_dim: int = 0
    quad_seed: int = 0
    w_init: float = 0.0
    shuffle: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}; got {self.kind!r}")
        if self.n_features <= 0:
            raise ValueError(f"n_features must be positive; got {self.n_features}")
        if self.n_classes <= 0:
            raise ValueError(f"n_classes must be positive; got {self.n_classes}")
        if self.lr < 0.0:
            raise ValueError(f"lr must be >= 0; got {self.lr}")
        if self.ridge < 0.0:
            raise ValueError(f"ridge must be >= 0; got {self.ridge}")
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0; got {self.epochs}")
        if self.quad_dim < 0:
            raise ValueError(f"quad_dim must be >= 0; got {self.quad_dim}")

    @property
    def is_local(self) -> bool:
        return self.kind in LOCAL_KINDS

    @property
    def is_closed_form(self) -> bool:
        return self.kind in CLOSED_FORM_KINDS

    @property
    def uses_quadratic(self) -> bool:
        return self.kind in QUAD_KINDS


class Readout:
    """Trainable map from a sparse code ``X`` to ``n_classes`` scores.

    The class is a drop-in replacement for the readout half of
    ``CortexClassifier``: the same online, per-sample, no-replay schedule, but
    with the learning rule and the feature space selectable so that
    representation, rule and linearity can be varied one at a time.

    ``partial_fit`` is called once per task (or per batch) and is the only
    training entry point::

        r = Readout(ReadoutConfig(kind="local_delta", n_features=800))
        r.partial_fit(X_task0, Y_task0)          # sees only this task's samples
        acc = r.score(X_task0_test, y_task0_test)

    Every local rule is a single forward pass with no backward pass through any
    network and no weight transport: the update for ``W[i, j]`` reads the
    pre-synaptic feature ``x_i`` and the post-synaptic error of class ``j``
    only.
    """

    #: Feature rows are expanded in blocks of this many rows when the quadratic
    #: space is active, to bound peak memory on large batches.
    _CHUNK: int = 4096

    def __init__(self, cfg: ReadoutConfig, backend: Backend | str | None = None):
        if not isinstance(cfg, ReadoutConfig):
            raise TypeError(f"cfg must be a ReadoutConfig; got {type(cfg).__name__}")
        self.cfg = cfg
        self.be = backend if isinstance(backend, Backend) else get_backend(backend)

        # ---- quadratic index structure (fixed, seeded, identical for every
        # arm that shares a quad_seed: this is what makes ridge_quad and
        # local_quad comparable feature-space-for-feature-space).
        n_pairs_possible = cfg.n_features * (cfg.n_features - 1) // 2
        self._quad_dim = int(min(cfg.quad_dim, n_pairs_possible))
        self._quad_truncated = bool(self._quad_dim < cfg.quad_dim)
        if self._quad_dim:
            rng = np.random.default_rng(cfg.quad_seed)
            flat = rng.choice(n_pairs_possible, size=self._quad_dim, replace=False)
            # invert the (i<j) column-major triangular index
            i_idx, j_idx = _triangular_unravel(flat, cfg.n_features)
            self._quad_i = i_idx.astype(np.int64)
            self._quad_j = j_idx.astype(np.int64)
        else:
            self._quad_i = np.zeros(0, dtype=np.int64)
            self._quad_j = np.zeros(0, dtype=np.int64)

        self.reset()

    # ------------------------------------------------------------- geometry
    @property
    def quad_dim(self) -> int:
        """Pairwise products actually used (after truncation to the available
        number of distinct pairs)."""
        return self._quad_dim

    @property
    def feature_dim(self) -> int:
        """Width of the vector the rule actually learns on.

        Equals ``n_features`` for the linear kinds, and
        ``2*n_features + quad_dim`` for the quadratic kinds (raw features,
        their squares, and the sampled pairwise products).
        """
        if self.cfg.uses_quadratic:
            return 2 * self.cfg.n_features + self._quad_dim
        return self.cfg.n_features

    # --------------------------------------------------------------- state
    def reset(self) -> None:
        """Zero the weights and drop every accumulated statistic."""
        cfg = self.cfg
        if cfg.w_init:
            rng = np.random.default_rng(cfg.seed)
            self.W = (rng.standard_normal((self.feature_dim, cfg.n_classes))
                      * cfg.w_init).astype(np.float32)
        else:
            self.W = np.zeros((self.feature_dim, cfg.n_classes), dtype=np.float32)
        self._updates = 0
        # Running (drifting) normaliser, Welford - mirrors CortexClassifier.
        self._n_seen = 0
        self._mean = np.zeros(cfg.n_features, dtype=np.float64)
        self._m2 = np.zeros(cfg.n_features, dtype=np.float64)
        # Closed-form accumulation state (kept only for diagnostics; each
        # partial_fit call fits on its own batch, per the frozen contract).
        self._last_gram: np.ndarray | None = None
        self._solve_method = "none"
        self._fitted_classes: np.ndarray = np.zeros(cfg.n_classes, dtype=bool)
        # Diagnostics (never used to change behaviour).
        self._n_nonfinite = 0
        self._n_zero_rows = 0
        self._diverged_at: int | None = None
        self._mean_loss = float("nan")
        self._epochs_run = 0

    # ------------------------------------------------------------ properties
    @property
    def n_params(self) -> int:
        """Trainable scalars actually used: ``feature_dim * n_classes``."""
        return int(self.feature_dim * self.cfg.n_classes)

    @property
    def updates(self) -> int:
        """Number of weight-matrix writes applied.

        Counting convention per kind, because the two families are different
        objects and conflating them would flatter one arm:

        * local rules - one update per sample per epoch, i.e. the same counter
          as ``CortexClassifier.readout_updates``. This is the number of times
          an online learner saw a sample.
        * closed-form rules - one update per solve (one per ``partial_fit``
          call). A solve has no notion of a per-sample step; reporting it as a
          sample count would invent a schedule that does not exist.
        """
        return int(self._updates)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Read-only extras for experiment bookkeeping (not part of the API)."""
        return {
            "kind": self.cfg.kind,
            "feature_dim": self.feature_dim,
            "quad_dim": self._quad_dim,
            "quad_dim_requested": int(self.cfg.quad_dim),
            "quad_truncated": self._quad_truncated,
            "solve_method": self._solve_method,
            "updates": self.updates,
            "n_nonfinite_features": int(self._n_nonfinite),
            "n_zero_rows": int(self._n_zero_rows),
            "mean_loss": self._mean_loss,
            "epochs_run": int(self._epochs_run),
            "n_seen": int(self._n_seen),
            "diverged": self._diverged_at is not None,
            "diverged_at_update": self._diverged_at,
            "weight_absmax": float(np.abs(self.W).max()) if self.W.size else 0.0,
        }

    # -------------------------------------------------------------- helpers
    def _as_numpy(self, X: Any) -> np.ndarray:
        """Bring an array to the host, forcing MLX evaluation first."""
        if self.be.is_mlx:
            return np.asarray(self.be.to_numpy(X), dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    def _labels(self, Y: Any, n: int) -> np.ndarray:
        """Accept one-hot ``(n, n_classes)`` or integer ``(n,)`` labels."""
        arr = self.be.to_numpy(Y) if self.be.is_mlx else np.asarray(Y)
        if arr.ndim == 1:
            lab = arr.astype(np.int64).reshape(-1)
            if lab.shape[0] != n:
                raise ValueError(f"Y has {lab.shape[0]} labels for {n} rows")
            if lab.min(initial=0) < 0 or lab.max(initial=-1) >= self.cfg.n_classes:
                raise ValueError(f"labels must be in [0, {self.cfg.n_classes})")
            return lab
        if arr.ndim == 2:
            if arr.shape[0] != n:
                raise ValueError(f"Y has {arr.shape[0]} rows for {n} features rows")
            if arr.shape[1] != self.cfg.n_classes:
                raise ValueError(
                    f"Y must have {self.cfg.n_classes} columns; got {arr.shape[1]}"
                )
            return np.argmax(arr, axis=1).astype(np.int64)
        raise ValueError(f"Y must be 1-D labels or 2-D one-hot; got ndim {arr.ndim}")

    def _onehot(self, labels: np.ndarray) -> np.ndarray:
        Y = np.zeros((labels.shape[0], self.cfg.n_classes), dtype=np.float64)
        Y[np.arange(labels.shape[0]), labels] = 1.0
        return Y

    def _update_running_stats(self, Xc: np.ndarray) -> None:
        """Welford update, one row at a time - identical to CortexClassifier.

        Row-at-a-time (not a batch update) is deliberate: it reproduces the
        drifting normaliser that the substrate arm actually used, including the
        fact that the estimate at sample ``i`` has never seen sample ``i+1``.
        """
        for row in Xc:
            self._n_seen += 1
            if self._n_seen == 1:
                self._mean = row.copy()
                self._m2 = np.zeros_like(row)
                continue
            d = row - self._mean
            self._mean += d / self._n_seen
            self._m2 += d * (row - self._mean)

    def _standardise(self, Xc: np.ndarray, *, update: bool) -> np.ndarray:
        """Running z-score, matching ``CortexClassifier._normalise`` exactly.

        Two behaviours are reproduced on purpose rather than fixed, because
        they are candidate causes of the substrate's low readout accuracy and
        should be *seen* in the numbers, not hidden by a nicer implementation:

        * the first sample normalises to the zero vector (the mean equals the
          sample and ``sd`` is forced to 1.0), contributing nothing;
        * a feature that has never varied gets ``sd = 0``, so the ``+1e-6``
          guard turns it into an enormous value. Such rows are counted in
          ``diagnostics["n_nonfinite_features"]`` instead of being clipped.
        """
        if update:
            self._update_running_stats(Xc)
        sd = (np.sqrt(self._m2 / max(1, self._n_seen - 1))
              if self._n_seen > 1 else np.ones(self.cfg.n_features))
        out = (Xc - self._mean) / (sd + 1e-6)
        self._n_nonfinite += int(np.count_nonzero(~np.isfinite(out)))
        self._n_zero_rows += int(np.count_nonzero(np.all(out == 0.0, axis=1)))
        return out

    def expand(self, Xc: np.ndarray) -> np.ndarray:
        """Apply the (fixed, non-learned) feature expansion.

        Quadratic kinds get ``[x, x**2, x_i * x_j for sampled pairs]``. The
        expansion is fixed before the rule sees it: nothing here is trained, so
        the comparison between a linear and a quadratic arm is a comparison of
        *feature spaces*, not of parameter counts doing different amounts of
        work per parameter.
        """
        if not self.cfg.uses_quadratic:
            return Xc
        n = Xc.shape[0]
        if n == 0:
            return np.zeros((0, self.feature_dim), dtype=Xc.dtype)
        out = np.empty((n, self.feature_dim), dtype=Xc.dtype)
        step = max(1, self._CHUNK)
        for lo in range(0, n, step):
            hi = min(n, lo + step)
            blk = Xc[lo:hi]
            out[lo:hi, : self.cfg.n_features] = blk
            out[lo:hi, self.cfg.n_features:2 * self.cfg.n_features] = blk * blk
            if self._quad_dim:
                out[lo:hi, 2 * self.cfg.n_features:] = (
                    blk[:, self._quad_i] * blk[:, self._quad_j]
                )
        return out

    def _features(self, X: Any, *, update: bool) -> np.ndarray:
        """``X`` -> the exact matrix the rule learns on (normalise, expand)."""
        Xc = self._as_numpy(X)
        if Xc.ndim != 2:
            Xc = Xc.reshape(1, -1)
        if Xc.shape[1] != self.cfg.n_features:
            raise ValueError(
                f"X must have {self.cfg.n_features} columns; got {Xc.shape[1]}"
            )
        if self.cfg.normalize:
            Xc = self._standardise(Xc, update=update)
        return self.expand(Xc)

    # ----------------------------------------------------------------- rules
    def _local_step(self, F: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Errors for one batch under the configured local rule.

        Returns ``(n, n_classes)`` errors. This is the one place the three local
        rules differ - the update ``dW = lr * F^T @ err / n`` is shared, which
        is the point: they are the same three-factor, synapse-local rule with
        different post-synaptic error terms.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            out = F @ self.W.astype(np.float64)
        # ``local_quad`` is the delta rule on the expanded space, so it shares
        # the delta error term exactly; only ``F`` is wider.
        kind = _LOCAL_ERROR_OF.get(self.cfg.kind, self.cfg.kind)
        if kind == "local_perceptron":
            # The rule under test, reproduced verbatim from cortex.py: the
            # output is binarised, so a class already scoring > 0 has error
            # exactly 0 and stops learning. No margin, no graded correction.
            return Y - (out > 0.0).astype(np.float64)
        if kind == "local_delta":
            return Y - out
        if kind == "local_softmax":
            z = out - out.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            return Y - p
        raise ValueError(f"no local update for kind {kind!r}")

    def _fit_local(self, F: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
        cfg = self.cfg
        n = F.shape[0]
        if n == 0:
            return {"updates": 0, "mean_loss": float("nan")}
        rng = np.random.default_rng(cfg.seed)
        order = np.arange(n)
        # The initial error is measured before any update, on the same batch,
        # so a diverged run is visible as a rising loss rather than silently
        # producing a plausible-looking accuracy.
        losses: list[float] = []
        for _ep in range(cfg.epochs):
            if cfg.shuffle and n > 1:
                rng.shuffle(order)
            for i in order:
                row = F[i:i + 1]
                err = self._local_step(row, Y[i:i + 1])
                # Unbounded error terms (delta, softmax-with-large-logits) can
                # overflow fp32 at a step size the *bounded* perceptron rule
                # handles fine. That divergence is a real property of the rule
                # and is reported, not hidden - but it is suppressed from
                # numpy's warning channel so a single diverged sweep point
                # cannot spam a long experiment run.
                with np.errstate(over="ignore", invalid="ignore"):
                    self.W += (cfg.lr * row.T @ err).astype(np.float32)
                self._updates += 1
                if self._updates == 1 or self._updates % 64 == 0:
                    if not np.isfinite(self.W).all():
                        self._diverged_at = self._updates
                        losses.append(float("inf"))
                        return {"updates": int(self._updates), "mean_loss": float("inf"),
                                "diverged": True}
                losses.append(float(np.mean(err ** 2)))
            self._epochs_run += 1
        return {
            "updates": int(n * cfg.epochs),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "diverged": False,
        }

    def _fit_ridge(self, F: np.ndarray, Y: np.ndarray,
                   labels: np.ndarray) -> dict[str, Any]:
        """Exact ridge solve on this batch only, writing present classes.

        Columns for classes absent from ``Y`` keep their previous weights. That
        mirrors ``brain.baselines.FrozenFeaturesReadout`` and keeps the split
        benchmark (two classes per task) meaningful: a class is only re-fitted
        where there is data to fit it with.
        """
        cfg = self.cfg
        n, d = F.shape
        present = np.zeros(cfg.n_classes, dtype=bool)
        present[np.unique(labels)] = True
        if n == 0 or not present.any():
            self._solve_method = "empty"
            return {"updates": 0, "mean_loss": float("nan")}

        # Gram in float32 (memory), solved in float64 (accuracy).
        G = (F.T @ F).astype(np.float32)
        B = (F.T @ Y).astype(np.float32)
        self._last_gram = G
        # Ridge on the diagonal. The floor scales with the trace so that a code
        # with unstandardised (e.g. raw spike-count) units is regularised
        # comparably to a standardised one; without this the penalty would be
        # effectively absent on large-magnitude features and overwhelming on
        # small ones, which is a unit-dependent result, not a scientific one.
        trace = float(np.trace(G)) / max(1, d)
        lam = float(cfg.ridge) * max(trace, 1e-12)
        A = G.astype(np.float64) + lam * np.eye(d)
        cols = np.flatnonzero(present)
        try:
            sol = np.linalg.solve(A, B[:, cols].astype(np.float64))
            self._solve_method = "solve"
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, B[:, cols].astype(np.float64), rcond=None)[0]
            self._solve_method = "lstsq"
        self.W[:, cols] = sol.astype(np.float32)
        self._fitted_classes[cols] = True
        self._updates += 1
        resid = F @ sol - Y[:, cols]
        return {
            "updates": 1,
            "mean_loss": float(np.mean(resid ** 2)),
            "ridge_lambda": lam,
            "solve_method": self._solve_method,
            "trace_over_dim": trace,
        }

    # ---------------------------------------------------------------- public
    def partial_fit(self, X: Any, Y: Any) -> dict:
        """Fit on this batch, in one online pass (local) or one solve (close).

        Args:
            X: ``(n, n_features)`` float32 codes. MLX arrays are evaluated
                before use.
            Y: ``(n, n_classes)`` one-hot float32, or ``(n,)`` integer labels
                (accepted for convenience; the contract specifies one-hot).

        Returns:
            Stats dict: ``n_samples``, ``updates`` (this call),
            ``updates_total``, ``mean_loss`` (squared error of the last batch
            before/at update time), ``feature_dim``, plus rule-specific keys.
        """
        Xn = self.be.to_numpy(X) if self.be.is_mlx else np.asarray(X)
        Xn = np.asarray(Xn, dtype=np.float32)
        if Xn.ndim != 2:
            Xn = Xn.reshape(1, -1)
        labels = self._labels(Y, Xn.shape[0])
        F = self._features(Xn, update=True)
        Yoh = self._onehot(labels)

        if self.cfg.is_local:
            info = self._fit_local(F, Yoh)
        else:
            info = self._fit_ridge(F, Yoh, labels)
        self._mean_loss = float(info.get("mean_loss", float("nan")))
        self.be.eval()  # keep the MLX-eval discipline explicit (no-op on NumPy)
        return {
            "kind": self.cfg.kind,
            "n_samples": int(Xn.shape[0]),
            "updates": int(info.get("updates", 0)),
            "updates_total": self.updates,
            "mean_loss": float(info.get("mean_loss", float("nan"))),
            "feature_dim": self.feature_dim,
            "n_params": self.n_params,
            "normalize": bool(self.cfg.normalize),
            "solve_method": info.get("solve_method", "-"),
            "quad_dim": self._quad_dim,
        }

    def scores(self, X: Any) -> np.ndarray:
        """Raw class scores ``(n, n_classes)`` (not part of the frozen API)."""
        Xn = self.be.to_numpy(X) if self.be.is_mlx else np.asarray(X)
        Xn = np.asarray(Xn, dtype=np.float32)
        if Xn.ndim != 2:
            Xn = Xn.reshape(1, -1)
        F = self._features(Xn, update=False)
        with np.errstate(over="ignore", invalid="ignore"):
            return (F @ self.W.astype(np.float64)).astype(np.float64)

    def predict(self, X: Any) -> np.ndarray:
        """``(n,)`` int labels: argmax over class scores."""
        return np.argmax(self.scores(X), axis=1).astype(np.int64)

    def score(self, X: Any, y: Any) -> float:
        """Accuracy of :meth:`predict` against ``y`` (labels or one-hot)."""
        Xn = self.be.to_numpy(X) if self.be.is_mlx else np.asarray(X)
        Xn = np.asarray(Xn)
        if Xn.ndim != 2:
            Xn = Xn.reshape(1, -1)
        lab = self._labels(y, Xn.shape[0])
        if lab.size == 0:
            return float("nan")
        return float(np.mean(self.predict(Xn) == lab))


# --------------------------------------------------------------- index helper
def _triangular_unravel(flat: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert the ``i < j`` upper-triangular index used to sample pairs.

    ``flat`` indexes the strict upper triangle of an ``n x n`` matrix in
    row-major order, i.e. position ``k`` maps to the pair
    ``(i, j) = (row, col)`` with ``i < j``. Computed in closed form (no
    search) so the same pair set is reconstructible from the seed alone by any
    reader of the JSON, which is what makes the quadratic arms auditable.
    """
    flat = np.asarray(flat, dtype=np.int64)
    # Solve (n-1)*i - i(i-1)/2 <= k  by stepping i up from the analytic guess.
    i = (2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8.0 * flat)) / 2.0
    i = np.floor(i).astype(np.int64)
    i = np.clip(i, 0, n - 2)
    row_start = i * (2 * n - i - 1) // 2
    # Correct any off-by-one from the float sqrt (robustness, not branch logic).
    over = row_start > flat
    while over.any():
        i[over] -= 1
        row_start = i * (2 * n - i - 1) // 2
        over = row_start > flat
    j = (flat - row_start) + i + 1
    return i, j


# ------------------------------------------------------------------ self-check
def _check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(f"readout self-check failed: {what}")


def _check_close(got: float, want: float, what: str, tol: float = 1e-9) -> None:
    if not np.isfinite(got) or abs(got - want) > tol:
        raise AssertionError(
            f"readout self-check failed: {what}: got {got!r}, want {want!r} "
            f"(tol {tol})"
        )


def self_check(*, verbose: bool = True) -> dict[str, float]:
    """Hand-computed assertions for every rule. Returns the values checked.

    ``python3 brain/readout.py`` runs this from a clean checkout; it is also
    importable so the test-suite can call ``self_check(verbose=False)``.
    """
    out: dict[str, float] = {}

    # ------------------------------------------------- index inversion is exact
    n = 12
    _, jj = np.triu_indices(n, k=1)
    ii_all = np.repeat(np.arange(n - 1), np.arange(n - 1, 0, -1))
    for k in range(len(ii_all)):
        i, j = _triangular_unravel(np.array([k]), n)
        _check(int(i[0]) == int(ii_all[k]) and int(j[0]) == int(jj[k]),
               f"triangular index {k} -> ({ii_all[k]},{jj[k]})")

    # ------------------------------------------------------ param accounting
    r = Readout(ReadoutConfig(kind="ridge", n_features=7, n_classes=3))
    _check(r.n_params == 21, "linear n_params = feature_dim * n_classes")
    rq = Readout(ReadoutConfig(kind="ridge_quad", n_features=7, n_classes=3,
                               quad_dim=5))
    _check(rq.feature_dim == 2 * 7 + 5, "quad feature_dim = 2n + quad_dim")
    _check(rq.n_params == 19 * 3, "quad n_params")
    # quad_dim=0 must NOT silently look like a quadratic arm with real gain
    rq0 = Readout(ReadoutConfig(kind="ridge_quad", n_features=7, n_classes=3,
                                quad_dim=0))
    _check(rq0.feature_dim == 14, "quad_dim=0 degenerates to squares only")
    out["quad_feature_dim"] = float(rq.feature_dim)

    # ------------------------------- local_delta potentiation / depression ----
    # Two features, two classes, one sample whose feature is 1.0. Column 1 is
    # already positive (0.5) and column 0 is at zero, and the sample is class 0.
    #   target-y   = [1, 0]
    #   out        = [0, 0.5]
    #   err_delta  = [1, -0.5]
    # so class 0 (the target) must be POTENTIATED and class 1 (a class that is
    # active but not the target) must be DEPRESSED, in the same step: this is
    # the direction check for both signs of the plasticity rule.
    cfg = ReadoutConfig(kind="local_delta", n_features=2, n_classes=2, lr=0.5,
                        normalize=False, epochs=1, shuffle=False, seed=0)
    r = Readout(cfg)
    r.W[:] = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.float32)
    r.partial_fit(np.array([[1.0, 0.0]], dtype=np.float32),
                  np.array([[1.0, 0.0]]))
    _check_close(float(r.W[0, 0]), 0.5, "delta potentiation +lr*err")
    _check_close(float(r.W[0, 1]), 0.25, "delta depression -lr*err")
    _check(float(r.W[0, 0] > 0.0 > (r.W[0, 1] - 0.5)), "delta signs are opposite")
    out["delta_potentiation_dw"] = float(r.W[0, 0])
    out["delta_depression_dw"] = float(r.W[0, 1] - 0.5)

    # ----------------------------------------- local_perceptron STOPPING BUG --
    # The diagnosis as an executable assertion. One feature (x = 1), two
    # classes, weights [0.05, 0.0], target = class 0. The target class already
    # scores positive (0.05) and the other class scores exactly 0:
    #
    #   perceptron: err = [1 - 1, 0 - 0] = [0, 0]        -> ZERO update
    #   delta     : err = [1 - 0.05, 0 - 0.0] = [0.95, 0] -> still learning
    #
    # Same data, same lr, same features, same schedule. The only difference is
    # the error term, and the perceptron's correct-but-unconfident answer is
    # frozen forever with no margin. H-rule in one assertion.
    W0 = np.array([[0.05, 0.0]], dtype=np.float32)
    x1 = np.array([[1.0]], dtype=np.float32)
    y1 = np.array([[1.0, 0.0]], dtype=np.float32)
    rp = Readout(ReadoutConfig(kind="local_perceptron", n_features=1, n_classes=2,
                               lr=0.5, normalize=False, epochs=1, shuffle=False))
    rp.W[:] = W0
    rp.partial_fit(x1, y1)
    rd = Readout(ReadoutConfig(kind="local_delta", n_features=1, n_classes=2,
                               lr=0.5, normalize=False, epochs=1, shuffle=False))
    rd.W[:] = W0
    rd.partial_fit(x1, y1)
    _check(np.array_equal(rp.W, W0),
           "perceptron must apply ZERO correction when every class already "
           "sits on the correct side of the threshold")
    _check_close(float(rd.W[0, 0]) - 0.05, 0.475,
                 "delta applies the graded correction", 1e-6)
    out["perceptron_frozen_dw"] = float(np.abs(rp.W - W0).max())
    out["delta_correction_dw"] = float(rd.W[0, 0] - 0.05)

    # ------------------------------------------------ softmax gradient sign
    # One-hot target on class 0 with both scores equal: p = [0.5, 0.5], so the
    # error on class 0 is +0.5 and on class 1 it is -0.5. Weight for class 0
    # rises, for class 1 falls.
    cfg_s = ReadoutConfig(kind="local_softmax", n_features=1, n_classes=2, lr=1.0,
                          normalize=False, epochs=1, shuffle=False)
    rs = Readout(cfg_s)
    rs.partial_fit(np.array([[1.0]], dtype=np.float32), np.array([[1.0, 0.0]]))
    _check_close(float(rs.W[0, 0]), 0.5, "softmax up on target class")
    _check_close(float(rs.W[0, 1]), -0.5, "softmax down on non-target class")
    _check(float(rs.W[0, 0] > rs.W[0, 1]), "softmax raises target above others")
    out["softmax_dw_target"] = float(rs.W[0, 0])

    # ------------------------------------------------ perceptron two-class swap
    # On a linearly separable 2-class problem the binarised rule must still
    # learn something; this guards against "it is broken everywhere" nonsense.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.5, 0.1, (40, 2)), rng.normal(-0.5, 0.1, (40, 2))]).astype(np.float32)
    y = np.array([0] * 40 + [1] * 40)
    Y = np.eye(2, dtype=np.float32)[y]
    for kind in ("local_perceptron", "local_delta", "local_softmax"):
        r = Readout(ReadoutConfig(kind=kind, n_features=2, n_classes=2, lr=0.1,
                                  normalize=False, epochs=5, seed=0))
        r.partial_fit(X, Y)
        acc = r.score(X, y)
        _check(acc == 1.0, f"{kind} must solve a separable 2-class problem")
        out[f"separable_acc_{kind}"] = acc

    # ---------------------------------- ridge matches the normal equations ---
    # Independent re-derivation of the closed-form solve, written out longhand
    # against a hand-built one-hot target. Catches sign errors, a missing
    # transpose, and a mis-scaled penalty, none of which accuracy alone would.
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])
    Yoh = np.eye(2, dtype=np.float64)[np.array([0, 1, 1, 0])]
    r2 = Readout(ReadoutConfig(kind="ridge", n_features=2, n_classes=2,
                               ridge=0.0, normalize=False))
    r2.partial_fit(A.astype(np.float32), Yoh.astype(np.float32))
    want = np.linalg.solve(A.T @ A, A.T @ Yoh)
    _check(np.all(np.isfinite(r2.W)), "ridge produces finite weights")
    _check_close(float(np.abs(r2.W - want).max()), 0.0,
                 "ridge == normal equations", 1e-5)
    out["ridge_vs_normal_equations_maxabs"] = float(np.abs(r2.W - want).max())

    # A penalty must actually shrink the solution toward zero.
    r3 = Readout(ReadoutConfig(kind="ridge", n_features=2, n_classes=2,
                               ridge=1.0, normalize=False))
    r3.partial_fit(A.astype(np.float32), Yoh.astype(np.float32))
    _check(float(np.abs(r3.W).max()) < float(np.abs(r2.W).max()),
           "a larger ridge must shrink the weights")
    out["ridge_shrinks_weights"] = float(np.abs(r3.W).max())

    # ------------------------------------------- collapsed-code degeneracy ----
    # A constant feature column with normalize=False and ridge=0 makes the Gram
    # singular. The solver must fall back rather than raise: a crashed arm would
    # silently drop out of the comparison and flatter the remaining ones.
    r4 = Readout(ReadoutConfig(kind="ridge", n_features=3, n_classes=2, ridge=0.0,
                               normalize=False))
    Xc = np.ones((6, 3), dtype=np.float32)
    Yc = np.eye(2, dtype=np.float32)[np.array([0, 1, 0, 1, 0, 1])]
    r4.partial_fit(Xc, Yc)
    _check(np.all(np.isfinite(r4.W)), "singular Gram must not produce NaNs")
    out["singular_gram_solve_method"] = 1.0 if r4.diagnostics["solve_method"] else 0.0

    # ------------------------------------------------- determinism given a seed
    a = Readout(ReadoutConfig(kind="local_delta", n_features=4, n_classes=3, lr=0.05,
                              seed=7, normalize=False, epochs=2))
    b_ = Readout(ReadoutConfig(kind="local_delta", n_features=4, n_classes=3, lr=0.05,
                               seed=7, normalize=False, epochs=2))
    Xd = np.random.default_rng(3).normal(size=(30, 4)).astype(np.float32)
    yd = np.random.default_rng(4).integers(0, 3, size=30)
    Yd = np.eye(3, dtype=np.float32)[yd]
    a.partial_fit(Xd, Yd)
    b_.partial_fit(Xd, Yd)
    _check(np.array_equal(a.W, b_.W), "same seed -> bit-identical weights")
    out["determinism"] = 1.0

    if verbose:
        print("brain/readout.py self-check: OK")
        for k, v in out.items():
            print(f"  {k:38s} {v!r}")
    return out


if __name__ == "__main__":
    raise SystemExit(0 if self_check() is not None else 1)
