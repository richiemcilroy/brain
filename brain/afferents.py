"""Frozen afferent projections: how the world gets *into* the cortex.

The problem this module exists to fix
-------------------------------------
``brain/cortex.py`` injects the image with a 1:1 identity map::

    drive[: cfg.n_input] = pix * cfg.gain       # cortex.py

Under that routing, neuron ``i`` sees exactly pixel ``i`` and neurons
``n_input..n_neurons`` receive *nothing at all*. No neuron ever combines two
pixels, so no neuron can be selective for any image feature richer than a single
pixel. That is not a cortical input stage. Cortical afferents are massively
**convergent** (each neuron receives thousands of synapses from a much larger
presynaptic pool) and **divergent** (each afferent axon branches onto many
postsynaptic targets), which is what gives a cortical column its mixed,
overlapping receptive fields.

This module supplies that missing stage as a *fixed random* front end. Nothing
here is trained: the projection is sampled once from ``seed`` and frozen. Only
the downstream readout in ``CortexClassifier`` learns. That is deliberate - it
makes the projection a controlled experimental factor rather than a second
learning system, and it matches the frozen-random-features arm the repo already
uses as a control.

The scaling rule (stated once, used everywhere)
-----------------------------------------------
All kinds multiply the frozen projection by ``gain`` so the *per-neuron current*
stays comparable across kinds:

* ``identity``: ``drive[j] = gain * x[j]`` for ``j < n_input``, else 0.
  Exactly the existing behaviour, bit-for-bit (see the note below).
* ``dense``:  ``drive = gain * (x @ W)`` with ``W[i,j] ~ N(0, 1/n_input)``, so
  each neuron's input is an average of all pixels with unit-variance weights
  and the current RMS is ``gain * RMS(x)`` - the same order as ``identity``.
* ``sparse``: same, but each neuron draws ``fan_in`` distinct pixels and uses
  ``N(0, 1/fan_in)``; the current RMS is again ``gain * RMS(x)``, so fan-in
  changes *which* pixels are combined, not how loud the neuron is. This is what
  makes the fan-in sweep a test of convergence rather than of gain.
* ``conv``:  ``drive = gain * (ReLU(z) + b)`` where ``z`` is a random 5x5-ish
  patch filter (``patch = round(sqrt(fan_in))``) with ``N(0, 1/patch**2)``
  weights, shared across ``n_filters`` random filters and applied at random
  retinotopic positions. Rectification makes it a random-feature CNN front end,
  not a linear map.

``normalize=False`` drops the ``1/fan_in`` normalisation (weights are ``N(0,1)``),
which is offered only so the normalisation itself can be ablated. It is not the
default and it is not recommended: unscaled weights make the current magnitude
scale with ``sqrt(fan_in)`` and silently confound gain with fan-in.

``bias`` adds one constant input per neuron (``b_j ~ N(0, 1/fan_in)``) so a
neuron with a near-zero-weight draw is not silent. Without it, a fraction of
neurons are effectively dead at small fan-in.

Notes for reviewers
-------------------
* ``kind="identity"`` intentionally **ignores** ``normalize`` and ``bias``. Both
  would change its output, and the whole point of the identity arm is to be
  bit-identical to the original ``drive[:n_input] = pix * gain`` line so the
  comparison is apples-to-apples. Use another kind if you want those knobs to do
  something.
* The projection runs on the host in NumPy even when the simulator runs on MLX.
  That is deliberate: a projection whose arithmetic (and therefore whose seed
  stream) changed with the backend would make the identity control and the mixed
  arms incomparable. Measured on an M4 Max for the sizes used here
  (784x900), NumPy is also ~5x faster than MLX at this shape, because a single
  matmul that small does not amortise the GPU dispatch.
* The weights are signed Gaussians. This is a *current* projection, not a
  synaptic matrix: it is not Dale-compliant, and it makes no claim to be. The
  Dale-compliant spiking synapses live in ``brain/connectivity.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["AfferentConfig", "Afferents", "KINDS"]

KINDS = ("identity", "dense", "sparse", "conv")

#: Fixed number of shared patch filters for ``kind="conv"``.
N_CONV_FILTERS = 16


@dataclass
class AfferentConfig:
    """Configuration for the frozen input projection.

    Attributes:
        n_input: Number of input features (pixels) the sensory sheet provides.
        n_neurons: Size of the postsynaptic population the projection targets.
        kind: ``"identity"`` (1:1 routing, the original behaviour),
            ``"dense"`` (all-to-all random), ``"sparse"`` (random ``fan_in``-of-
            ``n_input`` convergent), or ``"conv"`` (random patch filters with
            weight sharing and rectification).
        fan_in: Presynaptic convergence per neuron for ``kind="sparse"``, and
            the patch area (``round(sqrt(fan_in))**2``) for ``kind="conv"``.
            Ignored by ``identity`` and ``dense`` (dense uses ``n_input``).
        gain: Multiplier applied to the whole projection. Kept as a *separate*
            factor from the weight scale so that activity level can be swept
            independently of the mixing pattern (see the matched-activity
            control in ``docs/AFFERENTS.md``).
        normalize: ``True`` -> weights are scaled ``1/sqrt(fan_in)`` so the
            per-neuron current RMS is independent of fan-in. ``False`` -> raw
            ``N(0, 1)`` weights. Ignored by ``identity``.
        seed: RNG seed for the frozen draw.
        bias: Add a per-neuron constant input. Ignored by ``identity`` (which
            must stay bit-exact with the original code).
    """

    n_input: int = 784
    n_neurons: int = 900
    kind: str = "identity"  # "identity" | "dense" | "sparse" | "conv"
    fan_in: int = 64
    gain: float = 2.2
    normalize: bool = True
    seed: int = 0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.n_input <= 0 or self.n_neurons <= 0:
            raise ValueError("n_input and n_neurons must be positive")
        if not np.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("gain must be finite and > 0")
        if self.kind == "identity" and self.n_input > self.n_neurons:
            # Mirrors CortexConfig.__post_init__, which cannot route n_input
            # pixels into a smaller population without dropping some.
            raise ValueError("identity projection needs n_input <= n_neurons")
        if self.kind == "sparse" and not 0 < self.fan_in <= self.n_input:
            raise ValueError("sparse fan_in must satisfy 0 < fan_in <= n_input")
        if self.kind == "conv":
            self._check_conv()

    def _check_conv(self) -> None:
        side = math.isqrt(self.n_input)
        if side * side != self.n_input:
            raise ValueError(
                f"conv needs a square input sheet; n_input={self.n_input} is not a "
                "perfect square (784 = 28x28 is)"
            )
        patch = int(round(math.sqrt(self.fan_in)))
        if patch < 2 or patch > side:
            raise ValueError(f"conv patch {patch} (from fan_in={self.fan_in}) must be in [2, {side}]")


class Afferents:
    """A frozen random projection from an input sheet to dendritic current.

    The projection is sampled once from ``cfg.seed`` and never trained. Call
    :meth:`project` per sample to get the ``(n_neurons,)`` float32 current that
    a caller feeds to ``Brain.step(external_dend=...)``.
    """

    def __init__(self, cfg: AfferentConfig, backend: Any = None):
        self.cfg = cfg
        # Accepted for interface parity with the rest of the package. The frozen
        # projection is computed on the host in NumPy (see module docstring);
        # ``backend`` is recorded so callers can inspect the mismatch.
        self.be = backend
        self._draw()

    # ------------------------------------------------------------ construction
    def _draw(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        self.W: np.ndarray | None = None        # identity: None (no coefficients)
        self.idx: np.ndarray | None = None      # sparse/conv: (n_neurons, fan_in)
        self.conv_W: np.ndarray | None = None   # conv: (n_filters, patch*patch)
        self.conv_f: np.ndarray | None = None   # conv: filter id per neuron

        if cfg.kind == "identity":
            # Bit-exact reproduction of cortex.py's drive construction. No
            # weights are stored: neuron j is wired to pixel j with unit gain.
            self.n_connections = int(cfg.n_input)
            self.n_scalars = int(cfg.n_input)
            return

        if cfg.kind == "dense":
            scale = 1.0 / math.sqrt(cfg.n_input) if cfg.normalize else 1.0
            self.W = (rng.standard_normal((cfg.n_input, cfg.n_neurons)) * scale).astype(np.float32)
            self.n_connections = int(cfg.n_input) * int(cfg.n_neurons)
            self.n_scalars = self.n_connections
        elif cfg.kind == "sparse":
            k = int(cfg.fan_in)
            scale = 1.0 / math.sqrt(k) if cfg.normalize else 1.0
            # Distinct presynaptic partners per neuron: convergent, not a
            # dropout mask over a dense matrix.
            self.idx = np.empty((cfg.n_neurons, k), dtype=np.int32)
            self.W = (rng.standard_normal((cfg.n_neurons, k)) * scale).astype(np.float32)
            for j in range(cfg.n_neurons):
                self.idx[j] = rng.choice(cfg.n_input, size=k, replace=False)
            self.n_connections = int(cfg.n_neurons) * k
            self.n_scalars = self.n_connections
        else:  # conv
            side = math.isqrt(cfg.n_input)
            patch = int(round(math.sqrt(cfg.fan_in)))
            positions = side - patch + 1
            scale = 1.0 / math.sqrt(patch * patch) if cfg.normalize else 1.0
            n_filters = N_CONV_FILTERS
            self.patch_side = patch
            self.n_filters = n_filters
            self.conv_W = (rng.standard_normal((n_filters, patch * patch)) * scale).astype(np.float32)
            # Gather indices for every (row, col) patch origin, flattened.
            base = (np.arange(side * side, dtype=np.int32).reshape(side, side))
            origins_r = np.repeat(np.arange(positions), positions)
            origins_c = np.tile(np.arange(positions), positions)
            offs_r = np.repeat(np.arange(patch), patch)
            offs_c = np.tile(np.arange(patch), patch)
            flat = np.empty((positions * positions, patch * patch), dtype=np.int32)
            for p, (r0, c0) in enumerate(zip(origins_r, origins_c)):
                flat[p] = base[r0 + offs_r, c0 + offs_c]
            n_pos = positions * positions
            total = n_pos * n_filters
            replace = cfg.n_neurons > total
            if replace:
                choice = rng.integers(0, total, size=cfg.n_neurons)
            else:
                choice = rng.choice(total, size=cfg.n_neurons, replace=False)
            self.idx = flat[choice // n_filters]          # (n_neurons, patch*patch)
            self.conv_f = (choice % n_filters).astype(np.int32)
            self.n_connections = int(cfg.n_neurons) * patch * patch
            self.n_scalars = int(n_filters * patch * patch)   # filters are shared

        # Per-neuron constant input, scaled like the weights.
        self.b = None
        if cfg.bias:
            b_scale = 1.0 / math.sqrt(max(1, self.n_connections // max(1, cfg.n_neurons))) \
                if cfg.normalize else 1.0
            self.b = (rng.standard_normal(cfg.n_neurons) * b_scale).astype(np.float32)
            self.n_scalars += int(cfg.n_neurons)

    # -------------------------------------------------------------- projection
    def project(self, x: Any) -> np.ndarray:
        """Map one input vector to ``(n_neurons,)`` float32 dendritic current.

        The frozen projection, i.e. no part of this is trained.
        """
        cfg = self.cfg
        pix = np.asarray(x, dtype=np.float32).reshape(-1)
        if pix.size != cfg.n_input:
            raise ValueError(f"expected {cfg.n_input} inputs, got {pix.size}")
        out = np.zeros(cfg.n_neurons, dtype=np.float32)

        if cfg.kind == "identity":
            # Must stay bit-identical to cortex.py: `drive[:n_input] = pix * gain`.
            out[: cfg.n_input] = pix * cfg.gain
            return out

        if cfg.kind in ("dense", "sparse"):
            if cfg.kind == "dense":
                self._eval(self.W, pix)
                acc = pix @ self.W
            else:
                acc = (np.take(pix, self.idx) * self.W).sum(axis=1)
            if self.b is not None:
                acc = acc + self.b
            np.multiply(acc, cfg.gain, out=out)
            return out

        # conv: shared random filters, rectified, sampled at random positions.
        z = (np.take(pix, self.idx) * self.conv_W[self.conv_f]).sum(axis=1)
        z = np.maximum(z, 0.0)
        if self.b is not None:
            z = z + self.b
        np.multiply(z, cfg.gain, out=out)
        return out

    def _eval(self, *arrays: Any) -> None:
        """Force lazy-backend evaluation before any host read (MLX trap)."""
        be = self.be
        if be is not None and getattr(be, "is_mlx", False):
            be.eval(*arrays)

    # ------------------------------------------------------------------ stats
    @property
    def n_weights(self) -> int:
        """Stored scalar coefficients of the frozen projection.

        Counts each stored number once: weight-shared conv filters count once,
        the bias vector counts per neuron, and ``identity`` counts its
        ``n_input`` explicit unit gains (there are no other coefficients).
        Use :attr:`n_connections` for the total input->neuron connection count.
        """
        return int(self.n_scalars)

    @property
    def n_connections(self) -> int:
        """Total input->neuron connections (sum of fan-in over neurons)."""
        return int(self._n_connections)

    @n_connections.setter
    def n_connections(self, value: int) -> None:
        self._n_connections = int(value)

    @property
    def fan_in_effective(self) -> float:
        """Mean presynaptic convergence per postsynaptic neuron."""
        return self.n_connections / max(1, self.cfg.n_neurons)

    @property
    def divergence(self) -> float:
        """Mean number of postsynaptic neurons reached by one input unit."""
        return self.n_connections / max(1, self.cfg.n_input)

    @property
    def macs_per_sample(self) -> int:
        """Multiply-accumulates the projection performs per sample.

        ``identity`` is 0: it is a routing copy, not a multiply-accumulate.
        """
        if self.cfg.kind == "identity":
            return 0
        return int(self.n_connections)

    def rebuild(self, seed: int) -> None:
        """Re-sample the frozen projection from a new seed (identity: no-op)."""
        self.cfg.seed = int(seed)
        self._draw()

    def stats(self) -> dict[str, Any]:
        """Plain-dict summary for JSON result files."""
        return {
            "kind": self.cfg.kind,
            "n_input": int(self.cfg.n_input),
            "n_neurons": int(self.cfg.n_neurons),
            "gain": float(self.cfg.gain),
            "fan_in": int(self.cfg.fan_in),
            "normalize": bool(self.cfg.normalize),
            "bias": bool(self.cfg.bias),
            "seed": int(self.cfg.seed),
            "n_weights": int(self.n_weights),
            "n_connections": int(self.n_connections),
            "fan_in_effective": float(self.fan_in_effective),
            "divergence": float(self.divergence),
            "macs_per_sample": int(self.macs_per_sample),
            "patch_side": int(getattr(self, "patch_side", 0)),
            "n_filters": int(getattr(self, "n_filters", 0)),
        }


# ------------------------------------------------------------------- self-check
def _self_check() -> None:
    """Verify the identity arm is bit-exact and the other kinds are sane."""
    import sys
    import time
    from pathlib import Path

    if __package__:
        from .cortex import CortexConfig  # local import: avoids a circular import
    else:  # `python3 brain/afferents.py` from the repo root
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from brain.cortex import CortexConfig

    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    x = rng.random(784).astype(np.float32)
    n_neurons = 900

    # 1. identity is bit-identical to cortex.py's drive construction.
    for gain in (2.2, 6.0, 12.0):
        ccfg = CortexConfig(n_neurons=n_neurons, n_classes=10, gain=gain, seed=0)
        ref = np.zeros(ccfg.n_neurons, dtype=np.float32)
        ref[: ccfg.n_input] = x * ccfg.gain
        got = Afferents(AfferentConfig(n_neurons=n_neurons, kind="identity", gain=gain)).project(x)
        assert got.shape == (n_neurons,) and got.dtype == np.float32
        assert np.array_equal(got, ref), "identity must be bit-exact with cortex.py"
        assert got[784:].sum() == 0.0, "identity must leave neurons >= n_input silent"
    print("identity projection: bit-exact vs cortex.py drive (3 gains)")

    # 2. sparse has exactly fan_in distinct partners per neuron.
    a = Afferents(AfferentConfig(n_neurons=64, kind="sparse", fan_in=32, seed=0))
    assert a.idx.shape == (64, 32)
    assert all(len(set(row.tolist())) == 32 for row in a.idx), "sparse partners must be distinct"
    assert a.n_connections == 64 * 32

    # 3. scaling rule: dense and sparse currents are the same order at equal gain.
    d = Afferents(AfferentConfig(n_neurons=64, kind="dense", seed=1)).project(x)
    s = Afferents(AfferentConfig(n_neurons=64, kind="sparse", fan_in=64, seed=1)).project(x)
    ratio = float(np.std(s) / np.std(d))
    assert 0.5 < ratio < 2.0, f"dense/sparse current magnitudes not comparable: {ratio:.3f}"
    print(f"dense vs sparse current RMS ratio at equal gain: {ratio:.3f} (1.0 = identical scale)")

    # 4. conv: local receptive fields only, and rectified (non-negative pre-bias).
    c = Afferents(AfferentConfig(n_neurons=32, kind="conv", fan_in=25, seed=2))
    assert c.patch_side == 5 and c.idx.shape == (32, 25)
    rows = np.unique(c.idx[0] // 28), np.unique(c.idx[0] % 28)
    assert np.ptp(rows[0]) < 5 and np.ptp(rows[1]) < 5, "conv patches must be retinotopic"
    print(f"conv: patch {c.patch_side}x{c.patch_side}, {c.n_filters} shared filters, "
          f"{c.n_connections} connections, {c.n_weights} stored scalars")

    # 5. determinism + rebuild.
    a1 = Afferents(AfferentConfig(n_neurons=64, kind="sparse", fan_in=16, seed=5)).project(x)
    a2 = Afferents(AfferentConfig(n_neurons=64, kind="sparse", fan_in=16, seed=5)).project(x)
    assert np.array_equal(a1, a2), "same seed must give identical projections"
    a3 = Afferents(AfferentConfig(n_neurons=64, kind="sparse", fan_in=16, seed=6)).project(x)
    assert not np.array_equal(a1, a3), "different seed must give a different projection"
    a4 = Afferents(AfferentConfig(n_neurons=64, kind="sparse", fan_in=16, seed=5))
    a4.rebuild(6)
    assert np.array_equal(a4.project(x), a3), "rebuild(seed) must equal a fresh seed"
    print("determinism: same seed identical, different seed differs, rebuild(seed) matches")

    # 6. bad configs fail loudly.
    bad = (
        dict(kind="nope"), dict(kind="sparse", fan_in=0),
        dict(kind="identity", n_input=1000, n_neurons=900),
        dict(kind="conv", n_input=784, fan_in=1),          # patch < 2
        dict(kind="conv", n_input=98, fan_in=25),             # non-square sheet
        dict(kind="dense", gain=0.0), dict(kind="sparse", fan_in=2000),
    )
    for kwargs in bad:
        try:
            Afferents(AfferentConfig(**kwargs))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    print("config validation: invalid kinds/fan-ins/identity overflow rejected")
    print(f"all afferent checks passed in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
