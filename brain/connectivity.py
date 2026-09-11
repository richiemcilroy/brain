"""Sparse synaptic connectivity with axonal delays.

Storage is a fixed fan-out ("block CSR") layout: every pre-synaptic neuron owns
exactly ``k_out`` targets. This matters for performance. Real cortical
connectivity is sparse and irregular, but *variable-length* adjacency forces
either padding or per-spike gather loops, both of which defeat vectorisation on
Metal. A fixed fan-out keeps every spike's delivery a single rectangular gather
of shape ``(n_spikes, k_out)``, which is what makes the measured throughput in
this repo possible.

Biologically this is defensible rather than merely convenient: with a large
post-synaptic pool and low connection probability, in-degree concentrates
around its mean, so a fixed expected fan-out is a reasonable first-order model.

Dale's principle is enforced: every outgoing synapse of a neuron has the same
sign, set by whether that neuron is excitatory or inhibitory.

Delays are modelled with a ring buffer, so a spike sent with delay ``d``
arrives at step ``t + d``. Delays are integers in ``[1, max_delay]`` ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import Backend


@dataclass
class SynapseConfig:
    n_pre: int
    n_post: int
    k_out: int = 64
    excitatory_frac: float = 0.8
    w_exc: float = 0.08
    w_inh: float = 0.32
    w_noise: float = 0.5
    w_max_exc: float = 0.4
    w_max_inh: float = 0.8
    delay_min: int = 1
    delay_max: int = 4
    structure: str = "random"
    local_span: float = 0.05
    seed: int = 0

    @property
    def n_synapses(self) -> int:
        return self.n_pre * self.k_out

    def __post_init__(self) -> None:
        if self.k_out > self.n_post:
            raise ValueError("k_out cannot exceed n_post")
        if not 0.0 <= self.excitatory_frac <= 1.0:
            raise ValueError("excitatory_frac must be in [0, 1]")
        if self.delay_min < 1:
            raise ValueError("axonal delays must be >= 1 ms")
        if self.delay_max < self.delay_min:
            raise ValueError("delay_max must be >= delay_min")
        if self.structure not in ("random", "local"):
            raise ValueError("structure must be random|local")


class Synapses:
    """Fixed fan-out synaptic matrix with signed weights and delays."""

    def __init__(self, cfg: SynapseConfig, be: Backend):
        self.cfg = cfg
        self.be = be
        be.seed(cfg.seed)
        n_pre, n_post, k = cfg.n_pre, cfg.n_post, cfg.k_out

        n_exc = int(round(n_pre * cfg.excitatory_frac))
        is_exc = be.arange(n_pre, dtype=be.idx_dtype) < n_exc
        exc_col = is_exc.reshape((n_pre, 1))

        if cfg.structure == "local":
            span = max(k, int(cfg.local_span * n_post))
            centre = (be.arange(n_pre) / max(1, n_pre - 1)) * (n_post - 1)
            offs = be.randint(0, span, (n_pre, k)).astype(be.float_dtype)
            targets = be.clip(
                centre.reshape((n_pre, 1)) + offs - span / 2.0, 0.0, n_post - 1
            ).astype(be.idx_dtype)
        else:
            targets = be.randint(0, n_post, (n_pre, k))

        mag = be.abs(be.normal((n_pre, k), 1.0))
        base = be.where(exc_col, be.full((n_pre, k), cfg.w_exc),
                        be.full((n_pre, k), cfg.w_inh))
        weights = base * (1.0 + cfg.w_noise * mag)
        weights = be.where(exc_col, weights, -weights)

        delays = be.randint(cfg.delay_min, cfg.delay_max + 1, (n_pre, k))

        self.targets = targets
        self.weights = weights.astype(be.float_dtype)
        self.delays = delays
        self.is_exc = is_exc
        be.eval(self.targets, self.weights, self.delays)

    @property
    def n_synapses(self) -> int:
        return self.cfg.n_synapses

    def weight_bounds(self) -> tuple[Any, Any]:
        """Per-synapse (min, max) allowed weights, respecting sign and Dale."""
        cfg, be = self.cfg, self.be
        exc = self.is_exc.reshape((cfg.n_pre, 1))
        shape = (cfg.n_pre, cfg.k_out)
        lo = be.where(exc, be.zeros(shape), be.full(shape, -cfg.w_max_inh))
        hi = be.where(exc, be.full(shape, cfg.w_max_exc), be.zeros(shape))
        return lo, hi

    def clamp_weights(self) -> None:
        lo, hi = self.weight_bounds()
        w = self.weights
        w = self.be.where(w < lo, lo, w)
        w = self.be.where(w > hi, hi, w)
        self.weights = w

    def mean_abs_weight(self) -> float:
        tot = self.be.sum(self.be.abs(self.weights))
        return float(self.be.to_numpy(tot)) / self.n_synapses

    def sign_violations(self) -> int:
        """Count synapses violating Dale's principle (must be 0)."""
        be = self.be
        exc = self.is_exc.reshape((self.cfg.n_pre, 1))
        bad = be.logical_or(
            be.logical_and(exc, self.weights < 0.0),
            be.logical_and(be.logical_not(exc), self.weights > 0.0),
        )
        return int(be.to_numpy(be.sum(be.astype(bad, be.float_dtype))))

    def stats(self) -> dict[str, float]:
        mean_exc = float(self.be.to_numpy(
            self.be.sum(self.be.astype(self.is_exc, self.be.float_dtype))
        )) / self.cfg.n_pre
        return {
            "n_synapses": float(self.n_synapses),
            "mean_abs_weight": self.mean_abs_weight(),
            "sign_violations": float(self.sign_violations()),
            "excitatory_fraction": mean_exc,
        }


class DelayBuffer:
    """Ring buffer delivering currents to the correct future timestep."""

    def __init__(self, n_post: int, max_delay: int, be: Backend):
        self.n_post = n_post
        self.depth = max_delay + 1
        self.be = be
        self.flat = be.zeros((self.depth * n_post,))

    def read_and_clear(self, t: int) -> Any:
        """Return the (n_post,) current arriving now, zeroing that slot."""
        be, n, d = self.be, self.n_post, self.depth
        slot = (t % d) * n
        idx = be.arange(slot, slot + n, dtype=be.idx_dtype)
        cur = be.take(self.flat, idx)
        self.flat = be.scatter_add(self.flat, idx, -cur)
        return cur

    def schedule(self, t: int, targets: Any, delays: Any, weights: Any) -> None:
        """Scatter a spike batch so it arrives at ``t + delay``."""
        be, n, d = self.be, self.n_post, self.depth
        ring = (t + delays) % d
        flat_idx = be.reshape(ring * n + targets, (-1,)).astype(be.idx_dtype)
        self.flat = be.scatter_add(self.flat, flat_idx, be.reshape(weights, (-1,)))

    def pending(self) -> float:
        return float(self.be.to_numpy(self.be.sum(self.be.abs(self.flat))))
