"""Two-compartment neuron model: adaptive soma + nonlinear distal dendrite.

Why this model, and not a point LIF
-----------------------------------
Two independent findings motivate the extra compartment:

1. **Dendritic Ca2+ action potentials are *tuned*, not monotonic.** Gidon et al.
   2020 (Science 367:83) showed that human L2/3 pyramidal dendrites emit dCaAPs
   that peak for a preferred input magnitude and *attenuate* for stronger input.
   A non-monotonic subunit means a single neuron can solve linearly
   non-separable (XOR-class) problems that a point neuron cannot.
2. **Adaptive thresholds provide temporal memory.** Bellec et al. 2020
   (Nat Commun 11:3625) showed LSNNs with adaptive neurons acquire LSTM-like
   temporal capacity, which plain LIF units lack.

The dendritic nonlinearity is therefore the *causal hypothesis* under test, and
``dend_mode`` makes it directly ablatable (``dcaap`` -> ``linear``). If the
ablation performs as well as the full model, the dendritic claim is false.

Equations (explicit Euler, dt in ms)
------------------------------------
soma::

    dv_s/dt = ( -(v_s - E_L) + I_soma + g_d * phi(v_d) - adapt ) / tau_soma
    dadapt/dt = -adapt / tau_adapt              ( + adapt_inc on each spike )

dendrite::

    dv_d/dt = ( -(v_d - E_d) + I_dend ) / tau_dend

activation (``dcaap``)::

    phi(x) = (u/s) * exp(1 - u/s),  u = max(x, 0)

which peaks at ``u = s`` with value 1 and decays for larger input. The ablated
``linear`` mode uses ``phi(x) = u/s`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backend import Backend


@dataclass
class NeuronConfig:
    """Parameters for a population of two-compartment neurons."""

    n: int
    tau_soma: float = 20.0
    tau_dend: float = 10.0
    tau_adapt: float = 100.0
    e_leak: float = 0.0
    e_dend: float = 0.0
    v_reset: float = 0.0
    v_thresh: float = 1.0
    adapt_base: float = 0.0
    adapt_inc: float = 0.05
    dend_scale: float = 1.0
    dend_gain: float = 1.0
    dend_mode: str = "dcaap"
    refractory_ms: float = 2.0
    noise_std: float = 0.05
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.dend_mode not in ("dcaap", "linear", "none"):
            raise ValueError(f"dend_mode must be dcaap|linear|none, got {self.dend_mode!r}")
        if self.n <= 0:
            raise ValueError("n must be positive")


class NeuronState:
    """Mutable per-neuron state arrays (structure-of-arrays layout)."""

    __slots__ = ("cfg", "be", "v_soma", "v_dend", "adapt", "refrac",
                 "last_spike", "spike_count", "v_pre_reset")

    def __init__(self, cfg: NeuronConfig, be: Backend, *, init_noise: float = 0.0):
        self.cfg = cfg
        self.be = be
        n = cfg.n
        self.v_soma = be.full((n,), cfg.e_leak)
        if init_noise:
            self.v_soma = self.v_soma + be.normal((n,), init_noise)
        self.v_dend = be.full((n,), cfg.e_dend)
        self.adapt = be.full((n,), cfg.adapt_base)
        self.refrac = be.zeros((n,))
        self.last_spike = be.full((n,), -1e9)
        self.spike_count = be.zeros((n,))
        self.v_pre_reset = be.full((n,), cfg.e_leak)
        be.eval(self.v_soma, self.v_dend, self.adapt, self.refrac)

    # ------------------------------------------------------------------ math
    def dendritic_activation(self) -> Any:
        """Apply the dendritic nonlinearity to the current dendritic potential."""
        cfg = self.cfg
        if cfg.dend_mode == "none":
            return self.be.zeros((cfg.n,))
        u = self.be.maximum(self.v_dend - cfg.e_dend, 0.0)
        if cfg.dend_mode == "linear":
            return u / cfg.dend_scale
        z = u / cfg.dend_scale
        # Non-monotonic (tuned) dCaAP-like response, peak value 1 at z == 1.
        return z * self.be.exp(1.0 - z)

    # ---------------------------------------------------------------- update
    def update(self, i_soma: Any, i_dend: Any) -> Any:
        """Advance one timestep. Returns a boolean spike mask."""
        cfg, be = self.cfg, self.be
        dt = cfg.dt

        # --- dendrite
        self.v_dend = self.v_dend + (dt / cfg.tau_dend) * (
            -(self.v_dend - cfg.e_dend) + i_dend
        )
        dend_drive = cfg.dend_gain * self.dendritic_activation()

        # --- adaptation and refractory countdown
        self.adapt = self.adapt * (1.0 - dt / cfg.tau_adapt)
        self.refrac = be.maximum(self.refrac - dt, 0.0)

        # --- soma
        drive = i_soma
        if cfg.noise_std > 0.0:
            drive = drive + be.normal((cfg.n,), cfg.noise_std)
        self.v_soma = self.v_soma + (dt / cfg.tau_soma) * (
            -(self.v_soma - cfg.e_leak) + drive + dend_drive - self.adapt
        )

        blocked = self.refrac > 0.0
        spikes = be.logical_and(self.v_soma >= cfg.v_thresh, be.logical_not(blocked))

        # Keep the potential that was actually thresholded. k-WTA must rank by
        # this, not by the post-reset v_soma, or the neurons that just fired
        # (already reset to rest) look like the *weakest* competitors and are
        # the first to be suppressed.
        self.v_pre_reset = self.v_soma

        # --- reset spiked neurons
        self.v_soma = be.where(spikes, cfg.v_reset, self.v_soma)
        self.adapt = be.where(spikes, self.adapt + cfg.adapt_inc, self.adapt)
        self.refrac = be.where(spikes, cfg.refractory_ms, self.refrac)
        self.spike_count = self.spike_count + be.astype(spikes, be.float_dtype)
        return spikes

    # ------------------------------------------------------------- utilities
    def reset(self, *, keep_weights: bool = True) -> None:
        cfg, be = self.cfg, self.be
        self.v_soma = be.full((cfg.n,), cfg.e_leak)
        self.v_dend = be.full((cfg.n,), cfg.e_dend)
        self.adapt = be.full((cfg.n,), cfg.adapt_base)
        self.refrac = be.zeros((cfg.n,))
        self.last_spike = be.full((cfg.n,), -1e9)

    def mean_rate_hz(self, steps: int) -> float:
        be = self.be
        total = be.sum(self.spike_count)
        n = self.cfg.n
        duration_s = steps * self.cfg.dt / 1000.0
        return float(be.to_numpy(total)) / n / duration_s


def xor_reference_table() -> list[tuple[float, float, float]]:
    """Ground truth for the single-neuron XOR test used in the test-suite."""
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
