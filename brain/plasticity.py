"""Three-factor plasticity: local Hebbian eligibility x neuromodulation.

The learning rule
-----------------
Credit assignment in this substrate is local. Each synapse maintains an
*eligibility* trace built only from quantities available at that synapse - the
recent activity of its pre- and post-synaptic partners::

    e_ij  <-  decay * e_ij  +  A_plus * s_post_j * x_pre_i  -  A_minus * y_post_j

and the weight change is the product of that eligibility with a *third factor* -
a scalar neuromodulatory signal M(t) broadcast to the population::

    dw_ij  =  lr * M(t) * e_ij

This is the standard three-factor / R-STDP formulation (Frémaux & Gerstner 2015;
Gerstner et al. 2018). It is the mechanistic answer to "how do you learn without
backprop": there is no backward pass and no global weight transport. The
eligibility trace is local, and the only non-local quantity is a single scalar
that is broadcast identically to every synapse - orders of magnitude less
information than a backward pass carries.

Two important caveats, stated up front so results are not over-claimed
---------------------------------------------------------------------
1. **Local rules do not simply match backprop at scale.** Bartunov et al. 2018
   (NeurIPS) measured local rules (feedback alignment, target propagation)
   reaching 93-99% top-1 error on ImageNet where backprop reaches ~71%. Small
   benchmark wins for local rules are real, but they do not automatically
   transfer. Any claim here is a claim about continual learning at small scale.
2. **Two-factor STDP alone diverges in recurrent nets.** Feedback inhibition and
   homeostasis are not optional extras; without them weights run away. We
   therefore pair the rule with synaptic scaling and weight bounds.

``eligibility="dense"`` keeps an explicit ``(n_pre, k_out)`` eligibility matrix
and supports reward delayed relative to the Hebbian event. It costs O(n_pre *
k_out) per step, so it is intended for the learning-scale networks in this
repo. ``eligibility="off"`` applies the rule immediately at spiking synapses
only, which is O(active) and therefore compatible with a million-neuron sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import Backend


@dataclass
class PlasticityConfig:
    a_plus: float = 0.010
    a_minus: float = 0.012
    tau_pre: float = 20.0
    tau_post: float = 20.0
    tau_elig: float = 60.0
    lr: float = 1.0
    dt: float = 1.0
    eligibility: str = "dense"  # "dense" | "off"
    enabled: bool = True
    w_min: float = 0.0
    w_max: float = 0.4
    # homeostatic synaptic scaling
    homeostasis: bool = True
    homeo_every: int = 200
    homeo_rate: float = 0.02
    target_mean_w: float = 0.08

    def __post_init__(self) -> None:
        if self.eligibility not in ("dense", "off"):
            raise ValueError("eligibility must be dense|off")


class ThreeFactorPlasticity:
    """Maintains pre/post traces and applies three-factor weight updates."""

    def __init__(self, cfg: PlasticityConfig, n_pre: int, n_post: int, k_out: int,
                 be: Backend):
        self.cfg = cfg
        self.be = be
        self.n_pre, self.n_post, self.k_out = n_pre, n_post, k_out
        self.x_pre = be.zeros((n_pre,))
        self.y_post = be.zeros((n_post,))
        self.elig = be.zeros((n_pre, k_out)) if cfg.eligibility == "dense" else None
        self._row = be.arange(k_out, dtype=be.idx_dtype).reshape((1, k_out))
        self._steps = 0
        self.stats = {"potentiated": 0, "depressed": 0, "updates": 0}

    # ------------------------------------------------------------ trace decay
    def decay(self) -> None:
        cfg, be = self.cfg, self.be
        self.x_pre = self.x_pre * (1.0 - cfg.dt / cfg.tau_pre)
        self.y_post = self.y_post * (1.0 - cfg.dt / cfg.tau_post)
        if self.elig is not None:
            self.elig = self.elig * (1.0 - cfg.dt / cfg.tau_elig)

    def record_pre_spikes(self, spike_idx: Any) -> None:
        be = self.be
        one = be.ones(spike_idx.shape)
        self.x_pre = be.scatter_add(self.x_pre, spike_idx.astype(be.idx_dtype), one)

    def record_post_spikes(self, post_mask: Any) -> None:
        be = self.be
        self.y_post = be.scatter_add(
            self.y_post, be.arange(self.n_post, dtype=be.idx_dtype),
            be.astype(post_mask, be.float_dtype),
        )

    # -------------------------------------------------------- hebbian update
    def hebbian_delta(self, spike_idx: Any, targets: Any, post_mask: Any) -> Any:
        """Per-synapse Hebbian change for the synapses that just fired.

        Ordering matters and is subtle. ``x_pre`` already includes the current
        pre-synaptic spike, while ``y_post`` must be the post-synaptic trace
        from *before* this step's post spikes are recorded. If the current post
        spike were also folded into ``y_post``, a coincident pre+post pair
        would contribute to the LTP term *and* the LTD term at once, and with
        ``a_minus > a_plus`` a perfectly correlated pair would depress instead
        of potentiate. The caller therefore applies this rule before
        ``record_post_spikes``.
        """
        cfg, be = self.cfg, self.be
        x_pre = be.take(self.x_pre, spike_idx)
        y_post = be.take(self.y_post, targets)
        s_post = be.astype(be.take(post_mask, targets), be.float_dtype)
        return cfg.a_plus * s_post * x_pre.reshape((-1, 1)) - cfg.a_minus * y_post

    def apply(self, spike_idx: Any, targets: Any, post_mask: Any,
              synapses: Any, neuromod: float | Any) -> None:
        """Run one plasticity step at the synapses whose pre-neuron just spiked.

        This method is deliberately self-contained: it records the pre-synaptic
        spikes, evaluates the rule against the *pre-update* post-synaptic trace,
        writes the weights, and only then records this step's post-synaptic
        spikes. Doing the ordering here rather than in the caller matters -
        if the post trace is advanced before the rule is evaluated, a perfectly
        coincident pre+post pair contributes to the LTP *and* the LTD term and
        (because a_minus > a_plus) depression wins. Handing that footgun to
        callers is how a learning rule silently inverts, so it is not exposed.

        When ``neuromod`` is zero the eligibility traces are still updated and
        the traces still advance; only the weight write is skipped. Skipping the
        trace updates instead would desynchronise the traces from the spikes.
        """
        if not self.cfg.enabled:
            return
        cfg, be = self.cfg, self.be

        # 1. pre-synaptic trace INCLUDING the current spikes (needed for LTP)
        self.record_pre_spikes(spike_idx)

        # 2. evaluate the rule against the post trace from *before* this step
        delta = self.hebbian_delta(spike_idx, targets, post_mask)
        n_spk = delta.shape[0]
        flat_idx = (spike_idx.reshape((-1, 1)) * self.k_out + self._row)
        flat_idx = be.reshape(flat_idx, (-1,)).astype(be.idx_dtype)
        flat_delta = be.reshape(delta, (-1,))

        if self.elig is not None:
            self.elig = be.scatter_add(
                be.reshape(self.elig, (-1,)), flat_idx, flat_delta
            ).reshape((self.n_pre, self.k_out))
            apply_val = be.take(be.reshape(self.elig, (-1,)), flat_idx)
        else:
            apply_val = flat_delta

        # 3. post-synaptic trace for the NEXT step's LTD term
        self.record_post_spikes(post_mask)

        mod = neuromod if isinstance(neuromod, float) else be.item(neuromod)
        if mod == 0.0:
            self._steps += 1
            return
        dw = cfg.lr * mod * apply_val
        flat_w = be.reshape(synapses.weights, (-1,))

        # Dale's principle means an inhibitory synapse is a NEGATIVE weight, and
        # potentiating inhibition must make that weight MORE negative. The
        # Hebbian term is therefore a magnitude change that must be signed by
        # the synapse's own sign. Omitting this silently turns Hebbian
        # potentiation into a weakening of every inhibitory connection.
        sign = be.where(be.take(flat_w, flat_idx) < 0.0, -1.0, 1.0)
        dw = dw * sign

        flat_w = be.scatter_add(flat_w, flat_idx, dw)
        lo, hi = synapses.weight_bounds()
        w = flat_w.reshape((self.n_pre, self.k_out))
        w = be.where(w < lo, lo, w)
        w = be.where(w > hi, hi, w)
        synapses.weights = w

        # ``dw`` is already sign-corrected, so its sign now means "connection
        # got stronger" (+) or "weaker" (-). Comparing it against the *pre*
        # update weight would be meaningless, which is why the previous version
        # never incremented "depressed".
        n_up = be.sum(be.astype(dw != 0.0, be.float_dtype))
        n_pot = be.sum(be.astype(dw > 0.0, be.float_dtype))
        n_dep = be.sum(be.astype(dw < 0.0, be.float_dtype))
        self.stats["updates"] += int(be.to_numpy(n_up))
        self.stats["potentiated"] += int(be.to_numpy(n_pot))
        self.stats["depressed"] += int(be.to_numpy(n_dep))
        self._steps += 1

    # ---------------------------------------------------------- homeostasis
    def maybe_homeostasis(self, synapses: Any) -> bool:
        """Periodic synaptic scaling. Returns True if applied."""
        if not (self.cfg.enabled and self.cfg.homeostasis):
            return False
        if self.cfg.homeo_every <= 0 or self._steps % self.cfg.homeo_every:
            return False
        be, cfg = self.be, self.cfg
        w = synapses.weights
        mean_abs = be.sum(be.abs(w)) / float(synapses.n_synapses)
        scale = 1.0 + cfg.homeo_rate * (cfg.target_mean_w / (mean_abs + 1e-9) - 1.0)
        scale = float(min(max(scale, 0.5), 2.0))
        synapses.weights = w * scale
        synapses.clamp_weights()
        return True

    def reset_traces(self) -> None:
        be = self.be
        self.x_pre = be.zeros((self.n_pre,))
        self.y_post = be.zeros((self.n_post,))
        if self.elig is not None:
            self.elig = be.zeros((self.n_pre, self.k_out))

    def summary(self) -> dict[str, float]:
        return {
            "potentiated": float(self.stats["potentiated"]),
            "depressed": float(self.stats["depressed"]),
            "weight_updates": float(self.stats["updates"]),
        }
