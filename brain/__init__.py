"""A locally-runnable, biologically-grounded cortical simulation substrate.

The package implements an event-driven spiking network with:

* two-compartment neurons (soma + nonlinear distal dendrite),
* adaptive thresholds (ALIF-style temporal memory),
* conductance-style synaptic currents with axonal delays,
* three-factor plasticity (eligibility trace x neuromodulation),
* divisive inhibition / k-WTA sparse coding and homeostatic regulation.

Compute runs on MLX (Metal GPU) when available, with a NumPy fallback so the
package is portable and every result can be re-derived on CPU.
"""

from __future__ import annotations

from .backend import Backend, get_backend
from .neurons import NeuronConfig, NeuronState
from .connectivity import SynapseConfig, Synapses
from .plasticity import PlasticityConfig, ThreeFactorPlasticity
from .simulator import Brain, SimConfig

__all__ = [
    "Backend",
    "get_backend",
    "NeuronConfig",
    "NeuronState",
    "SynapseConfig",
    "Synapses",
    "PlasticityConfig",
    "ThreeFactorPlasticity",
    "Brain",
    "SimConfig",
]

__version__ = "0.1.0"
