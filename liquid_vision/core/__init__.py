"""Core liquid neural network implementations."""

import warnings

# Graceful imports with progressive fallbacks
try:
    from .liquid_neurons import LiquidNeuron, LiquidNet, register_neuron
    _TORCH_AVAILABLE = True
    _IMPLEMENTATION = "pytorch"
except ImportError:
    try:
        from .torch_fallback import LiquidNeuronFallback as LiquidNeuron, LiquidNetFallback as LiquidNet, create_liquid_net_fallback as create_liquid_net
        register_neuron = lambda name: lambda cls: cls
        _TORCH_AVAILABLE = False
        _IMPLEMENTATION = "numpy"
        warnings.warn("Using NumPy fallback implementations - some features may be limited")
    except ImportError:
        try:
            from .minimal_fallback import MinimalLiquidNeuron as LiquidNeuron, MinimalLiquidNet as LiquidNet, create_minimal_liquid_net as create_liquid_net
            register_neuron = lambda name: lambda cls: cls
            _TORCH_AVAILABLE = False
            _IMPLEMENTATION = "minimal"
            warnings.warn("Using minimal fallback implementations - zero dependencies mode")
        except ImportError:
            LiquidNeuron = None
            LiquidNet = None
            create_liquid_net = None
            register_neuron = None
            _TORCH_AVAILABLE = False
            _IMPLEMENTATION = "none"

try:
    from .event_encoding import EventEncoder, TemporalEncoder
except ImportError:
    EventEncoder = None
    TemporalEncoder = None

try:
    from .temporal_dynamics import ODESolver, EulerSolver, RK4Solver
except ImportError:
    ODESolver = None
    EulerSolver = None
    RK4Solver = None

__all__ = [
    "LiquidNeuron",
    "LiquidNet",
    "create_liquid_net",
    "register_neuron", 
    "EventEncoder",
    "TemporalEncoder",
    "ODESolver",
    "EulerSolver", 
    "RK4Solver",
    "_TORCH_AVAILABLE",
    "_IMPLEMENTATION",
]