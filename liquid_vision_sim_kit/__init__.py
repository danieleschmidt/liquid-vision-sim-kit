"""
liquid-vision-sim-kit
=====================

Neuromorphic event camera simulation and liquid state network inference
kit for embedded / edge deployment.

Classes
-------
EventCameraSimulator
    Simulate a DVS event camera from consecutive grayscale frames.
NeuromorphicDataset
    Synthetic event-stream dataset generator.
LiquidStateNetwork
    Echo-state / liquid state reservoir computing model.
EmbeddedProfiler
    Estimate parameter count, memory, and Cortex-M cycle budget.
CHeaderExporter
    Export weight matrices as C header files for embedded deployment.
"""

from .event_camera_simulator import EventCameraSimulator
from .neuromorphic_dataset import NeuromorphicDataset
from .liquid_state_network import LiquidStateNetwork
from .embedded_profiler import EmbeddedProfiler
from .c_header_exporter import CHeaderExporter

__all__ = [
    "EventCameraSimulator",
    "NeuromorphicDataset",
    "LiquidStateNetwork",
    "EmbeddedProfiler",
    "CHeaderExporter",
]

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
