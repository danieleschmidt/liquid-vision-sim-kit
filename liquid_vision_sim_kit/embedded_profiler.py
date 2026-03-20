"""
Embedded Profiler

Estimates deployment metrics for models targeting microcontrollers (e.g. ARM Cortex-M):
  - Parameter count
  - Memory footprint in bytes (float32)
  - Approximate Cortex-M cycle count based on multiply-accumulate (MAC) operations
"""

import numpy as np
from typing import Any


class EmbeddedProfiler:
    """
    Profile a model represented as a dict of named weight arrays.

    The model_dict maps layer/weight names to numpy arrays (or lists/sequences
    that can be converted to arrays).

    Example
    -------
    >>> profiler = EmbeddedProfiler()
    >>> model = {"W_in": np.zeros((128, 4)), "W_out": np.zeros((1, 128))}
    >>> stats = profiler.profile(model)
    """

    # Bytes per element for float32
    BYTES_PER_PARAM: int = 4

    # ------------------------------------------------------------------
    # Core profiling methods
    # ------------------------------------------------------------------

    def param_count(self, model_dict: dict[str, Any]) -> int:
        """
        Count total trainable / storable parameters.

        Parameters
        ----------
        model_dict : dict
            Maps weight names → array-like values.

        Returns
        -------
        int  Total element count across all arrays.
        """
        total = 0
        for v in model_dict.values():
            arr = np.asarray(v)
            total += int(arr.size)
        return total

    def memory_bytes(self, model_dict: dict[str, Any]) -> int:
        """
        Estimate flash / SRAM usage assuming float32 storage.

        Returns
        -------
        int  Bytes required to store all parameters.
        """
        return self.param_count(model_dict) * self.BYTES_PER_PARAM

    def cortex_m_cycles(
        self,
        model_dict: dict[str, Any],
        ops_per_cycle: int = 4,
    ) -> int:
        """
        Estimate Cortex-M inference cycles.

        Assumes each parameter contributes one multiply-accumulate (MAC)
        operation per inference step, and that the target core can execute
        `ops_per_cycle` MACs per clock cycle (accounts for SIMD/DSP extensions
        on M4/M33/M55 etc.).

        Parameters
        ----------
        model_dict : dict
        ops_per_cycle : int
            MACs per cycle (default 4 for Cortex-M4 with DSP extension).

        Returns
        -------
        int  Estimated cycle count.
        """
        macs = self.param_count(model_dict)
        return int(np.ceil(macs / ops_per_cycle))

    def profile(
        self,
        model_dict: dict[str, Any],
        ops_per_cycle: int = 4,
    ) -> dict[str, int]:
        """
        Return a dict with all profiling statistics.

        Returns
        -------
        dict with keys: param_count, memory_bytes, cortex_m_cycles
        """
        return {
            "param_count": self.param_count(model_dict),
            "memory_bytes": self.memory_bytes(model_dict),
            "cortex_m_cycles": self.cortex_m_cycles(model_dict, ops_per_cycle),
        }
