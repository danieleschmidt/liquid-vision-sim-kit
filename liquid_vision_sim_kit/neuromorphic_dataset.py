"""
Neuromorphic Dataset

Generates synthetic event sequences for training and evaluation of
neuromorphic / spiking neural network models.
"""

import numpy as np
from .event_camera_simulator import EventCameraSimulator


class NeuromorphicDataset:
    """
    Synthetic neuromorphic event-stream dataset.

    Generates a sequence of event arrays by simulating an event camera on
    randomly generated grayscale frame pairs.

    Parameters
    ----------
    n_frames : int
        Number of frame pairs to generate.
    height : int
        Frame height in pixels.
    width : int
        Frame width in pixels.
    threshold : float
        Event threshold forwarded to EventCameraSimulator (default 0.1).
    seed : int or None
        Random seed for reproducibility (default None).
    """

    def __init__(
        self,
        n_frames: int = 10,
        height: int = 32,
        width: int = 32,
        threshold: float = 0.1,
        seed: int | None = None,
    ):
        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.threshold = threshold
        self._rng = np.random.default_rng(seed)
        self._simulator = EventCameraSimulator()
        self._sequences = self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sequence(self, n_frames: int, height: int, width: int):
        """
        Generate a list of event arrays for n_frames consecutive frame pairs.

        Parameters
        ----------
        n_frames : int
        height : int
        width : int

        Returns
        -------
        list of np.ndarray
            Each element is an array of shape (N, 4) with columns
            [x, y, polarity, timestamp], or an empty array if no events fired.
        """
        rng = np.random.default_rng()
        sequences = []
        prev_frame = rng.random((height, width))
        for t in range(n_frames):
            curr_frame = rng.random((height, width))
            events = self._simulator.generate_events(
                prev_frame, curr_frame, self.threshold, timestamp=float(t)
            )
            if events:
                arr = np.array(events, dtype=np.float32)  # (N, 4)
            else:
                arr = np.empty((0, 4), dtype=np.float32)
            sequences.append(arr)
            prev_frame = curr_frame
        return sequences

    # ------------------------------------------------------------------
    # Sequence-like interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._sequences[idx]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build(self):
        """Pre-build the dataset sequences on init."""
        sequences = []
        prev_frame = self._rng.random((self.height, self.width))
        for t in range(self.n_frames):
            curr_frame = self._rng.random((self.height, self.width))
            events = self._simulator.generate_events(
                prev_frame, curr_frame, self.threshold, timestamp=float(t)
            )
            if events:
                arr = np.array(events, dtype=np.float32)
            else:
                arr = np.empty((0, 4), dtype=np.float32)
            sequences.append(arr)
            prev_frame = curr_frame
        return sequences
