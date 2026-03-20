"""
Event Camera Simulator

Simulates a Dynamic Vision Sensor (DVS) / event camera by computing frame
differences and emitting spike events where |diff| > threshold.
"""

import numpy as np


class EventCameraSimulator:
    """
    Simulates an event (neuromorphic) camera.

    An event camera fires asynchronous events when pixel brightness changes
    exceed a threshold. This simulator computes the difference between two
    consecutive grayscale frames and emits events for pixels where the
    absolute difference exceeds the threshold.
    """

    def generate_events(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: float = 0.1,
        timestamp: float = 0.0,
    ):
        """
        Generate spike events from two consecutive grayscale frames.

        Parameters
        ----------
        frame1 : np.ndarray
            First (earlier) grayscale frame, shape (H, W), values in [0, 1].
        frame2 : np.ndarray
            Second (later) grayscale frame, shape (H, W), values in [0, 1].
        threshold : float
            Minimum absolute brightness change to emit an event (default 0.1).
        timestamp : float
            Timestamp to assign to all generated events (default 0.0).

        Returns
        -------
        list of tuples (x, y, polarity, timestamp)
            x : int  — column index
            y : int  — row index
            polarity : int  — +1 (brightness increase) or -1 (decrease)
            timestamp : float  — event timestamp
        """
        frame1 = np.asarray(frame1, dtype=np.float64)
        frame2 = np.asarray(frame2, dtype=np.float64)

        if frame1.shape != frame2.shape:
            raise ValueError(
                f"frame1 and frame2 must have the same shape, "
                f"got {frame1.shape} and {frame2.shape}"
            )

        diff = frame2 - frame1
        rows, cols = np.where(np.abs(diff) > threshold)

        events = []
        for r, c in zip(rows, cols):
            polarity = 1 if diff[r, c] > 0 else -1
            events.append((int(c), int(r), polarity, float(timestamp)))

        return events
