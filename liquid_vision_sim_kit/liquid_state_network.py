"""
Liquid State Network (LSN / Echo State Network)

Reservoir computing model with:
- Fixed random sparse reservoir  W_res  (spectral radius scaled < 1)
- Fixed random input weights     W_in
- Trainable linear readout       W_out  (fit via pseudo-inverse)
"""

import numpy as np


class LiquidStateNetwork:
    """
    Echo-State / Liquid State Network for neuromorphic event streams.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input vector (default 4: x, y, polarity, t).
    reservoir_size : int
        Number of reservoir neurons (default 128).
    output_dim : int
        Number of output units (default 1).
    spectral_radius : float
        Target spectral radius of W_res — must be < 1 for echo-state property
        (default 0.9).
    sparsity : float
        Fraction of reservoir connections set to zero (default 0.8).
    seed : int or None
        Random seed (default 42).
    """

    def __init__(
        self,
        input_dim: int = 4,
        reservoir_size: int = 128,
        output_dim: int = 1,
        spectral_radius: float = 0.9,
        sparsity: float = 0.8,
        seed: int | None = 42,
    ):
        if spectral_radius >= 1.0:
            raise ValueError("spectral_radius must be < 1 for echo-state property")

        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity

        rng = np.random.default_rng(seed)

        # Input weight matrix: reservoir_size × input_dim
        self.W_in = rng.standard_normal((reservoir_size, input_dim)) * 0.1

        # Sparse reservoir weight matrix: reservoir_size × reservoir_size
        W = rng.standard_normal((reservoir_size, reservoir_size))
        mask = rng.random((reservoir_size, reservoir_size)) < sparsity
        W[mask] = 0.0
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_sr = np.max(np.abs(eigenvalues))
        if current_sr > 1e-10:
            W = W * (spectral_radius / current_sr)
        self.W_res = W

        # Readout weights: output_dim × reservoir_size  (initialised to zeros)
        self.W_out = np.zeros((output_dim, reservoir_size))

        # Reservoir state
        self._state = np.zeros(reservoir_size)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._state = np.zeros(self.reservoir_size)

    def _step(self, x: np.ndarray) -> np.ndarray:
        """Single reservoir update for input vector x, shape (input_dim,)."""
        pre = self.W_res @ self._state + self.W_in @ x
        self._state = np.tanh(pre)
        return self._state.copy()

    def forward(self, events: np.ndarray) -> np.ndarray:
        """
        Run events through the reservoir and return the readout output.

        Parameters
        ----------
        events : np.ndarray, shape (N, input_dim) or (input_dim,)
            Each row is one event vector [x, y, polarity, timestamp].
            An empty (0, 4) array produces a zero output.

        Returns
        -------
        np.ndarray, shape (output_dim,)
        """
        self._reset_state()
        events = np.atleast_2d(events)

        if events.shape[0] == 0:
            return np.zeros(self.output_dim)

        for event in events:
            x = event[: self.input_dim].astype(np.float64)
            self._step(x)

        output = self.W_out @ self._state  # (output_dim,)
        return output

    def collect_states(self, X: list) -> np.ndarray:
        """
        Run a list of event arrays through the reservoir and collect final states.

        Parameters
        ----------
        X : list of np.ndarray
            Each element has shape (N_i, input_dim).

        Returns
        -------
        np.ndarray, shape (len(X), reservoir_size)
        """
        states = []
        for events in X:
            self._reset_state()
            events = np.atleast_2d(events)
            if events.shape[0] > 0:
                for event in events:
                    self._step(event[: self.input_dim].astype(np.float64))
            states.append(self._state.copy())
        return np.array(states)

    def fit_readout(self, X: list, y: np.ndarray) -> None:
        """
        Train the linear readout W_out via ridge pseudo-inverse.

        Parameters
        ----------
        X : list of np.ndarray
            Training event sequences.
        y : np.ndarray, shape (n_samples,) or (n_samples, output_dim)
        """
        states = self.collect_states(X)          # (n_samples, reservoir_size)
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]                 # (n_samples, output_dim)
        # Least squares: W_out = (y^T S)(S^T S)^{-1}
        # Use numpy lstsq for numerical stability
        W_out_T, _, _, _ = np.linalg.lstsq(states, y, rcond=None)
        self.W_out = W_out_T.T                   # (output_dim, reservoir_size)

    @property
    def actual_spectral_radius(self) -> float:
        """Compute the actual spectral radius of W_res."""
        eigenvalues = np.linalg.eigvals(self.W_res)
        return float(np.max(np.abs(eigenvalues)))
