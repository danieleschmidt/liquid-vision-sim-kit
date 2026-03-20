"""
Tests for liquid-vision-sim-kit
"""

import numpy as np
import os
import sys
import tempfile
import pytest

# Ensure the package is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from liquid_vision_sim_kit import (
    EventCameraSimulator,
    NeuromorphicDataset,
    LiquidStateNetwork,
    EmbeddedProfiler,
    CHeaderExporter,
)


# ──────────────────────────────────────────────────────────────────────────────
# EventCameraSimulator
# ──────────────────────────────────────────────────────────────────────────────

class TestEventCameraSimulator:

    def test_generates_events_from_two_frames(self):
        """Simulator should return a non-empty event list for differing frames."""
        sim = EventCameraSimulator()
        frame1 = np.zeros((8, 8))
        frame2 = np.ones((8, 8))
        events = sim.generate_events(frame1, frame2, threshold=0.1)
        assert len(events) > 0

    def test_events_have_correct_structure(self):
        """Each event must be a tuple (x, y, polarity, timestamp)."""
        sim = EventCameraSimulator()
        frame1 = np.zeros((8, 8))
        frame2 = np.ones((8, 8))
        events = sim.generate_events(frame1, frame2, threshold=0.1, timestamp=1.5)
        assert len(events) > 0
        for event in events:
            x, y, polarity, ts = event
            assert isinstance(x, int)
            assert isinstance(y, int)
            assert polarity in (1, -1)
            assert ts == pytest.approx(1.5)

    def test_polarity_positive_for_brightness_increase(self):
        """Bright pixel in frame2 → polarity +1."""
        sim = EventCameraSimulator()
        frame1 = np.zeros((4, 4))
        frame2 = np.zeros((4, 4))
        frame2[2, 3] = 0.9   # large increase at row=2, col=3
        events = sim.generate_events(frame1, frame2, threshold=0.1)
        # All events should have polarity +1 (only brightening pixels)
        polarities = [e[2] for e in events]
        assert all(p == 1 for p in polarities)

    def test_polarity_negative_for_brightness_decrease(self):
        """Dim pixel in frame2 → polarity -1."""
        sim = EventCameraSimulator()
        frame1 = np.ones((4, 4))
        frame2 = np.zeros((4, 4))
        events = sim.generate_events(frame1, frame2, threshold=0.1)
        polarities = [e[2] for e in events]
        assert all(p == -1 for p in polarities)

    def test_no_events_below_threshold(self):
        """Very similar frames should produce no events."""
        sim = EventCameraSimulator()
        frame1 = np.full((8, 8), 0.5)
        frame2 = np.full((8, 8), 0.5 + 0.01)   # change << threshold
        events = sim.generate_events(frame1, frame2, threshold=0.1)
        assert len(events) == 0

    def test_mismatched_frames_raise_error(self):
        sim = EventCameraSimulator()
        frame1 = np.zeros((4, 4))
        frame2 = np.zeros((8, 8))
        with pytest.raises(ValueError):
            sim.generate_events(frame1, frame2)

    def test_event_coordinates_within_frame_bounds(self):
        sim = EventCameraSimulator()
        H, W = 16, 32
        rng = np.random.default_rng(0)
        frame1 = rng.random((H, W))
        frame2 = rng.random((H, W))
        events = sim.generate_events(frame1, frame2, threshold=0.1)
        for x, y, pol, ts in events:
            assert 0 <= x < W, f"x={x} out of bounds"
            assert 0 <= y < H, f"y={y} out of bounds"


# ──────────────────────────────────────────────────────────────────────────────
# NeuromorphicDataset
# ──────────────────────────────────────────────────────────────────────────────

class TestNeuromorphicDataset:

    def test_generates_sequences(self):
        """generate_sequence should return a list of arrays."""
        ds = NeuromorphicDataset(n_frames=5, height=16, width=16, seed=0)
        sequences = ds.generate_sequence(n_frames=5, height=16, width=16)
        assert isinstance(sequences, list)
        assert len(sequences) == 5

    def test_sequence_arrays_have_correct_columns(self):
        """Each event array must have 4 columns (x, y, polarity, ts)."""
        ds = NeuromorphicDataset(n_frames=5, height=16, width=16, seed=1)
        sequences = ds.generate_sequence(n_frames=5, height=16, width=16)
        for arr in sequences:
            assert arr.ndim == 2
            assert arr.shape[1] == 4

    def test_dataset_len(self):
        """__len__ should equal n_frames."""
        ds = NeuromorphicDataset(n_frames=8, height=16, width=16, seed=2)
        assert len(ds) == 8

    def test_dataset_getitem(self):
        """__getitem__ should return a 2-D array with 4 columns."""
        ds = NeuromorphicDataset(n_frames=6, height=16, width=16, seed=3)
        item = ds[0]
        assert isinstance(item, np.ndarray)
        assert item.ndim == 2
        assert item.shape[1] == 4


# ──────────────────────────────────────────────────────────────────────────────
# LiquidStateNetwork
# ──────────────────────────────────────────────────────────────────────────────

class TestLiquidStateNetwork:

    def _make_events(self, n=20, input_dim=4, seed=42):
        rng = np.random.default_rng(seed)
        return rng.random((n, input_dim)).astype(np.float32)

    def test_forward_pass_shape(self):
        """forward() should return a 1-D array of length output_dim."""
        lsn = LiquidStateNetwork(input_dim=4, reservoir_size=64, output_dim=2, seed=0)
        events = self._make_events(n=10)
        out = lsn.forward(events)
        assert out.shape == (2,)

    def test_reservoir_spectral_radius_less_than_one(self):
        """W_res must have spectral radius < 1 for echo-state property."""
        lsn = LiquidStateNetwork(reservoir_size=64, spectral_radius=0.9, seed=7)
        assert lsn.actual_spectral_radius < 1.0

    def test_forward_on_empty_events(self):
        """forward() on empty event array should return zeros."""
        lsn = LiquidStateNetwork(input_dim=4, reservoir_size=32, output_dim=1, seed=5)
        empty = np.empty((0, 4), dtype=np.float32)
        out = lsn.forward(empty)
        assert out.shape == (1,)
        assert np.allclose(out, 0.0)

    def test_fit_readout_trains_correctly(self):
        """After fit_readout, the network should predict training labels closely."""
        rng = np.random.default_rng(99)
        lsn = LiquidStateNetwork(
            input_dim=4, reservoir_size=128, output_dim=1, seed=10
        )
        # Simple binary labels: 20 samples
        X = [rng.random((15, 4)) for _ in range(20)]
        y = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(20)])
        lsn.fit_readout(X, y)
        # W_out should no longer be all zeros
        assert not np.allclose(lsn.W_out, 0.0)

    def test_fit_readout_reduces_error(self):
        """After training, predictions should be closer to targets than random."""
        rng = np.random.default_rng(77)
        lsn = LiquidStateNetwork(reservoir_size=128, output_dim=1, seed=11)
        X = [rng.random((10, 4)) for _ in range(30)]
        y = rng.standard_normal(30)

        # Collect error before training
        preds_before = np.array([lsn.forward(x)[0] for x in X])
        err_before = np.mean((preds_before - y) ** 2)

        lsn.fit_readout(X, y)

        preds_after = np.array([lsn.forward(x)[0] for x in X])
        err_after = np.mean((preds_after - y) ** 2)

        assert err_after < err_before

    def test_spectral_radius_above_one_raises(self):
        with pytest.raises(ValueError):
            LiquidStateNetwork(spectral_radius=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddedProfiler
# ──────────────────────────────────────────────────────────────────────────────

class TestEmbeddedProfiler:

    def _make_model(self):
        return {
            "W_in": np.zeros((128, 4)),
            "W_res": np.zeros((128, 128)),
            "W_out": np.zeros((1, 128)),
        }

    def test_param_count(self):
        prof = EmbeddedProfiler()
        model = self._make_model()
        expected = 128 * 4 + 128 * 128 + 1 * 128  # 512 + 16384 + 128
        assert prof.param_count(model) == expected

    def test_memory_bytes_is_positive(self):
        prof = EmbeddedProfiler()
        model = self._make_model()
        mem = prof.memory_bytes(model)
        assert mem > 0
        assert mem == prof.param_count(model) * 4  # float32

    def test_profile_returns_dict_with_all_keys(self):
        prof = EmbeddedProfiler()
        model = self._make_model()
        stats = prof.profile(model)
        assert "param_count" in stats
        assert "memory_bytes" in stats
        assert "cortex_m_cycles" in stats

    def test_cortex_m_cycles_proportional_to_params(self):
        prof = EmbeddedProfiler()
        model = {"W": np.zeros((16, 16))}   # 256 params
        cycles_4 = prof.cortex_m_cycles(model, ops_per_cycle=4)
        cycles_8 = prof.cortex_m_cycles(model, ops_per_cycle=8)
        assert cycles_4 > cycles_8   # fewer ops/cycle → more cycles


# ──────────────────────────────────────────────────────────────────────────────
# CHeaderExporter
# ──────────────────────────────────────────────────────────────────────────────

class TestCHeaderExporter:

    def _make_model(self):
        return {
            "W_in": np.eye(4, dtype=np.float32),
            "W_out": np.zeros((1, 4), dtype=np.float32),
        }

    def test_creates_valid_c_header_file(self):
        """export() should create a file with include guards and float arrays."""
        exporter = CHeaderExporter()
        model = self._make_model()
        with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
            fname = f.name
        try:
            exporter.export(model, fname)
            assert os.path.exists(fname)
            with open(fname) as f:
                content = f.read()
            assert "#ifndef" in content
            assert "#define" in content
            assert "#endif" in content
            assert "float" in content
        finally:
            os.unlink(fname)

    def test_header_contains_weight_names(self):
        """Exported header should contain variable names for each weight."""
        exporter = CHeaderExporter()
        model = {"W_in": np.zeros((4, 4)), "W_out": np.zeros((1, 4))}
        with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
            fname = f.name
        try:
            exporter.export(model, fname)
            with open(fname) as f:
                content = f.read()
            assert "W_in" in content
            assert "W_out" in content
        finally:
            os.unlink(fname)

    def test_header_values_are_float32(self):
        """Values in the header should be formatted as float32 (f suffix)."""
        exporter = CHeaderExporter()
        model = {"weights": np.array([1.0, 2.0, 3.0])}
        with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
            fname = f.name
        try:
            exporter.export(model, fname)
            with open(fname) as f:
                content = f.read()
            assert "f" in content  # float literal suffix
        finally:
            os.unlink(fname)
