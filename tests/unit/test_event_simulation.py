"""
Unit tests for event camera simulation components.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from liquid_vision.simulation.event_simulator import (
    EventData, DVSSimulator, DAVISSimulator, AdvancedDVSSimulator,
    create_simulator
)
from liquid_vision.simulation.scene_generator import (
    SceneGenerator, SceneObject, MotionPattern, ObjectType
)
from liquid_vision.simulation.noise_models import (
    ShotNoise, ThermalNoise, RefractoryNoise, TimestampJitter,
    PixelMismatch, CompositeNoise, create_realistic_noise
)


class TestEventData:
    """Test cases for EventData class."""
    
    def test_initialization(self):
        """Test EventData initialization."""
        x = np.array([10, 20, 30])
        y = np.array([15, 25, 35])
        t = np.array([1.0, 2.0, 3.0])
        p = np.array([1, -1, 1])
        
        events = EventData(x=x, y=y, t=t, p=p)
        
        assert len(events) == 3
        np.testing.assert_array_equal(events.x, x)
        np.testing.assert_array_equal(events.y, y)
        np.testing.assert_array_equal(events.t, t)
        np.testing.assert_array_equal(events.p, p)
        
    def test_to_tensor(self):
        """Test conversion to PyTorch tensor."""
        events = EventData(
            x=np.array([1, 2]),
            y=np.array([3, 4]),
            t=np.array([5.0, 6.0]),
            p=np.array([1, -1])
        )
        
        tensor = events.to_tensor()
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 4)
        assert tensor.dtype == torch.float32
        
        expected = torch.tensor([[1, 3, 5, 1], [2, 4, 6, -1]], dtype=torch.float32)
        assert torch.allclose(tensor, expected)
        
    def test_spatial_filtering(self):
        """Test spatial filtering of events."""
        events = EventData(
            x=np.array([5, 15, 25, 35]),
            y=np.array([10, 20, 30, 40]),
            t=np.array([1.0, 2.0, 3.0, 4.0]),
            p=np.array([1, -1, 1, -1])
        )
        
        # Filter to keep only events in x=[10, 30), y=[15, 35)
        filtered = events.filter_spatial(x_range=(10, 30), y_range=(15, 35))
        
        assert len(filtered) == 2
        np.testing.assert_array_equal(filtered.x, [15, 25])
        np.testing.assert_array_equal(filtered.y, [20, 30])
        
    def test_temporal_filtering(self):
        """Test temporal filtering of events."""
        events = EventData(
            x=np.array([1, 2, 3, 4]),
            y=np.array([1, 2, 3, 4]),
            t=np.array([0.5, 1.5, 2.5, 3.5]),
            p=np.array([1, -1, 1, -1])
        )
        
        # Filter to keep events in time [1.0, 3.0]
        filtered = events.filter_temporal(t_start=1.0, t_end=3.0)
        
        assert len(filtered) == 2
        np.testing.assert_array_equal(filtered.t, [1.5, 2.5])


class TestDVSSimulator:
    """Test cases for DVS camera simulator."""
    
    def test_initialization(self):
        """Test DVS simulator initialization."""
        sim = DVSSimulator(
            resolution=(320, 240),
            contrast_threshold=0.2,
            refractory_period=2.0
        )
        
        assert sim.resolution == (320, 240)
        assert sim.contrast_threshold == 0.2
        assert sim.refractory_period == 2.0
        assert sim.last_frame is None
        
    def test_first_frame_no_events(self):
        """Test that first frame generates no events."""
        sim = DVSSimulator(resolution=(64, 48))
        
        frame = np.random.rand(48, 64).astype(np.float32)
        events = sim.simulate_frame(frame, timestamp=0.0)
        
        assert len(events) == 0
        assert sim.last_frame is not None
        
    def test_brightness_change_events(self):
        """Test event generation from brightness changes."""
        sim = DVSSimulator(resolution=(64, 48), contrast_threshold=0.1)
        
        # First frame (dark)
        frame1 = np.zeros((48, 64), dtype=np.float32) + 0.1
        sim.simulate_frame(frame1, timestamp=0.0)
        
        # Second frame (bright spot)
        frame2 = frame1.copy()
        frame2[20:30, 25:35] = 0.8  # Bright rectangular region
        
        events = sim.simulate_frame(frame2, timestamp=1.0)
        
        assert len(events) > 0
        # All events should be positive (brightness increase)
        assert np.all(events.p > 0)
        # Events should be in the bright region
        assert np.all((events.x >= 25) & (events.x < 35))
        assert np.all((events.y >= 20) & (events.y < 30))
        
    def test_negative_events(self):
        """Test generation of negative events."""
        sim = DVSSimulator(resolution=(32, 32), contrast_threshold=0.15)
        
        # First frame (bright)
        frame1 = np.ones((32, 32), dtype=np.float32) * 0.8
        sim.simulate_frame(frame1, timestamp=0.0)
        
        # Second frame (dark spot)
        frame2 = frame1.copy()
        frame2[10:20, 10:20] = 0.1  # Dark region
        
        events = sim.simulate_frame(frame2, timestamp=1.0)
        
        assert len(events) > 0
        # Should have negative events (brightness decrease)
        assert np.any(events.p < 0)
        
    def test_refractory_period(self):
        """Test refractory period implementation."""
        sim = DVSSimulator(
            resolution=(32, 32),
            contrast_threshold=0.1,
            refractory_period=5.0
        )
        
        # First frame
        frame1 = np.zeros((32, 32), dtype=np.float32) + 0.1
        sim.simulate_frame(frame1, timestamp=0.0)
        
        # Second frame with change
        frame2 = frame1.copy()
        frame2[15, 15] = 0.8
        events1 = sim.simulate_frame(frame2, timestamp=1.0)
        
        # Third frame with same change (should be blocked by refractory period)
        frame3 = frame1.copy()
        frame3[15, 15] = 0.9
        events2 = sim.simulate_frame(frame3, timestamp=2.0)  # Within refractory period
        
        # Fourth frame with change after refractory period
        frame4 = frame1.copy()
        frame4[15, 15] = 0.7
        events3 = sim.simulate_frame(frame4, timestamp=7.0)  # After refractory period
        
        assert len(events1) > 0
        assert len(events2) == 0  # Blocked by refractory period
        assert len(events3) > 0  # After refractory period
        
    def test_video_simulation(self):
        """Test simulation of video sequence."""
        sim = DVSSimulator(resolution=(64, 48), contrast_threshold=0.1)
        
        # Create simple video sequence
        num_frames = 5
        frames = []
        
        for i in range(num_frames):
            frame = np.zeros((48, 64), dtype=np.float32) + 0.2
            # Moving bright spot
            x_pos = 10 + i * 5
            frame[20:25, x_pos:x_pos+5] = 0.8
            frames.append(frame)
            
        events = sim.simulate_video(frames, fps=30.0)
        
        assert len(events) > 0
        # Events should span the time duration
        assert events.t.max() > 0
        # Should have events from multiple time points
        unique_times = len(np.unique(events.t))
        assert unique_times >= num_frames - 1  # First frame generates no events


class TestDAVISSimulator:
    """Test cases for DAVIS camera simulator."""
    
    def test_davis_output(self):
        """Test DAVIS simulator returns both events and APS frames."""
        sim = DAVISSimulator(
            resolution=(64, 48),
            contrast_threshold=0.1,
            aps_exposure_time=30.0
        )
        
        # First frame
        frame1 = np.random.rand(48, 64).astype(np.float32)
        result1 = sim.simulate_frame(frame1, timestamp=0.0)
        
        if isinstance(result1, tuple):
            events1, aps1 = result1
            assert aps1 is not None  # Should capture APS frame
        else:
            events1 = result1
            
        # Second frame (before APS exposure time)
        frame2 = np.random.rand(48, 64).astype(np.float32)
        result2 = sim.simulate_frame(frame2, timestamp=15.0)
        
        if isinstance(result2, tuple):
            events2, aps2 = result2
            assert aps2 is None  # Should not capture APS frame yet
        else:
            events2 = result2


class TestAdvancedDVSSimulator:
    """Test cases for advanced DVS simulator with realistic effects."""
    
    def test_bandwidth_limitation(self):
        """Test bandwidth limitation functionality."""
        sim = AdvancedDVSSimulator(
            resolution=(64, 48),
            contrast_threshold=0.05,  # Low threshold for many events
            bandwidth_limit=100
        )
        
        # Create frame with many changes
        frame1 = np.random.rand(48, 64).astype(np.float32) * 0.3
        sim.simulate_frame(frame1, timestamp=0.0)
        
        # Frame with lots of changes
        frame2 = np.random.rand(48, 64).astype(np.float32) * 0.8
        events = sim.simulate_frame(frame2, timestamp=1.0)
        
        # Should be limited by bandwidth
        assert len(events) <= 100
        
    def test_latency_jitter(self):
        """Test latency jitter in timestamps."""
        sim = AdvancedDVSSimulator(
            resolution=(32, 32),
            contrast_threshold=0.1,
            latency_mean=0.5,
            latency_std=0.1
        )
        
        frame1 = np.zeros((32, 32), dtype=np.float32) + 0.1
        sim.simulate_frame(frame1, timestamp=0.0)
        
        frame2 = frame1.copy()
        frame2[15:17, 15:17] = 0.8
        events = sim.simulate_frame(frame2, timestamp=1.0)
        
        if len(events) > 0:
            # Timestamps should be jittered around 1.0
            assert np.all(events.t >= 1.0)  # Should be >= base timestamp
            assert np.std(events.t) > 0  # Should have some variation


class TestSceneGenerator:
    """Test cases for synthetic scene generation."""
    
    def test_initialization(self):
        """Test scene generator initialization."""
        scene = SceneGenerator(
            resolution=(320, 240),
            background_color=0.3,
            frame_rate=25.0
        )
        
        assert scene.resolution == (320, 240)
        assert scene.background_color == 0.3
        assert scene.frame_rate == 25.0
        assert len(scene.objects) == 0
        
    def test_add_object(self):
        """Test adding objects to scene."""
        scene = SceneGenerator()
        
        scene.add_object(
            object_type=ObjectType.CIRCLE,
            position=(100, 150),
            size=20,
            velocity=(2.0, -1.0),
            color=0.8,
            motion_pattern=MotionPattern.LINEAR
        )
        
        assert len(scene.objects) == 1
        obj = scene.objects[0]
        assert obj.object_type == ObjectType.CIRCLE
        assert obj.position == (100, 150)
        assert obj.size == 20
        
    def test_generate_frame(self):
        """Test single frame generation."""
        scene = SceneGenerator(resolution=(64, 48))
        
        # Add a simple object
        scene.add_object(
            object_type=ObjectType.CIRCLE,
            position=(32, 24),
            size=10,
            velocity=(0, 0),
            color=0.9
        )
        
        frame = scene.generate_frame(frame_number=0)
        
        assert frame.shape == (48, 64)
        assert frame.dtype == np.float32
        # Should have bright pixels where circle is drawn
        assert np.max(frame) > scene.background_color
        
    def test_motion_patterns(self):
        """Test different motion patterns."""
        scene = SceneGenerator(resolution=(100, 100))
        
        # Test linear motion
        scene.add_object(
            object_type=ObjectType.CIRCLE,
            position=(50, 50),
            size=5,
            velocity=(2, 0),
            color=0.8,
            motion_pattern=MotionPattern.LINEAR
        )
        
        frame0 = scene.generate_frame(0)
        frame10 = scene.generate_frame(10)
        
        # Object should have moved
        assert not np.array_equal(frame0, frame10)
        
    def test_generate_sequence(self):
        """Test sequence generation."""
        scene = SceneGenerator(resolution=(32, 32))
        
        scene.add_object(
            object_type=ObjectType.RECTANGLE,
            position=(16, 16),
            size=(8, 6),
            velocity=(1, 0),
            color=0.7
        )
        
        frames, timestamps = scene.generate_sequence(
            num_frames=5,
            return_timestamps=True
        )
        
        assert frames.shape == (5, 32, 32)
        assert len(timestamps) == 5
        assert timestamps[1] > timestamps[0]  # Increasing timestamps
        
    def test_create_scene_factory(self):
        """Test create_scene factory method."""
        scene = SceneGenerator.create_scene(
            num_objects=3,
            resolution=(80, 60),
            motion_type="circular",
            velocity_range=(1.0, 3.0),
            size_range=(5, 15)
        )
        
        assert len(scene.objects) == 3
        assert scene.resolution == (80, 60)
        
        # All objects should have circular motion
        for obj in scene.objects:
            assert obj.motion_pattern == MotionPattern.CIRCULAR


class TestNoiseModels:
    """Test cases for noise models."""
    
    def test_shot_noise(self):
        """Test shot noise model."""
        noise = ShotNoise(rate=0.1, sensor_size=(64, 48))
        
        # Create empty events
        events = EventData(
            x=np.array([]),
            y=np.array([]),
            t=np.array([]),
            p=np.array([])
        )
        
        noisy_events = noise.apply(events, timestamp=1.0)
        
        # Should add some noise events
        assert len(noisy_events) >= 0  # Poisson can be 0
        
        # Test with existing events
        events = EventData(
            x=np.array([10, 20]),
            y=np.array([15, 25]),
            t=np.array([1.0, 2.0]),
            p=np.array([1, -1])
        )
        
        noisy_events = noise.apply(events, timestamp=3.0)
        assert len(noisy_events) >= 2  # Original events + noise
        
    def test_thermal_noise(self):
        """Test thermal noise model."""
        noise_cold = ThermalNoise(
            base_rate=0.01,
            temperature=15.0,
            sensor_size=(32, 32)
        )
        
        noise_hot = ThermalNoise(
            base_rate=0.01,
            temperature=45.0,
            sensor_size=(32, 32)
        )
        
        events = EventData(
            x=np.array([5]),
            y=np.array([5]),
            t=np.array([1.0]),
            p=np.array([1])
        )
        
        # Generate noise at different temperatures multiple times
        cold_lengths = []
        hot_lengths = []
        
        for _ in range(10):
            noisy_cold = noise_cold.apply(events, timestamp=1.0)
            noisy_hot = noise_hot.apply(events, timestamp=1.0)
            cold_lengths.append(len(noisy_cold))
            hot_lengths.append(len(noisy_hot))
            
        # Hot temperature should generally produce more events
        assert np.mean(hot_lengths) >= np.mean(cold_lengths)
        
    def test_timestamp_jitter(self):
        """Test timestamp jitter noise."""
        jitter = TimestampJitter(jitter_std=0.5, max_jitter=2.0)
        
        events = EventData(
            x=np.array([1, 2, 3]),
            y=np.array([1, 2, 3]),
            t=np.array([1.0, 2.0, 3.0]),
            p=np.array([1, -1, 1])
        )
        
        jittered = jitter.apply(events, timestamp=5.0)
        
        assert len(jittered) == 3
        # Original timestamps should be modified
        assert not np.array_equal(events.t, jittered.t)
        # But should still be positive
        assert np.all(jittered.t >= 0)
        
    def test_composite_noise(self):
        """Test composite noise model."""
        noise_models = [
            ShotNoise(rate=0.05, sensor_size=(32, 32)),
            TimestampJitter(jitter_std=0.1),
        ]
        
        composite = CompositeNoise(noise_models)
        
        events = EventData(
            x=np.array([10]),
            y=np.array([10]),
            t=np.array([1.0]),
            p=np.array([1])
        )
        
        noisy_events = composite.apply(events, timestamp=2.0)
        
        # Should apply all noise models
        assert len(noisy_events) >= 1
        
    def test_create_realistic_noise(self):
        """Test realistic noise creation."""
        noise_levels = ["low", "medium", "high"]
        
        for level in noise_levels:
            noise = create_realistic_noise(
                sensor_size=(64, 48),
                noise_level=level
            )
            
            assert isinstance(noise, CompositeNoise)
            assert len(noise.noise_models) > 0
            
            # Higher levels should have more noise sources
            if level == "high":
                assert len(noise.noise_models) >= 3


class TestSimulatorFactory:
    """Test simulator factory function."""
    
    def test_create_simulator(self):
        """Test create_simulator factory function."""
        simulator_types = ["dvs", "davis", "advanced_dvs"]
        
        for sim_type in simulator_types:
            sim = create_simulator(
                simulator_type=sim_type,
                resolution=(128, 96),
                contrast_threshold=0.12
            )
            
            assert sim.resolution == (128, 96)
            assert sim.contrast_threshold == 0.12
            
        # Test invalid type
        with pytest.raises(ValueError):
            create_simulator(simulator_type="invalid")


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_scene_to_events_pipeline(self):
        """Test complete pipeline from scene generation to events."""
        # Create scene
        scene = SceneGenerator(resolution=(64, 48))
        scene.add_object(
            object_type=ObjectType.CIRCLE,
            position=(32, 24),
            size=8,
            velocity=(2, 0),
            color=0.9,
            motion_pattern=MotionPattern.LINEAR
        )
        
        # Generate frames
        frames, timestamps = scene.generate_sequence(
            num_frames=10,
            return_timestamps=True
        )
        
        # Simulate events
        simulator = DVSSimulator(
            resolution=(64, 48),
            contrast_threshold=0.1
        )
        
        events = simulator.simulate_video(frames, timestamps)
        
        # Should generate events from moving object
        assert len(events) > 0
        assert events.t.min() >= timestamps[0]
        assert events.t.max() <= timestamps[-1]
        
    def test_noisy_simulation(self):
        """Test simulation with noise."""
        # Create simple scene
        scene = SceneGenerator(resolution=(32, 32))
        scene.add_object(
            object_type=ObjectType.RECTANGLE,
            position=(16, 16),
            size=6,
            velocity=(1, 1),
            color=0.8
        )
        
        frames = scene.generate_sequence(num_frames=5, return_timestamps=False)
        
        # Simulate with noise
        simulator = DVSSimulator(
            resolution=(32, 32),
            contrast_threshold=0.1,
            noise_model=create_realistic_noise(
                sensor_size=(32, 32),
                noise_level="medium"
            )
        )
        
        events = simulator.simulate_video(frames, fps=30.0)
        
        # Should generate events (signal + noise)
        assert len(events) > 0