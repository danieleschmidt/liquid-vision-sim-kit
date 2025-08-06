#!/usr/bin/env python3
"""
Comprehensive tests for new event encoders.
Tests VoxelEncoder, SAEEncoder, and EventImageEncoder implementations.
"""

import sys
import os
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from liquid_vision.core.event_encoding import (
        VoxelEncoder, SAEEncoder, EventImageEncoder, create_encoder
    )
except ImportError as e:
    print(f"Import error (expected without PyTorch): {e}")
    pytest.skip("PyTorch not available", allow_module_level=True)


class TestVoxelEncoder:
    """Test VoxelEncoder functionality."""
    
    def test_voxel_encoder_initialization(self):
        """Test VoxelEncoder initialization."""
        encoder = VoxelEncoder(
            sensor_size=(64, 48),
            voxel_grid=(8, 6, 3),
            normalize=True
        )
        
        assert encoder.sensor_size == (64, 48)
        assert encoder.voxel_grid == (8, 6, 3)
        assert encoder.normalize == True
        assert encoder.x_scale == 8 / 64
        assert encoder.y_scale == 6 / 48
        assert encoder.t_scale == 3 / 50.0  # default time_window
    
    def test_voxel_encoder_empty_events(self):
        """Test VoxelEncoder with empty events."""
        encoder = VoxelEncoder(voxel_grid=(4, 4, 2))
        empty_events = torch.empty(0, 4)
        
        output = encoder(empty_events)
        
        expected_shape = (2, 2, 4, 4)  # (polarities, T, H, W)
        assert output.shape == expected_shape
        assert torch.all(output == 0)
    
    def test_voxel_encoder_with_events(self):
        """Test VoxelEncoder with actual events."""
        encoder = VoxelEncoder(
            sensor_size=(32, 24),
            voxel_grid=(4, 3, 2),
            normalize=True
        )
        
        # Create test events
        events = torch.tensor([
            [10, 8, 10.0, 1],   # positive event
            [20, 16, 30.0, -1], # negative event
            [15, 12, 20.0, 1],  # another positive
        ], dtype=torch.float32)
        
        output = encoder(events)
        
        assert output.shape == (2, 2, 3, 4)  # (polarities, T, H, W)
        assert output.sum() > 0  # Should have some non-zero values
        
        if encoder.normalize:
            assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-6)


class TestSAEEncoder:
    """Test SAEEncoder functionality."""
    
    def test_sae_encoder_initialization(self):
        """Test SAEEncoder initialization."""
        encoder = SAEEncoder(
            sensor_size=(64, 48),
            mixed_polarity=False
        )
        
        assert encoder.sensor_size == (64, 48)
        assert encoder.mixed_polarity == False
        assert encoder.sae_surface.shape == (2, 48, 64)
        
        # Test mixed polarity mode
        encoder_mixed = SAEEncoder(
            sensor_size=(32, 24),
            mixed_polarity=True
        )
        assert encoder_mixed.sae_surface.shape == (1, 24, 32)
    
    def test_sae_encoder_empty_events(self):
        """Test SAEEncoder with empty events."""
        encoder = SAEEncoder(sensor_size=(10, 8))
        empty_events = torch.empty(0, 4)
        
        output = encoder(empty_events)
        
        assert output.shape == (2, 8, 10)
        assert torch.all(output == 0)
    
    def test_sae_encoder_with_events(self):
        """Test SAEEncoder with actual events."""
        encoder = SAEEncoder(sensor_size=(16, 12), mixed_polarity=False)
        
        # Create test events with timestamps
        events = torch.tensor([
            [5, 3, 100.0, 1],    # positive event at t=100
            [8, 7, 150.0, -1],   # negative event at t=150
            [5, 3, 200.0, 1],    # overwrite previous at same location
        ], dtype=torch.float32)
        
        output = encoder(events)
        
        assert output.shape == (2, 12, 16)
        
        # Check that latest timestamp overwrote previous
        assert output[0, 3, 5] == 200.0  # positive channel
        assert output[1, 7, 8] == 150.0  # negative channel
    
    def test_sae_encoder_mixed_polarity(self):
        """Test SAEEncoder in mixed polarity mode."""
        encoder = SAEEncoder(sensor_size=(8, 6), mixed_polarity=True)
        
        events = torch.tensor([
            [2, 1, 50.0, 1],   # positive event
            [4, 3, 75.0, -1],  # negative event
        ], dtype=torch.float32)
        
        output = encoder(events)
        
        assert output.shape == (1, 6, 8)
        assert output[0, 1, 2] == 50.0   # positive timestamp
        assert output[0, 3, 4] == -75.0  # negative timestamp with sign
    
    def test_sae_encoder_reset(self):
        """Test SAEEncoder reset functionality."""
        encoder = SAEEncoder(sensor_size=(4, 4))
        
        events = torch.tensor([[1, 1, 10.0, 1]], dtype=torch.float32)
        encoder(events)
        
        assert encoder.sae_surface[0, 1, 1] == 10.0
        
        encoder.reset()
        assert torch.all(encoder.sae_surface == 0)


class TestEventImageEncoder:
    """Test EventImageEncoder functionality."""
    
    def test_event_image_encoder_initialization(self):
        """Test EventImageEncoder initialization."""
        encoder = EventImageEncoder(
            sensor_size=(32, 24),
            accumulation_mode="count",
            decay_rate=0.1
        )
        
        assert encoder.sensor_size == (32, 24)
        assert encoder.accumulation_mode == "count"
        assert encoder.decay_rate == 0.1
        assert encoder.event_image.shape == (2, 24, 32)
    
    def test_event_image_encoder_count_mode(self):
        """Test EventImageEncoder in count accumulation mode."""
        encoder = EventImageEncoder(
            sensor_size=(8, 6),
            accumulation_mode="count"
        )
        
        # Create events at same location
        events = torch.tensor([
            [2, 2, 10.0, 1],  # positive
            [2, 2, 20.0, 1],  # another positive at same location
            [3, 3, 15.0, -1], # negative
        ], dtype=torch.float32)
        
        output = encoder(events)
        
        assert output.shape == (2, 6, 8)
        assert output[0, 2, 2] == 2.0  # two positive events
        assert output[1, 3, 3] == 1.0  # one negative event
    
    def test_event_image_encoder_binary_mode(self):
        """Test EventImageEncoder in binary accumulation mode."""
        encoder = EventImageEncoder(
            sensor_size=(6, 4),
            accumulation_mode="binary"
        )
        
        events = torch.tensor([
            [1, 1, 10.0, 1],
            [1, 1, 20.0, 1],  # multiple events at same location
        ], dtype=torch.float32)
        
        output = encoder(events)
        
        assert output[0, 1, 1] == 1.0  # binary accumulation
    
    def test_event_image_encoder_exponential_mode(self):
        """Test EventImageEncoder in exponential accumulation mode."""
        encoder = EventImageEncoder(
            sensor_size=(4, 4),
            accumulation_mode="exponential",
            decay_rate=0.1
        )
        
        # First batch of events
        events1 = torch.tensor([[1, 1, 10.0, 1]], dtype=torch.float32)
        output1 = encoder(events1)
        initial_value = output1[0, 1, 1].item()
        
        # Second batch (should apply decay)
        events2 = torch.tensor([[2, 2, 20.0, 1]], dtype=torch.float32)
        output2 = encoder(events2)
        
        # First location should have decayed
        assert output2[0, 1, 1] < initial_value
        assert output2[0, 2, 2] > 0  # new event added
    
    def test_event_image_encoder_reset(self):
        """Test EventImageEncoder reset functionality."""
        encoder = EventImageEncoder(sensor_size=(4, 4))
        
        events = torch.tensor([[1, 1, 10.0, 1]], dtype=torch.float32)
        encoder(events)
        
        assert encoder.event_image[0, 1, 1] > 0
        
        encoder.reset()
        assert torch.all(encoder.event_image == 0)
        assert encoder.last_update == 0


class TestEncoderFactory:
    """Test create_encoder factory function."""
    
    def test_create_voxel_encoder(self):
        """Test creating VoxelEncoder through factory."""
        encoder = create_encoder(
            "voxel",
            sensor_size=(64, 48),
            voxel_grid=(8, 6, 4)
        )
        
        assert isinstance(encoder, VoxelEncoder)
        assert encoder.sensor_size == (64, 48)
        assert encoder.voxel_grid == (8, 6, 4)
    
    def test_create_sae_encoder(self):
        """Test creating SAEEncoder through factory."""
        encoder = create_encoder(
            "sae",
            sensor_size=(32, 24),
            mixed_polarity=True
        )
        
        assert isinstance(encoder, SAEEncoder)
        assert encoder.sensor_size == (32, 24)
        assert encoder.mixed_polarity == True
    
    def test_create_event_image_encoder(self):
        """Test creating EventImageEncoder through factory."""
        encoder = create_encoder(
            "event_image",
            sensor_size=(16, 12),
            accumulation_mode="binary"
        )
        
        assert isinstance(encoder, EventImageEncoder)
        assert encoder.sensor_size == (16, 12)
        assert encoder.accumulation_mode == "binary"
    
    def test_invalid_encoder_type(self):
        """Test factory with invalid encoder type."""
        with pytest.raises(ValueError, match="Unknown encoding type"):
            create_encoder("invalid_encoder")
    
    def test_all_encoder_types_available(self):
        """Test that all new encoder types are available in factory."""
        expected_types = ["temporal", "spatial", "timeslice", "adaptive", 
                         "voxel", "sae", "event_image"]
        
        for encoder_type in expected_types:
            try:
                encoder = create_encoder(encoder_type, sensor_size=(32, 24))
                assert encoder is not None
            except ValueError:
                pytest.fail(f"Encoder type '{encoder_type}' not available in factory")


class TestEncoderIntegration:
    """Integration tests for all encoders."""
    
    def test_encoder_consistency(self):
        """Test that all encoders handle edge cases consistently."""
        sensor_size = (16, 12)
        
        encoders = [
            create_encoder("voxel", sensor_size=sensor_size),
            create_encoder("sae", sensor_size=sensor_size),  
            create_encoder("event_image", sensor_size=sensor_size),
        ]
        
        # Test with empty events
        empty_events = torch.empty(0, 4)
        for encoder in encoders:
            output = encoder(empty_events)
            assert output is not None
            assert len(output.shape) >= 2
        
        # Test with valid events
        events = torch.tensor([
            [5, 3, 10.0, 1],
            [8, 7, 20.0, -1],
        ], dtype=torch.float32)
        
        for encoder in encoders:
            output = encoder(events)
            assert output is not None
            assert torch.isfinite(output).all()
    
    def test_encoder_reset_functionality(self):
        """Test reset functionality across all encoders."""
        encoders = [
            create_encoder("sae", sensor_size=(8, 6)),
            create_encoder("event_image", sensor_size=(8, 6)),
        ]
        
        events = torch.tensor([[2, 2, 10.0, 1]], dtype=torch.float32)
        
        for encoder in encoders:
            # Process events
            output1 = encoder(events)
            
            # Reset
            encoder.reset()
            
            # Process same events again
            output2 = encoder(events)
            
            # Results should be identical after reset
            assert torch.allclose(output1, output2)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])