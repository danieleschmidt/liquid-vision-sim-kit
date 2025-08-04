"""
Event encoding schemes for converting event-based camera data into neural network inputs.
Supports various temporal and spatial encoding strategies for neuromorphic computing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math


class EventEncoder(nn.Module):
    """
    Base class for event encoding schemes.
    Converts raw event data (x, y, t, polarity) into tensor representations.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,  # milliseconds
        dt: float = 1.0,
    ):
        super().__init__()
        self.sensor_size = sensor_size
        self.time_window = time_window
        self.dt = dt
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Encode events into tensor representation.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            Encoded tensor representation
        """
        raise NotImplementedError
        
    def reset(self) -> None:
        """Reset encoder state."""
        pass


class TemporalEncoder(EventEncoder):
    """
    Temporal surface encoding that maintains exponentially decaying activations
    for each pixel location, creating a time-aware spatial representation.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,
        tau_decay: float = 20.0,  # decay time constant in ms
        polarity_separate: bool = True,
        **kwargs
    ):
        super().__init__(sensor_size, time_window, **kwargs)
        
        self.tau_decay = tau_decay
        self.polarity_separate = polarity_separate
        
        # Initialize temporal surfaces
        num_channels = 2 if polarity_separate else 1
        self.register_buffer(
            'temporal_surface',
            torch.zeros(num_channels, sensor_size[1], sensor_size[0])
        )
        self.register_buffer('last_timestamp', torch.tensor(0.0))
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Update temporal surface with new events.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            Temporal surface [channels, height, width]
        """
        if events.numel() == 0:
            return self.temporal_surface.clone()
            
        # Extract event components
        x_coords = events[:, 0].long()
        y_coords = events[:, 1].long()
        timestamps = events[:, 2]
        polarities = events[:, 3]
        
        # Ensure coordinates are within bounds
        valid_mask = (
            (x_coords >= 0) & (x_coords < self.sensor_size[0]) &
            (y_coords >= 0) & (y_coords < self.sensor_size[1])
        )
        
        if not valid_mask.any():
            return self.temporal_surface.clone()
            
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        timestamps = timestamps[valid_mask]
        polarities = polarities[valid_mask]
        
        # Apply temporal decay
        current_time = timestamps.max()
        dt = current_time - self.last_timestamp
        
        if dt > 0:
            decay_factor = torch.exp(-dt / self.tau_decay)
            self.temporal_surface *= decay_factor
            
        # Update surface with new events
        if self.polarity_separate:
            # Separate positive and negative events
            pos_mask = polarities > 0
            neg_mask = polarities <= 0
            
            if pos_mask.any():
                self.temporal_surface[0, y_coords[pos_mask], x_coords[pos_mask]] = 1.0
            if neg_mask.any():
                self.temporal_surface[1, y_coords[neg_mask], x_coords[neg_mask]] = 1.0
        else:
            # Combine polarities with sign
            self.temporal_surface[0, y_coords, x_coords] = polarities
            
        self.last_timestamp = current_time
        return self.temporal_surface.clone()
        
    def reset(self) -> None:
        """Reset temporal surface and timestamp."""
        self.temporal_surface.zero_()
        self.last_timestamp.zero_()


class SpatialEncoder(EventEncoder):
    """
    Spatial histogram encoding that accumulates events in spatial bins
    over a fixed time window.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        spatial_bins: Tuple[int, int] = (32, 24),
        time_window: float = 50.0,
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(sensor_size, time_window, **kwargs)
        
        self.spatial_bins = spatial_bins
        self.normalize = normalize
        
        # Compute spatial binning factors
        self.x_factor = spatial_bins[0] / sensor_size[0]
        self.y_factor = spatial_bins[1] / sensor_size[1]
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Create spatial histogram of events.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            Spatial histogram [2, spatial_bins[1], spatial_bins[0]]
        """
        if events.numel() == 0:
            return torch.zeros(2, self.spatial_bins[1], self.spatial_bins[0])
            
        # Extract coordinates and polarities
        x_coords = events[:, 0]
        y_coords = events[:, 1]
        polarities = events[:, 3]
        
        # Convert to spatial bins
        x_bins = torch.clamp((x_coords * self.x_factor).long(), 0, self.spatial_bins[0] - 1)
        y_bins = torch.clamp((y_coords * self.y_factor).long(), 0, self.spatial_bins[1] - 1)
        
        # Create histograms for positive and negative events
        pos_mask = polarities > 0
        neg_mask = polarities <= 0
        
        histogram = torch.zeros(2, self.spatial_bins[1], self.spatial_bins[0])
        
        if pos_mask.any():
            pos_hist = torch.histogramdd(
                torch.stack([y_bins[pos_mask].float(), x_bins[pos_mask].float()], dim=1),
                bins=self.spatial_bins[::-1],  # histogramdd expects (y, x) order
                range=[(0, self.spatial_bins[1]), (0, self.spatial_bins[0])]
            )[0]
            histogram[0] = pos_hist
            
        if neg_mask.any():
            neg_hist = torch.histogramdd(
                torch.stack([y_bins[neg_mask].float(), x_bins[neg_mask].float()], dim=1),
                bins=self.spatial_bins[::-1],
                range=[(0, self.spatial_bins[1]), (0, self.spatial_bins[0])]
            )[0]
            histogram[1] = neg_hist
            
        if self.normalize and histogram.sum() > 0:
            histogram = histogram / histogram.sum()
            
        return histogram


class TimeSliceEncoder(EventEncoder):
    """
    Time slice encoding that creates multiple temporal slices
    within the time window for capturing temporal dynamics.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,
        num_slices: int = 5,
        spatial_downsampling: int = 8,
        **kwargs
    ):
        super().__init__(sensor_size, time_window, **kwargs)
        
        self.num_slices = num_slices
        self.spatial_downsampling = spatial_downsampling
        
        # Compute downsampled dimensions
        self.down_height = sensor_size[1] // spatial_downsampling
        self.down_width = sensor_size[0] // spatial_downsampling
        
        self.slice_duration = time_window / num_slices
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Create time slice representation.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            Time slices [num_slices * 2, down_height, down_width]
        """
        if events.numel() == 0:
            return torch.zeros(
                self.num_slices * 2, 
                self.down_height, 
                self.down_width
            )
            
        # Normalize timestamps to [0, time_window]
        timestamps = events[:, 2]
        if timestamps.numel() > 1:
            t_min, t_max = timestamps.min(), timestamps.max()
            if t_max > t_min:
                timestamps = (timestamps - t_min) / (t_max - t_min) * self.time_window
                
        # Assign events to time slices
        slice_indices = torch.clamp(
            (timestamps / self.slice_duration).long(),
            0, self.num_slices - 1
        )
        
        # Downsample spatial coordinates
        x_coords = (events[:, 0] / self.spatial_downsampling).long()
        y_coords = (events[:, 1] / self.spatial_downsampling).long()
        polarities = events[:, 3]
        
        # Clip to valid range
        x_coords = torch.clamp(x_coords, 0, self.down_width - 1)
        y_coords = torch.clamp(y_coords, 0, self.down_height - 1)
        
        # Create time slice tensor
        time_slices = torch.zeros(
            self.num_slices * 2,
            self.down_height,
            self.down_width
        )
        
        # Fill time slices
        for slice_idx in range(self.num_slices):
            slice_mask = slice_indices == slice_idx
            if not slice_mask.any():
                continue
                
            slice_x = x_coords[slice_mask]
            slice_y = y_coords[slice_mask]
            slice_pol = polarities[slice_mask]
            
            # Positive events
            pos_mask = slice_pol > 0
            if pos_mask.any():
                time_slices[slice_idx * 2, slice_y[pos_mask], slice_x[pos_mask]] = 1.0
                
            # Negative events
            neg_mask = slice_pol <= 0
            if neg_mask.any():
                time_slices[slice_idx * 2 + 1, slice_y[neg_mask], slice_x[neg_mask]] = 1.0
                
        return time_slices


class AdaptiveEncoder(EventEncoder):
    """
    Adaptive encoder that learns optimal encoding parameters
    through differentiable operations.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        feature_dim: int = 64,
        time_window: float = 50.0,
        learnable_tau: bool = True,
        **kwargs
    ):
        super().__init__(sensor_size, time_window, **kwargs)
        
        self.feature_dim = feature_dim
        
        # Learnable spatial transformation
        self.spatial_transform = nn.Conv2d(2, feature_dim, kernel_size=3, padding=1)
        
        # Learnable temporal parameters
        if learnable_tau:
            self.tau_decay = nn.Parameter(torch.tensor(20.0))
        else:
            self.register_buffer('tau_decay', torch.tensor(20.0))
            
        # Feature normalization
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Learn adaptive encoding representation.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            Learned features [feature_dim, height, width]
        """
        # First create temporal surface
        temporal_surface = self._create_temporal_surface(events)
        
        # Apply learnable spatial transformation
        features = self.spatial_transform(temporal_surface.unsqueeze(0))
        features = features.squeeze(0)
        
        # Normalize features
        b, h, w = features.shape
        features_flat = features.view(b, -1).transpose(0, 1)  # [h*w, feature_dim]
        features_norm = self.norm(features_flat)
        features = features_norm.transpose(0, 1).view(b, h, w)
        
        return features
        
    def _create_temporal_surface(self, events: torch.Tensor) -> torch.Tensor:
        """Create basic temporal surface for spatial transformation."""
        if events.numel() == 0:
            return torch.zeros(2, self.sensor_size[1], self.sensor_size[0])
            
        surface = torch.zeros(2, self.sensor_size[1], self.sensor_size[0])
        
        x_coords = events[:, 0].long()
        y_coords = events[:, 1].long()
        polarities = events[:, 3]
        
        # Clip coordinates
        x_coords = torch.clamp(x_coords, 0, self.sensor_size[0] - 1)
        y_coords = torch.clamp(y_coords, 0, self.sensor_size[1] - 1)
        
        # Fill surface
        pos_mask = polarities > 0
        neg_mask = polarities <= 0
        
        if pos_mask.any():
            surface[0, y_coords[pos_mask], x_coords[pos_mask]] = 1.0
        if neg_mask.any():
            surface[1, y_coords[neg_mask], x_coords[neg_mask]] = 1.0
            
        return surface


def create_encoder(
    encoding_type: str,
    sensor_size: Tuple[int, int] = (640, 480),
    **kwargs
) -> EventEncoder:
    """
    Factory function to create event encoders.
    
    Args:
        encoding_type: Type of encoder ("temporal", "spatial", "timeslice", "adaptive")
        sensor_size: Camera sensor dimensions
        **kwargs: Additional encoder-specific arguments
        
    Returns:
        Configured encoder instance
    """
    encoders = {
        "temporal": TemporalEncoder,
        "spatial": SpatialEncoder,
        "timeslice": TimeSliceEncoder,
        "adaptive": AdaptiveEncoder,
    }
    
    if encoding_type not in encoders:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
        
    return encoders[encoding_type](sensor_size=sensor_size, **kwargs)


def events_to_tensor(
    events: Union[np.ndarray, List],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert event data to PyTorch tensor.
    
    Args:
        events: Event data as numpy array or list
        device: Target device for tensor
        
    Returns:
        Event tensor [N, 4] with columns [x, y, t, polarity]
    """
    if isinstance(events, list):
        events = np.array(events)
        
    if events.ndim == 1 and len(events) == 0:
        tensor = torch.empty(0, 4)
    else:
        tensor = torch.from_numpy(events).float()
        
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
        
    if device is not None:
        tensor = tensor.to(device)
        
    return tensor