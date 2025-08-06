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


class VoxelEncoder(EventEncoder):
    """
    Voxel grid encoding that discretizes events in 3D space-time volumes,
    commonly used in event-based optical flow and object recognition.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,
        voxel_grid: Tuple[int, int, int] = (32, 24, 5),  # (W, H, T)
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(sensor_size, time_window, **kwargs)
        
        self.voxel_grid = voxel_grid
        self.normalize = normalize
        
        # Compute voxel scaling factors
        self.x_scale = voxel_grid[0] / sensor_size[0]
        self.y_scale = voxel_grid[1] / sensor_size[1]
        self.t_scale = voxel_grid[2] / time_window
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Create voxel grid representation of events.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            Voxel grid [2, T, H, W] for positive/negative events
        """
        if events.numel() == 0:
            return torch.zeros(2, self.voxel_grid[2], self.voxel_grid[1], self.voxel_grid[0])
            
        # Extract event components
        x_coords = events[:, 0]
        y_coords = events[:, 1]
        timestamps = events[:, 2]
        polarities = events[:, 3]
        
        # Normalize timestamps to [0, time_window]
        if timestamps.numel() > 1:
            t_min, t_max = timestamps.min(), timestamps.max()
            if t_max > t_min:
                timestamps = (timestamps - t_min) / (t_max - t_min) * self.time_window
        
        # Convert to voxel coordinates
        x_voxels = torch.clamp((x_coords * self.x_scale).long(), 0, self.voxel_grid[0] - 1)
        y_voxels = torch.clamp((y_coords * self.y_scale).long(), 0, self.voxel_grid[1] - 1)
        t_voxels = torch.clamp((timestamps * self.t_scale).long(), 0, self.voxel_grid[2] - 1)
        
        # Create voxel grids for both polarities
        voxel_grids = torch.zeros(2, self.voxel_grid[2], self.voxel_grid[1], self.voxel_grid[0])
        
        # Positive events
        pos_mask = polarities > 0
        if pos_mask.any():
            pos_x, pos_y, pos_t = x_voxels[pos_mask], y_voxels[pos_mask], t_voxels[pos_mask]
            voxel_grids[0, pos_t, pos_y, pos_x] += 1.0
            
        # Negative events  
        neg_mask = polarities <= 0
        if neg_mask.any():
            neg_x, neg_y, neg_t = x_voxels[neg_mask], y_voxels[neg_mask], t_voxels[neg_mask]
            voxel_grids[1, neg_t, neg_y, neg_x] += 1.0
            
        if self.normalize and voxel_grids.sum() > 0:
            voxel_grids = voxel_grids / voxel_grids.sum()
            
        return voxel_grids


class SAEEncoder(EventEncoder):
    """
    Surface of Active Events (SAE) encoding that maintains the most recent
    timestamp for each pixel, creating motion-sensitive representations.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,
        mixed_polarity: bool = False,
        **kwargs
    ):
        super().__init__(sensor_size, time_window, **kwargs)
        
        self.mixed_polarity = mixed_polarity
        
        # Initialize SAE surfaces
        if mixed_polarity:
            # Single channel with polarity encoded as sign
            self.register_buffer('sae_surface', torch.zeros(1, sensor_size[1], sensor_size[0]))
        else:
            # Separate channels for each polarity
            self.register_buffer('sae_surface', torch.zeros(2, sensor_size[1], sensor_size[0]))
            
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Update SAE surface with most recent timestamps.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            SAE surface [1 or 2, height, width] with recent timestamps
        """
        if events.numel() == 0:
            return self.sae_surface.clone()
            
        # Extract event components
        x_coords = events[:, 0].long()
        y_coords = events[:, 1].long()  
        timestamps = events[:, 2]
        polarities = events[:, 3]
        
        # Filter valid coordinates
        valid_mask = (
            (x_coords >= 0) & (x_coords < self.sensor_size[0]) &
            (y_coords >= 0) & (y_coords < self.sensor_size[1])
        )
        
        if not valid_mask.any():
            return self.sae_surface.clone()
            
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        timestamps = timestamps[valid_mask]
        polarities = polarities[valid_mask]
        
        if self.mixed_polarity:
            # Single channel with polarity as sign of timestamp
            self.sae_surface[0, y_coords, x_coords] = timestamps * polarities.sign()
        else:
            # Separate channels for each polarity
            pos_mask = polarities > 0
            neg_mask = polarities <= 0
            
            if pos_mask.any():
                self.sae_surface[0, y_coords[pos_mask], x_coords[pos_mask]] = timestamps[pos_mask]
            if neg_mask.any():
                self.sae_surface[1, y_coords[neg_mask], x_coords[neg_mask]] = timestamps[neg_mask]
                
        return self.sae_surface.clone()
        
    def reset(self) -> None:
        """Reset SAE surface."""
        self.sae_surface.zero_()


class EventImageEncoder(EventEncoder):
    """
    Event image encoding that accumulates events into traditional image-like
    representations for compatibility with CNN architectures.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,
        accumulation_mode: str = "count",  # "count", "binary", "exponential"
        decay_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(sensor_size, time_window, **kwargs)
        
        self.accumulation_mode = accumulation_mode
        self.decay_rate = decay_rate
        
        # Initialize event image
        self.register_buffer('event_image', torch.zeros(2, sensor_size[1], sensor_size[0]))
        self.register_buffer('last_update', torch.tensor(0.0))
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Accumulate events into image representation.
        
        Args:
            events: Event tensor [N, 4] with columns [x, y, t, polarity]
            
        Returns:
            Event image [2, height, width]
        """
        if events.numel() == 0:
            return self.event_image.clone()
            
        # Extract event components
        x_coords = events[:, 0].long()
        y_coords = events[:, 1].long()
        timestamps = events[:, 2]
        polarities = events[:, 3]
        
        # Filter valid coordinates
        valid_mask = (
            (x_coords >= 0) & (x_coords < self.sensor_size[0]) &
            (y_coords >= 0) & (y_coords < self.sensor_size[1])
        )
        
        if not valid_mask.any():
            return self.event_image.clone()
            
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        timestamps = timestamps[valid_mask]
        polarities = polarities[valid_mask]
        
        # Apply temporal decay if using exponential mode
        if self.accumulation_mode == "exponential":
            current_time = timestamps.max()
            dt = current_time - self.last_update
            if dt > 0:
                decay_factor = torch.exp(-dt * self.decay_rate)
                self.event_image *= decay_factor
                self.last_update = current_time
        
        # Accumulate events
        pos_mask = polarities > 0
        neg_mask = polarities <= 0
        
        if self.accumulation_mode == "count":
            # Count-based accumulation
            if pos_mask.any():
                self.event_image[0, y_coords[pos_mask], x_coords[pos_mask]] += 1.0
            if neg_mask.any():
                self.event_image[1, y_coords[neg_mask], x_coords[neg_mask]] += 1.0
        elif self.accumulation_mode == "binary":
            # Binary accumulation  
            if pos_mask.any():
                self.event_image[0, y_coords[pos_mask], x_coords[pos_mask]] = 1.0
            if neg_mask.any():
                self.event_image[1, y_coords[neg_mask], x_coords[neg_mask]] = 1.0
        elif self.accumulation_mode == "exponential":
            # Exponential accumulation with timestamps as weights
            if pos_mask.any():
                weights = torch.exp(timestamps[pos_mask] * self.decay_rate)
                self.event_image[0, y_coords[pos_mask], x_coords[pos_mask]] += weights
            if neg_mask.any():
                weights = torch.exp(timestamps[neg_mask] * self.decay_rate)
                self.event_image[1, y_coords[neg_mask], x_coords[neg_mask]] += weights
                
        return self.event_image.clone()
        
    def reset(self) -> None:
        """Reset event image and timestamp."""
        self.event_image.zero_()
        self.last_update.zero_()


def create_encoder(
    encoding_type: str,
    sensor_size: Tuple[int, int] = (640, 480),
    **kwargs
) -> EventEncoder:
    """
    Factory function to create event encoders.
    
    Args:
        encoding_type: Type of encoder ("temporal", "spatial", "timeslice", "adaptive", 
                      "voxel", "sae", "event_image")
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
        "voxel": VoxelEncoder,
        "sae": SAEEncoder,
        "event_image": EventImageEncoder,
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