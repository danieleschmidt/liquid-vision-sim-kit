"""
Event-based camera simulation for generating synthetic neuromorphic data.
Implements DVS (Dynamic Vision Sensor) and DAVIS camera models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cv2
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .noise_models import NoiseModel, ShotNoise


@dataclass
class EventData:
    """Container for event data."""
    x: np.ndarray  # X coordinates
    y: np.ndarray  # Y coordinates
    t: np.ndarray  # Timestamps (milliseconds)
    p: np.ndarray  # Polarity (+1 or -1)
    
    def __len__(self) -> int:
        return len(self.x)
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to PyTorch tensor [N, 4] format."""
        events = np.stack([self.x, self.y, self.t, self.p], axis=1)
        tensor = torch.from_numpy(events).float()
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def filter_spatial(self, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> 'EventData':
        """Filter events by spatial coordinates."""
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        mask = (
            (self.x >= x_min) & (self.x < x_max) &
            (self.y >= y_min) & (self.y < y_max)
        )
        
        return EventData(
            x=self.x[mask],
            y=self.y[mask], 
            t=self.t[mask],
            p=self.p[mask]
        )
    
    def filter_temporal(self, t_start: float, t_end: float) -> 'EventData':
        """Filter events by time window."""
        mask = (self.t >= t_start) & (self.t <= t_end)
        
        return EventData(
            x=self.x[mask],
            y=self.y[mask],
            t=self.t[mask],
            p=self.p[mask]
        )


class EventSimulator(ABC):
    """Base class for event-based camera simulators."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        contrast_threshold: float = 0.15,
        noise_model: Optional[NoiseModel] = None,
        refractory_period: float = 1.0,  # milliseconds
    ):
        self.resolution = resolution
        self.contrast_threshold = contrast_threshold
        self.noise_model = noise_model or ShotNoise(rate=0.01)
        self.refractory_period = refractory_period
        
        # State variables
        self.last_frame = None
        self.last_log_frame = None
        self.last_event_time = np.zeros(resolution[::-1])  # (height, width)
        self.reset()
        
    def reset(self) -> None:
        """Reset simulator state."""
        self.last_frame = None
        self.last_log_frame = None
        self.last_event_time.fill(0.0)
        
    @abstractmethod
    def simulate_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> EventData:
        """
        Simulate events for a single frame.
        
        Args:
            frame: Input frame as grayscale image [height, width]
            timestamp: Frame timestamp in milliseconds
            
        Returns:
            Generated events
        """
        pass
        
    def simulate_video(
        self,
        frames: Union[np.ndarray, List[np.ndarray]],
        timestamps: Optional[np.ndarray] = None,
        fps: float = 30.0,
    ) -> EventData:
        """
        Simulate events for video sequence.
        
        Args:
            frames: Video frames [num_frames, height, width] or list of frames
            timestamps: Frame timestamps in milliseconds
            fps: Frame rate if timestamps not provided
            
        Returns:
            All generated events
        """
        if isinstance(frames, list):
            frames = np.array(frames)
            
        num_frames = len(frames)
        
        if timestamps is None:
            timestamps = np.arange(num_frames) * (1000.0 / fps)
            
        all_events = []
        
        for frame, timestamp in zip(frames, timestamps):
            events = self.simulate_frame(frame, timestamp)
            if len(events) > 0:
                all_events.append(events)
                
        if not all_events:
            return EventData(
                x=np.array([], dtype=np.int32),
                y=np.array([], dtype=np.int32),
                t=np.array([], dtype=np.float32),
                p=np.array([], dtype=np.int8)
            )
            
        # Concatenate all events
        return EventData(
            x=np.concatenate([e.x for e in all_events]),
            y=np.concatenate([e.y for e in all_events]),
            t=np.concatenate([e.t for e in all_events]),
            p=np.concatenate([e.p for e in all_events])
        )


class DVSSimulator(EventSimulator):
    """
    Dynamic Vision Sensor (DVS) simulator.
    Generates events based on brightness changes exceeding threshold.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        contrast_threshold: float = 0.15,
        **kwargs
    ):
        super().__init__(resolution, contrast_threshold, **kwargs)
        
    def simulate_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> EventData:
        """
        Generate DVS events for single frame.
        
        Events are generated where log brightness change exceeds threshold:
        |log(I_new) - log(I_old)| > threshold
        """
        # Ensure frame is grayscale and float
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        
        # Add small epsilon to avoid log(0)
        frame = np.clip(frame, 1e-6, 1.0)
        
        if self.last_frame is None:
            self.last_frame = frame.copy()
            self.last_log_frame = np.log(frame)
            return EventData(
                x=np.array([], dtype=np.int32),
                y=np.array([], dtype=np.int32),
                t=np.array([], dtype=np.float32),
                p=np.array([], dtype=np.int8)
            )
            
        # Compute log brightness difference
        current_log_frame = np.log(frame)
        log_diff = current_log_frame - self.last_log_frame
        
        # Find pixels exceeding threshold
        pos_events = log_diff > self.contrast_threshold
        neg_events = log_diff < -self.contrast_threshold
        
        # Apply refractory period
        time_since_last = timestamp - self.last_event_time
        refractory_mask = time_since_last >= self.refractory_period
        
        pos_events = pos_events & refractory_mask
        neg_events = neg_events & refractory_mask
        
        # Extract event coordinates
        pos_coords = np.where(pos_events)
        neg_coords = np.where(neg_events)
        
        # Create event arrays
        num_pos = len(pos_coords[0])
        num_neg = len(neg_coords[0])
        
        x_coords = np.concatenate([
            pos_coords[1],  # x = column
            neg_coords[1]
        ])
        y_coords = np.concatenate([
            pos_coords[0],  # y = row
            neg_coords[0]
        ])
        timestamps_arr = np.full(num_pos + num_neg, timestamp, dtype=np.float32)
        polarities = np.concatenate([
            np.ones(num_pos, dtype=np.int8),
            -np.ones(num_neg, dtype=np.int8)
        ])
        
        # Update state
        self.last_frame = frame.copy()
        self.last_log_frame = current_log_frame.copy()
        
        # Update last event times for refractory period
        if len(x_coords) > 0:
            self.last_event_time[y_coords, x_coords] = timestamp
            
        # Create events
        events = EventData(
            x=x_coords.astype(np.int32),
            y=y_coords.astype(np.int32),
            t=timestamps_arr,
            p=polarities
        )
        
        # Apply noise model if specified
        if self.noise_model is not None:
            events = self.noise_model.apply(events, timestamp)
            
        return events


class DAVISSimulator(DVSSimulator):
    """
    DAVIS (Dynamic and Active Pixel Vision Sensor) simulator.
    Combines DVS events with APS (Active Pixel Sensor) frames.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        contrast_threshold: float = 0.15,
        aps_exposure_time: float = 33.33,  # milliseconds (30 FPS)
        **kwargs
    ):
        super().__init__(resolution, contrast_threshold, **kwargs)
        self.aps_exposure_time = aps_exposure_time
        self.last_aps_time = 0.0
        
    def simulate_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> Tuple[EventData, Optional[np.ndarray]]:
        """
        Generate DAVIS events and APS frame.
        
        Returns:
            Tuple of (events, aps_frame)
            aps_frame is None if not time for APS capture
        """
        # Generate DVS events
        events = super().simulate_frame(frame, timestamp)
        
        # Check if time for APS frame
        aps_frame = None
        if timestamp - self.last_aps_time >= self.aps_exposure_time:
            aps_frame = frame.copy()
            self.last_aps_time = timestamp
            
        return events, aps_frame


class AdvancedDVSSimulator(DVSSimulator):
    """
    Advanced DVS simulator with additional realistic effects.
    Includes spatial and temporal filtering, latency, and bandwidth limitations.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        contrast_threshold: float = 0.15,
        temporal_filter_tau: float = 10.0,  # milliseconds
        spatial_filter_sigma: float = 0.5,
        latency_mean: float = 0.1,  # milliseconds
        latency_std: float = 0.05,
        bandwidth_limit: Optional[int] = None,  # max events per frame
        **kwargs
    ):
        super().__init__(resolution, contrast_threshold, **kwargs)
        
        self.temporal_filter_tau = temporal_filter_tau
        self.spatial_filter_sigma = spatial_filter_sigma
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.bandwidth_limit = bandwidth_limit
        
        # Temporal filter state
        self.filtered_log_frame = None
        
    def simulate_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> EventData:
        """Advanced DVS simulation with realistic effects."""
        # Preprocess frame
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.clip(frame, 1e-6, 1.0)
        
        # Apply spatial filtering (simulates pixel cross-talk)
        if self.spatial_filter_sigma > 0:
            kernel_size = int(6 * self.spatial_filter_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), self.spatial_filter_sigma)
            
        current_log_frame = np.log(frame)
        
        if self.last_log_frame is None:
            self.last_log_frame = current_log_frame.copy()
            self.filtered_log_frame = current_log_frame.copy()
            return EventData(
                x=np.array([], dtype=np.int32),
                y=np.array([], dtype=np.int32),
                t=np.array([], dtype=np.float32),
                p=np.array([], dtype=np.int8)
            )
            
        # Apply temporal filtering
        if self.filtered_log_frame is None:
            self.filtered_log_frame = current_log_frame.copy()
        else:
            alpha = np.exp(-1.0 / self.temporal_filter_tau)
            self.filtered_log_frame = (
                alpha * self.filtered_log_frame + 
                (1 - alpha) * current_log_frame
            )
            
        # Compute brightness changes using filtered frame
        log_diff = self.filtered_log_frame - self.last_log_frame
        
        # Event detection
        pos_events = log_diff > self.contrast_threshold
        neg_events = log_diff < -self.contrast_threshold
        
        # Apply refractory period
        time_since_last = timestamp - self.last_event_time
        refractory_mask = time_since_last >= self.refractory_period
        
        pos_events = pos_events & refractory_mask
        neg_events = neg_events & refractory_mask
        
        # Extract coordinates
        pos_coords = np.where(pos_events)
        neg_coords = np.where(neg_events)
        
        num_pos = len(pos_coords[0])
        num_neg = len(neg_coords[0])
        total_events = num_pos + num_neg
        
        # Apply bandwidth limitation
        if self.bandwidth_limit is not None and total_events > self.bandwidth_limit:
            # Randomly subsample events
            indices = np.random.choice(total_events, self.bandwidth_limit, replace=False)
            indices.sort()
            
            # Split indices for pos/neg events
            pos_indices = indices[indices < num_pos]
            neg_indices = indices[indices >= num_pos] - num_pos
            
            pos_coords = (pos_coords[0][pos_indices], pos_coords[1][pos_indices])
            neg_coords = (neg_coords[0][neg_indices], neg_coords[1][neg_indices])
            
            num_pos = len(pos_coords[0])
            num_neg = len(neg_coords[0])
            
        # Create event arrays
        x_coords = np.concatenate([pos_coords[1], neg_coords[1]])
        y_coords = np.concatenate([pos_coords[0], neg_coords[0]])
        polarities = np.concatenate([
            np.ones(num_pos, dtype=np.int8),
            -np.ones(num_neg, dtype=np.int8)
        ])
        
        # Add latency jitter
        latencies = np.random.normal(
            self.latency_mean, 
            self.latency_std, 
            size=len(x_coords)
        )
        latencies = np.maximum(latencies, 0)  # Ensure non-negative
        timestamps_arr = np.full(len(x_coords), timestamp, dtype=np.float32) + latencies
        
        # Update state
        self.last_log_frame = self.filtered_log_frame.copy()
        if len(x_coords) > 0:
            self.last_event_time[y_coords, x_coords] = timestamp
            
        # Create events
        events = EventData(
            x=x_coords.astype(np.int32),
            y=y_coords.astype(np.int32),
            t=timestamps_arr,
            p=polarities
        )
        
        # Apply noise model
        if self.noise_model is not None:
            events = self.noise_model.apply(events, timestamp)
            
        return events


def create_simulator(
    simulator_type: str = "dvs",
    resolution: Tuple[int, int] = (640, 480),
    **kwargs
) -> EventSimulator:
    """
    Factory function to create event simulators.
    
    Args:
        simulator_type: Type of simulator ("dvs", "davis", "advanced_dvs")
        resolution: Camera resolution
        **kwargs: Additional simulator-specific arguments
        
    Returns:
        Configured simulator instance
    """
    simulators = {
        "dvs": DVSSimulator,
        "davis": DAVISSimulator,
        "advanced_dvs": AdvancedDVSSimulator,
    }
    
    if simulator_type not in simulators:
        raise ValueError(f"Unknown simulator type: {simulator_type}")
        
    return simulators[simulator_type](resolution=resolution, **kwargs)