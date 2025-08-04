"""
Event-based data loading utilities for liquid neural network training.
Handles temporal batching and event stream processing.
"""

import torch
import torch.utils.data as data
import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path
import random

from ..core.event_encoding import EventEncoder, create_encoder


class EventDataset(data.Dataset):
    """
    Dataset for event-based data with temporal windowing.
    Supports various event encodings and label formats.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        time_window: float = 50.0,  # milliseconds
        encoder_type: str = "temporal",
        encoder_kwargs: Optional[Dict] = None,
        label_type: str = "classification",  # "classification", "regression", "detection"
        stride: Optional[float] = None,  # Time stride for sliding window
        min_events: int = 10,  # Minimum events per sample
        max_events: int = 10000,  # Maximum events per sample
        augmentation: bool = False,
    ):
        self.data_path = Path(data_path)
        self.time_window = time_window
        self.label_type = label_type
        self.stride = stride or time_window / 2  # 50% overlap by default
        self.min_events = min_events
        self.max_events = max_events
        self.augmentation = augmentation
        
        # Initialize encoder
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = create_encoder(encoder_type, **encoder_kwargs)
        
        # Load data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load event data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        if self.data_path.suffix == '.h5':
            self._load_h5_data()
        elif self.data_path.suffix == '.npz':
            self._load_npz_data()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
        # Create temporal windows
        self._create_windows()
        
    def _load_h5_data(self) -> None:
        """Load data from HDF5 file."""
        with h5py.File(self.data_path, 'r') as f:
            self.events = {
                'x': f['events/x'][:],
                'y': f['events/y'][:],
                't': f['events/t'][:],
                'p': f['events/p'][:]
            }
            
            if 'labels' in f:
                self.labels = f['labels'][:]
            else:
                self.labels = None
                
            # Load metadata if available
            self.metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].keys():
                    self.metadata[key] = f['metadata'][key][()]
                    
    def _load_npz_data(self) -> None:
        """Load data from NPZ file."""
        data = np.load(self.data_path)
        
        self.events = {
            'x': data['x'],
            'y': data['y'], 
            't': data['t'],
            'p': data['p']
        }
        
        self.labels = data.get('labels', None)
        self.metadata = {k: v for k, v in data.items() 
                        if k not in ['x', 'y', 't', 'p', 'labels']}
        
    def _create_windows(self) -> None:
        """Create temporal windows for training samples."""
        timestamps = self.events['t']
        t_start = timestamps.min()
        t_end = timestamps.max()
        
        # Generate window start times
        window_starts = []
        current_time = t_start
        
        while current_time + self.time_window <= t_end:
            window_starts.append(current_time)
            current_time += self.stride
            
        self.window_starts = np.array(window_starts)
        
    def __len__(self) -> int:
        return len(self.window_starts)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        window_start = self.window_starts[idx]
        window_end = window_start + self.time_window
        
        # Extract events in time window
        timestamps = self.events['t']
        mask = (timestamps >= window_start) & (timestamps < window_end)
        
        if not mask.any() or mask.sum() < self.min_events:
            # Return empty sample if insufficient events
            return self._get_empty_sample()
            
        # Extract event data
        events_tensor = torch.stack([
            torch.from_numpy(self.events['x'][mask]),
            torch.from_numpy(self.events['y'][mask]),
            torch.from_numpy(self.events['t'][mask]),
            torch.from_numpy(self.events['p'][mask])
        ], dim=1).float()
        
        # Subsample if too many events
        if len(events_tensor) > self.max_events:
            indices = torch.randperm(len(events_tensor))[:self.max_events]
            events_tensor = events_tensor[indices]
            
        # Apply data augmentation
        if self.augmentation:
            events_tensor = self._augment_events(events_tensor)
            
        # Encode events
        encoded = self.encoder(events_tensor)
        
        # Get label
        label = self._get_label(idx, window_start, window_end)
        
        return encoded, label
        
    def _get_empty_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return empty sample for windows with insufficient events."""
        # Create empty encoded representation
        if hasattr(self.encoder, 'sensor_size'):
            if hasattr(self.encoder, 'num_slices'):
                # TimeSliceEncoder
                encoded = torch.zeros(
                    self.encoder.num_slices * 2,
                    self.encoder.down_height,
                    self.encoder.down_width
                )
            elif hasattr(self.encoder, 'feature_dim'):
                # AdaptiveEncoder  
                encoded = torch.zeros(
                    self.encoder.feature_dim,
                    self.encoder.sensor_size[1],
                    self.encoder.sensor_size[0]
                )
            else:
                # TemporalEncoder or SpatialEncoder
                channels = 2 if getattr(self.encoder, 'polarity_separate', True) else 1
                encoded = torch.zeros(
                    channels,
                    self.encoder.sensor_size[1],
                    self.encoder.sensor_size[0]
                )
        else:
            # Default empty tensor
            encoded = torch.zeros(2, 64, 64)
            
        # Default label
        if self.label_type == "classification":
            label = torch.tensor(0, dtype=torch.long)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)
            
        return encoded, label
        
    def _get_label(
        self, 
        idx: int, 
        window_start: float, 
        window_end: float
    ) -> torch.Tensor:
        """Get label for time window."""
        if self.labels is None:
            # No labels available - return dummy label
            if self.label_type == "classification":
                return torch.tensor(0, dtype=torch.long)
            else:
                return torch.tensor(0.0, dtype=torch.float32)
                
        # Simple approach: use label at window center
        window_center = (window_start + window_end) / 2
        
        if self.labels.ndim == 1:
            # Single label per sample
            label_idx = min(idx, len(self.labels) - 1)
            label = self.labels[label_idx]
        else:
            # Time-synchronized labels
            # Find closest label timestamp
            label_times = self.labels[:, 0]  # Assume first column is time
            time_diffs = np.abs(label_times - window_center)
            closest_idx = np.argmin(time_diffs)
            label = self.labels[closest_idx, 1:]  # Skip time column
            
        if self.label_type == "classification":
            return torch.tensor(label, dtype=torch.long)
        else:
            return torch.tensor(label, dtype=torch.float32)
            
    def _augment_events(self, events: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to events."""
        if not self.training:
            return events
            
        # Spatial augmentation
        if random.random() < 0.5:
            # Horizontal flip
            if hasattr(self.encoder, 'sensor_size'):
                events[:, 0] = self.encoder.sensor_size[0] - 1 - events[:, 0]
                
        # Temporal augmentation
        if random.random() < 0.3:
            # Add temporal jitter
            jitter = torch.randn(len(events)) * 0.5
            events[:, 2] += jitter
            
        # Polarity flip
        if random.random() < 0.2:
            events[:, 3] *= -1
            
        return events


class EventDataLoader:
    """
    Data loader for event-based datasets with specialized batching.
    Handles variable-length event sequences and temporal alignment.
    """
    
    def __init__(
        self,
        dataset: EventDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        temporal_batching: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.temporal_batching = temporal_batching
        
        # Create PyTorch DataLoader
        self.dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=self._collate_fn if temporal_batching else None
        )
        
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for temporal batching."""
        encoded_list, labels_list = zip(*batch)
        
        # Stack encoded features
        encoded_batch = torch.stack(encoded_list, dim=0)
        
        # Handle labels
        if labels_list[0].dim() == 0:
            # Scalar labels
            labels_batch = torch.stack(labels_list, dim=0)
        else:
            # Vector labels - pad if necessary
            max_len = max(label.size(0) for label in labels_list)
            padded_labels = []
            
            for label in labels_list:
                if label.size(0) < max_len:
                    padding = torch.zeros(max_len - label.size(0), *label.shape[1:])
                    padded = torch.cat([label, padding], dim=0)
                else:
                    padded = label
                padded_labels.append(padded)
                
            labels_batch = torch.stack(padded_labels, dim=0)
            
        return encoded_batch, labels_batch
        
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.dataloader)
        
    def __len__(self) -> int:
        return len(self.dataloader)


class SyntheticEventDataset(EventDataset):
    """
    Synthetic event dataset generator for testing and development.
    Creates simple patterns for classification and regression tasks.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        resolution: Tuple[int, int] = (64, 64),
        time_window: float = 50.0,
        task_type: str = "classification",  # "classification", "counting", "motion"
        **kwargs
    ):
        self.num_samples = num_samples
        self.resolution = resolution
        self.task_type = task_type
        
        # Don't call parent __init__ as we generate data differently
        self.time_window = time_window
        self.encoder = create_encoder(
            kwargs.get('encoder_type', 'temporal'),
            sensor_size=resolution,
            **kwargs.get('encoder_kwargs', {})
        )
        
        # Generate synthetic data
        self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic event data and labels."""
        self.samples = []
        
        for i in range(self.num_samples):
            events, label = self._generate_sample(i)
            self.samples.append((events, label))
            
    def _generate_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single synthetic sample."""
        if self.task_type == "classification":
            return self._generate_classification_sample(idx)
        elif self.task_type == "counting":
            return self._generate_counting_sample(idx)
        elif self.task_type == "motion":
            return self._generate_motion_sample(idx)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
            
    def _generate_classification_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample for classification task (detect shapes)."""
        # Random class (0: circle, 1: square, 2: triangle)
        class_id = idx % 3
        
        # Generate events based on class
        if class_id == 0:  # Circle
            events = self._generate_circle_events()
        elif class_id == 1:  # Square
            events = self._generate_square_events()
        else:  # Triangle
            events = self._generate_triangle_events()
            
        label = torch.tensor(class_id, dtype=torch.long)
        return events, label
        
    def _generate_counting_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample for counting task (count objects)."""
        # Random number of objects (1-5)
        num_objects = (idx % 5) + 1
        
        events_list = []
        for _ in range(num_objects):
            obj_events = self._generate_circle_events()
            # Add random spatial offset
            offset_x = random.randint(-20, 20)
            offset_y = random.randint(-20, 20)
            obj_events[:, 0] += offset_x
            obj_events[:, 1] += offset_y
            
            # Clip to sensor bounds
            obj_events[:, 0] = torch.clamp(obj_events[:, 0], 0, self.resolution[0] - 1)
            obj_events[:, 1] = torch.clamp(obj_events[:, 1], 0, self.resolution[1] - 1)
            
            events_list.append(obj_events)
            
        events = torch.cat(events_list, dim=0)
        label = torch.tensor(num_objects, dtype=torch.float32)
        return events, label
        
    def _generate_motion_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample for motion detection task."""
        # Random motion direction (0: left, 1: right, 2: up, 3: down)
        motion_dir = idx % 4
        
        # Generate moving object
        center_x = self.resolution[0] // 2
        center_y = self.resolution[1] // 2
        
        events_list = []
        num_time_steps = 10
        
        for t in range(num_time_steps):
            # Update position based on motion direction
            if motion_dir == 0:  # Left
                pos_x = center_x - t * 2
                pos_y = center_y
            elif motion_dir == 1:  # Right
                pos_x = center_x + t * 2
                pos_y = center_y
            elif motion_dir == 2:  # Up
                pos_x = center_x
                pos_y = center_y - t * 2
            else:  # Down
                pos_x = center_x
                pos_y = center_y + t * 2
                
            # Generate events at current position
            obj_events = self._generate_circle_events(center=(pos_x, pos_y), radius=5)
            obj_events[:, 2] += t * (self.time_window / num_time_steps)  # Add temporal offset
            
            events_list.append(obj_events)
            
        events = torch.cat(events_list, dim=0)
        label = torch.tensor(motion_dir, dtype=torch.long)
        return events, label
        
    def _generate_circle_events(
        self, 
        center: Optional[Tuple[int, int]] = None,
        radius: int = 10,
        num_events: int = 100
    ) -> torch.Tensor:
        """Generate events forming a circle."""
        if center is None:
            center = (self.resolution[0] // 2, self.resolution[1] // 2)
            
        # Generate points on circle
        angles = torch.rand(num_events) * 2 * np.pi
        radii = torch.rand(num_events) * radius
        
        x_coords = center[0] + radii * torch.cos(angles)
        y_coords = center[1] + radii * torch.sin(angles)
        
        # Random timestamps
        timestamps = torch.rand(num_events) * self.time_window
        
        # Random polarities
        polarities = torch.randint(0, 2, (num_events,)) * 2 - 1
        
        events = torch.stack([
            x_coords, y_coords, timestamps, polarities.float()
        ], dim=1)
        
        return events
        
    def _generate_square_events(self, size: int = 20, num_events: int = 100) -> torch.Tensor:
        """Generate events forming a square."""
        center_x = self.resolution[0] // 2
        center_y = self.resolution[1] // 2
        
        # Generate points on square perimeter
        x_coords = []
        y_coords = []
        
        for _ in range(num_events):
            side = random.randint(0, 3)
            
            if side == 0:  # Top
                x = random.randint(-size//2, size//2)
                y = -size//2
            elif side == 1:  # Right
                x = size//2
                y = random.randint(-size//2, size//2)
            elif side == 2:  # Bottom
                x = random.randint(-size//2, size//2)
                y = size//2
            else:  # Left
                x = -size//2
                y = random.randint(-size//2, size//2)
                
            x_coords.append(center_x + x)
            y_coords.append(center_y + y)
            
        x_coords = torch.tensor(x_coords, dtype=torch.float32)
        y_coords = torch.tensor(y_coords, dtype=torch.float32)
        timestamps = torch.rand(num_events) * self.time_window
        polarities = torch.randint(0, 2, (num_events,)) * 2 - 1
        
        events = torch.stack([
            x_coords, y_coords, timestamps, polarities.float()
        ], dim=1)
        
        return events
        
    def _generate_triangle_events(self, size: int = 20, num_events: int = 100) -> torch.Tensor:
        """Generate events forming a triangle."""
        center_x = self.resolution[0] // 2
        center_y = self.resolution[1] // 2
        
        # Triangle vertices
        vertices = [
            (0, -size//2),      # Top
            (-size//2, size//2), # Bottom left
            (size//2, size//2)   # Bottom right
        ]
        
        x_coords = []
        y_coords = []
        
        for _ in range(num_events):
            # Pick random edge
            edge = random.randint(0, 2)
            t = random.random()
            
            v1 = vertices[edge]
            v2 = vertices[(edge + 1) % 3]
            
            # Linear interpolation along edge
            x = v1[0] + t * (v2[0] - v1[0])
            y = v1[1] + t * (v2[1] - v1[1])
            
            x_coords.append(center_x + x)
            y_coords.append(center_y + y)
            
        x_coords = torch.tensor(x_coords, dtype=torch.float32)
        y_coords = torch.tensor(y_coords, dtype=torch.float32)
        timestamps = torch.rand(num_events) * self.time_window
        polarities = torch.randint(0, 2, (num_events,)) * 2 - 1
        
        events = torch.stack([
            x_coords, y_coords, timestamps, polarities.float()
        ], dim=1)
        
        return events
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoded sample."""
        events, label = self.samples[idx]
        encoded = self.encoder(events)
        return encoded, label