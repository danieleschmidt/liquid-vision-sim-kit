"""
Custom dataset loader for event-based data from various formats.
Supports HDF5, NumPy, and other common neuromorphic data formats.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Union, Tuple, Any
from pathlib import Path

from ..core.event_encoding import create_encoder, events_to_tensor


class CustomEventDataset(Dataset):
    """
    Dataset for loading custom event data from various formats.
    Supports HDF5, NumPy arrays, and structured event data.
    """
    
    def __init__(
        self,
        events_data: Union[np.ndarray, list],
        labels_data: Optional[Union[np.ndarray, list]] = None,
        encoder_type: str = "temporal",
        resolution: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,
        normalize: bool = True,
        augment: bool = False,
        **encoder_kwargs
    ):
        """
        Initialize custom event dataset.
        
        Args:
            events_data: Event data as numpy array [N, 4] or list of event arrays
            labels_data: Optional labels for supervised learning
            encoder_type: Type of event encoder to use
            resolution: Sensor resolution (width, height)
            time_window: Time window for encoding (ms)
            normalize: Whether to normalize event coordinates
            augment: Whether to apply data augmentation
            **encoder_kwargs: Additional arguments for encoder
        """
        
        # Store raw data
        if isinstance(events_data, list):
            self.events_data = events_data
        else:
            # Assume single array - need to chunk into sequences
            self.events_data = self._chunk_events(events_data)
            
        self.labels_data = labels_data
        self.resolution = resolution
        self.time_window = time_window
        self.normalize = normalize
        self.augment = augment
        
        # Create encoder
        self.encoder = create_encoder(
            encoder_type,
            sensor_size=resolution,
            time_window=time_window,
            **encoder_kwargs
        )
        
        # Validate data consistency
        if labels_data is not None:
            assert len(self.events_data) == len(labels_data), \
                "Events and labels must have same length"
                
        print(f"Loaded {len(self.events_data)} event sequences")
        if labels_data is not None:
            print(f"Labels shape: {np.array(labels_data).shape}")
            
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.events_data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get encoded events and optional label.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (encoded_events, label)
        """
        # Get event data
        events = self.events_data[idx]
        
        # Convert to tensor
        if not isinstance(events, torch.Tensor):
            events = events_to_tensor(events)
            
        # Normalize coordinates if requested
        if self.normalize:
            events = self._normalize_events(events)
            
        # Apply augmentation if requested
        if self.augment:
            events = self._augment_events(events)
            
        # Encode events
        self.encoder.reset()
        encoded = self.encoder(events)
        
        # Get label if available
        label = None
        if self.labels_data is not None:
            label = torch.tensor(self.labels_data[idx])
            
        return encoded, label
        
    def _chunk_events(self, events_array: np.ndarray, chunk_size: int = 10000) -> list:
        """
        Chunk a large event array into sequences.
        
        Args:
            events_array: Large event array [N, 4]
            chunk_size: Events per chunk
            
        Returns:
            List of event chunks
        """
        if events_array.shape[1] != 4:
            raise ValueError("Events must have shape [N, 4] with [x, y, t, polarity]")
            
        chunks = []
        for i in range(0, len(events_array), chunk_size):
            chunk = events_array[i:i + chunk_size]
            chunks.append(chunk)
            
        return chunks
        
    def _normalize_events(self, events: torch.Tensor) -> torch.Tensor:
        """
        Normalize event coordinates to [0, 1] range.
        
        Args:
            events: Event tensor [N, 4]
            
        Returns:
            Normalized event tensor
        """
        if events.numel() == 0:
            return events
            
        normalized = events.clone()
        
        # Normalize spatial coordinates
        normalized[:, 0] = events[:, 0] / self.resolution[0]  # x
        normalized[:, 1] = events[:, 1] / self.resolution[1]  # y
        
        # Normalize timestamps to [0, time_window]
        if events.size(0) > 1:
            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            if t_max > t_min:
                normalized[:, 2] = (events[:, 2] - t_min) / (t_max - t_min) * self.time_window
                
        return normalized
        
    def _augment_events(self, events: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to events.
        
        Args:
            events: Event tensor [N, 4]
            
        Returns:
            Augmented event tensor
        """
        if events.numel() == 0:
            return events
            
        augmented = events.clone()
        
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            augmented[:, 0] = self.resolution[0] - augmented[:, 0] - 1
            
        # Random temporal jitter (±5ms)
        if torch.rand(1).item() < 0.3:
            jitter = torch.randn(events.size(0)) * 5.0  # ±5ms
            augmented[:, 2] = torch.clamp(
                augmented[:, 2] + jitter, 
                min=0, 
                max=self.time_window
            )
            
        # Random polarity flip
        if torch.rand(1).item() < 0.1:
            augmented[:, 3] = -augmented[:, 3]
            
        return augmented
        
    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        encoder_type: str = "temporal",
        resolution: Tuple[int, int] = (640, 480),
        **kwargs
    ) -> 'CustomEventDataset':
        """
        Load dataset directly from file.
        
        Args:
            file_path: Path to data file
            encoder_type: Event encoder type
            resolution: Sensor resolution
            **kwargs: Additional dataset arguments
            
        Returns:
            CustomEventDataset instance
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.h5':
            import h5py
            with h5py.File(file_path, 'r') as f:
                events_data = f['events'][:]
                labels_data = f.get('labels', None)
                if labels_data is not None:
                    labels_data = labels_data[:]
                    
        elif file_path.suffix.lower() in ['.npz', '.npy']:
            data = np.load(file_path, allow_pickle=True)
            events_data = data['events'] if 'events' in data else data
            labels_data = data.get('labels', None)
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        return cls(
            events_data=events_data,
            labels_data=labels_data,
            encoder_type=encoder_type,
            resolution=resolution,
            **kwargs
        )
        
    def get_sample_info(self, idx: int = 0) -> dict:
        """
        Get information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample statistics
        """
        events = self.events_data[idx]
        if not isinstance(events, np.ndarray):
            events = np.array(events)
            
        info = {
            'num_events': len(events),
            'time_span': events[:, 2].max() - events[:, 2].min() if len(events) > 0 else 0,
            'spatial_range': {
                'x_min': events[:, 0].min() if len(events) > 0 else 0,
                'x_max': events[:, 0].max() if len(events) > 0 else 0,
                'y_min': events[:, 1].min() if len(events) > 0 else 0,
                'y_max': events[:, 1].max() if len(events) > 0 else 0,
            },
            'polarity_ratio': {
                'positive': np.sum(events[:, 3] > 0) / len(events) if len(events) > 0 else 0,
                'negative': np.sum(events[:, 3] <= 0) / len(events) if len(events) > 0 else 0,
            }
        }
        
        return info