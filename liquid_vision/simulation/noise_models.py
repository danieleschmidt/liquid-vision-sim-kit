"""
Noise models for realistic event camera simulation.
Implements various noise sources found in real neuromorphic sensors.
"""

import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .event_simulator import EventData


class NoiseModel(ABC):
    """Base class for event camera noise models."""
    
    @abstractmethod
    def apply(self, events: "EventData", timestamp: float) -> "EventData":
        """
        Apply noise to event data.
        
        Args:
            events: Input events
            timestamp: Current timestamp
            
        Returns:
            Events with noise applied
        """
        pass


class ShotNoise(NoiseModel):
    """
    Shot noise model that generates random spurious events.
    Simulates thermal noise and dark current in the sensor.
    """
    
    def __init__(
        self,
        rate: float = 0.01,  # Events per pixel per millisecond
        sensor_size: tuple = (640, 480),
    ):
        self.rate = rate
        self.sensor_size = sensor_size
        
    def apply(self, events: "EventData", timestamp: float) -> "EventData":
        """Add shot noise events."""
        # Calculate expected number of noise events
        total_pixels = self.sensor_size[0] * self.sensor_size[1]
        expected_noise = self.rate * total_pixels
        
        # Generate Poisson-distributed number of noise events
        num_noise = np.random.poisson(expected_noise)
        
        if num_noise == 0:
            return events
            
        # Generate random noise events
        noise_x = np.random.randint(0, self.sensor_size[0], num_noise)
        noise_y = np.random.randint(0, self.sensor_size[1], num_noise)
        noise_t = np.full(num_noise, timestamp, dtype=np.float32)
        noise_p = np.random.choice([-1, 1], num_noise)
        
        # Combine with original events
        combined_x = np.concatenate([events.x, noise_x])
        combined_y = np.concatenate([events.y, noise_y])
        combined_t = np.concatenate([events.t, noise_t])
        combined_p = np.concatenate([events.p, noise_p])
        
        # Import EventData dynamically to avoid circular imports
        from .event_simulator import EventData
        return EventData(
            x=combined_x,
            y=combined_y,
            t=combined_t,
            p=combined_p
        )


class ThermalNoise(NoiseModel):
    """
    Thermal noise model with temperature-dependent characteristics.
    Higher temperatures increase noise rates exponentially.
    """
    
    def __init__(
        self,
        base_rate: float = 0.005,  # Base rate at reference temperature
        temperature: float = 25.0,  # Celsius
        reference_temp: float = 25.0,
        temperature_coefficient: float = 0.1,  # Rate doubling per 10°C
        sensor_size: tuple = (640, 480),
    ):
        self.base_rate = base_rate
        self.temperature = temperature
        self.reference_temp = reference_temp
        self.temperature_coefficient = temperature_coefficient
        self.sensor_size = sensor_size
        
    def apply(self, events: "EventData", timestamp: float) -> "EventData":
        """Add temperature-dependent thermal noise."""
        # Calculate temperature-adjusted rate
        temp_diff = self.temperature - self.reference_temp
        rate_multiplier = np.exp(self.temperature_coefficient * temp_diff / 10.0)
        current_rate = self.base_rate * rate_multiplier
        
        # Generate noise events
        total_pixels = self.sensor_size[0] * self.sensor_size[1]
        expected_noise = current_rate * total_pixels
        num_noise = np.random.poisson(expected_noise)
        
        if num_noise == 0:
            return events
            
        noise_x = np.random.randint(0, self.sensor_size[0], num_noise)
        noise_y = np.random.randint(0, self.sensor_size[1], num_noise)
        noise_t = np.full(num_noise, timestamp, dtype=np.float32)
        noise_p = np.random.choice([-1, 1], num_noise)
        
        # Combine events
        combined_x = np.concatenate([events.x, noise_x])
        combined_y = np.concatenate([events.y, noise_y])
        combined_t = np.concatenate([events.t, noise_t])
        combined_p = np.concatenate([events.p, noise_p])
        
        # Import EventData dynamically to avoid circular imports
        from .event_simulator import EventData
        return EventData(
            x=combined_x,
            y=combined_y,
            t=combined_t,
            p=combined_p
        )


class RefractoryNoise(NoiseModel):
    """
    Refractory period violation noise.
    Sometimes pixels fire again before refractory period ends.
    """
    
    def __init__(
        self,
        violation_rate: float = 0.01,  # Probability of refractory violation
        refractory_period: float = 1.0,  # milliseconds
    ):
        self.violation_rate = violation_rate
        self.refractory_period = refractory_period
        
    def apply(self, events: "EventData", timestamp: float) -> "EventData":
        """Add refractory period violations."""
        if len(events) == 0:
            return events
            
        # Identify events that might cause violations
        violation_mask = np.random.random(len(events)) < self.violation_rate
        
        if not violation_mask.any():
            return events
            
        # Create violation events slightly after original events
        violation_indices = np.where(violation_mask)[0]
        num_violations = len(violation_indices)
        
        # Violation events occur shortly after original events
        violation_delay = np.random.exponential(
            self.refractory_period / 3, 
            num_violations
        )
        
        violation_x = events.x[violation_indices]
        violation_y = events.y[violation_indices]
        violation_t = events.t[violation_indices] + violation_delay
        violation_p = events.p[violation_indices]  # Same polarity
        
        # Combine events
        combined_x = np.concatenate([events.x, violation_x])
        combined_y = np.concatenate([events.y, violation_y])
        combined_t = np.concatenate([events.t, violation_t])
        combined_p = np.concatenate([events.p, violation_p])
        
        # Import EventData dynamically to avoid circular imports
        from .event_simulator import EventData
        return EventData(
            x=combined_x,
            y=combined_y,
            t=combined_t,
            p=combined_p
        )


class TimestampJitter(NoiseModel):
    """
    Timestamp jitter noise model.
    Adds random timing errors to event timestamps.
    """
    
    def __init__(
        self,
        jitter_std: float = 0.1,  # Standard deviation in milliseconds
        max_jitter: float = 1.0,  # Maximum jitter (clips outliers)
    ):
        self.jitter_std = jitter_std
        self.max_jitter = max_jitter
        
    def apply(self, events: "EventData", timestamp: float) -> "EventData":
        """Add timestamp jitter."""
        if len(events) == 0:
            return events
            
        # Generate jitter
        jitter = np.random.normal(0, self.jitter_std, len(events))
        jitter = np.clip(jitter, -self.max_jitter, self.max_jitter)
        
        # Apply jitter to timestamps
        jittered_t = events.t + jitter
        
        # Ensure timestamps remain positive
        jittered_t = np.maximum(jittered_t, 0)
        
        return EventData(
            x=events.x.copy(),
            y=events.y.copy(),
            t=jittered_t,
            p=events.p.copy()
        )


class PixelMismatch(NoiseModel):
    """
    Pixel mismatch noise model.
    Simulates variations in pixel sensitivity and threshold.
    """
    
    def __init__(
        self,
        mismatch_std: float = 0.05,  # Standard deviation of threshold mismatch
        sensor_size: tuple = (640, 480),
        seed: Optional[int] = None,
    ):
        self.mismatch_std = mismatch_std
        self.sensor_size = sensor_size
        
        # Generate fixed mismatch pattern
        if seed is not None:
            np.random.seed(seed)
            
        self.mismatch_map = np.random.normal(
            1.0, 
            mismatch_std, 
            (sensor_size[1], sensor_size[0])
        )
        
        # Restore random state
        if seed is not None:
            np.random.seed(None)
            
    def apply(self, events: "EventData", timestamp: float) -> "EventData":
        """Apply pixel mismatch effects."""
        if len(events) == 0:
            return events
            
        # Get mismatch factors for event pixels
        valid_mask = (
            (events.x >= 0) & (events.x < self.sensor_size[0]) &
            (events.y >= 0) & (events.y < self.sensor_size[1])
        )
        
        if not valid_mask.any():
            return events
            
        # Apply mismatch - some events get dropped, some get duplicated
        mismatch_factors = self.mismatch_map[events.y[valid_mask], events.x[valid_mask]]
        
        # Events with mismatch < 0.5 get dropped
        # Events with mismatch > 1.5 might get duplicated
        drop_mask = mismatch_factors < 0.5
        duplicate_mask = mismatch_factors > 1.5
        
        # Create filtered event indices
        keep_indices = np.where(valid_mask)[0]
        keep_indices = keep_indices[~drop_mask]
        
        # Add duplicates
        if duplicate_mask.any():
            duplicate_indices = np.where(valid_mask)[0]
            duplicate_indices = duplicate_indices[duplicate_mask]
            
            # Add small time offset to duplicates
            duplicate_offset = np.random.exponential(0.1, len(duplicate_indices))
            
            # Combine indices
            all_indices = np.concatenate([keep_indices, duplicate_indices])
            
            # Create event arrays
            filtered_x = np.concatenate([
                events.x[keep_indices],
                events.x[duplicate_indices]
            ])
            filtered_y = np.concatenate([
                events.y[keep_indices],
                events.y[duplicate_indices]
            ])
            filtered_t = np.concatenate([
                events.t[keep_indices],
                events.t[duplicate_indices] + duplicate_offset
            ])
            filtered_p = np.concatenate([
                events.p[keep_indices],
                events.p[duplicate_indices]
            ])
        else:
            filtered_x = events.x[keep_indices]
            filtered_y = events.y[keep_indices]
            filtered_t = events.t[keep_indices]
            filtered_p = events.p[keep_indices]
            
        return EventData(
            x=filtered_x,
            y=filtered_y,
            t=filtered_t,
            p=filtered_p
        )


class CompositeNoise(NoiseModel):
    """
    Composite noise model that combines multiple noise sources.
    """
    
    def __init__(self, noise_models: list):
        self.noise_models = noise_models
        
    def apply(self, events: "EventData", timestamp: float) -> "EventData":
        """Apply all noise models sequentially."""
        current_events = events
        
        for noise_model in self.noise_models:
            current_events = noise_model.apply(current_events, timestamp)
            
        return current_events


def create_realistic_noise(
    sensor_size: tuple = (640, 480),
    noise_level: str = "medium"
) -> CompositeNoise:
    """
    Create a realistic composite noise model.
    
    Args:
        sensor_size: Sensor dimensions
        noise_level: Noise level ("low", "medium", "high")
        
    Returns:
        Configured composite noise model
    """
    if noise_level == "low":
        noise_models = [
            ShotNoise(rate=0.001, sensor_size=sensor_size),
            TimestampJitter(jitter_std=0.05),
        ]
    elif noise_level == "medium":
        noise_models = [
            ShotNoise(rate=0.005, sensor_size=sensor_size),
            ThermalNoise(base_rate=0.003, sensor_size=sensor_size),
            TimestampJitter(jitter_std=0.1),
            RefractoryNoise(violation_rate=0.01),
        ]
    elif noise_level == "high":
        noise_models = [
            ShotNoise(rate=0.01, sensor_size=sensor_size),
            ThermalNoise(base_rate=0.008, temperature=40.0, sensor_size=sensor_size),
            TimestampJitter(jitter_std=0.2),
            RefractoryNoise(violation_rate=0.02),
            PixelMismatch(mismatch_std=0.1, sensor_size=sensor_size),
        ]
    else:
        raise ValueError(f"Unknown noise level: {noise_level}")
        
    return CompositeNoise(noise_models)