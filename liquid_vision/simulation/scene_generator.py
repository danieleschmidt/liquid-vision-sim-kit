"""
Synthetic scene generation for event-based camera simulation.
Creates realistic scenes with moving objects for training liquid neural networks.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math


class MotionPattern(Enum):
    """Supported motion patterns for objects."""
    LINEAR = "linear"
    CIRCULAR = "circular"
    RANDOM_WALK = "random_walk"
    SINUSOIDAL = "sinusoidal"
    SPIRAL = "spiral"
    OSCILLATORY = "oscillatory"


class ObjectType(Enum):
    """Supported object types."""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    LINE = "line"
    POLYGON = "polygon"
    TEXTURE_PATCH = "texture_patch"


@dataclass
class SceneObject:
    """Represents a single object in the scene."""
    object_type: ObjectType
    position: Tuple[float, float]  # (x, y)
    size: Union[float, Tuple[float, float]]  # radius or (width, height)
    velocity: Tuple[float, float]  # (vx, vy) pixels per frame
    color: float  # Grayscale intensity [0, 1]
    motion_pattern: MotionPattern = MotionPattern.LINEAR
    motion_params: Optional[Dict] = None  # Pattern-specific parameters
    
    def __post_init__(self):
        if self.motion_params is None:
            self.motion_params = {}


class SceneGenerator:
    """
    Generates synthetic scenes with moving objects for event camera simulation.
    Supports various motion patterns and object types for diverse training data.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        background_color: float = 0.5,
        frame_rate: float = 30.0,
    ):
        self.resolution = resolution
        self.background_color = background_color
        self.frame_rate = frame_rate
        self.objects: List[SceneObject] = []
        
    def add_object(
        self,
        object_type: ObjectType,
        position: Tuple[float, float],
        size: Union[float, Tuple[float, float]],
        velocity: Tuple[float, float],
        color: float,
        motion_pattern: MotionPattern = MotionPattern.LINEAR,
        **motion_params
    ) -> None:
        """Add an object to the scene."""
        obj = SceneObject(
            object_type=object_type,
            position=position,
            size=size,
            velocity=velocity,
            color=color,
            motion_pattern=motion_pattern,
            motion_params=motion_params
        )
        self.objects.append(obj)
        
    def clear_objects(self) -> None:
        """Remove all objects from the scene."""
        self.objects.clear()
        
    def generate_frame(self, frame_number: int) -> np.ndarray:
        """
        Generate a single frame of the scene.
        
        Args:
            frame_number: Frame index
            
        Returns:
            Generated frame as grayscale image [height, width]
        """
        # Create background
        frame = np.full(
            (self.resolution[1], self.resolution[0]), 
            self.background_color, 
            dtype=np.float32
        )
        
        # Render each object
        for obj in self.objects:
            position = self._update_object_position(obj, frame_number)
            self._render_object(frame, obj, position)
            
        return frame
        
    def generate_sequence(
        self,
        num_frames: int,
        return_timestamps: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a sequence of frames.
        
        Args:
            num_frames: Number of frames to generate
            return_timestamps: Whether to return timestamps
            
        Returns:
            Frames array [num_frames, height, width] and optionally timestamps
        """
        frames = []
        
        for frame_idx in range(num_frames):
            frame = self.generate_frame(frame_idx)
            frames.append(frame)
            
        frames_array = np.array(frames)
        
        if return_timestamps:
            timestamps = np.arange(num_frames) * (1000.0 / self.frame_rate)
            return frames_array, timestamps
        else:
            return frames_array
            
    def _update_object_position(
        self, 
        obj: SceneObject, 
        frame_number: int
    ) -> Tuple[float, float]:
        """Update object position based on motion pattern."""
        t = frame_number / self.frame_rate  # Time in seconds
        
        if obj.motion_pattern == MotionPattern.LINEAR:
            x = obj.position[0] + obj.velocity[0] * frame_number
            y = obj.position[1] + obj.velocity[1] * frame_number
            
        elif obj.motion_pattern == MotionPattern.CIRCULAR:
            radius = obj.motion_params.get('radius', 100)
            angular_freq = obj.motion_params.get('angular_frequency', 1.0)
            center_x = obj.motion_params.get('center_x', self.resolution[0] // 2)
            center_y = obj.motion_params.get('center_y', self.resolution[1] // 2)
            
            angle = angular_freq * t
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
        elif obj.motion_pattern == MotionPattern.SINUSOIDAL:
            amplitude_x = obj.motion_params.get('amplitude_x', 50)
            amplitude_y = obj.motion_params.get('amplitude_y', 30)
            freq_x = obj.motion_params.get('frequency_x', 0.5)
            freq_y = obj.motion_params.get('frequency_y', 0.7)
            
            x = obj.position[0] + amplitude_x * np.sin(2 * np.pi * freq_x * t)
            y = obj.position[1] + amplitude_y * np.sin(2 * np.pi * freq_y * t)
            
        elif obj.motion_pattern == MotionPattern.SPIRAL:
            radius_rate = obj.motion_params.get('radius_rate', 2.0)
            angular_freq = obj.motion_params.get('angular_frequency', 2.0)
            center_x = obj.motion_params.get('center_x', self.resolution[0] // 2)
            center_y = obj.motion_params.get('center_y', self.resolution[1] // 2)
            
            radius = radius_rate * t
            angle = angular_freq * t
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
        elif obj.motion_pattern == MotionPattern.OSCILLATORY:
            amplitude = obj.motion_params.get('amplitude', 100)
            frequency = obj.motion_params.get('frequency', 1.0)
            axis = obj.motion_params.get('axis', 'x')  # 'x' or 'y'
            
            if axis == 'x':
                x = obj.position[0] + amplitude * np.sin(2 * np.pi * frequency * t)
                y = obj.position[1]
            else:
                x = obj.position[0]
                y = obj.position[1] + amplitude * np.sin(2 * np.pi * frequency * t)
                
        elif obj.motion_pattern == MotionPattern.RANDOM_WALK:
            # For random walk, we need to maintain state
            if not hasattr(obj, '_random_position'):
                obj._random_position = list(obj.position)
                
            step_size = obj.motion_params.get('step_size', 2.0)
            
            # Random step
            angle = np.random.uniform(0, 2 * np.pi)
            dx = step_size * np.cos(angle)
            dy = step_size * np.sin(angle)
            
            obj._random_position[0] += dx
            obj._random_position[1] += dy
            
            # Boundary conditions
            obj._random_position[0] = np.clip(
                obj._random_position[0], 0, self.resolution[0] - 1
            )
            obj._random_position[1] = np.clip(
                obj._random_position[1], 0, self.resolution[1] - 1
            )
            
            x, y = obj._random_position
            
        else:
            x, y = obj.position
            
        # Apply boundary wrapping or clipping
        boundary_mode = obj.motion_params.get('boundary_mode', 'wrap')
        
        if boundary_mode == 'wrap':
            x = x % self.resolution[0]
            y = y % self.resolution[1]
        elif boundary_mode == 'clip':
            x = np.clip(x, 0, self.resolution[0] - 1)
            y = np.clip(y, 0, self.resolution[1] - 1)
        elif boundary_mode == 'bounce':
            if not hasattr(obj, '_velocity_sign'):
                obj._velocity_sign = [1, 1]
                
            if x < 0 or x >= self.resolution[0]:
                obj._velocity_sign[0] *= -1
                x = np.clip(x, 0, self.resolution[0] - 1)
            if y < 0 or y >= self.resolution[1]:
                obj._velocity_sign[1] *= -1
                y = np.clip(y, 0, self.resolution[1] - 1)
                
        return (x, y)
        
    def _render_object(
        self, 
        frame: np.ndarray, 
        obj: SceneObject, 
        position: Tuple[float, float]
    ) -> None:
        """Render an object onto the frame."""
        x, y = position
        x_int, y_int = int(round(x)), int(round(y))
        
        if obj.object_type == ObjectType.CIRCLE:
            radius = int(obj.size) if isinstance(obj.size, (int, float)) else int(obj.size[0])
            cv2.circle(frame, (x_int, y_int), radius, obj.color, -1)
            
        elif obj.object_type == ObjectType.RECTANGLE:
            if isinstance(obj.size, (int, float)):
                width = height = int(obj.size)
            else:
                width, height = int(obj.size[0]), int(obj.size[1])
                
            top_left = (x_int - width // 2, y_int - height // 2)
            bottom_right = (x_int + width // 2, y_int + height // 2)
            cv2.rectangle(frame, top_left, bottom_right, obj.color, -1)
            
        elif obj.object_type == ObjectType.TRIANGLE:
            size = int(obj.size) if isinstance(obj.size, (int, float)) else int(obj.size[0])
            
            # Equilateral triangle
            height = int(size * np.sqrt(3) / 2)
            pts = np.array([
                [x_int, y_int - height // 2],
                [x_int - size // 2, y_int + height // 2],
                [x_int + size // 2, y_int + height // 2]
            ], np.int32)
            
            cv2.fillPoly(frame, [pts], obj.color)
            
        elif obj.object_type == ObjectType.LINE:
            length = int(obj.size) if isinstance(obj.size, (int, float)) else int(obj.size[0])
            angle = obj.motion_params.get('angle', 0)  # Line angle in radians
            
            x1 = int(x - length // 2 * np.cos(angle))
            y1 = int(y - length // 2 * np.sin(angle))
            x2 = int(x + length // 2 * np.cos(angle))
            y2 = int(y + length // 2 * np.sin(angle))
            
            thickness = obj.motion_params.get('thickness', 2)
            cv2.line(frame, (x1, y1), (x2, y2), obj.color, thickness)
            
        elif obj.object_type == ObjectType.POLYGON:
            vertices = obj.motion_params.get('vertices', 6)
            radius = int(obj.size) if isinstance(obj.size, (int, float)) else int(obj.size[0])
            
            angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
            pts = []
            for angle in angles:
                px = int(x + radius * np.cos(angle))
                py = int(y + radius * np.sin(angle))
                pts.append([px, py])
                
            pts = np.array(pts, np.int32)
            cv2.fillPoly(frame, [pts], obj.color)
            
        elif obj.object_type == ObjectType.TEXTURE_PATCH:
            # Simple texture patch (checkerboard pattern)
            if isinstance(obj.size, (int, float)):
                width = height = int(obj.size)
            else:
                width, height = int(obj.size[0]), int(obj.size[1])
                
            checker_size = obj.motion_params.get('checker_size', 4)
            
            for i in range(height):
                for j in range(width):
                    px = x_int - width // 2 + j
                    py = y_int - height // 2 + i
                    
                    if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                        checker_x = j // checker_size
                        checker_y = i // checker_size
                        
                        if (checker_x + checker_y) % 2 == 0:
                            frame[py, px] = obj.color
                        else:
                            frame[py, px] = 1.0 - obj.color
    
    @staticmethod
    def create_scene(
        num_objects: int = 5,
        resolution: Tuple[int, int] = (640, 480),
        motion_type: str = "mixed",
        velocity_range: Tuple[float, float] = (0.5, 5.0),
        size_range: Tuple[float, float] = (10, 50),
        **kwargs
    ) -> 'SceneGenerator':
        """
        Factory method to create a scene with random objects.
        
        Args:
            num_objects: Number of objects to create
            resolution: Scene resolution
            motion_type: Motion pattern ("linear", "circular", "mixed")
            velocity_range: Range of velocities
            size_range: Range of object sizes
            
        Returns:
            Configured SceneGenerator instance
        """
        scene = SceneGenerator(resolution=resolution, **kwargs)
        
        object_types = list(ObjectType)
        motion_patterns = list(MotionPattern)
        
        for _ in range(num_objects):
            # Random object type
            obj_type = np.random.choice(object_types)
            
            # Random position
            position = (
                np.random.uniform(0, resolution[0]),
                np.random.uniform(0, resolution[1])
            )
            
            # Random size
            size = np.random.uniform(size_range[0], size_range[1])
            
            # Random velocity
            speed = np.random.uniform(velocity_range[0], velocity_range[1])
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = (speed * np.cos(angle), speed * np.sin(angle))
            
            # Random color
            color = np.random.uniform(0.1, 0.9)
            
            # Motion pattern
            if motion_type == "mixed":
                motion_pattern = np.random.choice(motion_patterns)
            elif motion_type == "linear":
                motion_pattern = MotionPattern.LINEAR
            elif motion_type == "circular":
                motion_pattern = MotionPattern.CIRCULAR
            else:
                motion_pattern = MotionPattern(motion_type)
                
            # Motion parameters
            motion_params = {}
            if motion_pattern == MotionPattern.CIRCULAR:
                motion_params = {
                    'radius': np.random.uniform(50, 150),
                    'angular_frequency': np.random.uniform(0.5, 3.0),
                    'center_x': np.random.uniform(100, resolution[0] - 100),
                    'center_y': np.random.uniform(100, resolution[1] - 100),
                }
            elif motion_pattern == MotionPattern.SINUSOIDAL:
                motion_params = {
                    'amplitude_x': np.random.uniform(20, 100),
                    'amplitude_y': np.random.uniform(20, 100),
                    'frequency_x': np.random.uniform(0.1, 2.0),
                    'frequency_y': np.random.uniform(0.1, 2.0),
                }
                
            scene.add_object(
                object_type=obj_type,
                position=position,
                size=size,
                velocity=velocity,
                color=color,
                motion_pattern=motion_pattern,
                **motion_params
            )
            
        return scene


class TexturedSceneGenerator(SceneGenerator):
    """
    Enhanced scene generator with textured backgrounds and complex patterns.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        background_texture: Optional[str] = None,
        **kwargs
    ):
        super().__init__(resolution, **kwargs)
        self.background_texture = background_texture
        
    def generate_frame(self, frame_number: int) -> np.ndarray:
        """Generate frame with textured background."""
        # Create textured background
        if self.background_texture == "checkerboard":
            frame = self._create_checkerboard_background()
        elif self.background_texture == "gradient":
            frame = self._create_gradient_background()
        elif self.background_texture == "noise":
            frame = self._create_noise_background()
        else:
            frame = np.full(
                (self.resolution[1], self.resolution[0]), 
                self.background_color, 
                dtype=np.float32
            )
            
        # Render objects
        for obj in self.objects:
            position = self._update_object_position(obj, frame_number)
            self._render_object(frame, obj, position)
            
        return frame
        
    def _create_checkerboard_background(self) -> np.ndarray:
        """Create checkerboard pattern background."""
        frame = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        checker_size = 32
        
        for i in range(0, self.resolution[1], checker_size):
            for j in range(0, self.resolution[0], checker_size):
                if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                    frame[i:i+checker_size, j:j+checker_size] = 0.3
                else:
                    frame[i:i+checker_size, j:j+checker_size] = 0.7
                    
        return frame
        
    def _create_gradient_background(self) -> np.ndarray:
        """Create gradient background."""
        x = np.linspace(0, 1, self.resolution[0])
        y = np.linspace(0, 1, self.resolution[1])
        X, Y = np.meshgrid(x, y)
        
        frame = 0.3 + 0.4 * (X + Y) / 2
        return frame.astype(np.float32)
        
    def _create_noise_background(self) -> np.ndarray:
        """Create noisy background."""
        frame = np.random.normal(
            self.background_color, 
            0.1, 
            (self.resolution[1], self.resolution[0])
        )
        return np.clip(frame, 0, 1).astype(np.float32)