"""
Configuration validation utilities.
"""

from typing import Any, List, Dict, Optional, Union
from dataclasses import fields
import re
import logging


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Configuration validation error."""
    pass


def validate_config(config: Any) -> None:
    """
    Validate configuration object.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValidationError: If configuration is invalid
    """
    config_type = type(config).__name__
    
    # Get validation rules for this config type
    validator_map = {
        'TrainingConfig': validate_training_config,
        'DeploymentConfig': validate_deployment_config,
        'SimulationConfig': validate_simulation_config,
    }
    
    validator = validator_map.get(config_type)
    if validator:
        validator(config)
    else:
        logger.warning(f"No specific validator for {config_type}")


def validate_training_config(config) -> None:
    """Validate training configuration."""
    errors = []
    
    # Architecture validation
    valid_architectures = ["liquid_net", "rnn", "lstm", "gru"]
    if config.architecture not in valid_architectures:
        errors.append(f"Invalid architecture: {config.architecture}. Must be one of {valid_architectures}")
    
    # Dimension validation
    if config.input_dim <= 0:
        errors.append("input_dim must be positive")
    if config.output_dim <= 0:
        errors.append("output_dim must be positive")
    
    # Hidden units validation
    if not config.hidden_units or not all(h > 0 for h in config.hidden_units):
        errors.append("hidden_units must be a list of positive integers")
    
    # Training parameter validation
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    if not 0 < config.learning_rate < 1:
        errors.append("learning_rate must be between 0 and 1")
    
    # Encoder validation
    valid_encoders = ["temporal", "spatial", "timeslice", "adaptive", "voxel", "sae", "event_image"]
    if config.encoder_type not in valid_encoders:
        errors.append(f"Invalid encoder_type: {config.encoder_type}. Must be one of {valid_encoders}")
    
    # Time parameters
    if config.time_window <= 0:
        errors.append("time_window must be positive")
    if config.liquid_time_constant <= 0:
        errors.append("liquid_time_constant must be positive")
    
    # Resolution validation
    if len(config.resolution) != 2 or any(r <= 0 for r in config.resolution):
        errors.append("resolution must be [width, height] with positive values")
    
    # Dropout validation
    if not 0 <= config.dropout_rate < 1:
        errors.append("dropout_rate must be between 0 and 1")
    if not 0 <= config.liquid_dropout < 1:
        errors.append("liquid_dropout must be between 0 and 1")
    
    # Device validation
    valid_devices = ["auto", "cpu", "cuda", "mps"]
    if config.device not in valid_devices and not re.match(r"cuda:\d+", config.device):
        errors.append(f"Invalid device: {config.device}")
    
    # Optimizer validation
    valid_optimizers = ["adam", "sgd", "adamw", "rmsprop"]
    if config.optimizer not in valid_optimizers:
        errors.append(f"Invalid optimizer: {config.optimizer}. Must be one of {valid_optimizers}")
    
    # Scheduler validation
    valid_schedulers = ["none", "cosine", "step", "exponential", "plateau"]
    if config.scheduler not in valid_schedulers:
        errors.append(f"Invalid scheduler: {config.scheduler}. Must be one of {valid_schedulers}")
    
    if errors:
        raise ValidationError(f"Training configuration errors: {'; '.join(errors)}")


def validate_deployment_config(config) -> None:
    """Validate deployment configuration."""
    errors = []
    
    # Target validation
    valid_targets = ["esp32", "esp32s3", "cortex_m4", "cortex_m7", "linux_arm", "raspberry_pi"]
    if config.target not in valid_targets:
        errors.append(f"Invalid target: {config.target}. Must be one of {valid_targets}")
    
    # Quantization validation
    valid_quantizations = ["none", "int8", "int16", "float16"]
    if config.quantization not in valid_quantizations:
        errors.append(f"Invalid quantization: {config.quantization}. Must be one of {valid_quantizations}")
    
    # Memory validation
    if config.max_memory_kb <= 0:
        errors.append("max_memory_kb must be positive")
    if config.max_memory_kb < 32:
        errors.append("max_memory_kb should be at least 32KB for practical deployment")
    
    # Performance validation
    if config.max_inference_time_ms <= 0:
        errors.append("max_inference_time_ms must be positive")
    
    # Sensor validation
    valid_sensors = ["dvs128", "dvs240", "dvs640", "davis346", "prophesee", "generic_spi", "generic_i2c"]
    if config.sensor_type not in valid_sensors:
        errors.append(f"Invalid sensor_type: {config.sensor_type}. Must be one of {valid_sensors}")
    
    # Frequency validation
    if config.spi_frequency <= 0:
        errors.append("spi_frequency must be positive")
    if config.i2c_frequency <= 0:
        errors.append("i2c_frequency must be positive")
    
    # Pruning validation
    if not 0 <= config.prune_weights < 1:
        errors.append("prune_weights must be between 0 and 1")
    
    # Build system validation
    valid_build_systems = ["cmake", "platformio", "makefile"]
    if config.build_system not in valid_build_systems:
        errors.append(f"Invalid build_system: {config.build_system}. Must be one of {valid_build_systems}")
    
    if errors:
        raise ValidationError(f"Deployment configuration errors: {'; '.join(errors)}")


def validate_simulation_config(config) -> None:
    """Validate simulation configuration."""
    errors = []
    
    # Scene validation
    if config.num_objects <= 0:
        errors.append("num_objects must be positive")
    
    valid_motions = ["linear", "circular", "random", "oscillatory"]
    if config.motion_type not in valid_motions:
        errors.append(f"Invalid motion_type: {config.motion_type}. Must be one of {valid_motions}")
    
    # Velocity validation
    if len(config.velocity_range) != 2:
        errors.append("velocity_range must have exactly 2 values [min, max]")
    elif config.velocity_range[0] >= config.velocity_range[1]:
        errors.append("velocity_range min must be less than max")
    elif any(v < 0 for v in config.velocity_range):
        errors.append("velocity_range values must be non-negative")
    
    # Resolution validation
    if len(config.resolution) != 2 or any(r <= 0 for r in config.resolution):
        errors.append("resolution must be [width, height] with positive values")
    
    # Camera parameters
    if config.fps <= 0:
        errors.append("fps must be positive")
    if config.fps > 1000:
        errors.append("fps should not exceed 1000 for practical simulation")
    
    # DVS parameters
    if not 0 < config.contrast_threshold < 1:
        errors.append("contrast_threshold must be between 0 and 1")
    if config.refractory_period < 0:
        errors.append("refractory_period must be non-negative")
    if not 0 <= config.noise_rate < 1:
        errors.append("noise_rate must be between 0 and 1")
    
    # Output validation
    if config.num_frames <= 0:
        errors.append("num_frames must be positive")
    
    valid_formats = ["h5", "npz", "mat", "csv"]
    if config.output_format not in valid_formats:
        errors.append(f"Invalid output_format: {config.output_format}. Must be one of {valid_formats}")
    
    if errors:
        raise ValidationError(f"Simulation configuration errors: {'; '.join(errors)}")


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate that value is within range."""
    if not min_val <= value <= max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_positive(value: Union[int, float], name: str) -> None:
    """Validate that value is positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: Union[int, float], name: str) -> None:
    """Validate that value is non-negative."""
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_list_elements(values: List[Any], validator_func, name: str) -> None:
    """Validate all elements in a list."""
    for i, value in enumerate(values):
        try:
            validator_func(value, f"{name}[{i}]")
        except ValidationError:
            raise


def validate_choice(value: Any, choices: List[Any], name: str) -> None:
    """Validate that value is one of the allowed choices."""
    if value not in choices:
        raise ValidationError(f"{name} must be one of {choices}, got {value}")