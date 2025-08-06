"""
Input validation and sanitization utilities for security and robustness.
"""

import re
import os
import numpy as np
import torch
from pathlib import Path
from typing import Any, List, Dict, Union, Optional, Tuple
import logging


logger = logging.getLogger('liquid_vision.validation')


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_file_path(
        path: Union[str, Path], 
        must_exist: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        max_size_mb: Optional[float] = None
    ) -> Path:
        """
        Validate and sanitize file paths.
        
        Args:
            path: File path to validate
            must_exist: Whether file must exist
            allowed_extensions: List of allowed file extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(path, (str, Path)):
            raise ValidationError(f"Path must be string or Path object, got {type(path)}")
        
        path_obj = Path(path)
        
        # Check for path traversal attacks
        try:
            path_obj.resolve(strict=must_exist)
        except (OSError, FileNotFoundError) as e:
            if must_exist:
                raise ValidationError(f"File not found or inaccessible: {path}")
            # For non-existent files, check parent directory
            try:
                path_obj.parent.resolve(strict=True)
            except OSError:
                raise ValidationError(f"Parent directory not accessible: {path_obj.parent}")
        
        # Prevent path traversal
        if '..' in str(path_obj) or str(path_obj).startswith('/'):
            # Only allow absolute paths in specific directories
            allowed_roots = ['/tmp', '/var/tmp', str(Path.home()), str(Path.cwd())]
            if not any(str(path_obj).startswith(root) for root in allowed_roots):
                logger.warning(f"Potentially unsafe path blocked: {path}")
                raise ValidationError(f"Unsafe path detected: {path}")
        
        # Check file extension
        if allowed_extensions and path_obj.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"File extension '{path_obj.suffix}' not allowed. "
                f"Allowed: {allowed_extensions}"
            )
        
        # Check file size
        if must_exist and max_size_mb and path_obj.exists():
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValidationError(f"File too large: {size_mb:.1f}MB > {max_size_mb}MB")
        
        logger.debug(f"File path validated: {path_obj}")
        return path_obj
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> torch.Tensor:
        """
        Validate tensor properties.
        
        Args:
            tensor: Input tensor to validate
            expected_shape: Expected tensor shape (None dimensions ignored)
            expected_dtype: Expected data type
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_nan: Whether NaN values are allowed
            allow_inf: Whether infinite values are allowed
            
        Returns:
            Validated tensor
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # Check shape
        if expected_shape is not None:
            if len(tensor.shape) != len(expected_shape):
                raise ValidationError(
                    f"Shape mismatch: expected {len(expected_shape)} dimensions, "
                    f"got {len(tensor.shape)}"
                )
            
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise ValidationError(
                        f"Shape mismatch at dimension {i}: expected {expected}, got {actual}"
                    )
        
        # Check dtype
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            raise ValidationError(f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
        
        # Check for NaN values
        if not allow_nan and torch.isnan(tensor).any():
            raise ValidationError("Tensor contains NaN values")
        
        # Check for infinite values
        if not allow_inf and torch.isinf(tensor).any():
            raise ValidationError("Tensor contains infinite values")
        
        # Check value range
        if min_val is not None:
            min_actual = tensor.min().item()
            if min_actual < min_val:
                raise ValidationError(f"Tensor values below minimum: {min_actual} < {min_val}")
        
        if max_val is not None:
            max_actual = tensor.max().item()
            if max_actual > max_val:
                raise ValidationError(f"Tensor values above maximum: {max_actual} > {max_val}")
        
        return tensor
    
    @staticmethod
    def validate_events(events: torch.Tensor) -> torch.Tensor:
        """
        Validate event tensor format.
        
        Args:
            events: Event tensor [N, 4] with [x, y, t, polarity]
            
        Returns:
            Validated events tensor
        """
        if events.numel() == 0:
            return events
        
        # Check basic shape
        if events.dim() != 2 or events.size(1) != 4:
            raise ValidationError(f"Events must have shape [N, 4], got {events.shape}")
        
        # Check coordinates are non-negative
        if (events[:, :2] < 0).any():
            raise ValidationError("Event coordinates must be non-negative")
        
        # Check timestamps are monotonic (optional - might not be required)
        if events.size(0) > 1:
            time_diffs = events[1:, 2] - events[:-1, 2]
            if (time_diffs < 0).any():
                logger.warning("Events are not temporally sorted")
        
        # Check polarity values
        unique_polarities = torch.unique(events[:, 3])
        valid_polarities = torch.tensor([-1.0, 0.0, 1.0])
        if not torch.isin(unique_polarities, valid_polarities).all():
            raise ValidationError(f"Invalid polarity values: {unique_polarities.tolist()}")
        
        return events
    
    @staticmethod
    def validate_string(
        value: str,
        max_length: int = 1000,
        allowed_chars: Optional[str] = None,
        forbidden_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: String to validate
            max_length: Maximum allowed length
            allowed_chars: Regex pattern for allowed characters
            forbidden_patterns: List of forbidden regex patterns
            
        Returns:
            Validated string
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")
        
        if len(value) > max_length:
            raise ValidationError(f"String too long: {len(value)} > {max_length}")
        
        # Check allowed characters
        if allowed_chars and not re.match(allowed_chars, value):
            raise ValidationError(f"String contains invalid characters: {value[:50]}...")
        
        # Check forbidden patterns
        if forbidden_patterns:
            for pattern in forbidden_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValidationError(f"String contains forbidden pattern: {pattern}")
        
        return value
    
    @staticmethod
    def validate_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary for security issues.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
        """
        if not isinstance(config, dict):
            raise ValidationError(f"Config must be dictionary, got {type(config)}")
        
        # Check for suspicious keys
        suspicious_keys = ['__', 'eval', 'exec', 'import', 'open', 'file']
        for key in config.keys():
            if any(suspicious in str(key).lower() for suspicious in suspicious_keys):
                raise ValidationError(f"Suspicious configuration key: {key}")
        
        # Recursively validate nested dictionaries
        validated_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                validated_config[key] = InputValidator.validate_config_dict(value)
            elif isinstance(value, str):
                # Basic string validation for config values
                validated_config[key] = InputValidator.validate_string(
                    value, 
                    max_length=10000,
                    forbidden_patterns=[r'<script', r'javascript:', r'data:']
                )
            else:
                validated_config[key] = value
        
        return validated_config
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent security issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Prevent hidden files
        if filename.startswith('.'):
            filename = '_' + filename[1:]
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename


class ModelValidator:
    """Validator for model-related inputs."""
    
    @staticmethod
    def validate_model_architecture(architecture: str) -> str:
        """Validate model architecture name."""
        allowed_architectures = [
            'liquid_net', 'rnn', 'lstm', 'gru', 'transformer',
            'tiny', 'small', 'base', 'large'
        ]
        
        if architecture not in allowed_architectures:
            raise ValidationError(
                f"Invalid architecture '{architecture}'. "
                f"Allowed: {allowed_architectures}"
            )
        
        return architecture
    
    @staticmethod
    def validate_model_dimensions(
        input_dim: int,
        output_dim: int,
        hidden_units: List[int]
    ) -> Tuple[int, int, List[int]]:
        """Validate model dimensions."""
        if input_dim <= 0 or input_dim > 100000:
            raise ValidationError(f"Invalid input_dim: {input_dim}")
        
        if output_dim <= 0 or output_dim > 10000:
            raise ValidationError(f"Invalid output_dim: {output_dim}")
        
        if not isinstance(hidden_units, list) or not hidden_units:
            raise ValidationError("hidden_units must be non-empty list")
        
        for i, units in enumerate(hidden_units):
            if not isinstance(units, int) or units <= 0 or units > 10000:
                raise ValidationError(f"Invalid hidden_units[{i}]: {units}")
        
        return input_dim, output_dim, hidden_units
    
    @staticmethod
    def validate_training_params(
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Tuple[int, int, float]:
        """Validate training parameters."""
        if epochs <= 0 or epochs > 10000:
            raise ValidationError(f"Invalid epochs: {epochs}")
        
        if batch_size <= 0 or batch_size > 1024:
            raise ValidationError(f"Invalid batch_size: {batch_size}")
        
        if learning_rate <= 0 or learning_rate >= 1:
            raise ValidationError(f"Invalid learning_rate: {learning_rate}")
        
        return epochs, batch_size, learning_rate


def validate_and_sanitize_input(
    input_data: Any,
    input_type: str,
    **validation_kwargs
) -> Any:
    """
    Main validation function that routes to appropriate validators.
    
    Args:
        input_data: Data to validate
        input_type: Type of input for appropriate validation
        **validation_kwargs: Additional validation parameters
        
    Returns:
        Validated and sanitized input
    """
    validator_map = {
        'file_path': InputValidator.validate_file_path,
        'tensor': InputValidator.validate_tensor,
        'events': InputValidator.validate_events,
        'string': InputValidator.validate_string,
        'config': InputValidator.validate_config_dict,
        'architecture': ModelValidator.validate_model_architecture,
    }
    
    validator = validator_map.get(input_type)
    if validator is None:
        logger.warning(f"No validator found for input_type: {input_type}")
        return input_data
    
    try:
        return validator(input_data, **validation_kwargs)
    except Exception as e:
        logger.error(f"Validation failed for {input_type}: {e}")
        raise ValidationError(f"Input validation failed: {e}")


# Decorator for automatic input validation
def validate_inputs(**validators):
    """
    Decorator for automatic input validation.
    
    Args:
        **validators: Mapping of parameter names to validation configs
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate specified parameters
            for param_name, validation_config in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    input_type = validation_config.get('type')
                    validation_kwargs = {k: v for k, v in validation_config.items() if k != 'type'}
                    
                    validated_value = validate_and_sanitize_input(
                        value, input_type, **validation_kwargs
                    )
                    bound_args.arguments[param_name] = validated_value
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator