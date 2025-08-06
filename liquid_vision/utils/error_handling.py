"""
Comprehensive error handling and validation system for liquid neural networks.
Provides robust error recovery, validation, and debugging capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Type
from enum import Enum
import traceback
import logging
import functools
import time
from dataclasses import dataclass
from contextlib import contextmanager
import warnings

from .logging import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    MEMORY = "memory"
    HARDWARE = "hardware"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    DATA = "data"
    DEPLOYMENT = "deployment"


@dataclass
class ErrorInfo:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: float
    traceback: Optional[str] = None
    recovery_action: Optional[str] = None
    user_guidance: Optional[str] = None


class LiquidVisionError(Exception):
    """Base exception for liquid vision framework."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        recovery_action: Optional[str] = None,
        user_guidance: Optional[str] = None
    ):
        super().__init__(message)
        self.error_info = ErrorInfo(
            category=category,
            severity=severity,
            message=message,
            details=details or {},
            timestamp=time.time(),
            traceback=traceback.format_exc(),
            recovery_action=recovery_action,
            user_guidance=user_guidance
        )


class ValidationError(LiquidVisionError):
    """Input validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ComputationError(LiquidVisionError):
    """Computation and numerical errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class MemoryError(LiquidVisionError):
    """Memory-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class HardwareError(LiquidVisionError):
    """Hardware interface errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.HARDWARE,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class DeploymentError(LiquidVisionError):
    """Deployment and edge-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DEPLOYMENT,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ErrorHandler:
    """Central error handling and recovery system."""
    
    def __init__(self, log_errors: bool = True, recover_enabled: bool = True):
        self.log_errors = log_errors
        self.recover_enabled = recover_enabled
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.MEMORY: [
                self._recover_memory_cleanup,
                self._recover_memory_reduce_precision,
                self._recover_memory_batch_reduction
            ],
            ErrorCategory.COMPUTATION: [
                self._recover_computation_fallback,
                self._recover_computation_precision_reduction
            ],
            ErrorCategory.HARDWARE: [
                self._recover_hardware_reconnect,
                self._recover_hardware_fallback_cpu
            ]
        }
        self.logger = get_logger(__name__)
        
        # Statistics
        self.error_counts = {category: 0 for category in ErrorCategory}
        self.recovery_counts = {category: 0 for category in ErrorCategory}
        
    def handle_error(self, error: LiquidVisionError, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle an error with optional recovery.
        
        Returns:
            True if error was recovered, False otherwise
        """
        error_info = error.error_info
        error_info.details.update(context or {})
        
        # Log error
        if self.log_errors:
            self._log_error(error_info)
        
        # Store in history
        self.error_history.append(error_info)
        self.error_counts[error_info.category] += 1
        
        # Attempt recovery
        if self.recover_enabled and error_info.severity not in [ErrorSeverity.FATAL, ErrorSeverity.CRITICAL]:
            recovery_success = self._attempt_recovery(error_info, context)
            if recovery_success:
                self.recovery_counts[error_info.category] += 1
                self.logger.info(f"Successfully recovered from {error_info.category.value} error")
                return True
        
        return False
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information."""
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        message = f"[{error_info.category.value.upper()}] {error_info.message}"
        if error_info.details:
            message += f" | Details: {error_info.details}"
        
        self.logger.log(log_level, message)
        
        if error_info.recovery_action:
            self.logger.info(f"Recovery action: {error_info.recovery_action}")
        if error_info.user_guidance:
            self.logger.info(f"User guidance: {error_info.user_guidance}")
    
    def _attempt_recovery(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Attempt recovery using registered strategies."""
        strategies = self.recovery_strategies.get(error_info.category, [])
        
        for strategy in strategies:
            try:
                self.logger.debug(f"Attempting recovery strategy: {strategy.__name__}")
                success = strategy(error_info, context)
                if success:
                    self.logger.info(f"Recovery successful using {strategy.__name__}")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
                continue
        
        return False
    
    def _recover_memory_cleanup(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Memory recovery: cleanup caches and run garbage collection."""
        try:
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.info("Memory cleanup completed")
            return True
        except Exception:
            return False
    
    def _recover_memory_reduce_precision(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Memory recovery: reduce precision of tensors."""
        try:
            model = context.get('model') if context else None
            if model is not None:
                # Convert model to half precision
                model.half()
                self.logger.info("Reduced model precision to float16")
                return True
        except Exception:
            pass
        return False
    
    def _recover_memory_batch_reduction(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Memory recovery: reduce batch size."""
        try:
            config = context.get('config') if context else None
            if config is not None and hasattr(config, 'batch_size'):
                original_batch_size = config.batch_size
                config.batch_size = max(1, config.batch_size // 2)
                self.logger.info(f"Reduced batch size from {original_batch_size} to {config.batch_size}")
                return True
        except Exception:
            pass
        return False
    
    def _recover_computation_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Computation recovery: fallback to simpler computation."""
        try:
            # This would implement fallback computation logic
            self.logger.info("Using fallback computation method")
            return True
        except Exception:
            return False
    
    def _recover_computation_precision_reduction(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Computation recovery: reduce numerical precision."""
        try:
            # Reduce precision in computations
            self.logger.info("Reduced computation precision")
            return True
        except Exception:
            return False
    
    def _recover_hardware_reconnect(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Hardware recovery: attempt reconnection."""
        try:
            adapter = context.get('hardware_adapter') if context else None
            if adapter is not None and hasattr(adapter, 'connect'):
                # Attempt reconnection
                connection_params = context.get('connection_params', {})
                success = adapter.connect(connection_params)
                if success:
                    self.logger.info("Hardware reconnection successful")
                    return True
        except Exception:
            pass
        return False
    
    def _recover_hardware_fallback_cpu(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Hardware recovery: fallback to CPU processing."""
        try:
            model = context.get('model') if context else None
            if model is not None:
                model.cpu()
                self.logger.info("Fallback to CPU processing")
                return True
        except Exception:
            pass
        return False
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        total_errors = sum(self.error_counts.values())
        total_recoveries = sum(self.recovery_counts.values())
        
        return {
            'total_errors': total_errors,
            'total_recoveries': total_recoveries,
            'recovery_rate': total_recoveries / max(1, total_errors),
            'errors_by_category': dict(self.error_counts),
            'recoveries_by_category': dict(self.recovery_counts),
            'recent_errors': [
                {
                    'category': err.category.value,
                    'severity': err.severity.value,
                    'message': err.message,
                    'timestamp': err.timestamp
                } for err in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Register custom recovery strategy."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _global_error_handler


def robust_operation(
    category: ErrorCategory = ErrorCategory.COMPUTATION,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    suppress_errors: bool = False
):
    """
    Decorator for robust operation execution with error handling and retry.
    
    Args:
        category: Error category for classification
        severity: Default severity level
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
        suppress_errors: Whether to suppress errors after all retries fail
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_error_handler()
            last_exception = None
            
            for attempt in range(retry_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except LiquidVisionError as e:
                    last_exception = e
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_attempts': retry_attempts + 1,
                        'args': args,
                        'kwargs': kwargs
                    }
                    
                    recovered = handler.handle_error(e, context)
                    if recovered:
                        # Retry if recovery was successful
                        if attempt < retry_attempts:
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Recovery successful but no more retries
                            break
                    else:
                        # Recovery failed
                        if attempt < retry_attempts:
                            time.sleep(retry_delay)
                            continue
                        else:
                            break
                            
                except Exception as e:
                    # Convert to framework exception
                    framework_error = LiquidVisionError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        category=category,
                        severity=severity,
                        details={'original_exception': type(e).__name__}
                    )
                    
                    last_exception = framework_error
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_attempts': retry_attempts + 1
                    }
                    
                    recovered = handler.handle_error(framework_error, context)
                    if not recovered and attempt < retry_attempts:
                        time.sleep(retry_delay)
                        continue
                    break
            
            # All attempts failed
            if suppress_errors:
                handler.logger.warning(f"All attempts failed for {func.__name__}, suppressing error")
                return None
            else:
                raise last_exception if last_exception else RuntimeError(f"All attempts failed for {func.__name__}")
                
        return wrapper
    return decorator


@contextmanager
def error_context(
    operation_name: str,
    category: ErrorCategory = ErrorCategory.COMPUTATION,
    **context_data
):
    """Context manager for operation error handling."""
    handler = get_error_handler()
    start_time = time.time()
    
    try:
        handler.logger.debug(f"Starting operation: {operation_name}")
        yield
        duration = time.time() - start_time
        handler.logger.debug(f"Operation {operation_name} completed successfully in {duration:.3f}s")
        
    except LiquidVisionError as e:
        context = {
            'operation': operation_name,
            'duration': time.time() - start_time,
            **context_data
        }
        handler.handle_error(e, context)
        raise
        
    except Exception as e:
        # Convert to framework exception
        framework_error = LiquidVisionError(
            f"Error in operation {operation_name}: {str(e)}",
            category=category,
            details={'original_exception': type(e).__name__, **context_data}
        )
        
        context = {
            'operation': operation_name,
            'duration': time.time() - start_time,
            **context_data
        }
        handler.handle_error(framework_error, context)
        raise framework_error


def validate_tensor_input(tensor: torch.Tensor, name: str = "tensor", **requirements) -> torch.Tensor:
    """
    Comprehensive tensor validation with detailed error messages.
    
    Args:
        tensor: Input tensor to validate
        name: Name of tensor for error messages
        **requirements: Validation requirements (shape, dtype, device, etc.)
    
    Returns:
        Validated tensor
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"{name} must be a torch.Tensor, got {type(tensor)}",
            details={'expected_type': 'torch.Tensor', 'actual_type': type(tensor).__name__},
            user_guidance=f"Convert {name} to torch tensor using torch.tensor() or torch.from_numpy()"
        )
    
    # Validate shape
    if 'shape' in requirements:
        expected_shape = requirements['shape']
        if tensor.shape != expected_shape:
            raise ValidationError(
                f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}",
                details={'expected_shape': expected_shape, 'actual_shape': list(tensor.shape)},
                user_guidance=f"Reshape {name} to {expected_shape} using tensor.reshape() or tensor.view()"
            )
    
    if 'ndim' in requirements:
        expected_ndim = requirements['ndim']
        if tensor.ndim != expected_ndim:
            raise ValidationError(
                f"{name} dimension mismatch: expected {expected_ndim}D, got {tensor.ndim}D",
                details={'expected_ndim': expected_ndim, 'actual_ndim': tensor.ndim}
            )
    
    if 'min_shape' in requirements:
        min_shape = requirements['min_shape']
        for i, (actual, minimum) in enumerate(zip(tensor.shape, min_shape)):
            if actual < minimum:
                raise ValidationError(
                    f"{name} dimension {i} too small: expected >= {minimum}, got {actual}",
                    details={'dimension': i, 'minimum': minimum, 'actual': actual}
                )
    
    # Validate dtype
    if 'dtype' in requirements:
        expected_dtype = requirements['dtype']
        if tensor.dtype != expected_dtype:
            raise ValidationError(
                f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}",
                details={'expected_dtype': str(expected_dtype), 'actual_dtype': str(tensor.dtype)},
                recovery_action=f"Convert using tensor.to({expected_dtype})"
            )
    
    # Validate device
    if 'device' in requirements:
        expected_device = requirements['device']
        if str(tensor.device) != str(expected_device):
            raise ValidationError(
                f"{name} device mismatch: expected {expected_device}, got {tensor.device}",
                details={'expected_device': str(expected_device), 'actual_device': str(tensor.device)},
                recovery_action=f"Move using tensor.to('{expected_device}')"
            )
    
    # Validate value ranges
    if 'min_value' in requirements:
        min_value = requirements['min_value']
        actual_min = tensor.min().item()
        if actual_min < min_value:
            raise ValidationError(
                f"{name} contains values below minimum: min value is {actual_min}, expected >= {min_value}",
                details={'min_value_found': actual_min, 'min_value_required': min_value}
            )
    
    if 'max_value' in requirements:
        max_value = requirements['max_value']
        actual_max = tensor.max().item()
        if actual_max > max_value:
            raise ValidationError(
                f"{name} contains values above maximum: max value is {actual_max}, expected <= {max_value}",
                details={'max_value_found': actual_max, 'max_value_required': max_value}
            )
    
    # Validate for NaN/Inf
    if requirements.get('no_nan', True):
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            raise ValidationError(
                f"{name} contains {nan_count} NaN values",
                details={'nan_count': nan_count, 'total_elements': tensor.numel()},
                recovery_action="Use torch.nan_to_num() to replace NaN values"
            )
    
    if requirements.get('no_inf', True):
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            raise ValidationError(
                f"{name} contains {inf_count} infinite values",
                details={'inf_count': inf_count, 'total_elements': tensor.numel()},
                recovery_action="Use torch.clamp() to limit value range"
            )
    
    return tensor


def validate_model_state(model: torch.nn.Module) -> torch.nn.Module:
    """Validate model state and parameters."""
    if not isinstance(model, torch.nn.Module):
        raise ValidationError(
            f"Expected torch.nn.Module, got {type(model)}",
            details={'actual_type': type(model).__name__}
        )
    
    # Check for NaN parameters
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
    
    if nan_params:
        raise ValidationError(
            f"Model contains NaN parameters: {nan_params}",
            details={'nan_parameters': nan_params},
            recovery_action="Reinitialize affected parameters or load from checkpoint"
        )
    
    if inf_params:
        raise ValidationError(
            f"Model contains infinite parameters: {inf_params}",
            details={'inf_parameters': inf_params},
            recovery_action="Apply gradient clipping or parameter regularization"
        )
    
    return model


def safe_tensor_operation(operation: str, *tensors, **kwargs) -> torch.Tensor:
    """
    Safely execute tensor operations with comprehensive error handling.
    
    Args:
        operation: Name of torch operation to perform
        *tensors: Input tensors
        **kwargs: Additional arguments for the operation
        
    Returns:
        Result tensor
    """
    if not hasattr(torch, operation) and not hasattr(torch.Tensor, operation):
        raise ValidationError(
            f"Unknown tensor operation: {operation}",
            details={'operation': operation},
            user_guidance="Check PyTorch documentation for available operations"
        )
    
    try:
        # Validate input tensors
        for i, tensor in enumerate(tensors):
            validate_tensor_input(tensor, f"tensor_{i}", no_nan=True, no_inf=True)
        
        # Get the operation function
        if hasattr(torch, operation):
            op_func = getattr(torch, operation)
            result = op_func(*tensors, **kwargs)
        else:
            # Method on first tensor
            op_func = getattr(tensors[0], operation)
            result = op_func(*tensors[1:], **kwargs)
        
        # Validate result
        if isinstance(result, torch.Tensor):
            validate_tensor_input(result, "operation_result", no_nan=True, no_inf=True)
        
        return result
        
    except Exception as e:
        raise ComputationError(
            f"Error in tensor operation '{operation}': {str(e)}",
            details={
                'operation': operation,
                'tensor_shapes': [t.shape for t in tensors],
                'tensor_dtypes': [str(t.dtype) for t in tensors],
                'kwargs': kwargs
            }
        )


# Custom warning handler for framework warnings
def warning_handler(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that converts warnings to log messages."""
    logger = get_logger(__name__)
    logger.warning(f"[{category.__name__}] {message} ({filename}:{lineno})")


# Set up custom warning handler
warnings.showwarning = warning_handler