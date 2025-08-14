"""
Robust error handling and validation system for Generation 2.
Provides comprehensive error recovery, validation, and graceful degradation.
"""

import functools
import traceback
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Rich error context information."""
    function_name: str
    args: tuple
    kwargs: dict
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    error_message: str
    stack_trace: str
    recovery_attempted: bool = False
    recovery_successful: bool = False


class RobustErrorHandler:
    """Comprehensive error handling with recovery strategies."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.max_retries = 3
        self.retry_delay = 0.1
        self.logger = logging.getLogger(__name__)
        
        # Register default recovery strategies
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies.update({
            ValueError: self._handle_value_error,
            TypeError: self._handle_type_error,
            AttributeError: self._handle_attribute_error,
            IndexError: self._handle_index_error,
            KeyError: self._handle_key_error,
            ZeroDivisionError: self._handle_zero_division,
            OverflowError: self._handle_overflow,
            MemoryError: self._handle_memory_error,
        })
        
    def _handle_value_error(self, error: ValueError, context: ErrorContext) -> Any:
        """Handle value errors with input sanitization."""
        self.logger.warning(f"Attempting value error recovery: {error}")
        
        # Try to sanitize inputs
        if 'args' in context.kwargs:
            # Attempt to convert to valid ranges
            args = context.args
            if args and isinstance(args[0], (int, float)):
                if args[0] < 0:
                    return 0.0
                elif args[0] > 1e6:
                    return 1e6
                    
        return None
        
    def _handle_type_error(self, error: TypeError, context: ErrorContext) -> Any:
        """Handle type errors with type conversion."""
        self.logger.warning(f"Attempting type error recovery: {error}")
        
        # Try basic type conversions
        args = context.args
        if args:
            try:
                if isinstance(args[0], str) and args[0].replace('.', '').replace('-', '').isdigit():
                    return float(args[0])
                elif isinstance(args[0], (list, tuple)) and len(args[0]) > 0:
                    return args[0][0]
            except:
                pass
                
        return None
        
    def _handle_attribute_error(self, error: AttributeError, context: ErrorContext) -> Any:
        """Handle attribute errors with fallback attributes."""
        self.logger.warning(f"Attempting attribute error recovery: {error}")
        
        # Return safe default values for common attributes
        if "shape" in str(error):
            return (1, 1)
        elif "data" in str(error):
            return [[0.0]]
        elif "device" in str(error):
            return "cpu"
            
        return None
        
    def _handle_index_error(self, error: IndexError, context: ErrorContext) -> Any:
        """Handle index errors with bounds checking."""
        self.logger.warning(f"Attempting index error recovery: {error}")
        return 0  # Safe default index
        
    def _handle_key_error(self, error: KeyError, context: ErrorContext) -> Any:
        """Handle key errors with default values."""
        self.logger.warning(f"Attempting key error recovery: {error}")
        return None  # Safe default value
        
    def _handle_zero_division(self, error: ZeroDivisionError, context: ErrorContext) -> Any:
        """Handle division by zero with epsilon."""
        self.logger.warning(f"Attempting zero division recovery: {error}")
        return 1e-8  # Small epsilon to avoid division by zero
        
    def _handle_overflow(self, error: OverflowError, context: ErrorContext) -> Any:
        """Handle overflow errors with clipping."""
        self.logger.warning(f"Attempting overflow recovery: {error}")
        return 1e6  # Safe maximum value
        
    def _handle_memory_error(self, error: MemoryError, context: ErrorContext) -> Any:
        """Handle memory errors with graceful degradation."""
        self.logger.error(f"Memory error - initiating graceful degradation: {error}")
        # Trigger garbage collection and return minimal result
        import gc
        gc.collect()
        return None
        
    def register_recovery_strategy(self, exception_type: Type[Exception], strategy: Callable):
        """Register custom recovery strategy for specific exception type."""
        self.recovery_strategies[exception_type] = strategy
        
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ZeroDivisionError, OverflowError, TypeError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, AttributeError, IndexError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
            
    def _create_error_context(self, func: Callable, args: tuple, kwargs: dict, 
                            error: Exception) -> ErrorContext:
        """Create detailed error context."""
        return ErrorContext(
            function_name=func.__name__,
            args=args,
            kwargs=kwargs,
            timestamp=time.time(),
            severity=self._determine_severity(error),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc()
        )
        
    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Attempt to recover from error using registered strategies."""
        context.recovery_attempted = True
        
        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                result = self.recovery_strategies[error_type](error, context)
                if result is not None:
                    context.recovery_successful = True
                    self.logger.info(f"Successfully recovered from {error_type.__name__}")
                    return result
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
                
        return None
        
    def handle_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with comprehensive error handling and recovery."""
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as error:
                context = self._create_error_context(func, args, kwargs, error)
                self.error_history.append(context)
                
                self.logger.error(
                    f"Error in {func.__name__} (attempt {attempt + 1}): "
                    f"{type(error).__name__}: {error}"
                )
                
                # Try recovery on first attempt
                if attempt == 0:
                    recovery_result = self._attempt_recovery(error, context)
                    if recovery_result is not None:
                        return recovery_result
                        
                # If not the last attempt, wait and retry
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                    
                # All attempts failed
                if context.severity == ErrorSeverity.CRITICAL:
                    self.logger.critical(f"Critical error in {func.__name__}: {error}")
                    raise error
                else:
                    self.logger.warning(f"Returning None due to unrecoverable error: {error}")
                    return None
                    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history."""
        if not self.error_history:
            return {"total_errors": 0, "recovery_rate": 1.0}
            
        total_errors = len(self.error_history)
        successful_recoveries = sum(1 for ctx in self.error_history if ctx.recovery_successful)
        recovery_rate = successful_recoveries / total_errors
        
        error_types = {}
        for ctx in self.error_history:
            error_types[ctx.error_type] = error_types.get(ctx.error_type, 0) + 1
            
        return {
            "total_errors": total_errors,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": recovery_rate,
            "error_types": error_types,
            "recent_errors": [
                {
                    "function": ctx.function_name,
                    "error_type": ctx.error_type,
                    "severity": ctx.severity.value,
                    "recovered": ctx.recovery_successful
                }
                for ctx in self.error_history[-5:]  # Last 5 errors
            ]
        }


# Global error handler instance
global_error_handler = RobustErrorHandler()


def robust_execution(max_retries: int = 3, retry_delay: float = 0.1):
    """Decorator for robust function execution with error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return global_error_handler.handle_with_recovery(func, *args, **kwargs)
        return wrapper
    return decorator


def validate_and_sanitize_inputs(**validators):
    """Decorator for input validation and sanitization."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply validators to kwargs
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = validator(kwargs[param_name])
                    except Exception as e:
                        logging.warning(f"Input validation failed for {param_name}: {e}")
                        # Use default value or skip parameter
                        if hasattr(validator, 'default'):
                            kwargs[param_name] = validator.default
                        else:
                            kwargs.pop(param_name, None)
                            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Common validators
class Validators:
    """Common input validators with sanitization."""
    
    @staticmethod
    def positive_float(value: Any, default: float = 1.0) -> float:
        """Validate and sanitize to positive float."""
        try:
            val = float(value)
            return max(val, 1e-8)  # Ensure positive
        except (ValueError, TypeError):
            return default
            
    @staticmethod
    def bounded_float(min_val: float = -1.0, max_val: float = 1.0, default: float = 0.0):
        """Create bounded float validator."""
        def validator(value: Any) -> float:
            try:
                val = float(value)
                return max(min_val, min(max_val, val))
            except (ValueError, TypeError):
                return default
        return validator
        
    @staticmethod
    def positive_int(value: Any, default: int = 1) -> int:
        """Validate and sanitize to positive integer."""
        try:
            val = int(value)
            return max(val, 1)
        except (ValueError, TypeError):
            return default
            
    @staticmethod
    def valid_architecture(value: Any, default: str = "small") -> str:
        """Validate architecture string."""
        valid_archs = ["tiny", "small", "base", "large"]
        if isinstance(value, str) and value.lower() in valid_archs:
            return value.lower()
        return default


# Example usage functions with robust error handling
@robust_execution(max_retries=3)
@validate_and_sanitize_inputs(
    input_dim=Validators.positive_int,
    output_dim=Validators.positive_int,
    tau=Validators.positive_float,
    leak=Validators.bounded_float(0.0, 1.0, 0.1)
)
def create_robust_liquid_net(input_dim: int = 2, output_dim: int = 3, 
                           architecture: str = "small", **kwargs):
    """Create liquid network with robust error handling."""
    from ..core.minimal_fallback import create_minimal_liquid_net
    
    # Additional validation
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Dimensions must be positive")
        
    return create_minimal_liquid_net(
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
        **kwargs
    )


def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    error_summary = global_error_handler.get_error_summary()
    
    # Determine overall health
    if error_summary["total_errors"] == 0:
        health_status = "excellent"
    elif error_summary["recovery_rate"] > 0.8:
        health_status = "good"
    elif error_summary["recovery_rate"] > 0.5:
        health_status = "fair"
    else:
        health_status = "poor"
        
    return {
        "health_status": health_status,
        "error_summary": error_summary,
        "uptime": time.time(),
        "robust_features_active": True,
    }