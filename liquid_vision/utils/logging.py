"""
Advanced logging utilities for liquid vision simulation kit.
Supports structured logging, performance metrics, and error tracking.
"""

import logging
import logging.config
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from functools import wraps
import traceback


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'msg', 'args']:
                log_data[key] = value
        
        # Add location info for debug level
        if record.levelno == logging.DEBUG:
            log_data['location'] = f"{record.filename}:{record.lineno}"
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger_name: str = "liquid_vision.performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.metrics[operation] = {'start_time': time.time()}
        
    def end_timer(self, operation: str, **kwargs):
        """End timing and log the operation."""
        if operation in self.metrics:
            duration = time.time() - self.metrics[operation]['start_time']
            self.logger.info(
                f"Performance: {operation}",
                extra={
                    'operation': operation,
                    'duration_ms': duration * 1000,
                    'duration_s': duration,
                    **kwargs
                }
            )
            del self.metrics[operation]
        else:
            self.logger.warning(f"Timer not found for operation: {operation}")
    
    @contextmanager
    def time_operation(self, operation: str, **kwargs):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(
                f"Performance: {operation}",
                extra={
                    'operation': operation,
                    'duration_ms': duration * 1000,
                    'duration_s': duration,
                    **kwargs
                }
            )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    structured: bool = False,
    performance_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured: Whether to use structured JSON logging
        performance_logging: Whether to enable performance logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Create formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    handlers = [console_handler]
    
    # Create file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all messages, handlers filter
        handlers=handlers,
        force=True
    )
    
    # Create specialized loggers
    loggers = {
        'main': logging.getLogger('liquid_vision'),
        'training': logging.getLogger('liquid_vision.training'),
        'simulation': logging.getLogger('liquid_vision.simulation'),
        'deployment': logging.getLogger('liquid_vision.deployment'),
        'performance': logging.getLogger('liquid_vision.performance'),
        'security': logging.getLogger('liquid_vision.security'),
    }
    
    # Set levels for specialized loggers
    for logger_name, logger in loggers.items():
        if logger_name == 'performance' and not performance_logging:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(getattr(logging, level.upper()))
    
    return loggers


def log_exceptions(logger: Optional[logging.Logger] = None):
    """Decorator to log exceptions in functions."""
    if logger is None:
        logger = logging.getLogger('liquid_vision.errors')
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    extra={
                        'function': func.__name__,
                        'args': str(args)[:200],  # Truncate for safety
                        'kwargs': str(kwargs)[:200],
                        'exception_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                )
                raise
        return wrapper
    return decorator


def log_performance(operation_name: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """Decorator to log performance metrics for functions."""
    if logger is None:
        logger = logging.getLogger('liquid_vision.performance')
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise
            finally:
                duration = time.time() - start_time
                logger.info(
                    f"Performance: {op_name}",
                    extra={
                        'operation': op_name,
                        'duration_ms': duration * 1000,
                        'duration_s': duration,
                        'success': success,
                        'function': func.__name__,
                    }
                )
            
            return result
        return wrapper
    return decorator


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self):
        self.logger = logging.getLogger('liquid_vision.security')
        
    def log_input_validation_error(self, input_type: str, value: str, error: str):
        """Log input validation errors."""
        self.logger.warning(
            f"Input validation failed: {error}",
            extra={
                'event_type': 'input_validation_error',
                'input_type': input_type,
                'input_value': value[:100],  # Truncate for safety
                'error': error
            }
        )
        
    def log_file_access(self, file_path: str, operation: str, success: bool):
        """Log file access attempts."""
        self.logger.info(
            f"File access: {operation} {file_path}",
            extra={
                'event_type': 'file_access',
                'file_path': str(file_path),
                'operation': operation,
                'success': success
            }
        )
        
    def log_config_change(self, config_type: str, changes: Dict[str, Any], user: str = "unknown"):
        """Log configuration changes."""
        self.logger.info(
            f"Configuration changed: {config_type}",
            extra={
                'event_type': 'config_change',
                'config_type': config_type,
                'changes': changes,
                'user': user
            }
        )
        
    def log_model_deployment(self, model_path: str, target: str, success: bool):
        """Log model deployment events."""
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Model deployment: {'success' if success else 'failed'}",
            extra={
                'event_type': 'model_deployment',
                'model_path': str(model_path),
                'target': target,
                'success': success
            }
        )


class ErrorTracker:
    """Track and analyze errors across the system."""
    
    def __init__(self):
        self.logger = logging.getLogger('liquid_vision.errors')
        self.error_counts = {}
        self.error_history = []
        
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track an error occurrence."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Add to history
        error_record = {
            'timestamp': time.time(),
            'error_type': error_type,
            'message': error_message,
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        self.error_history.append(error_record)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # Log the error
        self.logger.error(
            f"Error tracked: {error_type} - {error_message}",
            extra={
                'error_type': error_type,
                'error_message': error_message,
                'error_count': self.error_counts[error_type],
                'context': context,
            }
        )
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of tracked errors."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': dict(self.error_counts),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
        
    def reset_tracking(self):
        """Reset error tracking."""
        self.error_counts.clear()
        self.error_history.clear()


# Global instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
error_tracker = ErrorTracker()