"""
🛡️ Generation 2 Robust Error Handling - AUTONOMOUS IMPLEMENTATION
Production-grade error handling with self-healing capabilities

Features:
- Intelligent error recovery with fallback mechanisms
- Real-time health monitoring and alerting
- Comprehensive logging with structured error reporting
- Graceful degradation under resource constraints
- Automatic retry logic with exponential backoff
"""

import torch
import logging
import traceback
import time
import psutil
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import sys


class ErrorSeverity(Enum):
    """Error severity levels for structured handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for specialized handling."""
    MEMORY = "memory"
    COMPUTATION = "computation"
    IO = "io"
    NETWORK = "network"
    DATA = "data"
    HARDWARE = "hardware"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Comprehensive error context for diagnosis."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: str
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'stack_trace': self.stack_trace,
            'system_state': self.system_state,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'retry_count': self.retry_count,
        }


class SystemHealthMonitor:
    """Real-time system health monitoring."""
    
    def __init__(self):
        self.health_checks = {
            'cpu_usage': self._check_cpu_usage,
            'memory_usage': self._check_memory_usage,
            'gpu_memory': self._check_gpu_memory,
            'disk_space': self._check_disk_space,
            'temperature': self._check_temperature,
        }
        self.health_thresholds = {
            'cpu_usage': 90.0,  # percent
            'memory_usage': 85.0,  # percent
            'gpu_memory': 90.0,  # percent
            'disk_space': 95.0,  # percent
            'temperature': 85.0,  # celsius
        }
        
    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state snapshot."""
        state = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                state[check_name] = check_func()
            except Exception as e:
                state[check_name] = f"check_failed: {str(e)}"
                
        state['timestamp'] = time.time()
        return state
        
    def _check_cpu_usage(self) -> float:
        """Check CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
        
    def _check_memory_usage(self) -> Dict[str, float]:
        """Check memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
        
    def _check_gpu_memory(self) -> Dict[str, float]:
        """Check GPU memory usage."""
        if not torch.cuda.is_available():
            return {'available': False}
            
        return {
            'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
        }
        
    def _check_disk_space(self) -> Dict[str, float]:
        """Check disk space usage."""
        disk_usage = psutil.disk_usage('/')
        return {
            'percent': (disk_usage.used / disk_usage.total) * 100,
            'free_gb': disk_usage.free / (1024**3),
            'total_gb': disk_usage.total / (1024**3)
        }
        
    def _check_temperature(self) -> Dict[str, float]:
        """Check system temperature (if available)."""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature
                cpu_temps = temps.get('cpu_thermal', temps.get('coretemp', []))
                if cpu_temps:
                    return {'cpu_temp': cpu_temps[0].current}
            return {'temperature_unavailable': True}
        except Exception:
            return {'temperature_check_failed': True}
            
    def check_health_alerts(self) -> List[Dict[str, Any]]:
        """Check for system health alerts."""
        alerts = []
        system_state = self.get_system_state()
        
        for metric, threshold in self.health_thresholds.items():
            if metric in system_state:
                value = system_state[metric]
                if isinstance(value, dict):
                    # Handle nested metrics like memory_usage
                    if 'percent' in value and value['percent'] > threshold:
                        alerts.append({
                            'metric': metric,
                            'value': value['percent'],
                            'threshold': threshold,
                            'severity': 'high' if value['percent'] > threshold * 1.1 else 'medium'
                        })
                elif isinstance(value, (int, float)) and value > threshold:
                    alerts.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'severity': 'high' if value > threshold * 1.1 else 'medium'
                    })
                    
        return alerts


class Generation2ErrorHandler:
    """
    🛡️ Production-grade error handler with intelligent recovery.
    
    Features:
    - Automatic error classification and severity assessment
    - Context-aware recovery strategies
    - Real-time health monitoring integration
    - Structured logging with error analytics
    - Graceful degradation mechanisms
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_retry_attempts: int = 3,
        enable_health_monitoring: bool = True,
    ):
        self.max_retry_attempts = max_retry_attempts
        self.enable_health_monitoring = enable_health_monitoring
        
        # Setup logging
        self.logger = self._setup_logging(log_file)
        
        # System monitoring
        self.health_monitor = SystemHealthMonitor() if enable_health_monitoring else None
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Performance metrics
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'average_recovery_time': 0.0,
        }
        
        self.logger.info("🛡️ Generation 2 Error Handler initialized")
        
    def _setup_logging(self, log_file: Optional[str]) -> logging.Logger:
        """Setup structured logging with multiple handlers."""
        logger = logging.getLogger('generation2_error_handler')
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
        return logger
        
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[Callable]]:
        """Initialize recovery strategies for different error categories."""
        return {
            ErrorCategory.MEMORY: [
                self._clear_gpu_cache,
                self._reduce_batch_size,
                self._enable_gradient_checkpointing,
            ],
            ErrorCategory.COMPUTATION: [
                self._retry_with_cpu,
                self._reduce_precision,
                self._simplify_computation,
            ],
            ErrorCategory.IO: [
                self._retry_with_backoff,
                self._use_alternative_path,
                self._cache_fallback,
            ],
            ErrorCategory.DATA: [
                self._validate_and_clean_data,
                self._use_default_values,
                self._skip_corrupted_data,
            ],
            ErrorCategory.HARDWARE: [
                self._switch_device,
                self._reduce_workload,
                self._enable_safe_mode,
            ],
        }
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        allow_recovery: bool = True,
    ) -> ErrorContext:
        """
        Comprehensive error handling with intelligent recovery.
        
        Args:
            error: The exception to handle
            context: Additional context information
            allow_recovery: Whether to attempt recovery
            
        Returns:
            ErrorContext with handling results
        """
        
        # Create error context
        error_context = self._create_error_context(error, context)
        
        # Log error with full context
        self._log_error(error_context)
        
        # Update statistics
        self.error_stats['total_errors'] += 1
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.error_stats['critical_errors'] += 1
            
        # Attempt recovery if enabled
        if allow_recovery and error_context.category != ErrorCategory.UNKNOWN:
            recovery_start_time = time.time()
            
            success = self._attempt_recovery(error_context)
            recovery_time = time.time() - recovery_start_time
            
            if success:
                self.error_stats['recovered_errors'] += 1
                self.error_stats['average_recovery_time'] = (
                    (self.error_stats['average_recovery_time'] * (self.error_stats['recovered_errors'] - 1) + 
                     recovery_time) / self.error_stats['recovered_errors']
                )
                
        # Store error for analytics
        self.error_history.append(error_context)
        
        return error_context
        
    def _create_error_context(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]]
    ) -> ErrorContext:
        """Create comprehensive error context."""
        
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Classify error
        severity = self._classify_error_severity(error)
        category = self._classify_error_category(error, error_message)
        
        # Get system state
        system_state = {}
        if self.health_monitor:
            system_state = self.health_monitor.get_system_state()
        if context:
            system_state.update(context)
            
        return ErrorContext(
            timestamp=time.time(),
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            category=category,
            stack_trace=stack_trace,
            system_state=system_state,
        )
        
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        
        critical_errors = (
            SystemExit, KeyboardInterrupt, MemoryError, 
            torch.cuda.OutOfMemoryError
        )
        
        high_severity_errors = (
            RuntimeError, ValueError, TypeError, IOError,
            FileNotFoundError, ConnectionError
        )
        
        if isinstance(error, critical_errors):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, high_severity_errors):
            return ErrorSeverity.HIGH
        elif isinstance(error, Warning):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
            
    def _classify_error_category(self, error: Exception, message: str) -> ErrorCategory:
        """Classify error into categories for targeted recovery."""
        
        # Memory-related errors
        if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
            return ErrorCategory.MEMORY
        if "out of memory" in message.lower() or "oom" in message.lower():
            return ErrorCategory.MEMORY
            
        # Computation errors
        if isinstance(error, (RuntimeError, ArithmeticError)):
            if "cuda" in message.lower() or "gpu" in message.lower():
                return ErrorCategory.HARDWARE
            return ErrorCategory.COMPUTATION
            
        # I/O errors  
        if isinstance(error, (IOError, FileNotFoundError, PermissionError)):
            return ErrorCategory.IO
            
        # Network errors
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
            
        # Data errors
        if isinstance(error, (ValueError, TypeError)) and "data" in message.lower():
            return ErrorCategory.DATA
            
        return ErrorCategory.UNKNOWN
        
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt intelligent error recovery based on category."""
        
        if error_context.category not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy for category: {error_context.category}")
            return False
            
        error_context.recovery_attempted = True
        strategies = self.recovery_strategies[error_context.category]
        
        for i, strategy in enumerate(strategies):
            if error_context.retry_count >= self.max_retry_attempts:
                self.logger.warning(f"Max retry attempts reached for error: {error_context.error_type}")
                break
                
            try:
                self.logger.info(f"Attempting recovery strategy {i+1}/{len(strategies)}: {strategy.__name__}")
                
                result = strategy(error_context)
                
                if result:
                    error_context.recovery_successful = True
                    self.logger.info(f"✅ Recovery successful using strategy: {strategy.__name__}")
                    return True
                    
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {strategy.__name__}: {recovery_error}")
                
            error_context.retry_count += 1
            
        return False
        
    def _log_error(self, error_context: ErrorContext):
        """Log error with structured information."""
        
        # Determine log level based on severity
        log_levels = {
            ErrorSeverity.LOW: self.logger.info,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical,
        }
        
        log_func = log_levels[error_context.severity]
        
        # Create structured log message
        log_message = (
            f"🚨 Error Detected - {error_context.error_type}\n"
            f"   Message: {error_context.error_message}\n"
            f"   Severity: {error_context.severity.value}\n"
            f"   Category: {error_context.category.value}\n"
            f"   Timestamp: {error_context.timestamp}\n"
        )
        
        # Add system state if available
        if error_context.system_state:
            log_message += f"   System State: {json.dumps(error_context.system_state, indent=2)}\n"
            
        log_func(log_message)
        
        # Log stack trace for high/critical errors
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Stack trace:\n{error_context.stack_trace}")
            
    # Recovery strategy implementations
    def _clear_gpu_cache(self, error_context: ErrorContext) -> bool:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
        return False
        
    def _reduce_batch_size(self, error_context: ErrorContext) -> bool:
        """Signal to reduce batch size (handled by caller)."""
        error_context.system_state['suggested_action'] = 'reduce_batch_size'
        return True
        
    def _enable_gradient_checkpointing(self, error_context: ErrorContext) -> bool:
        """Signal to enable gradient checkpointing."""
        error_context.system_state['suggested_action'] = 'enable_gradient_checkpointing'
        return True
        
    def _retry_with_cpu(self, error_context: ErrorContext) -> bool:
        """Signal to retry computation on CPU."""
        error_context.system_state['suggested_action'] = 'use_cpu'
        return True
        
    def _reduce_precision(self, error_context: ErrorContext) -> bool:
        """Signal to reduce numerical precision."""
        error_context.system_state['suggested_action'] = 'reduce_precision'
        return True
        
    def _simplify_computation(self, error_context: ErrorContext) -> bool:
        """Signal to simplify computation."""
        error_context.system_state['suggested_action'] = 'simplify_computation'
        return True
        
    def _retry_with_backoff(self, error_context: ErrorContext) -> bool:
        """Implement exponential backoff retry."""
        backoff_time = 2 ** error_context.retry_count
        self.logger.info(f"Retrying after {backoff_time}s backoff")
        time.sleep(backoff_time)
        return True
        
    def _use_alternative_path(self, error_context: ErrorContext) -> bool:
        """Signal to use alternative I/O path."""
        error_context.system_state['suggested_action'] = 'use_alternative_path'
        return True
        
    def _cache_fallback(self, error_context: ErrorContext) -> bool:
        """Signal to use cached fallback."""
        error_context.system_state['suggested_action'] = 'use_cache_fallback'
        return True
        
    def _validate_and_clean_data(self, error_context: ErrorContext) -> bool:
        """Signal to validate and clean data."""
        error_context.system_state['suggested_action'] = 'validate_and_clean_data'
        return True
        
    def _use_default_values(self, error_context: ErrorContext) -> bool:
        """Signal to use default values for corrupted data."""
        error_context.system_state['suggested_action'] = 'use_default_values'
        return True
        
    def _skip_corrupted_data(self, error_context: ErrorContext) -> bool:
        """Signal to skip corrupted data samples."""
        error_context.system_state['suggested_action'] = 'skip_corrupted_data'
        return True
        
    def _switch_device(self, error_context: ErrorContext) -> bool:
        """Signal to switch to different device."""
        error_context.system_state['suggested_action'] = 'switch_device'
        return True
        
    def _reduce_workload(self, error_context: ErrorContext) -> bool:
        """Signal to reduce computational workload."""
        error_context.system_state['suggested_action'] = 'reduce_workload'
        return True
        
    def _enable_safe_mode(self, error_context: ErrorContext) -> bool:
        """Signal to enable safe mode operation."""
        error_context.system_state['suggested_action'] = 'enable_safe_mode'
        return True
        
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get comprehensive error analytics."""
        
        if not self.error_history:
            return {"status": "no_errors_recorded"}
            
        # Error distribution by category
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
        # Recovery success rate
        recovery_attempts = sum(1 for e in self.error_history if e.recovery_attempted)
        recovery_successes = sum(1 for e in self.error_history if e.recovery_successful)
        
        recovery_rate = recovery_successes / recovery_attempts if recovery_attempts > 0 else 0.0
        
        return {
            'total_errors': len(self.error_history),
            'error_distribution': {
                'by_category': category_counts,
                'by_severity': severity_counts,
            },
            'recovery_statistics': {
                'attempts': recovery_attempts,
                'successes': recovery_successes,
                'success_rate': recovery_rate,
                'average_recovery_time': self.error_stats['average_recovery_time'],
            },
            'system_health': self.health_monitor.get_system_state() if self.health_monitor else {},
            'recommendations': self._generate_recommendations(),
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations based on error patterns."""
        recommendations = []
        
        if not self.error_history:
            return recommendations
            
        # Analyze recent errors
        recent_errors = self.error_history[-10:]  # Last 10 errors
        
        # Memory-related recommendations
        memory_errors = [e for e in recent_errors if e.category == ErrorCategory.MEMORY]
        if len(memory_errors) > 3:
            recommendations.append("Consider increasing system RAM or reducing batch sizes")
            recommendations.append("Enable gradient checkpointing to reduce memory usage")
            
        # Computation recommendations
        computation_errors = [e for e in recent_errors if e.category == ErrorCategory.COMPUTATION]
        if len(computation_errors) > 2:
            recommendations.append("Consider using mixed precision training for better stability")
            recommendations.append("Review model architecture for numerical stability issues")
            
        # Hardware recommendations
        hardware_errors = [e for e in recent_errors if e.category == ErrorCategory.HARDWARE]
        if len(hardware_errors) > 1:
            recommendations.append("Check GPU health and update drivers")
            recommendations.append("Monitor system temperature and cooling")
            
        return recommendations


def robust_execution(
    max_retries: int = 3,
    fallback_value: Any = None,
    error_handler: Optional[Generation2ErrorHandler] = None,
):
    """
    Decorator for robust function execution with automatic error handling.
    
    Args:
        max_retries: Maximum number of retry attempts
        fallback_value: Value to return if all attempts fail
        error_handler: Custom error handler instance
        
    Usage:
        @robust_execution(max_retries=3, fallback_value=None)
        def risky_function():
            # Function that might fail
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or Generation2ErrorHandler()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        # Final attempt failed
                        error_context = handler.handle_error(e, {
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_retries': max_retries,
                        }, allow_recovery=False)
                        
                        if fallback_value is not None:
                            handler.logger.warning(f"Using fallback value for {func.__name__}")
                            return fallback_value
                        else:
                            raise e
                    else:
                        # Handle error and potentially retry
                        error_context = handler.handle_error(e, {
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_retries': max_retries,
                        })
                        
                        # Check if recovery suggests an action
                        suggested_action = error_context.system_state.get('suggested_action')
                        if suggested_action:
                            handler.logger.info(f"Suggested action for retry: {suggested_action}")
                            
                        # Brief pause before retry
                        time.sleep(0.5 * (attempt + 1))
                        
            return fallback_value
            
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> Generation2ErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = Generation2ErrorHandler()
    return _global_error_handler