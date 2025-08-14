"""
Advanced monitoring and observability system for Generation 2.
Provides real-time metrics, health monitoring, and performance tracking.
"""

import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import logging


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: float
    latency_ms: float = 0.0
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected by monitoring system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"  
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """System alert with context and recommendations."""
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    source: str = "autonomous_system"
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        def wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except self.expected_exception as e:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    self.last_failure_time = time.time()
                raise e
        return wrapper


class AutonomousMonitor:
    """
    Autonomous monitoring system with self-healing capabilities.
    
    Features:
    - Real-time metrics collection and analysis
    - Distributed tracing and performance profiling  
    - Automatic anomaly detection and alerting
    - Self-healing and optimization recommendations
    - Circuit breaker patterns for fault tolerance
    """
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.alerts: deque = deque(maxlen=1000)
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.running = False
        self._monitor_thread = None
        
        # System health metrics
        self.system_stats = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_io": 0.0,
        }
        
        self._setup_default_thresholds()
        self._register_default_health_checks()
        
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds."""
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "response_time": {"warning": 200.0, "critical": 500.0},
            "error_rate": {"warning": 5.0, "critical": 10.0},
        }
        
    def _register_default_health_checks(self):
        """Register default system health checks."""
        self.health_checks = {
            "system_resources": self._check_system_resources,
            "memory_leaks": self._check_memory_leaks,
            "disk_space": self._check_disk_space,
            "process_health": self._check_process_health,
        }
    
    def start_monitoring(self) -> None:
        """Start autonomous monitoring in background thread."""
        if self.running:
            return
            
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🔍 Autonomous monitoring system started")
        
    def stop_monitoring(self) -> None:
        """Stop monitoring system gracefully."""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("🔍 Monitoring system stopped")
        
    def record_metric(self, name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     metric_type: MetricType = MetricType.GAUGE) -> None:
        """Record a metric data point."""
        metric = Metric(
            name=name,
            value=value,
            tags=tags or {},
            metric_type=metric_type
        )
        self.metrics.append(metric)
        
        # Check for threshold violations
        self._check_thresholds(metric)
        
    def create_alert(self, level: AlertLevel, message: str,
                    source: str = "system", context: Dict[str, Any] = None,
                    recommendations: List[str] = None) -> None:
        """Create system alert with recommendations."""
        alert = Alert(
            level=level,
            message=message,
            source=source,
            context=context or {},
            recommendations=recommendations or []
        )
        self.alerts.append(alert)
        
        # Log based on severity
        log_msg = f"[{level.value.upper()}] {message}"
        if level == AlertLevel.CRITICAL:
            logger.critical(log_msg)
        elif level == AlertLevel.ERROR:
            logger.error(log_msg)
        elif level == AlertLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register custom health check function."""
        self.health_checks[name] = check_func
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.metrics:
            return {"total_metrics": 0, "latest_timestamp": None}
            
        latest_metrics = {}
        metric_counts = defaultdict(int)
        
        for metric in reversed(list(self.metrics)[-100:]):  # Last 100 metrics
            if metric.name not in latest_metrics:
                latest_metrics[metric.name] = metric.value
            metric_counts[metric.metric_type.value] += 1
            
        return {
            "total_metrics": len(self.metrics),
            "latest_metrics": latest_metrics,
            "metric_type_counts": dict(metric_counts),
            "system_health": self.system_stats.copy(),
            "active_alerts": len([a for a in self.alerts if time.time() - a.timestamp < 3600]),
            "circuit_breaker_states": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }
        
    def get_recent_alerts(self, hours: float = 1.0) -> List[Alert]:
        """Get alerts from recent time period."""
        cutoff = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self.running:
            try:
                # Update system statistics
                self._update_system_stats()
                
                # Run health checks
                self._run_health_checks()
                
                # Check for anomalies
                self._detect_anomalies()
                
                # Self-healing actions
                self._execute_self_healing()
                
                # Sleep before next iteration
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(30)  # Back off on errors
                
    def _update_system_stats(self) -> None:
        """Update system performance statistics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_stats["cpu_usage"] = cpu_percent
            self.record_metric("system.cpu_usage", cpu_percent, {"unit": "percent"})
            
            # Memory usage  
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_stats["memory_usage"] = memory_percent
            self.record_metric("system.memory_usage", memory_percent, {"unit": "percent"})
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_stats["disk_usage"] = disk_percent
            self.record_metric("system.disk_usage", disk_percent, {"unit": "percent"})
            
            # Network I/O (simplified)
            net_io = psutil.net_io_counters()
            network_total = net_io.bytes_sent + net_io.bytes_recv
            self.system_stats["network_io"] = network_total
            self.record_metric("system.network_io", network_total, {"unit": "bytes"})
            
        except Exception as e:
            logger.warning(f"Failed to update system stats: {e}")
            
    def _run_health_checks(self) -> None:
        """Execute all registered health checks."""
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                self.record_metric(f"health_check.{check_name}", 1.0 if result else 0.0)
                
                if not result:
                    self.create_alert(
                        AlertLevel.WARNING,
                        f"Health check failed: {check_name}",
                        source="health_check",
                        recommendations=[f"Investigate {check_name} health issue"]
                    )
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                self.record_metric(f"health_check.{check_name}", 0.0)
                
    def _check_thresholds(self, metric: Metric) -> None:
        """Check if metric violates configured thresholds."""
        if metric.name in self.thresholds:
            thresholds = self.thresholds[metric.name]
            
            if metric.value >= thresholds.get("critical", float('inf')):
                self.create_alert(
                    AlertLevel.CRITICAL,
                    f"Critical threshold exceeded: {metric.name} = {metric.value}",
                    context={"metric": metric.name, "value": metric.value},
                    recommendations=[f"Immediate action required for {metric.name}"]
                )
            elif metric.value >= thresholds.get("warning", float('inf')):
                self.create_alert(
                    AlertLevel.WARNING,
                    f"Warning threshold exceeded: {metric.name} = {metric.value}",
                    context={"metric": metric.name, "value": metric.value}
                )
                
    def _detect_anomalies(self) -> None:
        """Detect performance anomalies and unusual patterns."""
        # Simple anomaly detection based on recent metrics
        if len(self.metrics) < 10:
            return
            
        recent_metrics = list(self.metrics)[-10:]
        metric_groups = defaultdict(list)
        
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
            
        for metric_name, values in metric_groups.items():
            if len(values) >= 5:
                avg_value = sum(values) / len(values)
                recent_value = values[-1]
                
                # Detect sudden spikes (>50% increase)
                if recent_value > avg_value * 1.5:
                    self.create_alert(
                        AlertLevel.WARNING,
                        f"Anomaly detected: {metric_name} spike to {recent_value:.2f}",
                        source="anomaly_detection",
                        context={"metric": metric_name, "spike_ratio": recent_value / avg_value}
                    )
                    
    def _execute_self_healing(self) -> None:
        """Execute self-healing actions based on system state."""
        # Memory pressure relief
        if self.system_stats["memory_usage"] > 90:
            self._trigger_memory_cleanup()
            
        # Circuit breaker recovery attempts
        for name, cb in self.circuit_breakers.items():
            if cb.state == "open" and time.time() - cb.last_failure_time > cb.timeout * 2:
                logger.info(f"Attempting circuit breaker recovery for {name}")
                cb.state = "half-open"
                cb.failure_count = max(0, cb.failure_count - 1)
                
    def _trigger_memory_cleanup(self) -> None:
        """Trigger memory cleanup procedures."""
        import gc
        collected = gc.collect()
        self.record_metric("system.gc_collected", collected)
        logger.info(f"Memory cleanup collected {collected} objects")
        
    # Health check implementations
    def _check_system_resources(self) -> bool:
        """Check overall system resource health."""
        return (self.system_stats["cpu_usage"] < 95 and 
                self.system_stats["memory_usage"] < 95)
    
    def _check_memory_leaks(self) -> bool:
        """Basic memory leak detection."""
        return self.system_stats["memory_usage"] < 90
        
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        return self.system_stats["disk_usage"] < 90
        
    def _check_process_health(self) -> bool:
        """Check current process health."""
        try:
            process = psutil.Process()
            return process.is_running() and process.status() != 'zombie'
        except:
            return False


# Global monitoring instance
_monitor: Optional[AutonomousMonitor] = None


def get_monitor() -> AutonomousMonitor:
    """Get or create global monitoring instance."""
    global _monitor
    if _monitor is None:
        _monitor = AutonomousMonitor()
        _monitor.start_monitoring()
    return _monitor


def record_metric(name: str, value: float, **kwargs) -> None:
    """Global function to record metrics."""
    get_monitor().record_metric(name, value, **kwargs)


def create_alert(level: AlertLevel, message: str, **kwargs) -> None:
    """Global function to create alerts.""" 
    get_monitor().create_alert(level, message, **kwargs)


def circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Global function to get circuit breaker."""
    return get_monitor().get_circuit_breaker(name, **kwargs)


class MonitoringMiddleware:
    """Middleware for automatic request/response monitoring."""
    
    def __init__(self, monitor: AutonomousMonitor):
        self.monitor = monitor
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add monitoring to functions."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.monitor.record_metric(
                    f"function.{function_name}.duration",
                    execution_time * 1000,  # Convert to ms
                    tags={"function": function_name, "status": "success"},
                    metric_type=MetricType.TIMER
                )
                
                self.monitor.record_metric(
                    f"function.{function_name}.calls",
                    1.0,
                    tags={"function": function_name, "status": "success"},
                    metric_type=MetricType.COUNTER
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                self.monitor.record_metric(
                    f"function.{function_name}.duration",
                    execution_time * 1000,
                    tags={"function": function_name, "status": "error"},
                    metric_type=MetricType.TIMER
                )
                
                self.monitor.record_metric(
                    f"function.{function_name}.errors",
                    1.0,
                    tags={"function": function_name, "error": str(type(e).__name__)},
                    metric_type=MetricType.COUNTER
                )
                
                self.monitor.create_alert(
                    AlertLevel.ERROR,
                    f"Function {function_name} failed: {str(e)}",
                    source="monitoring_middleware",
                    context={"function": function_name, "error": str(e)}
                )
                
                raise e
        return wrapper


def monitor_function(func: Callable) -> Callable:
    """Decorator for automatic function monitoring."""
    middleware = MonitoringMiddleware(get_monitor())
    return middleware(func)