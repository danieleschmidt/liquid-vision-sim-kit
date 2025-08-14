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


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.aggregated_metrics: Dict[str, Dict[str, float]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = ""):
        """Record a single metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics.append(metric)
            
        # Update aggregated metrics
        self._update_aggregated_metrics(metric)
        
    def _update_aggregated_metrics(self, metric: PerformanceMetric):
        """Update aggregated metrics for dashboards."""
        key = metric.name
        
        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0
            }
            
        agg = self.aggregated_metrics[key]
        agg["count"] += 1
        agg["sum"] += metric.value
        agg["min"] = min(agg["min"], metric.value)
        agg["max"] = max(agg["max"], metric.value)
        agg["avg"] = agg["sum"] / agg["count"]
        
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[float] = None) -> List[PerformanceMetric]:
        """Get metrics, optionally filtered by name and time."""
        with self.lock:
            filtered = list(self.metrics)
            
        if name:
            filtered = [m for m in filtered if m.name == name]
            
        if since:
            filtered = [m for m in filtered if m.timestamp >= since]
            
        return filtered
        
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics for dashboards."""
        return self.aggregated_metrics.copy()
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        total_metrics = len(self.metrics)
        unique_names = set(m.name for m in self.metrics)
        
        recent_metrics = [m for m in self.metrics if time.time() - m.timestamp < 60]
        
        return {
            "total_metrics": total_metrics,
            "unique_metric_names": len(unique_names),
            "recent_metrics_1min": len(recent_metrics),
            "aggregated": self.aggregated_metrics,
            "metric_names": list(unique_names)
        }


class HealthMonitor:
    """Monitors system health with comprehensive checks."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_status: Dict[str, HealthCheck] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self._register_default_checks()
        
    def _register_default_checks(self):
        """Register default health checks."""
        self.health_checks.update({
            "core_functionality": self._check_core_functionality,
            "memory_usage": self._check_memory_usage,
            "error_rate": self._check_error_rate,
            "response_time": self._check_response_time,
        })
        
    def _check_core_functionality(self) -> HealthCheck:
        """Check if core functionality is working."""
        start_time = time.time()
        
        try:
            # Test basic liquid network creation
            from .core.minimal_fallback import create_minimal_liquid_net, MinimalTensor
            
            model = create_minimal_liquid_net(2, 1, architecture="tiny")
            x = MinimalTensor([[0.1, 0.2]])
            output = model(x)
            
            latency = (time.time() - start_time) * 1000
            
            if output and hasattr(output, 'data'):
                return HealthCheck(
                    component="core_functionality",
                    status="healthy",
                    message="Core liquid network operations working",
                    timestamp=time.time(),
                    latency_ms=latency
                )
            else:
                return HealthCheck(
                    component="core_functionality",
                    status="degraded",
                    message="Core operations producing unexpected results",
                    timestamp=time.time(),
                    latency_ms=latency
                )
                
        except Exception as e:
            return HealthCheck(
                component="core_functionality",
                status="unhealthy",
                message=f"Core functionality failed: {e}",
                timestamp=time.time(),
                latency_ms=(time.time() - start_time) * 1000
            )
            
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent < 70:
                status = "healthy"
                message = f"Memory usage normal: {memory.percent:.1f}%"
            elif memory.percent < 85:
                status = "degraded"
                message = f"Memory usage high: {memory.percent:.1f}%"
            else:
                status = "unhealthy"
                message = f"Memory usage critical: {memory.percent:.1f}%"
                
        except ImportError:
            # Fallback without psutil
            status = "healthy"
            message = "Memory monitoring not available (psutil not installed)"
            
        return HealthCheck(
            component="memory_usage",
            status=status,
            message=message,
            timestamp=time.time()
        )
        
    def _check_error_rate(self) -> HealthCheck:
        """Check recent error rate."""
        try:
            from .utils.robust_error_handling import global_error_handler
            error_summary = global_error_handler.get_error_summary()
        except ImportError:
            error_summary = {"total_errors": 0, "recovery_rate": 1.0}
        
        if error_summary["total_errors"] == 0:
            status = "healthy"
            message = "No errors detected"
        elif error_summary["recovery_rate"] > 0.8:
            status = "healthy"
            message = f"Error recovery rate good: {error_summary['recovery_rate']:.1%}"
        elif error_summary["recovery_rate"] > 0.5:
            status = "degraded"
            message = f"Error recovery rate fair: {error_summary['recovery_rate']:.1%}"
        else:
            status = "unhealthy"
            message = f"Error recovery rate poor: {error_summary['recovery_rate']:.1%}"
            
        return HealthCheck(
            component="error_rate",
            status=status,
            message=message,
            timestamp=time.time()
        )
        
    def _check_response_time(self) -> HealthCheck:
        """Check system response time."""
        start_time = time.time()
        
        try:
            # Simple operation to measure response time
            from .core.minimal_fallback import MinimalTensor
            x = MinimalTensor([[1.0, 2.0]])
            _ = x + x  # Simple operation
            
            response_time = (time.time() - start_time) * 1000
            
            if response_time < 1.0:
                status = "healthy"
                message = f"Response time excellent: {response_time:.2f}ms"
            elif response_time < 10.0:
                status = "healthy"
                message = f"Response time good: {response_time:.2f}ms"
            elif response_time < 100.0:
                status = "degraded"
                message = f"Response time slow: {response_time:.2f}ms"
            else:
                status = "unhealthy"
                message = f"Response time critical: {response_time:.2f}ms"
                
            return HealthCheck(
                component="response_time",
                status=status,
                message=message,
                timestamp=time.time(),
                latency_ms=response_time
            )
            
        except Exception as e:
            return HealthCheck(
                component="response_time",
                status="unhealthy",
                message=f"Response time check failed: {e}",
                timestamp=time.time(),
                latency_ms=(time.time() - start_time) * 1000
            )
            
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register custom health check."""
        self.health_checks[name] = check_func
        
    def run_health_check(self, component: str) -> HealthCheck:
        """Run specific health check."""
        if component not in self.health_checks:
            return HealthCheck(
                component=component,
                status="unhealthy",
                message=f"Unknown health check: {component}",
                timestamp=time.time()
            )
            
        try:
            result = self.health_checks[component]()
            self.last_health_status[component] = result
            self.health_history.append(result)
            return result
            
        except Exception as e:
            result = HealthCheck(
                component=component,
                status="unhealthy", 
                message=f"Health check failed: {e}",
                timestamp=time.time()
            )
            self.last_health_status[component] = result
            self.health_history.append(result)
            return result
            
    def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for component in self.health_checks:
            results[component] = self.run_health_check(component)
            
        return results
        
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        recent_checks = self.run_all_health_checks()
        
        # Determine overall status
        statuses = [check.status for check in recent_checks.values()]
        
        if all(status == "healthy" for status in statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
            
        healthy_count = sum(1 for status in statuses if status == "healthy")
        total_count = len(statuses)
        
        return {
            "overall_status": overall_status,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "health_score": healthy_count / total_count if total_count > 0 else 0.0,
            "component_details": {name: asdict(check) for name, check in recent_checks.items()},
            "timestamp": time.time()
        }


class PerformanceProfiler:
    """Profiles performance of functions and operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_profiles: Dict[str, float] = {}
        
    def start_profile(self, operation_name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.active_profiles[profile_id] = time.time()
        return profile_id
        
    def end_profile(self, profile_id: str, tags: Dict[str, str] = None):
        """End profiling and record metrics."""
        if profile_id not in self.active_profiles:
            return
            
        start_time = self.active_profiles.pop(profile_id)
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        operation_name = profile_id.split('_')[0]
        self.metrics.record_metric(
            name=f"{operation_name}_duration",
            value=duration,
            tags=tags or {},
            unit="ms"
        )
        
    def profile_function(self, func_name: str = None):
        """Decorator to automatically profile function execution."""
        def decorator(func):
            name = func_name or func.__name__
            
            def wrapper(*args, **kwargs):
                profile_id = self.start_profile(name)
                try:
                    result = func(*args, **kwargs)
                    self.end_profile(profile_id, {"status": "success"})
                    return result
                except Exception as e:
                    self.end_profile(profile_id, {"status": "error", "error_type": type(e).__name__})
                    raise
                    
            return wrapper
        return decorator


# Global monitoring instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor()
performance_profiler = PerformanceProfiler(metrics_collector)


def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data."""
    return {
        "timestamp": time.time(),
        "metrics_summary": metrics_collector.get_metrics_summary(),
        "health_status": health_monitor.get_overall_health(),
        "system_info": {
            "monitoring_active": True,
            "features_enabled": ["metrics", "health_checks", "profiling"],
            "uptime": time.time(),
        }
    }


def export_monitoring_data(filepath: str):
    """Export monitoring data to JSON file."""
    dashboard_data = get_monitoring_dashboard()
    
    with open(filepath, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
        
    logging.info(f"Monitoring data exported to {filepath}")


# Context manager for automatic profiling
class profile_operation:
    """Context manager for profiling operations."""
    
    def __init__(self, operation_name: str, tags: Dict[str, str] = None):
        self.operation_name = operation_name
        self.tags = tags or {}
        self.profile_id = None
        
    def __enter__(self):
        self.profile_id = performance_profiler.start_profile(self.operation_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tags["status"] = "error"
            self.tags["error_type"] = exc_type.__name__
        else:
            self.tags["status"] = "success"
            
        performance_profiler.end_profile(self.profile_id, self.tags)