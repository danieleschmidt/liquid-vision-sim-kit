"""
Comprehensive monitoring and metrics system for liquid neural networks.
Tracks performance, resource usage, model behavior, and system health.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
import time
import threading
import queue
import psutil
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import warnings

from .logging import get_logger


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: Union[float, int]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    power_consumption_watts: Optional[float] = None
    temperature_celsius: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'inference_time_ms': self.inference_time_ms,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'cpu_utilization_percent': self.cpu_utilization_percent,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'power_consumption_watts': self.power_consumption_watts,
            'temperature_celsius': self.temperature_celsius
        }


@dataclass
class ModelMetrics:
    """Model-specific metrics."""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    gradient_norm: Optional[float] = None
    parameter_norm: Optional[float] = None
    activation_mean: Optional[float] = None
    activation_std: Optional[float] = None
    liquid_state_stability: Optional[float] = None
    temporal_consistency: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricsCollector:
    """Collects and stores metrics from various sources."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.custom_collectors: List[Callable[[], Dict[str, float]]] = []
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def record_metric(self, name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None, **metadata):
        """Record a single metric value."""
        with self.lock:
            metric_point = MetricPoint(
                name=name,
                value=float(value),
                timestamp=time.time(),
                tags=tags or {},
                metadata=metadata
            )
            self.metrics_history[name].append(metric_point)
    
    def record_metrics_batch(self, metrics: Dict[str, Union[float, int]], tags: Optional[Dict[str, str]] = None):
        """Record multiple metrics at once."""
        timestamp = time.time()
        with self.lock:
            for name, value in metrics.items():
                metric_point = MetricPoint(
                    name=name,
                    value=float(value),
                    timestamp=timestamp,
                    tags=tags or {}
                )
                self.metrics_history[name].append(metric_point)
    
    def add_custom_collector(self, collector: Callable[[], Dict[str, float]]):
        """Add custom metrics collector function."""
        self.custom_collectors.append(collector)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        metrics = {}
        
        # Collect from custom collectors
        for collector in self.custom_collectors:
            try:
                custom_metrics = collector()
                metrics.update(custom_metrics)
            except Exception as e:
                self.logger.warning(f"Custom collector failed: {e}")
        
        return metrics
    
    def get_metric_history(self, name: str, duration_seconds: Optional[float] = None) -> List[MetricPoint]:
        """Get metric history for specified duration."""
        with self.lock:
            if name not in self.metrics_history:
                return []
            
            history = list(self.metrics_history[name])
            
            if duration_seconds is not None:
                cutoff_time = time.time() - duration_seconds
                history = [point for point in history if point.timestamp >= cutoff_time]
            
            return history
    
    def get_metric_stats(self, name: str, duration_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistical summary of metric."""
        history = self.get_metric_history(name, duration_seconds)
        
        if not history:
            return {}
        
        values = [point.value for point in history]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def get_all_metrics_summary(self, duration_seconds: float = 300) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        with self.lock:
            for metric_name in self.metrics_history.keys():
                stats = self.get_metric_stats(metric_name, duration_seconds)
                if stats:
                    summary[metric_name] = stats
        
        return summary
    
    def clear_history(self, metric_name: Optional[str] = None):
        """Clear metrics history."""
        with self.lock:
            if metric_name:
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].clear()
            else:
                for history in self.metrics_history.values():
                    history.clear()


class SystemMonitor:
    """Monitor system resources and health."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_collector = MetricsCollector()
        self.logger = get_logger(__name__)
        
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.metrics_collector.record_metrics_batch(system_metrics, tags={'source': 'system'})
                
                # Collect GPU metrics if available
                if torch.cuda.is_available():
                    gpu_metrics = self._collect_gpu_metrics()
                    self.metrics_collector.record_metrics_batch(gpu_metrics, tags={'source': 'gpu'})
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network metrics (if available)
            network_stats = psutil.net_io_counters()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_mb': memory_used_mb,
                'memory_available_mb': memory_available_mb,
                'disk_percent': disk_percent,
                'network_bytes_sent': network_stats.bytes_sent,
                'network_bytes_recv': network_stats.bytes_recv
            }
            
            # Temperature (if available)
            try:
                temperatures = psutil.sensors_temperatures()
                if temperatures:
                    # Get CPU temperature
                    for name, entries in temperatures.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            if entries:
                                metrics['cpu_temperature_c'] = entries[0].current
                                break
            except (AttributeError, OSError):
                pass  # Temperature sensors not available
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
            return {}
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU metrics."""
        try:
            if not torch.cuda.is_available():
                return {}
            
            metrics = {}
            
            for i in range(torch.cuda.device_count()):
                device = f"cuda:{i}"
                
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
                memory_cached = torch.cuda.memory_reserved(i) / (1024 * 1024)  # MB
                
                metrics[f'gpu_{i}_memory_allocated_mb'] = memory_allocated
                metrics[f'gpu_{i}_memory_cached_mb'] = memory_cached
                
                # GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f'gpu_{i}_utilization_percent'] = utilization.gpu
                    metrics[f'gpu_{i}_memory_utilization_percent'] = utilization.memory
                    
                    # Temperature
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics[f'gpu_{i}_temperature_c'] = temperature
                    
                    # Power consumption
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        metrics[f'gpu_{i}_power_watts'] = power
                    except pynvml.NVMLError:
                        pass  # Power monitoring not supported
                    
                    pynvml.nvmlShutdown()
                    
                except ImportError:
                    pass  # pynvml not available
                except Exception as e:
                    self.logger.warning(f"Failed to get NVML metrics for GPU {i}: {e}")
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to collect GPU metrics: {e}")
            return {}
    
    def get_current_performance(self) -> PerformanceMetrics:
        """Get current performance snapshot."""
        system_metrics = self._collect_system_metrics()
        gpu_metrics = self._collect_gpu_metrics() if torch.cuda.is_available() else {}
        
        return PerformanceMetrics(
            memory_usage_mb=system_metrics.get('memory_used_mb', 0),
            cpu_utilization_percent=system_metrics.get('cpu_percent', 0),
            gpu_memory_mb=sum(v for k, v in gpu_metrics.items() if 'memory_allocated_mb' in k),
            gpu_utilization_percent=np.mean([v for k, v in gpu_metrics.items() if 'utilization_percent' in k]) or 0,
            power_consumption_watts=sum(v for k, v in gpu_metrics.items() if 'power_watts' in k) or None,
            temperature_celsius=system_metrics.get('cpu_temperature_c') or 
                               np.mean([v for k, v in gpu_metrics.items() if 'temperature_c' in k]) or None
        )


class ModelMonitor:
    """Monitor model-specific metrics during training and inference."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.hooks = []
        self.logger = get_logger(__name__)
    
    def attach_to_model(self, model: torch.nn.Module):
        """Attach monitoring hooks to model."""
        self.detach_from_model()  # Remove existing hooks
        
        # Register forward hooks for activation monitoring
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name: self._activation_hook(name, output)
                )
                self.hooks.append(hook)
        
        # Register backward hooks for gradient monitoring
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(name, grad)
                )
                self.hooks.append(hook)
        
        self.logger.info(f"Attached {len(self.hooks)} monitoring hooks to model")
    
    def detach_from_model(self):
        """Remove all monitoring hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _activation_hook(self, layer_name: str, activation: torch.Tensor):
        """Hook for monitoring layer activations."""
        try:
            with torch.no_grad():
                activation_flat = activation.view(-1)
                
                # Compute statistics
                mean_val = activation_flat.mean().item()
                std_val = activation_flat.std().item()
                abs_mean = activation_flat.abs().mean().item()
                
                # Check for numerical issues
                nan_count = torch.isnan(activation_flat).sum().item()
                inf_count = torch.isinf(activation_flat).sum().item()
                
                self.metrics_collector.record_metrics_batch({
                    f'activation_{layer_name}_mean': mean_val,
                    f'activation_{layer_name}_std': std_val,
                    f'activation_{layer_name}_abs_mean': abs_mean,
                    f'activation_{layer_name}_nan_count': nan_count,
                    f'activation_{layer_name}_inf_count': inf_count
                })
                
                # Alert if numerical issues detected
                if nan_count > 0 or inf_count > 0:
                    self.logger.warning(f"Numerical issues in {layer_name}: {nan_count} NaNs, {inf_count} Infs")
                    
        except Exception as e:
            self.logger.warning(f"Error in activation hook for {layer_name}: {e}")
    
    def _gradient_hook(self, param_name: str, gradient: torch.Tensor):
        """Hook for monitoring parameter gradients."""
        try:
            with torch.no_grad():
                grad_flat = gradient.view(-1)
                
                # Compute gradient statistics
                grad_norm = grad_flat.norm().item()
                grad_mean = grad_flat.mean().item()
                grad_std = grad_flat.std().item()
                
                # Check for numerical issues
                nan_count = torch.isnan(grad_flat).sum().item()
                inf_count = torch.isinf(grad_flat).sum().item()
                
                self.metrics_collector.record_metrics_batch({
                    f'gradient_{param_name}_norm': grad_norm,
                    f'gradient_{param_name}_mean': grad_mean,
                    f'gradient_{param_name}_std': grad_std,
                    f'gradient_{param_name}_nan_count': nan_count,
                    f'gradient_{param_name}_inf_count': inf_count
                })
                
                # Alert for gradient issues
                if nan_count > 0 or inf_count > 0:
                    self.logger.warning(f"Gradient issues in {param_name}: {nan_count} NaNs, {inf_count} Infs")
                elif grad_norm > 1000:
                    self.logger.warning(f"Large gradient norm in {param_name}: {grad_norm}")
                elif grad_norm < 1e-8:
                    self.logger.warning(f"Vanishing gradient in {param_name}: {grad_norm}")
                    
        except Exception as e:
            self.logger.warning(f"Error in gradient hook for {param_name}: {e}")
    
    def record_model_metrics(self, model: torch.nn.Module, loss: Optional[torch.Tensor] = None) -> ModelMetrics:
        """Record comprehensive model metrics."""
        metrics = ModelMetrics()
        
        try:
            with torch.no_grad():
                # Parameter statistics
                all_params = torch.cat([p.view(-1) for p in model.parameters()])
                metrics.parameter_norm = all_params.norm().item()
                
                # Gradient statistics (if available)
                grad_params = [p for p in model.parameters() if p.grad is not None]
                if grad_params:
                    all_grads = torch.cat([p.grad.view(-1) for p in grad_params])
                    metrics.gradient_norm = all_grads.norm().item()
                
                # Loss value
                if loss is not None:
                    metrics.loss = loss.item()
                
                # Record to collector
                metrics_dict = metrics.to_dict()
                self.metrics_collector.record_metrics_batch(
                    metrics_dict, tags={'source': 'model'}
                )
                
        except Exception as e:
            self.logger.warning(f"Error recording model metrics: {e}")
        
        return metrics
    
    def check_model_health(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Comprehensive model health check."""
        health_report = {
            'healthy': True,
            'issues': [],
            'warnings': [],
            'parameter_stats': {},
            'gradient_stats': {}
        }
        
        try:
            with torch.no_grad():
                # Check parameters
                for name, param in model.named_parameters():
                    param_flat = param.view(-1)
                    
                    # NaN/Inf check
                    nan_count = torch.isnan(param_flat).sum().item()
                    inf_count = torch.isinf(param_flat).sum().item()
                    
                    if nan_count > 0:
                        health_report['healthy'] = False
                        health_report['issues'].append(f"Parameter {name} has {nan_count} NaN values")
                    
                    if inf_count > 0:
                        health_report['healthy'] = False
                        health_report['issues'].append(f"Parameter {name} has {inf_count} infinite values")
                    
                    # Parameter norm
                    param_norm = param_flat.norm().item()
                    health_report['parameter_stats'][name] = {
                        'norm': param_norm,
                        'mean': param_flat.mean().item(),
                        'std': param_flat.std().item(),
                        'nan_count': nan_count,
                        'inf_count': inf_count
                    }
                    
                    # Large parameter warning
                    if param_norm > 1000:
                        health_report['warnings'].append(f"Large parameter norm in {name}: {param_norm}")
                    
                    # Check gradients
                    if param.grad is not None:
                        grad_flat = param.grad.view(-1)
                        grad_norm = grad_flat.norm().item()
                        
                        health_report['gradient_stats'][name] = {
                            'norm': grad_norm,
                            'mean': grad_flat.mean().item(),
                            'std': grad_flat.std().item()
                        }
                        
                        if grad_norm > 1000:
                            health_report['warnings'].append(f"Large gradient norm in {name}: {grad_norm}")
                        elif grad_norm < 1e-8:
                            health_report['warnings'].append(f"Vanishing gradient in {name}: {grad_norm}")
                
        except Exception as e:
            health_report['healthy'] = False
            health_report['issues'].append(f"Error during health check: {str(e)}")
        
        return health_report


@contextmanager
def performance_monitor(operation_name: str, metrics_collector: Optional[MetricsCollector] = None):
    """Context manager for monitoring operation performance."""
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    
    start_time = time.time()
    start_memory = 0
    start_gpu_memory = 0
    
    # Get initial memory usage
    try:
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    except:
        pass
    
    if torch.cuda.is_available():
        start_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    
    try:
        yield metrics_collector
        
    finally:
        # Calculate performance metrics
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        end_memory = start_memory
        try:
            process = psutil.Process()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except:
            pass
        
        end_gpu_memory = start_gpu_memory
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        # Record metrics
        metrics_collector.record_metrics_batch({
            f'{operation_name}_duration_ms': duration_ms,
            f'{operation_name}_memory_delta_mb': end_memory - start_memory,
            f'{operation_name}_gpu_memory_delta_mb': end_gpu_memory - start_gpu_memory
        }, tags={'operation': operation_name})


class MetricsDashboard:
    """Simple metrics dashboard for monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = get_logger(__name__)
    
    def generate_report(self, duration_minutes: int = 5) -> str:
        """Generate text-based metrics report."""
        duration_seconds = duration_minutes * 60
        summary = self.metrics_collector.get_all_metrics_summary(duration_seconds)
        
        report = f"\n{'='*60}\n"
        report += f"METRICS REPORT - Last {duration_minutes} minutes\n"
        report += f"{'='*60}\n\n"
        
        if not summary:
            report += "No metrics data available.\n"
            return report
        
        # Group metrics by category
        categories = {
            'Performance': ['inference_time_ms', 'throughput_samples_per_sec'],
            'System Resources': ['cpu_percent', 'memory_percent', 'memory_used_mb'],
            'GPU': [k for k in summary.keys() if 'gpu' in k],
            'Model': [k for k in summary.keys() if any(x in k for x in ['activation', 'gradient', 'parameter'])],
            'Other': []
        }
        
        # Classify metrics
        classified = set()
        for category, patterns in categories.items():
            if category == 'Other':
                continue
            for metric in summary.keys():
                if any(pattern in metric for pattern in patterns):
                    classified.add(metric)
        
        categories['Other'] = [k for k in summary.keys() if k not in classified]
        
        # Generate report by category
        for category, metric_names in categories.items():
            relevant_metrics = {k: v for k, v in summary.items() if k in metric_names}
            
            if not relevant_metrics:
                continue
                
            report += f"{category}:\n"
            report += "-" * len(category) + "\n"
            
            for metric_name, stats in relevant_metrics.items():
                report += f"  {metric_name}:\n"
                report += f"    Mean: {stats['mean']:.3f} | Std: {stats['std']:.3f}\n"
                report += f"    Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | P95: {stats['p95']:.3f}\n"
                report += f"    Count: {stats['count']}\n\n"
        
        return report
    
    def export_metrics(self, filepath: Union[str, Path], duration_minutes: int = 60):
        """Export metrics to JSON file."""
        duration_seconds = duration_minutes * 60
        summary = self.metrics_collector.get_all_metrics_summary(duration_seconds)
        
        export_data = {
            'timestamp': time.time(),
            'duration_seconds': duration_seconds,
            'summary': summary,
            'raw_data': {}
        }
        
        # Include raw data for key metrics
        key_metrics = ['inference_time_ms', 'memory_used_mb', 'cpu_percent']
        for metric in key_metrics:
            if metric in self.metrics_collector.metrics_history:
                history = self.metrics_collector.get_metric_history(metric, duration_seconds)
                export_data['raw_data'][metric] = [
                    {'timestamp': p.timestamp, 'value': p.value} for p in history
                ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")


def create_monitoring_suite() -> Dict[str, Any]:
    """Create complete monitoring suite."""
    system_monitor = SystemMonitor()
    model_monitor = ModelMonitor()
    metrics_collector = MetricsCollector()
    dashboard = MetricsDashboard(metrics_collector)
    
    # Start system monitoring
    system_monitor.start_monitoring()
    
    return {
        'system_monitor': system_monitor,
        'model_monitor': model_monitor,
        'metrics_collector': metrics_collector,
        'dashboard': dashboard
    }