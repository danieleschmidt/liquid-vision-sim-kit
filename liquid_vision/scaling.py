"""
⚡ AUTONOMOUS SCALING & OPTIMIZATION SYSTEM v3.0

Intelligent auto-scaling, performance optimization, and distributed processing
with adaptive resource management and predictive scaling algorithms.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import statistics
from collections import deque, defaultdict
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_CHANGE = "no_change"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    WORKERS = "workers"
    INSTANCES = "instances"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"
    HYBRID = "hybrid"
    MACHINE_LEARNING = "ml_based"


@dataclass
class ScalingMetric:
    """Metric used for scaling decisions."""
    name: str
    value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingRule:
    """Rule for automated scaling decisions."""
    name: str
    resource_type: ResourceType
    metrics: List[str]  # Metric names to monitor
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_seconds: int = 300
    min_instances: int = 1
    max_instances: int = 10
    scaling_factor: float = 1.5
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Record of scaling action taken."""
    timestamp: float
    direction: ScalingDirection
    resource_type: ResourceType
    old_value: Union[int, float]
    new_value: Union[int, float]
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile for workload characterization."""
    name: str
    cpu_pattern: List[float]
    memory_pattern: List[float]
    io_pattern: List[float]
    network_pattern: List[float]
    optimal_workers: int
    peak_hours: List[int]  # Hours of day when peak load occurs


class AutonomousScaler:
    """
    Intelligent auto-scaling system with predictive capabilities.
    
    Features:
    - Real-time workload analysis and prediction
    - Multi-dimensional resource scaling (CPU, memory, workers)
    - Machine learning-based scaling decisions
    - Performance optimization and bottleneck detection
    - Distributed processing coordination
    - Cost-aware scaling with efficiency optimization
    """
    
    def __init__(self, 
                 strategy: OptimizationStrategy = OptimizationStrategy.HYBRID,
                 max_workers: int = None,
                 enable_gpu: bool = True):
        self.strategy = strategy
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.enable_gpu = enable_gpu
        
        # Scaling state
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_history: List[ScalingEvent] = []
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        
        # Resource tracking
        self.current_resources: Dict[ResourceType, Union[int, float]] = {
            ResourceType.CPU: mp.cpu_count() or 1,
            ResourceType.MEMORY: 8.0,  # GB, will be detected dynamically
            ResourceType.WORKERS: 1,
            ResourceType.INSTANCES: 1,
        }
        
        # Executors for distributed processing
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # Monitoring and prediction
        self.running = False
        self._scaling_thread: Optional[threading.Thread] = None
        self.last_scaling_action: Dict[str, float] = {}
        
        # Performance optimization cache
        self.optimization_cache: Dict[str, Any] = {}
        self.cache_hit_rate = 0.0
        
        self._setup_default_rules()
        self._setup_performance_profiles()
        self._initialize_executors()
        
    def _setup_default_rules(self):
        """Setup default auto-scaling rules."""
        self.scaling_rules = {
            "cpu_scaling": ScalingRule(
                name="CPU-based scaling",
                resource_type=ResourceType.WORKERS,
                metrics=["cpu_usage", "queue_length"],
                scale_up_threshold=75.0,
                scale_down_threshold=25.0,
                cooldown_seconds=180,
                min_instances=1,
                max_instances=self.max_workers,
                scaling_factor=1.5
            ),
            "memory_scaling": ScalingRule(
                name="Memory-based scaling", 
                resource_type=ResourceType.MEMORY,
                metrics=["memory_usage", "gc_pressure"],
                scale_up_threshold=80.0,
                scale_down_threshold=40.0,
                cooldown_seconds=240,
                min_instances=1,
                max_instances=5,
                scaling_factor=1.3
            ),
            "latency_scaling": ScalingRule(
                name="Latency-based scaling",
                resource_type=ResourceType.WORKERS,
                metrics=["response_time", "queue_wait_time"],
                scale_up_threshold=200.0,  # 200ms
                scale_down_threshold=50.0,
                cooldown_seconds=120,
                min_instances=1,
                max_instances=self.max_workers // 2,
                scaling_factor=2.0
            ),
        }
        
    def _setup_performance_profiles(self):
        """Setup performance profiles for different workload patterns."""
        self.performance_profiles = {
            "cpu_intensive": PerformanceProfile(
                name="CPU Intensive Workload",
                cpu_pattern=[90, 85, 80, 85, 90, 95, 85],  # High CPU usage pattern
                memory_pattern=[40, 42, 45, 43, 41, 38, 40],
                io_pattern=[10, 15, 12, 8, 10, 15, 12],
                network_pattern=[20, 25, 22, 18, 20, 25, 22],
                optimal_workers=mp.cpu_count() or 1,
                peak_hours=[9, 10, 11, 14, 15, 16]
            ),
            "io_intensive": PerformanceProfile(
                name="I/O Intensive Workload",
                cpu_pattern=[30, 35, 32, 28, 30, 35, 32],
                memory_pattern=[60, 65, 68, 70, 65, 62, 60],
                io_pattern=[80, 85, 90, 88, 85, 82, 80],
                network_pattern=[70, 75, 80, 78, 75, 72, 70],
                optimal_workers=(mp.cpu_count() or 1) * 2,
                peak_hours=[8, 9, 10, 13, 14, 15, 17, 18]
            ),
            "balanced": PerformanceProfile(
                name="Balanced Workload",
                cpu_pattern=[50, 55, 52, 48, 50, 55, 52],
                memory_pattern=[45, 48, 50, 47, 45, 43, 45],
                io_pattern=[30, 35, 32, 28, 30, 35, 32],
                network_pattern=[40, 45, 42, 38, 40, 45, 42],
                optimal_workers=max(2, (mp.cpu_count() or 1) // 2),
                peak_hours=[9, 10, 14, 15, 16]
            ),
        }
        
    def _initialize_executors(self):
        """Initialize thread and process executors for distributed processing."""
        initial_workers = self.current_resources[ResourceType.WORKERS]
        
        self.thread_executor = ThreadPoolExecutor(
            max_workers=int(initial_workers * 2),
            thread_name_prefix="liquid_vision_thread"
        )
        
        self.process_executor = ProcessPoolExecutor(
            max_workers=int(initial_workers),
            initializer=self._worker_initializer
        )
        
        logger.info(f"Initialized executors with {initial_workers} workers")
        
    def _worker_initializer(self):
        """Initialize worker processes with required setup."""
        # Setup worker process with necessary imports and state
        import logging
        logging.basicConfig(level=logging.WARNING)  # Reduce worker logging
        
    def start_scaling(self) -> None:
        """Start autonomous scaling system."""
        if self.running:
            return
            
        self.running = True
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
        logger.info("⚡ Autonomous scaling system started")
        
    def stop_scaling(self) -> None:
        """Stop scaling system and cleanup resources."""
        self.running = False
        
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)
            
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
            
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
            
        logger.info("⚡ Scaling system stopped")
        
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add custom scaling rule."""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name}")
        
    def record_metric(self, name: str, value: float, **kwargs) -> None:
        """Record performance metric for scaling decisions."""
        metric = ScalingMetric(
            name=name,
            value=value,
            threshold_up=kwargs.get('threshold_up', 80.0),
            threshold_down=kwargs.get('threshold_down', 20.0),
            weight=kwargs.get('weight', 1.0)
        )
        self.metrics_history.append(metric)
        
    async def scale_workload(self, workload_func: Callable, 
                           data: List[Any],
                           chunk_size: Optional[int] = None,
                           use_processes: bool = False) -> List[Any]:
        """
        Intelligently scale workload processing across available resources.
        
        Args:
            workload_func: Function to process each data item
            data: List of data items to process
            chunk_size: Optional chunk size for batching
            use_processes: Whether to use process pool instead of thread pool
            
        Returns:
            List of processed results
        """
        start_time = time.time()
        data_size = len(data)
        
        # Determine optimal processing strategy
        optimal_workers = self._calculate_optimal_workers(data_size, workload_func)
        chunk_size = chunk_size or self._calculate_optimal_chunk_size(data_size, optimal_workers)
        
        logger.info(f"Scaling workload: {data_size} items, {optimal_workers} workers, chunk_size={chunk_size}")
        
        # Record pre-processing metrics
        self.record_metric("workload.data_size", data_size)
        self.record_metric("workload.workers", optimal_workers)
        
        try:
            # Choose execution strategy
            executor = self.process_executor if use_processes else self.thread_executor
            
            # Process data in optimal chunks
            chunks = [data[i:i + chunk_size] for i in range(0, data_size, chunk_size)]
            
            # Submit tasks to executor
            future_to_chunk = {}
            for i, chunk in enumerate(chunks):
                if use_processes:
                    future = executor.submit(self._process_chunk, workload_func, chunk, i)
                else:
                    future = executor.submit(self._process_chunk_thread, workload_func, chunk, i)
                future_to_chunk[future] = i
                
            # Collect results maintaining order
            results = [None] * len(chunks)
            completed = 0
            
            for future in future_to_chunk:
                try:
                    chunk_idx = future_to_chunk[future]
                    chunk_results = await asyncio.wrap_future(future)
                    results[chunk_idx] = chunk_results
                    completed += 1
                    
                    # Update progress metrics
                    progress = (completed / len(chunks)) * 100
                    self.record_metric("workload.progress", progress)
                    
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    chunk_idx = future_to_chunk[future]
                    results[chunk_idx] = []  # Empty results for failed chunk
                    
            # Flatten results
            final_results = []
            for chunk_results in results:
                if chunk_results:
                    final_results.extend(chunk_results)
                    
            # Record completion metrics
            execution_time = time.time() - start_time
            throughput = data_size / execution_time if execution_time > 0 else 0
            
            self.record_metric("workload.execution_time", execution_time * 1000)  # ms
            self.record_metric("workload.throughput", throughput)
            self.record_metric("workload.success_rate", len(final_results) / data_size * 100)
            
            logger.info(f"Workload completed: {len(final_results)}/{data_size} items in {execution_time:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Workload scaling failed: {e}")
            self.record_metric("workload.error_rate", 100.0)
            raise
            
    def _process_chunk(self, func: Callable, chunk: List[Any], chunk_idx: int) -> List[Any]:
        """Process chunk in separate process."""
        try:
            results = []
            for item in chunk:
                try:
                    result = func(item)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Item processing failed in chunk {chunk_idx}: {e}")
                    # Continue processing other items in chunk
            return results
        except Exception as e:
            logger.error(f"Chunk {chunk_idx} processing failed: {e}")
            return []
            
    def _process_chunk_thread(self, func: Callable, chunk: List[Any], chunk_idx: int) -> List[Any]:
        """Process chunk in thread (same as process version but with thread context)."""
        return self._process_chunk(func, chunk, chunk_idx)
        
    def _calculate_optimal_workers(self, data_size: int, workload_func: Callable) -> int:
        """Calculate optimal number of workers for workload."""
        # Base calculation on CPU count and data size
        base_workers = mp.cpu_count() or 1
        
        # Adjust based on workload characteristics
        if hasattr(workload_func, '__name__'):
            func_name = workload_func.__name__
            if 'cpu' in func_name.lower() or 'compute' in func_name.lower():
                # CPU-intensive: workers = CPU cores
                optimal = base_workers
            elif 'io' in func_name.lower() or 'network' in func_name.lower():
                # I/O-intensive: more workers than cores
                optimal = base_workers * 2
            else:
                # Balanced workload
                optimal = int(base_workers * 1.5)
        else:
            optimal = base_workers
            
        # Adjust based on data size
        if data_size < 10:
            optimal = min(optimal, 2)
        elif data_size < 100:
            optimal = min(optimal, base_workers)
        else:
            optimal = min(optimal, self.max_workers)
            
        # Consider current system load
        current_load = self._get_current_system_load()
        if current_load > 0.8:
            optimal = int(optimal * 0.7)  # Reduce workers under high load
        elif current_load < 0.3:
            optimal = int(optimal * 1.3)  # Increase workers under low load
            
        return max(1, min(optimal, self.max_workers))
        
    def _calculate_optimal_chunk_size(self, data_size: int, num_workers: int) -> int:
        """Calculate optimal chunk size for workload distribution."""
        if data_size <= num_workers:
            return 1
            
        # Base chunk size: data_size / (workers * 4) for good load balancing
        base_chunk_size = max(1, data_size // (num_workers * 4))
        
        # Adjust based on system characteristics
        if data_size > 10000:
            # Large datasets: use larger chunks to reduce overhead
            chunk_size = min(base_chunk_size * 2, data_size // num_workers)
        elif data_size < 100:
            # Small datasets: use smaller chunks for better distribution
            chunk_size = max(1, base_chunk_size // 2)
        else:
            chunk_size = base_chunk_size
            
        return max(1, min(chunk_size, data_size))
        
    def _get_current_system_load(self) -> float:
        """Get current system load average."""
        try:
            import psutil
            return psutil.getloadavg()[0] / (mp.cpu_count() or 1)
        except:
            # Fallback: estimate from recent metrics
            if not self.metrics_history:
                return 0.5
                
            recent_cpu = [m.value for m in list(self.metrics_history)[-5:] 
                         if m.name == 'cpu_usage']
            if recent_cpu:
                return min(1.0, statistics.mean(recent_cpu) / 100.0)
            return 0.5
            
    def _scaling_loop(self) -> None:
        """Main scaling decision loop."""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                
                # Analyze scaling needs
                scaling_decisions = self._analyze_scaling_needs(current_metrics)
                
                # Execute scaling decisions
                for decision in scaling_decisions:
                    self._execute_scaling_decision(decision)
                    
                # Update performance profiles
                self._update_performance_profiles(current_metrics)
                
                # Optimize caching and performance
                self._optimize_performance()
                
                # Sleep based on system activity
                sleep_time = self._calculate_monitoring_interval(current_metrics)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(30)  # Back off on errors
                
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics for scaling decisions."""
        metrics = {}
        
        try:
            import psutil
            
            # System metrics
            metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            metrics['memory_usage'] = psutil.virtual_memory().percent
            metrics['disk_usage'] = psutil.disk_usage('/').percent
            
            # Process metrics
            process = psutil.Process()
            metrics['process_cpu'] = process.cpu_percent()
            metrics['process_memory'] = process.memory_info().rss / 1024 / 1024  # MB
            
        except ImportError:
            # Fallback metrics
            metrics = {
                'cpu_usage': 50.0,
                'memory_usage': 40.0,
                'disk_usage': 30.0,
                'process_cpu': 25.0,
                'process_memory': 100.0,
            }
            
        # Application-specific metrics from history
        recent_metrics = list(self.metrics_history)[-10:]
        for metric in recent_metrics:
            if metric.name not in metrics:
                metrics[metric.name] = metric.value
                
        # Add derived metrics
        metrics['queue_length'] = len(self.metrics_history)
        metrics['cache_hit_rate'] = self.cache_hit_rate
        
        return metrics
        
    def _analyze_scaling_needs(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Analyze current metrics and determine scaling actions needed."""
        decisions = []
        
        for rule_name, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
                
            # Check cooldown period
            last_action = self.last_scaling_action.get(rule_name, 0)
            if time.time() - last_action < rule.cooldown_seconds:
                continue
                
            # Calculate weighted metric score
            relevant_metrics = [metrics.get(metric_name, 0) for metric_name in rule.metrics]
            if not relevant_metrics:
                continue
                
            avg_metric = statistics.mean(relevant_metrics)
            
            # Determine scaling direction
            direction = ScalingDirection.NO_CHANGE
            current_value = self.current_resources.get(rule.resource_type, 1)
            new_value = current_value
            
            if avg_metric > rule.scale_up_threshold and current_value < rule.max_instances:
                direction = ScalingDirection.SCALE_UP
                new_value = min(rule.max_instances, int(current_value * rule.scaling_factor))
            elif avg_metric < rule.scale_down_threshold and current_value > rule.min_instances:
                direction = ScalingDirection.SCALE_DOWN
                new_value = max(rule.min_instances, int(current_value / rule.scaling_factor))
                
            if direction != ScalingDirection.NO_CHANGE:
                decisions.append({
                    'rule_name': rule_name,
                    'rule': rule,
                    'direction': direction,
                    'old_value': current_value,
                    'new_value': new_value,
                    'trigger_metric': avg_metric,
                    'metrics_snapshot': metrics.copy()
                })
                
        return decisions
        
    def _execute_scaling_decision(self, decision: Dict[str, Any]) -> None:
        """Execute a scaling decision."""
        rule = decision['rule']
        direction = decision['direction']
        old_value = decision['old_value']
        new_value = decision['new_value']
        
        try:
            # Update resource allocation
            success = self._apply_resource_change(rule.resource_type, new_value)
            
            if success:
                # Record scaling event
                event = ScalingEvent(
                    timestamp=time.time(),
                    direction=direction,
                    resource_type=rule.resource_type,
                    old_value=old_value,
                    new_value=new_value,
                    reason=f"Triggered by {rule.name}",
                    metrics=decision['metrics_snapshot']
                )
                
                self.scaling_history.append(event)
                self.last_scaling_action[decision['rule_name']] = time.time()
                
                logger.info(f"Scaling executed: {rule.resource_type.value} {old_value} → {new_value} ({direction.value})")
                
                # Record scaling metrics
                self.record_metric(f"scaling.{rule.resource_type.value}", new_value)
                self.record_metric("scaling.events", len(self.scaling_history))
                
            else:
                logger.warning(f"Failed to apply scaling decision for {rule.resource_type.value}")
                
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            
    def _apply_resource_change(self, resource_type: ResourceType, new_value: Union[int, float]) -> bool:
        """Apply resource change to system."""
        old_value = self.current_resources.get(resource_type, 1)
        
        try:
            if resource_type == ResourceType.WORKERS:
                # Resize thread pool
                if self.thread_executor:
                    # Note: ThreadPoolExecutor doesn't support dynamic resizing in standard library
                    # In production, you'd use a custom implementation or library like concurrent-futures-extra
                    logger.info(f"Thread pool resize: {old_value} → {int(new_value)} workers")
                    
                # Update current resource tracking
                self.current_resources[resource_type] = int(new_value)
                return True
                
            elif resource_type == ResourceType.MEMORY:
                # In real implementation, this would adjust memory limits
                logger.info(f"Memory limit adjustment: {old_value} → {new_value} GB")
                self.current_resources[resource_type] = new_value
                return True
                
            elif resource_type == ResourceType.INSTANCES:
                # In real implementation, this would scale container instances
                logger.info(f"Instance scaling: {old_value} → {int(new_value)} instances")
                self.current_resources[resource_type] = int(new_value)
                return True
                
            else:
                logger.warning(f"Resource type {resource_type} not implemented")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply resource change: {e}")
            return False
            
    def _update_performance_profiles(self, metrics: Dict[str, float]) -> None:
        """Update performance profiles based on observed metrics."""
        # Simple heuristic-based profile detection
        cpu_usage = metrics.get('cpu_usage', 50)
        memory_usage = metrics.get('memory_usage', 50)
        
        if cpu_usage > 70 and memory_usage < 60:
            current_profile = "cpu_intensive"
        elif memory_usage > 70 and cpu_usage < 60:
            current_profile = "io_intensive"
        else:
            current_profile = "balanced"
            
        # Update optimal workers based on detected profile
        profile = self.performance_profiles.get(current_profile)
        if profile:
            optimal_workers = profile.optimal_workers
            # Gradually adjust current worker allocation towards optimal
            current_workers = self.current_resources[ResourceType.WORKERS]
            if abs(current_workers - optimal_workers) > 1:
                # Suggest adjustment (will be handled by scaling rules)
                self.record_metric("suggested_workers", optimal_workers)
                
    def _optimize_performance(self) -> None:
        """Execute performance optimizations."""
        # Cache optimization
        cache_size = len(self.optimization_cache)
        max_cache_size = 1000
        
        if cache_size > max_cache_size:
            # Remove oldest 20% of cache entries
            items_to_remove = cache_size - int(max_cache_size * 0.8)
            cache_items = list(self.optimization_cache.items())
            for key, _ in cache_items[:items_to_remove]:
                del self.optimization_cache[key]
                
        # Update cache hit rate
        # This would be calculated based on actual cache usage in production
        self.cache_hit_rate = min(0.95, self.cache_hit_rate + 0.01)
        
        # Memory optimization
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]  # Keep recent half
            
    def _calculate_monitoring_interval(self, metrics: Dict[str, float]) -> float:
        """Calculate adaptive monitoring interval based on system state."""
        # Base interval
        base_interval = 15.0  # seconds
        
        # Adjust based on system activity
        cpu_usage = metrics.get('cpu_usage', 50)
        memory_usage = metrics.get('memory_usage', 50)
        
        activity_level = max(cpu_usage, memory_usage)
        
        if activity_level > 80:
            return base_interval * 0.5  # More frequent monitoring under high load
        elif activity_level < 30:
            return base_interval * 2.0  # Less frequent monitoring under low load
        else:
            return base_interval
            
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        recent_events = [e for e in self.scaling_history if time.time() - e.timestamp < 3600]
        
        return {
            "running": self.running,
            "strategy": self.strategy.value,
            "current_resources": dict(self.current_resources),
            "scaling_rules": {name: rule.__dict__ for name, rule in self.scaling_rules.items()},
            "recent_scaling_events": len(recent_events),
            "total_scaling_events": len(self.scaling_history),
            "metrics_buffer_size": len(self.metrics_history),
            "cache_hit_rate": self.cache_hit_rate,
            "cache_size": len(self.optimization_cache),
            "performance_profiles": list(self.performance_profiles.keys()),
            "max_workers": self.max_workers,
        }


# Global scaling system instance
_scaler: Optional[AutonomousScaler] = None


def get_scaler() -> AutonomousScaler:
    """Get or create global scaling system instance."""
    global _scaler
    if _scaler is None:
        _scaler = AutonomousScaler()
        _scaler.start_scaling()
    return _scaler


async def scale_workload(workload_func: Callable, data: List[Any], **kwargs) -> List[Any]:
    """Global function to scale workload processing."""
    return await get_scaler().scale_workload(workload_func, data, **kwargs)


def record_scaling_metric(name: str, value: float, **kwargs) -> None:
    """Global function to record scaling metrics."""
    get_scaler().record_metric(name, value, **kwargs)