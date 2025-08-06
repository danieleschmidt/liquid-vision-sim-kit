"""
Auto-scaling and adaptive optimization for liquid neural networks.
Dynamic resource allocation, load balancing, and performance optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import psutil
import math

from ..core.liquid_neurons import LiquidNet
from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector, SystemMonitor
from ..utils.error_handling import robust_operation, ErrorCategory


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    INFERENCE_LATENCY = "inference_latency"
    THROUGHPUT = "throughput"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"


class ScalingAction(Enum):
    """Available scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    OPTIMIZE_MODEL = "optimize_model"
    ADJUST_BATCH_SIZE = "adjust_batch_size"
    REDISTRIBUTE_LOAD = "redistribute_load"
    NO_ACTION = "no_action"


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    metric: ScalingMetric
    threshold_up: float
    threshold_down: float
    action_up: ScalingAction
    action_down: ScalingAction
    cooldown_seconds: float = 300.0
    evaluation_window: int = 10  # Number of measurements to consider
    weight: float = 1.0  # Rule importance weight


@dataclass
class ScalingState:
    """Current scaling state."""
    active_instances: int = 1
    batch_size: int = 32
    model_complexity: str = "base"  # "tiny", "small", "base", "large"
    last_scaling_time: float = 0.0
    scaling_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))


class AdaptiveOptimizer:
    """Adaptive optimizer that adjusts hyperparameters based on performance."""
    
    def __init__(self, initial_lr: float = 1e-3, adaptation_rate: float = 0.1):
        self.initial_lr = initial_lr
        self.adaptation_rate = adaptation_rate
        self.performance_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.current_lr = initial_lr
        self.best_performance = float('inf')
        self.patience_counter = 0
        self.max_patience = 10
        self.logger = get_logger(__name__)
    
    def adapt_learning_rate(self, current_loss: float, optimizer: torch.optim.Optimizer) -> float:
        """Adapt learning rate based on performance trends."""
        self.performance_history.append(current_loss)
        
        if len(self.performance_history) < 5:
            return self.current_lr
        
        # Check if performance is improving
        recent_avg = np.mean(list(self.performance_history)[-5:])
        older_avg = np.mean(list(self.performance_history)[-10:-5]) if len(self.performance_history) >= 10 else recent_avg
        
        if recent_avg < self.best_performance:
            self.best_performance = recent_avg
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Adjust learning rate based on trends
        if recent_avg < older_avg * 0.95:  # Improving significantly
            self.current_lr *= 1.05  # Increase LR slightly
            self.current_lr = min(self.current_lr, self.initial_lr * 10)
        elif recent_avg > older_avg * 1.05:  # Getting worse
            self.current_lr *= 0.9  # Decrease LR
            self.current_lr = max(self.current_lr, self.initial_lr * 0.01)
        elif self.patience_counter >= self.max_patience:  # Plateau
            self.current_lr *= 0.5  # Significant reduction
            self.patience_counter = 0
        
        # Update optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        self.lr_history.append(self.current_lr)
        
        if len(self.lr_history) % 20 == 0:  # Log every 20 adaptations
            self.logger.info(f"Adapted learning rate to {self.current_lr:.6f} (best loss: {self.best_performance:.6f})")
        
        return self.current_lr
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            'current_lr': self.current_lr,
            'initial_lr': self.initial_lr,
            'best_performance': self.best_performance,
            'patience_counter': self.patience_counter,
            'lr_changes': len(self.lr_history),
            'performance_trend': (
                'improving' if len(self.performance_history) >= 10 and 
                np.mean(list(self.performance_history)[-5:]) < np.mean(list(self.performance_history)[-10:-5])
                else 'declining' if len(self.performance_history) >= 10 and
                np.mean(list(self.performance_history)[-5:]) > np.mean(list(self.performance_history)[-10:-5])
                else 'stable'
            )
        }


class AutoScaler:
    """Automatic scaling system for liquid neural networks."""
    
    def __init__(self, model: LiquidNet, metrics_collector: Optional[MetricsCollector] = None):
        self.model = model
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = get_logger(__name__)
        
        # Scaling configuration
        self.scaling_rules = self._create_default_rules()
        self.scaling_state = ScalingState()
        
        # Monitoring
        self.system_monitor = SystemMonitor()
        self.monitoring_thread = None
        self.scaling_thread = None
        self.running = False
        
        # Performance tracking
        self.request_queue = queue.Queue()
        self.processing_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0
        
        # Model variants for scaling
        self.model_variants = {}
        self._initialize_model_variants()
        
        # Load balancer
        self.load_balancer = LoadBalancer()
        
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            ScalingRule(
                metric=ScalingMetric.CPU_UTILIZATION,
                threshold_up=80.0,
                threshold_down=30.0,
                action_up=ScalingAction.SCALE_UP,
                action_down=ScalingAction.SCALE_DOWN,
                weight=1.0
            ),
            ScalingRule(
                metric=ScalingMetric.MEMORY_UTILIZATION,
                threshold_up=85.0,
                threshold_down=40.0,
                action_up=ScalingAction.OPTIMIZE_MODEL,
                action_down=ScalingAction.NO_ACTION,
                weight=1.2
            ),
            ScalingRule(
                metric=ScalingMetric.INFERENCE_LATENCY,
                threshold_up=100.0,  # ms
                threshold_down=20.0,
                action_up=ScalingAction.SCALE_UP,
                action_down=ScalingAction.SCALE_DOWN,
                weight=1.5
            ),
            ScalingRule(
                metric=ScalingMetric.QUEUE_LENGTH,
                threshold_up=50.0,
                threshold_down=5.0,
                action_up=ScalingAction.REDISTRIBUTE_LOAD,
                action_down=ScalingAction.NO_ACTION,
                weight=1.3
            ),
            ScalingRule(
                metric=ScalingMetric.ERROR_RATE,
                threshold_up=0.05,  # 5% error rate
                threshold_down=0.01,
                action_up=ScalingAction.OPTIMIZE_MODEL,
                action_down=ScalingAction.NO_ACTION,
                weight=2.0
            )
        ]
    
    def _initialize_model_variants(self):
        """Initialize different model complexity variants."""
        try:
            # Create smaller variants
            self.model_variants['tiny'] = self._create_model_variant([8], 'tiny')
            self.model_variants['small'] = self._create_model_variant([16, 8], 'small')
            self.model_variants['base'] = self.model  # Original model
            
            # Create larger variant if needed
            larger_units = [unit * 2 for unit in self.model.hidden_units]
            self.model_variants['large'] = self._create_model_variant(larger_units, 'large')
            
            self.logger.info(f"Initialized {len(self.model_variants)} model variants")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model variants: {e}")
            self.model_variants['base'] = self.model
    
    def _create_model_variant(self, hidden_units: List[int], variant_name: str) -> LiquidNet:
        """Create a model variant with different complexity."""
        try:
            variant = LiquidNet(
                input_dim=self.model.input_dim,
                hidden_units=hidden_units,
                output_dim=self.model.output_dim,
                tau=getattr(self.model, 'tau', 10.0),
                leak=getattr(self.model, 'leak', 0.1)
            )
            
            # Initialize with similar weights if possible
            self._transfer_weights(self.model, variant)
            
            self.logger.info(f"Created {variant_name} model variant with {sum(p.numel() for p in variant.parameters())} parameters")
            return variant
            
        except Exception as e:
            self.logger.error(f"Failed to create {variant_name} variant: {e}")
            return self.model
    
    def _transfer_weights(self, source: LiquidNet, target: LiquidNet):
        """Transfer weights between model variants where possible."""
        try:
            source_state = source.state_dict()
            target_state = target.state_dict()
            
            for name, target_param in target_state.items():
                if name in source_state:
                    source_param = source_state[name]
                    
                    # Transfer compatible weights
                    if target_param.shape == source_param.shape:
                        target_param.data.copy_(source_param.data)
                    elif len(target_param.shape) == len(source_param.shape):
                        # Transfer subset of weights for size differences
                        slices = tuple(slice(0, min(t, s)) for t, s in zip(target_param.shape, source_param.shape))
                        target_param.data[slices].copy_(source_param.data[slices])
            
            target.load_state_dict(target_state)
            
        except Exception as e:
            self.logger.warning(f"Weight transfer failed: {e}")
    
    def start_auto_scaling(self):
        """Start the auto-scaling system."""
        if self.running:
            return
        
        self.running = True
        self.system_monitor.start_monitoring()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start scaling decision thread
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        self.logger.info("Auto-scaling system started")
    
    def stop_auto_scaling(self):
        """Stop the auto-scaling system."""
        self.running = False
        self.system_monitor.stop_monitoring()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        if self.scaling_thread:
            self.scaling_thread.join(timeout=2.0)
        
        self.logger.info("Auto-scaling system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_current_metrics()
                
                # Store metrics for scaling decisions
                for metric_name, value in system_metrics.items():
                    self.scaling_state.performance_metrics[metric_name].append(value)
                
                # Record to metrics collector
                self.metrics_collector.record_metrics_batch(system_metrics, tags={'source': 'autoscaler'})
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.running:
            try:
                # Make scaling decisions
                action = self._evaluate_scaling_rules()
                
                if action != ScalingAction.NO_ACTION:
                    self._execute_scaling_action(action)
                
                time.sleep(30)  # Evaluate scaling every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Scaling loop error: {e}")
                time.sleep(30)
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        metrics = {}
        
        try:
            # System metrics
            metrics['cpu_utilization'] = psutil.cpu_percent()
            
            memory = psutil.virtual_memory()
            metrics['memory_utilization'] = memory.percent
            
            # GPU metrics
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                metrics['gpu_utilization'] = gpu_memory
            
            # Application metrics
            metrics['queue_length'] = self.request_queue.qsize()
            
            # Inference latency (average of recent processing times)
            if self.processing_times:
                metrics['inference_latency'] = np.mean(self.processing_times)
            
            # Throughput (requests per second)
            if len(self.processing_times) > 0:
                metrics['throughput'] = 1000 / np.mean(self.processing_times)  # Convert ms to requests/sec
            
            # Error rate
            metrics['error_rate'] = self.error_count / max(1, self.total_requests)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect some metrics: {e}")
        
        return metrics
    
    def _evaluate_scaling_rules(self) -> ScalingAction:
        """Evaluate scaling rules and decide on action."""
        if time.time() - self.scaling_state.last_scaling_time < 60:  # Minimum 1-minute cooldown
            return ScalingAction.NO_ACTION
        
        action_scores = defaultdict(float)
        
        for rule in self.scaling_rules:
            metric_values = self.scaling_state.performance_metrics.get(rule.metric.value, deque())
            
            if len(metric_values) < rule.evaluation_window:
                continue
            
            # Calculate average over evaluation window
            recent_values = list(metric_values)[-rule.evaluation_window:]
            avg_value = np.mean(recent_values)
            
            # Determine if thresholds are crossed
            if avg_value > rule.threshold_up:
                action_scores[rule.action_up] += rule.weight
            elif avg_value < rule.threshold_down:
                action_scores[rule.action_down] += rule.weight
        
        # Find the highest scoring action
        if not action_scores:
            return ScalingAction.NO_ACTION
        
        best_action = max(action_scores.items(), key=lambda x: x[1])
        
        # Only take action if score is significant
        if best_action[1] >= 1.0:  # Minimum threshold for action
            return best_action[0]
        
        return ScalingAction.NO_ACTION
    
    @robust_operation(category=ErrorCategory.COMPUTATION)
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute a scaling action."""
        self.logger.info(f"Executing scaling action: {action.value}")
        
        try:
            if action == ScalingAction.SCALE_UP:
                self._scale_up()
            elif action == ScalingAction.SCALE_DOWN:
                self._scale_down()
            elif action == ScalingAction.OPTIMIZE_MODEL:
                self._optimize_model()
            elif action == ScalingAction.ADJUST_BATCH_SIZE:
                self._adjust_batch_size()
            elif action == ScalingAction.REDISTRIBUTE_LOAD:
                self._redistribute_load()
            
            # Record scaling action
            self.scaling_state.last_scaling_time = time.time()
            self.scaling_state.scaling_history.append({
                'timestamp': time.time(),
                'action': action.value,
                'metrics': self._collect_current_metrics()
            })
            
            # Keep only recent history
            if len(self.scaling_state.scaling_history) > 100:
                self.scaling_state.scaling_history = self.scaling_state.scaling_history[-50:]
                
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action {action.value}: {e}")
    
    def _scale_up(self):
        """Scale up resources."""
        # Increase active instances (would integrate with orchestration system)
        self.scaling_state.active_instances += 1
        
        # Or switch to more complex model
        if self.scaling_state.model_complexity != 'large':
            complexity_order = ['tiny', 'small', 'base', 'large']
            current_idx = complexity_order.index(self.scaling_state.model_complexity)
            if current_idx < len(complexity_order) - 1:
                new_complexity = complexity_order[current_idx + 1]
                self._switch_model_variant(new_complexity)
        
        self.logger.info(f"Scaled up: {self.scaling_state.active_instances} instances, complexity: {self.scaling_state.model_complexity}")
    
    def _scale_down(self):
        """Scale down resources."""
        # Decrease active instances
        if self.scaling_state.active_instances > 1:
            self.scaling_state.active_instances -= 1
        
        # Or switch to simpler model
        complexity_order = ['tiny', 'small', 'base', 'large']
        current_idx = complexity_order.index(self.scaling_state.model_complexity)
        if current_idx > 0:
            new_complexity = complexity_order[current_idx - 1]
            self._switch_model_variant(new_complexity)
        
        self.logger.info(f"Scaled down: {self.scaling_state.active_instances} instances, complexity: {self.scaling_state.model_complexity}")
    
    def _optimize_model(self):
        """Optimize current model for better performance."""
        try:
            # Apply model optimizations
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Enable inference optimizations
            if torch.cuda.is_available():
                self.model = self.model.half()  # Use FP16
                torch.backends.cudnn.benchmark = True
            
            # Could also apply quantization, pruning, etc.
            self.logger.info("Applied model optimizations")
            
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
    
    def _adjust_batch_size(self):
        """Adjust batch size based on current performance."""
        current_latency = np.mean(self.processing_times) if self.processing_times else 50
        
        if current_latency > 100:  # High latency, reduce batch size
            self.scaling_state.batch_size = max(1, self.scaling_state.batch_size // 2)
        elif current_latency < 20:  # Low latency, can increase batch size
            self.scaling_state.batch_size = min(128, self.scaling_state.batch_size * 2)
        
        self.logger.info(f"Adjusted batch size to {self.scaling_state.batch_size}")
    
    def _redistribute_load(self):
        """Redistribute load across available resources."""
        # This would integrate with load balancer
        self.load_balancer.rebalance_load()
        self.logger.info("Redistributed load across instances")
    
    def _switch_model_variant(self, complexity: str):
        """Switch to different model complexity variant."""
        if complexity in self.model_variants:
            self.model = self.model_variants[complexity]
            self.scaling_state.model_complexity = complexity
            self.logger.info(f"Switched to {complexity} model variant")
        else:
            self.logger.warning(f"Model variant {complexity} not available")
    
    @robust_operation(category=ErrorCategory.COMPUTATION)
    def process_request(self, input_data: torch.Tensor) -> torch.Tensor:
        """Process request with performance tracking."""
        start_time = time.time()
        
        try:
            self.total_requests += 1
            
            # Add to queue for monitoring
            self.request_queue.put(input_data, timeout=0.1)
            
            # Process with current model
            with torch.no_grad():
                output = self.model(input_data)
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_times.append(processing_time)
            
            return output
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Request processing failed: {e}")
            raise
        finally:
            # Remove from queue
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                pass
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling system statistics."""
        return {
            'scaling_state': {
                'active_instances': self.scaling_state.active_instances,
                'batch_size': self.scaling_state.batch_size,
                'model_complexity': self.scaling_state.model_complexity,
                'last_scaling_time': self.scaling_state.last_scaling_time
            },
            'performance_metrics': {
                metric: {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values) if values else 0,
                    'min': np.min(values) if values else 0,
                    'max': np.max(values) if values else 0
                } for metric, values in self.scaling_state.performance_metrics.items()
            },
            'request_stats': {
                'total_requests': self.total_requests,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(1, self.total_requests),
                'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
                'queue_length': self.request_queue.qsize()
            },
            'scaling_history': self.scaling_state.scaling_history[-10:],  # Last 10 actions
            'model_variants': list(self.model_variants.keys())
        }


class LoadBalancer:
    """Simple load balancer for distributing requests."""
    
    def __init__(self):
        self.instances = []
        self.current_instance = 0
        self.instance_loads = defaultdict(int)
        self.logger = get_logger(__name__)
    
    def add_instance(self, instance_id: str, capacity: int = 100):
        """Add a processing instance."""
        self.instances.append({'id': instance_id, 'capacity': capacity})
        self.logger.info(f"Added instance {instance_id} with capacity {capacity}")
    
    def remove_instance(self, instance_id: str):
        """Remove a processing instance."""
        self.instances = [inst for inst in self.instances if inst['id'] != instance_id]
        if instance_id in self.instance_loads:
            del self.instance_loads[instance_id]
        self.logger.info(f"Removed instance {instance_id}")
    
    def get_next_instance(self) -> Optional[str]:
        """Get next instance using round-robin with load awareness."""
        if not self.instances:
            return None
        
        # Find instance with lowest load
        best_instance = min(self.instances, 
                           key=lambda x: self.instance_loads[x['id']] / x['capacity'])
        
        return best_instance['id']
    
    def record_request_completion(self, instance_id: str):
        """Record that a request has completed on an instance."""
        if instance_id in self.instance_loads:
            self.instance_loads[instance_id] = max(0, self.instance_loads[instance_id] - 1)
    
    def rebalance_load(self):
        """Rebalance load across instances."""
        # Simple rebalancing by resetting load counters
        # In a real system, this would migrate active requests
        total_load = sum(self.instance_loads.values())
        if total_load > 0:
            avg_load = total_load / len(self.instances) if self.instances else 0
            for instance in self.instances:
                self.instance_loads[instance['id']] = avg_load
        
        self.logger.info("Rebalanced load across instances")


class PerformanceOptimizer:
    """Performance optimization utilities for liquid neural networks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.optimization_cache = {}
    
    def optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference performance."""
        try:
            # Set to evaluation mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad_(False)
            
            # Apply torch.jit if possible
            try:
                # Create example input
                example_input = torch.randn(1, model.input_dim)
                traced_model = torch.jit.trace(model, example_input)
                traced_model = torch.jit.optimize_for_inference(traced_model)
                self.logger.info("Applied TorchScript optimization")
                return traced_model
            except Exception as e:
                self.logger.warning(f"TorchScript optimization failed: {e}")
            
            # Apply other optimizations
            if torch.cuda.is_available():
                model = model.half()  # FP16
                torch.backends.cudnn.benchmark = True
                self.logger.info("Applied CUDA optimizations")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Inference optimization failed: {e}")
            return model
    
    def optimize_memory_usage(self, model: torch.nn.Module, target_memory_mb: float = 512) -> torch.nn.Module:
        """Optimize model memory usage."""
        try:
            # Estimate current memory usage
            current_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            if current_memory <= target_memory_mb:
                return model
            
            reduction_factor = target_memory_mb / current_memory
            
            # Apply model compression techniques
            if reduction_factor < 0.5:
                # Aggressive compression needed
                model = self._apply_quantization(model, bits=8)
                model = self._apply_pruning(model, sparsity=0.5)
            elif reduction_factor < 0.8:
                # Moderate compression
                model = self._apply_quantization(model, bits=16)
                model = self._apply_pruning(model, sparsity=0.3)
            
            self.logger.info(f"Optimized model memory from {current_memory:.1f}MB to target {target_memory_mb:.1f}MB")
            return model
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return model
    
    def _apply_quantization(self, model: torch.nn.Module, bits: int = 8) -> torch.nn.Module:
        """Apply quantization to reduce model size."""
        try:
            if bits == 8:
                # Apply dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                return quantized_model
            else:
                # Apply half precision
                return model.half()
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: torch.nn.Module, sparsity: float = 0.3) -> torch.nn.Module:
        """Apply weight pruning to reduce model complexity."""
        try:
            import torch.nn.utils.prune as prune
            
            # Apply magnitude-based pruning to linear layers
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')  # Make pruning permanent
            
            self.logger.info(f"Applied {sparsity:.1%} pruning")
            return model
            
        except ImportError:
            self.logger.warning("Pruning not available (requires torch.nn.utils.prune)")
            return model
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return model
    
    def benchmark_model(self, model: torch.nn.Module, input_shape: Tuple[int, ...], 
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance."""
        try:
            model.eval()
            device = next(model.parameters()).device
            
            # Warmup
            dummy_input = torch.randn(input_shape).to(device)
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    output = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_inference_time = (total_time / num_iterations) * 1000  # ms
            throughput = num_iterations / total_time  # inferences/sec
            
            # Memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            results = {
                'avg_inference_time_ms': avg_inference_time,
                'throughput_per_sec': throughput,
                'memory_usage_mb': memory_mb,
                'total_parameters': sum(p.numel() for p in model.parameters())
            }
            
            self.logger.info(f"Benchmark results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return {}


def create_scaling_suite(model: LiquidNet) -> Dict[str, Any]:
    """Create comprehensive scaling and optimization suite."""
    return {
        'auto_scaler': AutoScaler(model),
        'adaptive_optimizer': AdaptiveOptimizer(),
        'performance_optimizer': PerformanceOptimizer(),
        'load_balancer': LoadBalancer()
    }