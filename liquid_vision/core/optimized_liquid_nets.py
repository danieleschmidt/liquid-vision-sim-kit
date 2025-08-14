"""
Optimized Liquid Neural Networks for Generation 3.
Integrates performance optimizations, caching, and parallel processing.
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import functools

from .minimal_fallback import (
    MinimalTensor, MinimalLinear, MinimalLiquidNeuron, 
    MinimalLiquidNet, create_minimal_liquid_net
)
from ..optimization.performance_optimizer import (
    performance_optimizer, cached, memory_optimized, batch_optimized
)


class OptimizedMinimalTensor(MinimalTensor):
    """Enhanced MinimalTensor with optimization features."""
    
    def __init__(self, data, shape=None, cached=True):
        super().__init__(data, shape)
        self._cached = cached
        self._operation_cache = {}
        
    @cached(lambda self, other: f"add_{id(self)}_{id(other)}")
    def __add__(self, other):
        return super().__add__(other)
        
    @cached(lambda self, other: f"mul_{id(self)}_{id(other)}")
    def __mul__(self, other):
        return super().__mul__(other)
        
    @cached(lambda self, other: f"matmul_{id(self)}_{id(other)}")
    def matmul(self, other):
        return super().matmul(other)
        
    def clear_cache(self):
        """Clear operation cache for this tensor."""
        self._operation_cache.clear()


class OptimizedMinimalLinear(MinimalLinear):
    """Enhanced linear layer with caching and optimization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self._weight_cache = {}
        self._last_input_hash = None
        
    @memory_optimized
    def __call__(self, x: MinimalTensor) -> MinimalTensor:
        """Optimized forward pass with caching."""
        # Create cache key based on input
        input_hash = hash(str(x.data))
        
        # Check if we can reuse previous computation
        if input_hash == self._last_input_hash and 'last_output' in self._weight_cache:
            performance_optimizer.cache.cache.hits += 1
            return self._weight_cache['last_output']
        
        # Compute forward pass
        result = super().__call__(x)
        
        # Cache result
        self._last_input_hash = input_hash
        self._weight_cache['last_output'] = result
        
        return result


class OptimizedMinimalLiquidNeuron(MinimalLiquidNeuron):
    """Enhanced liquid neuron with performance optimizations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, tau: float = 10.0, 
                 leak: float = 0.1, activation: str = "tanh", dt: float = 1.0):
        super().__init__(input_dim, hidden_dim, tau, leak, activation, dt)
        
        # Replace linear layers with optimized versions
        self.W_in = OptimizedMinimalLinear(input_dim, hidden_dim, bias=False)
        self.W_rec = OptimizedMinimalLinear(hidden_dim, hidden_dim, bias=False)
        
        # State caching for temporal consistency
        self._state_cache = {}
        self._computation_cache = {}
        
    def _cache_key(self, x: MinimalTensor, hidden: Optional[MinimalTensor], dt: float) -> str:
        """Generate cache key for computation."""
        x_hash = hash(str(x.data))
        hidden_hash = hash(str(hidden.data)) if hidden is not None else 0
        return f"liquid_{x_hash}_{hidden_hash}_{dt}_{self.tau}_{self.leak}"
        
    @memory_optimized
    def __call__(self, x: MinimalTensor, hidden: Optional[MinimalTensor] = None, 
                 dt: Optional[float] = None) -> MinimalTensor:
        """Optimized forward pass with state caching."""
        dt = dt if dt is not None else self.dt
        batch_size = x.size(0)
        
        # Generate cache key
        cache_key = self._cache_key(x, hidden, dt)
        
        # Check computation cache
        if cache_key in self._computation_cache:
            performance_optimizer.cache.cache.hits += 1
            return self._computation_cache[cache_key]
        
        # Initialize hidden state if not provided
        if hidden is None:
            if 'default_hidden' in self._state_cache:
                hidden = self._state_cache['default_hidden']
            else:
                from .minimal_fallback import zeros
                hidden = zeros((batch_size, self.hidden_dim))
                self._state_cache['default_hidden'] = hidden
        
        # Optimized forward computation
        input_contrib = self.W_in(x)
        recurrent_contrib = self.W_rec(hidden)
        
        # Liquid dynamics with optimized integration
        total_input = input_contrib.data + recurrent_contrib.data + self.bias.data
        target_state = self.activation(OptimizedMinimalTensor(total_input))
        
        # Optimized Euler integration
        dhdt_data = []
        for i, row in enumerate(hidden.data):
            dhdt_row = []
            for j, h_val in enumerate(row):
                dhdt = (-h_val * self.leak + target_state.data[i][j]) / self.tau
                dhdt_row.append(dhdt)
            dhdt_data.append(dhdt_row)
        
        # Update hidden state
        hidden_new_data = []
        for i, (hidden_row, dhdt_row) in enumerate(zip(hidden.data, dhdt_data)):
            new_row = []
            for h_val, dhdt_val in zip(hidden_row, dhdt_row):
                new_row.append(h_val + dt * dhdt_val)
            hidden_new_data.append(new_row)
        
        result = OptimizedMinimalTensor(hidden_new_data)
        
        # Cache result
        self._computation_cache[cache_key] = result
        
        # Limit cache size
        if len(self._computation_cache) > 100:
            # Remove oldest entries
            old_keys = list(self._computation_cache.keys())[:-50]
            for key in old_keys:
                del self._computation_cache[key]
        
        return result
        
    def reset_cache(self):
        """Reset all caches."""
        self._state_cache.clear()
        self._computation_cache.clear()


class OptimizedMinimalLiquidNet(MinimalLiquidNet):
    """Enhanced liquid network with batch optimization and caching."""
    
    def __init__(self, input_dim: int, hidden_units: List[int], output_dim: int,
                 tau: float = 10.0, leak: float = 0.1, activation: str = "tanh",
                 dt: float = 1.0, enable_batch_optimization: bool = True):
        # Initialize with optimized components
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.num_layers = len(hidden_units)
        self.dt = dt
        self.enable_batch_optimization = enable_batch_optimization
        
        # Build optimized liquid layers
        self.liquid_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_units:
            layer = OptimizedMinimalLiquidNeuron(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                tau=tau,
                leak=leak,
                activation=activation,
                dt=dt,
            )
            self.liquid_layers.append(layer)
            prev_dim = hidden_dim
            
        # Optimized readout layer
        self.readout = OptimizedMinimalLinear(hidden_units[-1], output_dim)
        
        # Hidden states with optimization
        self.hidden_states = [None] * self.num_layers
        self._batch_cache = {}
        
    def _forward_step_optimized(self, x: OptimizedMinimalTensor, dt: Optional[float] = None) -> OptimizedMinimalTensor:
        """Optimized single forward step."""
        current_input = x
        
        # Pass through liquid layers with state management
        for i, layer in enumerate(self.liquid_layers):
            self.hidden_states[i] = layer(current_input, self.hidden_states[i], dt)
            current_input = self.hidden_states[i]
            
        # Readout layer
        output = self.readout(current_input)
        return output
        
    @memory_optimized
    def __call__(self, x: MinimalTensor, reset_state: bool = False, 
                 dt: Optional[float] = None) -> MinimalTensor:
        """Optimized forward pass."""
        if reset_state:
            self.reset_states()
            
        # Convert to optimized tensor
        if not isinstance(x, OptimizedMinimalTensor):
            x = OptimizedMinimalTensor(x.data, x.shape)
            
        # Handle 2D input with optimization
        if x.dim() == 2:
            return self._forward_step_optimized(x, dt)
        else:
            raise ValueError("Optimized implementation only supports 2D input")
            
    def reset_states(self):
        """Reset all hidden states and caches."""
        self.hidden_states = [None] * self.num_layers
        self._batch_cache.clear()
        
        # Reset layer caches
        for layer in self.liquid_layers:
            if hasattr(layer, 'reset_cache'):
                layer.reset_cache()
                
    @batch_optimized
    def process_batch(self, batch_inputs: List[MinimalTensor]) -> List[MinimalTensor]:
        """Optimized batch processing."""
        if not batch_inputs:
            return []
            
        def process_single(x):
            return self(x)
            
        return [process_single(x) for x in batch_inputs]
        
    def benchmark_performance(self, test_inputs: List[MinimalTensor], 
                            iterations: int = 100) -> Dict[str, float]:
        """Benchmark network performance."""
        if not test_inputs:
            return {}
            
        # Warmup
        for _ in range(10):
            for x in test_inputs[:5]:
                self(x)
                
        # Benchmark single inference
        start_time = time.time()
        for _ in range(iterations):
            for x in test_inputs:
                output = self(x)
        single_time = time.time() - start_time
        
        # Benchmark batch processing
        self.reset_states()
        start_time = time.time()
        for _ in range(iterations):
            batch_outputs = self.process_batch(test_inputs)
        batch_time = time.time() - start_time
        
        # Calculate metrics
        total_operations = iterations * len(test_inputs)
        single_throughput = total_operations / single_time
        batch_throughput = total_operations / batch_time
        
        speedup = single_time / batch_time if batch_time > 0 else 1.0
        
        return {
            "single_inference_time": single_time,
            "batch_processing_time": batch_time,
            "single_throughput": single_throughput,
            "batch_throughput": batch_throughput,
            "batch_speedup": speedup,
            "total_operations": total_operations
        }


def create_optimized_liquid_net(input_dim: int, output_dim: int, 
                              architecture: str = "small", **kwargs) -> OptimizedMinimalLiquidNet:
    """Factory function for optimized liquid networks."""
    architectures = {
        "tiny": {"hidden_units": [8], "tau": 8.0},
        "small": {"hidden_units": [16, 8], "tau": 10.0}, 
        "base": {"hidden_units": [32, 16, 8], "tau": 12.0},
        "large": {"hidden_units": [64, 32, 16], "tau": 15.0},
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
        
    config = architectures[architecture]
    config.update(kwargs)
    
    return OptimizedMinimalLiquidNet(
        input_dim=input_dim,
        output_dim=output_dim,
        **config
    )


class AdaptiveModelScaler:
    """Automatically scales model complexity based on workload."""
    
    def __init__(self):
        self.performance_history = []
        self.current_architecture = "small"
        self.workload_metrics = {
            "avg_batch_size": 1,
            "avg_latency": 0.0,
            "throughput_target": 100.0
        }
        
    def record_performance(self, batch_size: int, latency: float, throughput: float):
        """Record performance metrics for scaling decisions."""
        self.performance_history.append({
            "batch_size": batch_size,
            "latency": latency,
            "throughput": throughput,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
            
        # Update workload metrics
        recent_metrics = self.performance_history[-10:]
        self.workload_metrics.update({
            "avg_batch_size": sum(m["batch_size"] for m in recent_metrics) / len(recent_metrics),
            "avg_latency": sum(m["latency"] for m in recent_metrics) / len(recent_metrics),
        })
        
    def recommend_architecture(self) -> str:
        """Recommend optimal architecture based on workload."""
        if len(self.performance_history) < 5:
            return self.current_architecture
            
        avg_latency = self.workload_metrics["avg_latency"]
        avg_batch_size = self.workload_metrics["avg_batch_size"]
        
        # Scale up if high latency or large batches
        if avg_latency > 0.1 or avg_batch_size > 10:
            if self.current_architecture == "tiny":
                return "small"
            elif self.current_architecture == "small":
                return "base"
            elif self.current_architecture == "base":
                return "large"
                
        # Scale down if very fast and small batches
        elif avg_latency < 0.01 and avg_batch_size < 3:
            if self.current_architecture == "large":
                return "base"
            elif self.current_architecture == "base":
                return "small"
            elif self.current_architecture == "small":
                return "tiny"
                
        return self.current_architecture
        
    def auto_scale_model(self, current_model: OptimizedMinimalLiquidNet) -> OptimizedMinimalLiquidNet:
        """Automatically scale model if needed."""
        recommended_arch = self.recommend_architecture()
        
        if recommended_arch != self.current_architecture:
            print(f"🔄 Auto-scaling from {self.current_architecture} to {recommended_arch}")
            
            # Create new model with recommended architecture
            new_model = create_optimized_liquid_net(
                input_dim=current_model.input_dim,
                output_dim=current_model.output_dim,
                architecture=recommended_arch
            )
            
            self.current_architecture = recommended_arch
            return new_model
        else:
            return current_model


# Global model scaler
adaptive_scaler = AdaptiveModelScaler()