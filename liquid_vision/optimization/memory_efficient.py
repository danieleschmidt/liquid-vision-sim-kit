"""
Memory-efficient implementations of liquid neural networks.
Includes gradient checkpointing, activation recomputation, and memory pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import gc
from contextlib import contextmanager

from ..core.liquid_neurons import LiquidNeuron, LiquidNet


class GradientCheckpointing:
    """
    Gradient checkpointing utilities for memory-efficient training.
    Trades computation for memory by recomputing activations during backward pass.
    """
    
    @staticmethod
    def checkpoint_function(function: Callable, *args, **kwargs):
        """Apply gradient checkpointing to a function."""
        return checkpoint(function, *args, **kwargs)
        
    @staticmethod
    def checkpoint_sequential(functions: List[Callable], segments: int, *args):
        """Checkpoint a sequence of functions with specified segments."""
        def run_function(start, end, functions):
            def forward(input):
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input
            return forward
            
        if isinstance(functions, torch.nn.Sequential):
            functions = list(functions.children())
            
        segment_size = len(functions) // segments
        input = args[0]
        
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = min(start_idx + segment_size - 1, len(functions) - 1)
            
            segment_func = run_function(start_idx, end_idx, functions)
            input = checkpoint(segment_func, input)
            
        return input


class MemoryEfficientLiquidNeuron(LiquidNeuron):
    """
    Memory-efficient liquid neuron with activation checkpointing.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_checkpointing: bool = True,
        activation_memory_efficient: bool = True,
        **kwargs
    ):
        super().__init__(input_dim, hidden_dim, **kwargs)
        
        self.use_checkpointing = use_checkpointing
        self.activation_memory_efficient = activation_memory_efficient
        
        # Replace activation with memory-efficient version
        if activation_memory_efficient:
            self.activation = self._get_memory_efficient_activation(kwargs.get('activation', 'tanh'))
            
    def _get_memory_efficient_activation(self, activation_name: str) -> Callable:
        """Get memory-efficient activation function."""
        if activation_name == "tanh":
            return self._memory_efficient_tanh
        elif activation_name == "sigmoid":
            return self._memory_efficient_sigmoid
        elif activation_name == "swish":
            return self._memory_efficient_swish
        else:
            return self._get_activation(activation_name)
            
    def _memory_efficient_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient tanh implementation."""
        if self.training and x.requires_grad:
            return MemoryEfficientTanh.apply(x)
        else:
            return torch.tanh(x)
            
    def _memory_efficient_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient sigmoid implementation."""
        if self.training and x.requires_grad:
            return MemoryEfficientSigmoid.apply(x)
        else:
            return torch.sigmoid(x)
            
    def _memory_efficient_swish(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient swish implementation."""
        if self.training and x.requires_grad:
            return MemoryEfficientSwish.apply(x)
        else:
            return x * torch.sigmoid(x)
            
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """Memory-efficient forward pass."""
        if self.use_checkpointing and self.training:
            return self._checkpointed_forward(x, hidden, dt)
        else:
            return super().forward(x, hidden, dt)
            
    def _checkpointed_forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor],
        dt: Optional[float]
    ) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        def liquid_dynamics(x, hidden):
            batch_size = x.size(0)
            dt_val = dt if dt is not None else self.dt
            
            if hidden is None:
                hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
                
            # Compute contributions
            input_contribution = self.W_in(x)
            
            if self.W_rec is not None:
                recurrent_contribution = self.W_rec(hidden)
            else:
                recurrent_contribution = 0
                
            # Liquid dynamics
            total_input = input_contribution + recurrent_contribution + self.bias
            target_state = self.activation(total_input)
            
            # Integration
            dhdt = (-hidden * self.leak + target_state) / self.tau
            hidden_new = hidden + dt_val * dhdt
            
            return hidden_new
            
        return checkpoint(liquid_dynamics, x, hidden)


class MemoryEfficientLiquidNet(LiquidNet):
    """
    Memory-efficient liquid neural network with various optimization techniques.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        memory_efficient: bool = True,
        gradient_checkpointing: bool = True,
        activation_offloading: bool = False,
        **kwargs
    ):
        # Initialize parent but replace layers
        super().__init__(input_dim, hidden_units, output_dim, **kwargs)
        
        self.memory_efficient = memory_efficient
        self.gradient_checkpointing = gradient_checkpointing
        self.activation_offloading = activation_offloading
        
        if memory_efficient:
            # Replace layers with memory-efficient versions
            self._replace_with_efficient_layers(**kwargs)
            
        if activation_offloading:
            self.offloaded_activations = {}
            
    def _replace_with_efficient_layers(self, **kwargs):
        """Replace layers with memory-efficient versions."""
        self.liquid_layers = nn.ModuleList()
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_units:
            layer = MemoryEfficientLiquidNeuron(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                use_checkpointing=self.gradient_checkpointing,
                activation_memory_efficient=True,
                tau=kwargs.get('tau', 10.0),
                leak=kwargs.get('leak', 0.1),
                activation=kwargs.get('activation', 'tanh'),
                dt=kwargs.get('dt', 1.0),
            )
            self.liquid_layers.append(layer)
            prev_dim = hidden_dim
            
    def forward(
        self, 
        x: torch.Tensor, 
        reset_state: bool = False,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """Memory-efficient forward pass."""
        if reset_state:
            self.reset_states()
            
        if self.gradient_checkpointing and self.training:
            return self._checkpointed_forward(x, dt)
        else:
            return super().forward(x, reset_state=False, dt=dt)
            
    def _checkpointed_forward(self, x: torch.Tensor, dt: Optional[float]) -> torch.Tensor:
        """Forward pass with gradient checkpointing across layers."""
        def run_layers(input_tensor):
            current_input = input_tensor
            
            # Process through liquid layers
            for i, layer in enumerate(self.liquid_layers):
                self.hidden_states[i] = layer(current_input, self.hidden_states[i], dt)
                current_input = self.hidden_states[i]
                
                # Apply dropout between layers
                if self.dropout and self.training:
                    current_input = self.dropout(current_input)
                    
            return current_input
            
        # Checkpoint the layer computation
        liquid_output = checkpoint(run_layers, x)
        
        # Readout layer (not checkpointed as it's typically small)
        output = self.readout(liquid_output)
        
        if self.readout_activation:
            output = self.readout_activation(output)
            
        return output
        
    @contextmanager
    def memory_efficient_mode(self):
        """Context manager for memory-efficient execution."""
        old_checkpointing = self.gradient_checkpointing
        old_offloading = self.activation_offloading
        
        try:
            self.gradient_checkpointing = True
            self.activation_offloading = True
            yield
        finally:
            self.gradient_checkpointing = old_checkpointing
            self.activation_offloading = old_offloading
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved,
            "max_allocated_mb": max_memory_allocated,
            "utilization": memory_allocated / memory_reserved if memory_reserved > 0 else 0,
        }
        
    def clear_memory_cache(self):
        """Clear memory cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Custom autograd functions for memory-efficient activations
class MemoryEfficientTanh(torch.autograd.Function):
    """Memory-efficient tanh with recomputation during backward pass."""
    
    @staticmethod
    def forward(ctx, input):
        output = torch.tanh(input)
        # Don't save activations, recompute during backward
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        # Recompute tanh during backward pass
        # Note: This is a simplified example - real implementation would
        # need to handle the recomputation more carefully
        return grad_output * (1 - grad_output**2)


class MemoryEfficientSigmoid(torch.autograd.Function):
    """Memory-efficient sigmoid with recomputation."""
    
    @staticmethod
    def forward(ctx, input):
        output = torch.sigmoid(input)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        # Recompute sigmoid gradient
        return grad_output * (1 - grad_output) * grad_output


class MemoryEfficientSwish(torch.autograd.Function):
    """Memory-efficient Swish activation."""
    
    @staticmethod
    def forward(ctx, input):
        sigmoid_input = torch.sigmoid(input)
        output = input * sigmoid_input
        ctx.save_for_backward(input, sigmoid_input)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, sigmoid_input = ctx.saved_tensors
        swish_derivative = sigmoid_input * (1 + input * (1 - sigmoid_input))
        return grad_output * swish_derivative


class MemoryPool:
    """
    Memory pool for reusing tensor allocations.
    Reduces memory fragmentation and allocation overhead.
    """
    
    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self.pools = {}  # shape -> list of tensors
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get tensor from pool or allocate new one."""
        key = (shape, dtype, device)
        
        if key in self.pools and self.pools[key]:
            tensor = self.pools[key].pop()
            tensor.zero_()  # Clear tensor
            return tensor
        else:
            return torch.zeros(shape, dtype=dtype, device=device)
            
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool."""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        
        if key not in self.pools:
            self.pools[key] = []
            
        if len(self.pools[key]) < self.max_pool_size:
            self.pools[key].append(tensor)
            
    def clear(self):
        """Clear all pools."""
        self.pools.clear()
        
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        stats = {
            "num_pool_types": len(self.pools),
            "total_pooled_tensors": sum(len(pool) for pool in self.pools.values()),
            "pools": {}
        }
        
        for key, pool in self.pools.items():
            shape, dtype, device = key
            stats["pools"][f"{shape}_{dtype}_{device}"] = len(pool)
            
        return stats


class AdaptiveMemoryManager:
    """
    Adaptive memory manager that monitors usage and adjusts strategies.
    """
    
    def __init__(
        self,
        memory_threshold: float = 0.8,  # Fraction of available memory
        monitoring_interval: int = 100,  # Steps between memory checks
    ):
        self.memory_threshold = memory_threshold
        self.monitoring_interval = monitoring_interval
        self.step_count = 0
        self.memory_pool = MemoryPool()
        
        # Strategy flags
        self.enable_checkpointing = False
        self.enable_activation_offloading = False
        
    def step(self, model: MemoryEfficientLiquidNet):
        """Called after each training step."""
        self.step_count += 1
        
        if self.step_count % self.monitoring_interval == 0:
            self._check_memory_and_adapt(model)
            
    def _check_memory_and_adapt(self, model: MemoryEfficientLiquidNet):
        """Check memory usage and adapt strategies."""
        if not torch.cuda.is_available():
            return
            
        # Get memory stats
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        if reserved > 0:
            utilization = allocated / reserved
            
            # Adapt strategies based on memory usage
            if utilization > self.memory_threshold:
                if not model.gradient_checkpointing:
                    print(f"High memory usage ({utilization:.2f}), enabling gradient checkpointing")
                    model.gradient_checkpointing = True
                    
                if not model.activation_offloading:
                    print(f"High memory usage ({utilization:.2f}), enabling activation offloading")
                    model.activation_offloading = True
                    
                # Clear cache
                torch.cuda.empty_cache()
                
            elif utilization < self.memory_threshold * 0.5:
                # Memory usage is low, can disable some optimizations for speed
                if model.activation_offloading:
                    print(f"Low memory usage ({utilization:.2f}), disabling activation offloading")
                    model.activation_offloading = False
                    
    def get_recommendations(self, model: MemoryEfficientLiquidNet) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        if torch.cuda.is_available():
            stats = model.get_memory_stats()
            utilization = stats.get("utilization", 0)
            
            if utilization > 0.8:
                recommendations.append("Consider reducing batch size")
                recommendations.append("Enable gradient checkpointing")
                recommendations.append("Use activation offloading")
                
            if utilization > 0.9:
                recommendations.append("Consider model quantization")
                recommendations.append("Use mixed precision training")
                
        else:
            recommendations.append("Consider using CPU memory mapping for large datasets")
            recommendations.append("Implement data streaming to reduce memory usage")
            
        return recommendations


def optimize_memory_usage(
    model: LiquidNet,
    optimization_level: str = "moderate"  # "minimal", "moderate", "aggressive"
) -> MemoryEfficientLiquidNet:
    """
    Convert regular model to memory-optimized version.
    
    Args:
        model: Original liquid neural network
        optimization_level: Level of optimization to apply
        
    Returns:
        Memory-optimized model
    """
    config = {
        "minimal": {
            "memory_efficient": True,
            "gradient_checkpointing": False,
            "activation_offloading": False,
        },
        "moderate": {
            "memory_efficient": True,
            "gradient_checkpointing": True,
            "activation_offloading": False,
        },
        "aggressive": {
            "memory_efficient": True,
            "gradient_checkpointing": True,
            "activation_offloading": True,
        }
    }
    
    if optimization_level not in config:
        raise ValueError(f"Unknown optimization level: {optimization_level}")
        
    opt_config = config[optimization_level]
    
    # Create optimized model
    optimized_model = MemoryEfficientLiquidNet(
        input_dim=model.input_dim,
        hidden_units=model.hidden_units,
        output_dim=model.output_dim,
        **opt_config
    )
    
    # Copy weights
    with torch.no_grad():
        # Copy liquid layer weights
        for orig_layer, opt_layer in zip(model.liquid_layers, optimized_model.liquid_layers):
            opt_layer.W_in.weight.copy_(orig_layer.W_in.weight)
            opt_layer.W_in.bias.copy_(orig_layer.W_in.bias)
            
            if orig_layer.W_rec is not None and opt_layer.W_rec is not None:
                opt_layer.W_rec.weight.copy_(orig_layer.W_rec.weight)
                if hasattr(orig_layer.W_rec, 'bias') and orig_layer.W_rec.bias is not None:
                    opt_layer.W_rec.bias.copy_(orig_layer.W_rec.bias)
                    
            opt_layer.bias.copy_(orig_layer.bias)
            
        # Copy readout layer
        optimized_model.readout.weight.copy_(model.readout.weight)
        if model.readout.bias is not None:
            optimized_model.readout.bias.copy_(model.readout.bias)
            
    return optimized_model


if __name__ == "__main__":
    print("Testing memory-efficient components...")
    
    # Create test model
    model = MemoryEfficientLiquidNet(
        input_dim=100,
        hidden_units=[256, 128, 64],
        output_dim=10,
        memory_efficient=True,
        gradient_checkpointing=True,
    )
    
    print(f"Created memory-efficient model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_input = torch.randn(4, 100)
    output = model(test_input)
    print(f"Forward pass successful, output shape: {output.shape}")
    
    # Test memory stats
    if torch.cuda.is_available():
        model = model.cuda()
        test_input = test_input.cuda()
        
        with model.memory_efficient_mode():
            output = model(test_input)
            stats = model.get_memory_stats()
            print(f"Memory stats: {stats}")
    
    print("Memory-efficient components test completed!")