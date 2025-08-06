"""
Memory-efficient implementations of liquid neural networks for edge devices.
Optimized for minimal memory footprint with gradient checkpointing, sparse operations,
quantization, and adaptive memory management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import gc
from contextlib import contextmanager
import numpy as np
import psutil
import os
import time
import threading

from ..core.liquid_neurons import LiquidNeuron, LiquidNet
from ..utils.logging import log_performance


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


class SparseLinear(nn.Module):
    """Memory-efficient sparse linear layer with configurable sparsity patterns."""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        sparsity: float = 0.7,
        structured: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.structured = structured
        
        # Initialize dense weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Create sparsity mask
        self._create_sparsity_mask()
        self.reset_parameters()
    
    def _create_sparsity_mask(self):
        """Create sparsity mask for weights."""
        if self.structured:
            # Structured sparsity: remove entire neurons/channels
            num_remove = int(self.out_features * self.sparsity)
            mask = torch.ones(self.out_features, self.in_features)
            if num_remove > 0:
                remove_indices = torch.randperm(self.out_features)[:num_remove]
                mask[remove_indices, :] = 0
        else:
            # Unstructured sparsity: random weight removal
            mask = torch.rand(self.out_features, self.in_features)
            mask = (mask >= self.sparsity).float()
        
        self.register_buffer('mask', mask)
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sparse_weight = self.weight * self.mask
        return F.linear(input, sparse_weight, self.bias)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        weight_memory = self.weight.nelement() * self.weight.element_size() / (1024**2)
        mask_memory = self.mask.nelement() * self.mask.element_size() / (1024**2)
        bias_memory = 0
        if self.bias is not None:
            bias_memory = self.bias.nelement() * self.bias.element_size() / (1024**2)
        
        return {
            'weight_mb': weight_memory,
            'mask_mb': mask_memory,
            'bias_mb': bias_memory,
            'total_mb': weight_memory + mask_memory + bias_memory,
            'effective_sparsity': (1 - self.mask.sum() / self.mask.numel()).item()
        }


class QuantizedLinear(nn.Module):
    """Quantized linear layer for edge deployment."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits: int = 8,
        bias_bits: int = 16,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        
        # Quantized parameters
        self.register_parameter('weight_scale', nn.Parameter(torch.ones(1)))
        self.register_parameter('weight_zero_point', nn.Parameter(torch.zeros(1, dtype=torch.int32)))
        self.register_buffer('quantized_weight', torch.zeros(out_features, in_features, dtype=torch.int8))
        
        if bias:
            self.register_parameter('bias_scale', nn.Parameter(torch.ones(1)))
            self.register_parameter('bias_zero_point', nn.Parameter(torch.zeros(1, dtype=torch.int32)))
            self.register_buffer('quantized_bias', torch.zeros(out_features, dtype=torch.int16))
        else:
            self.register_parameter('bias_scale', None)
            self.register_parameter('bias_zero_point', None)
            self.register_buffer('quantized_bias', None)
    
    def quantize_weights(self, weight: torch.Tensor):
        """Quantize floating point weights to integer representation."""
        w_min, w_max = weight.min(), weight.max()
        if self.weight_bits == 8:
            q_min, q_max = -128, 127
        else:
            q_min, q_max = -(2**(self.weight_bits-1)), 2**(self.weight_bits-1) - 1
        
        scale = (w_max - w_min) / (q_max - q_min)
        zero_point = q_min - w_min / scale
        zero_point = torch.round(zero_point).clamp(q_min, q_max).to(torch.int32)
        
        self.weight_scale.data = scale
        self.weight_zero_point.data = zero_point
        self.quantized_weight.data = torch.round(weight / scale + zero_point).clamp(q_min, q_max).to(torch.int8)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        dequant_weight = (self.quantized_weight.float() - self.weight_zero_point.float()) * self.weight_scale
        
        # Dequantize bias if present
        dequant_bias = None
        if self.quantized_bias is not None:
            dequant_bias = (self.quantized_bias.float() - self.bias_zero_point.float()) * self.bias_scale
        
        return F.linear(input, dequant_weight, dequant_bias)


class UltraLowMemoryLiquidNeuron(LiquidNeuron):
    """Ultra-low memory liquid neuron for extreme edge devices (< 64KB RAM)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        fixed_point: bool = True,
        weight_sharing: bool = True,
        **kwargs
    ):
        super().__init__(input_dim, hidden_dim, **kwargs)
        
        self.fixed_point = fixed_point
        self.weight_sharing = weight_sharing
        
        if fixed_point:
            # Convert to fixed-point arithmetic
            self._convert_to_fixed_point()
        
        if weight_sharing:
            # Share weights between input and recurrent connections
            self._setup_weight_sharing()
    
    def _convert_to_fixed_point(self, scale: int = 2**14):
        """Convert weights to fixed-point representation."""
        with torch.no_grad():
            self.W_in.weight.data = torch.round(self.W_in.weight * scale).clamp(-32768, 32767) / scale
            if self.W_rec is not None:
                self.W_rec.weight.data = torch.round(self.W_rec.weight * scale).clamp(-32768, 32767) / scale
    
    def _setup_weight_sharing(self):
        """Setup weight sharing to reduce memory usage."""
        if self.W_rec is not None:
            # Share subset of weights between input and recurrent matrices
            shared_dim = min(self.input_dim, self.hidden_dim) // 2
            if shared_dim > 0:
                self.W_rec.weight.data[:shared_dim, :shared_dim] = self.W_in.weight.data[:shared_dim, :shared_dim]


class HardwareAwareMemoryManager:
    """Hardware-aware memory management for different edge platforms."""
    
    def __init__(self, platform: str = "generic"):
        self.platform = platform
        self.memory_profiles = {
            "esp32": {"total_ram_kb": 520, "available_kb": 300, "stack_kb": 32},
            "esp32s3": {"total_ram_kb": 512, "available_kb": 350, "stack_kb": 32},
            "cortex_m4": {"total_ram_kb": 256, "available_kb": 180, "stack_kb": 16},
            "cortex_m7": {"total_ram_kb": 512, "available_kb": 400, "stack_kb": 32},
            "arduino_nano": {"total_ram_kb": 32, "available_kb": 20, "stack_kb": 2},
            "generic": {"total_ram_kb": 1024, "available_kb": 800, "stack_kb": 64}
        }
        
        self.profile = self.memory_profiles.get(platform, self.memory_profiles["generic"])
        self.allocated_kb = 0
        self.allocation_history = []
    
    def can_allocate(self, size_kb: float) -> bool:
        """Check if allocation is possible within memory constraints."""
        return (self.allocated_kb + size_kb) <= self.profile["available_kb"]
    
    def allocate_model(self, model: nn.Module) -> Dict[str, Any]:
        """Allocate model and return memory usage report."""
        model_size_kb = self.estimate_model_memory(model) / 1024
        
        if not self.can_allocate(model_size_kb):
            return {
                "success": False,
                "required_kb": model_size_kb,
                "available_kb": self.profile["available_kb"] - self.allocated_kb,
                "recommendations": self._get_memory_reduction_recommendations(model_size_kb)
            }
        
        self.allocated_kb += model_size_kb
        return {
            "success": True,
            "allocated_kb": model_size_kb,
            "total_allocated_kb": self.allocated_kb,
            "remaining_kb": self.profile["available_kb"] - self.allocated_kb
        }
    
    def estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate model memory usage in bytes."""
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Add overhead for activations (estimated)
        activation_overhead = param_bytes * 0.1
        
        return param_bytes + buffer_bytes + activation_overhead
    
    def _get_memory_reduction_recommendations(self, required_kb: float) -> List[str]:
        """Get recommendations for reducing memory usage."""
        excess_kb = required_kb - (self.profile["available_kb"] - self.allocated_kb)
        reduction_needed = excess_kb / required_kb
        
        recommendations = []
        
        if reduction_needed > 0.5:
            recommendations.append("Consider using a smaller model architecture")
            recommendations.append("Enable aggressive quantization (4-bit weights)")
            recommendations.append("Use structured pruning (50%+ sparsity)")
        elif reduction_needed > 0.2:
            recommendations.append("Enable weight quantization (8-bit)")
            recommendations.append("Use unstructured pruning (30%+ sparsity)")
            recommendations.append("Enable weight sharing")
        else:
            recommendations.append("Enable gradient checkpointing")
            recommendations.append("Use mixed precision (float16)")
        
        return recommendations


class StreamingMemoryManager:
    """Memory manager for streaming/continuous processing scenarios."""
    
    def __init__(self, max_memory_mb: float = 64.0):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        self.memory_warning_threshold = 0.8
        self.memory_critical_threshold = 0.95
        self.cleanup_callbacks = []
        self.monitoring_active = False
        self.monitor_thread = None
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register callback for memory cleanup operations."""
        self.cleanup_callbacks.append(callback)
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                self._check_memory_usage()
                time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed."""
        if torch.cuda.is_available():
            current_bytes = torch.cuda.memory_allocated()
        else:
            process = psutil.Process(os.getpid())
            current_bytes = process.memory_info().rss
        
        current_mb = current_bytes / (1024**2)
        utilization = current_mb / self.max_memory_mb
        
        if utilization > self.memory_critical_threshold:
            self._trigger_emergency_cleanup()
        elif utilization > self.memory_warning_threshold:
            self._trigger_cleanup()
    
    def _trigger_cleanup(self):
        """Trigger normal memory cleanup."""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Cleanup callback failed: {e}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency memory cleanup."""
        self._trigger_cleanup()
        
        # Additional emergency measures
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def create_edge_optimized_model(
    input_dim: int,
    output_dim: int,
    target_platform: str = "esp32",
    memory_budget_kb: float = 200
) -> nn.Module:
    """Create edge-optimized model within memory budget."""
    
    manager = HardwareAwareMemoryManager(target_platform)
    
    # Start with minimal architecture
    architectures = [
        ([8], 0.9),      # Ultra-tiny
        ([16], 0.8),     # Tiny
        ([32], 0.7),     # Small
        ([32, 16], 0.6), # Medium
    ]
    
    for hidden_units, sparsity in architectures:
        if target_platform.startswith("arduino") or memory_budget_kb < 50:
            # Use ultra-low memory variant
            model = nn.Sequential(
                SparseLinear(input_dim, hidden_units[0], sparsity=sparsity),
                nn.Tanh(),
                SparseLinear(hidden_units[0], output_dim, sparsity=sparsity)
            )
        else:
            model = MemoryEfficientLiquidNet(
                input_dim=input_dim,
                hidden_units=hidden_units,
                output_dim=output_dim,
                memory_efficient=True,
                gradient_checkpointing=True,
                activation_offloading=True
            )
        
        # Check if model fits in memory budget
        allocation_result = manager.allocate_model(model)
        
        if allocation_result["success"]:
            print(f"Created {target_platform} model: {hidden_units} hidden units, "
                  f"{allocation_result['allocated_kb']:.1f}KB")
            return model
    
    # If no architecture fits, return ultra-minimal model
    print(f"Warning: Using ultra-minimal model for {target_platform} (memory budget too small)")
    return nn.Sequential(
        SparseLinear(input_dim, 4, sparsity=0.9),
        nn.Tanh(),
        SparseLinear(4, output_dim, sparsity=0.9)
    )


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