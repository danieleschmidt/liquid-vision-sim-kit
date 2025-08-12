"""
Lightweight Liquid Neural Network implementation using pure NumPy.
Designed for edge deployment and environments without PyTorch.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import logging
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LightweightConfig:
    """Configuration for lightweight liquid networks."""
    input_dim: int
    hidden_units: List[int]
    output_dim: int
    tau: float = 10.0
    leak: float = 0.1
    activation: str = "tanh"
    dt: float = 1.0
    quantization_bits: int = 8


class ActivationFunctions:
    """Collection of activation functions optimized for NumPy."""
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Prevent overflow
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        return x * ActivationFunctions.sigmoid(x)
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LightweightLiquidNeuron:
    """
    Pure NumPy implementation of Liquid Neural Network cell.
    Optimized for edge deployment and resource-constrained environments.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau: float = 10.0,
        leak: float = 0.1,
        activation: str = "tanh",
        dt: float = 1.0,
        quantization_bits: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.leak = leak
        self.dt = dt
        self.quantization_bits = quantization_bits
        
        # Initialize weights
        self.W_in = self._xavier_init(hidden_dim, input_dim)
        self.W_rec = self._xavier_init(hidden_dim, hidden_dim)
        self.bias = np.zeros(hidden_dim)
        
        # Apply quantization if specified
        if quantization_bits:
            self._quantize_weights()
        
        # Activation function
        self.activation = getattr(ActivationFunctions, activation)
        
        # State tracking
        self.hidden_state = None
        
    def _xavier_init(self, out_dim: int, in_dim: int) -> np.ndarray:
        """Xavier/Glorot initialization."""
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, (out_dim, in_dim))
    
    def _quantize_weights(self) -> None:
        """Apply quantization to weights for edge deployment."""
        if self.quantization_bits is None:
            return
            
        def quantize_array(arr: np.ndarray) -> np.ndarray:
            min_val, max_val = arr.min(), arr.max()
            scale = (max_val - min_val) / (2 ** self.quantization_bits - 1)
            quantized = np.round((arr - min_val) / scale)
            return quantized * scale + min_val
        
        self.W_in = quantize_array(self.W_in)
        self.W_rec = quantize_array(self.W_rec)
        self.bias = quantize_array(self.bias)
    
    def forward(
        self, 
        x: np.ndarray, 
        hidden: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Forward pass through liquid neuron.
        
        Args:
            x: Input array [batch_size, input_dim]
            hidden: Previous hidden state [batch_size, hidden_dim]
            dt: Time step override
            
        Returns:
            Updated hidden state [batch_size, hidden_dim]
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        batch_size = x.shape[0]
        dt = dt if dt is not None else self.dt
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = np.zeros((batch_size, self.hidden_dim))
            
        # Input transformation
        input_contribution = np.dot(x, self.W_in.T)
        
        # Recurrent contribution  
        recurrent_contribution = np.dot(hidden, self.W_rec.T)
        
        # Liquid state dynamics
        total_input = input_contribution + recurrent_contribution + self.bias
        target_state = self.activation(total_input)
        
        # Euler integration with leak
        dhdt = (-hidden * self.leak + target_state) / self.tau
        hidden_new = hidden + dt * dhdt
        
        return hidden_new
    
    def reset_state(self) -> None:
        """Reset internal state."""
        self.hidden_state = None


class LightweightLiquidNet:
    """
    Multi-layer Lightweight Liquid Neural Network.
    Pure NumPy implementation for edge deployment.
    """
    
    def __init__(self, config: LightweightConfig):
        self.config = config
        self.layers = []
        self.readout_layer = None
        self.hidden_states = []
        
        # Build liquid layers
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_units:
            layer = LightweightLiquidNeuron(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                tau=config.tau,
                leak=config.leak,
                activation=config.activation,
                dt=config.dt,
                quantization_bits=config.quantization_bits,
            )
            self.layers.append(layer)
            self.hidden_states.append(None)
            prev_dim = hidden_dim
        
        # Readout layer
        self.readout_layer = {
            'weight': self._xavier_init(config.output_dim, config.hidden_units[-1]),
            'bias': np.zeros(config.output_dim)
        }
        
    def _xavier_init(self, out_dim: int, in_dim: int) -> np.ndarray:
        """Xavier initialization for readout layer."""
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, (out_dim, in_dim))
    
    def forward(
        self, 
        x: np.ndarray, 
        reset_state: bool = False,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Forward pass through lightweight liquid network.
        
        Args:
            x: Input array [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            reset_state: Whether to reset hidden states
            dt: Time step override
            
        Returns:
            Output array [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """
        if reset_state:
            self.reset_states()
        
        # Handle different input dimensions
        if x.ndim == 2:
            return self._forward_step(x, dt)
        elif x.ndim == 3:
            batch_size, seq_len, _ = x.shape
            outputs = []
            
            for t in range(seq_len):
                output = self._forward_step(x[:, t], dt)
                outputs.append(output)
                
            return np.stack(outputs, axis=1)
        else:
            raise ValueError(f"Input must be 2D or 3D, got {x.ndim}D")
    
    def _forward_step(self, x: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """Single forward step."""
        current_input = x
        
        # Pass through liquid layers
        for i, layer in enumerate(self.layers):
            self.hidden_states[i] = layer.forward(current_input, self.hidden_states[i], dt)
            current_input = self.hidden_states[i]
        
        # Readout layer
        output = np.dot(current_input, self.readout_layer['weight'].T) + self.readout_layer['bias']
        
        return output
    
    def reset_states(self) -> None:
        """Reset all hidden states."""
        self.hidden_states = [None] * len(self.layers)
    
    def get_model_size(self) -> Dict[str, int]:
        """Calculate model size metrics."""
        total_params = 0
        
        for layer in self.layers:
            total_params += layer.W_in.size + layer.W_rec.size + layer.bias.size
        
        total_params += self.readout_layer['weight'].size + self.readout_layer['bias'].size
        
        return {
            'total_parameters': total_params,
            'memory_bytes': total_params * 4,  # Assuming float32
            'layers': len(self.layers),
        }
    
    def export_to_c(self, output_path: str) -> None:
        """Export model to C code for embedded deployment."""
        model_size = self.get_model_size()
        
        c_code = f"""
// Generated Lightweight Liquid Neural Network
// Model size: {model_size['total_parameters']} parameters, {model_size['memory_bytes']} bytes

#include <math.h>
#include <string.h>

#define NUM_LAYERS {len(self.layers)}
#define INPUT_DIM {self.config.input_dim}
#define OUTPUT_DIM {self.config.output_dim}

// Model parameters
"""
        
        # Export layer weights
        for i, layer in enumerate(self.layers):
            c_code += f"\n// Layer {i} weights\n"
            c_code += f"const float layer_{i}_w_in[{layer.hidden_dim}][{layer.input_dim}] = {{\n"
            for row in layer.W_in:
                c_code += "  {" + ", ".join(f"{val:.6f}f" for val in row) + "},\n"
            c_code += "};\n"
            
            c_code += f"const float layer_{i}_w_rec[{layer.hidden_dim}][{layer.hidden_dim}] = {{\n"
            for row in layer.W_rec:
                c_code += "  {" + ", ".join(f"{val:.6f}f" for val in row) + "},\n"
            c_code += "};\n"
            
            c_code += f"const float layer_{i}_bias[{layer.hidden_dim}] = {{\n"
            c_code += "  " + ", ".join(f"{val:.6f}f" for val in layer.bias) + "\n};\n"
        
        # Export readout weights
        c_code += f"""
// Readout layer weights
const float readout_weight[{self.config.output_dim}][{self.config.hidden_units[-1]}] = {{
"""
        for row in self.readout_layer['weight']:
            c_code += "  {" + ", ".join(f"{val:.6f}f" for val in row) + "},\n"
        c_code += "};\n"
        
        c_code += f"const float readout_bias[{self.config.output_dim}] = {{\n"
        c_code += "  " + ", ".join(f"{val:.6f}f" for val in self.readout_layer['bias']) + "\n};\n"
        
        # Add inference function
        c_code += """
// Activation functions
float tanh_activation(float x) {
    return tanhf(x);
}

float sigmoid_activation(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Inference function
void liquid_net_inference(const float* input, float* output) {
    // Implementation would go here
    // This is a template - full implementation requires specific layer handling
}
"""
        
        with open(output_path, 'w') as f:
            f.write(c_code)
        
        logger.info(f"Model exported to C code: {output_path}")


def create_lightweight_net(
    input_dim: int,
    output_dim: int,
    architecture: str = "small",
    quantization_bits: Optional[int] = None,
    **kwargs
) -> LightweightLiquidNet:
    """
    Factory function to create lightweight liquid networks.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        architecture: Architecture size ("tiny", "small", "base")
        quantization_bits: Number of bits for quantization (None for no quantization)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LightweightLiquidNet instance
    """
    architectures = {
        "tiny": {"hidden_units": [8], "tau": 5.0},
        "small": {"hidden_units": [16, 8], "tau": 8.0},
        "base": {"hidden_units": [32, 16], "tau": 10.0},
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    config_dict = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "quantization_bits": quantization_bits,
        **architectures[architecture],
        **kwargs
    }
    
    config = LightweightConfig(**config_dict)
    return LightweightLiquidNet(config)


def benchmark_inference_speed(
    model: LightweightLiquidNet,
    input_shape: Tuple[int, ...],
    num_iterations: int = 1000
) -> Dict[str, float]:
    """Benchmark inference speed for performance optimization."""
    import time
    
    # Generate test data
    test_input = np.random.randn(*input_shape)
    
    # Warmup
    for _ in range(10):
        _ = model.forward(test_input)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model.forward(test_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_inference_time = total_time / num_iterations
    fps = 1.0 / avg_inference_time
    
    return {
        'total_time_s': total_time,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'fps': fps,
        'iterations': num_iterations,
    }