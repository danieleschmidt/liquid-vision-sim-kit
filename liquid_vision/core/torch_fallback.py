"""
Torch-independent fallback implementations for core liquid neuron functionality.
Enables basic operations without PyTorch dependencies for lightweight deployment.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Callable, Any, Union


class TensorFallback:
    """Lightweight tensor implementation using NumPy."""
    
    def __init__(self, data: np.ndarray, device: str = "cpu"):
        self.data = data
        self.device = device
        
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
        
    def size(self, dim: int = None) -> Union[Tuple[int, ...], int]:
        if dim is None:
            return self.shape
        return self.shape[dim]
        
    def dim(self) -> int:
        return len(self.shape)
        
    def __add__(self, other):
        if isinstance(other, TensorFallback):
            return TensorFallback(self.data + other.data)
        return TensorFallback(self.data + other)
        
    def __mul__(self, other):
        if isinstance(other, TensorFallback):
            return TensorFallback(self.data * other.data)
        return TensorFallback(self.data * other)
        
    def __matmul__(self, other):
        if isinstance(other, TensorFallback):
            return TensorFallback(self.data @ other.data)
        return TensorFallback(self.data @ other)
        
    def std(self, axis=None, keepdims=False):
        return TensorFallback(np.std(self.data, axis=axis, keepdims=keepdims))
        
    def zeros_like(self):
        return TensorFallback(np.zeros_like(self.data))
        
    def to(self, device: str):
        return TensorFallback(self.data, device)


def tensor(data, device="cpu") -> TensorFallback:
    """Create tensor from data."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return TensorFallback(data, device)


def zeros(shape: Tuple[int, ...], device="cpu") -> TensorFallback:
    """Create zero tensor."""
    return TensorFallback(np.zeros(shape), device)


def tanh(x: TensorFallback) -> TensorFallback:
    """Tanh activation."""
    return TensorFallback(np.tanh(x.data))


def sigmoid(x: TensorFallback) -> TensorFallback:
    """Sigmoid activation."""
    return TensorFallback(1 / (1 + np.exp(-x.data)))


def relu(x: TensorFallback) -> TensorFallback:
    """ReLU activation."""
    return TensorFallback(np.maximum(0, x.data))


class LinearFallback:
    """Lightweight linear layer implementation."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Initialize weights with Xavier uniform
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = TensorFallback(
            np.random.uniform(-limit, limit, (out_features, in_features))
        )
        
        if bias:
            self.bias = TensorFallback(np.zeros(out_features))
        else:
            self.bias = None
    
    def __call__(self, x: TensorFallback) -> TensorFallback:
        """Forward pass."""
        output = x @ self.weight.data.T
        if self.bias is not None:
            output = output + self.bias.data
        return TensorFallback(output)


class LiquidNeuronFallback:
    """
    Fallback Liquid Neural Network cell using NumPy.
    Provides basic functionality without PyTorch dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau: float = 10.0,
        leak: float = 0.1,
        activation: str = "tanh",
        dt: float = 1.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.leak = leak
        self.dt = dt
        
        # Linear layers
        self.W_in = LinearFallback(input_dim, hidden_dim, bias=False)
        self.W_rec = LinearFallback(hidden_dim, hidden_dim, bias=False)
        self.bias = TensorFallback(np.zeros(hidden_dim))
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Normalize recurrent weights for stability
        self._normalize_recurrent_weights()
        
    def _get_activation(self, activation: str) -> Callable:
        """Get activation function."""
        activations = {
            "tanh": tanh,
            "sigmoid": sigmoid,
            "relu": relu,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation]
        
    def _normalize_recurrent_weights(self):
        """Normalize recurrent weights for stability."""
        # Simple spectral normalization approximation
        W = self.W_rec.weight.data
        spectral_radius = np.linalg.norm(W, ord=2)
        if spectral_radius > 0:
            self.W_rec.weight.data = W / spectral_radius * 0.9
    
    def __call__(
        self, 
        x: TensorFallback, 
        hidden: Optional[TensorFallback] = None,
        dt: Optional[float] = None
    ) -> TensorFallback:
        """Forward pass through liquid neuron."""
        batch_size = x.size(0)
        dt = dt if dt is not None else self.dt
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = zeros((batch_size, self.hidden_dim))
            
        # Input and recurrent transformations
        input_contrib = self.W_in(x)
        recurrent_contrib = self.W_rec(hidden)
        
        # Liquid state dynamics
        total_input = input_contrib.data + recurrent_contrib.data + self.bias.data
        target_state = self.activation(TensorFallback(total_input))
        
        # Euler integration
        dhdt = (-hidden.data * self.leak + target_state.data) / self.tau
        hidden_new = TensorFallback(hidden.data + dt * dhdt)
        
        return hidden_new


class LiquidNetFallback:
    """
    Multi-layer Liquid Neural Network fallback implementation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        tau: float = 10.0,
        leak: float = 0.1,
        activation: str = "tanh",
        dt: float = 1.0,
    ):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.num_layers = len(hidden_units)
        self.dt = dt
        
        # Build liquid layers
        self.liquid_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_units:
            layer = LiquidNeuronFallback(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                tau=tau,
                leak=leak,
                activation=activation,
                dt=dt,
            )
            self.liquid_layers.append(layer)
            prev_dim = hidden_dim
            
        # Readout layer
        self.readout = LinearFallback(hidden_units[-1], output_dim)
        
        # Hidden states
        self.hidden_states = [None] * self.num_layers
        
    def __call__(
        self, 
        x: TensorFallback, 
        reset_state: bool = False,
        dt: Optional[float] = None,
    ) -> TensorFallback:
        """Forward pass through liquid network."""
        if reset_state:
            self.reset_states()
            
        # Handle 2D input
        if x.dim() == 2:
            return self._forward_step(x, dt)
        else:
            raise ValueError("Fallback implementation only supports 2D input")
            
    def _forward_step(self, x: TensorFallback, dt: Optional[float] = None) -> TensorFallback:
        """Single forward step."""
        current_input = x
        
        # Pass through liquid layers
        for i, layer in enumerate(self.liquid_layers):
            self.hidden_states[i] = layer(current_input, self.hidden_states[i], dt)
            current_input = self.hidden_states[i]
            
        # Readout layer
        output = self.readout(current_input)
        return output
        
    def reset_states(self) -> None:
        """Reset all hidden states."""
        self.hidden_states = [None] * self.num_layers


def create_liquid_net_fallback(
    input_dim: int,
    output_dim: int,
    architecture: str = "small",
    **kwargs
) -> LiquidNetFallback:
    """
    Factory function for fallback liquid networks.
    """
    architectures = {
        "tiny": {"hidden_units": [16], "tau": 8.0},
        "small": {"hidden_units": [32, 16], "tau": 10.0}, 
        "base": {"hidden_units": [64, 32, 16], "tau": 12.0},
        "large": {"hidden_units": [128, 64, 32], "tau": 15.0},
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
        
    config = architectures[architecture]
    config.update(kwargs)
    
    return LiquidNetFallback(
        input_dim=input_dim,
        output_dim=output_dim,
        **config
    )


# Demo functionality
def demo_fallback_functionality():
    """Demonstrate fallback functionality."""
    print("🧠 Liquid Vision Fallback Demo")
    
    # Create a simple liquid network
    model = create_liquid_net_fallback(
        input_dim=2,
        output_dim=3,
        architecture="tiny"
    )
    
    # Generate sample input
    x = tensor(np.random.randn(1, 2))
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.data}")
    
    return True


if __name__ == "__main__":
    demo_fallback_functionality()