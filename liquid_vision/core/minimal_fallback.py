"""
Minimal fallback implementations using only Python standard library.
No external dependencies required - enables basic functionality everywhere.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Callable, Any, Union


class MinimalTensor:
    """Ultra-lightweight tensor using Python lists."""
    
    def __init__(self, data: List[List[float]], shape: Optional[Tuple[int, ...]] = None):
        if isinstance(data, (int, float)):
            data = [[data]]
        elif isinstance(data, list) and not isinstance(data[0], list):
            data = [data]
        
        self.data = data
        self._shape = shape or self._compute_shape(data)
        
    def _compute_shape(self, data: List) -> Tuple[int, ...]:
        """Compute shape of nested list."""
        if not isinstance(data[0], list):
            return (len(data),)
        return (len(data), len(data[0]))
        
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
        
    def size(self, dim: int = None) -> Union[Tuple[int, ...], int]:
        if dim is None:
            return self.shape
        return self.shape[dim]
        
    def dim(self) -> int:
        return len(self.shape)
        
    def __add__(self, other):
        if isinstance(other, MinimalTensor):
            result = []
            for i, row in enumerate(self.data):
                new_row = []
                for j, val in enumerate(row):
                    new_row.append(val + other.data[i][j])
                result.append(new_row)
            return MinimalTensor(result)
        else:
            result = []
            for row in self.data:
                new_row = [val + other for val in row]
                result.append(new_row)
            return MinimalTensor(result)
            
    def __mul__(self, other):
        if isinstance(other, MinimalTensor):
            result = []
            for i, row in enumerate(self.data):
                new_row = []
                for j, val in enumerate(row):
                    new_row.append(val * other.data[i][j])
                result.append(new_row)
            return MinimalTensor(result)
        else:
            result = []
            for row in self.data:
                new_row = [val * other for val in row]
                result.append(new_row)
            return MinimalTensor(result)
            
    def matmul(self, other):
        """Matrix multiplication."""
        if not isinstance(other, MinimalTensor):
            raise ValueError("Matrix multiplication requires two tensors")
            
        # self: [m, n], other: [n, p] -> result: [m, p]
        m, n = self.shape
        n2, p = other.shape
        
        if n != n2:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} and {other.shape}")
            
        result = []
        for i in range(m):
            row = []
            for j in range(p):
                val = 0
                for k in range(n):
                    val += self.data[i][k] * other.data[k][j]
                row.append(val)
            result.append(row)
            
        return MinimalTensor(result)
        
    def std(self):
        """Compute standard deviation."""
        all_vals = []
        for row in self.data:
            all_vals.extend(row)
            
        mean_val = sum(all_vals) / len(all_vals)
        variance = sum((x - mean_val) ** 2 for x in all_vals) / len(all_vals)
        std_val = math.sqrt(variance)
        
        return MinimalTensor([[std_val]])
        
    def transpose(self):
        """Transpose the tensor."""
        if len(self.shape) != 2:
            raise ValueError("Transpose only supported for 2D tensors")
            
        m, n = self.shape
        result = []
        for j in range(n):
            row = []
            for i in range(m):
                row.append(self.data[i][j])
            result.append(row)
            
        return MinimalTensor(result)


def zeros(shape: Tuple[int, ...]) -> MinimalTensor:
    """Create zero tensor."""
    if len(shape) == 1:
        return MinimalTensor([[0.0] * shape[0]])
    elif len(shape) == 2:
        return MinimalTensor([[0.0] * shape[1] for _ in range(shape[0])])
    else:
        raise ValueError("Only 1D and 2D shapes supported")


def random_tensor(shape: Tuple[int, ...], std: float = 1.0) -> MinimalTensor:
    """Create random tensor."""
    if len(shape) == 1:
        data = [[random.gauss(0, std) for _ in range(shape[0])]]
        return MinimalTensor(data)
    elif len(shape) == 2:
        data = [[random.gauss(0, std) for _ in range(shape[1])] for _ in range(shape[0])]
        return MinimalTensor(data)
    else:
        raise ValueError("Only 1D and 2D shapes supported")


def tanh(x: MinimalTensor) -> MinimalTensor:
    """Tanh activation."""
    result = []
    for row in x.data:
        new_row = [math.tanh(val) for val in row]
        result.append(new_row)
    return MinimalTensor(result)


def sigmoid(x: MinimalTensor) -> MinimalTensor:
    """Sigmoid activation."""
    result = []
    for row in x.data:
        new_row = []
        for val in row:
            try:
                new_row.append(1 / (1 + math.exp(-val)))
            except OverflowError:
                new_row.append(0.0 if val < 0 else 1.0)
        result.append(new_row)
    return MinimalTensor(result)


def relu(x: MinimalTensor) -> MinimalTensor:
    """ReLU activation."""
    result = []
    for row in x.data:
        new_row = [max(0, val) for val in row]
        result.append(new_row)
    return MinimalTensor(result)


class MinimalLinear:
    """Minimal linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Xavier initialization
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = random_tensor((out_features, in_features), limit)
        
        if bias:
            self.bias = zeros((1, out_features))
        else:
            self.bias = None
    
    def __call__(self, x: MinimalTensor) -> MinimalTensor:
        """Forward pass."""
        # x @ W.T + b
        output = x.matmul(self.weight.transpose())
        if self.bias is not None:
            output = output + self.bias
        return output


class MinimalLiquidNeuron:
    """
    Minimal Liquid Neural Network cell using only Python standard library.
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
        self.W_in = MinimalLinear(input_dim, hidden_dim, bias=False)
        self.W_rec = MinimalLinear(hidden_dim, hidden_dim, bias=False)
        self.bias = zeros((1, hidden_dim))
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Normalize recurrent weights
        self._normalize_weights()
        
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
        
    def _normalize_weights(self):
        """Simple weight normalization."""
        # Scale recurrent weights to prevent instability
        scale_factor = 0.9 / math.sqrt(self.hidden_dim)
        self.W_rec.weight = self.W_rec.weight * scale_factor
    
    def __call__(
        self, 
        x: MinimalTensor, 
        hidden: Optional[MinimalTensor] = None,
        dt: Optional[float] = None
    ) -> MinimalTensor:
        """Forward pass."""
        batch_size = x.size(0)
        dt = dt if dt is not None else self.dt
        
        # Initialize hidden state
        if hidden is None:
            hidden = zeros((batch_size, self.hidden_dim))
            
        # Forward pass
        input_contrib = self.W_in(x)
        recurrent_contrib = self.W_rec(hidden)
        
        # Liquid dynamics
        total_input = input_contrib + recurrent_contrib + self.bias
        target_state = self.activation(total_input)
        
        # Euler integration
        dhdt = (hidden * (-self.leak) + target_state) * (1.0 / self.tau)
        hidden_new = hidden + dhdt * dt
        
        return hidden_new


class MinimalLiquidNet:
    """
    Minimal multi-layer Liquid Neural Network.
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
            layer = MinimalLiquidNeuron(
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
        self.readout = MinimalLinear(hidden_units[-1], output_dim)
        
        # Hidden states
        self.hidden_states = [None] * self.num_layers
        
    def __call__(
        self, 
        x: MinimalTensor, 
        reset_state: bool = False,
        dt: Optional[float] = None,
    ) -> MinimalTensor:
        """Forward pass."""
        if reset_state:
            self.reset_states()
            
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


def create_minimal_liquid_net(
    input_dim: int,
    output_dim: int,
    architecture: str = "small",
    **kwargs
) -> MinimalLiquidNet:
    """Factory function for minimal liquid networks."""
    architectures = {
        "tiny": {"hidden_units": [8], "tau": 8.0},
        "small": {"hidden_units": [16, 8], "tau": 10.0}, 
        "base": {"hidden_units": [32, 16, 8], "tau": 12.0},
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
        
    config = architectures[architecture]
    config.update(kwargs)
    
    return MinimalLiquidNet(
        input_dim=input_dim,
        output_dim=output_dim,
        **config
    )


def demo_minimal_functionality():
    """Demonstrate minimal fallback functionality."""
    print("🧠 Minimal Liquid Vision Demo (Zero Dependencies)")
    
    # Create a simple liquid network
    model = create_minimal_liquid_net(
        input_dim=2,
        output_dim=3,
        architecture="tiny"
    )
    
    # Generate sample input
    x = MinimalTensor([[0.5, -0.3]])
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.data}")
    
    # Test sequence processing
    print("\nTesting sequence processing:")
    for i in range(3):
        x_i = MinimalTensor([[random.random() - 0.5, random.random() - 0.5]])
        output_i = model(x_i)
        print(f"Step {i}: input={x_i.data[0][:2]}, output={output_i.data[0][:2]}")
    
    return True


if __name__ == "__main__":
    demo_minimal_functionality()