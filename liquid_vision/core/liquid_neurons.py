"""
Liquid Neural Network implementations optimized for event-based processing.
Based on Liquid AI architectures with temporal dynamics for neuromorphic computing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
import math
import logging

from ..utils.logging import log_exceptions, log_performance
from ..utils.validation import validate_inputs, ValidationError


# Global registry for custom neuron types
_NEURON_REGISTRY: Dict[str, type] = {}


def register_neuron(name: str) -> Callable:
    """Decorator to register custom neuron types."""
    def decorator(cls: type) -> type:
        _NEURON_REGISTRY[name] = cls
        return cls
    return decorator


class LiquidNeuron(nn.Module):
    """
    Base Liquid Neural Network cell with continuous-time dynamics.
    
    Implements the core liquid state machine with configurable time constants
    and nonlinear activation functions optimized for temporal processing.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau: float = 10.0,
        leak: float = 0.1,
        activation: str = "tanh",
        recurrent_connection: bool = True,
        dt: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.leak = leak
        self.dt = dt
        self.recurrent_connection = recurrent_connection
        
        # Input transformation
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Recurrent connections (if enabled)
        if recurrent_connection:
            self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.register_parameter('W_rec', None)
            
        # Bias term
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._init_weights()
        
    def _get_activation(self, activation: str) -> Callable:
        """Get activation function by name."""
        activations = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "relu": F.relu,
            "swish": F.silu,
            "gelu": F.gelu,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation]
        
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.W_in.weight)
        if self.W_rec is not None:
            # Spectral radius normalization for stability
            with torch.no_grad():
                nn.init.xavier_uniform_(self.W_rec.weight)
                spectral_radius = torch.linalg.matrix_norm(self.W_rec.weight, ord=2)
                self.W_rec.weight.data = self.W_rec.weight.data / spectral_radius * 0.9
                
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass through liquid neuron.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            hidden: Previous hidden state [batch_size, hidden_dim]
            dt: Time step (overrides default)
            
        Returns:
            Updated hidden state [batch_size, hidden_dim]
        """
        batch_size = x.size(0)
        dt = dt if dt is not None else self.dt
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            
        # Input transformation
        input_contribution = self.W_in(x)
        
        # Recurrent contribution
        if self.W_rec is not None:
            recurrent_contribution = self.W_rec(hidden)
        else:
            recurrent_contribution = 0
            
        # Liquid state dynamics: dh/dt = (-h + activation(W_in*x + W_rec*h + b)) / tau
        total_input = input_contribution + recurrent_contribution + self.bias
        target_state = self.activation(total_input)
        
        # Euler integration with leak
        dhdt = (-hidden * self.leak + target_state) / self.tau
        hidden_new = hidden + dt * dhdt
        
        return hidden_new
        
    def reset_state(self) -> None:
        """Reset internal state (useful for stateful processing)."""
        # This implementation is stateless, but subclasses might maintain state
        pass


class LiquidNet(nn.Module):
    """
    Multi-layer Liquid Neural Network for event-based processing.
    
    Combines multiple liquid neuron layers with optional readout layers
    for classification and regression tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        tau: float = 10.0,
        leak: float = 0.1,
        activation: str = "tanh",
        readout_activation: Optional[str] = None,
        dropout: float = 0.0,
        dt: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.num_layers = len(hidden_units)
        self.dt = dt
        
        # Build liquid layers
        self.liquid_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_units:
            layer = LiquidNeuron(
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
        self.readout = nn.Linear(hidden_units[-1], output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Readout activation
        if readout_activation:
            self.readout_activation = self._get_activation(readout_activation)
        else:
            self.readout_activation = None
            
        # Hidden states storage
        self.hidden_states: List[Optional[torch.Tensor]] = [None] * self.num_layers
        
    def _get_activation(self, activation: str) -> Callable:
        """Get activation function by name."""
        activations = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "relu": F.relu,
            "softmax": lambda x: F.softmax(x, dim=-1),
            "log_softmax": lambda x: F.log_softmax(x, dim=-1),
        }
        return activations.get(activation, lambda x: x)
        
    def forward(
        self, 
        x: torch.Tensor, 
        reset_state: bool = False,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass through liquid network.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            reset_state: Whether to reset hidden states
            dt: Time step override
            
        Returns:
            Output tensor [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """
        if reset_state:
            self.reset_states()
            
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            return self._forward_step(x, dt)
        elif x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            outputs = []
            
            for t in range(seq_len):
                output = self._forward_step(x[:, t], dt)
                outputs.append(output)
                
            return torch.stack(outputs, dim=1)
        else:
            raise ValueError(f"Input must be 2D or 3D, got {x.dim()}D")
            
    def _forward_step(self, x: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """Single forward step."""
        current_input = x
        
        # Pass through liquid layers
        for i, layer in enumerate(self.liquid_layers):
            self.hidden_states[i] = layer(current_input, self.hidden_states[i], dt)
            current_input = self.hidden_states[i]
            
            # Apply dropout between layers
            if self.dropout and self.training:
                current_input = self.dropout(current_input)
                
        # Readout layer
        output = self.readout(current_input)
        
        if self.readout_activation:
            output = self.readout_activation(output)
            
        return output
        
    def reset_states(self) -> None:
        """Reset all hidden states."""
        self.hidden_states = [None] * self.num_layers
        
    def get_liquid_states(self) -> List[Optional[torch.Tensor]]:
        """Get current liquid states for analysis."""
        return self.hidden_states.copy()
        
    def set_time_constants(self, tau: float) -> None:
        """Update time constants for all layers."""
        for layer in self.liquid_layers:
            layer.tau = tau


@register_neuron("adaptive_liquid")
class AdaptiveLiquidNeuron(LiquidNeuron):
    """
    Adaptive Liquid Neuron with learnable time constants.
    Time constants adapt based on input statistics for improved performance.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau_range: Tuple[float, float] = (5.0, 50.0),
        **kwargs
    ):
        super().__init__(input_dim, hidden_dim, **kwargs)
        
        self.tau_range = tau_range
        self.tau_adapter = nn.Linear(1, hidden_dim)
        
        # Initialize tau adapter
        nn.init.zeros_(self.tau_adapter.weight)
        nn.init.constant_(self.tau_adapter.bias, 0.5)  # Start at middle of range
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """Forward pass with adaptive time constants."""
        batch_size = x.size(0)
        dt = dt if dt is not None else self.dt
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            
        # Compute adaptive time constants based on input statistics
        input_std = x.std(dim=-1, keepdim=True)  # [batch_size, 1]
        tau_logits = self.tau_adapter(input_std)  # [batch_size, hidden_dim]
        tau_normalized = torch.sigmoid(tau_logits)
        
        # Scale to tau_range
        tau_min, tau_max = self.tau_range
        adaptive_tau = tau_min + tau_normalized * (tau_max - tau_min)
        
        # Input and recurrent contributions
        input_contribution = self.W_in(x)
        if self.W_rec is not None:
            recurrent_contribution = self.W_rec(hidden)
        else:
            recurrent_contribution = 0
            
        # Liquid dynamics with adaptive tau
        total_input = input_contribution + recurrent_contribution + self.bias
        target_state = self.activation(total_input)
        
        # Adaptive integration
        dhdt = (-hidden * self.leak + target_state) / adaptive_tau
        hidden_new = hidden + dt * dhdt
        
        return hidden_new


def create_liquid_net(
    input_dim: int,
    output_dim: int,
    architecture: str = "small",
    **kwargs
) -> LiquidNet:
    """
    Factory function to create pre-configured liquid networks.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        architecture: Architecture size ("tiny", "small", "base", "large")
        **kwargs: Additional arguments passed to LiquidNet
        
    Returns:
        Configured LiquidNet instance
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
    
    return LiquidNet(
        input_dim=input_dim,
        output_dim=output_dim,
        **config
    )


def get_model_info(model: LiquidNet) -> Dict[str, Any]:
    """Get comprehensive model information for analysis."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_info = []
    for i, layer in enumerate(model.liquid_layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        layer_info.append({
            "layer": i,
            "input_dim": layer.input_dim,
            "hidden_dim": layer.hidden_dim,
            "tau": layer.tau,
            "parameters": layer_params,
        })
        
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "num_layers": model.num_layers,
        "layer_info": layer_info,
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
    }