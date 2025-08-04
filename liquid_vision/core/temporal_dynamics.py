"""
ODE solvers and temporal dynamics for liquid neural networks.
Implements various numerical integration schemes for continuous-time neural dynamics.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod
import math


class ODESolver(ABC):
    """
    Abstract base class for ODE solvers used in liquid neural networks.
    """
    
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        
    @abstractmethod
    def step(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        t: float,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform one integration step.
        
        Args:
            func: Function dy/dt = func(t, y, *args)
            y: Current state
            t: Current time
            
        Returns:
            Updated state y + dy
        """
        pass


class EulerSolver(ODESolver):
    """
    Forward Euler method for ODE integration.
    Simple but fast first-order method: y_{n+1} = y_n + dt * f(t_n, y_n)
    """
    
    def step(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        t: float,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Forward Euler step."""
        dydt = func(t, y, *args, **kwargs)
        return y + self.dt * dydt


class RK4Solver(ODESolver):
    """
    Fourth-order Runge-Kutta method for ODE integration.
    Higher accuracy but more computationally expensive than Euler.
    """
    
    def step(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        t: float,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Fourth-order Runge-Kutta step."""
        dt = self.dt
        
        k1 = func(t, y, *args, **kwargs)
        k2 = func(t + dt/2, y + dt*k1/2, *args, **kwargs)
        k3 = func(t + dt/2, y + dt*k2/2, *args, **kwargs)
        k4 = func(t + dt, y + dt*k3, *args, **kwargs)
        
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


class AdaptiveRKSolver(ODESolver):
    """
    Adaptive Runge-Kutta solver with error control.
    Automatically adjusts step size based on local truncation error.
    """
    
    def __init__(
        self,
        dt: float = 1.0,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_dt: float = 10.0,
        min_dt: float = 1e-3,
    ):
        super().__init__(dt)
        self.rtol = rtol
        self.atol = atol
        self.max_dt = max_dt
        self.min_dt = min_dt
        
    def step(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        t: float,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Adaptive step with error control."""
        dt = self.dt
        
        while True:
            # Take one full step
            y_full = self._rk4_step(func, y, t, dt, *args, **kwargs)
            
            # Take two half steps
            y_half1 = self._rk4_step(func, y, t, dt/2, *args, **kwargs)
            y_half2 = self._rk4_step(func, y_half1, t + dt/2, dt/2, *args, **kwargs)
            
            # Estimate error
            error = torch.abs(y_full - y_half2)
            tolerance = self.atol + self.rtol * torch.max(torch.abs(y), torch.abs(y_full))
            
            # Check if error is acceptable
            if torch.all(error <= tolerance):
                # Accept step and update dt for next iteration
                self.dt = min(self.max_dt, dt * 1.2)
                return y_half2  # Use more accurate estimate
            else:
                # Reject step and reduce dt
                dt = max(self.min_dt, dt * 0.5)
                if dt <= self.min_dt:
                    # Accept step with minimum dt to avoid infinite loop
                    return y_full
                    
    def _rk4_step(
        self,
        func: Callable,
        y: torch.Tensor,
        t: float,
        dt: float,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Single RK4 step with specified dt."""
        k1 = func(t, y, *args, **kwargs)
        k2 = func(t + dt/2, y + dt*k1/2, *args, **kwargs)
        k3 = func(t + dt/2, y + dt*k2/2, *args, **kwargs)
        k4 = func(t + dt, y + dt*k3, *args, **kwargs)
        
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


class LiquidDynamics(nn.Module):
    """
    Continuous-time dynamics for liquid neural networks.
    Implements various liquid state equations with configurable nonlinearities.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        tau: float = 10.0,
        leak: float = 0.1,
        coupling_strength: float = 1.0,
        noise_level: float = 0.0,
        activation: str = "tanh",
        solver: str = "euler",
        dt: float = 1.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.leak = leak
        self.coupling_strength = coupling_strength
        self.noise_level = noise_level
        self.dt = dt
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # ODE solver
        self.solver = self._get_solver(solver, dt)
        
        # Learnable parameters
        self.W_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        
        # Initialize recurrent weights with controlled spectral radius
        self._init_recurrent_weights()
        
    def _get_activation(self, activation: str) -> Callable:
        """Get activation function."""
        activations = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "swish": lambda x: x * torch.sigmoid(x),
            "gelu": torch.nn.functional.gelu,
        }
        return activations.get(activation, torch.tanh)
        
    def _get_solver(self, solver: str, dt: float) -> ODESolver:
        """Get ODE solver."""
        solvers = {
            "euler": EulerSolver,
            "rk4": RK4Solver,
            "adaptive": AdaptiveRKSolver,
        }
        
        if solver not in solvers:
            raise ValueError(f"Unknown solver: {solver}")
            
        return solvers[solver](dt)
        
    def _init_recurrent_weights(self) -> None:
        """Initialize recurrent weights with controlled spectral radius."""
        with torch.no_grad():
            # Random initialization
            nn.init.xavier_uniform_(self.W_rec)
            
            # Control spectral radius for stability
            eigenvals = torch.linalg.eigvals(self.W_rec).real
            spectral_radius = eigenvals.abs().max()
            
            if spectral_radius > 0.9:
                self.W_rec.data = self.W_rec.data * (0.9 / spectral_radius)
                
    def liquid_ode(
        self,
        t: float,
        hidden: torch.Tensor,
        external_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Liquid state ODE: dh/dt = (-leak*h + activation(W_rec*h + input)) / tau + noise
        
        Args:
            t: Current time
            hidden: Current hidden state [batch_size, hidden_dim]
            external_input: External input [batch_size, hidden_dim]
            
        Returns:
            Time derivative dh/dt
        """
        # Recurrent contribution
        recurrent_input = hidden @ self.W_rec.T * self.coupling_strength
        
        # Total input
        total_input = recurrent_input + external_input
        
        # Nonlinear transformation
        activated = self.activation(total_input)
        
        # Liquid dynamics
        dhdt = (-self.leak * hidden + activated) / self.tau
        
        # Add noise if specified
        if self.noise_level > 0 and self.training:
            noise = torch.randn_like(hidden) * self.noise_level
            dhdt = dhdt + noise
            
        return dhdt
        
    def forward(
        self,
        hidden: torch.Tensor,
        external_input: torch.Tensor,
        t: float = 0.0,
    ) -> torch.Tensor:
        """
        Integrate liquid dynamics for one time step.
        
        Args:
            hidden: Current hidden state [batch_size, hidden_dim]
            external_input: External input [batch_size, hidden_dim]
            t: Current time
            
        Returns:
            Updated hidden state
        """
        return self.solver.step(
            self.liquid_ode,
            hidden,
            t,
            external_input
        )


class MultiTimescaleDynamics(nn.Module):
    """
    Multi-timescale liquid dynamics with different time constants
    for different state components.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_timescales: int = 3,
        tau_range: Tuple[float, float] = (5.0, 50.0),
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_timescales = num_timescales
        
        # Divide hidden dimensions across timescales
        dims_per_scale = hidden_dim // num_timescales
        self.scale_dims = [dims_per_scale] * num_timescales
        self.scale_dims[-1] += hidden_dim - sum(self.scale_dims)  # Handle remainder
        
        # Create time constants
        tau_min, tau_max = tau_range
        self.tau_values = torch.logspace(
            math.log10(tau_min),
            math.log10(tau_max),
            num_timescales
        )
        
        # Create dynamics for each timescale
        self.dynamics = nn.ModuleList()
        start_idx = 0
        
        for i, (dim, tau) in enumerate(zip(self.scale_dims, self.tau_values)):
            dynamics = LiquidDynamics(
                hidden_dim=dim,
                tau=tau.item(),
                **kwargs
            )
            self.dynamics.append(dynamics)
            
    def forward(
        self,
        hidden: torch.Tensor,
        external_input: torch.Tensor,
        t: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass through multi-timescale dynamics."""
        batch_size = hidden.size(0)
        updated_states = []
        
        # Split hidden state and input across timescales
        hidden_splits = torch.split(hidden, self.scale_dims, dim=1)
        input_splits = torch.split(external_input, self.scale_dims, dim=1)
        
        # Process each timescale
        for dynamics, h_split, i_split in zip(self.dynamics, hidden_splits, input_splits):
            updated = dynamics(h_split, i_split, t)
            updated_states.append(updated)
            
        return torch.cat(updated_states, dim=1)


class AdaptiveDynamics(nn.Module):
    """
    Adaptive liquid dynamics with learnable time constants
    that adapt based on input statistics.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        tau_range: Tuple[float, float] = (5.0, 50.0),
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.tau_range = tau_range
        self.adaptation_rate = adaptation_rate
        
        # Base dynamics
        self.base_dynamics = LiquidDynamics(hidden_dim, **kwargs)
        
        # Adaptation network
        self.tau_adaptation = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        # Running statistics for input
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_var', torch.ones(input_dim))
        self.register_buffer('update_count', torch.tensor(0))
        
    def _update_input_stats(self, external_input: torch.Tensor) -> None:
        """Update running input statistics."""
        if self.training:
            batch_mean = external_input.mean(dim=0)
            batch_var = external_input.var(dim=0, unbiased=False)
            
            # Update running averages
            alpha = self.adaptation_rate
            self.input_mean = (1 - alpha) * self.input_mean + alpha * batch_mean
            self.input_var = (1 - alpha) * self.input_var + alpha * batch_var
            self.update_count += 1
            
    def forward(
        self,
        hidden: torch.Tensor,
        external_input: torch.Tensor,
        t: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass with adaptive time constants."""
        # Update input statistics
        self._update_input_stats(external_input)
        
        # Normalize input
        normalized_input = (external_input - self.input_mean) / (self.input_var.sqrt() + 1e-8)
        
        # Compute adaptive tau values
        adaptation_input = torch.cat([normalized_input, hidden], dim=1)
        tau_logits = self.tau_adaptation(adaptation_input)
        tau_normalized = torch.sigmoid(tau_logits)
        
        # Scale to tau_range
        tau_min, tau_max = self.tau_range
        adaptive_tau = tau_min + tau_normalized * (tau_max - tau_min)
        
        # Override base dynamics tau (this is a simplified approach)
        # In practice, you might want to modify the ODE directly
        original_tau = self.base_dynamics.tau
        self.base_dynamics.tau = adaptive_tau.mean().item()
        
        # Forward pass
        result = self.base_dynamics(hidden, external_input, t)
        
        # Restore original tau
        self.base_dynamics.tau = original_tau
        
        return result


def create_dynamics(
    dynamics_type: str,
    hidden_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create liquid dynamics.
    
    Args:
        dynamics_type: Type of dynamics ("liquid", "multiscale", "adaptive")
        hidden_dim: Hidden state dimension
        **kwargs: Additional arguments
        
    Returns:
        Configured dynamics module
    """
    dynamics_map = {
        "liquid": LiquidDynamics,
        "multiscale": MultiTimescaleDynamics,
        "adaptive": AdaptiveDynamics,
    }
    
    if dynamics_type not in dynamics_map:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")
        
    return dynamics_map[dynamics_type](hidden_dim=hidden_dim, **kwargs)