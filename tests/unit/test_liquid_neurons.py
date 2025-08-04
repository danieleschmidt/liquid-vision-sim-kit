"""
Unit tests for liquid neural network components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from liquid_vision.core.liquid_neurons import (
    LiquidNeuron, LiquidNet, AdaptiveLiquidNeuron,
    create_liquid_net, get_model_info, register_neuron
)


class TestLiquidNeuron:
    """Test cases for LiquidNeuron class."""
    
    def test_initialization(self):
        """Test neuron initialization with various parameters."""
        neuron = LiquidNeuron(
            input_dim=10,
            hidden_dim=20,
            tau=15.0,
            leak=0.2,
            activation="tanh"
        )
        
        assert neuron.input_dim == 10
        assert neuron.hidden_dim == 20
        assert neuron.tau == 15.0
        assert neuron.leak == 0.2
        assert neuron.W_in.in_features == 10
        assert neuron.W_in.out_features == 20
        assert neuron.W_rec.in_features == 20
        assert neuron.W_rec.out_features == 20
        
    def test_forward_pass(self):
        """Test forward pass with different input shapes."""
        neuron = LiquidNeuron(input_dim=5, hidden_dim=10)
        
        # Test with batch input
        batch_size = 8
        x = torch.randn(batch_size, 5)
        
        # First call (no hidden state)
        h1 = neuron(x)
        assert h1.shape == (batch_size, 10)
        
        # Second call (with hidden state)
        h2 = neuron(x, h1)
        assert h2.shape == (batch_size, 10)
        
        # States should be different due to dynamics
        assert not torch.allclose(h1, h2)
        
    def test_different_activations(self):
        """Test neuron with different activation functions."""
        activations = ["tanh", "sigmoid", "relu", "swish", "gelu"]
        
        for activation in activations:
            neuron = LiquidNeuron(
                input_dim=3,
                hidden_dim=5,
                activation=activation
            )
            
            x = torch.randn(4, 3)
            h = neuron(x)
            assert h.shape == (4, 5)
            
    def test_no_recurrent_connection(self):
        """Test neuron without recurrent connections."""
        neuron = LiquidNeuron(
            input_dim=3,
            hidden_dim=5,
            recurrent_connection=False
        )
        
        assert neuron.W_rec is None
        
        x = torch.randn(2, 3)
        h = neuron(x)
        assert h.shape == (2, 5)
        
    def test_custom_dt(self):
        """Test neuron with custom time step."""
        neuron = LiquidNeuron(input_dim=3, hidden_dim=5, dt=0.5)
        
        x = torch.randn(2, 3)
        h1 = neuron(x, dt=0.1)  # Small time step
        h2 = neuron(x, dt=1.0)  # Large time step
        
        # Different time steps should produce different results
        assert not torch.allclose(h1, h2, atol=1e-6)
        
    def test_spectral_radius_initialization(self):
        """Test that recurrent weights have controlled spectral radius."""
        neuron = LiquidNeuron(input_dim=10, hidden_dim=20)
        
        # Check spectral radius is <= 0.9
        W_rec = neuron.W_rec.weight.data
        eigenvals = torch.linalg.eigvals(W_rec)
        spectral_radius = torch.max(torch.abs(eigenvals.real))
        
        assert spectral_radius <= 0.95  # Allow small tolerance


class TestLiquidNet:
    """Test cases for LiquidNet class."""
    
    def test_single_layer_network(self):
        """Test network with single liquid layer."""
        net = LiquidNet(
            input_dim=10,
            hidden_units=[20],
            output_dim=5
        )
        
        assert net.input_dim == 10
        assert net.output_dim == 5
        assert net.num_layers == 1
        assert len(net.liquid_layers) == 1
        
        # Test forward pass
        x = torch.randn(8, 10)
        output = net(x)
        assert output.shape == (8, 5)
        
    def test_multi_layer_network(self):
        """Test network with multiple liquid layers."""
        net = LiquidNet(
            input_dim=15,
            hidden_units=[32, 16, 8],
            output_dim=3
        )
        
        assert net.num_layers == 3
        assert len(net.liquid_layers) == 3
        
        x = torch.randn(4, 15)
        output = net(x)
        assert output.shape == (4, 3)
        
    def test_sequence_processing(self):
        """Test processing of sequence data."""
        net = LiquidNet(
            input_dim=5,
            hidden_units=[10, 8],
            output_dim=2
        )
        
        # 3D input: [batch, sequence, features]
        batch_size, seq_len = 6, 12
        x = torch.randn(batch_size, seq_len, 5)
        
        output = net(x)
        assert output.shape == (batch_size, seq_len, 2)
        
    def test_state_reset(self):
        """Test hidden state reset functionality."""
        net = LiquidNet(
            input_dim=5,
            hidden_units=[10],
            output_dim=2
        )
        
        x = torch.randn(3, 5)
        
        # First forward pass
        output1 = net(x)
        states1 = net.get_liquid_states()
        
        # Second forward pass (states should evolve)
        output2 = net(x)
        states2 = net.get_liquid_states()
        
        # Reset states
        net.reset_states()
        states_reset = net.get_liquid_states()
        
        # Third forward pass after reset
        output3 = net(x)
        
        # States should be different before/after reset
        assert not torch.allclose(states1[0], states2[0])
        assert states_reset[0] is None
        assert torch.allclose(output1, output3, atol=1e-6)
        
    def test_dropout(self):
        """Test dropout functionality."""
        net = LiquidNet(
            input_dim=10,
            hidden_units=[20, 15],
            output_dim=5,
            dropout=0.5
        )
        
        x = torch.randn(4, 10)
        
        # Training mode (dropout active)
        net.train()
        output_train = net(x)
        
        # Eval mode (dropout inactive)
        net.eval()
        output_eval = net(x)
        
        assert output_train.shape == output_eval.shape == (4, 5)
        
    def test_readout_activation(self):
        """Test different readout activations."""
        net = LiquidNet(
            input_dim=5,
            hidden_units=[10],
            output_dim=3,
            readout_activation="softmax"
        )
        
        x = torch.randn(2, 5)
        output = net(x)
        
        # Check softmax properties
        assert torch.allclose(torch.sum(output, dim=1), torch.ones(2), atol=1e-6)
        assert torch.all(output >= 0)
        
    def test_time_constant_update(self):
        """Test updating time constants."""
        net = LiquidNet(
            input_dim=5,
            hidden_units=[10, 8],
            output_dim=2,
            tau=10.0
        )
        
        # Check initial tau values
        for layer in net.liquid_layers:
            assert layer.tau == 10.0
            
        # Update tau values
        net.set_time_constants(20.0)
        
        # Check updated tau values
        for layer in net.liquid_layers:
            assert layer.tau == 20.0


class TestAdaptiveLiquidNeuron:
    """Test cases for AdaptiveLiquidNeuron class."""
    
    def test_initialization(self):
        """Test adaptive neuron initialization."""
        neuron = AdaptiveLiquidNeuron(
            input_dim=8,
            hidden_dim=12,
            tau_range=(5.0, 25.0)
        )
        
        assert neuron.input_dim == 8
        assert neuron.hidden_dim == 12
        assert neuron.tau_range == (5.0, 25.0)
        assert neuron.tau_adapter.in_features == 1
        assert neuron.tau_adapter.out_features == 12
        
    def test_adaptive_forward_pass(self):
        """Test forward pass with adaptive time constants."""
        neuron = AdaptiveLiquidNeuron(
            input_dim=5,
            hidden_dim=8,
            tau_range=(3.0, 15.0)
        )
        
        # Different input statistics should lead to different behaviors
        x1 = torch.randn(4, 5) * 0.1  # Low variance
        x2 = torch.randn(4, 5) * 2.0  # High variance
        
        h1 = neuron(x1)
        h2 = neuron(x2)
        
        assert h1.shape == h2.shape == (4, 8)
        # Results should be different due to adaptive tau
        assert not torch.allclose(h1, h2, atol=1e-4)


class TestFactoryFunctions:
    """Test factory functions and utilities."""
    
    def test_create_liquid_net(self):
        """Test create_liquid_net factory function."""
        architectures = ["tiny", "small", "base", "large"]
        
        for arch in architectures:
            net = create_liquid_net(
                input_dim=10,
                output_dim=5,
                architecture=arch
            )
            
            assert isinstance(net, LiquidNet)
            assert net.input_dim == 10
            assert net.output_dim == 5
            
        # Test invalid architecture
        with pytest.raises(ValueError):
            create_liquid_net(10, 5, architecture="invalid")
            
    def test_get_model_info(self):
        """Test model information extraction."""
        net = LiquidNet(
            input_dim=15,
            hidden_units=[32, 16],
            output_dim=8
        )
        
        info = get_model_info(net)
        
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "num_layers" in info
        assert "layer_info" in info
        assert "input_dim" in info
        assert "output_dim" in info
        
        assert info["input_dim"] == 15
        assert info["output_dim"] == 8
        assert info["num_layers"] == 2
        assert len(info["layer_info"]) == 2
        assert info["total_parameters"] > 0
        
    def test_neuron_registration(self):
        """Test custom neuron registration."""
        @register_neuron("test_neuron")
        class TestNeuron(LiquidNeuron):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.test_attribute = "test_value"
                
        # Check that neuron was registered
        from liquid_vision.core.liquid_neurons import _NEURON_REGISTRY
        assert "test_neuron" in _NEURON_REGISTRY
        assert _NEURON_REGISTRY["test_neuron"] == TestNeuron


class TestBatchProcessing:
    """Test batch processing capabilities."""
    
    def test_variable_batch_sizes(self):
        """Test processing with different batch sizes."""
        net = LiquidNet(
            input_dim=6,
            hidden_units=[12],
            output_dim=4
        )
        
        batch_sizes = [1, 3, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 6)
            output = net(x, reset_state=True)
            assert output.shape == (batch_size, 4)
            
    def test_memory_efficiency(self):
        """Test memory usage with large batches."""
        net = LiquidNet(
            input_dim=10,
            hidden_units=[50],
            output_dim=5
        )
        
        # Process large batch
        large_batch = torch.randn(1000, 10)
        
        # Should not raise memory error
        output = net(large_batch, reset_state=True)
        assert output.shape == (1000, 5)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_device_compatibility(device):
    """Test model works on different devices."""
    net = LiquidNet(
        input_dim=5,
        hidden_units=[10],
        output_dim=3
    ).to(device)
    
    x = torch.randn(4, 5).to(device)
    output = net(x)
    
    assert output.device.type == device
    assert output.shape == (4, 3)


def test_numerical_stability():
    """Test numerical stability with extreme inputs."""
    net = LiquidNet(
        input_dim=5,
        hidden_units=[10],
        output_dim=3
    )
    
    # Test with very large inputs
    x_large = torch.randn(2, 5) * 100
    output_large = net(x_large, reset_state=True)
    assert torch.isfinite(output_large).all()
    
    # Test with very small inputs
    x_small = torch.randn(2, 5) * 0.001
    output_small = net(x_small, reset_state=True)
    assert torch.isfinite(output_small).all()
    
    # Test with zero inputs
    x_zero = torch.zeros(2, 5)
    output_zero = net(x_zero, reset_state=True)
    assert torch.isfinite(output_zero).all()


def test_gradient_flow():
    """Test gradient computation and backpropagation."""
    net = LiquidNet(
        input_dim=5,
        hidden_units=[10],
        output_dim=3
    )
    
    x = torch.randn(2, 5, requires_grad=True)
    target = torch.randint(0, 3, (2,))
    
    output = net(x)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    for param in net.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()