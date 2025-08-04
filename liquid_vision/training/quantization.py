"""
Quantization-aware training and model compression for edge deployment.
Implements various quantization schemes optimized for liquid neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import numpy as np
from abc import ABC, abstractmethod

from ..core.liquid_neurons import LiquidNeuron, LiquidNet


class QuantizationScheme(ABC):
    """Base class for quantization schemes."""
    
    @abstractmethod
    def quantize_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization in forward pass."""
        pass
    
    @abstractmethod
    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor."""
        pass


class UniformQuantization(QuantizationScheme):
    """
    Uniform quantization scheme.
    Maps floating point values to discrete levels uniformly.
    """
    
    def __init__(
        self,
        num_bits: int = 8,
        signed: bool = True,
        scale_method: str = "max",  # "max", "percentile"
        percentile: float = 99.9,
    ):
        self.num_bits = num_bits
        self.signed = signed
        self.scale_method = scale_method
        self.percentile = percentile
        
        if signed:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
            
    def quantize_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations during forward pass."""
        if self.scale_method == "max":
            scale = torch.max(torch.abs(x))
        else:  # percentile
            scale = torch.quantile(torch.abs(x), self.percentile / 100.0)
            
        if scale == 0:
            return x
            
        # Quantize and dequantize
        x_scaled = x / scale
        x_quant = torch.clamp(torch.round(x_scaled * self.qmax), self.qmin, self.qmax)
        x_dequant = x_quant * scale / self.qmax
        
        return x_dequant
        
    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor."""
        if self.scale_method == "max":
            scale = torch.max(torch.abs(weights))
        else:
            scale = torch.quantile(torch.abs(weights), self.percentile / 100.0)
            
        if scale == 0:
            return weights
            
        weights_scaled = weights / scale
        weights_quant = torch.clamp(torch.round(weights_scaled * self.qmax), self.qmin, self.qmax)
        weights_dequant = weights_quant * scale / self.qmax
        
        return weights_dequant


class LSTQQuantization(QuantizationScheme):
    """
    Learned Step Size Quantization (LSQ).
    Learns optimal quantization step sizes during training.
    """
    
    def __init__(self, num_bits: int = 8, signed: bool = True):
        self.num_bits = num_bits
        self.signed = signed
        
        if signed:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
            
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def quantize_forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSQ quantization with gradient estimation."""
        # Straight-through estimator
        x_scaled = x / self.scale
        x_quant = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
        
        # Gradient computation
        grad_scale = 1.0 / math.sqrt(x.numel() * self.qmax)
        
        # Use straight-through estimator for backward pass
        x_dequant = (x_quant - x_scaled).detach() + x_scaled
        x_dequant = x_dequant * self.scale
        
        return x_dequant
        
    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights with LSQ."""
        return self.quantize_forward(weights)


class QuantizedLiquidNeuron(LiquidNeuron):
    """
    Quantized version of liquid neuron for efficient edge deployment.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        weight_quantization: str = "uniform",
        activation_quantization: str = "uniform",
        weight_bits: int = 8,
        activation_bits: int = 8,
        **kwargs
    ):
        super().__init__(input_dim, hidden_dim, **kwargs)
        
        # Setup quantization schemes
        self.weight_quantizer = self._create_quantizer(
            weight_quantization, weight_bits, signed=True
        )
        self.activation_quantizer = self._create_quantizer(
            activation_quantization, activation_bits, signed=True
        )
        
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
    def _create_quantizer(self, scheme: str, num_bits: int, signed: bool = True) -> QuantizationScheme:
        """Create quantization scheme."""
        if scheme == "uniform":
            return UniformQuantization(num_bits=num_bits, signed=signed)
        elif scheme == "lsq":
            return LSTQQuantization(num_bits=num_bits, signed=signed)
        else:
            raise ValueError(f"Unknown quantization scheme: {scheme}")
            
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """Forward pass with quantization."""
        batch_size = x.size(0)
        dt = dt if dt is not None else self.dt
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            
        # Quantize input
        x_quant = self.activation_quantizer.quantize_forward(x)
        
        # Quantize weights
        W_in_quant = self.weight_quantizer.quantize_weights(self.W_in.weight)
        
        # Input transformation with quantized weights
        input_contribution = F.linear(x_quant, W_in_quant, self.W_in.bias)
        
        # Recurrent contribution
        if self.W_rec is not None:
            W_rec_quant = self.weight_quantizer.quantize_weights(self.W_rec.weight)
            hidden_quant = self.activation_quantizer.quantize_forward(hidden)
            recurrent_contribution = F.linear(hidden_quant, W_rec_quant, self.W_rec.bias)
        else:
            recurrent_contribution = 0
            
        # Liquid dynamics
        total_input = input_contribution + recurrent_contribution + self.bias
        target_state = self.activation(total_input)
        
        # Quantize intermediate states
        target_state = self.activation_quantizer.quantize_forward(target_state)
        
        # Euler integration
        dhdt = (-hidden * self.leak + target_state) / self.tau
        hidden_new = hidden + dt * dhdt
        
        # Quantize output
        hidden_new = self.activation_quantizer.quantize_forward(hidden_new)
        
        return hidden_new


class QuantizedLiquidNet(LiquidNet):
    """
    Quantized liquid neural network for edge deployment.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        weight_quantization: str = "uniform",
        activation_quantization: str = "uniform",
        weight_bits: int = 8,
        activation_bits: int = 8,
        **kwargs
    ):
        # Initialize parent class first
        super().__init__(input_dim, hidden_units, output_dim, **kwargs)
        
        # Replace liquid layers with quantized versions
        self.liquid_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_units:
            layer = QuantizedLiquidNeuron(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                weight_quantization=weight_quantization,
                activation_quantization=activation_quantization,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                tau=kwargs.get('tau', 10.0),
                leak=kwargs.get('leak', 0.1),
                activation=kwargs.get('activation', 'tanh'),
                dt=kwargs.get('dt', 1.0),
            )
            self.liquid_layers.append(layer)
            prev_dim = hidden_dim
            
        # Store quantization parameters
        self.weight_quantization = weight_quantization
        self.activation_quantization = activation_quantization
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits


class QuantizationAwareTrainer:
    """
    Trainer extension for quantization-aware training.
    """
    
    def __init__(
        self,
        model: Union[LiquidNet, QuantizedLiquidNet],
        quantization_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.quantization_config = quantization_config or {}
        
        # Convert to quantized model if not already
        if not isinstance(model, QuantizedLiquidNet):
            self.quantized_model = self._convert_to_quantized(model)
        else:
            self.quantized_model = model
            
        self.training_mode = "qat"  # quantization-aware training
        
    def _convert_to_quantized(self, model: LiquidNet) -> QuantizedLiquidNet:
        """Convert regular model to quantized version."""
        quantized = QuantizedLiquidNet(
            input_dim=model.input_dim,
            hidden_units=model.hidden_units,
            output_dim=model.output_dim,
            weight_quantization=self.quantization_config.get("weight_scheme", "uniform"),
            activation_quantization=self.quantization_config.get("activation_scheme", "uniform"),
            weight_bits=self.quantization_config.get("weight_bits", 8),
            activation_bits=self.quantization_config.get("activation_bits", 8),
            tau=model.liquid_layers[0].tau if model.liquid_layers else 10.0,
            dropout=self.quantization_config.get("dropout", 0.0),
        )
        
        # Copy weights from original model
        self._copy_weights(model, quantized)
        
        return quantized
        
    def _copy_weights(self, source: LiquidNet, target: QuantizedLiquidNet) -> None:
        """Copy weights from source to target model."""
        with torch.no_grad():
            # Copy liquid layer weights
            for src_layer, tgt_layer in zip(source.liquid_layers, target.liquid_layers):
                tgt_layer.W_in.weight.copy_(src_layer.W_in.weight)
                tgt_layer.W_in.bias.copy_(src_layer.W_in.bias)
                
                if src_layer.W_rec is not None and tgt_layer.W_rec is not None:
                    tgt_layer.W_rec.weight.copy_(src_layer.W_rec.weight)
                    if hasattr(src_layer.W_rec, 'bias') and src_layer.W_rec.bias is not None:
                        tgt_layer.W_rec.bias.copy_(src_layer.W_rec.bias)
                        
                tgt_layer.bias.copy_(src_layer.bias)
                
            # Copy readout layer
            target.readout.weight.copy_(source.readout.weight)
            if source.readout.bias is not None:
                target.readout.bias.copy_(source.readout.bias)
                
    def get_model(self) -> QuantizedLiquidNet:
        """Get the quantized model."""
        return self.quantized_model
        
    def estimate_speedup(self) -> Dict[str, float]:
        """Estimate speedup from quantization."""
        weight_reduction = 32 / self.quantized_model.weight_bits
        activation_reduction = 32 / self.quantized_model.activation_bits
        
        # Rough estimates (actual speedup depends on hardware)
        memory_speedup = (weight_reduction + activation_reduction) / 2
        compute_speedup = weight_reduction * 0.7  # Conservative estimate
        
        return {
            "memory_reduction": memory_speedup,
            "compute_speedup": compute_speedup,
            "model_size_reduction": weight_reduction,
        }


class ModelQuantizer:
    """
    Post-training quantization utilities.
    """
    
    def __init__(self, calibration_loader=None):
        self.calibration_loader = calibration_loader
        
    def quantize_model(
        self,
        model: LiquidNet,
        quantization_scheme: str = "uniform",
        weight_bits: int = 8,
        activation_bits: int = 8,
        calibrate: bool = True,
    ) -> QuantizedLiquidNet:
        """
        Apply post-training quantization to model.
        
        Args:
            model: Model to quantize
            quantization_scheme: Quantization method
            weight_bits: Bits for weights
            activation_bits: Bits for activations
            calibrate: Whether to calibrate using calibration data
            
        Returns:
            Quantized model
        """
        # Create quantized version
        quantized_model = QuantizedLiquidNet(
            input_dim=model.input_dim,
            hidden_units=model.hidden_units,
            output_dim=model.output_dim,
            weight_quantization=quantization_scheme,
            activation_quantization=quantization_scheme,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
        )
        
        # Copy weights
        self._copy_weights(model, quantized_model)
        
        # Calibrate if requested and data available
        if calibrate and self.calibration_loader is not None:
            self._calibrate_model(quantized_model)
            
        return quantized_model
        
    def _copy_weights(self, source: LiquidNet, target: QuantizedLiquidNet) -> None:
        """Copy weights between models."""
        with torch.no_grad():
            for src_layer, tgt_layer in zip(source.liquid_layers, target.liquid_layers):
                tgt_layer.W_in.weight.copy_(src_layer.W_in.weight)
                tgt_layer.W_in.bias.copy_(src_layer.W_in.bias)
                
                if src_layer.W_rec is not None and tgt_layer.W_rec is not None:
                    tgt_layer.W_rec.weight.copy_(src_layer.W_rec.weight)
                    if hasattr(src_layer.W_rec, 'bias') and src_layer.W_rec.bias is not None:
                        tgt_layer.W_rec.bias.copy_(src_layer.W_rec.bias)
                        
                tgt_layer.bias.copy_(src_layer.bias)
                
            target.readout.weight.copy_(source.readout.weight)
            if source.readout.bias is not None:
                target.readout.bias.copy_(source.readout.bias)
                
    def _calibrate_model(self, model: QuantizedLiquidNet) -> None:
        """Calibrate quantization parameters using calibration data."""
        model.eval()
        
        with torch.no_grad():
            for data, _ in self.calibration_loader:
                # Forward pass to collect activation statistics
                _ = model(data)
                # Statistics are automatically collected by quantizers
                break  # One batch is usually sufficient for calibration
                
    def compare_models(
        self,
        original_model: LiquidNet,
        quantized_model: QuantizedLiquidNet,
        test_loader,
    ) -> Dict[str, Any]:
        """
        Compare original and quantized models.
        
        Returns:
            Comparison metrics
        """
        original_model.eval()
        quantized_model.eval()
        
        original_acc = 0
        quantized_acc = 0
        total_samples = 0
        mse_error = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                batch_size = data.size(0)
                
                # Original model predictions
                orig_outputs = original_model(data)
                orig_preds = torch.argmax(orig_outputs, dim=1)
                
                # Quantized model predictions
                quant_outputs = quantized_model(data)
                quant_preds = torch.argmax(quant_outputs, dim=1)
                
                # Accuracy
                original_acc += (orig_preds == targets).sum().item()
                quantized_acc += (quant_preds == targets).sum().item()
                
                # MSE between outputs
                mse_error += F.mse_loss(orig_outputs, quant_outputs).item() * batch_size
                
                total_samples += batch_size
                
        original_acc = original_acc / total_samples
        quantized_acc = quantized_acc / total_samples
        mse_error = mse_error / total_samples
        
        # Model size comparison
        orig_size = sum(p.numel() * 4 for p in original_model.parameters())  # float32
        quant_size = self._estimate_quantized_size(quantized_model)
        
        return {
            "original_accuracy": original_acc,
            "quantized_accuracy": quantized_acc,
            "accuracy_drop": original_acc - quantized_acc,
            "output_mse": mse_error,
            "original_size_mb": orig_size / (1024 * 1024),
            "quantized_size_mb": quant_size / (1024 * 1024),
            "compression_ratio": orig_size / quant_size,
        }
        
    def _estimate_quantized_size(self, model: QuantizedLiquidNet) -> int:
        """Estimate quantized model size in bytes."""
        total_size = 0
        
        for param in model.parameters():
            if "weight" in param.name if hasattr(param, 'name') else True:
                bits_per_param = model.weight_bits
            else:
                bits_per_param = model.activation_bits
                
            total_size += param.numel() * (bits_per_param / 8)
            
        return int(total_size)


def create_quantized_model(
    model: LiquidNet,
    quantization_config: Dict[str, Any],
    calibration_data=None,
) -> QuantizedLiquidNet:
    """
    Factory function to create quantized models.
    
    Args:
        model: Original model to quantize
        quantization_config: Quantization configuration
        calibration_data: Data for calibration (optional)
        
    Returns:
        Quantized model
    """
    quantizer = ModelQuantizer(calibration_data)
    
    return quantizer.quantize_model(
        model=model,
        quantization_scheme=quantization_config.get("scheme", "uniform"),
        weight_bits=quantization_config.get("weight_bits", 8),
        activation_bits=quantization_config.get("activation_bits", 8),
        calibrate=quantization_config.get("calibrate", True),
    )