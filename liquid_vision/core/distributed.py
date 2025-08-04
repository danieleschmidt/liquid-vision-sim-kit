"""
Distributed training and multi-device support for liquid neural networks.
Implements data and model parallelism optimizations.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Tuple, Any, Union
import os
from pathlib import Path

from .liquid_neurons import LiquidNet


class DistributedLiquidNet(nn.Module):
    """
    Distributed wrapper for liquid neural networks.
    Supports data and model parallelism across multiple devices.
    """
    
    def __init__(
        self,
        model: LiquidNet,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        find_unused_parameters: bool = True,
    ):
        super().__init__()
        
        self.model = model
        self.device_ids = device_ids
        self.output_device = output_device
        
        # Wrap with DDP if distributed training is initialized
        if dist.is_initialized():
            self.ddp_model = DDP(
                model,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.ddp_model = model
            
    def forward(self, *args, **kwargs):
        """Forward pass through distributed model."""
        return self.ddp_model(*args, **kwargs)
        
    def reset_states(self):
        """Reset liquid states across all devices."""
        if hasattr(self.ddp_model, 'module'):
            # DDP case
            self.ddp_model.module.reset_states()
        else:
            # Non-distributed case
            self.ddp_model.reset_states()
            
    def get_liquid_states(self):
        """Get liquid states from the underlying model."""
        if hasattr(self.ddp_model, 'module'):
            return self.ddp_model.module.get_liquid_states()
        else:
            return self.ddp_model.get_liquid_states()


class ModelParallelLiquidNet(nn.Module):
    """
    Model parallel implementation for very large liquid networks.
    Splits layers across multiple devices.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        device_map: Optional[Dict[int, str]] = None,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.num_layers = len(hidden_units)
        
        # Default device mapping
        if device_map is None:
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            device_map = self._create_default_device_map(num_devices)
            
        self.device_map = device_map
        
        # Create layers on specified devices
        self.liquid_layers = nn.ModuleList()
        self.layer_devices = []
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_units):
            device = device_map.get(i, 'cpu')
            
            # Import here to avoid circular imports
            from .liquid_neurons import LiquidNeuron
            
            layer = LiquidNeuron(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                tau=kwargs.get('tau', 10.0),
                leak=kwargs.get('leak', 0.1),
                activation=kwargs.get('activation', 'tanh'),
                dt=kwargs.get('dt', 1.0),
            ).to(device)
            
            self.liquid_layers.append(layer)
            self.layer_devices.append(device)
            prev_dim = hidden_dim
            
        # Readout layer
        readout_device = device_map.get(self.num_layers, self.layer_devices[-1])
        self.readout = nn.Linear(hidden_units[-1], output_dim).to(readout_device)
        self.readout_device = readout_device
        
        # Hidden states storage
        self.hidden_states = [None] * self.num_layers
        
    def _create_default_device_map(self, num_devices: int) -> Dict[int, str]:
        """Create default device mapping."""
        if num_devices <= 1:
            return {i: 'cpu' for i in range(self.num_layers + 1)}
            
        device_map = {}
        layers_per_device = max(1, self.num_layers // num_devices)
        
        for i in range(self.num_layers):
            device_id = min(i // layers_per_device, num_devices - 1)
            device_map[i] = f'cuda:{device_id}'
            
        # Place readout on last device
        device_map[self.num_layers] = f'cuda:{num_devices - 1}'
        
        return device_map
        
    def forward(self, x: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """Forward pass with model parallelism."""
        if reset_state:
            self.reset_states()
            
        current_input = x
        
        # Process through liquid layers
        for i, layer in enumerate(self.liquid_layers):
            # Move input to layer's device
            device = self.layer_devices[i]
            current_input = current_input.to(device)
            
            # Forward through layer
            self.hidden_states[i] = layer(current_input, self.hidden_states[i])
            current_input = self.hidden_states[i]
            
        # Readout layer
        current_input = current_input.to(self.readout_device)
        output = self.readout(current_input)
        
        return output
        
    def reset_states(self):
        """Reset all hidden states."""
        self.hidden_states = [None] * self.num_layers
        
    def get_liquid_states(self):
        """Get current liquid states."""
        return self.hidden_states.copy()


class DataParallelTrainer:
    """
    Enhanced trainer with data parallelism support.
    """
    
    def __init__(
        self,
        model: LiquidNet,
        local_rank: int = 0,
        world_size: int = 1,
        backend: str = "nccl",
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.backend = backend
        
        # Initialize distributed training if needed
        if world_size > 1:
            self._init_distributed()
            
        # Setup device
        if torch.cuda.is_available() and world_size > 1:
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')
            
        # Move model to device and wrap with DDP
        self.model = model.to(self.device)
        if world_size > 1:
            self.model = DistributedLiquidNet(
                self.model,
                device_ids=[local_rank] if torch.cuda.is_available() else None
            )
            
    def _init_distributed(self):
        """Initialize distributed training."""
        if not dist.is_initialized():
            # Use environment variables or defaults
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '12355')
            
            # Initialize process group
            init_method = f'tcp://{master_addr}:{master_port}'
            dist.init_process_group(
                backend=self.backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.local_rank,
            )
            
    def create_distributed_sampler(self, dataset):
        """Create distributed sampler for dataset."""
        if self.world_size > 1:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )
        else:
            return None
            
    def all_reduce_gradients(self):
        """Manually reduce gradients across processes."""
        if self.world_size > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
                    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


class AsyncLiquidNet(nn.Module):
    """
    Asynchronous liquid neural network for pipeline parallelism.
    Allows overlapping computation across layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        pipeline_stages: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.pipeline_stages = pipeline_stages
        
        # Split layers into pipeline stages
        layers_per_stage = max(1, len(hidden_units) // pipeline_stages)
        
        self.stages = nn.ModuleList()
        self.stage_devices = []
        
        prev_dim = input_dim
        stage_start = 0
        
        for stage in range(pipeline_stages):
            stage_end = min(stage_start + layers_per_stage, len(hidden_units))
            stage_hidden_units = hidden_units[stage_start:stage_end]
            
            if not stage_hidden_units:
                break
                
            # Create stage
            stage_model = self._create_stage(prev_dim, stage_hidden_units, **kwargs)
            
            # Place on appropriate device
            device = f'cuda:{stage}' if torch.cuda.device_count() > stage else 'cpu'
            stage_model = stage_model.to(device)
            
            self.stages.append(stage_model)
            self.stage_devices.append(device)
            
            prev_dim = stage_hidden_units[-1]
            stage_start = stage_end
            
        # Readout layer on last device
        self.readout = nn.Linear(prev_dim, output_dim).to(self.stage_devices[-1])
        
    def _create_stage(self, input_dim: int, hidden_units: List[int], **kwargs):
        """Create a pipeline stage."""
        from .liquid_neurons import LiquidNet
        
        return LiquidNet(
            input_dim=input_dim,
            hidden_units=hidden_units,
            output_dim=hidden_units[-1],  # Output is last hidden dim
            **kwargs
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pipeline stages."""
        current_input = x
        
        for i, stage in enumerate(self.stages):
            # Move to stage device
            device = self.stage_devices[i]
            current_input = current_input.to(device)
            
            # Process through stage
            current_input = stage(current_input)
            
        # Final readout
        current_input = current_input.to(self.stage_devices[-1])
        output = self.readout(current_input)
        
        return output
        
    def reset_states(self):
        """Reset states in all stages."""
        for stage in self.stages:
            if hasattr(stage, 'reset_states'):
                stage.reset_states()


class GradientCompression:
    """
    Gradient compression for distributed training.
    Reduces communication overhead in distributed settings.
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.1,
        method: str = "topk",  # "topk", "randomk", "threshold"
    ):
        self.compression_ratio = compression_ratio
        self.method = method
        
    def compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compress gradients for communication."""
        compressed = {}
        
        for name, grad in gradients.items():
            if grad is None:
                compressed[name] = None
                continue
                
            if self.method == "topk":
                compressed[name] = self._topk_compress(grad)
            elif self.method == "randomk":
                compressed[name] = self._randomk_compress(grad)
            elif self.method == "threshold":
                compressed[name] = self._threshold_compress(grad)
            else:
                compressed[name] = grad
                
        return compressed
        
    def decompress_gradients(
        self, 
        compressed_gradients: Dict[str, Any],
        original_shapes: Dict[str, torch.Size]
    ) -> Dict[str, torch.Tensor]:
        """Decompress gradients after communication."""
        decompressed = {}
        
        for name, compressed in compressed_gradients.items():
            if compressed is None:
                decompressed[name] = None
                continue
                
            original_shape = original_shapes[name]
            
            if isinstance(compressed, dict):
                if compressed.get("method") == "topk":
                    decompressed[name] = self._topk_decompress(compressed, original_shape)
                elif compressed.get("method") == "randomk":
                    decompressed[name] = self._randomk_decompress(compressed, original_shape)
                elif compressed.get("method") == "threshold":
                    decompressed[name] = self._threshold_decompress(compressed, original_shape)
                else:
                    decompressed[name] = compressed
            else:
                decompressed[name] = compressed
                
        return decompressed
        
    def _topk_compress(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Top-k compression."""
        flat_tensor = tensor.flatten()
        k = max(1, int(len(flat_tensor) * self.compression_ratio))
        
        values, indices = torch.topk(torch.abs(flat_tensor), k)
        selected_values = flat_tensor[indices]
        
        return {
            "method": "topk",
            "values": selected_values,
            "indices": indices,
            "shape": tensor.shape,
        }
        
    def _topk_decompress(self, compressed: Dict[str, Any], shape: torch.Size) -> torch.Tensor:
        """Top-k decompression."""
        tensor = torch.zeros(shape).flatten()
        tensor[compressed["indices"]] = compressed["values"]
        return tensor.reshape(shape)
        
    def _randomk_compress(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Random-k compression."""
        flat_tensor = tensor.flatten()
        k = max(1, int(len(flat_tensor) * self.compression_ratio))
        
        indices = torch.randperm(len(flat_tensor))[:k]
        selected_values = flat_tensor[indices]
        
        return {
            "method": "randomk",
            "values": selected_values,
            "indices": indices,
            "shape": tensor.shape,
        }
        
    def _randomk_decompress(self, compressed: Dict[str, Any], shape: torch.Size) -> torch.Tensor:
        """Random-k decompression."""
        tensor = torch.zeros(shape).flatten()
        tensor[compressed["indices"]] = compressed["values"]
        return tensor.reshape(shape)
        
    def _threshold_compress(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Threshold-based compression."""
        flat_tensor = tensor.flatten()
        threshold = torch.quantile(torch.abs(flat_tensor), 1.0 - self.compression_ratio)
        
        mask = torch.abs(flat_tensor) >= threshold
        indices = torch.where(mask)[0]
        values = flat_tensor[indices]
        
        return {
            "method": "threshold",
            "values": values,
            "indices": indices,
            "shape": tensor.shape,
        }
        
    def _threshold_decompress(self, compressed: Dict[str, Any], shape: torch.Size) -> torch.Tensor:
        """Threshold decompression."""
        tensor = torch.zeros(shape).flatten()
        tensor[compressed["indices"]] = compressed["values"]
        return tensor.reshape(shape)


def setup_distributed_training(
    local_rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """
    Setup distributed training environment.
    
    Args:
        local_rank: Local rank of the process
        world_size: Total number of processes
        backend: Communication backend
        master_addr: Master node address
        master_port: Master node port
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=local_rank,
        world_size=world_size,
    )
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_distributed_training() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# Example usage functions
def create_distributed_model(
    input_dim: int,
    hidden_units: List[int],
    output_dim: int,
    parallelism_type: str = "data",  # "data", "model", "pipeline"
    **kwargs
) -> nn.Module:
    """
    Factory function to create distributed models.
    
    Args:
        input_dim: Input dimension
        hidden_units: Hidden layer sizes
        output_dim: Output dimension
        parallelism_type: Type of parallelism
        **kwargs: Additional model arguments
        
    Returns:
        Distributed model
    """
    if parallelism_type == "data":
        from .liquid_neurons import LiquidNet
        model = LiquidNet(input_dim, hidden_units, output_dim, **kwargs)
        return DistributedLiquidNet(model)
        
    elif parallelism_type == "model":
        return ModelParallelLiquidNet(input_dim, hidden_units, output_dim, **kwargs)
        
    elif parallelism_type == "pipeline":
        pipeline_stages = kwargs.pop("pipeline_stages", 2)
        return AsyncLiquidNet(
            input_dim, hidden_units, output_dim, 
            pipeline_stages=pipeline_stages, **kwargs
        )
        
    else:
        raise ValueError(f"Unknown parallelism type: {parallelism_type}")


if __name__ == "__main__":
    # Example distributed training setup
    print("Testing distributed components...")
    
    # Test model parallel version
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        
        model = ModelParallelLiquidNet(
            input_dim=100,
            hidden_units=[256, 128, 64],
            output_dim=10
        )
        
        print("Model parallel model created successfully")
        
        # Test forward pass
        test_input = torch.randn(4, 100)
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        
    else:
        print("CUDA not available or insufficient devices for model parallelism")
        
    print("Distributed components test completed!")