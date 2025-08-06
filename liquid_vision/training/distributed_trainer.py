"""
Distributed training implementation for liquid neural networks.
Supports multi-GPU, multi-node training with gradient synchronization.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
import time
import json
from pathlib import Path

from ..core.liquid_neurons import LiquidNet
from ..utils.logging import log_performance, performance_logger
from ..utils.validation import validate_inputs
from .liquid_trainer import LiquidTrainer, TrainingConfig


logger = logging.getLogger('liquid_vision.training.distributed')


class DistributedTrainingConfig:
    """Configuration for distributed training."""
    
    def __init__(
        self,
        backend: str = "nccl",  # nccl for GPU, gloo for CPU
        init_method: str = "tcp://localhost:23456",
        world_size: int = 1,
        rank: int = 0,
        local_rank: int = 0,
        find_unused_parameters: bool = True,
        gradient_compression: bool = False,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.find_unused_parameters = find_unused_parameters
        self.gradient_compression = gradient_compression
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        
        # Additional configuration
        for key, value in kwargs.items():
            setattr(self, key, value)


class DistributedLiquidTrainer:
    """
    Distributed trainer for liquid neural networks with advanced optimization.
    """
    
    def __init__(
        self,
        model: LiquidNet,
        training_config: TrainingConfig,
        distributed_config: DistributedTrainingConfig
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Liquid neural network model
            training_config: Training configuration
            distributed_config: Distributed training configuration
        """
        self.model = model
        self.training_config = training_config
        self.distributed_config = distributed_config
        
        # Initialize distributed training
        self._setup_distributed()
        
        # Setup device and model
        self.device = self._setup_device()
        self.model = self._setup_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if distributed_config.mixed_precision else None
        
        # Performance tracking
        self.training_metrics = {}
        
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        if not dist.is_available():
            raise RuntimeError("Distributed training not available")
        
        # Initialize process group
        dist.init_process_group(
            backend=self.distributed_config.backend,
            init_method=self.distributed_config.init_method,
            world_size=self.distributed_config.world_size,
            rank=self.distributed_config.rank
        )
        
        logger.info(
            f"Distributed training initialized: rank {self.distributed_config.rank} "
            f"of {self.distributed_config.world_size}"
        )
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.distributed_config.local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        logger.info(f"Training on device: {device}")
        return device
    
    def _setup_model(self) -> DDP:
        """Setup distributed model."""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing if configured
        if self.distributed_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Wrap in DDP
        ddp_model = DDP(
            self.model,
            device_ids=[self.distributed_config.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=self.distributed_config.find_unused_parameters
        )
        
        return ddp_model
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup distributed optimizer."""
        # Scale learning rate by world size
        base_lr = self.training_config.learning_rate
        scaled_lr = base_lr * self.distributed_config.world_size
        
        if self.training_config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=scaled_lr,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=scaled_lr,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=scaled_lr,
                momentum=0.9,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer}")
        
        logger.info(f"Optimizer setup: {self.training_config.optimizer} with lr={scaled_lr}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.training_config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.training_config.epochs
            )
        elif self.training_config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.training_config.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        elif self.training_config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10
            )
        else:
            scheduler = None
        
        return scheduler
    
    @log_performance("distributed_training")
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Train model with distributed training.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            callbacks: Training callbacks
            
        Returns:
            Training results
        """
        # Setup distributed data loaders
        train_loader = self._setup_distributed_dataloader(train_loader)
        if val_loader is not None:
            val_loader = self._setup_distributed_dataloader(val_loader)
        
        # Training loop
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.training_config.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, epoch)
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch results (only on rank 0)
            if self.distributed_config.rank == 0:
                epoch_time = time.time() - epoch_start_time
                
                training_history['train_loss'].append(train_metrics['loss'])
                training_history['train_accuracy'].append(train_metrics['accuracy'])
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_accuracy'].append(val_metrics['accuracy'])
                training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                training_history['epoch_times'].append(epoch_time)
                
                logger.info(
                    f"Epoch {epoch+1}/{self.training_config.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Save checkpoint if validation improved
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics)
        
        # Final synchronization
        dist.barrier()
        
        if self.distributed_config.rank == 0:
            logger.info("Distributed training completed successfully")
        
        return training_history
    
    def _setup_distributed_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Setup distributed data loader."""
        # Create distributed sampler
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.distributed_config.world_size,
            rank=self.distributed_config.rank,
            shuffle=True
        )
        
        # Create new dataloader with distributed sampler
        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=True  # Important for DDP
        )
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.training_config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.training_config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # Log progress
            if batch_idx % 100 == 0 and self.distributed_config.rank == 0:
                logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Synchronize metrics across all processes
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        # All-reduce for global metrics
        loss_tensor = torch.tensor(avg_loss).to(self.device)
        acc_tensor = torch.tensor(accuracy).to(self.device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        
        global_loss = loss_tensor.item() / self.distributed_config.world_size
        global_accuracy = acc_tensor.item() / self.distributed_config.world_size
        
        return {'loss': global_loss, 'accuracy': global_accuracy}
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        # Synchronize validation metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        loss_tensor = torch.tensor(avg_loss).to(self.device)
        acc_tensor = torch.tensor(accuracy).to(self.device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        
        global_loss = loss_tensor.item() / self.distributed_config.world_size
        global_accuracy = acc_tensor.item() / self.distributed_config.world_size
        
        return {'loss': global_loss, 'accuracy': global_accuracy}
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Save model checkpoint."""
        if self.distributed_config.rank != 0:
            return  # Only save on rank 0
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_config': self.training_config.__dict__,
            'distributed_config': self.distributed_config.__dict__
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = Path(self.training_config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


def launch_distributed_training(
    rank: int,
    world_size: int,
    model_fn: Callable,
    train_dataset,
    val_dataset,
    training_config: TrainingConfig,
    distributed_config: DistributedTrainingConfig
):
    """
    Launch distributed training on a single process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        model_fn: Function to create model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_config: Training configuration
        distributed_config: Distributed training configuration
    """
    # Update distributed config for this process
    distributed_config.rank = rank
    distributed_config.world_size = world_size
    distributed_config.local_rank = rank % torch.cuda.device_count()
    
    try:
        # Create model
        model = model_fn()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,  # Will be handled by DistributedSampler
            num_workers=training_config.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            pin_memory=torch.cuda.is_available()
        ) if val_dataset is not None else None
        
        # Create distributed trainer
        trainer = DistributedLiquidTrainer(
            model=model,
            training_config=training_config,
            distributed_config=distributed_config
        )
        
        # Train model
        results = trainer.train(train_loader, val_loader)
        
        # Cleanup
        trainer.cleanup()
        
        return results
        
    except Exception as e:
        logger.error(f"Distributed training failed on rank {rank}: {e}")
        raise


def train_distributed_multiprocessing(
    model_fn: Callable,
    train_dataset,
    val_dataset,
    training_config: TrainingConfig,
    distributed_config: Optional[DistributedTrainingConfig] = None
) -> Dict[str, Any]:
    """
    Launch distributed training using multiprocessing.
    
    Args:
        model_fn: Function to create model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_config: Training configuration
        distributed_config: Distributed training configuration
        
    Returns:
        Training results
    """
    if distributed_config is None:
        distributed_config = DistributedTrainingConfig()
    
    # Determine world size
    if torch.cuda.is_available():
        world_size = min(distributed_config.world_size, torch.cuda.device_count())
    else:
        world_size = 1
    
    if world_size == 1:
        # Single process training
        return launch_distributed_training(
            rank=0,
            world_size=1,
            model_fn=model_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
            distributed_config=distributed_config
        )
    
    # Multi-process training
    logger.info(f"Starting distributed training with {world_size} processes")
    
    # Launch processes
    mp.spawn(
        launch_distributed_training,
        args=(
            world_size,
            model_fn,
            train_dataset,
            val_dataset, 
            training_config,
            distributed_config
        ),
        nprocs=world_size,
        join=True
    )
    
    logger.info("Distributed training completed")
    return {"status": "completed", "world_size": world_size}


# GPU memory optimization utilities
class GPUMemoryOptimizer:
    """Utilities for optimizing GPU memory usage."""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {"available": 0, "total": 0}
        
        available = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        
        return {
            "total": available / 1024**3,  # GB
            "reserved": reserved / 1024**3,
            "allocated": allocated / 1024**3,
            "free": (available - reserved) / 1024**3
        }
    
    @staticmethod
    def optimize_batch_size(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        max_memory_gb: float = 10.0
    ) -> int:
        """
        Find optimal batch size for given memory constraint.
        
        Args:
            model: Model to optimize for
            input_shape: Input tensor shape (without batch dimension)
            max_memory_gb: Maximum memory to use in GB
            
        Returns:
            Optimal batch size
        """
        device = next(model.parameters()).device
        
        # Start with batch size 1
        batch_size = 1
        max_batch_size = 1
        
        model.eval()
        
        while batch_size <= 512:  # Reasonable upper limit
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Test batch
                test_input = torch.randn(batch_size, *input_shape).to(device)
                
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                memory_info = GPUMemoryOptimizer.get_memory_info()
                if memory_info["allocated"] < max_memory_gb:
                    max_batch_size = batch_size
                    batch_size *= 2
                else:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise
        
        # Clean up
        del test_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return max_batch_size