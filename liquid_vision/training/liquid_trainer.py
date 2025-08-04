"""
Training framework for liquid neural networks with event-based data.
Supports various training configurations and optimization strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import logging
from tqdm import tqdm

from ..core.liquid_neurons import LiquidNet
from .event_dataloader import EventDataLoader
from .losses import LiquidLoss, TemporalLoss


@dataclass 
class TrainingConfig:
    """Configuration for liquid neural network training."""
    
    # Model parameters
    model_name: str = "liquid_net"
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    gradient_clip: Optional[float] = 1.0
    
    # Liquid-specific parameters
    dt: float = 1.0  # Time step for liquid dynamics
    reset_state_frequency: int = 10  # Reset liquid states every N batches
    
    # Loss function
    loss_type: str = "cross_entropy"  # "cross_entropy", "mse", "liquid", "temporal"
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Optimizer
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    scheduler: Optional[str] = "cosine"  # "step", "cosine", "plateau", None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    dropout: float = 0.1
    liquid_regularization: float = 0.01  # Regularization on liquid states
    
    # Validation
    validation_frequency: int = 5  # Validate every N epochs
    early_stopping_patience: int = 15
    
    # Logging and checkpointing
    log_frequency: int = 10  # Log every N batches
    checkpoint_frequency: int = 10  # Save checkpoint every N epochs
    save_best_only: bool = True
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    mixed_precision: bool = False
    
    # Quantization
    quantization_aware: bool = False
    quantization_backend: str = "fbgemm"
    
    # Output directories
    output_dir: str = "experiments"
    experiment_name: Optional[str] = None


class LiquidTrainer:
    """
    Trainer for liquid neural networks with event-based vision data.
    Handles training loop, validation, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: LiquidNet,
        config: TrainingConfig,
        train_loader: EventDataLoader,
        val_loader: Optional[EventDataLoader] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or self._setup_logger()
        
        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = self._setup_loss()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision if enabled
        self.scaler = None
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            
        # Setup output directories
        self.output_dir = Path(self.config.output_dir)
        if self.config.experiment_name:
            self.output_dir = self.output_dir / self.config.experiment_name
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.output_dir / f"{self.config.model_name}_{timestamp}"
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Save config
        self._save_config()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup training logger."""
        logger = logging.getLogger(f"LiquidTrainer_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
            
        self.logger.info(f"Using device: {device}")
        return device
        
    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        if self.config.loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(**self.config.loss_kwargs)
        elif self.config.loss_type == "mse":
            return nn.MSELoss(**self.config.loss_kwargs)
        elif self.config.loss_type == "liquid":
            return LiquidLoss(**self.config.loss_kwargs)
        elif self.config.loss_type == "temporal":
            return TemporalLoss(**self.config.loss_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
            
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        params = self.model.parameters()
        
        if self.config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.scheduler is None:
            return None
            
        if self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_kwargs.get("step_size", 30),
                gamma=self.config.scheduler_kwargs.get("gamma", 0.1)
            )
        elif self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                **self.config.scheduler_kwargs
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                **self.config.scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
            
    def _save_config(self) -> None:
        """Save training configuration."""
        config_dict = {
            "model_info": {
                "type": self.model.__class__.__name__,
                "input_dim": self.model.input_dim,
                "hidden_units": self.model.hidden_units,
                "output_dim": self.model.output_dim,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
            },
            "training_config": {
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "optimizer": self.config.optimizer,
                "loss_type": self.config.loss_type,
                "device": str(self.device),
            }
        }
        
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
            
    def fit(
        self,
        epochs: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the liquid neural network.
        
        Args:
            epochs: Number of epochs to train (overrides config)
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training history dictionary
        """
        if epochs is not None:
            self.config.epochs = epochs
            
        if resume_from is not None:
            self._load_checkpoint(resume_from)
            
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        self.logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                
                # Training
                train_metrics = self._train_epoch()
                self.train_metrics.append(train_metrics)
                
                # Validation
                if (self.val_loader is not None and 
                    epoch % self.config.validation_frequency == 0):
                    val_metrics = self._validate_epoch()
                    self.val_metrics.append(val_metrics)
                    
                    # Early stopping check
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.patience_counter = 0
                        
                        if self.config.save_best_only:
                            self._save_checkpoint("best_model.pth")
                    else:
                        self.patience_counter += 1
                        
                    if self.patience_counter >= self.config.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                        
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        val_loss = self.val_metrics[-1]["loss"] if self.val_metrics else train_metrics["loss"]
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                        
                # Checkpointing
                if epoch % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            
        # Final checkpoint
        self._save_checkpoint("final_model.pth")
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Close tensorboard writer
        self.writer.close()
        
        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics
        }
        
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Reset liquid states periodically
            if batch_idx % self.config.reset_state_frequency == 0:
                self.model.reset_states()
                
            # Forward pass
            if self.config.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data, dt=self.config.dt)
                    loss = self.criterion(outputs, targets)
                    
                    # Add liquid regularization
                    if self.config.liquid_regularization > 0:
                        liquid_states = self.model.get_liquid_states()
                        reg_loss = sum(
                            torch.norm(state) for state in liquid_states if state is not None
                        )
                        loss = loss + self.config.liquid_regularization * reg_loss
                        
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data, dt=self.config.dt)
                loss = self.criterion(outputs, targets)
                
                # Add liquid regularization
                if self.config.liquid_regularization > 0:
                    liquid_states = self.model.get_liquid_states()
                    reg_loss = sum(
                        torch.norm(state) for state in liquid_states if state is not None
                    )
                    loss = loss + self.config.liquid_regularization * reg_loss
                    
                # Backward pass
                loss.backward()
                
                if self.config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            # Calculate accuracy for classification tasks
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs.data, 1)
                if targets.dim() == 1:  # Classification labels
                    correct_predictions += (predicted == targets).sum().item()
                    
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
                self.writer.add_scalar("Train/LearningRate", current_lr, self.global_step)
                
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / total_samples:.4f}'
            })
            
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if correct_predictions > 0 else 0.0
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
        
        self.logger.info(
            f"Epoch {self.current_epoch + 1}: "
            f"Train Loss: {avg_loss:.4f}, "
            f"Train Acc: {accuracy:.4f}"
        )
        
        return metrics
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation"):
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Reset states for validation
                self.model.reset_states()
                
                outputs = self.model(data, dt=self.config.dt)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # Calculate accuracy
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    _, predicted = torch.max(outputs.data, 1)
                    if targets.dim() == 1:
                        correct_predictions += (predicted == targets).sum().item()
                        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if correct_predictions > 0 else 0.0
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
        
        # Log validation metrics
        self.writer.add_scalar("Val/Loss", avg_loss, self.current_epoch)
        self.writer.add_scalar("Val/Accuracy", accuracy, self.current_epoch)
        
        self.logger.info(
            f"Epoch {self.current_epoch + 1}: "
            f"Val Loss: {avg_loss:.4f}, "
            f"Val Acc: {accuracy:.4f}"
        )
        
        return metrics
        
    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
            
        torch.save(checkpoint, self.output_dir / filename)
        self.logger.info(f"Checkpoint saved: {filename}")
        
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_metrics = checkpoint.get("train_metrics", [])
        self.val_metrics = checkpoint.get("val_metrics", [])
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        
    def evaluate(self, test_loader: EventDataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Testing"):
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                self.model.reset_states()
                outputs = self.model(data, dt=self.config.dt)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # Store predictions and targets
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    if targets.dim() == 1:
                        correct_predictions += (predicted == targets).sum().item()
                        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if correct_predictions > 0 else 0.0
        
        metrics = {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "predictions": all_predictions,
            "targets": all_targets,
        }
        
        self.logger.info(f"Test Results: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return metrics