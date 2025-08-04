"""
Unit tests for training components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from liquid_vision.training.liquid_trainer import LiquidTrainer, TrainingConfig
from liquid_vision.training.event_dataloader import (
    EventDataset, EventDataLoader, SyntheticEventDataset
)
from liquid_vision.training.losses import (
    TemporalLoss, LiquidLoss, ContrastiveLoss, EventSequenceLoss,
    create_loss_function
)
from liquid_vision.core.liquid_neurons import LiquidNet


class TestTrainingConfig:
    """Test cases for TrainingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.epochs == 100
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.optimizer == "adam"
        assert config.device == "auto"
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            epochs=50,
            learning_rate=1e-4,
            batch_size=16,
            optimizer="sgd",
            loss_type="liquid"
        )
        
        assert config.epochs == 50
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.optimizer == "sgd"
        assert config.loss_type == "liquid"


class TestSyntheticEventDataset:
    """Test cases for synthetic event dataset."""
    
    def test_classification_dataset(self):
        """Test synthetic classification dataset."""
        dataset = SyntheticEventDataset(
            num_samples=50,
            resolution=(32, 32),
            task_type="classification",
            encoder_type="temporal"
        )
        
        assert len(dataset) == 50
        
        # Test sample
        encoded, label = dataset[0]
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert 0 <= label.item() <= 2  # 3 classes (circle, square, triangle)
        
    def test_counting_dataset(self):
        """Test synthetic counting dataset."""
        dataset = SyntheticEventDataset(
            num_samples=25,
            resolution=(48, 48),
            task_type="counting",
            encoder_type="spatial"
        )
        
        assert len(dataset) == 25
        
        encoded, label = dataset[0]
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.float32
        assert 1 <= label.item() <= 5  # 1-5 objects
        
    def test_motion_dataset(self):
        """Test synthetic motion detection dataset."""
        dataset = SyntheticEventDataset(
            num_samples=20,
            resolution=(64, 64),
            task_type="motion",
            encoder_type="timeslice"
        )
        
        assert len(dataset) == 20
        
        encoded, label = dataset[0]
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert 0 <= label.item() <= 3  # 4 motion directions
        
    def test_different_encoders(self):
        """Test dataset with different encoders."""
        encoder_types = ["temporal", "spatial", "timeslice", "adaptive"]
        
        for encoder_type in encoder_types:
            dataset = SyntheticEventDataset(
                num_samples=10,
                resolution=(32, 32),
                task_type="classification",
                encoder_type=encoder_type
            )
            
            encoded, label = dataset[0]
            assert isinstance(encoded, torch.Tensor)
            assert encoded.dim() >= 2  # At least 2D


class TestEventDataLoader:
    """Test cases for event data loader."""
    
    def test_dataloader_creation(self):
        """Test creation of event data loader."""
        dataset = SyntheticEventDataset(
            num_samples=100,
            resolution=(64, 48),
            task_type="classification"
        )
        
        dataloader = EventDataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        assert len(dataloader) == 100 // 8
        
        # Test iteration
        for batch_data, batch_labels in dataloader:
            assert batch_data.shape[0] == 8  # Batch size
            assert batch_labels.shape[0] == 8
            break  # Just test first batch
            
    def test_temporal_batching(self):
        """Test temporal batching functionality."""
        dataset = SyntheticEventDataset(
            num_samples=24,
            resolution=(32, 32),
            task_type="classification"
        )
        
        dataloader = EventDataLoader(
            dataset=dataset,
            batch_size=4,
            temporal_batching=True
        )
        
        # Should work without errors
        for batch_data, batch_labels in dataloader:
            assert batch_data.shape[0] == 4
            break


class TestLossFunctions:
    """Test cases for loss functions."""
    
    def test_temporal_loss(self):
        """Test temporal loss function."""
        loss_fn = TemporalLoss(
            base_loss="cross_entropy",
            temporal_weight=0.1,
            smoothness_weight=0.05
        )
        
        batch_size, seq_len, num_classes = 4, 10, 3
        
        # Test with sequence output
        outputs = torch.randn(batch_size, seq_len, num_classes)
        targets = torch.randint(0, num_classes, (batch_size, seq_len))
        
        loss = loss_fn(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        
        # Test with single output
        outputs_single = torch.randn(batch_size, num_classes)
        targets_single = torch.randint(0, num_classes, (batch_size,))
        
        loss_single = loss_fn(outputs_single, targets_single)
        assert isinstance(loss_single, torch.Tensor)
        assert loss_single.item() > 0
        
    def test_liquid_loss(self):
        """Test liquid loss function."""
        loss_fn = LiquidLoss(
            base_loss="cross_entropy",
            stability_weight=0.01,
            sparsity_weight=0.001,
            energy_weight=0.005
        )
        
        batch_size, num_classes = 6, 4
        outputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Test without liquid states
        loss = loss_fn(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        
        # Test with liquid states
        liquid_states = [
            torch.randn(batch_size, 20),
            torch.randn(batch_size, 15)
        ]
        
        loss_with_states = loss_fn(outputs, targets, liquid_states=liquid_states)
        assert isinstance(loss_with_states, torch.Tensor)
        assert loss_with_states.item() > loss.item()  # Should be higher due to regularization
        
    def test_contrastive_loss(self):
        """Test contrastive loss function."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        batch_size, embedding_dim = 8, 64
        embeddings = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, 3, (batch_size,))
        
        loss = loss_fn(embeddings, labels=labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        
    def test_sequence_loss(self):
        """Test event sequence loss."""
        loss_fn = EventSequenceLoss(
            base_loss="cross_entropy",
            sequence_weight=1.0,
            alignment_weight=0.1
        )
        
        batch_size, max_seq_len, num_classes = 4, 12, 5
        predictions = torch.randn(batch_size, max_seq_len, num_classes)
        targets = torch.randint(0, num_classes, (batch_size, max_seq_len))
        
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        
    def test_loss_factory(self):
        """Test loss function factory."""
        loss_types = ["cross_entropy", "mse", "temporal", "liquid", "contrastive", "sequence"]
        
        for loss_type in loss_types:
            loss_fn = create_loss_function(loss_type)
            assert isinstance(loss_fn, nn.Module)
            
        # Test invalid loss type
        with pytest.raises(ValueError):
            create_loss_function("invalid_loss")


class TestLiquidTrainer:
    """Test cases for liquid trainer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def sample_model(self):
        """Create sample liquid network for testing."""
        return LiquidNet(
            input_dim=10,
            hidden_units=[20, 15],
            output_dim=3
        )
        
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        dataset = SyntheticEventDataset(
            num_samples=100,
            resolution=(32, 32),
            task_type="classification",
            encoder_type="temporal"
        )
        
        train_loader = EventDataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create smaller validation set
        val_dataset = SyntheticEventDataset(
            num_samples=30,
            resolution=(32, 32),
            task_type="classification",
            encoder_type="temporal"
        )
        val_loader = EventDataLoader(val_dataset, batch_size=8, shuffle=False)
        
        return train_loader, val_loader
        
    def test_trainer_initialization(self, sample_model, sample_data, temp_dir):
        """Test trainer initialization."""
        train_loader, val_loader = sample_data
        
        config = TrainingConfig(
            epochs=5,
            batch_size=8,
            output_dir=temp_dir,
            experiment_name="test_experiment"
        )
        
        trainer = LiquidTrainer(
            model=sample_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        assert trainer.model is sample_model
        assert trainer.config is config
        assert trainer.train_loader is train_loader
        assert trainer.val_loader is val_loader
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        
    def test_device_setup(self, sample_model, sample_data, temp_dir):
        """Test device setup."""
        train_loader, _ = sample_data
        
        # Test auto device selection
        config = TrainingConfig(device="auto", output_dir=temp_dir)
        trainer = LiquidTrainer(sample_model, config, train_loader)
        
        assert trainer.device.type in ["cpu", "cuda", "mps"]
        
        # Test specific device
        config_cpu = TrainingConfig(device="cpu", output_dir=temp_dir)
        trainer_cpu = LiquidTrainer(sample_model, config_cpu, train_loader)
        
        assert trainer_cpu.device.type == "cpu"
        
    def test_optimizer_setup(self, sample_model, sample_data, temp_dir):
        """Test optimizer setup."""
        train_loader, _ = sample_data
        
        optimizers = ["adam", "adamw", "sgd"]
        
        for opt_name in optimizers:
            config = TrainingConfig(optimizer=opt_name, output_dir=temp_dir)
            trainer = LiquidTrainer(sample_model, config, train_loader)
            
            assert trainer.optimizer is not None
            
        # Test invalid optimizer
        with pytest.raises(ValueError):
            config = TrainingConfig(optimizer="invalid", output_dir=temp_dir)
            LiquidTrainer(sample_model, config, train_loader)
            
    def test_scheduler_setup(self, sample_model, sample_data, temp_dir):
        """Test learning rate scheduler setup."""
        train_loader, _ = sample_data
        
        schedulers = ["step", "cosine", "plateau", None]
        
        for sched_name in schedulers:
            config = TrainingConfig(scheduler=sched_name, output_dir=temp_dir)
            trainer = LiquidTrainer(sample_model, config, train_loader)
            
            if sched_name is None:
                assert trainer.scheduler is None
            else:
                assert trainer.scheduler is not None
                
    def test_loss_setup(self, sample_model, sample_data, temp_dir):
        """Test loss function setup."""
        train_loader, _ = sample_data
        
        loss_types = ["cross_entropy", "mse", "liquid", "temporal"]
        
        for loss_type in loss_types:
            config = TrainingConfig(loss_type=loss_type, output_dir=temp_dir)
            trainer = LiquidTrainer(sample_model, config, train_loader)
            
            assert trainer.criterion is not None
            
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_training_single_epoch(self, mock_writer, sample_model, sample_data, temp_dir):
        """Test single epoch training."""
        train_loader, _ = sample_data
        
        config = TrainingConfig(
            epochs=1,
            batch_size=8,
            log_frequency=1,
            output_dir=temp_dir
        )
        
        trainer = LiquidTrainer(sample_model, config, train_loader)
        
        # Mock the writer to avoid actual tensorboard calls
        trainer.writer = Mock()
        
        # Train single epoch
        metrics = trainer._train_epoch()
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0 <= metrics["accuracy"] <= 1
        
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_validation_epoch(self, mock_writer, sample_model, sample_data, temp_dir):
        """Test validation epoch."""
        train_loader, val_loader = sample_data
        
        config = TrainingConfig(output_dir=temp_dir)
        trainer = LiquidTrainer(sample_model, config, train_loader, val_loader)
        trainer.writer = Mock()
        
        metrics = trainer._validate_epoch()
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0 <= metrics["accuracy"] <= 1
        
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_full_training_loop(self, mock_writer, sample_model, sample_data, temp_dir):
        """Test complete training loop."""
        train_loader, val_loader = sample_data
        
        config = TrainingConfig(
            epochs=2,
            batch_size=8,
            validation_frequency=1,
            checkpoint_frequency=1,
            output_dir=temp_dir
        )
        
        trainer = LiquidTrainer(sample_model, config, train_loader, val_loader)
        trainer.writer = Mock()
        
        # Run training
        history = trainer.fit()
        
        assert "train_metrics" in history
        assert "val_metrics" in history
        assert len(history["train_metrics"]) == 2  # 2 epochs
        assert len(history["val_metrics"]) == 2   # Validated every epoch
        
    def test_checkpoint_save_load(self, sample_model, sample_data, temp_dir):
        """Test checkpoint saving and loading."""
        train_loader, _ = sample_data
        
        config = TrainingConfig(epochs=1, output_dir=temp_dir)
        trainer = LiquidTrainer(sample_model, config, train_loader)
        trainer.writer = Mock()
        
        # Save initial state
        initial_state = trainer.model.state_dict()
        
        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"
        trainer._save_checkpoint(checkpoint_path.name)
        
        assert checkpoint_path.exists()
        
        # Modify model
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.add_(1.0)
                
        # Load checkpoint
        trainer._load_checkpoint(str(checkpoint_path))
        
        # Model should be restored
        loaded_state = trainer.model.state_dict()
        for key in initial_state:
            assert torch.allclose(initial_state[key], loaded_state[key])
            
    def test_evaluation(self, sample_model, sample_data, temp_dir):
        """Test model evaluation."""
        train_loader, val_loader = sample_data
        
        config = TrainingConfig(output_dir=temp_dir)
        trainer = LiquidTrainer(sample_model, config, train_loader)
        trainer.writer = Mock()
        
        metrics = trainer.evaluate(val_loader)
        
        assert "test_loss" in metrics
        assert "test_accuracy" in metrics
        assert "predictions" in metrics
        assert "targets" in metrics
        assert metrics["test_loss"] > 0
        assert 0 <= metrics["test_accuracy"] <= 1
        
    def test_mixed_precision_training(self, sample_model, sample_data, temp_dir):
        """Test mixed precision training (if CUDA available)."""
        train_loader, _ = sample_data
        
        config = TrainingConfig(
            epochs=1,
            mixed_precision=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir=temp_dir
        )
        
        trainer = LiquidTrainer(sample_model, config, train_loader)
        trainer.writer = Mock()
        
        if torch.cuda.is_available():
            assert trainer.scaler is not None
            
        # Should train without errors
        history = trainer.fit()
        assert len(history["train_metrics"]) == 1


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    def test_end_to_end_training(self):
        """Test complete end-to-end training pipeline."""
        # Create synthetic dataset
        dataset = SyntheticEventDataset(
            num_samples=50,
            resolution=(32, 32),
            task_type="classification",
            encoder_type="temporal"
        )
        
        train_loader = EventDataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model
        # Note: input_dim should match encoder output
        model = LiquidNet(
            input_dim=2 * 32 * 32,  # 2 channels * height * width for temporal encoder
            hidden_units=[16],
            output_dim=3
        )
        
        # Flatten the encoded input for the model
        class FlattenModel(nn.Module):
            def __init__(self, liquid_net):
                super().__init__()
                self.liquid_net = liquid_net
                
            def forward(self, x, **kwargs):
                # Flatten spatial dimensions
                batch_size = x.size(0)
                x_flat = x.view(batch_size, -1)
                return self.liquid_net(x_flat, **kwargs)
                
            def reset_states(self):
                return self.liquid_net.reset_states()
                
            def get_liquid_states(self):
                return self.liquid_net.get_liquid_states()
        
        model = FlattenModel(model)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                output_dir=temp_dir
            )
            
            trainer = LiquidTrainer(model, config, train_loader)
            trainer.writer = Mock()
            
            # Run training
            history = trainer.fit()
            
            # Check that training completed
            assert len(history["train_metrics"]) == 2
            assert all(m["loss"] > 0 for m in history["train_metrics"])
            
    def test_different_encoder_model_combinations(self):
        """Test training with different encoder-model combinations."""
        encoder_types = ["temporal", "spatial"]
        
        for encoder_type in encoder_types:
            dataset = SyntheticEventDataset(
                num_samples=20,
                resolution=(16, 16),
                task_type="classification",
                encoder_type=encoder_type
            )
            
            train_loader = EventDataLoader(dataset, batch_size=2)
            
            # Get sample to determine input size
            sample_encoded, _ = dataset[0]
            input_size = sample_encoded.numel()
            
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 10),
                nn.ReLU(),
                nn.Linear(10, 3)
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config = TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    output_dir=temp_dir
                )
                
                # Create simple trainer wrapper
                class SimpleTrainer:
                    def __init__(self, model, config, train_loader):
                        self.model = model
                        self.device = torch.device("cpu")
                        self.model.to(self.device)
                        self.criterion = nn.CrossEntropyLoss()
                        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                        self.train_loader = train_loader
                        
                    def train_step(self):
                        self.model.train()
                        total_loss = 0
                        
                        for data, targets in self.train_loader:
                            data = data.to(self.device)
                            targets = targets.to(self.device)
                            
                            self.optimizer.zero_grad()
                            outputs = self.model(data)
                            loss = self.criterion(outputs, targets)
                            loss.backward()
                            self.optimizer.step()
                            
                            total_loss += loss.item()
                            
                        return total_loss / len(self.train_loader)
                
                trainer = SimpleTrainer(model, config, train_loader)
                loss = trainer.train_step()
                
                # Should complete without errors
                assert loss > 0


@pytest.mark.slow
class TestLongRunningTraining:
    """Long-running training tests (marked as slow)."""
    
    def test_convergence_on_simple_task(self):
        """Test that model can converge on a simple task."""
        # Create simple pattern recognition task
        dataset = SyntheticEventDataset(
            num_samples=200,
            resolution=(24, 24),
            task_type="classification",
            encoder_type="spatial"
        )
        
        train_loader = EventDataLoader(dataset, batch_size=8, shuffle=True)
        
        # Simple model
        sample_encoded, _ = dataset[0]
        input_size = sample_encoded.numel()
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Train for several epochs
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(20):
            epoch_loss = 0
            
            for data, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            
            if epoch == 0:
                initial_loss = avg_loss
            if epoch == 19:
                final_loss = avg_loss
                
        # Loss should decrease
        assert final_loss < initial_loss
        assert final_loss < 1.0  # Should achieve reasonable loss