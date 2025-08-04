#!/usr/bin/env python3
"""
Gesture recognition example using liquid neural networks.
Demonstrates classification of hand gestures from event-based camera data.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from liquid_vision import LiquidNet, EventSimulator, SceneGenerator
from liquid_vision.training import LiquidTrainer, TrainingConfig
from liquid_vision.training.event_dataloader import EventDataLoader, SyntheticEventDataset
from liquid_vision.core.event_encoding import create_encoder
from liquid_vision.simulation import ObjectType, MotionPattern


class GestureDataset(SyntheticEventDataset):
    """
    Synthetic gesture dataset with different hand motion patterns.
    """
    
    def __init__(self, num_samples=1000, resolution=(128, 128), **kwargs):
        self.gesture_types = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "circle", "tap"]
        self.num_classes = len(self.gesture_types)
        
        super().__init__(
            num_samples=num_samples,
            resolution=resolution,
            task_type="classification",
            **kwargs
        )
    
    def _generate_sample(self, idx):
        """Generate gesture-specific event patterns."""
        gesture_id = idx % self.num_classes
        gesture_type = self.gesture_types[gesture_id]
        
        if gesture_type == "swipe_left":
            events = self._generate_swipe_gesture(direction="left")
        elif gesture_type == "swipe_right":
            events = self._generate_swipe_gesture(direction="right")
        elif gesture_type == "swipe_up":
            events = self._generate_swipe_gesture(direction="up")
        elif gesture_type == "swipe_down":
            events = self._generate_swipe_gesture(direction="down")
        elif gesture_type == "circle":
            events = self._generate_circle_gesture()
        elif gesture_type == "tap":
            events = self._generate_tap_gesture()
        else:
            events = self._generate_circle_events()
            
        label = torch.tensor(gesture_id, dtype=torch.long)
        return events, label
    
    def _generate_swipe_gesture(self, direction="right", num_events=200):
        """Generate swipe gesture events."""
        center_x, center_y = self.resolution[0] // 2, self.resolution[1] // 2
        
        if direction == "left":
            start_x, end_x = center_x + 30, center_x - 30
            start_y = end_y = center_y
        elif direction == "right":
            start_x, end_x = center_x - 30, center_x + 30
            start_y = end_y = center_y
        elif direction == "up":
            start_x = end_x = center_x
            start_y, end_y = center_y + 30, center_y - 30
        else:  # down
            start_x = end_x = center_x
            start_y, end_y = center_y - 30, center_y + 30
            
        # Create trajectory
        t_values = np.linspace(0, self.time_window * 0.8, num_events)
        progress = t_values / (self.time_window * 0.8)
        
        x_coords = start_x + progress * (end_x - start_x)
        y_coords = start_y + progress * (end_y - start_y)
        
        # Add some noise and finger width
        finger_width = 3
        x_noise = np.random.normal(0, finger_width, num_events)
        y_noise = np.random.normal(0, finger_width, num_events)
        
        x_coords += x_noise
        y_coords += y_noise
        
        # Clip to bounds
        x_coords = np.clip(x_coords, 0, self.resolution[0] - 1)
        y_coords = np.clip(y_coords, 0, self.resolution[1] - 1)
        
        # Random polarities (finger contact/lift events)
        polarities = np.random.choice([-1, 1], num_events, p=[0.3, 0.7])
        
        return torch.stack([
            torch.from_numpy(x_coords.astype(np.float32)),
            torch.from_numpy(y_coords.astype(np.float32)),
            torch.from_numpy(t_values.astype(np.float32)),
            torch.from_numpy(polarities.astype(np.float32))
        ], dim=1)
    
    def _generate_circle_gesture(self, num_events=250):
        """Generate circular gesture events."""
        center_x, center_y = self.resolution[0] // 2, self.resolution[1] // 2
        radius = 25
        
        # Create circular trajectory
        t_values = np.linspace(0, self.time_window * 0.9, num_events)
        angles = np.linspace(0, 2 * np.pi, num_events)
        
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)
        
        # Add finger width noise
        finger_width = 2
        x_coords += np.random.normal(0, finger_width, num_events)
        y_coords += np.random.normal(0, finger_width, num_events)
        
        x_coords = np.clip(x_coords, 0, self.resolution[0] - 1)
        y_coords = np.clip(y_coords, 0, self.resolution[1] - 1)
        
        polarities = np.random.choice([-1, 1], num_events, p=[0.2, 0.8])
        
        return torch.stack([
            torch.from_numpy(x_coords.astype(np.float32)),
            torch.from_numpy(y_coords.astype(np.float32)),
            torch.from_numpy(t_values.astype(np.float32)),
            torch.from_numpy(polarities.astype(np.float32))
        ], dim=1)
    
    def _generate_tap_gesture(self, num_events=50):
        """Generate tap gesture events."""
        # Random tap location
        tap_x = np.random.randint(20, self.resolution[0] - 20)
        tap_y = np.random.randint(20, self.resolution[1] - 20)
        
        # Short burst of events at tap location
        tap_radius = 5
        
        angles = np.random.uniform(0, 2 * np.pi, num_events)
        radii = np.random.uniform(0, tap_radius, num_events)
        
        x_coords = tap_x + radii * np.cos(angles)
        y_coords = tap_y + radii * np.sin(angles)
        
        x_coords = np.clip(x_coords, 0, self.resolution[0] - 1)
        y_coords = np.clip(y_coords, 0, self.resolution[1] - 1)
        
        # Concentrated in time (short tap duration)
        t_values = np.random.uniform(
            self.time_window * 0.4, 
            self.time_window * 0.6, 
            num_events
        )
        
        polarities = np.random.choice([-1, 1], num_events, p=[0.4, 0.6])
        
        return torch.stack([
            torch.from_numpy(x_coords.astype(np.float32)),
            torch.from_numpy(y_coords.astype(np.float32)),
            torch.from_numpy(t_values.astype(np.float32)),
            torch.from_numpy(polarities.astype(np.float32))
        ], dim=1)


class GestureClassifier(nn.Module):
    """
    Gesture classifier using liquid neural networks.
    """
    
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        
        # Event encoder
        self.encoder_flatten = nn.Flatten()
        
        # Liquid neural network layers
        self.liquid_net = LiquidNet(
            input_dim=input_dim,
            hidden_units=[128, 64, 32],
            output_dim=num_classes,
            tau=15.0,
            leak=0.1,
            activation="swish",
            dropout=0.2
        )
        
    def forward(self, x, reset_state=False):
        # Flatten encoded events
        x_flat = self.encoder_flatten(x)
        
        # Process through liquid network
        output = self.liquid_net(x_flat, reset_state=reset_state)
        
        return output
    
    def reset_states(self):
        return self.liquid_net.reset_states()
    
    def get_liquid_states(self):
        return self.liquid_net.get_liquid_states()


def create_datasets(train_size=2000, val_size=400, test_size=400):
    """Create train, validation, and test datasets."""
    print("Creating datasets...")
    
    # Training dataset
    train_dataset = GestureDataset(
        num_samples=train_size,
        resolution=(64, 64),
        encoder_type="timeslice",
        encoder_kwargs={
            "num_slices": 5,
            "spatial_downsampling": 4
        },
        time_window=100.0
    )
    
    # Validation dataset
    val_dataset = GestureDataset(
        num_samples=val_size,
        resolution=(64, 64),
        encoder_type="timeslice",
        encoder_kwargs={
            "num_slices": 5,
            "spatial_downsampling": 4
        },
        time_window=100.0
    )
    
    # Test dataset
    test_dataset = GestureDataset(
        num_samples=test_size,
        resolution=(64, 64),
        encoder_type="timeslice",
        encoder_kwargs={
            "num_slices": 5,
            "spatial_downsampling": 4
        },
        time_window=100.0
    )
    
    print(f"✓ Created datasets: {train_size} train, {val_size} val, {test_size} test")
    return train_dataset, val_dataset, test_dataset


def train_gesture_classifier(train_dataset, val_dataset, output_dir="output/gesture_recognition"):
    """Train the gesture classifier."""
    print("Training gesture classifier...")
    
    # Create data loaders
    train_loader = EventDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = EventDataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Get input dimension
    sample_encoded, _ = train_dataset[0]
    input_dim = sample_encoded.numel()
    
    # Create model
    model = GestureClassifier(input_dim=input_dim, num_classes=6)
    
    # Training configuration
    config = TrainingConfig(
        epochs=25,
        learning_rate=1e-3,
        batch_size=32,
        optimizer="adamw",
        scheduler="cosine",
        loss_type="cross_entropy",
        validation_frequency=2,
        early_stopping_patience=8,
        gradient_clip=1.0,
        liquid_regularization=0.01,
        output_dir=output_dir,
        experiment_name="gesture_classifier"
    )
    
    # Create trainer
    trainer = LiquidTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train model
    history = trainer.fit()
    
    print("✓ Training completed")
    return model, trainer, history


def evaluate_gesture_classifier(model, test_dataset):
    """Evaluate the trained classifier."""
    print("Evaluating gesture classifier...")
    
    test_loader = EventDataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 6
    class_total = [0] * 6
    
    gesture_names = ["Swipe Left", "Swipe Right", "Swipe Up", "Swipe Down", "Circle", "Tap"]
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data, reset_state=True)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
    
    overall_accuracy = 100 * correct / total
    print(f"✓ Overall Test Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
    
    print("\nPer-class accuracy:")
    for i, gesture_name in enumerate(gesture_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {gesture_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return overall_accuracy


def visualize_results(train_dataset, model, output_dir="output/gesture_recognition"):
    """Create visualizations of results."""
    print("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    gesture_names = ["Swipe Left", "Swipe Right", "Swipe Up", "Swipe Down", "Circle", "Tap"]
    
    # Visualize sample gestures
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    model.eval()
    with torch.no_grad():
        for i in range(6):
            # Get a sample of each gesture type
            sample_idx = i  # First sample of each class
            encoded, true_label = train_dataset[sample_idx]
            
            # Get raw events for visualization
            events, _ = train_dataset._generate_sample(sample_idx)
            
            # Model prediction
            output = model(encoded.unsqueeze(0), reset_state=True)
            predicted_prob = torch.softmax(output, dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            
            # Plot events
            ax = axes[i]
            pos_mask = events[:, 3] > 0
            neg_mask = events[:, 3] <= 0
            
            if pos_mask.any():
                ax.scatter(events[pos_mask, 0], events[pos_mask, 1], 
                          c=events[pos_mask, 2], cmap='Reds', s=2, alpha=0.7, label='Positive')
            if neg_mask.any():
                ax.scatter(events[neg_mask, 0], events[neg_mask, 1], 
                          c=events[neg_mask, 2], cmap='Blues', s=2, alpha=0.7, label='Negative')
            
            ax.set_title(f'{gesture_names[i]}\nPred: {gesture_names[predicted_label]} ({predicted_prob[0, predicted_label]:.2f})')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "gesture_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Gesture Recognition with Liquid Neural Networks")
    parser.add_argument("--train-size", type=int, default=1200, help="Training dataset size")
    parser.add_argument("--val-size", type=int, default=300, help="Validation dataset size") 
    parser.add_argument("--test-size", type=int, default=300, help="Test dataset size")
    parser.add_argument("--output-dir", type=str, default="output/gesture_recognition", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Training device")
    
    args = parser.parse_args()
    
    print("🤚 Gesture Recognition with Liquid Neural Networks")
    print("=" * 55)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size
    )
    
    # Train model
    model, trainer, history = train_gesture_classifier(
        train_dataset, val_dataset, output_dir=args.output_dir
    )
    
    # Evaluate model
    test_accuracy = evaluate_gesture_classifier(model, test_dataset)
    
    # Create visualizations
    visualize_results(train_dataset, model, output_dir=args.output_dir)
    
    # Summary
    print(f"\n🎉 Gesture recognition training completed!")
    print(f"📊 Final test accuracy: {test_accuracy:.2f}%")
    print(f"📁 Results saved to: {args.output_dir}")
    
    # Save model
    model_path = Path(args.output_dir) / "gesture_classifier.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.encoder_flatten.numel() if hasattr(model.encoder_flatten, 'numel') else 'unknown',
            'num_classes': 6
        },
        'test_accuracy': test_accuracy,
        'gesture_names': ["Swipe Left", "Swipe Right", "Swipe Up", "Swipe Down", "Circle", "Tap"]
    }, model_path)
    
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    main()