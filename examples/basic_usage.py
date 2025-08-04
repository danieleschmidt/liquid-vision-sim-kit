#!/usr/bin/env python3
"""
Basic usage example for liquid-vision-sim-kit.
Demonstrates core functionality: scene generation, event simulation, and training.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import liquid vision components
from liquid_vision import (
    LiquidNet, EventSimulator, SceneGenerator, LiquidTrainer,
    EventDataLoader, SyntheticEventDataset
)
from liquid_vision.training import TrainingConfig
from liquid_vision.simulation import create_simulator, MotionPattern, ObjectType


def main():
    """Run basic usage example."""
    print("🧠 Liquid Vision Sim-Kit - Basic Usage Example")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Scene Generation
    print("\n1. Generating synthetic scene...")
    scene = create_simple_scene()
    frames, timestamps = scene.generate_sequence(num_frames=20, return_timestamps=True)
    print(f"✓ Generated {len(frames)} frames with resolution {frames.shape[1:3]}")
    
    # 2. Event Camera Simulation
    print("\n2. Simulating event camera...")
    events = simulate_events(frames, timestamps)
    print(f"✓ Generated {len(events)} events")
    print(f"  - Time range: {events.t.min():.1f} - {events.t.max():.1f} ms")
    print(f"  - Spatial range: ({events.x.min()}, {events.y.min()}) to ({events.x.max()}, {events.y.max()})")
    print(f"  - Polarity distribution: {np.sum(events.p > 0)} positive, {np.sum(events.p < 0)} negative")
    
    # 3. Liquid Neural Network Training
    print("\n3. Training liquid neural network...")
    model, history = train_liquid_network()
    print(f"✓ Training completed")
    print(f"  - Final training loss: {history['train_metrics'][-1]['loss']:.4f}")
    print(f"  - Final training accuracy: {history['train_metrics'][-1]['accuracy']:.4f}")
    
    # 4. Model Evaluation
    print("\n4. Evaluating model...")
    evaluate_model(model)
    
    # 5. Visualization
    print("\n5. Creating visualizations...")
    create_visualizations(scene, events, history)
    print("✓ Visualizations saved to 'output/' directory")
    
    print("\n🎉 Basic usage example completed successfully!")
    print("\nNext steps:")
    print("- Explore examples/gesture_recognition/ for advanced usage")
    print("- Check examples/optical_flow/ for motion estimation")
    print("- See examples/object_tracking/ for real-time tracking")


def create_simple_scene():
    """Create a simple scene with moving objects."""
    scene = SceneGenerator(
        resolution=(128, 96),
        background_color=0.3,
        frame_rate=30.0
    )
    
    # Add circular moving object
    scene.add_object(
        object_type=ObjectType.CIRCLE,
        position=(30, 48),
        size=8,
        velocity=(3, 0),
        color=0.9,
        motion_pattern=MotionPattern.LINEAR,
        boundary_mode="wrap"
    )
    
    # Add oscillating rectangle
    scene.add_object(
        object_type=ObjectType.RECTANGLE,
        position=(64, 30),
        size=(12, 8),
        velocity=(0, 0),
        color=0.7,
        motion_pattern=MotionPattern.OSCILLATORY,
        amplitude=20,
        frequency=1.5,
        axis='y'
    )
    
    return scene


def simulate_events(frames, timestamps):
    """Simulate events from video frames."""
    simulator = create_simulator(
        simulator_type="dvs",
        resolution=(128, 96),
        contrast_threshold=0.1,
        refractory_period=1.0
    )
    
    events = simulator.simulate_video(frames, timestamps)
    return events


def train_liquid_network():
    """Train a simple liquid neural network on synthetic data."""
    # Create synthetic dataset
    dataset = SyntheticEventDataset(
        num_samples=500,
        resolution=(64, 48),
        task_type="classification",
        encoder_type="temporal",
        time_window=50.0
    )
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = EventDataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = EventDataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    # Get input dimension from dataset
    sample_encoded, _ = dataset[0]
    input_dim = sample_encoded.numel()
    
    model = LiquidNet(
        input_dim=input_dim,
        hidden_units=[32, 16],
        output_dim=3,  # 3 classes in synthetic classification task
        tau=10.0,
        dropout=0.1
    )
    
    # Wrapper to handle flattening
    class FlattenWrapper(torch.nn.Module):
        def __init__(self, liquid_net):
            super().__init__()
            self.liquid_net = liquid_net
            
        def forward(self, x, **kwargs):
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            return self.liquid_net(x_flat, **kwargs)
            
        def reset_states(self):
            return self.liquid_net.reset_states()
            
        def get_liquid_states(self):
            return self.liquid_net.get_liquid_states()
    
    model = FlattenWrapper(model)
    
    # Training configuration
    config = TrainingConfig(
        epochs=10,
        learning_rate=1e-3,
        batch_size=16,
        optimizer="adam",
        loss_type="cross_entropy",
        validation_frequency=2,
        output_dir="output/training",
        experiment_name="basic_example"
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
    
    return model, history


def evaluate_model(model):
    """Evaluate trained model on test data."""
    # Create test dataset
    test_dataset = SyntheticEventDataset(
        num_samples=100,
        resolution=(64, 48),
        task_type="classification",
        encoder_type="temporal"
    )
    
    test_loader = EventDataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            batch_size = data.size(0)
            data_flat = data.view(batch_size, -1)
            outputs = model(data_flat)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100 * correct / total
    print(f"✓ Test accuracy: {accuracy:.2f}% ({correct}/{total})")


def create_visualizations(scene, events, history):
    """Create and save visualizations."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Scene visualization
    plt.figure(figsize=(12, 4))
    
    # Show first, middle, and last frames
    frames, _ = scene.generate_sequence(num_frames=20, return_timestamps=False)
    frame_indices = [0, 10, 19]
    
    for i, idx in enumerate(frame_indices):
        plt.subplot(1, 3, i + 1)
        plt.imshow(frames[idx], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Frame {idx}')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "scene_frames.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Event visualization
    plt.figure(figsize=(12, 5))
    
    # Spatial distribution
    plt.subplot(1, 2, 1)
    pos_events = events.p > 0
    neg_events = events.p < 0
    
    plt.scatter(events.x[pos_events], events.y[pos_events], 
               c='red', s=1, alpha=0.6, label='Positive')
    plt.scatter(events.x[neg_events], events.y[neg_events], 
               c='blue', s=1, alpha=0.6, label='Negative')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Event spatial distribution')
    plt.legend()
    plt.axis('equal')
    
    # Temporal distribution
    plt.subplot(1, 2, 2)
    plt.hist(events.t[pos_events], bins=50, alpha=0.6, color='red', label='Positive')
    plt.hist(events.t[neg_events], bins=50, alpha=0.6, color='blue', label='Negative')
    plt.xlabel('Time (ms)')
    plt.ylabel('Event count')
    plt.title('Event temporal distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "event_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Training history
    plt.figure(figsize=(12, 4))
    
    epochs = range(1, len(history['train_metrics']) + 1)
    train_losses = [m['loss'] for m in history['train_metrics']]
    train_accs = [m['accuracy'] for m in history['train_metrics']]
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if history['val_metrics']:
        val_epochs = range(1, len(history['val_metrics']) + 1)
        val_losses = [m['loss'] for m in history['val_metrics']]
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    if history['val_metrics']:
        val_accs = [m['accuracy'] for m in history['val_metrics']]
        plt.plot(val_epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()