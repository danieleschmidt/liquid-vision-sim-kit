#!/usr/bin/env python3
"""
Command-line interface for liquid-vision-sim-kit.
Provides easy access to training, simulation, and evaluation functionality.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

from . import __version__
from .core.liquid_neurons import create_liquid_net, get_model_info
from .simulation import create_simulator, SceneGenerator
from .training import LiquidTrainer, TrainingConfig
from .training.event_dataloader import SyntheticEventDataset, EventDataLoader
from .config import ConfigManager, TrainingConfig as AdvancedTrainingConfig, DeploymentConfig, SimulationConfig
from .config.defaults import get_preset_config, list_presets


def cmd_info(args):
    """Display system and library information."""
    print(f"Liquid Vision Sim-Kit v{__version__}")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    print(f"NumPy version: {np.__version__}")


def cmd_simulate(args):
    """Generate event data from synthetic scenes."""
    print(f"Generating {args.frames} frames at {args.resolution}...")
    
    # Create scene
    scene = SceneGenerator.create_scene(
        num_objects=args.objects,
        resolution=args.resolution,
        motion_type=args.motion,
        velocity_range=(args.min_velocity, args.max_velocity)
    )
    
    # Generate frames
    frames, timestamps = scene.generate_sequence(
        num_frames=args.frames,
        return_timestamps=True
    )
    
    # Simulate events
    simulator = create_simulator(
        simulator_type=args.simulator,
        resolution=args.resolution,
        contrast_threshold=args.threshold
    )
    
    events = simulator.simulate_video(frames, timestamps)
    
    print(f"Generated {len(events)} events")
    
    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_path,
            x=events.x,
            y=events.y,
            t=events.t,
            p=events.p,
            frames=frames,
            timestamps=timestamps
        )
        print(f"Saved to {output_path}")


def cmd_train(args):
    """Train a liquid neural network."""
    print(f"Training {args.architecture} model for {args.epochs} epochs...")
    
    # Create dataset
    if args.data:
        # Load custom data
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
            
        if data_path.suffix.lower() == '.h5':
            # Load HDF5 dataset
            import h5py
            with h5py.File(data_path, 'r') as f:
                events_data = f['events'][:]
                labels_data = f.get('labels', None)
                
            # Create custom dataset
            from .training.custom_dataset import CustomEventDataset
            dataset = CustomEventDataset(
                events_data,
                labels_data,
                encoder_type=args.encoder,
                resolution=args.resolution
            )
        elif data_path.suffix.lower() in ['.npz', '.npy']:
            # Load NumPy dataset
            data = np.load(data_path, allow_pickle=True)
            events_data = data['events'] if 'events' in data else data
            labels_data = data.get('labels', None)
            
            from .training.custom_dataset import CustomEventDataset
            dataset = CustomEventDataset(
                events_data,
                labels_data,
                encoder_type=args.encoder,
                resolution=args.resolution
            )
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
    else:
        # Use synthetic data
        dataset = SyntheticEventDataset(
            num_samples=args.samples,
            resolution=args.resolution,
            task_type=args.task,
            encoder_type=args.encoder
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = EventDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = EventDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    sample_encoded, _ = dataset[0]
    input_dim = sample_encoded.numel()
    
    model = create_liquid_net(
        input_dim=input_dim,
        output_dim=args.num_classes,
        architecture=args.architecture
    )
    
    # Wrapper for flattening
    class ModelWrapper(torch.nn.Module):
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
    
    model = ModelWrapper(model)
    
    # Training config
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        device=args.device,
        output_dir=args.output,
        experiment_name=args.name
    )
    
    # Create trainer
    trainer = LiquidTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train
    history = trainer.fit()
    
    final_acc = history['train_metrics'][-1]['accuracy']
    print(f"Training completed! Final accuracy: {final_acc:.4f}")


def cmd_model_info(args):
    """Display model architecture information."""
    model = create_liquid_net(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        architecture=args.architecture
    )
    
    info = get_model_info(model)
    
    print(f"Model Architecture: {args.architecture}")
    print("=" * 30)
    print(f"Input dimension: {info['input_dim']}")
    print(f"Output dimension: {info['output_dim']}")
    print(f"Number of layers: {info['num_layers']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    
    print("\nLayer details:")
    for layer_info in info['layer_info']:
        print(f"  Layer {layer_info['layer']}: "
              f"{layer_info['input_dim']} → {layer_info['hidden_dim']} "
              f"(τ={layer_info['tau']:.1f}, {layer_info['parameters']} params)")


def cmd_config(args):
    """Configuration management commands."""
    config_manager = ConfigManager(args.config_dir)
    
    if args.config_action == "list":
        # List available configurations
        configs = config_manager.list_configs()
        if configs:
            print("Loaded configurations:")
            for config_type in configs:
                print(f"  - {config_type}")
        else:
            print("No configurations loaded.")
            
    elif args.config_action == "show":
        # Show configuration details
        if args.config_type:
            config = config_manager.get_config(args.config_type)
            if config:
                print(f"{args.config_type.title()} Configuration:")
                print("=" * 40)
                if args.format == "yaml":
                    print(config.to_yaml())
                else:
                    print(config.to_json())
            else:
                print(f"Configuration '{args.config_type}' not loaded.")
        else:
            print("Please specify --type for configuration to show")
            
    elif args.config_action == "validate":
        # Validate all configurations
        results = config_manager.validate_all_configs()
        all_valid = all(results.values())
        
        print("Configuration Validation Results:")
        print("=" * 40)
        for config_type, is_valid in results.items():
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"{config_type}: {status}")
        
        if not all_valid:
            sys.exit(1)
            
    elif args.config_action == "preset":
        # Apply preset configuration
        if args.preset_name:
            try:
                # Show available presets
                if args.preset_name == "list":
                    presets = list_presets()
                    print("Available Configuration Presets:")
                    print("=" * 40)
                    for preset_name, config_types in presets.items():
                        print(f"{preset_name}:")
                        for config_type in config_types:
                            print(f"  - {config_type}")
                        print()
                else:
                    # Apply specific preset
                    config_type = args.config_type or "training"
                    preset_config = get_preset_config(args.preset_name, config_type)
                    
                    # Save to file
                    output_file = Path(args.output or f"{config_type}_preset_{args.preset_name}.yaml")
                    
                    config_classes = {
                        "training": AdvancedTrainingConfig,
                        "deployment": DeploymentConfig,
                        "simulation": SimulationConfig
                    }
                    
                    config_class = config_classes[config_type]
                    config = config_class.from_dict(preset_config)
                    config.save(output_file)
                    
                    print(f"Applied preset '{args.preset_name}' to {output_file}")
                    
            except KeyError as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            print("Please specify preset name (or 'list' to show available presets)")


def cmd_profile(args):
    """Configuration profile management."""
    config_manager = ConfigManager(args.config_dir)
    
    if args.profile_action == "create":
        if not args.profile_name:
            print("Please specify --name for profile to create")
            sys.exit(1)
            
        # Load configurations to include in profile
        configs = {}
        
        # Load training config if specified
        if args.training_config:
            training_config = AdvancedTrainingConfig.from_file(args.training_config)
            configs["training"] = training_config
            
        # Load deployment config if specified  
        if args.deployment_config:
            deployment_config = DeploymentConfig.from_file(args.deployment_config)
            configs["deployment"] = deployment_config
            
        # Load simulation config if specified
        if args.simulation_config:
            simulation_config = SimulationConfig.from_file(args.simulation_config)
            configs["simulation"] = simulation_config
        
        if not configs:
            print("Please specify at least one configuration file to include in profile")
            sys.exit(1)
            
        config_manager.create_profile(args.profile_name, configs)
        print(f"Created profile '{args.profile_name}' with {len(configs)} configurations")
        
    elif args.profile_action == "load":
        if not args.profile_name:
            print("Please specify --name for profile to load")
            sys.exit(1)
            
        try:
            configs = config_manager.load_profile(args.profile_name)
            print(f"Loaded profile '{args.profile_name}' with configurations:")
            for config_type in configs:
                print(f"  - {config_type}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif args.profile_action == "list":
        profiles_dir = Path(config_manager.config_dir) / "profiles"
        if profiles_dir.exists():
            profiles = [p.name for p in profiles_dir.iterdir() if p.is_dir()]
            if profiles:
                print("Available configuration profiles:")
                for profile in sorted(profiles):
                    print(f"  - {profile}")
            else:
                print("No configuration profiles found.")
        else:
            print("No configuration profiles directory found.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Liquid Vision Sim-Kit - Neuromorphic ML Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  liquid-vision info                    # Show system info
  liquid-vision simulate --frames 100  # Generate 100 frames of events
  liquid-vision train --epochs 50      # Train model for 50 epochs
  liquid-vision model-info --arch base # Show model architecture info
        """
    )
    
    parser.add_argument("--version", action="version", version=f"liquid-vision-sim-kit {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display system information")
    info_parser.set_defaults(func=cmd_info)
    
    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Generate synthetic event data")
    sim_parser.add_argument("--frames", type=int, default=50, help="Number of frames to generate")
    sim_parser.add_argument("--resolution", type=int, nargs=2, default=[128, 96], 
                           metavar=("WIDTH", "HEIGHT"), help="Frame resolution")
    sim_parser.add_argument("--objects", type=int, default=3, help="Number of moving objects")
    sim_parser.add_argument("--motion", choices=["linear", "circular", "mixed"], default="mixed",
                           help="Motion pattern")
    sim_parser.add_argument("--simulator", choices=["dvs", "davis", "advanced_dvs"], default="dvs",
                           help="Simulator type")
    sim_parser.add_argument("--threshold", type=float, default=0.1, help="Contrast threshold")
    sim_parser.add_argument("--min-velocity", type=float, default=0.5, help="Minimum velocity")
    sim_parser.add_argument("--max-velocity", type=float, default=3.0, help="Maximum velocity")
    sim_parser.add_argument("--output", "-o", help="Output file path (.npz)")
    sim_parser.set_defaults(func=cmd_simulate)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train liquid neural network")
    train_parser.add_argument("--data", help="Path to training data (uses synthetic if not provided)")
    train_parser.add_argument("--task", choices=["classification", "counting", "motion"], 
                             default="classification", help="Task type for synthetic data")
    train_parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    train_parser.add_argument("--resolution", type=int, nargs=2, default=[64, 64],
                             metavar=("WIDTH", "HEIGHT"), help="Input resolution")
    train_parser.add_argument("--encoder", choices=["temporal", "spatial", "timeslice", "adaptive"],
                             default="temporal", help="Event encoder type")
    train_parser.add_argument("--architecture", choices=["tiny", "small", "base", "large"],
                             default="small", help="Model architecture")
    train_parser.add_argument("--num-classes", type=int, default=3, help="Number of output classes")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam",
                             help="Optimizer")
    train_parser.add_argument("--device", default="auto", help="Training device")
    train_parser.add_argument("--output", "-o", default="experiments", help="Output directory")
    train_parser.add_argument("--name", help="Experiment name")
    train_parser.set_defaults(func=cmd_train)
    
    # Model info command
    model_parser = subparsers.add_parser("model-info", help="Display model architecture information")
    model_parser.add_argument("--architecture", choices=["tiny", "small", "base", "large"],
                             default="small", help="Model architecture")
    model_parser.add_argument("--input-dim", type=int, default=1000, help="Input dimension")
    model_parser.add_argument("--output-dim", type=int, default=10, help="Output dimension")
    model_parser.set_defaults(func=cmd_model_info)
    
    # Configuration management command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--config-dir", default=".", help="Configuration directory")
    config_subparsers = config_parser.add_subparsers(dest="config_action", help="Configuration actions")
    
    # List configurations
    list_parser = config_subparsers.add_parser("list", help="List available configurations")
    
    # Show configuration
    show_parser = config_subparsers.add_parser("show", help="Show configuration details")
    show_parser.add_argument("--type", dest="config_type", 
                            choices=["training", "deployment", "simulation"],
                            help="Configuration type to show")
    show_parser.add_argument("--format", choices=["json", "yaml"], default="yaml",
                            help="Output format")
    
    # Validate configurations
    validate_parser = config_subparsers.add_parser("validate", help="Validate all configurations")
    
    # Preset configurations
    preset_parser = config_subparsers.add_parser("preset", help="Apply preset configurations")
    preset_parser.add_argument("preset_name", help="Preset name (or 'list' to show available)")
    preset_parser.add_argument("--type", dest="config_type",
                              choices=["training", "deployment", "simulation"],
                              help="Configuration type")
    preset_parser.add_argument("--output", help="Output file path")
    
    config_parser.set_defaults(func=cmd_config)
    
    # Profile management command
    profile_parser = subparsers.add_parser("profile", help="Configuration profile management")
    profile_parser.add_argument("--config-dir", default=".", help="Configuration directory")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_action", help="Profile actions")
    
    # Create profile
    create_profile_parser = profile_subparsers.add_parser("create", help="Create configuration profile")
    create_profile_parser.add_argument("--name", dest="profile_name", required=True,
                                      help="Profile name")
    create_profile_parser.add_argument("--training-config", help="Training configuration file")
    create_profile_parser.add_argument("--deployment-config", help="Deployment configuration file")  
    create_profile_parser.add_argument("--simulation-config", help="Simulation configuration file")
    
    # Load profile
    load_profile_parser = profile_subparsers.add_parser("load", help="Load configuration profile")
    load_profile_parser.add_argument("--name", dest="profile_name", required=True,
                                    help="Profile name")
    
    # List profiles
    list_profiles_parser = profile_subparsers.add_parser("list", help="List available profiles")
    
    profile_parser.set_defaults(func=cmd_profile)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
        
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())