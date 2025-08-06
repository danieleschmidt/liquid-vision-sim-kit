"""
Default configuration values for liquid vision simulation kit.
"""

DEFAULT_CONFIGS = {
    "training": {
        "architecture": "liquid_net",
        "input_dim": 64,
        "hidden_units": [128, 64],
        "output_dim": 10,
        "liquid_time_constant": 20.0,
        
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "gradient_clip_norm": 1.0,
        
        "encoder_type": "temporal",
        "time_window": 50.0,
        "resolution": [640, 480],
        "augment_data": False,
        
        "optimizer": "adam",
        "scheduler": "cosine",
        "warmup_epochs": 10,
        
        "dropout_rate": 0.1,
        "liquid_dropout": 0.05,
        
        "device": "auto",
        "num_workers": 4,
        "mixed_precision": True,
        
        "save_every": 10,
        "checkpoint_dir": "./checkpoints",
    },
    
    "deployment": {
        "target": "esp32",
        "quantization": "int8",
        "max_memory_kb": 256,
        
        "optimize_for_inference": True,
        "fuse_operations": True,
        "prune_weights": 0.1,
        
        "sensor_type": "dvs240",
        "spi_frequency": 2000000,
        "i2c_frequency": 100000,
        
        "max_inference_time_ms": 10.0,
        "batch_inference": False,
        
        "output_dir": "./deployment",
        "project_name": "liquid_vision_edge",
        
        "build_system": "cmake",
        "enable_debug": False,
    },
    
    "simulation": {
        "num_objects": 3,
        "motion_type": "linear",
        "velocity_range": [10.0, 50.0],
        
        "resolution": [640, 480],
        "fps": 30.0,
        
        "contrast_threshold": 0.2,
        "refractory_period": 1.0,
        "noise_rate": 0.01,
        
        "num_frames": 1000,
        "output_format": "h5",
        "save_aps_frames": False,
    },
}


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "research_high_quality": {
        "training": {
            "epochs": 500,
            "batch_size": 16,
            "learning_rate": 0.0005,
            "hidden_units": [256, 128, 64],
            "mixed_precision": True,
            "augment_data": True,
        }
    },
    
    "development_fast": {
        "training": {
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.002,
            "hidden_units": [64, 32],
            "mixed_precision": False,
        }
    },
    
    "edge_optimized": {
        "deployment": {
            "quantization": "int8",
            "max_memory_kb": 128,
            "optimize_for_inference": True,
            "fuse_operations": True,
            "prune_weights": 0.3,
        }
    },
    
    "high_performance_edge": {
        "deployment": {
            "target": "cortex_m7",
            "quantization": "int16", 
            "max_memory_kb": 512,
            "max_inference_time_ms": 5.0,
        }
    },
    
    "gesture_recognition": {
        "training": {
            "encoder_type": "temporal",
            "time_window": 100.0,
            "output_dim": 5,  # 5 gesture classes
            "hidden_units": [128, 64],
        },
        "simulation": {
            "motion_type": "oscillatory",
            "velocity_range": [5.0, 25.0],
            "num_objects": 1,  # Single hand
        }
    },
    
    "optical_flow": {
        "training": {
            "encoder_type": "voxel",
            "output_dim": 2,  # x, y flow vectors
            "hidden_units": [256, 128],
        },
        "simulation": {
            "motion_type": "linear",
            "velocity_range": [20.0, 80.0],
            "contrast_threshold": 0.15,
        }
    },
    
    "object_tracking": {
        "training": {
            "encoder_type": "sae",
            "output_dim": 4,  # bounding box coordinates
            "hidden_units": [128, 64, 32],
        },
        "simulation": {
            "motion_type": "random",
            "num_objects": 1,
            "velocity_range": [15.0, 45.0],
        }
    },
}


def get_preset_config(preset_name: str, config_type: str) -> dict:
    """
    Get preset configuration for specific use case.
    
    Args:
        preset_name: Name of preset configuration
        config_type: Type of config (training, deployment, simulation)
        
    Returns:
        Configuration dictionary
        
    Raises:
        KeyError: If preset or config type not found
    """
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise KeyError(f"Preset '{preset_name}' not found. Available: {available}")
    
    preset = PRESET_CONFIGS[preset_name]
    if config_type not in preset:
        available = list(preset.keys())
        raise KeyError(f"Config type '{config_type}' not in preset '{preset_name}'. Available: {available}")
    
    # Merge with defaults
    base_config = DEFAULT_CONFIGS[config_type].copy()
    base_config.update(preset[config_type])
    
    return base_config


def list_presets() -> dict:
    """
    List all available preset configurations.
    
    Returns:
        Dictionary mapping preset names to their config types
    """
    return {name: list(config.keys()) for name, config in PRESET_CONFIGS.items()}