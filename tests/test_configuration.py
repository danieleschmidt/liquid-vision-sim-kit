#!/usr/bin/env python3
"""
Comprehensive tests for advanced configuration management.
Tests ConfigManager, validation, and preset configurations.
"""

import sys
import os
import pytest
import tempfile
import json
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liquid_vision.config import (
    ConfigManager, TrainingConfig, DeploymentConfig, SimulationConfig, 
    ValidationError, validate_config
)
from liquid_vision.config.defaults import get_preset_config, list_presets, PRESET_CONFIGS


class TestTrainingConfig:
    """Test TrainingConfig functionality."""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig with default values."""
        config = TrainingConfig()
        
        # Check essential defaults
        assert config.architecture == "liquid_net"
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert isinstance(config.hidden_units, list)
        assert isinstance(config.resolution, list)
        assert len(config.resolution) == 2
    
    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            architecture="rnn",
            epochs=200,
            batch_size=64,
            learning_rate=0.01,
            hidden_units=[256, 128, 64],
            encoder_type="voxel"
        )
        
        assert config.architecture == "rnn"
        assert config.epochs == 200
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.hidden_units == [256, 128, 64]
        assert config.encoder_type == "voxel"
    
    def test_training_config_validation_valid(self):
        """Test validation of valid TrainingConfig."""
        config = TrainingConfig(
            architecture="liquid_net",
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            encoder_type="temporal"
        )
        
        # Should not raise exception
        validate_config(config)
    
    def test_training_config_validation_invalid(self):
        """Test validation of invalid TrainingConfig."""
        # Invalid architecture
        with pytest.raises(ValidationError):
            config = TrainingConfig(architecture="invalid_arch")
            validate_config(config)
        
        # Invalid learning rate
        with pytest.raises(ValidationError):
            config = TrainingConfig(learning_rate=2.0)  # > 1
            validate_config(config)
        
        # Invalid encoder type
        with pytest.raises(ValidationError):
            config = TrainingConfig(encoder_type="invalid_encoder")
            validate_config(config)
    
    def test_training_config_serialization(self):
        """Test TrainingConfig serialization."""
        config = TrainingConfig(
            epochs=150,
            batch_size=64,
            encoder_type="sae"
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["epochs"] == 150
        assert config_dict["encoder_type"] == "sae"
        
        # Test to_json
        config_json = config.to_json()
        assert isinstance(config_json, str)
        assert "150" in config_json
        
        # Test to_yaml
        config_yaml = config.to_yaml()
        assert isinstance(config_yaml, str)
        assert "epochs: 150" in config_yaml
    
    def test_training_config_from_dict(self):
        """Test TrainingConfig creation from dictionary."""
        config_data = {
            "architecture": "liquid_net",
            "epochs": 75,
            "batch_size": 16,
            "learning_rate": 0.005,
            "encoder_type": "event_image"
        }
        
        config = TrainingConfig.from_dict(config_data)
        
        assert config.architecture == "liquid_net"
        assert config.epochs == 75
        assert config.batch_size == 16
        assert config.learning_rate == 0.005
        assert config.encoder_type == "event_image"


class TestDeploymentConfig:
    """Test DeploymentConfig functionality."""
    
    def test_deployment_config_defaults(self):
        """Test DeploymentConfig with default values."""
        config = DeploymentConfig()
        
        assert config.target == "esp32"
        assert config.quantization == "int8"
        assert config.max_memory_kb == 256
        assert config.sensor_type == "dvs240"
        assert config.build_system == "cmake"
    
    def test_deployment_config_validation_valid(self):
        """Test validation of valid DeploymentConfig."""
        config = DeploymentConfig(
            target="cortex_m7",
            quantization="int16",
            max_memory_kb=512,
            sensor_type="dvs640"
        )
        
        validate_config(config)  # Should not raise
    
    def test_deployment_config_validation_invalid(self):
        """Test validation of invalid DeploymentConfig."""
        # Invalid target
        with pytest.raises(ValidationError):
            config = DeploymentConfig(target="invalid_target")
            validate_config(config)
        
        # Invalid memory size
        with pytest.raises(ValidationError):
            config = DeploymentConfig(max_memory_kb=0)
            validate_config(config)
        
        # Invalid quantization
        with pytest.raises(ValidationError):
            config = DeploymentConfig(quantization="invalid_quant")
            validate_config(config)


class TestSimulationConfig:
    """Test SimulationConfig functionality."""
    
    def test_simulation_config_defaults(self):
        """Test SimulationConfig with default values."""
        config = SimulationConfig()
        
        assert config.num_objects == 3
        assert config.motion_type == "linear"
        assert isinstance(config.velocity_range, list)
        assert len(config.velocity_range) == 2
        assert isinstance(config.resolution, list)
        assert len(config.resolution) == 2
    
    def test_simulation_config_validation_valid(self):
        """Test validation of valid SimulationConfig."""
        config = SimulationConfig(
            num_objects=5,
            motion_type="circular",
            velocity_range=[20.0, 80.0],
            fps=60.0
        )
        
        validate_config(config)  # Should not raise
    
    def test_simulation_config_validation_invalid(self):
        """Test validation of invalid SimulationConfig."""
        # Invalid motion type
        with pytest.raises(ValidationError):
            config = SimulationConfig(motion_type="invalid_motion")
            validate_config(config)
        
        # Invalid velocity range
        with pytest.raises(ValidationError):
            config = SimulationConfig(velocity_range=[50.0, 20.0])  # max < min
            validate_config(config)
        
        # Invalid fps
        with pytest.raises(ValidationError):
            config = SimulationConfig(fps=0.0)
            validate_config(config)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_from_dict(self):
        """Test loading config from dictionary (default behavior)."""
        config = self.config_manager.load_config(
            "training", TrainingConfig
        )
        
        assert isinstance(config, TrainingConfig)
        assert config.architecture == "liquid_net"  # Default value
    
    def test_load_config_from_yaml_file(self):
        """Test loading config from YAML file."""
        # Create test YAML file
        config_data = {
            "architecture": "rnn",
            "epochs": 200,
            "batch_size": 128,
            "learning_rate": 0.01,
            "encoder_type": "spatial"
        }
        
        config_file = Path(self.temp_dir) / "training.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        config = self.config_manager.load_config(
            "training", TrainingConfig
        )
        
        assert config.architecture == "rnn"
        assert config.epochs == 200
        assert config.batch_size == 128
        assert config.encoder_type == "spatial"
    
    def test_load_config_from_json_file(self):
        """Test loading config from JSON file."""
        # Create test JSON file
        config_data = {
            "target": "cortex_m4",
            "quantization": "int16",
            "max_memory_kb": 512,
            "sensor_type": "davis346"
        }
        
        config_file = Path(self.temp_dir) / "deployment.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load config
        config = self.config_manager.load_config(
            "deployment", DeploymentConfig
        )
        
        assert config.target == "cortex_m4"
        assert config.quantization == "int16"
        assert config.max_memory_kb == 512
        assert config.sensor_type == "davis346"
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = TrainingConfig(
            epochs=300,
            batch_size=256,
            encoder_type="voxel"
        )
        
        config_file = Path(self.temp_dir) / "saved_training.yaml"
        
        self.config_manager.save_config("training", config, config_file)
        
        # Verify file was created
        assert config_file.exists()
        
        # Load and verify content
        with open(config_file, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["epochs"] == 300
        assert loaded_data["batch_size"] == 256
        assert loaded_data["encoder_type"] == "voxel"
    
    def test_validate_all_configs(self):
        """Test validation of all loaded configs."""
        # Load valid configs
        training_config = self.config_manager.load_config("training", TrainingConfig)
        deployment_config = self.config_manager.load_config("deployment", DeploymentConfig)
        
        # Validate all
        results = self.config_manager.validate_all_configs()
        
        assert isinstance(results, dict)
        assert "training" in results
        assert "deployment" in results
        assert results["training"] == True
        assert results["deployment"] == True
    
    def test_create_and_load_profile(self):
        """Test creating and loading configuration profiles."""
        # Create configs
        training_config = TrainingConfig(epochs=100, encoder_type="temporal")
        deployment_config = DeploymentConfig(target="esp32s3", max_memory_kb=512)
        
        configs = {
            "training": training_config,
            "deployment": deployment_config
        }
        
        # Create profile
        profile_name = "test_profile"
        self.config_manager.create_profile(profile_name, configs)
        
        # Verify profile directory was created
        profile_dir = Path(self.temp_dir) / "profiles" / profile_name
        assert profile_dir.exists()
        assert (profile_dir / "profile.json").exists()
        assert (profile_dir / "training.yaml").exists()
        assert (profile_dir / "deployment.yaml").exists()
        
        # Load profile
        loaded_configs = self.config_manager.load_profile(profile_name)
        
        assert "training" in loaded_configs
        assert "deployment" in loaded_configs
        assert loaded_configs["training"].epochs == 100
        assert loaded_configs["training"].encoder_type == "temporal"
        assert loaded_configs["deployment"].target == "esp32s3"
        assert loaded_configs["deployment"].max_memory_kb == 512


class TestPresetConfigurations:
    """Test preset configuration functionality."""
    
    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_presets()
        
        assert isinstance(presets, dict)
        assert len(presets) > 0
        
        # Check that expected presets exist
        expected_presets = ["research_high_quality", "development_fast", "edge_optimized"]
        for preset in expected_presets:
            assert preset in presets
    
    def test_get_preset_config_training(self):
        """Test getting training preset configuration."""
        preset_config = get_preset_config("research_high_quality", "training")
        
        assert isinstance(preset_config, dict)
        assert "epochs" in preset_config
        assert "batch_size" in preset_config
        assert "learning_rate" in preset_config
        
        # Should have preset-specific values
        assert preset_config["epochs"] == 500  # Higher than default
        assert preset_config["mixed_precision"] == True
        assert preset_config["augment_data"] == True
    
    def test_get_preset_config_deployment(self):
        """Test getting deployment preset configuration."""
        preset_config = get_preset_config("edge_optimized", "deployment")
        
        assert isinstance(preset_config, dict)
        assert "quantization" in preset_config
        assert "max_memory_kb" in preset_config
        
        # Should have optimization-specific values
        assert preset_config["quantization"] == "int8"
        assert preset_config["optimize_for_inference"] == True
        assert preset_config["prune_weights"] == 0.3
    
    def test_get_preset_config_invalid(self):
        """Test getting invalid preset configuration."""
        # Invalid preset name
        with pytest.raises(KeyError):
            get_preset_config("invalid_preset", "training")
        
        # Invalid config type for existing preset
        with pytest.raises(KeyError):
            get_preset_config("research_high_quality", "invalid_config_type")
    
    def test_preset_config_merging(self):
        """Test that presets properly merge with defaults."""
        preset_config = get_preset_config("development_fast", "training")
        
        # Should have preset values
        assert preset_config["epochs"] == 50  # Preset value
        assert preset_config["batch_size"] == 64  # Preset value
        
        # Should also have default values for unspecified fields
        assert "architecture" in preset_config
        assert "device" in preset_config
        assert "dropout_rate" in preset_config
    
    def test_domain_specific_presets(self):
        """Test domain-specific preset configurations."""
        # Test gesture recognition preset
        gesture_config = get_preset_config("gesture_recognition", "training")
        assert gesture_config["encoder_type"] == "temporal"
        assert gesture_config["time_window"] == 100.0
        assert gesture_config["output_dim"] == 5  # 5 gesture classes
        
        # Test optical flow preset
        optical_flow_config = get_preset_config("optical_flow", "training")
        assert optical_flow_config["encoder_type"] == "voxel"
        assert optical_flow_config["output_dim"] == 2  # x, y flow vectors


class TestEnvironmentVariableOverrides:
    """Test environment variable configuration overrides."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
        
        # Store original environment
        self.original_env = dict(os.environ)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_environment_variable_override(self):
        """Test configuration override with environment variables."""
        # Set environment variables
        os.environ["LIQUID_VISION_TRAINING_EPOCHS"] = "999"
        os.environ["LIQUID_VISION_TRAINING_BATCH_SIZE"] = "512"
        os.environ["LIQUID_VISION_TRAINING_ENCODER_TYPE"] = "sae"
        
        # Load config
        config = self.config_manager.load_config("training", TrainingConfig)
        
        # Should have environment overrides
        assert config.epochs == 999
        assert config.batch_size == 512
        assert config.encoder_type == "sae"
    
    def test_environment_variable_type_conversion(self):
        """Test proper type conversion of environment variables."""
        # Set various types
        os.environ["LIQUID_VISION_TRAINING_MIXED_PRECISION"] = "false"
        os.environ["LIQUID_VISION_TRAINING_LEARNING_RATE"] = "0.005"
        os.environ["LIQUID_VISION_TRAINING_HIDDEN_UNITS"] = "[128, 64, 32]"
        
        config = self.config_manager.load_config("training", TrainingConfig)
        
        assert config.mixed_precision == False  # Boolean conversion
        assert config.learning_rate == 0.005   # Float conversion  
        assert config.hidden_units == [128, 64, 32]  # List conversion


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])