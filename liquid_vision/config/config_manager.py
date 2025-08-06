"""
Advanced configuration management with hierarchical configs and validation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, fields
import logging

from .validators import validate_config, ValidationError
from .defaults import DEFAULT_CONFIGS


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Base configuration class with validation."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        try:
            validate_config(self)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create from dictionary."""
        # Filter only valid fields for this config class
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """Load configuration from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls.from_dict(data)
    
    def save(self, file_path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                f.write(self.to_yaml())
            elif path.suffix.lower() == '.json':
                f.write(self.to_json())
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")


@dataclass
class TrainingConfig(Config):
    """Configuration for training liquid neural networks."""
    
    # Model architecture
    architecture: str = "liquid_net"
    input_dim: int = 64
    hidden_units: List[int] = None
    output_dim: int = 10
    liquid_time_constant: float = 20.0
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Data parameters
    encoder_type: str = "temporal"
    time_window: float = 50.0
    resolution: List[int] = None
    augment_data: bool = False
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    
    # Regularization
    dropout_rate: float = 0.1
    liquid_dropout: float = 0.05
    
    # Hardware
    device: str = "auto"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "./checkpoints"
    
    def __post_init__(self):
        """Set defaults and validate."""
        if self.hidden_units is None:
            self.hidden_units = [128, 64]
        if self.resolution is None:
            self.resolution = [640, 480]
        super().__post_init__()


@dataclass
class DeploymentConfig(Config):
    """Configuration for edge deployment."""
    
    # Target platform
    target: str = "esp32"
    quantization: str = "int8"
    max_memory_kb: int = 256
    
    # Model optimization
    optimize_for_inference: bool = True
    fuse_operations: bool = True
    prune_weights: float = 0.1
    
    # Hardware interface
    sensor_type: str = "dvs240"
    spi_frequency: int = 2000000
    i2c_frequency: int = 100000
    
    # Performance
    max_inference_time_ms: float = 10.0
    batch_inference: bool = False
    
    # Output
    output_dir: str = "./deployment"
    project_name: str = "liquid_vision_edge"
    
    # Build system
    build_system: str = "cmake"  # cmake, platformio
    enable_debug: bool = False


@dataclass
class SimulationConfig(Config):
    """Configuration for event simulation."""
    
    # Scene parameters
    num_objects: int = 3
    motion_type: str = "linear"
    velocity_range: List[float] = None
    
    # Camera parameters
    resolution: List[int] = None
    fps: float = 30.0
    
    # DVS parameters
    contrast_threshold: float = 0.2
    refractory_period: float = 1.0
    noise_rate: float = 0.01
    
    # Output
    num_frames: int = 1000
    output_format: str = "h5"
    save_aps_frames: bool = False
    
    def __post_init__(self):
        """Set defaults and validate."""
        if self.velocity_range is None:
            self.velocity_range = [10.0, 50.0]
        if self.resolution is None:
            self.resolution = [640, 480]
        super().__post_init__()


class ConfigManager:
    """
    Advanced configuration manager with hierarchical configs,
    environment variable support, and validation.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to search for config files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self._configs: Dict[str, Config] = {}
        self._search_paths = [
            self.config_dir,
            Path.home() / ".liquid_vision",
            Path("/etc/liquid_vision"),
            Path(__file__).parent / "defaults"
        ]
        
    def load_config(
        self, 
        config_type: str,
        config_class: type,
        file_name: Optional[str] = None,
        search_paths: Optional[List[Path]] = None
    ) -> Config:
        """
        Load configuration with hierarchical search and environment variable support.
        
        Args:
            config_type: Type of configuration (training, deployment, simulation)
            config_class: Configuration class to instantiate
            file_name: Specific file name to load
            search_paths: Override search paths
            
        Returns:
            Loaded and validated configuration
        """
        # Use provided search paths or default
        paths = search_paths or self._search_paths
        
        # Load base configuration
        config_data = DEFAULT_CONFIGS.get(config_type, {}).copy()
        
        # Search for config files in hierarchical order
        if file_name is None:
            possible_names = [
                f"{config_type}.yaml",
                f"{config_type}.yml", 
                f"{config_type}.json",
                "config.yaml",
                "config.yml",
                "config.json"
            ]
        else:
            possible_names = [file_name]
            
        for search_path in paths:
            if not search_path.exists():
                continue
                
            for name in possible_names:
                config_file = search_path / name
                if config_file.exists():
                    logger.info(f"Loading config from: {config_file}")
                    try:
                        file_config = self._load_config_file(config_file)
                        # Merge with existing config (file takes precedence)
                        config_data.update(file_config)
                    except Exception as e:
                        logger.warning(f"Failed to load {config_file}: {e}")
        
        # Override with environment variables
        env_config = self._load_env_config(config_type)
        config_data.update(env_config)
        
        # Create and validate configuration
        try:
            config = config_class.from_dict(config_data)
            self._configs[config_type] = config
            logger.info(f"Successfully loaded {config_type} configuration")
            return config
        except Exception as e:
            logger.error(f"Failed to create {config_type} configuration: {e}")
            raise
    
    def get_config(self, config_type: str) -> Optional[Config]:
        """Get cached configuration."""
        return self._configs.get(config_type)
    
    def save_config(self, config_type: str, config: Config, file_path: Optional[Path] = None):
        """Save configuration to file."""
        if file_path is None:
            file_path = self.config_dir / f"{config_type}.yaml"
        
        config.save(file_path)
        self._configs[config_type] = config
        logger.info(f"Saved {config_type} configuration to: {file_path}")
    
    def list_configs(self) -> List[str]:
        """List available configuration types."""
        return list(self._configs.keys())
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all loaded configurations."""
        results = {}
        for config_type, config in self._configs.items():
            try:
                validate_config(config)
                results[config_type] = True
                logger.info(f"Configuration {config_type} is valid")
            except ValidationError as e:
                results[config_type] = False
                logger.error(f"Configuration {config_type} is invalid: {e}")
        return results
    
    def create_profile(self, profile_name: str, configs: Dict[str, Config]):
        """Create a configuration profile with multiple configs."""
        profile_dir = self.config_dir / "profiles" / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        for config_type, config in configs.items():
            config_file = profile_dir / f"{config_type}.yaml"
            config.save(config_file)
        
        # Save profile metadata
        metadata = {
            "name": profile_name,
            "created": str(Path(__file__).parent),
            "configs": list(configs.keys())
        }
        
        with open(profile_dir / "profile.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created configuration profile: {profile_name}")
    
    def load_profile(self, profile_name: str) -> Dict[str, Config]:
        """Load a configuration profile."""
        profile_dir = self.config_dir / "profiles" / profile_name
        
        if not profile_dir.exists():
            raise FileNotFoundError(f"Profile not found: {profile_name}")
        
        # Load profile metadata
        metadata_file = profile_dir / "profile.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            config_types = metadata.get("configs", [])
        else:
            # Discover config files
            config_types = []
            for config_file in profile_dir.glob("*.yaml"):
                if config_file.stem != "profile":
                    config_types.append(config_file.stem)
        
        # Load configurations
        configs = {}
        config_classes = {
            "training": TrainingConfig,
            "deployment": DeploymentConfig,
            "simulation": SimulationConfig
        }
        
        for config_type in config_types:
            config_class = config_classes.get(config_type)
            if config_class:
                config_file = profile_dir / f"{config_type}.yaml"
                if config_file.exists():
                    configs[config_type] = config_class.from_file(config_file)
                    self._configs[config_type] = configs[config_type]
        
        logger.info(f"Loaded configuration profile: {profile_name}")
        return configs
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_file.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")
    
    def _load_env_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        prefix = f"LIQUID_VISION_{config_type.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Convert string values to appropriate types
                env_config[config_key] = self._convert_env_value(value)
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try to parse as JSON first (handles lists, dicts, etc.)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value