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

# Global-first configuration extensions
SUPPORTED_REGIONS = {
    'us-east-1': {'name': 'US East (N. Virginia)', 'compliance': ['CCPA', 'COPPA']},
    'us-west-2': {'name': 'US West (Oregon)', 'compliance': ['CCPA', 'COPPA']},
    'eu-west-1': {'name': 'Europe (Ireland)', 'compliance': ['GDPR', 'DSGVO']},
    'ap-northeast-1': {'name': 'Asia Pacific (Tokyo)', 'compliance': ['PDPA']},
    'ap-southeast-1': {'name': 'Asia Pacific (Singapore)', 'compliance': ['PDPA']},
}

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français', 
    'de': 'Deutsch',
    'ja': '日本語',
    'zh': '中文',
    'pt': 'Português',
    'ru': 'Русский',
}


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
    environment variable support, validation, and global-first capabilities.
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
        
        # Enhanced global configuration for worldwide deployment
        self.current_region = os.getenv('LIQUID_VISION_REGION', 'us-east-1')
        self.current_language = os.getenv('LIQUID_VISION_LANG', 'en')
        self.enabled_compliance = set()
        self.global_settings = {
            'multi_region': True,
            'i18n_enabled': True,
            'compliance_mode': True,
            'edge_cdn': True,
        }
        
        # Enhanced global configuration for worldwide deployment  
        self.global_config = {
            "regions": [
                # Americas
                "us-east-1", "us-west-2", "us-gov-west-1",
                "ca-central-1", "sa-east-1",
                # Europe  
                "eu-west-1", "eu-west-2", "eu-central-1", "eu-north-1",
                "eu-south-1", "eu-west-3",
                # Asia Pacific
                "ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
                "ap-northeast-2", "ap-south-1", "ap-east-1",
                # Middle East & Africa
                "me-south-1", "af-south-1",
                # China (special regions)
                "cn-north-1", "cn-northwest-1"
            ],
            "languages": [
                # Major languages with ISO 639-1 codes
                "en", "es", "fr", "de", "ja", "zh-cn", "zh-tw",
                "ko", "pt", "ru", "ar", "hi", "it", "nl", "sv",
                "no", "da", "fi", "pl", "tr", "th", "vi", "id"
            ],
            "compliance": {
                "gdpr": {
                    "enabled": True,
                    "data_retention_days": 1095,
                    "pseudonymization": True,
                    "right_to_be_forgotten": True,
                    "privacy_by_design": True
                },
                "ccpa": {
                    "enabled": True,
                    "opt_out_enabled": True,
                    "do_not_sell": True,
                    "privacy_rights_disclosure": True
                },
                "pdpa": {
                    "enabled": True,
                    "consent_required": True,
                    "data_breach_notification": True,
                    "singapore_compliance": True
                },
                "lgpd": {  # Brazil's General Data Protection Law
                    "enabled": True,
                    "lawful_basis_required": True,
                    "data_protection_officer": True
                },
                "pipeda": {  # Canada's Personal Information Protection
                    "enabled": True,
                    "purpose_limitation": True,
                    "consent_requirements": True
                },
                "privacy_act": {  # Australia Privacy Act
                    "enabled": True,
                    "notifiable_data_breaches": True,
                    "privacy_policy_required": True
                },
                "kvkk": {  # Turkey's Data Protection Law
                    "enabled": True,
                    "explicit_consent": True,
                    "data_controller_registration": True
                },
                "lei_geral": {  # Mexico's data protection
                    "enabled": True,
                    "privacy_notice_required": True
                },
                "popi": {  # South Africa Protection of Personal Information
                    "enabled": True,
                    "information_officer_required": True
                }
            },
            "localization": {
                "timezone_aware": True,
                "currency_support": [
                    "USD", "EUR", "JPY", "GBP", "CAD", "AUD", "CHF",
                    "CNY", "KRW", "SGD", "HKD", "INR", "BRL", "MXN",
                    "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "TRY"
                ],
                "date_formats": {
                    "us": "MM/DD/YYYY",
                    "eu": "DD/MM/YYYY",
                    "iso": "YYYY-MM-DD",
                    "japan": "YYYY年MM月DD日",
                    "china": "YYYY-MM-DD",
                    "korea": "YYYY.MM.DD"
                },
                "number_formats": {
                    "decimal_separator": {"us": ".", "eu": ","},
                    "thousands_separator": {"us": ",", "eu": "."},
                    "currency_position": {"us": "before", "eu": "after"}
                },
                "rtl_languages": ["ar", "he", "fa", "ur"],
                "accessibility": {
                    "wcag_compliance": "2.1_AA",
                    "screen_reader_support": True,
                    "high_contrast_mode": True,
                    "keyboard_navigation": True
                }
            },
            "edge_deployment": {
                "supported_architectures": [
                    "arm64", "armv7", "x86_64", "aarch64",
                    "riscv64", "mips", "esp32", "cortex-m"
                ],
                "optimization_levels": ["O0", "O1", "O2", "O3", "Os"],
                "memory_constraints": {
                    "tiny": "32KB",
                    "small": "128KB", 
                    "medium": "512KB",
                    "large": "2MB"
                }
            },
            "security": {
                "encryption": {
                    "at_rest": "AES-256-GCM",
                    "in_transit": "TLS-1.3",
                    "keys": "RSA-4096"
                },
                "authentication": {
                    "multi_factor": True,
                    "biometric_support": True,
                    "oauth2_providers": ["google", "microsoft", "apple", "github"]
                },
                "audit_logging": {
                    "enabled": True,
                    "retention_days": 2555,  # 7 years
                    "real_time_monitoring": True
                }
            }
        }
        
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
    
    # Global-First Implementation Methods
    def set_region(self, region: str) -> None:
        """Set the current deployment region."""
        if region not in SUPPORTED_REGIONS:
            raise ValueError(f"Unsupported region: {region}. Supported regions: {list(SUPPORTED_REGIONS.keys())}")
        
        self.current_region = region
        # Automatically enable compliance for the region
        region_compliance = SUPPORTED_REGIONS[region]['compliance']
        for compliance in region_compliance:
            self.enabled_compliance.add(compliance)
        
        logger.info(f"Region set to {region} ({SUPPORTED_REGIONS[region]['name']})")
        logger.info(f"Enabled compliance: {list(self.enabled_compliance)}")
    
    def set_language(self, language: str) -> None:
        """Set the current language/locale."""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
        
        self.current_language = language
        logger.info(f"Language set to {language} ({SUPPORTED_LANGUAGES[language]})")
    
    def enable_compliance(self, compliance_standard: str) -> None:
        """Enable a specific compliance standard."""
        valid_standards = set()
        for region_data in SUPPORTED_REGIONS.values():
            valid_standards.update(region_data['compliance'])
        
        if compliance_standard not in valid_standards:
            raise ValueError(f"Unsupported compliance: {compliance_standard}. Valid standards: {list(valid_standards)}")
        
        self.enabled_compliance.add(compliance_standard)
        logger.info(f"Enabled compliance standard: {compliance_standard}")
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get current global configuration status."""
        return {
            'region': {
                'current': self.current_region,
                'name': SUPPORTED_REGIONS[self.current_region]['name'],
                'supported': list(SUPPORTED_REGIONS.keys())
            },
            'language': {
                'current': self.current_language,
                'name': SUPPORTED_LANGUAGES[self.current_language],
                'supported': list(SUPPORTED_LANGUAGES.keys())
            },
            'compliance': {
                'enabled': list(self.enabled_compliance),
                'available': list(set().union(*[r['compliance'] for r in SUPPORTED_REGIONS.values()]))
            },
            'global_settings': self.global_settings.copy()
        }
    
    def is_gdpr_compliant(self) -> bool:
        """Check if GDPR compliance is enabled."""
        return 'GDPR' in self.enabled_compliance
    
    def is_ccpa_compliant(self) -> bool:
        """Check if CCPA compliance is enabled."""
        return 'CCPA' in self.enabled_compliance
    
    def get_region_optimized_config(self, config_type: str) -> Dict[str, Any]:
        """Get region-optimized configuration settings."""
        base_config = DEFAULT_CONFIGS.get(config_type, {}).copy()
        
        # Apply region-specific optimizations
        if self.current_region.startswith('eu-'):
            # EU region optimizations
            base_config.update({
                'privacy_enhanced': True,
                'data_retention_days': 30 if 'GDPR' in self.enabled_compliance else 90,
                'encryption_required': True,
                'user_consent_required': True
            })
        elif self.current_region.startswith('ap-'):
            # Asia-Pacific optimizations
            base_config.update({
                'latency_optimized': True,
                'edge_processing_preferred': True,
                'bandwidth_conscious': True
            })
        else:
            # Default optimizations
            base_config.update({
                'performance_optimized': True,
                'cost_optimized': True
            })
        
        # Apply compliance requirements
        if self.enabled_compliance:
            base_config.update({
                'audit_logging': True,
                'secure_transport': True,
                'data_minimization': True,
                'user_rights_enabled': True
            })
        
        return base_config


# Global Configuration Manager Instance
GlobalConfigManager = ConfigManager