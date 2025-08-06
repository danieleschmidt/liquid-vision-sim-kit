"""
Advanced input sanitization and security hardening for liquid neural networks.
Comprehensive protection against various security threats and vulnerabilities.
"""

import re
import os
import json
import html
import urllib.parse
import hashlib
import base64
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
import logging
import torch
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.logging import get_logger
from ..utils.error_handling import ValidationError, LiquidVisionError, ErrorCategory

logger = get_logger(__name__)


class SanitizationError(Exception):
    """Exception raised when input sanitization fails."""
    pass


class InputSanitizer:
    """
    Comprehensive input sanitization to prevent injection attacks
    and ensure safe handling of user inputs.
    """
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',               # JavaScript URLs
        r'data:text/html',           # HTML data URLs
        r'vbscript:',                # VBScript
        r'onload\s*=',               # Event handlers
        r'onerror\s*=',
        r'onclick\s*=',
        r'__import__',               # Python imports
        r'eval\s*\(',                # Code evaluation
        r'exec\s*\(',
        r'subprocess\.',             # System commands
        r'os\.',
        r'sys\.',
        r'\.\./|\.\.\\',             # Path traversal
        r'file:///',                 # File URLs
        r'ftp://',                   # FTP URLs
    ]
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r"union\s+select",
        r"drop\s+table",
        r"delete\s+from",
        r"insert\s+into",
        r"update\s+.+set",
        r"create\s+table",
        r"alter\s+table",
        r";.*--",
        r"'.*or.*'.*'",
        r'".*or.*".*"',
    ]
    
    # File extension whitelist
    SAFE_EXTENSIONS = {
        '.txt', '.json', '.yaml', '.yml', '.csv', '.log',
        '.h5', '.hdf5', '.npz', '.npy', '.mat', '.pkl',
        '.png', '.jpg', '.jpeg', '.bmp', '.tiff',
        '.py', '.c', '.cpp', '.h', '.hpp',
        '.md', '.rst', '.pdf'
    }
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize input sanitizer.
        
        Args:
            strict_mode: If True, applies stricter sanitization rules
        """
        self.strict_mode = strict_mode
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.dangerous_regex = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.DANGEROUS_PATTERNS
        ]
        
        self.sql_regex = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SQL_PATTERNS
        ]
    
    def sanitize_string(
        self, 
        value: str, 
        max_length: int = 10000,
        allow_html: bool = False,
        allow_unicode: bool = True
    ) -> str:
        """
        Sanitize string input to prevent injection attacks.
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            allow_unicode: Whether to allow Unicode characters
            
        Returns:
            Sanitized string
            
        Raises:
            SanitizationError: If string contains dangerous content
        """
        if not isinstance(value, str):
            raise SanitizationError(f"Expected string, got {type(value)}")
        
        # Check length
        if len(value) > max_length:
            if self.strict_mode:
                raise SanitizationError(f"String too long: {len(value)} > {max_length}")
            else:
                value = value[:max_length]
                logger.warning(f"String truncated to {max_length} characters")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_regex:
            if pattern.search(value):
                raise SanitizationError(f"Dangerous pattern detected: {pattern.pattern}")
        
        # Check for SQL injection patterns
        for pattern in self.sql_regex:
            if pattern.search(value):
                raise SanitizationError(f"SQL injection pattern detected: {pattern.pattern}")
        
        # HTML escaping
        if not allow_html:
            value = html.escape(value)
        
        # Unicode handling
        if not allow_unicode:
            # Keep only ASCII characters
            value = value.encode('ascii', 'ignore').decode('ascii')
        
        # URL decode to catch encoded attacks
        try:
            decoded = urllib.parse.unquote(value)
            if decoded != value:
                # Recursively check decoded content
                return self.sanitize_string(decoded, max_length, allow_html, allow_unicode)
        except Exception:
            pass  # If URL decoding fails, continue with original
        
        return value
    
    def sanitize_file_path(
        self, 
        path: Union[str, Path],
        allowed_roots: Optional[List[str]] = None,
        require_safe_extension: bool = True
    ) -> Path:
        """
        Sanitize file paths to prevent directory traversal attacks.
        
        Args:
            path: File path to sanitize
            allowed_roots: List of allowed root directories
            require_safe_extension: Whether to require safe file extensions
            
        Returns:
            Sanitized Path object
            
        Raises:
            SanitizationError: If path is unsafe
        """
        if not isinstance(path, (str, Path)):
            raise SanitizationError(f"Path must be string or Path, got {type(path)}")
        
        path_obj = Path(path)
        path_str = str(path_obj)
        
        # Check for dangerous patterns in path
        dangerous_chars = ['..', '<', '>', '|', '&', ';', '$', '`', '"', "'"]
        for char in dangerous_chars:
            if char in path_str:
                raise SanitizationError(f"Dangerous character in path: {char}")
        
        # Check for null bytes
        if '\x00' in path_str:
            raise SanitizationError("Null byte in path")
        
        # Resolve path to catch traversal attempts
        try:
            resolved_path = path_obj.resolve()
        except OSError as e:
            raise SanitizationError(f"Cannot resolve path: {e}")
        
        # Check allowed roots
        if allowed_roots:
            allowed = False
            for root in allowed_roots:
                try:
                    resolved_path.relative_to(Path(root).resolve())
                    allowed = True
                    break
                except ValueError:
                    continue
            
            if not allowed:
                raise SanitizationError(f"Path outside allowed directories: {resolved_path}")
        
        # Check file extension
        if require_safe_extension and path_obj.suffix:
            if path_obj.suffix.lower() not in self.SAFE_EXTENSIONS:
                raise SanitizationError(f"Unsafe file extension: {path_obj.suffix}")
        
        return resolved_path
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            raise SanitizationError(f"Filename must be string, got {type(filename)}")
        
        # Remove directory components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Remove leading dots and spaces
        filename = filename.lstrip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        # Ensure not empty
        if not filename:
            filename = 'sanitized_file'
        
        return filename
    
    def sanitize_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize configuration dictionary.
        
        Args:
            config: Configuration dictionary to sanitize
            
        Returns:
            Sanitized configuration dictionary
        """
        if not isinstance(config, dict):
            raise SanitizationError(f"Config must be dictionary, got {type(config)}")
        
        sanitized = {}
        
        for key, value in config.items():
            # Sanitize key
            if not isinstance(key, str):
                raise SanitizationError(f"Config key must be string, got {type(key)}")
            
            sanitized_key = self.sanitize_string(key, max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized_value = self.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized_value = self.sanitize_config_dict(value)
            elif isinstance(value, list):
                sanitized_value = self.sanitize_list(value)
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized_value = value
            else:
                # Convert unknown types to string and sanitize
                sanitized_value = self.sanitize_string(str(value))
                logger.warning(f"Unknown config value type {type(value)}, converted to string")
            
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized
    
    def sanitize_list(self, items: List[Any]) -> List[Any]:
        """
        Sanitize list of items.
        
        Args:
            items: List to sanitize
            
        Returns:
            Sanitized list
        """
        if not isinstance(items, list):
            raise SanitizationError(f"Expected list, got {type(items)}")
        
        sanitized = []
        for item in items:
            if isinstance(item, str):
                sanitized.append(self.sanitize_string(item))
            elif isinstance(item, dict):
                sanitized.append(self.sanitize_config_dict(item))
            elif isinstance(item, list):
                sanitized.append(self.sanitize_list(item))
            elif isinstance(item, (int, float, bool, type(None))):
                sanitized.append(item)
            else:
                sanitized.append(self.sanitize_string(str(item)))
        
        return sanitized
    
    def check_content_safety(self, content: bytes, max_size: int = 100 * 1024 * 1024) -> bool:
        """
        Check if file content is safe to process.
        
        Args:
            content: File content as bytes
            max_size: Maximum allowed file size
            
        Returns:
            True if content is safe, False otherwise
        """
        # Check file size
        if len(content) > max_size:
            logger.warning(f"File too large: {len(content)} > {max_size}")
            return False
        
        # Check for binary executable signatures
        dangerous_signatures = [
            b'\x7fELF',          # Linux executable
            b'MZ',               # Windows executable
            b'\xfe\xed\xfa',     # Mach-O executable
            b'PK',               # ZIP archive (could contain malware)
        ]
        
        for sig in dangerous_signatures:
            if content.startswith(sig):
                logger.warning(f"Dangerous file signature detected: {sig}")
                return False
        
        # Check for embedded scripts in common file types
        content_str = content.decode('utf-8', errors='ignore').lower()
        
        script_patterns = [
            '<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=',
            '<?php', '<%', 'exec(', 'eval(', 'system(', 'shell_exec('
        ]
        
        for pattern in script_patterns:
            if pattern in content_str:
                logger.warning(f"Script pattern detected in file: {pattern}")
                return False
        
        return True
    
    def generate_safe_hash(self, data: Union[str, bytes]) -> str:
        """
        Generate a safe hash for data identification.
        
        Args:
            data: Data to hash
            
        Returns:
            SHA-256 hash as hex string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hash_obj = hashlib.sha256(data)
        return hash_obj.hexdigest()
    
    def create_sandbox_filename(self, original_filename: str) -> str:
        """
        Create a sandbox-safe filename with hash.
        
        Args:
            original_filename: Original filename
            
        Returns:
            Safe filename with hash prefix
        """
        sanitized_name = self.sanitize_filename(original_filename)
        name_hash = self.generate_safe_hash(sanitized_name)[:8]
        
        # Combine hash with sanitized name
        name, ext = os.path.splitext(sanitized_name)
        safe_name = f"{name_hash}_{name}"
        
        # Limit total length
        if len(safe_name + ext) > 200:
            safe_name = safe_name[:200-len(ext)]
        
        return safe_name + ext


# Global sanitizer instance
default_sanitizer = InputSanitizer(strict_mode=True)


def sanitize_user_input(
    data: Any,
    input_type: str = "string",
    **kwargs
) -> Any:
    """
    Convenient function to sanitize user input.
    
    Args:
        data: Input data to sanitize
        input_type: Type of input (string, path, filename, config, list)
        **kwargs: Additional sanitization parameters
        
    Returns:
        Sanitized data
    """
    sanitizer = default_sanitizer
    
    method_map = {
        'string': sanitizer.sanitize_string,
        'path': sanitizer.sanitize_file_path,
        'filename': sanitizer.sanitize_filename,
        'config': sanitizer.sanitize_config_dict,
        'list': sanitizer.sanitize_list,
    }
    
    method = method_map.get(input_type)
    if method is None:
        logger.warning(f"No sanitizer for input type: {input_type}")
        return data
    
    try:
        return method(data, **kwargs)
    except Exception as e:
        logger.error(f"Sanitization failed for {input_type}: {e}")
        raise SanitizationError(f"Failed to sanitize {input_type}: {e}")


def sanitize_decorator(input_type: str = "string", **sanitize_kwargs):
    """
    Decorator to automatically sanitize function arguments.
    
    Args:
        input_type: Type of input to sanitize
        **sanitize_kwargs: Additional sanitization parameters
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Sanitize string arguments
            sanitized_args = []
            for arg in args:
                if isinstance(arg, str):
                    sanitized_args.append(
                        sanitize_user_input(arg, input_type, **sanitize_kwargs)
                    )
                else:
                    sanitized_args.append(arg)
            
            # Sanitize keyword arguments
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    sanitized_kwargs[key] = sanitize_user_input(
                        value, input_type, **sanitize_kwargs
                    )
                else:
                    sanitized_kwargs[key] = value
            
            return func(*sanitized_args, **sanitized_kwargs)
        
        return wrapper
    return decorator


class SecureModelValidator:
    """Comprehensive security validation for model inputs and outputs."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.threat_patterns = self._load_threat_patterns()
        self.validation_cache = {}
        self.validation_stats = {
            'total_validations': 0,
            'threats_detected': 0,
            'cache_hits': 0
        }
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load known threat patterns for detection."""
        return {
            'adversarial_indicators': [
                'epsilon', 'perturbation', 'fgsm', 'pgd', 'carlini',
                'deepfool', 'adversarial', 'attack', 'poison'
            ],
            'model_extraction': [
                'query', 'extract', 'steal', 'replica', 'distill',
                'membership_inference', 'model_inversion'
            ],
            'backdoor_indicators': [
                'trigger', 'backdoor', 'trojan', 'poison', 'watermark'
            ]
        }
    
    def validate_tensor_input(self, tensor: torch.Tensor, context: str = "unknown") -> bool:
        """
        Validate tensor input for security threats.
        
        Args:
            tensor: Input tensor to validate
            context: Context description for logging
            
        Returns:
            True if tensor is safe, raises exception if threats detected
        """
        self.validation_stats['total_validations'] += 1
        
        # Generate validation key for caching
        tensor_hash = self._compute_tensor_hash(tensor)
        cache_key = f"{tensor_hash}_{context}"
        
        if cache_key in self.validation_cache:
            self.validation_stats['cache_hits'] += 1
            return self.validation_cache[cache_key]
        
        try:
            # Check for NaN/Inf attacks
            self._check_numerical_stability(tensor, context)
            
            # Check for adversarial patterns
            self._check_adversarial_patterns(tensor, context)
            
            # Check tensor properties
            self._check_tensor_properties(tensor, context)
            
            # Check for unusual statistical properties
            self._check_statistical_anomalies(tensor, context)
            
            # Cache successful validation
            self.validation_cache[cache_key] = True
            return True
            
        except ValidationError as e:
            self.validation_stats['threats_detected'] += 1
            self.logger.warning(f"Security threat detected in tensor input ({context}): {e}")
            raise
    
    def _compute_tensor_hash(self, tensor: torch.Tensor) -> str:
        """Compute hash of tensor for caching."""
        # Use tensor statistics for hash to avoid memory issues
        stats = torch.tensor([
            tensor.mean().item(),
            tensor.std().item(),
            tensor.min().item(),
            tensor.max().item(),
            float(tensor.shape[0]) if tensor.numel() > 0 else 0
        ])
        return hashlib.md5(stats.numpy().tobytes()).hexdigest()[:16]
    
    def _check_numerical_stability(self, tensor: torch.Tensor, context: str):
        """Check for numerical stability attacks."""
        if torch.isnan(tensor).any():
            raise ValidationError(
                f"NaN values detected in {context}",
                category=ErrorCategory.VALIDATION,
                details={'nan_count': torch.isnan(tensor).sum().item()}
            )
        
        if torch.isinf(tensor).any():
            raise ValidationError(
                f"Infinite values detected in {context}",
                category=ErrorCategory.VALIDATION,
                details={'inf_count': torch.isinf(tensor).sum().item()}
            )
        
        # Check for extreme values that might cause overflow
        max_val = tensor.max().abs().item()
        if max_val > 1e6:
            self.logger.warning(f"Extremely large values in tensor ({context}): max={max_val}")
    
    def _check_adversarial_patterns(self, tensor: torch.Tensor, context: str):
        """Check for known adversarial attack patterns."""
        # Check for uniform random noise (common in adversarial attacks)
        if tensor.numel() > 100:  # Only check for reasonably sized tensors
            tensor_flat = tensor.view(-1)
            
            # Check if values are suspiciously uniform
            hist, _ = torch.histogram(tensor_flat, bins=50)
            uniformity = hist.std() / (hist.mean() + 1e-8)
            
            if uniformity < 0.1 and tensor_flat.std() > 0.01:
                self.logger.warning(f"Suspiciously uniform distribution in {context}")
        
        # Check for high-frequency noise patterns
        if tensor.dim() >= 2 and tensor.shape[-1] > 8 and tensor.shape[-2] > 8:
            # Compute high-frequency components using simple differences
            diff_x = torch.diff(tensor, dim=-1).abs().mean()
            diff_y = torch.diff(tensor, dim=-2).abs().mean()
            
            mean_val = tensor.abs().mean()
            if mean_val > 0 and (diff_x / mean_val > 0.5 or diff_y / mean_val > 0.5):
                self.logger.warning(f"High-frequency patterns detected in {context}")
    
    def _check_tensor_properties(self, tensor: torch.Tensor, context: str):
        """Check basic tensor properties for anomalies."""
        # Check tensor size limits
        max_elements = 100_000_000  # 100M elements
        if tensor.numel() > max_elements:
            raise ValidationError(
                f"Tensor too large in {context}: {tensor.numel()} > {max_elements}",
                category=ErrorCategory.VALIDATION,
                details={'tensor_shape': list(tensor.shape)}
            )
        
        # Check for unusual tensor shapes
        if tensor.dim() > 6:
            self.logger.warning(f"Unusual tensor dimensionality in {context}: {tensor.dim()}D")
        
        # Check for very sparse tensors (might indicate crafted input)
        if tensor.numel() > 1000:
            zero_fraction = (tensor == 0).float().mean().item()
            if zero_fraction > 0.99:
                self.logger.warning(f"Extremely sparse tensor in {context}: {zero_fraction:.1%} zeros")
    
    def _check_statistical_anomalies(self, tensor: torch.Tensor, context: str):
        """Check for statistical anomalies that might indicate attacks."""
        if tensor.numel() < 10:
            return  # Skip for very small tensors
        
        tensor_flat = tensor.view(-1)
        
        # Check for extreme statistical properties
        mean_val = tensor_flat.mean().item()
        std_val = tensor_flat.std().item()
        
        # Check for extreme skewness
        if std_val > 1e-6:  # Avoid division by zero
            centered = tensor_flat - mean_val
            skewness = (centered ** 3).mean() / (std_val ** 3)
            
            if abs(skewness) > 10:
                self.logger.warning(f"Extreme skewness in {context}: {skewness:.3f}")
        
        # Check for bimodal distributions (might indicate mixed legitimate/adversarial data)
        if tensor.numel() > 1000:
            # Simple bimodality check using Hartigan's dip test approximation
            sorted_vals, _ = torch.sort(tensor_flat)
            n = len(sorted_vals)
            
            # Check for gaps in the distribution
            diffs = torch.diff(sorted_vals)
            large_gaps = (diffs > 3 * diffs.median()).sum().item()
            
            if large_gaps > n * 0.05:  # More than 5% large gaps
                self.logger.warning(f"Unusual distribution gaps in {context}")
    
    def validate_model_outputs(self, outputs: torch.Tensor, expected_shape: Optional[tuple] = None) -> bool:
        """
        Validate model outputs for security and correctness.
        
        Args:
            outputs: Model output tensor
            expected_shape: Expected output shape
            
        Returns:
            True if outputs are valid
        """
        # Basic tensor validation
        self.validate_tensor_input(outputs, "model_output")
        
        # Shape validation
        if expected_shape and outputs.shape != expected_shape:
            raise ValidationError(
                f"Model output shape mismatch: expected {expected_shape}, got {outputs.shape}",
                category=ErrorCategory.VALIDATION
            )
        
        # Check for reasonable output ranges
        if outputs.dtype.is_floating_point:
            min_val, max_val = outputs.min().item(), outputs.max().item()
            
            # Check for extremely large outputs (potential overflow)
            if max_val > 1e6 or min_val < -1e6:
                self.logger.warning(f"Extreme output values: min={min_val:.3e}, max={max_val:.3e}")
            
            # Check for outputs stuck at extreme values (might indicate attack)
            if outputs.numel() > 10:
                extreme_fraction = ((outputs.abs() > 1e3).float().mean()).item()
                if extreme_fraction > 0.9:
                    self.logger.warning(f"Most outputs are extreme values: {extreme_fraction:.1%}")
        
        return True
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self.validation_stats,
            'cache_size': len(self.validation_cache),
            'threat_detection_rate': (
                self.validation_stats['threats_detected'] / 
                max(1, self.validation_stats['total_validations'])
            ),
            'cache_hit_rate': (
                self.validation_stats['cache_hits'] / 
                max(1, self.validation_stats['total_validations'])
            )
        }
    
    def clear_cache(self):
        """Clear validation cache."""
        self.validation_cache.clear()


class SecureDataEncryption:
    """Encryption utilities for secure data handling."""
    
    def __init__(self, password: Optional[str] = None):
        self.logger = get_logger(__name__)
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password."""
        if salt is None:
            salt = b'liquid_vision_salt_2023'  # Use consistent salt for reproducibility
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: Union[str, bytes, Dict, torch.Tensor]) -> bytes:
        """
        Encrypt various types of data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as bytes
        """
        try:
            # Convert different data types to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            elif isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, torch.Tensor):
                data_bytes = data.detach().cpu().numpy().tobytes()
            else:
                # Convert to string first
                data_bytes = str(data).encode('utf-8')
            
            # Encrypt the data
            encrypted_data = self.cipher.encrypt(data_bytes)
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise LiquidVisionError(
                f"Failed to encrypt data: {e}",
                category=ErrorCategory.CONFIGURATION
            )
    
    def decrypt_data(self, encrypted_data: bytes, data_type: str = "string") -> Any:
        """
        Decrypt data and convert back to specified type.
        
        Args:
            encrypted_data: Encrypted data bytes
            data_type: Expected data type (string, bytes, dict, tensor)
            
        Returns:
            Decrypted data in specified format
        """
        try:
            # Decrypt the data
            decrypted_bytes = self.cipher.decrypt(encrypted_data)
            
            # Convert back to requested type
            if data_type == "string":
                return decrypted_bytes.decode('utf-8')
            elif data_type == "bytes":
                return decrypted_bytes
            elif data_type == "dict":
                return json.loads(decrypted_bytes.decode('utf-8'))
            elif data_type == "tensor":
                # This would need shape information to reconstruct properly
                return torch.from_numpy(np.frombuffer(decrypted_bytes, dtype=np.float32))
            else:
                return decrypted_bytes.decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise LiquidVisionError(
                f"Failed to decrypt data: {e}",
                category=ErrorCategory.CONFIGURATION
            )
    
    def encrypt_model_weights(self, model: torch.nn.Module) -> Dict[str, bytes]:
        """
        Encrypt model weights for secure storage.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary of encrypted weight data
        """
        encrypted_weights = {}
        
        try:
            for name, param in model.state_dict().items():
                # Convert parameter to bytes
                param_bytes = param.detach().cpu().numpy().tobytes()
                
                # Add shape and dtype metadata
                metadata = {
                    'shape': list(param.shape),
                    'dtype': str(param.dtype),
                    'data': param_bytes.hex()  # Convert to hex for JSON serialization
                }
                
                # Encrypt the metadata
                encrypted_weights[name] = self.encrypt_data(metadata)
            
            self.logger.info(f"Encrypted {len(encrypted_weights)} model parameters")
            return encrypted_weights
            
        except Exception as e:
            self.logger.error(f"Model weight encryption failed: {e}")
            raise LiquidVisionError(
                f"Failed to encrypt model weights: {e}",
                category=ErrorCategory.CONFIGURATION
            )
    
    def decrypt_model_weights(self, encrypted_weights: Dict[str, bytes]) -> Dict[str, torch.Tensor]:
        """
        Decrypt model weights for loading.
        
        Args:
            encrypted_weights: Dictionary of encrypted weight data
            
        Returns:
            Dictionary of decrypted tensors
        """
        decrypted_weights = {}
        
        try:
            for name, encrypted_data in encrypted_weights.items():
                # Decrypt metadata
                metadata = self.decrypt_data(encrypted_data, "dict")
                
                # Reconstruct tensor
                shape = metadata['shape']
                dtype_str = metadata['dtype']
                data_hex = metadata['data']
                
                # Convert hex back to bytes
                param_bytes = bytes.fromhex(data_hex)
                
                # Reconstruct numpy array
                param_np = np.frombuffer(param_bytes, dtype=np.float32).reshape(shape)
                
                # Convert to tensor with correct dtype
                if 'float' in dtype_str:
                    tensor = torch.from_numpy(param_np.astype(np.float32))
                else:
                    tensor = torch.from_numpy(param_np)
                
                decrypted_weights[name] = tensor
            
            self.logger.info(f"Decrypted {len(decrypted_weights)} model parameters")
            return decrypted_weights
            
        except Exception as e:
            self.logger.error(f"Model weight decryption failed: {e}")
            raise LiquidVisionError(
                f"Failed to decrypt model weights: {e}",
                category=ErrorCategory.CONFIGURATION
            )


class RateLimiter:
    """Rate limiting for API endpoints and resource access."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = defaultdict(deque)
        self.logger = get_logger(__name__)
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier for the client/request
            
        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        client_requests = self.requests[identifier]
        
        # Remove old requests outside time window
        while client_requests and now - client_requests[0] > self.time_window:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) >= self.max_requests:
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        client_requests = self.requests[identifier]
        
        # Clean old requests
        while client_requests and now - client_requests[0] > self.time_window:
            client_requests.popleft()
        
        return max(0, self.max_requests - len(client_requests))
    
    def reset_client(self, identifier: str):
        """Reset rate limit for specific client."""
        self.requests[identifier].clear()


def create_security_suite() -> Dict[str, Any]:
    """Create comprehensive security suite."""
    return {
        'input_sanitizer': InputSanitizer(strict_mode=True),
        'model_validator': SecureModelValidator(),
        'data_encryption': SecureDataEncryption(),
        'rate_limiter': RateLimiter(),
        'threat_patterns': SecureModelValidator()._load_threat_patterns()
    }


# Security decorators
def secure_input(input_type: str = "string", **kwargs):
    """Decorator for automatic input sanitization."""
    return sanitize_decorator(input_type, **kwargs)


def rate_limited(max_requests: int = 100, time_window: int = 3600):
    """Decorator for rate limiting function calls."""
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use function name as identifier (could be enhanced with user context)
            identifier = f"{func.__module__}.{func.__name__}"
            
            if not limiter.is_allowed(identifier):
                raise LiquidVisionError(
                    f"Rate limit exceeded for {func.__name__}",
                    category=ErrorCategory.VALIDATION
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator