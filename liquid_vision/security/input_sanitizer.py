"""
Advanced input sanitization for security hardening.
"""

import re
import os
import json
import html
import urllib.parse
import hashlib
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger('liquid_vision.security')


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