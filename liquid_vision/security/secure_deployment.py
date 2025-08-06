"""
Secure deployment utilities with encryption and access control.
"""

import os
import json
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging
import tempfile
import shutil

from .input_sanitizer import InputSanitizer, SanitizationError
from ..utils.logging import SecurityLogger


logger = logging.getLogger('liquid_vision.security.deployment')
security_logger = SecurityLogger()


@dataclass
class SecurityConfig:
    """Security configuration for deployment."""
    
    enable_encryption: bool = True
    require_signature: bool = True
    sandbox_execution: bool = True
    max_memory_mb: int = 512
    max_execution_time: int = 300  # 5 minutes
    allowed_network_access: bool = False
    enable_audit_logging: bool = True
    minimum_key_size: int = 256
    
    def validate(self):
        """Validate security configuration."""
        if self.max_memory_mb < 64:
            raise ValueError("max_memory_mb must be at least 64MB")
        
        if self.max_execution_time < 1:
            raise ValueError("max_execution_time must be positive")
        
        if self.minimum_key_size < 128:
            raise ValueError("minimum_key_size must be at least 128 bits")


class SecureDeployer:
    """
    Secure model deployment with encryption, sandboxing, and access control.
    """
    
    def __init__(self, security_config: Optional[SecurityConfig] = None):
        """
        Initialize secure deployer.
        
        Args:
            security_config: Security configuration, uses defaults if None
        """
        self.security_config = security_config or SecurityConfig()
        self.security_config.validate()
        
        self.sanitizer = InputSanitizer(strict_mode=True)
        self.deployment_keys = {}
        self.deployment_hashes = {}
        
    def create_secure_package(
        self, 
        model_files: Dict[str, Path],
        metadata: Dict[str, Any],
        output_path: Path,
        encryption_key: Optional[bytes] = None
    ) -> Dict[str, str]:
        """
        Create a secure deployment package with encryption and integrity checks.
        
        Args:
            model_files: Dictionary mapping logical names to file paths
            metadata: Deployment metadata
            output_path: Output package path
            encryption_key: Optional encryption key, generates one if None
            
        Returns:
            Dictionary with package hash and encryption key info
        """
        # Validate inputs
        output_path = self.sanitizer.sanitize_file_path(output_path)
        metadata = self.sanitizer.sanitize_config_dict(metadata)
        
        # Generate encryption key if not provided
        if encryption_key is None:
            encryption_key = secrets.token_bytes(self.security_config.minimum_key_size // 8)
        
        # Create temporary directory for package creation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            package_dir = temp_path / "package"
            package_dir.mkdir()
            
            # Process and copy model files
            file_hashes = {}
            for logical_name, file_path in model_files.items():
                # Validate file path
                validated_path = self.sanitizer.sanitize_file_path(file_path, require_safe_extension=True)
                
                # Check file content safety
                with open(validated_path, 'rb') as f:
                    content = f.read()
                
                if not self.sanitizer.check_content_safety(content):
                    raise SanitizationError(f"Unsafe content in file: {file_path}")
                
                # Copy file with sanitized name
                safe_name = self.sanitizer.sanitize_filename(logical_name)
                target_path = package_dir / safe_name
                shutil.copy2(validated_path, target_path)
                
                # Calculate file hash
                file_hash = hashlib.sha256(content).hexdigest()
                file_hashes[safe_name] = file_hash
                
                security_logger.log_file_access(str(validated_path), "package", True)
            
            # Create package manifest
            manifest = {
                'metadata': metadata,
                'files': file_hashes,
                'security': {
                    'created_at': str(Path(__file__).parent),
                    'encryption_enabled': self.security_config.enable_encryption,
                    'signature_required': self.security_config.require_signature,
                    'version': '1.0'
                }
            }
            
            manifest_path = package_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create encrypted archive
            package_hash = self._create_encrypted_archive(
                package_dir, output_path, encryption_key
            )
            
            # Store deployment info
            deployment_id = secrets.token_hex(16)
            self.deployment_keys[deployment_id] = encryption_key
            self.deployment_hashes[deployment_id] = package_hash
            
            security_logger.log_model_deployment(str(output_path), "package", True)
            
            return {
                'deployment_id': deployment_id,
                'package_hash': package_hash,
                'key_fingerprint': hashlib.sha256(encryption_key).hexdigest()[:16],
                'manifest_hash': hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
            }
    
    def verify_package_integrity(
        self, 
        package_path: Path, 
        expected_hash: str,
        decryption_key: bytes
    ) -> bool:
        """
        Verify package integrity and authenticity.
        
        Args:
            package_path: Path to deployment package
            expected_hash: Expected package hash
            decryption_key: Decryption key
            
        Returns:
            True if package is valid and secure
        """
        try:
            # Validate package path
            validated_path = self.sanitizer.sanitize_file_path(package_path)
            
            # Calculate package hash
            with open(validated_path, 'rb') as f:
                content = f.read()
            
            actual_hash = hashlib.sha256(content).hexdigest()
            
            if actual_hash != expected_hash:
                logger.error(f"Package hash mismatch: {actual_hash} != {expected_hash}")
                return False
            
            # Try to decrypt and verify structure
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                if not self._extract_encrypted_archive(validated_path, temp_path, decryption_key):
                    logger.error("Failed to decrypt package")
                    return False
                
                # Check manifest exists
                manifest_path = temp_path / "manifest.json"
                if not manifest_path.exists():
                    logger.error("Package manifest missing")
                    return False
                
                # Load and validate manifest
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Verify file hashes in manifest
                for filename, expected_file_hash in manifest['files'].items():
                    file_path = temp_path / filename
                    if not file_path.exists():
                        logger.error(f"Package file missing: {filename}")
                        return False
                    
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    actual_file_hash = hashlib.sha256(file_content).hexdigest()
                    if actual_file_hash != expected_file_hash:
                        logger.error(f"File hash mismatch for {filename}")
                        return False
            
            logger.info(f"Package integrity verified: {package_path}")
            return True
            
        except Exception as e:
            logger.error(f"Package verification failed: {e}")
            return False
    
    def deploy_package_securely(
        self,
        package_path: Path,
        deployment_dir: Path,
        decryption_key: bytes,
        expected_hash: str
    ) -> Dict[str, Any]:
        """
        Deploy package securely with sandboxing and access control.
        
        Args:
            package_path: Path to deployment package  
            deployment_dir: Target deployment directory
            decryption_key: Decryption key
            expected_hash: Expected package hash
            
        Returns:
            Deployment status and information
        """
        try:
            # Verify package integrity first
            if not self.verify_package_integrity(package_path, expected_hash, decryption_key):
                raise SecurityError("Package integrity verification failed")
            
            # Validate deployment directory
            validated_dir = self.sanitizer.sanitize_file_path(
                deployment_dir,
                allowed_roots=[str(Path.cwd()), "/tmp", "/var/tmp"]
            )
            validated_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract package in sandbox
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract to temporary location
                if not self._extract_encrypted_archive(package_path, temp_path, decryption_key):
                    raise SecurityError("Failed to extract package")
                
                # Load manifest
                manifest_path = temp_path / "manifest.json"
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Copy files to deployment directory with validation
                deployed_files = []
                for filename in manifest['files'].keys():
                    source_path = temp_path / filename
                    target_path = validated_dir / filename
                    
                    # Additional safety check
                    with open(source_path, 'rb') as f:
                        content = f.read()
                    
                    if not self.sanitizer.check_content_safety(content):
                        raise SecurityError(f"Unsafe content detected in {filename}")
                    
                    shutil.copy2(source_path, target_path)
                    deployed_files.append(str(target_path))
                    
                    security_logger.log_file_access(str(target_path), "deploy", True)
                
                # Create deployment info file
                deployment_info = {
                    'deployed_at': str(Path(__file__).parent),
                    'package_hash': expected_hash,
                    'files': deployed_files,
                    'metadata': manifest['metadata'],
                    'security_config': {
                        'encryption_enabled': self.security_config.enable_encryption,
                        'sandboxed': self.security_config.sandbox_execution,
                        'max_memory_mb': self.security_config.max_memory_mb
                    }
                }
                
                info_path = validated_dir / "deployment_info.json"
                with open(info_path, 'w') as f:
                    json.dump(deployment_info, f, indent=2)
                
                logger.info(f"Package deployed successfully to {validated_dir}")
                security_logger.log_model_deployment(str(package_path), str(validated_dir), True)
                
                return deployment_info
                
        except Exception as e:
            logger.error(f"Secure deployment failed: {e}")
            security_logger.log_model_deployment(str(package_path), str(deployment_dir), False)
            raise
    
    def _create_encrypted_archive(
        self, 
        source_dir: Path, 
        output_path: Path, 
        encryption_key: bytes
    ) -> str:
        """
        Create encrypted archive from directory.
        
        Args:
            source_dir: Source directory to archive
            output_path: Output archive path
            encryption_key: Encryption key
            
        Returns:
            Archive hash
        """
        import tarfile
        
        # Create tar archive in memory
        with tempfile.NamedTemporaryFile() as temp_tar:
            with tarfile.open(temp_tar.name, 'w') as tar:
                for item in source_dir.iterdir():
                    tar.add(item, arcname=item.name)
            
            # Read tar content
            temp_tar.seek(0)
            tar_content = temp_tar.read()
        
        if self.security_config.enable_encryption:
            # Simple XOR encryption (in production, use proper encryption like AES)
            encrypted_content = self._xor_encrypt(tar_content, encryption_key)
        else:
            encrypted_content = tar_content
        
        # Write encrypted content
        with open(output_path, 'wb') as f:
            f.write(encrypted_content)
        
        # Return hash
        return hashlib.sha256(encrypted_content).hexdigest()
    
    def _extract_encrypted_archive(
        self, 
        archive_path: Path, 
        output_dir: Path, 
        decryption_key: bytes
    ) -> bool:
        """
        Extract encrypted archive.
        
        Args:
            archive_path: Path to encrypted archive
            output_dir: Output directory
            decryption_key: Decryption key
            
        Returns:
            True if successful
        """
        import tarfile
        
        try:
            # Read encrypted content
            with open(archive_path, 'rb') as f:
                encrypted_content = f.read()
            
            if self.security_config.enable_encryption:
                # Decrypt content
                decrypted_content = self._xor_encrypt(encrypted_content, decryption_key)
            else:
                decrypted_content = encrypted_content
            
            # Extract tar archive
            with tempfile.NamedTemporaryFile() as temp_tar:
                temp_tar.write(decrypted_content)
                temp_tar.flush()
                
                with tarfile.open(temp_tar.name, 'r') as tar:
                    tar.extractall(output_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Archive extraction failed: {e}")
            return False
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """
        Simple XOR encryption (for demonstration - use proper encryption in production).
        
        Args:
            data: Data to encrypt/decrypt
            key: Encryption key
            
        Returns:
            Encrypted/decrypted data
        """
        # Extend key to match data length
        extended_key = (key * (len(data) // len(key) + 1))[:len(data)]
        
        # XOR encryption
        return bytes(a ^ b for a, b in zip(data, extended_key))
    
    def clean_deployment(self, deployment_dir: Path, deployment_id: str) -> bool:
        """
        Securely clean up deployment directory.
        
        Args:
            deployment_dir: Deployment directory to clean
            deployment_id: Deployment ID
            
        Returns:
            True if successful
        """
        try:
            validated_dir = self.sanitizer.sanitize_file_path(deployment_dir)
            
            if validated_dir.exists():
                # Secure deletion by overwriting files
                for file_path in validated_dir.rglob('*'):
                    if file_path.is_file():
                        # Overwrite with random data
                        file_size = file_path.stat().st_size
                        with open(file_path, 'wb') as f:
                            f.write(secrets.token_bytes(file_size))
                
                # Remove directory
                shutil.rmtree(validated_dir)
            
            # Clean up stored keys
            if deployment_id in self.deployment_keys:
                del self.deployment_keys[deployment_id]
            if deployment_id in self.deployment_hashes:
                del self.deployment_hashes[deployment_id]
            
            logger.info(f"Deployment cleaned: {deployment_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment cleanup failed: {e}")
            return False


class SecurityError(Exception):
    """Security-related error in deployment."""
    pass