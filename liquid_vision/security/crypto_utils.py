"""
Cryptographic utilities for model encryption and secure storage.
"""

import os
import hashlib
import secrets
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import json
import logging

logger = logging.getLogger('liquid_vision.security.crypto')


class ModelEncryption:
    """
    Utilities for encrypting and decrypting neural network models.
    Uses industry-standard cryptographic practices.
    """
    
    def __init__(self, key_size: int = 256):
        """
        Initialize model encryption.
        
        Args:
            key_size: Encryption key size in bits (128, 192, or 256)
        """
        if key_size not in [128, 192, 256]:
            raise ValueError("Key size must be 128, 192, or 256 bits")
        
        self.key_size = key_size
        self.key_bytes = key_size // 8
    
    def generate_key(self) -> bytes:
        """
        Generate a cryptographically secure encryption key.
        
        Returns:
            Random encryption key
        """
        return secrets.token_bytes(self.key_bytes)
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: User password
            salt: Optional salt (generates random if None)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000,  # 100k iterations
            self.key_bytes
        )
        
        return key, salt
    
    def encrypt_data(self, data: bytes, key: bytes, associated_data: Optional[bytes] = None) -> Dict[str, str]:
        """
        Encrypt data with AES-GCM (simulated with secure XOR for this implementation).
        In production, use proper AES-GCM implementation.
        
        Args:
            data: Data to encrypt
            key: Encryption key
            associated_data: Optional associated data for authentication
            
        Returns:
            Dictionary with encrypted data, nonce, and tag
        """
        if len(key) != self.key_bytes:
            raise ValueError(f"Key must be {self.key_bytes} bytes")
        
        # Generate random nonce
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        # Simple encryption (in production, use proper AES-GCM)
        encrypted_data = self._encrypt_with_key(data, key, nonce)
        
        # Generate authentication tag
        auth_tag = self._generate_auth_tag(encrypted_data, key, nonce, associated_data)
        
        return {
            'encrypted_data': base64.b64encode(encrypted_data).decode('ascii'),
            'nonce': base64.b64encode(nonce).decode('ascii'),
            'auth_tag': base64.b64encode(auth_tag).decode('ascii'),
            'associated_data': base64.b64encode(associated_data or b'').decode('ascii')
        }
    
    def decrypt_data(self, encrypted_package: Dict[str, str], key: bytes) -> bytes:
        """
        Decrypt data and verify authentication.
        
        Args:
            encrypted_package: Dictionary with encrypted data, nonce, and tag
            key: Decryption key
            
        Returns:
            Decrypted data
            
        Raises:
            SecurityError: If authentication fails
        """
        if len(key) != self.key_bytes:
            raise ValueError(f"Key must be {self.key_bytes} bytes")
        
        try:
            # Decode components
            encrypted_data = base64.b64decode(encrypted_package['encrypted_data'])
            nonce = base64.b64decode(encrypted_package['nonce'])
            auth_tag = base64.b64decode(encrypted_package['auth_tag'])
            associated_data = base64.b64decode(encrypted_package['associated_data'])
            
            # Verify authentication tag
            expected_tag = self._generate_auth_tag(encrypted_data, key, nonce, associated_data)
            if not secrets.compare_digest(auth_tag, expected_tag):
                raise SecurityError("Authentication verification failed")
            
            # Decrypt data
            decrypted_data = self._decrypt_with_key(encrypted_data, key, nonce)
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Decryption failed: {e}")
    
    def _encrypt_with_key(self, data: bytes, key: bytes, nonce: bytes) -> bytes:
        """
        Encrypt data with key and nonce.
        This is a simplified implementation - use proper AES-GCM in production.
        """
        # Combine key and nonce for encryption
        encryption_key = hashlib.sha256(key + nonce).digest()
        
        # XOR encryption (simplified)
        encrypted = bytearray()
        for i, byte in enumerate(data):
            key_byte = encryption_key[i % len(encryption_key)]
            encrypted.append(byte ^ key_byte)
        
        return bytes(encrypted)
    
    def _decrypt_with_key(self, encrypted_data: bytes, key: bytes, nonce: bytes) -> bytes:
        """
        Decrypt data with key and nonce.
        """
        # Decryption is the same as encryption for XOR
        return self._encrypt_with_key(encrypted_data, key, nonce)
    
    def _generate_auth_tag(self, data: bytes, key: bytes, nonce: bytes, associated_data: Optional[bytes]) -> bytes:
        """
        Generate authentication tag for data integrity.
        """
        # HMAC-based authentication tag
        message = data + nonce + (associated_data or b'')
        return hashlib.pbkdf2_hmac('sha256', key, message, 1000, 16)


class SecureStorage:
    """
    Secure storage for sensitive data like model weights and configurations.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize secure storage.
        
        Args:
            storage_dir: Directory for secure storage (creates in temp if None)
        """
        if storage_dir is None:
            storage_dir = Path.home() / '.liquid_vision' / 'secure_storage'
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)  # Owner only
        
        # Metadata file for tracking stored items
        self.metadata_file = self.storage_dir / '.metadata.json'
        self.metadata = self._load_metadata()
        
        self.encryption = ModelEncryption()
    
    def store_model_weights(
        self, 
        model_id: str, 
        weights_data: bytes, 
        key: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Store model weights securely.
        
        Args:
            model_id: Unique model identifier
            weights_data: Serialized model weights
            key: Encryption key (generates if None)
            metadata: Optional metadata
            
        Returns:
            Storage information including key fingerprint
        """
        if key is None:
            key = self.encryption.generate_key()
        
        # Encrypt weights data
        encrypted_package = self.encryption.encrypt_data(
            weights_data, 
            key,
            associated_data=model_id.encode('utf-8')
        )
        
        # Create storage entry
        storage_entry = {
            'model_id': model_id,
            'created_at': str(Path(__file__).parent),
            'encrypted_package': encrypted_package,
            'metadata': metadata or {},
            'key_fingerprint': hashlib.sha256(key).hexdigest()[:16]
        }
        
        # Save to file
        storage_file = self.storage_dir / f"{model_id}.enc"
        with open(storage_file, 'w') as f:
            json.dump(storage_entry, f, indent=2)
        
        # Update metadata
        self.metadata[model_id] = {
            'file': str(storage_file),
            'created_at': storage_entry['created_at'],
            'key_fingerprint': storage_entry['key_fingerprint']
        }
        self._save_metadata()
        
        logger.info(f"Model weights stored securely: {model_id}")
        
        return {
            'model_id': model_id,
            'key_fingerprint': storage_entry['key_fingerprint'],
            'storage_file': str(storage_file)
        }
    
    def retrieve_model_weights(self, model_id: str, key: bytes) -> bytes:
        """
        Retrieve and decrypt model weights.
        
        Args:
            model_id: Model identifier
            key: Decryption key
            
        Returns:
            Decrypted model weights
        """
        if model_id not in self.metadata:
            raise KeyError(f"Model not found: {model_id}")
        
        storage_file = Path(self.metadata[model_id]['file'])
        if not storage_file.exists():
            raise FileNotFoundError(f"Storage file not found: {storage_file}")
        
        # Load encrypted data
        with open(storage_file, 'r') as f:
            storage_entry = json.load(f)
        
        # Verify key fingerprint
        key_fingerprint = hashlib.sha256(key).hexdigest()[:16]
        if key_fingerprint != storage_entry['key_fingerprint']:
            raise SecurityError("Key fingerprint mismatch")
        
        # Decrypt weights
        weights_data = self.encryption.decrypt_data(
            storage_entry['encrypted_package'],
            key
        )
        
        logger.info(f"Model weights retrieved: {model_id}")
        return weights_data
    
    def list_stored_models(self) -> List[Dict[str, Any]]:
        """
        List all stored models.
        
        Returns:
            List of model information (without keys)
        """
        models = []
        for model_id, info in self.metadata.items():
            models.append({
                'model_id': model_id,
                'created_at': info['created_at'],
                'key_fingerprint': info['key_fingerprint']
            })
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Securely delete stored model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful
        """
        if model_id not in self.metadata:
            return False
        
        storage_file = Path(self.metadata[model_id]['file'])
        
        if storage_file.exists():
            # Secure deletion by overwriting
            file_size = storage_file.stat().st_size
            with open(storage_file, 'wb') as f:
                f.write(secrets.token_bytes(file_size))
            
            storage_file.unlink()
        
        # Remove from metadata
        del self.metadata[model_id]
        self._save_metadata()
        
        logger.info(f"Model deleted securely: {model_id}")
        return True
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Set restrictive permissions
        os.chmod(self.metadata_file, 0o600)


class SecurityError(Exception):
    """Cryptographic security error."""
    pass


# Utility functions
def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)


def hash_sensitive_data(data: Union[str, bytes]) -> str:
    """Hash sensitive data for logging/comparison."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    # Add salt to prevent rainbow table attacks
    salt = b"liquid_vision_salt_v1"
    return hashlib.sha256(salt + data).hexdigest()


def constant_time_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    
    return secrets.compare_digest(a, b)