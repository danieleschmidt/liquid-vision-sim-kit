#!/usr/bin/env python3
"""
Comprehensive tests for security modules.
Tests input sanitization, secure deployment, and cryptographic utilities.
"""

import sys
import os
import pytest
import tempfile
import secrets
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liquid_vision.security.input_sanitizer import (
    InputSanitizer, SanitizationError, sanitize_user_input
)
from liquid_vision.security.secure_deployment import (
    SecureDeployer, SecurityConfig, SecurityError
)
from liquid_vision.security.crypto_utils import (
    ModelEncryption, SecureStorage, SecurityError as CryptoSecurityError
)


class TestInputSanitizer:
    """Test InputSanitizer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sanitizer = InputSanitizer(strict_mode=True)
    
    def test_sanitize_safe_string(self):
        """Test sanitizing safe strings."""
        safe_string = "Hello World 123"
        result = self.sanitizer.sanitize_string(safe_string)
        assert result == safe_string
    
    def test_sanitize_dangerous_script_tags(self):
        """Test blocking script tags."""
        dangerous_string = "<script>alert('xss')</script>"
        
        with pytest.raises(SanitizationError, match="Dangerous pattern detected"):
            self.sanitizer.sanitize_string(dangerous_string)
    
    def test_sanitize_javascript_urls(self):
        """Test blocking JavaScript URLs."""
        dangerous_string = "javascript:alert('xss')"
        
        with pytest.raises(SanitizationError, match="Dangerous pattern detected"):
            self.sanitizer.sanitize_string(dangerous_string)
    
    def test_sanitize_sql_injection(self):
        """Test blocking SQL injection patterns."""
        dangerous_strings = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords"
        ]
        
        for dangerous_string in dangerous_strings:
            with pytest.raises(SanitizationError, match="SQL injection pattern detected"):
                self.sanitizer.sanitize_string(dangerous_string)
    
    def test_sanitize_string_length_limit(self):
        """Test string length limits."""
        long_string = "A" * 11000  # Exceeds default limit of 10000
        
        with pytest.raises(SanitizationError, match="String too long"):
            self.sanitizer.sanitize_string(long_string)
    
    def test_sanitize_html_escaping(self):
        """Test HTML escaping."""
        html_string = "<div>Hello & goodbye</div>"
        result = self.sanitizer.sanitize_string(html_string, allow_html=False)
        assert "&lt;div&gt;" in result
        assert "&amp;" in result
    
    def test_sanitize_file_path_safe(self):
        """Test sanitizing safe file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            
            result = self.sanitizer.sanitize_file_path(test_file)
            assert result == test_file.resolve()
    
    def test_sanitize_file_path_traversal(self):
        """Test blocking path traversal attacks."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
        ]
        
        for dangerous_path in dangerous_paths:
            with pytest.raises(SanitizationError):
                self.sanitizer.sanitize_file_path(dangerous_path)
    
    def test_sanitize_file_path_null_byte(self):
        """Test blocking null bytes in paths."""
        dangerous_path = "test\x00.txt"
        
        with pytest.raises(SanitizationError, match="Null byte in path"):
            self.sanitizer.sanitize_file_path(dangerous_path)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        dangerous_filename = "<script>.txt"
        result = self.sanitizer.sanitize_filename(dangerous_filename)
        assert "<" not in result
        assert ">" not in result
        assert result.endswith(".txt")
    
    def test_sanitize_config_dict(self):
        """Test configuration dictionary sanitization."""
        config = {
            "safe_key": "safe_value",
            "dangerous_value": "<script>alert('xss')</script>",
            "nested": {
                "safe": "value",
                "dangerous": "javascript:alert(1)"
            }
        }
        
        with pytest.raises(SanitizationError):
            self.sanitizer.sanitize_config_dict(config)
    
    def test_check_content_safety_text(self):
        """Test content safety checking for text files."""
        safe_content = b"This is safe text content."
        assert self.sanitizer.check_content_safety(safe_content) == True
        
        # Test various dangerous signatures
        dangerous_contents = [
            b'\x7fELF',  # Linux executable
            b'MZ',       # Windows executable
            b'<script>alert("xss")</script>'.encode(),
        ]
        
        for dangerous_content in dangerous_contents:
            assert self.sanitizer.check_content_safety(dangerous_content) == False
    
    def test_generate_safe_hash(self):
        """Test hash generation."""
        data = "test data"
        hash_result = self.sanitizer.generate_safe_hash(data)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 hex length
        
        # Same input should produce same hash
        hash_result2 = self.sanitizer.generate_safe_hash(data)
        assert hash_result == hash_result2
    
    def test_create_sandbox_filename(self):
        """Test sandbox filename creation."""
        original_filename = "test file.txt"
        safe_filename = self.sanitizer.create_sandbox_filename(original_filename)
        
        assert safe_filename != original_filename
        assert safe_filename.endswith(".txt")
        assert len(safe_filename) <= 200
        assert "_" in safe_filename  # Should have hash prefix


class TestSecureDeployer:
    """Test SecureDeployer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.security_config = SecurityConfig(
            enable_encryption=True,
            require_signature=True
        )
        self.deployer = SecureDeployer(self.security_config)
    
    def test_security_config_validation(self):
        """Test SecurityConfig validation."""
        # Valid config should not raise
        config = SecurityConfig()
        config.validate()
        
        # Invalid config should raise
        with pytest.raises(ValueError):
            invalid_config = SecurityConfig(max_memory_mb=0)
            invalid_config.validate()
    
    def test_create_secure_package(self):
        """Test secure package creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            test_file1 = temp_path / "test1.txt"
            test_file1.write_text("Test content 1")
            
            test_file2 = temp_path / "test2.py"
            test_file2.write_text("print('Hello World')")
            
            model_files = {
                "model.txt": test_file1,
                "script.py": test_file2
            }
            
            metadata = {
                "model_name": "test_model",
                "version": "1.0"
            }
            
            output_path = temp_path / "secure_package.tar.enc"
            
            # Create secure package
            result = self.deployer.create_secure_package(
                model_files, metadata, output_path
            )
            
            assert "deployment_id" in result
            assert "package_hash" in result
            assert "key_fingerprint" in result
            assert output_path.exists()
    
    def test_verify_package_integrity(self):
        """Test package integrity verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create and package files
            test_file = temp_path / "test.txt"
            test_file.write_text("Test content")
            
            model_files = {"test.txt": test_file}
            metadata = {"test": "data"}
            output_path = temp_path / "package.tar.enc"
            
            # Create package
            result = self.deployer.create_secure_package(
                model_files, metadata, output_path
            )
            
            # Verify with correct hash and key
            deployment_id = result["deployment_id"]
            package_hash = result["package_hash"]
            decryption_key = self.deployer.deployment_keys[deployment_id]
            
            is_valid = self.deployer.verify_package_integrity(
                output_path, package_hash, decryption_key
            )
            assert is_valid == True
            
            # Verify with wrong hash should fail
            wrong_hash = "0" * 64
            is_valid_wrong = self.deployer.verify_package_integrity(
                output_path, wrong_hash, decryption_key
            )
            assert is_valid_wrong == False


class TestModelEncryption:
    """Test ModelEncryption functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.encryption = ModelEncryption(key_size=256)
    
    def test_generate_key(self):
        """Test encryption key generation."""
        key = self.encryption.generate_key()
        
        assert isinstance(key, bytes)
        assert len(key) == 32  # 256 bits = 32 bytes
    
    def test_derive_key_from_password(self):
        """Test key derivation from password."""
        password = "test_password_123"
        key, salt = self.encryption.derive_key_from_password(password)
        
        assert isinstance(key, bytes)
        assert isinstance(salt, bytes)
        assert len(key) == 32  # 256 bits
        assert len(salt) == 32
        
        # Same password and salt should produce same key
        key2, _ = self.encryption.derive_key_from_password(password, salt)
        assert key == key2
    
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption."""
        test_data = b"This is secret test data for encryption!"
        key = self.encryption.generate_key()
        
        # Encrypt data
        encrypted_package = self.encryption.encrypt_data(test_data, key)
        
        assert "encrypted_data" in encrypted_package
        assert "nonce" in encrypted_package
        assert "auth_tag" in encrypted_package
        
        # Decrypt data
        decrypted_data = self.encryption.decrypt_data(encrypted_package, key)
        
        assert decrypted_data == test_data
    
    def test_decrypt_with_wrong_key(self):
        """Test decryption with wrong key fails."""
        test_data = b"Secret data"
        correct_key = self.encryption.generate_key()
        wrong_key = self.encryption.generate_key()
        
        # Encrypt with correct key
        encrypted_package = self.encryption.encrypt_data(test_data, correct_key)
        
        # Decrypt with wrong key should fail
        with pytest.raises(CryptoSecurityError):
            self.encryption.decrypt_data(encrypted_package, wrong_key)
    
    def test_encrypt_with_associated_data(self):
        """Test encryption with associated data for authentication."""
        test_data = b"Secret data"
        key = self.encryption.generate_key()
        associated_data = b"associated_metadata"
        
        # Encrypt with associated data
        encrypted_package = self.encryption.encrypt_data(
            test_data, key, associated_data
        )
        
        # Decrypt should succeed
        decrypted_data = self.encryption.decrypt_data(encrypted_package, key)
        assert decrypted_data == test_data
        
        # Modifying associated data should cause decryption to fail
        encrypted_package["associated_data"] = "modified_data"
        
        with pytest.raises(CryptoSecurityError):
            self.encryption.decrypt_data(encrypted_package, key)


class TestSecureStorage:
    """Test SecureStorage functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = SecureStorage(storage_dir=Path(self.temp_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve_model_weights(self):
        """Test storing and retrieving model weights."""
        model_id = "test_model_v1"
        test_weights = b"fake_model_weights_data" * 100  # Make it substantial
        
        # Store weights
        result = self.storage.store_model_weights(model_id, test_weights)
        
        assert result["model_id"] == model_id
        assert "key_fingerprint" in result
        assert "storage_file" in result
        
        # The key should be available in metadata
        assert model_id in self.storage.metadata
        
        # Retrieve weights (we need the key from the store operation)
        # In real usage, the key would be managed separately
        encryption_key = list(self.storage.encryption._encrypt_with_key.__defaults__)[0]
        # For this test, we'll generate a new key since we can't access the original
        key = self.storage.encryption.generate_key()
        
        # Store again with known key
        result2 = self.storage.store_model_weights(model_id + "_2", test_weights, key)
        
        # Retrieve with known key
        retrieved_weights = self.storage.retrieve_model_weights(model_id + "_2", key)
        assert retrieved_weights == test_weights
    
    def test_list_stored_models(self):
        """Test listing stored models."""
        # Initially empty
        models = self.storage.list_stored_models()
        assert len(models) == 0
        
        # Store some models
        model_weights = b"test_weights"
        self.storage.store_model_weights("model1", model_weights)
        self.storage.store_model_weights("model2", model_weights)
        
        # List should return both
        models = self.storage.list_stored_models()
        assert len(models) == 2
        
        model_ids = [m["model_id"] for m in models]
        assert "model1" in model_ids
        assert "model2" in model_ids
    
    def test_delete_model(self):
        """Test secure model deletion."""
        model_id = "test_delete_model"
        test_weights = b"test_weights_for_deletion"
        
        # Store model
        self.storage.store_model_weights(model_id, test_weights)
        
        # Verify it exists
        models = self.storage.list_stored_models()
        assert any(m["model_id"] == model_id for m in models)
        
        # Delete model
        success = self.storage.delete_model(model_id)
        assert success == True
        
        # Verify it's gone
        models = self.storage.list_stored_models()
        assert not any(m["model_id"] == model_id for m in models)
        
        # Deleting non-existent model should return False
        success = self.storage.delete_model("non_existent_model")
        assert success == False


class TestSanitizeUserInputFunction:
    """Test sanitize_user_input convenience function."""
    
    def test_sanitize_string_input(self):
        """Test sanitizing string input."""
        safe_string = "Hello World"
        result = sanitize_user_input(safe_string, "string")
        assert result == safe_string
        
        dangerous_string = "<script>alert('xss')</script>"
        with pytest.raises(SanitizationError):
            sanitize_user_input(dangerous_string, "string")
    
    def test_sanitize_filename_input(self):
        """Test sanitizing filename input."""
        dangerous_filename = "../../../etc/passwd"
        result = sanitize_user_input(dangerous_filename, "filename")
        
        assert "../" not in result
        assert result != dangerous_filename
    
    def test_sanitize_config_input(self):
        """Test sanitizing config input."""
        safe_config = {"key": "value", "number": 42}
        result = sanitize_user_input(safe_config, "config")
        assert result == safe_config
        
        dangerous_config = {"key": "<script>alert('xss')</script>"}
        with pytest.raises(SanitizationError):
            sanitize_user_input(dangerous_config, "config")
    
    def test_unknown_input_type(self):
        """Test handling unknown input types."""
        result = sanitize_user_input("test", "unknown_type")
        assert result == "test"  # Should return unchanged with warning


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])