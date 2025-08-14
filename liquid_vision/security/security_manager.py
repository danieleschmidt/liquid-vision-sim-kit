"""
Comprehensive security management system for Generation 2.
Provides input sanitization, access control, and security monitoring.
"""

import hashlib
import hmac
import time
import re
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event for logging and monitoring."""
    event_type: str
    severity: SecurityLevel
    message: str
    source: str
    timestamp: float
    metadata: Dict[str, Any]


class InputSanitizer:
    """Sanitizes and validates inputs to prevent security vulnerabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',  # JavaScript URLs
            r'data:',  # Data URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Eval calls
            r'exec\s*\(',  # Exec calls
            r'import\s+',  # Import statements (when not expected)
            r'__.*__',  # Dunder methods
            r'\.\./',  # Directory traversal
            r'[<>"\']',  # Basic XSS characters
        ]
        
    def sanitize_string(self, value: str, max_length: int = 1000, 
                       allow_html: bool = False) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be string")
            
        # Length check
        if len(value) > max_length:
            self.logger.warning(f"Input truncated from {len(value)} to {max_length} characters")
            value = value[:max_length]
            
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Check for dangerous patterns
        if not allow_html:
            for pattern in self.dangerous_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    self.logger.warning(f"Dangerous pattern detected: {pattern}")
                    # Remove the dangerous content
                    value = re.sub(pattern, '', value, flags=re.IGNORECASE)
                    
        # Basic HTML escaping if not allowing HTML
        if not allow_html:
            value = (value.replace('&', '&amp;')
                         .replace('<', '&lt;')
                         .replace('>', '&gt;')
                         .replace('"', '&quot;')
                         .replace("'", '&#x27;'))
            
        return value
        
    def sanitize_numeric(self, value: Union[int, float, str], 
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> float:
        """Sanitize numeric input."""
        try:
            if isinstance(value, str):
                # Remove any non-numeric characters except . and -
                cleaned = re.sub(r'[^\d\.-]', '', value)
                numeric_value = float(cleaned) if cleaned else 0.0
            else:
                numeric_value = float(value)
                
            # Apply bounds
            if min_val is not None:
                numeric_value = max(numeric_value, min_val)
            if max_val is not None:
                numeric_value = min(numeric_value, max_val)
                
            # Check for special values
            if not (-1e10 < numeric_value < 1e10):
                self.logger.warning(f"Numeric value {numeric_value} out of safe range")
                return 0.0
                
            return numeric_value
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid numeric input: {value}, error: {e}")
            return 0.0
            
    def sanitize_list(self, value: List[Any], max_length: int = 1000,
                     item_sanitizer: Optional[Callable] = None) -> List[Any]:
        """Sanitize list input."""
        if not isinstance(value, (list, tuple)):
            raise ValueError("Input must be list or tuple")
            
        # Length check
        if len(value) > max_length:
            self.logger.warning(f"List truncated from {len(value)} to {max_length} items")
            value = value[:max_length]
            
        # Sanitize individual items if sanitizer provided
        if item_sanitizer:
            value = [item_sanitizer(item) for item in value]
            
        return list(value)
        
    def sanitize_dict(self, value: Dict[str, Any], max_keys: int = 100,
                     key_sanitizer: Optional[Callable] = None,
                     value_sanitizer: Optional[Callable] = None) -> Dict[str, Any]:
        """Sanitize dictionary input."""
        if not isinstance(value, dict):
            raise ValueError("Input must be dictionary")
            
        # Size check
        if len(value) > max_keys:
            self.logger.warning(f"Dictionary truncated from {len(value)} to {max_keys} keys")
            items = list(value.items())[:max_keys]
            value = dict(items)
            
        # Sanitize keys and values
        sanitized = {}
        for k, v in value.items():
            clean_key = key_sanitizer(k) if key_sanitizer else str(k)
            clean_value = value_sanitizer(v) if value_sanitizer else v
            sanitized[clean_key] = clean_value
            
        return sanitized


class AccessController:
    """Controls access to system resources and operations."""
    
    def __init__(self):
        self.permissions: Dict[str, Dict[str, bool]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
    def add_permission(self, user: str, resource: str, allowed: bool = True):
        """Add permission for user to access resource."""
        if user not in self.permissions:
            self.permissions[user] = {}
        self.permissions[user][resource] = allowed
        
    def check_permission(self, user: str, resource: str) -> bool:
        """Check if user has permission to access resource."""
        if user not in self.permissions:
            return False
        return self.permissions[user].get(resource, False)
        
    def set_rate_limit(self, user: str, max_requests: int, time_window: int):
        """Set rate limit for user."""
        self.rate_limits[user] = {
            "max_requests": max_requests,
            "time_window": time_window,
            "requests": [],
        }
        
    def check_rate_limit(self, user: str) -> bool:
        """Check if user is within rate limits."""
        if user not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[user]
        current_time = time.time()
        
        # Clean old requests
        limit_info["requests"] = [
            req_time for req_time in limit_info["requests"]
            if current_time - req_time < limit_info["time_window"]
        ]
        
        # Check if under limit
        if len(limit_info["requests"]) >= limit_info["max_requests"]:
            return False
            
        # Record this request
        limit_info["requests"].append(current_time)
        return True


class SecurityAuditor:
    """Audits and logs security events."""
    
    def __init__(self, max_events: int = 10000):
        self.security_events: List[SecurityEvent] = []
        self.max_events = max_events
        self.logger = logging.getLogger(__name__)
        
    def log_security_event(self, event_type: str, severity: SecurityLevel,
                          message: str, source: str = "unknown",
                          metadata: Dict[str, Any] = None):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Trim old events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
            
        # Log to standard logger based on severity
        if severity == SecurityLevel.CRITICAL:
            self.logger.critical(f"SECURITY: {event_type} - {message}")
        elif severity == SecurityLevel.HIGH:
            self.logger.error(f"SECURITY: {event_type} - {message}")
        elif severity == SecurityLevel.MEDIUM:
            self.logger.warning(f"SECURITY: {event_type} - {message}")
        else:
            self.logger.info(f"SECURITY: {event_type} - {message}")
            
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security events."""
        if not self.security_events:
            return {
                "total_events": 0,
                "critical_events": 0,
                "high_events": 0,
                "security_score": 100.0
            }
            
        total_events = len(self.security_events)
        critical_events = sum(1 for e in self.security_events if e.severity == SecurityLevel.CRITICAL)
        high_events = sum(1 for e in self.security_events if e.severity == SecurityLevel.HIGH)
        
        # Calculate security score (0-100)
        penalty = critical_events * 10 + high_events * 5
        security_score = max(0, 100 - penalty)
        
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]
        
        return {
            "total_events": total_events,
            "critical_events": critical_events,
            "high_events": high_events,
            "recent_events_1h": len(recent_events),
            "security_score": security_score,
            "event_types": list(set(e.event_type for e in self.security_events)),
        }


class SecureHashManager:
    """Manages secure hashing and verification."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or self._generate_secret_key()
        self.logger = logging.getLogger(__name__)
        
    def _generate_secret_key(self) -> str:
        """Generate a random secret key."""
        import os
        return hashlib.sha256(os.urandom(32)).hexdigest()
        
    def hash_data(self, data: str) -> str:
        """Create secure hash of data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
        
    def create_hmac(self, data: str) -> str:
        """Create HMAC signature for data."""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    def verify_hmac(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.create_hmac(data)
        return hmac.compare_digest(expected, signature)


class SecurityManager:
    """Main security management system."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.access_controller = AccessController()
        self.auditor = SecurityAuditor()
        self.hash_manager = SecureHashManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize default security settings
        self._initialize_defaults()
        
    def _initialize_defaults(self):
        """Initialize default security settings."""
        # Add default permissions for common operations
        self.access_controller.add_permission("anonymous", "read_status", True)
        self.access_controller.add_permission("anonymous", "basic_inference", True)
        
        # Set default rate limits
        self.access_controller.set_rate_limit("anonymous", 100, 60)  # 100 requests per minute
        
        self.auditor.log_security_event(
            "initialization",
            SecurityLevel.LOW,
            "Security manager initialized",
            "system"
        )
        
    def secure_function_call(self, func: Callable, user: str = "anonymous",
                           resource: str = "default", *args, **kwargs):
        """Execute function with security checks."""
        # Check permissions
        if not self.access_controller.check_permission(user, resource):
            self.auditor.log_security_event(
                "access_denied",
                SecurityLevel.MEDIUM,
                f"User {user} denied access to {resource}",
                "access_controller"
            )
            raise PermissionError(f"Access denied to {resource}")
            
        # Check rate limits
        if not self.access_controller.check_rate_limit(user):
            self.auditor.log_security_event(
                "rate_limit_exceeded",
                SecurityLevel.MEDIUM,
                f"Rate limit exceeded for user {user}",
                "access_controller"
            )
            raise ValueError("Rate limit exceeded")
            
        # Sanitize inputs
        try:
            sanitized_args = []
            for arg in args:
                if isinstance(arg, str):
                    sanitized_args.append(self.sanitizer.sanitize_string(arg))
                elif isinstance(arg, (int, float)):
                    sanitized_args.append(self.sanitizer.sanitize_numeric(arg))
                elif isinstance(arg, list):
                    sanitized_args.append(self.sanitizer.sanitize_list(arg))
                elif isinstance(arg, dict):
                    sanitized_args.append(self.sanitizer.sanitize_dict(arg))
                else:
                    sanitized_args.append(arg)
                    
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                clean_key = self.sanitizer.sanitize_string(str(key), max_length=100)
                if isinstance(value, str):
                    clean_value = self.sanitizer.sanitize_string(value)
                elif isinstance(value, (int, float)):
                    clean_value = self.sanitizer.sanitize_numeric(value)
                elif isinstance(value, list):
                    clean_value = self.sanitizer.sanitize_list(value)
                elif isinstance(value, dict):
                    clean_value = self.sanitizer.sanitize_dict(value)
                else:
                    clean_value = value
                sanitized_kwargs[clean_key] = clean_value
                
        except Exception as e:
            self.auditor.log_security_event(
                "input_sanitization_failed",
                SecurityLevel.HIGH,
                f"Input sanitization failed: {e}",
                "sanitizer"
            )
            raise ValueError(f"Input sanitization failed: {e}")
            
        # Execute function
        try:
            result = func(*sanitized_args, **sanitized_kwargs)
            
            self.auditor.log_security_event(
                "secure_function_executed",
                SecurityLevel.LOW,
                f"Function {func.__name__} executed successfully",
                "security_manager"
            )
            
            return result
            
        except Exception as e:
            self.auditor.log_security_event(
                "function_execution_failed",
                SecurityLevel.MEDIUM,
                f"Function {func.__name__} failed: {e}",
                "security_manager"
            )
            raise
            
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "security_summary": self.auditor.get_security_summary(),
            "permissions_count": len(self.access_controller.permissions),
            "rate_limits_count": len(self.access_controller.rate_limits),
            "security_level": "high",
            "features_enabled": [
                "input_sanitization",
                "access_control", 
                "rate_limiting",
                "security_auditing",
                "secure_hashing"
            ],
            "timestamp": time.time()
        }


# Global security manager instance
security_manager = SecurityManager()


def secure_operation(resource: str = "default", user: str = "anonymous"):
    """Decorator for securing function operations."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return security_manager.secure_function_call(
                func, user, resource, *args, **kwargs
            )
        return wrapper
    return decorator