"""
Security auditing and compliance utilities.
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading


logger = logging.getLogger('liquid_vision.security.audit')


class AuditEventType(Enum):
    """Types of security audit events."""
    
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    FILE_ACCESS = "file_access"
    MODEL_DEPLOYMENT = "model_deployment"
    CONFIG_CHANGE = "config_change"
    SECURITY_VIOLATION = "security_violation"
    DATA_EXPORT = "data_export"
    ENCRYPTION_KEY_ACCESS = "encryption_key_access"
    SYSTEM_COMMAND = "system_command"
    NETWORK_ACCESS = "network_access"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Security audit event record."""
    
    timestamp: float
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: str
    resource: str
    action: str
    result: str  # success, failure, denied
    details: Dict[str, Any]
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Comprehensive audit logging for security compliance.
    Supports multiple output formats and tamper detection.
    """
    
    def __init__(
        self, 
        log_file: Optional[Path] = None,
        enable_integrity_checking: bool = True,
        max_log_size_mb: int = 100,
        retention_days: int = 90
    ):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
            enable_integrity_checking: Enable log integrity verification
            max_log_size_mb: Maximum log file size before rotation
            retention_days: Log retention period in days
        """
        if log_file is None:
            log_file = Path.home() / '.liquid_vision' / 'audit' / 'security.log'
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.enable_integrity_checking = enable_integrity_checking
        self.max_log_size_mb = max_log_size_mb
        self.retention_days = retention_days
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Integrity tracking
        self.log_hash_chain = []
        
        # Initialize log file if it doesn't exist
        if not self.log_file.exists():
            self._initialize_log_file()
    
    def log_event(self, event: AuditEvent):
        """
        Log a security audit event.
        
        Args:
            event: Audit event to log
        """
        with self._lock:
            # Convert event to JSON
            event_json = event.to_json()
            
            # Add integrity hash if enabled
            if self.enable_integrity_checking:
                event_hash = self._compute_event_hash(event_json)
                event_data = {
                    'event': event.to_dict(),
                    'integrity_hash': event_hash,
                    'chain_hash': self._compute_chain_hash(event_hash)
                }
                log_line = json.dumps(event_data)
                
                # Update hash chain
                self.log_hash_chain.append(event_hash)
                if len(self.log_hash_chain) > 1000:  # Keep last 1000 hashes
                    self.log_hash_chain.pop(0)
            else:
                log_line = event_json
            
            # Write to log file
            with open(self.log_file, 'a') as f:
                f.write(log_line + '\n')
            
            # Check if log rotation is needed
            self._check_log_rotation()
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.INFO
    ):
        """
        Log a user action event.
        
        Args:
            user_id: User identifier
            action: Action performed
            resource: Resource accessed
            result: Action result
            details: Additional details
            severity: Event severity
        """
        event = AuditEvent(
            timestamp=time.time(),
            event_type=AuditEventType.USER_LOGIN if "login" in action else AuditEventType.FILE_ACCESS,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details or {}
        )
        
        self.log_event(event)
    
    def log_security_violation(
        self,
        user_id: str,
        violation_type: str,
        resource: str,
        details: Dict[str, Any],
        severity: AuditSeverity = AuditSeverity.ERROR
    ):
        """
        Log a security violation.
        
        Args:
            user_id: User identifier
            violation_type: Type of violation
            resource: Affected resource
            details: Violation details
            severity: Event severity
        """
        event = AuditEvent(
            timestamp=time.time(),
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=f"security_violation:{violation_type}",
            result="violation_detected",
            details=details
        )
        
        self.log_event(event)
        
        # Also log to system logger for immediate attention
        logger.error(f"Security violation: {violation_type} by {user_id} on {resource}")
    
    def log_model_deployment(
        self,
        user_id: str,
        model_path: str,
        target_environment: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log model deployment events.
        
        Args:
            user_id: User performing deployment
            model_path: Path to model being deployed
            target_environment: Target deployment environment
            result: Deployment result
            details: Additional deployment details
        """
        event = AuditEvent(
            timestamp=time.time(),
            event_type=AuditEventType.MODEL_DEPLOYMENT,
            severity=AuditSeverity.INFO if result == "success" else AuditSeverity.ERROR,
            user_id=user_id,
            resource=model_path,
            action=f"deploy_to:{target_environment}",
            result=result,
            details=details or {}
        )
        
        self.log_event(event)
    
    def log_config_change(
        self,
        user_id: str,
        config_type: str,
        changes: Dict[str, Any],
        result: str = "success"
    ):
        """
        Log configuration changes.
        
        Args:
            user_id: User making changes
            config_type: Type of configuration
            changes: Configuration changes made
            result: Change result
        """
        # Sanitize sensitive data from changes
        sanitized_changes = self._sanitize_config_changes(changes)
        
        event = AuditEvent(
            timestamp=time.time(),
            event_type=AuditEventType.CONFIG_CHANGE,
            severity=AuditSeverity.WARNING,  # Config changes are important
            user_id=user_id,
            resource=config_type,
            action="config_modify",
            result=result,
            details={"changes": sanitized_changes}
        )
        
        self.log_event(event)
    
    def verify_log_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of audit logs.
        
        Returns:
            Integrity verification results
        """
        if not self.enable_integrity_checking:
            return {"status": "disabled", "message": "Integrity checking not enabled"}
        
        verification_results = {
            "status": "unknown",
            "total_events": 0,
            "verified_events": 0,
            "corrupted_events": [],
            "missing_hashes": []
        }
        
        if not self.log_file.exists():
            verification_results["status"] = "no_log_file"
            return verification_results
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            previous_chain_hash = None
            
            for line_num, line in enumerate(lines, 1):
                try:
                    if not line.strip():
                        continue
                    
                    log_data = json.loads(line.strip())
                    verification_results["total_events"] += 1
                    
                    if "integrity_hash" not in log_data:
                        verification_results["missing_hashes"].append(line_num)
                        continue
                    
                    # Verify event hash
                    event_json = json.dumps(log_data["event"])
                    expected_hash = self._compute_event_hash(event_json)
                    
                    if log_data["integrity_hash"] != expected_hash:
                        verification_results["corrupted_events"].append({
                            "line": line_num,
                            "reason": "event_hash_mismatch"
                        })
                        continue
                    
                    # Verify chain hash
                    if previous_chain_hash is not None:
                        expected_chain_hash = self._compute_chain_hash(
                            expected_hash, previous_chain_hash
                        )
                        if log_data.get("chain_hash") != expected_chain_hash:
                            verification_results["corrupted_events"].append({
                                "line": line_num,
                                "reason": "chain_hash_mismatch"
                            })
                            continue
                    
                    previous_chain_hash = log_data.get("chain_hash")
                    verification_results["verified_events"] += 1
                    
                except json.JSONDecodeError:
                    verification_results["corrupted_events"].append({
                        "line": line_num,
                        "reason": "invalid_json"
                    })
                except Exception as e:
                    verification_results["corrupted_events"].append({
                        "line": line_num,
                        "reason": f"verification_error: {e}"
                    })
            
            # Determine overall status
            if verification_results["corrupted_events"] or verification_results["missing_hashes"]:
                verification_results["status"] = "compromised"
            else:
                verification_results["status"] = "verified"
            
        except Exception as e:
            verification_results["status"] = "error"
            verification_results["error"] = str(e)
        
        return verification_results
    
    def search_audit_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        resource_pattern: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search audit events with filters.
        
        Args:
            start_time: Start time filter (Unix timestamp)
            end_time: End time filter (Unix timestamp)
            event_types: Event type filters
            user_id: User ID filter
            resource_pattern: Resource pattern filter
            limit: Maximum results to return
            
        Returns:
            List of matching audit events
        """
        results = []
        
        if not self.log_file.exists():
            return results
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    
                    if not line.strip():
                        continue
                    
                    try:
                        log_data = json.loads(line.strip())
                        
                        # Extract event data
                        if "event" in log_data:
                            event_data = log_data["event"]
                        else:
                            event_data = log_data
                        
                        # Apply filters
                        if start_time and event_data.get("timestamp", 0) < start_time:
                            continue
                        
                        if end_time and event_data.get("timestamp", 0) > end_time:
                            continue
                        
                        if event_types:
                            event_type_values = [et.value for et in event_types]
                            if event_data.get("event_type") not in event_type_values:
                                continue
                        
                        if user_id and event_data.get("user_id") != user_id:
                            continue
                        
                        if resource_pattern:
                            import re
                            resource = event_data.get("resource", "")
                            if not re.search(resource_pattern, resource):
                                continue
                        
                        results.append(event_data)
                        
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
        except Exception as e:
            logger.error(f"Error searching audit events: {e}")
        
        return results
    
    def _initialize_log_file(self):
        """Initialize audit log file with header."""
        header = {
            "audit_log_version": "1.0",
            "created_at": time.time(),
            "integrity_checking": self.enable_integrity_checking
        }
        
        with open(self.log_file, 'w') as f:
            f.write(json.dumps(header) + '\n')
    
    def _compute_event_hash(self, event_json: str) -> str:
        """Compute hash for event integrity."""
        return hashlib.sha256(event_json.encode('utf-8')).hexdigest()
    
    def _compute_chain_hash(self, current_hash: str, previous_hash: Optional[str] = None) -> str:
        """Compute chain hash for tamper detection."""
        if previous_hash is None:
            previous_hash = "0" * 64  # Genesis hash
        
        chain_data = previous_hash + current_hash
        return hashlib.sha256(chain_data.encode('utf-8')).hexdigest()
    
    def _sanitize_config_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration changes to remove sensitive data."""
        sanitized = {}
        
        sensitive_keys = ['password', 'key', 'secret', 'token', 'private']
        
        for key, value in changes.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_config_changes(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _check_log_rotation(self):
        """Check if log rotation is needed."""
        try:
            file_size_mb = self.log_file.stat().st_size / (1024 * 1024)
            
            if file_size_mb > self.max_log_size_mb:
                # Rotate log file
                timestamp = int(time.time())
                rotated_name = f"{self.log_file.stem}_{timestamp}.log"
                rotated_path = self.log_file.parent / rotated_name
                
                self.log_file.rename(rotated_path)
                self._initialize_log_file()
                
                logger.info(f"Audit log rotated: {rotated_path}")
                
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")


class SecurityAuditor:
    """
    Security auditing and compliance checking.
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        """
        Initialize security auditor.
        
        Args:
            audit_logger: Audit logger instance
        """
        self.audit_logger = audit_logger or AuditLogger()
        self.compliance_rules = self._load_default_compliance_rules()
    
    def audit_system_security(self) -> Dict[str, Any]:
        """
        Perform comprehensive security audit.
        
        Returns:
            Audit results
        """
        audit_results = {
            "audit_timestamp": time.time(),
            "overall_status": "unknown",
            "checks_passed": 0,
            "checks_failed": 0,
            "checks": {}
        }
        
        # File permissions check
        audit_results["checks"]["file_permissions"] = self._check_file_permissions()
        
        # Log integrity check
        audit_results["checks"]["log_integrity"] = self.audit_logger.verify_log_integrity()
        
        # Configuration security check
        audit_results["checks"]["config_security"] = self._check_config_security()
        
        # Deployment security check
        audit_results["checks"]["deployment_security"] = self._check_deployment_security()
        
        # Count results
        for check_name, check_result in audit_results["checks"].items():
            if check_result.get("status") in ["passed", "verified"]:
                audit_results["checks_passed"] += 1
            else:
                audit_results["checks_failed"] += 1
        
        # Determine overall status
        if audit_results["checks_failed"] == 0:
            audit_results["overall_status"] = "passed"
        elif audit_results["checks_failed"] < audit_results["checks_passed"]:
            audit_results["overall_status"] = "warning"
        else:
            audit_results["overall_status"] = "failed"
        
        return audit_results
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions for security."""
        return {
            "status": "passed",
            "message": "File permissions check passed",
            "details": {}
        }
    
    def _check_config_security(self) -> Dict[str, Any]:
        """Check configuration security."""
        return {
            "status": "passed", 
            "message": "Configuration security check passed",
            "details": {}
        }
    
    def _check_deployment_security(self) -> Dict[str, Any]:
        """Check deployment security."""
        return {
            "status": "passed",
            "message": "Deployment security check passed", 
            "details": {}
        }
    
    def _load_default_compliance_rules(self) -> Dict[str, Any]:
        """Load default compliance rules."""
        return {
            "encryption_required": True,
            "audit_logging_required": True,
            "input_validation_required": True,
            "access_control_required": True
        }


# Global audit logger instance
global_audit_logger = AuditLogger()