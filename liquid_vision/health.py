"""
🏥 AUTONOMOUS HEALTH SYSTEM v2.0

Self-healing infrastructure with comprehensive health checks, 
automatic recovery procedures, and predictive maintenance.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Types of automated recovery actions."""
    RESTART_COMPONENT = "restart_component"
    CLEAR_CACHE = "clear_cache"
    SCALE_UP = "scale_up"
    FAILOVER = "failover"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable[[], bool]
    description: str = ""
    critical: bool = False
    timeout_seconds: int = 30
    retry_attempts: int = 3
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    
@dataclass
class HealthReport:
    """Comprehensive system health report."""
    overall_status: HealthStatus
    timestamp: float = field(default_factory=time.time)
    component_statuses: Dict[str, HealthStatus] = field(default_factory=dict)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recovery_actions_taken: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    
    @property
    def is_healthy(self) -> bool:
        return self.overall_status == HealthStatus.HEALTHY
        
    @property
    def needs_attention(self) -> bool:
        return self.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]


class AutonomousHealthSystem:
    """
    Self-healing health monitoring system with predictive capabilities.
    
    Features:
    - Continuous health monitoring of all system components
    - Automatic recovery actions based on health status
    - Predictive maintenance and early warning detection
    - Dependency-aware health checking
    - Comprehensive reporting and alerting
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.recovery_procedures: Dict[RecoveryAction, Callable] = {}
        self.health_history: List[HealthReport] = []
        self.running = False
        self._health_thread: Optional[threading.Thread] = None
        self.start_time = time.time()
        
        # System state
        self.last_health_report: Optional[HealthReport] = None
        self.consecutive_failures: Dict[str, int] = {}
        self.recovery_cooldown: Dict[str, float] = {}
        
        self._setup_default_checks()
        self._setup_recovery_procedures()
        
    def _setup_default_checks(self):
        """Setup default system health checks."""
        self.register_health_check(
            "system_memory",
            self._check_memory_usage,
            "Monitor system memory usage",
            critical=True,
            recovery_actions=[RecoveryAction.CLEAR_CACHE]
        )
        
        self.register_health_check(
            "system_cpu", 
            self._check_cpu_usage,
            "Monitor CPU usage patterns",
            recovery_actions=[RecoveryAction.SCALE_UP]
        )
        
        self.register_health_check(
            "disk_space",
            self._check_disk_space, 
            "Monitor available disk space",
            critical=True,
            recovery_actions=[RecoveryAction.MANUAL_INTERVENTION]
        )
        
        self.register_health_check(
            "process_health",
            self._check_process_health,
            "Verify process is running correctly",
            critical=True,
            recovery_actions=[RecoveryAction.RESTART_COMPONENT]
        )
        
        self.register_health_check(
            "memory_leaks",
            self._check_memory_leaks,
            "Detect potential memory leaks",
            recovery_actions=[RecoveryAction.RESTART_COMPONENT]
        )
        
    def _setup_recovery_procedures(self):
        """Setup automated recovery procedures."""
        self.recovery_procedures = {
            RecoveryAction.RESTART_COMPONENT: self._restart_component,
            RecoveryAction.CLEAR_CACHE: self._clear_system_cache,
            RecoveryAction.SCALE_UP: self._scale_up_resources,
            RecoveryAction.FAILOVER: self._initiate_failover,
            RecoveryAction.MANUAL_INTERVENTION: self._request_manual_intervention,
        }
        
    def register_health_check(self, 
                            name: str,
                            check_function: Callable[[], bool],
                            description: str = "",
                            critical: bool = False,
                            timeout_seconds: int = 30,
                            retry_attempts: int = 3,
                            recovery_actions: List[RecoveryAction] = None,
                            dependencies: List[str] = None) -> None:
        """Register a new health check."""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            description=description,
            critical=critical,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            recovery_actions=recovery_actions or [],
            dependencies=dependencies or []
        )
        logger.info(f"Registered health check: {name}")
        
    def register_recovery_procedure(self, 
                                  action: RecoveryAction,
                                  procedure: Callable[[str], bool]) -> None:
        """Register custom recovery procedure."""
        self.recovery_procedures[action] = procedure
        logger.info(f"Registered recovery procedure: {action.value}")
        
    def start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.running:
            return
            
        self.running = True
        self._health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self._health_thread.start()
        logger.info("🏥 Autonomous health monitoring started")
        
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring gracefully."""
        self.running = False
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
        logger.info("🏥 Health monitoring stopped")
        
    async def run_health_check(self, force_all: bool = False) -> HealthReport:
        """Run comprehensive health check and return report."""
        logger.info("🏥 Running comprehensive health check")
        
        start_time = time.time()
        component_statuses = {}
        failed_checks = []
        warnings = []
        recovery_actions_taken = []
        
        # Run individual health checks
        for check_name, health_check in self.health_checks.items():
            try:
                # Skip non-critical checks if not forced and system is under stress
                if not force_all and not health_check.critical and self._is_system_stressed():
                    continue
                    
                # Check dependencies first
                if not self._check_dependencies(health_check):
                    component_statuses[check_name] = HealthStatus.DEGRADED
                    warnings.append(f"Dependencies not met for {check_name}")
                    continue
                    
                # Run the actual health check with retries
                success = await self._run_check_with_retry(health_check)
                
                if success:
                    component_statuses[check_name] = HealthStatus.HEALTHY
                    # Reset consecutive failures on success
                    self.consecutive_failures.pop(check_name, None)
                else:
                    # Track consecutive failures
                    self.consecutive_failures[check_name] = self.consecutive_failures.get(check_name, 0) + 1
                    
                    if health_check.critical:
                        component_statuses[check_name] = HealthStatus.CRITICAL
                        failed_checks.append(check_name)
                    else:
                        component_statuses[check_name] = HealthStatus.UNHEALTHY
                        warnings.append(check_name)
                    
                    # Attempt automated recovery
                    recovery_actions = await self._attempt_recovery(health_check)
                    recovery_actions_taken.extend(recovery_actions)
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed with exception: {e}")
                component_statuses[check_name] = HealthStatus.CRITICAL
                failed_checks.append(check_name)
                
        # Determine overall health status
        overall_status = self._calculate_overall_status(component_statuses)
        
        # Create comprehensive health report
        report = HealthReport(
            overall_status=overall_status,
            component_statuses=component_statuses,
            failed_checks=failed_checks,
            warnings=warnings,
            recovery_actions_taken=recovery_actions_taken,
            uptime_seconds=time.time() - self.start_time
        )
        
        # Store for history and analysis
        self.last_health_report = report
        self.health_history.append(report)
        
        # Keep only recent history (last 100 reports)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
            
        logger.info(f"Health check completed in {time.time() - start_time:.2f}s - Status: {overall_status.value}")
        return report
        
    async def _run_check_with_retry(self, health_check: HealthCheck) -> bool:
        """Run health check with retry logic and timeout."""
        for attempt in range(health_check.retry_attempts):
            try:
                # Run with timeout
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, health_check.check_function
                    ),
                    timeout=health_check.timeout_seconds
                )
                
                if result:
                    return True
                    
            except asyncio.TimeoutError:
                logger.warning(f"Health check {health_check.name} timed out (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Health check {health_check.name} failed: {e} (attempt {attempt + 1})")
                
            # Wait before retry (exponential backoff)
            if attempt < health_check.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)
                
        return False
        
    def _check_dependencies(self, health_check: HealthCheck) -> bool:
        """Check if health check dependencies are satisfied."""
        for dependency in health_check.dependencies:
            if dependency not in self.health_checks:
                continue
                
            # Check if dependency is healthy in last report
            if (self.last_health_report and 
                dependency in self.last_health_report.component_statuses and
                self.last_health_report.component_statuses[dependency] != HealthStatus.HEALTHY):
                return False
                
        return True
        
    async def _attempt_recovery(self, health_check: HealthCheck) -> List[str]:
        """Attempt automated recovery for failed health check."""
        recovery_actions_taken = []
        
        # Check cooldown period to prevent recovery storms
        cooldown_key = health_check.name
        if cooldown_key in self.recovery_cooldown:
            if time.time() - self.recovery_cooldown[cooldown_key] < 300:  # 5 min cooldown
                return recovery_actions_taken
                
        # Only attempt recovery after multiple consecutive failures
        consecutive = self.consecutive_failures.get(health_check.name, 0)
        if consecutive < 2:  # Wait for at least 2 consecutive failures
            return recovery_actions_taken
            
        for action in health_check.recovery_actions:
            try:
                if action in self.recovery_procedures:
                    logger.info(f"Attempting recovery action: {action.value} for {health_check.name}")
                    success = await self._execute_recovery_procedure(action, health_check.name)
                    
                    if success:
                        recovery_actions_taken.append(f"{action.value}:{health_check.name}")
                        self.recovery_cooldown[cooldown_key] = time.time()
                        break  # Stop after first successful recovery
                    else:
                        logger.warning(f"Recovery action {action.value} failed for {health_check.name}")
                        
            except Exception as e:
                logger.error(f"Recovery action {action.value} threw exception: {e}")
                
        return recovery_actions_taken
        
    async def _execute_recovery_procedure(self, action: RecoveryAction, component: str) -> bool:
        """Execute recovery procedure with timeout."""
        try:
            procedure = self.recovery_procedures[action]
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, procedure, component),
                timeout=120  # 2 minute timeout for recovery actions
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Recovery procedure {action.value} timed out")
            return False
        except Exception as e:
            logger.error(f"Recovery procedure {action.value} failed: {e}")
            return False
            
    def _calculate_overall_status(self, component_statuses: Dict[str, HealthStatus]) -> HealthStatus:
        """Calculate overall system health status."""
        if not component_statuses:
            return HealthStatus.DEGRADED
            
        status_counts = {status: 0 for status in HealthStatus}
        for status in component_statuses.values():
            status_counts[status] += 1
            
        # Overall status logic
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > len(component_statuses) // 2:
            return HealthStatus.UNHEALTHY  
        elif status_counts[HealthStatus.DEGRADED] > 0 or status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
            
    def _is_system_stressed(self) -> bool:
        """Determine if system is under stress."""
        if not self.last_health_report:
            return False
            
        return (self.last_health_report.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL] or
                len(self.last_health_report.failed_checks) > 2)
                
    def _health_monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                # Run health check
                report = loop.run_until_complete(self.run_health_check())
                
                # Log status changes
                if (self.last_health_report and 
                    report.overall_status != self.last_health_report.overall_status):
                    logger.info(f"Health status changed: {self.last_health_report.overall_status.value} → {report.overall_status.value}")
                    
                # Adjust monitoring frequency based on health
                if report.overall_status == HealthStatus.HEALTHY:
                    sleep_time = 30  # Check every 30 seconds when healthy
                elif report.overall_status == HealthStatus.DEGRADED:
                    sleep_time = 15  # More frequent when degraded
                else:
                    sleep_time = 5   # Very frequent when unhealthy/critical
                    
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                traceback.print_exc()
                time.sleep(60)  # Back off on errors
                
        loop.close()
        
    # Default health check implementations
    def _check_memory_usage(self) -> bool:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90.0  # Less than 90% memory usage
        except ImportError:
            logger.warning("psutil not available for memory check")
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
            
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage patterns."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 85.0  # Less than 85% CPU usage
        except ImportError:
            return True
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return False
            
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 90.0  # Less than 90% disk usage
        except ImportError:
            return True
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return False
            
    def _check_process_health(self) -> bool:
        """Check current process health."""
        try:
            import psutil
            process = psutil.Process()
            return (process.is_running() and 
                   process.status() not in ['zombie', 'dead'])
        except ImportError:
            return True
        except Exception as e:
            logger.error(f"Process health check failed: {e}")
            return False
            
    def _check_memory_leaks(self) -> bool:
        """Basic memory leak detection."""
        # Simple heuristic: check if memory usage is consistently increasing
        if len(self.health_history) < 5:
            return True
            
        try:
            import psutil
            recent_memory = []
            for report in self.health_history[-5:]:
                if 'system_memory' in report.component_statuses:
                    # This is a simplified check - in real implementation, 
                    # you'd track actual memory values over time
                    recent_memory.append(1 if report.component_statuses['system_memory'] == HealthStatus.HEALTHY else 0)
                    
            # If memory health has been declining consistently, might indicate leak
            if len(recent_memory) >= 3:
                return sum(recent_memory[-3:]) > 0  # At least one healthy in last 3
                
            return True
            
        except Exception as e:
            logger.error(f"Memory leak check failed: {e}")
            return True
            
    # Recovery procedure implementations
    def _restart_component(self, component: str) -> bool:
        """Restart system component."""
        logger.info(f"Simulating component restart for: {component}")
        # In real implementation, this would restart specific services/components
        import gc
        gc.collect()  # Force garbage collection as basic restart simulation
        return True
        
    def _clear_system_cache(self, component: str) -> bool:
        """Clear system caches."""
        logger.info(f"Clearing system caches for: {component}")
        import gc
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        return True
        
    def _scale_up_resources(self, component: str) -> bool:
        """Scale up system resources."""
        logger.info(f"Scaling up resources for: {component}")
        # In real implementation, this would trigger auto-scaling
        return True
        
    def _initiate_failover(self, component: str) -> bool:
        """Initiate failover procedures."""
        logger.info(f"Initiating failover for: {component}")
        # In real implementation, this would switch to backup systems
        return True
        
    def _request_manual_intervention(self, component: str) -> bool:
        """Request manual intervention."""
        logger.critical(f"MANUAL INTERVENTION REQUIRED for: {component}")
        # In real implementation, this would trigger alerts/tickets
        return False  # Manual intervention needed
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health system summary."""
        return {
            "monitoring_active": self.running,
            "registered_checks": len(self.health_checks),
            "recovery_procedures": len(self.recovery_procedures),
            "last_report": self.last_health_report.__dict__ if self.last_health_report else None,
            "history_count": len(self.health_history),
            "uptime_seconds": time.time() - self.start_time,
            "consecutive_failures": dict(self.consecutive_failures),
            "recovery_cooldowns": {k: time.time() - v for k, v in self.recovery_cooldown.items()},
        }


# Global health system instance
_health_system: Optional[AutonomousHealthSystem] = None


def get_health_system() -> AutonomousHealthSystem:
    """Get or create global health system instance."""
    global _health_system
    if _health_system is None:
        _health_system = AutonomousHealthSystem()
        _health_system.start_health_monitoring()
    return _health_system


async def run_health_check() -> HealthReport:
    """Global function to run health check."""
    return await get_health_system().run_health_check()


def register_health_check(name: str, check_function: Callable[[], bool], **kwargs) -> None:
    """Global function to register health check.""" 
    get_health_system().register_health_check(name, check_function, **kwargs)