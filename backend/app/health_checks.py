"""
Comprehensive Health Check System.
Replaces sleep-based readiness checks with proper service dependency verification,
health monitoring, and startup sequence management for production deployment.
"""

import asyncio
import aiohttp
import logging
import time
import socket
import subprocess
import psutil
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks."""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    SYSTEM_RESOURCES = "system_resources"
    DEPENDENCY_SERVICE = "dependency_service"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    check_type: CheckType
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'type': self.check_type.value,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error
        }


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_type: CheckType
    check_function: Callable[[], Any]
    timeout_seconds: int = 10
    critical: bool = True  # Whether failure should mark service as unhealthy
    interval_seconds: int = 30
    retries: int = 3
    retry_delay_seconds: int = 1
    
    # Dependency information
    depends_on: List[str] = field(default_factory=list)
    
    # Last check info
    last_result: Optional[HealthCheckResult] = None
    last_check_time: Optional[datetime] = None


class ServiceHealthMonitor:
    """Comprehensive health monitoring system for all service dependencies."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # System thresholds
        self.system_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'max_response_time_ms': 5000.0
        }
        
        # Overall service status
        self.service_status = HealthStatus.UNKNOWN
        self.startup_complete = False
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default health checks for common dependencies."""
        
        # Database health check
        self.register_health_check(
            name="postgresql",
            check_type=CheckType.DATABASE,
            check_function=self._check_postgresql,
            timeout_seconds=15,
            critical=True,
            interval_seconds=30
        )
        
        # Valkey health check
        self.register_health_check(
            name="valkey",
            check_type=CheckType.CACHE,
            check_function=self._check_valkey,
            timeout_seconds=10,
            critical=True,
            interval_seconds=30
        )
        
        # Ollama health check
        self.register_health_check(
            name="ollama",
            check_type=CheckType.EXTERNAL_API,
            check_function=self._check_ollama,
            timeout_seconds=30,
            critical=True,
            interval_seconds=60,
            depends_on=["network"]
        )
        
        # System resources check
        self.register_health_check(
            name="system_resources",
            check_type=CheckType.SYSTEM_RESOURCES,
            check_function=self._check_system_resources,
            timeout_seconds=5,
            critical=False,
            interval_seconds=30
        )
        
        # Filesystem health check
        self.register_health_check(
            name="filesystem",
            check_type=CheckType.FILESYSTEM,
            check_function=self._check_filesystem,
            timeout_seconds=10,
            critical=True,
            interval_seconds=60
        )
        
        # Network connectivity check
        self.register_health_check(
            name="network",
            check_type=CheckType.NETWORK,
            check_function=self._check_network,
            timeout_seconds=15,
            critical=True,
            interval_seconds=60
        )
    
    def register_health_check(
        self,
        name: str,
        check_type: CheckType,
        check_function: Callable,
        timeout_seconds: int = 10,
        critical: bool = True,
        interval_seconds: int = 30,
        retries: int = 3,
        retry_delay_seconds: int = 1,
        depends_on: List[str] = None
    ):
        """Register a custom health check."""
        health_check = HealthCheck(
            name=name,
            check_type=check_type,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            critical=critical,
            interval_seconds=interval_seconds,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            depends_on=depends_on or []
        )
        
        self.health_checks[name] = health_check
        logger.info(f"Registered health check: {name}")
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Run all health checks
                await self._run_all_health_checks()
                
                # Update overall service status
                self._update_service_status()
                
                # Wait before next check cycle
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _run_all_health_checks(self):
        """Run all registered health checks."""
        # Group checks by interval
        now = datetime.now()
        checks_to_run = []
        
        for check_name, health_check in self.health_checks.items():
            # Check if it's time to run this check
            if (health_check.last_check_time is None or
                (now - health_check.last_check_time).total_seconds() >= health_check.interval_seconds):
                checks_to_run.append(check_name)
        
        if not checks_to_run:
            return
        
        # Run checks in dependency order
        ordered_checks = self._order_checks_by_dependencies(checks_to_run)
        
        # Execute checks
        for check_name in ordered_checks:
            if check_name in self.health_checks:
                await self._run_single_health_check(check_name)
    
    def _order_checks_by_dependencies(self, check_names: List[str]) -> List[str]:
        """Order health checks by their dependencies."""
        ordered = []
        remaining = set(check_names)
        
        while remaining:
            # Find checks with no unresolved dependencies
            ready = []
            for check_name in remaining:
                health_check = self.health_checks[check_name]
                if not any(dep in remaining for dep in health_check.depends_on):
                    ready.append(check_name)
            
            if not ready:
                # Circular dependency or missing dependency, add all remaining
                ready = list(remaining)
            
            ordered.extend(ready)
            remaining -= set(ready)
        
        return ordered
    
    async def _run_single_health_check(self, check_name: str):
        """Run a single health check with retries."""
        health_check = self.health_checks[check_name]
        
        for attempt in range(health_check.retries + 1):
            try:
                start_time = time.time()
                
                # Run the check with timeout
                result = await asyncio.wait_for(
                    self._execute_health_check(health_check),
                    timeout=health_check.timeout_seconds
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Create result
                health_check.last_result = HealthCheckResult(
                    name=check_name,
                    check_type=health_check.check_type,
                    status=result.get('status', HealthStatus.UNKNOWN),
                    message=result.get('message', 'Check completed'),
                    details=result.get('details', {}),
                    duration_ms=duration_ms
                )
                
                health_check.last_check_time = datetime.now()
                
                # If successful, break retry loop
                if health_check.last_result.status != HealthStatus.UNHEALTHY:
                    break
                    
            except asyncio.TimeoutError:
                health_check.last_result = HealthCheckResult(
                    name=check_name,
                    check_type=health_check.check_type,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out after {health_check.timeout_seconds}s",
                    error="timeout"
                )
                health_check.last_check_time = datetime.now()
                
            except Exception as e:
                health_check.last_result = HealthCheckResult(
                    name=check_name,
                    check_type=health_check.check_type,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    error=str(e)
                )
                health_check.last_check_time = datetime.now()
            
            # Wait before retry
            if attempt < health_check.retries:
                await asyncio.sleep(health_check.retry_delay_seconds)
    
    async def _execute_health_check(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Execute a health check function."""
        if asyncio.iscoroutinefunction(health_check.check_function):
            return await health_check.check_function()
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, health_check.check_function)
    
    def _update_service_status(self):
        """Update overall service health status."""
        if not self.health_checks:
            self.service_status = HealthStatus.UNKNOWN
            return
        
        critical_checks = [
            check for check in self.health_checks.values()
            if check.critical and check.last_result is not None
        ]
        
        if not critical_checks:
            self.service_status = HealthStatus.UNKNOWN
            return
        
        # Check critical services
        unhealthy_critical = [
            check for check in critical_checks
            if check.last_result.status == HealthStatus.UNHEALTHY
        ]
        
        degraded_critical = [
            check for check in critical_checks
            if check.last_result.status == HealthStatus.DEGRADED
        ]
        
        if unhealthy_critical:
            self.service_status = HealthStatus.UNHEALTHY
        elif degraded_critical:
            self.service_status = HealthStatus.DEGRADED
        else:
            self.service_status = HealthStatus.HEALTHY
    
    # Specific health check implementations
    async def _check_postgresql(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity."""
        try:
            # Try to import the database module
            from .database import engine
            
            # Test connection
            async with engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'PostgreSQL connection successful',
                'details': {
                    'database_url': settings.database_url.split('@')[0] + '@***'  # Hide password
                }
            }
            
        except ImportError as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': 'Database module not available',
                'details': {'error': str(e)}
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Database connection failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _check_valkey(self) -> Dict[str, Any]:
        """Check Valkey cache connectivity."""
        try:
            from .valkey_integration import valkey_connection_manager
            
            # Test connection
            async with valkey_connection_manager.get_client() as client:
                result = await client.ping()
                if not result:
                    raise Exception("Ping returned False")
                
                # Test basic operations
                test_key = "health_check_test"
                await client.setex(test_key, 5, "test_value")
                value = await client.get(test_key)
                await client.delete(test_key)
                
                if value != "test_value":
                    raise Exception("Set/get test failed")
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Valkey connection successful',
                'details': {
                    'valkey_url': settings.valkey_url.split('@')[0] + '@***' if '@' in settings.valkey_url else settings.valkey_url
                }
            }
            
        except ImportError as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': 'Valkey module not available',
                'details': {'error': str(e)}
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Valkey connection failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _check_ollama(self) -> Dict[str, Any]:
        """Check Ollama API connectivity."""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if Ollama is running
                async with session.get(
                    f"{settings.ollama_base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    data = await response.json()
                    models = data.get('models', [])
                    
                    # Check if required models are available
                    model_names = [model.get('name', '') for model in models]
                    required_models = [
                        settings.ollama_generator_model,
                        settings.ollama_explainer_model,
                        settings.ollama_default_model
                    ]
                    
                    missing_models = [
                        model for model in required_models 
                        if not any(model in name for name in model_names)
                    ]
                    
                    if missing_models:
                        return {
                            'status': HealthStatus.DEGRADED,
                            'message': f'Some required models missing: {missing_models}',
                            'details': {
                                'available_models': model_names,
                                'missing_models': missing_models,
                                'ollama_url': settings.ollama_base_url
                            }
                        }
                    
                    return {
                        'status': HealthStatus.HEALTHY,
                        'message': 'Ollama API accessible, all required models available',
                        'details': {
                            'available_models': model_names,
                            'ollama_url': settings.ollama_base_url
                        }
                    }
                    
        except aiohttp.ClientError as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Ollama API connection failed: {str(e)}',
                'details': {
                    'error': str(e),
                    'ollama_url': settings.ollama_base_url
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Ollama health check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (Unix only)
            load_avg = None
            try:
                if hasattr(os, 'getloadavg'):
                    load_avg = os.getloadavg()
            except OSError:
                pass  # Not available on this system
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.system_thresholds['cpu_percent']:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.system_thresholds['memory_percent']:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.system_thresholds['disk_percent']:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "System resources within normal limits"
            if issues:
                message = f"Resource issues detected: {', '.join(issues)}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk.free / (1024**3),
                    'load_average': load_avg
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': f'Failed to check system resources: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem health and required directories."""
        try:
            issues = []
            
            # Check required directories
            required_dirs = [
                '/tmp',
                '/var/log',
                Path(__file__).parent.parent  # App directory
            ]
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                if not path.exists():
                    issues.append(f"Required directory missing: {dir_path}")
                elif not path.is_dir():
                    issues.append(f"Path is not a directory: {dir_path}")
                elif not os.access(str(path), os.R_OK | os.W_OK):
                    issues.append(f"Insufficient permissions: {dir_path}")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                issues.append(f"Low disk space: {free_gb:.2f}GB free")
            
            status = HealthStatus.UNHEALTHY if issues else HealthStatus.HEALTHY
            message = "Filesystem checks passed" if not issues else f"Filesystem issues: {', '.join(issues)}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'free_space_gb': free_gb,
                    'issues': issues
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': f'Filesystem check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            # Test DNS resolution
            try:
                socket.gethostbyname('google.com')
                dns_ok = True
            except socket.gaierror:
                dns_ok = False
            
            # Test HTTP connectivity
            http_ok = False
            response_time_ms = None
            
            try:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        'http://httpbin.org/status/200',
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            http_ok = True
                            response_time_ms = (time.time() - start_time) * 1000
            except Exception:
                pass
            
            # Determine status
            if not dns_ok and not http_ok:
                status = HealthStatus.UNHEALTHY
                message = "Network connectivity issues: DNS and HTTP failed"
            elif not dns_ok:
                status = HealthStatus.DEGRADED
                message = "DNS resolution failed"
            elif not http_ok:
                status = HealthStatus.DEGRADED
                message = "HTTP connectivity failed"
            else:
                status = HealthStatus.HEALTHY
                message = "Network connectivity OK"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'dns_resolution': dns_ok,
                    'http_connectivity': http_ok,
                    'response_time_ms': response_time_ms
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': f'Network check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def wait_for_dependencies(
        self,
        required_services: List[str] = None,
        timeout_seconds: int = 300,
        check_interval_seconds: int = 5
    ) -> bool:
        """Wait for required services to be healthy before starting."""
        required_services = required_services or ['postgresql', 'valkey']
        
        logger.info(f"Waiting for dependencies: {required_services}")
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            all_healthy = True
            
            for service_name in required_services:
                if service_name not in self.health_checks:
                    logger.warning(f"Unknown service in dependencies: {service_name}")
                    continue
                
                # Run health check
                await self._run_single_health_check(service_name)
                
                health_check = self.health_checks[service_name]
                if (not health_check.last_result or 
                    health_check.last_result.status == HealthStatus.UNHEALTHY):
                    all_healthy = False
                    logger.info(f"Waiting for {service_name} to become healthy...")
                    break
            
            if all_healthy:
                logger.info("All dependencies are healthy")
                self.startup_complete = True
                return True
            
            await asyncio.sleep(check_interval_seconds)
        
        logger.error(f"Timeout waiting for dependencies after {timeout_seconds}s")
        return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        checks = {}
        for name, health_check in self.health_checks.items():
            if health_check.last_result:
                checks[name] = health_check.last_result.to_dict()
            else:
                checks[name] = {
                    'name': name,
                    'type': health_check.check_type.value,
                    'status': HealthStatus.UNKNOWN.value,
                    'message': 'Not yet checked'
                }
        
        return {
            'service_status': self.service_status.value,
            'startup_complete': self.startup_complete,
            'timestamp': datetime.now().isoformat(),
            'checks': checks
        }
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.service_status == HealthStatus.HEALTHY
    
    def is_ready(self) -> bool:
        """Check if service is ready to accept traffic."""
        return self.startup_complete and self.service_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


# Global health monitor instance
health_monitor = ServiceHealthMonitor()


async def initialize_health_monitor():
    """Initialize health monitoring."""
    await health_monitor.start_monitoring()
    logger.info("Health monitor initialized")


async def shutdown_health_monitor():
    """Shutdown health monitoring."""
    await health_monitor.stop_monitoring()
    logger.info("Health monitor shut down")


async def wait_for_startup(
    required_services: List[str] = None,
    timeout_seconds: int = 300
) -> bool:
    """Wait for all required services to be ready."""
    return await health_monitor.wait_for_dependencies(required_services, timeout_seconds)


def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    return health_monitor.get_health_status()


def is_healthy() -> bool:
    """Check if application is healthy."""
    return health_monitor.is_healthy()


def is_ready() -> bool:
    """Check if application is ready."""
    return health_monitor.is_ready()


# Health check endpoints for web framework
async def health_endpoint():
    """Health endpoint for load balancers."""
    status = health_monitor.get_health_status()
    if health_monitor.is_healthy():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    else:
        return {"status": status["service_status"], "timestamp": datetime.now().isoformat()}


async def readiness_endpoint():
    """Readiness endpoint for Kubernetes."""
    status = health_monitor.get_health_status()
    if health_monitor.is_ready():
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    else:
        return {"status": "not_ready", "details": status, "timestamp": datetime.now().isoformat()}


async def liveness_endpoint():
    """Liveness endpoint for Kubernetes."""
    # Simple liveness check - service is alive if monitoring is running
    if health_monitor.is_monitoring:
        return {"status": "alive", "timestamp": datetime.now().isoformat()}
    else:
        return {"status": "dead", "timestamp": datetime.now().isoformat()}


# Startup sequence helper
async def perform_startup_sequence(
    required_services: List[str] = None,
    timeout_seconds: int = 300
) -> bool:
    """Perform complete startup sequence with proper health checks."""
    logger.info("Starting application startup sequence")
    
    # Initialize health monitoring
    await initialize_health_monitor()
    
    # Wait for dependencies
    success = await wait_for_startup(required_services, timeout_seconds)
    
    if success:
        logger.info("Startup sequence completed successfully")
    else:
        logger.error("Startup sequence failed")
    
    return success