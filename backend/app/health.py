"""
Health check endpoints for monitoring system health and dependencies.
Provides comprehensive health status for production monitoring.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import psutil
import redis
import sqlalchemy
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.database import get_db
from app.config import get_settings

router = APIRouter()
settings = get_settings()

class HealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis client for health checks."""
        try:
            import redis
            self.redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
        except Exception:
            self.redis_client = None
    
    async def check_database(self, db) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            # Basic connectivity test
            result = await db.execute(text("SELECT 1"))
            connectivity_time = time.time() - start_time
            
            # Pool status
            pool_info = {
                "size": db.get_bind().pool.size(),
                "checked_in": db.get_bind().pool.checkedin(),
                "checked_out": db.get_bind().pool.checkedout(),
                "overflow": db.get_bind().pool.overflow(),
                "invalid": db.get_bind().pool.invalid()
            }
            
            # Performance test
            start_time = time.time()
            await db.execute(text("SELECT COUNT(*) FROM information_schema.tables"))
            query_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "connectivity_time_ms": round(connectivity_time * 1000, 2),
                "query_time_ms": round(query_time * 1000, 2),
                "pool": pool_info,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except OperationalError as e:
            return {
                "status": "unhealthy",
                "error": f"Database connection error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Database check failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        if not self.redis_client:
            return {
                "status": "unhealthy",
                "error": "Redis client not initialized",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        try:
            start_time = time.time()
            
            # Basic connectivity test
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            ping_time = time.time() - start_time
            
            # Get Redis info
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.info
            )
            
            # Test set/get operation
            test_key = "health_check_test"
            test_value = str(int(time.time()))
            
            start_time = time.time()
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.setex, test_key, 60, test_value
            )
            retrieved_value = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, test_key
            )
            operation_time = time.time() - start_time
            
            # Cleanup
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, test_key
            )
            
            return {
                "status": "healthy",
                "ping_time_ms": round(ping_time * 1000, 2),
                "operation_time_ms": round(operation_time * 1000, 2),
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "test_successful": retrieved_value == test_value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Redis check failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources (CPU, memory, disk)."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network statistics
            network = psutil.net_io_counters()
            
            # Load average (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                pass  # Not available on Windows
            
            return {
                "status": "healthy",
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": load_avg
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": round((disk.used / disk.total) * 100, 2)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"System resource check failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_application_info(self) -> Dict[str, Any]:
        """Get application-specific health information."""
        uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime_seconds, 2),
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now(timezone.utc).isoformat(),
            "version": getattr(settings, 'APP_VERSION', '1.0.0'),
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "features": {
                "ollama_enabled": bool(settings.OLLAMA_BASE_URL),
                "redis_enabled": self.redis_client is not None,
                "metrics_enabled": getattr(settings, 'PROMETHEUS_METRICS', False)
            }
        }

# Global health checker instance
health_checker = HealthChecker()

@router.get("/health", tags=["health"])
async def basic_health_check():
    """Basic health check endpoint - returns OK if service is running."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "automata-backend"
        }
    )

@router.get("/health/ready", tags=["health"])
async def readiness_check(db=Depends(get_db)):
    """
    Readiness check - returns OK if service is ready to handle requests.
    Checks database connectivity and essential services.
    """
    checks = {}
    overall_status = "healthy"
    status_code = 200
    
    try:
        # Check database
        checks["database"] = await health_checker.check_database(db)
        if checks["database"]["status"] != "healthy":
            overall_status = "unhealthy"
            status_code = 503
        
        # Check Redis if available
        checks["redis"] = await health_checker.check_redis()
        if checks["redis"]["status"] != "healthy":
            # Redis is not critical for basic functionality
            overall_status = "degraded" if overall_status == "healthy" else overall_status
        
        response_data = {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks
        }
        
        return JSONResponse(status_code=status_code, content=response_data)
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/health/live", tags=["health"])
async def liveness_check():
    """
    Liveness check - returns OK if the application process is alive.
    Used by orchestrators to determine if the container should be restarted.
    """
    try:
        # Simple check to ensure the application is responsive
        app_info = health_checker.get_application_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "alive",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "application": app_info
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "dead",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/health/detailed", tags=["health"])
async def detailed_health_check(db=Depends(get_db)):
    """
    Detailed health check - comprehensive status of all system components.
    Includes resource usage, dependency status, and performance metrics.
    """
    checks = {}
    overall_status = "healthy"
    warnings = []
    
    try:
        # Application info
        checks["application"] = health_checker.get_application_info()
        
        # Database check
        checks["database"] = await health_checker.check_database(db)
        if checks["database"]["status"] != "healthy":
            overall_status = "unhealthy"
        
        # Redis check
        checks["redis"] = await health_checker.check_redis()
        if checks["redis"]["status"] != "healthy":
            warnings.append("Redis is not available - caching disabled")
        
        # System resources
        checks["system"] = health_checker.check_system_resources()
        if checks["system"]["status"] == "healthy":
            # Check for resource warnings
            cpu_percent = checks["system"]["cpu"]["usage_percent"]
            memory_percent = checks["system"]["memory"]["percent"]
            disk_percent = checks["system"]["disk"]["percent"]
            
            if cpu_percent > 80:
                warnings.append(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 85:
                warnings.append(f"High memory usage: {memory_percent}%")
            if disk_percent > 90:
                warnings.append(f"High disk usage: {disk_percent}%")
            
            if warnings and overall_status == "healthy":
                overall_status = "degraded"
        
        # Overall health determination
        unhealthy_services = [
            service for service, check in checks.items() 
            if check["status"] == "unhealthy"
        ]
        
        if unhealthy_services:
            overall_status = "unhealthy"
        elif warnings:
            overall_status = "degraded"
        
        status_code = 200
        if overall_status == "unhealthy":
            status_code = 503
        elif overall_status == "degraded":
            status_code = 200  # Still functional
        
        response_data = {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "warnings": warnings,
            "summary": {
                "total_checks": len(checks),
                "healthy_checks": sum(1 for check in checks.values() if check["status"] == "healthy"),
                "unhealthy_services": unhealthy_services
            }
        }
        
        return JSONResponse(status_code=status_code, content=response_data)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Health check failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/metrics", tags=["monitoring"])
async def prometheus_metrics():
    """
    Prometheus-compatible metrics endpoint.
    Returns metrics in Prometheus format for monitoring integration.
    """
    try:
        # Basic application metrics
        uptime_seconds = (datetime.now(timezone.utc) - health_checker.start_time).total_seconds()
        
        # System metrics
        system_info = health_checker.check_system_resources()
        
        metrics = []
        
        # Application metrics
        metrics.append(f"app_uptime_seconds {uptime_seconds}")
        metrics.append(f"app_info{{version=\"{getattr(settings, 'APP_VERSION', '1.0.0')}\",environment=\"{settings.ENVIRONMENT}\"}} 1")
        
        if system_info["status"] == "healthy":
            # System metrics
            metrics.append(f"system_cpu_usage_percent {system_info['cpu']['usage_percent']}")
            metrics.append(f"system_memory_usage_percent {system_info['memory']['percent']}")
            metrics.append(f"system_disk_usage_percent {system_info['disk']['percent']}")
            metrics.append(f"system_memory_total_bytes {system_info['memory']['total']}")
            metrics.append(f"system_memory_available_bytes {system_info['memory']['available']}")
            metrics.append(f"system_disk_total_bytes {system_info['disk']['total']}")
            metrics.append(f"system_disk_free_bytes {system_info['disk']['free']}")
        
        metrics_text = '\n'.join(metrics) + '\n'
        
        return JSONResponse(
            status_code=200,
            content=metrics_text,
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate metrics: {str(e)}"
        )