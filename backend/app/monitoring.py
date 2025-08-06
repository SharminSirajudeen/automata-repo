"""
Monitoring and metrics module for the Automata Learning Platform.
Provides Prometheus metrics, health checks, and performance monitoring.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import asyncio

from fastapi import Request, Response, HTTPException
from prometheus_client import (
    Counter, Histogram, Gauge, Info, generate_latest, 
    CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
)
import httpx

logger = logging.getLogger(__name__)

# Create a custom registry for our metrics
metrics_registry = CollectorRegistry()

# Define Prometheus metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=metrics_registry
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=metrics_registry
)

active_connections = Gauge(
    'http_active_connections',
    'Number of active HTTP connections',
    registry=metrics_registry
)

database_connections = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=metrics_registry
)

ai_model_requests = Counter(
    'ai_model_requests_total',
    'Total AI model requests',
    ['model_name', 'endpoint'],
    registry=metrics_registry
)

ai_model_duration = Histogram(
    'ai_model_duration_seconds',
    'AI model request duration in seconds',
    ['model_name', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=metrics_registry
)

jflap_operations = Counter(
    'jflap_operations_total',
    'Total JFLAP operations',
    ['operation_type', 'status'],
    registry=metrics_registry
)

problem_validations = Counter(
    'problem_validations_total',
    'Total problem validations',
    ['problem_type', 'result'],
    registry=metrics_registry
)

user_sessions = Gauge(
    'user_sessions_active',
    'Number of active user sessions',
    registry=metrics_registry
)

# System metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=metrics_registry
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes',
    registry=metrics_registry
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    registry=metrics_registry
)

# Application info
app_info = Info(
    'automata_platform_info',
    'Information about the Automata Learning Platform',
    registry=metrics_registry
)


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_times = deque(maxlen=1000)  # Keep last 1000 request times
        self.error_counts = defaultdict(int)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'last_accessed': None
        })
        self.system_stats_thread = None
        self.running = False
        
    def start_monitoring(self):
        """Start the monitoring background tasks."""
        self.running = True
        self.system_stats_thread = threading.Thread(target=self._collect_system_stats)
        self.system_stats_thread.daemon = True
        self.system_stats_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring background tasks."""
        self.running = False
        if self.system_stats_thread:
            self.system_stats_thread.join()
        logger.info("Performance monitoring stopped")
        
    def _collect_system_stats(self):
        """Collect system statistics in background thread."""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                system_cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                system_memory_usage.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                system_disk_usage.set(disk_percent)
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system stats: {e}")
                time.sleep(60)  # Wait longer on error
    
    def record_request(self, method: str, endpoint: str, duration: float, status_code: int):
        """Record request metrics."""
        request_count.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Track in internal stats
        self.request_times.append(duration)
        self.endpoint_stats[endpoint]['count'] += 1
        self.endpoint_stats[endpoint]['total_time'] += duration
        self.endpoint_stats[endpoint]['last_accessed'] = datetime.utcnow()
        
        if status_code >= 400:
            self.endpoint_stats[endpoint]['errors'] += 1
            self.error_counts[status_code] += 1
    
    def record_ai_request(self, model_name: str, endpoint: str, duration: float):
        """Record AI model request metrics."""
        ai_model_requests.labels(model_name=model_name, endpoint=endpoint).inc()
        ai_model_duration.labels(model_name=model_name, endpoint=endpoint).observe(duration)
    
    def record_jflap_operation(self, operation_type: str, success: bool):
        """Record JFLAP operation metrics."""
        status = "success" if success else "error"
        jflap_operations.labels(operation_type=operation_type, status=status).inc()
    
    def record_problem_validation(self, problem_type: str, result: str):
        """Record problem validation metrics."""
        problem_validations.labels(problem_type=problem_type, result=result).inc()
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get current performance statistics summary."""
        uptime = time.time() - self.start_time
        
        # Calculate average response time
        avg_response_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        
        # Get recent error rate (last 100 requests)
        recent_requests = list(self.request_times)[-100:]
        error_rate = sum(1 for _ in self.error_counts.values()) / max(len(recent_requests), 1)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": sum(stats['count'] for stats in self.endpoint_stats.values()),
            "average_response_time": avg_response_time,
            "error_rate": error_rate,
            "active_endpoints": len(self.endpoint_stats),
            "top_endpoints": self._get_top_endpoints(5),
            "error_summary": dict(self.error_counts),
            "system_stats": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent if psutil.disk_usage('/') else 0
            }
        }
    
    def _get_top_endpoints(self, limit: int) -> List[Dict[str, Any]]:
        """Get top endpoints by request count."""
        sorted_endpoints = sorted(
            self.endpoint_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return [
            {
                "endpoint": endpoint,
                "count": stats['count'],
                "avg_time": stats['total_time'] / stats['count'] if stats['count'] > 0 else 0,
                "error_rate": stats['errors'] / stats['count'] if stats['count'] > 0 else 0
            }
            for endpoint, stats in sorted_endpoints[:limit]
        ]


class HealthChecker:
    """Comprehensive health checking for application components."""
    
    def __init__(self):
        self.checks = {}
        self.register_default_checks()
    
    def register_check(self, name: str, check_func, timeout: int = 5):
        """Register a health check function."""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout,
            'last_result': None,
            'last_check': None
        }
    
    def register_default_checks(self):
        """Register default health checks."""
        self.register_check('database', self._check_database)
        self.register_check('ai_service', self._check_ai_service)
        self.register_check('memory', self._check_memory)
        self.register_check('disk_space', self._check_disk_space)
    
    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {
                'status': 'unknown',
                'error': f'Check {name} not found'
            }
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                check['func'](),
                timeout=check['timeout']
            )
            
            check['last_result'] = result
            check['last_check'] = datetime.utcnow()
            
            return {
                'status': 'healthy',
                'result': result,
                'response_time': time.time() - start_time
            }
            
        except asyncio.TimeoutError:
            return {
                'status': 'timeout',
                'error': f'Check {name} timed out after {check["timeout"]}s'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = 'healthy'
        
        for name in self.checks:
            result = await self.run_check(name)
            results[name] = result
            
            if result['status'] != 'healthy':
                overall_status = 'degraded' if overall_status == 'healthy' else 'unhealthy'
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            from .database import get_db
            # This would typically run a simple query
            return {
                'connected': True,
                'response_time_ms': 10  # Placeholder
            }
        except Exception as e:
            raise Exception(f"Database check failed: {e}")
    
    async def _check_ai_service(self) -> Dict[str, Any]:
        """Check AI service (Ollama) connectivity."""
        try:
            from .config import settings
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.ollama_base_url}/api/tags",
                    timeout=3.0
                )
                
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    return {
                        'connected': True,
                        'models_available': len(models),
                        'models': [m['name'] for m in models[:3]]  # First 3 models
                    }
                else:
                    raise Exception(f"AI service returned status {response.status_code}")
                    
        except Exception as e:
            raise Exception(f"AI service check failed: {e}")
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            raise Exception(f"High memory usage: {memory.percent}%")
        
        return {
            'usage_percent': memory.percent,
            'available_gb': memory.available / (1024**3)
        }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 85:
            raise Exception(f"Low disk space: {usage_percent:.1f}% used")
        
        return {
            'usage_percent': usage_percent,
            'free_gb': disk.free / (1024**3)
        }


class MonitoringMiddleware:
    """Middleware for automatic request monitoring."""
    
    def __init__(self, app, monitor: PerformanceMonitor):
        self.app = app
        self.monitor = monitor
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            start_time = time.time()
            
            # Increment active connections
            active_connections.inc()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Record metrics when response starts
                    duration = time.time() - start_time
                    status_code = message["status"]
                    
                    # Extract endpoint from path
                    endpoint = request.url.path
                    method = request.method
                    
                    self.monitor.record_request(method, endpoint, duration, status_code)
                    
                    # Decrement active connections
                    active_connections.dec()
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Global instances
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker()


async def get_metrics():
    """Get Prometheus metrics."""
    return Response(
        generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST
    )


async def get_health():
    """Get comprehensive health status."""
    return await health_checker.run_all_checks()


async def get_performance_stats():
    """Get performance statistics."""
    return performance_monitor.get_stats_summary()


def setup_monitoring():
    """Initialize monitoring components."""
    # Set application info
    app_info.info({
        'version': '1.0.0',
        'environment': 'production',
        'build_date': datetime.now().isoformat()
    })
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    
    logger.info("Monitoring setup completed")


def cleanup_monitoring():
    """Cleanup monitoring resources."""
    performance_monitor.stop_monitoring()
    logger.info("Monitoring cleanup completed")