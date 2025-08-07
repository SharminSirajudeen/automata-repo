"""
Application-level load balancing for Automata Learning Platform.
Provides client-side load balancing with service discovery and health monitoring.
"""

import asyncio
import time
import random
import logging
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import httpx
import json
from datetime import datetime, timedelta
import statistics
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    RESOURCE_BASED = "resource_based"

class ServerStatus(Enum):
    """Server health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

@dataclass
class ServerInstance:
    """Represents a backend server instance."""
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    status: ServerStatus = ServerStatus.HEALTHY
    last_check: datetime = field(default_factory=datetime.utcnow)
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    circuit_breaker_open: bool = False
    circuit_breaker_open_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        """Get the full URL for this server."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times[-10:])  # Last 10 requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    def is_available(self) -> bool:
        """Check if server is available for requests."""
        return (
            self.status == ServerStatus.HEALTHY and
            not self.circuit_breaker_open and
            self.current_connections < self.max_connections
        )
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics."""
        self.response_times.append(response_time)
        if len(self.response_times) > 100:  # Keep last 100 measurements
            self.response_times = self.response_times[-100:]
        
        if success:
            self.success_count += 1
            self.error_count = max(0, self.error_count - 1)  # Gradual recovery
        else:
            self.error_count += 1
        
        # Circuit breaker logic
        if self.error_count >= 5 and self.success_rate < 0.5:
            self.circuit_breaker_open = True
            self.circuit_breaker_open_time = datetime.utcnow()
            logger.warning(f"Circuit breaker opened for {self.url}")

class ConsistentHash:
    """Consistent hashing implementation for load balancing."""
    
    def __init__(self, nodes: List[str], replicas: int = 150):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        
        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key: str) -> int:
        """Simple hash function."""
        return hash(key) % (2**32)
    
    def add_node(self, node: str):
        """Add a node to the ring."""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node: str):
        """Remove a node from the ring."""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for the key."""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find the first node clockwise
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]

class ServiceDiscovery:
    """Service discovery for dynamic server registration."""
    
    def __init__(self):
        self.services: Dict[str, List[ServerInstance]] = {}
        self.watchers: List[Callable] = []
    
    def register_service(self, service_name: str, server: ServerInstance):
        """Register a service instance."""
        if service_name not in self.services:
            self.services[service_name] = []
        
        # Remove existing instance if it exists
        self.services[service_name] = [
            s for s in self.services[service_name] 
            if not (s.host == server.host and s.port == server.port)
        ]
        
        self.services[service_name].append(server)
        self._notify_watchers(service_name, 'register', server)
        logger.info(f"Registered service {service_name}: {server.url}")
    
    def deregister_service(self, service_name: str, host: str, port: int):
        """Deregister a service instance."""
        if service_name in self.services:
            original_count = len(self.services[service_name])
            self.services[service_name] = [
                s for s in self.services[service_name] 
                if not (s.host == host and s.port == port)
            ]
            
            if len(self.services[service_name]) < original_count:
                self._notify_watchers(service_name, 'deregister', None)
                logger.info(f"Deregistered service {service_name}: {host}:{port}")
    
    def get_services(self, service_name: str) -> List[ServerInstance]:
        """Get all instances of a service."""
        return self.services.get(service_name, [])
    
    def watch_service(self, callback: Callable):
        """Watch for service changes."""
        self.watchers.append(callback)
    
    def _notify_watchers(self, service_name: str, action: str, server: Optional[ServerInstance]):
        """Notify watchers of service changes."""
        for callback in self.watchers:
            try:
                callback(service_name, action, server)
            except Exception as e:
                logger.error(f"Error in service watcher: {e}")

class HealthChecker:
    """Health checker for monitoring server availability."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self.servers: List[ServerInstance] = []
        self._check_task = None
    
    def add_server(self, server: ServerInstance):
        """Add server to health check monitoring."""
        if server not in self.servers:
            self.servers.append(server)
    
    def remove_server(self, server: ServerInstance):
        """Remove server from health check monitoring."""
        if server in self.servers:
            self.servers.remove(server)
    
    async def start(self):
        """Start health checking."""
        self.running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")
    
    async def stop(self):
        """Stop health checking."""
        self.running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")
    
    async def _health_check_loop(self):
        """Main health check loop."""
        while self.running:
            try:
                await self._check_all_servers()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _check_all_servers(self):
        """Check health of all servers."""
        tasks = []
        for server in self.servers.copy():  # Copy to avoid modification during iteration
            tasks.append(self._check_server_health(server))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_server_health(self, server: ServerInstance):
        """Check health of a single server."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{server.url}/healthz")
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    server.status = ServerStatus.HEALTHY
                    server.record_request(response_time, True)
                    
                    # Reset circuit breaker if enough time has passed
                    if (server.circuit_breaker_open and 
                        server.circuit_breaker_open_time and
                        datetime.utcnow() - server.circuit_breaker_open_time > timedelta(minutes=5)):
                        server.circuit_breaker_open = False
                        server.circuit_breaker_open_time = None
                        logger.info(f"Circuit breaker reset for {server.url}")
                        
                else:
                    server.status = ServerStatus.DEGRADED
                    server.record_request(response_time, False)
                    
        except Exception as e:
            response_time = time.time() - start_time
            server.status = ServerStatus.UNHEALTHY
            server.record_request(response_time, False)
            logger.warning(f"Health check failed for {server.url}: {e}")
        
        server.last_check = datetime.utcnow()

class LoadBalancer:
    """Advanced application-level load balancer."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.servers: List[ServerInstance] = []
        self.service_discovery = ServiceDiscovery()
        self.health_checker = HealthChecker()
        self.consistent_hash = None
        self.round_robin_index = 0
        self._lock = threading.Lock()
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = datetime.utcnow()
        
        # Service discovery watcher
        self.service_discovery.watch_service(self._on_service_change)
    
    def add_server(self, server: ServerInstance):
        """Add a server to the load balancer."""
        with self._lock:
            if server not in self.servers:
                self.servers.append(server)
                self.health_checker.add_server(server)
                
                # Update consistent hash ring
                if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    self._update_consistent_hash()
                
                logger.info(f"Added server: {server.url}")
    
    def remove_server(self, server: ServerInstance):
        """Remove a server from the load balancer."""
        with self._lock:
            if server in self.servers:
                self.servers.remove(server)
                self.health_checker.remove_server(server)
                
                # Update consistent hash ring
                if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    self._update_consistent_hash()
                
                logger.info(f"Removed server: {server.url}")
    
    def _update_consistent_hash(self):
        """Update consistent hash ring."""
        if self.servers:
            node_names = [f"{s.host}:{s.port}" for s in self.servers if s.is_available()]
            self.consistent_hash = ConsistentHash(node_names)
    
    def _on_service_change(self, service_name: str, action: str, server: Optional[ServerInstance]):
        """Handle service discovery changes."""
        if action == 'register' and server:
            self.add_server(server)
        elif action == 'deregister':
            # Find and remove the server
            for s in self.servers.copy():
                if s.host == server.host and s.port == server.port:
                    self.remove_server(s)
                    break
    
    def get_server(self, client_id: Optional[str] = None) -> Optional[ServerInstance]:
        """Get next server based on load balancing strategy."""
        available_servers = [s for s in self.servers if s.is_available()]
        
        if not available_servers:
            logger.error("No available servers")
            return None
        
        with self._lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                server = self._round_robin_selection(available_servers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                server = self._least_connections_selection(available_servers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                server = self._weighted_round_robin_selection(available_servers)
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                server = random.choice(available_servers)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                server = self._least_response_time_selection(available_servers)
            elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                server = self._consistent_hash_selection(available_servers, client_id)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                server = self._resource_based_selection(available_servers)
            else:
                server = available_servers[0]
            
            if server:
                server.current_connections += 1
            
            return server
    
    def _round_robin_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Round-robin server selection."""
        if not servers:
            return None
        
        server = servers[self.round_robin_index % len(servers)]
        self.round_robin_index += 1
        return server
    
    def _least_connections_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Least connections server selection."""
        return min(servers, key=lambda s: s.current_connections)
    
    def _weighted_round_robin_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Weighted round-robin selection."""
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.weight)
        
        if not weighted_servers:
            return servers[0] if servers else None
        
        server = weighted_servers[self.round_robin_index % len(weighted_servers)]
        self.round_robin_index += 1
        return server
    
    def _least_response_time_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Least response time selection."""
        return min(servers, key=lambda s: s.average_response_time)
    
    def _consistent_hash_selection(self, servers: List[ServerInstance], client_id: Optional[str]) -> ServerInstance:
        """Consistent hash selection."""
        if not client_id or not self.consistent_hash:
            return random.choice(servers)
        
        node_name = self.consistent_hash.get_node(client_id)
        if not node_name:
            return random.choice(servers)
        
        # Find server by node name
        host, port = node_name.split(':')
        for server in servers:
            if server.host == host and server.port == int(port):
                return server
        
        return random.choice(servers)
    
    def _resource_based_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Resource-based selection considering multiple factors."""
        def score(server: ServerInstance) -> float:
            """Calculate server score (lower is better)."""
            connection_factor = server.current_connections / server.max_connections
            response_time_factor = server.average_response_time / 1000.0  # Normalize to seconds
            error_rate_factor = 1.0 - server.success_rate
            
            return connection_factor + response_time_factor + error_rate_factor
        
        return min(servers, key=score)
    
    def release_server(self, server: ServerInstance):
        """Release a server connection."""
        with self._lock:
            if server.current_connections > 0:
                server.current_connections -= 1
    
    @asynccontextmanager
    async def get_client(self, client_id: Optional[str] = None) -> httpx.AsyncClient:
        """Get an HTTP client with load balancing."""
        server = self.get_server(client_id)
        if not server:
            raise RuntimeError("No available servers")
        
        start_time = time.time()
        success = False
        
        try:
            async with httpx.AsyncClient(
                base_url=server.url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_connections=10)
            ) as client:
                yield client
                success = True
                self.successful_requests += 1
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Request failed for server {server.url}: {e}")
            raise
        finally:
            response_time = time.time() - start_time
            server.record_request(response_time, success)
            self.release_server(server)
            self.total_requests += 1
    
    async def start(self):
        """Start the load balancer."""
        await self.health_checker.start()
        logger.info("Load balancer started")
    
    async def stop(self):
        """Stop the load balancer."""
        await self.health_checker.stop()
        logger.info("Load balancer stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get load balancer metrics."""
        uptime = datetime.utcnow() - self.start_time
        
        return {
            "strategy": self.strategy.value,
            "total_servers": len(self.servers),
            "healthy_servers": len([s for s in self.servers if s.status == ServerStatus.HEALTHY]),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "uptime_seconds": uptime.total_seconds(),
            "servers": [
                {
                    "url": s.url,
                    "status": s.status.value,
                    "current_connections": s.current_connections,
                    "success_rate": s.success_rate,
                    "average_response_time": s.average_response_time,
                    "circuit_breaker_open": s.circuit_breaker_open
                }
                for s in self.servers
            ]
        }

# Global load balancer instance
load_balancer = None

async def initialize_load_balancer(config: Dict[str, Any]):
    """Initialize the global load balancer."""
    global load_balancer
    
    strategy = LoadBalancingStrategy(config.get('strategy', 'least_connections'))
    load_balancer = LoadBalancer(strategy)
    
    # Add configured servers
    for server_config in config.get('servers', []):
        server = ServerInstance(
            host=server_config['host'],
            port=server_config['port'],
            weight=server_config.get('weight', 1),
            max_connections=server_config.get('max_connections', 100)
        )
        load_balancer.add_server(server)
    
    await load_balancer.start()
    logger.info("Load balancer initialized")

async def cleanup_load_balancer():
    """Cleanup the global load balancer."""
    global load_balancer
    if load_balancer:
        await load_balancer.stop()
        load_balancer = None

# Health check endpoint for graceful shutdown
async def health_check():
    """Health check endpoint for the load balancer."""
    if not load_balancer:
        return {"status": "not_initialized"}
    
    metrics = load_balancer.get_metrics()
    return {
        "status": "healthy" if metrics["healthy_servers"] > 0 else "unhealthy",
        "metrics": metrics
    }

# Graceful shutdown support
_shutdown_event = asyncio.Event()

def signal_shutdown():
    """Signal graceful shutdown."""
    _shutdown_event.set()

async def wait_for_shutdown():
    """Wait for shutdown signal."""
    await _shutdown_event.wait()

# Circuit breaker decorator
def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60):
    """Circuit breaker decorator for functions."""
    def decorator(func):
        func._circuit_breaker_failures = 0
        func._circuit_breaker_last_failure = None
        func._circuit_breaker_open = False
        
        async def wrapper(*args, **kwargs):
            # Check if circuit breaker is open
            if func._circuit_breaker_open:
                if (func._circuit_breaker_last_failure and 
                    time.time() - func._circuit_breaker_last_failure > recovery_timeout):
                    func._circuit_breaker_open = False
                    func._circuit_breaker_failures = 0
                    logger.info(f"Circuit breaker reset for {func.__name__}")
                else:
                    raise RuntimeError(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                func._circuit_breaker_failures = 0
                return result
            except Exception as e:
                func._circuit_breaker_failures += 1
                func._circuit_breaker_last_failure = time.time()
                
                if func._circuit_breaker_failures >= failure_threshold:
                    func._circuit_breaker_open = True
                    logger.warning(f"Circuit breaker opened for {func.__name__}")
                
                raise e
        
        return wrapper
    return decorator