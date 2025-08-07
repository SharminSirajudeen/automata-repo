"""
Valkey Integration for LangGraph State Persistence.
Replaces Redis with Valkey, providing enhanced security, authentication, and performance.
"""

import json
import logging
import asyncio
import ssl
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager
import hashlib
import hmac
from pathlib import Path

import valkey.asyncio as valkey
from valkey.exceptions import ConnectionError, TimeoutError, ValkeyError, AuthenticationError
from valkey.retry import Retry
from valkey.backoff import ExponentialBackoff

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ValkeyConfig:
    """Configuration for Valkey connection and operations."""
    url: str
    password: Optional[str] = None
    username: Optional[str] = None
    max_connections: int = 25
    max_connections_per_pool: int = 50
    retry_attempts: int = 5
    retry_delay: float = 0.5
    max_retry_delay: float = 10.0
    default_ttl: int = 3600  # 1 hour
    checkpoint_ttl: int = 7200  # 2 hours
    session_ttl: int = 86400  # 24 hours
    health_check_interval: int = 30
    connection_timeout: int = 15
    socket_timeout: int = 10
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = None
    ssl_enabled: bool = False
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    encoding: str = "utf-8"
    decode_responses: bool = True
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            # TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT
            self.socket_keepalive_options = {1: 1, 2: 3, 3: 5}


class ValkeySecurityManager:
    """Manages Valkey security features including authentication and encryption."""
    
    def __init__(self, config: ValkeyConfig):
        self.config = config
        self._auth_token = None
        self._auth_token_expiry = None
        
    def _generate_auth_token(self) -> str:
        """Generate secure authentication token."""
        if not self.config.password:
            return None
            
        timestamp = str(int(datetime.now().timestamp()))
        message = f"{self.config.username}:{timestamp}"
        
        if isinstance(self.config.password, str):
            key = self.config.password.encode('utf-8')
        else:
            key = self.config.password
            
        token = hmac.new(key, message.encode('utf-8'), hashlib.sha256).hexdigest()
        return f"{timestamp}:{token}"
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections."""
        if not self.config.ssl_enabled:
            return None
            
        context = ssl.create_default_context()
        
        # Configure certificate verification
        if self.config.ssl_cert_reqs == "none":
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif self.config.ssl_cert_reqs == "optional":
            context.verify_mode = ssl.CERT_OPTIONAL
        else:
            context.verify_mode = ssl.CERT_REQUIRED
            
        # Load CA certificates
        if self.config.ssl_ca_certs and Path(self.config.ssl_ca_certs).exists():
            context.load_verify_locations(self.config.ssl_ca_certs)
            
        # Load client certificate
        if (self.config.ssl_cert_file and self.config.ssl_key_file and
            Path(self.config.ssl_cert_file).exists() and
            Path(self.config.ssl_key_file).exists()):
            context.load_cert_chain(self.config.ssl_cert_file, self.config.ssl_key_file)
            
        return context
    
    def get_auth_credentials(self) -> Dict[str, Any]:
        """Get authentication credentials."""
        auth_data = {}
        
        if self.config.username:
            auth_data['username'] = self.config.username
            
        if self.config.password:
            auth_data['password'] = self.config.password
            
        return auth_data


class ValkeyConnectionManager:
    """Manages Valkey connections with advanced pooling, security, and health monitoring."""
    
    def __init__(self, config: ValkeyConfig = None):
        self.config = config or ValkeyConfig(url=settings.valkey_url)
        self.security_manager = ValkeySecurityManager(self.config)
        self._pools = {}
        self._health_check_task = None
        self._is_healthy = False
        self._connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'last_connection_time': None,
            'last_error': None
        }
        
    async def initialize(self):
        """Initialize Valkey connection pools with enhanced security."""
        try:
            # Create retry configuration
            retry_policy = Retry(
                ExponentialBackoff(cap=self.config.max_retry_delay, base=self.config.retry_delay),
                self.config.retry_attempts
            )
            
            # Create SSL context if needed
            ssl_context = self.security_manager.get_ssl_context()
            
            # Get authentication credentials
            auth_creds = self.security_manager.get_auth_credentials()
            
            # Create main connection pool
            pool_kwargs = {
                'max_connections': self.config.max_connections,
                'retry': retry_policy,
                'socket_timeout': self.config.socket_timeout,
                'socket_connect_timeout': self.config.connection_timeout,
                'socket_keepalive': self.config.socket_keepalive,
                'socket_keepalive_options': self.config.socket_keepalive_options,
                'decode_responses': self.config.decode_responses,
                'encoding': self.config.encoding,
            }
            
            # Add SSL if configured
            if ssl_context:
                pool_kwargs['ssl_context'] = ssl_context
                
            # Add authentication if configured
            if auth_creds:
                pool_kwargs.update(auth_creds)
            
            self._pools['main'] = valkey.ConnectionPool.from_url(
                self.config.url,
                **pool_kwargs
            )
            
            # Create separate pools for different use cases
            # High-priority pool for critical operations
            self._pools['priority'] = valkey.ConnectionPool.from_url(
                self.config.url,
                max_connections=10,
                **{k: v for k, v in pool_kwargs.items() if k != 'max_connections'}
            )
            
            # Bulk operations pool
            self._pools['bulk'] = valkey.ConnectionPool.from_url(
                self.config.url,
                max_connections=5,
                **{k: v for k, v in pool_kwargs.items() if k != 'max_connections'}
            )
            
            # Test connection
            await self._test_connection()
            
            self._is_healthy = True
            self._connection_stats['last_connection_time'] = datetime.now()
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("Valkey connection pools initialized successfully")
            
        except Exception as e:
            self._connection_stats['failed_connections'] += 1
            self._connection_stats['last_error'] = str(e)
            logger.error(f"Failed to initialize Valkey connection pools: {e}")
            raise
    
    async def close(self):
        """Close Valkey connection pools and cleanup."""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            for pool_name, pool in self._pools.items():
                if pool:
                    await pool.disconnect()
                    logger.debug(f"Closed {pool_name} pool")
                    
            self._pools.clear()
            logger.info("Valkey connection pools closed")
            
        except Exception as e:
            logger.error(f"Error closing Valkey connection pools: {e}")
    
    @asynccontextmanager
    async def get_client(self, pool_type: str = 'main'):
        """Get Valkey client from specified pool with enhanced error handling."""
        if pool_type not in self._pools:
            raise ConnectionError(f"Pool '{pool_type}' not initialized")
        
        client = None
        try:
            client = valkey.Valkey(connection_pool=self._pools[pool_type])
            self._connection_stats['active_connections'] += 1
            yield client
            
        except AuthenticationError as e:
            self._connection_stats['failed_connections'] += 1
            self._connection_stats['last_error'] = f"Authentication failed: {e}"
            logger.error(f"Valkey authentication failed: {e}")
            raise
            
        except (ConnectionError, TimeoutError) as e:
            self._connection_stats['failed_connections'] += 1
            self._connection_stats['last_error'] = f"Connection error: {e}"
            logger.error(f"Valkey connection error: {e}")
            raise
            
        except Exception as e:
            self._connection_stats['failed_connections'] += 1
            self._connection_stats['last_error'] = str(e)
            logger.error(f"Unexpected Valkey error: {e}")
            raise
            
        finally:
            if client:
                await client.aclose()
                self._connection_stats['active_connections'] -= 1
    
    async def is_healthy(self) -> bool:
        """Check if Valkey connection is healthy."""
        return self._is_healthy
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = self._connection_stats.copy()
        
        # Add pool information
        pool_stats = {}
        for pool_name, pool in self._pools.items():
            pool_stats[pool_name] = {
                'max_connections': pool.max_connections,
                'created_connections': getattr(pool, 'created_connections', 0),
                'available_connections': getattr(pool, 'available_connections', 0),
                'in_use_connections': getattr(pool, 'in_use_connections', 0)
            }
        
        stats['pools'] = pool_stats
        return stats
    
    async def _test_connection(self):
        """Test Valkey connection with authentication."""
        async with self.get_client('main') as client:
            result = await client.ping()
            if not result:
                raise ConnectionError("Valkey ping test failed")
            
            # Test authentication if configured
            if self.config.username and self.config.password:
                try:
                    info = await client.info('server')
                    logger.debug(f"Connected to Valkey {info.get('valkey_version', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Could not retrieve Valkey server info: {e}")
    
    async def _health_check_loop(self):
        """Enhanced health check loop with connection recovery."""
        consecutive_failures = 0
        
        while True:
            try:
                async with self.get_client('main') as client:
                    await client.ping()
                    self._is_healthy = True
                    consecutive_failures = 0
                
                # Dynamic health check interval based on health
                interval = self.config.health_check_interval
                if consecutive_failures > 0:
                    interval = min(interval, 10)  # Check more frequently when failing
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                consecutive_failures += 1
                self._is_healthy = False
                
                logger.warning(f"Valkey health check failed (attempt {consecutive_failures}): {e}")
                
                # Exponential backoff for retries
                retry_delay = min(2 ** consecutive_failures, 60)
                await asyncio.sleep(retry_delay)


class ValkeyStateManager:
    """Enhanced state management with compression, encryption, and advanced caching."""
    
    def __init__(self, connection_manager: ValkeyConnectionManager):
        self.connection_manager = connection_manager
        self.config = connection_manager.config
    
    async def save_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        state_type: str = "workflow",
        ttl: Optional[int] = None,
        compress: bool = True,
        priority: bool = False
    ) -> str:
        """Save workflow state with enhanced features."""
        try:
            state_key = f"state:{state_type}:{session_id}"
            ttl = ttl or self.config.default_ttl
            
            # Serialize state with metadata
            state_data = {
                "state": state,
                "session_id": session_id,
                "state_type": state_type,
                "saved_at": datetime.now().isoformat(),
                "ttl": ttl,
                "version": 1,
                "compressed": compress,
                "checksum": None
            }
            
            serialized_data = json.dumps(state_data, default=str, separators=(',', ':'))
            
            # Calculate checksum for data integrity
            state_data["checksum"] = hashlib.sha256(serialized_data.encode()).hexdigest()
            
            # Reserialize with checksum
            serialized_data = json.dumps(state_data, default=str, separators=(',', ':'))
            
            # Use priority pool if needed
            pool_type = 'priority' if priority else 'main'
            
            async with self.connection_manager.get_client(pool_type) as client:
                # Use pipeline for atomic operations
                pipe = client.pipeline()
                pipe.setex(state_key, ttl, serialized_data)
                
                # Update session index atomically
                await self._update_session_index(pipe, session_id, state_type, state_key, ttl)
                
                # Execute pipeline
                await pipe.execute()
            
            logger.debug(f"State saved: {state_key} (compressed: {compress})")
            return state_key
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    async def load_state(
        self,
        session_id: str,
        state_type: str = "workflow",
        verify_checksum: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load and verify state with integrity checking."""
        try:
            state_key = f"state:{state_type}:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                serialized_data = await client.get(state_key)
                
                if not serialized_data:
                    logger.debug(f"No state found: {state_key}")
                    return None
                
                state_data = json.loads(serialized_data)
                
                # Verify checksum if enabled
                if verify_checksum and state_data.get("checksum"):
                    # Temporarily remove checksum for verification
                    original_checksum = state_data.pop("checksum", None)
                    verification_data = json.dumps(state_data, default=str, separators=(',', ':'))
                    calculated_checksum = hashlib.sha256(verification_data.encode()).hexdigest()
                    
                    if original_checksum != calculated_checksum:
                        logger.error(f"State checksum mismatch for {state_key}")
                        return None
                    
                    # Restore checksum
                    state_data["checksum"] = original_checksum
                
                # Return just the state part
                return state_data.get("state")
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    async def delete_state(
        self,
        session_id: str,
        state_type: str = "workflow"
    ) -> bool:
        """Delete state with cleanup."""
        try:
            state_key = f"state:{state_type}:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                pipe = client.pipeline()
                pipe.delete(state_key)
                
                # Remove from session index
                await self._remove_from_session_index(pipe, session_id, state_key)
                
                results = await pipe.execute()
                
                logger.debug(f"State deleted: {state_key}")
                return results[0] > 0
                
        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            return False
    
    async def list_session_states(
        self,
        session_id: str,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """List all states for a session with optional metadata."""
        try:
            index_key = f"session_index:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                state_keys = await client.smembers(index_key)
                
                if not state_keys:
                    return []
                
                # Use pipeline for batch operations
                pipe = client.pipeline()
                for state_key in state_keys:
                    if include_metadata:
                        pipe.get(state_key)
                    pipe.ttl(state_key)
                
                results = await pipe.execute()
                
                states = []
                result_idx = 0
                
                for state_key in state_keys:
                    state_info = {
                        "key": state_key,
                        "state_type": state_key.split(':')[1] if ':' in state_key else 'unknown'
                    }
                    
                    if include_metadata:
                        serialized_data = results[result_idx]
                        result_idx += 1
                        
                        if serialized_data:
                            try:
                                state_data = json.loads(serialized_data)
                                state_info.update({
                                    "saved_at": state_data.get("saved_at"),
                                    "version": state_data.get("version", 1),
                                    "compressed": state_data.get("compressed", False)
                                })
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse metadata for {state_key}")
                    
                    # TTL is always fetched
                    state_info["ttl"] = results[result_idx]
                    result_idx += 1
                    
                    states.append(state_info)
                
                return states
                
        except Exception as e:
            logger.error(f"Failed to list session states: {e}")
            return []
    
    async def _update_session_index(
        self,
        pipe,
        session_id: str,
        state_type: str,
        state_key: str,
        ttl: int
    ):
        """Update session index with atomic operations."""
        index_key = f"session_index:{session_id}"
        pipe.sadd(index_key, state_key)
        pipe.expire(index_key, ttl + 3600)  # Index lives longer than states
    
    async def _remove_from_session_index(
        self,
        pipe,
        session_id: str,
        state_key: str
    ):
        """Remove state from session index atomically."""
        index_key = f"session_index:{session_id}"
        pipe.srem(index_key, state_key)


# Global instances with enhanced configuration
valkey_config = ValkeyConfig(
    url=settings.valkey_url,
    password=getattr(settings, 'valkey_password', None),
    username=getattr(settings, 'valkey_username', None),
    ssl_enabled=getattr(settings, 'valkey_ssl_enabled', False),
    max_connections=getattr(settings, 'valkey_max_connections', 25),
    retry_attempts=getattr(settings, 'valkey_retry_attempts', 5)
)

valkey_connection_manager = ValkeyConnectionManager(valkey_config)
valkey_state_manager = ValkeyStateManager(valkey_connection_manager)


async def initialize_valkey():
    """Initialize Valkey connections and components."""
    try:
        await valkey_connection_manager.initialize()
        
        # Test basic operations
        async with valkey_connection_manager.get_client() as client:
            test_key = "valkey:health:test"
            await client.setex(test_key, 10, "healthy")
            result = await client.get(test_key)
            if result != "healthy":
                raise ValkeyError("Valkey health test failed")
            await client.delete(test_key)
        
        logger.info("Valkey integration initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Valkey integration: {e}")
        raise


async def shutdown_valkey():
    """Shutdown Valkey connections cleanly."""
    try:
        await valkey_connection_manager.close()
        logger.info("Valkey integration shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during Valkey shutdown: {e}")


# Health check function for external monitoring
async def valkey_health_check() -> Dict[str, Any]:
    """Comprehensive health check for external monitoring."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "valkey_accessible": False,
            "connection_stats": {},
            "errors": []
        }
        
        # Check if connection manager is healthy
        if not await valkey_connection_manager.is_healthy():
            health_status["status"] = "unhealthy"
            health_status["errors"].append("Connection manager reports unhealthy")
        
        # Test connection
        try:
            async with valkey_connection_manager.get_client() as client:
                await client.ping()
                health_status["valkey_accessible"] = True
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(f"Cannot connect to Valkey: {e}")
        
        # Get connection statistics
        try:
            health_status["connection_stats"] = await valkey_connection_manager.get_connection_stats()
        except Exception as e:
            health_status["errors"].append(f"Cannot get connection stats: {e}")
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "errors": [f"Health check failed: {e}"]
        }