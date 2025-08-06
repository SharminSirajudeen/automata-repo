"""
Redis Integration for LangGraph State Persistence.
Provides Redis setup, connection management, and data persistence utilities.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Configuration for Redis connection and operations."""
    url: str
    max_connections: int = 20
    retry_attempts: int = 3
    retry_delay: float = 1.0
    default_ttl: int = 3600  # 1 hour
    checkpoint_ttl: int = 7200  # 2 hours
    session_ttl: int = 86400  # 24 hours
    health_check_interval: int = 30
    connection_timeout: int = 10
    socket_timeout: int = 5


class RedisConnectionManager:
    """Manages Redis connections with connection pooling and health checks."""
    
    def __init__(self, config: RedisConfig = None):
        self.config = config or RedisConfig(url=settings.redis_url)
        self._pool = None
        self._health_check_task = None
        self._is_healthy = False
        
    async def initialize(self):
        """Initialize Redis connection pool."""
        try:
            self._pool = redis.ConnectionPool.from_url(
                self.config.url,
                max_connections=self.config.max_connections,
                retry_on_timeout=True,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout,
                decode_responses=True
            )
            
            # Test connection
            async with self.get_client() as client:
                await client.ping()
                
            self._is_healthy = True
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("Redis connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise
    
    async def close(self):
        """Close Redis connection pool and cleanup."""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._pool:
                await self._pool.disconnect()
                
            logger.info("Redis connection pool closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis connection pool: {e}")
    
    @asynccontextmanager
    async def get_client(self):
        """Get Redis client from pool with context management."""
        if not self._pool:
            raise ConnectionError("Redis pool not initialized")
        
        client = redis.Redis(connection_pool=self._pool)
        try:
            yield client
        finally:
            await client.close()
    
    async def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        return self._is_healthy
    
    async def _health_check_loop(self):
        """Periodic health check for Redis connection."""
        while True:
            try:
                async with self.get_client() as client:
                    await client.ping()
                    self._is_healthy = True
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                self._is_healthy = False
                await asyncio.sleep(5)  # Quick retry on failure


class RedisStateManager:
    """Manages workflow state persistence in Redis."""
    
    def __init__(self, connection_manager: RedisConnectionManager):
        self.connection_manager = connection_manager
        self.config = connection_manager.config
    
    async def save_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        state_type: str = "workflow",
        ttl: Optional[int] = None
    ) -> str:
        """Save workflow state to Redis."""
        try:
            state_key = f"state:{state_type}:{session_id}"
            ttl = ttl or self.config.default_ttl
            
            # Serialize state with metadata
            state_data = {
                "state": state,
                "session_id": session_id,
                "state_type": state_type,
                "saved_at": datetime.now().isoformat(),
                "ttl": ttl
            }
            
            serialized_data = json.dumps(state_data, default=str)
            
            async with self.connection_manager.get_client() as client:
                await client.setex(state_key, ttl, serialized_data)
                
                # Also save to session index
                await self._update_session_index(client, session_id, state_type, state_key)
            
            logger.debug(f"State saved: {state_key}")
            return state_key
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    async def load_state(
        self,
        session_id: str,
        state_type: str = "workflow"
    ) -> Optional[Dict[str, Any]]:
        """Load workflow state from Redis."""
        try:
            state_key = f"state:{state_type}:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                serialized_data = await client.get(state_key)
                
                if not serialized_data:
                    logger.debug(f"No state found: {state_key}")
                    return None
                
                state_data = json.loads(serialized_data)
                
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
        """Delete workflow state from Redis."""
        try:
            state_key = f"state:{state_type}:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                result = await client.delete(state_key)
                
                # Remove from session index
                await self._remove_from_session_index(client, session_id, state_key)
                
                logger.debug(f"State deleted: {state_key}")
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            return False
    
    async def list_session_states(self, session_id: str) -> List[Dict[str, Any]]:
        """List all states for a session."""
        try:
            index_key = f"session_index:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                state_keys = await client.smembers(index_key)
                
                states = []
                for state_key in state_keys:
                    serialized_data = await client.get(state_key)
                    if serialized_data:
                        state_data = json.loads(serialized_data)
                        states.append({
                            "key": state_key,
                            "state_type": state_data.get("state_type"),
                            "saved_at": state_data.get("saved_at"),
                            "ttl": await client.ttl(state_key)
                        })
                
                return states
                
        except Exception as e:
            logger.error(f"Failed to list session states: {e}")
            return []
    
    async def _update_session_index(
        self,
        client: redis.Redis,
        session_id: str,
        state_type: str,
        state_key: str
    ):
        """Update session index with new state."""
        index_key = f"session_index:{session_id}"
        await client.sadd(index_key, state_key)
        await client.expire(index_key, self.config.session_ttl)
    
    async def _remove_from_session_index(
        self,
        client: redis.Redis,
        session_id: str,
        state_key: str
    ):
        """Remove state from session index."""
        index_key = f"session_index:{session_id}"
        await client.srem(index_key, state_key)


class RedisCheckpointStore:
    """Enhanced checkpoint storage with versioning and recovery."""
    
    def __init__(self, connection_manager: RedisConnectionManager):
        self.connection_manager = connection_manager
        self.config = connection_manager.config
    
    async def save_checkpoint(
        self,
        session_id: str,
        checkpoint_data: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """Save checkpoint with versioning."""
        try:
            # Generate version if not provided
            if not version:
                timestamp = int(datetime.now().timestamp() * 1000)
                version = f"v{timestamp}"
            
            checkpoint_id = f"checkpoint:{session_id}:{version}"
            
            # Prepare checkpoint data
            checkpoint_record = {
                "session_id": session_id,
                "version": version,
                "data": checkpoint_data,
                "created_at": datetime.now().isoformat(),
                "checkpoint_id": checkpoint_id
            }
            
            serialized_data = json.dumps(checkpoint_record, default=str)
            
            async with self.connection_manager.get_client() as client:
                # Save checkpoint
                await client.setex(
                    checkpoint_id, 
                    self.config.checkpoint_ttl, 
                    serialized_data
                )
                
                # Update checkpoint list for session
                await self._update_checkpoint_list(client, session_id, version)
                
                # Maintain checkpoint limit (keep last 20)
                await self._maintain_checkpoint_limit(client, session_id, 20)
            
            logger.debug(f"Checkpoint saved: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    async def load_checkpoint(
        self,
        session_id: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint by session and version."""
        try:
            if version:
                checkpoint_id = f"checkpoint:{session_id}:{version}"
            else:
                # Get latest checkpoint
                checkpoint_id = await self._get_latest_checkpoint_id(session_id)
                if not checkpoint_id:
                    return None
            
            async with self.connection_manager.get_client() as client:
                serialized_data = await client.get(checkpoint_id)
                
                if not serialized_data:
                    return None
                
                checkpoint_record = json.loads(serialized_data)
                return checkpoint_record.get("data")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    async def list_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a session."""
        try:
            list_key = f"checkpoint_list:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                checkpoint_versions = await client.lrange(list_key, 0, -1)
                
                checkpoints = []
                for version in checkpoint_versions:
                    checkpoint_id = f"checkpoint:{session_id}:{version}"
                    serialized_data = await client.get(checkpoint_id)
                    
                    if serialized_data:
                        checkpoint_record = json.loads(serialized_data)
                        checkpoints.append({
                            "checkpoint_id": checkpoint_id,
                            "version": version,
                            "created_at": checkpoint_record.get("created_at"),
                            "ttl": await client.ttl(checkpoint_id)
                        })
                
                # Sort by creation time (newest first)
                checkpoints.sort(
                    key=lambda x: x.get("created_at", ""), 
                    reverse=True
                )
                
                return checkpoints
                
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def delete_checkpoint(
        self,
        session_id: str,
        version: str
    ) -> bool:
        """Delete a specific checkpoint."""
        try:
            checkpoint_id = f"checkpoint:{session_id}:{version}"
            
            async with self.connection_manager.get_client() as client:
                result = await client.delete(checkpoint_id)
                
                # Remove from checkpoint list
                list_key = f"checkpoint_list:{session_id}"
                await client.lrem(list_key, 1, version)
                
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    async def _get_latest_checkpoint_id(self, session_id: str) -> Optional[str]:
        """Get the latest checkpoint ID for a session."""
        try:
            list_key = f"checkpoint_list:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                latest_version = await client.lindex(list_key, 0)
                
                if latest_version:
                    return f"checkpoint:{session_id}:{latest_version}"
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None
    
    async def _update_checkpoint_list(
        self,
        client: redis.Redis,
        session_id: str,
        version: str
    ):
        """Update checkpoint list with new version."""
        list_key = f"checkpoint_list:{session_id}"
        
        # Add to front of list (latest first)
        await client.lpush(list_key, version)
        await client.expire(list_key, self.config.checkpoint_ttl)
    
    async def _maintain_checkpoint_limit(
        self,
        client: redis.Redis,
        session_id: str,
        limit: int
    ):
        """Maintain checkpoint limit by removing old ones."""
        list_key = f"checkpoint_list:{session_id}"
        
        # Get current length
        current_length = await client.llen(list_key)
        
        if current_length > limit:
            # Remove excess checkpoints
            excess_versions = await client.lrange(list_key, limit, -1)
            
            # Delete the actual checkpoint data
            for version in excess_versions:
                checkpoint_id = f"checkpoint:{session_id}:{version}"
                await client.delete(checkpoint_id)
            
            # Trim the list
            await client.ltrim(list_key, 0, limit - 1)


class RedisSessionManager:
    """Manages user sessions and workflow tracking."""
    
    def __init__(self, connection_manager: RedisConnectionManager):
        self.connection_manager = connection_manager
        self.config = connection_manager.config
    
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        session_type: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new session."""
        try:
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "session_type": session_type,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "status": "active"
            }
            
            session_key = f"session:{session_id}"
            serialized_data = json.dumps(session_data, default=str)
            
            async with self.connection_manager.get_client() as client:
                await client.setex(session_key, self.config.session_ttl, serialized_data)
                
                # Add to user's session list
                await self._add_to_user_sessions(client, user_id, session_id)
            
            logger.info(f"Session created: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data."""
        try:
            session_key = f"session:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                serialized_data = await client.get(session_key)
                
                if not serialized_data:
                    return False
                
                session_data = json.loads(serialized_data)
                session_data.update(updates)
                session_data["last_activity"] = datetime.now().isoformat()
                
                updated_data = json.dumps(session_data, default=str)
                await client.setex(session_key, self.config.session_ttl, updated_data)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        try:
            session_key = f"session:{session_id}"
            
            async with self.connection_manager.get_client() as client:
                serialized_data = await client.get(session_key)
                
                if not serialized_data:
                    return None
                
                return json.loads(serialized_data)
                
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def list_user_sessions(
        self,
        user_id: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all sessions for a user."""
        try:
            user_sessions_key = f"user_sessions:{user_id}"
            
            async with self.connection_manager.get_client() as client:
                session_ids = await client.smembers(user_sessions_key)
                
                sessions = []
                for session_id in session_ids:
                    session_data = await self.get_session(session_id)
                    if session_data:
                        if not status or session_data.get("status") == status:
                            sessions.append(session_data)
                
                # Sort by last activity
                sessions.sort(
                    key=lambda x: x.get("last_activity", ""), 
                    reverse=True
                )
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to list user sessions: {e}")
            return []
    
    async def close_session(self, session_id: str) -> bool:
        """Close/deactivate a session."""
        try:
            updates = {
                "status": "closed",
                "closed_at": datetime.now().isoformat()
            }
            
            return await self.update_session(session_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to close session: {e}")
            return False
    
    async def _add_to_user_sessions(
        self,
        client: redis.Redis,
        user_id: str,
        session_id: str
    ):
        """Add session to user's session list."""
        user_sessions_key = f"user_sessions:{user_id}"
        await client.sadd(user_sessions_key, session_id)
        await client.expire(user_sessions_key, self.config.session_ttl * 2)


class RedisMonitor:
    """Monitors Redis performance and provides metrics."""
    
    def __init__(self, connection_manager: RedisConnectionManager):
        self.connection_manager = connection_manager
    
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            async with self.connection_manager.get_client() as client:
                info = await client.info()
                
                return {
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                    "uptime_in_seconds": info.get("uptime_in_seconds")
                }
                
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    async def get_key_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored keys."""
        try:
            async with self.connection_manager.get_client() as client:
                # Count keys by type
                key_counts = {
                    "sessions": 0,
                    "states": 0,
                    "checkpoints": 0,
                    "total": 0
                }
                
                # Use SCAN to iterate through keys
                async for key in client.scan_iter(match="*", count=100):
                    key_counts["total"] += 1
                    
                    if key.startswith("session:"):
                        key_counts["sessions"] += 1
                    elif key.startswith("state:"):
                        key_counts["states"] += 1
                    elif key.startswith("checkpoint:"):
                        key_counts["checkpoints"] += 1
                
                return key_counts
                
        except Exception as e:
            logger.error(f"Failed to get key statistics: {e}")
            return {}
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired keys that Redis hasn't removed yet."""
        try:
            cleaned_count = 0
            
            async with self.connection_manager.get_client() as client:
                # Check for expired sessions, states, checkpoints
                for pattern in ["session:*", "state:*", "checkpoint:*"]:
                    async for key in client.scan_iter(match=pattern, count=50):
                        ttl = await client.ttl(key)
                        if ttl == -2:  # Key doesn't exist (expired)
                            cleaned_count += 1
                        elif ttl == 0:  # Key exists but no expiry set
                            # Set default expiry
                            await client.expire(key, 3600)
            
            logger.info(f"Cleaned up {cleaned_count} expired keys")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return 0


# Global instances
redis_config = RedisConfig(url=settings.redis_url)
redis_connection_manager = RedisConnectionManager(redis_config)
redis_state_manager = RedisStateManager(redis_connection_manager)
redis_checkpoint_store = RedisCheckpointStore(redis_connection_manager)
redis_session_manager = RedisSessionManager(redis_connection_manager)
redis_monitor = RedisMonitor(redis_connection_manager)


async def initialize_redis():
    """Initialize Redis connections and components."""
    try:
        await redis_connection_manager.initialize()
        logger.info("Redis integration initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis integration: {e}")
        raise


async def shutdown_redis():
    """Shutdown Redis connections cleanly."""
    try:
        await redis_connection_manager.close()
        logger.info("Redis integration shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during Redis shutdown: {e}")