"""
Valkey Integration for Caching, Session Management, and LangGraph Checkpointing.

This module provides a centralized interface for interacting with Valkey,
the open-source fork of Redis. It offers managers for handling different
aspects of caching and data storage required by the application.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import valkey.asyncio as valkey
from valkey.asyncio.connection import ConnectionPool
from langgraph.checkpoint.redis import RedisSaver

from .config import settings

logger = logging.getLogger(__name__)

class ValkeyConnectionManager:
    """Manages the connection pool to the Valkey server."""
    _pool: Optional[ConnectionPool] = None

    @classmethod
    async def get_pool(cls) -> ConnectionPool:
        """Get the Valkey connection pool, creating it if it doesn't exist."""
        if cls._pool is None:
            try:
                logger.info(f"Creating Valkey connection pool for URL: {settings.valkey_url}")
                cls._pool = ConnectionPool.from_url(
                    settings.valkey_url,
                    max_connections=50,
                    socket_connect_timeout=5,
                    decode_responses=True
                )
            except Exception as e:
                logger.error(f"Failed to create Valkey connection pool: {e}")
                raise
        return cls._pool

    @classmethod
    async def get_client(cls) -> valkey.Valkey:
        """Get a Valkey client from the connection pool."""
        pool = await cls.get_pool()
        return valkey.Valkey(connection_pool=pool)

    @classmethod
    async def close_pool(cls):
        """Close the Valkey connection pool."""
        if cls._pool:
            await cls._pool.disconnect()
            cls._pool = None
            logger.info("Valkey connection pool closed.")

class ValkeyCheckpointStore:
    """A LangGraph checkpoint store using Valkey as the backend."""
    def __init__(self, connection_manager: ValkeyConnectionManager):
        self._connection_manager = connection_manager
        self._saver: Optional[RedisSaver] = None

    async def get_saver(self) -> RedisSaver:
        """Get the LangGraph RedisSaver instance, configured for Valkey."""
        if self._saver is None:
            client = await self._connection_manager.get_client()
            self._saver = RedisSaver(client)
        return self._saver

    async def list_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a given session."""
        saver = await self.get_saver()
        checkpoints = []
        async for checkpoint in saver.list({"configurable": {"thread_id": session_id}}):
            checkpoints.append({
                "session_id": session_id,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "timestamp": checkpoint["ts"],
            })
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints

    async def load_checkpoint(self, session_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Loads a specific checkpoint state."""
        config = {"configurable": {"thread_id": session_id, "checkpoint_id": checkpoint_id}}
        saver = await self.get_saver()
        return await saver.get(config)

class ValkeySessionManager:
    """Manages user sessions in Valkey."""
    def __init__(self, connection_manager: ValkeyConnectionManager):
        self._connection_manager = connection_manager
        self.prefix = "session:"
        self.user_session_prefix = "user_sessions:"

    async def create_session(self, session_id: str, user_id: str, session_type: str, metadata: Dict[str, Any]) -> bool:
        """Create a new session record in Valkey."""
        client = await self._connection_manager.get_client()
        session_key = f"{self.prefix}{session_id}"
        session_data = {
            "user_id": user_id,
            "session_type": session_type,
            "metadata": json.dumps(metadata),
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
        }
        # Use a transaction to ensure atomicity
        async with client.pipeline(transaction=True) as pipe:
            pipe.hmset(session_key, session_data)
            pipe.sadd(f"{self.user_session_prefix}{user_id}", session_id)
            await pipe.execute()
        return True

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from Valkey."""
        client = await self._connection_manager.get_client()
        session_data = await client.hgetall(f"{self.prefix}{session_id}")
        if session_data and "metadata" in session_data:
            session_data["metadata"] = json.loads(session_data["metadata"])
        return session_data or None

    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing session in Valkey."""
        client = await self._connection_manager.get_client()
        session_key = f"{self.prefix}{session_id}"
        if "metadata" in updates:
            updates["metadata"] = json.dumps(updates["metadata"])
        updates["last_activity"] = datetime.now().isoformat()
        await client.hmset(session_key, updates)
        return True

    async def close_session(self, session_id: str) -> bool:
        """Mark a session as closed."""
        return await self.update_session(session_id, {"status": "closed"})

    async def list_user_sessions(self, user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions for a user, optionally filtering by status."""
        client = await self._connection_manager.get_client()
        session_ids = await client.smembers(f"{self.user_session_prefix}{user_id}")
        sessions = []
        for session_id in session_ids:
            session_data = await self.get_session(session_id)
            if session_data:
                if status is None or session_data.get("status") == status:
                    sessions.append(session_data)
        sessions.sort(key=lambda x: x.get("last_activity", ""), reverse=True)
        return sessions

class ValkeyMonitor:
    """Provides monitoring and administrative functions for Valkey."""
    def __init__(self, connection_manager: ValkeyConnectionManager):
        self._connection_manager = connection_manager

    async def get_valkey_info(self) -> Dict[str, Any]:
        """Get server information and stats from Valkey."""
        client = await self._connection_manager.get_client()
        return await client.info()

    async def get_key_statistics(self) -> Dict[str, int]:
        """Get statistics about the keys in the database."""
        client = await self._connection_manager.get_client()
        total_keys = await client.dbsize()
        session_keys = await client.keys(f"{ValkeySessionManager().prefix}*")

        return {
            "total_keys": total_keys,
            "session_keys": len(session_keys),
        }

    async def cleanup_expired_keys(self) -> int:
        """
        This is a conceptual placeholder. Valkey handles TTL automatically.
        This function could be used for more complex cleanup logic if needed.
        """
        logger.info("Valkey handles key expiration automatically via TTLs. No manual cleanup needed.")
        return 0

# Instantiate singleton managers
valkey_connection_manager = ValkeyConnectionManager()
valkey_checkpoint_store = ValkeyCheckpointStore(valkey_connection_manager)
valkey_session_manager = ValkeySessionManager(valkey_connection_manager)
valkey_monitor = ValkeyMonitor(valkey_connection_manager)

# This object is not used by the router, but good to have a placeholder
class ValkeyStateManager:
    """Manages generic state in Valkey (e.g., for application-wide flags)."""
    def __init__(self, connection_manager: ValkeyConnectionManager):
        self._connection_manager = connection_manager

    async def set_state(self, key: str, value: Any):
        client = await self._connection_manager.get_client()
        await client.set(f"state:{key}", json.dumps(value))

    async def get_state(self, key: str) -> Optional[Any]:
        client = await self._connection_manager.get_client()
        value = await client.get(f"state:{key}")
        return json.loads(value) if value else None

valkey_state_manager = ValkeyStateManager(valkey_connection_manager)
