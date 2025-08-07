"""
Advanced Checkpoint Compression and Management System.
Implements LZ4 compression, delta checkpointing, size limits, and automatic cleanup
for optimal storage efficiency and performance in educational applications.
"""

import asyncio
import json
import logging
import lz4.frame
import pickle
import hashlib
import time
import zlib
from typing import Dict, List, Optional, Any, Tuple, Union, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import os
from pathlib import Path
import threading

from .config import settings
from .valkey_integration import valkey_connection_manager
from .memory_manager import memory_manager, BoundedDict

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Available compression methods."""
    NONE = "none"
    LZ4 = "lz4"
    ZLIB = "zlib"
    GZIP = "gzip"


class CheckpointType(Enum):
    """Types of checkpoints."""
    FULL = "full"           # Complete state snapshot
    DELTA = "delta"         # Changes from previous checkpoint
    INCREMENTAL = "incremental"  # Accumulated changes
    COMPRESSED_FULL = "compressed_full"  # Compressed complete snapshot


@dataclass
class CompressionStats:
    """Statistics about compression performance."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: float = 0.0
    
    @property
    def space_saved_percent(self) -> float:
        """Calculate percentage of space saved."""
        if self.original_size == 0:
            return 0.0
        return ((self.original_size - self.compressed_size) / self.original_size) * 100


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoints."""
    checkpoint_id: str
    session_id: str
    checkpoint_type: CheckpointType
    compression_method: CompressionMethod
    created_at: datetime
    expires_at: Optional[datetime]
    
    # Size information
    original_size: int
    compressed_size: int
    compression_stats: CompressionStats
    
    # Delta information
    parent_checkpoint_id: Optional[str] = None
    delta_chain_length: int = 0
    
    # Quality metrics
    fidelity_score: float = 1.0  # How accurate the checkpoint is
    restoration_time_ms: float = 0.0
    
    # Access statistics
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def record_access(self):
        """Record checkpoint access."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    @property
    def age_hours(self) -> float:
        """Get checkpoint age in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    @property
    def is_expired(self) -> bool:
        """Check if checkpoint is expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at


class DeltaCalculator:
    """Calculates and applies deltas between states."""
    
    @staticmethod
    def calculate_delta(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate delta between two states."""
        delta = {
            'added': {},
            'modified': {},
            'removed': set(),
            'metadata': {
                'old_keys': set(old_state.keys()),
                'new_keys': set(new_state.keys()),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        old_keys = set(old_state.keys())
        new_keys = set(new_state.keys())
        
        # Find added keys
        for key in new_keys - old_keys:
            delta['added'][key] = new_state[key]
        
        # Find removed keys
        delta['removed'] = old_keys - new_keys
        
        # Find modified keys
        for key in old_keys & new_keys:
            if old_state[key] != new_state[key]:
                delta['modified'][key] = {
                    'old': old_state[key],
                    'new': new_state[key]
                }
        
        return delta
    
    @staticmethod
    def apply_delta(base_state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
        """Apply delta to a base state."""
        result = base_state.copy()
        
        # Remove keys
        for key in delta.get('removed', set()):
            if key in result:
                del result[key]
        
        # Add new keys
        for key, value in delta.get('added', {}).items():
            result[key] = value
        
        # Modify existing keys
        for key, changes in delta.get('modified', {}).items():
            if key in result:
                result[key] = changes['new']
        
        return result
    
    @staticmethod
    def estimate_delta_size(delta: Dict[str, Any]) -> int:
        """Estimate size of delta in bytes."""
        try:
            return len(json.dumps(delta, separators=(',', ':')).encode())
        except Exception:
            return len(str(delta).encode())


class CompressionEngine:
    """Handles different compression methods."""
    
    @staticmethod
    def compress(
        data: bytes,
        method: CompressionMethod = CompressionMethod.LZ4,
        level: int = 4
    ) -> Tuple[bytes, CompressionStats]:
        """Compress data using specified method."""
        start_time = time.time()
        original_size = len(data)
        
        if method == CompressionMethod.NONE:
            compressed_data = data
        elif method == CompressionMethod.LZ4:
            compressed_data = lz4.frame.compress(
                data,
                compression_level=level,
                auto_flush=True,
                block_size=lz4.frame.BLOCKSIZE_MAX1MB
            )
        elif method == CompressionMethod.ZLIB:
            compressed_data = zlib.compress(data, level)
        elif method == CompressionMethod.GZIP:
            import gzip
            compressed_data = gzip.compress(data, compresslevel=level)
        else:
            raise ValueError(f"Unsupported compression method: {method}")
        
        compression_time_ms = (time.time() - start_time) * 1000
        compressed_size = len(compressed_data)
        
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time_ms
        )
        
        return compressed_data, stats
    
    @staticmethod
    def decompress(
        compressed_data: bytes,
        method: CompressionMethod = CompressionMethod.LZ4
    ) -> Tuple[bytes, float]:
        """Decompress data and return decompression time."""
        start_time = time.time()
        
        if method == CompressionMethod.NONE:
            decompressed_data = compressed_data
        elif method == CompressionMethod.LZ4:
            decompressed_data = lz4.frame.decompress(compressed_data)
        elif method == CompressionMethod.ZLIB:
            decompressed_data = zlib.decompress(compressed_data)
        elif method == CompressionMethod.GZIP:
            import gzip
            decompressed_data = gzip.decompress(compressed_data)
        else:
            raise ValueError(f"Unsupported compression method: {method}")
        
        decompression_time_ms = (time.time() - start_time) * 1000
        
        return decompressed_data, decompression_time_ms
    
    @staticmethod
    def choose_best_compression(
        data: bytes,
        methods: List[CompressionMethod] = None
    ) -> Tuple[CompressionMethod, CompressionStats]:
        """Choose the best compression method for given data."""
        methods = methods or [CompressionMethod.LZ4, CompressionMethod.ZLIB]
        
        best_method = CompressionMethod.NONE
        best_stats = CompressionStats(len(data), len(data), 1.0, 0.0)
        
        for method in methods:
            try:
                _, stats = CompressionEngine.compress(data, method)
                
                # Score based on compression ratio and speed
                speed_bonus = max(0, 100 - stats.compression_time_ms) / 100
                compression_bonus = (1.0 - stats.compression_ratio) * 2
                score = compression_bonus + speed_bonus * 0.3
                
                best_score = (1.0 - best_stats.compression_ratio) * 2
                if score > best_score:
                    best_method = method
                    best_stats = stats
                    
            except Exception as e:
                logger.warning(f"Compression test failed for {method}: {e}")
        
        return best_method, best_stats


class CheckpointManager:
    """Advanced checkpoint management with compression and delta support."""
    
    def __init__(
        self,
        max_checkpoints_per_session: int = 50,
        max_total_size_mb: int = 1000,
        default_compression: CompressionMethod = CompressionMethod.LZ4,
        enable_delta_compression: bool = True
    ):
        self.max_checkpoints_per_session = max_checkpoints_per_session
        self.max_total_size_mb = max_total_size_mb
        self.default_compression = default_compression
        self.enable_delta_compression = enable_delta_compression
        
        # Storage
        self.metadata_cache = BoundedDict(
            "checkpoint_metadata",
            max_size=10000,
            ttl_seconds=86400  # 24 hours
        )
        
        self.compression_engine = CompressionEngine()
        self.delta_calculator = DeltaCalculator()
        
        # Session tracking
        self.session_checkpoints: Dict[str, List[str]] = defaultdict(list)
        self.checkpoint_chains: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_checkpoints': 0,
            'total_size_saved_mb': 0.0,
            'avg_compression_ratio': 0.0,
            'total_delta_checkpoints': 0,
            'avg_delta_chain_length': 0.0
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(f"Checkpoint manager initialized with {default_compression.value} compression")
    
    async def initialize(self):
        """Initialize the checkpoint manager."""
        try:
            # Load existing metadata from storage
            await self._load_metadata_from_storage()
            
            # Start background tasks
            self.is_running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Checkpoint manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint manager: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the checkpoint manager."""
        try:
            self.is_running = False
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Save metadata to storage
            await self._save_metadata_to_storage()
            
            logger.info("Checkpoint manager shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during checkpoint manager shutdown: {e}")
    
    async def create_checkpoint(
        self,
        session_id: str,
        state_data: Dict[str, Any],
        checkpoint_type: CheckpointType = CheckpointType.FULL,
        compression_method: Optional[CompressionMethod] = None,
        expires_in_hours: Optional[float] = None
    ) -> str:
        """Create a new checkpoint."""
        compression_method = compression_method or self.default_compression
        checkpoint_id = self._generate_checkpoint_id(session_id)
        
        try:
            # Determine if we should create a delta checkpoint
            should_use_delta = (
                self.enable_delta_compression and
                checkpoint_type == CheckpointType.FULL and
                len(self.session_checkpoints[session_id]) > 0
            )
            
            if should_use_delta:
                parent_checkpoint_id = self.session_checkpoints[session_id][-1]
                parent_metadata = await self.metadata_cache.get(parent_checkpoint_id)
                
                if parent_metadata and parent_metadata.checkpoint_type in [CheckpointType.FULL, CheckpointType.DELTA]:
                    # Try to create delta checkpoint
                    try:
                        parent_state = await self._load_checkpoint_data(parent_checkpoint_id)
                        delta = self.delta_calculator.calculate_delta(parent_state, state_data)
                        delta_size = self.delta_calculator.estimate_delta_size(delta)
                        
                        # Use delta if it's significantly smaller
                        if delta_size < len(json.dumps(state_data).encode()) * 0.5:
                            return await self._create_delta_checkpoint(
                                checkpoint_id, session_id, delta, parent_checkpoint_id,
                                compression_method, expires_in_hours
                            )
                    except Exception as e:
                        logger.warning(f"Delta checkpoint creation failed, using full: {e}")
            
            # Create full checkpoint
            return await self._create_full_checkpoint(
                checkpoint_id, session_id, state_data, compression_method, expires_in_hours
            )
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    async def _create_full_checkpoint(
        self,
        checkpoint_id: str,
        session_id: str,
        state_data: Dict[str, Any],
        compression_method: CompressionMethod,
        expires_in_hours: Optional[float]
    ) -> str:
        """Create a full checkpoint."""
        # Serialize state data
        serialized_data = json.dumps(state_data, separators=(',', ':')).encode()
        
        # Compress data
        compressed_data, compression_stats = self.compression_engine.compress(
            serialized_data, compression_method
        )
        
        # Calculate expiration
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            checkpoint_type=CheckpointType.FULL,
            compression_method=compression_method,
            created_at=datetime.now(),
            expires_at=expires_at,
            original_size=len(serialized_data),
            compressed_size=len(compressed_data),
            compression_stats=compression_stats
        )
        
        # Store checkpoint data
        await self._store_checkpoint_data(checkpoint_id, compressed_data)
        
        # Store metadata
        await self.metadata_cache.set(checkpoint_id, metadata)
        
        # Update session tracking
        self.session_checkpoints[session_id].append(checkpoint_id)
        
        # Manage session checkpoint limits
        await self._enforce_session_limits(session_id)
        
        # Update statistics
        self._update_stats(metadata)
        
        logger.debug(
            f"Created full checkpoint {checkpoint_id[:8]}... "
            f"({compression_stats.space_saved_percent:.1f}% compression)"
        )
        
        return checkpoint_id
    
    async def _create_delta_checkpoint(
        self,
        checkpoint_id: str,
        session_id: str,
        delta: Dict[str, Any],
        parent_checkpoint_id: str,
        compression_method: CompressionMethod,
        expires_in_hours: Optional[float]
    ) -> str:
        """Create a delta checkpoint."""
        # Serialize delta
        serialized_delta = json.dumps(delta, separators=(',', ':')).encode()
        
        # Compress delta
        compressed_data, compression_stats = self.compression_engine.compress(
            serialized_delta, compression_method
        )
        
        # Get parent metadata for chain length
        parent_metadata = await self.metadata_cache.get(parent_checkpoint_id)
        delta_chain_length = (parent_metadata.delta_chain_length + 1) if parent_metadata else 1
        
        # Calculate expiration
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            checkpoint_type=CheckpointType.DELTA,
            compression_method=compression_method,
            created_at=datetime.now(),
            expires_at=expires_at,
            original_size=len(serialized_delta),
            compressed_size=len(compressed_data),
            compression_stats=compression_stats,
            parent_checkpoint_id=parent_checkpoint_id,
            delta_chain_length=delta_chain_length
        )
        
        # Store checkpoint data
        await self._store_checkpoint_data(checkpoint_id, compressed_data)
        
        # Store metadata
        await self.metadata_cache.set(checkpoint_id, metadata)
        
        # Update session tracking
        self.session_checkpoints[session_id].append(checkpoint_id)
        self.checkpoint_chains[parent_checkpoint_id].append(checkpoint_id)
        
        # Manage session checkpoint limits
        await self._enforce_session_limits(session_id)
        
        # Update statistics
        self._update_stats(metadata)
        self.stats['total_delta_checkpoints'] += 1
        
        logger.debug(
            f"Created delta checkpoint {checkpoint_id[:8]}... "
            f"(chain length: {delta_chain_length}, "
            f"{compression_stats.space_saved_percent:.1f}% compression)"
        )
        
        return checkpoint_id
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        try:
            # Get metadata
            metadata = await self.metadata_cache.get(checkpoint_id)
            if not metadata:
                logger.warning(f"Checkpoint metadata not found: {checkpoint_id}")
                return None
            
            # Check if expired
            if metadata.is_expired:
                logger.info(f"Checkpoint expired: {checkpoint_id}")
                await self._delete_checkpoint(checkpoint_id)
                return None
            
            # Record access
            metadata.record_access()
            
            start_time = time.time()
            
            if metadata.checkpoint_type == CheckpointType.FULL:
                # Load full checkpoint
                state_data = await self._load_full_checkpoint(checkpoint_id, metadata)
            elif metadata.checkpoint_type == CheckpointType.DELTA:
                # Load delta checkpoint (reconstruct from chain)
                state_data = await self._load_delta_checkpoint(checkpoint_id, metadata)
            else:
                raise ValueError(f"Unsupported checkpoint type: {metadata.checkpoint_type}")
            
            # Update restoration time
            metadata.restoration_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                f"Loaded checkpoint {checkpoint_id[:8]}... "
                f"({metadata.restoration_time_ms:.1f}ms)"
            )
            
            return state_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    async def _load_full_checkpoint(
        self,
        checkpoint_id: str,
        metadata: CheckpointMetadata
    ) -> Dict[str, Any]:
        """Load a full checkpoint."""
        # Load compressed data
        compressed_data = await self._load_checkpoint_data(checkpoint_id)
        if not compressed_data:
            raise ValueError(f"Checkpoint data not found: {checkpoint_id}")
        
        # Decompress data
        decompressed_data, decompression_time = self.compression_engine.decompress(
            compressed_data, metadata.compression_method
        )
        
        # Update decompression time in stats
        metadata.compression_stats.decompression_time_ms = decompression_time
        
        # Deserialize state data
        state_data = json.loads(decompressed_data.decode())
        
        return state_data
    
    async def _load_delta_checkpoint(
        self,
        checkpoint_id: str,
        metadata: CheckpointMetadata
    ) -> Dict[str, Any]:
        """Load a delta checkpoint by reconstructing from parent chain."""
        # Build chain from root to target checkpoint
        chain = await self._build_checkpoint_chain(checkpoint_id)
        
        if not chain:
            raise ValueError(f"Could not build checkpoint chain for {checkpoint_id}")
        
        # Load root checkpoint (must be full)
        root_id = chain[0]
        root_metadata = await self.metadata_cache.get(root_id)
        
        if not root_metadata or root_metadata.checkpoint_type != CheckpointType.FULL:
            raise ValueError(f"Invalid root checkpoint: {root_id}")
        
        # Load base state
        state_data = await self._load_full_checkpoint(root_id, root_metadata)
        
        # Apply deltas in sequence
        for delta_id in chain[1:]:
            delta_metadata = await self.metadata_cache.get(delta_id)
            if not delta_metadata:
                raise ValueError(f"Delta checkpoint metadata not found: {delta_id}")
            
            # Load and decompress delta
            compressed_delta = await self._load_checkpoint_data(delta_id)
            if not compressed_delta:
                raise ValueError(f"Delta checkpoint data not found: {delta_id}")
            
            decompressed_delta, _ = self.compression_engine.decompress(
                compressed_delta, delta_metadata.compression_method
            )
            
            delta = json.loads(decompressed_delta.decode())
            
            # Apply delta
            state_data = self.delta_calculator.apply_delta(state_data, delta)
        
        return state_data
    
    async def _build_checkpoint_chain(self, checkpoint_id: str) -> List[str]:
        """Build chain of checkpoints from root to target."""
        chain = []
        current_id = checkpoint_id
        
        # Walk backwards to find root
        while current_id:
            metadata = await self.metadata_cache.get(current_id)
            if not metadata:
                break
            
            chain.append(current_id)
            
            if metadata.checkpoint_type == CheckpointType.FULL:
                # Found root
                break
            
            current_id = metadata.parent_checkpoint_id
            
            # Prevent infinite loops
            if len(chain) > 100:
                logger.error(f"Checkpoint chain too long for {checkpoint_id}")
                return []
        
        # Reverse to get root-to-target order
        return list(reversed(chain))
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint and its dependents."""
        try:
            return await self._delete_checkpoint(checkpoint_id, cascade=True)
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def _delete_checkpoint(self, checkpoint_id: str, cascade: bool = False) -> bool:
        """Internal checkpoint deletion."""
        metadata = await self.metadata_cache.get(checkpoint_id)
        if not metadata:
            return False
        
        # Delete dependent checkpoints if cascading
        if cascade:
            dependents = self.checkpoint_chains.get(checkpoint_id, [])
            for dependent_id in dependents:
                await self._delete_checkpoint(dependent_id, cascade=True)
        
        # Delete checkpoint data
        await self._delete_checkpoint_data(checkpoint_id)
        
        # Remove metadata
        await self.metadata_cache.delete(checkpoint_id)
        
        # Update session tracking
        if checkpoint_id in self.session_checkpoints[metadata.session_id]:
            self.session_checkpoints[metadata.session_id].remove(checkpoint_id)
        
        # Update chain tracking
        if checkpoint_id in self.checkpoint_chains:
            del self.checkpoint_chains[checkpoint_id]
        
        logger.debug(f"Deleted checkpoint {checkpoint_id[:8]}...")
        return True
    
    async def _enforce_session_limits(self, session_id: str):
        """Enforce checkpoint limits per session."""
        checkpoints = self.session_checkpoints[session_id]
        
        while len(checkpoints) > self.max_checkpoints_per_session:
            # Remove oldest checkpoint
            oldest_id = checkpoints.pop(0)
            await self._delete_checkpoint(oldest_id, cascade=True)
    
    async def _store_checkpoint_data(self, checkpoint_id: str, data: bytes):
        """Store checkpoint data in Valkey."""
        key = f"checkpoint_data:{checkpoint_id}"
        
        async with valkey_connection_manager.get_client() as client:
            # Store with 7-day expiration
            await client.setex(key, 604800, data)
    
    async def _load_checkpoint_data(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint data from Valkey."""
        key = f"checkpoint_data:{checkpoint_id}"
        
        async with valkey_connection_manager.get_client() as client:
            return await client.get(key)
    
    async def _delete_checkpoint_data(self, checkpoint_id: str):
        """Delete checkpoint data from Valkey."""
        key = f"checkpoint_data:{checkpoint_id}"
        
        async with valkey_connection_manager.get_client() as client:
            await client.delete(key)
    
    def _generate_checkpoint_id(self, session_id: str) -> str:
        """Generate unique checkpoint ID."""
        timestamp = int(time.time() * 1000)
        content = f"{session_id}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _update_stats(self, metadata: CheckpointMetadata):
        """Update global statistics."""
        self.stats['total_checkpoints'] += 1
        
        space_saved_mb = (
            metadata.original_size - metadata.compressed_size
        ) / (1024 * 1024)
        self.stats['total_size_saved_mb'] += space_saved_mb
        
        # Update average compression ratio
        total_checkpoints = self.stats['total_checkpoints']
        current_avg = self.stats['avg_compression_ratio']
        new_ratio = metadata.compression_stats.compression_ratio
        
        self.stats['avg_compression_ratio'] = (
            (current_avg * (total_checkpoints - 1) + new_ratio) / total_checkpoints
        )
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired checkpoints."""
        while self.is_running:
            try:
                await self._cleanup_expired_checkpoints()
                await self._optimize_delta_chains()
                await asyncio.sleep(1800)  # Clean up every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(1800)
    
    async def _cleanup_expired_checkpoints(self):
        """Remove expired checkpoints."""
        expired_count = 0
        
        # Check all cached metadata
        for checkpoint_id in list(self.metadata_cache.keys()):
            metadata = await self.metadata_cache.get(checkpoint_id)
            if metadata and metadata.is_expired:
                await self._delete_checkpoint(checkpoint_id, cascade=True)
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired checkpoints")
    
    async def _optimize_delta_chains(self):
        """Optimize long delta chains by creating full checkpoints."""
        optimized_count = 0
        
        for session_id, checkpoints in self.session_checkpoints.items():
            if not checkpoints:
                continue
            
            # Check latest checkpoint
            latest_id = checkpoints[-1]
            metadata = await self.metadata_cache.get(latest_id)
            
            if (metadata and 
                metadata.checkpoint_type == CheckpointType.DELTA and
                metadata.delta_chain_length > 10):  # Optimize chains longer than 10
                
                try:
                    # Load current state
                    state_data = await self.load_checkpoint(latest_id)
                    if state_data:
                        # Create new full checkpoint
                        await self._create_full_checkpoint(
                            self._generate_checkpoint_id(session_id),
                            session_id,
                            state_data,
                            self.default_compression,
                            24.0  # 24 hour expiration
                        )
                        
                        optimized_count += 1
                        logger.debug(f"Optimized delta chain for session {session_id}")
                        
                except Exception as e:
                    logger.warning(f"Failed to optimize delta chain: {e}")
        
        if optimized_count > 0:
            logger.info(f"Optimized {optimized_count} delta chains")
    
    async def _save_metadata_to_storage(self):
        """Save metadata cache to persistent storage."""
        try:
            metadata_dict = {}
            
            for checkpoint_id in self.metadata_cache.keys():
                metadata = await self.metadata_cache.get(checkpoint_id)
                if metadata:
                    metadata_dict[checkpoint_id] = {
                        'checkpoint_id': metadata.checkpoint_id,
                        'session_id': metadata.session_id,
                        'checkpoint_type': metadata.checkpoint_type.value,
                        'compression_method': metadata.compression_method.value,
                        'created_at': metadata.created_at.isoformat(),
                        'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                        'original_size': metadata.original_size,
                        'compressed_size': metadata.compressed_size,
                        'parent_checkpoint_id': metadata.parent_checkpoint_id,
                        'delta_chain_length': metadata.delta_chain_length
                    }
            
            # Store metadata
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    "checkpoint_metadata_backup",
                    86400,  # 24 hours
                    json.dumps({
                        'metadata': metadata_dict,
                        'session_checkpoints': dict(self.session_checkpoints),
                        'stats': self.stats
                    }, separators=(',', ':'))
                )
                
            logger.debug(f"Saved {len(metadata_dict)} checkpoint metadata entries")
            
        except Exception as e:
            logger.warning(f"Failed to save metadata to storage: {e}")
    
    async def _load_metadata_from_storage(self):
        """Load metadata cache from persistent storage."""
        try:
            async with valkey_connection_manager.get_client() as client:
                data = await client.get("checkpoint_metadata_backup")
                
                if not data:
                    return
                
                backup_data = json.loads(data)
                
                # Restore metadata
                for checkpoint_id, meta_dict in backup_data.get('metadata', {}).items():
                    metadata = CheckpointMetadata(
                        checkpoint_id=meta_dict['checkpoint_id'],
                        session_id=meta_dict['session_id'],
                        checkpoint_type=CheckpointType(meta_dict['checkpoint_type']),
                        compression_method=CompressionMethod(meta_dict['compression_method']),
                        created_at=datetime.fromisoformat(meta_dict['created_at']),
                        expires_at=datetime.fromisoformat(meta_dict['expires_at']) if meta_dict['expires_at'] else None,
                        original_size=meta_dict['original_size'],
                        compressed_size=meta_dict['compressed_size'],
                        compression_stats=CompressionStats(
                            meta_dict['original_size'],
                            meta_dict['compressed_size'],
                            meta_dict['compressed_size'] / meta_dict['original_size'],
                            0.0
                        ),
                        parent_checkpoint_id=meta_dict.get('parent_checkpoint_id'),
                        delta_chain_length=meta_dict.get('delta_chain_length', 0)
                    )
                    
                    # Only restore non-expired metadata
                    if not metadata.is_expired:
                        await self.metadata_cache.set(checkpoint_id, metadata)
                
                # Restore session tracking
                self.session_checkpoints = defaultdict(list, backup_data.get('session_checkpoints', {}))
                
                # Restore statistics
                self.stats.update(backup_data.get('stats', {}))
                
                logger.info(f"Loaded {len(self.metadata_cache)} checkpoint metadata entries")
                
        except Exception as e:
            logger.warning(f"Failed to load metadata from storage: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint statistics."""
        return {
            'total_checkpoints': self.stats['total_checkpoints'],
            'total_size_saved_mb': round(self.stats['total_size_saved_mb'], 2),
            'avg_compression_ratio': round(self.stats['avg_compression_ratio'], 3),
            'total_delta_checkpoints': self.stats['total_delta_checkpoints'],
            'active_sessions': len(self.session_checkpoints),
            'total_cached_metadata': len(self.metadata_cache),
            'compression_method': self.default_compression.value,
            'delta_compression_enabled': self.enable_delta_compression,
            'max_checkpoints_per_session': self.max_checkpoints_per_session
        }
    
    async def get_session_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all checkpoints for a session."""
        checkpoints = []
        
        for checkpoint_id in self.session_checkpoints.get(session_id, []):
            metadata = await self.metadata_cache.get(checkpoint_id)
            if metadata:
                checkpoints.append({
                    'checkpoint_id': checkpoint_id,
                    'type': metadata.checkpoint_type.value,
                    'created_at': metadata.created_at.isoformat(),
                    'size_mb': metadata.compressed_size / (1024 * 1024),
                    'compression_ratio': metadata.compression_stats.compression_ratio,
                    'access_count': metadata.access_count,
                    'delta_chain_length': metadata.delta_chain_length
                })
        
        return checkpoints


# Global checkpoint manager instance
checkpoint_manager = CheckpointManager()


async def initialize_checkpoint_manager():
    """Initialize the checkpoint manager."""
    await checkpoint_manager.initialize()


async def shutdown_checkpoint_manager():
    """Shutdown the checkpoint manager."""
    await checkpoint_manager.shutdown()


# Convenience functions
async def create_checkpoint(
    session_id: str,
    state_data: Dict[str, Any],
    compression: CompressionMethod = CompressionMethod.LZ4,
    expires_in_hours: float = 24.0
) -> str:
    """Create a checkpoint with optimal compression."""
    return await checkpoint_manager.create_checkpoint(
        session_id, state_data, CheckpointType.FULL, compression, expires_in_hours
    )


async def load_checkpoint(checkpoint_id: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint data."""
    return await checkpoint_manager.load_checkpoint(checkpoint_id)


async def delete_checkpoint(checkpoint_id: str) -> bool:
    """Delete a checkpoint."""
    return await checkpoint_manager.delete_checkpoint(checkpoint_id)


def get_checkpoint_stats() -> Dict[str, Any]:
    """Get checkpoint statistics."""
    return checkpoint_manager.get_statistics()