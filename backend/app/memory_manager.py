"""
Advanced Memory Management System.
Fixes unbounded collections, implements cleanup tasks, monitors memory usage,
and provides automatic garbage collection for optimal resource utilization.
"""

import asyncio
import gc
import logging
import psutil
import resource
import sys
import threading
import time
import weakref
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CollectionType(Enum):
    """Types of managed collections."""
    LRU_CACHE = "lru_cache"
    TIME_BASED_CACHE = "time_based_cache"
    DEQUE = "deque"
    DICT = "dict"
    LIST = "list"
    SET = "set"
    CUSTOM = "custom"


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    free_mb: float
    percent_used: float
    swap_total_mb: float
    swap_used_mb: float
    swap_percent: float
    
    # Process-specific stats
    process_rss_mb: float
    process_vms_mb: float
    process_percent: float
    
    # Garbage collection stats
    gc_collections: Dict[int, int]
    gc_objects: int
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CollectionStats:
    """Statistics for managed collections."""
    name: str
    collection_type: CollectionType
    current_size: int
    max_size: Optional[int]
    memory_estimate_mb: float
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    creation_time: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.creation_time).total_seconds()


class BoundedCollection:
    """Base class for bounded collections with automatic cleanup."""
    
    def __init__(
        self,
        name: str,
        max_size: int,
        collection_type: CollectionType,
        ttl_seconds: Optional[int] = None,
        cleanup_callback: Optional[Callable] = None
    ):
        self.name = name
        self.max_size = max_size
        self.collection_type = collection_type
        self.ttl_seconds = ttl_seconds
        self.cleanup_callback = cleanup_callback
        
        self.stats = CollectionStats(name, collection_type, 0, max_size, 0.0)
        self._lock = asyncio.Lock()
        
        # Register with memory manager
        memory_manager.register_collection(self)
    
    def _record_hit(self):
        """Record cache hit."""
        self.stats.hit_count += 1
        self.stats.last_access = datetime.now()
    
    def _record_miss(self):
        """Record cache miss."""
        self.stats.miss_count += 1
        self.stats.last_access = datetime.now()
    
    def _record_eviction(self):
        """Record eviction."""
        self.stats.eviction_count += 1
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Override in subclasses for more accurate estimation
        return self.stats.current_size * 0.001  # 1KB per item estimate
    
    def update_stats(self):
        """Update collection statistics."""
        self.stats.memory_estimate_mb = self._estimate_memory_usage()
    
    async def cleanup_expired(self):
        """Remove expired items (if TTL is set)."""
        if not self.ttl_seconds:
            return
        
        # Override in subclasses
        pass
    
    async def force_cleanup(self, target_size: Optional[int] = None):
        """Force cleanup to target size."""
        # Override in subclasses
        pass


class BoundedDict(BoundedCollection):
    """Dictionary with size limits and TTL support."""
    
    def __init__(
        self,
        name: str,
        max_size: int = 10000,
        ttl_seconds: Optional[int] = None,
        cleanup_callback: Optional[Callable] = None
    ):
        super().__init__(name, max_size, CollectionType.DICT, ttl_seconds, cleanup_callback)
        self._data: Dict[Any, Any] = {}
        self._access_times: Dict[Any, datetime] = {}
        self._insert_order: deque = deque()
    
    async def get(self, key: Any, default: Any = None) -> Any:
        """Get item with hit/miss tracking."""
        async with self._lock:
            if key in self._data:
                self._record_hit()
                self._access_times[key] = datetime.now()
                return self._data[key]
            else:
                self._record_miss()
                return default
    
    async def set(self, key: Any, value: Any):
        """Set item with size management."""
        async with self._lock:
            # Remove existing key if present
            if key in self._data:
                self._insert_order.remove(key)
            
            # Add new item
            self._data[key] = value
            self._access_times[key] = datetime.now()
            self._insert_order.append(key)
            
            # Enforce size limit
            while len(self._data) > self.max_size:
                await self._evict_lru()
            
            self.stats.current_size = len(self._data)
            self.update_stats()
    
    async def delete(self, key: Any) -> bool:
        """Delete item."""
        async with self._lock:
            if key in self._data:
                del self._data[key]
                del self._access_times[key]
                self._insert_order.remove(key)
                self.stats.current_size = len(self._data)
                self.update_stats()
                return True
            return False
    
    async def _evict_lru(self):
        """Evict least recently used item."""
        if not self._insert_order:
            return
        
        # Find LRU item
        lru_key = None
        lru_time = datetime.now()
        
        for key in self._insert_order:
            access_time = self._access_times.get(key, datetime.min)
            if access_time < lru_time:
                lru_time = access_time
                lru_key = key
        
        if lru_key:
            if self.cleanup_callback:
                try:
                    await self.cleanup_callback(lru_key, self._data[lru_key])
                except Exception as e:
                    logger.warning(f"Cleanup callback failed for {self.name}: {e}")
            
            del self._data[lru_key]
            del self._access_times[lru_key]
            self._insert_order.remove(lru_key)
            self._record_eviction()
    
    async def cleanup_expired(self):
        """Remove expired items."""
        if not self.ttl_seconds:
            return
        
        async with self._lock:
            now = datetime.now()
            expired_keys = []
            
            for key, access_time in self._access_times.items():
                if (now - access_time).total_seconds() > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if self.cleanup_callback:
                    try:
                        await self.cleanup_callback(key, self._data[key])
                    except Exception:
                        pass
                
                del self._data[key]
                del self._access_times[key]
                self._insert_order.remove(key)
            
            if expired_keys:
                self.stats.current_size = len(self._data)
                self.update_stats()
                logger.debug(f"Cleaned up {len(expired_keys)} expired items from {self.name}")
    
    async def force_cleanup(self, target_size: Optional[int] = None):
        """Force cleanup to target size."""
        target_size = target_size or self.max_size // 2
        
        async with self._lock:
            while len(self._data) > target_size:
                await self._evict_lru()
            
            self.stats.current_size = len(self._data)
            self.update_stats()
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage."""
        if not self._data:
            return 0.0
        
        # Sample a few items to estimate size
        sample_size = min(10, len(self._data))
        sample_items = list(self._data.items())[:sample_size]
        
        total_size = 0
        for key, value in sample_items:
            try:
                key_size = sys.getsizeof(key)
                value_size = sys.getsizeof(value)
                total_size += key_size + value_size
            except Exception:
                total_size += 1024  # 1KB estimate
        
        avg_item_size = total_size / sample_size
        estimated_total = avg_item_size * len(self._data)
        
        return estimated_total / (1024 * 1024)  # Convert to MB
    
    def __len__(self) -> int:
        return len(self._data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


class BoundedDeque(BoundedCollection):
    """Deque with size limits."""
    
    def __init__(
        self,
        name: str,
        max_size: int = 10000,
        ttl_seconds: Optional[int] = None
    ):
        super().__init__(name, max_size, CollectionType.DEQUE, ttl_seconds)
        self._data: deque = deque(maxlen=max_size)
        self._timestamps: deque = deque(maxlen=max_size)
    
    async def append(self, item: Any):
        """Add item to right side."""
        async with self._lock:
            self._data.append(item)
            self._timestamps.append(datetime.now())
            self.stats.current_size = len(self._data)
            self.update_stats()
    
    async def appendleft(self, item: Any):
        """Add item to left side."""
        async with self._lock:
            self._data.appendleft(item)
            self._timestamps.appendleft(datetime.now())
            self.stats.current_size = len(self._data)
            self.update_stats()
    
    async def pop(self) -> Any:
        """Remove and return rightmost item."""
        async with self._lock:
            if self._data:
                self._timestamps.pop()
                self.stats.current_size = len(self._data)
                self.update_stats()
                return self._data.pop()
            return None
    
    async def popleft(self) -> Any:
        """Remove and return leftmost item."""
        async with self._lock:
            if self._data:
                self._timestamps.popleft()
                self.stats.current_size = len(self._data)
                self.update_stats()
                return self._data.popleft()
            return None
    
    async def cleanup_expired(self):
        """Remove expired items."""
        if not self.ttl_seconds:
            return
        
        async with self._lock:
            now = datetime.now()
            expired_count = 0
            
            # Remove expired items from left (oldest)
            while (self._timestamps and 
                   (now - self._timestamps[0]).total_seconds() > self.ttl_seconds):
                self._data.popleft()
                self._timestamps.popleft()
                expired_count += 1
            
            if expired_count > 0:
                self.stats.current_size = len(self._data)
                self.update_stats()
                logger.debug(f"Cleaned up {expired_count} expired items from {self.name}")
    
    async def force_cleanup(self, target_size: Optional[int] = None):
        """Force cleanup to target size."""
        target_size = target_size or self.max_size // 2
        
        async with self._lock:
            while len(self._data) > target_size:
                self._data.popleft()
                self._timestamps.popleft()
            
            self.stats.current_size = len(self._data)
            self.update_stats()
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)


class MemoryManager:
    """Central memory management system."""
    
    def __init__(self):
        self.collections: Dict[str, BoundedCollection] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Memory thresholds (percentage)
        self.memory_thresholds = {
            MemoryPressureLevel.LOW: 50,
            MemoryPressureLevel.MEDIUM: 70,
            MemoryPressureLevel.HIGH: 85,
            MemoryPressureLevel.CRITICAL: 95
        }
        
        # Current memory pressure
        self.current_pressure = MemoryPressureLevel.LOW
        
        # Statistics
        self.stats_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.gc_stats: Dict[str, Any] = {}
        
        # Callbacks for memory pressure
        self.pressure_callbacks: Dict[MemoryPressureLevel, List[Callable]] = defaultdict(list)
        
        # Process reference for monitoring
        try:
            self.process = psutil.Process()
        except Exception:
            self.process = None
            logger.warning("Could not initialize process monitoring")
    
    async def initialize(self):
        """Initialize memory management."""
        try:
            # Set up garbage collection
            self._configure_garbage_collection()
            
            # Start monitoring tasks
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Memory manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown memory management."""
        try:
            self.is_monitoring = False
            
            # Cancel tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Wait for tasks to complete
            for task in [self.monitoring_task, self.cleanup_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Force cleanup of all collections
            await self._force_cleanup_all()
            
            logger.info("Memory manager shut down")
            
        except Exception as e:
            logger.error(f"Error during memory manager shutdown: {e}")
    
    def _configure_garbage_collection(self):
        """Configure garbage collection for optimal performance."""
        # Enable garbage collection debugging in development
        if __debug__:
            gc.set_debug(gc.DEBUG_STATS)
        
        # Tune garbage collection thresholds
        # (threshold0, threshold1, threshold2)
        gc.set_threshold(1000, 10, 10)  # More aggressive collection
        
        # Force initial collection
        collected = gc.collect()
        logger.info(f"Initial garbage collection freed {collected} objects")
    
    def register_collection(self, collection: BoundedCollection):
        """Register a collection for management."""
        self.collections[collection.name] = collection
        logger.debug(f"Registered collection: {collection.name}")
    
    def unregister_collection(self, name: str):
        """Unregister a collection."""
        if name in self.collections:
            del self.collections[name]
            logger.debug(f"Unregistered collection: {name}")
    
    def register_pressure_callback(
        self,
        level: MemoryPressureLevel,
        callback: Callable
    ):
        """Register callback for memory pressure level."""
        self.pressure_callbacks[level].append(callback)
    
    async def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process memory
            process_memory = None
            if self.process:
                process_memory = self.process.memory_info()
            
            # Garbage collection stats
            gc_stats = {}
            for generation in range(3):
                gc_stats[generation] = gc.get_count()[generation]
            
            return MemoryStats(
                total_mb=memory.total / (1024 * 1024),
                available_mb=memory.available / (1024 * 1024),
                used_mb=memory.used / (1024 * 1024),
                free_mb=memory.free / (1024 * 1024),
                percent_used=memory.percent,
                swap_total_mb=swap.total / (1024 * 1024),
                swap_used_mb=swap.used / (1024 * 1024),
                swap_percent=swap.percent,
                process_rss_mb=process_memory.rss / (1024 * 1024) if process_memory else 0.0,
                process_vms_mb=process_memory.vms / (1024 * 1024) if process_memory else 0.0,
                process_percent=self.process.memory_percent() if self.process else 0.0,
                gc_collections=gc_stats,
                gc_objects=len(gc.get_objects())
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, 0)
    
    def _determine_memory_pressure(self, stats: MemoryStats) -> MemoryPressureLevel:
        """Determine current memory pressure level."""
        # Use process memory percentage if available, otherwise system memory
        memory_percent = stats.process_percent if stats.process_percent > 0 else stats.percent_used
        
        if memory_percent >= self.memory_thresholds[MemoryPressureLevel.CRITICAL]:
            return MemoryPressureLevel.CRITICAL
        elif memory_percent >= self.memory_thresholds[MemoryPressureLevel.HIGH]:
            return MemoryPressureLevel.HIGH
        elif memory_percent >= self.memory_thresholds[MemoryPressureLevel.MEDIUM]:
            return MemoryPressureLevel.MEDIUM
        else:
            return MemoryPressureLevel.LOW
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get memory statistics
                stats = await self.get_memory_stats()
                self.stats_history.append(stats)
                
                # Determine memory pressure
                new_pressure = self._determine_memory_pressure(stats)
                
                # Handle pressure level changes
                if new_pressure != self.current_pressure:
                    await self._handle_pressure_change(self.current_pressure, new_pressure)
                    self.current_pressure = new_pressure
                
                # Update collection statistics
                for collection in self.collections.values():
                    collection.update_stats()
                
                # Log memory status periodically
                if len(self.stats_history) % 60 == 0:  # Every hour
                    logger.info(
                        f"Memory: {stats.process_percent:.1f}% process, "
                        f"{stats.percent_used:.1f}% system, "
                        f"pressure: {self.current_pressure.value}"
                    )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Periodic cleanup loop."""
        while self.is_monitoring:
            try:
                # Clean up expired items in all collections
                for collection in self.collections.values():
                    try:
                        await collection.cleanup_expired()
                    except Exception as e:
                        logger.warning(f"Cleanup failed for {collection.name}: {e}")
                
                # Force garbage collection
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collection freed {collected} objects")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _handle_pressure_change(
        self,
        old_level: MemoryPressureLevel,
        new_level: MemoryPressureLevel
    ):
        """Handle memory pressure level changes."""
        logger.info(f"Memory pressure changed: {old_level.value} -> {new_level.value}")
        
        # Execute pressure-specific actions
        if new_level == MemoryPressureLevel.CRITICAL:
            await self._handle_critical_pressure()
        elif new_level == MemoryPressureLevel.HIGH:
            await self._handle_high_pressure()
        elif new_level == MemoryPressureLevel.MEDIUM:
            await self._handle_medium_pressure()
        
        # Execute registered callbacks
        for callback in self.pressure_callbacks[new_level]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(new_level)
                else:
                    callback(new_level)
            except Exception as e:
                logger.error(f"Pressure callback failed: {e}")
    
    async def _handle_critical_pressure(self):
        """Handle critical memory pressure."""
        logger.warning("Critical memory pressure - aggressive cleanup")
        
        # Force cleanup all collections to 25% capacity
        for collection in self.collections.values():
            try:
                target_size = collection.max_size // 4
                await collection.force_cleanup(target_size)
            except Exception as e:
                logger.error(f"Critical cleanup failed for {collection.name}: {e}")
        
        # Force garbage collection
        for generation in range(3):
            gc.collect(generation)
        
        # Log memory usage after cleanup
        stats = await self.get_memory_stats()
        logger.warning(
            f"After critical cleanup: {stats.process_percent:.1f}% process memory"
        )
    
    async def _handle_high_pressure(self):
        """Handle high memory pressure."""
        logger.warning("High memory pressure - moderate cleanup")
        
        # Force cleanup all collections to 50% capacity
        for collection in self.collections.values():
            try:
                target_size = collection.max_size // 2
                await collection.force_cleanup(target_size)
            except Exception as e:
                logger.error(f"High pressure cleanup failed for {collection.name}: {e}")
        
        # Force garbage collection
        gc.collect()
    
    async def _handle_medium_pressure(self):
        """Handle medium memory pressure."""
        logger.info("Medium memory pressure - light cleanup")
        
        # Clean up expired items more aggressively
        for collection in self.collections.values():
            try:
                await collection.cleanup_expired()
            except Exception as e:
                logger.warning(f"Medium pressure cleanup failed for {collection.name}: {e}")
    
    async def _force_cleanup_all(self):
        """Force cleanup of all collections."""
        for collection in self.collections.values():
            try:
                await collection.force_cleanup(0)  # Clear all
            except Exception as e:
                logger.error(f"Force cleanup failed for {collection.name}: {e}")
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all managed collections."""
        stats = {}
        
        for name, collection in self.collections.items():
            stats[name] = {
                'type': collection.collection_type.value,
                'current_size': collection.stats.current_size,
                'max_size': collection.stats.max_size,
                'memory_mb': collection.stats.memory_estimate_mb,
                'hit_rate': collection.stats.hit_rate,
                'eviction_count': collection.stats.eviction_count,
                'age_seconds': collection.stats.age_seconds
            }
        
        return stats
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary memory management statistics."""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-1]
        
        # Collection summary
        total_collections = len(self.collections)
        total_items = sum(c.stats.current_size for c in self.collections.values())
        total_memory_mb = sum(c.stats.memory_estimate_mb for c in self.collections.values())
        
        return {
            'memory_pressure': self.current_pressure.value,
            'process_memory_percent': recent_stats.process_percent,
            'system_memory_percent': recent_stats.percent_used,
            'process_memory_mb': recent_stats.process_rss_mb,
            'total_collections': total_collections,
            'total_managed_items': total_items,
            'total_managed_memory_mb': total_memory_mb,
            'gc_objects': recent_stats.gc_objects,
            'monitoring_active': self.is_monitoring
        }


# Global memory manager instance
memory_manager = MemoryManager()


async def initialize_memory_manager():
    """Initialize the memory manager."""
    await memory_manager.initialize()


async def shutdown_memory_manager():
    """Shutdown the memory manager."""
    await memory_manager.shutdown()


def create_bounded_dict(
    name: str,
    max_size: int = 10000,
    ttl_seconds: Optional[int] = None,
    cleanup_callback: Optional[Callable] = None
) -> BoundedDict:
    """Create a managed bounded dictionary."""
    return BoundedDict(name, max_size, ttl_seconds, cleanup_callback)


def create_bounded_deque(
    name: str,
    max_size: int = 10000,
    ttl_seconds: Optional[int] = None
) -> BoundedDeque:
    """Create a managed bounded deque."""
    return BoundedDeque(name, max_size, ttl_seconds)


def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics."""
    return memory_manager.get_summary_stats()


def register_pressure_callback(level: MemoryPressureLevel, callback: Callable):
    """Register callback for memory pressure."""
    memory_manager.register_pressure_callback(level, callback)


def force_garbage_collection() -> int:
    """Force garbage collection and return number of collected objects."""
    return gc.collect()


# Decorators for memory management
def memory_limit(max_mb: float):
    """Decorator to enforce memory limits on functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get initial memory
            initial_stats = await memory_manager.get_memory_stats()
            
            try:
                result = await func(*args, **kwargs)
                
                # Check memory usage after function
                final_stats = await memory_manager.get_memory_stats()
                memory_used = final_stats.process_rss_mb - initial_stats.process_rss_mb
                
                if memory_used > max_mb:
                    logger.warning(
                        f"Function {func.__name__} used {memory_used:.1f}MB "
                        f"(limit: {max_mb}MB)"
                    )
                
                return result
                
            except Exception as e:
                # Force cleanup on error
                force_garbage_collection()
                raise
        
        return wrapper
    return decorator


# Context manager for temporary memory monitoring
class memory_monitor:
    """Context manager for monitoring memory usage in a code block."""
    
    def __init__(self, name: str, log_usage: bool = True):
        self.name = name
        self.log_usage = log_usage
        self.initial_stats = None
        self.final_stats = None
    
    async def __aenter__(self):
        self.initial_stats = await memory_manager.get_memory_stats()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.final_stats = await memory_manager.get_memory_stats()
        
        if self.log_usage and self.initial_stats and self.final_stats:
            memory_used = self.final_stats.process_rss_mb - self.initial_stats.process_rss_mb
            
            if memory_used > 0:
                logger.info(f"Memory usage for {self.name}: +{memory_used:.1f}MB")
            elif memory_used < 0:
                logger.info(f"Memory usage for {self.name}: {memory_used:.1f}MB (freed)")
    
    @property
    def memory_used_mb(self) -> float:
        """Get memory used during monitored period."""
        if self.initial_stats and self.final_stats:
            return self.final_stats.process_rss_mb - self.initial_stats.process_rss_mb
        return 0.0