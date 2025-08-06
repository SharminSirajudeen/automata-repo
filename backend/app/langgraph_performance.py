"""
Performance Optimization and Monitoring for LangGraph Workflows.
Provides metrics collection, performance analysis, and optimization strategies.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
from statistics import mean, median
import psutil

from .langgraph_core import ConversationState, WorkflowStatus
from .redis_integration import redis_state_manager

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    QUEUE_SIZE = "queue_size"
    CHECKPOINT_SIZE = "checkpoint_size"
    CACHE_HIT_RATE = "cache_hit_rate"


class OptimizationStrategy(str, Enum):
    """Performance optimization strategies."""
    CACHING = "caching"
    PARALLELIZATION = "parallelization"
    RESOURCE_POOLING = "resource_pooling"
    BATCH_PROCESSING = "batch_processing"
    LAZY_LOADING = "lazy_loading"
    COMPRESSION = "compression"
    PREFETCHING = "prefetching"
    LOAD_BALANCING = "load_balancing"


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    session_id: Optional[str] = None
    node_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile for a workflow or node."""
    name: str
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    avg_memory_usage: float
    error_rate: float
    total_executions: int
    last_updated: datetime
    trends: Dict[str, List[float]] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation based on performance analysis."""
    strategy: OptimizationStrategy
    description: str
    expected_improvement: str
    implementation_effort: str  # low, medium, high
    priority: str  # low, medium, high, critical
    parameters: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
        self.session_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.node_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
    
    def collect_metric(self, metric: PerformanceMetric):
        """Collect a performance metric."""
        self.metrics.append(metric)
        
        # Aggregate by type
        self.aggregated_metrics[metric.metric_type.value].append(metric.value)
        
        # Aggregate by session
        if metric.session_id:
            self.session_metrics[metric.session_id].append(metric)
        
        # Aggregate by node
        if metric.node_name:
            self.node_metrics[metric.node_name].append(metric)
    
    def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        session_id: Optional[str] = None,
        node_name: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> List[PerformanceMetric]:
        """Get metrics with optional filtering."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        filtered_metrics = []
        
        for metric in self.metrics:
            # Time filter
            if metric.timestamp < cutoff_time:
                continue
            
            # Type filter
            if metric_type and metric.metric_type != metric_type:
                continue
            
            # Session filter
            if session_id and metric.session_id != session_id:
                continue
            
            # Node filter
            if node_name and metric.node_name != node_name:
                continue
            
            filtered_metrics.append(metric)
        
        return filtered_metrics
    
    def get_aggregated_stats(
        self,
        metric_type: MetricType,
        time_window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get aggregated statistics for a metric type."""
        metrics = self.get_metrics(metric_type, time_window_minutes=time_window_minutes)
        values = [m.value for m in metrics]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "avg": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "total": sum(values)
        }


class PerformanceMonitor:
    """Monitors workflow performance and provides analysis."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.resource_usage: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def start_monitoring_session(self, session_id: str, metadata: Dict[str, Any] = None):
        """Start monitoring a workflow session."""
        self.active_sessions[session_id] = {
            "start_time": time.time(),
            "start_memory": psutil.virtual_memory().used,
            "start_cpu": psutil.cpu_percent(),
            "metadata": metadata or {},
            "node_times": {},
            "checkpoints": 0
        }
        
        logger.debug(f"Started monitoring session: {session_id}")
    
    async def stop_monitoring_session(self, session_id: str) -> Dict[str, Any]:
        """Stop monitoring a workflow session and return summary."""
        if session_id not in self.active_sessions:
            return {}
        
        session_data = self.active_sessions.pop(session_id)
        end_time = time.time()
        
        total_time = end_time - session_data["start_time"]
        
        # Collect final metrics
        final_memory = psutil.virtual_memory().used
        memory_delta = final_memory - session_data["start_memory"]
        
        summary = {
            "session_id": session_id,
            "total_execution_time": total_time,
            "memory_delta_bytes": memory_delta,
            "node_execution_times": session_data["node_times"],
            "checkpoints_created": session_data["checkpoints"],
            "metadata": session_data["metadata"]
        }
        
        # Store summary
        await self._store_session_summary(session_id, summary)
        
        return summary
    
    async def record_node_execution(
        self,
        session_id: str,
        node_name: str,
        execution_time: float,
        memory_used: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record node execution metrics."""
        # Collect execution time metric
        metric = PerformanceMetric(
            metric_type=MetricType.EXECUTION_TIME,
            value=execution_time,
            timestamp=datetime.now(),
            session_id=session_id,
            node_name=node_name,
            metadata=metadata or {}
        )
        
        self.metrics_collector.collect_metric(metric)
        
        # Update session tracking
        if session_id in self.active_sessions:
            if node_name not in self.active_sessions[session_id]["node_times"]:
                self.active_sessions[session_id]["node_times"][node_name] = []
            self.active_sessions[session_id]["node_times"][node_name].append(execution_time)
        
        # Record memory usage if provided
        if memory_used is not None:
            memory_metric = PerformanceMetric(
                metric_type=MetricType.MEMORY_USAGE,
                value=memory_used,
                timestamp=datetime.now(),
                session_id=session_id,
                node_name=node_name,
                metadata=metadata or {}
            )
            self.metrics_collector.collect_metric(memory_metric)
    
    async def record_checkpoint_creation(self, session_id: str, checkpoint_size_bytes: int):
        """Record checkpoint creation metrics."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["checkpoints"] += 1
        
        # Record checkpoint size metric
        metric = PerformanceMetric(
            metric_type=MetricType.CHECKPOINT_SIZE,
            value=checkpoint_size_bytes,
            timestamp=datetime.now(),
            session_id=session_id,
            metadata={"type": "checkpoint_size"}
        )
        
        self.metrics_collector.collect_metric(metric)
    
    async def get_performance_report(
        self,
        session_id: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "time_window_minutes": time_window_minutes,
            "system_metrics": await self._get_system_metrics(),
            "workflow_metrics": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Get workflow-specific metrics
        if session_id:
            report["workflow_metrics"] = await self._get_session_metrics(session_id, time_window_minutes)
        else:
            report["workflow_metrics"] = await self._get_aggregate_metrics(time_window_minutes)
        
        # Identify bottlenecks
        report["bottlenecks"] = await self._identify_bottlenecks(time_window_minutes)
        
        # Generate recommendations
        report["recommendations"] = await self._generate_recommendations(report)
        
        return report
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100,
                "active_sessions": len(self.active_sessions)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    async def _get_session_metrics(self, session_id: str, time_window_minutes: int) -> Dict[str, Any]:
        """Get metrics for a specific session."""
        metrics = self.metrics_collector.get_metrics(
            session_id=session_id,
            time_window_minutes=time_window_minutes
        )
        
        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type.value].append(metric.value)
        
        # Calculate statistics for each type
        session_stats = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                session_stats[metric_type] = {
                    "count": len(values),
                    "avg": mean(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values)
                }
        
        return session_stats
    
    async def _get_aggregate_metrics(self, time_window_minutes: int) -> Dict[str, Any]:
        """Get aggregate metrics across all sessions."""
        aggregate_stats = {}
        
        for metric_type in MetricType:
            stats = self.metrics_collector.get_aggregated_stats(metric_type, time_window_minutes)
            if stats:
                aggregate_stats[metric_type.value] = stats
        
        return aggregate_stats
    
    async def _identify_bottlenecks(self, time_window_minutes: int) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check for slow nodes
        execution_metrics = self.metrics_collector.get_metrics(
            MetricType.EXECUTION_TIME,
            time_window_minutes=time_window_minutes
        )
        
        # Group by node
        node_times = defaultdict(list)
        for metric in execution_metrics:
            if metric.node_name:
                node_times[metric.node_name].append(metric.value)
        
        # Identify slow nodes (avg > 30 seconds)
        for node_name, times in node_times.items():
            avg_time = mean(times)
            if avg_time > 30:
                bottlenecks.append({
                    "type": "slow_node",
                    "node_name": node_name,
                    "avg_execution_time": avg_time,
                    "executions": len(times),
                    "severity": "high" if avg_time > 60 else "medium"
                })
        
        # Check for memory issues
        memory_metrics = self.metrics_collector.get_metrics(
            MetricType.MEMORY_USAGE,
            time_window_minutes=time_window_minutes
        )
        
        if memory_metrics:
            memory_values = [m.value for m in memory_metrics]
            max_memory = max(memory_values)
            
            # Memory usage > 1GB is concerning
            if max_memory > 1024**3:
                bottlenecks.append({
                    "type": "high_memory_usage",
                    "max_memory_gb": max_memory / (1024**3),
                    "avg_memory_gb": mean(memory_values) / (1024**3),
                    "severity": "critical" if max_memory > 2 * (1024**3) else "high"
                })
        
        # Check for large checkpoints
        checkpoint_metrics = self.metrics_collector.get_metrics(
            MetricType.CHECKPOINT_SIZE,
            time_window_minutes=time_window_minutes
        )
        
        if checkpoint_metrics:
            checkpoint_sizes = [m.value for m in checkpoint_metrics]
            avg_size = mean(checkpoint_sizes)
            
            # Average checkpoint > 100MB is concerning
            if avg_size > 100 * 1024 * 1024:
                bottlenecks.append({
                    "type": "large_checkpoints",
                    "avg_size_mb": avg_size / (1024**2),
                    "max_size_mb": max(checkpoint_sizes) / (1024**2),
                    "severity": "medium"
                })
        
        return bottlenecks
    
    async def _generate_recommendations(self, report: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        bottlenecks = report.get("bottlenecks", [])
        system_metrics = report.get("system_metrics", {})
        
        # Recommendations based on bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_node":
                recommendations.append(OptimizationRecommendation(
                    strategy=OptimizationStrategy.CACHING,
                    description=f"Implement caching for slow node: {bottleneck['node_name']}",
                    expected_improvement=f"Reduce execution time from {bottleneck['avg_execution_time']:.1f}s",
                    implementation_effort="medium",
                    priority="high" if bottleneck.get("severity") == "high" else "medium",
                    parameters={"node_name": bottleneck["node_name"]}
                ))
                
                recommendations.append(OptimizationRecommendation(
                    strategy=OptimizationStrategy.PARALLELIZATION,
                    description=f"Parallelize operations in node: {bottleneck['node_name']}",
                    expected_improvement="50-70% reduction in execution time",
                    implementation_effort="high",
                    priority="medium",
                    parameters={"node_name": bottleneck["node_name"]}
                ))
            
            elif bottleneck["type"] == "high_memory_usage":
                recommendations.append(OptimizationRecommendation(
                    strategy=OptimizationStrategy.COMPRESSION,
                    description="Implement state compression to reduce memory usage",
                    expected_improvement=f"Reduce memory usage by 30-50% from {bottleneck['max_memory_gb']:.1f}GB",
                    implementation_effort="medium",
                    priority="high",
                    parameters={"compression_type": "gzip"}
                ))
                
                recommendations.append(OptimizationRecommendation(
                    strategy=OptimizationStrategy.LAZY_LOADING,
                    description="Implement lazy loading for state components",
                    expected_improvement="Reduce initial memory footprint",
                    implementation_effort="high",
                    priority="medium"
                ))
            
            elif bottleneck["type"] == "large_checkpoints":
                recommendations.append(OptimizationRecommendation(
                    strategy=OptimizationStrategy.COMPRESSION,
                    description="Compress checkpoint data before storage",
                    expected_improvement=f"Reduce checkpoint size by 60-80% from {bottleneck['avg_size_mb']:.1f}MB",
                    implementation_effort="low",
                    priority="medium",
                    parameters={"compression_algorithm": "lz4"}
                ))
        
        # System-level recommendations
        if system_metrics.get("memory_percent", 0) > 80:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.RESOURCE_POOLING,
                description="Implement resource pooling to manage memory usage",
                expected_improvement="Better resource utilization and reduced memory spikes",
                implementation_effort="high",
                priority="high"
            ))
        
        if system_metrics.get("active_sessions", 0) > 10:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.LOAD_BALANCING,
                description="Implement load balancing for multiple concurrent sessions",
                expected_improvement="Better distribution of computational load",
                implementation_effort="high",
                priority="medium"
            ))
        
        return recommendations
    
    async def _store_session_summary(self, session_id: str, summary: Dict[str, Any]):
        """Store session performance summary."""
        try:
            await redis_state_manager.save_state(
                session_id,
                summary,
                state_type="performance_summary",
                ttl=7 * 24 * 3600  # Keep for 7 days
            )
        except Exception as e:
            logger.error(f"Failed to store session summary: {e}")


class PerformanceOptimizer:
    """Implements performance optimizations."""
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
    async def optimize_state_serialization(self, state: Dict[str, Any]) -> bytes:
        """Optimize state serialization with compression."""
        try:
            import gzip
            
            # Serialize to JSON
            json_data = json.dumps(state, default=str)
            
            # Compress
            compressed_data = gzip.compress(json_data.encode('utf-8'))
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Failed to optimize state serialization: {e}")
            # Fallback to regular JSON
            return json.dumps(state, default=str).encode('utf-8')
    
    async def optimize_state_deserialization(self, compressed_data: bytes) -> Dict[str, Any]:
        """Optimize state deserialization with decompression."""
        try:
            import gzip
            
            # Decompress
            json_data = gzip.decompress(compressed_data).decode('utf-8')
            
            # Deserialize
            state = json.loads(json_data)
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to optimize state deserialization: {e}")
            # Fallback to regular JSON
            try:
                return json.loads(compressed_data.decode('utf-8'))
            except:
                return {}
    
    async def cache_result(self, key: str, result: Any, ttl_seconds: int = 3600):
        """Cache a result with TTL."""
        self.cache[key] = {
            "result": result,
            "expires_at": time.time() + ttl_seconds
        }
    
    async def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if key in self.cache:
            cache_entry = self.cache[key]
            if time.time() < cache_entry["expires_at"]:
                self.cache_stats["hits"] += 1
                return cache_entry["result"]
            else:
                # Expired
                del self.cache[key]
        
        self.cache_stats["misses"] += 1
        return None
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests) if total_requests > 0 else 0.0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "total_cached_items": len(self.cache),
            "cache_size_estimate": len(self.cache) * 1024  # Rough estimate
        }
    
    async def clear_expired_cache(self):
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time >= entry["expires_at"]
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")


# Global instances
performance_monitor = PerformanceMonitor()
performance_optimizer = PerformanceOptimizer()