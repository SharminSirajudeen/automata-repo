"""
Ollama Cost Tracking and Performance Optimization System.
Maximizes performance while minimizing resource usage through intelligent tracking,
caching, and optimization strategies for local Ollama deployments.
"""

import asyncio
import json
import logging
import time
import psutil
import hashlib
from typing import Dict, List, Optional, Any, NamedTuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from pathlib import Path
import pickle
import gzip

from .config import settings
from .valkey_integration import valkey_connection_manager

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers based on resource requirements and capabilities."""
    MICRO = "micro"      # < 1B parameters (e.g., TinyLlama)
    SMALL = "small"      # 1-7B parameters (e.g., Llama 3.1 8B, CodeLlama 7B)
    MEDIUM = "medium"    # 7-15B parameters (e.g., CodeLlama 13B)
    LARGE = "large"      # 15-35B parameters (e.g., CodeLlama 34B)
    XLARGE = "xlarge"    # > 35B parameters (e.g., Llama 3.1 70B)


@dataclass
class TokenUsage:
    """Detailed token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # Tokens served from cache
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.cached_tokens / self.total_tokens


@dataclass
class ResourceUsage:
    """System resource usage during inference."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    inference_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    @property
    def total_time_ms(self) -> float:
        return self.queue_time_ms + self.inference_time_ms


@dataclass
class CostMetrics:
    """Cost calculation based on resource usage."""
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    time_cost: float = 0.0
    total_cost: float = 0.0
    efficiency_score: float = 0.0  # Tokens per cost unit


@dataclass
class ModelPerformanceProfile:
    """Performance profile for specific models."""
    model_name: str
    tier: ModelTier
    avg_tokens_per_second: float = 0.0
    avg_memory_usage_mb: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_inference_time_ms: float = 0.0
    cache_effectiveness: float = 0.0
    cost_per_1k_tokens: float = 0.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class OllamaCostTracker:
    """Comprehensive cost tracking and optimization for Ollama deployments."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Cost configuration (adjustable based on actual resource costs)
        self.cost_config = {
            "cpu_cost_per_hour": 0.10,          # $0.10 per CPU hour
            "memory_cost_per_gb_hour": 0.05,    # $0.05 per GB RAM hour
            "gpu_cost_per_hour": 2.00,          # $2.00 per GPU hour
            "storage_cost_per_gb": 0.02,        # $0.02 per GB storage
            "network_cost_per_gb": 0.01,        # $0.01 per GB network
            "efficiency_weight": 0.3             # Weight for efficiency scoring
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "min_tokens_per_second": 10,
            "max_inference_time_ms": 30000,  # 30 seconds
            "max_memory_mb": 16000,          # 16GB
            "max_cpu_percent": 90
        }
        
        # Tracking data structures
        self.session_costs: Dict[str, List[Dict]] = defaultdict(list)
        self.user_costs: Dict[str, Dict] = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_tokens": 0,
            "session_count": 0,
            "daily_limit": 10.0,  # $10 daily limit per user
            "monthly_limit": 200.0  # $200 monthly limit per user
        })
        self.model_profiles: Dict[str, ModelPerformanceProfile] = {}
        self.recent_requests: deque = deque(maxlen=1000)
        self.optimization_suggestions: List[Dict] = []
        
        # Real-time monitoring
        self.monitoring_active = True
        self.monitoring_task = None
        
        # Cache for cost calculations
        self.cost_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("Ollama Cost Tracker initialized with performance optimization focus")
    
    def get_model_tier(self, model_name: str) -> ModelTier:
        """Classify model into performance tiers."""
        model_lower = model_name.lower()
        
        # Micro models (fastest, lowest cost)
        if any(x in model_lower for x in ['tiny', '1b', '0.5b', 'nano']):
            return ModelTier.MICRO
        
        # Small models (good balance)
        elif any(x in model_lower for x in ['7b', '8b', '3.1:8b', 'small']):
            return ModelTier.SMALL
        
        # Medium models
        elif any(x in model_lower for x in ['13b', '15b', 'medium']):
            return ModelTier.MEDIUM
        
        # Large models
        elif any(x in model_lower for x in ['34b', '33b', 'large']):
            return ModelTier.LARGE
        
        # XLarge models (highest capability, highest cost)
        elif any(x in model_lower for x in ['70b', '65b', '72b', 'xlarge']):
            return ModelTier.XLARGE
        
        # Default to small for unknown models
        return ModelTier.SMALL
    
    async def start_request_tracking(
        self,
        session_id: str,
        user_id: str,
        model_name: str,
        prompt: str,
        request_metadata: Dict[str, Any] = None
    ) -> str:
        """Start tracking a new request."""
        request_id = f"{session_id}_{int(time.time() * 1000)}"
        
        # Check user budget limits
        user_data = self.user_costs[user_id]
        daily_spent = self._calculate_daily_spending(user_id)
        monthly_spent = self._calculate_monthly_spending(user_id)
        
        if daily_spent >= user_data["daily_limit"]:
            raise ValueError(f"Daily budget limit exceeded: ${daily_spent:.2f} >= ${user_data['daily_limit']:.2f}")
        
        if monthly_spent >= user_data["monthly_limit"]:
            raise ValueError(f"Monthly budget limit exceeded: ${monthly_spent:.2f} >= ${user_data['monthly_limit']:.2f}")
        
        # Create prompt hash for caching
        prompt_hash = hashlib.md5(f"{model_name}:{prompt}".encode()).hexdigest()
        
        tracking_data = {
            "request_id": request_id,
            "session_id": session_id,
            "user_id": user_id,
            "model_name": model_name,
            "model_tier": self.get_model_tier(model_name).value,
            "prompt_hash": prompt_hash,
            "prompt_length": len(prompt),
            "start_time": time.time(),
            "metadata": request_metadata or {},
            "resource_start": self._capture_resource_snapshot()
        }
        
        # Store in recent requests
        self.recent_requests.append(tracking_data)
        
        # Check if we can serve from cache
        cached_result = await self._check_semantic_cache(prompt_hash, model_name)
        if cached_result:
            tracking_data["cache_hit"] = True
            tracking_data["cached_tokens"] = cached_result.get("token_count", 0)
            logger.info(f"Cache hit for request {request_id}")
        else:
            tracking_data["cache_hit"] = False
            tracking_data["cached_tokens"] = 0
        
        return request_id
    
    async def complete_request_tracking(
        self,
        request_id: str,
        response: str,
        token_usage: TokenUsage,
        error: Optional[str] = None
    ) -> CostMetrics:
        """Complete tracking and calculate costs for a request."""
        # Find the tracking data
        tracking_data = None
        for req in self.recent_requests:
            if req.get("request_id") == request_id:
                tracking_data = req
                break
        
        if not tracking_data:
            logger.error(f"Tracking data not found for request {request_id}")
            return CostMetrics()
        
        end_time = time.time()
        total_time_ms = (end_time - tracking_data["start_time"]) * 1000
        
        # Capture final resource usage
        resource_end = self._capture_resource_snapshot()
        resource_usage = self._calculate_resource_delta(
            tracking_data["resource_start"], 
            resource_end, 
            total_time_ms
        )
        
        # Calculate token metrics
        if not tracking_data["cache_hit"]:
            # Fresh inference
            resource_usage.tokens_per_second = (
                token_usage.total_tokens / (total_time_ms / 1000.0) 
                if total_time_ms > 0 else 0.0
            )
        else:
            # Cache hit - minimal resource usage
            resource_usage.inference_time_ms = min(resource_usage.inference_time_ms, 50)
            resource_usage.cpu_percent = min(resource_usage.cpu_percent, 5)
            token_usage.cached_tokens = tracking_data["cached_tokens"]
        
        # Calculate costs
        cost_metrics = self._calculate_cost_metrics(resource_usage, token_usage)
        
        # Update model performance profile
        await self._update_model_profile(
            tracking_data["model_name"],
            resource_usage,
            token_usage,
            cost_metrics
        )
        
        # Store cost data
        cost_record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "session_id": tracking_data["session_id"],
            "user_id": tracking_data["user_id"],
            "model_name": tracking_data["model_name"],
            "model_tier": tracking_data["model_tier"],
            "prompt_length": tracking_data["prompt_length"],
            "token_usage": token_usage.__dict__,
            "resource_usage": resource_usage.__dict__,
            "cost_metrics": cost_metrics.__dict__,
            "cache_hit": tracking_data["cache_hit"],
            "error": error,
            "optimization_score": self._calculate_optimization_score(
                resource_usage, token_usage, cost_metrics
            )
        }
        
        # Store in session and user costs
        self.session_costs[tracking_data["session_id"]].append(cost_record)
        self.user_costs[tracking_data["user_id"]]["total_cost"] += cost_metrics.total_cost
        self.user_costs[tracking_data["user_id"]]["total_tokens"] += token_usage.total_tokens
        
        # Cache response for future use (if not from cache and successful)
        if not tracking_data["cache_hit"] and not error and response:
            await self._cache_response(
                tracking_data["prompt_hash"],
                tracking_data["model_name"],
                response,
                token_usage.total_tokens
            )
        
        # Generate optimization suggestions
        await self._generate_optimization_suggestions(cost_record)
        
        logger.info(f"Request {request_id} completed: ${cost_metrics.total_cost:.4f}, "
                   f"{token_usage.total_tokens} tokens, {resource_usage.tokens_per_second:.1f} t/s")
        
        return cost_metrics
    
    def _capture_resource_snapshot(self) -> Dict[str, float]:
        """Capture current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Try to get GPU memory if available
            gpu_memory_mb = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory_mb = sum(gpu.memoryUsed for gpu in gpus)
            except ImportError:
                pass  # GPU monitoring not available
            
            return {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_mb": memory.used / 1024 / 1024,
                "gpu_memory_mb": gpu_memory_mb
            }
            
        except Exception as e:
            logger.warning(f"Failed to capture resource snapshot: {e}")
            return {
                "timestamp": time.time(),
                "cpu_percent": 0.0,
                "memory_mb": 0.0,
                "gpu_memory_mb": 0.0
            }
    
    def _calculate_resource_delta(
        self,
        start_snapshot: Dict[str, float],
        end_snapshot: Dict[str, float],
        duration_ms: float
    ) -> ResourceUsage:
        """Calculate resource usage delta."""
        duration_hours = duration_ms / 1000.0 / 3600.0
        
        avg_cpu_percent = (start_snapshot["cpu_percent"] + end_snapshot["cpu_percent"]) / 2
        avg_memory_mb = (start_snapshot["memory_mb"] + end_snapshot["memory_mb"]) / 2
        avg_gpu_memory_mb = (start_snapshot["gpu_memory_mb"] + end_snapshot["gpu_memory_mb"]) / 2
        
        return ResourceUsage(
            cpu_percent=avg_cpu_percent,
            memory_mb=avg_memory_mb,
            gpu_memory_mb=avg_gpu_memory_mb,
            inference_time_ms=duration_ms,
            queue_time_ms=0.0,  # Calculated separately if needed
            tokens_per_second=0.0  # Calculated later with token info
        )
    
    def _calculate_cost_metrics(
        self,
        resource_usage: ResourceUsage,
        token_usage: TokenUsage
    ) -> CostMetrics:
        """Calculate detailed cost metrics."""
        duration_hours = resource_usage.total_time_ms / 1000.0 / 3600.0
        
        # CPU cost (based on usage percentage)
        cpu_cost = (
            (resource_usage.cpu_percent / 100.0) * 
            self.cost_config["cpu_cost_per_hour"] * 
            duration_hours
        )
        
        # Memory cost
        memory_cost = (
            (resource_usage.memory_mb / 1024.0) * 
            self.cost_config["memory_cost_per_gb_hour"] * 
            duration_hours
        )
        
        # GPU cost (if applicable)
        gpu_cost = 0.0
        if resource_usage.gpu_memory_mb > 0:
            gpu_cost = (
                self.cost_config["gpu_cost_per_hour"] * 
                duration_hours
            )
        
        # Time-based cost (opportunity cost)
        time_cost = duration_hours * 0.01  # $0.01 per hour base time cost
        
        total_cost = cpu_cost + memory_cost + gpu_cost + time_cost
        
        # Calculate efficiency score
        efficiency_score = 0.0
        if total_cost > 0 and token_usage.total_tokens > 0:
            efficiency_score = token_usage.total_tokens / total_cost
        
        return CostMetrics(
            compute_cost=cpu_cost + gpu_cost,
            memory_cost=memory_cost,
            time_cost=time_cost,
            total_cost=total_cost,
            efficiency_score=efficiency_score
        )
    
    async def _update_model_profile(
        self,
        model_name: str,
        resource_usage: ResourceUsage,
        token_usage: TokenUsage,
        cost_metrics: CostMetrics
    ):
        """Update performance profile for a model."""
        if model_name not in self.model_profiles:
            self.model_profiles[model_name] = ModelPerformanceProfile(
                model_name=model_name,
                tier=self.get_model_tier(model_name)
            )
        
        profile = self.model_profiles[model_name]
        
        # Update running averages
        n = profile.sample_count
        profile.avg_tokens_per_second = (
            (profile.avg_tokens_per_second * n + resource_usage.tokens_per_second) / (n + 1)
        )
        profile.avg_memory_usage_mb = (
            (profile.avg_memory_usage_mb * n + resource_usage.memory_mb) / (n + 1)
        )
        profile.avg_cpu_usage = (
            (profile.avg_cpu_usage * n + resource_usage.cpu_percent) / (n + 1)
        )
        profile.avg_inference_time_ms = (
            (profile.avg_inference_time_ms * n + resource_usage.inference_time_ms) / (n + 1)
        )
        profile.cost_per_1k_tokens = (
            cost_metrics.total_cost / (token_usage.total_tokens / 1000.0)
            if token_usage.total_tokens > 0 else 0.0
        )
        
        profile.sample_count += 1
        profile.last_updated = datetime.now()
        
        # Save to persistent storage periodically
        if profile.sample_count % 10 == 0:
            await self._save_model_profiles()
    
    def _calculate_optimization_score(
        self,
        resource_usage: ResourceUsage,
        token_usage: TokenUsage,
        cost_metrics: CostMetrics
    ) -> float:
        """Calculate optimization score (0-100, higher is better)."""
        score = 100.0
        
        # Penalize high resource usage
        if resource_usage.cpu_percent > 80:
            score -= 20
        if resource_usage.memory_mb > 12000:  # > 12GB
            score -= 15
        if resource_usage.tokens_per_second < 15:
            score -= 20
        if resource_usage.inference_time_ms > 10000:  # > 10 seconds
            score -= 25
        
        # Bonus for cache hits
        if token_usage.cache_hit_rate > 0.8:
            score += 15
        
        # Bonus for efficiency
        if cost_metrics.efficiency_score > 1000:  # > 1000 tokens per cost unit
            score += 10
        
        return max(0.0, min(100.0, score))
    
    async def _check_semantic_cache(
        self,
        prompt_hash: str,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Check if response exists in semantic cache."""
        try:
            cache_key = f"ollama_cache:{model_name}:{prompt_hash}"
            
            async with valkey_connection_manager.get_client() as client:
                cached_data = await client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None
    
    async def _cache_response(
        self,
        prompt_hash: str,
        model_name: str,
        response: str,
        token_count: int
    ):
        """Cache response for future use."""
        try:
            cache_key = f"ollama_cache:{model_name}:{prompt_hash}"
            cache_data = {
                "response": response,
                "token_count": token_count,
                "cached_at": datetime.now().isoformat(),
                "model_name": model_name
            }
            
            # Cache for 6 hours
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    cache_key, 
                    21600,  # 6 hours
                    json.dumps(cache_data, separators=(',', ':'))
                )
                
            logger.debug(f"Cached response for {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    async def _generate_optimization_suggestions(self, cost_record: Dict[str, Any]):
        """Generate optimization suggestions based on performance data."""
        suggestions = []
        
        resource_usage = ResourceUsage(**cost_record["resource_usage"])
        token_usage = TokenUsage(**cost_record["token_usage"])
        model_tier = cost_record["model_tier"]
        
        # Model tier optimization
        if resource_usage.tokens_per_second < 10 and model_tier in ["large", "xlarge"]:
            suggestions.append({
                "type": "model_downgrade",
                "message": f"Consider using a smaller model for faster responses",
                "current_model": cost_record["model_name"],
                "suggested_models": self._suggest_smaller_models(cost_record["model_name"]),
                "potential_savings": "30-60% cost reduction"
            })
        
        # Memory optimization
        if resource_usage.memory_mb > 12000:
            suggestions.append({
                "type": "memory_optimization",
                "message": "High memory usage detected",
                "suggestions": [
                    "Reduce context length",
                    "Use model quantization",
                    "Implement request batching"
                ]
            })
        
        # Cache optimization
        if token_usage.cache_hit_rate < 0.3:
            suggestions.append({
                "type": "cache_optimization",
                "message": "Low cache hit rate",
                "suggestions": [
                    "Implement semantic similarity caching",
                    "Precompute common responses",
                    "Use prompt templates"
                ]
            })
        
        if suggestions:
            self.optimization_suggestions.extend(suggestions)
            # Keep only recent suggestions
            self.optimization_suggestions = self.optimization_suggestions[-100:]
    
    def _suggest_smaller_models(self, current_model: str) -> List[str]:
        """Suggest smaller, more efficient models."""
        current_tier = self.get_model_tier(current_model)
        
        suggestions = []
        if current_tier == ModelTier.XLARGE:
            suggestions = ["codellama:34b", "deepseek-coder:33b", "llama3.1:8b"]
        elif current_tier == ModelTier.LARGE:
            suggestions = ["codellama:13b", "llama3.1:8b", "deepseek-coder:7b"]
        elif current_tier == ModelTier.MEDIUM:
            suggestions = ["llama3.1:8b", "codellama:7b", "tinyllama:1b"]
        elif current_tier == ModelTier.SMALL:
            suggestions = ["tinyllama:1b", "phi:2.7b"]
        
        return suggestions
    
    def _calculate_daily_spending(self, user_id: str) -> float:
        """Calculate user's spending today."""
        today = datetime.now().date()
        total = 0.0
        
        for session_costs in self.session_costs.values():
            for cost_record in session_costs:
                if (cost_record["user_id"] == user_id and
                    datetime.fromisoformat(cost_record["timestamp"]).date() == today):
                    total += cost_record["cost_metrics"]["total_cost"]
        
        return total
    
    def _calculate_monthly_spending(self, user_id: str) -> float:
        """Calculate user's spending this month."""
        current_month = datetime.now().replace(day=1)
        total = 0.0
        
        for session_costs in self.session_costs.values():
            for cost_record in session_costs:
                record_date = datetime.fromisoformat(cost_record["timestamp"])
                if (cost_record["user_id"] == user_id and
                    record_date >= current_month):
                    total += cost_record["cost_metrics"]["total_cost"]
        
        return total
    
    async def get_cost_report(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive cost report."""
        time_range = time_range or timedelta(days=7)
        cutoff_time = datetime.now() - time_range
        
        # Collect relevant cost records
        relevant_records = []
        
        if session_id:
            relevant_records = [
                r for r in self.session_costs.get(session_id, [])
                if datetime.fromisoformat(r["timestamp"]) >= cutoff_time
            ]
        else:
            for session_costs in self.session_costs.values():
                for record in session_costs:
                    record_time = datetime.fromisoformat(record["timestamp"])
                    if (record_time >= cutoff_time and
                        (not user_id or record["user_id"] == user_id)):
                        relevant_records.append(record)
        
        if not relevant_records:
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_requests": 0,
                "avg_cost_per_request": 0.0,
                "avg_cost_per_token": 0.0,
                "model_breakdown": {},
                "optimization_suggestions": []
            }
        
        # Calculate totals
        total_cost = sum(r["cost_metrics"]["total_cost"] for r in relevant_records)
        total_tokens = sum(r["token_usage"]["total_tokens"] for r in relevant_records)
        total_requests = len(relevant_records)
        
        # Model breakdown
        model_breakdown = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "requests": 0})
        for record in relevant_records:
            model = record["model_name"]
            model_breakdown[model]["cost"] += record["cost_metrics"]["total_cost"]
            model_breakdown[model]["tokens"] += record["token_usage"]["total_tokens"]
            model_breakdown[model]["requests"] += 1
        
        # Performance metrics
        avg_tokens_per_second = sum(
            r["resource_usage"]["tokens_per_second"] for r in relevant_records
        ) / len(relevant_records)
        
        avg_optimization_score = sum(
            r.get("optimization_score", 0) for r in relevant_records
        ) / len(relevant_records)
        
        return {
            "time_range": str(time_range),
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "avg_cost_per_request": round(total_cost / total_requests, 4),
            "avg_cost_per_token": round(total_cost / total_tokens, 6) if total_tokens > 0 else 0.0,
            "avg_tokens_per_second": round(avg_tokens_per_second, 2),
            "avg_optimization_score": round(avg_optimization_score, 2),
            "model_breakdown": dict(model_breakdown),
            "model_profiles": {
                name: profile.__dict__ for name, profile in self.model_profiles.items()
            },
            "optimization_suggestions": self.optimization_suggestions[-10:],  # Last 10
            "cache_stats": self._get_cache_stats(),
            "budget_status": self._get_budget_status(user_id) if user_id else {}
        }
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.recent_requests:
            return {"hit_rate": 0.0, "total_requests": 0}
        
        cache_hits = sum(1 for req in self.recent_requests if req.get("cache_hit", False))
        total_requests = len(self.recent_requests)
        
        return {
            "hit_rate": cache_hits / total_requests if total_requests > 0 else 0.0,
            "cache_hits": cache_hits,
            "total_requests": total_requests
        }
    
    def _get_budget_status(self, user_id: str) -> Dict[str, Any]:
        """Get user budget status."""
        user_data = self.user_costs[user_id]
        daily_spent = self._calculate_daily_spending(user_id)
        monthly_spent = self._calculate_monthly_spending(user_id)
        
        return {
            "daily_spent": round(daily_spent, 4),
            "daily_limit": user_data["daily_limit"],
            "daily_remaining": round(user_data["daily_limit"] - daily_spent, 4),
            "monthly_spent": round(monthly_spent, 4),
            "monthly_limit": user_data["monthly_limit"],
            "monthly_remaining": round(user_data["monthly_limit"] - monthly_spent, 4),
            "total_lifetime_cost": round(user_data["total_cost"], 4),
            "total_lifetime_tokens": user_data["total_tokens"]
        }
    
    async def _save_model_profiles(self):
        """Save model profiles to persistent storage."""
        try:
            profiles_data = {
                name: {
                    **profile.__dict__,
                    "last_updated": profile.last_updated.isoformat()
                }
                for name, profile in self.model_profiles.items()
            }
            
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    "ollama_model_profiles",
                    86400,  # 24 hours
                    json.dumps(profiles_data, separators=(',', ':'))
                )
                
        except Exception as e:
            logger.warning(f"Failed to save model profiles: {e}")
    
    async def load_model_profiles(self):
        """Load model profiles from persistent storage."""
        try:
            async with valkey_connection_manager.get_client() as client:
                profiles_data = await client.get("ollama_model_profiles")
                
                if profiles_data:
                    data = json.loads(profiles_data)
                    for name, profile_dict in data.items():
                        profile_dict["last_updated"] = datetime.fromisoformat(profile_dict["last_updated"])
                        profile_dict["tier"] = ModelTier(profile_dict["tier"])
                        self.model_profiles[name] = ModelPerformanceProfile(**profile_dict)
                    
                    logger.info(f"Loaded {len(self.model_profiles)} model profiles")
                    
        except Exception as e:
            logger.warning(f"Failed to load model profiles: {e}")
    
    async def update_user_limits(
        self,
        user_id: str,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None
    ):
        """Update user budget limits."""
        user_data = self.user_costs[user_id]
        
        if daily_limit is not None:
            user_data["daily_limit"] = daily_limit
        if monthly_limit is not None:
            user_data["monthly_limit"] = monthly_limit
        
        logger.info(f"Updated limits for user {user_id}: "
                   f"daily=${user_data['daily_limit']}, monthly=${user_data['monthly_limit']}")


# Global cost tracker instance
cost_tracker = OllamaCostTracker()


async def initialize_cost_tracker():
    """Initialize the cost tracker."""
    try:
        await cost_tracker.load_model_profiles()
        logger.info("Ollama cost tracker initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize cost tracker: {e}")
        raise


async def shutdown_cost_tracker():
    """Shutdown the cost tracker and save data."""
    try:
        await cost_tracker._save_model_profiles()
        logger.info("Cost tracker shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during cost tracker shutdown: {e}")


# Convenience functions for easy integration
async def track_ollama_request(
    session_id: str,
    user_id: str,
    model_name: str,
    prompt: str,
    response: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    metadata: Dict[str, Any] = None
) -> CostMetrics:
    """Simple function to track a complete Ollama request."""
    request_id = await cost_tracker.start_request_tracking(
        session_id, user_id, model_name, prompt, metadata
    )
    
    token_usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )
    
    return await cost_tracker.complete_request_tracking(
        request_id, response, token_usage
    )