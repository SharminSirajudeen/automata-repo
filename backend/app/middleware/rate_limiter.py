"""
Advanced Rate Limiting Middleware with Token Bucket Implementation.
Provides per-user, per-endpoint, and adaptive rate limiting for cost optimization
and abuse prevention in educational applications.
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import user_agents

from ..config import settings
from ..valkey_integration import valkey_connection_manager

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of rate limits."""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    TOKENS_PER_HOUR = "tokens_per_hour"
    COMPUTE_TIME_PER_HOUR = "compute_time_per_hour"
    COST_PER_HOUR = "cost_per_hour"


class RateLimitScope(Enum):
    """Scope of rate limiting."""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    USER_ENDPOINT = "user_endpoint"
    IP_ENDPOINT = "ip_endpoint"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float
    
    def consume(self, tokens_requested: int = 1) -> bool:
        """Attempt to consume tokens from the bucket."""
        now = time.time()
        
        # Refill bucket based on time elapsed
        time_elapsed = now - self.last_refill
        tokens_to_add = time_elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Check if we can consume the requested tokens
        if self.tokens >= tokens_requested:
            self.tokens -= tokens_requested
            return True
        
        return False
    
    def get_wait_time(self, tokens_requested: int = 1) -> float:
        """Get time to wait before tokens are available."""
        if self.tokens >= tokens_requested:
            return 0.0
        
        tokens_needed = tokens_requested - self.tokens
        return tokens_needed / self.refill_rate


@dataclass
class RateLimit:
    """Rate limit configuration."""
    limit_type: LimitType
    limit_value: int
    window_seconds: int
    scope: RateLimitScope
    burst_multiplier: float = 1.5
    priority: int = 1  # Higher priority = more strict
    
    def get_bucket_capacity(self) -> int:
        """Get token bucket capacity based on burst multiplier."""
        return int(self.limit_value * self.burst_multiplier)
    
    def get_refill_rate(self) -> float:
        """Get token refill rate per second."""
        return self.limit_value / self.window_seconds


@dataclass
class RateLimitStatus:
    """Status of rate limiting for a request."""
    allowed: bool
    limit_type: str
    limit_value: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None
    cost_estimate: float = 0.0
    warning_threshold: float = 0.8  # Warn when 80% of limit used


class AdaptiveRateLimiter:
    """Advanced rate limiter with adaptive limits and educational focus."""
    
    def __init__(self):
        # Configuration for different user tiers
        self.default_limits = {
            "free": {
                LimitType.REQUESTS_PER_MINUTE: RateLimit(
                    LimitType.REQUESTS_PER_MINUTE, 30, 60, RateLimitScope.USER, 2.0, 1
                ),
                LimitType.REQUESTS_PER_HOUR: RateLimit(
                    LimitType.REQUESTS_PER_HOUR, 200, 3600, RateLimitScope.USER, 1.5, 2
                ),
                LimitType.TOKENS_PER_MINUTE: RateLimit(
                    LimitType.TOKENS_PER_MINUTE, 1000, 60, RateLimitScope.USER, 2.0, 1
                ),
                LimitType.COST_PER_HOUR: RateLimit(
                    LimitType.COST_PER_HOUR, 100, 3600, RateLimitScope.USER, 1.2, 3  # $1.00/hour
                ),
            },
            "student": {
                LimitType.REQUESTS_PER_MINUTE: RateLimit(
                    LimitType.REQUESTS_PER_MINUTE, 60, 60, RateLimitScope.USER, 2.0, 1
                ),
                LimitType.REQUESTS_PER_HOUR: RateLimit(
                    LimitType.REQUESTS_PER_HOUR, 500, 3600, RateLimitScope.USER, 1.5, 2
                ),
                LimitType.TOKENS_PER_MINUTE: RateLimit(
                    LimitType.TOKENS_PER_MINUTE, 2000, 60, RateLimitScope.USER, 2.0, 1
                ),
                LimitType.COST_PER_HOUR: RateLimit(
                    LimitType.COST_PER_HOUR, 500, 3600, RateLimitScope.USER, 1.2, 3  # $5.00/hour
                ),
            },
            "educator": {
                LimitType.REQUESTS_PER_MINUTE: RateLimit(
                    LimitType.REQUESTS_PER_MINUTE, 120, 60, RateLimitScope.USER, 2.0, 1
                ),
                LimitType.REQUESTS_PER_HOUR: RateLimit(
                    LimitType.REQUESTS_PER_HOUR, 1000, 3600, RateLimitScope.USER, 1.5, 2
                ),
                LimitType.TOKENS_PER_MINUTE: RateLimit(
                    LimitType.TOKENS_PER_MINUTE, 5000, 60, RateLimitScope.USER, 2.0, 1
                ),
                LimitType.COST_PER_HOUR: RateLimit(
                    LimitType.COST_PER_HOUR, 2000, 3600, RateLimitScope.USER, 1.2, 3  # $20.00/hour
                ),
            }
        }
        
        # Endpoint-specific limits (stricter for expensive operations)
        self.endpoint_limits = {
            "/api/ai/generate": {
                LimitType.REQUESTS_PER_MINUTE: RateLimit(
                    LimitType.REQUESTS_PER_MINUTE, 10, 60, RateLimitScope.USER_ENDPOINT, 1.2, 4
                )
            },
            "/api/ai/proof": {
                LimitType.REQUESTS_PER_MINUTE: RateLimit(
                    LimitType.REQUESTS_PER_MINUTE, 5, 60, RateLimitScope.USER_ENDPOINT, 1.2, 5
                ),
                LimitType.COMPUTE_TIME_PER_HOUR: RateLimit(
                    LimitType.COMPUTE_TIME_PER_HOUR, 3600, 3600, RateLimitScope.USER_ENDPOINT, 1.1, 5  # 1 hour compute time
                )
            },
            "/api/jflap/simulate": {
                LimitType.REQUESTS_PER_MINUTE: RateLimit(
                    LimitType.REQUESTS_PER_MINUTE, 20, 60, RateLimitScope.USER_ENDPOINT, 1.5, 3
                )
            }
        }
        
        # IP-based limits for abuse prevention
        self.ip_limits = {
            LimitType.REQUESTS_PER_MINUTE: RateLimit(
                LimitType.REQUESTS_PER_MINUTE, 100, 60, RateLimitScope.IP, 1.2, 2
            ),
            LimitType.REQUESTS_PER_HOUR: RateLimit(
                LimitType.REQUESTS_PER_HOUR, 2000, 3600, RateLimitScope.IP, 1.2, 3
            )
        }
        
        # In-memory buckets (with Valkey backup)
        self.buckets: Dict[str, TokenBucket] = {}
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        self.adaptive_multipliers: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Performance tracking for adaptive limits
        self.performance_stats = {
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "load_factor": 0.0
        }
        
        # Cleanup task
        self.cleanup_task = None
    
    async def initialize(self):
        """Initialize the rate limiter."""
        try:
            # Load existing buckets from Valkey
            await self._load_buckets_from_storage()
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Rate limiter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the rate limiter."""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Save buckets to Valkey
            await self._save_buckets_to_storage()
            
            logger.info("Rate limiter shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during rate limiter shutdown: {e}")
    
    def get_user_tier(self, request: Request) -> str:
        """Determine user tier from request."""
        # In a real implementation, this would check user authentication
        # For now, return based on headers or default to "free"
        tier = getattr(request.state, 'user_tier', None)
        if tier:
            return tier
        
        # Check for tier in headers (for testing)
        tier_header = request.headers.get('X-User-Tier', 'free')
        return tier_header.lower()
    
    def get_user_id(self, request: Request) -> str:
        """Extract user ID from request."""
        # Try to get from request state (set by auth middleware)
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return str(user_id)
        
        # Fallback to IP address for anonymous users
        client_ip = self._get_client_ip(request)
        return f"anon_{hashlib.md5(client_ip.encode()).hexdigest()[:8]}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    def _get_bucket_key(
        self,
        scope: RateLimitScope,
        limit_type: LimitType,
        user_id: str = "",
        ip: str = "",
        endpoint: str = ""
    ) -> str:
        """Generate bucket key for rate limiting."""
        components = [scope.value, limit_type.value]
        
        if scope in [RateLimitScope.USER, RateLimitScope.USER_ENDPOINT]:
            components.append(user_id)
        
        if scope in [RateLimitScope.IP, RateLimitScope.IP_ENDPOINT]:
            components.append(ip)
        
        if scope in [RateLimitScope.ENDPOINT, RateLimitScope.USER_ENDPOINT, RateLimitScope.IP_ENDPOINT]:
            components.append(endpoint)
        
        return ":".join(components)
    
    def _create_bucket(self, rate_limit: RateLimit) -> TokenBucket:
        """Create a new token bucket."""
        return TokenBucket(
            capacity=rate_limit.get_bucket_capacity(),
            tokens=float(rate_limit.get_bucket_capacity()),  # Start full
            refill_rate=rate_limit.get_refill_rate(),
            last_refill=time.time()
        )
    
    async def check_rate_limit(
        self,
        request: Request,
        endpoint: str,
        tokens_requested: int = 1,
        cost_estimate: float = 0.0
    ) -> RateLimitStatus:
        """Check rate limits for a request."""
        user_id = self.get_user_id(request)
        user_tier = self.get_user_tier(request)
        client_ip = self._get_client_ip(request)
        
        # Get applicable rate limits
        applicable_limits = self._get_applicable_limits(user_tier, endpoint)
        
        # Check each limit (prioritized by strictness)
        for rate_limit in sorted(applicable_limits, key=lambda x: x.priority, reverse=True):
            bucket_key = self._get_bucket_key(
                rate_limit.scope, rate_limit.limit_type, user_id, client_ip, endpoint
            )
            
            # Get or create bucket
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = self._create_bucket(rate_limit)
            
            bucket = self.buckets[bucket_key]
            
            # Apply adaptive multiplier
            adaptive_key = f"{user_tier}:{rate_limit.limit_type.value}"
            multiplier = self.adaptive_multipliers[adaptive_key]
            adjusted_tokens = max(1, int(tokens_requested * multiplier))
            
            # Check if tokens can be consumed
            if not bucket.consume(adjusted_tokens):
                wait_time = bucket.get_wait_time(adjusted_tokens)
                
                return RateLimitStatus(
                    allowed=False,
                    limit_type=rate_limit.limit_type.value,
                    limit_value=rate_limit.limit_value,
                    remaining=int(bucket.tokens),
                    reset_time=int(time.time() + wait_time),
                    retry_after=int(wait_time) + 1,
                    cost_estimate=cost_estimate
                )
        
        # All limits passed
        remaining_tokens = min(
            int(self.buckets[self._get_bucket_key(
                limit.scope, limit.limit_type, user_id, client_ip, endpoint
            )].tokens)
            for limit in applicable_limits
            if self._get_bucket_key(limit.scope, limit.limit_type, user_id, client_ip, endpoint) in self.buckets
        )
        
        # Update request history for adaptive limits
        await self._update_request_history(user_id, endpoint)
        
        return RateLimitStatus(
            allowed=True,
            limit_type="combined",
            limit_value=min(limit.limit_value for limit in applicable_limits),
            remaining=remaining_tokens,
            reset_time=int(time.time() + 60),  # Next minute
            cost_estimate=cost_estimate
        )
    
    def _get_applicable_limits(self, user_tier: str, endpoint: str) -> List[RateLimit]:
        """Get all applicable rate limits for a user tier and endpoint."""
        limits = []
        
        # User tier limits
        tier_limits = self.default_limits.get(user_tier, self.default_limits["free"])
        limits.extend(tier_limits.values())
        
        # Endpoint-specific limits
        if endpoint in self.endpoint_limits:
            limits.extend(self.endpoint_limits[endpoint].values())
        
        # IP-based limits
        limits.extend(self.ip_limits.values())
        
        return limits
    
    async def _update_request_history(self, user_id: str, endpoint: str):
        """Update request history for adaptive rate limiting."""
        now = time.time()
        history_key = f"{user_id}:{endpoint}"
        
        # Add current request
        self.request_history[history_key].append(now)
        
        # Keep only last hour of history
        cutoff = now - 3600
        self.request_history[history_key] = [
            t for t in self.request_history[history_key] if t > cutoff
        ]
    
    async def update_performance_stats(
        self,
        response_time: float,
        status_code: int,
        load_factor: float
    ):
        """Update performance statistics for adaptive rate limiting."""
        # Update running averages
        alpha = 0.1  # Smoothing factor
        
        self.performance_stats["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * self.performance_stats["avg_response_time"]
        )
        
        # Update error rate
        is_error = 1.0 if status_code >= 400 else 0.0
        self.performance_stats["error_rate"] = (
            alpha * is_error + 
            (1 - alpha) * self.performance_stats["error_rate"]
        )
        
        self.performance_stats["load_factor"] = load_factor
        
        # Adjust adaptive multipliers based on performance
        await self._adjust_adaptive_multipliers()
    
    async def _adjust_adaptive_multipliers(self):
        """Adjust rate limits based on system performance."""
        # If system is under high load, make limits stricter
        if self.performance_stats["load_factor"] > 0.8:
            for key in self.adaptive_multipliers:
                self.adaptive_multipliers[key] = min(2.0, self.adaptive_multipliers[key] * 1.1)
        
        # If error rate is high, make limits stricter
        elif self.performance_stats["error_rate"] > 0.1:
            for key in self.adaptive_multipliers:
                self.adaptive_multipliers[key] = min(2.0, self.adaptive_multipliers[key] * 1.05)
        
        # If system is performing well, relax limits slightly
        elif (self.performance_stats["load_factor"] < 0.3 and 
              self.performance_stats["error_rate"] < 0.01):
            for key in self.adaptive_multipliers:
                self.adaptive_multipliers[key] = max(0.5, self.adaptive_multipliers[key] * 0.99)
    
    async def _load_buckets_from_storage(self):
        """Load token buckets from Valkey storage."""
        try:
            async with valkey_connection_manager.get_client() as client:
                bucket_data = await client.get("rate_limiter:buckets")
                
                if bucket_data:
                    data = json.loads(bucket_data)
                    for key, bucket_dict in data.items():
                        self.buckets[key] = TokenBucket(**bucket_dict)
                    
                    logger.info(f"Loaded {len(self.buckets)} token buckets from storage")
                    
        except Exception as e:
            logger.warning(f"Failed to load buckets from storage: {e}")
    
    async def _save_buckets_to_storage(self):
        """Save token buckets to Valkey storage."""
        try:
            # Convert buckets to serializable format
            bucket_data = {
                key: {
                    "capacity": bucket.capacity,
                    "tokens": bucket.tokens,
                    "refill_rate": bucket.refill_rate,
                    "last_refill": bucket.last_refill
                }
                for key, bucket in self.buckets.items()
            }
            
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    "rate_limiter:buckets",
                    3600,  # 1 hour TTL
                    json.dumps(bucket_data, separators=(',', ':'))
                )
                
            logger.debug(f"Saved {len(self.buckets)} token buckets to storage")
            
        except Exception as e:
            logger.warning(f"Failed to save buckets to storage: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old buckets and data."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                now = time.time()
                cutoff = now - 7200  # 2 hours
                
                # Remove old buckets that haven't been used recently
                old_buckets = [
                    key for key, bucket in self.buckets.items()
                    if bucket.last_refill < cutoff
                ]
                
                for key in old_buckets:
                    del self.buckets[key]
                
                # Clean up old request history
                for key in list(self.request_history.keys()):
                    self.request_history[key] = [
                        t for t in self.request_history[key] 
                        if t > cutoff
                    ]
                    
                    if not self.request_history[key]:
                        del self.request_history[key]
                
                # Save to storage periodically
                if len(self.buckets) > 0:
                    await self._save_buckets_to_storage()
                
                if old_buckets:
                    logger.debug(f"Cleaned up {len(old_buckets)} old buckets")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a user."""
        user_buckets = {
            key: {
                "tokens": bucket.tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
                "utilization": (bucket.capacity - bucket.tokens) / bucket.capacity
            }
            for key, bucket in self.buckets.items()
            if user_id in key
        }
        
        return {
            "user_id": user_id,
            "active_buckets": len(user_buckets),
            "bucket_details": user_buckets,
            "adaptive_multipliers": {
                key: multiplier 
                for key, multiplier in self.adaptive_multipliers.items()
                if user_id in key
            },
            "performance_stats": self.performance_stats.copy()
        }


# Global rate limiter instance
rate_limiter = AdaptiveRateLimiter()


# Middleware function
async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting."""
    start_time = time.time()
    
    try:
        # Extract endpoint path
        endpoint = request.url.path
        
        # Estimate tokens needed (basic estimation)
        tokens_needed = 1
        if "ai" in endpoint or "generate" in endpoint:
            tokens_needed = 5  # AI operations need more tokens
        elif "simulate" in endpoint:
            tokens_needed = 2  # Simulations need moderate tokens
        
        # Check rate limits
        rate_status = await rate_limiter.check_rate_limit(
            request, endpoint, tokens_needed
        )
        
        if not rate_status.allowed:
            # Rate limit exceeded
            headers = {
                "X-RateLimit-Limit": str(rate_status.limit_value),
                "X-RateLimit-Remaining": str(rate_status.remaining),
                "X-RateLimit-Reset": str(rate_status.reset_time),
                "Retry-After": str(rate_status.retry_after or 60)
            }
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests for {rate_status.limit_type}",
                    "retry_after": rate_status.retry_after,
                    "limit_value": rate_status.limit_value,
                    "remaining": rate_status.remaining
                },
                headers=headers
            )
        
        # Add rate limit info to request
        request.state.rate_limit_status = rate_status
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_status.limit_value)
        response.headers["X-RateLimit-Remaining"] = str(rate_status.remaining)
        response.headers["X-RateLimit-Reset"] = str(rate_status.reset_time)
        
        # Warn if approaching limit
        utilization = 1.0 - (rate_status.remaining / rate_status.limit_value)
        if utilization > rate_status.warning_threshold:
            response.headers["X-RateLimit-Warning"] = (
                f"Approaching limit: {int(utilization * 100)}% used"
            )
        
        # Update performance stats
        processing_time = time.time() - start_time
        await rate_limiter.update_performance_stats(
            processing_time,
            response.status_code,
            load_factor=min(1.0, processing_time / 5.0)  # Normalize to 5-second max
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in rate limit middleware: {e}")
        # Don't block requests if rate limiter fails
        return await call_next(request)


# Decorator for function-based rate limiting
def rate_limited(
    tokens_needed: int = 1,
    limit_type: LimitType = LimitType.REQUESTS_PER_MINUTE,
    custom_limits: Optional[Dict[str, RateLimit]] = None
):
    """Decorator for function-based rate limiting."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            endpoint = f"{func.__module__}.{func.__name__}"
            
            rate_status = await rate_limiter.check_rate_limit(
                request, endpoint, tokens_needed
            )
            
            if not rate_status.allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": rate_status.retry_after,
                        "limit_type": rate_status.limit_type
                    }
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


# Initialize and shutdown functions
async def initialize_rate_limiter():
    """Initialize the rate limiter."""
    await rate_limiter.initialize()


async def shutdown_rate_limiter():
    """Shutdown the rate limiter."""
    await rate_limiter.shutdown()


# Utility functions
async def get_user_rate_limit_status(user_id: str) -> Dict[str, Any]:
    """Get rate limit status for a specific user."""
    return await rate_limiter.get_user_stats(user_id)


def create_custom_rate_limit(
    requests_per_minute: int,
    requests_per_hour: int,
    user_tier: str = "custom"
) -> Dict[LimitType, RateLimit]:
    """Create custom rate limits for specific use cases."""
    return {
        LimitType.REQUESTS_PER_MINUTE: RateLimit(
            LimitType.REQUESTS_PER_MINUTE, requests_per_minute, 60, 
            RateLimitScope.USER, 1.5, 1
        ),
        LimitType.REQUESTS_PER_HOUR: RateLimit(
            LimitType.REQUESTS_PER_HOUR, requests_per_hour, 3600,
            RateLimitScope.USER, 1.2, 2
        )
    }