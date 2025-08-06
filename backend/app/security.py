"""
Security and rate limiting module for the Automata Learning Platform.
Provides comprehensive security features including rate limiting, API key management,
input validation, and security headers.
"""

import hashlib
import hmac
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import logging

from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import redis

from .config import settings

logger = logging.getLogger(__name__)

# Initialize Redis for rate limiting storage
try:
    redis_client = redis.Redis(
        host=getattr(settings, 'redis_host', 'localhost'),
        port=getattr(settings, 'redis_port', 6379),
        db=getattr(settings, 'redis_db', 0),
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    logger.info("Redis connection established for rate limiting")
except Exception as e:
    logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
    redis_client = None

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379" if redis_client else "memory://",
    default_limits=["1000 per hour"]
)

# Security bearer for API key authentication
security = HTTPBearer(auto_error=False)

# API key storage (in production, this would be in a secure database)
API_KEYS = {
    "ai_service_key": {
        "key_hash": "hashed_key_here",
        "scopes": ["ai:read", "ai:write"],
        "rate_limit": "100/minute",
        "created_at": "2025-08-05T16:27:32Z",
        "last_used": None,
        "usage_count": 0
    }
}

# Security configurations
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

# Input validation patterns
VALIDATION_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "username": r"^[a-zA-Z0-9_-]{3,30}$",
    "safe_string": r"^[a-zA-Z0-9\s\-_.(){}[\],:;!?@#$%^&*+=<>/\\|`~\"']*$",
    "problem_id": r"^[a-zA-Z0-9_-]{1,50}$",
    "session_id": r"^[a-zA-Z0-9_-]{10,100}$"
}

# Rate limiting configurations
RATE_LIMITS = {
    "auth": {
        "login": "5/minute",
        "register": "3/minute",
        "refresh": "10/minute"
    },
    "ai": {
        "generate": "10/minute",
        "query": "20/minute",
        "hint": "15/minute"
    },
    "problems": {
        "validate": "30/minute",
        "submit": "20/minute",
        "hint": "10/minute"
    },
    "jflap": {
        "convert": "30/minute",
        "simulate": "50/minute",
        "minimize": "20/minute"
    },
    "general": {
        "api": "100/minute",
        "health": "60/minute"
    }
}


class SecurityManager:
    """Central security management class."""
    
    def __init__(self):
        self.blocked_ips = set()
        self.suspicious_activity = {}
        
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP address is blocked."""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str, reason: str = "Security violation"):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        logger.warning(f"IP {ip} blocked: {reason}")
    
    def track_suspicious_activity(self, ip: str, activity: str):
        """Track suspicious activity from an IP."""
        if ip not in self.suspicious_activity:
            self.suspicious_activity[ip] = []
        
        self.suspicious_activity[ip].append({
            "activity": activity,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Auto-block if too many suspicious activities
        if len(self.suspicious_activity[ip]) > 10:
            self.block_ip(ip, "Too many suspicious activities")


security_manager = SecurityManager()


def hash_api_key(key: str) -> str:
    """Hash an API key securely."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, key_hash: str) -> bool:
    """Verify an API key against its hash."""
    return hmac.compare_digest(hash_api_key(key), key_hash)


async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Extract and validate API key from request."""
    if not credentials:
        return None
    
    key = credentials.credentials
    for key_id, key_data in API_KEYS.items():
        if verify_api_key(key, key_data["key_hash"]):
            # Update usage statistics
            key_data["last_used"] = datetime.utcnow().isoformat()
            key_data["usage_count"] += 1
            return {"key_id": key_id, **key_data}
    
    return None


def require_api_key(scopes: List[str] = None):
    """Decorator to require API key authentication."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or args[0] if args else None
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            credentials = await security(request)
            api_key_data = await get_api_key(credentials)
            
            if not api_key_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Valid API key required",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Check scopes if specified
            if scopes:
                key_scopes = api_key_data.get("scopes", [])
                if not any(scope in key_scopes for scope in scopes):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
            
            kwargs["api_key_data"] = api_key_data
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(text: str, pattern_name: str, max_length: int = 1000) -> str:
    """Validate and sanitize input text."""
    import re
    
    if not text or len(text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long (max {max_length} characters)"
        )
    
    if pattern_name in VALIDATION_PATTERNS:
        pattern = VALIDATION_PATTERNS[pattern_name]
        if not re.match(pattern, text):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid {pattern_name} format"
            )
    
    # Remove potentially dangerous characters
    sanitized = text.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    return sanitized


class SecurityMiddleware:
    """Custom security middleware."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Check if IP is blocked
            client_ip = request.client.host
            if security_manager.is_ip_blocked(client_ip):
                response = {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [[b"content-type", b"application/json"]]
                }
                await send(response)
                
                body = json.dumps({"error": "Access denied"}).encode()
                await send({
                    "type": "http.response.body",
                    "body": body
                })
                return
            
            # Add security headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    for name, value in SECURITY_HEADERS.items():
                        headers.append([name.encode(), value.encode()])
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


def detect_sql_injection(text: str) -> bool:
    """Detect potential SQL injection attempts."""
    sql_patterns = [
        r"(?i)(union\s+select)",
        r"(?i)(drop\s+table)",
        r"(?i)(delete\s+from)",
        r"(?i)(insert\s+into)",
        r"(?i)(update\s+.*\s+set)",
        r"(?i)(exec\s*\()",
        r"(?i)(script\s*>)",
        r"(\-\-|\#|\/\*)"
    ]
    
    import re
    for pattern in sql_patterns:
        if re.search(pattern, text):
            return True
    return False


def detect_xss_attempt(text: str) -> bool:
    """Detect potential XSS attempts."""
    xss_patterns = [
        r"(?i)<script.*?>",
        r"(?i)javascript:",
        r"(?i)on\w+\s*=",
        r"(?i)<iframe.*?>",
        r"(?i)<object.*?>",
        r"(?i)<embed.*?>"
    ]
    
    import re
    for pattern in xss_patterns:
        if re.search(pattern, text):
            return True
    return False


async def security_scan_request(request: Request):
    """Scan incoming request for security threats."""
    client_ip = request.client.host
    
    # Check request size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
        security_manager.track_suspicious_activity(client_ip, "Large request body")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request too large"
        )
    
    # Check for suspicious headers
    suspicious_headers = ["x-forwarded-for", "x-real-ip", "x-cluster-client-ip"]
    for header in suspicious_headers:
        if header in request.headers:
            # Log but don't block - these can be legitimate
            logger.info(f"Request with {header} header from {client_ip}")
    
    # Scan request body if present
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            if body:
                body_str = body.decode('utf-8', errors='ignore')
                
                if detect_sql_injection(body_str):
                    security_manager.track_suspicious_activity(client_ip, "SQL injection attempt")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Malicious request detected"
                    )
                
                if detect_xss_attempt(body_str):
                    security_manager.track_suspicious_activity(client_ip, "XSS attempt")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Malicious request detected"
                    )
        except UnicodeDecodeError:
            # Binary content, skip scanning
            pass


# Rate limiting decorators for different endpoint types
def rate_limit_auth(endpoint: str = "general"):
    """Rate limit for authentication endpoints."""
    limit = RATE_LIMITS["auth"].get(endpoint, "5/minute")
    return limiter.limit(limit)


def rate_limit_ai(endpoint: str = "general"):
    """Rate limit for AI endpoints."""
    limit = RATE_LIMITS["ai"].get(endpoint, "10/minute")
    return limiter.limit(limit)


def rate_limit_problems(endpoint: str = "general"):
    """Rate limit for problem endpoints."""
    limit = RATE_LIMITS["problems"].get(endpoint, "30/minute")
    return limiter.limit(limit)


def rate_limit_jflap(endpoint: str = "general"):
    """Rate limit for JFLAP endpoints."""
    limit = RATE_LIMITS["jflap"].get(endpoint, "30/minute")
    return limiter.limit(limit)


def rate_limit_general():
    """General rate limit for all endpoints."""
    return limiter.limit(RATE_LIMITS["general"]["api"])


# Security audit logging
class SecurityLogger:
    """Security-focused logging utility."""
    
    @staticmethod
    def log_auth_attempt(user_id: str, success: bool, ip: str, user_agent: str = None):
        """Log authentication attempt."""
        logger.info(f"Auth attempt - User: {user_id}, Success: {success}, IP: {ip}, UA: {user_agent}")
    
    @staticmethod
    def log_api_key_usage(key_id: str, endpoint: str, ip: str):
        """Log API key usage."""
        logger.info(f"API key usage - Key: {key_id}, Endpoint: {endpoint}, IP: {ip}")
    
    @staticmethod
    def log_rate_limit_exceeded(ip: str, endpoint: str):
        """Log rate limit exceeded."""
        logger.warning(f"Rate limit exceeded - IP: {ip}, Endpoint: {endpoint}")
    
    @staticmethod
    def log_security_violation(violation_type: str, ip: str, details: str):
        """Log security violation."""
        logger.error(f"Security violation - Type: {violation_type}, IP: {ip}, Details: {details}")


security_logger = SecurityLogger()


def get_client_info(request: Request) -> Dict[str, str]:
    """Extract client information from request."""
    return {
        "ip": request.client.host,
        "user_agent": request.headers.get("user-agent", "Unknown"),
        "referer": request.headers.get("referer", "None"),
        "forwarded_for": request.headers.get("x-forwarded-for", "None")
    }