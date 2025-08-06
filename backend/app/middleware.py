"""
Middleware for error handling, logging, and request validation.
"""
import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to logs
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Log response time
            process_time = time.time() - start_time
            logger.info(
                f"Request {request_id} completed in {process_time:.3f}s "
                f"with status {response.status_code}"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            logger.warning(
                f"Request {request_id} failed with HTTP {exc.status_code}: {exc.detail}"
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "request_id": request_id,
                    "status_code": exc.status_code
                },
                headers={"X-Request-ID": request_id}
            )
            
        except Exception as exc:
            # Handle unexpected exceptions
            logger.error(
                f"Request {request_id} failed with unexpected error: {str(exc)}",
                exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "status_code": 500
                },
                headers={"X-Request-ID": request_id}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}  # In production, use Redis
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier (IP address)
        client = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old entries
        self.clients = {
            ip: times for ip, times in self.clients.items()
            if any(t > now - self.period for t in times)
        }
        
        # Check rate limit
        if client in self.clients:
            recent_calls = [t for t in self.clients[client] if t > now - self.period]
            if len(recent_calls) >= self.calls:
                logger.warning(f"Rate limit exceeded for client {client}")
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded. Please try again later."}
                )
            self.clients[client] = recent_calls + [now]
        else:
            self.clients[client] = [now]
        
        return await call_next(request)