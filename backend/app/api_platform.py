"""
API Platform for third-party integrations with OAuth2 authentication,
rate limiting, webhook support, and SDK generation capabilities.
"""

import json
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import aioredis
import asyncio
import hmac
import httpx
from .database import Base, get_db
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class APIScope(str, Enum):
    """Available API scopes."""
    READ_PROBLEMS = "read:problems"
    WRITE_PROBLEMS = "write:problems"
    READ_SOLUTIONS = "read:solutions"
    WRITE_SOLUTIONS = "write:solutions"
    READ_USERS = "read:users"
    WRITE_USERS = "write:users"
    ADMIN = "admin"
    WEBHOOKS = "webhooks"
    EXPORT = "export"

class RateLimitTier(str, Enum):
    """Rate limiting tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class WebhookEventType(str, Enum):
    """Webhook event types."""
    SOLUTION_SUBMITTED = "solution.submitted"
    PROBLEM_COMPLETED = "problem.completed"
    USER_REGISTERED = "user.registered"
    LEARNING_MILESTONE = "learning.milestone"
    SYSTEM_ALERT = "system.alert"

# Database Models
class APIClient(Base):
    """API client model for OAuth2 applications."""
    __tablename__ = "api_clients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String(255), unique=True, nullable=False, index=True)
    client_secret_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # OAuth2 configuration
    redirect_uris = Column(JSON, default=list)
    scopes = Column(JSON, default=list)
    grant_types = Column(JSON, default=["authorization_code", "refresh_token"])
    
    # Rate limiting
    rate_limit_tier = Column(String(50), default=RateLimitTier.FREE.value)
    requests_per_minute = Column(Integer, default=60)
    requests_per_hour = Column(Integer, default=1000)
    requests_per_day = Column(Integer, default=10000)
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    is_confidential = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used_at = Column(DateTime)
    
    # Owner information
    owner_email = Column(String(255), nullable=False)
    contact_email = Column(String(255))
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="client")
    access_tokens = relationship("APIAccessToken", back_populates="client")
    webhooks = relationship("WebhookEndpoint", back_populates="client")
    
    __table_args__ = (
        Index('idx_client_active', 'is_active'),
    )

class APIKey(Base):
    """API key model for simplified authentication."""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("api_clients.id"), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    key_prefix = Column(String(20), nullable=False)  # For identification
    name = Column(String(255), nullable=False)
    
    # Permissions
    scopes = Column(JSON, default=list)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    client = relationship("APIClient", back_populates="api_keys")
    
    __table_args__ = (
        Index('idx_api_key_prefix', 'key_prefix'),
        Index('idx_api_key_active', 'is_active'),
    )

class APIAccessToken(Base):
    """OAuth2 access token model."""
    __tablename__ = "api_access_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("api_clients.id"), nullable=False)
    token_hash = Column(String(255), nullable=False, unique=True)
    refresh_token_hash = Column(String(255))
    
    # Token details
    scopes = Column(JSON, default=list)
    expires_at = Column(DateTime, nullable=False)
    refresh_expires_at = Column(DateTime)
    
    # User context (if applicable)
    user_id = Column(UUID(as_uuid=True))
    
    # Status
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    client = relationship("APIClient", back_populates="access_tokens")
    
    __table_args__ = (
        Index('idx_token_expires', 'expires_at'),
        Index('idx_token_revoked', 'is_revoked'),
    )

class WebhookEndpoint(Base):
    """Webhook endpoint configuration."""
    __tablename__ = "webhook_endpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("api_clients.id"), nullable=False)
    url = Column(String(2048), nullable=False)
    secret = Column(String(255), nullable=False)
    
    # Event configuration
    events = Column(JSON, default=list)  # List of WebhookEventType values
    is_active = Column(Boolean, default=True)
    
    # Delivery settings
    max_delivery_attempts = Column(Integer, default=3)
    retry_backoff_seconds = Column(Integer, default=60)
    
    # Statistics
    total_deliveries = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    last_delivery_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    client = relationship("APIClient", back_populates="webhooks")
    deliveries = relationship("WebhookDelivery", back_populates="endpoint")

class WebhookDelivery(Base):
    """Webhook delivery attempt tracking."""
    __tablename__ = "webhook_deliveries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint_id = Column(UUID(as_uuid=True), ForeignKey("webhook_endpoints.id"), nullable=False)
    
    # Delivery details
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False)
    attempt_number = Column(Integer, default=1)
    
    # Response details
    status_code = Column(Integer)
    response_body = Column(Text)
    response_headers = Column(JSON)
    
    # Status
    is_successful = Column(Boolean, default=False)
    delivered_at = Column(DateTime, default=datetime.utcnow)
    next_retry_at = Column(DateTime)
    
    # Relationships
    endpoint = relationship("WebhookEndpoint", back_populates="deliveries")
    
    __table_args__ = (
        Index('idx_delivery_status', 'is_successful'),
        Index('idx_delivery_retry', 'next_retry_at'),
    )

class RateLimitEntry(Base):
    """Rate limiting tracking."""
    __tablename__ = "rate_limit_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("api_clients.id"), nullable=False)
    
    # Rate limiting windows
    requests_this_minute = Column(Integer, default=0)
    requests_this_hour = Column(Integer, default=0)
    requests_this_day = Column(Integer, default=0)
    
    # Window timestamps
    minute_window_start = Column(DateTime, default=datetime.utcnow)
    hour_window_start = Column(DateTime, default=datetime.utcnow)
    day_window_start = Column(DateTime, default=datetime.utcnow)
    
    # Last request
    last_request_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_rate_limit_client', 'client_id'),
    )

# Pydantic Models
class ClientRegistrationRequest(BaseModel):
    """Request model for client registration."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    redirect_uris: List[str] = Field(default_factory=list)
    scopes: List[APIScope] = Field(default_factory=list)
    owner_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    contact_email: Optional[str] = None
    rate_limit_tier: RateLimitTier = RateLimitTier.FREE
    
    @validator('redirect_uris')
    def validate_redirect_uris(cls, v):
        for uri in v:
            if not uri.startswith(('http://', 'https://')):
                raise ValueError('Redirect URIs must use HTTP or HTTPS')
        return v

class ClientResponse(BaseModel):
    """Response model for client information."""
    id: str
    client_id: str
    name: str
    description: Optional[str]
    scopes: List[str]
    rate_limit_tier: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime]

class APIKeyRequest(BaseModel):
    """Request model for API key creation."""
    name: str = Field(..., min_length=1, max_length=255)
    scopes: List[APIScope] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)

class APIKeyResponse(BaseModel):
    """Response model for API key creation."""
    id: str
    key: str  # Only returned once
    key_prefix: str
    name: str
    scopes: List[str]
    expires_at: Optional[datetime]
    created_at: datetime

class WebhookEndpointRequest(BaseModel):
    """Request model for webhook endpoint registration."""
    url: str = Field(..., regex=r'^https?://.+')
    events: List[WebhookEventType]
    secret: Optional[str] = None
    max_delivery_attempts: int = Field(default=3, ge=1, le=10)

class WebhookEndpointResponse(BaseModel):
    """Response model for webhook endpoint."""
    id: str
    url: str
    events: List[str]
    is_active: bool
    created_at: datetime
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int

# Main API Platform Class
class APIPlatform:
    """Main API platform management class."""
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.rate_limits = {
            RateLimitTier.FREE: {"minute": 60, "hour": 1000, "day": 10000},
            RateLimitTier.BASIC: {"minute": 300, "hour": 10000, "day": 100000},
            RateLimitTier.PREMIUM: {"minute": 1000, "hour": 50000, "day": 500000},
            RateLimitTier.ENTERPRISE: {"minute": 5000, "hour": 200000, "day": 2000000}
        }
    
    async def initialize(self):
        """Initialize API platform components."""
        try:
            # Initialize Redis for rate limiting
            self.redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("API Platform initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API Platform: {e}")
            raise
    
    async def register_client(self, request: ClientRegistrationRequest,
                            db: Session) -> Tuple[str, str]:
        """Register a new API client and return client_id and client_secret."""
        try:
            # Generate client credentials
            client_id = f"ac_{secrets.token_urlsafe(32)}"
            client_secret = secrets.token_urlsafe(64)
            client_secret_hash = pwd_context.hash(client_secret)
            
            # Create client record
            client = APIClient(
                client_id=client_id,
                client_secret_hash=client_secret_hash,
                name=request.name,
                description=request.description,
                redirect_uris=request.redirect_uris,
                scopes=[scope.value for scope in request.scopes],
                owner_email=request.owner_email,
                contact_email=request.contact_email,
                rate_limit_tier=request.rate_limit_tier.value
            )
            
            # Set rate limits based on tier
            limits = self.rate_limits[request.rate_limit_tier]
            client.requests_per_minute = limits["minute"]
            client.requests_per_hour = limits["hour"]
            client.requests_per_day = limits["day"]
            
            db.add(client)
            db.commit()
            db.refresh(client)
            
            logger.info(f"Registered new API client: {client_id}")
            return client_id, client_secret
            
        except Exception as e:
            logger.error(f"Error registering client: {e}")
            db.rollback()
            raise
    
    async def authenticate_client(self, client_id: str, client_secret: str,
                                db: Session) -> Optional[APIClient]:
        """Authenticate API client credentials."""
        try:
            client = db.query(APIClient).filter(
                APIClient.client_id == client_id,
                APIClient.is_active == True
            ).first()
            
            if not client:
                return None
            
            if not pwd_context.verify(client_secret, client.client_secret_hash):
                return None
            
            # Update last used timestamp
            client.last_used_at = datetime.utcnow()
            db.commit()
            
            return client
            
        except Exception as e:
            logger.error(f"Error authenticating client: {e}")
            return None
    
    async def create_api_key(self, client_id: str, request: APIKeyRequest,
                           db: Session) -> Tuple[str, APIKey]:
        """Create a new API key for a client."""
        try:
            # Get client
            client = db.query(APIClient).filter(
                APIClient.id == client_id,
                APIClient.is_active == True
            ).first()
            
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            
            # Generate API key
            key = f"ak_{secrets.token_urlsafe(48)}"
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            key_prefix = key[:12] + "..."
            
            # Calculate expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
            
            # Validate scopes
            valid_scopes = [scope.value for scope in request.scopes 
                          if scope.value in client.scopes]
            
            # Create API key record
            api_key = APIKey(
                client_id=client.id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=request.name,
                scopes=valid_scopes,
                expires_at=expires_at
            )
            
            db.add(api_key)
            db.commit()
            db.refresh(api_key)
            
            logger.info(f"Created API key for client {client.client_id}: {key_prefix}")
            return key, api_key
            
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            db.rollback()
            raise
    
    async def validate_api_key(self, api_key: str, db: Session) -> Optional[Tuple[APIClient, APIKey]]:
        """Validate API key and return associated client and key objects."""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Find API key
            api_key_obj = db.query(APIKey).filter(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            ).first()
            
            if not api_key_obj:
                return None
            
            # Check expiration
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                return None
            
            # Get associated client
            client = db.query(APIClient).filter(
                APIClient.id == api_key_obj.client_id,
                APIClient.is_active == True
            ).first()
            
            if not client:
                return None
            
            # Update usage statistics
            api_key_obj.usage_count += 1
            api_key_obj.last_used_at = datetime.utcnow()
            client.last_used_at = datetime.utcnow()
            db.commit()
            
            return client, api_key_obj
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None
    
    async def check_rate_limit(self, client: APIClient, db: Session) -> bool:
        """Check if client has exceeded rate limits."""
        try:
            if not self.redis_client:
                # Fallback to database-based rate limiting
                return await self._check_db_rate_limit(client, db)
            
            # Use Redis for efficient rate limiting
            return await self._check_redis_rate_limit(client)
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    async def _check_redis_rate_limit(self, client: APIClient) -> bool:
        """Redis-based rate limiting."""
        now = datetime.utcnow()
        client_key = f"rate_limit:{client.client_id}"
        
        # Check minute window
        minute_key = f"{client_key}:minute:{now.strftime('%Y%m%d%H%M')}"
        minute_count = await self.redis_client.incr(minute_key)
        if minute_count == 1:
            await self.redis_client.expire(minute_key, 60)
        
        if minute_count > client.requests_per_minute:
            return False
        
        # Check hour window
        hour_key = f"{client_key}:hour:{now.strftime('%Y%m%d%H')}"
        hour_count = await self.redis_client.incr(hour_key)
        if hour_count == 1:
            await self.redis_client.expire(hour_key, 3600)
        
        if hour_count > client.requests_per_hour:
            return False
        
        # Check day window
        day_key = f"{client_key}:day:{now.strftime('%Y%m%d')}"
        day_count = await self.redis_client.incr(day_key)
        if day_count == 1:
            await self.redis_client.expire(day_key, 86400)
        
        if day_count > client.requests_per_day:
            return False
        
        return True
    
    async def _check_db_rate_limit(self, client: APIClient, db: Session) -> bool:
        """Database-based rate limiting fallback."""
        now = datetime.utcnow()
        
        # Get or create rate limit entry
        rate_entry = db.query(RateLimitEntry).filter(
            RateLimitEntry.client_id == client.id
        ).first()
        
        if not rate_entry:
            rate_entry = RateLimitEntry(
                client_id=client.id,
                minute_window_start=now,
                hour_window_start=now,
                day_window_start=now
            )
            db.add(rate_entry)
        
        # Check and update minute window
        if (now - rate_entry.minute_window_start).total_seconds() >= 60:
            rate_entry.requests_this_minute = 0
            rate_entry.minute_window_start = now
        
        rate_entry.requests_this_minute += 1
        if rate_entry.requests_this_minute > client.requests_per_minute:
            return False
        
        # Check and update hour window
        if (now - rate_entry.hour_window_start).total_seconds() >= 3600:
            rate_entry.requests_this_hour = 0
            rate_entry.hour_window_start = now
        
        rate_entry.requests_this_hour += 1
        if rate_entry.requests_this_hour > client.requests_per_hour:
            return False
        
        # Check and update day window
        if (now - rate_entry.day_window_start).total_seconds() >= 86400:
            rate_entry.requests_this_day = 0
            rate_entry.day_window_start = now
        
        rate_entry.requests_this_day += 1
        if rate_entry.requests_this_day > client.requests_per_day:
            return False
        
        rate_entry.last_request_at = now
        db.commit()
        
        return True
    
    async def register_webhook(self, client_id: str, request: WebhookEndpointRequest,
                             db: Session) -> WebhookEndpoint:
        """Register a webhook endpoint for a client."""
        try:
            # Get client
            client = db.query(APIClient).filter(
                APIClient.id == client_id,
                APIClient.is_active == True
            ).first()
            
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            
            # Generate webhook secret if not provided
            webhook_secret = request.secret or secrets.token_urlsafe(32)
            
            # Create webhook endpoint
            endpoint = WebhookEndpoint(
                client_id=client.id,
                url=request.url,
                secret=webhook_secret,
                events=[event.value for event in request.events],
                max_delivery_attempts=request.max_delivery_attempts
            )
            
            db.add(endpoint)
            db.commit()
            db.refresh(endpoint)
            
            logger.info(f"Registered webhook for client {client.client_id}: {request.url}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error registering webhook: {e}")
            db.rollback()
            raise
    
    async def send_webhook(self, event_type: WebhookEventType, payload: Dict[str, Any],
                         db: Session):
        """Send webhook to all registered endpoints for the event type."""
        try:
            # Find all active webhook endpoints for this event type
            endpoints = db.query(WebhookEndpoint).filter(
                WebhookEndpoint.is_active == True,
                WebhookEndpoint.events.contains([event_type.value])
            ).all()
            
            # Send to all endpoints concurrently
            tasks = [
                self._deliver_webhook(endpoint, event_type.value, payload, db)
                for endpoint in endpoints
            ]
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error sending webhooks: {e}")
    
    async def _deliver_webhook(self, endpoint: WebhookEndpoint, event_type: str,
                             payload: Dict[str, Any], db: Session, attempt: int = 1):
        """Deliver webhook to a specific endpoint."""
        try:
            # Create delivery record
            delivery = WebhookDelivery(
                endpoint_id=endpoint.id,
                event_type=event_type,
                payload=payload,
                attempt_number=attempt
            )
            
            # Prepare webhook payload
            webhook_payload = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": payload
            }
            
            # Calculate signature
            signature = self._calculate_webhook_signature(
                json.dumps(webhook_payload, sort_keys=True),
                endpoint.secret
            )
            
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-Event": event_type,
                "X-Webhook-Delivery": str(delivery.id),
                "User-Agent": "AutomataAPI-Webhooks/1.0"
            }
            
            # Send webhook
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint.url,
                    json=webhook_payload,
                    headers=headers
                )
                
                # Update delivery record
                delivery.status_code = response.status_code
                delivery.response_body = response.text[:1000]  # Truncate
                delivery.response_headers = dict(response.headers)
                delivery.is_successful = 200 <= response.status_code < 300
                
                # Update endpoint statistics
                endpoint.total_deliveries += 1
                endpoint.last_delivery_at = datetime.utcnow()
                
                if delivery.is_successful:
                    endpoint.successful_deliveries += 1
                else:
                    endpoint.failed_deliveries += 1
                    
                    # Schedule retry if needed
                    if attempt < endpoint.max_delivery_attempts:
                        backoff_seconds = endpoint.retry_backoff_seconds * (2 ** (attempt - 1))
                        delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=backoff_seconds)
                
                db.add(delivery)
                db.commit()
                
                logger.info(f"Webhook delivered: {endpoint.url} [{response.status_code}]")
                
                # Retry if failed and attempts remaining
                if not delivery.is_successful and attempt < endpoint.max_delivery_attempts:
                    await asyncio.sleep(endpoint.retry_backoff_seconds)
                    await self._deliver_webhook(endpoint, event_type, payload, db, attempt + 1)
                
        except Exception as e:
            logger.error(f"Error delivering webhook to {endpoint.url}: {e}")
            
            # Record failed delivery
            delivery.is_successful = False
            delivery.response_body = str(e)
            endpoint.failed_deliveries += 1
            db.add(delivery)
            db.commit()
    
    def _calculate_webhook_signature(self, payload: str, secret: str) -> str:
        """Calculate HMAC signature for webhook payload."""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def generate_openapi_spec(self, client: APIClient) -> Dict[str, Any]:
        """Generate OpenAPI specification for client's accessible endpoints."""
        try:
            # Base OpenAPI spec
            spec = {
                "openapi": "3.0.3",
                "info": {
                    "title": "Automata Learning Platform API",
                    "version": "1.0.0",
                    "description": f"API access for {client.name}",
                    "contact": {
                        "email": client.contact_email or client.owner_email
                    }
                },
                "servers": [
                    {"url": "https://api.automata-platform.com", "description": "Production"},
                    {"url": "https://api-staging.automata-platform.com", "description": "Staging"}
                ],
                "components": {
                    "securitySchemes": {
                        "ApiKeyAuth": {
                            "type": "apiKey",
                            "in": "header",
                            "name": "Authorization",
                            "description": "Use 'Bearer {api_key}'"
                        }
                    },
                    "schemas": {}
                },
                "security": [{"ApiKeyAuth": []}],
                "paths": {}
            }
            
            # Add paths based on client scopes
            if APIScope.READ_PROBLEMS.value in client.scopes:
                spec["paths"]["/api/problems"] = {
                    "get": {
                        "summary": "List problems",
                        "description": "Retrieve a list of available problems",
                        "responses": {
                            "200": {"description": "Successful response"}
                        }
                    }
                }
            
            if APIScope.READ_SOLUTIONS.value in client.scopes:
                spec["paths"]["/api/solutions"] = {
                    "get": {
                        "summary": "List solutions",
                        "description": "Retrieve user solutions",
                        "responses": {
                            "200": {"description": "Successful response"}
                        }
                    }
                }
            
            if APIScope.EXPORT.value in client.scopes:
                spec["paths"]["/api/export/latex"] = {
                    "post": {
                        "summary": "Export to LaTeX",
                        "description": "Export automata and solutions to LaTeX format",
                        "responses": {
                            "200": {"description": "LaTeX document generated"}
                        }
                    }
                }
            
            return spec
            
        except Exception as e:
            logger.error(f"Error generating OpenAPI spec: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("API Platform cleaned up")

# Global API platform instance
api_platform = APIPlatform()

# Authentication dependencies
async def get_current_client(credentials: HTTPAuthorizationCredentials = Depends(security),
                           db: Session = Depends(get_db)) -> Tuple[APIClient, Optional[APIKey]]:
    """Dependency to get current authenticated API client."""
    try:
        token = credentials.credentials
        
        # Try API key authentication first
        if token.startswith("ak_"):
            result = await api_platform.validate_api_key(token, db)
            if result:
                client, api_key = result
                # Check rate limits
                if not await api_platform.check_rate_limit(client, db):
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
                return client, api_key
        
        # OAuth2 access token authentication would go here
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

def require_scope(required_scope: APIScope):
    """Dependency factory to require specific API scope."""
    def _check_scope(client_info: Tuple[APIClient, Optional[APIKey]] = Depends(get_current_client)):
        client, api_key = client_info
        
        # Check client scopes
        if required_scope.value not in client.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope.value}"
            )
        
        # Check API key scopes if applicable
        if api_key and required_scope.value not in api_key.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key lacks required scope: {required_scope.value}"
            )
        
        return client, api_key
    
    return _check_scope


# Utility functions
async def initialize_api_platform():
    """Initialize the API platform."""
    await api_platform.initialize()

async def cleanup_api_platform():
    """Cleanup the API platform."""
    await api_platform.cleanup()

async def send_webhook_event(event_type: WebhookEventType, payload: Dict[str, Any],
                           db: Session):
    """Send webhook event to all registered endpoints."""
    await api_platform.send_webhook(event_type, payload, db)