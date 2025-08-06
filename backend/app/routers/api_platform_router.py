"""
API Platform router for third-party integrations.
Handles OAuth2, API keys, rate limiting, and webhook management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from ..database import get_db
from ..api_platform import (
    api_platform, ClientRegistrationRequest, ClientResponse,
    APIKeyRequest, APIKeyResponse, WebhookEndpointRequest, 
    WebhookEndpointResponse, APIClient, APIKey, WebhookEndpoint,
    get_current_client, require_scope, APIScope, WebhookEventType
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/platform", tags=["API Platform"])
security = HTTPBasic()

@router.post("/clients/register", response_model=Dict[str, str])
async def register_client(
    request: ClientRegistrationRequest,
    db: Session = Depends(get_db)
):
    """Register a new API client application."""
    try:
        client_id, client_secret = await api_platform.register_client(request, db)
        
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "message": "Client registered successfully. Store the client_secret securely - it won't be shown again."
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error registering client: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register client"
        )

@router.get("/clients/me", response_model=ClientResponse)
async def get_current_client_info(
    client_info = Depends(get_current_client)
):
    """Get information about the current authenticated client."""
    client, _ = client_info
    
    return ClientResponse(
        id=str(client.id),
        client_id=client.client_id,
        name=client.name,
        description=client.description,
        scopes=client.scopes,
        rate_limit_tier=client.rate_limit_tier,
        is_active=client.is_active,
        created_at=client.created_at,
        last_used_at=client.last_used_at
    )

@router.post("/clients/{client_id}/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    client_id: str,
    request: APIKeyRequest,
    credentials: HTTPBasicCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Create a new API key for a client."""
    try:
        # Authenticate client with basic auth
        client = await api_platform.authenticate_client(
            credentials.username, credentials.password, db
        )
        
        if not client or str(client.id) != client_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials"
            )
        
        api_key, api_key_obj = await api_platform.create_api_key(
            client_id, request, db
        )
        
        return APIKeyResponse(
            id=str(api_key_obj.id),
            key=api_key,  # Only shown once
            key_prefix=api_key_obj.key_prefix,
            name=api_key_obj.name,
            scopes=api_key_obj.scopes,
            expires_at=api_key_obj.expires_at,
            created_at=api_key_obj.created_at
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )

@router.get("/clients/{client_id}/api-keys")
async def list_api_keys(
    client_id: str,
    credentials: HTTPBasicCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """List API keys for a client."""
    try:
        # Authenticate client
        client = await api_platform.authenticate_client(
            credentials.username, credentials.password, db
        )
        
        if not client or str(client.id) != client_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials"
            )
        
        api_keys = db.query(APIKey).filter(
            APIKey.client_id == client.id,
            APIKey.is_active == True
        ).all()
        
        return {
            "api_keys": [
                {
                    "id": str(key.id),
                    "key_prefix": key.key_prefix,
                    "name": key.name,
                    "scopes": key.scopes,
                    "usage_count": key.usage_count,
                    "last_used_at": key.last_used_at,
                    "expires_at": key.expires_at,
                    "created_at": key.created_at
                }
                for key in api_keys
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )

@router.delete("/clients/{client_id}/api-keys/{key_id}")
async def revoke_api_key(
    client_id: str,
    key_id: str,
    credentials: HTTPBasicCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Revoke an API key."""
    try:
        # Authenticate client
        client = await api_platform.authenticate_client(
            credentials.username, credentials.password, db
        )
        
        if not client or str(client.id) != client_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials"
            )
        
        # Find and revoke API key
        api_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.client_id == client.id
        ).first()
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        api_key.is_active = False
        db.commit()
        
        return {"message": "API key revoked successfully"}
        
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )

@router.post("/webhooks", response_model=WebhookEndpointResponse)
async def register_webhook(
    request: WebhookEndpointRequest,
    client_info = Depends(require_scope(APIScope.WEBHOOKS)),
    db: Session = Depends(get_db)
):
    """Register a webhook endpoint."""
    try:
        client, _ = client_info
        
        endpoint = await api_platform.register_webhook(
            str(client.id), request, db
        )
        
        return WebhookEndpointResponse(
            id=str(endpoint.id),
            url=endpoint.url,
            events=endpoint.events,
            is_active=endpoint.is_active,
            created_at=endpoint.created_at,
            total_deliveries=endpoint.total_deliveries,
            successful_deliveries=endpoint.successful_deliveries,
            failed_deliveries=endpoint.failed_deliveries
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error registering webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register webhook"
        )

@router.get("/webhooks")
async def list_webhooks(
    client_info = Depends(require_scope(APIScope.WEBHOOKS)),
    db: Session = Depends(get_db)
):
    """List webhook endpoints for the client."""
    try:
        client, _ = client_info
        
        endpoints = db.query(WebhookEndpoint).filter(
            WebhookEndpoint.client_id == client.id
        ).all()
        
        return {
            "webhooks": [
                {
                    "id": str(endpoint.id),
                    "url": endpoint.url,
                    "events": endpoint.events,
                    "is_active": endpoint.is_active,
                    "created_at": endpoint.created_at,
                    "total_deliveries": endpoint.total_deliveries,
                    "successful_deliveries": endpoint.successful_deliveries,
                    "failed_deliveries": endpoint.failed_deliveries,
                    "last_delivery_at": endpoint.last_delivery_at
                }
                for endpoint in endpoints
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list webhooks"
        )

@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    client_info = Depends(require_scope(APIScope.WEBHOOKS)),
    db: Session = Depends(get_db)
):
    """Delete a webhook endpoint."""
    try:
        client, _ = client_info
        
        endpoint = db.query(WebhookEndpoint).filter(
            WebhookEndpoint.id == webhook_id,
            WebhookEndpoint.client_id == client.id
        ).first()
        
        if not endpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Webhook endpoint not found"
            )
        
        db.delete(endpoint)
        db.commit()
        
        return {"message": "Webhook endpoint deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete webhook"
        )

@router.post("/webhooks/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    client_info = Depends(require_scope(APIScope.WEBHOOKS)),
    db: Session = Depends(get_db)
):
    """Test a webhook endpoint with sample data."""
    try:
        client, _ = client_info
        
        endpoint = db.query(WebhookEndpoint).filter(
            WebhookEndpoint.id == webhook_id,
            WebhookEndpoint.client_id == client.id
        ).first()
        
        if not endpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Webhook endpoint not found"
            )
        
        # Send test webhook
        test_payload = {
            "test": True,
            "message": "This is a test webhook delivery",
            "client_id": client.client_id,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        await api_platform._deliver_webhook(
            endpoint, "test.event", test_payload, db
        )
        
        return {"message": "Test webhook sent successfully"}
        
    except Exception as e:
        logger.error(f"Error testing webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test webhook"
        )

@router.get("/openapi-spec")
async def get_openapi_spec(
    client_info = Depends(get_current_client)
):
    """Get OpenAPI specification for the client's accessible endpoints."""
    try:
        client, _ = client_info
        
        spec = await api_platform.generate_openapi_spec(client)
        
        return spec
        
    except Exception as e:
        logger.error(f"Error generating OpenAPI spec: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate OpenAPI specification"
        )

@router.get("/rate-limits")
async def get_rate_limits(
    client_info = Depends(get_current_client)
):
    """Get current rate limit status for the client."""
    try:
        client, _ = client_info
        
        return {
            "client_id": client.client_id,
            "rate_limit_tier": client.rate_limit_tier,
            "limits": {
                "requests_per_minute": client.requests_per_minute,
                "requests_per_hour": client.requests_per_hour,
                "requests_per_day": client.requests_per_day
            },
            "last_used_at": client.last_used_at
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get rate limit information"
        )

@router.get("/scopes")
async def get_available_scopes():
    """Get list of available API scopes."""
    return {
        "scopes": {
            scope.value: {
                "name": scope.value,
                "description": f"Access to {scope.value.split(':')[1]} operations"
            }
            for scope in APIScope
        }
    }

@router.get("/webhook-events")
async def get_webhook_events():
    """Get list of available webhook event types."""
    return {
        "events": {
            event.value: {
                "name": event.value,
                "description": f"Triggered when {event.value.replace('.', ' ')}"
            }
            for event in WebhookEventType
        }
    }

@router.get("/usage-statistics")
async def get_usage_statistics(
    client_info = Depends(get_current_client),
    db: Session = Depends(get_db)
):
    """Get usage statistics for the client."""
    try:
        client, _ = client_info
        
        # Get API key usage
        api_keys = db.query(APIKey).filter(
            APIKey.client_id == client.id,
            APIKey.is_active == True
        ).all()
        
        total_requests = sum(key.usage_count for key in api_keys)
        
        # Get webhook statistics
        webhooks = db.query(WebhookEndpoint).filter(
            WebhookEndpoint.client_id == client.id
        ).all()
        
        total_webhooks = len(webhooks)
        total_deliveries = sum(w.total_deliveries for w in webhooks)
        successful_deliveries = sum(w.successful_deliveries for w in webhooks)
        
        return {
            "client_id": client.client_id,
            "api_requests": {
                "total": total_requests,
                "active_keys": len(api_keys)
            },
            "webhooks": {
                "total_endpoints": total_webhooks,
                "total_deliveries": total_deliveries,
                "successful_deliveries": successful_deliveries,
                "success_rate": (successful_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
            },
            "account_status": {
                "is_active": client.is_active,
                "created_at": client.created_at,
                "last_used_at": client.last_used_at,
                "rate_limit_tier": client.rate_limit_tier
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting usage statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get usage statistics"
        )