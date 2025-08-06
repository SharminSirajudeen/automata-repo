"""
WebSocket router for real-time collaboration API endpoints.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..websocket_server import collaboration_manager, websocket_health_check
from ..auth import get_current_user
from ..security import rate_limit_general

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket", "collaboration"])

# Pydantic models for WebSocket operations
class CreateRoomRequest(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    document_type: str = Field(default="automaton", description="Type of document")
    title: str = Field(..., min_length=1, max_length=100, description="Room title")
    description: str = Field(default="", max_length=500, description="Room description")
    max_users: int = Field(default=10, ge=1, le=50, description="Maximum users allowed")
    is_public: bool = Field(default=True, description="Whether room is public")

class RoomInfo(BaseModel):
    room_id: str
    document_id: str
    document_type: str
    created_at: datetime
    owner_id: str
    title: str
    description: str
    max_users: int
    is_public: bool

class UserPresence(BaseModel):
    user_id: str
    username: str
    avatar: Optional[str]
    color: str
    cursor_position: Optional[Dict[str, Any]]
    selection: Optional[Dict[str, Any]]
    last_seen: Optional[datetime]
    is_active: bool

class RoomUsers(BaseModel):
    room_id: str
    users: List[UserPresence]
    total_users: int

@router.post("/rooms", response_model=RoomInfo)
@rate_limit_general()
async def create_room(
    room_request: CreateRoomRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new collaboration room."""
    try:
        # Generate unique room ID
        import uuid
        room_id = f"room_{uuid.uuid4().hex[:12]}"
        
        room_data = {
            "room_id": room_id,
            "document_id": room_request.document_id,
            "document_type": room_request.document_type,
            "owner_id": current_user["user_id"],
            "title": room_request.title,
            "description": room_request.description,
            "max_users": room_request.max_users,
            "is_public": room_request.is_public
        }
        
        room_info = await collaboration_manager.create_room(room_data)
        
        return RoomInfo(
            room_id=room_info.room_id,
            document_id=room_info.document_id,
            document_type=room_info.document_type,
            created_at=room_info.created_at,
            owner_id=room_info.owner_id,
            title=room_info.title,
            description=room_info.description,
            max_users=room_info.max_users,
            is_public=room_info.is_public
        )
        
    except Exception as e:
        logger.error(f"Failed to create room: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rooms", response_model=List[RoomInfo])
@rate_limit_general()
async def list_rooms(
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    is_public: Optional[bool] = Query(None, description="Filter by public rooms"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of rooms to return")
):
    """List available collaboration rooms."""
    try:
        rooms = []
        for room in collaboration_manager.rooms.values():
            # Apply filters
            if document_type and room.document_type != document_type:
                continue
            if is_public is not None and room.is_public != is_public:
                continue
                
            rooms.append(RoomInfo(
                room_id=room.room_id,
                document_id=room.document_id,
                document_type=room.document_type,
                created_at=room.created_at,
                owner_id=room.owner_id,
                title=room.title,
                description=room.description,
                max_users=room.max_users,
                is_public=room.is_public
            ))
        
        # Sort by creation time (newest first) and apply limit
        rooms.sort(key=lambda r: r.created_at, reverse=True)
        return rooms[:limit]
        
    except Exception as e:
        logger.error(f"Failed to list rooms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rooms/{room_id}", response_model=RoomInfo)
@rate_limit_general()
async def get_room(room_id: str):
    """Get specific room information."""
    try:
        if room_id not in collaboration_manager.rooms:
            raise HTTPException(status_code=404, detail="Room not found")
            
        room = collaboration_manager.rooms[room_id]
        return RoomInfo(
            room_id=room.room_id,
            document_id=room.document_id,
            document_type=room.document_type,
            created_at=room.created_at,
            owner_id=room.owner_id,
            title=room.title,
            description=room.description,
            max_users=room.max_users,
            is_public=room.is_public
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get room {room_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rooms/{room_id}/users", response_model=RoomUsers)
@rate_limit_general()
async def get_room_users(room_id: str):
    """Get all users currently in a room."""
    try:
        if room_id not in collaboration_manager.rooms:
            raise HTTPException(status_code=404, detail="Room not found")
            
        users_data = await collaboration_manager.get_room_users(room_id)
        users = [UserPresence(**user) for user in users_data]
        
        return RoomUsers(
            room_id=room_id,
            users=users,
            total_users=len(users)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get users for room {room_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rooms/{room_id}")
@rate_limit_general()
async def delete_room(
    room_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a collaboration room (owner only)."""
    try:
        if room_id not in collaboration_manager.rooms:
            raise HTTPException(status_code=404, detail="Room not found")
            
        room = collaboration_manager.rooms[room_id]
        if room.owner_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Only room owner can delete the room")
        
        # Remove room and notify users
        del collaboration_manager.rooms[room_id]
        if room_id in collaboration_manager.room_users:
            del collaboration_manager.room_users[room_id]
        
        # TODO: Emit room_deleted event to all connected clients
        
        return {"message": "Room deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete room {room_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}/state")
@rate_limit_general()
async def get_document_state(document_id: str):
    """Get current document state for synchronization."""
    try:
        state = await collaboration_manager.get_document_state(document_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Document state not found")
            
        return {
            "document_id": document_id,
            "state": state,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document state for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/{document_id}/state")
@rate_limit_general()
async def update_document_state(
    document_id: str,
    state: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Update document state."""
    try:
        await collaboration_manager.store_document_state(document_id, state)
        
        return {
            "message": "Document state updated successfully",
            "document_id": document_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update document state for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def websocket_health():
    """Get WebSocket server health status."""
    try:
        return await websocket_health_check()
    except Exception as e:
        logger.error(f"WebSocket health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
@rate_limit_general()
async def get_collaboration_stats():
    """Get collaboration statistics."""
    try:
        total_rooms = len(collaboration_manager.rooms)
        active_users = sum(len(users) for users in collaboration_manager.room_users.values())
        
        # Room statistics by type
        room_types = {}
        for room in collaboration_manager.rooms.values():
            room_types[room.document_type] = room_types.get(room.document_type, 0) + 1
        
        return {
            "total_rooms": total_rooms,
            "active_users": active_users,
            "room_types": room_types,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get collaboration stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))