"""
WebSocket server for real-time collaboration features.
Implements Socket.IO with room management, presence system, and Y.js integration.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import socketio
import redis.asyncio as redis
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Configure Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    async_mode='asgi',
    ping_timeout=60,
    ping_interval=25
)

# Redis connection for persistence and scaling
redis_client: Optional[redis.Redis] = None

@dataclass
class UserPresence:
    """User presence information."""
    user_id: str
    username: str
    avatar: Optional[str]
    color: str
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    last_seen: datetime = None
    is_active: bool = True

@dataclass
class RoomInfo:
    """Room information and metadata."""
    room_id: str
    document_id: str
    document_type: str  # 'automaton', 'grammar', 'problem', etc.
    created_at: datetime
    owner_id: str
    title: str
    description: str = ""
    max_users: int = 10
    is_public: bool = True

class CollaborationManager:
    """Manages real-time collaboration sessions."""
    
    def __init__(self):
        self.rooms: Dict[str, RoomInfo] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> user_data
        self.room_users: Dict[str, Dict[str, UserPresence]] = {}  # room_id -> {user_id: presence}
        self.document_cache: Dict[str, Dict[str, Any]] = {}  # document_id -> document_data
        self.user_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
            "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA"
        ]
        self.color_index = 0

    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis connection."""
        global redis_client
        try:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            redis_client = None

    def get_next_color(self) -> str:
        """Get next available color for user."""
        color = self.user_colors[self.color_index % len(self.user_colors)]
        self.color_index += 1
        return color

    async def create_room(self, room_data: Dict[str, Any]) -> RoomInfo:
        """Create a new collaboration room."""
        room_info = RoomInfo(
            room_id=room_data['room_id'],
            document_id=room_data['document_id'],
            document_type=room_data.get('document_type', 'automaton'),
            created_at=datetime.utcnow(),
            owner_id=room_data['owner_id'],
            title=room_data.get('title', 'Untitled Document'),
            description=room_data.get('description', ''),
            max_users=room_data.get('max_users', 10),
            is_public=room_data.get('is_public', True)
        )
        
        self.rooms[room_info.room_id] = room_info
        self.room_users[room_info.room_id] = {}
        
        # Persist to Redis if available
        if redis_client:
            await redis_client.hset(
                f"room:{room_info.room_id}",
                mapping={
                    "data": json.dumps(asdict(room_info), default=str),
                    "created_at": room_info.created_at.isoformat()
                }
            )
            
        logger.info(f"Created room {room_info.room_id} for document {room_info.document_id}")
        return room_info

    async def join_room(self, session_id: str, room_id: str, user_data: Dict[str, Any]) -> UserPresence:
        """Add user to collaboration room."""
        if room_id not in self.rooms:
            raise ValueError(f"Room {room_id} does not exist")
            
        room = self.rooms[room_id]
        if len(self.room_users[room_id]) >= room.max_users:
            raise ValueError(f"Room {room_id} is full")
            
        # Create user presence
        user_presence = UserPresence(
            user_id=user_data['user_id'],
            username=user_data.get('username', f"User {user_data['user_id'][:8]}"),
            avatar=user_data.get('avatar'),
            color=self.get_next_color(),
            last_seen=datetime.utcnow()
        )
        
        # Store session and user data
        self.user_sessions[session_id] = {
            'user_id': user_data['user_id'],
            'room_id': room_id,
            'joined_at': datetime.utcnow()
        }
        
        self.room_users[room_id][user_data['user_id']] = user_presence
        
        # Persist to Redis
        if redis_client:
            await redis_client.hset(
                f"room:{room_id}:users",
                user_data['user_id'],
                json.dumps(asdict(user_presence), default=str)
            )
            
        logger.info(f"User {user_data['user_id']} joined room {room_id}")
        return user_presence

    async def leave_room(self, session_id: str) -> Optional[str]:
        """Remove user from their current room."""
        if session_id not in self.user_sessions:
            return None
            
        session_data = self.user_sessions[session_id]
        room_id = session_data['room_id']
        user_id = session_data['user_id']
        
        # Remove from room
        if room_id in self.room_users and user_id in self.room_users[room_id]:
            del self.room_users[room_id][user_id]
            
        # Remove session
        del self.user_sessions[session_id]
        
        # Clean up Redis
        if redis_client:
            await redis_client.hdel(f"room:{room_id}:users", user_id)
            
        logger.info(f"User {user_id} left room {room_id}")
        return room_id

    async def update_cursor(self, session_id: str, cursor_data: Dict[str, Any]):
        """Update user's cursor position."""
        if session_id not in self.user_sessions:
            return
            
        session_data = self.user_sessions[session_id]
        room_id = session_data['room_id']
        user_id = session_data['user_id']
        
        if room_id in self.room_users and user_id in self.room_users[room_id]:
            self.room_users[room_id][user_id].cursor_position = cursor_data
            self.room_users[room_id][user_id].last_seen = datetime.utcnow()

    async def update_selection(self, session_id: str, selection_data: Dict[str, Any]):
        """Update user's selection."""
        if session_id not in self.user_sessions:
            return
            
        session_data = self.user_sessions[session_id]
        room_id = session_data['room_id']
        user_id = session_data['user_id']
        
        if room_id in self.room_users and user_id in self.room_users[room_id]:
            self.room_users[room_id][user_id].selection = selection_data
            self.room_users[room_id][user_id].last_seen = datetime.utcnow()

    async def get_room_users(self, room_id: str) -> List[Dict[str, Any]]:
        """Get all users in a room."""
        if room_id not in self.room_users:
            return []
            
        users = []
        for user_presence in self.room_users[room_id].values():
            user_data = asdict(user_presence)
            user_data['last_seen'] = user_presence.last_seen.isoformat() if user_presence.last_seen else None
            users.append(user_data)
            
        return users

    async def store_document_state(self, document_id: str, state: Dict[str, Any]):
        """Store document state for Y.js synchronization."""
        self.document_cache[document_id] = {
            'state': state,
            'last_updated': datetime.utcnow(),
            'version': state.get('version', 0)
        }
        
        # Persist to Redis
        if redis_client:
            await redis_client.hset(
                f"document:{document_id}",
                mapping={
                    "state": json.dumps(state),
                    "last_updated": datetime.utcnow().isoformat(),
                    "version": str(state.get('version', 0))
                }
            )

    async def get_document_state(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document state for Y.js synchronization."""
        if document_id in self.document_cache:
            return self.document_cache[document_id]['state']
            
        # Try Redis
        if redis_client:
            state_data = await redis_client.hget(f"document:{document_id}", "state")
            if state_data:
                return json.loads(state_data)
                
        return None

# Global collaboration manager instance
collaboration_manager = CollaborationManager()

# Socket.IO event handlers
@sio.event
async def connect(sid, environ, auth):
    """Handle client connection."""
    logger.info(f"Client {sid} connected")
    
    # Authenticate user if auth token provided
    if auth and 'token' in auth:
        try:
            # Here you would validate the JWT token
            # For now, we'll extract user info from auth
            user_id = auth.get('user_id', f"anonymous_{sid[:8]}")
            logger.info(f"Authenticated user {user_id} for session {sid}")
        except Exception as e:
            logger.error(f"Authentication failed for {sid}: {e}")
            await sio.disconnect(sid)
            return False
    
    return True

@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"Client {sid} disconnected")
    
    # Leave room and notify others
    room_id = await collaboration_manager.leave_room(sid)
    if room_id:
        await sio.emit('user_left', {
            'room_id': room_id,
            'users': await collaboration_manager.get_room_users(room_id)
        }, room=room_id, skip_sid=sid)

@sio.event
async def create_room(sid, data):
    """Create a new collaboration room."""
    try:
        room_info = await collaboration_manager.create_room(data)
        await sio.emit('room_created', {
            'success': True,
            'room': asdict(room_info)
        }, room=sid)
        
        logger.info(f"Room {room_info.room_id} created by {sid}")
    except Exception as e:
        logger.error(f"Failed to create room: {e}")
        await sio.emit('room_created', {
            'success': False,
            'error': str(e)
        }, room=sid)

@sio.event
async def join_room(sid, data):
    """Join a collaboration room."""
    try:
        room_id = data['room_id']
        user_data = data['user']
        
        user_presence = await collaboration_manager.join_room(sid, room_id, user_data)
        
        # Join Socket.IO room
        await sio.enter_room(sid, room_id)
        
        # Get all users in room
        room_users = await collaboration_manager.get_room_users(room_id)
        
        # Notify user of successful join
        await sio.emit('joined_room', {
            'success': True,
            'room_id': room_id,
            'user': asdict(user_presence),
            'users': room_users
        }, room=sid)
        
        # Notify others of new user
        await sio.emit('user_joined', {
            'room_id': room_id,
            'user': asdict(user_presence),
            'users': room_users
        }, room=room_id, skip_sid=sid)
        
        # Send current document state if available
        document_id = collaboration_manager.rooms[room_id].document_id
        document_state = await collaboration_manager.get_document_state(document_id)
        if document_state:
            await sio.emit('document_state', {
                'document_id': document_id,
                'state': document_state
            }, room=sid)
        
        logger.info(f"User {user_data['user_id']} joined room {room_id}")
        
    except Exception as e:
        logger.error(f"Failed to join room: {e}")
        await sio.emit('joined_room', {
            'success': False,
            'error': str(e)
        }, room=sid)

@sio.event
async def leave_room(sid, data):
    """Leave current room."""
    try:
        room_id = await collaboration_manager.leave_room(sid)
        if room_id:
            await sio.leave_room(sid, room_id)
            
            # Notify others
            await sio.emit('user_left', {
                'room_id': room_id,
                'users': await collaboration_manager.get_room_users(room_id)
            }, room=room_id)
            
        await sio.emit('left_room', {'success': True}, room=sid)
        
    except Exception as e:
        logger.error(f"Failed to leave room: {e}")
        await sio.emit('left_room', {
            'success': False,
            'error': str(e)
        }, room=sid)

@sio.event
async def cursor_update(sid, data):
    """Update user's cursor position."""
    try:
        await collaboration_manager.update_cursor(sid, data['cursor'])
        
        # Get user's room
        if sid in collaboration_manager.user_sessions:
            room_id = collaboration_manager.user_sessions[sid]['room_id']
            user_id = collaboration_manager.user_sessions[sid]['user_id']
            
            # Broadcast cursor update
            await sio.emit('cursor_updated', {
                'user_id': user_id,
                'cursor': data['cursor']
            }, room=room_id, skip_sid=sid)
            
    except Exception as e:
        logger.error(f"Failed to update cursor: {e}")

@sio.event
async def selection_update(sid, data):
    """Update user's selection."""
    try:
        await collaboration_manager.update_selection(sid, data['selection'])
        
        # Get user's room
        if sid in collaboration_manager.user_sessions:
            room_id = collaboration_manager.user_sessions[sid]['room_id']
            user_id = collaboration_manager.user_sessions[sid]['user_id']
            
            # Broadcast selection update
            await sio.emit('selection_updated', {
                'user_id': user_id,
                'selection': data['selection']
            }, room=room_id, skip_sid=sid)
            
    except Exception as e:
        logger.error(f"Failed to update selection: {e}")

@sio.event
async def document_change(sid, data):
    """Handle document changes for Y.js synchronization."""
    try:
        if sid in collaboration_manager.user_sessions:
            room_id = collaboration_manager.user_sessions[sid]['room_id']
            user_id = collaboration_manager.user_sessions[sid]['user_id']
            document_id = data['document_id']
            
            # Store document state
            await collaboration_manager.store_document_state(document_id, data['state'])
            
            # Broadcast changes to other users
            await sio.emit('document_changed', {
                'document_id': document_id,
                'changes': data['changes'],
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }, room=room_id, skip_sid=sid)
            
    except Exception as e:
        logger.error(f"Failed to handle document change: {e}")

@sio.event
async def sync_request(sid, data):
    """Handle Y.js sync request."""
    try:
        document_id = data['document_id']
        document_state = await collaboration_manager.get_document_state(document_id)
        
        await sio.emit('sync_response', {
            'document_id': document_id,
            'state': document_state,
            'timestamp': datetime.utcnow().isoformat()
        }, room=sid)
        
    except Exception as e:
        logger.error(f"Failed to handle sync request: {e}")

@sio.event
async def awareness_update(sid, data):
    """Handle awareness updates (cursors, selections, etc.)."""
    try:
        if sid in collaboration_manager.user_sessions:
            room_id = collaboration_manager.user_sessions[sid]['room_id']
            user_id = collaboration_manager.user_sessions[sid]['user_id']
            
            # Update awareness info
            if 'cursor' in data:
                await collaboration_manager.update_cursor(sid, data['cursor'])
            if 'selection' in data:
                await collaboration_manager.update_selection(sid, data['selection'])
            
            # Broadcast awareness update
            await sio.emit('awareness_updated', {
                'user_id': user_id,
                'awareness': data,
                'timestamp': datetime.utcnow().isoformat()
            }, room=room_id, skip_sid=sid)
            
    except Exception as e:
        logger.error(f"Failed to handle awareness update: {e}")

# ASGI app for FastAPI integration
socket_app = socketio.ASGIApp(sio, socketio_path='/ws/socket.io')

async def init_websocket_server():
    """Initialize the WebSocket server."""
    try:
        await collaboration_manager.initialize_redis()
        logger.info("WebSocket server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket server: {e}")
        raise

async def cleanup_websocket_server():
    """Clean up WebSocket server resources."""
    try:
        if redis_client:
            await redis_client.close()
        logger.info("WebSocket server cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up WebSocket server: {e}")

# Health check for WebSocket server
async def websocket_health_check() -> Dict[str, Any]:
    """Get WebSocket server health status."""
    try:
        active_rooms = len(collaboration_manager.rooms)
        total_users = sum(len(users) for users in collaboration_manager.room_users.values())
        
        redis_status = "healthy"
        if redis_client:
            try:
                await redis_client.ping()
            except Exception:
                redis_status = "unhealthy"
        else:
            redis_status = "not_configured"
        
        return {
            "status": "healthy",
            "active_rooms": active_rooms,
            "total_users": total_users,
            "redis_status": redis_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }