# WebSocket Infrastructure for Real-Time Collaboration

## Overview

This document describes the comprehensive WebSocket infrastructure implemented for real-time collaborative features in the Automata Learning Platform. The system enables multiple users to work together simultaneously on automata problems, grammars, and other theory of computation exercises.

## Architecture

### Backend Components

#### 1. WebSocket Server (`websocket_server.py`)
- **Socket.IO Integration**: Uses python-socketio for WebSocket communication
- **Room Management**: Supports creating, joining, and managing collaboration rooms
- **User Presence**: Tracks active users and their status in real-time
- **Redis Persistence**: Optional Redis integration for scaling across multiple servers
- **Authentication**: Supports JWT token authentication

**Key Features:**
- Real-time user presence tracking
- Cursor position and selection synchronization
- Document state synchronization
- Room-based collaboration
- Auto-scaling with Redis

#### 2. Y.js Integration (`yjs_integration.py`)
- **Document Synchronization**: Handles Y.js document state management
- **Conflict Resolution**: Implements multiple conflict resolution strategies
- **Version Control**: Tracks document versions and updates
- **Merge Operations**: Handles collaborative document merging

**Conflict Resolution Strategies:**
- Last Write Wins
- First Write Wins
- Intelligent Merge
- User Priority Based

#### 3. WebSocket Router (`routers/websocket_router.py`)
- **REST API**: Provides REST endpoints for room management
- **Room CRUD**: Create, read, update, delete collaboration rooms
- **User Management**: Handle user joining/leaving rooms
- **Document State**: Manage document persistence and retrieval

### Frontend Components

#### 1. WebSocket Hook (`useWebSocket.ts`)
- **Connection Management**: Handles WebSocket connection lifecycle
- **Event Handling**: Manages all WebSocket events and callbacks
- **Reconnection Logic**: Automatic reconnection with exponential backoff
- **TypeScript Support**: Fully typed interfaces for type safety

**Key Features:**
```typescript
const {
  isConnected,
  currentRoom,
  roomUsers,
  createRoom,
  joinRoom,
  leaveRoom,
  sendDocumentChange,
  updateCursor,
  updateSelection
} = useWebSocket(events, options);
```

#### 2. Collaboration Provider (`CollaborationProvider.tsx`)
- **Context Management**: React Context for global collaboration state
- **Y.js Integration**: Client-side Y.js document management
- **Real-time Updates**: Handles document synchronization
- **User Awareness**: Manages user presence and awareness data

#### 3. Collaborative Canvas (`CollaborativeCanvas.tsx`)
- **Real-time Cursors**: Shows other users' cursor positions
- **User Presence**: Visual indicators for active collaborators
- **Selection Highlighting**: Shows user selections in real-time
- **Awareness System**: Comprehensive user awareness features

#### 4. Room Manager (`RoomManager.tsx`)
- **Room Creation**: Interface for creating new collaboration rooms
- **Room Discovery**: Browse and join existing public rooms
- **Room Settings**: Configure room properties and permissions
- **User Management**: View and manage room participants

#### 5. Conflict Resolver (`ConflictResolver.tsx`)
- **Conflict Detection**: Identifies document conflicts automatically
- **Resolution Interface**: User-friendly conflict resolution UI
- **Multiple Strategies**: Support for different resolution approaches
- **Manual Resolution**: Advanced manual conflict resolution options

#### 6. Collaborative Workspace (`CollaborativeWorkspace.tsx`)
- **Integrated Interface**: Complete collaboration workspace
- **Document Management**: Import/export collaborative documents
- **Real-time Canvas**: Collaborative automata canvas
- **Tab Interface**: Organized collaboration features

## Installation and Setup

### Backend Dependencies

Add to `pyproject.toml`:
```toml
python-socketio = "^5.11.4"
python-multipart = "^0.0.16"
aiofiles = "^24.1.0"
redis = "^5.0.8"
pydantic-settings = "^2.5.2"
```

Install:
```bash
cd backend
poetry install
```

### Frontend Dependencies

Add to `package.json`:
```json
{
  "socket.io-client": "^4.8.1",
  "yjs": "^13.6.20",
  "y-websocket": "^2.0.4",
  "y-protocols": "^1.0.6",
  "@types/uuid": "^10.0.0",
  "uuid": "^11.0.3"
}
```

Install:
```bash
cd frontend
npm install
```

### Redis Setup (Optional)

For production scaling, install and configure Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Docker
docker run -d -p 6379:6379 redis:alpine
```

## Configuration

### Backend Configuration

Set environment variables:
```bash
export WEBSOCKET_URL="ws://localhost:8000/ws"
export REDIS_URL="redis://localhost:6379"
export CORS_ORIGINS="http://localhost:5173,http://localhost:3000"
```

### Frontend Configuration

Set in `.env`:
```bash
VITE_WS_URL="ws://localhost:8000/ws"
```

## Usage Examples

### Basic Collaboration Setup

```typescript
// Wrap your app with CollaborationProvider
function App() {
  return (
    <CollaborationProvider
      userId="user123"
      username="John Doe"
      autoConnect={true}
    >
      <YourAppComponents />
    </CollaborationProvider>
  );
}
```

### Create and Join Rooms

```typescript
const { createRoom, joinRoom } = useCollaboration();

// Create a room
await createRoom({
  title: "DFA Problem Session",
  description: "Working on DFA construction",
  documentType: "automaton",
  isPublic: true
});

// Join a room
await joinRoom("room_abc123", {
  username: "Jane Smith"
});
```

### Real-time Document Updates

```typescript
const { updateDocument, currentDocument } = useCollaboration();

// Update document data
updateDocument("doc_123", {
  states: newStates,
  transitions: newTransitions,
  lastModified: Date.now()
});

// Listen for changes
useEffect(() => {
  if (currentDocument) {
    currentDocument.data.observe((event) => {
      console.log('Document changed:', event);
    });
  }
}, [currentDocument]);
```

### Cursor and Selection Tracking

```typescript
const { updateCursor, updateSelection } = useCollaboration();

// Update cursor position
const handleMouseMove = (e) => {
  updateCursor({
    x: e.clientX,
    y: e.clientY,
    element: "canvas"
  });
};

// Update selection
const handleSelection = (start, end) => {
  updateSelection({
    start,
    end,
    element: "text-editor"
  });
};
```

## API Endpoints

### WebSocket Events

#### Client to Server
- `create_room`: Create new collaboration room
- `join_room`: Join existing room
- `leave_room`: Leave current room
- `document_change`: Send document changes
- `cursor_update`: Update cursor position
- `selection_update`: Update selection
- `sync_request`: Request document synchronization

#### Server to Client
- `room_created`: Room creation confirmation
- `joined_room`: Room join confirmation
- `user_joined`: New user joined notification
- `user_left`: User left notification
- `document_changed`: Document change notification
- `cursor_updated`: Cursor position update
- `selection_updated`: Selection update

### REST API Endpoints

- `POST /ws/rooms`: Create new room
- `GET /ws/rooms`: List available rooms
- `GET /ws/rooms/{room_id}`: Get room details
- `DELETE /ws/rooms/{room_id}`: Delete room
- `GET /ws/rooms/{room_id}/users`: Get room users
- `GET /ws/documents/{doc_id}/state`: Get document state
- `POST /ws/documents/{doc_id}/state`: Update document state
- `GET /ws/health`: WebSocket health check
- `GET /ws/stats`: Collaboration statistics

## Security Considerations

### Authentication
- JWT token validation for WebSocket connections
- User identity verification for room access
- Rate limiting on WebSocket events

### Authorization
- Room ownership permissions
- Public/private room access control
- Document modification permissions

### Data Protection
- Input validation for all WebSocket messages
- XSS protection in collaborative content
- CORS configuration for cross-origin requests

## Performance Optimization

### Backend Optimizations
- Redis caching for room and user data
- Connection pooling for database operations
- Message queuing for high-throughput scenarios
- Horizontal scaling with multiple server instances

### Frontend Optimizations
- Debounced cursor updates (50ms intervals)
- Efficient Y.js document synchronization
- Memory management for large documents
- Connection state optimization

### Network Optimizations
- Binary protocol for Y.js updates
- Compressed message payloads
- Efficient reconnection strategies
- Bandwidth-aware update throttling

## Monitoring and Debugging

### Health Checks
```bash
# Check WebSocket server health
curl http://localhost:8000/ws/health

# Get collaboration statistics
curl http://localhost:8000/ws/stats
```

### Debug Logging
```python
# Enable debug logging
logging.getLogger('socketio').setLevel(logging.DEBUG)
logging.getLogger('engineio').setLevel(logging.DEBUG)
```

### Browser DevTools
- WebSocket connection inspection
- Real-time message monitoring
- Performance profiling
- Memory usage tracking

## Deployment

### Production Deployment
1. Set up Redis cluster for scaling
2. Configure load balancer for WebSocket sticky sessions
3. Set up monitoring and alerting
4. Configure SSL/TLS for secure connections
5. Set appropriate CORS policies

### Docker Deployment
```dockerfile
# Backend WebSocket support
EXPOSE 8000
ENV WEBSOCKET_URL=ws://0.0.0.0:8000/ws
ENV REDIS_URL=redis://redis:6379

# Frontend build
ENV VITE_WS_URL=ws://your-domain.com/ws
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check CORS configuration
   - Verify WebSocket URL
   - Test Redis connectivity

2. **Performance Issues**
   - Monitor message frequency
   - Check Y.js document size
   - Optimize update batching

3. **Synchronization Problems**
   - Verify Y.js version compatibility
   - Check conflict resolution settings
   - Monitor network latency

### Debug Commands
```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/ws/socket.io/?EIO=4&transport=websocket

# Monitor Redis activity
redis-cli monitor

# Check backend logs
tail -f backend/logs/websocket.log
```

## Future Enhancements

### Planned Features
- Voice/video chat integration
- Screen sharing capabilities
- Advanced permission systems
- Analytics and usage tracking
- Mobile app support
- Offline synchronization

### Performance Improvements
- WebRTC for peer-to-peer communication
- CRDT alternatives to Y.js
- Advanced caching strategies
- Machine learning for conflict resolution

This WebSocket infrastructure provides a robust foundation for real-time collaboration in the Automata Learning Platform, enabling seamless multi-user interactions while maintaining data consistency and performance.