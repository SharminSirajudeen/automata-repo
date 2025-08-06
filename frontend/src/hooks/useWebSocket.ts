/**
 * WebSocket hook for real-time collaboration features.
 * Provides Socket.IO client with room management and presence system.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';

export interface UserPresence {
  user_id: string;
  username: string;
  avatar?: string;
  color: string;
  cursor_position?: {
    x: number;
    y: number;
    element?: string;
  };
  selection?: {
    start: number;
    end: number;
    element?: string;
  };
  last_seen?: string;
  is_active: boolean;
}

export interface RoomInfo {
  room_id: string;
  document_id: string;
  document_type: string;
  created_at: string;
  owner_id: string;
  title: string;
  description: string;
  max_users: number;
  is_public: boolean;
}

export interface DocumentChange {
  document_id: string;
  changes: any[];
  user_id: string;
  timestamp: string;
}

export interface WebSocketEvents {
  // Room events
  onRoomCreated?: (data: { success: boolean; room?: RoomInfo; error?: string }) => void;
  onJoinedRoom?: (data: { success: boolean; room_id: string; user: UserPresence; users: UserPresence[] }) => void;
  onLeftRoom?: (data: { success: boolean }) => void;
  onUserJoined?: (data: { room_id: string; user: UserPresence; users: UserPresence[] }) => void;
  onUserLeft?: (data: { room_id: string; users: UserPresence[] }) => void;
  
  // Document events
  onDocumentChanged?: (data: DocumentChange) => void;
  onDocumentState?: (data: { document_id: string; state: any }) => void;
  onSyncResponse?: (data: { document_id: string; state: any; timestamp: string }) => void;
  
  // Awareness events
  onCursorUpdated?: (data: { user_id: string; cursor: any }) => void;
  onSelectionUpdated?: (data: { user_id: string; selection: any }) => void;
  onAwarenessUpdated?: (data: { user_id: string; awareness: any; timestamp: string }) => void;
  
  // Connection events
  onConnected?: () => void;
  onDisconnected?: () => void;
  onReconnected?: () => void;
  onError?: (error: any) => void;
}

export interface UseWebSocketOptions {
  url?: string;
  auth?: {
    token?: string;
    user_id?: string;
  };
  autoConnect?: boolean;
  reconnection?: boolean;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
}

export interface UseWebSocketReturn {
  // Connection state
  socket: Socket | null;
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: string | null;
  
  // Room state
  currentRoom: RoomInfo | null;
  roomUsers: UserPresence[];
  isInRoom: boolean;
  
  // Methods
  connect: () => void;
  disconnect: () => void;
  createRoom: (roomData: Partial<RoomInfo>) => Promise<void>;
  joinRoom: (roomId: string, userData: Partial<UserPresence>) => Promise<void>;
  leaveRoom: () => Promise<void>;
  
  // Document methods
  sendDocumentChange: (documentId: string, changes: any[], state: any) => void;
  requestSync: (documentId: string) => void;
  
  // Awareness methods
  updateCursor: (cursor: { x: number; y: number; element?: string }) => void;
  updateSelection: (selection: { start: number; end: number; element?: string }) => void;
  updateAwareness: (awareness: any) => void;
  
  // Utility methods
  sendMessage: (event: string, data: any) => void;
}

export function useWebSocket(
  events: WebSocketEvents = {},
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    url = process.env.NODE_ENV === 'production' 
      ? `${window.location.protocol}//${window.location.host}/ws` 
      : 'http://localhost:8000/ws',
    auth,
    autoConnect = true,
    reconnection = true,
    reconnectionAttempts = 5,
    reconnectionDelay = 1000
  } = options;

  // Connection state
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  
  // Room state
  const [currentRoom, setCurrentRoom] = useState<RoomInfo | null>(null);
  const [roomUsers, setRoomUsers] = useState<UserPresence[]>([]);
  const [isInRoom, setIsInRoom] = useState(false);
  
  // Refs to prevent stale closures
  const eventsRef = useRef(events);
  const reconnectAttempts = useRef(0);
  
  // Update events ref when events change
  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  // Connect to WebSocket server
  const connect = useCallback(() => {
    if (socket?.connected) return;
    
    setIsConnecting(true);
    setConnectionError(null);
    
    const socketOptions: any = {
      transports: ['websocket', 'polling'],
      reconnection,
      reconnectionAttempts,
      reconnectionDelay,
      timeout: 20000,
    };
    
    if (auth?.token) {
      socketOptions.auth = {
        token: auth.token,
        user_id: auth.user_id || uuidv4()
      };
    }
    
    const newSocket = io(url, socketOptions);
    
    // Connection event handlers
    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setIsConnecting(false);
      setConnectionError(null);
      reconnectAttempts.current = 0;
      eventsRef.current.onConnected?.();
    });
    
    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setIsConnected(false);
      setIsConnecting(false);
      setCurrentRoom(null);
      setRoomUsers([]);
      setIsInRoom(false);
      eventsRef.current.onDisconnected?.();
    });
    
    newSocket.on('reconnect', () => {
      console.log('WebSocket reconnected');
      setIsConnected(true);
      setIsConnecting(false);
      setConnectionError(null);
      eventsRef.current.onReconnected?.();
    });
    
    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setIsConnecting(false);
      setConnectionError(error.message);
      reconnectAttempts.current++;
      eventsRef.current.onError?.(error);
    });
    
    // Room event handlers
    newSocket.on('room_created', (data) => {
      eventsRef.current.onRoomCreated?.(data);
    });
    
    newSocket.on('joined_room', (data) => {
      if (data.success) {
        setIsInRoom(true);
        setRoomUsers(data.users);
        // currentRoom will be set by the caller
      }
      eventsRef.current.onJoinedRoom?.(data);
    });
    
    newSocket.on('left_room', (data) => {
      if (data.success) {
        setIsInRoom(false);
        setCurrentRoom(null);
        setRoomUsers([]);
      }
      eventsRef.current.onLeftRoom?.(data);
    });
    
    newSocket.on('user_joined', (data) => {
      setRoomUsers(data.users);
      eventsRef.current.onUserJoined?.(data);
    });
    
    newSocket.on('user_left', (data) => {
      setRoomUsers(data.users);
      eventsRef.current.onUserLeft?.(data);
    });
    
    // Document event handlers
    newSocket.on('document_changed', (data) => {
      eventsRef.current.onDocumentChanged?.(data);
    });
    
    newSocket.on('document_state', (data) => {
      eventsRef.current.onDocumentState?.(data);
    });
    
    newSocket.on('sync_response', (data) => {
      eventsRef.current.onSyncResponse?.(data);
    });
    
    // Awareness event handlers
    newSocket.on('cursor_updated', (data) => {
      eventsRef.current.onCursorUpdated?.(data);
    });
    
    newSocket.on('selection_updated', (data) => {
      eventsRef.current.onSelectionUpdated?.(data);
    });
    
    newSocket.on('awareness_updated', (data) => {
      eventsRef.current.onAwarenessUpdated?.(data);
    });
    
    setSocket(newSocket);
  }, [url, auth, reconnection, reconnectionAttempts, reconnectionDelay]);
  
  // Disconnect from WebSocket server
  const disconnect = useCallback(() => {
    if (socket) {
      socket.disconnect();
      setSocket(null);
      setIsConnected(false);
      setIsConnecting(false);
      setCurrentRoom(null);
      setRoomUsers([]);
      setIsInRoom(false);
    }
  }, [socket]);
  
  // Create a new room
  const createRoom = useCallback(async (roomData: Partial<RoomInfo>) => {
    if (!socket?.connected) {
      throw new Error('WebSocket not connected');
    }
    
    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Create room timeout'));
      }, 10000);
      
      const handleResponse = (data: { success: boolean; room?: RoomInfo; error?: string }) => {
        clearTimeout(timeout);
        if (data.success && data.room) {
          setCurrentRoom(data.room);
          resolve();
        } else {
          reject(new Error(data.error || 'Failed to create room'));
        }
      };
      
      socket.once('room_created', handleResponse);
      socket.emit('create_room', {
        ...roomData,
        room_id: roomData.room_id || `room_${uuidv4()}`,
        document_id: roomData.document_id || `doc_${uuidv4()}`,
      });
    });
  }, [socket]);
  
  // Join an existing room
  const joinRoom = useCallback(async (roomId: string, userData: Partial<UserPresence>) => {
    if (!socket?.connected) {
      throw new Error('WebSocket not connected');
    }
    
    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Join room timeout'));
      }, 10000);
      
      const handleResponse = (data: { success: boolean; room_id: string; user: UserPresence; users: UserPresence[]; error?: string }) => {
        clearTimeout(timeout);
        if (data.success) {
          resolve();
        } else {
          reject(new Error(data.error || 'Failed to join room'));
        }
      };
      
      socket.once('joined_room', handleResponse);
      socket.emit('join_room', {
        room_id: roomId,
        user: {
          user_id: userData.user_id || auth?.user_id || uuidv4(),
          username: userData.username || 'Anonymous',
          avatar: userData.avatar,
          ...userData
        }
      });
    });
  }, [socket, auth]);
  
  // Leave current room
  const leaveRoom = useCallback(async () => {
    if (!socket?.connected || !isInRoom) {
      return;
    }
    
    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Leave room timeout'));
      }, 5000);
      
      const handleResponse = (data: { success: boolean; error?: string }) => {
        clearTimeout(timeout);
        if (data.success) {
          resolve();
        } else {
          reject(new Error(data.error || 'Failed to leave room'));
        }
      };
      
      socket.once('left_room', handleResponse);
      socket.emit('leave_room', {});
    });
  }, [socket, isInRoom]);
  
  // Send document changes
  const sendDocumentChange = useCallback((documentId: string, changes: any[], state: any) => {
    if (!socket?.connected || !isInRoom) return;
    
    socket.emit('document_change', {
      document_id: documentId,
      changes,
      state
    });
  }, [socket, isInRoom]);
  
  // Request document sync
  const requestSync = useCallback((documentId: string) => {
    if (!socket?.connected) return;
    
    socket.emit('sync_request', {
      document_id: documentId
    });
  }, [socket]);
  
  // Update cursor position
  const updateCursor = useCallback((cursor: { x: number; y: number; element?: string }) => {
    if (!socket?.connected || !isInRoom) return;
    
    socket.emit('cursor_update', { cursor });
  }, [socket, isInRoom]);
  
  // Update selection
  const updateSelection = useCallback((selection: { start: number; end: number; element?: string }) => {
    if (!socket?.connected || !isInRoom) return;
    
    socket.emit('selection_update', { selection });
  }, [socket, isInRoom]);
  
  // Update awareness info
  const updateAwareness = useCallback((awareness: any) => {
    if (!socket?.connected || !isInRoom) return;
    
    socket.emit('awareness_update', awareness);
  }, [socket, isInRoom]);
  
  // Send custom message
  const sendMessage = useCallback((event: string, data: any) => {
    if (!socket?.connected) return;
    
    socket.emit(event, data);
  }, [socket]);
  
  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, [socket]);
  
  return {
    // Connection state
    socket,
    isConnected,
    isConnecting,
    connectionError,
    
    // Room state
    currentRoom,
    roomUsers,
    isInRoom,
    
    // Methods
    connect,
    disconnect,
    createRoom,
    joinRoom,
    leaveRoom,
    
    // Document methods
    sendDocumentChange,
    requestSync,
    
    // Awareness methods
    updateCursor,
    updateSelection,
    updateAwareness,
    
    // Utility methods
    sendMessage
  };
}