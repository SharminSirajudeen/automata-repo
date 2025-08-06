/**
 * Collaboration Provider for real-time collaborative editing.
 * Provides Y.js integration, conflict resolution, and user presence.
 */

import React, { 
  createContext, 
  useContext, 
  useState, 
  useEffect, 
  useCallback, 
  useRef,
  ReactNode 
} from 'react';
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { useWebSocket, UserPresence, RoomInfo, WebSocketEvents } from '../hooks/useWebSocket';
import { useToast } from '../hooks/use-toast';

// Y.js document types
export interface CollaborativeDocument {
  id: string;
  type: 'automaton' | 'grammar' | 'problem' | 'solution';
  ydoc: Y.Doc;
  provider?: WebsocketProvider;
  data: Y.Map<any>;
  awareness?: Y.Map<any>;
}

export interface CollaborationContextType {
  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: string | null;
  
  // Room state
  currentRoom: RoomInfo | null;
  roomUsers: UserPresence[];
  isInRoom: boolean;
  
  // Document state
  documents: Map<string, CollaborativeDocument>;
  currentDocument: CollaborativeDocument | null;
  
  // Collaboration methods
  createRoom: (roomData: {
    title: string;
    description?: string;
    documentType: string;
    isPublic?: boolean;
  }) => Promise<void>;
  joinRoom: (roomId: string, userData?: Partial<UserPresence>) => Promise<void>;
  leaveRoom: () => Promise<void>;
  
  // Document methods
  createDocument: (id: string, type: CollaborativeDocument['type'], initialData?: any) => CollaborativeDocument;
  getDocument: (id: string) => CollaborativeDocument | null;
  switchDocument: (id: string) => void;
  updateDocument: (id: string, updates: any) => void;
  
  // Awareness methods
  updateCursor: (position: { x: number; y: number; element?: string }) => void;
  updateSelection: (selection: { start: number; end: number; element?: string }) => void;
  updateUserStatus: (status: Partial<UserPresence>) => void;
  
  // Conflict resolution
  resolveConflict: (documentId: string, strategy: 'last-write-wins' | 'first-write-wins' | 'merge') => void;
  
  // Utilities
  exportDocument: (documentId: string) => any;
  importDocument: (documentId: string, data: any) => void;
}

const CollaborationContext = createContext<CollaborationContextType | null>(null);

export interface CollaborationProviderProps {
  children: ReactNode;
  userId?: string;
  username?: string;
  avatar?: string;
  wsUrl?: string;
  autoConnect?: boolean;
}

export function CollaborationProvider({
  children,
  userId,
  username = 'Anonymous User',
  avatar,
  wsUrl,
  autoConnect = true
}: CollaborationProviderProps) {
  const { toast } = useToast();
  
  // Document state
  const [documents, setDocuments] = useState<Map<string, CollaborativeDocument>>(new Map());
  const [currentDocument, setCurrentDocument] = useState<CollaborativeDocument | null>(null);
  
  // Refs for stable references
  const documentsRef = useRef(documents);
  const conflictResolutionQueue = useRef<Set<string>>(new Set());
  
  // Update ref when documents change
  useEffect(() => {
    documentsRef.current = documents;
  }, [documents]);
  
  // WebSocket event handlers
  const wsEvents: WebSocketEvents = {
    onConnected: () => {
      toast({
        title: "Connected",
        description: "Real-time collaboration is now active",
      });
    },
    
    onDisconnected: () => {
      toast({
        title: "Disconnected",
        description: "Connection lost. Attempting to reconnect...",
        variant: "destructive"
      });
    },
    
    onReconnected: () => {
      toast({
        title: "Reconnected",
        description: "Real-time collaboration restored",
      });
    },
    
    onError: (error) => {
      console.error('WebSocket error:', error);
      toast({
        title: "Connection Error",
        description: error.message || "Failed to connect to collaboration server",
        variant: "destructive"
      });
    },
    
    onUserJoined: (data) => {
      const user = data.user;
      toast({
        title: "User Joined",
        description: `${user.username} joined the collaboration`,
      });
    },
    
    onUserLeft: (data) => {
      // We don't have user info in this event, so just show generic message
      toast({
        title: "User Left",
        description: "A user left the collaboration",
      });
    },
    
    onDocumentChanged: (data) => {
      handleDocumentChange(data);
    },
    
    onDocumentState: (data) => {
      handleDocumentState(data);
    },
    
    onSyncResponse: (data) => {
      handleSyncResponse(data);
    },
    
    onAwarenessUpdated: (data) => {
      handleAwarenessUpdate(data);
    }
  };
  
  // Initialize WebSocket
  const {
    isConnected,
    isConnecting,
    connectionError,
    currentRoom,
    roomUsers,
    isInRoom,
    createRoom: wsCreateRoom,
    joinRoom: wsJoinRoom,
    leaveRoom: wsLeaveRoom,
    sendDocumentChange,
    requestSync,
    updateCursor: wsUpdateCursor,
    updateSelection: wsUpdateSelection,
    updateAwareness
  } = useWebSocket(wsEvents, {
    url: wsUrl,
    auth: {
      user_id: userId,
    },
    autoConnect
  });
  
  // Handle document changes from WebSocket
  const handleDocumentChange = useCallback((data: any) => {
    const doc = documentsRef.current.get(data.document_id);
    if (!doc) return;
    
    try {
      // Apply Y.js update
      if (data.changes && data.changes.length > 0) {
        // Convert changes to Y.js updates and apply them
        // This is a simplified version - in reality, you'd need proper Y.js update encoding
        doc.data.set('lastUpdate', {
          timestamp: data.timestamp,
          user: data.user_id,
          changes: data.changes
        });
      }
    } catch (error) {
      console.error('Failed to apply document change:', error);
      // Queue for conflict resolution
      conflictResolutionQueue.current.add(data.document_id);
    }
  }, []);
  
  // Handle document state from WebSocket
  const handleDocumentState = useCallback((data: any) => {
    const doc = documentsRef.current.get(data.document_id);
    if (!doc || !data.state) return;
    
    try {
      // Update document with server state
      Object.entries(data.state).forEach(([key, value]) => {
        doc.data.set(key, value);
      });
    } catch (error) {
      console.error('Failed to apply document state:', error);
    }
  }, []);
  
  // Handle sync response from WebSocket
  const handleSyncResponse = useCallback((data: any) => {
    if (data.state) {
      handleDocumentState(data);
    }
  }, [handleDocumentState]);
  
  // Handle awareness updates
  const handleAwarenessUpdate = useCallback((data: any) => {
    // Update user awareness in Y.js documents
    documentsRef.current.forEach((doc) => {
      if (doc.awareness) {
        doc.awareness.set(data.user_id, {
          ...data.awareness,
          timestamp: data.timestamp
        });
      }
    });
  }, []);
  
  // Create a new room
  const createRoom = useCallback(async (roomData: {
    title: string;
    description?: string;
    documentType: string;
    isPublic?: boolean;
  }) => {
    try {
      await wsCreateRoom({
        title: roomData.title,
        description: roomData.description || '',
        document_type: roomData.documentType,
        is_public: roomData.isPublic ?? true,
        max_users: 10
      });
    } catch (error) {
      console.error('Failed to create room:', error);
      throw error;
    }
  }, [wsCreateRoom]);
  
  // Join an existing room
  const joinRoom = useCallback(async (roomId: string, userData?: Partial<UserPresence>) => {
    try {
      await wsJoinRoom(roomId, {
        user_id: userId,
        username,
        avatar,
        ...userData
      });
    } catch (error) {
      console.error('Failed to join room:', error);
      throw error;
    }
  }, [wsJoinRoom, userId, username, avatar]);
  
  // Leave current room
  const leaveRoom = useCallback(async () => {
    try {
      await wsLeaveRoom();
      // Clean up documents
      documents.forEach((doc) => {
        if (doc.provider) {
          doc.provider.destroy();
        }
      });
      setDocuments(new Map());
      setCurrentDocument(null);
    } catch (error) {
      console.error('Failed to leave room:', error);
      throw error;
    }
  }, [wsLeaveRoom, documents]);
  
  // Create a new collaborative document
  const createDocument = useCallback((
    id: string, 
    type: CollaborativeDocument['type'], 
    initialData?: any
  ): CollaborativeDocument => {
    // Create Y.js document
    const ydoc = new Y.Doc();
    const data = ydoc.getMap('data');
    const awareness = ydoc.getMap('awareness');
    
    // Initialize with data
    if (initialData) {
      Object.entries(initialData).forEach(([key, value]) => {
        data.set(key, value);
      });
    }
    
    // Set document metadata
    data.set('id', id);
    data.set('type', type);
    data.set('created_at', new Date().toISOString());
    data.set('version', 1);
    
    const doc: CollaborativeDocument = {
      id,
      type,
      ydoc,
      data,
      awareness
    };
    
    // Set up Y.js event handlers
    data.observe((event) => {
      if (isInRoom && currentRoom) {
        // Convert Y.js event to changes format
        const changes = Array.from(event.changes.keys.entries()).map(([key, change]) => ({
          key,
          action: change.action,
          oldValue: change.oldValue,
          newValue: data.get(key)
        }));
        
        // Send changes via WebSocket
        sendDocumentChange(id, changes, Object.fromEntries(data.entries()));
      }
    });
    
    // Add to documents map
    setDocuments(prev => new Map(prev).set(id, doc));
    
    return doc;
  }, [isInRoom, currentRoom, sendDocumentChange]);
  
  // Get document by ID
  const getDocument = useCallback((id: string): CollaborativeDocument | null => {
    return documents.get(id) || null;
  }, [documents]);
  
  // Switch to a different document
  const switchDocument = useCallback((id: string) => {
    const doc = documents.get(id);
    if (doc) {
      setCurrentDocument(doc);
      // Request sync from server
      requestSync(id);
    }
  }, [documents, requestSync]);
  
  // Update document data
  const updateDocument = useCallback((id: string, updates: any) => {
    const doc = documents.get(id);
    if (!doc) return;
    
    // Apply updates to Y.js document
    Object.entries(updates).forEach(([key, value]) => {
      doc.data.set(key, value);
    });
  }, [documents]);
  
  // Update cursor position
  const updateCursor = useCallback((position: { x: number; y: number; element?: string }) => {
    wsUpdateCursor(position);
    
    // Update local awareness
    if (currentDocument?.awareness) {
      currentDocument.awareness.set(userId || 'anonymous', {
        cursor: position,
        timestamp: new Date().toISOString()
      });
    }
  }, [wsUpdateCursor, currentDocument, userId]);
  
  // Update selection
  const updateSelection = useCallback((selection: { start: number; end: number; element?: string }) => {
    wsUpdateSelection(selection);
    
    // Update local awareness
    if (currentDocument?.awareness) {
      currentDocument.awareness.set(userId || 'anonymous', {
        selection,
        timestamp: new Date().toISOString()
      });
    }
  }, [wsUpdateSelection, currentDocument, userId]);
  
  // Update user status
  const updateUserStatus = useCallback((status: Partial<UserPresence>) => {
    updateAwareness({
      ...status,
      timestamp: new Date().toISOString()
    });
  }, [updateAwareness]);
  
  // Resolve conflicts
  const resolveConflict = useCallback((
    documentId: string, 
    strategy: 'last-write-wins' | 'first-write-wins' | 'merge'
  ) => {
    const doc = documents.get(documentId);
    if (!doc) return;
    
    try {
      switch (strategy) {
        case 'last-write-wins':
          // Request latest state from server
          requestSync(documentId);
          break;
          
        case 'first-write-wins':
          // Keep current local state, don't sync
          break;
          
        case 'merge':
          // Implement merge strategy
          // This would require more sophisticated conflict resolution
          requestSync(documentId);
          break;
      }
      
      // Remove from conflict queue
      conflictResolutionQueue.current.delete(documentId);
      
      toast({
        title: "Conflict Resolved",
        description: `Used ${strategy} strategy for document ${documentId}`,
      });
    } catch (error) {
      console.error('Failed to resolve conflict:', error);
      toast({
        title: "Conflict Resolution Failed",
        description: "Failed to resolve document conflict",
        variant: "destructive"
      });
    }
  }, [documents, requestSync, toast]);
  
  // Export document
  const exportDocument = useCallback((documentId: string) => {
    const doc = documents.get(documentId);
    if (!doc) return null;
    
    return {
      id: documentId,
      type: doc.type,
      data: Object.fromEntries(doc.data.entries()),
      awareness: Object.fromEntries(doc.awareness?.entries() || []),
      exported_at: new Date().toISOString()
    };
  }, [documents]);
  
  // Import document
  const importDocument = useCallback((documentId: string, data: any) => {
    let doc = documents.get(documentId);
    if (!doc && data.type) {
      doc = createDocument(documentId, data.type);
    }
    
    if (doc && data.data) {
      Object.entries(data.data).forEach(([key, value]) => {
        doc!.data.set(key, value);
      });
    }
  }, [documents, createDocument]);
  
  // Context value
  const contextValue: CollaborationContextType = {
    // Connection state
    isConnected,
    isConnecting,
    connectionError,
    
    // Room state
    currentRoom,
    roomUsers,
    isInRoom,
    
    // Document state
    documents,
    currentDocument,
    
    // Collaboration methods
    createRoom,
    joinRoom,
    leaveRoom,
    
    // Document methods
    createDocument,
    getDocument,
    switchDocument,
    updateDocument,
    
    // Awareness methods
    updateCursor,
    updateSelection,
    updateUserStatus,
    
    // Conflict resolution
    resolveConflict,
    
    // Utilities
    exportDocument,
    importDocument
  };
  
  return (
    <CollaborationContext.Provider value={contextValue}>
      {children}
    </CollaborationContext.Provider>
  );
}

export function useCollaboration(): CollaborationContextType {
  const context = useContext(CollaborationContext);
  if (!context) {
    throw new Error('useCollaboration must be used within a CollaborationProvider');
  }
  return context;
}