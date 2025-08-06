/**
 * Collaborative Canvas Component
 * Extends AutomataCanvas with real-time collaboration features
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useCollaboration } from './CollaborationProvider';
import { UserPresence } from '../hooks/useWebSocket';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { Button } from './ui/button';
import { Users, Share2, Eye, EyeOff } from 'lucide-react';

interface CollaborativeCanvasProps {
  documentId: string;
  documentType: 'automaton' | 'grammar' | 'problem' | 'solution';
  className?: string;
  onDocumentChange?: (changes: any) => void;
  children?: React.ReactNode;
}

interface CursorPosition {
  x: number;
  y: number;
  user: UserPresence;
}

export function CollaborativeCanvas({
  documentId,
  documentType,
  className = '',
  onDocumentChange,
  children
}: CollaborativeCanvasProps) {
  const {
    isConnected,
    currentDocument,
    roomUsers,
    isInRoom,
    updateCursor,
    updateSelection,
    createDocument,
    switchDocument,
    getDocument
  } = useCollaboration();
  
  // Local state
  const [cursors, setCursors] = useState<Map<string, CursorPosition>>(new Map());
  const [showCursors, setShowCursors] = useState(true);
  const [isMouseOver, setIsMouseOver] = useState(false);
  
  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);
  const mousePosition = useRef({ x: 0, y: 0 });
  const updateTimeoutRef = useRef<NodeJS.Timeout>();
  
  // Initialize or switch to document
  useEffect(() => {
    let doc = getDocument(documentId);
    if (!doc) {
      doc = createDocument(documentId, documentType);
    } else {
      switchDocument(documentId);
    }
  }, [documentId, documentType, getDocument, createDocument, switchDocument]);
  
  // Handle mouse movement for cursor tracking
  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (!isConnected || !isInRoom || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    mousePosition.current = { x, y };
    
    // Throttle cursor updates
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    
    updateTimeoutRef.current = setTimeout(() => {
      updateCursor({
        x: x / rect.width,  // Normalize to 0-1 range
        y: y / rect.height,
        element: documentId
      });
    }, 50); // Update every 50ms
  }, [isConnected, isInRoom, documentId, updateCursor]);
  
  // Handle mouse enter/leave
  const handleMouseEnter = useCallback(() => {
    setIsMouseOver(true);
  }, []);
  
  const handleMouseLeave = useCallback(() => {
    setIsMouseOver(false);
    // Clear cursor position when mouse leaves
    if (isConnected && isInRoom) {
      updateCursor({
        x: -1, // Use -1 to indicate cursor is not in canvas
        y: -1,
        element: documentId
      });
    }
  }, [isConnected, isInRoom, documentId, updateCursor]);
  
  // Handle selection changes
  const handleSelection = useCallback((start: number, end: number) => {
    if (!isConnected || !isInRoom) return;
    
    updateSelection({
      start,
      end,
      element: documentId
    });
  }, [isConnected, isInRoom, documentId, updateSelection]);
  
  // Update cursors from room users
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const newCursors = new Map<string, CursorPosition>();
    
    roomUsers.forEach(user => {
      if (user.cursor_position && 
          user.cursor_position.element === documentId &&
          user.cursor_position.x >= 0 && 
          user.cursor_position.y >= 0) {
        
        newCursors.set(user.user_id, {
          x: user.cursor_position.x * rect.width,  // Denormalize from 0-1 range
          y: user.cursor_position.y * rect.height,
          user
        });
      }
    });
    
    setCursors(newCursors);
  }, [roomUsers, documentId]);
  
  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, []);
  
  // Render collaborative cursors
  const renderCursors = () => {
    if (!showCursors || !isMouseOver) return null;
    
    return Array.from(cursors.values()).map(({ x, y, user }) => (
      <div
        key={user.user_id}
        className="absolute pointer-events-none z-50"
        style={{
          left: x,
          top: y,
          transform: 'translate(-2px, -2px)'
        }}
      >
        {/* Cursor pointer */}
        <div
          className="w-4 h-4 relative"
          style={{ color: user.color }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M0 0L0 11.2L3.2 8L6.4 11.2L8 9.6L4.8 6.4L8 6.4L0 0Z"
              fill="currentColor"
            />
          </svg>
        </div>
        
        {/* User label */}
        <div
          className="absolute top-4 left-2 px-2 py-1 rounded text-xs text-white whitespace-nowrap"
          style={{ backgroundColor: user.color }}
        >
          {user.username}
        </div>
      </div>
    ));
  };
  
  // Render user presence indicators
  const renderPresenceIndicators = () => {
    if (!isInRoom || roomUsers.length === 0) return null;
    
    return (
      <div className="absolute top-4 right-4 z-40">
        <div className="flex items-center gap-2 bg-white/90 backdrop-blur-sm rounded-lg p-2 shadow-lg">
          <div className="flex items-center gap-1">
            <Users className="w-4 h-4" />
            <span className="text-sm font-medium">{roomUsers.length}</span>
          </div>
          
          <div className="flex -space-x-2">
            {roomUsers.slice(0, 5).map((user) => (
              <TooltipProvider key={user.user_id}>
                <Tooltip>
                  <TooltipTrigger>
                    <Avatar className="w-8 h-8 border-2 border-white">
                      <AvatarImage src={user.avatar} alt={user.username} />
                      <AvatarFallback 
                        className="text-xs"
                        style={{ backgroundColor: user.color, color: 'white' }}
                      >
                        {user.username.slice(0, 2).toUpperCase()}
                      </AvatarFallback>
                    </Avatar>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: user.color }}
                      />
                      <span>{user.username}</span>
                      {user.is_active && (
                        <Badge variant="secondary" className="text-xs">
                          Active
                        </Badge>
                      )}
                    </div>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ))}
            
            {roomUsers.length > 5 && (
              <div className="w-8 h-8 rounded-full bg-gray-200 border-2 border-white flex items-center justify-center">
                <span className="text-xs font-medium">+{roomUsers.length - 5}</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowCursors(!showCursors)}
              className="h-8 w-8 p-0"
            >
              {showCursors ? (
                <Eye className="w-4 h-4" />
              ) : (
                <EyeOff className="w-4 h-4" />
              )}
            </Button>
            
            {isConnected && (
              <div className="w-2 h-2 bg-green-500 rounded-full" title="Connected" />
            )}
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div 
      ref={canvasRef}
      className={`relative w-full h-full ${className}`}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Connection status indicator */}
      {!isConnected && (
        <div className="absolute top-4 left-4 z-40">
          <Badge variant="destructive" className="flex items-center gap-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            Disconnected
          </Badge>
        </div>
      )}
      
      {/* Main content */}
      {children}
      
      {/* Collaborative cursors */}
      {renderCursors()}
      
      {/* User presence indicators */}
      {renderPresenceIndicators()}
      
      {/* Selection overlay (for future implementation) */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Selection highlights would go here */}
      </div>
    </div>
  );
}

// Hook for using collaborative canvas features in other components
export function useCollaborativeCanvas(documentId: string) {
  const {
    currentDocument,
    updateDocument,
    exportDocument,
    importDocument
  } = useCollaboration();
  
  const updateCanvas = useCallback((changes: any) => {
    updateDocument(documentId, changes);
  }, [documentId, updateDocument]);
  
  const exportCanvas = useCallback(() => {
    return exportDocument(documentId);
  }, [documentId, exportDocument]);
  
  const importCanvas = useCallback((data: any) => {
    importDocument(documentId, data);
  }, [documentId, importDocument]);
  
  const getCanvasData = useCallback(() => {
    if (!currentDocument || currentDocument.id !== documentId) {
      return null;
    }
    
    return Object.fromEntries(currentDocument.data.entries());
  }, [currentDocument, documentId]);
  
  return {
    canvasData: getCanvasData(),
    updateCanvas,
    exportCanvas,
    importCanvas
  };
}