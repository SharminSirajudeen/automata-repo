/**
 * Room Manager Component
 * Handles creating, joining, and managing collaboration rooms
 */

import React, { useState, useEffect } from 'react';
import { useCollaboration } from './CollaborationProvider';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Switch } from './ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { useToast } from '../hooks/use-toast';
import { 
  Plus, 
  Users, 
  Globe, 
  Lock, 
  Copy, 
  ExternalLink, 
  Settings,
  LogOut,
  Trash2
} from 'lucide-react';

interface RoomInfo {
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

export function RoomManager() {
  const {
    currentRoom,
    isInRoom,
    roomUsers,
    isConnected,
    createRoom,
    joinRoom,
    leaveRoom
  } = useCollaboration();
  
  const { toast } = useToast();
  
  // State
  const [availableRooms, setAvailableRooms] = useState<RoomInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showJoinDialog, setShowJoinDialog] = useState(false);
  
  // Create room form state
  const [createForm, setCreateForm] = useState({
    title: '',
    description: '',
    documentType: 'automaton',
    isPublic: true,
    maxUsers: 10
  });
  
  // Join room form state
  const [joinForm, setJoinForm] = useState({
    roomId: '',
    username: ''
  });
  
  // Fetch available rooms
  const fetchRooms = async () => {
    if (!isConnected) return;
    
    try {
      const response = await fetch('/ws/rooms');
      if (response.ok) {
        const rooms = await response.json();
        setAvailableRooms(rooms);
      }
    } catch (error) {
      console.error('Failed to fetch rooms:', error);
    }
  };
  
  // Load rooms on component mount
  useEffect(() => {
    fetchRooms();
  }, [isConnected]);
  
  // Handle create room
  const handleCreateRoom = async () => {
    if (!createForm.title.trim()) {
      toast({
        title: "Validation Error",
        description: "Room title is required",
        variant: "destructive"
      });
      return;
    }
    
    setIsLoading(true);
    try {
      await createRoom({
        title: createForm.title,
        description: createForm.description,
        documentType: createForm.documentType,
        isPublic: createForm.isPublic
      });
      
      toast({
        title: "Room Created",
        description: `Successfully created room "${createForm.title}"`,
      });
      
      setShowCreateDialog(false);
      setCreateForm({
        title: '',
        description: '',
        documentType: 'automaton',
        isPublic: true,
        maxUsers: 10
      });
      
      await fetchRooms();
    } catch (error) {
      toast({
        title: "Failed to Create Room",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle join room
  const handleJoinRoom = async (roomId?: string) => {
    const targetRoomId = roomId || joinForm.roomId;
    
    if (!targetRoomId.trim()) {
      toast({
        title: "Validation Error",
        description: "Room ID is required",
        variant: "destructive"
      });
      return;
    }
    
    setIsLoading(true);
    try {
      await joinRoom(targetRoomId, {
        username: joinForm.username || 'Anonymous User'
      });
      
      toast({
        title: "Joined Room",
        description: `Successfully joined room ${targetRoomId}`,
      });
      
      if (!roomId) { // Only close dialog if called from form
        setShowJoinDialog(false);
        setJoinForm({ roomId: '', username: '' });
      }
    } catch (error) {
      toast({
        title: "Failed to Join Room",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle leave room
  const handleLeaveRoom = async () => {
    if (!isInRoom) return;
    
    setIsLoading(true);
    try {
      await leaveRoom();
      toast({
        title: "Left Room",
        description: "Successfully left the collaboration room",
      });
      await fetchRooms();
    } catch (error) {
      toast({
        title: "Failed to Leave Room",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Copy room ID to clipboard
  const copyRoomId = (roomId: string) => {
    navigator.clipboard.writeText(roomId);
    toast({
      title: "Copied",
      description: "Room ID copied to clipboard",
    });
  };
  
  // Render create room dialog
  const renderCreateDialog = () => (
    <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
      <DialogTrigger asChild>
        <Button className="flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Create Room
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create Collaboration Room</DialogTitle>
          <DialogDescription>
            Set up a new room for real-time collaboration
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4">
          <div>
            <Label htmlFor="title">Room Title</Label>
            <Input
              id="title"
              value={createForm.title}
              onChange={(e) => setCreateForm(prev => ({ ...prev, title: e.target.value }))}
              placeholder="Enter room title"
            />
          </div>
          
          <div>
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={createForm.description}
              onChange={(e) => setCreateForm(prev => ({ ...prev, description: e.target.value }))}
              placeholder="Optional room description"
              rows={3}
            />
          </div>
          
          <div>
            <Label htmlFor="documentType">Document Type</Label>
            <Select 
              value={createForm.documentType} 
              onValueChange={(value) => setCreateForm(prev => ({ ...prev, documentType: value }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="automaton">Automaton</SelectItem>
                <SelectItem value="grammar">Grammar</SelectItem>
                <SelectItem value="problem">Problem</SelectItem>
                <SelectItem value="solution">Solution</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="flex items-center space-x-2">
            <Switch
              id="isPublic"
              checked={createForm.isPublic}
              onCheckedChange={(checked) => setCreateForm(prev => ({ ...prev, isPublic: checked }))}
            />
            <Label htmlFor="isPublic">Public Room</Label>
          </div>
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
            Cancel
          </Button>
          <Button onClick={handleCreateRoom} disabled={isLoading}>
            {isLoading ? 'Creating...' : 'Create Room'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
  
  // Render join room dialog
  const renderJoinDialog = () => (
    <Dialog open={showJoinDialog} onOpenChange={setShowJoinDialog}>
      <DialogTrigger asChild>
        <Button variant="outline" className="flex items-center gap-2">
          <ExternalLink className="w-4 h-4" />
          Join Room
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Join Collaboration Room</DialogTitle>
          <DialogDescription>
            Enter a room ID to join an existing collaboration
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4">
          <div>
            <Label htmlFor="roomId">Room ID</Label>
            <Input
              id="roomId"
              value={joinForm.roomId}
              onChange={(e) => setJoinForm(prev => ({ ...prev, roomId: e.target.value }))}
              placeholder="Enter room ID"
            />
          </div>
          
          <div>
            <Label htmlFor="username">Your Name</Label>
            <Input
              id="username"
              value={joinForm.username}
              onChange={(e) => setJoinForm(prev => ({ ...prev, username: e.target.value }))}
              placeholder="Enter your name"
            />
          </div>
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowJoinDialog(false)}>
            Cancel
          </Button>
          <Button onClick={() => handleJoinRoom()} disabled={isLoading}>
            {isLoading ? 'Joining...' : 'Join Room'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
  
  // Render current room info
  const renderCurrentRoom = () => {
    if (!isInRoom || !currentRoom) return null;
    
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>{currentRoom.title}</span>
            <div className="flex items-center gap-2">
              <Badge variant={currentRoom.is_public ? "default" : "secondary"}>
                {currentRoom.is_public ? (
                  <>
                    <Globe className="w-3 h-3 mr-1" />
                    Public
                  </>
                ) : (
                  <>
                    <Lock className="w-3 h-3 mr-1" />
                    Private
                  </>
                )}
              </Badge>
            </div>
          </CardTitle>
          <CardDescription>
            {currentRoom.description || 'No description'}
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Room ID:</span>
              <div className="flex items-center gap-2">
                <code className="text-sm bg-muted px-2 py-1 rounded">
                  {currentRoom.room_id}
                </code>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyRoomId(currentRoom.room_id)}
                >
                  <Copy className="w-4 h-4" />
                </Button>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Users:</span>
              <div className="flex items-center gap-1">
                <Users className="w-4 h-4" />
                <span>{roomUsers.length}/{currentRoom.max_users}</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Document Type:</span>
              <Badge variant="outline">{currentRoom.document_type}</Badge>
            </div>
          </div>
        </CardContent>
        
        <CardFooter>
          <Button
            variant="destructive"
            onClick={handleLeaveRoom}
            disabled={isLoading}
            className="w-full"
          >
            <LogOut className="w-4 h-4 mr-2" />
            Leave Room
          </Button>
        </CardFooter>
      </Card>
    );
  };
  
  // Render available rooms
  const renderAvailableRooms = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Available Rooms</h3>
        <Button variant="outline" size="sm" onClick={fetchRooms}>
          Refresh
        </Button>
      </div>
      
      {availableRooms.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              No public rooms available
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {availableRooms.map((room) => (
            <Card key={room.room_id}>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center justify-between">
                  <span>{room.title}</span>
                  <Badge variant="outline">{room.document_type}</Badge>
                </CardTitle>
                {room.description && (
                  <CardDescription>{room.description}</CardDescription>
                )}
              </CardHeader>
              
              <CardContent className="pt-0">
                <div className="flex items-center justify-between text-sm text-muted-foreground">
                  <div className="flex items-center gap-4">
                    <span>ID: {room.room_id}</span>
                    <div className="flex items-center gap-1">
                      <Users className="w-3 h-3" />
                      <span>0/{room.max_users}</span>
                    </div>
                  </div>
                  
                  <Button
                    size="sm"
                    onClick={() => handleJoinRoom(room.room_id)}
                    disabled={isLoading || isInRoom}
                  >
                    Join
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
  
  if (!isConnected) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center">
            <div className="text-muted-foreground">
              Connecting to collaboration server...
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Action buttons */}
      <div className="flex gap-2">
        {renderCreateDialog()}
        {renderJoinDialog()}
      </div>
      
      {/* Current room or available rooms */}
      <Tabs value={isInRoom ? "current" : "available"}>
        <TabsList>
          <TabsTrigger value="current" disabled={!isInRoom}>
            Current Room
          </TabsTrigger>
          <TabsTrigger value="available">
            Available Rooms
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="current" className="mt-4">
          {renderCurrentRoom()}
        </TabsContent>
        
        <TabsContent value="available" className="mt-4">
          {renderAvailableRooms()}
        </TabsContent>
      </Tabs>
    </div>
  );
}