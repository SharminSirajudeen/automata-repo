/**
 * Collaborative Workspace Component
 * Main interface for real-time collaboration features
 */

import React, { useState, useEffect } from 'react';
import { useCollaboration } from './CollaborationProvider';
import { CollaborativeCanvas } from './CollaborativeCanvas';
import { RoomManager } from './RoomManager';
import { ConflictResolver } from './ConflictResolver';
import AutomataCanvas from './AutomataCanvas';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { useToast } from '../hooks/use-toast';
import { 
  Users, 
  FileText, 
  Settings, 
  AlertCircle,
  Wifi,
  WifiOff,
  Download,
  Upload,
  Share2
} from 'lucide-react';

interface CollaborativeWorkspaceProps {
  problem?: any; // The current problem being worked on
  onProblemChange?: (problem: any) => void;
}

export function CollaborativeWorkspace({ 
  problem,
  onProblemChange 
}: CollaborativeWorkspaceProps) {
  const {
    isConnected,
    isInRoom,
    currentRoom,
    roomUsers,
    currentDocument,
    createDocument,
    switchDocument,
    exportDocument,
    importDocument
  } = useCollaboration();
  
  const { toast } = useToast();
  
  // Local state
  const [activeTab, setActiveTab] = useState('canvas');
  const [documentId, setDocumentId] = useState<string>('');
  
  // Initialize document ID from problem
  useEffect(() => {
    if (problem?.id) {
      const docId = `problem_${problem.id}`;
      setDocumentId(docId);
      
      // Create document if it doesn't exist
      if (!currentDocument || currentDocument.id !== docId) {
        createDocument(docId, 'automaton', {
          problemId: problem.id,
          title: problem.title,
          description: problem.description,
          initialData: problem
        });
        switchDocument(docId);
      }
    } else {
      // Generate random document ID for new problems
      const docId = `doc_${Date.now()}`;
      setDocumentId(docId);
      createDocument(docId, 'automaton');
    }
  }, [problem, currentDocument, createDocument, switchDocument]);
  
  // Handle document export
  const handleExport = () => {
    try {
      const exportedData = exportDocument(documentId);
      if (exportedData) {
        const blob = new Blob([JSON.stringify(exportedData, null, 2)], {
          type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${documentId}_${new Date().getTime()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        toast({
          title: "Document Exported",
          description: "Document has been downloaded successfully",
        });
      }
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export document",
        variant: "destructive"
      });
    }
  };
  
  // Handle document import
  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const data = JSON.parse(e.target?.result as string);
            importDocument(documentId, data);
            toast({
              title: "Document Imported",
              description: "Document has been imported successfully",
            });
          } catch (error) {
            toast({
              title: "Import Failed",
              description: "Failed to parse imported file",
              variant: "destructive"
            });
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };
  
  // Handle share document
  const handleShare = async () => {
    if (!currentRoom) {
      toast({
        title: "Not in Room",
        description: "Join a room first to share documents",
        variant: "destructive"
      });
      return;
    }
    
    try {
      await navigator.clipboard.writeText(currentRoom.room_id);
      toast({
        title: "Room ID Copied",
        description: "Share this ID with others to collaborate",
      });
    } catch (error) {
      toast({
        title: "Share Failed",
        description: "Failed to copy room ID",
        variant: "destructive"
      });
    }
  };
  
  // Render connection status
  const renderConnectionStatus = () => (
    <div className="flex items-center gap-2">
      {isConnected ? (
        <>
          <Wifi className="w-4 h-4 text-green-500" />
          <span className="text-sm text-green-600">Connected</span>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4 text-red-500" />
          <span className="text-sm text-red-600">Disconnected</span>
        </>
      )}
      
      {isInRoom && (
        <>
          <span className="text-muted-foreground">•</span>
          <Users className="w-4 h-4" />
          <span className="text-sm">{roomUsers.length} user{roomUsers.length !== 1 ? 's' : ''}</span>
        </>
      )}
    </div>
  );
  
  // Render toolbar
  const renderToolbar = () => (
    <div className="flex items-center justify-between p-4 border-b">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold">
          {currentDocument?.data.get('title') || 'Collaborative Workspace'}
        </h2>
        {renderConnectionStatus()}
      </div>
      
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={handleImport}
          className="flex items-center gap-2"
        >
          <Upload className="w-4 h-4" />
          Import
        </Button>
        
        <Button
          variant="outline"
          size="sm"
          onClick={handleExport}
          className="flex items-center gap-2"
          disabled={!currentDocument}
        >
          <Download className="w-4 h-4" />
          Export
        </Button>
        
        <Button
          variant="outline"
          size="sm"
          onClick={handleShare}
          className="flex items-center gap-2"
          disabled={!isInRoom}
        >
          <Share2 className="w-4 h-4" />
          Share
        </Button>
      </div>
    </div>
  );
  
  // Render collaborative canvas
  const renderCanvas = () => (
    <CollaborativeCanvas
      documentId={documentId}
      documentType="automaton"
      className="h-[600px] border rounded-lg"
    >
      <AutomataCanvas
        problem={problem}
        onSolutionChange={(solution) => {
          if (currentDocument) {
            currentDocument.data.set('solution', solution);
            onProblemChange?.({ ...problem, solution });
          }
        }}
        className="w-full h-full"
      />
    </CollaborativeCanvas>
  );
  
  // Main workspace content
  const renderWorkspace = () => {
    if (!isConnected) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <WifiOff className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Connection Required</h3>
              <p className="text-muted-foreground">
                Please connect to the collaboration server to use real-time features.
              </p>
            </div>
          </CardContent>
        </Card>
      );
    }
    
    if (!isInRoom) {
      return (
        <div className="space-y-6">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Join or create a collaboration room to start working together in real-time.
            </AlertDescription>
          </Alert>
          <RoomManager />
        </div>
      );
    }
    
    return (
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="canvas">Canvas</TabsTrigger>
          <TabsTrigger value="conflicts">Conflicts</TabsTrigger>
          <TabsTrigger value="room">Room</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>
        
        <TabsContent value="canvas" className="mt-4">
          {renderCanvas()}
        </TabsContent>
        
        <TabsContent value="conflicts" className="mt-4">
          <ConflictResolver />
        </TabsContent>
        
        <TabsContent value="room" className="mt-4">
          <RoomManager />
        </TabsContent>
        
        <TabsContent value="settings" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Collaboration Settings</CardTitle>
              <CardDescription>
                Configure your collaboration preferences
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Document Information</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Document ID:</span>
                      <code className="bg-muted px-2 py-1 rounded">{documentId}</code>
                    </div>
                    {currentDocument && (
                      <>
                        <div className="flex justify-between">
                          <span>Document Type:</span>
                          <Badge variant="outline">{currentDocument.type}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>Version:</span>
                          <span>{currentDocument.data.get('version') || 1}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Last Updated:</span>
                          <span>
                            {new Date(currentDocument.data.get('last_updated') || Date.now())
                              .toLocaleString()}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
                
                {currentRoom && (
                  <div>
                    <h4 className="font-medium mb-2">Room Information</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Room ID:</span>
                        <code className="bg-muted px-2 py-1 rounded">{currentRoom.room_id}</code>
                      </div>
                      <div className="flex justify-between">
                        <span>Room Title:</span>
                        <span>{currentRoom.title}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Document Type:</span>
                        <span>{currentRoom.document_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Max Users:</span>
                        <span>{roomUsers.length}/{currentRoom.max_users}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    );
  };
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      {renderToolbar()}
      <div className="p-4">
        {renderWorkspace()}
      </div>
    </div>
  );
}