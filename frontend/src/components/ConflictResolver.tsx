/**
 * Conflict Resolver Component
 * Handles conflict detection and resolution for collaborative editing
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useCollaboration } from './CollaborationProvider';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { RadioGroup, RadioGroupItem } from './ui/radio-group';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ScrollArea } from './ui/scroll-area';
import { useToast } from '../hooks/use-toast';
import { 
  AlertTriangle, 
  Clock, 
  User, 
  FileText,
  GitBranch,
  ChevronRight,
  CheckCircle,
  XCircle
} from 'lucide-react';

interface ConflictData {
  id: string;
  documentId: string;
  conflictType: 'concurrent_edit' | 'version_mismatch' | 'data_corruption';
  timestamp: string;
  users: Array<{
    userId: string;
    username: string;
    changes: any[];
    timestamp: string;
  }>;
  affectedKeys: string[];
  currentValue: any;
  conflictingValues: Array<{
    value: any;
    user: string;
    timestamp: string;
  }>;
}

type ResolutionStrategy = 'last-write-wins' | 'first-write-wins' | 'merge' | 'manual';

export function ConflictResolver() {
  const { 
    documents, 
    currentDocument, 
    roomUsers,
    resolveConflict 
  } = useCollaboration();
  
  const { toast } = useToast();
  
  // State
  const [conflicts, setConflicts] = useState<ConflictData[]>([]);
  const [selectedConflict, setSelectedConflict] = useState<ConflictData | null>(null);
  const [resolutionStrategy, setResolutionStrategy] = useState<ResolutionStrategy>('last-write-wins');
  const [showResolutionDialog, setShowResolutionDialog] = useState(false);
  const [manualResolutionValue, setManualResolutionValue] = useState<any>(null);
  
  // Detect conflicts in documents
  const detectConflicts = useCallback(() => {
    const newConflicts: ConflictData[] = [];
    
    // Check each document for conflicts
    documents.forEach((doc, documentId) => {
      try {
        // Check for version conflicts
        const lastUpdate = doc.data.get('lastUpdate');
        const version = doc.data.get('version') || 0;
        
        // Simulate conflict detection logic
        // In a real implementation, this would check Y.js update history
        
        // Check for concurrent edits
        const awareness = doc.awareness;
        if (awareness) {
          const activeUsers = Array.from(awareness.entries())
            .filter(([userId, data]: [string, any]) => {
              const userLastSeen = new Date(data.timestamp || 0);
              const now = new Date();
              return now.getTime() - userLastSeen.getTime() < 30000; // Active in last 30 seconds
            });
          
          if (activeUsers.length > 1) {
            // Check if users are editing the same elements
            const editingConflicts = detectEditingConflicts(activeUsers, doc);
            newConflicts.push(...editingConflicts);
          }
        }
        
        // Check for data inconsistencies
        const dataConflicts = detectDataConflicts(doc);
        newConflicts.push(...dataConflicts);
        
      } catch (error) {
        console.error(`Failed to check conflicts for document ${documentId}:`, error);
      }
    });
    
    setConflicts(newConflicts);
  }, [documents]);
  
  // Detect editing conflicts between users
  const detectEditingConflicts = (activeUsers: Array<[string, any]>, doc: any): ConflictData[] => {
    const conflicts: ConflictData[] = [];
    
    // Group users by the elements they're editing
    const elementUsers: { [key: string]: Array<{ userId: string; username: string; timestamp: string }> } = {};
    
    activeUsers.forEach(([userId, userData]) => {
      const user = roomUsers.find(u => u.user_id === userId);
      if (!user) return;
      
      const editingElement = userData.selection?.element || userData.cursor?.element;
      if (editingElement) {
        if (!elementUsers[editingElement]) {
          elementUsers[editingElement] = [];
        }
        elementUsers[editingElement].push({
          userId,
          username: user.username,
          timestamp: userData.timestamp
        });
      }
    });
    
    // Find elements with multiple users
    Object.entries(elementUsers).forEach(([element, users]) => {
      if (users.length > 1) {
        conflicts.push({
          id: `concurrent_${doc.id}_${element}_${Date.now()}`,
          documentId: doc.id,
          conflictType: 'concurrent_edit',
          timestamp: new Date().toISOString(),
          users: users.map(u => ({
            userId: u.userId,
            username: u.username,
            changes: [], // Would contain actual changes
            timestamp: u.timestamp
          })),
          affectedKeys: [element],
          currentValue: doc.data.get(element),
          conflictingValues: users.map(u => ({
            value: doc.data.get(element), // Would be different for each user
            user: u.username,
            timestamp: u.timestamp
          }))
        });
      }
    });
    
    return conflicts;
  };
  
  // Detect data conflicts in document
  const detectDataConflicts = (doc: any): ConflictData[] => {
    const conflicts: ConflictData[] = [];
    
    try {
      // Check for version mismatches
      const version = doc.data.get('version') || 0;
      const lastUpdate = doc.data.get('lastUpdate');
      
      if (lastUpdate && lastUpdate.version && lastUpdate.version !== version) {
        conflicts.push({
          id: `version_mismatch_${doc.id}_${Date.now()}`,
          documentId: doc.id,
          conflictType: 'version_mismatch',
          timestamp: new Date().toISOString(),
          users: [{
            userId: 'system',
            username: 'System',
            changes: [],
            timestamp: new Date().toISOString()
          }],
          affectedKeys: ['version'],
          currentValue: version,
          conflictingValues: [{
            value: lastUpdate.version,
            user: 'Server',
            timestamp: lastUpdate.timestamp
          }]
        });
      }
      
    } catch (error) {
      console.error('Failed to detect data conflicts:', error);
    }
    
    return conflicts;
  };
  
  // Run conflict detection periodically
  useEffect(() => {
    const interval = setInterval(detectConflicts, 5000); // Check every 5 seconds
    return () => clearInterval(interval);
  }, [detectConflicts]);
  
  // Handle conflict resolution
  const handleResolveConflict = async (conflict: ConflictData, strategy: ResolutionStrategy) => {
    try {
      if (strategy === 'manual') {
        // Open manual resolution dialog
        setSelectedConflict(conflict);
        setManualResolutionValue(conflict.currentValue);
        setShowResolutionDialog(true);
        return;
      }
      
      // Apply automatic resolution
      resolveConflict(conflict.documentId, strategy);
      
      // Remove conflict from list
      setConflicts(prev => prev.filter(c => c.id !== conflict.id));
      
      toast({
        title: "Conflict Resolved",
        description: `Applied ${strategy} resolution strategy`,
      });
      
    } catch (error) {
      toast({
        title: "Resolution Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
    }
  };
  
  // Handle manual resolution
  const handleManualResolution = async () => {
    if (!selectedConflict) return;
    
    try {
      // Apply manual resolution value
      const doc = documents.get(selectedConflict.documentId);
      if (doc) {
        selectedConflict.affectedKeys.forEach(key => {
          doc.data.set(key, manualResolutionValue);
        });
      }
      
      // Remove conflict from list
      setConflicts(prev => prev.filter(c => c.id !== selectedConflict.id));
      setShowResolutionDialog(false);
      setSelectedConflict(null);
      
      toast({
        title: "Conflict Resolved",
        description: "Applied manual resolution",
      });
      
    } catch (error) {
      toast({
        title: "Manual Resolution Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
    }
  };
  
  // Render conflict item
  const renderConflictItem = (conflict: ConflictData) => (
    <Card key={conflict.id} className="mb-4">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-base">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-500" />
            <span>
              {conflict.conflictType === 'concurrent_edit' && 'Concurrent Edit'}
              {conflict.conflictType === 'version_mismatch' && 'Version Mismatch'}
              {conflict.conflictType === 'data_corruption' && 'Data Corruption'}
            </span>
          </div>
          <Badge variant={
            conflict.conflictType === 'concurrent_edit' ? 'default' :
            conflict.conflictType === 'version_mismatch' ? 'secondary' : 'destructive'
          }>
            {conflict.conflictType.replace('_', ' ')}
          </Badge>
        </CardTitle>
        <CardDescription className="flex items-center gap-2">
          <Clock className="w-3 h-3" />
          {new Date(conflict.timestamp).toLocaleString()}
          <span>•</span>
          <FileText className="w-3 h-3" />
          {conflict.documentId}
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-medium mb-2">Affected Elements:</h4>
            <div className="flex flex-wrap gap-1">
              {conflict.affectedKeys.map(key => (
                <Badge key={key} variant="outline" className="text-xs">
                  {key}
                </Badge>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-medium mb-2">Involved Users:</h4>
            <div className="flex flex-wrap gap-2">
              {conflict.users.map(user => (
                <div key={user.userId} className="flex items-center gap-1">
                  <User className="w-3 h-3" />
                  <span className="text-sm">{user.username}</span>
                </div>
              ))}
            </div>
          </div>
          
          {conflict.conflictingValues.length > 0 && (
            <div>
              <h4 className="text-sm font-medium mb-2">Conflicting Values:</h4>
              <div className="space-y-1">
                {conflict.conflictingValues.map((value, index) => (
                  <div key={index} className="flex items-center justify-between text-sm bg-muted p-2 rounded">
                    <span className="truncate flex-1">{JSON.stringify(value.value)}</span>
                    <span className="text-muted-foreground ml-2">{value.user}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
      
      <CardFooter className="pt-0">
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            onClick={() => handleResolveConflict(conflict, 'last-write-wins')}
          >
            Last Write Wins
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleResolveConflict(conflict, 'first-write-wins')}
          >
            First Write Wins
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleResolveConflict(conflict, 'merge')}
          >
            Auto Merge
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => handleResolveConflict(conflict, 'manual')}
          >
            Manual
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
  
  // Render manual resolution dialog
  const renderManualResolutionDialog = () => (
    <Dialog open={showResolutionDialog} onOpenChange={setShowResolutionDialog}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Manual Conflict Resolution</DialogTitle>
          <DialogDescription>
            Choose how to resolve this conflict manually
          </DialogDescription>
        </DialogHeader>
        
        {selectedConflict && (
          <div className="space-y-4">
            <div>
              <Label>Conflicting Values:</Label>
              <Tabs defaultValue="0" className="mt-2">
                <TabsList>
                  {selectedConflict.conflictingValues.map((_, index) => (
                    <TabsTrigger key={index} value={index.toString()}>
                      Option {index + 1}
                    </TabsTrigger>
                  ))}
                </TabsList>
                {selectedConflict.conflictingValues.map((value, index) => (
                  <TabsContent key={index} value={index.toString()}>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{value.user}</span>
                            <span className="text-sm text-muted-foreground">
                              {new Date(value.timestamp).toLocaleString()}
                            </span>
                          </div>
                          <ScrollArea className="h-32 w-full border rounded p-2 font-mono text-sm">
                            <pre>{JSON.stringify(value.value, null, 2)}</pre>
                          </ScrollArea>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>
                ))}
              </Tabs>
            </div>
            
            <div>
              <Label>Resolution Strategy:</Label>
              <RadioGroup
                value={resolutionStrategy}
                onValueChange={(value: ResolutionStrategy) => setResolutionStrategy(value)}
                className="mt-2"
              >
                {selectedConflict.conflictingValues.map((value, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <RadioGroupItem 
                      value={`use-${index}`}
                      id={`use-${index}`}
                    />
                    <Label htmlFor={`use-${index}`}>
                      Use {value.user}'s version
                    </Label>
                  </div>
                ))}
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="custom" id="custom" />
                  <Label htmlFor="custom">
                    Use custom value
                  </Label>
                </div>
              </RadioGroup>
            </div>
          </div>
        )}
        
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowResolutionDialog(false)}>
            Cancel
          </Button>
          <Button onClick={handleManualResolution}>
            Apply Resolution
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
  
  // If no conflicts, show empty state
  if (conflicts.length === 0) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center">
            <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No Conflicts Detected</h3>
            <p className="text-muted-foreground">
              All documents are in sync and no conflicts have been found.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          {conflicts.length} conflict{conflicts.length !== 1 ? 's' : ''} detected.
          Please resolve them to continue collaborating safely.
        </AlertDescription>
      </Alert>
      
      <div className="space-y-4">
        {conflicts.map(renderConflictItem)}
      </div>
      
      {renderManualResolutionDialog()}
    </div>
  );
}