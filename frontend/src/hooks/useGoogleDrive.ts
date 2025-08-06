import { useState, useEffect, useCallback } from 'react';
import { GoogleDriveStorage } from '@/services/googleDriveStorage';
import { useToast } from '@/components/ui/use-toast';

interface UseGoogleDriveReturn {
  isSignedIn: boolean;
  isLoading: boolean;
  user: { email: string; name: string } | null;
  signIn: () => Promise<void>;
  signOut: () => Promise<void>;
  saveProgress: (progress: any) => Promise<void>;
  loadProgress: () => Promise<any>;
  lastSyncTime: Date | null;
}

export const useGoogleDrive = (): UseGoogleDriveReturn => {
  const [isSignedIn, setIsSignedIn] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState<{ email: string; name: string } | null>(null);
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null);
  const { toast } = useToast();
  const storage = GoogleDriveStorage.getInstance();

  // Initialize Google Drive API
  useEffect(() => {
    const init = async () => {
      try {
        await storage.initialize();
        const signedIn = storage.isSignedIn();
        setIsSignedIn(signedIn);
        if (signedIn) {
          setUser(storage.getCurrentUser());
        }
      } catch (error) {
        console.error('Failed to initialize Google Drive:', error);
        toast({
          title: "Google Drive Unavailable",
          description: "You can still use the app with local storage.",
          variant: "default"
        });
      } finally {
        setIsLoading(false);
      }
    };
    init();
  }, []);

  const signIn = useCallback(async () => {
    try {
      setIsLoading(true);
      await storage.signIn();
      setIsSignedIn(true);
      setUser(storage.getCurrentUser());
      
      toast({
        title: "Connected to Google Drive! 🎉",
        description: "Your progress will now sync automatically.",
        variant: "default"
      });

      // Try to load existing progress
      const existingProgress = await storage.loadProgress();
      if (existingProgress) {
        toast({
          title: "Progress Restored",
          description: `Welcome back! Loaded ${existingProgress.statistics.completedProblems} completed problems.`,
          variant: "default"
        });
      }
    } catch (error) {
      console.error('Sign in failed:', error);
      toast({
        title: "Connection Failed",
        description: "Could not connect to Google Drive. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const signOut = useCallback(async () => {
    try {
      setIsLoading(true);
      await storage.signOut();
      setIsSignedIn(false);
      setUser(null);
      setLastSyncTime(null);
      
      toast({
        title: "Disconnected",
        description: "Your progress is now stored locally only.",
        variant: "default"
      });
    } catch (error) {
      console.error('Sign out failed:', error);
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const saveProgress = useCallback(async (progress: any) => {
    if (!isSignedIn) {
      throw new Error('User must be signed in to save progress');
    }

    try {
      await storage.saveProgress(progress);
      setLastSyncTime(new Date());
      
      // Silent save - don't show toast for automatic saves
      console.log('Progress saved to Google Drive');
    } catch (error) {
      console.error('Save failed:', error);
      toast({
        title: "Sync Failed",
        description: "Could not save to Google Drive. Progress saved locally.",
        variant: "destructive"
      });
      throw error;
    }
  }, [isSignedIn, toast]);

  const loadProgress = useCallback(async () => {
    if (!isSignedIn) {
      throw new Error('User must be signed in to load progress');
    }

    try {
      const progress = await storage.loadProgress();
      if (progress) {
        setLastSyncTime(new Date(progress.lastUpdated));
      }
      return progress;
    } catch (error) {
      console.error('Load failed:', error);
      toast({
        title: "Load Failed",
        description: "Could not load from Google Drive.",
        variant: "destructive"
      });
      throw error;
    }
  }, [isSignedIn, toast]);

  return {
    isSignedIn,
    isLoading,
    user,
    signIn,
    signOut,
    saveProgress,
    loadProgress,
    lastSyncTime
  };
};