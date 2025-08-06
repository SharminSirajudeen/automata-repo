/**
 * Ultra-Efficient Google Drive Storage Service
 * Stores user progress with maximum compression and delta updates
 * Target: Keep storage under 10KB even after years of usage
 */

import { CompressionEngine, CompactProgress } from './compressionEngine';
import { DeltaManager, DeltaStorage } from './deltaStorage';
import { ProblemMapper } from './optimizedStorage';

interface UserProgress {
  version: string;
  lastUpdated: string;
  user: {
    email?: string;
    displayName?: string;
  };
  statistics: {
    totalProblems: number;
    completedProblems: number;
    currentStreak: number;
    longestStreak: number;
    totalTimeSpent: number; // in seconds
    lastActiveDate: string;
  };
  completedProblems: {
    problemId: string;
    completedAt: string;
    attempts: number;
    timeSpent: number;
    score: number;
    solution: any; // Automaton structure
  }[];
  currentProblem?: {
    problemId: string;
    startedAt: string;
    attempts: number;
    currentSolution: any;
  };
  achievements: {
    id: string;
    unlockedAt: string;
    type: string;
  }[];
  preferences: {
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    enableHints: boolean;
    enableAnimations: boolean;
  };
}

// Ultra-compact storage format
interface StoragePayload {
  v: number; // format version
  d: string; // base64-encoded compressed data
  m: string[]; // problem ID mapping
  s: number; // uncompressed size for validation
}

export class GoogleDriveStorage {
  private static instance: GoogleDriveStorage;
  private gapi: any;
  private isInitialized = false;
  private readonly APP_NAME = 'Automata Learning Tool';
  private readonly FOLDER_NAME = 'Automata Progress';
  private readonly FILE_NAME = 'automata-progress.bin';
  private readonly MAPPING_FILE = 'problem-mapping.json';
  private folderId: string | null = null;
  private lastKnownProgress: CompactProgress | null = null;

  private constructor() {}

  static getInstance(): GoogleDriveStorage {
    if (!GoogleDriveStorage.instance) {
      GoogleDriveStorage.instance = new GoogleDriveStorage();
    }
    return GoogleDriveStorage.instance;
  }

  /**
   * Initialize Google API client
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://apis.google.com/js/api.js';
      script.onload = () => {
        window.gapi.load('client:auth2', async () => {
          try {
            await window.gapi.client.init({
              apiKey: import.meta.env.VITE_GOOGLE_API_KEY,
              clientId: import.meta.env.VITE_GOOGLE_CLIENT_ID,
              discoveryDocs: ['https://www.googleapis.com/discovery/v1/apis/drive/v3/rest'],
              scope: 'https://www.googleapis.com/auth/drive.file'
            });
            this.gapi = window.gapi;
            this.isInitialized = true;
            resolve();
          } catch (error) {
            reject(error);
          }
        });
      };
      document.body.appendChild(script);
    });
  }

  /**
   * Sign in with Google
   */
  async signIn(): Promise<void> {
    if (!this.isInitialized) await this.initialize();
    
    const auth = this.gapi.auth2.getAuthInstance();
    if (!auth.isSignedIn.get()) {
      await auth.signIn();
    }
  }

  /**
   * Sign out from Google
   */
  async signOut(): Promise<void> {
    const auth = this.gapi.auth2.getAuthInstance();
    if (auth.isSignedIn.get()) {
      await auth.signOut();
    }
  }

  /**
   * Check if user is signed in
   */
  isSignedIn(): boolean {
    if (!this.isInitialized) return false;
    const auth = this.gapi.auth2.getAuthInstance();
    return auth.isSignedIn.get();
  }

  /**
   * Get current user info
   */
  getCurrentUser(): { email: string; name: string } | null {
    if (!this.isSignedIn()) return null;
    
    const user = this.gapi.auth2.getAuthInstance().currentUser.get();
    const profile = user.getBasicProfile();
    
    return {
      email: profile.getEmail(),
      name: profile.getName()
    };
  }

  /**
   * Create or get the app folder in Google Drive
   */
  private async getOrCreateFolder(): Promise<string> {
    if (this.folderId) return this.folderId;

    // Search for existing folder
    const response = await this.gapi.client.drive.files.list({
      q: `name='${this.FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false`,
      fields: 'files(id, name)',
      spaces: 'drive'
    });

    if (response.result.files && response.result.files.length > 0) {
      this.folderId = response.result.files[0].id;
      return this.folderId;
    }

    // Create new folder
    const createResponse = await this.gapi.client.drive.files.create({
      resource: {
        name: this.FOLDER_NAME,
        mimeType: 'application/vnd.google-apps.folder'
      },
      fields: 'id'
    });

    this.folderId = createResponse.result.id;
    return this.folderId;
  }

  /**
   * Save progress to Google Drive with ultra-compression
   */
  async saveProgress(progress: UserProgress): Promise<void> {
    if (!this.isSignedIn()) {
      throw new Error('User must be signed in to save progress');
    }

    const folderId = await this.getOrCreateFolder();
    const user = this.getCurrentUser();
    
    // Update user info and timestamp
    progress.user = {
      email: user?.email,
      displayName: user?.name
    };
    progress.lastUpdated = new Date().toISOString();

    // Compress to ultra-compact format
    const compactProgress = CompressionEngine.compressProgress(progress);
    
    // Use delta updates if we have previous state
    let deltaStorage: DeltaStorage;
    
    if (this.lastKnownProgress) {
      // Create deltas from last known state
      const deltas = DeltaManager.createDelta(this.lastKnownProgress, compactProgress);
      
      // Load existing delta storage or create new one
      const existingDeltaStorage = await this.loadDeltaStorage();
      
      if (existingDeltaStorage && !DeltaManager.shouldCreateSnapshot(existingDeltaStorage)) {
        // Add new deltas to existing chain
        deltaStorage = {
          ...existingDeltaStorage,
          deltas: [...existingDeltaStorage.deltas, ...deltas]
        };
      } else {
        // Create new snapshot
        deltaStorage = {
          baseSnapshot: compactProgress,
          deltas: [],
          lastSync: Date.now()
        };
      }
    } else {
      // First save - create initial snapshot
      deltaStorage = {
        baseSnapshot: compactProgress,
        deltas: [],
        lastSync: Date.now()
      };
    }

    // Compress delta storage to binary
    const compressedData = DeltaManager.compressDeltaStorage(deltaStorage);
    const base64Data = CompressionEngine.encodeToBase64(compressedData);
    
    // Create ultra-compact payload
    const payload: StoragePayload = {
      v: 2, // version 2 with compression
      d: base64Data,
      m: ProblemMapper.exportMapping(),
      s: compressedData.byteLength
    };
    
    // Convert to minimal JSON (no formatting)
    const content = JSON.stringify(payload);
    const blob = new Blob([content], { type: 'application/json' });

    console.log(`Storage size: ${DeltaManager.getHumanReadableSize(content.length)} (${((1 - content.length / JSON.stringify(progress).length) * 100).toFixed(1)}% reduction)`);

    // Save to Drive
    await this.uploadFile(this.FILE_NAME, blob, folderId);
    
    // Update local cache
    this.lastKnownProgress = compactProgress;
  }

  /**
   * Upload or update file helper
   */
  private async uploadFile(fileName: string, blob: Blob, folderId: string): Promise<void> {
    // Search for existing file
    const searchResponse = await this.gapi.client.drive.files.list({
      q: `name='${fileName}' and '${folderId}' in parents and trashed=false`,
      fields: 'files(id)',
      spaces: 'drive'
    });

    if (searchResponse.result.files && searchResponse.result.files.length > 0) {
      // Update existing file
      const fileId = searchResponse.result.files[0].id;
      await this.gapi.client.request({
        path: `/upload/drive/v3/files/${fileId}`,
        method: 'PATCH',
        params: {
          uploadType: 'media'
        },
        body: blob
      });
    } else {
      // Create new file
      const metadata = {
        name: fileName,
        parents: [folderId]
      };
      
      await this.gapi.client.request({
        path: '/upload/drive/v3/files',
        method: 'POST',
        params: {
          uploadType: 'multipart'
        },
        headers: {
          'Content-Type': 'multipart/related; boundary=foo_bar_baz'
        },
        body: this.createMultipartBody(metadata, await blob.text())
      });
    }
  }

  /**
   * Load progress from Google Drive with decompression
   */
  async loadProgress(): Promise<UserProgress | null> {
    if (!this.isSignedIn()) {
      throw new Error('User must be signed in to load progress');
    }

    try {
      const deltaStorage = await this.loadDeltaStorage();
      if (!deltaStorage) {
        return null;
      }

      // Apply deltas to get current state
      const compactProgress = DeltaManager.applyDeltas(deltaStorage.baseSnapshot, deltaStorage.deltas);
      
      // Cache for delta calculations
      this.lastKnownProgress = compactProgress;
      
      // Convert back to full format
      const user = this.getCurrentUser();
      const fullProgress = CompressionEngine.decompressProgress(compactProgress, {
        email: user?.email,
        displayName: user?.name
      });

      return fullProgress;
    } catch (error) {
      console.warn('Failed to load compressed progress, trying legacy format:', error);
      return await this.loadLegacyProgress();
    }
  }

  /**
   * Load delta storage from Drive
   */
  private async loadDeltaStorage(): Promise<DeltaStorage | null> {
    const folderId = await this.getOrCreateFolder();

    // Search for progress file
    const searchResponse = await this.gapi.client.drive.files.list({
      q: `name='${this.FILE_NAME}' and '${folderId}' in parents and trashed=false`,
      fields: 'files(id)',
      spaces: 'drive'
    });

    if (!searchResponse.result.files || searchResponse.result.files.length === 0) {
      return null;
    }

    // Download file content
    const fileId = searchResponse.result.files[0].id;
    const response = await this.gapi.client.drive.files.get({
      fileId: fileId,
      alt: 'media'
    });

    const payload = JSON.parse(response.result) as StoragePayload;
    
    // Import problem mapping
    ProblemMapper.importMapping(payload.m);
    
    // Decompress data
    const compressedBuffer = CompressionEngine.decodeFromBase64(payload.d);
    const deltaStorage = DeltaManager.decompressDeltaStorage(compressedBuffer);
    
    return deltaStorage;
  }

  /**
   * Load legacy uncompressed progress (fallback)
   */
  private async loadLegacyProgress(): Promise<UserProgress | null> {
    const folderId = await this.getOrCreateFolder();

    // Search for legacy JSON file
    const searchResponse = await this.gapi.client.drive.files.list({
      q: `name='automata-progress.json' and '${folderId}' in parents and trashed=false`,
      fields: 'files(id)',
      spaces: 'drive'
    });

    if (!searchResponse.result.files || searchResponse.result.files.length === 0) {
      return null;
    }

    const fileId = searchResponse.result.files[0].id;
    const response = await this.gapi.client.drive.files.get({
      fileId: fileId,
      alt: 'media'
    });

    return response.result as UserProgress;
  }

  /**
   * Create multipart request body
   */
  private createMultipartBody(metadata: any, content: string): string {
    const boundary = 'foo_bar_baz';
    const delimiter = '\r\n--' + boundary + '\r\n';
    const close_delim = '\r\n--' + boundary + '--';

    return delimiter +
      'Content-Type: application/json\r\n\r\n' +
      JSON.stringify(metadata) +
      delimiter +
      'Content-Type: application/json\r\n\r\n' +
      content +
      close_delim;
  }

  /**
   * Export progress as downloadable file
   */
  async exportProgress(progress: UserProgress): Promise<void> {
    const content = JSON.stringify(progress, null, 2);
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `automata-progress-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /**
   * Import progress from file
   */
  async importProgress(file: File): Promise<UserProgress> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          let progress: UserProgress;
          
          try {
            // Try parsing as compressed format first
            const payload = JSON.parse(content) as StoragePayload;
            if (payload.v && payload.d) {
              ProblemMapper.importMapping(payload.m);
              const compressedBuffer = CompressionEngine.decodeFromBase64(payload.d);
              const deltaStorage = DeltaManager.decompressDeltaStorage(compressedBuffer);
              const compactProgress = DeltaManager.applyDeltas(deltaStorage.baseSnapshot, deltaStorage.deltas);
              progress = CompressionEngine.decompressProgress(compactProgress);
            } else {
              throw new Error('Not compressed format');
            }
          } catch {
            // Fallback to legacy JSON format
            progress = JSON.parse(content) as UserProgress;
          }
          
          resolve(progress);
        } catch (error) {
          reject(new Error('Invalid progress file'));
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  /**
   * Get storage analytics
   */
  async getStorageAnalytics(): Promise<{
    totalSize: number;
    compressionRatio: number;
    estimatedYearlyGrowth: number;
    problemCount: number;
    deltaChainLength: number;
  }> {
    try {
      const deltaStorage = await this.loadDeltaStorage();
      if (!deltaStorage) {
        return {
          totalSize: 0,
          compressionRatio: 0,
          estimatedYearlyGrowth: 0,
          problemCount: 0,
          deltaChainLength: 0
        };
      }

      const currentSize = DeltaManager.calculateSize(deltaStorage);
      const currentState = DeltaManager.applyDeltas(deltaStorage.baseSnapshot, deltaStorage.deltas);
      const uncompressedSize = JSON.stringify(CompressionEngine.decompressProgress(currentState)).length;
      
      // Estimate yearly growth (assume 1 problem per day)
      const avgProblemSize = currentSize / Math.max(currentState.p.length, 1);
      const estimatedYearlyGrowth = avgProblemSize * 365;

      return {
        totalSize: currentSize,
        compressionRatio: uncompressedSize / currentSize,
        estimatedYearlyGrowth,
        problemCount: currentState.p.length,
        deltaChainLength: deltaStorage.deltas.length
      };
    } catch {
      return {
        totalSize: 0,
        compressionRatio: 0,
        estimatedYearlyGrowth: 0,
        problemCount: 0,
        deltaChainLength: 0
      };
    }
  }

  /**
   * Force snapshot creation (manual optimization)
   */
  async optimizeStorage(): Promise<void> {
    try {
      const deltaStorage = await this.loadDeltaStorage();
      if (!deltaStorage) return;

      // Create fresh snapshot
      const optimizedStorage = DeltaManager.createSnapshot(deltaStorage);
      
      // Save optimized version
      const compressedData = DeltaManager.compressDeltaStorage(optimizedStorage);
      const base64Data = CompressionEngine.encodeToBase64(compressedData);
      
      const payload: StoragePayload = {
        v: 2,
        d: base64Data,
        m: ProblemMapper.exportMapping(),
        s: compressedData.byteLength
      };
      
      const content = JSON.stringify(payload);
      const blob = new Blob([content], { type: 'application/json' });
      const folderId = await this.getOrCreateFolder();
      
      await this.uploadFile(this.FILE_NAME, blob, folderId);
      
      console.log('Storage optimized - delta chain reset');
    } catch (error) {
      console.error('Failed to optimize storage:', error);
      throw error;
    }
  }

  /**
   * Get human readable storage report
   */
  async getStorageReport(): Promise<string> {
    const analytics = await this.getStorageAnalytics();
    
    return `
Storage Analytics Report
=======================
Current Size: ${DeltaManager.getHumanReadableSize(analytics.totalSize)}
Compression Ratio: ${analytics.compressionRatio.toFixed(1)}x
Problems Completed: ${analytics.problemCount}
Delta Chain Length: ${analytics.deltaChainLength}
Estimated Yearly Growth: ${DeltaManager.getHumanReadableSize(analytics.estimatedYearlyGrowth)}

Storage Efficiency: ${analytics.totalSize < 10 * 1024 ? '✅ Excellent' : analytics.totalSize < 50 * 1024 ? '⚠️ Good' : '❌ Needs Optimization'}
${analytics.deltaChainLength > 30 ? '\n⚠️ Consider running optimizeStorage() to compress delta chain' : ''}
    `.trim();
  }
}