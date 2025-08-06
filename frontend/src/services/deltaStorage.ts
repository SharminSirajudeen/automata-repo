/**
 * Delta Storage System
 * Only uploads changes, dramatically reducing bandwidth and storage
 */

import { CompressionEngine, CompactProgress } from './compressionEngine';

export interface Delta {
  timestamp: number;
  type: 'problem_completed' | 'streak_updated' | 'achievement_unlocked' | 'preference_changed';
  data: any;
}

export interface DeltaStorage {
  baseSnapshot: CompactProgress;
  deltas: Delta[];
  lastSync: number;
}

export class DeltaManager {
  private static readonly MAX_DELTAS = 50; // Limit delta chain length
  private static readonly SNAPSHOT_INTERVAL = 24 * 60 * 60 * 1000; // 24 hours
  
  /**
   * Create delta from two progress states
   */
  static createDelta(oldProgress: CompactProgress, newProgress: CompactProgress): Delta[] {
    const deltas: Delta[] = [];
    const now = Date.now();
    
    // Check for completed problems
    if (newProgress.p.length > oldProgress.p.length) {
      const newProblems = newProgress.p.slice(oldProgress.p.length);
      const newDeltas = newProgress.d.slice(oldProgress.d.length);
      
      newProblems.forEach((problemIndex, i) => {
        deltas.push({
          timestamp: now,
          type: 'problem_completed',
          data: {
            problemIndex,
            delta: newDeltas[i] || 0
          }
        });
      });
    }
    
    // Check for streak changes
    if (newProgress.s.cs !== oldProgress.s.cs || newProgress.s.ls !== oldProgress.s.ls) {
      deltas.push({
        timestamp: now,
        type: 'streak_updated',
        data: {
          currentStreak: newProgress.s.cs,
          longestStreak: newProgress.s.ls
        }
      });
    }
    
    // Check for new achievements
    if (newProgress.a !== oldProgress.a) {
      const newAchievements = newProgress.a & ~oldProgress.a; // XOR to get new bits
      if (newAchievements) {
        deltas.push({
          timestamp: now,
          type: 'achievement_unlocked',
          data: {
            achievementFlags: newAchievements
          }
        });
      }
    }
    
    // Check for preference changes
    if (newProgress.pref !== oldProgress.pref) {
      deltas.push({
        timestamp: now,
        type: 'preference_changed',
        data: {
          preferences: newProgress.pref
        }
      });
    }
    
    return deltas;
  }
  
  /**
   * Apply deltas to base snapshot
   */
  static applyDeltas(baseSnapshot: CompactProgress, deltas: Delta[]): CompactProgress {
    const result = JSON.parse(JSON.stringify(baseSnapshot)); // Deep clone
    
    deltas.forEach(delta => {
      switch (delta.type) {
        case 'problem_completed':
          result.p.push(delta.data.problemIndex);
          result.d.push(delta.data.delta);
          result.s.c++;
          result.s.t = Math.max(result.s.t, result.s.c);
          
          // Update total time and score
          const deltaData = delta.data.delta;
          const timeMinutes = (deltaData >> 8) & 0xFF;
          const score = deltaData & 0xFF;
          result.s.ts += timeMinutes;
          result.s.sc += score;
          break;
          
        case 'streak_updated':
          result.s.cs = delta.data.currentStreak;
          result.s.ls = Math.max(result.s.ls, delta.data.longestStreak);
          break;
          
        case 'achievement_unlocked':
          result.a |= delta.data.achievementFlags;
          break;
          
        case 'preference_changed':
          result.pref = delta.data.preferences;
          break;
      }
      
      result.u = Math.floor(delta.timestamp / 1000);
    });
    
    return result;
  }
  
  /**
   * Compress delta storage
   */
  static compressDeltaStorage(deltaStorage: DeltaStorage): ArrayBuffer {
    const serializedBase = CompressionEngine.serialize(deltaStorage.baseSnapshot);
    const deltasJson = JSON.stringify(deltaStorage.deltas);
    const deltasBytes = new TextEncoder().encode(deltasJson);
    
    const buffer = new ArrayBuffer(8 + serializedBase.byteLength + deltasBytes.byteLength);
    const view = new DataView(buffer);
    let offset = 0;
    
    // Write metadata
    view.setUint32(offset, serializedBase.byteLength); offset += 4;
    view.setUint32(offset, deltaStorage.lastSync); offset += 4;
    
    // Write base snapshot
    new Uint8Array(buffer, offset).set(new Uint8Array(serializedBase));
    offset += serializedBase.byteLength;
    
    // Write deltas
    new Uint8Array(buffer, offset).set(deltasBytes);
    
    return buffer;
  }
  
  /**
   * Decompress delta storage
   */
  static decompressDeltaStorage(buffer: ArrayBuffer): DeltaStorage {
    const view = new DataView(buffer);
    let offset = 0;
    
    const baseSize = view.getUint32(offset); offset += 4;
    const lastSync = view.getUint32(offset); offset += 4;
    
    const baseBuffer = buffer.slice(offset, offset + baseSize);
    const baseSnapshot = CompressionEngine.deserialize(baseBuffer);
    offset += baseSize;
    
    const deltasBytes = new Uint8Array(buffer, offset);
    const deltasJson = new TextDecoder().decode(deltasBytes);
    const deltas = JSON.parse(deltasJson);
    
    return {
      baseSnapshot,
      deltas,
      lastSync
    };
  }
  
  /**
   * Check if snapshot should be created
   */
  static shouldCreateSnapshot(deltaStorage: DeltaStorage): boolean {
    const now = Date.now();
    return (
      deltaStorage.deltas.length >= this.MAX_DELTAS ||
      (now - deltaStorage.lastSync) > this.SNAPSHOT_INTERVAL
    );
  }
  
  /**
   * Create new snapshot from current state
   */
  static createSnapshot(deltaStorage: DeltaStorage): DeltaStorage {
    const currentState = this.applyDeltas(deltaStorage.baseSnapshot, deltaStorage.deltas);
    
    return {
      baseSnapshot: currentState,
      deltas: [],
      lastSync: Date.now()
    };
  }
  
  /**
   * Optimize delta chain by removing redundant deltas
   */
  static optimizeDeltaChain(deltas: Delta[]): Delta[] {
    const optimized: Delta[] = [];
    const seen = new Set<string>();
    
    // Remove duplicate preference changes (keep only latest)
    const prefChanges = deltas.filter(d => d.type === 'preference_changed');
    if (prefChanges.length > 0) {
      optimized.push(prefChanges[prefChanges.length - 1]);
      seen.add('preference_changed');
    }
    
    // Keep all other deltas but deduplicate
    deltas.forEach(delta => {
      const key = `${delta.type}_${JSON.stringify(delta.data)}`;
      if (!seen.has(key) && delta.type !== 'preference_changed') {
        optimized.push(delta);
        seen.add(key);
      }
    });
    
    return optimized.sort((a, b) => a.timestamp - b.timestamp);
  }
  
  /**
   * Calculate storage size in bytes
   */
  static calculateSize(deltaStorage: DeltaStorage): number {
    const compressed = this.compressDeltaStorage(deltaStorage);
    return compressed.byteLength;
  }
  
  /**
   * Get human-readable size
   */
  static getHumanReadableSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}