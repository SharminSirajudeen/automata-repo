/**
 * Ultra-Efficient Compression Engine
 * Combines multiple compression strategies for maximum space savings
 */

import { CompactProgress, AutomatonCompressor, ProblemMapper, PreferencesPacker } from './optimizedStorage';

export { CompactProgress };

export class CompressionEngine {
  
  /**
   * Convert full UserProgress to ultra-compact format
   * Target reduction: 90-95% size reduction
   */
  static compressProgress(progress: any): CompactProgress {
    // Map problem IDs to indices
    const completedIndices = progress.completedProblems?.map((p: any) => 
      ProblemMapper.addProblem(p.problemId)
    ) || [];
    
    // Pack completion data as deltas
    const deltas = progress.completedProblems?.map((p: any) => {
      // Pack attempts (max 255), timeMinutes (max 65535), score (max 255)
      const timeMinutes = Math.min(Math.floor((p.timeSpent || 0) / 60), 65535);
      const attempts = Math.min(p.attempts || 1, 255);
      const score = Math.min(p.score || 0, 255);
      
      return (attempts << 16) | (timeMinutes << 8) | score;
    }) || [];
    
    // Pack achievement flags
    let achievementFlags = 0;
    progress.achievements?.forEach((ach: any) => {
      // Map achievement types to bit flags
      switch (ach.type) {
        case 'first_problem': achievementFlags |= 1 << 0; break;
        case 'streak_5': achievementFlags |= 1 << 1; break;
        case 'streak_10': achievementFlags |= 1 << 2; break;
        case 'perfect_score': achievementFlags |= 1 << 3; break;
        case 'speed_demon': achievementFlags |= 1 << 4; break;
        case 'master_builder': achievementFlags |= 1 << 5; break;
      }
    });
    
    const compact: CompactProgress = {
      v: 1,
      u: Math.floor(new Date(progress.lastUpdated || Date.now()).getTime() / 1000),
      s: {
        t: Math.min(progress.statistics?.totalProblems || 0, 65535),
        c: Math.min(progress.statistics?.completedProblems || 0, 65535),
        cs: Math.min(progress.statistics?.currentStreak || 0, 255),
        ls: Math.min(progress.statistics?.longestStreak || 0, 255),
        ts: Math.min(Math.floor((progress.statistics?.totalTimeSpent || 0) / 60), 2147483647),
        la: Math.floor(new Date(progress.statistics?.lastActiveDate || Date.now()).getTime() / 1000),
        sc: Math.min(
          progress.completedProblems?.reduce((sum: number, p: any) => sum + (p.score || 0), 0) || 0,
          65535
        )
      },
      p: completedIndices,
      d: deltas,
      a: achievementFlags,
      pref: PreferencesPacker.pack(progress.preferences || {})
    };
    
    // Add current problem if exists
    if (progress.currentProblem) {
      compact.curr = {
        id: ProblemMapper.addProblem(progress.currentProblem.problemId),
        st: Math.floor(new Date(progress.currentProblem.startedAt).getTime() / 1000),
        att: Math.min(progress.currentProblem.attempts || 0, 255),
        sol: AutomatonCompressor.compress(progress.currentProblem.currentSolution).slice(0, 50) // Truncate if too long
      };
    }
    
    return compact;
  }
  
  /**
   * Convert compact format back to full UserProgress
   */
  static decompressProgress(compact: CompactProgress, userInfo?: any): any {
    const completedProblems = compact.p.map((problemIndex, i) => {
      const delta = compact.d[i] || 0;
      const attempts = (delta >> 16) & 0xFF;
      const timeMinutes = (delta >> 8) & 0xFF;
      const score = delta & 0xFF;
      
      return {
        problemId: ProblemMapper.getProblemId(problemIndex),
        completedAt: new Date(compact.u * 1000).toISOString(),
        attempts,
        timeSpent: timeMinutes * 60,
        score,
        solution: null // Omit full solution to save space
      };
    });
    
    // Reconstruct achievements
    const achievements = [];
    const achievementTypes = [
      'first_problem', 'streak_5', 'streak_10', 'perfect_score', 'speed_demon', 'master_builder'
    ];
    
    achievementTypes.forEach((type, index) => {
      if (compact.a & (1 << index)) {
        achievements.push({
          id: type,
          unlockedAt: new Date(compact.u * 1000).toISOString(),
          type
        });
      }
    });
    
    const progress = {
      version: '2.0',
      lastUpdated: new Date(compact.u * 1000).toISOString(),
      user: userInfo || {},
      statistics: {
        totalProblems: compact.s.t,
        completedProblems: compact.s.c,
        currentStreak: compact.s.cs,
        longestStreak: compact.s.ls,
        totalTimeSpent: compact.s.ts * 60, // Convert back to seconds
        lastActiveDate: new Date(compact.s.la * 1000).toISOString()
      },
      completedProblems,
      achievements,
      preferences: PreferencesPacker.unpack(compact.pref)
    };
    
    // Add current problem if exists
    if (compact.curr) {
      progress.currentProblem = {
        problemId: ProblemMapper.getProblemId(compact.curr.id),
        startedAt: new Date(compact.curr.st * 1000).toISOString(),
        attempts: compact.curr.att,
        currentSolution: AutomatonCompressor.decompress(compact.curr.sol)
      };
    }
    
    return progress;
  }
  
  /**
   * Binary serialization for maximum compression
   * Uses ArrayBuffer for ultra-compact storage
   */
  static serialize(compact: CompactProgress): ArrayBuffer {
    // Calculate required buffer size
    const baseSize = 32; // Fixed fields
    const problemsSize = compact.p.length * 2; // 2 bytes per problem index
    const deltasSize = compact.d.length * 4; // 4 bytes per delta
    const currentProblemSize = compact.curr ? 8 + compact.curr.sol.length : 0;
    
    const buffer = new ArrayBuffer(baseSize + problemsSize + deltasSize + currentProblemSize);
    const view = new DataView(buffer);
    let offset = 0;
    
    // Write fixed fields
    view.setUint8(offset++, compact.v);
    view.setUint32(offset, compact.u); offset += 4;
    
    // Statistics
    view.setUint16(offset, compact.s.t); offset += 2;
    view.setUint16(offset, compact.s.c); offset += 2;
    view.setUint8(offset++, compact.s.cs);
    view.setUint8(offset++, compact.s.ls);
    view.setUint32(offset, compact.s.ts); offset += 4;
    view.setUint32(offset, compact.s.la); offset += 4;
    view.setUint16(offset, compact.s.sc); offset += 2;
    
    // Achievement flags and preferences
    view.setUint16(offset, compact.a); offset += 2;
    view.setUint8(offset++, compact.pref);
    
    // Array lengths
    view.setUint16(offset, compact.p.length); offset += 2;
    view.setUint16(offset, compact.d.length); offset += 2;
    
    // Problem indices
    compact.p.forEach(problemIndex => {
      view.setUint16(offset, problemIndex);
      offset += 2;
    });
    
    // Deltas
    compact.d.forEach(delta => {
      view.setUint32(offset, delta);
      offset += 4;
    });
    
    // Current problem (optional)
    if (compact.curr) {
      view.setUint8(offset++, 1); // Has current problem flag
      view.setUint16(offset, compact.curr.id); offset += 2;
      view.setUint32(offset, compact.curr.st); offset += 4;
      view.setUint8(offset++, compact.curr.att);
      
      // Solution string
      const solBytes = new TextEncoder().encode(compact.curr.sol);
      view.setUint8(offset++, solBytes.length);
      new Uint8Array(buffer, offset).set(solBytes);
      offset += solBytes.length;
    } else {
      view.setUint8(offset++, 0); // No current problem flag
    }
    
    return buffer.slice(0, offset);
  }
  
  /**
   * Deserialize from binary format
   */
  static deserialize(buffer: ArrayBuffer): CompactProgress {
    const view = new DataView(buffer);
    let offset = 0;
    
    const compact: CompactProgress = {
      v: view.getUint8(offset++),
      u: view.getUint32(offset), 
      s: {
        t: 0, c: 0, cs: 0, ls: 0, ts: 0, la: 0, sc: 0
      },
      p: [],
      d: [],
      a: 0,
      pref: 0
    };
    
    offset += 4;
    
    // Read statistics
    compact.s.t = view.getUint16(offset); offset += 2;
    compact.s.c = view.getUint16(offset); offset += 2;
    compact.s.cs = view.getUint8(offset++);
    compact.s.ls = view.getUint8(offset++);
    compact.s.ts = view.getUint32(offset); offset += 4;
    compact.s.la = view.getUint32(offset); offset += 4;
    compact.s.sc = view.getUint16(offset); offset += 2;
    
    // Read flags
    compact.a = view.getUint16(offset); offset += 2;
    compact.pref = view.getUint8(offset++);
    
    // Read array lengths
    const problemsLength = view.getUint16(offset); offset += 2;
    const deltasLength = view.getUint16(offset); offset += 2;
    
    // Read problem indices
    for (let i = 0; i < problemsLength; i++) {
      compact.p.push(view.getUint16(offset));
      offset += 2;
    }
    
    // Read deltas
    for (let i = 0; i < deltasLength; i++) {
      compact.d.push(view.getUint32(offset));
      offset += 4;
    }
    
    // Read current problem if exists
    const hasCurrentProblem = view.getUint8(offset++);
    if (hasCurrentProblem) {
      const solLength = view.getUint8(offset + 7);
      compact.curr = {
        id: view.getUint16(offset),
        st: view.getUint32(offset + 2),
        att: view.getUint8(offset + 6),
        sol: new TextDecoder().decode(new Uint8Array(buffer, offset + 8, solLength))
      };
    }
    
    return compact;
  }
  
  /**
   * Base64 encode binary data for JSON storage
   */
  static encodeToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
  
  /**
   * Base64 decode to binary data
   */
  static decodeFromBase64(base64: string): ArrayBuffer {
    const binary = atob(base64);
    const buffer = new ArrayBuffer(binary.length);
    const view = new Uint8Array(buffer);
    for (let i = 0; i < binary.length; i++) {
      view[i] = binary.charCodeAt(i);
    }
    return buffer;
  }
}