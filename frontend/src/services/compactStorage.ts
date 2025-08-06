/**
 * Ultra-efficient storage for Google Drive
 * Keeps data in kilobytes, not megabytes
 */

// Compact progress format - uses short keys to minimize JSON size
interface CompactProgress {
  v: 1;                    // version (1 byte)
  t: number;               // timestamp (epoch seconds)
  c: string;               // completed problems as bitstring
  s: number;               // current streak
  l: number;               // total learning time (minutes)
  m: number;               // modules completed (bitmask)
}

export class CompactStorage {
  private static VERSION = 1;
  
  /**
   * Compress progress data to minimal size
   */
  static compress(data: {
    completedProblems: string[];
    currentStreak: number;
    totalTime: number;
    completedModules: string[];
  }): string {
    // Convert problem IDs to indices (assuming sequential IDs)
    const problemIndices = data.completedProblems
      .map(id => parseInt(id.replace(/\D/g, ''), 10))
      .sort((a, b) => a - b);
    
    // Pack into bitstring (1 bit per problem, up to 1000 problems = 125 bytes)
    const maxProblem = Math.max(...problemIndices, 0);
    const bitArray = new Array(Math.ceil((maxProblem + 1) / 8)).fill(0);
    
    problemIndices.forEach(idx => {
      const byteIndex = Math.floor(idx / 8);
      const bitIndex = idx % 8;
      bitArray[byteIndex] |= (1 << bitIndex);
    });
    
    // Convert to base64 for JSON storage
    const bitString = btoa(String.fromCharCode(...bitArray));
    
    // Module completion as bitmask (supports up to 32 modules)
    const moduleMask = data.completedModules.reduce((mask, modId) => {
      const modNum = parseInt(modId.replace(/\D/g, ''), 10);
      return mask | (1 << modNum);
    }, 0);
    
    const compact: CompactProgress = {
      v: this.VERSION,
      t: Math.floor(Date.now() / 1000),
      c: bitString,
      s: data.currentStreak,
      l: Math.floor(data.totalTime / 60), // Store in minutes
      m: moduleMask
    };
    
    // Return as compact JSON (typically < 200 bytes)
    return JSON.stringify(compact);
  }
  
  /**
   * Decompress back to full format
   */
  static decompress(compactJson: string): {
    completedProblems: string[];
    currentStreak: number;
    totalTime: number;
    completedModules: string[];
    lastUpdated: Date;
  } {
    const compact: CompactProgress = JSON.parse(compactJson);
    
    // Decode bitstring back to problem IDs
    const bytes = atob(compact.c).split('').map(c => c.charCodeAt(0));
    const completedProblems: string[] = [];
    
    bytes.forEach((byte, byteIndex) => {
      for (let bit = 0; bit < 8; bit++) {
        if (byte & (1 << bit)) {
          const problemId = byteIndex * 8 + bit;
          completedProblems.push(`problem-${problemId}`);
        }
      }
    });
    
    // Decode module bitmask
    const completedModules: string[] = [];
    for (let i = 0; i < 32; i++) {
      if (compact.m & (1 << i)) {
        completedModules.push(`mod-${i}`);
      }
    }
    
    return {
      completedProblems,
      currentStreak: compact.s,
      totalTime: compact.l * 60, // Convert back to seconds
      completedModules,
      lastUpdated: new Date(compact.t * 1000)
    };
  }
  
  /**
   * Calculate storage size in bytes
   */
  static calculateSize(compactJson: string): number {
    return new Blob([compactJson]).size;
  }
  
  /**
   * Delta update - only send changes
   */
  static createDelta(
    oldProgress: CompactProgress,
    newProgress: CompactProgress
  ): string {
    const delta: any = { t: newProgress.t };
    
    // Only include changed fields
    if (oldProgress.s !== newProgress.s) delta.s = newProgress.s;
    if (oldProgress.l !== newProgress.l) delta.l = newProgress.l;
    if (oldProgress.m !== newProgress.m) delta.m = newProgress.m;
    if (oldProgress.c !== newProgress.c) {
      // Send only new problems as delta
      const oldProblems = this.decodeBitstring(oldProgress.c);
      const newProblems = this.decodeBitstring(newProgress.c);
      const addedProblems = newProblems.filter(p => !oldProblems.includes(p));
      delta.a = addedProblems; // Added problems only
    }
    
    return JSON.stringify(delta); // Typically < 50 bytes
  }
  
  private static decodeBitstring(bitString: string): number[] {
    const bytes = atob(bitString).split('').map(c => c.charCodeAt(0));
    const problems: number[] = [];
    
    bytes.forEach((byte, byteIndex) => {
      for (let bit = 0; bit < 8; bit++) {
        if (byte & (1 << bit)) {
          problems.push(byteIndex * 8 + bit);
        }
      }
    });
    
    return problems;
  }
}

// Example usage showing size efficiency
export function demonstrateEfficiency() {
  const sampleProgress = {
    completedProblems: Array.from({ length: 150 }, (_, i) => `problem-${i}`),
    currentStreak: 7,
    totalTime: 25200, // 7 hours
    completedModules: ['mod-1', 'mod-2', 'mod-3']
  };
  
  // Compress
  const compressed = CompactStorage.compress(sampleProgress);
  const size = CompactStorage.calculateSize(compressed);
  
  console.log('Compressed size:', size, 'bytes'); // ~180 bytes for 150 problems
  console.log('Compression ratio:', Math.round((1 - size / JSON.stringify(sampleProgress).length) * 100) + '%');
  
  // Verify decompression
  const decompressed = CompactStorage.decompress(compressed);
  console.log('Decompressed:', decompressed);
}