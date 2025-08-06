/**
 * Ultra-Efficient Storage Data Structures
 * Target: Keep total storage under 10KB even after years of usage
 */

// Compact binary flags for achievements and settings
export const ACHIEVEMENT_FLAGS = {
  FIRST_PROBLEM: 1 << 0,
  STREAK_5: 1 << 1,
  STREAK_10: 1 << 2,
  PERFECT_SCORE: 1 << 3,
  SPEED_DEMON: 1 << 4,
  MASTER_BUILDER: 1 << 5,
} as const;

// Ultra-compact progress data structure
export interface CompactProgress {
  v: number; // version (1 byte)
  u: number; // lastUpdated unix timestamp (4 bytes)
  s: {
    // statistics - total 16 bytes
    t: number; // totalProblems (2 bytes)
    c: number; // completedProblems (2 bytes)
    cs: number; // currentStreak (1 byte)
    ls: number; // longestStreak (1 byte)
    ts: number; // totalTimeSpent in minutes (4 bytes)
    la: number; // lastActiveDate unix timestamp (4 bytes)
    sc: number; // total score (2 bytes)
  };
  p: number[]; // completed problem IDs as indices (2 bytes each)
  d: number[]; // completion deltas: [attempts, timeMinutes, score] packed (3 bytes each)
  a: number; // achievement flags bitmask (2 bytes)
  pref: number; // preferences packed into single byte
  curr?: {
    // current problem - 8 bytes max
    id: number; // problem index (2 bytes)
    st: number; // started timestamp (4 bytes)
    att: number; // attempts (1 byte)
    sol: string; // compressed solution hash (1 byte)
  };
}

// Automaton compression utilities
export class AutomatonCompressor {
  
  /**
   * Compress automaton to minimal representation
   * Average size: 20-50 bytes vs 500-2000 bytes uncompressed
   */
  static compress(automaton: any): string {
    if (!automaton) return '';
    
    const states = automaton.states || [];
    const transitions = automaton.transitions || [];
    const alphabet = automaton.alphabet || [];
    
    // Use base36 encoding for compactness
    const stateMap = new Map(states.map((s: any, i: number) => [s.id, i.toString(36)]));
    const alphaMap = new Map(alphabet.map((a: string, i: number) => [a, i.toString(36)]));
    
    // Encode: stateCount|alphabetSize|startState|acceptStates|transitions
    const parts = [
      states.length.toString(36),
      alphabet.length.toString(36),
      stateMap.get(automaton.startState) || '0',
      (automaton.acceptStates || []).map((s: string) => stateMap.get(s)).join(''),
      transitions.map((t: any) => 
        `${stateMap.get(t.from)}${alphaMap.get(t.symbol)}${stateMap.get(t.to)}`
      ).join('')
    ];
    
    return parts.join('|');
  }
  
  /**
   * Decompress automaton from minimal representation
   */
  static decompress(compressed: string): any {
    if (!compressed) return null;
    
    const parts = compressed.split('|');
    if (parts.length !== 5) return null;
    
    const [stateCountStr, alphaSizeStr, startStateStr, acceptStatesStr, transitionsStr] = parts;
    const stateCount = parseInt(stateCountStr, 36);
    const alphaSize = parseInt(alphaSizeStr, 36);
    
    // Reconstruct alphabet (assume standard: a, b, c, ...)
    const alphabet = Array.from({length: alphaSize}, (_, i) => String.fromCharCode(97 + i));
    
    // Reconstruct states
    const states = Array.from({length: stateCount}, (_, i) => ({
      id: `q${i}`,
      x: 0, y: 0, // Positions not stored to save space
      isStart: i.toString(36) === startStateStr,
      isAccept: acceptStatesStr.includes(i.toString(36))
    }));
    
    // Reconstruct transitions
    const transitions = [];
    for (let i = 0; i < transitionsStr.length; i += 3) {
      const fromIdx = parseInt(transitionsStr[i], 36);
      const symbolIdx = parseInt(transitionsStr[i + 1], 36);
      const toIdx = parseInt(transitionsStr[i + 2], 36);
      
      if (fromIdx < stateCount && symbolIdx < alphaSize && toIdx < stateCount) {
        transitions.push({
          from: `q${fromIdx}`,
          to: `q${toIdx}`,
          symbol: alphabet[symbolIdx]
        });
      }
    }
    
    return {
      states,
      transitions,
      alphabet,
      startState: `q${parseInt(startStateStr, 36)}`,
      acceptStates: acceptStatesStr.split('').map(s => `q${parseInt(s, 36)}`)
    };
  }
}

// Problem ID mapper - maps string IDs to indices
export class ProblemMapper {
  private static problemIds: string[] = [];
  private static idToIndex = new Map<string, number>();
  
  static addProblem(problemId: string): number {
    if (this.idToIndex.has(problemId)) {
      return this.idToIndex.get(problemId)!;
    }
    
    const index = this.problemIds.length;
    this.problemIds.push(problemId);
    this.idToIndex.set(problemId, index);
    return index;
  }
  
  static getIndex(problemId: string): number {
    return this.idToIndex.get(problemId) ?? -1;
  }
  
  static getProblemId(index: number): string {
    return this.problemIds[index] || '';
  }
  
  static exportMapping(): string[] {
    return [...this.problemIds];
  }
  
  static importMapping(ids: string[]): void {
    this.problemIds = [...ids];
    this.idToIndex.clear();
    ids.forEach((id, index) => this.idToIndex.set(id, index));
  }
}

// Preferences packing (all preferences in 1 byte)
export class PreferencesPacker {
  static pack(prefs: any): number {
    let packed = 0;
    
    // Difficulty: 2 bits (0-3)
    const difficultyMap = { beginner: 0, intermediate: 1, advanced: 2 };
    packed |= (difficultyMap[prefs.difficulty as keyof typeof difficultyMap] || 0) << 0;
    
    // Boolean flags
    if (prefs.enableHints) packed |= 1 << 2;
    if (prefs.enableAnimations) packed |= 1 << 3;
    
    return packed;
  }
  
  static unpack(packed: number): any {
    const difficultyNames = ['beginner', 'intermediate', 'advanced'];
    
    return {
      difficulty: difficultyNames[packed & 0x3] || 'beginner',
      enableHints: Boolean(packed & (1 << 2)),
      enableAnimations: Boolean(packed & (1 << 3))
    };
  }
}