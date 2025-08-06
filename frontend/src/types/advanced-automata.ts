import { State } from './automata';

export interface MultiTapeTransition {
  from_state: string;
  to_state: string;
  tape_operations: TapeOperation[];
}

export interface TapeOperation {
  tape_index: number;
  read_symbol: string;
  write_symbol: string;
  head_direction: 'L' | 'R' | 'S';
}

export interface MultiTapeTMAutomaton {
  type: 'multi-tape-tm';
  states: State[];
  transitions: MultiTapeTransition[];
  tape_alphabet: string[];
  blank_symbol: string;
  num_tapes: number;
  input_tape_index: number;
}

export interface TapeState {
  contents: string[];
  head_position: number;
  tape_index: number;
}

export interface UnrestrictedGrammar {
  type: 'unrestricted-grammar';
  terminals: string[];
  non_terminals: string[];
  productions: UnrestrictedProduction[];
  start_symbol: string;
}

export interface UnrestrictedProduction {
  id: string;
  left_side: string[]; // Multiple symbols allowed
  right_side: string[];
  context_sensitive?: boolean;
}

export interface SLRParserState {
  id: string;
  items: LRItem[];
  transitions: { [symbol: string]: string };
}

export interface LRItem {
  production: UnrestrictedProduction;
  dot_position: number;
  lookahead?: string[];
}

export interface SLRParser {
  type: 'slr-parser';
  grammar: UnrestrictedGrammar;
  states: SLRParserState[];
  action_table: { [state: string]: { [symbol: string]: Action } };
  goto_table: { [state: string]: { [symbol: string]: string } };
}

export interface Action {
  type: 'shift' | 'reduce' | 'accept' | 'error';
  value?: string | number;
}

export interface UniversalTM {
  type: 'universal-tm';
  encoded_tm: string;
  encoding_scheme: 'standard' | 'binary' | 'decimal';
  control_states: string[];
}

export interface LSystem {
  type: 'l-system';
  axiom: string;
  productions: { [symbol: string]: string };
  iterations: number;
  angle: number;
  turtle_commands: TurtleCommand[];
  render_3d: boolean;
}

export interface TurtleCommand {
  symbol: string;
  action: 'forward' | 'turn_left' | 'turn_right' | 'push' | 'pop' | 'up' | 'down';
  value?: number;
}

export interface ParseNode {
  id: string;
  symbol: string;
  children: ParseNode[];
  is_terminal: boolean;
  position: { x: number; y: number };
  is_bracketed?: boolean;
}

export type AdvancedAutomaton = MultiTapeTMAutomaton | UnrestrictedGrammar | SLRParser | UniversalTM | LSystem;

// Re-export base types
export * from './automata';