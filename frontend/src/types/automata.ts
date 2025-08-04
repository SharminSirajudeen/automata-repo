export interface State {
  id: string;
  x: number;
  y: number;
  is_start: boolean;
  is_accept: boolean;
  label?: string;
}

export interface Transition {
  from_state: string;
  to_state: string;
  symbol: string;
  x?: number;
  y?: number;
}

export interface Automaton {
  states: State[];
  transitions: Transition[];
  alphabet: string[];
  type?: 'dfa' | 'nfa' | 'enfa';
}

export type AutomataType = 'dfa' | 'nfa' | 'enfa' | 'pda' | 'cfg' | 'tm' | 'regex' | 'pumping';

export interface PDATransition extends Transition {
  stack_pop: string;
  stack_push: string;
}

export interface PDAAutomaton {
  type: 'pda';
  states: State[];
  transitions: PDATransition[];
  alphabet: string[];
  stack_alphabet: string[];
  start_stack_symbol: string;
}

export interface CFGProduction {
  id: string;
  left_side: string;
  right_side: string;
}

export interface CFGAutomaton {
  type: 'cfg';
  terminals: string[];
  non_terminals: string[];
  productions: CFGProduction[];
  start_symbol: string;
}

export interface TMTransition {
  from_state: string;
  to_state: string;
  read_symbol: string;
  write_symbol: string;
  head_direction: 'L' | 'R' | 'S';
  tape_index?: number;
}

export interface TMAutomaton {
  type: 'tm';
  states: State[];
  transitions: TMTransition[];
  tape_alphabet: string[];
  blank_symbol: string;
  num_tapes?: number;
}

export interface RegexAutomaton {
  type: 'regex';
  pattern: string;
  alphabet: string[];
  equivalent_nfa?: Automaton;
  equivalent_dfa?: Automaton;
}

export interface PumpingLemmaAutomaton {
  type: 'pumping';
  language_type: 'regular' | 'context_free';
  language_description: string;
  pumping_length?: number;
  example_string?: string;
  decomposition?: {
    x: string;
    y: string;
    z: string;
  };
}

export type ExtendedAutomaton = Automaton | PDAAutomaton | CFGAutomaton | TMAutomaton | RegexAutomaton | PumpingLemmaAutomaton;

export interface Problem {
  id: string;
  type: AutomataType;
  title: string;
  description: string;
  language_description: string;
  alphabet: string[];
  test_strings: Array<{
    string: string;
    should_accept: boolean;
    trace_path?: string[];
    stack_trace?: string[];
    tape_trace?: string[];
  }>;
  hints?: string[];
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  category?: string;
  reference_solution?: ExtendedAutomaton;
}

export interface ValidationResult {
  is_correct: boolean;
  score: number;
  feedback: string;
  test_results: Array<{
    string: string;
    expected: boolean;
    actual: boolean;
    correct: boolean;
    path: string[];
    stack_trace?: string[];
    tape_trace?: string[];
    parse_tree?: any;
  }>;
  mistakes: string[];
  ai_explanation?: string;
  ai_hints?: string[];
  minimization_suggestions?: string[];
  unreachable_states?: string[];
}

export interface AIFeedbackRequest {
  problem_description: string;
  user_automaton: ExtendedAutomaton;
  test_results: Array<any>;
  mistakes: string[];
  automata_type: AutomataType;
}

export interface Solution {
  problem_id: string;
  automaton: ExtendedAutomaton;
  user_id?: string;
}

export interface SimulationStep {
  step_number: number;
  current_state: string;
  input_position: number;
  remaining_input: string;
  stack_contents?: string[];
  tape_contents?: string[];
  head_position?: number;
  action_description: string;
}

export interface SimulationResult {
  accepted: boolean;
  steps: SimulationStep[];
  final_state: string;
  execution_path: string[];
  error_message?: string;
}

export interface CodeExportOptions {
  language: 'python' | 'javascript' | 'java';
  include_tests: boolean;
  include_visualization: boolean;
  format: 'class' | 'function';
}

export interface ExportResult {
  code: string;
  filename: string;
  language: string;
  test_cases?: string;
}
