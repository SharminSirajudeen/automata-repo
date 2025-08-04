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
}

export interface Problem {
  id: string;
  type: string;
  title: string;
  description: string;
  language_description: string;
  alphabet: string[];
  test_strings: Array<{
    string: string;
    should_accept: boolean;
  }>;
  hints?: string[];
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
  }>;
  mistakes: string[];
  ai_explanation?: string;
  ai_hints?: string[];
}

export interface AIFeedbackRequest {
  problem_description: string;
  user_automaton: Automaton;
  test_results: Array<any>;
  mistakes: string[];
}

export interface Solution {
  problem_id: string;
  automaton: Automaton;
  user_id?: string;
}
