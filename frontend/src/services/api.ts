import { Automaton, Problem, ValidationResult, AIFeedbackRequest, Solution } from '../types/automata';

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  async getProblems(): Promise<{ problems: Problem[] }> {
    const response = await fetch(`${API_BASE_URL}/problems`);
    if (!response.ok) throw new Error('Failed to fetch problems');
    return response.json();
  }

  async getProblem(problemId: string): Promise<Problem> {
    const response = await fetch(`${API_BASE_URL}/problems/${problemId}`);
    if (!response.ok) throw new Error('Failed to fetch problem');
    return response.json();
  }

  async validateSolution(problemId: string, automaton: Automaton, userId: string = 'anonymous'): Promise<ValidationResult> {
    const response = await fetch(`${API_BASE_URL}/problems/${problemId}/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        problem_id: problemId,
        automaton,
        user_id: userId,
      }),
    });
    if (!response.ok) throw new Error('Failed to validate solution');
    return response.json();
  }

  async getHint(problemId: string, hintIndex: number = 0): Promise<{ hint: string; total_hints: number }> {
    const response = await fetch(`${API_BASE_URL}/problems/${problemId}/hint?hint_index=${hintIndex}`);
    if (!response.ok) throw new Error('Failed to fetch hint');
    return response.json();
  }

  async getAIHint(problemId: string, request: AIFeedbackRequest): Promise<{ ai_hint: string }> {
    const response = await fetch(`${API_BASE_URL}/problems/${problemId}/ai-hint`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error('Failed to fetch AI hint');
    return response.json();
  }

  async checkAIStatus(): Promise<{ available: boolean; models?: string[]; current_model?: string; error?: string }> {
    const response = await fetch(`${API_BASE_URL}/ai/status`);
    if (!response.ok) throw new Error('Failed to check AI status');
    return response.json();
  }

  async generateSolution(problemId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/problems/${problemId}/generate-solution`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error('Failed to generate solution');
    return response.json();
  }

  async explainSolution(problemId: string, solution: Solution): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/problems/${problemId}/explain-solution`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(solution)
    });
    if (!response.ok) throw new Error('Failed to explain solution');
    return response.json();
  }

  async getGuidedStep(problemId: string, request: AIFeedbackRequest): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/problems/${problemId}/guided-step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    if (!response.ok) throw new Error('Failed to get guided step');
    return response.json();
  }
}

export const apiService = new ApiService();
