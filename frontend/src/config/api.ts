/**
 * API configuration for the frontend
 */

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || '';
export const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

export const API_ENDPOINTS = {
  // Problem endpoints
  problems: `${API_BASE_URL}/problems`,
  problemById: (id: string) => `${API_BASE_URL}/problems/${id}`,
  validateSolution: (id: string) => `${API_BASE_URL}/problems/${id}/validate`,
  
  // AI endpoints
  analyzeProblem: `${API_BASE_URL}/api/analyze-problem`,
  generateSolution: (id: string) => `${API_BASE_URL}/problems/${id}/generate-solution`,
  getHint: (id: string) => `${API_BASE_URL}/problems/${id}/ai-hint`,
  explainSolution: (id: string) => `${API_BASE_URL}/problems/${id}/explain-solution`,
  guidedStep: (id: string) => `${API_BASE_URL}/problems/${id}/guided-step`,
  
  // Automata operations
  simulate: `${API_BASE_URL}/api/simulate`,
  export: `${API_BASE_URL}/api/export`,
  minimize: `${API_BASE_URL}/api/minimize`,
  convertToDFA: `${API_BASE_URL}/api/convert-to-dfa`,
  validateEquivalence: `${API_BASE_URL}/api/validate-equivalence`,
} as const;