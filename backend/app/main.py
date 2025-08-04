from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Set, Any
import re
import httpx
import json
from .agents import AutomataGenerator, AutomataExplainer

app = FastAPI(title="Theory of Computation Tutor", version="1.0.0")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"  # Default model

problems_db = {}
solutions_db = {}

class State(BaseModel):
    id: str
    x: float
    y: float
    is_start: bool = False
    is_accept: bool = False
    label: Optional[str] = None

class Transition(BaseModel):
    from_state: str
    to_state: str
    symbol: str
    x: Optional[float] = None
    y: Optional[float] = None

class Automaton(BaseModel):
    states: List[State]
    transitions: List[Transition]
    alphabet: List[str]

class Problem(BaseModel):
    id: str
    type: str  # "dfa", "nfa", "pda", "regex", "grammar"
    title: str
    description: str
    language_description: str
    alphabet: List[str]
    test_strings: List[Dict[str, Any]]  # {"string": "ab", "should_accept": True}
    hints: Optional[List[str]] = []

class Solution(BaseModel):
    problem_id: str
    automaton: Automaton
    user_id: Optional[str] = "anonymous"

class ValidationResult(BaseModel):
    is_correct: bool
    score: float
    feedback: str
    test_results: List[Dict[str, Any]]
    mistakes: List[str]
    ai_explanation: Optional[str] = None
    ai_hints: Optional[List[str]] = None

class AIFeedbackRequest(BaseModel):
    problem_description: str
    user_automaton: Automaton
    test_results: List[Dict[str, Any]]
    mistakes: List[str]

async def get_ai_feedback(problem: Problem, automaton: Automaton, test_results: List[Dict[str, Any]], mistakes: List[str]) -> Dict[str, Any]:
    """Get AI-powered feedback using Ollama"""
    try:
        prompt = f"""
You are an expert in Theory of Computation and automata theory. A student has submitted a DFA solution for the following problem:

Problem: {problem.title}
Description: {problem.description}
Language: {problem.language_description}
Alphabet: {problem.alphabet}

Student's DFA:
- States: {len(automaton.states)} states
- Transitions: {len(automaton.transitions)} transitions
- Start states: {[s.id for s in automaton.states if s.is_start]}
- Accept states: {[s.id for s in automaton.states if s.is_accept]}

Test Results:
{json.dumps(test_results, indent=2)}

Mistakes found:
{mistakes if mistakes else "No structural mistakes found"}

Please provide:
1. A clear explanation of what the student did right and wrong
2. Specific hints to help them improve their solution
3. Educational insights about the automata theory concepts involved

Be encouraging but precise. Focus on helping them understand the underlying concepts.
"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "")
                
                lines = ai_response.split('\n')
                explanation = ai_response
                hints = []
                
                in_hints_section = False
                for line in lines:
                    if "hints" in line.lower() or "suggestions" in line.lower():
                        in_hints_section = True
                    elif in_hints_section and line.strip().startswith(('-', '•', '*')):
                        hints.append(line.strip()[1:].strip())
                
                return {
                    "explanation": explanation,
                    "hints": hints if hints else [
                        "Review the problem requirements carefully",
                        "Check your state transitions for each symbol",
                        "Verify your accept states match the language definition"
                    ]
                }
            else:
                return {
                    "explanation": "AI feedback temporarily unavailable. Please check your solution against the test cases.",
                    "hints": ["Review failed test cases", "Check state transitions", "Verify accept states"]
                }
                
    except Exception as e:
        return {
            "explanation": "AI feedback temporarily unavailable. Please check your solution against the test cases.",
            "hints": ["Review failed test cases", "Check state transitions", "Verify accept states"]
        }

def validate_dfa(automaton: Automaton, problem: Problem) -> ValidationResult:
    """Validate a DFA solution against a problem"""
    mistakes = []
    test_results = []
    correct_count = 0
    
    start_states = [s for s in automaton.states if s.is_start]
    if len(start_states) != 1:
        mistakes.append(f"DFA must have exactly one start state, found {len(start_states)}")
        return ValidationResult(
            is_correct=False,
            score=0.0,
            feedback="Invalid DFA structure",
            test_results=[],
            mistakes=mistakes
        )
    
    start_state = start_states[0]
    
    transitions = {}
    for state in automaton.states:
        transitions[state.id] = {}
    
    for trans in automaton.transitions:
        if trans.from_state not in transitions:
            transitions[trans.from_state] = {}
        transitions[trans.from_state][trans.symbol] = trans.to_state
    
    for test_case in problem.test_strings:
        test_string = test_case["string"]
        should_accept = test_case["should_accept"]
        
        current_state = start_state.id
        accepted = True
        path = [current_state]
        
        for symbol in str(test_string):
            if current_state not in transitions or symbol not in transitions[current_state]:
                accepted = False
                break
            current_state = transitions[current_state][symbol]
            path.append(current_state)
        
        if accepted:
            final_state = next((s for s in automaton.states if s.id == current_state), None)
            accepted = final_state and final_state.is_accept
        
        is_correct = accepted == should_accept
        if is_correct:
            correct_count += 1
        
        test_results.append({
            "string": test_string,
            "expected": should_accept,
            "actual": accepted,
            "correct": is_correct,
            "path": path
        })
    
    score = correct_count / len(problem.test_strings) if problem.test_strings else 0
    is_correct = score == 1.0
    
    feedback = f"Passed {correct_count}/{len(problem.test_strings)} test cases"
    if not is_correct:
        feedback += ". Check the failed test cases and adjust your automaton."
    
    return ValidationResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        test_results=test_results,
        mistakes=mistakes
    )

def init_sample_problems():
    problems_db["dfa_ending_ab"] = Problem(
        id="dfa_ending_ab",
        type="dfa",
        title="DFA: Strings ending in 'ab'",
        description="Construct a DFA that recognizes the language of strings over the alphabet {a, b} that end with 'ab'.",
        language_description="All words that end with 'ab'",
        alphabet=["a", "b"],
        test_strings=[
            {"string": "ab", "should_accept": True},
            {"string": "aab", "should_accept": True},
            {"string": "bab", "should_accept": True},
            {"string": "abab", "should_accept": True},
            {"string": "a", "should_accept": False},
            {"string": "b", "should_accept": False},
            {"string": "aa", "should_accept": False},
            {"string": "bb", "should_accept": False},
            {"string": "ba", "should_accept": False},
            {"string": "aba", "should_accept": False}
        ],
        hints=[
            "Think about what states you need to track",
            "You need to remember the last two characters",
            "Consider what happens when you see 'a' vs 'b'"
        ]
    )
    
    problems_db["dfa_even_as"] = Problem(
        id="dfa_even_as",
        type="dfa",
        title="DFA: Even number of a's",
        description="Construct a DFA that accepts strings with an even number of 'a's over the alphabet {a, b}.",
        language_description="All words with an even number of 'a's",
        alphabet=["a", "b"],
        test_strings=[
            {"string": "", "should_accept": True},
            {"string": "b", "should_accept": True},
            {"string": "bb", "should_accept": True},
            {"string": "aa", "should_accept": True},
            {"string": "abab", "should_accept": True},
            {"string": "a", "should_accept": False},
            {"string": "ab", "should_accept": False},
            {"string": "aaa", "should_accept": False},
            {"string": "baba", "should_accept": False}
        ],
        hints=[
            "You only need two states",
            "Track whether you've seen an even or odd number of 'a's",
            "The 'b' transitions don't change the count"
        ]
    )
    
    problems_db["dfa_contains_101"] = Problem(
        id="dfa_contains_101",
        type="dfa",
        title="DFA: Contains substring '101'",
        description="Construct a DFA that accepts strings containing the substring '101' over the alphabet {0, 1}.",
        language_description="All words containing '101' as a substring",
        alphabet=["0", "1"],
        test_strings=[
            {"string": "101", "should_accept": True},
            {"string": "1101", "should_accept": True},
            {"string": "1011", "should_accept": True},
            {"string": "01010", "should_accept": True},
            {"string": "11010", "should_accept": True},
            {"string": "", "should_accept": False},
            {"string": "0", "should_accept": False},
            {"string": "1", "should_accept": False},
            {"string": "10", "should_accept": False},
            {"string": "110", "should_accept": False},
            {"string": "1100", "should_accept": False}
        ],
        hints=[
            "Think about tracking progress toward finding '101'",
            "You need states for: nothing, '1', '10', and 'found 101'",
            "Once you find '101', you stay in the accepting state"
        ]
    )

init_sample_problems()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/problems")
async def get_problems():
    """Get all available problems"""
    return {"problems": list(problems_db.values())}

@app.get("/problems/{problem_id}")
async def get_problem(problem_id: str):
    """Get a specific problem by ID"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    return problems_db[problem_id]

@app.post("/problems/{problem_id}/validate")
async def validate_solution(problem_id: str, solution: Solution):
    """Validate a solution against a problem"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    
    if problem.type == "dfa":
        result = validate_dfa(solution.automaton, problem)
    else:
        raise HTTPException(status_code=400, detail=f"Validation for {problem.type} not yet implemented")
    
    solution_key = f"{problem_id}_{solution.user_id}"
    solutions_db[solution_key] = {
        "solution": solution,
        "result": result,
        "timestamp": "2025-08-04T16:27:32Z"  # In a real app, use datetime.now()
    }
    
    return result

@app.get("/problems/{problem_id}/hint")
async def get_hint(problem_id: str, hint_index: int = 0):
    """Get a hint for a specific problem"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    if hint_index >= len(problem.hints):
        raise HTTPException(status_code=404, detail="Hint not found")
    
    return {"hint": problem.hints[hint_index], "total_hints": len(problem.hints)}

@app.post("/problems/{problem_id}/ai-hint")
async def get_ai_hint(problem_id: str, request: AIFeedbackRequest):
    """Get AI-powered personalized hint based on current progress"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    explainer = AutomataExplainer()
    
    current_progress = {
        "states": len(request.user_automaton.states),
        "transitions": len(request.user_automaton.transitions),
        "start_states": len([s for s in request.user_automaton.states if s.is_start]),
        "accept_states": len([s for s in request.user_automaton.states if s.is_accept])
    }
    
    ai_hint = await explainer.provide_step_guidance(problem.description, current_progress)
    return {"ai_hint": ai_hint}

@app.get("/ai/status")
async def check_ai_status():
    """Check if Ollama AI service is available"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return {
                    "available": True,
                    "models": available_models,
                    "current_model": OLLAMA_MODEL,
                    "generator_model": "codellama:34b",
                    "explainer_model": "deepseek-coder:33b"
                }
            else:
                return {"available": False, "error": "Ollama service not responding"}
    except Exception as e:
        return {"available": False, "error": str(e)}

@app.post("/problems/{problem_id}/generate-solution")
async def generate_solution(problem_id: str):
    """Generate a complete automaton solution using AI"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    generator = AutomataGenerator()
    explainer = AutomataExplainer()
    
    automaton_data = await generator.generate_automaton(problem.description, problem.type)
    
    explanation_data = await explainer.explain_automaton(problem.description, automaton_data)
    
    return {
        "problem_id": problem_id,
        "generated_automaton": automaton_data,
        "explanation": explanation_data,
        "note": "This is a reference solution. Try building your own first!"
    }

@app.post("/problems/{problem_id}/explain-solution")
async def explain_solution(problem_id: str, solution: Solution):
    """Get detailed explanation of a user's automaton solution"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    explainer = AutomataExplainer()
    
    automaton_data = {
        "states": [{"id": s.id, "is_start": s.is_start, "is_accept": s.is_accept} for s in solution.automaton.states],
        "transitions": [{"from": t.from_state, "to": t.to_state, "symbol": t.symbol} for t in solution.automaton.transitions],
        "alphabet": solution.automaton.alphabet
    }
    
    explanation_data = await explainer.explain_automaton(problem.description, automaton_data, solution.automaton)
    
    return explanation_data
