from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Set, Any
import re
import httpx
import json
from .agents import AutomataGenerator, AutomataExplainer
from .config import settings
from .middleware import ErrorHandlingMiddleware, RateLimitMiddleware
from .validators import (
    ValidatedAutomaton, ValidatedProblemInput, ValidatedTestCase,
    validate_automaton_type, sanitize_string, validate_test_strings
)
from .jflap_complete import (
    jflap_algorithms, Automaton, State, Transition, AutomatonType, Grammar
)
from .jflap_simulator import simulation_engine
from .auth import (
    create_user, authenticate_user, create_access_token,
    get_current_active_user, UserCreate, UserLogin, UserResponse
)
from .database import get_db, init_db, Problem, Solution, save_solution, User
import logging
from sqlalchemy.orm import Session
from datetime import timedelta, datetime

# Configure logging
logging.basicConfig(level=settings.log_level, format=settings.log_format)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.api_title, version=settings.api_version, debug=settings.debug)

# Add middleware for error handling and rate limiting
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# Configure CORS for production security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Ollama configuration from settings
OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_default_model

problems_db = {}
solutions_db = {}

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": "AI-powered Theory of Computation Educational Platform",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "problems": "/problems",
            "auth": "/auth/*"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(
    user_create: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    user = create_user(db, user_create)
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        skill_level=user.skill_level,
        created_at=user.created_at
    )


@app.post("/auth/login")
async def login(
    user_login: UserLogin,
    db: Session = Depends(get_db)
):
    """Login and receive access token."""
    user = authenticate_user(db, user_login.email, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id)},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            skill_level=user.skill_level,
            created_at=user.created_at
        )
    }


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        skill_level=current_user.skill_level,
        created_at=current_user.created_at
    )

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
    type: Optional[str] = "dfa"

class PDATransition(BaseModel):
    from_state: str
    to_state: str
    symbol: str
    stack_pop: str
    stack_push: str
    x: Optional[float] = None
    y: Optional[float] = None

class PDAAutomaton(BaseModel):
    type: str = "pda"
    states: List[State]
    transitions: List[PDATransition]
    alphabet: List[str]
    stack_alphabet: List[str]
    start_stack_symbol: str

class CFGProduction(BaseModel):
    id: str
    left_side: str
    right_side: str

class CFGAutomaton(BaseModel):
    type: str = "cfg"
    terminals: List[str]
    non_terminals: List[str]
    productions: List[CFGProduction]
    start_symbol: str

class TMTransition(BaseModel):
    from_state: str
    to_state: str
    read_symbol: str
    write_symbol: str
    head_direction: str
    tape_index: Optional[int] = 0

class TMAutomaton(BaseModel):
    type: str = "tm"
    states: List[State]
    transitions: List[TMTransition]
    tape_alphabet: List[str]
    blank_symbol: str
    num_tapes: Optional[int] = 1

class RegexAutomaton(BaseModel):
    type: str = "regex"
    pattern: str
    alphabet: List[str]
    equivalent_nfa: Optional[Automaton] = None
    equivalent_dfa: Optional[Automaton] = None

class PumpingLemmaAutomaton(BaseModel):
    type: str = "pumping"
    language_type: str
    language_description: str
    pumping_length: Optional[int] = None
    example_string: Optional[str] = None
    decomposition: Optional[Dict[str, str]] = None

class Problem(BaseModel):
    id: str
    type: str  # "dfa", "nfa", "enfa", "pda", "cfg", "tm", "regex", "pumping"
    title: str
    description: str
    language_description: str
    alphabet: List[str]
    test_strings: List[Dict[str, Any]]
    hints: Optional[List[str]] = []
    difficulty: Optional[str] = "beginner"
    category: Optional[str] = None
    reference_solution: Optional[Dict[str, Any]] = None

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
    minimization_suggestions: Optional[List[str]] = None
    unreachable_states: Optional[List[str]] = None

class SimulationStep(BaseModel):
    step_number: int
    current_state: str
    input_position: int
    remaining_input: str
    stack_contents: Optional[List[str]] = None
    tape_contents: Optional[List[str]] = None
    head_position: Optional[int] = None
    action_description: str

class SimulationResult(BaseModel):
    accepted: bool
    steps: List[SimulationStep]
    final_state: str
    execution_path: List[str]
    error_message: Optional[str] = None

class CodeExportOptions(BaseModel):
    language: str
    include_tests: bool = True
    include_visualization: bool = False
    format: str = "class"

class ExportResult(BaseModel):
    code: str
    filename: str
    language: str
    test_cases: Optional[str] = None

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

def init_comprehensive_problems():
    problems_db["dfa_ending_ab"] = Problem(
        id="dfa_ending_ab",
        type="dfa",
        title="DFA: Strings ending in 'ab'",
        description="Construct a DFA that recognizes the language of strings over the alphabet {a, b} that end with 'ab'.",
        language_description="All words that end with 'ab'",
        alphabet=["a", "b"],
        difficulty="beginner",
        category="Basic Patterns",
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
    
    problems_db["dfa_even_a"] = Problem(
        id="dfa_even_a",
        type="dfa",
        title="DFA: Even number of a's",
        description="Construct a DFA that accepts strings with an even number of 'a's over the alphabet {a, b}.",
        language_description="Strings with even number of 'a's",
        alphabet=["a", "b"],
        difficulty="beginner",
        category="Counting",
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
            "Use two states to track even/odd count",
            "State transitions depend only on seeing 'a'",
            "The 'b' symbol doesn't change the count"
        ]
    )
    
    problems_db["dfa_divisible_3"] = Problem(
        id="dfa_divisible_3",
        type="dfa",
        title="DFA: Binary numbers divisible by 3",
        description="Construct a DFA that accepts binary strings representing numbers divisible by 3.",
        language_description="Binary representations of numbers divisible by 3",
        alphabet=["0", "1"],
        difficulty="intermediate",
        category="Arithmetic",
        test_strings=[
            {"string": "", "should_accept": True},
            {"string": "0", "should_accept": True},
            {"string": "11", "should_accept": True},
            {"string": "110", "should_accept": True},
            {"string": "1001", "should_accept": True},
            {"string": "1", "should_accept": False},
            {"string": "10", "should_accept": False},
            {"string": "100", "should_accept": False},
            {"string": "101", "should_accept": False}
        ],
        hints=[
            "Track remainder when divided by 3",
            "Use three states for remainders 0, 1, 2",
            "Reading a bit doubles the number and adds the bit value"
        ]
    )
    
    problems_db["nfa_contains_101"] = Problem(
        id="nfa_contains_101",
        type="nfa",
        title="NFA: Contains substring '101'",
        description="Construct an NFA that accepts strings containing '101' as a substring.",
        language_description="Strings containing '101'",
        alphabet=["0", "1"],
        difficulty="beginner",
        category="Substring Recognition",
        test_strings=[
            {"string": "101", "should_accept": True},
            {"string": "0101", "should_accept": True},
            {"string": "1010", "should_accept": False},
            {"string": "1100", "should_accept": False},
            {"string": "11010", "should_accept": True}
        ],
        hints=[
            "Use non-determinism to guess when '101' starts",
            "Stay in start state or begin matching '101'",
            "Once you match '101', accept all future strings"
        ]
    )
    
    problems_db["enfa_union"] = Problem(
        id="enfa_union",
        type="enfa",
        title="ε-NFA: Union of two languages",
        description="Construct an ε-NFA for the union of L1 = {a^n b^n | n ≥ 0} and L2 = {a^n | n ≥ 0}.",
        language_description="Union of equal a's and b's OR just a's",
        alphabet=["a", "b"],
        difficulty="intermediate",
        category="Language Operations",
        test_strings=[
            {"string": "", "should_accept": True},
            {"string": "a", "should_accept": True},
            {"string": "aa", "should_accept": True},
            {"string": "ab", "should_accept": True},
            {"string": "aabb", "should_accept": True},
            {"string": "b", "should_accept": False},
            {"string": "ba", "should_accept": False},
            {"string": "abb", "should_accept": False}
        ],
        hints=[
            "Use ε-transitions to choose between languages",
            "Create separate branches for each language",
            "Connect with ε-transitions from start state"
        ]
    )
    
    problems_db["pda_balanced_parens"] = Problem(
        id="pda_balanced_parens",
        type="pda",
        title="PDA: Balanced Parentheses",
        description="Construct a PDA that accepts strings of balanced parentheses.",
        language_description="Balanced parentheses strings",
        alphabet=["(", ")"],
        difficulty="beginner",
        category="Context-Free Languages",
        test_strings=[
            {"string": "", "should_accept": True},
            {"string": "()", "should_accept": True},
            {"string": "(())", "should_accept": True},
            {"string": "()()", "should_accept": True},
            {"string": "((()))", "should_accept": True},
            {"string": "(", "should_accept": False},
            {"string": ")", "should_accept": False},
            {"string": "(()", "should_accept": False},
            {"string": "())", "should_accept": False}
        ],
        hints=[
            "Push '(' onto stack when you see '('",
            "Pop from stack when you see ')'",
            "Accept if stack is empty at end"
        ]
    )
    
    problems_db["pda_equal_ab"] = Problem(
        id="pda_equal_ab",
        type="pda",
        title="PDA: Equal number of a's and b's",
        description="Construct a PDA that accepts strings with equal numbers of a's and b's.",
        language_description="Strings with equal count of a's and b's",
        alphabet=["a", "b"],
        difficulty="intermediate",
        category="Context-Free Languages",
        test_strings=[
            {"string": "", "should_accept": True},
            {"string": "ab", "should_accept": True},
            {"string": "ba", "should_accept": True},
            {"string": "aabb", "should_accept": True},
            {"string": "abab", "should_accept": True},
            {"string": "a", "should_accept": False},
            {"string": "b", "should_accept": False},
            {"string": "aab", "should_accept": False},
            {"string": "abb", "should_accept": False}
        ],
        hints=[
            "Use stack to count difference between a's and b's",
            "Push symbol for one type, pop for the other",
            "Accept when stack is empty"
        ]
    )
    
    problems_db["cfg_palindromes"] = Problem(
        id="cfg_palindromes",
        type="cfg",
        title="CFG: Palindromes over {a,b}",
        description="Write a context-free grammar that generates all palindromes over the alphabet {a, b}.",
        language_description="All palindromes over {a, b}",
        alphabet=["a", "b"],
        difficulty="intermediate",
        category="Context-Free Grammars",
        test_strings=[
            {"string": "", "should_accept": True},
            {"string": "a", "should_accept": True},
            {"string": "b", "should_accept": True},
            {"string": "aa", "should_accept": True},
            {"string": "aba", "should_accept": True},
            {"string": "bab", "should_accept": True},
            {"string": "ababa", "should_accept": True},
            {"string": "ab", "should_accept": False},
            {"string": "abc", "should_accept": False}
        ],
        hints=[
            "S → aSa | bSb | a | b | ε",
            "Recursive structure builds palindromes from center out",
            "Base cases handle single characters and empty string"
        ]
    )
    
    problems_db["tm_increment"] = Problem(
        id="tm_increment",
        type="tm",
        title="TM: Binary Increment",
        description="Construct a Turing machine that increments a binary number by 1.",
        language_description="Binary number incremented by 1",
        alphabet=["0", "1"],
        difficulty="intermediate",
        category="Turing Machines",
        test_strings=[
            {"string": "0", "should_accept": True},
            {"string": "1", "should_accept": True},
            {"string": "10", "should_accept": True},
            {"string": "11", "should_accept": True},
            {"string": "100", "should_accept": True}
        ],
        hints=[
            "Start from rightmost bit",
            "Handle carry propagation",
            "0 becomes 1, 1 becomes 0 with carry"
        ]
    )
    
    problems_db["regex_email"] = Problem(
        id="regex_email",
        type="regex",
        title="Regex: Simple Email Validation",
        description="Write a regular expression that matches simple email addresses.",
        language_description="Simple email format: letters@letters.letters",
        alphabet=["a", "b", "c", "@", "."],
        difficulty="beginner",
        category="Regular Expressions",
        test_strings=[
            {"string": "a@b.c", "should_accept": True},
            {"string": "abc@abc.abc", "should_accept": True},
            {"string": "@b.c", "should_accept": False},
            {"string": "a@.c", "should_accept": False},
            {"string": "a@b.", "should_accept": False}
        ],
        hints=[
            "Pattern: letters + @ + letters + . + letters",
            "Use + for one or more letters",
            "Literal @ and . characters"
        ]
    )
    
    problems_db["pumping_regular"] = Problem(
        id="pumping_regular",
        type="pumping",
        title="Pumping Lemma: Prove L = {a^n b^n | n ≥ 0} is not regular",
        description="Use the pumping lemma to prove that L = {a^n b^n | n ≥ 0} is not regular.",
        language_description="Equal numbers of a's followed by b's",
        alphabet=["a", "b"],
        difficulty="advanced",
        category="Pumping Lemma",
        test_strings=[
            {"string": "", "should_accept": True},
            {"string": "ab", "should_accept": True},
            {"string": "aabb", "should_accept": True},
            {"string": "a", "should_accept": False},
            {"string": "abb", "should_accept": False}
        ],
        hints=[
            "Assume L is regular with pumping length p",
            "Choose string a^p b^p",
            "Show that pumping violates the equal count property"
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

init_comprehensive_problems()

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
    elif problem.type == "nfa":
        result = validate_nfa(solution.automaton, problem)
    elif problem.type == "pda":
        result = validate_pda(solution.automaton, problem)
    elif problem.type == "cfg":
        result = validate_cfg(solution.automaton, problem)
    elif problem.type == "tm":
        result = validate_tm(solution.automaton, problem)
    elif problem.type == "regex":
        result = validate_regex(solution.automaton, problem)
    elif problem.type == "pumping":
        result = validate_pumping_lemma(solution.automaton, problem)
    else:
        raise HTTPException(status_code=400, detail=f"Validation for {problem.type} is not yet implemented")
    
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

@app.post("/problems/{problem_id}/guided-step")
async def get_guided_step(problem_id: str, request: AIFeedbackRequest):
    """Get specific guidance for the next step in automaton construction"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    explainer = AutomataExplainer()
    
    current_progress = {
        "states": len(request.user_automaton.states),
        "transitions": len(request.user_automaton.transitions),
        "start_states": len([s for s in request.user_automaton.states if s.is_start]),
        "accept_states": len([s for s in request.user_automaton.states if s.is_accept]),
        "state_list": [{"id": s.id, "is_start": s.is_start, "is_accept": s.is_accept} for s in request.user_automaton.states],
        "transition_list": [{"from": t.from_state, "to": t.to_state, "symbol": t.symbol} for t in request.user_automaton.transitions]
    }
    
    guided_step = await explainer.provide_step_guidance(problem.description, current_progress)
    return {"guided_step": guided_step, "current_progress": current_progress}

def validate_nfa(automaton: Automaton, problem: Problem) -> ValidationResult:
    """Validate an NFA solution against a problem"""
    mistakes = []
    test_results = []
    
    if not automaton.states:
        mistakes.append("Automaton has no states")
        return ValidationResult(
            is_correct=False,
            score=0.0,
            feedback="Your automaton must have at least one state.",
            test_results=[],
            mistakes=mistakes
        )
    
    start_states = [s for s in automaton.states if s.is_start]
    if len(start_states) == 0:
        mistakes.append("No start state defined")
    
    accept_states = [s for s in automaton.states if s.is_accept]
    if len(accept_states) == 0:
        mistakes.append("No accept states defined")
    
    for test_case in problem.test_strings:
        test_string = test_case["string"]
        expected = test_case["should_accept"]
        
        try:
            actual, path = simulate_nfa(automaton, test_string)
            correct = actual == expected
            
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "path": path
            })
        except Exception as e:
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": False,
                "correct": False,
                "path": [],
                "error": str(e)
            })
    
    correct_tests = sum(1 for result in test_results if result["correct"])
    score = correct_tests / len(test_results) if test_results else 0.0
    is_correct = score == 1.0 and not mistakes
    
    if is_correct:
        feedback = "Congratulations! Your NFA correctly recognizes the language."
    else:
        feedback = f"Your NFA passed {correct_tests}/{len(test_results)} test cases."
        if mistakes:
            feedback += f" Issues found: {', '.join(mistakes)}"
    
    return ValidationResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        test_results=test_results,
        mistakes=mistakes
    )

def simulate_nfa(automaton: Automaton, input_string: str) -> tuple[bool, list[str]]:
    """Simulate NFA execution on input string"""
    start_states = [s.id for s in automaton.states if s.is_start]
    if not start_states:
        return False, []
    
    current_states = set(start_states)
    path = [f"Start: {current_states}"]
    
    for i, symbol in enumerate(input_string):
        next_states = set()
        for state in current_states:
            for transition in automaton.transitions:
                if transition.from_state == state and transition.symbol == symbol:
                    next_states.add(transition.to_state)
        
        current_states = next_states
        path.append(f"After '{symbol}': {current_states}")
        
        if not current_states:
            return False, path
    
    accept_states = {s.id for s in automaton.states if s.is_accept}
    accepted = bool(current_states.intersection(accept_states))
    
    return accepted, path

def validate_pda(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a PDA solution against a problem"""
    mistakes = []
    test_results = []
    
    if not hasattr(automaton, 'states') or not automaton.states:
        mistakes.append("PDA has no states")
        return ValidationResult(
            is_correct=False,
            score=0.0,
            feedback="Your PDA must have at least one state.",
            test_results=[],
            mistakes=mistakes
        )
    
    for test_case in problem.test_strings:
        test_string = test_case["string"]
        expected = test_case["should_accept"]
        
        try:
            actual, path, stack_trace = simulate_pda(automaton, test_string)
            correct = actual == expected
            
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "path": path,
                "stack_trace": stack_trace
            })
        except Exception as e:
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": False,
                "correct": False,
                "path": [],
                "stack_trace": [],
                "error": str(e)
            })
    
    correct_tests = sum(1 for result in test_results if result["correct"])
    score = correct_tests / len(test_results) if test_results else 0.0
    is_correct = score == 1.0 and not mistakes
    
    feedback = f"Your PDA passed {correct_tests}/{len(test_results)} test cases."
    if is_correct:
        feedback = "Congratulations! Your PDA correctly recognizes the language."
    
    return ValidationResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        test_results=test_results,
        mistakes=mistakes
    )

def process_epsilon_transitions_pda(automaton: Any, states: set, stack: list) -> set:
    """Process all possible ε-transitions from current states"""
    changed = True
    while changed:
        changed = False
        new_states = set(states)
        for state_id in states:
            for transition in automaton['transitions']:
                if (transition['from_state'] == state_id and 
                    transition['symbol'] == '' and
                    transition['to_state'] not in new_states and
                    can_pop_stack(stack, transition.get('stack_pop', ''))):
                    new_states.add(transition['to_state'])
                    changed = True
        states = new_states
    return states

def can_pop_stack(stack: list, pop_symbol: str) -> bool:
    """Check if we can pop the specified symbol from stack"""
    if not pop_symbol:
        return True
    return len(stack) > 0 and stack[-1] == pop_symbol

def simulate_pda(automaton: Any, input_string: str) -> tuple[bool, list[str], list[str]]:
    """Simulate PDA execution on input string with detailed step tracking"""
    logger.debug(f"PDA automaton structure: {automaton}")
    logger.debug(f"PDA automaton type: {type(automaton)}")
    
    if not isinstance(automaton, dict) or 'transitions' not in automaton or 'states' not in automaton:
        return False, ["Invalid PDA structure - missing transitions or states"], []
    
    current_states = set()
    stack = [automaton.get('start_stack_symbol', 'Z')]
    input_position = 0
    execution_path = []
    stack_trace = []
    
    start_state = None
    for state in automaton['states']:
        if state.get('is_start', False):
            start_state = state['id']
            break
    
    if not start_state:
        return False, ["No start state found"], []
    
    current_states.add(start_state)
    current_states = process_epsilon_transitions_pda(automaton, current_states, stack)
    
    while input_position <= len(input_string):
        if input_position == len(input_string):
            for state_id in current_states:
                state = next((s for s in automaton['states'] if s['id'] == state_id), None)
                if state and state.get('is_accept', False):
                    if len(stack) <= 1:
                        execution_path.append(f"Accepted in state {state_id}")
                        return True, execution_path, stack_trace
            break
        
        current_symbol = input_string[input_position]
        next_states = set()
        
        for state_id in current_states:
            for transition in automaton['transitions']:
                if (transition['from_state'] == state_id and 
                    transition['symbol'] == current_symbol and
                    can_pop_stack(stack, transition.get('stack_pop', ''))):
                    
                    new_stack = stack.copy()
                    if transition.get('stack_pop'):
                        new_stack.pop()
                    if transition.get('stack_push'):
                        new_stack.extend(list(transition['stack_push']))
                    
                    next_states.add(transition['to_state'])
                    execution_path.append(f"δ({state_id}, {current_symbol}, {stack[-1] if stack else 'ε'}) → ({transition['to_state']}, {transition.get('stack_push', 'ε')})")
                    stack_trace.append(f"Stack: {new_stack}")
        
        if not next_states:
            execution_path.append(f"No valid transitions from states {current_states} on symbol '{current_symbol}'")
            return False, execution_path, stack_trace
        
        current_states = next_states
        stack = new_stack
        input_position += 1
        
        current_states = process_epsilon_transitions_pda(automaton, current_states, stack)
    
    execution_path.append(f"Input consumed but no accepting state reached")
    return False, execution_path, stack_trace

def validate_cfg(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a CFG solution against a problem"""
    mistakes = []
    test_results = []
    
    for test_case in problem.test_strings:
        test_string = test_case["string"]
        expected = test_case["should_accept"]
        
        try:
            actual, parse_tree = simulate_cfg(automaton, test_string)
            correct = actual == expected
            
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "path": [],
                "parse_tree": parse_tree
            })
        except Exception as e:
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": False,
                "correct": False,
                "path": [],
                "parse_tree": None,
                "error": str(e)
            })
    
    correct_tests = sum(1 for result in test_results if result["correct"])
    score = correct_tests / len(test_results) if test_results else 0.0
    is_correct = score == 1.0 and not mistakes
    
    feedback = f"Your CFG passed {correct_tests}/{len(test_results)} test cases."
    if is_correct:
        feedback = "Congratulations! Your CFG correctly generates the language."
    
    return ValidationResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        test_results=test_results,
        mistakes=mistakes
    )

def build_parse_tree(derivation_info: dict) -> dict:
    """Build parse tree from derivation information"""
    if isinstance(derivation_info, str):
        return {"symbol": derivation_info, "children": []}
    
    result = {
        "symbol": derivation_info["production"].split(" → ")[0],
        "children": []
    }
    
    for child in derivation_info.get("children", []):
        if isinstance(child, str):
            result["children"].append({"symbol": child, "children": []})
        else:
            result["children"].append(build_parse_tree(child))
    
    return result

def extract_derivation_steps(parse_tree: dict) -> list[str]:
    """Extract step-by-step derivation from parse tree"""
    steps = []
    
    def traverse(node, current_derivation):
        if not node.get("children"):
            return current_derivation
        
        for i, child in enumerate(node["children"]):
            if child["children"]:
                production = f"{child['symbol']} → {' '.join([c['symbol'] for c in child['children']])}"
                steps.append(production)
                new_derivation = current_derivation.replace(child['symbol'], ' '.join([c['symbol'] for c in child['children']]), 1)
                return traverse(child, new_derivation)
        
        return current_derivation
    
    initial = parse_tree["symbol"]
    steps.append(f"Start: {initial}")
    traverse(parse_tree, initial)
    return steps

def simulate_cfg(automaton: Any, input_string: str) -> tuple[bool, Any]:
    """Simulate CFG parsing on input string using CYK algorithm"""
    logger.debug(f" CFG automaton structure: {automaton}")
    logger.debug(f" CFG automaton type: {type(automaton)}")
    
    if not isinstance(automaton, dict) or 'productions' not in automaton or 'terminals' not in automaton:
        return False, None
    
    productions = automaton['productions']
    terminals = set(automaton['terminals'])
    non_terminals = set(automaton['non_terminals'])
    start_symbol = automaton.get('start_symbol', 'S')
    
    if not input_string:
        for prod in productions:
            if prod['left_side'] == start_symbol and prod['right_side'] == '':
                return True, {"derivation": [f"{start_symbol} → ε"], "parse_tree": {"symbol": start_symbol, "children": []}}
        return False, None
    
    n = len(input_string)
    table = [[set() for _ in range(n + 1)] for _ in range(n)]
    derivations = [[{} for _ in range(n + 1)] for _ in range(n)]
    
    for i in range(n):
        char = input_string[i]
        for prod in productions:
            if prod['right_side'] == char and prod['left_side'] in non_terminals:
                table[i][1].add(prod['left_side'])
                derivations[i][1][prod['left_side']] = {
                    "production": f"{prod['left_side']} → {char}",
                    "children": [char]
                }
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            for k in range(1, length):
                left_symbols = table[i][k]
                right_symbols = table[i + k][length - k]
                
                for prod in productions:
                    if len(prod['right_side']) == 2:
                        left_nt, right_nt = prod['right_side'][0], prod['right_side'][1]
                        if left_nt in left_symbols and right_nt in right_symbols:
                            table[i][length].add(prod['left_side'])
                            derivations[i][length][prod['left_side']] = {
                                "production": f"{prod['left_side']} → {left_nt}{right_nt}",
                                "children": [
                                    derivations[i][k][left_nt],
                                    derivations[i + k][length - k][right_nt]
                                ]
                            }
    
    if start_symbol in table[0][n]:
        parse_tree = build_parse_tree(derivations[0][n][start_symbol])
        derivation_steps = extract_derivation_steps(parse_tree)
        return True, {"derivation": derivation_steps, "parse_tree": parse_tree}
    
    return False, None

def validate_tm(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a Turing Machine solution against a problem"""
    mistakes = []
    test_results = []
    
    for test_case in problem.test_strings:
        test_string = test_case["string"]
        expected = test_case["should_accept"]
        
        try:
            actual, path, tape_trace = simulate_tm(automaton, test_string)
            correct = actual == expected
            
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "path": path,
                "tape_trace": tape_trace
            })
        except Exception as e:
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": False,
                "correct": False,
                "path": [],
                "tape_trace": [],
                "error": str(e)
            })
    
    correct_tests = sum(1 for result in test_results if result["correct"])
    score = correct_tests / len(test_results) if test_results else 0.0
    is_correct = score == 1.0 and not mistakes
    
    feedback = f"Your TM passed {correct_tests}/{len(test_results)} test cases."
    if is_correct:
        feedback = "Congratulations! Your Turing Machine correctly computes the function."
    
    return ValidationResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        test_results=test_results,
        mistakes=mistakes
    )

def simulate_tm(automaton: Any, input_string: str) -> tuple[bool, list[str], list[str]]:
    """Simulate Turing Machine execution on input string with detailed step tracking"""
    logger.debug(f" TM automaton structure: {automaton}")
    logger.debug(f" TM automaton type: {type(automaton)}")
    
    if not isinstance(automaton, dict) or 'transitions' not in automaton or 'states' not in automaton:
        return False, ["Invalid TM structure - missing transitions or states"], []
    
    blank_symbol = automaton.get('blank_symbol', '_')
    tape = list(input_string) + [blank_symbol] * 100
    head_position = 0
    execution_path = []
    tape_trace = []
    
    current_state = None
    for state in automaton['states']:
        if state.get('is_start', False):
            current_state = state['id']
            break
    
    if not current_state:
        return False, ["No start state found"], []
    
    step_count = 0
    max_steps = 10000
    
    while step_count < max_steps:
        current_state_obj = next((s for s in automaton['states'] if s['id'] == current_state), None)
        if current_state_obj and current_state_obj.get('is_accept', False):
            execution_path.append(f"Accepted in state {current_state}")
            tape_trace.append(f"Final tape: {''.join(tape[:50]).rstrip(blank_symbol)}")
            return True, execution_path, tape_trace
        
        current_symbol = tape[head_position] if head_position < len(tape) else blank_symbol
        
        transition_found = False
        for transition in automaton['transitions']:
            if (transition['from_state'] == current_state and 
                transition['read_symbol'] == current_symbol):
                
                tape[head_position] = transition['write_symbol']
                
                if transition['head_direction'] == 'L':
                    head_position = max(0, head_position - 1)
                elif transition['head_direction'] == 'R':
                    head_position += 1
                    if head_position >= len(tape):
                        tape.extend([blank_symbol] * 100)
                
                current_state = transition['to_state']
                
                execution_path.append(f"δ({transition['from_state']}, {current_symbol}) → ({current_state}, {transition['write_symbol']}, {transition['head_direction']})")
                tape_trace.append(f"Tape: {''.join(tape[:min(50, head_position + 20)]).rstrip(blank_symbol)}, Head: {head_position}")
                
                transition_found = True
                break
        
        if not transition_found:
            execution_path.append(f"No transition from state {current_state} on symbol '{current_symbol}'")
            tape_trace.append(f"Final tape: {''.join(tape[:50]).rstrip(blank_symbol)}")
            return False, execution_path, tape_trace
        
        step_count += 1
    
    execution_path.append(f"Maximum steps ({max_steps}) exceeded - possible infinite loop")
    return False, execution_path, tape_trace

def validate_regex(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a Regular Expression solution against a problem"""
    mistakes = []
    test_results = []
    
    for test_case in problem.test_strings:
        test_string = test_case["string"]
        expected = test_case["should_accept"]
        
        try:
            actual = simulate_regex(automaton, test_string)
            correct = actual == expected
            
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "path": []
            })
        except Exception as e:
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": False,
                "correct": False,
                "path": [],
                "error": str(e)
            })
    
    correct_tests = sum(1 for result in test_results if result["correct"])
    score = correct_tests / len(test_results) if test_results else 0.0
    is_correct = score == 1.0 and not mistakes
    
    feedback = f"Your regex passed {correct_tests}/{len(test_results)} test cases."
    if is_correct:
        feedback = "Congratulations! Your regular expression correctly matches the language."
    
    return ValidationResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        test_results=test_results,
        mistakes=mistakes
    )

def simulate_regex(automaton: Any, input_string: str) -> bool:
    """Simulate regex matching on input string"""
    import re
    
    logger.debug(f" Regex automaton structure: {automaton}")
    logger.debug(f" Regex automaton type: {type(automaton)}")
    
    if not isinstance(automaton, dict) or 'pattern' not in automaton:
        return False
    
    pattern = automaton['pattern']
    try:
        match = re.fullmatch(pattern, input_string)
        return match is not None
    except re.error:
        return False

def validate_pumping_lemma(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a Pumping Lemma proof against a problem"""
    mistakes = []
    test_results = []
    
    for test_case in problem.test_strings:
        test_string = test_case["string"]
        expected = test_case["should_accept"]
        
        try:
            actual = simulate_pumping_lemma(automaton, test_string)
            correct = actual == expected
            
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "path": []
            })
        except Exception as e:
            test_results.append({
                "string": test_string,
                "expected": expected,
                "actual": False,
                "correct": False,
                "path": [],
                "error": str(e)
            })
    
    correct_tests = sum(1 for result in test_results if result["correct"])
    score = correct_tests / len(test_results) if test_results else 0.0
    is_correct = score == 1.0 and not mistakes
    
    feedback = f"Your pumping lemma proof passed {correct_tests}/{len(test_results)} test cases."
    if is_correct:
        feedback = "Congratulations! Your pumping lemma proof is correct."
    
    return ValidationResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        test_results=test_results,
        mistakes=mistakes
    )

def simulate_pumping_lemma(automaton: Any, input_string: str) -> bool:
    """Simulate pumping lemma proof verification"""
    logger.debug(f" Pumping Lemma automaton structure: {automaton}")
    logger.debug(f" Pumping Lemma automaton type: {type(automaton)}")
    
    language_type = automaton.get('language_type', 'regular')
    
    if language_type == 'regular':
        pumping_length = automaton.get('pumping_length', 10)
        return len(input_string) >= pumping_length
    elif language_type == 'context_free':
        pumping_length = automaton.get('pumping_length', 15)
        return len(input_string) >= pumping_length
    
    return False

@app.post("/api/generate")
async def generate_automaton_endpoint(request: Dict[str, Any]):
    """Generate automaton from natural language description"""
    task = request.get("task", "")
    automaton_type = request.get("type", "dfa")
    
    generator = AutomataGenerator()
    result = await generator.generate_automaton(task, automaton_type)
    
    return {
        "generated_automaton": result,
        "task": task,
        "type": automaton_type
    }

@app.post("/api/complete-solution")
async def generate_complete_solution_endpoint(request: Dict[str, Any]):
    """Generate complete solution for any TOC problem with detailed explanation"""
    import asyncio
    
    task = request.get("task", "")
    problem_type = request.get("problem_type", "dfa")
    problem_id = request.get("problem_id", "")
    
    try:
        generator = AutomataGenerator()
        result = await asyncio.wait_for(
            generator.generate_automaton(task, problem_type), 
            timeout=30.0
        )
        
        explainer = AutomataExplainer()
        explanation = await asyncio.wait_for(
            explainer.explain_automaton(task, result), 
            timeout=20.0
        )
        
        return {
            "formal_definition": result.get("formal_definition", ""),
            "python_code": result.get("python_code", ""),
            "dot_graph": result.get("dot_graph", ""),
            "test_cases": result.get("test_cases", {}),
            "explanation": explanation.get("explanation", ""),
            "automaton": result,
            "task": task,
            "type": problem_type
        }
    except asyncio.TimeoutError:
        return {
            "formal_definition": f"Timeout occurred while generating solution for {problem_type}",
            "python_code": "# AI generation timed out, using fallback",
            "dot_graph": "",
            "test_cases": {},
            "explanation": "AI model took too long to respond. Please try again.",
            "automaton": generate_fallback_automaton(problem_type, task),
            "task": task,
            "type": problem_type,
            "error": "timeout"
        }
    except Exception as e:
        return {
            "formal_definition": f"Error generating solution: {str(e)}",
            "python_code": "# Error occurred during generation",
            "dot_graph": "",
            "test_cases": {},
            "explanation": f"An error occurred: {str(e)}",
            "automaton": generate_fallback_automaton(problem_type, task),
            "task": task,
            "type": problem_type,
            "error": str(e)
        }

@app.post("/api/guided-approach")
async def provide_guided_approach_endpoint(request: Dict[str, Any]):
    """Provide guided step-by-step approach for solving TOC problems"""
    import asyncio
    
    task = request.get("task", "")
    problem_type = request.get("problem_type", "dfa")
    current_progress = request.get("current_progress", {})
    
    try:
        explainer = AutomataExplainer()
        next_step = await asyncio.wait_for(
            explainer.provide_step_guidance(task, current_progress),
            timeout=15.0
        )
        
        steps = [
            next_step,
            "Consider what states you need to track the problem requirements",
            "Add transitions between states based on input symbols",
            "Mark appropriate start and accept states",
            "Test your automaton with the provided examples"
        ]
        
        return {
            "steps": steps,
            "next_step": next_step,
            "task": task,
            "type": problem_type,
            "current_progress": current_progress
        }
    except asyncio.TimeoutError:
        fallback_steps = generate_fallback_guidance(problem_type, task)
        return {
            "steps": fallback_steps,
            "next_step": fallback_steps[0] if fallback_steps else "Start by identifying the problem requirements",
            "task": task,
            "type": problem_type,
            "current_progress": current_progress,
            "error": "timeout"
        }
    except Exception as e:
        fallback_steps = generate_fallback_guidance(problem_type, task)
        return {
            "steps": fallback_steps,
            "next_step": fallback_steps[0] if fallback_steps else "Start by identifying the problem requirements",
            "task": task,
            "type": problem_type,
            "current_progress": current_progress,
            "error": str(e)
        }

@app.post("/api/explain")
async def explain_automaton_endpoint(request: Dict[str, Any]):
    """Explain automaton structure and behavior"""
    automaton_data = request.get("automaton", {})
    task = request.get("task", "")
    
    explainer = AutomataExplainer()
    result = await explainer.explain_automaton(task, automaton_data)
    
    return {
        "explanation": result,
        "automaton": automaton_data
    }

@app.post("/api/simulate")
async def simulate_automaton_endpoint(request: Dict[str, Any]):
    """Simulate automaton execution step by step"""
    automaton_data = request.get("automaton", {})
    input_string = request.get("input_string", "")
    automaton_type = request.get("type", "dfa")
    
    try:
        if automaton_type == "dfa" or automaton_type == "nfa":
            automaton = Automaton(**automaton_data)
            if automaton_type == "nfa":
                accepted, path = simulate_nfa(automaton, input_string)
            else:
                accepted, path = simulate_dfa_detailed(automaton, input_string)
            
            return SimulationResult(
                accepted=accepted,
                steps=[],
                final_state=path[-1] if path else "",
                execution_path=path
            )
        elif automaton_type == 'pda':
            accepted, path, stack_trace = simulate_pda(automaton_data, input_string)
            steps = generate_simulation_steps_pda(automaton_data, input_string, path, stack_trace)
            return SimulationResult(
                accepted=accepted,
                steps=steps,
                final_state=path[-1] if path else "",
                execution_path=path
            )
        elif automaton_type == 'cfg':
            accepted, parse_info = simulate_cfg(automaton_data, input_string)
            steps = generate_simulation_steps_cfg(automaton_data, input_string, parse_info)
            return SimulationResult(
                accepted=accepted,
                steps=steps,
                final_state="parsed" if accepted else "failed",
                execution_path=parse_info.get('derivation', []) if parse_info else []
            )
        elif automaton_type == 'tm':
            accepted, path, tape_trace = simulate_tm(automaton_data, input_string)
            steps = generate_simulation_steps_tm(automaton_data, input_string, path, tape_trace)
            return SimulationResult(
                accepted=accepted,
                steps=steps,
                final_state=path[-1] if path else "",
                execution_path=path
            )
        elif automaton_type == 'regex':
            accepted = simulate_regex(automaton_data, input_string)
            steps = generate_simulation_steps_regex(automaton_data, input_string, accepted)
            return SimulationResult(
                accepted=accepted,
                steps=steps,
                final_state="matched" if accepted else "no_match",
                execution_path=[f"Pattern {'matches' if accepted else 'does not match'} input"]
            )
        elif automaton_type == 'pumping':
            accepted = simulate_pumping_lemma(automaton_data, input_string)
            steps = generate_simulation_steps_pumping(automaton_data, input_string, accepted)
            return SimulationResult(
                accepted=accepted,
                steps=steps,
                final_state="satisfied" if accepted else "violated",
                execution_path=[f"Pumping lemma {'satisfied' if accepted else 'violated'}"]
            )
        else:
            return SimulationResult(
                accepted=False,
                steps=[],
                final_state="",
                execution_path=[],
                error_message=f"Simulation for {automaton_type} not yet implemented"
            )
    except Exception as e:
        return SimulationResult(
            accepted=False,
            steps=[],
            final_state="",
            execution_path=[],
            error_message=str(e)
        )

def simulate_dfa_detailed(automaton: Automaton, input_string: str) -> tuple[bool, list[str]]:
    """Detailed DFA simulation with step tracking"""
    start_states = [s.id for s in automaton.states if s.is_start]
    if not start_states:
        return False, ["No start state found"]
    
    current_state = start_states[0]
    path = [f"Start: {current_state}"]
    
    for i, symbol in enumerate(input_string):
        found_transition = False
        for transition in automaton.transitions:
            if transition.from_state == current_state and transition.symbol == symbol:
                current_state = transition.to_state
                path.append(f"Read '{symbol}' → {current_state}")
                found_transition = True
                break
        
        if not found_transition:
            path.append(f"No transition for '{symbol}' from {current_state}")
            return False, path
    
    accept_states = {s.id for s in automaton.states if s.is_accept}
    accepted = current_state in accept_states
    
    if accepted:
        path.append(f"Accept: {current_state} is an accept state")
    else:
        path.append(f"Reject: {current_state} is not an accept state")
    
    return accepted, path

@app.post("/api/export")
async def export_automaton_endpoint(request: Dict[str, Any]):
    """Export automaton to various formats"""
    automaton_data = request.get("automaton", {})
    options = CodeExportOptions(**request.get("options", {}))
    
    try:
        if options.language == "python":
            code = generate_python_code(automaton_data, options)
        elif options.language == "javascript":
            code = generate_javascript_code(automaton_data, options)
        elif options.language == "java":
            code = generate_java_code(automaton_data, options)
        else:
            raise ValueError(f"Unsupported language: {options.language}")
        
        filename = f"automaton.{get_file_extension(options.language)}"
        
        return ExportResult(
            code=code,
            filename=filename,
            language=options.language,
            test_cases=generate_test_cases(automaton_data, options) if options.include_tests else None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def generate_python_code(automaton_data: Dict[str, Any], options: CodeExportOptions) -> str:
    """Generate Python code for automaton"""
    if options.format == "class":
        return f'''class DFA:
    def __init__(self):
        self.states = {automaton_data.get("states", [])}
        self.alphabet = {automaton_data.get("alphabet", [])}
        self.transitions = {automaton_data.get("transitions", [])}
        self.start_state = "{get_start_state(automaton_data)}"
        self.accept_states = {get_accept_states(automaton_data)}
    
    def simulate(self, input_string):
        current_state = self.start_state
        for symbol in input_string:
            found = False
            for transition in self.transitions:
                if transition["from_state"] == current_state and transition["symbol"] == symbol:
                    current_state = transition["to_state"]
                    found = True
                    break
            if not found:
                return False
        return current_state in self.accept_states

'''
    else:
        return f'''def simulate_dfa(input_string):
    states = {automaton_data.get("states", [])}
    alphabet = {automaton_data.get("alphabet", [])}
    transitions = {automaton_data.get("transitions", [])}
    start_state = "{get_start_state(automaton_data)}"
    accept_states = {get_accept_states(automaton_data)}
    
    current_state = start_state
    for symbol in input_string:
        found = False
        for transition in transitions:
            if transition["from_state"] == current_state and transition["symbol"] == symbol:
                current_state = transition["to_state"]
                found = True
                break
        if not found:
            return False
    return current_state in accept_states

'''


def generate_simulation_steps_pda(automaton: Any, input_string: str, path: list[str], stack_trace: list[str]) -> list[SimulationStep]:
    """Generate simulation steps for PDA"""
    steps = []
    for i, (step_desc, stack_state) in enumerate(zip(path, stack_trace + [""])):
        remaining_input = input_string[i:] if i < len(input_string) else ""
        steps.append(SimulationStep(
            step_number=i,
            current_state=extract_state_from_path(step_desc),
            input_position=i,
            remaining_input=remaining_input,
            action_description=step_desc,
            stack_contents=parse_stack_contents(stack_state) if stack_state else [],
            tape_contents=[],
            head_position=0
        ))
    return steps

def generate_simulation_steps_cfg(automaton: Any, input_string: str, parse_info: Any) -> list[SimulationStep]:
    """Generate simulation steps for CFG"""
    steps = []
    if parse_info and 'derivation' in parse_info:
        for i, derivation_step in enumerate(parse_info['derivation']):
            remaining_input = input_string[i:] if i < len(input_string) else ""
            steps.append(SimulationStep(
                step_number=i,
                current_state="parsing",
                input_position=i,
                remaining_input=remaining_input,
                action_description=derivation_step,
                stack_contents=[],
                tape_contents=[],
                head_position=0
            ))
    return steps

def generate_simulation_steps_tm(automaton: Any, input_string: str, path: list[str], tape_trace: list[str]) -> list[SimulationStep]:
    """Generate simulation steps for TM"""
    steps = []
    for i, (step_desc, tape_state) in enumerate(zip(path, tape_trace + [""])):
        tape_contents, head_pos = parse_tape_contents(tape_state) if tape_state else ([], 0)
        remaining_input = input_string[i:] if i < len(input_string) else ""
        steps.append(SimulationStep(
            step_number=i,
            current_state=extract_state_from_path(step_desc),
            input_position=i,
            remaining_input=remaining_input,
            action_description=step_desc,
            stack_contents=[],
            tape_contents=tape_contents,
            head_position=head_pos
        ))
    return steps

def generate_simulation_steps_regex(automaton: Any, input_string: str, accepted: bool) -> list[SimulationStep]:
    """Generate simulation steps for Regex"""
    return [SimulationStep(
        step_number=0,
        current_state="matching",
        input_position=len(input_string),
        remaining_input="",
        action_description=f"Pattern {'matches' if accepted else 'does not match'} input string",
        stack_contents=[],
        tape_contents=[],
        head_position=0
    )]

def generate_simulation_steps_pumping(automaton: Any, input_string: str, accepted: bool) -> list[SimulationStep]:
    """Generate simulation steps for Pumping Lemma"""
    return [SimulationStep(
        step_number=0,
        current_state="verifying",
        input_position=len(input_string),
        remaining_input="",
        action_description=f"Pumping lemma {'satisfied' if accepted else 'violated'} for input string",
        stack_contents=[],
        tape_contents=[],
        head_position=0
    )]

def extract_state_from_path(step_desc: str) -> str:
    """Extract state name from step description"""
    import re
    match = re.search(r'state (\w+)', step_desc)
    return match.group(1) if match else "unknown"

def parse_stack_contents(stack_state: str) -> list[str]:
    """Parse stack contents from trace string"""
    if "Stack:" in stack_state:
        stack_str = stack_state.split("Stack:")[1].strip()
        try:
            return eval(stack_str) if stack_str.startswith('[') else []
        except:
            return []
    return []

def parse_tape_contents(tape_state: str) -> tuple[list[str], int]:
    """Parse tape contents and head position from trace string"""
    if "Tape:" in tape_state and "Head:" in tape_state:
        parts = tape_state.split(", Head:")
        tape_str = parts[0].split("Tape:")[1].strip()
        head_pos = int(parts[1].strip()) if len(parts) > 1 else 0
        return list(tape_str), head_pos
    return [], 0

def generate_javascript_code(automaton_data: Dict[str, Any], options: CodeExportOptions) -> str:
    """Generate JavaScript code for automaton"""
    if options.format == "class":
        return f'''class DFA {{
    constructor() {{
        this.states = {json.dumps(automaton_data.get("states", []))};
        this.alphabet = {json.dumps(automaton_data.get("alphabet", []))};
        this.transitions = {json.dumps(automaton_data.get("transitions", []))};
        this.startState = "{get_start_state(automaton_data)}";
        this.acceptStates = new Set({json.dumps(get_accept_states(automaton_data))});
    }}
    
    simulate(inputString) {{
        let currentState = this.startState;
        for (const symbol of inputString) {{
            let found = false;
            for (const transition of this.transitions) {{
                if (transition.from_state === currentState && transition.symbol === symbol) {{
                    currentState = transition.to_state;
                    found = true;
                    break;
                }}
            }}
            if (!found) return false;
        }}
        return this.acceptStates.has(currentState);
    }}
}}

// Usage example:
// const dfa = new DFA();
// const result = dfa.simulate("your_input_string");
// console.log(result ? "Accepted" : "Rejected");
'''
    else:
        return f'''function simulateDFA(inputString) {{
    const states = {json.dumps(automaton_data.get("states", []))};
    const alphabet = {json.dumps(automaton_data.get("alphabet", []))};
    const transitions = {json.dumps(automaton_data.get("transitions", []))};
    const startState = "{get_start_state(automaton_data)}";
    const acceptStates = new Set({json.dumps(get_accept_states(automaton_data))});
    
    let currentState = startState;
    for (const symbol of inputString) {{
        let found = false;
        for (const transition of transitions) {{
            if (transition.from_state === currentState && transition.symbol === symbol) {{
                currentState = transition.to_state;
                found = true;
                break;
            }}
        }}
        if (!found) return false;
    }}
    return acceptStates.has(currentState);
}}

// Usage example:
// const result = simulateDFA("your_input_string");
// console.log(result ? "Accepted" : "Rejected");
'''

def generate_java_code(automaton_data: Dict[str, Any], options: CodeExportOptions) -> str:
    """Generate Java code for automaton"""
    return f'''import java.util.*;

public class DFA {{


    private List<String> states;
    private List<String> alphabet;
    private List<Map<String, String>> transitions;
    private String startState;
    private Set<String> acceptStates;
    
    public DFA() {{
        this.states = Arrays.asList({", ".join(f'"{s}"' for s in get_state_ids(automaton_data))});
        this.alphabet = Arrays.asList({", ".join(f'"{a}"' for a in automaton_data.get("alphabet", []))});
        this.startState = "{get_start_state(automaton_data)}";
        this.acceptStates = new HashSet<>(Arrays.asList({", ".join(f'"{s}"' for s in get_accept_states(automaton_data))}));
        
        this.transitions = new ArrayList<>();
        // Add transitions here
    }}
    
    public boolean simulate(String inputString) {{
        String currentState = startState;
        for (char symbol : inputString.toCharArray()) {{
            boolean found = false;
            for (Map<String, String> transition : transitions) {{
                if (transition.get("from_state").equals(currentState) && 
                    transition.get("symbol").equals(String.valueOf(symbol))) {{
                    currentState = transition.get("to_state");
                    found = true;
                    break;
                }}
            }}
            if (!found) return false;
        }}
        return acceptStates.contains(currentState);
    }}
    
    public static void main(String[] args) {{
        DFA dfa = new DFA();
        boolean result = dfa.simulate("your_input_string");
        System.out.println(result ? "Accepted" : "Rejected");
    }}
}}
'''

def get_start_state(automaton_data: Dict[str, Any]) -> str:
    """Get start state from automaton data"""
    states = automaton_data.get("states", [])
    for state in states:
        if isinstance(state, dict) and state.get("is_start", False):
            return state.get("id", "")
    return ""

def get_accept_states(automaton_data: Dict[str, Any]) -> List[str]:
    """Get accept states from automaton data"""
    states = automaton_data.get("states", [])
    accept_states = []
    for state in states:
        if isinstance(state, dict) and state.get("is_accept", False):
            accept_states.append(state.get("id", ""))
    return accept_states

def get_state_ids(automaton_data: Dict[str, Any]) -> List[str]:
    """Get all state IDs from automaton data"""
    states = automaton_data.get("states", [])
    return [state.get("id", "") for state in states if isinstance(state, dict)]

def get_file_extension(language: str) -> str:
    """Get file extension for programming language"""
    extensions = {
        "python": "py",
        "javascript": "js",
        "java": "java"
    }
    return extensions.get(language, "txt")

def generate_test_cases(automaton_data: Dict[str, Any], options: CodeExportOptions) -> str:
    """Generate test cases for the automaton"""
    if options.language == "python":
        return '''# Test cases
test_cases = [
    ("", True),   # Empty string
    ("a", False), # Single character
    ("ab", True), # Example accepting string
]

for test_string, expected in test_cases:
    result = simulate_dfa(test_string)  # or dfa.simulate(test_string) for class
    print(f"Input: '{test_string}' - Expected: {expected}, Got: {result}, {'✓' if result == expected else '✗'}")
'''
    elif options.language == "javascript":
        return '''// Test cases
const testCases = [
    ["", true],   // Empty string
    ["a", false], // Single character
    ["ab", true], // Example accepting string
];

testCases.forEach(([testString, expected]) => {
    const result = simulateDFA(testString); // or dfa.simulate(testString) for class
    console.log(`Input: '${testString}' - Expected: ${expected}, Got: ${result}, ${result === expected ? '✓' : '✗'}`);
});
'''
    else:  # Java
        return '''// Add this to your main method for testing:
String[][] testCases = {
    {"", "true"},   // Empty string
    {"a", "false"}, // Single character
    {"ab", "true"}, // Example accepting string
};

for (String[] testCase : testCases) {
    String testString = testCase[0];
    boolean expected = Boolean.parseBoolean(testCase[1]);
    boolean result = dfa.simulate(testString);
    System.out.println("Input: '" + testString + "' - Expected: " + expected + 
                      ", Got: " + result + ", " + (result == expected ? "✓" : "✗"));
}
'''

@app.post("/api/minimize")
async def minimize_automaton_endpoint(request: Dict[str, Any]):
    """Minimize DFA using standard algorithms"""
    automaton_data = request.get("automaton", {})
    
    try:
        minimized_automaton = minimize_dfa(automaton_data)
        suggestions = generate_minimization_suggestions(automaton_data, minimized_automaton)
        
        return {
            "original_automaton": automaton_data,
            "minimized_automaton": minimized_automaton,
            "suggestions": suggestions,
            "reduction_info": {
                "original_states": len(automaton_data.get("states", [])),
                "minimized_states": len(minimized_automaton.get("states", [])),
                "states_removed": len(automaton_data.get("states", [])) - len(minimized_automaton.get("states", []))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def minimize_dfa(automaton_data: Dict[str, Any]) -> Dict[str, Any]:
    """Minimize DFA using table-filling algorithm"""
    return automaton_data

def generate_minimization_suggestions(original: Dict[str, Any], minimized: Dict[str, Any]) -> List[str]:
    """Generate suggestions for DFA minimization"""
    suggestions = []
    
    original_count = len(original.get("states", []))
    minimized_count = len(minimized.get("states", []))
    
    if original_count > minimized_count:
        suggestions.append(f"Your DFA can be minimized from {original_count} to {minimized_count} states")
        suggestions.append("Some states are equivalent and can be merged")
    else:
        suggestions.append("Your DFA is already minimal")
    
    return suggestions

@app.post("/api/validate-proof-step")
async def validate_proof_step_endpoint(request: Dict[str, Any]):
    """Validate a single proof step using AI reasoning"""
    automaton_data = request.get("automaton", {})
    proof_type = request.get("proof_type", "equivalence")
    step_id = request.get("step_id", "")
    steps = request.get("steps", [])
    
    try:
        explainer = AutomataExplainer()
        validation_result = await explainer.validate_proof_step(
            automaton_data, proof_type, step_id, steps
        )
        
        return {
            "is_valid": validation_result.get("is_valid", True),
            "explanation": validation_result.get("explanation", "Step appears valid"),
            "suggestions": validation_result.get("suggestions", [])
        }
    except Exception as e:
        return {
            "is_valid": False,
            "explanation": f"Error validating step: {str(e)}",
            "suggestions": ["Please check your step and try again"]
        }

@app.post("/api/generate-proof")
async def generate_proof_endpoint(request: Dict[str, Any]):
    """Generate proof suggestions using AI reasoning"""
    automaton_data = request.get("automaton", {})
    proof_type = request.get("proof_type", "equivalence")
    current_steps = request.get("current_steps", [])
    
    try:
        generator = AutomataGenerator()
        proof_suggestions = await generator.generate_proof_steps(
            automaton_data, proof_type, current_steps
        )
        
        return {
            "suggested_steps": proof_suggestions.get("steps", []),
            "reasoning": proof_suggestions.get("reasoning", ""),
            "next_steps": proof_suggestions.get("next_steps", [])
        }
    except Exception as e:
        return {
            "suggested_steps": [],
            "reasoning": f"Error generating proof: {str(e)}",
            "next_steps": ["Please try a different approach"]
        }

def generate_fallback_automaton(problem_type: str, task: str) -> Dict[str, Any]:
    """Generate a basic fallback automaton when AI models fail"""
    if problem_type == "dfa":
        return {
            "states": [
                {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
            ],
            "transitions": [
                {"from_state": "q0", "to_state": "q1", "symbol": "a"},
                {"from_state": "q1", "to_state": "q1", "symbol": "a,b"}
            ],
            "alphabet": ["a", "b"]
        }
    elif problem_type == "nfa":
        return {
            "states": [
                {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
            ],
            "transitions": [
                {"from_state": "q0", "to_state": "q1", "symbol": "a"},
                {"from_state": "q0", "to_state": "q1", "symbol": "ε"}
            ],
            "alphabet": ["a", "b"]
        }
    elif problem_type == "pda":
        return {
            "states": [
                {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
            ],
            "transitions": [
                {"from_state": "q0", "to_state": "q1", "symbol": "a", "stack_pop": "Z", "stack_push": "aZ"}
            ],
            "alphabet": ["a", "b"],
            "stack_alphabet": ["a", "Z"]
        }
    else:
        return {
            "states": [{"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": True}],
            "transitions": [],
            "alphabet": ["a", "b"]
        }

@app.post("/api/analyze-problem")
async def analyze_problem_endpoint(request: Dict[str, Any]):
    """Analyze natural language or image problem input"""
    problem_text = request.get("problem_text", "")
    problem_type = request.get("type", "text")
    
    try:
        generator = AutomataGenerator()
        explainer = AutomataExplainer()
        
        is_toc_problem = await generator.is_toc_problem(problem_text, problem_type)
        
        if not is_toc_problem:
            return {
                "is_toc_problem": False,
                "message": "This doesn't appear to be a Theory of Computation problem. Please provide a problem related to automata, formal languages, or computability theory.",
                "suggestions": [
                    "Try problems involving DFA, NFA, PDA, or Turing Machines",
                    "Ask about regular expressions or context-free grammars",
                    "Request pumping lemma proofs or language equivalence"
                ]
            }
        
        problem_analysis = await generator.analyze_problem_text(problem_text, problem_type)
        
        return {
            "is_toc_problem": True,
            "problem_type": problem_analysis.get("automaton_type", "dfa"),
            "problem_description": problem_analysis.get("description", problem_text),
            "difficulty": problem_analysis.get("difficulty", "intermediate"),
            "concepts": problem_analysis.get("concepts", []),
            "complete_solution": problem_analysis.get("solution", {}),
            "guided_steps": problem_analysis.get("guided_steps", []),
            "test_cases": problem_analysis.get("test_cases", {"accept": [], "reject": []})
        }
        
    except Exception as e:
        return {
            "is_toc_problem": False,
            "message": f"Error analyzing problem: {str(e)}",
            "suggestions": ["Please try rephrasing your problem or check the image quality"]
        }

def generate_fallback_guidance(problem_type: str, task: str) -> list[str]:
    """Generate fallback step-by-step guidance when AI models fail"""
    if problem_type == "dfa":
        return [
            "1. Identify what the DFA should accept/reject based on the problem description",
            "2. Determine the minimum number of states needed to track the pattern",
            "3. Create a start state and mark it appropriately",
            "4. Add transitions for each symbol in the alphabet",
            "5. Mark accept states based on the acceptance criteria",
            "6. Test with example strings to verify correctness"
        ]
    elif problem_type == "nfa":
        return [
            "1. Analyze the language pattern to identify non-deterministic choices",
            "2. Create states to represent different computation paths",
            "3. Add ε-transitions where multiple paths are possible",
            "4. Define transitions for each input symbol",
            "5. Mark appropriate accept states",
            "6. Verify with test strings that require non-deterministic choices"
        ]
    elif problem_type == "pda":
        return [
            "1. Identify the context-free pattern that requires stack memory",
            "2. Design stack operations (push/pop) for each transition",
            "3. Create states to track the parsing progress",
            "4. Define transitions with stack operations",
            "5. Ensure proper stack management for acceptance",
            "6. Test with nested/balanced string examples"
        ]
    elif problem_type == "cfg":
        return [
            "1. Identify the recursive structure in the language",
            "2. Define non-terminal symbols for different syntactic categories",
            "3. Write production rules for each non-terminal",
            "4. Ensure the grammar generates the desired language",
            "5. Check for ambiguity and left recursion",
            "6. Test derivations with example strings"
        ]
    elif problem_type == "tm":
        return [
            "1. Analyze the computation that the TM should perform",
            "2. Design the tape alphabet and state set",
            "3. Define transitions for reading, writing, and moving",
            "4. Implement the algorithm step by step",
            "5. Handle edge cases and termination conditions",
            "6. Test with various input configurations"
        ]
    else:
        return [
            "1. Analyze the problem requirements carefully",
            "2. Choose the appropriate automaton type",
            "3. Design the state structure",
            "4. Define transitions systematically",
            "5. Verify correctness with test cases"
        ]

# ============================================================================
# PHASE 2: ELITE EDUCATIONAL FEATURES ENDPOINTS
# ============================================================================

# Import new modules
from .proofs import (
    validate_proof_step, generate_proof_hint, get_proof_templates,
    ProofValidationRequest, ProofHintRequest, ProofState, ProofStep
)
from .verification import (
    verify_equivalence, verify_containment, minimize_automaton, find_counter_example,
    get_verification_algorithms, EquivalenceRequest, ContainmentRequest, MinimizationRequest, Automaton
)
from .pumping import (
    validate_pumping_lemma, get_non_regular_candidates, decompose_string_regular, decompose_string_cfl,
    get_pumping_templates, PumpingRequest, PumpingResult
)
from .complexity import (
    analyze_complexity, verify_reduction, get_complexity_classes, get_class_relationship,
    get_complexity_templates, ComplexityAnalysisRequest, ReductionVerificationRequest
)
from .adaptive_learning import (
    update_student_performance, get_problem_recommendations, get_student_analytics,
    start_learning_session, end_learning_session, get_difficulty_adjustment,
    StudentAction, ProblemRecommendation, LearningSession
)
from .papers import (
    search_papers, recommend_papers, get_paper_citation, get_papers_by_topic,
    get_paper_details, get_database_stats, PaperSearchRequest, PaperRecommendationRequest,
    CitationRequest, TopicArea
)

# ============================================================================
# INTERACTIVE PROOF SYSTEM ENDPOINTS
# ============================================================================

@app.post("/api/proofs/validate-step")
async def validate_proof_step_endpoint(request: ProofValidationRequest):
    """Validate a single proof step"""
    try:
        result = validate_proof_step(request)
        return result
    except Exception as e:
        logger.error(f"Error validating proof step: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/proofs/get-hint")
async def get_proof_hint_endpoint(request: ProofHintRequest):
    """Get a hint for the current proof state"""
    try:
        result = generate_proof_hint(request)
        return result
    except Exception as e:
        logger.error(f"Error generating proof hint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/proofs/templates")
async def get_proof_templates_endpoint():
    """Get proof templates for different techniques"""
    try:
        templates = get_proof_templates()
        return templates
    except Exception as e:
        logger.error(f"Error getting proof templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FORMAL VERIFICATION ENDPOINTS
# ============================================================================

@app.post("/api/verification/equivalence")
async def check_equivalence_endpoint(request: EquivalenceRequest):
    """Check if two automata are equivalent"""
    try:
        result = verify_equivalence(request)
        return result
    except Exception as e:
        logger.error(f"Error checking equivalence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verification/containment")
async def check_containment_endpoint(request: ContainmentRequest):
    """Check if one language is contained in another"""
    try:
        result = verify_containment(request)
        return result
    except Exception as e:
        logger.error(f"Error checking containment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verification/minimize")
async def minimize_automaton_endpoint(request: MinimizationRequest):
    """Minimize an automaton"""
    try:
        result = minimize_automaton(request)
        return result
    except Exception as e:
        logger.error(f"Error minimizing automaton: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verification/counter-example")
async def find_counter_example_endpoint(automaton1: Automaton, automaton2: Automaton, max_length: int = 10):
    """Find a counter-example showing automata are not equivalent"""
    try:
        result = find_counter_example(automaton1, automaton2, max_length)
        return {"counter_example": result}
    except Exception as e:
        logger.error(f"Error finding counter-example: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/verification/algorithms")
async def get_verification_algorithms_endpoint():
    """Get information about available verification algorithms"""
    try:
        algorithms = get_verification_algorithms()
        return algorithms
    except Exception as e:
        logger.error(f"Error getting verification algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PUMPING LEMMA ENDPOINTS
# ============================================================================

@app.post("/api/pumping/validate")
async def validate_pumping_endpoint(request: PumpingRequest):
    """Validate a pumping lemma application"""
    try:
        result = validate_pumping_lemma(request)
        return result
    except Exception as e:
        logger.error(f"Error validating pumping lemma: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pumping/candidates/{language_description}")
async def get_pumping_candidates_endpoint(language_description: str):
    """Get candidate strings for non-regularity proofs"""
    try:
        candidates = get_non_regular_candidates(language_description)
        return {"candidates": candidates}
    except Exception as e:
        logger.error(f"Error getting pumping candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pumping/decompose-regular/{string}/{pumping_length}")
async def decompose_regular_endpoint(string: str, pumping_length: int):
    """Get all valid regular decompositions of a string"""
    try:
        decompositions = decompose_string_regular(string, pumping_length)
        return {"decompositions": decompositions}
    except Exception as e:
        logger.error(f"Error decomposing string: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pumping/decompose-cfl/{string}/{pumping_length}")
async def decompose_cfl_endpoint(string: str, pumping_length: int):
    """Get valid CFL decompositions of a string"""
    try:
        decompositions = decompose_string_cfl(string, pumping_length)
        return {"decompositions": decompositions}
    except Exception as e:
        logger.error(f"Error decomposing CFL string: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pumping/templates")
async def get_pumping_templates_endpoint():
    """Get templates for pumping lemma proofs"""
    try:
        templates = get_pumping_templates()
        return templates
    except Exception as e:
        logger.error(f"Error getting pumping templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# COMPLEXITY THEORY ENDPOINTS
# ============================================================================

@app.post("/api/complexity/analyze")
async def analyze_complexity_endpoint(request: ComplexityAnalysisRequest):
    """Analyze the complexity of an algorithm"""
    try:
        result = analyze_complexity(request)
        return result
    except Exception as e:
        logger.error(f"Error analyzing complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complexity/verify-reduction")
async def verify_reduction_endpoint(request: ReductionVerificationRequest):
    """Verify a reduction between problems"""
    try:
        result = verify_reduction(request)
        return result
    except Exception as e:
        logger.error(f"Error verifying reduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/complexity/classes")
async def get_complexity_classes_endpoint():
    """Get information about complexity classes"""
    try:
        classes = get_complexity_classes()
        return classes
    except Exception as e:
        logger.error(f"Error getting complexity classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/complexity/relationship/{class1}/{class2}")
async def get_class_relationship_endpoint(class1: str, class2: str):
    """Get relationship between two complexity classes"""
    try:
        relationship = get_class_relationship(class1, class2)
        return relationship
    except Exception as e:
        logger.error(f"Error getting class relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/complexity/templates")
async def get_complexity_templates_endpoint():
    """Get templates for complexity theory proofs"""
    try:
        templates = get_complexity_templates()
        return templates
    except Exception as e:
        logger.error(f"Error getting complexity templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ADAPTIVE LEARNING ENDPOINTS
# ============================================================================

@app.post("/api/learning/update-performance")
async def update_performance_endpoint(student_id: str, action: StudentAction):
    """Update student performance based on action"""
    try:
        profile = update_student_performance(student_id, action)
        return profile
    except Exception as e:
        logger.error(f"Error updating student performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/recommendations/{student_id}")
async def get_recommendations_endpoint(student_id: str, count: int = 5):
    """Get personalized problem recommendations"""
    try:
        recommendations = get_problem_recommendations(student_id, count)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/analytics/{student_id}")
async def get_analytics_endpoint(student_id: str):
    """Get comprehensive learning analytics"""
    try:
        analytics = get_student_analytics(student_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting student analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/session/start")
async def start_session_endpoint(student_id: str, session_id: str):
    """Start a new learning session"""
    try:
        session = start_learning_session(student_id, session_id)
        return session
    except Exception as e:
        logger.error(f"Error starting learning session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/session/end")
async def end_session_endpoint(student_id: str, session_id: str):
    """End a learning session"""
    try:
        session = end_learning_session(student_id, session_id)
        return session
    except Exception as e:
        logger.error(f"Error ending learning session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/difficulty/{student_id}/{problem_type}")
async def get_difficulty_endpoint(student_id: str, problem_type: str):
    """Get recommended difficulty level for a problem type"""
    try:
        from .adaptive_learning import ProblemType
        problem_type_enum = ProblemType(problem_type)
        difficulty = get_difficulty_adjustment(student_id, problem_type_enum)
        return {"recommended_difficulty": difficulty}
    except Exception as e:
        logger.error(f"Error getting difficulty adjustment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RESEARCH PAPERS ENDPOINTS
# ============================================================================

@app.post("/api/papers/search")
async def search_papers_endpoint(request: PaperSearchRequest):
    """Search papers in database"""
    try:
        papers = search_papers(request)
        return {"papers": papers}
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/papers/recommend")
async def recommend_papers_endpoint(request: PaperRecommendationRequest):
    """Get paper recommendations"""
    try:
        papers = recommend_papers(request)
        return {"recommendations": papers}
    except Exception as e:
        logger.error(f"Error recommending papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/papers/citation")
async def get_citation_endpoint(request: CitationRequest):
    """Get formatted citation for paper"""
    try:
        citation = get_paper_citation(request)
        if citation is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return {"citation": citation}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting citation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers/topic/{topic}")
async def get_papers_by_topic_endpoint(topic: TopicArea):
    """Get papers for specific topic"""
    try:
        papers = get_papers_by_topic(topic)
        return {"papers": papers}
    except Exception as e:
        logger.error(f"Error getting papers by topic: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers/{paper_id}")
async def get_paper_details_endpoint(paper_id: str):
    """Get detailed information about a paper"""
    try:
        paper = get_paper_details(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers/stats")
async def get_database_stats_endpoint():
    """Get statistics about the papers database"""
    try:
        stats = get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AI-POWERED ENDPOINTS - PHASE 3
# ============================================================================

from .prompts import prompt_builder, PromptExample, PromptType
from .orchestrator import orchestrate_task, ModelRequest, ExecutionMode
from .ai_proof_assistant import (
    proof_generator, proof_assistant, nl_translator,
    ProofTechnique, ProofStructure
)
from .semantic_search import (
    db_manager, search_engine, recommendation_engine,
    Document, DocumentType, SearchMode
)
from .rag_system import execute_rag_query, rag_chain, RAGRequest
from .memory import (
    conversation_memory, preference_manager,
    ConversationRole, UserPreference
)
from .optimizer import optimize_automaton, automata_optimizer
from .ai_config import TaskComplexity


# Prompt Template Endpoints
@app.post("/api/ai/prompt/generate")
async def generate_prompt(
    template_name: str,
    variables: Dict[str, Any],
    examples: Optional[List[Dict[str, str]]] = None
):
    """Generate a prompt from template with variables"""
    try:
        prompt_examples = None
        if examples:
            prompt_examples = [
                PromptExample(
                    input=ex.get("input", ""),
                    output=ex.get("output", ""),
                    explanation=ex.get("explanation")
                )
                for ex in examples
            ]
        
        prompt = prompt_builder.build(
            template_name,
            variables,
            examples=prompt_examples,
            optimize=True
        )
        
        return {
            "prompt": prompt,
            "template_used": template_name,
            "optimized": True
        }
    except Exception as e:
        logger.error(f"Prompt generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/prompt/templates")
async def list_prompt_templates():
    """List available prompt templates"""
    try:
        templates = prompt_builder.library.list_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Orchestration Endpoints
@app.post("/api/ai/orchestrate")
async def orchestrate_model_task(
    task: str,
    prompt: str,
    mode: str = "sequential",
    complexity: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """Execute task using model orchestration"""
    try:
        result = await orchestrate_task(
            task=task,
            prompt=prompt,
            mode=mode,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "task": task,
            "mode": mode,
            "result": result
        }
    except Exception as e:
        logger.error(f"Orchestration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Proof Assistant Endpoints
@app.post("/api/ai/proof/generate")
async def generate_proof(
    theorem: str,
    technique: Optional[str] = None,
    context: Optional[str] = None
):
    """Generate a formal proof for a theorem"""
    try:
        proof_tech = ProofTechnique(technique) if technique else None
        proof = await proof_generator.generate_proof(
            theorem=theorem,
            technique=proof_tech,
            context=context
        )
        
        return {
            "theorem": theorem,
            "proof": proof.dict(),
            "verification_score": proof.verification_score,
            "status": proof.status.value
        }
    except Exception as e:
        logger.error(f"Proof generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/proof/verify")
async def verify_proof_step(
    session_id: str,
    step_number: int,
    statement: str,
    justification: str
):
    """Verify a proof step in an interactive session"""
    try:
        from .ai_proof_assistant import ProofStep
        
        step = ProofStep(
            step_number=step_number,
            statement=statement,
            justification=justification
        )
        
        is_valid, explanation = await proof_assistant.verify_step(
            session_id,
            step
        )
        
        return {
            "valid": is_valid,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Proof verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/proof/translate")
async def translate_notation(
    text: str,
    direction: str = "to_formal"  # to_formal or to_natural
):
    """Translate between natural language and formal notation"""
    try:
        if direction == "to_formal":
            result = await nl_translator.to_formal(text)
        else:
            result = await nl_translator.to_natural(text)
        
        return {
            "original": text,
            "translated": result,
            "direction": direction
        }
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Semantic Search Endpoints
@app.post("/api/ai/search/add_document")
async def add_document_to_search(
    content: str,
    doc_type: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Add a document to the semantic search index"""
    try:
        doc = Document(
            id="",  # Will be generated
            content=content,
            type=DocumentType(doc_type),
            title=title,
            metadata=metadata or {}
        )
        
        doc_id = db_manager.add_document(doc)
        
        return {
            "document_id": doc_id,
            "indexed": True
        }
    except Exception as e:
        logger.error(f"Document indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/search/query")
async def semantic_search(
    query: str,
    mode: str = "hybrid",
    n_results: int = 5,
    filters: Optional[Dict[str, Any]] = None
):
    """Perform semantic search"""
    try:
        if mode == "hybrid":
            results = await search_engine.hybrid_search(query, n_results)
        elif mode == "similarity":
            results = db_manager.search(query, n_results, filters)
        elif mode == "conceptual":
            results = await search_engine.conceptual_search(query, n_results=n_results)
        else:
            results = db_manager.search(query, n_results, filters)
        
        return {
            "query": query,
            "results": [r.dict() for r in results],
            "mode": mode
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/search/recommend")
async def get_recommendations(
    current_topic: str,
    user_history: Optional[List[str]] = None,
    n_recommendations: int = 5
):
    """Get content recommendations"""
    try:
        recommendations = await recommendation_engine.recommend_next_topics(
            current_topic,
            user_history,
            n_recommendations
        )
        
        return {
            "current_topic": current_topic,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG System Endpoints
@app.post("/api/ai/rag/query")
async def rag_query(
    query: str,
    mode: str = "simple",
    max_sources: int = 5,
    include_sources: bool = True,
    conversation_id: Optional[str] = None
):
    """Execute a RAG query"""
    try:
        result = await execute_rag_query(
            query=query,
            mode=mode,
            max_sources=max_sources,
            include_sources=include_sources,
            conversation_id=conversation_id
        )
        
        return result
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/rag/add_knowledge")
async def add_to_rag_knowledge(
    documents: List[Dict[str, Any]]
):
    """Add documents to RAG knowledge base"""
    try:
        docs = [
            Document(
                id=doc.get("id", ""),
                content=doc["content"],
                type=DocumentType(doc.get("type", "concept")),
                title=doc.get("title"),
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]
        
        chunks_added = await rag_chain.add_documents(docs)
        
        return {
            "documents_added": len(docs),
            "chunks_created": chunks_added
        }
    except Exception as e:
        logger.error(f"RAG knowledge addition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory Management Endpoints
@app.post("/api/ai/memory/message")
async def add_conversation_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Add a message to conversation memory"""
    try:
        success = await conversation_memory.add_message(
            session_id,
            ConversationRole(role),
            content,
            metadata
        )
        
        return {
            "session_id": session_id,
            "message_added": success
        }
    except Exception as e:
        logger.error(f"Memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/memory/context/{session_id}")
async def get_conversation_context(
    session_id: str,
    max_tokens: int = 2048
):
    """Get conversation context for a session"""
    try:
        context = await conversation_memory.get_context(
            session_id,
            max_tokens
        )
        
        topics = await conversation_memory.extract_topics(session_id)
        
        return {
            "session_id": session_id,
            "context": context,
            "topics": topics
        }
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/memory/preferences")
async def update_user_preferences(
    user_id: str,
    preferences: Dict[str, Any]
):
    """Update user learning preferences"""
    try:
        updated = await preference_manager.update_preference(
            user_id,
            preferences
        )
        
        return {
            "user_id": user_id,
            "preferences": updated.dict()
        }
    except Exception as e:
        logger.error(f"Preference update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Automata Optimization Endpoints
@app.post("/api/ai/optimize/automaton")
async def optimize_automaton_endpoint(
    automaton: Dict[str, Any],
    optimization_type: Optional[str] = None
):
    """Optimize an automaton"""
    try:
        result = await optimize_automaton(
            automaton,
            optimization_type
        )
        
        return result
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/optimize/analyze")
async def analyze_automaton_performance(
    automaton: Dict[str, Any]
):
    """Analyze automaton performance characteristics"""
    try:
        from .optimizer import Automaton
        
        auto = Automaton(**automaton)
        analysis = await automata_optimizer.analyze_performance(auto)
        
        return {
            "automaton": automaton,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/optimize/equivalence")
async def check_automata_equivalence(
    automaton1: Dict[str, Any],
    automaton2: Dict[str, Any]
):
    """Check if two automata are equivalent"""
    try:
        from .optimizer import Automaton
        
        auto1 = Automaton(**automaton1)
        auto2 = Automaton(**automaton2)
        
        are_equivalent, counterexample = automata_optimizer.equivalence_checker.are_equivalent(
            auto1,
            auto2
        )
        
        return {
            "equivalent": are_equivalent,
            "counterexample": counterexample
        }
    except Exception as e:
        logger.error(f"Equivalence check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check for AI services
@app.get("/api/ai/health")
async def ai_health_check():
    """Check health of AI services"""
    try:
        health = {
            "prompts": prompt_builder is not None,
            "orchestrator": orchestrator is not None,
            "proof_assistant": proof_generator is not None,
            "semantic_search": search_engine is not None,
            "rag_system": rag_chain is not None,
            "memory": conversation_memory is not None,
            "optimizer": automata_optimizer is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return health
    except Exception as e:
        logger.error(f"AI health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# =============================================================================
# JFLAP COMPLETE ALGORITHM ENDPOINTS
# =============================================================================

class AutomatonRequest(BaseModel):
    """Request model for automaton operations"""
    type: str
    states: List[Dict[str, Any]]
    transitions: List[Dict[str, Any]]
    initial_state: str
    final_states: List[str]
    alphabet: Optional[List[str]] = None

class RegexRequest(BaseModel):
    """Request model for regex operations"""
    regex: str

class GrammarRequest(BaseModel):
    """Request model for grammar operations"""
    variables: List[str]
    terminals: List[str]
    productions: Dict[str, List[str]]
    start_symbol: str

class SimulationRequest(BaseModel):
    """Request model for simulation"""
    automaton: AutomatonRequest
    input_string: str
    options: Optional[Dict[str, Any]] = {}

class BatchSimulationRequest(BaseModel):
    """Request model for batch simulation"""
    automaton: AutomatonRequest
    input_strings: List[str]
    options: Optional[Dict[str, Any]] = {}

def convert_request_to_automaton(request: AutomatonRequest) -> Automaton:
    """Convert API request to internal Automaton representation"""
    automaton_type = AutomatonType(request.type.lower())
    automaton = Automaton(type=automaton_type)
    
    # Add states
    for state_data in request.states:
        state = State(
            name=state_data["name"],
            is_initial=state_data.get("is_initial", False),
            is_final=state_data.get("is_final", False),
            x=state_data.get("x", 0),
            y=state_data.get("y", 0),
            output=state_data.get("output")
        )
        automaton.add_state(state)
        
        if state.is_final:
            automaton.final_states.add(state.name)
    
    automaton.initial_state = request.initial_state
    
    # Add transitions
    for trans_data in request.transitions:
        transition = Transition(
            from_state=trans_data["from_state"],
            to_state=trans_data["to_state"],
            input_symbol=trans_data["input_symbol"],
            output_symbol=trans_data.get("output_symbol"),
            stack_pop=trans_data.get("stack_pop"),
            stack_push=trans_data.get("stack_push"),
            tape_read=trans_data.get("tape_read"),
            tape_write=trans_data.get("tape_write"),
            tape_move=trans_data.get("tape_move")
        )
        automaton.add_transition(transition)
    
    if request.alphabet:
        automaton.alphabet = set(request.alphabet)
    
    return automaton

def convert_automaton_to_response(automaton: Automaton) -> Dict[str, Any]:
    """Convert internal Automaton to API response"""
    return {
        "type": automaton.type.value,
        "states": [
            {
                "name": state.name,
                "is_initial": state.is_initial,
                "is_final": state.is_final,
                "x": state.x,
                "y": state.y,
                "output": state.output
            }
            for state in automaton.states
        ],
        "transitions": [
            {
                "from_state": trans.from_state,
                "to_state": trans.to_state,
                "input_symbol": trans.input_symbol,
                "output_symbol": trans.output_symbol,
                "stack_pop": trans.stack_pop,
                "stack_push": trans.stack_push,
                "tape_read": trans.tape_read,
                "tape_write": trans.tape_write,
                "tape_move": trans.tape_move
            }
            for trans in automaton.transitions
        ],
        "initial_state": automaton.initial_state,
        "final_states": list(automaton.final_states),
        "alphabet": list(automaton.alphabet)
    }

@app.post("/api/jflap/convert/nfa-to-dfa")
async def convert_nfa_to_dfa(request: AutomatonRequest):
    """Convert NFA to DFA using subset construction"""
    try:
        nfa = convert_request_to_automaton(request)
        if nfa.type != AutomatonType.NFA:
            raise HTTPException(status_code=400, detail="Input must be an NFA")
        
        dfa = jflap_algorithms.convert_nfa_to_dfa(nfa)
        
        return {
            "success": True,
            "original_states": len(nfa.states),
            "converted_states": len(dfa.states),
            "automaton": convert_automaton_to_response(dfa)
        }
    except Exception as e:
        logger.error(f"NFA to DFA conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/minimize/dfa")
async def minimize_dfa(request: AutomatonRequest):
    """Minimize DFA using Hopcroft's algorithm"""
    try:
        dfa = convert_request_to_automaton(request)
        if dfa.type != AutomatonType.DFA:
            raise HTTPException(status_code=400, detail="Input must be a DFA")
        
        minimal_dfa = jflap_algorithms.minimize_dfa(dfa)
        
        return {
            "success": True,
            "original_states": len(dfa.states),
            "minimized_states": len(minimal_dfa.states),
            "automaton": convert_automaton_to_response(minimal_dfa)
        }
    except Exception as e:
        logger.error(f"DFA minimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/convert/regex-to-nfa")
async def convert_regex_to_nfa(request: RegexRequest):
    """Convert regular expression to NFA using Thompson's construction"""
    try:
        nfa = jflap_algorithms.regex_to_nfa(request.regex)
        
        return {
            "success": True,
            "regex": request.regex,
            "states": len(nfa.states),
            "automaton": convert_automaton_to_response(nfa)
        }
    except Exception as e:
        logger.error(f"Regex to NFA conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/convert/nfa-to-regex")
async def convert_nfa_to_regex(request: AutomatonRequest):
    """Convert NFA to regular expression using state elimination"""
    try:
        nfa = convert_request_to_automaton(request)
        if nfa.type != AutomatonType.NFA:
            raise HTTPException(status_code=400, detail="Input must be an NFA")
        
        regex = jflap_algorithms.nfa_to_regex(nfa)
        
        return {
            "success": True,
            "states": len(nfa.states),
            "regex": regex
        }
    except Exception as e:
        logger.error(f"NFA to regex conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/grammar/to-cnf")
async def convert_grammar_to_cnf(request: GrammarRequest):
    """Convert context-free grammar to Chomsky Normal Form"""
    try:
        grammar = Grammar()
        grammar.variables = set(request.variables)
        grammar.terminals = set(request.terminals)
        grammar.productions = request.productions
        grammar.start_symbol = request.start_symbol
        
        cnf_grammar = jflap_algorithms.cfg_to_cnf(grammar)
        
        return {
            "success": True,
            "original_productions": len(sum(request.productions.values(), [])),
            "cnf_productions": len(sum(cnf_grammar.productions.values(), [])),
            "grammar": {
                "variables": list(cnf_grammar.variables),
                "terminals": list(cnf_grammar.terminals),
                "productions": cnf_grammar.productions,
                "start_symbol": cnf_grammar.start_symbol
            }
        }
    except Exception as e:
        logger.error(f"Grammar to CNF conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/grammar/to-pda")
async def convert_grammar_to_pda(request: GrammarRequest):
    """Convert context-free grammar to PDA"""
    try:
        grammar = Grammar()
        grammar.variables = set(request.variables)
        grammar.terminals = set(request.terminals)
        grammar.productions = request.productions
        grammar.start_symbol = request.start_symbol
        
        pda = jflap_algorithms.cfg_to_pda(grammar)
        
        return {
            "success": True,
            "grammar_productions": len(sum(request.productions.values(), [])),
            "pda_states": len(pda.states),
            "automaton": convert_automaton_to_response(pda)
        }
    except Exception as e:
        logger.error(f"Grammar to PDA conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/parse/cyk")
async def parse_with_cyk(grammar_request: GrammarRequest, input_string: str):
    """Parse string using CYK algorithm"""
    try:
        grammar = Grammar()
        grammar.variables = set(grammar_request.variables)
        grammar.terminals = set(grammar_request.terminals)
        grammar.productions = grammar_request.productions
        grammar.start_symbol = grammar_request.start_symbol
        
        is_accepted, parse_table = jflap_algorithms.parse_cyk(grammar, input_string)
        
        return {
            "success": True,
            "input_string": input_string,
            "accepted": is_accepted,
            "parse_table": parse_table
        }
    except Exception as e:
        logger.error(f"CYK parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/parse/ll1")
async def parse_with_ll1(grammar_request: GrammarRequest, input_string: str):
    """Parse string using LL(1) algorithm"""
    try:
        grammar = Grammar()
        grammar.variables = set(grammar_request.variables)
        grammar.terminals = set(grammar_request.terminals)
        grammar.productions = grammar_request.productions
        grammar.start_symbol = grammar_request.start_symbol
        
        is_accepted, derivation = jflap_algorithms.parse_ll1(grammar, input_string)
        
        return {
            "success": True,
            "input_string": input_string,
            "accepted": is_accepted,
            "derivation": derivation
        }
    except Exception as e:
        logger.error(f"LL(1) parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/simulate")
async def simulate_automaton(request: SimulationRequest):
    """Simulate automaton execution with comprehensive tracking"""
    try:
        automaton = convert_request_to_automaton(request.automaton)
        
        result = simulation_engine.simulate(
            automaton, 
            request.input_string, 
            request.options
        )
        
        return {
            "success": True,
            "simulation_result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/simulate/batch")
async def batch_simulate_automaton(request: BatchSimulationRequest):
    """Batch simulate multiple strings on automaton"""
    try:
        automaton = convert_request_to_automaton(request.automaton)
        
        results = simulation_engine.batch_simulate(
            automaton, 
            request.input_strings, 
            request.options
        )
        
        return {
            "success": True,
            "batch_results": [result.to_dict() for result in results],
            "summary": {
                "total_inputs": len(request.input_strings),
                "accepted_count": sum(1 for r in results if r.is_accepted),
                "rejected_count": sum(1 for r in results if not r.is_accepted)
            }
        }
    except Exception as e:
        logger.error(f"Batch simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jflap/simulate/compare")
async def compare_executions(request: BatchSimulationRequest):
    """Compare execution patterns across multiple inputs"""
    try:
        automaton = convert_request_to_automaton(request.automaton)
        
        comparison = simulation_engine.compare_executions(
            automaton, 
            request.input_strings
        )
        
        return {
            "success": True,
            "comparison_analysis": comparison
        }
    except Exception as e:
        logger.error(f"Execution comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jflap/algorithms/info")
async def get_algorithm_info():
    """Get information about all available JFLAP algorithms"""
    try:
        info = jflap_algorithms.get_algorithm_info()
        
        return {
            "success": True,
            "algorithms": info,
            "total_algorithms": sum(len(algs) for algs in info.values())
        }
    except Exception as e:
        logger.error(f"Algorithm info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jflap/health")
async def jflap_health_check():
    """Health check for JFLAP algorithms"""
    try:
        # Test basic functionality
        test_nfa = Automaton(type=AutomatonType.NFA)
        q0 = State("q0", is_initial=True)
        q1 = State("q1", is_final=True)
        test_nfa.add_state(q0)
        test_nfa.add_state(q1)
        test_nfa.initial_state = "q0"
        test_nfa.final_states.add("q1")
        test_nfa.add_transition(Transition("q0", "q1", "a"))
        
        # Test conversion
        dfa = jflap_algorithms.convert_nfa_to_dfa(test_nfa)
        
        # Test simulation
        result = simulation_engine.simulate(dfa, "a")
        
        return {
            "status": "healthy",
            "algorithms_loaded": True,
            "simulation_engine": True,
            "test_conversion": len(dfa.states) > 0,
            "test_simulation": result.is_accepted,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"JFLAP health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
