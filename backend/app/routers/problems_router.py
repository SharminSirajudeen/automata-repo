"""
Problems router for the Automata Learning Platform.
Handles problem management, validation, and hint generation.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from ..database import Problem, Solution, save_solution, User
from ..agents import AutomataExplainer
from ..auth import get_current_active_user
from ..validators import ValidationResult, SolutionCreate
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/problems", tags=["problems"])

# Global problems database (in production, this would be in a real database)
problems_db = {}
solutions_db = {}


class AIFeedbackRequest(BaseModel):
    user_automaton: Any  # This should be properly typed based on your automaton structure
    difficulty_level: str = "beginner"


def validate_dfa(automaton: Any, problem: Problem) -> ValidationResult:
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


def validate_nfa(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate an NFA solution against a problem - placeholder implementation"""
    # This would contain the full NFA validation logic
    return ValidationResult(
        is_correct=True,
        score=1.0,
        feedback="NFA validation not fully implemented",
        test_results=[],
        mistakes=[]
    )


def validate_pda(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a PDA solution against a problem - placeholder implementation"""
    return ValidationResult(
        is_correct=True,
        score=1.0,
        feedback="PDA validation not fully implemented",
        test_results=[],
        mistakes=[]
    )


def validate_cfg(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a CFG solution against a problem - placeholder implementation"""
    return ValidationResult(
        is_correct=True,
        score=1.0,
        feedback="CFG validation not fully implemented",
        test_results=[],
        mistakes=[]
    )


def validate_tm(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a TM solution against a problem - placeholder implementation"""
    return ValidationResult(
        is_correct=True,
        score=1.0,
        feedback="TM validation not fully implemented",
        test_results=[],
        mistakes=[]
    )


def validate_regex(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a regex solution against a problem - placeholder implementation"""
    return ValidationResult(
        is_correct=True,
        score=1.0,
        feedback="Regex validation not fully implemented",
        test_results=[],
        mistakes=[]
    )


def validate_pumping_lemma(automaton: Any, problem: Problem) -> ValidationResult:
    """Validate a pumping lemma solution against a problem - placeholder implementation"""
    return ValidationResult(
        is_correct=True,
        score=1.0,
        feedback="Pumping lemma validation not fully implemented",
        test_results=[],
        mistakes=[]
    )


@router.get("/")
async def get_problems():
    """Get all available problems"""
    return {"problems": list(problems_db.values())}


@router.get("/{problem_id}")
async def get_problem(problem_id: str):
    """Get a specific problem by ID"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    return problems_db[problem_id]


@router.post("/{problem_id}/validate")
async def validate_solution(problem_id: str, solution: SolutionCreate):
    """Validate a solution against a problem"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    
    try:
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
            raise HTTPException(
                status_code=400, 
                detail=f"Validation for {problem.type} is not yet implemented"
            )
        
        solution_key = f"{problem_id}_{solution.user_id}"
        solutions_db[solution_key] = {
            "solution": solution,
            "result": result,
            "timestamp": "2025-08-05T16:27:32Z"  # In production, use datetime.now()
        }
        
        logger.info(f"Solution validated for problem {problem_id}, score: {result.score}")
        return result
        
    except Exception as e:
        logger.error(f"Validation failed for problem {problem_id}: {e}")
        raise HTTPException(status_code=500, detail="Validation failed")


@router.get("/{problem_id}/hint")
async def get_hint(problem_id: str, hint_index: int = 0):
    """Get a hint for a specific problem"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    if hint_index >= len(problem.hints):
        raise HTTPException(status_code=404, detail="Hint not found")
    
    return {"hint": problem.hints[hint_index], "total_hints": len(problem.hints)}


@router.post("/{problem_id}/ai-hint")
async def get_ai_hint(problem_id: str, request: AIFeedbackRequest):
    """Get AI-powered personalized hint based on current progress"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    problem = problems_db[problem_id]
    explainer = AutomataExplainer()
    
    try:
        current_progress = {
            "states": len(request.user_automaton.states),
            "transitions": len(request.user_automaton.transitions),
            "start_states": len([s for s in request.user_automaton.states if s.is_start]),
            "accept_states": len([s for s in request.user_automaton.states if s.is_accept])
        }
        
        ai_hint = await explainer.provide_step_guidance(problem.description, current_progress)
        logger.info(f"AI hint generated for problem {problem_id}")
        return {"ai_hint": ai_hint}
        
    except Exception as e:
        logger.error(f"AI hint generation failed for problem {problem_id}: {e}")
        raise HTTPException(status_code=500, detail="AI hint generation failed")


@router.get("/{problem_id}/generate-solution")
async def generate_solution(problem_id: str):
    """Generate a solution for the specified problem"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    # This would integrate with AI solution generation
    # Placeholder implementation
    return {"message": "Solution generation not implemented yet"}


@router.post("/{problem_id}/explain-solution")
async def explain_solution(problem_id: str, solution: SolutionCreate):
    """Explain a solution step by step"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    # This would integrate with AI explanation generation
    # Placeholder implementation
    return {"explanation": "Solution explanation not implemented yet"}


@router.post("/{problem_id}/guided-step")
async def guided_step(problem_id: str, current_progress: Dict[str, Any]):
    """Get the next guided step for solving a problem"""
    if problem_id not in problems_db:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    # This would integrate with AI guided learning
    # Placeholder implementation
    return {"next_step": "Guided step not implemented yet"}