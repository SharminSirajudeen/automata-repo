"""
Input validation and sanitization for the automata-repo application.
"""
import re
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field, validator, ValidationError
from fastapi import HTTPException


class ValidatedState(BaseModel):
    """Validated state model with constraints."""
    id: str = Field(..., min_length=1, max_length=50, description="State identifier")
    x: float = Field(..., ge=-10000, le=10000, description="X coordinate")
    y: float = Field(..., ge=-10000, le=10000, description="Y coordinate")
    is_start: bool = False
    is_accept: bool = False
    label: Optional[str] = Field(None, max_length=100)
    
    @validator('id')
    def validate_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('State ID must contain only alphanumeric characters, hyphens, and underscores')
        return v


class ValidatedTransition(BaseModel):
    """Validated transition model with constraints."""
    from_state: str = Field(..., min_length=1, max_length=50)
    to_state: str = Field(..., min_length=1, max_length=50)
    symbol: str = Field(..., min_length=0, max_length=10)
    x: Optional[float] = Field(None, ge=-10000, le=10000)
    y: Optional[float] = Field(None, ge=-10000, le=10000)
    
    @validator('from_state', 'to_state')
    def validate_state_refs(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('State reference must contain only alphanumeric characters, hyphens, and underscores')
        return v
    
    @validator('symbol')
    def validate_symbol(cls, v):
        # Allow epsilon transitions (empty string) and unicode symbols
        if len(v) > 10:
            raise ValueError('Symbol must be at most 10 characters')
        return v


class ValidatedAutomaton(BaseModel):
    """Base validated automaton model."""
    states: List[ValidatedState] = Field(..., min_items=1, max_items=1000)
    transitions: List[ValidatedTransition] = Field(..., max_items=10000)
    alphabet: List[str] = Field(..., max_items=100)
    
    @validator('alphabet')
    def validate_alphabet(cls, v):
        for symbol in v:
            if len(symbol) > 10:
                raise ValueError(f'Alphabet symbol "{symbol}" is too long (max 10 characters)')
        return v
    
    @validator('states')
    def validate_states(cls, v):
        state_ids = [state.id for state in v]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError('Duplicate state IDs found')
        
        start_states = [s for s in v if s.is_start]
        if len(start_states) != 1:
            raise ValueError('Exactly one start state is required')
        
        return v
    
    @validator('transitions')
    def validate_transitions(cls, v, values):
        if 'states' not in values:
            return v
            
        state_ids = {state.id for state in values['states']}
        for transition in v:
            if transition.from_state not in state_ids:
                raise ValueError(f'Invalid from_state: {transition.from_state}')
            if transition.to_state not in state_ids:
                raise ValueError(f'Invalid to_state: {transition.to_state}')
        
        return v


class ValidatedProblemInput(BaseModel):
    """Validated problem input."""
    text: str = Field(..., min_length=1, max_length=10000)
    problem_type: Optional[str] = Field(None, pattern=r'^(dfa|nfa|pda|cfg|tm|regex|pumping_lemma)$')
    
    @validator('text')
    def sanitize_text(cls, v):
        # Remove any potential script tags or HTML
        v = re.sub(r'<[^>]*>', '', v)
        # Limit consecutive whitespace
        v = re.sub(r'\s+', ' ', v)
        return v.strip()


class SolutionCreate(BaseModel):
    """Pydantic model for creating a solution."""
    user_id: str
    automaton: Dict[str, Any]


class ValidatedTestCase(BaseModel):
    """Validated test case."""
    input: str = Field(..., max_length=1000)
    expected: bool
    description: Optional[str] = Field(None, max_length=500)
    
    @validator('input')
    def validate_input(cls, v):
        # Ensure input doesn't contain control characters
        if any(ord(char) < 32 and char not in '\n\r\t' for char in v):
            raise ValueError('Input contains invalid control characters')
        return v


class ValidationResult(BaseModel):
    """Structured result of a validation check."""
    is_correct: bool
    score: float = Field(..., ge=0.0, le=1.0)
    feedback: str
    test_results: List[Dict[str, Any]]
    mistakes: List[str]


def validate_automaton_type(automaton: Dict[str, Any], expected_type: str) -> Dict[str, Any]:
    """Validate automaton based on its type."""
    try:
        if expected_type in ['dfa', 'nfa']:
            validated = ValidatedAutomaton(**automaton)
            return validated.dict()
        elif expected_type == 'pda':
            # Add PDA-specific validation
            if 'stack_alphabet' not in automaton:
                raise ValueError('PDA must have stack_alphabet')
            validated = ValidatedAutomaton(**automaton)
            return validated.dict()
        elif expected_type == 'cfg':
            # CFG has different structure
            if 'productions' not in automaton or 'start_symbol' not in automaton:
                raise ValueError('CFG must have productions and start_symbol')
            return automaton  # TODO: Add specific CFG validation
        elif expected_type == 'tm':
            # TM has tape alphabet
            if 'tape_alphabet' not in automaton:
                raise ValueError('TM must have tape_alphabet')
            validated = ValidatedAutomaton(**automaton)
            return validated.dict()
        else:
            raise ValueError(f'Unknown automaton type: {expected_type}')
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """Sanitize user input string."""
    if not text:
        return ""
    
    # Remove HTML/script tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit length
    text = text[:max_length]
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_test_strings(test_strings: List[str], max_strings: int = 1000, max_length: int = 1000) -> List[str]:
    """Validate a list of test strings."""
    if len(test_strings) > max_strings:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many test strings (max {max_strings})"
        )
    
    validated = []
    for s in test_strings:
        if len(s) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"Test string too long (max {max_length} characters)"
            )
        validated.append(sanitize_string(s, max_length))
    
    return validated