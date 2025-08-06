"""
AI-Enhanced JFLAP Features API Router
=====================================

FastAPI router for AI-powered JFLAP feature endpoints.
Provides REST API access to all AI enhancements.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import json
import asyncio
from datetime import datetime

from ..ai_jflap_integration import (
    ai_jflap,
    MultiTapeTMGenerator,
    GrammarAnalyzer,
    IntelligentErrorRecovery,
    AutomatedTestGenerator,
    NaturalLanguageConverter,
    StepByStepTutor
)
from ..jflap_complete import Grammar, Automaton, AutomatonType

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-jflap", tags=["AI-JFLAP"])


# ============================
# REQUEST/RESPONSE MODELS
# ============================

class MultiTapeTMRequest(BaseModel):
    """Request for multi-tape TM generation"""
    problem_description: str
    num_tapes: int = Field(default=2, ge=1, le=5)
    tape_purposes: Optional[List[str]] = None
    optimize: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "problem_description": "Recognize strings with equal number of a's and b's",
                "num_tapes": 2,
                "tape_purposes": ["Input reading", "Counter storage"],
                "optimize": True
            }
        }


class GrammarAnalysisRequest(BaseModel):
    """Request for grammar analysis"""
    variables: List[str]
    terminals: List[str]
    productions: Dict[str, List[str]]
    start_symbol: str = "S"
    
    class Config:
        schema_extra = {
            "example": {
                "variables": ["S", "A", "B"],
                "terminals": ["a", "b"],
                "productions": {
                    "S": ["aA", "bB"],
                    "A": ["aS", "ε"],
                    "B": ["bS", "ε"]
                },
                "start_symbol": "S"
            }
        }


class ErrorRecoveryRequest(BaseModel):
    """Request for parsing error recovery"""
    input_string: str
    grammar: GrammarAnalysisRequest
    error_type: str = "syntax_error"
    error_position: int = 0
    error_context: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "input_string": "((a+b)*c",
                "grammar": {
                    "variables": ["E", "T", "F"],
                    "terminals": ["(", ")", "+", "*", "id"],
                    "productions": {
                        "E": ["E+T", "T"],
                        "T": ["T*F", "F"],
                        "F": ["(E)", "id"]
                    },
                    "start_symbol": "E"
                },
                "error_type": "unmatched_parenthesis",
                "error_position": 7
            }
        }


class TestGenerationRequest(BaseModel):
    """Request for test case generation"""
    automaton_type: str
    description: str
    specification: Optional[Dict[str, Any]] = None
    coverage_target: float = Field(default=0.95, ge=0.0, le=1.0)
    focus_areas: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "automaton_type": "DFA",
                "description": "Accept strings ending with 'ab'",
                "coverage_target": 0.95
            }
        }


class NLConversionRequest(BaseModel):
    """Request for natural language conversion"""
    description: str
    target_formalism: str = Field(default="auto")
    detail_level: str = Field(default="medium")
    
    class Config:
        schema_extra = {
            "example": {
                "description": "Accept strings that contain at least two a's and end with b",
                "target_formalism": "DFA",
                "detail_level": "medium"
            }
        }


class TutoringRequest(BaseModel):
    """Request for tutoring content"""
    algorithm: str
    student_id: str
    student_level: str = Field(default="intermediate")
    specific_step: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm": "NFA to DFA conversion",
                "student_id": "student123",
                "student_level": "intermediate"
            }
        }


class HintRequest(BaseModel):
    """Request for problem hint"""
    problem: str
    current_attempt: str
    hint_level: int = Field(default=1, ge=1, le=3)
    
    class Config:
        schema_extra = {
            "example": {
                "problem": "Convert this NFA to DFA",
                "current_attempt": "I've identified the states but stuck on epsilon closure",
                "hint_level": 1
            }
        }


# ============================
# MULTI-TAPE TM ENDPOINTS
# ============================

@router.post("/multi-tape-tm/generate")
async def generate_multi_tape_tm(request: MultiTapeTMRequest):
    """
    Generate a multi-tape Turing Machine for a given problem.
    
    This endpoint uses AI to:
    - Analyze the problem and suggest optimal tape usage
    - Generate formal TM specification
    - Optimize the TM for minimal states
    - Provide implementation code
    - Generate test cases
    """
    try:
        result = await ai_jflap.process_request(
            "multi_tape_tm",
            {
                "problem": request.problem_description,
                "num_tapes": request.num_tapes,
                "tape_purposes": request.tape_purposes,
                "optimize": request.optimize
            }
        )
        
        if result["success"]:
            return result["result"]
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Multi-tape TM generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-tape-tm/optimize")
async def optimize_tm(tm_spec: Dict[str, Any]):
    """
    Optimize an existing Turing Machine specification.
    
    Reduces states and transitions while maintaining correctness.
    """
    try:
        generator = MultiTapeTMGenerator()
        optimized = await generator._optimize_tm(tm_spec)
        return optimized
        
    except Exception as e:
        logger.error(f"TM optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# GRAMMAR ANALYSIS ENDPOINTS
# ============================

@router.post("/grammar/analyze")
async def analyze_grammar(request: GrammarAnalysisRequest):
    """
    Analyze a context-free grammar to detect its type and properties.
    
    Returns:
    - Grammar type (regular, context-free, etc.)
    - Special forms (CNF, GNF, LL(1), etc.)
    - Properties (ambiguity, left recursion, etc.)
    - Conversion suggestions
    """
    try:
        # Create Grammar object
        grammar = Grammar()
        grammar.variables = set(request.variables)
        grammar.terminals = set(request.terminals)
        grammar.productions = request.productions
        grammar.start_symbol = request.start_symbol
        
        result = await ai_jflap.process_request(
            "grammar_analysis",
            {"grammar": grammar}
        )
        
        if result["success"]:
            return result["result"]
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Grammar analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/grammar/convert")
async def convert_grammar(
    request: GrammarAnalysisRequest,
    target_form: str = Query(..., description="Target form: CNF, GNF, LL1, etc.")
):
    """
    Convert grammar to specified form.
    
    Supported forms:
    - CNF (Chomsky Normal Form)
    - GNF (Greibach Normal Form)
    - Remove epsilon productions
    - Eliminate left recursion
    """
    try:
        grammar = Grammar()
        grammar.variables = set(request.variables)
        grammar.terminals = set(request.terminals)
        grammar.productions = request.productions
        grammar.start_symbol = request.start_symbol
        
        analyzer = GrammarAnalyzer()
        converted = await analyzer.convert_grammar(grammar, target_form)
        
        return {
            "original": request.dict(),
            "converted": {
                "variables": list(converted.variables),
                "terminals": list(converted.terminals),
                "productions": converted.productions,
                "start_symbol": converted.start_symbol
            },
            "target_form": target_form
        }
        
    except Exception as e:
        logger.error(f"Grammar conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# ERROR RECOVERY ENDPOINTS
# ============================

@router.post("/error-recovery/suggest")
async def suggest_error_recovery(request: ErrorRecoveryRequest):
    """
    Suggest recovery strategies for parsing errors.
    
    Provides:
    - Error diagnosis
    - Multiple recovery strategies with confidence scores
    - Valid alternative inputs
    - Educational explanation
    """
    try:
        grammar = Grammar()
        grammar.variables = set(request.grammar.variables)
        grammar.terminals = set(request.grammar.terminals)
        grammar.productions = request.grammar.productions
        grammar.start_symbol = request.grammar.start_symbol
        
        error_info = {
            "type": request.error_type,
            "position": request.error_position,
            "context": request.error_context
        }
        
        result = await ai_jflap.process_request(
            "error_recovery",
            {
                "input_string": request.input_string,
                "grammar": grammar,
                "error_info": error_info
            }
        )
        
        if result["success"]:
            return result["result"]
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error recovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/error-recovery/correct")
async def correct_input(
    input_string: str = Query(..., description="Malformed input"),
    expected_pattern: str = Query(..., description="Expected pattern/format")
):
    """
    Suggest corrections for malformed input.
    
    Returns list of corrected alternatives.
    """
    try:
        recovery = IntelligentErrorRecovery()
        corrections = await recovery.suggest_corrections(input_string, expected_pattern)
        
        return {
            "original": input_string,
            "expected_pattern": expected_pattern,
            "corrections": corrections
        }
        
    except Exception as e:
        logger.error(f"Input correction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# TEST GENERATION ENDPOINTS
# ============================

@router.post("/test/generate")
async def generate_tests(request: TestGenerationRequest):
    """
    Generate comprehensive test suite for automaton.
    
    Includes:
    - Positive and negative cases
    - Edge cases and boundary conditions
    - Coverage analysis
    - Test statistics
    """
    try:
        # Create mock automaton for test generation
        automaton = Automaton(type=AutomatonType[request.automaton_type.upper()])
        
        if request.specification:
            # Parse specification if provided
            # This would be enhanced with actual parsing
            pass
        
        result = await ai_jflap.process_request(
            "test_generation",
            {
                "automaton": automaton,
                "description": request.description,
                "coverage_target": request.coverage_target
            }
        )
        
        if result["success"]:
            return result["result"]
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/edge-cases")
async def generate_edge_cases(request: TestGenerationRequest):
    """
    Generate edge cases for specific focus areas.
    
    Focuses on unusual, boundary, and corner cases.
    """
    try:
        automaton = Automaton(type=AutomatonType[request.automaton_type.upper()])
        
        generator = AutomatedTestGenerator()
        edge_cases = await generator.generate_edge_cases(
            automaton,
            request.focus_areas
        )
        
        return {
            "automaton_type": request.automaton_type,
            "focus_areas": request.focus_areas,
            "edge_cases": edge_cases
        }
        
    except Exception as e:
        logger.error(f"Edge case generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# NL CONVERSION ENDPOINTS
# ============================

@router.post("/nl/to-formal")
async def natural_to_formal(request: NLConversionRequest):
    """
    Convert natural language description to formal specification.
    
    Automatically detects best formalism or uses specified target.
    """
    try:
        result = await ai_jflap.process_request(
            "nl_conversion",
            {
                "description": request.description,
                "target_formalism": request.target_formalism
            }
        )
        
        if result["success"]:
            return result["result"]
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"NL to formal conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nl/to-natural")
async def formal_to_natural(
    formal_spec: Dict[str, Any],
    detail_level: str = Query(default="medium", description="simple, medium, or detailed")
):
    """
    Convert formal specification to natural language description.
    
    Provides intuitive explanation at specified detail level.
    """
    try:
        converter = NaturalLanguageConverter()
        description = await converter.formal_to_natural(formal_spec, detail_level)
        
        return {
            "formal_spec": formal_spec,
            "natural_description": description,
            "detail_level": detail_level
        }
        
    except Exception as e:
        logger.error(f"Formal to NL conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# TUTORING ENDPOINTS
# ============================

@router.post("/tutor/create-tutorial")
async def create_tutorial(request: TutoringRequest):
    """
    Create personalized tutorial for algorithm.
    
    Includes:
    - Step-by-step explanation
    - Interactive elements
    - Practice problems
    - Assessment questions
    """
    try:
        result = await ai_jflap.process_request(
            "tutoring",
            {
                "algorithm": request.algorithm,
                "student_id": request.student_id,
                "student_level": request.student_level
            }
        )
        
        if result["success"]:
            return result["result"]
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Tutorial creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tutor/hint")
async def get_hint(request: HintRequest):
    """
    Get adaptive hint for problem solving.
    
    Hint levels:
    - Level 1: Subtle nudge
    - Level 2: Specific guidance
    - Level 3: Detailed help
    """
    try:
        tutor = StepByStepTutor()
        hint = await tutor.provide_hint(
            request.problem,
            request.current_attempt,
            request.hint_level
        )
        
        return {
            "problem": request.problem,
            "hint": hint,
            "hint_level": request.hint_level
        }
        
    except Exception as e:
        logger.error(f"Hint generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tutor/explain-step")
async def explain_step(
    algorithm: str,
    step_number: int,
    context: Optional[Dict[str, Any]] = None
):
    """
    Explain specific algorithm step in detail.
    
    Provides detailed explanation with examples.
    """
    try:
        tutor = StepByStepTutor()
        explanation = await tutor.explain_step(
            algorithm,
            step_number,
            context or {}
        )
        
        return {
            "algorithm": algorithm,
            "step_number": step_number,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.error(f"Step explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# STREAMING ENDPOINTS
# ============================

@router.get("/stream/tutorial/{algorithm}")
async def stream_tutorial(
    algorithm: str,
    student_id: str = Query(..., description="Student identifier"),
    student_level: str = Query(default="intermediate")
):
    """
    Stream tutorial content as it's generated.
    
    Useful for long tutorials to show progress.
    """
    async def generate():
        try:
            tutor = StepByStepTutor()
            tutorial = await tutor.create_tutorial(algorithm, student_id, student_level)
            
            # Stream sections as they're available
            sections = [
                ("introduction", tutorial.get("tutorial", "")[:500]),
                ("content", tutorial.get("tutorial", "")[500:]),
                ("interactive", json.dumps(tutorial.get("interactive", {}))),
                ("assessment", json.dumps(tutorial.get("assessment", [])))
            ]
            
            for section_name, content in sections:
                chunk = json.dumps({
                    "section": section_name,
                    "content": content
                }) + "\n"
                yield chunk.encode()
                await asyncio.sleep(0.1)  # Small delay for streaming effect
                
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode()
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ============================
# METRICS & MONITORING
# ============================

@router.get("/metrics")
async def get_metrics():
    """
    Get AI service performance metrics.
    
    Returns usage statistics and performance data.
    """
    return ai_jflap.get_metrics()


@router.get("/health")
async def health_check():
    """
    Check health of AI services.
    
    Verifies all components are operational.
    """
    try:
        # Quick test of each component
        components = {
            "tm_generator": ai_jflap.tm_generator is not None,
            "grammar_analyzer": ai_jflap.grammar_analyzer is not None,
            "error_recovery": ai_jflap.error_recovery is not None,
            "test_generator": ai_jflap.test_generator is not None,
            "nl_converter": ai_jflap.nl_converter is not None,
            "tutor": ai_jflap.tutor is not None
        }
        
        all_healthy = all(components.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }