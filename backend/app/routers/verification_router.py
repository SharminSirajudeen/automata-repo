"""
Verification router for the Automata Learning Platform.
Handles formal verification, pumping lemma, complexity theory, and proof validation.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from ..verification import (
    verification_engine,
    EquivalenceResult,
    ContainmentResult,
    CounterExampleResult
)
from ..pumping import pumping_lemma_engine, PumpingRequest, DecompositionResult
from ..complexity import complexity_analyzer, ComplexityClass, ReductionRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/verification", tags=["verification"])


class EquivalenceRequest(BaseModel):
    automaton1: Dict[str, Any]
    automaton2: Dict[str, Any]
    method: str = "cross_product"


class ContainmentRequest(BaseModel):
    automaton1: Dict[str, Any]  # L(A1) ⊆ L(A2)?
    automaton2: Dict[str, Any]
    method: str = "complement_intersection"


class MinimizationRequest(BaseModel):
    automaton: Dict[str, Any]
    algorithm: str = "hopcroft"


class CounterExampleRequest(BaseModel):
    claim: str
    automaton1: Dict[str, Any]
    automaton2: Dict[str, Any]
    max_length: int = 20


class PumpingValidationRequest(BaseModel):
    language_description: str
    pumping_constant: int
    witness_string: str
    decomposition: Dict[str, str]  # x, y, z parts
    claim: str  # "regular" or "not_regular"


class ComplexityAnalysisRequest(BaseModel):
    problem_description: str
    algorithm_description: Optional[str] = None
    input_size_param: str = "n"


class ReductionVerificationRequest(BaseModel):
    source_problem: str
    target_problem: str
    reduction_description: str
    reduction_type: str = "polynomial"


# Formal Verification Endpoints
@router.post("/equivalence")
async def check_equivalence(request: EquivalenceRequest):
    """Check if two automata recognize the same language"""
    try:
        result = await verification_engine.check_equivalence(
            automaton1=request.automaton1,
            automaton2=request.automaton2,
            method=request.method
        )
        
        return {
            "automata": {
                "automaton1": request.automaton1,
                "automaton2": request.automaton2
            },
            "equivalent": result.equivalent,
            "method": request.method,
            "proof_sketch": result.proof_sketch,
            "witness_strings": result.witness_strings if not result.equivalent else None,
            "confidence": result.confidence,
            "computational_steps": result.steps_taken
        }
        
    except Exception as e:
        logger.error(f"Equivalence check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/containment")
async def check_containment(request: ContainmentRequest):
    """Check if L(A1) ⊆ L(A2)"""
    try:
        result = await verification_engine.check_containment(
            automaton1=request.automaton1,
            automaton2=request.automaton2,
            method=request.method
        )
        
        return {
            "automata": {
                "subset_candidate": request.automaton1,
                "superset_candidate": request.automaton2
            },
            "contained": result.contained,
            "method": request.method,
            "counterexample": result.counterexample if not result.contained else None,
            "proof_outline": result.proof_outline,
            "verification_steps": result.verification_steps
        }
        
    except Exception as e:
        logger.error(f"Containment check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/minimize")
async def minimize_automaton(request: MinimizationRequest):
    """Minimize an automaton and verify the result"""
    try:
        result = await verification_engine.minimize_and_verify(
            automaton=request.automaton,
            algorithm=request.algorithm
        )
        
        return {
            "original_automaton": request.automaton,
            "minimized_automaton": result.minimized_automaton,
            "algorithm": request.algorithm,
            "equivalence_verified": result.equivalence_verified,
            "reduction_stats": {
                "original_states": result.original_state_count,
                "minimized_states": result.minimized_state_count,
                "reduction_percentage": result.reduction_percentage
            },
            "minimization_steps": result.steps
        }
        
    except Exception as e:
        logger.error(f"Minimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counter-example")
async def find_counter_example(request: CounterExampleRequest):
    """Find a counter-example to disprove a claim about automata"""
    try:
        result = await verification_engine.find_counter_example(
            claim=request.claim,
            automaton1=request.automaton1,
            automaton2=request.automaton2,
            max_length=request.max_length
        )
        
        return {
            "claim": request.claim,
            "counter_example_found": result.found,
            "counter_example": result.counter_example if result.found else None,
            "verification": result.verification_details,
            "search_space_explored": result.search_stats,
            "conclusion": result.conclusion
        }
        
    except Exception as e:
        logger.error(f"Counter-example search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/algorithms")
async def list_verification_algorithms():
    """Get information about available verification algorithms"""
    return {
        "equivalence_methods": {
            "cross_product": {
                "description": "Cross product construction with DFS",
                "complexity": "O(|Q1| × |Q2|)",
                "best_for": "Small to medium automata"
            },
            "bisimulation": {
                "description": "Bisimulation-based equivalence",
                "complexity": "O(|Q|^2 × |Σ|)",
                "best_for": "Minimized automata"
            },
            "table_filling": {
                "description": "Table-filling algorithm",
                "complexity": "O(|Q|^2 × |Σ|)",
                "best_for": "DFA equivalence"
            }
        },
        "containment_methods": {
            "complement_intersection": {
                "description": "L1 ⊆ L2 iff L1 ∩ L̄2 = ∅",
                "complexity": "Exponential (due to complementation)",
                "best_for": "Theoretical verification"
            },
            "simulation": {
                "description": "Simulation-based containment",
                "complexity": "Polynomial in many cases",
                "best_for": "Practical verification"
            }
        },
        "minimization_algorithms": {
            "hopcroft": {
                "description": "Hopcroft's minimization algorithm",
                "complexity": "O(n log n)",
                "best_for": "Large DFAs"
            },
            "moore": {
                "description": "Moore's algorithm",
                "complexity": "O(n^2)",
                "best_for": "Educational purposes"
            }
        }
    }


# Pumping Lemma Endpoints
@router.post("/pumping/validate")
async def validate_pumping_proof(request: PumpingValidationRequest):
    """Validate a pumping lemma proof or disproof"""
    try:
        result = await pumping_lemma_engine.validate_proof(
            language_description=request.language_description,
            pumping_constant=request.pumping_constant,
            witness_string=request.witness_string,
            decomposition=request.decomposition,
            claim=request.claim
        )
        
        return {
            "language_description": request.language_description,
            "pumping_constant": request.pumping_constant,
            "witness_string": request.witness_string,
            "decomposition": request.decomposition,
            "claim": request.claim,
            "proof_valid": result.valid,
            "verification_details": result.details,
            "errors": result.errors if not result.valid else None,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"Pumping lemma validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pumping/candidates/{language_description}")
async def get_pumping_candidates(language_description: str):
    """Get candidate strings for pumping lemma proofs"""
    try:
        candidates = await pumping_lemma_engine.generate_candidates(
            language_description
        )
        
        return {
            "language_description": language_description,
            "candidates": candidates,
            "generation_strategy": "pattern_analysis",
            "recommended_pumping_constant": candidates.get("recommended_p", 3)
        }
        
    except Exception as e:
        logger.error(f"Pumping candidate generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pumping/decompose-regular/{string}/{pumping_length}")
async def decompose_regular_string(string: str, pumping_length: int):
    """Generate all valid xyz decompositions for regular pumping lemma"""
    try:
        decompositions = await pumping_lemma_engine.decompose_regular(
            string=string,
            pumping_length=pumping_length
        )
        
        return {
            "string": string,
            "pumping_length": pumping_length,
            "decompositions": decompositions,
            "total_decompositions": len(decompositions),
            "constraints": {
                "xy_length_leq_p": True,
                "y_length_geq_1": True
            }
        }
        
    except Exception as e:
        logger.error(f"Regular decomposition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pumping/decompose-cfl/{string}/{pumping_length}")
async def decompose_cfl_string(string: str, pumping_length: int):
    """Generate all valid uvwxy decompositions for CFL pumping lemma"""
    try:
        decompositions = await pumping_lemma_engine.decompose_cfl(
            string=string,
            pumping_length=pumping_length
        )
        
        return {
            "string": string,
            "pumping_length": pumping_length,
            "decompositions": decompositions,
            "total_decompositions": len(decompositions),
            "constraints": {
                "vwx_length_leq_p": True,
                "vx_length_geq_1": True
            }
        }
        
    except Exception as e:
        logger.error(f"CFL decomposition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pumping/templates")
async def get_pumping_templates():
    """Get templates for common pumping lemma proofs"""
    return {
        "regular_templates": {
            "not_regular_proof": {
                "steps": [
                    "Assume L is regular",
                    "Let p be the pumping length",
                    "Choose string s ∈ L with |s| ≥ p",
                    "Consider all xyz decompositions where |xy| ≤ p and |y| ≥ 1",
                    "Show that xy^i z ∉ L for some i ≥ 0",
                    "Contradiction, so L is not regular"
                ],
                "common_strings": ["a^p b^p", "a^p b^p c^p", "ww^R"]
            }
        },
        "cfl_templates": {
            "not_cfl_proof": {
                "steps": [
                    "Assume L is context-free",
                    "Let p be the pumping length",
                    "Choose string s ∈ L with |s| ≥ p",
                    "Consider all uvwxy decompositions where |vwx| ≤ p and |vx| ≥ 1",
                    "Show that uv^i wx^i y ∉ L for some i ≥ 0",
                    "Contradiction, so L is not context-free"
                ],
                "common_strings": ["a^p b^p c^p", "a^i b^j c^i d^j"]
            }
        }
    }


# Complexity Theory Endpoints
@router.post("/complexity/analyze")
async def analyze_complexity(request: ComplexityAnalysisRequest):
    """Analyze the computational complexity of a problem or algorithm"""
    try:
        analysis = await complexity_analyzer.analyze(
            problem_description=request.problem_description,
            algorithm_description=request.algorithm_description,
            input_size_param=request.input_size_param
        )
        
        return {
            "problem_description": request.problem_description,
            "complexity_analysis": {
                "time_complexity": analysis.time_complexity,
                "space_complexity": analysis.space_complexity,
                "complexity_class": analysis.complexity_class,
                "decision_vs_optimization": analysis.problem_type
            },
            "analysis_details": analysis.details,
            "related_problems": analysis.related_problems,
            "reduction_possibilities": analysis.reductions
        }
        
    except Exception as e:
        logger.error(f"Complexity analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complexity/verify-reduction")
async def verify_reduction(request: ReductionVerificationRequest):
    """Verify a complexity reduction between problems"""
    try:
        verification = await complexity_analyzer.verify_reduction(
            source_problem=request.source_problem,
            target_problem=request.target_problem,
            reduction_description=request.reduction_description,
            reduction_type=request.reduction_type
        )
        
        return {
            "reduction": {
                "source": request.source_problem,
                "target": request.target_problem,
                "type": request.reduction_type
            },
            "valid_reduction": verification.valid,
            "verification_details": verification.details,
            "complexity_implications": verification.implications,
            "correctness_proof": verification.proof_outline
        }
        
    except Exception as e:
        logger.error(f"Reduction verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/complexity/classes")
async def list_complexity_classes():
    """Get information about complexity classes"""
    return {
        "time_complexity_classes": {
            "P": {
                "description": "Polynomial time",
                "definition": "DTIME(n^k) for some k",
                "examples": ["Sorting", "Graph connectivity", "Linear programming"]
            },
            "NP": {
                "description": "Nondeterministic polynomial time",
                "definition": "NTIME(n^k) for some k",
                "examples": ["SAT", "Hamiltonian path", "TSP decision"]
            },
            "PSPACE": {
                "description": "Polynomial space",
                "definition": "DSPACE(n^k) for some k",
                "examples": ["QBF", "Game tree evaluation"]
            },
            "EXPTIME": {
                "description": "Exponential time",
                "definition": "DTIME(2^n^k) for some k",
                "examples": ["Model checking", "Chess with n×n board"]
            }
        },
        "relationships": {
            "inclusions": "P ⊆ NP ⊆ PSPACE ⊆ EXPTIME",
            "open_questions": ["P vs NP", "NP vs PSPACE"]
        }
    }


@router.get("/complexity/relationship/{class1}/{class2}")
async def get_complexity_relationship(class1: str, class2: str):
    """Get the relationship between two complexity classes"""
    try:
        relationship = await complexity_analyzer.get_class_relationship(
            class1, class2
        )
        
        return {
            "class1": class1,
            "class2": class2,
            "relationship": relationship.relationship_type,
            "known_inclusions": relationship.inclusions,
            "separation_results": relationship.separations,
            "open_questions": relationship.open_questions,
            "key_results": relationship.key_theorems
        }
        
    except Exception as e:
        logger.error(f"Complexity relationship error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/complexity/templates")
async def get_complexity_templates():
    """Get templates for complexity theory proofs and reductions"""
    return {
        "reduction_templates": {
            "polynomial_reduction": {
                "structure": [
                    "Define transformation f: Instance(A) → Instance(B)",
                    "Prove f is computable in polynomial time",
                    "Prove x ∈ A iff f(x) ∈ B",
                    "Conclude A ≤_p B"
                ]
            },
            "many_one_reduction": {
                "structure": [
                    "Given instance x of problem A",
                    "Construct instance f(x) of problem B",
                    "Show construction takes polynomial time",
                    "Prove correctness: x ∈ A ↔ f(x) ∈ B"
                ]
            }
        },
        "proof_templates": {
            "np_completeness": {
                "steps": [
                    "Show problem is in NP",
                    "Choose known NP-complete problem",
                    "Construct polynomial reduction",
                    "Prove reduction correctness",
                    "Conclude NP-completeness"
                ]
            }
        }
    }