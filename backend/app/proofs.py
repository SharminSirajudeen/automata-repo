"""
Interactive Proof System for Automata Theory
Supports step-by-step proof validation with hints and multiple proof techniques.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class ProofTechnique(str, Enum):
    """Supported proof techniques"""
    CONTRADICTION = "contradiction"
    INDUCTION = "induction" 
    CONSTRUCTION = "construction"
    DIRECT = "direct"
    CONTRAPOSITIVE = "contrapositive"

class ProofStepType(str, Enum):
    """Types of proof steps"""
    ASSUMPTION = "assumption"
    GIVEN = "given"
    DEFINITION = "definition"
    THEOREM = "theorem"
    INFERENCE = "inference"
    CONTRADICTION_FOUND = "contradiction_found"
    BASE_CASE = "base_case"
    INDUCTIVE_STEP = "inductive_step"
    CONSTRUCTION_STEP = "construction_step"
    CONCLUSION = "conclusion"

class ProofStep(BaseModel):
    """Individual step in a proof"""
    step_number: int = Field(..., ge=1)
    step_type: ProofStepType
    statement: str = Field(..., min_length=1, max_length=1000)
    justification: Optional[str] = Field(None, max_length=500)
    references: List[int] = Field(default_factory=list)  # References to previous steps
    is_valid: Optional[bool] = None
    feedback: Optional[str] = None

class ProofState(BaseModel):
    """Current state of a proof"""
    theorem_statement: str = Field(..., min_length=1)
    technique: ProofTechnique
    steps: List[ProofStep] = Field(default_factory=list)
    is_complete: bool = False
    is_valid: bool = False
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    hints_used: int = Field(default=0, ge=0)

class ProofValidationRequest(BaseModel):
    """Request for proof step validation"""
    proof_state: ProofState
    new_step: ProofStep

class ProofHintRequest(BaseModel):
    """Request for proof hints"""
    proof_state: ProofState
    stuck_reason: Optional[str] = None

class ProofValidator:
    """Core proof validation engine"""
    
    def __init__(self):
        self.known_theorems = self._load_known_theorems()
        self.definitions = self._load_definitions()
        
    def _load_known_theorems(self) -> Dict[str, Dict[str, Any]]:
        """Load known theorems and their properties"""
        return {
            "pumping_lemma_regular": {
                "statement": "If L is regular, then there exists p such that for all w in L with |w| >= p, w = xyz where |xy| <= p, |y| >= 1, and xy^i z in L for all i >= 0",
                "domain": "regular_languages",
                "prerequisites": ["finite_automata", "regular_languages"]
            },
            "pumping_lemma_cfl": {
                "statement": "If L is context-free, then there exists p such that for all w in L with |w| >= p, w = uvxyz where |vxy| <= p, |vy| >= 1, and uv^i xy^i z in L for all i >= 0",
                "domain": "context_free_languages",
                "prerequisites": ["pushdown_automata", "context_free_grammars"]
            },
            "myhill_nerode": {
                "statement": "A language L is regular if and only if the relation ~_L has finite index",
                "domain": "regular_languages",
                "prerequisites": ["equivalence_relations", "regular_languages"]
            },
            "complement_regular": {
                "statement": "The class of regular languages is closed under complement",
                "domain": "regular_languages",
                "prerequisites": ["finite_automata", "regular_languages"]
            },
            "union_regular": {
                "statement": "The class of regular languages is closed under union",
                "domain": "regular_languages", 
                "prerequisites": ["finite_automata", "regular_languages"]
            },
            "intersection_regular": {
                "statement": "The class of regular languages is closed under intersection",
                "domain": "regular_languages",
                "prerequisites": ["finite_automata", "regular_languages"]
            }
        }
    
    def _load_definitions(self) -> Dict[str, str]:
        """Load important definitions"""
        return {
            "regular_language": "A language L is regular if there exists a finite automaton that recognizes L",
            "context_free_language": "A language L is context-free if there exists a context-free grammar that generates L",
            "finite_automaton": "A 5-tuple (Q, Σ, δ, q0, F) where Q is finite set of states, Σ is alphabet, δ is transition function, q0 is start state, F is set of accept states",
            "pushdown_automaton": "A 7-tuple (Q, Σ, Γ, δ, q0, Z0, F) with stack alphabet Γ and stack operations",
            "decidable_language": "A language L is decidable if there exists a Turing machine that decides L",
            "recognizable_language": "A language L is recognizable if there exists a Turing machine that recognizes L"
        }
    
    def validate_step(self, proof_state: ProofState, new_step: ProofStep) -> Tuple[bool, str]:
        """Validate a single proof step"""
        try:
            # Check step number sequence
            expected_step_num = len(proof_state.steps) + 1
            if new_step.step_number != expected_step_num:
                return False, f"Expected step number {expected_step_num}, got {new_step.step_number}"
            
            # Validate based on proof technique
            if proof_state.technique == ProofTechnique.CONTRADICTION:
                return self._validate_contradiction_step(proof_state, new_step)
            elif proof_state.technique == ProofTechnique.INDUCTION:
                return self._validate_induction_step(proof_state, new_step)
            elif proof_state.technique == ProofTechnique.CONSTRUCTION:
                return self._validate_construction_step(proof_state, new_step)
            elif proof_state.technique == ProofTechnique.DIRECT:
                return self._validate_direct_step(proof_state, new_step)
            else:
                return self._validate_generic_step(proof_state, new_step)
                
        except Exception as e:
            logger.error(f"Error validating proof step: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_contradiction_step(self, proof_state: ProofState, new_step: ProofStep) -> Tuple[bool, str]:
        """Validate contradiction proof step"""
        if new_step.step_number == 1:
            if new_step.step_type != ProofStepType.ASSUMPTION:
                return False, "First step in contradiction proof must be an assumption (assume the opposite of what we want to prove)"
            
            # Check if assumption is negation of theorem
            if not self._is_negation_of_theorem(new_step.statement, proof_state.theorem_statement):
                return False, "Assumption should be the negation of the theorem statement"
        
        elif new_step.step_type == ProofStepType.CONTRADICTION_FOUND:
            # Check if we actually have a contradiction
            if not self._has_contradiction(proof_state.steps):
                return False, "No valid contradiction found in previous steps"
            
            # This should be near the end
            if len(proof_state.steps) < 3:
                return False, "Contradiction found too early - need more development"
        
        return self._validate_generic_step(proof_state, new_step)
    
    def _validate_induction_step(self, proof_state: ProofState, new_step: ProofStep) -> Tuple[bool, str]:
        """Validate induction proof step"""
        if new_step.step_type == ProofStepType.BASE_CASE:
            # Should be early in proof
            if len(proof_state.steps) > 3:
                return False, "Base case should be established early in induction proof"
            
            # Check if it addresses the base case properly
            if "n = 0" not in new_step.statement and "n = 1" not in new_step.statement:
                return False, "Base case should specify the initial value (e.g., n = 0 or n = 1)"
        
        elif new_step.step_type == ProofStepType.INDUCTIVE_STEP:
            # Should have base case first
            if not any(step.step_type == ProofStepType.BASE_CASE for step in proof_state.steps):
                return False, "Must establish base case before inductive step"
            
            # Should assume inductive hypothesis
            if "assume" not in new_step.statement.lower() and "hypothesis" not in new_step.statement.lower():
                return False, "Inductive step should state the inductive hypothesis"
        
        return self._validate_generic_step(proof_state, new_step)
    
    def _validate_construction_step(self, proof_state: ProofState, new_step: ProofStep) -> Tuple[bool, str]:
        """Validate construction proof step"""
        if new_step.step_type == ProofStepType.CONSTRUCTION_STEP:
            # Should be building something concrete
            construction_keywords = ["construct", "build", "define", "create", "set"]
            if not any(keyword in new_step.statement.lower() for keyword in construction_keywords):
                return False, "Construction step should explicitly construct or define something"
            
            # Should have proper justification
            if not new_step.justification:
                return False, "Construction steps require justification"
        
        return self._validate_generic_step(proof_state, new_step)
    
    def _validate_direct_step(self, proof_state: ProofState, new_step: ProofStep) -> Tuple[bool, str]:
        """Validate direct proof step"""
        # Direct proofs should flow logically from premises to conclusion
        if new_step.step_type == ProofStepType.GIVEN and len(proof_state.steps) > 0:
            return False, "Given statements should be at the beginning of the proof"
        
        return self._validate_generic_step(proof_state, new_step)
    
    def _validate_generic_step(self, proof_state: ProofState, new_step: ProofStep) -> Tuple[bool, str]:
        """Generic step validation common to all proof techniques"""
        # Check references exist
        for ref in new_step.references:
            if ref < 1 or ref > len(proof_state.steps):
                return False, f"Invalid reference to step {ref}"
        
        # Check logical flow based on step type
        if new_step.step_type == ProofStepType.INFERENCE:
            if not new_step.references:
                return False, "Inference steps must reference previous steps"
            
            if not new_step.justification:
                return False, "Inference steps require justification"
        
        elif new_step.step_type == ProofStepType.THEOREM:
            # Should reference a known theorem
            theorem_name = self._extract_theorem_name(new_step.statement)
            if theorem_name and theorem_name not in self.known_theorems:
                return False, f"Unknown theorem: {theorem_name}"
        
        # Check statement quality
        if len(new_step.statement.strip()) < 10:
            return False, "Proof step statement is too brief"
        
        return True, "Valid step"
    
    def _is_negation_of_theorem(self, assumption: str, theorem: str) -> bool:
        """Check if assumption is negation of theorem"""
        # Simple heuristic - look for negation indicators
        negation_indicators = ["not", "does not", "cannot", "is not", "are not"]
        assumption_lower = assumption.lower()
        theorem_lower = theorem.lower()
        
        # If theorem has negation, assumption should not
        theorem_has_negation = any(neg in theorem_lower for neg in negation_indicators)
        assumption_has_negation = any(neg in assumption_lower for neg in negation_indicators)
        
        return theorem_has_negation != assumption_has_negation
    
    def _has_contradiction(self, steps: List[ProofStep]) -> bool:
        """Check if steps contain a logical contradiction"""
        statements = [step.statement.lower() for step in steps]
        
        # Look for explicit contradictions
        contradiction_patterns = [
            (r"(\w+) is regular", r"\1 is not regular"),
            (r"(\w+) is decidable", r"\1 is undecidable"),
            (r"(\w+) exists", r"\1 does not exist"),
            (r"(\w+) = (\w+)", r"\1 ≠ \2"),
        ]
        
        for pattern1, pattern2 in contradiction_patterns:
            for stmt1 in statements:
                for stmt2 in statements:
                    if re.search(pattern1, stmt1) and re.search(pattern2, stmt2):
                        return True
        
        return False
    
    def _extract_theorem_name(self, statement: str) -> Optional[str]:
        """Extract theorem name from statement"""
        theorem_patterns = [
            r"by (?:the )?(\w+(?:\s+\w+)*) (?:theorem|lemma)",
            r"using (?:the )?(\w+(?:\s+\w+)*) (?:theorem|lemma)",
            r"(?:theorem|lemma):?\s*(\w+(?:\s+\w+)*)"
        ]
        
        for pattern in theorem_patterns:
            match = re.search(pattern, statement.lower())
            if match:
                return match.group(1).replace(" ", "_")
        
        return None
    
    def check_completeness(self, proof_state: ProofState) -> Tuple[bool, str]:
        """Check if proof is complete and valid"""
        if not proof_state.steps:
            return False, "Proof has no steps"
        
        # Check technique-specific completeness
        if proof_state.technique == ProofTechnique.CONTRADICTION:
            return self._check_contradiction_completeness(proof_state)
        elif proof_state.technique == ProofTechnique.INDUCTION:
            return self._check_induction_completeness(proof_state)
        elif proof_state.technique == ProofTechnique.CONSTRUCTION:
            return self._check_construction_completeness(proof_state)
        else:
            return self._check_generic_completeness(proof_state)
    
    def _check_contradiction_completeness(self, proof_state: ProofState) -> Tuple[bool, str]:
        """Check if contradiction proof is complete"""
        has_assumption = any(step.step_type == ProofStepType.ASSUMPTION for step in proof_state.steps)
        has_contradiction = any(step.step_type == ProofStepType.CONTRADICTION_FOUND for step in proof_state.steps)
        has_conclusion = any(step.step_type == ProofStepType.CONCLUSION for step in proof_state.steps)
        
        if not has_assumption:
            return False, "Contradiction proof missing initial assumption"
        if not has_contradiction:
            return False, "Contradiction proof missing contradiction"
        if not has_conclusion:
            return False, "Proof missing conclusion"
        
        return True, "Complete contradiction proof"
    
    def _check_induction_completeness(self, proof_state: ProofState) -> Tuple[bool, str]:
        """Check if induction proof is complete"""
        has_base_case = any(step.step_type == ProofStepType.BASE_CASE for step in proof_state.steps)
        has_inductive_step = any(step.step_type == ProofStepType.INDUCTIVE_STEP for step in proof_state.steps)
        has_conclusion = any(step.step_type == ProofStepType.CONCLUSION for step in proof_state.steps)
        
        if not has_base_case:
            return False, "Induction proof missing base case"
        if not has_inductive_step:
            return False, "Induction proof missing inductive step"
        if not has_conclusion:
            return False, "Proof missing conclusion"
        
        return True, "Complete induction proof"
    
    def _check_construction_completeness(self, proof_state: ProofState) -> Tuple[bool, str]:
        """Check if construction proof is complete"""
        has_construction = any(step.step_type == ProofStepType.CONSTRUCTION_STEP for step in proof_state.steps)
        has_verification = len([step for step in proof_state.steps if "verify" in step.statement.lower() or "check" in step.statement.lower()]) > 0
        has_conclusion = any(step.step_type == ProofStepType.CONCLUSION for step in proof_state.steps)
        
        if not has_construction:
            return False, "Construction proof missing construction step"
        if not has_verification:
            return False, "Construction proof should verify the construction works"
        if not has_conclusion:
            return False, "Proof missing conclusion"
        
        return True, "Complete construction proof"
    
    def _check_generic_completeness(self, proof_state: ProofState) -> Tuple[bool, str]:
        """Check generic proof completeness"""
        has_conclusion = any(step.step_type == ProofStepType.CONCLUSION for step in proof_state.steps)
        
        if not has_conclusion:
            return False, "Proof missing conclusion"
        
        if len(proof_state.steps) < 3:
            return False, "Proof too short - needs more development"
        
        return True, "Complete proof"

class ProofHintGenerator:
    """Generates hints for students stuck on proofs"""
    
    def __init__(self, validator: ProofValidator):
        self.validator = validator
    
    def generate_hint(self, proof_state: ProofState, stuck_reason: Optional[str] = None) -> str:
        """Generate a helpful hint based on current proof state"""
        try:
            if not proof_state.steps:
                return self._get_starting_hint(proof_state)
            
            # Analyze current progress
            if proof_state.technique == ProofTechnique.CONTRADICTION:
                return self._get_contradiction_hint(proof_state, stuck_reason)
            elif proof_state.technique == ProofTechnique.INDUCTION:
                return self._get_induction_hint(proof_state, stuck_reason)
            elif proof_state.technique == ProofTechnique.CONSTRUCTION:
                return self._get_construction_hint(proof_state, stuck_reason)
            else:
                return self._get_generic_hint(proof_state, stuck_reason)
                
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return "Try breaking down the problem into smaller steps and consider what you need to prove."
    
    def _get_starting_hint(self, proof_state: ProofState) -> str:
        """Hint for starting a proof"""
        if proof_state.technique == ProofTechnique.CONTRADICTION:
            return "Start by assuming the opposite of what you want to prove. If your theorem states that something IS true, assume it is NOT true."
        elif proof_state.technique == ProofTechnique.INDUCTION:
            return "Begin with the base case. What is the smallest value for which your statement should hold? Prove it holds for that value."
        elif proof_state.technique == ProofTechnique.CONSTRUCTION:
            return "Start by clearly stating what you need to construct. What object (automaton, grammar, etc.) will demonstrate your theorem?"
        else:
            return "Begin by stating what is given and what you need to prove. Consider which definitions and theorems might be relevant."
    
    def _get_contradiction_hint(self, proof_state: ProofState, stuck_reason: Optional[str] = None) -> str:
        """Hint for contradiction proofs"""
        has_assumption = any(step.step_type == ProofStepType.ASSUMPTION for step in proof_state.steps)
        
        if not has_assumption:
            return "You need to start with an assumption that contradicts your theorem. What would it mean if your theorem were false?"
        
        # Check if we're developing the contradiction
        if len(proof_state.steps) < 4:
            return "Now develop the consequences of your assumption. What can you derive from it? Look for something that contradicts a known fact."
        
        # Look for contradiction
        if not any(step.step_type == ProofStepType.CONTRADICTION_FOUND for step in proof_state.steps):
            return "You've developed several steps from your assumption. Look for a contradiction with: 1) A known theorem, 2) A definition, or 3) Something you derived earlier."
        
        return "You found a contradiction! Now conclude that your original assumption must be false, which proves your theorem."
    
    def _get_induction_hint(self, proof_state: ProofState, stuck_reason: Optional[str] = None) -> str:
        """Hint for induction proofs"""
        has_base_case = any(step.step_type == ProofStepType.BASE_CASE for step in proof_state.steps)
        has_inductive_step = any(step.step_type == ProofStepType.INDUCTIVE_STEP for step in proof_state.steps)
        
        if not has_base_case:
            return "Establish the base case first. For what smallest value does your statement hold? Prove it directly."
        
        if not has_inductive_step:
            return "Now for the inductive step: assume your statement holds for some arbitrary k, then prove it holds for k+1."
        
        # In inductive step development
        inductive_steps = [step for step in proof_state.steps if step.step_type == ProofStepType.INDUCTIVE_STEP]
        if len(inductive_steps) == 1:
            return "You've assumed the inductive hypothesis. Now use this assumption to prove the statement for the next case (k+1)."
        
        return "Complete the inductive step by showing how the case k+1 follows from the case k. Then conclude by induction."
    
    def _get_construction_hint(self, proof_state: ProofState, stuck_reason: Optional[str] = None) -> str:
        """Hint for construction proofs"""
        has_construction = any(step.step_type == ProofStepType.CONSTRUCTION_STEP for step in proof_state.steps)
        
        if not has_construction:
            return "Describe exactly what you're going to construct. Be specific about the components (states, transitions, rules, etc.)."
        
        # Check if construction is complete
        construction_steps = [step for step in proof_state.steps if step.step_type == ProofStepType.CONSTRUCTION_STEP]
        if len(construction_steps) < 2:
            return "Continue building your construction. Make sure to define all necessary components clearly."
        
        # Check for verification
        has_verification = any("verify" in step.statement.lower() or "check" in step.statement.lower() for step in proof_state.steps)
        if not has_verification:
            return "Now verify that your construction works. Show that it satisfies the required properties."
        
        return "Complete your proof by explaining why your construction demonstrates the theorem."
    
    def _get_generic_hint(self, proof_state: ProofState, stuck_reason: Optional[str] = None) -> str:
        """Generic hint for any proof"""
        if len(proof_state.steps) < 3:
            return "Develop your proof further. What definitions, theorems, or properties can you apply?"
        
        last_step = proof_state.steps[-1]
        if last_step.step_type not in [ProofStepType.CONCLUSION]:
            return "You're making progress! Consider how to connect your current work to the final conclusion."
        
        return "Review your logical flow. Does each step follow clearly from the previous ones?"

# Global instances
proof_validator = ProofValidator()
proof_hint_generator = ProofHintGenerator(proof_validator)

def validate_proof_step(request: ProofValidationRequest) -> Dict[str, Any]:
    """Validate a single proof step and return feedback"""
    is_valid, feedback = proof_validator.validate_step(request.proof_state, request.new_step)
    
    # Update step validation status
    request.new_step.is_valid = is_valid
    request.new_step.feedback = feedback
    
    # Add step to proof state
    new_proof_state = request.proof_state.copy()
    new_proof_state.steps.append(request.new_step)
    
    # Check if proof is complete
    is_complete, completeness_feedback = proof_validator.check_completeness(new_proof_state)
    new_proof_state.is_complete = is_complete
    new_proof_state.is_valid = is_complete and is_valid
    
    # Update score
    if is_valid:
        new_proof_state.score = min(100.0, new_proof_state.score + (100.0 / max(1, len(new_proof_state.steps))))
    
    return {
        "is_valid": is_valid,
        "feedback": feedback,
        "proof_state": new_proof_state,
        "is_complete": is_complete,
        "completeness_feedback": completeness_feedback
    }

def generate_proof_hint(request: ProofHintRequest) -> Dict[str, Any]:
    """Generate a hint for the current proof state"""
    hint = proof_hint_generator.generate_hint(request.proof_state, request.stuck_reason)
    
    # Update hints used counter
    new_proof_state = request.proof_state.copy()
    new_proof_state.hints_used += 1
    
    # Slight score penalty for using hints
    if new_proof_state.score > 5:
        new_proof_state.score -= 5
    
    return {
        "hint": hint,
        "proof_state": new_proof_state,
        "hints_used": new_proof_state.hints_used
    }

def get_proof_templates() -> Dict[str, Any]:
    """Get templates for different proof techniques"""
    return {
        "contradiction": {
            "description": "Assume the opposite of what you want to prove, then derive a contradiction",
            "steps": [
                "Assume the negation of the theorem statement",
                "Derive consequences from this assumption",
                "Find a contradiction with known facts",
                "Conclude the assumption is false, proving the theorem"
            ]
        },
        "induction": {
            "description": "Prove a base case, then show the property is preserved",
            "steps": [
                "Prove the base case (usually n=0 or n=1)",
                "State the inductive hypothesis (assume true for k)",
                "Prove the inductive step (show true for k+1)",
                "Conclude by mathematical induction"
            ]
        },
        "construction": {
            "description": "Build an object that demonstrates the theorem",
            "steps": [
                "Define what you will construct",
                "Build the construction step by step",
                "Verify the construction has required properties",
                "Conclude the construction proves the theorem"
            ]
        },
        "direct": {
            "description": "Use logical steps to go directly from premises to conclusion",
            "steps": [
                "State what is given",
                "Apply relevant definitions and theorems",
                "Make valid logical inferences",
                "Reach the desired conclusion"
            ]
        }
    }