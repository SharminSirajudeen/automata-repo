"""
AI-Powered Proof Assistant for the Automata Learning Platform.
Integrates formal methods with AI for proof generation and verification.
"""
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import logging
from dataclasses import dataclass
import asyncio

from .orchestrator import orchestrator, ExecutionMode
from .prompts import prompt_builder, PromptExample
from .ai_config import ModelType

logger = logging.getLogger(__name__)


class ProofTechnique(str, Enum):
    """Available proof techniques."""
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CONSTRUCTION = "construction"
    CONTRAPOSITIVE = "contrapositive"
    EXHAUSTION = "exhaustion"
    DIAGONALIZATION = "diagonalization"
    PUMPING_LEMMA = "pumping_lemma"


class ProofStatus(str, Enum):
    """Status of a proof."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


class FormalStatement(BaseModel):
    """Formal mathematical statement."""
    statement: str
    variables: List[str] = Field(default_factory=list)
    quantifiers: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    goal: str
    domain: Optional[str] = None


class ProofStep(BaseModel):
    """Individual step in a proof."""
    step_number: int
    statement: str
    justification: str
    rule_applied: Optional[str] = None
    dependencies: List[int] = Field(default_factory=list)
    verified: bool = False
    confidence: float = 0.0


class ProofStructure(BaseModel):
    """Complete proof structure."""
    theorem: FormalStatement
    technique: ProofTechnique
    steps: List[ProofStep]
    conclusion: str
    status: ProofStatus = ProofStatus.PENDING
    verification_score: float = 0.0
    natural_language: Optional[str] = None
    formal_notation: Optional[str] = None
    counterexamples: List[str] = Field(default_factory=list)


class LogicalRule(BaseModel):
    """Logical inference rule."""
    name: str
    pattern: str
    application: str
    example: Optional[str] = None


class ProofValidator:
    """Validates formal proofs for logical consistency."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, LogicalRule]:
        """Initialize logical inference rules."""
        return {
            "modus_ponens": LogicalRule(
                name="Modus Ponens",
                pattern="P → Q, P ⊢ Q",
                application="If P implies Q and P is true, then Q is true"
            ),
            "modus_tollens": LogicalRule(
                name="Modus Tollens",
                pattern="P → Q, ¬Q ⊢ ¬P",
                application="If P implies Q and Q is false, then P is false"
            ),
            "universal_instantiation": LogicalRule(
                name="Universal Instantiation",
                pattern="∀x P(x) ⊢ P(a)",
                application="From universal statement, derive specific instance"
            ),
            "existential_generalization": LogicalRule(
                name="Existential Generalization",
                pattern="P(a) ⊢ ∃x P(x)",
                application="From specific instance, derive existential statement"
            ),
            "conjunction": LogicalRule(
                name="Conjunction",
                pattern="P, Q ⊢ P ∧ Q",
                application="Combine statements with AND"
            ),
            "disjunction": LogicalRule(
                name="Disjunction",
                pattern="P ⊢ P ∨ Q",
                application="Derive OR statement from single statement"
            ),
            "contradiction": LogicalRule(
                name="Contradiction",
                pattern="P, ¬P ⊢ ⊥",
                application="Derive contradiction from P and not P"
            ),
            "double_negation": LogicalRule(
                name="Double Negation",
                pattern="¬¬P ⊢ P",
                application="Eliminate double negation"
            )
        }
    
    def validate_step(
        self,
        step: ProofStep,
        previous_steps: List[ProofStep],
        context: FormalStatement
    ) -> Tuple[bool, str]:
        """
        Validate a single proof step.
        
        Args:
            step: The step to validate
            previous_steps: Previous steps in the proof
            context: The formal statement being proved
        
        Returns:
            Tuple of (is_valid, explanation)
        """
        # Check dependencies exist
        for dep in step.dependencies:
            if dep >= step.step_number or dep < 0:
                return False, f"Invalid dependency: step {dep}"
            if dep > 0 and dep > len(previous_steps):
                return False, f"Dependency on non-existent step {dep}"
        
        # Check if rule is valid
        if step.rule_applied and step.rule_applied not in self.rules:
            logger.warning(f"Unknown rule: {step.rule_applied}")
        
        # Basic logical consistency checks
        if "contradiction" in step.statement.lower() and "¬" not in step.statement:
            return False, "Contradiction claimed without negation"
        
        # Check circular reasoning
        if any(step.statement == prev.statement for prev in previous_steps):
            return False, "Circular reasoning detected"
        
        return True, "Step appears valid"
    
    def validate_proof(self, proof: ProofStructure) -> Tuple[bool, float, List[str]]:
        """
        Validate entire proof structure.
        
        Args:
            proof: The proof to validate
        
        Returns:
            Tuple of (is_valid, confidence_score, issues)
        """
        issues = []
        valid_steps = 0
        
        # Validate each step
        previous_steps = []
        for step in proof.steps:
            is_valid, explanation = self.validate_step(
                step,
                previous_steps,
                proof.theorem
            )
            if is_valid:
                valid_steps += 1
                step.verified = True
            else:
                issues.append(f"Step {step.step_number}: {explanation}")
                step.verified = False
            previous_steps.append(step)
        
        # Check if conclusion follows from steps
        if proof.conclusion not in str(proof.steps[-1].statement):
            issues.append("Conclusion doesn't follow from final step")
        
        # Check if goal is achieved
        if proof.theorem.goal not in proof.conclusion:
            issues.append("Proof doesn't achieve stated goal")
        
        # Calculate confidence score
        confidence = valid_steps / len(proof.steps) if proof.steps else 0
        
        # Adjust for technique appropriateness
        if proof.technique == ProofTechnique.INDUCTION:
            has_base = any("base" in s.statement.lower() for s in proof.steps)
            has_inductive = any("inductive" in s.statement.lower() for s in proof.steps)
            if not (has_base and has_inductive):
                confidence *= 0.5
                issues.append("Induction proof missing base case or inductive step")
        
        is_valid = len(issues) == 0
        return is_valid, confidence, issues


class ProofGenerator:
    """Generates formal proofs using AI."""
    
    def __init__(self):
        self.validator = ProofValidator()
    
    async def generate_proof(
        self,
        theorem: str,
        technique: Optional[ProofTechnique] = None,
        context: Optional[str] = None,
        examples: Optional[List[PromptExample]] = None
    ) -> ProofStructure:
        """
        Generate a formal proof for a theorem.
        
        Args:
            theorem: The theorem to prove
            technique: Preferred proof technique
            context: Additional context or constraints
            examples: Example proofs for few-shot learning
        
        Returns:
            Generated ProofStructure
        """
        # Determine best technique if not specified
        if not technique:
            technique = await self._suggest_technique(theorem, context)
        
        # Build proof generation prompt
        prompt_vars = {
            "theorem": theorem,
            "proof_technique": technique.value,
            "given_conditions": context or "Standard mathematical axioms"
        }
        
        if technique == ProofTechnique.INDUCTION:
            prompt_vars["base_case"] = "0 or minimal element"
        
        prompt = prompt_builder.build(
            "proof_assistant",
            prompt_vars,
            examples=examples
        )
        
        # Generate proof using orchestrator
        response = await orchestrator.execute(
            task="prove_theorem",
            prompt=prompt,
            mode=ExecutionMode.CASCADE,
            temperature=0.3,
            max_tokens=3072
        )
        
        # Parse response into proof structure
        proof = await self._parse_proof_response(response, theorem, technique)
        
        # Validate the proof
        is_valid, confidence, issues = self.validator.validate_proof(proof)
        proof.verification_score = confidence
        
        if not is_valid and proof.status != ProofStatus.FAILED:
            proof.status = ProofStatus.NEEDS_REVISION
            
            # Attempt to fix issues
            proof = await self._revise_proof(proof, issues)
        
        return proof
    
    async def _suggest_technique(
        self,
        theorem: str,
        context: Optional[str] = None
    ) -> ProofTechnique:
        """Suggest appropriate proof technique based on theorem."""
        prompt = f"""Analyze this theorem and suggest the best proof technique:

Theorem: {theorem}
Context: {context or 'General mathematical proof'}

Consider these techniques:
- Direct proof: Straightforward logical derivation
- Proof by contradiction: Assume negation leads to contradiction
- Proof by induction: For statements about natural numbers
- Proof by construction: Build explicit example
- Contrapositive: Prove ¬Q → ¬P instead of P → Q
- Exhaustion: Check all possible cases
- Diagonalization: For uncountability proofs
- Pumping lemma: For non-regular language proofs

Return only the technique name."""
        
        response = await orchestrator.execute(
            task="proof_technique_selection",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.2
        )
        
        # Extract technique from response
        technique_map = {
            "direct": ProofTechnique.DIRECT,
            "contradiction": ProofTechnique.CONTRADICTION,
            "induction": ProofTechnique.INDUCTION,
            "construction": ProofTechnique.CONSTRUCTION,
            "contrapositive": ProofTechnique.CONTRAPOSITIVE,
            "exhaustion": ProofTechnique.EXHAUSTION,
            "diagonalization": ProofTechnique.DIAGONALIZATION,
            "pumping": ProofTechnique.PUMPING_LEMMA
        }
        
        response_text = str(response[0].response).lower() if isinstance(response, list) else str(response.response).lower()
        
        for key, value in technique_map.items():
            if key in response_text:
                return value
        
        return ProofTechnique.DIRECT  # Default
    
    async def _parse_proof_response(
        self,
        response: Any,
        theorem: str,
        technique: ProofTechnique
    ) -> ProofStructure:
        """Parse AI response into structured proof."""
        # Extract response text
        if isinstance(response, list):
            response_text = response[0].response
        else:
            response_text = response.response if hasattr(response, 'response') else str(response)
        
        # Parse steps from response
        steps = []
        step_pattern = r'(\d+)\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(step_pattern, response_text, re.DOTALL)
        
        for i, (num, content) in enumerate(matches):
            # Extract justification if present
            just_match = re.search(r'\[(.+?)\]|\((.+?)\)', content)
            justification = just_match.group(1) if just_match else "Given"
            
            # Clean statement
            statement = re.sub(r'\[.+?\]|\(.+?\)', '', content).strip()
            
            steps.append(ProofStep(
                step_number=i + 1,
                statement=statement,
                justification=justification,
                confidence=0.8  # Default confidence
            ))
        
        # Extract conclusion
        conclusion_match = re.search(
            r'(conclusion|therefore|thus|hence):?\s*(.+)',
            response_text,
            re.IGNORECASE
        )
        conclusion = conclusion_match.group(2) if conclusion_match else "Proof complete"
        
        # Create formal statement
        formal_stmt = FormalStatement(
            statement=theorem,
            goal=theorem,
            assumptions=[]
        )
        
        return ProofStructure(
            theorem=formal_stmt,
            technique=technique,
            steps=steps,
            conclusion=conclusion,
            status=ProofStatus.COMPLETED,
            natural_language=response_text
        )
    
    async def _revise_proof(
        self,
        proof: ProofStructure,
        issues: List[str]
    ) -> ProofStructure:
        """Attempt to revise a proof to fix issues."""
        revision_prompt = f"""Revise this proof to fix the following issues:

Original Theorem: {proof.theorem.statement}
Technique: {proof.technique.value}

Issues found:
{chr(10).join(f'- {issue}' for issue in issues)}

Current proof steps:
{chr(10).join(f'{s.step_number}. {s.statement}' for s in proof.steps)}

Provide a corrected proof that addresses these issues."""
        
        response = await orchestrator.execute(
            task="proof_revision",
            prompt=revision_prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.4
        )
        
        # Parse revised proof
        revised_proof = await self._parse_proof_response(
            response,
            proof.theorem.statement,
            proof.technique
        )
        
        # Validate again
        is_valid, confidence, new_issues = self.validator.validate_proof(revised_proof)
        revised_proof.verification_score = confidence
        
        if len(new_issues) < len(issues):
            # Improvement made
            revised_proof.status = ProofStatus.COMPLETED
            return revised_proof
        else:
            # No improvement, return original with failed status
            proof.status = ProofStatus.FAILED
            return proof


class NaturalLanguageTranslator:
    """Translates between natural language and formal notation."""
    
    def __init__(self):
        self.notation_map = {
            "for all": "∀",
            "there exists": "∃",
            "implies": "→",
            "if and only if": "↔",
            "and": "∧",
            "or": "∨",
            "not": "¬",
            "element of": "∈",
            "subset": "⊆",
            "union": "∪",
            "intersection": "∩",
            "empty set": "∅"
        }
    
    async def to_formal(self, natural_text: str) -> str:
        """Convert natural language to formal notation."""
        prompt = f"""Convert this natural language statement to formal mathematical notation:

Natural language: {natural_text}

Use standard mathematical symbols:
- ∀ for "for all"
- ∃ for "there exists"
- → for "implies"
- ↔ for "if and only if"
- ∧ for "and"
- ∨ for "or"
- ¬ for "not"
- ∈ for "element of"
- ⊆ for "subset"

Return only the formal notation."""
        
        response = await orchestrator.execute(
            task="natural_to_formal",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.2
        )
        
        formal = response[0].response if isinstance(response, list) else response.response
        
        # Apply direct replacements as fallback
        for natural, symbol in self.notation_map.items():
            formal = formal.replace(natural, symbol)
        
        return formal
    
    async def to_natural(self, formal_text: str) -> str:
        """Convert formal notation to natural language."""
        prompt = f"""Convert this formal mathematical notation to clear natural language:

Formal notation: {formal_text}

Explain what this means in plain English, suitable for students.
Be precise but avoid jargon where possible."""
        
        response = await orchestrator.execute(
            task="formal_to_natural",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.5
        )
        
        return response[0].response if isinstance(response, list) else response.response


class InteractiveProofAssistant:
    """Interactive proof refinement assistant."""
    
    def __init__(self):
        self.generator = ProofGenerator()
        self.translator = NaturalLanguageTranslator()
        self.active_proofs: Dict[str, ProofStructure] = {}
    
    async def start_proof(
        self,
        theorem: str,
        session_id: str,
        technique: Optional[ProofTechnique] = None
    ) -> ProofStructure:
        """Start an interactive proof session."""
        proof = await self.generator.generate_proof(theorem, technique)
        self.active_proofs[session_id] = proof
        return proof
    
    async def get_hint(
        self,
        session_id: str,
        step_number: Optional[int] = None
    ) -> str:
        """Get a hint for the current or specific step."""
        if session_id not in self.active_proofs:
            return "No active proof session found"
        
        proof = self.active_proofs[session_id]
        
        if step_number:
            hint_prompt = f"""Provide a hint for step {step_number} in proving:
{proof.theorem.statement}

Current technique: {proof.technique.value}
Previous steps: {', '.join(s.statement for s in proof.steps[:step_number-1])}

Give a helpful hint without revealing the complete solution."""
        else:
            hint_prompt = f"""Provide a general hint for proving:
{proof.theorem.statement}

Technique being used: {proof.technique.value}

Give guidance on the next step without revealing the solution."""
        
        response = await orchestrator.execute(
            task="proof_hint",
            prompt=hint_prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.6
        )
        
        return response[0].response if isinstance(response, list) else response.response
    
    async def verify_step(
        self,
        session_id: str,
        step: ProofStep
    ) -> Tuple[bool, str]:
        """Verify a user-provided proof step."""
        if session_id not in self.active_proofs:
            return False, "No active proof session"
        
        proof = self.active_proofs[session_id]
        
        # Get previous steps
        previous_steps = [s for s in proof.steps if s.step_number < step.step_number]
        
        # Validate the step
        is_valid, explanation = proof.generator.validator.validate_step(
            step,
            previous_steps,
            proof.theorem
        )
        
        if is_valid:
            # Add to proof
            proof.steps.append(step)
            self.active_proofs[session_id] = proof
        
        return is_valid, explanation
    
    async def complete_proof(
        self,
        session_id: str
    ) -> ProofStructure:
        """Complete and finalize a proof."""
        if session_id not in self.active_proofs:
            raise ValueError("No active proof session")
        
        proof = self.active_proofs[session_id]
        
        # Validate complete proof
        is_valid, confidence, issues = proof.generator.validator.validate_proof(proof)
        proof.verification_score = confidence
        proof.status = ProofStatus.VERIFIED if is_valid else ProofStatus.NEEDS_REVISION
        
        # Convert to formal notation
        proof.formal_notation = await self.translator.to_formal(proof.natural_language or "")
        
        return proof


# Global instances
proof_generator = ProofGenerator()
proof_assistant = InteractiveProofAssistant()
nl_translator = NaturalLanguageTranslator()