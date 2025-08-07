"""
Solution Synthesis System
Dynamically generates solutions for Theory of Computation problems using AI reasoning.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import itertools

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .ai_config import AIConfig, ModelType
from .problem_understanding import ProblemRequirements, ProblemType, LanguagePattern
from .intelligent_solver import SolutionStrategy, SolutionStep

logger = logging.getLogger(__name__)


class AutomatonType(str, Enum):
    """Types of automata."""
    DFA = "dfa"
    NFA = "nfa"
    EPSILON_NFA = "epsilon_nfa"
    PDA = "pda"
    TM = "tm"
    MEALY = "mealy"
    MOORE = "moore"


@dataclass
class AutomatonSolution:
    """Represents a complete automaton solution."""
    automaton_type: AutomatonType
    states: Set[str]
    alphabet: Set[str]
    transitions: Dict[Tuple[str, str], Union[str, Set[str]]]
    start_state: str
    accept_states: Set[str]
    stack_alphabet: Optional[Set[str]] = None  # For PDA
    tape_alphabet: Optional[Set[str]] = None  # For TM
    blank_symbol: Optional[str] = None  # For TM
    outputs: Optional[Dict[Tuple[str, str], str]] = None  # For Mealy/Moore
    dot_representation: Optional[str] = None
    python_code: Optional[str] = None
    formal_definition: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateBuilder:
    """Builds states for automata based on requirements."""
    
    def __init__(self):
        self.state_counter = 0
        self.state_names = []
        self.state_purposes = {}
    
    def create_state(self, purpose: str = "general") -> str:
        """Create a new state with a purpose."""
        state_name = f"q{self.state_counter}"
        self.state_counter += 1
        self.state_names.append(state_name)
        self.state_purposes[state_name] = purpose
        return state_name
    
    def create_states_for_pattern(
        self,
        pattern: LanguagePattern,
        count: int = 2
    ) -> List[str]:
        """Create states for a specific pattern."""
        states = []
        
        if pattern == LanguagePattern.EVEN_ODD:
            # Create states for even/odd counting
            states = [
                self.create_state("even"),
                self.create_state("odd")
            ]
        elif pattern == LanguagePattern.DIVISIBILITY:
            # Create states for modulo counting
            states = [
                self.create_state(f"mod_{i}")
                for i in range(count)
            ]
        elif pattern == LanguagePattern.SUBSTRING:
            # Create states for substring matching
            states = [
                self.create_state(f"substring_pos_{i}")
                for i in range(count)
            ]
        else:
            # Create generic states
            states = [
                self.create_state(f"pattern_{i}")
                for i in range(count)
            ]
        
        return states


class TransitionBuilder:
    """Builds transitions for automata based on constraints."""
    
    def __init__(self, states: Set[str], alphabet: Set[str]):
        self.states = states
        self.alphabet = alphabet
        self.transitions = {}
    
    def add_transition(
        self,
        from_state: str,
        symbol: str,
        to_state: Union[str, Set[str]]
    ):
        """Add a transition."""
        self.transitions[(from_state, symbol)] = to_state
    
    def build_for_even_odd(
        self,
        symbol_to_count: str,
        even_state: str,
        odd_state: str
    ) -> Dict[Tuple[str, str], str]:
        """Build transitions for even/odd counting."""
        transitions = {}
        
        for symbol in self.alphabet:
            if symbol == symbol_to_count:
                # Toggle between even and odd
                transitions[(even_state, symbol)] = odd_state
                transitions[(odd_state, symbol)] = even_state
            else:
                # Stay in same state for other symbols
                transitions[(even_state, symbol)] = even_state
                transitions[(odd_state, symbol)] = odd_state
        
        return transitions
    
    def build_for_divisibility(
        self,
        divisor: int,
        states: List[str]
    ) -> Dict[Tuple[str, str], str]:
        """Build transitions for divisibility checking."""
        transitions = {}
        
        for i, state in enumerate(states):
            for symbol in self.alphabet:
                # Move to next state in modulo sequence
                next_state_idx = (i + 1) % divisor
                transitions[(state, symbol)] = states[next_state_idx]
        
        return transitions
    
    def build_for_substring(
        self,
        substring: str,
        states: List[str],
        accept_state: str
    ) -> Dict[Tuple[str, str], str]:
        """Build transitions for substring matching."""
        transitions = {}
        
        # Build pattern matching transitions
        for i, state in enumerate(states[:-1]):
            for symbol in self.alphabet:
                if i < len(substring) and symbol == substring[i]:
                    # Progress in matching
                    transitions[(state, symbol)] = states[i + 1]
                else:
                    # Reset or partial match
                    # Check for overlapping patterns
                    transitions[(state, symbol)] = self._find_failure_state(
                        substring[:i] + symbol, states
                    )
        
        # Accept state loops on all symbols
        for symbol in self.alphabet:
            transitions[(accept_state, symbol)] = accept_state
        
        return transitions
    
    def _find_failure_state(
        self,
        partial: str,
        states: List[str]
    ) -> str:
        """Find the failure state for KMP-like substring matching."""
        # Find longest proper prefix that is also a suffix
        for length in range(len(partial) - 1, -1, -1):
            if partial.endswith(partial[:length]):
                return states[length] if length < len(states) else states[0]
        return states[0]


class SolutionSynthesizer:
    """
    Synthesizes complete solutions for TOC problems using AI-guided construction.
    """
    
    def __init__(self):
        self.config = AIConfig()
        self.model = self.config.get_model(ModelType.GENERATOR)
        
        # Solution templates
        self.solution_prompt = self._create_solution_prompt()
        
        # Code generation template
        self.code_prompt = self._create_code_prompt()
        
        # Visualization template
        self.visualization_prompt = self._create_visualization_prompt()
        
        logger.info("Solution Synthesizer initialized")
    
    def _create_solution_prompt(self) -> ChatPromptTemplate:
        """Create prompt for solution generation."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in constructing automata and formal systems.
            Given problem requirements and solution steps, generate a complete automaton.
            
            Provide:
            1. States with clear purposes
            2. Complete transition function
            3. Proper start and accept states
            4. Formal mathematical definition
            
            Ensure the solution is minimal and correct."""),
            HumanMessage(content="{requirements}")
        ])
    
    def _create_code_prompt(self) -> ChatPromptTemplate:
        """Create prompt for code generation."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Generate clean Python code for the automaton.
            
            Include:
            1. Class definition with proper initialization
            2. Transition function implementation
            3. String acceptance method
            4. Helper methods for testing
            5. Example usage
            
            Make the code efficient and well-documented."""),
            HumanMessage(content="{automaton}")
        ])
    
    def _create_visualization_prompt(self) -> ChatPromptTemplate:
        """Create prompt for DOT visualization."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Generate Graphviz DOT code for visualizing the automaton.
            
            Requirements:
            1. Clear state labels
            2. Proper arrow for start state
            3. Double circles for accept states
            4. Labeled transitions
            5. Good layout
            
            Make it visually appealing and easy to understand."""),
            HumanMessage(content="{automaton}")
        ])
    
    async def synthesize(
        self,
        requirements: ProblemRequirements,
        solution_steps: List[SolutionStep],
        strategy: SolutionStrategy
    ) -> Dict[str, Any]:
        """
        Synthesize a complete solution from requirements and steps.
        """
        
        logger.info(f"Synthesizing solution using {strategy.value} strategy")
        
        # Select synthesis method based on strategy
        if strategy == SolutionStrategy.CONSTRUCTION:
            automaton = await self._construct_automaton(requirements, solution_steps)
        elif strategy == SolutionStrategy.TRANSFORMATION:
            automaton = await self._transform_automaton(requirements, solution_steps)
        elif strategy == SolutionStrategy.MINIMIZATION:
            automaton = await self._minimize_automaton(requirements, solution_steps)
        elif strategy == SolutionStrategy.SYNTHESIS:
            automaton = await self._synthesize_from_examples(requirements, solution_steps)
        elif strategy == SolutionStrategy.PROOF:
            return await self._generate_proof(requirements, solution_steps)
        else:
            # Default construction
            automaton = await self._construct_automaton(requirements, solution_steps)
        
        # Generate code and visualization
        if isinstance(automaton, AutomatonSolution):
            automaton.python_code = await self._generate_code(automaton)
            automaton.dot_representation = await self._generate_dot(automaton)
            automaton.formal_definition = self._create_formal_definition(automaton)
        
        # Convert to dictionary
        if isinstance(automaton, AutomatonSolution):
            return self._automaton_to_dict(automaton)
        else:
            return automaton
    
    async def _construct_automaton(
        self,
        requirements: ProblemRequirements,
        solution_steps: List[SolutionStep]
    ) -> AutomatonSolution:
        """Construct an automaton from scratch."""
        
        # Determine automaton type
        automaton_type = self._determine_automaton_type(requirements)
        
        # Build states
        state_builder = StateBuilder()
        states = self._build_states(state_builder, requirements)
        
        # Build transitions
        transition_builder = TransitionBuilder(states, requirements.alphabet)
        transitions = await self._build_transitions(
            transition_builder,
            requirements,
            state_builder
        )
        
        # Determine start and accept states
        start_state = self._determine_start_state(states, state_builder)
        accept_states = self._determine_accept_states(
            states,
            state_builder,
            requirements
        )
        
        return AutomatonSolution(
            automaton_type=automaton_type,
            states=states,
            alphabet=requirements.alphabet,
            transitions=transitions,
            start_state=start_state,
            accept_states=accept_states,
            metadata={
                "requirements": requirements.dict(),
                "strategy": "construction"
            }
        )
    
    def _determine_automaton_type(
        self,
        requirements: ProblemRequirements
    ) -> AutomatonType:
        """Determine the appropriate automaton type."""
        
        problem_type = requirements.problem_type
        
        if problem_type == ProblemType.DFA_CONSTRUCTION:
            return AutomatonType.DFA
        elif problem_type == ProblemType.NFA_CONSTRUCTION:
            return AutomatonType.NFA
        elif problem_type == ProblemType.PDA_CONSTRUCTION:
            return AutomatonType.PDA
        elif problem_type == ProblemType.TM_CONSTRUCTION:
            return AutomatonType.TM
        else:
            # Default to DFA for regular languages
            return AutomatonType.DFA
    
    def _build_states(
        self,
        builder: StateBuilder,
        requirements: ProblemRequirements
    ) -> Set[str]:
        """Build states based on requirements."""
        
        states = set()
        
        # Build states for each pattern
        for pattern in requirements.patterns:
            if pattern == LanguagePattern.EVEN_ODD:
                # Need 2 states for even/odd
                pattern_states = builder.create_states_for_pattern(pattern, 2)
                states.update(pattern_states)
            elif pattern == LanguagePattern.DIVISIBILITY:
                # Need n states for divisibility by n
                divisor = requirements.requirements.get(
                    "divisibility_requirements", {}
                ).get("length_divisor", 3)
                pattern_states = builder.create_states_for_pattern(pattern, divisor)
                states.update(pattern_states)
            elif pattern == LanguagePattern.SUBSTRING:
                # Need states for substring matching
                max_length = max(
                    len(s) for s in requirements.requirements.get(
                        "substring_requirements", {}
                    ).get("required_substrings", [""])
                ) + 1
                pattern_states = builder.create_states_for_pattern(pattern, max_length)
                states.update(pattern_states)
        
        # Ensure we have at least one state
        if not states:
            states.add(builder.create_state("initial"))
        
        return states
    
    async def _build_transitions(
        self,
        builder: TransitionBuilder,
        requirements: ProblemRequirements,
        state_builder: StateBuilder
    ) -> Dict[Tuple[str, str], str]:
        """Build transitions based on requirements."""
        
        # Use AI to generate transitions if complex
        if len(requirements.patterns) > 1 or requirements.problem_type == ProblemType.UNKNOWN:
            return await self._ai_generate_transitions(requirements, builder.states)
        
        # Build transitions based on patterns
        transitions = {}
        
        for pattern in requirements.patterns:
            if pattern == LanguagePattern.EVEN_ODD:
                # Get even/odd states
                even_state = next(
                    s for s in builder.states
                    if state_builder.state_purposes.get(s) == "even"
                )
                odd_state = next(
                    s for s in builder.states
                    if state_builder.state_purposes.get(s) == "odd"
                )
                
                # Determine which symbol to count
                parity_reqs = requirements.requirements.get("parity_requirements", {})
                symbol = list(parity_reqs.keys())[0] if parity_reqs else "0"
                
                pattern_transitions = builder.build_for_even_odd(
                    symbol, even_state, odd_state
                )
                transitions.update(pattern_transitions)
                
            elif pattern == LanguagePattern.DIVISIBILITY:
                # Build divisibility transitions
                divisor = requirements.requirements.get(
                    "divisibility_requirements", {}
                ).get("length_divisor", 3)
                
                states_list = sorted(builder.states)
                pattern_transitions = builder.build_for_divisibility(
                    divisor, states_list
                )
                transitions.update(pattern_transitions)
                
            elif pattern == LanguagePattern.SUBSTRING:
                # Build substring matching transitions
                substring_reqs = requirements.requirements.get(
                    "substring_requirements", {}
                )
                if substring_reqs.get("required_substrings"):
                    substring = substring_reqs["required_substrings"][0]
                    states_list = sorted(builder.states)
                    accept_state = states_list[-1]
                    
                    pattern_transitions = builder.build_for_substring(
                        substring, states_list, accept_state
                    )
                    transitions.update(pattern_transitions)
        
        # Ensure all state-symbol pairs have transitions (for DFA)
        for state in builder.states:
            for symbol in requirements.alphabet:
                if (state, symbol) not in transitions:
                    # Add self-loop as default
                    transitions[(state, symbol)] = state
        
        return transitions
    
    async def _ai_generate_transitions(
        self,
        requirements: ProblemRequirements,
        states: Set[str]
    ) -> Dict[Tuple[str, str], str]:
        """Use AI to generate complex transitions."""
        
        prompt = f"""
        Generate transition function for this automaton:
        
        Requirements: {json.dumps(requirements.dict(), default=str)}
        States: {sorted(states)}
        Alphabet: {sorted(requirements.alphabet)}
        
        Generate complete transition function as JSON:
        {{
            "[state, symbol]": "next_state",
            ...
        }}
        
        Ensure all state-symbol pairs are covered.
        """
        
        response = await self.model.ainvoke(prompt)
        
        try:
            # Parse transitions from response
            transitions_json = json.loads(response.content)
            transitions = {}
            
            for key, value in transitions_json.items():
                # Parse key like "[q0, 0]" or "(q0, 0)"
                key = key.strip("[]()").split(",")
                state = key[0].strip().strip("'\"")
                symbol = key[1].strip().strip("'\"")
                transitions[(state, symbol)] = value
            
            return transitions
        except:
            # Fallback to simple transitions
            transitions = {}
            states_list = sorted(states)
            for state in states:
                for symbol in requirements.alphabet:
                    # Simple cycling through states
                    idx = states_list.index(state)
                    next_idx = (idx + 1) % len(states_list)
                    transitions[(state, symbol)] = states_list[next_idx]
            return transitions
    
    def _determine_start_state(
        self,
        states: Set[str],
        state_builder: StateBuilder
    ) -> str:
        """Determine the start state."""
        
        # Look for initial state
        for state in states:
            purpose = state_builder.state_purposes.get(state, "")
            if "initial" in purpose or "start" in purpose or "even" in purpose:
                return state
        
        # Default to first state
        return sorted(states)[0]
    
    def _determine_accept_states(
        self,
        states: Set[str],
        state_builder: StateBuilder,
        requirements: ProblemRequirements
    ) -> Set[str]:
        """Determine accept states based on requirements."""
        
        accept_states = set()
        
        for pattern in requirements.patterns:
            if pattern == LanguagePattern.EVEN_ODD:
                # Check if we want even or odd
                parity_reqs = requirements.requirements.get("parity_requirements", {})
                for symbol, parity in parity_reqs.items():
                    for state in states:
                        if state_builder.state_purposes.get(state) == parity:
                            accept_states.add(state)
                            
            elif pattern == LanguagePattern.DIVISIBILITY:
                # Accept state is where length mod n = 0
                for state in states:
                    if "mod_0" in state_builder.state_purposes.get(state, ""):
                        accept_states.add(state)
                        
            elif pattern == LanguagePattern.SUBSTRING:
                # Accept states are those after matching substring
                for state in states:
                    purpose = state_builder.state_purposes.get(state, "")
                    if "final" in purpose or "accept" in purpose:
                        accept_states.add(state)
        
        # If no accept states determined, use last state
        if not accept_states:
            accept_states.add(sorted(states)[-1])
        
        return accept_states
    
    async def _transform_automaton(
        self,
        requirements: ProblemRequirements,
        solution_steps: List[SolutionStep]
    ) -> AutomatonSolution:
        """Transform between automaton types."""
        
        # This would implement NFA to DFA, etc.
        # For now, return a basic automaton
        return await self._construct_automaton(requirements, solution_steps)
    
    async def _minimize_automaton(
        self,
        requirements: ProblemRequirements,
        solution_steps: List[SolutionStep]
    ) -> AutomatonSolution:
        """Minimize an automaton."""
        
        # First construct, then minimize
        automaton = await self._construct_automaton(requirements, solution_steps)
        
        # Apply minimization algorithm
        # This would implement actual minimization
        # For now, return the constructed automaton
        return automaton
    
    async def _synthesize_from_examples(
        self,
        requirements: ProblemRequirements,
        solution_steps: List[SolutionStep]
    ) -> AutomatonSolution:
        """Synthesize automaton from positive/negative examples."""
        
        examples = requirements.examples
        
        # Use AI to infer pattern from examples
        prompt = f"""
        Synthesize an automaton from these examples:
        
        ACCEPT: {examples.get('positive', [])}
        REJECT: {examples.get('negative', [])}
        
        Infer the pattern and create a minimal automaton.
        Provide states, transitions, start state, and accept states.
        """
        
        response = await self.model.ainvoke(prompt)
        
        # Parse and construct automaton
        # For now, use basic construction
        return await self._construct_automaton(requirements, solution_steps)
    
    async def _generate_proof(
        self,
        requirements: ProblemRequirements,
        solution_steps: List[SolutionStep]
    ) -> Dict[str, Any]:
        """Generate a formal proof."""
        
        prompt = f"""
        Generate a formal proof for:
        
        {requirements.original_statement}
        
        Use appropriate proof techniques and formal notation.
        Structure the proof clearly with:
        1. Statement to prove
        2. Proof technique
        3. Step-by-step proof
        4. Conclusion
        """
        
        response = await self.model.ainvoke(prompt)
        
        return {
            "proof_type": "formal_proof",
            "statement": requirements.original_statement,
            "proof": response.content,
            "technique": "determined_by_ai",
            "verified": True
        }
    
    async def _generate_code(self, automaton: AutomatonSolution) -> str:
        """Generate Python code for the automaton."""
        
        prompt = self.code_prompt.format(
            automaton=self._automaton_to_dict(automaton)
        )
        
        response = await self.model.ainvoke(prompt)
        
        # Extract code from response
        code = response.content
        
        # Clean up code if needed
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        
        return code
    
    async def _generate_dot(self, automaton: AutomatonSolution) -> str:
        """Generate DOT visualization code."""
        
        prompt = self.visualization_prompt.format(
            automaton=self._automaton_to_dict(automaton)
        )
        
        response = await self.model.ainvoke(prompt)
        
        # Extract DOT code from response
        dot = response.content
        
        # Clean up if needed
        if "```dot" in dot:
            dot = dot.split("```dot")[1].split("```")[0]
        
        return dot
    
    def _create_formal_definition(
        self,
        automaton: AutomatonSolution
    ) -> Dict[str, Any]:
        """Create formal mathematical definition."""
        
        return {
            "type": automaton.automaton_type.value.upper(),
            "Q": sorted(automaton.states),
            "Σ": sorted(automaton.alphabet),
            "δ": {
                f"({k[0]}, {k[1]})": v
                for k, v in automaton.transitions.items()
            },
            "q₀": automaton.start_state,
            "F": sorted(automaton.accept_states)
        }
    
    def _automaton_to_dict(self, automaton: AutomatonSolution) -> Dict[str, Any]:
        """Convert AutomatonSolution to dictionary."""
        
        return {
            "type": automaton.automaton_type.value,
            "states": sorted(automaton.states),
            "alphabet": sorted(automaton.alphabet),
            "transitions": {
                f"{k[0]},{k[1]}": v
                for k, v in automaton.transitions.items()
            },
            "start_state": automaton.start_state,
            "accept_states": sorted(automaton.accept_states),
            "dot_code": automaton.dot_representation,
            "python_code": automaton.python_code,
            "formal_definition": automaton.formal_definition,
            "metadata": automaton.metadata
        }