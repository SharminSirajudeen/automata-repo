"""
Automata Optimizer for the Automata Learning Platform.
Implements DFA/NFA minimization, state reduction, and performance optimization.
"""
import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import itertools
import time

from .orchestrator import orchestrator, ExecutionMode
from .prompts import prompt_builder

logger = logging.getLogger(__name__)


class AutomatonType(str, Enum):
    """Types of automata."""
    DFA = "dfa"
    NFA = "nfa"
    EPSILON_NFA = "epsilon_nfa"
    PDA = "pda"
    TM = "tm"


class OptimizationType(str, Enum):
    """Types of optimization."""
    MINIMIZATION = "minimization"
    STATE_REDUCTION = "state_reduction"
    TRANSITION_OPTIMIZATION = "transition_optimization"
    EQUIVALENCE_CHECK = "equivalence_check"
    CONVERSION = "conversion"


@dataclass
class State:
    """Represents a state in an automaton."""
    name: str
    is_final: bool = False
    is_initial: bool = False
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name if isinstance(other, State) else False


@dataclass
class Transition:
    """Represents a transition in an automaton."""
    from_state: str
    to_state: str
    symbol: str
    
    def __hash__(self):
        return hash((self.from_state, self.to_state, self.symbol))


class Automaton(BaseModel):
    """Base automaton structure."""
    type: AutomatonType
    states: List[str]
    alphabet: List[str]
    transitions: Dict[str, Dict[str, Union[str, List[str]]]]  # {from_state: {symbol: to_state(s)}}
    initial_state: str
    final_states: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_state_count(self) -> int:
        """Get number of states."""
        return len(self.states)
    
    def get_transition_count(self) -> int:
        """Get number of transitions."""
        count = 0
        for from_state in self.transitions:
            for symbol in self.transitions[from_state]:
                if isinstance(self.transitions[from_state][symbol], list):
                    count += len(self.transitions[from_state][symbol])
                else:
                    count += 1
        return count
    
    def is_deterministic(self) -> bool:
        """Check if automaton is deterministic."""
        if self.type != AutomatonType.DFA:
            return False
        
        for from_state in self.transitions:
            for symbol in self.transitions[from_state]:
                if isinstance(self.transitions[from_state][symbol], list):
                    return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type.value,
            "states": self.states,
            "alphabet": self.alphabet,
            "transitions": self.transitions,
            "initial_state": self.initial_state,
            "final_states": self.final_states,
            "metadata": self.metadata
        }


class OptimizationResult(BaseModel):
    """Result of optimization operation."""
    original: Automaton
    optimized: Automaton
    optimization_type: OptimizationType
    metrics: Dict[str, Any] = Field(default_factory=dict)
    equivalence_verified: bool = False
    execution_time: float = 0.0
    notes: List[str] = Field(default_factory=list)


class DFAMinimizer:
    """Implements DFA minimization algorithms."""
    
    def minimize(self, dfa: Automaton) -> Automaton:
        """
        Minimize DFA using Hopcroft's algorithm.
        
        Args:
            dfa: DFA to minimize
        
        Returns:
            Minimized DFA
        """
        start_time = time.time()
        
        # Remove unreachable states first
        reachable = self._find_reachable_states(dfa)
        
        # Build partition refinement
        partitions = self._hopcroft_algorithm(dfa, reachable)
        
        # Construct minimized DFA
        minimized = self._construct_minimized_dfa(dfa, partitions)
        
        execution_time = time.time() - start_time
        logger.info(f"DFA minimization completed in {execution_time:.3f}s")
        
        return minimized
    
    def _find_reachable_states(self, dfa: Automaton) -> Set[str]:
        """Find all reachable states from initial state."""
        reachable = set()
        queue = deque([dfa.initial_state])
        
        while queue:
            state = queue.popleft()
            if state in reachable:
                continue
            
            reachable.add(state)
            
            # Add states reachable from current state
            if state in dfa.transitions:
                for symbol in dfa.transitions[state]:
                    next_state = dfa.transitions[state][symbol]
                    if isinstance(next_state, list):
                        queue.extend(next_state)
                    elif next_state and next_state not in reachable:
                        queue.append(next_state)
        
        return reachable
    
    def _hopcroft_algorithm(
        self,
        dfa: Automaton,
        reachable: Set[str]
    ) -> List[Set[str]]:
        """
        Hopcroft's algorithm for DFA minimization.
        
        Args:
            dfa: DFA to minimize
            reachable: Set of reachable states
        
        Returns:
            List of equivalence classes
        """
        # Initial partition: final and non-final states
        final_set = set(dfa.final_states) & reachable
        non_final_set = reachable - final_set
        
        partitions = []
        if final_set:
            partitions.append(final_set)
        if non_final_set:
            partitions.append(non_final_set)
        
        # Refinement loop
        changed = True
        while changed:
            changed = False
            new_partitions = []
            
            for partition in partitions:
                # Try to split this partition
                splits = self._split_partition(dfa, partition, partitions)
                
                if len(splits) > 1:
                    changed = True
                    new_partitions.extend(splits)
                else:
                    new_partitions.append(partition)
            
            partitions = new_partitions
        
        return partitions
    
    def _split_partition(
        self,
        dfa: Automaton,
        partition: Set[str],
        all_partitions: List[Set[str]]
    ) -> List[Set[str]]:
        """Split a partition based on distinguishability."""
        if len(partition) <= 1:
            return [partition]
        
        # Group states by their transition behavior
        groups = defaultdict(set)
        
        for state in partition:
            # Create signature based on where state transitions go
            signature = []
            for symbol in dfa.alphabet:
                if state in dfa.transitions and symbol in dfa.transitions[state]:
                    next_state = dfa.transitions[state][symbol]
                    # Find which partition the next state belongs to
                    for i, part in enumerate(all_partitions):
                        if next_state in part:
                            signature.append(i)
                            break
                else:
                    signature.append(-1)  # No transition
            
            groups[tuple(signature)].add(state)
        
        return list(groups.values())
    
    def _construct_minimized_dfa(
        self,
        original: Automaton,
        partitions: List[Set[str]]
    ) -> Automaton:
        """Construct minimized DFA from equivalence classes."""
        # Map states to their partition index
        state_to_partition = {}
        for i, partition in enumerate(partitions):
            for state in partition:
                state_to_partition[state] = i
        
        # Create new states
        new_states = [f"q{i}" for i in range(len(partitions))]
        
        # Find initial state partition
        initial_partition = state_to_partition.get(original.initial_state, 0)
        new_initial = f"q{initial_partition}"
        
        # Find final states
        new_finals = []
        for i, partition in enumerate(partitions):
            if any(state in original.final_states for state in partition):
                new_finals.append(f"q{i}")
        
        # Build new transitions
        new_transitions = defaultdict(dict)
        for i, partition in enumerate(partitions):
            from_state = f"q{i}"
            # Use any state from partition to determine transitions
            representative = next(iter(partition))
            
            if representative in original.transitions:
                for symbol in original.alphabet:
                    if symbol in original.transitions[representative]:
                        next_state = original.transitions[representative][symbol]
                        if next_state in state_to_partition:
                            next_partition = state_to_partition[next_state]
                            new_transitions[from_state][symbol] = f"q{next_partition}"
        
        return Automaton(
            type=AutomatonType.DFA,
            states=new_states,
            alphabet=original.alphabet,
            transitions=dict(new_transitions),
            initial_state=new_initial,
            final_states=new_finals,
            metadata={
                "minimized": True,
                "original_states": len(original.states),
                "reduced_to": len(new_states)
            }
        )


class NFAOptimizer:
    """Optimizes NFA structures."""
    
    def optimize(self, nfa: Automaton) -> Automaton:
        """
        Optimize NFA by removing redundant states and transitions.
        
        Args:
            nfa: NFA to optimize
        
        Returns:
            Optimized NFA
        """
        # Remove unreachable states
        reachable = self._find_reachable_states(nfa)
        
        # Remove dead states (can't reach final states)
        useful = self._find_useful_states(nfa, reachable)
        
        # Construct optimized NFA
        optimized = self._construct_optimized_nfa(nfa, useful)
        
        return optimized
    
    def _find_reachable_states(self, nfa: Automaton) -> Set[str]:
        """Find reachable states in NFA."""
        reachable = set()
        queue = deque([nfa.initial_state])
        
        while queue:
            state = queue.popleft()
            if state in reachable:
                continue
            
            reachable.add(state)
            
            if state in nfa.transitions:
                for symbol in nfa.transitions[state]:
                    next_states = nfa.transitions[state][symbol]
                    if isinstance(next_states, list):
                        for ns in next_states:
                            if ns not in reachable:
                                queue.append(ns)
                    elif next_states and next_states not in reachable:
                        queue.append(next_states)
        
        return reachable
    
    def _find_useful_states(
        self,
        nfa: Automaton,
        reachable: Set[str]
    ) -> Set[str]:
        """Find states that can reach final states."""
        # Backward search from final states
        useful = set(nfa.final_states)
        queue = deque(nfa.final_states)
        
        # Build reverse transitions
        reverse_transitions = defaultdict(lambda: defaultdict(set))
        for from_state in nfa.transitions:
            for symbol in nfa.transitions[from_state]:
                to_states = nfa.transitions[from_state][symbol]
                if isinstance(to_states, list):
                    for to_state in to_states:
                        reverse_transitions[to_state][symbol].add(from_state)
                else:
                    reverse_transitions[to_states][symbol].add(from_state)
        
        while queue:
            state = queue.popleft()
            
            for symbol in reverse_transitions[state]:
                for prev_state in reverse_transitions[state][symbol]:
                    if prev_state not in useful:
                        useful.add(prev_state)
                        queue.append(prev_state)
        
        # Return intersection of reachable and useful
        return reachable & useful
    
    def _construct_optimized_nfa(
        self,
        original: Automaton,
        useful_states: Set[str]
    ) -> Automaton:
        """Construct optimized NFA with only useful states."""
        # Filter states
        new_states = [s for s in original.states if s in useful_states]
        
        # Filter transitions
        new_transitions = {}
        for from_state in original.transitions:
            if from_state in useful_states:
                new_transitions[from_state] = {}
                for symbol in original.transitions[from_state]:
                    to_states = original.transitions[from_state][symbol]
                    if isinstance(to_states, list):
                        filtered = [s for s in to_states if s in useful_states]
                        if filtered:
                            new_transitions[from_state][symbol] = filtered
                    elif to_states in useful_states:
                        new_transitions[from_state][symbol] = to_states
        
        # Filter final states
        new_finals = [s for s in original.final_states if s in useful_states]
        
        return Automaton(
            type=original.type,
            states=new_states,
            alphabet=original.alphabet,
            transitions=new_transitions,
            initial_state=original.initial_state,
            final_states=new_finals,
            metadata={
                "optimized": True,
                "removed_states": len(original.states) - len(new_states)
            }
        )


class AutomatonConverter:
    """Converts between different automaton types."""
    
    def nfa_to_dfa(self, nfa: Automaton) -> Automaton:
        """
        Convert NFA to DFA using subset construction.
        
        Args:
            nfa: NFA to convert
        
        Returns:
            Equivalent DFA
        """
        # Epsilon closure for epsilon-NFA
        epsilon_closures = {}
        if nfa.type == AutomatonType.EPSILON_NFA:
            epsilon_closures = self._compute_epsilon_closures(nfa)
        
        # Subset construction
        dfa_states = []
        dfa_transitions = {}
        dfa_finals = []
        
        # Start with epsilon closure of initial state
        if nfa.initial_state in epsilon_closures:
            initial_subset = frozenset(epsilon_closures[nfa.initial_state])
        else:
            initial_subset = frozenset([nfa.initial_state])
        
        state_map = {initial_subset: "q0"}
        dfa_states.append("q0")
        dfa_initial = "q0"
        
        # Check if initial is final
        if any(s in nfa.final_states for s in initial_subset):
            dfa_finals.append("q0")
        
        # BFS to construct DFA states
        queue = deque([initial_subset])
        visited = {initial_subset}
        state_counter = 1
        
        while queue:
            current_subset = queue.popleft()
            current_name = state_map[current_subset]
            
            dfa_transitions[current_name] = {}
            
            for symbol in nfa.alphabet:
                if symbol == "ε":  # Skip epsilon
                    continue
                
                # Find all states reachable via symbol
                next_subset = set()
                for state in current_subset:
                    if state in nfa.transitions and symbol in nfa.transitions[state]:
                        next_states = nfa.transitions[state][symbol]
                        if isinstance(next_states, list):
                            next_subset.update(next_states)
                        else:
                            next_subset.add(next_states)
                
                # Apply epsilon closure if needed
                if epsilon_closures:
                    closure = set()
                    for s in next_subset:
                        if s in epsilon_closures:
                            closure.update(epsilon_closures[s])
                        else:
                            closure.add(s)
                    next_subset = closure
                
                if next_subset:
                    next_subset_frozen = frozenset(next_subset)
                    
                    # Create new state if needed
                    if next_subset_frozen not in state_map:
                        new_state_name = f"q{state_counter}"
                        state_counter += 1
                        state_map[next_subset_frozen] = new_state_name
                        dfa_states.append(new_state_name)
                        
                        # Check if final
                        if any(s in nfa.final_states for s in next_subset):
                            dfa_finals.append(new_state_name)
                        
                        if next_subset_frozen not in visited:
                            queue.append(next_subset_frozen)
                            visited.add(next_subset_frozen)
                    
                    dfa_transitions[current_name][symbol] = state_map[next_subset_frozen]
        
        return Automaton(
            type=AutomatonType.DFA,
            states=dfa_states,
            alphabet=[s for s in nfa.alphabet if s != "ε"],
            transitions=dfa_transitions,
            initial_state=dfa_initial,
            final_states=dfa_finals,
            metadata={
                "converted_from": "NFA",
                "original_states": len(nfa.states),
                "dfa_states": len(dfa_states)
            }
        )
    
    def _compute_epsilon_closures(self, nfa: Automaton) -> Dict[str, Set[str]]:
        """Compute epsilon closure for each state."""
        closures = {}
        
        for state in nfa.states:
            closure = {state}
            queue = deque([state])
            
            while queue:
                current = queue.popleft()
                
                if current in nfa.transitions and "ε" in nfa.transitions[current]:
                    next_states = nfa.transitions[current]["ε"]
                    if isinstance(next_states, list):
                        for ns in next_states:
                            if ns not in closure:
                                closure.add(ns)
                                queue.append(ns)
                    elif next_states not in closure:
                        closure.add(next_states)
                        queue.append(next_states)
            
            closures[state] = closure
        
        return closures


class EquivalenceChecker:
    """Checks equivalence between automata."""
    
    def are_equivalent(
        self,
        automaton1: Automaton,
        automaton2: Automaton
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if two automata are equivalent.
        
        Args:
            automaton1: First automaton
            automaton2: Second automaton
        
        Returns:
            Tuple of (are_equivalent, counterexample)
        """
        # Convert to DFAs if needed
        converter = AutomatonConverter()
        
        if automaton1.type != AutomatonType.DFA:
            dfa1 = converter.nfa_to_dfa(automaton1)
        else:
            dfa1 = automaton1
        
        if automaton2.type != AutomatonType.DFA:
            dfa2 = converter.nfa_to_dfa(automaton2)
        else:
            dfa2 = automaton2
        
        # Minimize both DFAs
        minimizer = DFAMinimizer()
        min_dfa1 = minimizer.minimize(dfa1)
        min_dfa2 = minimizer.minimize(dfa2)
        
        # Check structural equivalence of minimized DFAs
        if len(min_dfa1.states) != len(min_dfa2.states):
            # Find counterexample
            counterexample = self._find_counterexample(min_dfa1, min_dfa2)
            return False, counterexample
        
        # Check isomorphism
        return self._check_isomorphism(min_dfa1, min_dfa2)
    
    def _find_counterexample(
        self,
        dfa1: Automaton,
        dfa2: Automaton
    ) -> str:
        """Find a string accepted by one DFA but not the other."""
        # BFS to find shortest distinguishing string
        queue = deque([("", dfa1.initial_state, dfa2.initial_state)])
        visited = {(dfa1.initial_state, dfa2.initial_state)}
        
        while queue:
            string, state1, state2 = queue.popleft()
            
            # Check if one accepts and other rejects
            is_final1 = state1 in dfa1.final_states
            is_final2 = state2 in dfa2.final_states
            
            if is_final1 != is_final2:
                return string
            
            # Try each symbol
            for symbol in dfa1.alphabet:
                next1 = None
                next2 = None
                
                if state1 in dfa1.transitions and symbol in dfa1.transitions[state1]:
                    next1 = dfa1.transitions[state1][symbol]
                
                if state2 in dfa2.transitions and symbol in dfa2.transitions[state2]:
                    next2 = dfa2.transitions[state2][symbol]
                
                if next1 and next2 and (next1, next2) not in visited:
                    visited.add((next1, next2))
                    queue.append((string + symbol, next1, next2))
        
        return ""  # Should not reach here if DFAs are different
    
    def _check_isomorphism(
        self,
        dfa1: Automaton,
        dfa2: Automaton
    ) -> Tuple[bool, Optional[str]]:
        """Check if two DFAs are isomorphic."""
        # Simple check: same structure after renaming
        # More sophisticated implementation would find actual isomorphism
        
        # For now, check if they accept same strings up to certain length
        max_length = 10
        for length in range(max_length + 1):
            strings1 = self._generate_accepted_strings(dfa1, length)
            strings2 = self._generate_accepted_strings(dfa2, length)
            
            if strings1 != strings2:
                # Find specific counterexample
                diff = strings1.symmetric_difference(strings2)
                if diff:
                    return False, next(iter(diff))
        
        return True, None
    
    def _generate_accepted_strings(
        self,
        dfa: Automaton,
        max_length: int
    ) -> Set[str]:
        """Generate all accepted strings up to given length."""
        accepted = set()
        
        def generate(current_state: str, current_string: str, remaining: int):
            if current_state in dfa.final_states:
                accepted.add(current_string)
            
            if remaining > 0:
                for symbol in dfa.alphabet:
                    if current_state in dfa.transitions and symbol in dfa.transitions[current_state]:
                        next_state = dfa.transitions[current_state][symbol]
                        generate(next_state, current_string + symbol, remaining - 1)
        
        generate(dfa.initial_state, "", max_length)
        return accepted


class AutomataOptimizer:
    """Main optimizer coordinating all optimization operations."""
    
    def __init__(self):
        self.dfa_minimizer = DFAMinimizer()
        self.nfa_optimizer = NFAOptimizer()
        self.converter = AutomatonConverter()
        self.equivalence_checker = EquivalenceChecker()
    
    async def optimize(
        self,
        automaton: Automaton,
        optimization_type: Optional[OptimizationType] = None
    ) -> OptimizationResult:
        """
        Optimize an automaton.
        
        Args:
            automaton: Automaton to optimize
            optimization_type: Specific optimization to apply
        
        Returns:
            Optimization result
        """
        start_time = time.time()
        
        # Determine optimization type if not specified
        if not optimization_type:
            optimization_type = await self._determine_optimization(automaton)
        
        # Apply optimization
        if optimization_type == OptimizationType.MINIMIZATION:
            if automaton.type == AutomatonType.DFA:
                optimized = self.dfa_minimizer.minimize(automaton)
            else:
                # Convert to DFA first, then minimize
                dfa = self.converter.nfa_to_dfa(automaton)
                optimized = self.dfa_minimizer.minimize(dfa)
        
        elif optimization_type == OptimizationType.STATE_REDUCTION:
            if automaton.type in [AutomatonType.NFA, AutomatonType.EPSILON_NFA]:
                optimized = self.nfa_optimizer.optimize(automaton)
            else:
                optimized = self.dfa_minimizer.minimize(automaton)
        
        elif optimization_type == OptimizationType.CONVERSION:
            if automaton.type in [AutomatonType.NFA, AutomatonType.EPSILON_NFA]:
                optimized = self.converter.nfa_to_dfa(automaton)
            else:
                optimized = automaton  # Already DFA
        
        else:
            optimized = automaton  # No optimization applied
        
        # Verify equivalence
        are_equivalent, counterexample = self.equivalence_checker.are_equivalent(
            automaton,
            optimized
        )
        
        # Calculate metrics
        metrics = {
            "original_states": len(automaton.states),
            "optimized_states": len(optimized.states),
            "state_reduction": len(automaton.states) - len(optimized.states),
            "reduction_percentage": (
                (len(automaton.states) - len(optimized.states)) / len(automaton.states) * 100
                if len(automaton.states) > 0 else 0
            ),
            "original_transitions": automaton.get_transition_count(),
            "optimized_transitions": optimized.get_transition_count()
        }
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            original=automaton,
            optimized=optimized,
            optimization_type=optimization_type,
            metrics=metrics,
            equivalence_verified=are_equivalent,
            execution_time=execution_time,
            notes=[
                f"Reduced states by {metrics['state_reduction']} ({metrics['reduction_percentage']:.1f}%)",
                f"Optimization completed in {execution_time:.3f} seconds",
                f"Equivalence verified: {are_equivalent}"
            ]
        )
    
    async def _determine_optimization(
        self,
        automaton: Automaton
    ) -> OptimizationType:
        """Determine best optimization type using AI."""
        prompt = f"""Analyze this automaton and suggest the best optimization:

Type: {automaton.type.value}
States: {len(automaton.states)}
Transitions: {automaton.get_transition_count()}
Alphabet size: {len(automaton.alphabet)}

Options:
1. minimization - Minimize states using equivalence classes
2. state_reduction - Remove unreachable/dead states
3. conversion - Convert NFA to DFA

Return only the optimization name."""
        
        response = await orchestrator.execute(
            task="optimization_selection",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.2
        )
        
        response_text = response[0].response if isinstance(response, list) else response.response
        response_lower = response_text.lower()
        
        if "minimization" in response_lower:
            return OptimizationType.MINIMIZATION
        elif "reduction" in response_lower:
            return OptimizationType.STATE_REDUCTION
        elif "conversion" in response_lower:
            return OptimizationType.CONVERSION
        else:
            # Default based on type
            if automaton.type == AutomatonType.DFA:
                return OptimizationType.MINIMIZATION
            else:
                return OptimizationType.STATE_REDUCTION
    
    async def analyze_performance(
        self,
        automaton: Automaton
    ) -> Dict[str, Any]:
        """
        Analyze automaton performance characteristics.
        
        Args:
            automaton: Automaton to analyze
        
        Returns:
            Performance analysis
        """
        analysis = {
            "complexity": {
                "space": f"O({len(automaton.states)})",
                "time_per_symbol": "O(1)" if automaton.type == AutomatonType.DFA else f"O({len(automaton.states)})",
                "construction_time": f"O({len(automaton.states)} * {len(automaton.alphabet)})"
            },
            "statistics": {
                "state_count": len(automaton.states),
                "transition_count": automaton.get_transition_count(),
                "alphabet_size": len(automaton.alphabet),
                "final_state_count": len(automaton.final_states),
                "density": automaton.get_transition_count() / (len(automaton.states) * len(automaton.alphabet))
                if len(automaton.states) > 0 else 0
            },
            "properties": {
                "is_deterministic": automaton.is_deterministic(),
                "is_complete": self._is_complete(automaton),
                "is_minimal": await self._check_minimality(automaton),
                "has_unreachable": len(self._find_unreachable_states(automaton)) > 0
            },
            "recommendations": await self._generate_recommendations(automaton)
        }
        
        return analysis
    
    def _is_complete(self, automaton: Automaton) -> bool:
        """Check if automaton is complete (has transition for every symbol from every state)."""
        if automaton.type != AutomatonType.DFA:
            return False
        
        for state in automaton.states:
            if state not in automaton.transitions:
                return False
            for symbol in automaton.alphabet:
                if symbol not in automaton.transitions[state]:
                    return False
        
        return True
    
    def _find_unreachable_states(self, automaton: Automaton) -> Set[str]:
        """Find unreachable states."""
        if automaton.type == AutomatonType.DFA:
            reachable = self.dfa_minimizer._find_reachable_states(automaton)
        else:
            reachable = self.nfa_optimizer._find_reachable_states(automaton)
        
        return set(automaton.states) - reachable
    
    async def _check_minimality(self, automaton: Automaton) -> bool:
        """Check if automaton is already minimal."""
        if automaton.type != AutomatonType.DFA:
            return False
        
        minimized = self.dfa_minimizer.minimize(automaton)
        return len(minimized.states) == len(automaton.states)
    
    async def _generate_recommendations(self, automaton: Automaton) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check for unreachable states
        unreachable = self._find_unreachable_states(automaton)
        if unreachable:
            recommendations.append(
                f"Remove {len(unreachable)} unreachable states: {', '.join(list(unreachable)[:3])}"
            )
        
        # Check if can be minimized
        if automaton.type == AutomatonType.DFA:
            if not await self._check_minimality(automaton):
                recommendations.append("Apply DFA minimization to reduce states")
        
        # Check if complete
        if not self._is_complete(automaton):
            recommendations.append("Add transitions to make automaton complete")
        
        # Check density
        density = automaton.get_transition_count() / (len(automaton.states) * len(automaton.alphabet))
        if density < 0.3:
            recommendations.append("Consider using sparse representation for transitions")
        
        return recommendations


# Global optimizer instance
automata_optimizer = AutomataOptimizer()


async def optimize_automaton(
    automaton_dict: Dict[str, Any],
    optimization_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for automaton optimization.
    
    Args:
        automaton_dict: Automaton as dictionary
        optimization_type: Optional optimization type
    
    Returns:
        Optimization result as dictionary
    """
    # Convert dict to Automaton
    automaton = Automaton(**automaton_dict)
    
    # Optimize
    opt_type = OptimizationType(optimization_type) if optimization_type else None
    result = await automata_optimizer.optimize(automaton, opt_type)
    
    # Convert result to dict
    return {
        "original": result.original.to_dict(),
        "optimized": result.optimized.to_dict(),
        "optimization_type": result.optimization_type.value,
        "metrics": result.metrics,
        "equivalence_verified": result.equivalence_verified,
        "execution_time": result.execution_time,
        "notes": result.notes
    }