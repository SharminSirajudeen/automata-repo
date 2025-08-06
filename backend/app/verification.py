"""
Formal Verification Module for Automata Theory
Provides automata equivalence checking, language containment, DFA minimization, and counter-example generation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging
from collections import defaultdict, deque
import itertools

logger = logging.getLogger(__name__)

class AutomatonType(str, Enum):
    """Types of automata supported"""
    DFA = "dfa"
    NFA = "nfa"
    PDA = "pda"
    TM = "tm"

class VerificationResult(BaseModel):
    """Result of a verification operation"""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    counter_example: Optional[str] = None
    witness: Optional[str] = None

class AutomatonState(BaseModel):
    """State in an automaton"""
    id: str
    is_start: bool = False
    is_accept: bool = False
    x: float = 0.0
    y: float = 0.0

class AutomatonTransition(BaseModel):
    """Transition in an automaton"""
    from_state: str
    to_state: str
    symbol: str

class Automaton(BaseModel):
    """Generic automaton representation"""
    states: List[AutomatonState]
    transitions: List[AutomatonTransition]
    alphabet: List[str]
    automaton_type: AutomatonType = AutomatonType.DFA

class EquivalenceRequest(BaseModel):
    """Request for automata equivalence checking"""
    automaton1: Automaton
    automaton2: Automaton

class ContainmentRequest(BaseModel):
    """Request for language containment checking"""
    subset_automaton: Automaton  # L1 ⊆ L2?
    superset_automaton: Automaton

class MinimizationRequest(BaseModel):
    """Request for DFA minimization"""
    automaton: Automaton

class AutomataVerifier:
    """Core verification algorithms for automata"""
    
    def __init__(self):
        self.max_string_length = 1000  # Maximum length for brute force checks
        self.max_states = 1000  # Maximum states for practical algorithms
    
    def check_equivalence(self, request: EquivalenceRequest) -> VerificationResult:
        """Check if two automata are equivalent (recognize same language)"""
        try:
            aut1, aut2 = request.automaton1, request.automaton2
            
            # Validate inputs
            if not self._validate_automaton(aut1) or not self._validate_automaton(aut2):
                return VerificationResult(
                    is_valid=False,
                    message="Invalid automaton structure"
                )
            
            # Check if both are DFAs (required for efficient equivalence checking)
            if aut1.automaton_type != AutomatonType.DFA or aut2.automaton_type != AutomatonType.DFA:
                return VerificationResult(
                    is_valid=False,
                    message="Equivalence checking currently supports DFAs only"
                )
            
            # Convert to internal representation
            dfa1 = self._convert_to_dfa_dict(aut1)
            dfa2 = self._convert_to_dfa_dict(aut2)
            
            # Check equivalence using product construction
            return self._check_dfa_equivalence(dfa1, dfa2)
            
        except Exception as e:
            logger.error(f"Error in equivalence checking: {e}")
            return VerificationResult(
                is_valid=False,
                message=f"Verification error: {str(e)}"
            )
    
    def check_containment(self, request: ContainmentRequest) -> VerificationResult:
        """Check if L(subset_automaton) ⊆ L(superset_automaton)"""
        try:
            sub_aut = request.subset_automaton
            sup_aut = request.superset_automaton
            
            # Validate inputs
            if not self._validate_automaton(sub_aut) or not self._validate_automaton(sup_aut):
                return VerificationResult(
                    is_valid=False,
                    message="Invalid automaton structure"
                )
            
            # Check containment: L1 ⊆ L2 iff L1 ∩ L2^c = ∅
            # This is equivalent to checking if L1 - L2 = ∅
            
            if sub_aut.automaton_type != AutomatonType.DFA or sup_aut.automaton_type != AutomatonType.DFA:
                return VerificationResult(
                    is_valid=False,
                    message="Containment checking currently supports DFAs only"
                )
            
            dfa1 = self._convert_to_dfa_dict(sub_aut)
            dfa2 = self._convert_to_dfa_dict(sup_aut)
            
            return self._check_dfa_containment(dfa1, dfa2)
            
        except Exception as e:
            logger.error(f"Error in containment checking: {e}")
            return VerificationResult(
                is_valid=False,
                message=f"Verification error: {str(e)}"
            )
    
    def minimize_dfa(self, request: MinimizationRequest) -> Dict[str, Any]:
        """Minimize a DFA using Hopcroft's algorithm"""
        try:
            automaton = request.automaton
            
            if automaton.automaton_type != AutomatonType.DFA:
                return {
                    "success": False,
                    "message": "Minimization only supports DFAs"
                }
            
            dfa = self._convert_to_dfa_dict(automaton)
            minimized_dfa = self._minimize_dfa_hopcroft(dfa)
            minimized_automaton = self._convert_from_dfa_dict(minimized_dfa)
            
            original_states = len(automaton.states)
            minimized_states = len(minimized_automaton.states)
            
            return {
                "success": True,
                "original_automaton": automaton,
                "minimized_automaton": minimized_automaton,
                "reduction_info": {
                    "original_states": original_states,
                    "minimized_states": minimized_states,
                    "reduction_percentage": ((original_states - minimized_states) / original_states) * 100 if original_states > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in DFA minimization: {e}")
            return {
                "success": False,
                "message": f"Minimization error: {str(e)}"
            }
    
    def generate_counter_example(self, automaton1: Automaton, automaton2: Automaton, max_length: int = 10) -> Optional[str]:
        """Generate a counter-example showing automata are not equivalent"""
        try:
            if automaton1.automaton_type != AutomatonType.DFA or automaton2.automaton_type != AutomatonType.DFA:
                return None
            
            dfa1 = self._convert_to_dfa_dict(automaton1)
            dfa2 = self._convert_to_dfa_dict(automaton2)
            
            # Use BFS to find shortest distinguishing string
            return self._find_distinguishing_string(dfa1, dfa2, max_length)
            
        except Exception as e:
            logger.error(f"Error generating counter-example: {e}")
            return None
    
    def _validate_automaton(self, automaton: Automaton) -> bool:
        """Validate automaton structure"""
        try:
            # Check for start state
            start_states = [s for s in automaton.states if s.is_start]
            if len(start_states) != 1:
                return False
            
            # Check state IDs are unique
            state_ids = [s.id for s in automaton.states]
            if len(state_ids) != len(set(state_ids)):
                return False
            
            # Check transitions reference valid states
            state_id_set = set(state_ids)
            for trans in automaton.transitions:
                if trans.from_state not in state_id_set or trans.to_state not in state_id_set:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _convert_to_dfa_dict(self, automaton: Automaton) -> Dict[str, Any]:
        """Convert automaton to internal dictionary representation"""
        states = {s.id for s in automaton.states}
        start_state = next(s.id for s in automaton.states if s.is_start)
        accept_states = {s.id for s in automaton.states if s.is_accept}
        alphabet = set(automaton.alphabet)
        
        # Build transition function
        transitions = defaultdict(dict)
        for trans in automaton.transitions:
            transitions[trans.from_state][trans.symbol] = trans.to_state
        
        return {
            'states': states,
            'alphabet': alphabet,
            'transitions': dict(transitions),
            'start_state': start_state,
            'accept_states': accept_states
        }
    
    def _convert_from_dfa_dict(self, dfa_dict: Dict[str, Any]) -> Automaton:
        """Convert internal dictionary back to Automaton"""
        states = []
        transitions = []
        
        # Create states
        for i, state_id in enumerate(sorted(dfa_dict['states'])):
            states.append(AutomatonState(
                id=state_id,
                is_start=(state_id == dfa_dict['start_state']),
                is_accept=(state_id in dfa_dict['accept_states']),
                x=100 * (i % 5),  # Simple layout
                y=100 * (i // 5)
            ))
        
        # Create transitions
        for from_state, trans_dict in dfa_dict['transitions'].items():
            for symbol, to_state in trans_dict.items():
                transitions.append(AutomatonTransition(
                    from_state=from_state,
                    to_state=to_state,
                    symbol=symbol
                ))
        
        return Automaton(
            states=states,
            transitions=transitions,
            alphabet=list(dfa_dict['alphabet']),
            automaton_type=AutomatonType.DFA
        )
    
    def _check_dfa_equivalence(self, dfa1: Dict[str, Any], dfa2: Dict[str, Any]) -> VerificationResult:
        """Check equivalence using product construction and reachability"""
        try:
            # Build product automaton to find differences
            alphabet = dfa1['alphabet'] | dfa2['alphabet']
            
            # State pairs: (state1, state2)
            start_pair = (dfa1['start_state'], dfa2['start_state'])
            visited = set()
            queue = deque([start_pair])
            
            while queue:
                state1, state2 = queue.popleft()
                
                if (state1, state2) in visited:
                    continue
                visited.add((state1, state2))
                
                # Check if acceptance differs
                accept1 = state1 in dfa1['accept_states']
                accept2 = state2 in dfa2['accept_states']
                
                if accept1 != accept2:
                    # Found a distinguishing string
                    path = self._reconstruct_path_to_pair(dfa1, dfa2, start_pair, (state1, state2))
                    return VerificationResult(
                        is_valid=False,
                        message="Automata are not equivalent",
                        counter_example=path,
                        details={
                            "distinguishing_state_pair": (state1, state2),
                            "dfa1_accepts": accept1,
                            "dfa2_accepts": accept2
                        }
                    )
                
                # Explore transitions
                for symbol in alphabet:
                    next1 = dfa1['transitions'].get(state1, {}).get(symbol)
                    next2 = dfa2['transitions'].get(state2, {}).get(symbol)
                    
                    # Handle missing transitions (go to implicit dead state)
                    if next1 is None:
                        next1 = "DEAD"
                    if next2 is None:
                        next2 = "DEAD"
                    
                    if (next1, next2) not in visited:
                        queue.append((next1, next2))
            
            return VerificationResult(
                is_valid=True,
                message="Automata are equivalent",
                details={"explored_state_pairs": len(visited)}
            )
            
        except Exception as e:
            logger.error(f"Error in DFA equivalence check: {e}")
            return VerificationResult(
                is_valid=False,
                message=f"Equivalence check failed: {str(e)}"
            )
    
    def _check_dfa_containment(self, dfa1: Dict[str, Any], dfa2: Dict[str, Any]) -> VerificationResult:
        """Check if L(dfa1) ⊆ L(dfa2)"""
        try:
            # L1 ⊆ L2 iff there's no string accepted by L1 but rejected by L2
            alphabet = dfa1['alphabet'] | dfa2['alphabet']
            
            start_pair = (dfa1['start_state'], dfa2['start_state'])
            visited = set()
            queue = deque([start_pair])
            
            while queue:
                state1, state2 = queue.popleft()
                
                if (state1, state2) in visited:
                    continue
                visited.add((state1, state2))
                
                # Check for containment violation
                accept1 = state1 in dfa1['accept_states']
                accept2 = state2 in dfa2['accept_states']
                
                if accept1 and not accept2:
                    # Found string in L1 but not in L2
                    path = self._reconstruct_path_to_pair(dfa1, dfa2, start_pair, (state1, state2))
                    return VerificationResult(
                        is_valid=False,
                        message="Containment does not hold",
                        counter_example=path,
                        details={
                            "violating_state_pair": (state1, state2)
                        }
                    )
                
                # Explore transitions
                for symbol in alphabet:
                    next1 = dfa1['transitions'].get(state1, {}).get(symbol)
                    next2 = dfa2['transitions'].get(state2, {}).get(symbol)
                    
                    if next1 is None:
                        next1 = "DEAD"
                    if next2 is None:
                        next2 = "DEAD"
                    
                    if (next1, next2) not in visited:
                        queue.append((next1, next2))
            
            return VerificationResult(
                is_valid=True,
                message="Containment holds: L1 ⊆ L2",
                details={"explored_state_pairs": len(visited)}
            )
            
        except Exception as e:
            logger.error(f"Error in containment check: {e}")
            return VerificationResult(
                is_valid=False,
                message=f"Containment check failed: {str(e)}"
            )
    
    def _minimize_dfa_hopcroft(self, dfa: Dict[str, Any]) -> Dict[str, Any]:
        """Minimize DFA using Hopcroft's algorithm"""
        states = dfa['states']
        alphabet = dfa['alphabet']
        transitions = dfa['transitions']
        accept_states = dfa['accept_states']
        
        # Initial partition: accepting vs non-accepting states
        accepting = accept_states & states
        non_accepting = states - accept_states
        
        partitions = []
        if accepting:
            partitions.append(accepting)
        if non_accepting:
            partitions.append(non_accepting)
        
        # Refine partitions
        changed = True
        while changed:
            changed = False
            new_partitions = []
            
            for partition in partitions:
                # Try to split this partition
                splits = self._split_partition(partition, partitions, alphabet, transitions)
                if len(splits) > 1:
                    changed = True
                    new_partitions.extend(splits)
                else:
                    new_partitions.append(partition)
            
            partitions = new_partitions
        
        # Build minimized DFA
        return self._build_minimized_dfa(dfa, partitions)
    
    def _split_partition(self, partition: Set[str], all_partitions: List[Set[str]], 
                        alphabet: Set[str], transitions: Dict[str, Dict[str, str]]) -> List[Set[str]]:
        """Split a partition based on transition behavior"""
        if len(partition) == 1:
            return [partition]
        
        # Group states by their transition signatures
        signature_groups = defaultdict(set)
        
        for state in partition:
            signature = []
            for symbol in sorted(alphabet):
                target = transitions.get(state, {}).get(symbol, "DEAD")
                # Find which partition the target belongs to
                target_partition_idx = -1
                for i, part in enumerate(all_partitions):
                    if target in part:
                        target_partition_idx = i
                        break
                signature.append(target_partition_idx)
            
            signature_groups[tuple(signature)].add(state)
        
        return list(signature_groups.values())
    
    def _build_minimized_dfa(self, original_dfa: Dict[str, Any], partitions: List[Set[str]]) -> Dict[str, Any]:
        """Build minimized DFA from partitions"""
        # Create state mapping
        state_to_partition = {}
        partition_representatives = {}
        
        for i, partition in enumerate(partitions):
            rep = f"q{i}"
            partition_representatives[i] = rep
            for state in partition:
                state_to_partition[state] = i
        
        # Find start state partition
        start_partition = state_to_partition[original_dfa['start_state']]
        new_start_state = partition_representatives[start_partition]
        
        # Find accept states
        new_accept_states = set()
        for partition_idx, partition in enumerate(partitions):
            if partition & original_dfa['accept_states']:
                new_accept_states.add(partition_representatives[partition_idx])
        
        # Build new transitions
        new_transitions = defaultdict(dict)
        for partition_idx, partition in enumerate(partitions):
            rep_state = partition_representatives[partition_idx]
            # Take any state from partition to determine transitions
            sample_state = next(iter(partition))
            
            for symbol in original_dfa['alphabet']:
                target = original_dfa['transitions'].get(sample_state, {}).get(symbol)
                if target and target in state_to_partition:
                    target_partition = state_to_partition[target]
                    target_rep = partition_representatives[target_partition]
                    new_transitions[rep_state][symbol] = target_rep
        
        return {
            'states': set(partition_representatives.values()),
            'alphabet': original_dfa['alphabet'],
            'transitions': dict(new_transitions),
            'start_state': new_start_state,
            'accept_states': new_accept_states
        }
    
    def _find_distinguishing_string(self, dfa1: Dict[str, Any], dfa2: Dict[str, Any], max_length: int) -> Optional[str]:
        """Find shortest string that distinguishes two DFAs"""
        alphabet = list(dfa1['alphabet'] | dfa2['alphabet'])
        
        # BFS to find shortest distinguishing string
        queue = deque([(dfa1['start_state'], dfa2['start_state'], "")])
        visited = set()
        
        while queue and len(queue[0][2]) <= max_length:
            state1, state2, path = queue.popleft()
            
            if (state1, state2) in visited:
                continue
            visited.add((state1, state2))
            
            # Check if current states have different acceptance
            accept1 = state1 in dfa1['accept_states']
            accept2 = state2 in dfa2['accept_states']
            
            if accept1 != accept2:
                return path
            
            # Explore transitions
            if len(path) < max_length:
                for symbol in alphabet:
                    next1 = dfa1['transitions'].get(state1, {}).get(symbol, "DEAD")
                    next2 = dfa2['transitions'].get(state2, {}).get(symbol, "DEAD")
                    
                    new_path = path + symbol
                    if (next1, next2) not in visited:
                        queue.append((next1, next2, new_path))
        
        return None
    
    def _reconstruct_path_to_pair(self, dfa1: Dict[str, Any], dfa2: Dict[str, Any], 
                                 start_pair: Tuple[str, str], target_pair: Tuple[str, str]) -> str:
        """Reconstruct path from start to target state pair"""
        # Simple BFS to find path
        alphabet = list(dfa1['alphabet'] | dfa2['alphabet'])
        queue = deque([(start_pair[0], start_pair[1], "")])
        visited = set()
        
        while queue:
            state1, state2, path = queue.popleft()
            
            if (state1, state2) == target_pair:
                return path
            
            if (state1, state2) in visited:
                continue
            visited.add((state1, state2))
            
            for symbol in alphabet:
                next1 = dfa1['transitions'].get(state1, {}).get(symbol, "DEAD")
                next2 = dfa2['transitions'].get(state2, {}).get(symbol, "DEAD")
                
                if (next1, next2) not in visited:
                    queue.append((next1, next2, path + symbol))
        
        return ""  # Should not reach here

# Global verifier instance
automata_verifier = AutomataVerifier()

def verify_equivalence(request: EquivalenceRequest) -> VerificationResult:
    """Check if two automata are equivalent"""
    return automata_verifier.check_equivalence(request)

def verify_containment(request: ContainmentRequest) -> VerificationResult:
    """Check if one language is contained in another"""
    return automata_verifier.check_containment(request)

def minimize_automaton(request: MinimizationRequest) -> Dict[str, Any]:
    """Minimize an automaton"""
    return automata_verifier.minimize_dfa(request)

def find_counter_example(automaton1: Automaton, automaton2: Automaton, max_length: int = 10) -> Optional[str]:
    """Find a counter-example showing automata are not equivalent"""
    return automata_verifier.generate_counter_example(automaton1, automaton2, max_length)

def get_verification_algorithms() -> Dict[str, Any]:
    """Get information about available verification algorithms"""
    return {
        "equivalence_checking": {
            "name": "Product Construction Method",
            "description": "Uses product automaton construction to find distinguishing strings",
            "complexity": "O(|Q1| × |Q2| × |Σ|)",
            "supports": ["DFA"]
        },
        "containment_checking": {
            "name": "Product Construction with Complement",
            "description": "Checks L1 ⊆ L2 by verifying L1 ∩ L2^c = ∅",
            "complexity": "O(|Q1| × |Q2| × |Σ|)",
            "supports": ["DFA"]
        },
        "minimization": {
            "name": "Hopcroft's Algorithm",
            "description": "Partition refinement algorithm for DFA minimization",
            "complexity": "O(|Q| × |Σ| × log|Q|)",
            "supports": ["DFA"]
        },
        "counter_example_generation": {
            "name": "BFS Shortest Path",
            "description": "Finds shortest distinguishing string using breadth-first search",
            "complexity": "O(|Q1| × |Q2| × |Σ|^L) where L is max length",
            "supports": ["DFA"]
        }
    }