"""
JFLAP Advanced Simulator
========================

Comprehensive simulation engine for all automaton types with:
- Step-by-step execution with configuration tracking
- Non-deterministic branching visualization
- Instantaneous descriptions for TM/PDA
- Trace generation for debugging and education
- Multi-run support for non-deterministic automata
- Performance optimization for large inputs

Author: AegisX AI Software Engineer
Version: 1.0
"""

import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from copy import deepcopy
import time
import uuid
try:
    from .jflap_complete import (
        Automaton, State, Transition, AutomatonType, 
        Grammar, TuringMachine
    )
except ImportError:
    from jflap_complete import (
        Automaton, State, Transition, AutomatonType, 
        Grammar, TuringMachine
    )

# Configuration and Execution State Classes
# ==========================================

@dataclass
class ExecutionConfiguration:
    """
    Represents the complete state of an automaton at any point in execution
    """
    # Universal fields
    configuration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_states: Set[str] = field(default_factory=set)
    input_position: int = 0
    step_number: int = 0
    is_accepting: bool = False
    is_dead_end: bool = False
    parent_config_id: Optional[str] = None
    transition_taken: Optional[Transition] = None
    
    # PDA specific
    stack: List[str] = field(default_factory=list)
    stack_top: Optional[str] = None
    
    # TM specific
    tapes: List[List[str]] = field(default_factory=list)
    head_positions: List[int] = field(default_factory=list)
    
    # Execution metadata
    timestamp: float = field(default_factory=time.time)
    computation_path: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.stack:
            self.stack_top = self.stack[-1] if self.stack else None
    
    def clone(self) -> 'ExecutionConfiguration':
        """Create deep copy of configuration"""
        return ExecutionConfiguration(
            configuration_id=str(uuid.uuid4()),
            current_states=self.current_states.copy(),
            input_position=self.input_position,
            step_number=self.step_number + 1,
            is_accepting=self.is_accepting,
            is_dead_end=self.is_dead_end,
            parent_config_id=self.configuration_id,
            stack=self.stack.copy(),
            tapes=[tape.copy() for tape in self.tapes],
            head_positions=self.head_positions.copy(),
            computation_path=self.computation_path.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'configuration_id': self.configuration_id,
            'current_states': list(self.current_states),
            'input_position': self.input_position,
            'step_number': self.step_number,
            'is_accepting': self.is_accepting,
            'is_dead_end': self.is_dead_end,
            'parent_config_id': self.parent_config_id,
            'transition_taken': {
                'from_state': self.transition_taken.from_state,
                'to_state': self.transition_taken.to_state,
                'input_symbol': self.transition_taken.input_symbol,
                'output_symbol': self.transition_taken.output_symbol,
                'stack_pop': self.transition_taken.stack_pop,
                'stack_push': self.transition_taken.stack_push,
                'tape_read': self.transition_taken.tape_read,
                'tape_write': self.transition_taken.tape_write,
                'tape_move': self.transition_taken.tape_move
            } if self.transition_taken else None,
            'stack': self.stack,
            'stack_top': self.stack_top,
            'tapes': self.tapes,
            'head_positions': self.head_positions,
            'timestamp': self.timestamp,
            'computation_path': self.computation_path
        }

@dataclass
class SimulationResult:
    """Complete simulation result with all execution paths"""
    input_string: str
    automaton_type: str
    is_accepted: bool
    accepting_paths: List[List[ExecutionConfiguration]]
    rejecting_paths: List[List[ExecutionConfiguration]]
    all_configurations: List[ExecutionConfiguration]
    execution_tree: Dict[str, List[str]]  # parent_id -> [child_ids]
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'input_string': self.input_string,
            'automaton_type': self.automaton_type,
            'is_accepted': self.is_accepted,
            'accepting_paths': [[config.to_dict() for config in path] 
                              for path in self.accepting_paths],
            'rejecting_paths': [[config.to_dict() for config in path] 
                              for path in self.rejecting_paths],
            'all_configurations': [config.to_dict() for config in self.all_configurations],
            'execution_tree': self.execution_tree,
            'statistics': self.statistics
        }

# Advanced Simulator Classes
# ===========================

class JFLAPSimulator:
    """
    Advanced JFLAP-compatible simulator with comprehensive features
    """
    
    def __init__(self, automaton: Automaton):
        self.automaton = automaton
        self.configurations = []
        self.execution_tree = defaultdict(list)
        self.max_steps = 10000  # Prevent infinite loops
        self.max_configurations = 1000  # Limit non-deterministic explosion
        
    def simulate_string(self, input_string: str, 
                       mode: str = "all_paths") -> SimulationResult:
        """
        Simulate string execution with comprehensive tracking
        
        Args:
            input_string: Input string to process
            mode: "all_paths", "first_accepting", "shortest_path"
            
        Returns:
            Complete simulation result
        """
        start_time = time.time()
        
        # Initialize simulation
        initial_config = self._create_initial_configuration(input_string)
        
        # Run simulation based on automaton type
        if self.automaton.type in [AutomatonType.DFA, AutomatonType.NFA]:
            result = self._simulate_finite_automaton(initial_config, input_string, mode)
        elif self.automaton.type == AutomatonType.PDA:
            result = self._simulate_pushdown_automaton(initial_config, input_string, mode)
        elif self.automaton.type == AutomatonType.TM:
            result = self._simulate_turing_machine(initial_config, input_string, mode)
        else:
            raise ValueError(f"Unsupported automaton type: {self.automaton.type}")
        
        # Calculate statistics
        end_time = time.time()
        result.statistics = {
            'execution_time': end_time - start_time,
            'total_configurations': len(result.all_configurations),
            'max_branching_factor': self._calculate_max_branching_factor(result),
            'average_path_length': self._calculate_average_path_length(result),
            'memory_usage': self._estimate_memory_usage(result)
        }
        
        return result
    
    def _create_initial_configuration(self, input_string: str) -> ExecutionConfiguration:
        """Create initial configuration for simulation"""
        config = ExecutionConfiguration()
        config.current_states = {self.automaton.initial_state}
        config.input_position = 0
        config.step_number = 0
        
        # Initialize based on automaton type
        if self.automaton.type == AutomatonType.PDA:
            config.stack = ['Z']  # Bottom of stack marker
        elif self.automaton.type == AutomatonType.TM:
            # Initialize tape with input
            if input_string:
                config.tapes = [list(input_string) + [self.automaton.blank_symbol]]
            else:
                config.tapes = [[self.automaton.blank_symbol]]
            config.head_positions = [0]
        
        return config
    
    def _simulate_finite_automaton(self, initial_config: ExecutionConfiguration,
                                 input_string: str, mode: str) -> SimulationResult:
        """Simulate DFA/NFA execution"""
        accepting_paths = []
        rejecting_paths = []
        all_configurations = [initial_config]
        
        # BFS for exploring all paths
        queue = deque([initial_config])
        processed = set()
        
        while queue and len(all_configurations) < self.max_configurations:
            current_config = queue.popleft()
            config_key = self._get_config_key(current_config, input_string)
            
            if config_key in processed:
                continue
            processed.add(config_key)
            
            # Check if we've processed all input
            if current_config.input_position >= len(input_string):
                path = self._reconstruct_path(current_config, all_configurations)
                if any(state in self.automaton.final_states 
                       for state in current_config.current_states):
                    current_config.is_accepting = True
                    accepting_paths.append(path)
                    if mode == "first_accepting":
                        break
                else:
                    rejecting_paths.append(path)
                continue
            
            # Get current input symbol
            current_symbol = input_string[current_config.input_position]
            
            # Find all possible transitions
            next_configs = []
            for current_state in current_config.current_states:
                # Regular transitions
                for transition in self.automaton.transitions:
                    if (transition.from_state == current_state and
                        transition.input_symbol == current_symbol):
                        next_configs.append(self._apply_fa_transition(
                            current_config, transition, input_string
                        ))
                
                # Epsilon transitions (NFA only)
                if self.automaton.type == AutomatonType.NFA:
                    for transition in self.automaton.transitions:
                        if (transition.from_state == current_state and
                            transition.input_symbol == 'ε'):
                            next_configs.append(self._apply_fa_transition(
                                current_config, transition, input_string, epsilon=True
                            ))
            
            # Add next configurations to queue
            for next_config in next_configs:
                if next_config:
                    all_configurations.append(next_config)
                    queue.append(next_config)
                    self.execution_tree[current_config.configuration_id].append(
                        next_config.configuration_id
                    )
        
        # Handle remaining paths
        while queue:
            config = queue.popleft()
            path = self._reconstruct_path(config, all_configurations)
            config.is_dead_end = True
            rejecting_paths.append(path)
        
        return SimulationResult(
            input_string=input_string,
            automaton_type=self.automaton.type.value,
            is_accepted=len(accepting_paths) > 0,
            accepting_paths=accepting_paths,
            rejecting_paths=rejecting_paths,
            all_configurations=all_configurations,
            execution_tree=dict(self.execution_tree),
            statistics={}
        )
    
    def _simulate_pushdown_automaton(self, initial_config: ExecutionConfiguration,
                                   input_string: str, mode: str) -> SimulationResult:
        """Simulate PDA execution with stack tracking"""
        accepting_paths = []
        rejecting_paths = []
        all_configurations = [initial_config]
        
        queue = deque([initial_config])
        processed = set()
        
        while queue and len(all_configurations) < self.max_configurations:
            current_config = queue.popleft()
            config_key = self._get_pda_config_key(current_config, input_string)
            
            if config_key in processed:
                continue
            processed.add(config_key)
            
            # Check acceptance conditions
            input_consumed = current_config.input_position >= len(input_string)
            in_final_state = any(state in self.automaton.final_states 
                               for state in current_config.current_states)
            stack_empty = len(current_config.stack) <= 1  # Only Z remains
            
            if input_consumed and (in_final_state or stack_empty):
                path = self._reconstruct_path(current_config, all_configurations)
                current_config.is_accepting = True
                accepting_paths.append(path)
                if mode == "first_accepting":
                    break
                continue
            
            # Find applicable transitions
            next_configs = []
            current_symbol = (input_string[current_config.input_position] 
                            if current_config.input_position < len(input_string) 
                            else None)
            
            for current_state in current_config.current_states:
                for transition in self.automaton.transitions:
                    if transition.from_state == current_state:
                        next_config = self._apply_pda_transition(
                            current_config, transition, current_symbol
                        )
                        if next_config:
                            next_configs.append(next_config)
            
            # Add next configurations
            if not next_configs and not current_config.is_accepting:
                path = self._reconstruct_path(current_config, all_configurations)
                current_config.is_dead_end = True
                rejecting_paths.append(path)
            else:
                for next_config in next_configs:
                    all_configurations.append(next_config)
                    queue.append(next_config)
                    self.execution_tree[current_config.configuration_id].append(
                        next_config.configuration_id
                    )
        
        return SimulationResult(
            input_string=input_string,
            automaton_type=self.automaton.type.value,
            is_accepted=len(accepting_paths) > 0,
            accepting_paths=accepting_paths,
            rejecting_paths=rejecting_paths,
            all_configurations=all_configurations,
            execution_tree=dict(self.execution_tree),
            statistics={}
        )
    
    def _simulate_turing_machine(self, initial_config: ExecutionConfiguration,
                               input_string: str, mode: str) -> SimulationResult:
        """Simulate Turing Machine execution"""
        accepting_paths = []
        rejecting_paths = []
        all_configurations = [initial_config]
        
        queue = deque([initial_config])
        processed = set()
        
        while queue and len(all_configurations) < self.max_configurations:
            current_config = queue.popleft()
            
            # Prevent infinite loops
            if current_config.step_number > self.max_steps:
                path = self._reconstruct_path(current_config, all_configurations)
                current_config.is_dead_end = True
                rejecting_paths.append(path)
                continue
            
            config_key = self._get_tm_config_key(current_config)
            if config_key in processed:
                continue
            processed.add(config_key)
            
            # Check acceptance
            if any(state in self.automaton.final_states 
                   for state in current_config.current_states):
                path = self._reconstruct_path(current_config, all_configurations)
                current_config.is_accepting = True
                accepting_paths.append(path)
                if mode == "first_accepting":
                    break
                continue
            
            # Find applicable transitions
            next_configs = []
            for current_state in current_config.current_states:
                # Get current tape symbols
                tape_symbols = []
                for i, head_pos in enumerate(current_config.head_positions):
                    if 0 <= head_pos < len(current_config.tapes[i]):
                        tape_symbols.append(current_config.tapes[i][head_pos])
                    else:
                        tape_symbols.append(self.automaton.blank_symbol)
                
                for transition in self.automaton.transitions:
                    if (transition.from_state == current_state and
                        transition.tape_read == tape_symbols[0]):  # Single tape for now
                        next_config = self._apply_tm_transition(
                            current_config, transition
                        )
                        if next_config:
                            next_configs.append(next_config)
            
            # Add next configurations or mark as dead end
            if not next_configs:
                path = self._reconstruct_path(current_config, all_configurations)
                current_config.is_dead_end = True
                rejecting_paths.append(path)
            else:
                for next_config in next_configs:
                    all_configurations.append(next_config)
                    queue.append(next_config)
                    self.execution_tree[current_config.configuration_id].append(
                        next_config.configuration_id
                    )
        
        return SimulationResult(
            input_string=input_string,
            automaton_type=self.automaton.type.value,
            is_accepted=len(accepting_paths) > 0,
            accepting_paths=accepting_paths,
            rejecting_paths=rejecting_paths,
            all_configurations=all_configurations,
            execution_tree=dict(self.execution_tree),
            statistics={}
        )
    
    # Transition Application Methods
    # ==============================
    
    def _apply_fa_transition(self, config: ExecutionConfiguration, 
                           transition: Transition, input_string: str,
                           epsilon: bool = False) -> Optional[ExecutionConfiguration]:
        """Apply finite automaton transition"""
        next_config = config.clone()
        next_config.current_states = {transition.to_state}
        next_config.transition_taken = transition
        
        if not epsilon:
            next_config.input_position += 1
        
        next_config.computation_path.append(
            f"{transition.from_state} --{transition.input_symbol}--> {transition.to_state}"
        )
        
        return next_config
    
    def _apply_pda_transition(self, config: ExecutionConfiguration,
                            transition: Transition, 
                            current_symbol: Optional[str]) -> Optional[ExecutionConfiguration]:
        """Apply PDA transition with stack operations"""
        # Check if transition is applicable
        if transition.input_symbol != 'ε' and transition.input_symbol != current_symbol:
            return None
        
        if transition.stack_pop and (not config.stack or 
                                   config.stack[-1] != transition.stack_pop):
            return None
        
        next_config = config.clone()
        next_config.current_states = {transition.to_state}
        next_config.transition_taken = transition
        
        # Advance input if not epsilon transition
        if transition.input_symbol != 'ε':
            next_config.input_position += 1
        
        # Stack operations
        if transition.stack_pop and next_config.stack:
            next_config.stack.pop()
        
        if transition.stack_push and transition.stack_push != 'ε':
            # Push symbols in reverse order for multi-character pushes
            for symbol in reversed(transition.stack_push):
                next_config.stack.append(symbol)
        
        next_config.stack_top = next_config.stack[-1] if next_config.stack else None
        
        # Update computation path
        stack_op = f"[{transition.stack_pop or 'ε'}/{transition.stack_push or 'ε'}]"
        next_config.computation_path.append(
            f"{transition.from_state} --{transition.input_symbol},{stack_op}--> {transition.to_state}"
        )
        
        return next_config
    
    def _apply_tm_transition(self, config: ExecutionConfiguration,
                           transition: Transition) -> Optional[ExecutionConfiguration]:
        """Apply Turing Machine transition"""
        next_config = config.clone()
        next_config.current_states = {transition.to_state}
        next_config.transition_taken = transition
        
        # Write to tape
        if transition.tape_write:
            tape_idx = 0  # Single tape for now
            head_pos = next_config.head_positions[tape_idx]
            
            # Extend tape if necessary
            while len(next_config.tapes[tape_idx]) <= head_pos:
                next_config.tapes[tape_idx].append(self.automaton.blank_symbol)
            
            next_config.tapes[tape_idx][head_pos] = transition.tape_write
        
        # Move head
        if transition.tape_move == 'L':
            next_config.head_positions[0] = max(0, next_config.head_positions[0] - 1)
        elif transition.tape_move == 'R':
            next_config.head_positions[0] += 1
        # 'S' means stay
        
        # Update computation path
        tape_op = f"[{transition.tape_read or 'ε'}/{transition.tape_write or 'ε'},{transition.tape_move or 'S'}]"
        next_config.computation_path.append(
            f"{transition.from_state} --{tape_op}--> {transition.to_state}"
        )
        
        return next_config
    
    # Utility Methods
    # ===============
    
    def _get_config_key(self, config: ExecutionConfiguration, input_string: str) -> str:
        """Generate unique key for FA configuration"""
        return f"{sorted(config.current_states)}_{config.input_position}"
    
    def _get_pda_config_key(self, config: ExecutionConfiguration, input_string: str) -> str:
        """Generate unique key for PDA configuration"""
        stack_str = ''.join(config.stack)
        return f"{sorted(config.current_states)}_{config.input_position}_{stack_str}"
    
    def _get_tm_config_key(self, config: ExecutionConfiguration) -> str:
        """Generate unique key for TM configuration"""
        tape_str = ''.join([''.join(tape) for tape in config.tapes])
        head_str = '_'.join(map(str, config.head_positions))
        return f"{sorted(config.current_states)}_{tape_str}_{head_str}"
    
    def _reconstruct_path(self, final_config: ExecutionConfiguration,
                         all_configs: List[ExecutionConfiguration]) -> List[ExecutionConfiguration]:
        """Reconstruct execution path from initial to final configuration"""
        path = []
        current = final_config
        config_map = {config.configuration_id: config for config in all_configs}
        
        while current:
            path.append(current)
            if current.parent_config_id:
                current = config_map.get(current.parent_config_id)
            else:
                break
        
        return list(reversed(path))
    
    def _calculate_max_branching_factor(self, result: SimulationResult) -> int:
        """Calculate maximum branching factor in execution tree"""
        max_children = 0
        for parent_id, children in result.execution_tree.items():
            max_children = max(max_children, len(children))
        return max_children
    
    def _calculate_average_path_length(self, result: SimulationResult) -> float:
        """Calculate average path length"""
        all_paths = result.accepting_paths + result.rejecting_paths
        if not all_paths:
            return 0.0
        
        total_length = sum(len(path) for path in all_paths)
        return total_length / len(all_paths)
    
    def _estimate_memory_usage(self, result: SimulationResult) -> Dict[str, Any]:
        """Estimate memory usage of simulation"""
        return {
            'configurations_count': len(result.all_configurations),
            'tree_nodes': len(result.execution_tree),
            'estimated_bytes': len(str(result.to_dict()).encode('utf-8'))
        }

# Specialized Simulators
# ======================

class NonDeterministicTracker:
    """Track and visualize non-deterministic execution branches"""
    
    def __init__(self, simulation_result: SimulationResult):
        self.result = simulation_result
    
    def get_branching_points(self) -> List[Dict[str, Any]]:
        """Identify points where execution branches"""
        branching_points = []
        
        for parent_id, children in self.result.execution_tree.items():
            if len(children) > 1:
                parent_config = next(
                    (config for config in self.result.all_configurations 
                     if config.configuration_id == parent_id), None
                )
                
                if parent_config:
                    branching_points.append({
                        'configuration_id': parent_id,
                        'step_number': parent_config.step_number,
                        'current_states': list(parent_config.current_states),
                        'input_position': parent_config.input_position,
                        'branch_count': len(children),
                        'child_configurations': children
                    })
        
        return branching_points
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get detailed execution statistics"""
        accepting_configs = sum(1 for config in self.result.all_configurations 
                              if config.is_accepting)
        dead_end_configs = sum(1 for config in self.result.all_configurations 
                             if config.is_dead_end)
        
        return {
            'total_configurations': len(self.result.all_configurations),
            'accepting_configurations': accepting_configs,
            'dead_end_configurations': dead_end_configs,
            'active_configurations': (len(self.result.all_configurations) - 
                                    accepting_configs - dead_end_configs),
            'branching_points': len(self.get_branching_points()),
            'max_depth': max((config.step_number for config in self.result.all_configurations), 
                           default=0)
        }

class InstantaneousDescriptionGenerator:
    """Generate instantaneous descriptions for educational purposes"""
    
    @staticmethod
    def generate_fa_description(config: ExecutionConfiguration, 
                              input_string: str) -> str:
        """Generate FA instantaneous description"""
        states_str = "{" + ", ".join(sorted(config.current_states)) + "}"
        remaining_input = input_string[config.input_position:]
        return f"({states_str}, {remaining_input or 'ε'})"
    
    @staticmethod
    def generate_pda_description(config: ExecutionConfiguration, 
                               input_string: str) -> str:
        """Generate PDA instantaneous description"""
        states_str = "{" + ", ".join(sorted(config.current_states)) + "}"
        remaining_input = input_string[config.input_position:]
        stack_str = ''.join(config.stack) if config.stack else 'ε'
        return f"({states_str}, {remaining_input or 'ε'}, {stack_str})"
    
    @staticmethod
    def generate_tm_description(config: ExecutionConfiguration) -> str:
        """Generate TM instantaneous description"""
        if not config.tapes or not config.head_positions:
            return "Invalid TM configuration"
        
        # Format tape with head position marker
        tape = config.tapes[0]
        head_pos = config.head_positions[0]
        
        # Create visual representation
        tape_left = ''.join(tape[:head_pos]) if head_pos > 0 else ''
        current_symbol = tape[head_pos] if head_pos < len(tape) else '□'
        tape_right = ''.join(tape[head_pos+1:]) if head_pos < len(tape) - 1 else ''
        
        tape_visual = f"{tape_left}[{current_symbol}]{tape_right}"
        
        states_str = "{" + ", ".join(sorted(config.current_states)) + "}"
        return f"({states_str}, {tape_visual})"

# Main Simulation Engine
# ======================

class JFLAPSimulationEngine:
    """
    Main simulation engine with support for all automaton types
    and advanced features
    """
    
    def __init__(self):
        self.simulators = {}
        self.results_cache = {}
    
    def simulate(self, automaton: Automaton, input_string: str,
                options: Dict[str, Any] = None) -> SimulationResult:
        """
        Main simulation method with comprehensive options
        
        Args:
            automaton: Automaton to simulate
            input_string: Input string
            options: Simulation options
                - mode: "all_paths", "first_accepting", "shortest_path"
                - max_steps: Maximum steps for TM
                - max_configurations: Maximum configurations for non-deterministic
                - track_branches: Whether to track branching points
                - generate_descriptions: Whether to generate instantaneous descriptions
        
        Returns:
            Complete simulation result
        """
        options = options or {}
        
        # Create simulator
        simulator = JFLAPSimulator(automaton)
        simulator.max_steps = options.get('max_steps', 10000)
        simulator.max_configurations = options.get('max_configurations', 1000)
        
        # Run simulation
        result = simulator.simulate_string(
            input_string, 
            mode=options.get('mode', 'all_paths')
        )
        
        # Enhanced processing based on options
        if options.get('track_branches', True):
            tracker = NonDeterministicTracker(result)
            result.statistics['branching_analysis'] = tracker.get_execution_statistics()
            result.statistics['branching_points'] = tracker.get_branching_points()
        
        if options.get('generate_descriptions', True):
            self._add_instantaneous_descriptions(result, automaton, input_string)
        
        return result
    
    def _add_instantaneous_descriptions(self, result: SimulationResult,
                                      automaton: Automaton, input_string: str):
        """Add instantaneous descriptions to configurations"""
        generator = InstantaneousDescriptionGenerator()
        
        for config in result.all_configurations:
            if automaton.type in [AutomatonType.DFA, AutomatonType.NFA]:
                config.description = generator.generate_fa_description(config, input_string)
            elif automaton.type == AutomatonType.PDA:
                config.description = generator.generate_pda_description(config, input_string)
            elif automaton.type == AutomatonType.TM:
                config.description = generator.generate_tm_description(config)
    
    def batch_simulate(self, automaton: Automaton, 
                      input_strings: List[str],
                      options: Dict[str, Any] = None) -> List[SimulationResult]:
        """Simulate multiple strings efficiently"""
        results = []
        
        for input_string in input_strings:
            result = self.simulate(automaton, input_string, options)
            results.append(result)
        
        return results
    
    def compare_executions(self, automaton: Automaton,
                          input_strings: List[str]) -> Dict[str, Any]:
        """Compare execution patterns across multiple inputs"""
        results = self.batch_simulate(automaton, input_strings)
        
        comparison = {
            'inputs': input_strings,
            'acceptance_pattern': [result.is_accepted for result in results],
            'step_counts': [result.statistics.get('max_depth', 0) for result in results],
            'configuration_counts': [len(result.all_configurations) for result in results],
            'average_execution_time': sum(result.statistics.get('execution_time', 0) 
                                        for result in results) / len(results),
            'complexity_analysis': self._analyze_complexity_pattern(results)
        }
        
        return comparison
    
    def _analyze_complexity_pattern(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze computational complexity patterns"""
        input_lengths = [len(result.input_string) for result in results]
        step_counts = [result.statistics.get('max_depth', 0) for result in results]
        config_counts = [len(result.all_configurations) for result in results]
        
        return {
            'time_complexity_estimate': self._estimate_time_complexity(input_lengths, step_counts),
            'space_complexity_estimate': self._estimate_space_complexity(input_lengths, config_counts),
            'scaling_factor': max(config_counts) / min(config_counts) if min(config_counts) > 0 else 0
        }
    
    def _estimate_time_complexity(self, input_lengths: List[int], 
                                step_counts: List[int]) -> str:
        """Estimate time complexity based on input length vs steps"""
        if not input_lengths or not step_counts:
            return "insufficient_data"
        
        # Simple heuristic analysis
        max_ratio = max(steps / length if length > 0 else steps 
                       for steps, length in zip(step_counts, input_lengths))
        
        if max_ratio <= 2:
            return "O(n)"
        elif max_ratio <= 10:
            return "O(n²)"
        else:
            return "O(2^n) or worse"
    
    def _estimate_space_complexity(self, input_lengths: List[int], 
                                 config_counts: List[int]) -> str:
        """Estimate space complexity based on configuration counts"""
        if not input_lengths or not config_counts:
            return "insufficient_data"
        
        max_ratio = max(configs / length if length > 0 else configs 
                       for configs, length in zip(config_counts, input_lengths))
        
        if max_ratio <= 5:
            return "O(n)"
        elif max_ratio <= 50:
            return "O(n²)"
        else:
            return "O(2^n) or worse"

# Global simulation engine instance
simulation_engine = JFLAPSimulationEngine()