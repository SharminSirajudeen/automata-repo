"""
Knowledge Extraction System for Hardcoded Algorithms
===================================================

This system analyzes hardcoded algorithms to extract:
- Problem patterns they solve
- Step-by-step strategies
- Edge cases they handle
- Optimization techniques

The extracted knowledge is used to improve AI-generated solutions.

Author: AegisX AI Software Engineer
Version: 1.0
"""

import ast
import inspect
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from pathlib import Path

from .jflap_complete import (
    NFAToDFAConverter, DFAMinimizer, RegexConverter, CFGProcessor,
    ParsingAlgorithms, TuringMachine, JFLAPAlgorithms,
    AutomatonType, State, Transition, Automaton, Grammar
)
from .jflap_advanced import (
    MultiTapeTuringMachine, UniversalTuringMachine, UnrestrictedGrammar,
    ContextSensitiveGrammar, SLRParser, GNFConverter, EnhancedLSystem
)
from .problem_understanding import ProblemType, LanguagePattern

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmSignature:
    """Signature of an algorithm including inputs, outputs, and behavior."""
    name: str
    class_name: str
    input_types: List[str]
    output_types: List[str]
    complexity: Dict[str, str]  # time, space complexity
    preconditions: List[str]
    postconditions: List[str]
    error_conditions: List[str]


@dataclass
class SolutionPattern:
    """Pattern extracted from hardcoded solutions."""
    pattern_id: str
    pattern_name: str
    problem_types: List[ProblemType]
    language_patterns: List[LanguagePattern]
    algorithm_steps: List[str]
    key_insights: List[str]
    common_pitfalls: List[str]
    optimization_opportunities: List[str]
    code_examples: List[str]
    mathematical_properties: Dict[str, Any]
    performance_characteristics: Dict[str, Any]


@dataclass
class EdgeCase:
    """Edge case identified in hardcoded algorithms."""
    case_id: str
    description: str
    algorithm: str
    input_conditions: List[str]
    handling_strategy: str
    code_snippet: str
    importance_level: int  # 1-5, 5 being critical


@dataclass
class OptimizationTechnique:
    """Optimization technique used in hardcoded algorithms."""
    technique_id: str
    technique_name: str
    description: str
    applicable_contexts: List[str]
    performance_impact: Dict[str, float]  # time_improvement, space_improvement
    implementation_examples: List[str]
    trade_offs: List[str]


class CodeAnalyzer:
    """Analyzes source code to extract patterns and insights."""
    
    def __init__(self):
        self.ast_cache = {}
        self.pattern_extractors = {
            'loops': self._extract_loop_patterns,
            'conditionals': self._extract_conditional_patterns,
            'data_structures': self._extract_data_structure_patterns,
            'algorithms': self._extract_algorithm_patterns,
            'optimizations': self._extract_optimization_patterns
        }
    
    def analyze_class(self, cls) -> Dict[str, Any]:
        """Analyze a class to extract algorithmic patterns."""
        
        analysis = {
            'class_name': cls.__name__,
            'methods': {},
            'complexity_patterns': [],
            'data_flow_patterns': [],
            'optimization_patterns': []
        }
        
        # Get source code
        try:
            source = inspect.getsource(cls)
            tree = ast.parse(source)
            analysis['ast'] = tree
        except Exception as e:
            logger.warning(f"Could not analyze source for {cls.__name__}: {e}")
            return analysis
        
        # Analyze each method
        for method_name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
            if not method_name.startswith('_'):  # Skip private methods
                analysis['methods'][method_name] = self._analyze_method(method)
        
        # Extract high-level patterns
        analysis['complexity_patterns'] = self._extract_complexity_patterns(tree)
        analysis['data_flow_patterns'] = self._extract_data_flow_patterns(tree)
        analysis['optimization_patterns'] = self._extract_optimization_patterns(tree)
        
        return analysis
    
    def _analyze_method(self, method) -> Dict[str, Any]:
        """Analyze a single method."""
        
        analysis = {
            'signature': self._extract_signature(method),
            'complexity': self._estimate_complexity(method),
            'patterns': [],
            'edge_cases': [],
            'optimizations': []
        }
        
        try:
            source = inspect.getsource(method)
            tree = ast.parse(source)
            
            # Extract patterns
            for pattern_type, extractor in self.pattern_extractors.items():
                patterns = extractor(tree)
                analysis['patterns'].extend(patterns)
            
            # Identify edge cases
            analysis['edge_cases'] = self._identify_edge_cases(tree)
            
            # Find optimizations
            analysis['optimizations'] = self._identify_optimizations(tree)
            
        except Exception as e:
            logger.warning(f"Could not analyze method source: {e}")
        
        return analysis
    
    def _extract_signature(self, method) -> AlgorithmSignature:
        """Extract method signature information."""
        
        sig = inspect.signature(method)
        
        return AlgorithmSignature(
            name=method.__name__,
            class_name=method.__qualname__.split('.')[0] if '.' in method.__qualname__ else '',
            input_types=self._extract_parameter_types(sig),
            output_types=self._extract_return_type(sig),
            complexity={'time': 'unknown', 'space': 'unknown'},
            preconditions=[],
            postconditions=[],
            error_conditions=[]
        )
    
    def _extract_parameter_types(self, signature) -> List[str]:
        """Extract parameter types from signature."""
        types = []
        for param in signature.parameters.values():
            if param.annotation != param.empty:
                types.append(str(param.annotation))
            else:
                types.append('Any')
        return types
    
    def _extract_return_type(self, signature) -> List[str]:
        """Extract return type from signature."""
        if signature.return_annotation != signature.empty:
            return [str(signature.return_annotation)]
        return ['Any']
    
    def _estimate_complexity(self, method) -> Dict[str, str]:
        """Estimate time and space complexity."""
        
        complexity = {'time': 'O(1)', 'space': 'O(1)'}
        
        try:
            source = inspect.getsource(method)
            
            # Simple heuristics for complexity estimation
            if 'for' in source:
                nested_loops = source.count('for') - source.count('for ') + source.count('for ')
                if nested_loops == 1:
                    complexity['time'] = 'O(n)'
                elif nested_loops == 2:
                    complexity['time'] = 'O(n²)'
                else:
                    complexity['time'] = f'O(n^{nested_loops})'
            
            if 'while' in source:
                complexity['time'] = 'O(n)'  # Conservative estimate
            
            # Space complexity heuristics
            if any(ds in source for ds in ['dict', 'list', 'set', 'defaultdict']):
                complexity['space'] = 'O(n)'
            
            if 'recursive' in source or method.__name__ in source:
                complexity['space'] = 'O(n)'  # Stack space
        
        except Exception:
            pass
        
        return complexity
    
    def _extract_loop_patterns(self, tree) -> List[str]:
        """Extract loop patterns from AST."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                patterns.append("for_loop")
                # Check for nested loops
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        patterns.append("nested_loop")
                        break
            
            elif isinstance(node, ast.While):
                patterns.append("while_loop")
            
            elif isinstance(node, ast.ListComp):
                patterns.append("list_comprehension")
        
        return list(set(patterns))
    
    def _extract_conditional_patterns(self, tree) -> List[str]:
        """Extract conditional patterns from AST."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                patterns.append("conditional")
                
                # Check for elif chains
                current = node
                elif_count = 0
                while hasattr(current, 'orelse') and current.orelse:
                    if (len(current.orelse) == 1 and 
                        isinstance(current.orelse[0], ast.If)):
                        elif_count += 1
                        current = current.orelse[0]
                    else:
                        break
                
                if elif_count > 0:
                    patterns.append(f"elif_chain_{elif_count}")
                
                if current.orelse and not isinstance(current.orelse[0], ast.If):
                    patterns.append("if_else")
            
            elif isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    patterns.append("logical_and")
                elif isinstance(node.op, ast.Or):
                    patterns.append("logical_or")
        
        return list(set(patterns))
    
    def _extract_data_structure_patterns(self, tree) -> List[str]:
        """Extract data structure usage patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in ['dict', 'defaultdict']:
                    patterns.append("dictionary_usage")
                elif node.id in ['list', 'deque']:
                    patterns.append("list_usage")
                elif node.id in ['set']:
                    patterns.append("set_usage")
                elif node.id in ['heapq']:
                    patterns.append("priority_queue")
            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name in ['append', 'extend']:
                        patterns.append("list_append_pattern")
                    elif method_name in ['add', 'update']:
                        patterns.append("set_update_pattern")
                    elif method_name in ['get', 'setdefault']:
                        patterns.append("dict_access_pattern")
        
        return list(set(patterns))
    
    def _extract_algorithm_patterns(self, tree) -> List[str]:
        """Extract algorithmic patterns."""
        patterns = []
        
        # Look for common algorithmic patterns in variable names and comments
        source_text = ast.unparse(tree).lower() if hasattr(ast, 'unparse') else ''
        
        algorithmic_keywords = {
            'bfs': 'breadth_first_search',
            'dfs': 'depth_first_search',
            'dynamic': 'dynamic_programming',
            'memo': 'memoization',
            'cache': 'caching',
            'recursive': 'recursion',
            'iterative': 'iteration',
            'greedy': 'greedy_algorithm',
            'divide': 'divide_and_conquer',
            'backtrack': 'backtracking'
        }
        
        for keyword, pattern in algorithmic_keywords.items():
            if keyword in source_text:
                patterns.append(pattern)
        
        return list(set(patterns))
    
    def _extract_optimization_patterns(self, tree) -> List[str]:
        """Extract optimization patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            # Memoization pattern
            if (isinstance(node, ast.Assign) and 
                any('cache' in target.id for target in node.targets 
                    if isinstance(target, ast.Name))):
                patterns.append("memoization")
            
            # Early return pattern
            if isinstance(node, ast.Return) and isinstance(node.value, ast.IfExp):
                patterns.append("early_return")
            
            # Loop optimization patterns
            if isinstance(node, ast.For):
                # Check for range optimization
                if (isinstance(node.iter, ast.Call) and
                    isinstance(node.iter.func, ast.Name) and
                    node.iter.func.id == 'range'):
                    patterns.append("range_optimization")
            
            # Set/dict lookup optimization
            if (isinstance(node, ast.Compare) and
                any(isinstance(op, ast.In) for op in node.ops)):
                patterns.append("membership_test_optimization")
        
        return list(set(patterns))
    
    def _extract_complexity_patterns(self, tree) -> List[str]:
        """Extract complexity-related patterns."""
        patterns = []
        
        # Count nested structures
        max_nesting = self._calculate_max_nesting(tree)
        if max_nesting > 1:
            patterns.append(f"nested_structure_depth_{max_nesting}")
        
        # Identify recursive patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for child in ast.walk(node):
                    if (isinstance(child, ast.Call) and
                        isinstance(child.func, ast.Name) and
                        child.func.id == func_name):
                        patterns.append("recursive_call")
                        break
        
        return patterns
    
    def _extract_data_flow_patterns(self, tree) -> List[str]:
        """Extract data flow patterns."""
        patterns = []
        
        # Track variable assignments and usage
        assignments = set()
        usages = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.add(target.id)
            
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                usages.add(node.id)
        
        # Identify data flow patterns
        if len(assignments) > len(usages):
            patterns.append("data_accumulation")
        elif len(usages) > len(assignments) * 2:
            patterns.append("data_intensive_processing")
        
        return patterns
    
    def _calculate_max_nesting(self, tree) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(node, (ast.For, ast.While, ast.If, ast.With)):
                for child in ast.iter_child_nodes(node):
                    calculate_depth(child, current_depth + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    calculate_depth(child, current_depth)
        
        calculate_depth(tree)
        return max_depth
    
    def _identify_edge_cases(self, tree) -> List[EdgeCase]:
        """Identify edge cases in the code."""
        edge_cases = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Look for edge case conditions
                condition_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
                
                edge_case_patterns = [
                    ('empty', ['not ', 'len(', '== 0', 'is None']),
                    ('boundary', ['<= ', '>= ', '== 0', '== 1']),
                    ('null_check', ['is None', '== None', 'not ']),
                    ('size_check', ['len(', 'size', 'count'])
                ]
                
                for case_type, patterns in edge_case_patterns:
                    if any(pattern in condition_str for pattern in patterns):
                        edge_cases.append(EdgeCase(
                            case_id=f"{case_type}_{len(edge_cases)}",
                            description=f"Handles {case_type} condition",
                            algorithm="unknown",
                            input_conditions=[condition_str],
                            handling_strategy="conditional_check",
                            code_snippet=condition_str,
                            importance_level=3
                        ))
        
        return edge_cases
    
    def _identify_optimizations(self, tree) -> List[OptimizationTechnique]:
        """Identify optimization techniques used."""
        optimizations = []
        
        optimization_patterns = {
            'early_termination': ['return ', 'break', 'continue'],
            'caching': ['cache', 'memo', 'stored'],
            'lazy_evaluation': ['yield', 'generator'],
            'batch_processing': ['batch', 'chunk'],
            'precomputation': ['precompute', 'precalc']
        }
        
        source_str = ast.unparse(tree) if hasattr(ast, 'unparse') else str(tree)
        
        for opt_type, keywords in optimization_patterns.items():
            if any(keyword in source_str.lower() for keyword in keywords):
                optimizations.append(OptimizationTechnique(
                    technique_id=f"{opt_type}_{len(optimizations)}",
                    technique_name=opt_type.replace('_', ' ').title(),
                    description=f"Uses {opt_type} technique",
                    applicable_contexts=["general"],
                    performance_impact={'time_improvement': 0.2, 'space_improvement': 0.0},
                    implementation_examples=[],
                    trade_offs=[]
                ))
        
        return optimizations


class AlgorithmKnowledgeExtractor:
    """Extracts knowledge from hardcoded algorithms."""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.extracted_patterns = {}
        self.algorithm_signatures = {}
        self.edge_cases = []
        self.optimizations = []
        
        # Initialize algorithm mappings
        self.algorithm_classes = {
            'nfa_to_dfa': NFAToDFAConverter,
            'dfa_minimization': DFAMinimizer,
            'regex_conversion': RegexConverter,
            'cfg_processing': CFGProcessor,
            'parsing_algorithms': ParsingAlgorithms,
            'turing_machine': TuringMachine,
            'multi_tape_tm': MultiTapeTuringMachine,
            'universal_tm': UniversalTuringMachine,
            'slr_parser': SLRParser,
            'gnf_converter': GNFConverter
        }
    
    def extract_all_knowledge(self) -> Dict[str, Any]:
        """Extract knowledge from all hardcoded algorithms."""
        
        knowledge_base = {
            'algorithms': {},
            'patterns': {},
            'edge_cases': [],
            'optimizations': [],
            'cross_algorithm_insights': []
        }
        
        # Analyze each algorithm class
        for alg_name, alg_class in self.algorithm_classes.items():
            logger.info(f"Analyzing {alg_name}...")
            
            analysis = self.code_analyzer.analyze_class(alg_class)
            knowledge_base['algorithms'][alg_name] = analysis
            
            # Extract solution patterns
            patterns = self._extract_solution_patterns(alg_name, alg_class, analysis)
            knowledge_base['patterns'].update(patterns)
            
            # Extract edge cases
            edge_cases = self._extract_edge_cases(alg_name, analysis)
            knowledge_base['edge_cases'].extend(edge_cases)
            
            # Extract optimizations
            optimizations = self._extract_optimizations(alg_name, analysis)
            knowledge_base['optimizations'].extend(optimizations)
        
        # Generate cross-algorithm insights
        knowledge_base['cross_algorithm_insights'] = self._generate_cross_insights(
            knowledge_base
        )
        
        return knowledge_base
    
    def _extract_solution_patterns(
        self,
        algorithm_name: str,
        algorithm_class,
        analysis: Dict[str, Any]
    ) -> Dict[str, SolutionPattern]:
        """Extract solution patterns from an algorithm."""
        
        patterns = {}
        
        # Map algorithm to problem types and language patterns
        algorithm_mapping = {
            'nfa_to_dfa': {
                'problem_types': [ProblemType.DFA_CONSTRUCTION, ProblemType.NFA_CONSTRUCTION],
                'language_patterns': [LanguagePattern.REGULAR],
                'key_steps': [
                    'Compute epsilon closures',
                    'Apply subset construction',
                    'Build state transition table',
                    'Determine final states'
                ]
            },
            'dfa_minimization': {
                'problem_types': [ProblemType.DFA_CONSTRUCTION],
                'language_patterns': [LanguagePattern.REGULAR],
                'key_steps': [
                    'Remove unreachable states',
                    'Partition states by finality',
                    'Refine partitions iteratively',
                    'Build minimized automaton'
                ]
            },
            'regex_conversion': {
                'problem_types': [ProblemType.REGEX_CONVERSION],
                'language_patterns': [LanguagePattern.REGULAR],
                'key_steps': [
                    'Parse regex to postfix',
                    'Apply Thompson construction',
                    'Handle operators systematically',
                    'Manage epsilon transitions'
                ]
            },
            'cfg_processing': {
                'problem_types': [ProblemType.CFG_ANALYSIS],
                'language_patterns': [LanguagePattern.CONTEXT_FREE],
                'key_steps': [
                    'Remove epsilon productions',
                    'Remove unit productions',
                    'Remove useless symbols',
                    'Convert to normal form'
                ]
            },
            'parsing_algorithms': {
                'problem_types': [ProblemType.CFG_ANALYSIS],
                'language_patterns': [LanguagePattern.CONTEXT_FREE],
                'key_steps': [
                    'Compute FIRST sets',
                    'Compute FOLLOW sets',
                    'Build parsing table',
                    'Apply parsing algorithm'
                ]
            }
        }
        
        if algorithm_name in algorithm_mapping:
            mapping = algorithm_mapping[algorithm_name]
            
            pattern = SolutionPattern(
                pattern_id=f"{algorithm_name}_pattern",
                pattern_name=algorithm_name.replace('_', ' ').title(),
                problem_types=mapping['problem_types'],
                language_patterns=mapping['language_patterns'],
                algorithm_steps=mapping['key_steps'],
                key_insights=self._extract_key_insights(algorithm_name, analysis),
                common_pitfalls=self._extract_common_pitfalls(algorithm_name),
                optimization_opportunities=self._extract_optimization_opportunities(analysis),
                code_examples=self._extract_code_examples(algorithm_class),
                mathematical_properties=self._extract_mathematical_properties(algorithm_name),
                performance_characteristics=self._extract_performance_characteristics(analysis)
            )
            
            patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _extract_key_insights(self, algorithm_name: str, analysis: Dict[str, Any]) -> List[str]:
        """Extract key insights from algorithm analysis."""
        
        insights = []
        
        # Algorithm-specific insights
        algorithm_insights = {
            'nfa_to_dfa': [
                "Epsilon closures are fundamental for handling non-determinism",
                "Subset construction may lead to exponential state explosion",
                "State names in DFA reflect sets of NFA states",
                "Final states are determined by intersection with original final states"
            ],
            'dfa_minimization': [
                "Unreachable state removal is preprocessing step",
                "Hopcroft's algorithm uses partition refinement",
                "Equivalent states have identical future behavior",
                "Minimization preserves language recognition"
            ],
            'regex_conversion': [
                "Thompson construction creates epsilon-NFA",
                "Each regex operator has specific construction pattern",
                "Kleene star requires careful epsilon transition handling",
                "Union operations need new start/end states"
            ],
            'cfg_processing': [
                "CNF requires all productions to be A→BC or A→a",
                "Epsilon removal affects start symbol specially",
                "Unit production removal uses transitive closure",
                "Order of transformations matters for correctness"
            ]
        }
        
        if algorithm_name in algorithm_insights:
            insights.extend(algorithm_insights[algorithm_name])
        
        # Extract insights from code patterns
        if 'patterns' in analysis:
            for method_analysis in analysis['methods'].values():
                if 'patterns' in method_analysis:
                    patterns = method_analysis['patterns']
                    if 'nested_loop' in patterns:
                        insights.append("Algorithm uses nested iteration for completeness")
                    if 'memoization' in patterns:
                        insights.append("Caching is used to avoid redundant computation")
                    if 'early_return' in patterns:
                        insights.append("Early termination optimizes performance")
        
        return insights
    
    def _extract_common_pitfalls(self, algorithm_name: str) -> List[str]:
        """Extract common pitfalls for each algorithm."""
        
        pitfall_map = {
            'nfa_to_dfa': [
                "Forgetting to include initial state in epsilon closure",
                "Not handling empty alphabet correctly",
                "Missing epsilon transitions in closure computation",
                "Incorrect final state determination"
            ],
            'dfa_minimization': [
                "Not removing unreachable states first",
                "Incorrect partition refinement logic",
                "Missing edge cases in Hopcroft's algorithm",
                "Forgetting to update state references"
            ],
            'regex_conversion': [
                "Incorrect operator precedence handling",
                "Missing explicit concatenation operators",
                "Improper epsilon transition management",
                "Stack underflow in postfix evaluation"
            ],
            'cfg_processing': [
                "Incorrect epsilon production handling",
                "Missing special case for start symbol",
                "Wrong order of grammar transformations",
                "Not preserving language equivalence"
            ]
        }
        
        return pitfall_map.get(algorithm_name, [])
    
    def _extract_optimization_opportunities(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract optimization opportunities from analysis."""
        
        opportunities = []
        
        # Check for optimization patterns in methods
        for method_name, method_analysis in analysis.get('methods', {}).items():
            if 'optimizations' in method_analysis:
                for opt in method_analysis['optimizations']:
                    opportunities.append(f"{method_name}: {opt.description}")
        
        # General optimization opportunities
        if analysis.get('optimization_patterns'):
            opportunities.extend([
                "Consider memoization for repeated subproblems",
                "Early termination can reduce unnecessary computation",
                "Batch processing can improve cache performance",
                "Precomputation of common cases reduces runtime cost"
            ])
        
        return opportunities
    
    def _extract_code_examples(self, algorithm_class) -> List[str]:
        """Extract representative code examples."""
        
        examples = []
        
        try:
            # Get key method signatures
            for name, method in inspect.getmembers(algorithm_class, predicate=inspect.ismethod):
                if not name.startswith('_') and hasattr(method, '__doc__'):
                    sig = inspect.signature(method)
                    examples.append(f"{name}{sig}")
        except Exception as e:
            logger.warning(f"Could not extract code examples: {e}")
        
        return examples[:5]  # Limit to 5 examples
    
    def _extract_mathematical_properties(self, algorithm_name: str) -> Dict[str, Any]:
        """Extract mathematical properties of algorithms."""
        
        properties_map = {
            'nfa_to_dfa': {
                'time_complexity': 'O(2^n)',
                'space_complexity': 'O(2^n)',
                'deterministic': True,
                'preserves_language': True,
                'worst_case_states': 'exponential',
                'typical_case': 'much better than worst case'
            },
            'dfa_minimization': {
                'time_complexity': 'O(n log n)',
                'space_complexity': 'O(n)',
                'deterministic': True,
                'preserves_language': True,
                'optimality': 'produces unique minimal DFA',
                'algorithm': 'Hopcroft partition refinement'
            },
            'regex_conversion': {
                'time_complexity': 'O(n)',
                'space_complexity': 'O(n)',
                'deterministic': True,
                'construction': 'Thompson construction',
                'epsilon_transitions': 'creates epsilon-NFA',
                'states_per_operator': 'constant number of states'
            }
        }
        
        return properties_map.get(algorithm_name, {})
    
    def _extract_performance_characteristics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance characteristics from analysis."""
        
        characteristics = {}
        
        # Extract complexity information
        for method_name, method_analysis in analysis.get('methods', {}).items():
            if 'complexity' in method_analysis:
                characteristics[f"{method_name}_complexity"] = method_analysis['complexity']
        
        # Add general characteristics
        characteristics.update({
            'scalability': 'depends on input size',
            'memory_usage': 'proportional to state space',
            'parallelizable': False,
            'incremental': False
        })
        
        return characteristics
    
    def _extract_edge_cases(self, algorithm_name: str, analysis: Dict[str, Any]) -> List[EdgeCase]:
        """Extract edge cases from algorithm analysis."""
        
        edge_cases = []
        
        # Algorithm-specific edge cases
        edge_case_map = {
            'nfa_to_dfa': [
                EdgeCase(
                    case_id=f"{algorithm_name}_empty_alphabet",
                    description="Empty alphabet handling",
                    algorithm=algorithm_name,
                    input_conditions=["alphabet is empty"],
                    handling_strategy="return automaton with no transitions",
                    code_snippet="if not alphabet: return empty_automaton",
                    importance_level=4
                ),
                EdgeCase(
                    case_id=f"{algorithm_name}_no_initial_state",
                    description="Missing initial state",
                    algorithm=algorithm_name,
                    input_conditions=["initial_state is None"],
                    handling_strategy="raise ValueError",
                    code_snippet="if not initial_state: raise ValueError",
                    importance_level=5
                )
            ],
            'dfa_minimization': [
                EdgeCase(
                    case_id=f"{algorithm_name}_single_state",
                    description="DFA with single state",
                    algorithm=algorithm_name,
                    input_conditions=["len(states) == 1"],
                    handling_strategy="return original DFA",
                    code_snippet="if len(states) == 1: return dfa",
                    importance_level=3
                )
            ]
        }
        
        if algorithm_name in edge_case_map:
            edge_cases.extend(edge_case_map[algorithm_name])
        
        # Extract edge cases from code analysis
        for method_name, method_analysis in analysis.get('methods', {}).items():
            if 'edge_cases' in method_analysis:
                for edge_case in method_analysis['edge_cases']:
                    edge_case.algorithm = algorithm_name
                    edge_cases.append(edge_case)
        
        return edge_cases
    
    def _extract_optimizations(
        self,
        algorithm_name: str,
        analysis: Dict[str, Any]
    ) -> List[OptimizationTechnique]:
        """Extract optimization techniques from algorithm analysis."""
        
        optimizations = []
        
        # Extract from code analysis
        for method_name, method_analysis in analysis.get('methods', {}).items():
            if 'optimizations' in method_analysis:
                optimizations.extend(method_analysis['optimizations'])
        
        # Add algorithm-specific optimizations
        algorithm_optimizations = {
            'nfa_to_dfa': [
                OptimizationTechnique(
                    technique_id=f"{algorithm_name}_epsilon_closure_caching",
                    technique_name="Epsilon Closure Caching",
                    description="Cache epsilon closures to avoid recomputation",
                    applicable_contexts=["subset construction"],
                    performance_impact={'time_improvement': 0.3, 'space_improvement': -0.1},
                    implementation_examples=["self.epsilon_closures[state] = closure"],
                    trade_offs=["Uses more memory for better time complexity"]
                )
            ],
            'dfa_minimization': [
                OptimizationTechnique(
                    technique_id=f"{algorithm_name}_unreachable_removal",
                    technique_name="Unreachable State Removal",
                    description="Remove unreachable states before minimization",
                    applicable_contexts=["preprocessing"],
                    performance_impact={'time_improvement': 0.2, 'space_improvement': 0.2},
                    implementation_examples=["reachable_dfa = self._remove_unreachable_states()"],
                    trade_offs=["Additional preprocessing step"]
                )
            ]
        }
        
        if algorithm_name in algorithm_optimizations:
            optimizations.extend(algorithm_optimizations[algorithm_name])
        
        return optimizations
    
    def _generate_cross_insights(self, knowledge_base: Dict[str, Any]) -> List[str]:
        """Generate insights across multiple algorithms."""
        
        insights = []
        
        # Analyze common patterns across algorithms
        all_patterns = set()
        for alg_analysis in knowledge_base['algorithms'].values():
            for method_analysis in alg_analysis.get('methods', {}).values():
                all_patterns.update(method_analysis.get('patterns', []))
        
        # Common pattern insights
        if 'nested_loop' in all_patterns:
            insights.append("Many algorithms use nested iteration for complete exploration")
        
        if 'memoization' in all_patterns:
            insights.append("Caching is a common optimization across different algorithms")
        
        if 'early_return' in all_patterns:
            insights.append("Early termination patterns improve efficiency across algorithms")
        
        # Cross-algorithm optimization opportunities
        insights.extend([
            "State space exploration algorithms benefit from visited state tracking",
            "Iterative deepening can replace recursion in many contexts",
            "Batch processing improves performance for multiple similar operations",
            "Preprocessing (like unreachable state removal) often improves main algorithm",
            "Error checking and validation patterns are consistent across algorithms"
        ])
        
        return insights
    
    def generate_training_data(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training data for AI models from extracted knowledge."""
        
        training_data = {
            'problem_solution_pairs': [],
            'strategy_examples': [],
            'edge_case_handlers': [],
            'optimization_patterns': [],
            'verification_rules': []
        }
        
        # Generate problem-solution pairs
        for pattern_id, pattern in knowledge_base['patterns'].items():
            for problem_type in pattern.problem_types:
                training_example = {
                    'problem_type': problem_type.value,
                    'language_patterns': [lp.value for lp in pattern.language_patterns],
                    'solution_strategy': pattern.algorithm_steps,
                    'key_insights': pattern.key_insights,
                    'expected_complexity': pattern.performance_characteristics
                }
                training_data['problem_solution_pairs'].append(training_example)
        
        # Generate strategy examples
        for alg_name, alg_analysis in knowledge_base['algorithms'].items():
            strategy_example = {
                'algorithm': alg_name,
                'approach': 'deterministic',
                'steps': alg_analysis.get('complexity_patterns', []),
                'patterns': alg_analysis.get('optimization_patterns', [])
            }
            training_data['strategy_examples'].append(strategy_example)
        
        # Generate edge case handlers
        for edge_case in knowledge_base['edge_cases']:
            handler = {
                'condition': edge_case.input_conditions,
                'strategy': edge_case.handling_strategy,
                'importance': edge_case.importance_level,
                'code_pattern': edge_case.code_snippet
            }
            training_data['edge_case_handlers'].append(handler)
        
        # Generate optimization patterns
        for optimization in knowledge_base['optimizations']:
            opt_pattern = {
                'technique': optimization.technique_name,
                'context': optimization.applicable_contexts,
                'impact': optimization.performance_impact,
                'trade_offs': optimization.trade_offs
            }
            training_data['optimization_patterns'].append(opt_pattern)
        
        return training_data
    
    def save_knowledge_base(self, knowledge_base: Dict[str, Any], file_path: str):
        """Save extracted knowledge to file."""
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_kb = {}
        
        for key, value in knowledge_base.items():
            if key == 'patterns':
                serializable_kb[key] = {
                    pattern_id: {
                        'pattern_id': pattern.pattern_id,
                        'pattern_name': pattern.pattern_name,
                        'problem_types': [pt.value for pt in pattern.problem_types],
                        'language_patterns': [lp.value for lp in pattern.language_patterns],
                        'algorithm_steps': pattern.algorithm_steps,
                        'key_insights': pattern.key_insights,
                        'common_pitfalls': pattern.common_pitfalls,
                        'optimization_opportunities': pattern.optimization_opportunities,
                        'code_examples': pattern.code_examples,
                        'mathematical_properties': pattern.mathematical_properties,
                        'performance_characteristics': pattern.performance_characteristics
                    }
                    for pattern_id, pattern in value.items()
                }
            elif key == 'edge_cases':
                serializable_kb[key] = [
                    {
                        'case_id': case.case_id,
                        'description': case.description,
                        'algorithm': case.algorithm,
                        'input_conditions': case.input_conditions,
                        'handling_strategy': case.handling_strategy,
                        'code_snippet': case.code_snippet,
                        'importance_level': case.importance_level
                    }
                    for case in value
                ]
            elif key == 'optimizations':
                serializable_kb[key] = [
                    {
                        'technique_id': opt.technique_id,
                        'technique_name': opt.technique_name,
                        'description': opt.description,
                        'applicable_contexts': opt.applicable_contexts,
                        'performance_impact': opt.performance_impact,
                        'implementation_examples': opt.implementation_examples,
                        'trade_offs': opt.trade_offs
                    }
                    for opt in value
                ]
            else:
                serializable_kb[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(serializable_kb, f, indent=2)
        
        logger.info(f"Knowledge base saved to {file_path}")


# Global knowledge extractor instance
knowledge_extractor = AlgorithmKnowledgeExtractor()


def extract_hardcoded_knowledge() -> Dict[str, Any]:
    """
    Convenience function to extract all hardcoded knowledge.
    
    Returns:
        Comprehensive knowledge base from hardcoded algorithms
    """
    return knowledge_extractor.extract_all_knowledge()


def generate_ai_training_data() -> Dict[str, Any]:
    """
    Generate training data for AI models from hardcoded algorithms.
    
    Returns:
        Training data structured for AI model improvement
    """
    knowledge_base = extract_hardcoded_knowledge()
    return knowledge_extractor.generate_training_data(knowledge_base)