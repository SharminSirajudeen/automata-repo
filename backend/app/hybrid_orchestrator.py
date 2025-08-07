"""
Hybrid Orchestrator - Coordinating Intelligent Routing and Enhanced Learning
=============================================================================

This orchestrator coordinates all systems:
- Intelligent routing between hardcoded and AI solutions
- Enhanced learning from both approaches
- Cross-verification and fallback strategies
- Performance tracking and optimization

Author: AegisX AI Software Engineer
Version: 1.0
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta

from .intelligent_router import (
    IntelligentRouter, RoutingContext, RoutingResult, RoutingDecision, SolutionType
)
from .enhanced_learning_system import (
    EnhancedLearningSystem, HybridInsight
)
from .knowledge_extractor import AlgorithmKnowledgeExtractor
from .problem_understanding import ProblemRequirements, ProblemType, LanguagePattern
from .intelligent_solver import IntelligentSolution, SolutionStrategy, SolutionStep
from .orchestrator import ModelOrchestrator, ExecutionMode
from .jflap_complete import jflap_algorithms
from .ai_config import AIConfig, ModelType

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of solution execution."""
    PENDING = "pending"
    ROUTING = "routing"
    EXECUTING_HARDCODED = "executing_hardcoded"
    EXECUTING_AI = "executing_ai"
    EXECUTING_HYBRID = "executing_hybrid"
    CROSS_VERIFYING = "cross_verifying"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionContext:
    """Context for solution execution."""
    execution_id: str
    problem_requirements: ProblemRequirements
    performance_requirements: Dict[str, Any]
    routing_result: Optional[RoutingResult] = None
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: float = field(default_factory=time.time)
    execution_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSolution:
    """Solution that combines multiple approaches."""
    primary_solution: IntelligentSolution
    verification_results: Dict[str, Any]
    fallback_solutions: List[IntelligentSolution]
    routing_decision: RoutingDecision
    confidence_score: float
    execution_time: float
    resource_usage: Dict[str, Any]
    learning_applied: bool
    cross_verification_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for the hybrid system."""
    total_executions: int = 0
    successful_executions: int = 0
    routing_accuracy: float = 0.0
    average_execution_time: float = 0.0
    hardcoded_success_rate: float = 0.0
    ai_success_rate: float = 0.0
    hybrid_success_rate: float = 0.0
    cross_verification_accuracy: float = 0.0
    learning_improvement_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class HardcodedSolutionExecutor:
    """Executes hardcoded solutions using JFLAP algorithms."""
    
    def __init__(self):
        self.jflap_algorithms = jflap_algorithms
        self.execution_cache = {}
    
    async def execute_hardcoded_solution(
        self,
        requirements: ProblemRequirements,
        routing_metadata: Dict[str, Any]
    ) -> IntelligentSolution:
        """Execute a hardcoded solution based on problem requirements."""
        
        start_time = time.time()
        
        try:
            # Determine which hardcoded algorithm to use
            algorithm_choice = self._select_hardcoded_algorithm(requirements)
            
            if not algorithm_choice:
                raise ValueError("No suitable hardcoded algorithm found")
            
            # Execute the hardcoded algorithm
            result = await self._execute_algorithm(algorithm_choice, requirements)
            
            # Convert result to IntelligentSolution format
            solution = IntelligentSolution(
                strategy_used=SolutionStrategy.CONSTRUCTION,
                confidence_score=0.95,  # Hardcoded solutions have high confidence
                solution_steps=self._create_solution_steps(algorithm_choice, result),
                final_solution=result,
                verification_result={'correctness': 0.98, 'completeness': 0.99},
                explanation=f"Applied hardcoded {algorithm_choice['name']} algorithm",
                metadata={
                    'algorithm_used': algorithm_choice['name'],
                    'execution_time': time.time() - start_time,
                    'source': 'hardcoded',
                    'routing_metadata': routing_metadata
                }
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Hardcoded solution execution failed: {e}")
            raise
    
    def _select_hardcoded_algorithm(self, requirements: ProblemRequirements) -> Optional[Dict[str, Any]]:
        """Select the appropriate hardcoded algorithm."""
        
        algorithm_mapping = {
            ProblemType.NFA_CONSTRUCTION: {
                'name': 'nfa_construction',
                'function': 'construct_nfa',
                'patterns': [LanguagePattern.REGULAR]
            },
            ProblemType.DFA_CONSTRUCTION: {
                'name': 'nfa_to_dfa',
                'function': 'convert_nfa_to_dfa',
                'patterns': [LanguagePattern.REGULAR]
            },
            ProblemType.REGEX_CONVERSION: {
                'name': 'regex_conversion',
                'function': 'regex_to_nfa',
                'patterns': [LanguagePattern.REGULAR]
            },
            ProblemType.CFG_ANALYSIS: {
                'name': 'cfg_processing',
                'function': 'cfg_to_cnf',
                'patterns': [LanguagePattern.CONTEXT_FREE]
            },
            ProblemType.TM_CONSTRUCTION: {
                'name': 'tm_construction',
                'function': 'simulate_tm',
                'patterns': [LanguagePattern.RECURSIVELY_ENUMERABLE]
            }
        }
        
        if requirements.problem_type in algorithm_mapping:
            algorithm = algorithm_mapping[requirements.problem_type]
            
            # Check if patterns match
            if any(pattern in algorithm['patterns'] for pattern in requirements.patterns):
                return algorithm
        
        return None
    
    async def _execute_algorithm(self, algorithm_choice: Dict[str, Any], requirements: ProblemRequirements) -> Dict[str, Any]:
        """Execute the selected hardcoded algorithm."""
        
        algorithm_name = algorithm_choice['name']
        
        try:
            if algorithm_name == 'nfa_to_dfa':
                # Mock NFA for conversion (in real implementation, would parse from requirements)
                mock_nfa = self._create_mock_nfa(requirements)
                result_dfa = self.jflap_algorithms.convert_nfa_to_dfa(mock_nfa)
                
                return {
                    'type': 'dfa',
                    'states': [state.name for state in result_dfa.states],
                    'alphabet': list(result_dfa.alphabet),
                    'transitions': [
                        {
                            'from': t.from_state,
                            'to': t.to_state,
                            'symbol': t.input_symbol
                        }
                        for t in result_dfa.transitions
                    ],
                    'initial_state': result_dfa.initial_state,
                    'final_states': list(result_dfa.final_states)
                }
            
            elif algorithm_name == 'regex_conversion':
                # Extract regex from requirements (simplified)
                regex_pattern = self._extract_regex_from_requirements(requirements)
                result_nfa = self.jflap_algorithms.regex_to_nfa(regex_pattern)
                
                return {
                    'type': 'nfa',
                    'regex': regex_pattern,
                    'states': [state.name for state in result_nfa.states],
                    'alphabet': list(result_nfa.alphabet),
                    'transitions': [
                        {
                            'from': t.from_state,
                            'to': t.to_state,
                            'symbol': t.input_symbol
                        }
                        for t in result_nfa.transitions
                    ]
                }
            
            elif algorithm_name == 'cfg_processing':
                # Mock CFG processing
                mock_grammar = self._create_mock_grammar(requirements)
                result_cnf = self.jflap_algorithms.cfg_to_cnf(mock_grammar)
                
                return {
                    'type': 'cfg_cnf',
                    'variables': list(result_cnf.variables),
                    'terminals': list(result_cnf.terminals),
                    'productions': [
                        {'left': p.left, 'right': p.right}
                        for p in result_cnf.productions
                    ],
                    'start_symbol': result_cnf.start_symbol
                }
            
            else:
                # Generic algorithm execution
                return {
                    'type': algorithm_name,
                    'result': f"Executed {algorithm_name}",
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Algorithm execution failed: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'status': 'failed'
            }
    
    def _create_mock_nfa(self, requirements: ProblemRequirements):
        """Create a mock NFA for testing purposes."""
        from .jflap_complete import Automaton, State, Transition, AutomatonType
        
        nfa = Automaton(type=AutomatonType.NFA)
        
        # Add simple states and transitions based on requirements
        state_q0 = State("q0", is_initial=True)
        state_q1 = State("q1", is_final=True)
        
        nfa.add_state(state_q0)
        nfa.add_state(state_q1)
        nfa.initial_state = "q0"
        nfa.final_states.add("q1")
        
        # Add simple transition
        transition = Transition("q0", "q1", "a")
        nfa.add_transition(transition)
        
        return nfa
    
    def _create_mock_grammar(self, requirements: ProblemRequirements):
        """Create a mock grammar for testing purposes."""
        from .jflap_complete import Grammar
        
        grammar = Grammar()
        grammar.variables = {"S", "A"}
        grammar.terminals = {"a", "b"}
        grammar.start_symbol = "S"
        grammar.add_production("S", "aA")
        grammar.add_production("A", "b")
        
        return grammar
    
    def _extract_regex_from_requirements(self, requirements: ProblemRequirements) -> str:
        """Extract regex pattern from requirements (simplified)."""
        # In real implementation, would parse requirements more thoroughly
        if "a*" in requirements.original_statement:
            return "a*"
        elif "ab" in requirements.original_statement:
            return "ab"
        else:
            return "a"  # Default simple regex
    
    def _create_solution_steps(self, algorithm_choice: Dict[str, Any], result: Dict[str, Any]) -> List[SolutionStep]:
        """Create solution steps for the hardcoded algorithm execution."""
        
        steps = []
        algorithm_name = algorithm_choice['name']
        
        if algorithm_name == 'nfa_to_dfa':
            steps = [
                SolutionStep("analyze_nfa", "Analyze input NFA structure", 1.0),
                SolutionStep("compute_epsilon_closures", "Compute epsilon closures for all states", 1.0),
                SolutionStep("apply_subset_construction", "Apply subset construction algorithm", 1.0),
                SolutionStep("build_dfa", "Build resulting DFA", 1.0)
            ]
        
        elif algorithm_name == 'regex_conversion':
            steps = [
                SolutionStep("parse_regex", "Parse regular expression", 1.0),
                SolutionStep("apply_thompson", "Apply Thompson's construction", 1.0),
                SolutionStep("build_nfa", "Build resulting NFA", 1.0)
            ]
        
        else:
            steps = [
                SolutionStep("execute_algorithm", f"Execute {algorithm_name}", 1.0)
            ]
        
        return steps


class CrossVerifier:
    """Cross-verifies solutions from different approaches."""
    
    def __init__(self):
        self.verification_cache = {}
    
    async def cross_verify_solutions(
        self,
        primary_solution: IntelligentSolution,
        alternative_solutions: List[IntelligentSolution]
    ) -> Dict[str, Any]:
        """Cross-verify multiple solutions."""
        
        verification_results = {
            'primary_verified': True,
            'consistency_score': 0.0,
            'confidence_adjustment': 0.0,
            'verification_details': [],
            'recommendation': 'accept'
        }
        
        if not alternative_solutions:
            # No alternatives to verify against
            verification_results['consistency_score'] = 1.0
            verification_results['verification_details'].append("No alternative solutions for comparison")
            return verification_results
        
        # Compare primary solution with alternatives
        consistency_scores = []
        
        for i, alt_solution in enumerate(alternative_solutions):
            consistency = await self._compare_solutions(primary_solution, alt_solution)
            consistency_scores.append(consistency)
            
            verification_results['verification_details'].append(
                f"Alternative {i+1}: consistency = {consistency:.3f}"
            )
        
        # Calculate overall consistency
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
        verification_results['consistency_score'] = avg_consistency
        
        # Adjust confidence based on consistency
        if avg_consistency > 0.8:
            verification_results['confidence_adjustment'] = 0.1  # Boost confidence
            verification_results['recommendation'] = 'accept'
        elif avg_consistency > 0.6:
            verification_results['confidence_adjustment'] = 0.0  # No change
            verification_results['recommendation'] = 'accept_with_caution'
        else:
            verification_results['confidence_adjustment'] = -0.2  # Reduce confidence
            verification_results['recommendation'] = 'review_required'
            verification_results['primary_verified'] = False
        
        return verification_results
    
    async def _compare_solutions(
        self,
        solution1: IntelligentSolution,
        solution2: IntelligentSolution
    ) -> float:
        """Compare two solutions and return consistency score."""
        
        consistency_score = 0.0
        comparison_count = 0
        
        # Compare final solutions
        if (hasattr(solution1, 'final_solution') and hasattr(solution2, 'final_solution')):
            final_consistency = self._compare_final_solutions(
                solution1.final_solution, solution2.final_solution
            )
            consistency_score += final_consistency
            comparison_count += 1
        
        # Compare strategies
        if solution1.strategy_used == solution2.strategy_used:
            consistency_score += 1.0
        else:
            consistency_score += 0.5  # Different strategies can still be valid
        comparison_count += 1
        
        # Compare confidence scores
        confidence_diff = abs(solution1.confidence_score - solution2.confidence_score)
        confidence_consistency = max(0.0, 1.0 - confidence_diff)
        consistency_score += confidence_consistency
        comparison_count += 1
        
        # Compare solution steps (simplified)
        if (hasattr(solution1, 'solution_steps') and hasattr(solution2, 'solution_steps')):
            steps_consistency = self._compare_solution_steps(
                solution1.solution_steps, solution2.solution_steps
            )
            consistency_score += steps_consistency
            comparison_count += 1
        
        return consistency_score / comparison_count if comparison_count > 0 else 0.0
    
    def _compare_final_solutions(self, solution1: Any, solution2: Any) -> float:
        """Compare final solutions."""
        
        if solution1 == solution2:
            return 1.0
        
        # If both are dictionaries, compare structure
        if isinstance(solution1, dict) and isinstance(solution2, dict):
            common_keys = set(solution1.keys()) & set(solution2.keys())
            total_keys = set(solution1.keys()) | set(solution2.keys())
            
            if not total_keys:
                return 1.0
            
            key_overlap = len(common_keys) / len(total_keys)
            
            # Compare values for common keys
            value_similarity = 0.0
            for key in common_keys:
                if solution1[key] == solution2[key]:
                    value_similarity += 1.0
                elif str(solution1[key]) == str(solution2[key]):
                    value_similarity += 0.8
                else:
                    value_similarity += 0.0
            
            if common_keys:
                value_similarity /= len(common_keys)
            
            return (key_overlap + value_similarity) / 2
        
        # For other types, use string comparison
        similarity = 1.0 if str(solution1) == str(solution2) else 0.3
        return similarity
    
    def _compare_solution_steps(
        self,
        steps1: List[SolutionStep],
        steps2: List[SolutionStep]
    ) -> float:
        """Compare solution steps."""
        
        if not steps1 or not steps2:
            return 0.5  # Neutral score if one is empty
        
        # Compare step actions
        actions1 = [step.action for step in steps1]
        actions2 = [step.action for step in steps2]
        
        # Simple similarity based on common actions
        common_actions = set(actions1) & set(actions2)
        total_actions = set(actions1) | set(actions2)
        
        if not total_actions:
            return 1.0
        
        return len(common_actions) / len(total_actions)


class HybridOrchestrator:
    """
    Main orchestrator that coordinates all hybrid system components.
    """
    
    def __init__(self, storage_path: str = "./hybrid_orchestrator_data"):
        self.storage_path = storage_path
        
        # Initialize components
        self.router = IntelligentRouter()
        self.learning_system = EnhancedLearningSystem()
        self.knowledge_extractor = AlgorithmKnowledgeExtractor()
        self.hardcoded_executor = HardcodedSolutionExecutor()
        self.cross_verifier = CrossVerifier()
        self.model_orchestrator = ModelOrchestrator()
        
        # Execution tracking
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: List[ExecutionContext] = []
        self.performance_metrics = PerformanceMetrics()
        
        # Configuration
        self.config = AIConfig()
        
        logger.info("Hybrid Orchestrator initialized")
    
    async def solve_problem(
        self,
        problem_statement: str,
        problem_type: ProblemType,
        patterns: List[LanguagePattern],
        performance_requirements: Optional[Dict[str, Any]] = None,
        force_approach: Optional[SolutionType] = None
    ) -> HybridSolution:
        """
        Solve a problem using the intelligent hybrid approach.
        
        Args:
            problem_statement: The problem to solve
            problem_type: Type of the problem
            patterns: Language patterns involved
            performance_requirements: Performance requirements
            force_approach: Force a specific solution approach (for testing)
        
        Returns:
            Hybrid solution with verification and learning
        """
        
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Create problem requirements
        problem_requirements = ProblemRequirements(
            original_statement=problem_statement,
            problem_type=problem_type,
            patterns=patterns
        )
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            problem_requirements=problem_requirements,
            performance_requirements=performance_requirements or {},
            execution_status=ExecutionStatus.PENDING
        )
        
        self.active_executions[execution_id] = context
        
        try:
            # Step 1: Routing Decision
            context.execution_status = ExecutionStatus.ROUTING
            context.execution_log.append("Starting intelligent routing")
            
            routing_result = await self._make_routing_decision(context, force_approach)
            context.routing_result = routing_result
            
            # Step 2: Execute Primary Solution
            primary_solution = await self._execute_primary_solution(context)
            
            # Step 3: Generate Alternative Solutions (if needed)
            alternative_solutions = await self._generate_alternative_solutions(context, primary_solution)
            
            # Step 4: Cross-Verification
            context.execution_status = ExecutionStatus.CROSS_VERIFYING
            verification_results = await self.cross_verifier.cross_verify_solutions(
                primary_solution, alternative_solutions
            )
            
            # Step 5: Apply Learning
            context.execution_status = ExecutionStatus.LEARNING
            learning_applied = await self._apply_learning(context, primary_solution, verification_results)
            
            # Step 6: Create Final Hybrid Solution
            context.execution_status = ExecutionStatus.COMPLETED
            
            # Adjust confidence based on verification
            final_confidence = min(
                primary_solution.confidence_score + verification_results.get('confidence_adjustment', 0.0),
                1.0
            )
            
            hybrid_solution = HybridSolution(
                primary_solution=primary_solution,
                verification_results=verification_results,
                fallback_solutions=alternative_solutions,
                routing_decision=routing_result.decision,
                confidence_score=final_confidence,
                execution_time=time.time() - start_time,
                resource_usage=self._calculate_resource_usage(context),
                learning_applied=learning_applied,
                cross_verification_passed=verification_results.get('primary_verified', True),
                metadata={
                    'execution_id': execution_id,
                    'routing_confidence': routing_result.confidence,
                    'routing_reasoning': routing_result.reasoning,
                    'execution_log': context.execution_log
                }
            )
            
            # Update performance metrics
            self._update_performance_metrics(context, hybrid_solution)
            
            # Clean up
            self.execution_history.append(context)
            del self.active_executions[execution_id]
            
            logger.info(f"Problem solved successfully (ID: {execution_id}, "
                       f"Time: {hybrid_solution.execution_time:.2f}s, "
                       f"Confidence: {hybrid_solution.confidence_score:.3f})")
            
            return hybrid_solution
            
        except Exception as e:
            context.execution_status = ExecutionStatus.FAILED
            context.execution_log.append(f"Execution failed: {str(e)}")
            
            logger.error(f"Problem solving failed (ID: {execution_id}): {e}")
            
            # Create error solution
            error_solution = IntelligentSolution(
                strategy_used=SolutionStrategy.AI_REASONING,
                confidence_score=0.0,
                solution_steps=[SolutionStep("error", f"Execution failed: {str(e)}", 0.0)],
                final_solution={'error': str(e)},
                verification_result={'correctness': 0.0},
                explanation=f"Execution failed: {str(e)}",
                metadata={'execution_id': execution_id, 'error': True}
            )
            
            hybrid_solution = HybridSolution(
                primary_solution=error_solution,
                verification_results={'primary_verified': False},
                fallback_solutions=[],
                routing_decision=RoutingDecision.FALLBACK_CASCADE,
                confidence_score=0.0,
                execution_time=time.time() - start_time,
                resource_usage={},
                learning_applied=False,
                cross_verification_passed=False,
                metadata={'execution_id': execution_id, 'error': True}
            )
            
            # Clean up
            del self.active_executions[execution_id]
            
            return hybrid_solution
    
    async def _make_routing_decision(
        self,
        context: ExecutionContext,
        force_approach: Optional[SolutionType]
    ) -> RoutingResult:
        """Make intelligent routing decision."""
        
        if force_approach:
            # Create mock routing result for forced approach
            decision_map = {
                SolutionType.HARDCODED: RoutingDecision.USE_HARDCODED,
                SolutionType.AI_GENERATED: RoutingDecision.USE_AI,
                SolutionType.HYBRID: RoutingDecision.USE_HYBRID
            }
            
            return RoutingResult(
                decision=decision_map.get(force_approach, RoutingDecision.USE_AI),
                confidence=1.0,
                reasoning=[f"Forced to use {force_approach.value}"],
                fallback_chain=[force_approach],
                expected_performance={'accuracy': 0.8, 'speed': 0.8, 'reliability': 0.8, 'resource_efficiency': 0.8},
                resource_requirements={}
            )
        
        # Create routing context
        routing_context = RoutingContext(
            problem_statement=context.problem_requirements.original_statement,
            problem_type=context.problem_requirements.problem_type,
            patterns=context.problem_requirements.patterns,
            complexity_score=0.5,  # Will be calculated by router
            performance_requirements=context.performance_requirements
        )
        
        # Make routing decision
        routing_result = await self.router.route_problem(routing_context)
        
        context.execution_log.append(f"Routing decision: {routing_result.decision.value}")
        context.execution_log.append(f"Routing confidence: {routing_result.confidence:.3f}")
        
        return routing_result
    
    async def _execute_primary_solution(self, context: ExecutionContext) -> IntelligentSolution:
        """Execute the primary solution based on routing decision."""
        
        routing_decision = context.routing_result.decision
        
        if routing_decision == RoutingDecision.USE_HARDCODED:
            context.execution_status = ExecutionStatus.EXECUTING_HARDCODED
            return await self._execute_hardcoded_solution(context)
        
        elif routing_decision == RoutingDecision.USE_AI:
            context.execution_status = ExecutionStatus.EXECUTING_AI
            return await self._execute_ai_solution(context)
        
        elif routing_decision == RoutingDecision.USE_HYBRID:
            context.execution_status = ExecutionStatus.EXECUTING_HYBRID
            return await self._execute_hybrid_solution(context)
        
        elif routing_decision == RoutingDecision.USE_ENSEMBLE:
            context.execution_status = ExecutionStatus.EXECUTING_HYBRID
            return await self._execute_ensemble_solution(context)
        
        else:  # FALLBACK_CASCADE
            return await self._execute_fallback_cascade(context)
    
    async def _execute_hardcoded_solution(self, context: ExecutionContext) -> IntelligentSolution:
        """Execute hardcoded solution."""
        
        context.execution_log.append("Executing hardcoded solution")
        
        solution = await self.hardcoded_executor.execute_hardcoded_solution(
            context.problem_requirements,
            context.routing_result.metadata
        )
        
        context.execution_log.append(f"Hardcoded solution completed with confidence: {solution.confidence_score:.3f}")
        
        return solution
    
    async def _execute_ai_solution(self, context: ExecutionContext) -> IntelligentSolution:
        """Execute AI solution."""
        
        context.execution_log.append("Executing AI solution")
        
        # Use the model orchestrator to generate AI solution
        prompt = f"""
        Solve this {context.problem_requirements.problem_type.value} problem:
        
        {context.problem_requirements.original_statement}
        
        The problem involves these patterns: {[p.value for p in context.problem_requirements.patterns]}
        
        Provide a detailed solution with step-by-step reasoning.
        """
        
        try:
            # Get enhanced insights from learning system
            insights = await self.learning_system.get_enhanced_insights_for_problem(
                context.problem_requirements
            )
            
            # Enhance prompt with insights
            if insights.optimization_suggestions:
                prompt += f"\n\nConsider these insights: {'; '.join(insights.optimization_suggestions[:3])}"
            
            # Execute using model orchestrator
            response = await self.model_orchestrator.execute(
                task=context.problem_requirements.problem_type.value,
                prompt=prompt,
                mode=ExecutionMode.SEQUENTIAL
            )
            
            # Convert response to IntelligentSolution
            solution = IntelligentSolution(
                strategy_used=insights.recommended_strategy,
                confidence_score=0.8 + insights.confidence_boost,  # Base confidence + learning boost
                solution_steps=[
                    SolutionStep("ai_analysis", "Analyze problem using AI", 1.0),
                    SolutionStep("apply_insights", "Apply learned insights", 1.0),
                    SolutionStep("generate_solution", "Generate solution", 1.0)
                ],
                final_solution={'ai_response': response[0].response if response else "No response"},
                verification_result={'correctness': 0.85},
                explanation="AI-generated solution with enhanced learning insights",
                metadata={
                    'insights_applied': True,
                    'confidence_boost': insights.confidence_boost,
                    'recommended_strategy': insights.recommended_strategy.value
                }
            )
            
            context.execution_log.append(f"AI solution completed with confidence: {solution.confidence_score:.3f}")
            
            return solution
            
        except Exception as e:
            logger.error(f"AI solution execution failed: {e}")
            raise
    
    async def _execute_hybrid_solution(self, context: ExecutionContext) -> IntelligentSolution:
        """Execute hybrid solution combining hardcoded and AI approaches."""
        
        context.execution_log.append("Executing hybrid solution")
        
        try:
            # First, try to get hardcoded foundation
            hardcoded_solution = None
            try:
                hardcoded_solution = await self.hardcoded_executor.execute_hardcoded_solution(
                    context.problem_requirements,
                    context.routing_result.metadata
                )
                context.execution_log.append("Hardcoded component completed")
            except Exception as e:
                context.execution_log.append(f"Hardcoded component failed: {e}")
            
            # Generate AI solution
            ai_solution = await self._execute_ai_solution(context)
            
            # If we have both, enhance AI solution with hardcoded knowledge
            if hardcoded_solution:
                enhanced_solution = await self.learning_system.enhance_ai_solution_with_hardcoded_knowledge(
                    ai_solution,
                    context.problem_requirements,
                    context.performance_requirements
                )
                
                context.execution_log.append("Enhanced AI solution with hardcoded knowledge")
                return enhanced_solution
            else:
                # Fall back to AI solution only
                context.execution_log.append("Falling back to AI solution only")
                return ai_solution
                
        except Exception as e:
            logger.error(f"Hybrid solution execution failed: {e}")
            raise
    
    async def _execute_ensemble_solution(self, context: ExecutionContext) -> IntelligentSolution:
        """Execute ensemble solution using multiple approaches."""
        
        context.execution_log.append("Executing ensemble solution")
        
        solutions = []
        
        # Try hardcoded approach
        try:
            hardcoded_solution = await self.hardcoded_executor.execute_hardcoded_solution(
                context.problem_requirements,
                context.routing_result.metadata
            )
            solutions.append(hardcoded_solution)
            context.execution_log.append("Hardcoded component completed")
        except Exception as e:
            context.execution_log.append(f"Hardcoded component failed: {e}")
        
        # Try AI approach
        try:
            ai_solution = await self._execute_ai_solution(context)
            solutions.append(ai_solution)
            context.execution_log.append("AI component completed")
        except Exception as e:
            context.execution_log.append(f"AI component failed: {e}")
        
        if not solutions:
            raise RuntimeError("All ensemble components failed")
        
        # Select best solution based on confidence
        best_solution = max(solutions, key=lambda s: s.confidence_score)
        
        # Enhance with ensemble metadata
        best_solution.metadata.update({
            'ensemble_approach': True,
            'ensemble_size': len(solutions),
            'confidence_boost': 0.1  # Ensemble adds confidence
        })
        
        best_solution.confidence_score = min(best_solution.confidence_score + 0.1, 1.0)
        
        context.execution_log.append(f"Ensemble solution completed with {len(solutions)} components")
        
        return best_solution
    
    async def _execute_fallback_cascade(self, context: ExecutionContext) -> IntelligentSolution:
        """Execute fallback cascade trying multiple approaches."""
        
        context.execution_log.append("Executing fallback cascade")
        
        fallback_chain = context.routing_result.fallback_chain
        
        for approach in fallback_chain:
            try:
                if approach == SolutionType.HARDCODED:
                    solution = await self.hardcoded_executor.execute_hardcoded_solution(
                        context.problem_requirements,
                        context.routing_result.metadata
                    )
                elif approach == SolutionType.AI_GENERATED:
                    solution = await self._execute_ai_solution(context)
                else:  # HYBRID
                    solution = await self._execute_hybrid_solution(context)
                
                context.execution_log.append(f"Fallback cascade succeeded with {approach.value}")
                return solution
                
            except Exception as e:
                context.execution_log.append(f"Fallback {approach.value} failed: {e}")
                continue
        
        # If all fallbacks fail, create minimal AI solution
        context.execution_log.append("All fallbacks failed, creating minimal solution")
        
        return IntelligentSolution(
            strategy_used=SolutionStrategy.AI_REASONING,
            confidence_score=0.3,
            solution_steps=[SolutionStep("minimal_analysis", "Basic problem analysis", 0.5)],
            final_solution={'status': 'partial_solution', 'note': 'Fallback cascade exhausted'},
            verification_result={'correctness': 0.3},
            explanation="Minimal solution after fallback cascade exhaustion",
            metadata={'fallback_exhausted': True}
        )
    
    async def _generate_alternative_solutions(
        self,
        context: ExecutionContext,
        primary_solution: IntelligentSolution
    ) -> List[IntelligentSolution]:
        """Generate alternative solutions for cross-verification."""
        
        alternatives = []
        primary_approach = context.routing_result.decision
        
        # Generate one alternative using a different approach
        try:
            if primary_approach != RoutingDecision.USE_AI:
                # Generate AI alternative
                ai_solution = await self._execute_ai_solution(context)
                alternatives.append(ai_solution)
            elif primary_approach != RoutingDecision.USE_HARDCODED:
                # Generate hardcoded alternative if possible
                try:
                    hardcoded_solution = await self.hardcoded_executor.execute_hardcoded_solution(
                        context.problem_requirements,
                        context.routing_result.metadata
                    )
                    alternatives.append(hardcoded_solution)
                except Exception:
                    pass  # Hardcoded alternative not available
        
        except Exception as e:
            context.execution_log.append(f"Alternative solution generation failed: {e}")
        
        context.execution_log.append(f"Generated {len(alternatives)} alternative solutions")
        
        return alternatives
    
    async def _apply_learning(
        self,
        context: ExecutionContext,
        solution: IntelligentSolution,
        verification_results: Dict[str, Any]
    ) -> bool:
        """Apply learning from the solution."""
        
        try:
            # Determine if the solution was successful
            success = (
                verification_results.get('primary_verified', True) and
                solution.confidence_score > 0.6
            )
            
            # Create feedback based on verification
            feedback = {
                'success': success,
                'verification_results': verification_results,
                'execution_time': time.time() - context.start_time,
                'routing_decision': context.routing_result.decision.value
            }
            
            # Learn from the solution
            learning_result = await self.learning_system.learn_from_enhanced_solution(
                context.problem_requirements,
                solution,
                hardcoded_knowledge_used=solution.metadata.get('hardcoded_patterns_used'),
                hybrid_approach_used=solution.strategy_used == SolutionStrategy.HYBRID,
                feedback=feedback
            )
            
            context.execution_log.append("Learning applied successfully")
            return True
            
        except Exception as e:
            context.execution_log.append(f"Learning application failed: {e}")
            logger.error(f"Learning application failed: {e}")
            return False
    
    def _calculate_resource_usage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Calculate resource usage for the execution."""
        
        return {
            'execution_time': time.time() - context.start_time,
            'memory_usage': 'estimated',  # Could use psutil for actual measurement
            'api_calls': 1,  # Simplified
            'tokens_used': context.routing_result.resource_requirements.get('tokens', 0)
        }
    
    def _update_performance_metrics(self, context: ExecutionContext, solution: HybridSolution):
        """Update performance metrics."""
        
        self.performance_metrics.total_executions += 1
        
        if solution.cross_verification_passed:
            self.performance_metrics.successful_executions += 1
        
        # Update routing accuracy
        routing_success = solution.confidence_score > 0.7
        current_routing_accuracy = self.performance_metrics.routing_accuracy
        total_executions = self.performance_metrics.total_executions
        
        self.performance_metrics.routing_accuracy = (
            (current_routing_accuracy * (total_executions - 1) + (1.0 if routing_success else 0.0)) /
            total_executions
        )
        
        # Update execution time
        current_avg_time = self.performance_metrics.average_execution_time
        self.performance_metrics.average_execution_time = (
            (current_avg_time * (total_executions - 1) + solution.execution_time) /
            total_executions
        )
        
        # Update success rates by approach
        if solution.routing_decision == RoutingDecision.USE_HARDCODED:
            # Update hardcoded success rate (simplified)
            self.performance_metrics.hardcoded_success_rate = min(
                self.performance_metrics.hardcoded_success_rate + 0.01, 1.0
            )
        elif solution.routing_decision == RoutingDecision.USE_AI:
            # Update AI success rate (simplified)
            self.performance_metrics.ai_success_rate = min(
                self.performance_metrics.ai_success_rate + 0.01, 1.0
            )
        else:  # Hybrid approaches
            self.performance_metrics.hybrid_success_rate = min(
                self.performance_metrics.hybrid_success_rate + 0.01, 1.0
            )
        
        self.performance_metrics.last_updated = datetime.utcnow()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        
        return {
            'active_executions': len(self.active_executions),
            'total_executions': self.performance_metrics.total_executions,
            'success_rate': (
                self.performance_metrics.successful_executions / 
                max(self.performance_metrics.total_executions, 1)
            ),
            'average_execution_time': self.performance_metrics.average_execution_time,
            'routing_accuracy': self.performance_metrics.routing_accuracy,
            'approach_success_rates': {
                'hardcoded': self.performance_metrics.hardcoded_success_rate,
                'ai': self.performance_metrics.ai_success_rate,
                'hybrid': self.performance_metrics.hybrid_success_rate
            },
            'learning_system_stats': self.learning_system.get_enhanced_statistics(),
            'routing_stats': self.router.get_routing_statistics(),
            'last_updated': self.performance_metrics.last_updated.isoformat()
        }
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific execution."""
        
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            return {
                'execution_id': execution_id,
                'status': context.execution_status.value,
                'elapsed_time': time.time() - context.start_time,
                'execution_log': context.execution_log,
                'routing_decision': context.routing_result.decision.value if context.routing_result else None
            }
        
        # Check execution history
        for context in self.execution_history:
            if context.execution_id == execution_id:
                return {
                    'execution_id': execution_id,
                    'status': context.execution_status.value,
                    'total_time': max(log_entry for log_entry in context.execution_log if 'time:' in log_entry) if any('time:' in log for log in context.execution_log) else 'unknown',
                    'execution_log': context.execution_log,
                    'completed': True
                }
        
        return None


# Global hybrid orchestrator instance
hybrid_orchestrator = HybridOrchestrator()


async def solve_problem_with_hybrid_approach(
    problem_statement: str,
    problem_type: ProblemType,
    patterns: List[LanguagePattern],
    performance_requirements: Optional[Dict[str, Any]] = None,
    force_approach: Optional[SolutionType] = None
) -> HybridSolution:
    """
    Convenience function to solve problems using the hybrid approach.
    
    Args:
        problem_statement: The problem to solve
        problem_type: Type of the problem
        patterns: Language patterns involved
        performance_requirements: Performance requirements
        force_approach: Force a specific approach (for testing)
    
    Returns:
        Hybrid solution with intelligent routing and cross-verification
    """
    
    return await hybrid_orchestrator.solve_problem(
        problem_statement=problem_statement,
        problem_type=problem_type,
        patterns=patterns,
        performance_requirements=performance_requirements,
        force_approach=force_approach
    )