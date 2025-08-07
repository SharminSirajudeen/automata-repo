"""
Intelligent Router System for Dynamic Decision Making
====================================================

This system intelligently routes problems between hardcoded and AI solutions
based on deep analysis of problem characteristics, performance requirements,
and historical success patterns.

Author: AegisX AI Software Engineer
Version: 1.0
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import pickle
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from .ai_config import AIConfig, ModelType
from .problem_understanding import ProblemRequirements, ProblemType, LanguagePattern
from .orchestrator import ModelOrchestrator, ExecutionMode

logger = logging.getLogger(__name__)


class SolutionType(str, Enum):
    """Types of solutions available."""
    HARDCODED = "hardcoded"
    AI_GENERATED = "ai_generated"
    HYBRID = "hybrid"
    VERIFICATION_ONLY = "verification_only"


class RoutingDecision(str, Enum):
    """Routing decision types."""
    USE_HARDCODED = "use_hardcoded"
    USE_AI = "use_ai"
    USE_HYBRID = "use_hybrid"
    USE_ENSEMBLE = "use_ensemble"
    FALLBACK_CASCADE = "fallback_cascade"


@dataclass
class RoutingContext:
    """Context for routing decisions."""
    problem_statement: str
    problem_type: ProblemType
    patterns: List[LanguagePattern]
    complexity_score: float
    performance_requirements: Dict[str, Any]
    time_constraints: Optional[float] = None
    accuracy_requirements: float = 0.85
    confidence_threshold: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    decision: RoutingDecision
    confidence: float
    reasoning: List[str]
    fallback_chain: List[SolutionType]
    expected_performance: Dict[str, float]
    resource_requirements: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for different solution types."""
    accuracy: float
    speed: float
    reliability: float
    resource_usage: float
    confidence: float
    success_count: int
    total_attempts: int
    average_time: float
    last_updated: float


class ProblemComplexityAnalyzer:
    """Analyzes problem complexity using multiple dimensions."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.complexity_weights = {
            'statement_length': 0.1,
            'mathematical_complexity': 0.2,
            'pattern_complexity': 0.25,
            'logical_depth': 0.2,
            'algorithmic_complexity': 0.25
        }
    
    def analyze_complexity(self, context: RoutingContext) -> Dict[str, float]:
        """Analyze problem complexity across multiple dimensions."""
        
        scores = {}
        
        # Statement length complexity
        scores['statement_length'] = min(len(context.problem_statement) / 1000.0, 1.0)
        
        # Mathematical complexity (count of mathematical terms/operators)
        math_terms = self._count_mathematical_terms(context.problem_statement)
        scores['mathematical_complexity'] = min(math_terms / 20.0, 1.0)
        
        # Pattern complexity
        scores['pattern_complexity'] = self._analyze_pattern_complexity(context.patterns)
        
        # Logical depth (nested constructions, conditionals)
        scores['logical_depth'] = self._analyze_logical_depth(context.problem_statement)
        
        # Algorithmic complexity based on problem type
        scores['algorithmic_complexity'] = self._get_algorithmic_complexity(context.problem_type)
        
        # Weighted overall complexity
        overall = sum(
            scores[dim] * weight 
            for dim, weight in self.complexity_weights.items()
        )
        scores['overall'] = overall
        
        return scores
    
    def _count_mathematical_terms(self, text: str) -> int:
        """Count mathematical terms and operators."""
        math_indicators = [
            'state', 'transition', 'accept', 'reject', 'language', 'grammar',
            'production', 'derivation', 'concatenation', 'union', 'intersection',
            'complement', 'kleene', 'regular', 'context-free', 'decidable',
            'algorithm', 'turing', 'machine', 'automaton', 'proof', 'lemma'
        ]
        
        count = 0
        text_lower = text.lower()
        for term in math_indicators:
            count += text_lower.count(term)
        
        return count
    
    def _analyze_pattern_complexity(self, patterns: List[LanguagePattern]) -> float:
        """Analyze complexity based on language patterns."""
        if not patterns:
            return 0.1
        
        pattern_weights = {
            LanguagePattern.FINITE: 0.2,
            LanguagePattern.REGULAR: 0.3,
            LanguagePattern.CONTEXT_FREE: 0.5,
            LanguagePattern.CONTEXT_SENSITIVE: 0.7,
            LanguagePattern.RECURSIVELY_ENUMERABLE: 0.9,
            LanguagePattern.UNDECIDABLE: 1.0
        }
        
        max_complexity = max(
            pattern_weights.get(pattern, 0.5) for pattern in patterns
        )
        pattern_count_factor = min(len(patterns) * 0.1, 0.3)
        
        return min(max_complexity + pattern_count_factor, 1.0)
    
    def _analyze_logical_depth(self, text: str) -> float:
        """Analyze logical depth and nesting."""
        depth_indicators = ['if', 'then', 'else', 'for all', 'exists', 'such that',
                          'prove', 'assume', 'contradiction', 'induction']
        
        depth = 0
        for indicator in depth_indicators:
            depth += text.lower().count(indicator)
        
        # Count nested parentheses
        max_nesting = 0
        current_nesting = 0
        for char in text:
            if char == '(':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == ')':
                current_nesting = max(0, current_nesting - 1)
        
        return min((depth + max_nesting) / 15.0, 1.0)
    
    def _get_algorithmic_complexity(self, problem_type: ProblemType) -> float:
        """Get algorithmic complexity for problem type."""
        complexity_map = {
            ProblemType.DFA_CONSTRUCTION: 0.3,
            ProblemType.NFA_CONSTRUCTION: 0.4,
            ProblemType.REGEX_CONVERSION: 0.5,
            ProblemType.CFG_ANALYSIS: 0.6,
            ProblemType.PDA_CONSTRUCTION: 0.7,
            ProblemType.TM_CONSTRUCTION: 0.8,
            ProblemType.PUMPING_LEMMA_PROOF: 0.9,
            ProblemType.DECIDABILITY: 1.0,
            ProblemType.COMPLEXITY_ANALYSIS: 0.8
        }
        
        return complexity_map.get(problem_type, 0.5)


class HardcodedSolutionAnalyzer:
    """Analyzes hardcoded solution capabilities and patterns."""
    
    def __init__(self):
        self.algorithm_coverage = self._initialize_algorithm_coverage()
        self.performance_profiles = self._initialize_performance_profiles()
    
    def _initialize_algorithm_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Initialize coverage map for hardcoded algorithms."""
        return {
            'nfa_to_dfa': {
                'patterns': [LanguagePattern.REGULAR],
                'problem_types': [ProblemType.DFA_CONSTRUCTION, ProblemType.NFA_CONSTRUCTION],
                'complexity_range': (0.2, 0.8),
                'accuracy': 0.98,
                'speed_factor': 0.95
            },
            'dfa_minimization': {
                'patterns': [LanguagePattern.REGULAR],
                'problem_types': [ProblemType.DFA_CONSTRUCTION],
                'complexity_range': (0.1, 0.7),
                'accuracy': 0.99,
                'speed_factor': 0.9
            },
            'regex_conversion': {
                'patterns': [LanguagePattern.REGULAR],
                'problem_types': [ProblemType.REGEX_CONVERSION],
                'complexity_range': (0.2, 0.9),
                'accuracy': 0.95,
                'speed_factor': 0.8
            },
            'cfg_processing': {
                'patterns': [LanguagePattern.CONTEXT_FREE],
                'problem_types': [ProblemType.CFG_ANALYSIS],
                'complexity_range': (0.3, 0.8),
                'accuracy': 0.92,
                'speed_factor': 0.7
            },
            'cyk_parsing': {
                'patterns': [LanguagePattern.CONTEXT_FREE],
                'problem_types': [ProblemType.CFG_ANALYSIS],
                'complexity_range': (0.2, 0.9),
                'accuracy': 0.96,
                'speed_factor': 0.85
            },
            'tm_simulation': {
                'patterns': [LanguagePattern.RECURSIVELY_ENUMERABLE],
                'problem_types': [ProblemType.TM_CONSTRUCTION],
                'complexity_range': (0.4, 1.0),
                'accuracy': 0.94,
                'speed_factor': 0.6
            }
        }
    
    def _initialize_performance_profiles(self) -> Dict[str, PerformanceMetrics]:
        """Initialize performance profiles for hardcoded algorithms."""
        profiles = {}
        
        for algo_name, coverage in self.algorithm_coverage.items():
            profiles[algo_name] = PerformanceMetrics(
                accuracy=coverage['accuracy'],
                speed=coverage['speed_factor'],
                reliability=0.98,  # Hardcoded algorithms are very reliable
                resource_usage=0.3,  # Generally low resource usage
                confidence=0.95,  # High confidence in deterministic results
                success_count=100,  # Simulated historical data
                total_attempts=105,
                average_time=0.5,
                last_updated=time.time()
            )
        
        return profiles
    
    def can_handle_problem(self, context: RoutingContext) -> Tuple[bool, float, List[str]]:
        """Check if hardcoded solutions can handle the problem."""
        
        suitable_algorithms = []
        confidence_scores = []
        reasoning = []
        
        complexity_analysis = ProblemComplexityAnalyzer().analyze_complexity(context)
        overall_complexity = complexity_analysis['overall']
        
        for algo_name, coverage in self.algorithm_coverage.items():
            # Check pattern compatibility
            pattern_match = any(
                pattern in coverage['patterns'] 
                for pattern in context.patterns
            )
            
            # Check problem type compatibility
            type_match = context.problem_type in coverage['problem_types']
            
            # Check complexity range
            min_complexity, max_complexity = coverage['complexity_range']
            complexity_match = min_complexity <= overall_complexity <= max_complexity
            
            if pattern_match and type_match and complexity_match:
                confidence = (
                    coverage['accuracy'] * 0.4 +
                    coverage['speed_factor'] * 0.2 +
                    (1.0 - abs(overall_complexity - (min_complexity + max_complexity) / 2)) * 0.4
                )
                
                suitable_algorithms.append(algo_name)
                confidence_scores.append(confidence)
                reasoning.append(f"{algo_name}: pattern={pattern_match}, type={type_match}, complexity={complexity_match:.3f}")
        
        if suitable_algorithms:
            best_confidence = max(confidence_scores)
            best_algorithm = suitable_algorithms[confidence_scores.index(best_confidence)]
            reasoning.append(f"Best hardcoded option: {best_algorithm} (confidence: {best_confidence:.3f})")
            return True, best_confidence, reasoning
        else:
            reasoning.append("No suitable hardcoded algorithms found")
            return False, 0.0, reasoning
    
    def get_expected_performance(self, context: RoutingContext) -> Dict[str, float]:
        """Get expected performance metrics for hardcoded solution."""
        
        can_handle, confidence, _ = self.can_handle_problem(context)
        
        if not can_handle:
            return {
                'accuracy': 0.0,
                'speed': 0.0,
                'reliability': 0.0,
                'resource_efficiency': 0.0
            }
        
        # Find best matching algorithm
        best_algo = None
        best_score = 0.0
        
        complexity_analysis = ProblemComplexityAnalyzer().analyze_complexity(context)
        overall_complexity = complexity_analysis['overall']
        
        for algo_name, coverage in self.algorithm_coverage.items():
            pattern_match = any(p in coverage['patterns'] for p in context.patterns)
            type_match = context.problem_type in coverage['problem_types']
            
            if pattern_match and type_match:
                min_c, max_c = coverage['complexity_range']
                if min_c <= overall_complexity <= max_c:
                    score = coverage['accuracy'] * coverage['speed_factor']
                    if score > best_score:
                        best_score = score
                        best_algo = algo_name
        
        if best_algo:
            coverage = self.algorithm_coverage[best_algo]
            return {
                'accuracy': coverage['accuracy'],
                'speed': coverage['speed_factor'],
                'reliability': 0.98,
                'resource_efficiency': 0.85
            }
        
        return {
            'accuracy': 0.7,
            'speed': 0.8,
            'reliability': 0.9,
            'resource_efficiency': 0.8
        }


class AICapabilityAnalyzer:
    """Analyzes AI solution capabilities and performance predictions."""
    
    def __init__(self):
        self.model_capabilities = self._initialize_model_capabilities()
        self.performance_history = defaultdict(list)
        self.orchestrator = ModelOrchestrator()
    
    def _initialize_model_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize AI model capability profiles."""
        return {
            'general_ai': {
                'flexibility': 0.9,
                'creativity': 0.8,
                'pattern_recognition': 0.7,
                'logical_reasoning': 0.75,
                'mathematical_precision': 0.65,
                'speed_factor': 0.4,
                'reliability': 0.8
            },
            'specialist_ai': {
                'flexibility': 0.7,
                'creativity': 0.6,
                'pattern_recognition': 0.85,
                'logical_reasoning': 0.85,
                'mathematical_precision': 0.8,
                'speed_factor': 0.3,
                'reliability': 0.85
            },
            'ensemble_ai': {
                'flexibility': 0.95,
                'creativity': 0.9,
                'pattern_recognition': 0.9,
                'logical_reasoning': 0.9,
                'mathematical_precision': 0.75,
                'speed_factor': 0.15,
                'reliability': 0.9
            }
        }
    
    async def analyze_ai_suitability(self, context: RoutingContext) -> Tuple[bool, float, List[str]]:
        """Analyze if AI solutions are suitable for the problem."""
        
        reasoning = []
        complexity_analysis = ProblemComplexityAnalyzer().analyze_complexity(context)
        
        # AI is generally good at:
        # 1. Novel or creative problems
        # 2. Problems requiring flexible reasoning
        # 3. Problems where hardcoded solutions don't exist
        # 4. Complex multi-step problems
        
        suitability_score = 0.0
        
        # Check complexity - AI handles complex problems well
        if complexity_analysis['overall'] > 0.7:
            suitability_score += 0.3
            reasoning.append("High complexity favors AI solutions")
        
        # Check for novel patterns
        uncommon_patterns = [
            LanguagePattern.CONTEXT_SENSITIVE,
            LanguagePattern.UNDECIDABLE,
            LanguagePattern.RECURSIVELY_ENUMERABLE
        ]
        if any(pattern in uncommon_patterns for pattern in context.patterns):
            suitability_score += 0.25
            reasoning.append("Uncommon patterns favor AI flexibility")
        
        # Check problem type novelty
        complex_types = [
            ProblemType.PUMPING_LEMMA_PROOF,
            ProblemType.DECIDABILITY,
            ProblemType.COMPLEXITY_ANALYSIS
        ]
        if context.problem_type in complex_types:
            suitability_score += 0.2
            reasoning.append("Complex problem type suits AI reasoning")
        
        # Check if creative/proof-based reasoning is needed
        if any(keyword in context.problem_statement.lower() for keyword in 
               ['prove', 'proof', 'show that', 'demonstrate', 'argue']):
            suitability_score += 0.25
            reasoning.append("Proof-based problems suit AI reasoning")
        
        # Factor in performance requirements
        if context.accuracy_requirements > 0.9:
            suitability_score *= 0.8  # AI may be less precise than hardcoded
            reasoning.append("High accuracy requirements slightly favor hardcoded")
        
        if context.time_constraints and context.time_constraints < 5.0:
            suitability_score *= 0.7  # AI is generally slower
            reasoning.append("Tight time constraints favor hardcoded solutions")
        
        # Cap the score
        suitability_score = min(suitability_score, 1.0)
        
        is_suitable = suitability_score > 0.4
        reasoning.append(f"AI suitability score: {suitability_score:.3f}")
        
        return is_suitable, suitability_score, reasoning
    
    async def get_expected_performance(self, context: RoutingContext) -> Dict[str, float]:
        """Get expected performance metrics for AI solution."""
        
        complexity_analysis = ProblemComplexityAnalyzer().analyze_complexity(context)
        overall_complexity = complexity_analysis['overall']
        
        # Determine best AI approach
        if overall_complexity > 0.8:
            model_profile = self.model_capabilities['ensemble_ai']
        elif context.problem_type in [ProblemType.PUMPING_LEMMA_PROOF, ProblemType.DECIDABILITY]:
            model_profile = self.model_capabilities['specialist_ai']
        else:
            model_profile = self.model_capabilities['general_ai']
        
        # Adjust performance based on problem characteristics
        base_accuracy = model_profile['mathematical_precision']
        
        # Boost for creative/proof problems
        if any(keyword in context.problem_statement.lower() for keyword in 
               ['prove', 'creative', 'novel']):
            base_accuracy += 0.1
        
        # Penalty for very precise mathematical requirements
        if context.accuracy_requirements > 0.95:
            base_accuracy *= 0.9
        
        return {
            'accuracy': min(base_accuracy, 1.0),
            'speed': model_profile['speed_factor'],
            'reliability': model_profile['reliability'],
            'resource_efficiency': 0.6  # AI uses more resources
        }


class IntelligentRouter:
    """
    Main intelligent routing system that makes dynamic decisions between
    hardcoded and AI solutions based on deep problem analysis.
    """
    
    def __init__(self, storage_path: str = "./routing_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize analyzers
        self.complexity_analyzer = ProblemComplexityAnalyzer()
        self.hardcoded_analyzer = HardcodedSolutionAnalyzer()
        self.ai_analyzer = AICapabilityAnalyzer()
        
        # Initialize ML components
        self.decision_model = None
        self.feature_vectorizer = TfidfVectorizer(max_features=500)
        
        # Routing history and performance tracking
        self.routing_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(lambda: PerformanceMetrics(
            accuracy=0.5, speed=0.5, reliability=0.5, resource_usage=0.5,
            confidence=0.5, success_count=0, total_attempts=0, average_time=1.0,
            last_updated=time.time()
        ))
        
        # Decision templates
        self.decision_prompts = self._create_decision_prompts()
        
        # Load existing data
        self._load_routing_data()
        
        logger.info("Intelligent Router initialized")
    
    def _create_decision_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Create decision-making prompts."""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert system for routing computational theory problems.
            
            Analyze the following problem and determine the best solution approach:
            
            HARDCODED SOLUTIONS are best for:
            - Well-defined algorithmic problems with known solutions
            - Problems requiring high precision and speed
            - Standard constructions (NFA→DFA, regex conversions, etc.)
            - Problems with clear pattern matches
            
            AI SOLUTIONS are best for:
            - Novel or creative problems
            - Proof-based reasoning
            - Complex multi-step problems
            - Problems requiring flexible interpretation
            - Edge cases not covered by algorithms
            
            HYBRID APPROACHES are best for:
            - Problems that benefit from both precision and creativity
            - Verification of AI solutions with hardcoded checks
            - Complex problems with known sub-components
            
            Provide your analysis and recommendation."""),
            
            HumanMessage(content="""
            Problem: {problem_statement}
            Type: {problem_type}
            Patterns: {patterns}
            Complexity: {complexity_score}
            Requirements: {requirements}
            
            Hardcoded Analysis: {hardcoded_analysis}
            AI Analysis: {ai_analysis}
            
            What is your recommendation and why?
            """)
        ])
        
        return {
            'routing_analysis': analysis_prompt
        }
    
    async def route_problem(
        self,
        context: RoutingContext,
        use_ml_model: bool = True
    ) -> RoutingResult:
        """
        Route a problem to the best solution approach.
        
        Args:
            context: Routing context with problem details
            use_ml_model: Whether to use ML model for decision
        
        Returns:
            Routing result with decision and reasoning
        """
        
        start_time = time.time()
        
        # Step 1: Analyze problem complexity
        complexity_analysis = self.complexity_analyzer.analyze_complexity(context)
        
        # Step 2: Analyze hardcoded solution capabilities
        hardcoded_capable, hardcoded_confidence, hardcoded_reasoning = \
            self.hardcoded_analyzer.can_handle_problem(context)
        
        # Step 3: Analyze AI solution suitability
        ai_suitable, ai_confidence, ai_reasoning = \
            await self.ai_analyzer.analyze_ai_suitability(context)
        
        # Step 4: Get expected performance for each approach
        hardcoded_performance = self.hardcoded_analyzer.get_expected_performance(context)
        ai_performance = await self.ai_analyzer.get_expected_performance(context)
        
        # Step 5: Make routing decision
        if use_ml_model and self.decision_model is not None:
            decision = await self._make_ml_decision(context, complexity_analysis)
        else:
            decision = await self._make_rule_based_decision(
                context, hardcoded_capable, hardcoded_confidence,
                ai_suitable, ai_confidence, complexity_analysis
            )
        
        # Step 6: Build fallback chain
        fallback_chain = self._build_fallback_chain(
            decision, hardcoded_capable, ai_suitable
        )
        
        # Step 7: Calculate expected performance
        expected_performance = self._calculate_expected_performance(
            decision, hardcoded_performance, ai_performance
        )
        
        # Step 8: Create routing result
        reasoning = []
        reasoning.extend(hardcoded_reasoning)
        reasoning.extend(ai_reasoning)
        reasoning.append(f"Complexity analysis: {complexity_analysis['overall']:.3f}")
        reasoning.append(f"Decision: {decision.value}")
        
        result = RoutingResult(
            decision=decision,
            confidence=self._calculate_routing_confidence(
                decision, hardcoded_confidence, ai_confidence, complexity_analysis
            ),
            reasoning=reasoning,
            fallback_chain=fallback_chain,
            expected_performance=expected_performance,
            resource_requirements=self._estimate_resource_requirements(decision),
            metadata={
                'complexity_analysis': complexity_analysis,
                'hardcoded_performance': hardcoded_performance,
                'ai_performance': ai_performance,
                'routing_time': time.time() - start_time
            }
        )
        
        # Record routing decision for learning
        self._record_routing_decision(context, result)
        
        return result
    
    async def _make_rule_based_decision(
        self,
        context: RoutingContext,
        hardcoded_capable: bool,
        hardcoded_confidence: float,
        ai_suitable: bool,
        ai_confidence: float,
        complexity_analysis: Dict[str, float]
    ) -> RoutingDecision:
        """Make routing decision using rule-based logic."""
        
        overall_complexity = complexity_analysis['overall']
        
        # Rule 1: If hardcoded solution exists and has high confidence, prefer it
        if hardcoded_capable and hardcoded_confidence > 0.8 and overall_complexity < 0.7:
            if context.accuracy_requirements > 0.9:
                return RoutingDecision.USE_HARDCODED
        
        # Rule 2: For very complex problems, prefer AI or hybrid
        if overall_complexity > 0.8:
            if hardcoded_capable and ai_suitable:
                return RoutingDecision.USE_HYBRID
            elif ai_suitable:
                return RoutingDecision.USE_AI
        
        # Rule 3: For proof-based problems, prefer AI
        if any(keyword in context.problem_statement.lower() for keyword in 
               ['prove', 'proof', 'show that', 'demonstrate']):
            if ai_suitable:
                if hardcoded_capable and hardcoded_confidence > 0.7:
                    return RoutingDecision.USE_HYBRID
                else:
                    return RoutingDecision.USE_AI
        
        # Rule 4: Time-critical problems prefer hardcoded
        if context.time_constraints and context.time_constraints < 10.0:
            if hardcoded_capable:
                return RoutingDecision.USE_HARDCODED
        
        # Rule 5: High accuracy requirements prefer hardcoded or hybrid
        if context.accuracy_requirements > 0.95:
            if hardcoded_capable and hardcoded_confidence > 0.8:
                return RoutingDecision.USE_HARDCODED
            elif hardcoded_capable and ai_suitable:
                return RoutingDecision.USE_HYBRID
        
        # Rule 6: Novel/creative problems prefer AI
        if context.problem_type in [ProblemType.PUMPING_LEMMA_PROOF, ProblemType.DECIDABILITY]:
            if ai_suitable:
                return RoutingDecision.USE_AI
        
        # Rule 7: Use ensemble for high-stakes problems
        if (context.accuracy_requirements > 0.9 and 
            overall_complexity > 0.6 and 
            hardcoded_capable and ai_suitable):
            return RoutingDecision.USE_ENSEMBLE
        
        # Default decision logic
        if hardcoded_capable and ai_suitable:
            if hardcoded_confidence > ai_confidence * 1.2:
                return RoutingDecision.USE_HARDCODED
            elif ai_confidence > hardcoded_confidence * 1.2:
                return RoutingDecision.USE_AI
            else:
                return RoutingDecision.USE_HYBRID
        elif hardcoded_capable:
            return RoutingDecision.USE_HARDCODED
        elif ai_suitable:
            return RoutingDecision.USE_AI
        else:
            return RoutingDecision.FALLBACK_CASCADE
    
    async def _make_ml_decision(
        self,
        context: RoutingContext,
        complexity_analysis: Dict[str, float]
    ) -> RoutingDecision:
        """Make routing decision using trained ML model."""
        
        # Extract features for ML model
        features = self._extract_ml_features(context, complexity_analysis)
        
        # Predict using trained model
        prediction = self.decision_model.predict([features])[0]
        probabilities = self.decision_model.predict_proba([features])[0]
        
        # Map prediction to routing decision
        decision_mapping = {
            0: RoutingDecision.USE_HARDCODED,
            1: RoutingDecision.USE_AI,
            2: RoutingDecision.USE_HYBRID,
            3: RoutingDecision.USE_ENSEMBLE
        }
        
        decision = decision_mapping.get(prediction, RoutingDecision.USE_HYBRID)
        
        # If confidence is low, fall back to rule-based decision
        max_probability = max(probabilities)
        if max_probability < 0.6:
            hardcoded_capable, hardcoded_confidence, _ = \
                self.hardcoded_analyzer.can_handle_problem(context)
            ai_suitable, ai_confidence, _ = \
                await self.ai_analyzer.analyze_ai_suitability(context)
            
            decision = await self._make_rule_based_decision(
                context, hardcoded_capable, hardcoded_confidence,
                ai_suitable, ai_confidence, complexity_analysis
            )
        
        return decision
    
    def _extract_ml_features(
        self,
        context: RoutingContext,
        complexity_analysis: Dict[str, float]
    ) -> List[float]:
        """Extract features for ML model."""
        
        features = []
        
        # Complexity features
        features.extend([
            complexity_analysis['overall'],
            complexity_analysis['mathematical_complexity'],
            complexity_analysis['pattern_complexity'],
            complexity_analysis['logical_depth'],
            complexity_analysis['algorithmic_complexity']
        ])
        
        # Problem type features (one-hot encoded)
        problem_types = list(ProblemType)
        for ptype in problem_types:
            features.append(1.0 if context.problem_type == ptype else 0.0)
        
        # Pattern features
        all_patterns = list(LanguagePattern)
        for pattern in all_patterns:
            features.append(1.0 if pattern in context.patterns else 0.0)
        
        # Requirement features
        features.extend([
            context.accuracy_requirements,
            context.confidence_threshold,
            context.time_constraints or 60.0,  # Default 60 seconds
            len(context.problem_statement) / 1000.0,  # Statement length
        ])
        
        return features
    
    def _build_fallback_chain(
        self,
        primary_decision: RoutingDecision,
        hardcoded_capable: bool,
        ai_suitable: bool
    ) -> List[SolutionType]:
        """Build fallback chain for the routing decision."""
        
        chain = []
        
        if primary_decision == RoutingDecision.USE_HARDCODED:
            chain = [SolutionType.HARDCODED]
            if ai_suitable:
                chain.append(SolutionType.AI_GENERATED)
        
        elif primary_decision == RoutingDecision.USE_AI:
            chain = [SolutionType.AI_GENERATED]
            if hardcoded_capable:
                chain.append(SolutionType.HARDCODED)
        
        elif primary_decision == RoutingDecision.USE_HYBRID:
            chain = [SolutionType.HYBRID]
            if hardcoded_capable:
                chain.append(SolutionType.HARDCODED)
            if ai_suitable:
                chain.append(SolutionType.AI_GENERATED)
        
        elif primary_decision == RoutingDecision.USE_ENSEMBLE:
            chain = [SolutionType.HYBRID, SolutionType.HARDCODED, SolutionType.AI_GENERATED]
        
        else:  # FALLBACK_CASCADE
            if hardcoded_capable:
                chain.append(SolutionType.HARDCODED)
            if ai_suitable:
                chain.append(SolutionType.AI_GENERATED)
            if not chain:
                chain.append(SolutionType.AI_GENERATED)  # Last resort
        
        return chain
    
    def _calculate_expected_performance(
        self,
        decision: RoutingDecision,
        hardcoded_perf: Dict[str, float],
        ai_perf: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate expected performance for the routing decision."""
        
        if decision == RoutingDecision.USE_HARDCODED:
            return hardcoded_perf
        
        elif decision == RoutingDecision.USE_AI:
            return ai_perf
        
        elif decision == RoutingDecision.USE_HYBRID:
            # Hybrid combines strengths
            return {
                'accuracy': max(hardcoded_perf['accuracy'], ai_perf['accuracy']),
                'speed': (hardcoded_perf['speed'] + ai_perf['speed']) / 2,
                'reliability': max(hardcoded_perf['reliability'], ai_perf['reliability']),
                'resource_efficiency': min(hardcoded_perf['resource_efficiency'], 
                                         ai_perf['resource_efficiency']) * 0.8
            }
        
        elif decision == RoutingDecision.USE_ENSEMBLE:
            # Ensemble provides best overall performance but uses more resources
            return {
                'accuracy': max(hardcoded_perf['accuracy'], ai_perf['accuracy']) + 0.05,
                'speed': min(hardcoded_perf['speed'], ai_perf['speed']) * 0.7,
                'reliability': max(hardcoded_perf['reliability'], ai_perf['reliability']) + 0.02,
                'resource_efficiency': min(hardcoded_perf['resource_efficiency'], 
                                         ai_perf['resource_efficiency']) * 0.6
            }
        
        else:  # FALLBACK_CASCADE
            # Conservative estimates
            return {
                'accuracy': (hardcoded_perf['accuracy'] + ai_perf['accuracy']) / 2,
                'speed': min(hardcoded_perf['speed'], ai_perf['speed']),
                'reliability': min(hardcoded_perf['reliability'], ai_perf['reliability']),
                'resource_efficiency': min(hardcoded_perf['resource_efficiency'], 
                                         ai_perf['resource_efficiency']) * 0.9
            }
    
    def _calculate_routing_confidence(
        self,
        decision: RoutingDecision,
        hardcoded_confidence: float,
        ai_confidence: float,
        complexity_analysis: Dict[str, float]
    ) -> float:
        """Calculate confidence in the routing decision."""
        
        base_confidence = 0.5
        
        if decision == RoutingDecision.USE_HARDCODED:
            base_confidence = hardcoded_confidence
        elif decision == RoutingDecision.USE_AI:
            base_confidence = ai_confidence
        elif decision == RoutingDecision.USE_HYBRID:
            base_confidence = (hardcoded_confidence + ai_confidence) / 2 + 0.1
        elif decision == RoutingDecision.USE_ENSEMBLE:
            base_confidence = max(hardcoded_confidence, ai_confidence) + 0.15
        
        # Adjust based on complexity match
        complexity = complexity_analysis['overall']
        if complexity < 0.3 and decision == RoutingDecision.USE_HARDCODED:
            base_confidence += 0.1
        elif complexity > 0.8 and decision == RoutingDecision.USE_AI:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _estimate_resource_requirements(self, decision: RoutingDecision) -> Dict[str, Any]:
        """Estimate resource requirements for the routing decision."""
        
        if decision == RoutingDecision.USE_HARDCODED:
            return {
                'cpu_time': 'low',
                'memory': 'low',
                'tokens': 0,
                'parallel_processes': 1
            }
        
        elif decision == RoutingDecision.USE_AI:
            return {
                'cpu_time': 'medium',
                'memory': 'medium',
                'tokens': 2000,
                'parallel_processes': 1
            }
        
        elif decision == RoutingDecision.USE_HYBRID:
            return {
                'cpu_time': 'medium',
                'memory': 'medium',
                'tokens': 1500,
                'parallel_processes': 2
            }
        
        elif decision == RoutingDecision.USE_ENSEMBLE:
            return {
                'cpu_time': 'high',
                'memory': 'high',
                'tokens': 4000,
                'parallel_processes': 3
            }
        
        else:  # FALLBACK_CASCADE
            return {
                'cpu_time': 'variable',
                'memory': 'medium',
                'tokens': 3000,
                'parallel_processes': 2
            }
    
    def _record_routing_decision(self, context: RoutingContext, result: RoutingResult):
        """Record routing decision for future learning."""
        
        record = {
            'timestamp': time.time(),
            'problem_hash': self._hash_problem(context.problem_statement),
            'problem_type': context.problem_type.value,
            'patterns': [p.value for p in context.patterns],
            'complexity': result.metadata['complexity_analysis']['overall'],
            'decision': result.decision.value,
            'confidence': result.confidence,
            'expected_accuracy': result.expected_performance['accuracy']
        }
        
        self.routing_history.append(record)
        
        # Update performance metrics
        decision_key = result.decision.value
        metrics = self.performance_metrics[decision_key]
        metrics.total_attempts += 1
        metrics.last_updated = time.time()
    
    def learn_from_outcome(
        self,
        problem_hash: str,
        actual_performance: Dict[str, float],
        success: bool
    ):
        """Learn from actual problem-solving outcomes."""
        
        # Find the routing decision for this problem
        routing_record = None
        for record in reversed(self.routing_history):
            if record['problem_hash'] == problem_hash:
                routing_record = record
                break
        
        if not routing_record:
            logger.warning(f"No routing record found for problem hash: {problem_hash}")
            return
        
        # Update performance metrics
        decision_key = routing_record['decision']
        metrics = self.performance_metrics[decision_key]
        
        if success:
            metrics.success_count += 1
        
        # Update running averages
        metrics.accuracy = (
            (metrics.accuracy * (metrics.total_attempts - 1) + 
             actual_performance.get('accuracy', 0.5)) / metrics.total_attempts
        )
        
        metrics.speed = (
            (metrics.speed * (metrics.total_attempts - 1) + 
             actual_performance.get('speed', 0.5)) / metrics.total_attempts
        )
        
        # Add to training data for ML model
        if self.decision_model is None:
            self._initialize_ml_model()
        
        # Retrain model periodically
        if len(self.routing_history) % 50 == 0:
            asyncio.create_task(self._retrain_ml_model())
    
    async def _retrain_ml_model(self):
        """Retrain the ML model with accumulated data."""
        
        if len(self.routing_history) < 20:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for record in self.routing_history:
            # Create feature vector (simplified for this example)
            features = [
                record['complexity'],
                len(record['patterns']),
                1.0 if 'REGULAR' in record['patterns'] else 0.0,
                1.0 if 'CONTEXT_FREE' in record['patterns'] else 0.0,
                record['expected_accuracy']
            ]
            X.append(features)
            
            # Map decision to class
            decision_to_class = {
                'use_hardcoded': 0,
                'use_ai': 1,
                'use_hybrid': 2,
                'use_ensemble': 3
            }
            y.append(decision_to_class.get(record['decision'], 2))
        
        # Train model
        if len(set(y)) > 1:  # Need multiple classes
            self.decision_model.fit(X, y)
            logger.info(f"ML model retrained with {len(X)} samples")
    
    def _initialize_ml_model(self):
        """Initialize the ML model for routing decisions."""
        self.decision_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _hash_problem(self, problem_statement: str) -> str:
        """Create hash for problem identification."""
        return hashlib.md5(problem_statement.encode()).hexdigest()[:12]
    
    def _save_routing_data(self):
        """Save routing data to storage."""
        try:
            data = {
                'routing_history': list(self.routing_history),
                'performance_metrics': dict(self.performance_metrics),
                'model_state': self.decision_model.__getstate__() if self.decision_model else None
            }
            
            with open(self.storage_path / 'routing_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            logger.info("Routing data saved successfully")
        except Exception as e:
            logger.error(f"Failed to save routing data: {e}")
    
    def _load_routing_data(self):
        """Load routing data from storage."""
        try:
            with open(self.storage_path / 'routing_data.pkl', 'rb') as f:
                data = pickle.load(f)
            
            self.routing_history = deque(data.get('routing_history', []), maxlen=1000)
            
            # Restore performance metrics
            for key, metrics_data in data.get('performance_metrics', {}).items():
                self.performance_metrics[key] = PerformanceMetrics(**metrics_data)
            
            # Restore ML model
            if data.get('model_state'):
                self._initialize_ml_model()
                self.decision_model.__setstate__(data['model_state'])
            
            logger.info(f"Loaded routing data: {len(self.routing_history)} records")
        except Exception as e:
            logger.info(f"No existing routing data found: {e}")
            self._initialize_ml_model()
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics for analysis."""
        
        total_decisions = len(self.routing_history)
        if total_decisions == 0:
            return {"message": "No routing decisions recorded yet"}
        
        decision_counts = defaultdict(int)
        accuracy_by_decision = defaultdict(list)
        
        for record in self.routing_history:
            decision = record['decision']
            decision_counts[decision] += 1
            
            # Find corresponding performance metrics
            if decision in self.performance_metrics:
                accuracy_by_decision[decision].append(
                    self.performance_metrics[decision].accuracy
                )
        
        return {
            'total_decisions': total_decisions,
            'decision_distribution': dict(decision_counts),
            'average_accuracy_by_decision': {
                decision: sum(accuracies) / len(accuracies) if accuracies else 0
                for decision, accuracies in accuracy_by_decision.items()
            },
            'most_common_decision': max(decision_counts, key=decision_counts.get) if decision_counts else None,
            'performance_metrics': {
                decision: {
                    'success_rate': metrics.success_count / metrics.total_attempts if metrics.total_attempts > 0 else 0,
                    'average_accuracy': metrics.accuracy,
                    'average_speed': metrics.speed,
                    'reliability': metrics.reliability
                }
                for decision, metrics in self.performance_metrics.items()
            }
        }


# Global router instance
intelligent_router = IntelligentRouter()


async def route_problem(
    problem_statement: str,
    problem_type: ProblemType,
    patterns: List[LanguagePattern],
    performance_requirements: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RoutingResult:
    """
    Convenience function for routing problems.
    
    Args:
        problem_statement: The problem to solve
        problem_type: Type of the problem
        patterns: Language patterns involved
        performance_requirements: Performance requirements
        **kwargs: Additional context parameters
    
    Returns:
        Routing result with decision and reasoning
    """
    
    context = RoutingContext(
        problem_statement=problem_statement,
        problem_type=problem_type,
        patterns=patterns,
        complexity_score=0.5,  # Will be calculated
        performance_requirements=performance_requirements or {},
        **kwargs
    )
    
    return await intelligent_router.route_problem(context)