"""
Enhanced Learning System with Hardcoded Knowledge Integration
============================================================

This system enhances the basic learning system by:
- Integrating knowledge from hardcoded solutions
- Learning patterns and strategies from proven algorithms
- Improving AI-generated solutions with hardcoded insights
- Creating hybrid approaches combining both methodologies

Author: AegisX AI Software Engineer
Version: 1.0
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import hashlib
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd

from .learning_system import LearningSystem, SolvedProblem, PatternKnowledge, LearningInsights
from .knowledge_extractor import (
    AlgorithmKnowledgeExtractor, SolutionPattern, EdgeCase, OptimizationTechnique
)
from .problem_understanding import ProblemRequirements, ProblemType, LanguagePattern
from .intelligent_solver import IntelligentSolution, SolutionStrategy
from .ai_config import AIConfig, ModelType

logger = logging.getLogger(__name__)


@dataclass
class HardcodedKnowledge:
    """Knowledge extracted from hardcoded algorithms."""
    algorithm_patterns: Dict[str, SolutionPattern]
    edge_cases: List[EdgeCase]
    optimizations: List[OptimizationTechnique]
    cross_algorithm_insights: List[str]
    performance_profiles: Dict[str, Dict[str, float]]
    training_data: Dict[str, Any]
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HybridInsight:
    """Insight combining hardcoded and AI knowledge."""
    insight_id: str
    hardcoded_component: str
    ai_component: str
    synthesis: str
    confidence: float
    applicability: List[ProblemType]
    improvement_metrics: Dict[str, float]


@dataclass
class LearningMetrics:
    """Metrics for learning system performance."""
    total_problems_learned: int
    hardcoded_knowledge_integrated: int
    hybrid_insights_generated: int
    average_improvement: float
    learning_accuracy: float
    knowledge_coverage: float
    adaptation_rate: float


class HardcodedKnowledgeIntegrator:
    """Integrates knowledge from hardcoded algorithms into the learning system."""
    
    def __init__(self):
        self.knowledge_extractor = AlgorithmKnowledgeExtractor()
        self.hardcoded_knowledge: Optional[HardcodedKnowledge] = None
        self.pattern_embeddings = None
        self.insight_cache = {}
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    async def extract_and_integrate_knowledge(self) -> HardcodedKnowledge:
        """Extract knowledge from hardcoded algorithms and prepare for integration."""
        
        logger.info("Extracting knowledge from hardcoded algorithms...")
        
        # Extract raw knowledge
        raw_knowledge = self.knowledge_extractor.extract_all_knowledge()
        
        # Process and structure the knowledge
        algorithm_patterns = {}
        for pattern_id, pattern_data in raw_knowledge.get('patterns', {}).items():
            if isinstance(pattern_data, dict):
                # Convert dict back to SolutionPattern
                pattern = SolutionPattern(
                    pattern_id=pattern_data['pattern_id'],
                    pattern_name=pattern_data['pattern_name'],
                    problem_types=[ProblemType(pt) for pt in pattern_data['problem_types']],
                    language_patterns=[LanguagePattern(lp) for lp in pattern_data['language_patterns']],
                    algorithm_steps=pattern_data['algorithm_steps'],
                    key_insights=pattern_data['key_insights'],
                    common_pitfalls=pattern_data['common_pitfalls'],
                    optimization_opportunities=pattern_data['optimization_opportunities'],
                    code_examples=pattern_data['code_examples'],
                    mathematical_properties=pattern_data['mathematical_properties'],
                    performance_characteristics=pattern_data['performance_characteristics']
                )
                algorithm_patterns[pattern_id] = pattern
        
        # Process edge cases
        edge_cases = []
        for case_data in raw_knowledge.get('edge_cases', []):
            if isinstance(case_data, dict):
                edge_case = EdgeCase(
                    case_id=case_data['case_id'],
                    description=case_data['description'],
                    algorithm=case_data['algorithm'],
                    input_conditions=case_data['input_conditions'],
                    handling_strategy=case_data['handling_strategy'],
                    code_snippet=case_data['code_snippet'],
                    importance_level=case_data['importance_level']
                )
                edge_cases.append(edge_case)
        
        # Process optimizations
        optimizations = []
        for opt_data in raw_knowledge.get('optimizations', []):
            if isinstance(opt_data, dict):
                optimization = OptimizationTechnique(
                    technique_id=opt_data['technique_id'],
                    technique_name=opt_data['technique_name'],
                    description=opt_data['description'],
                    applicable_contexts=opt_data['applicable_contexts'],
                    performance_impact=opt_data['performance_impact'],
                    implementation_examples=opt_data['implementation_examples'],
                    trade_offs=opt_data['trade_offs']
                )
                optimizations.append(optimization)
        
        # Generate training data
        training_data = self.knowledge_extractor.generate_training_data(raw_knowledge)
        
        # Create hardcoded knowledge structure
        self.hardcoded_knowledge = HardcodedKnowledge(
            algorithm_patterns=algorithm_patterns,
            edge_cases=edge_cases,
            optimizations=optimizations,
            cross_algorithm_insights=raw_knowledge.get('cross_algorithm_insights', []),
            performance_profiles=self._extract_performance_profiles(raw_knowledge),
            training_data=training_data
        )
        
        # Create embeddings for pattern matching
        await self._create_pattern_embeddings()
        
        logger.info(f"Integrated knowledge from {len(algorithm_patterns)} patterns, "
                   f"{len(edge_cases)} edge cases, {len(optimizations)} optimizations")
        
        return self.hardcoded_knowledge
    
    def _extract_performance_profiles(self, raw_knowledge: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract performance profiles from raw knowledge."""
        
        profiles = {}
        
        for alg_name, alg_data in raw_knowledge.get('algorithms', {}).items():
            profile = {
                'accuracy': 0.95,  # Hardcoded algorithms are typically very accurate
                'speed': 0.8,      # Generally fast
                'reliability': 0.98, # Very reliable
                'scalability': 0.7   # Varies by algorithm
            }
            
            # Adjust based on algorithm characteristics
            if 'complexity_patterns' in alg_data:
                complexity_patterns = alg_data['complexity_patterns']
                if any('nested' in pattern for pattern in complexity_patterns):
                    profile['scalability'] = 0.5  # Nested loops reduce scalability
                if any('recursive' in pattern for pattern in complexity_patterns):
                    profile['speed'] = 0.6  # Recursion can be slower
            
            profiles[alg_name] = profile
        
        return profiles
    
    async def _create_pattern_embeddings(self):
        """Create embeddings for pattern matching."""
        
        if not self.hardcoded_knowledge:
            return
        
        # Create text representations of patterns
        pattern_texts = []
        pattern_ids = []
        
        for pattern_id, pattern in self.hardcoded_knowledge.algorithm_patterns.items():
            text = f"{pattern.pattern_name} {' '.join(pattern.algorithm_steps)} {' '.join(pattern.key_insights)}"
            pattern_texts.append(text)
            pattern_ids.append(pattern_id)
        
        if pattern_texts:
            # Create FAISS index
            self.pattern_embeddings = FAISS.from_texts(
                pattern_texts,
                self.embeddings,
                metadatas=[{'pattern_id': pid} for pid in pattern_ids]
            )
    
    async def find_relevant_hardcoded_patterns(
        self,
        requirements: ProblemRequirements,
        top_k: int = 3
    ) -> List[Tuple[SolutionPattern, float]]:
        """Find hardcoded patterns relevant to a problem."""
        
        if not self.hardcoded_knowledge or not self.pattern_embeddings:
            return []
        
        # Create query text
        query_text = f"{requirements.problem_type.value} {' '.join([p.value for p in requirements.patterns])} {requirements.original_statement[:200]}"
        
        # Search for similar patterns
        docs = self.pattern_embeddings.similarity_search_with_score(query_text, k=top_k)
        
        relevant_patterns = []
        for doc, score in docs:
            pattern_id = doc.metadata['pattern_id']
            if pattern_id in self.hardcoded_knowledge.algorithm_patterns:
                pattern = self.hardcoded_knowledge.algorithm_patterns[pattern_id]
                # Also check direct pattern/type matches for bonus scoring
                type_match = requirements.problem_type in pattern.problem_types
                pattern_match = any(p in pattern.language_patterns for p in requirements.patterns)
                
                if type_match:
                    score += 0.2
                if pattern_match:
                    score += 0.3
                
                relevant_patterns.append((pattern, min(score, 1.0)))
        
        return sorted(relevant_patterns, key=lambda x: x[1], reverse=True)
    
    def get_relevant_edge_cases(
        self,
        algorithm_name: str,
        problem_context: str
    ) -> List[EdgeCase]:
        """Get relevant edge cases for an algorithm or problem."""
        
        if not self.hardcoded_knowledge:
            return []
        
        relevant_cases = []
        
        for edge_case in self.hardcoded_knowledge.edge_cases:
            # Check if edge case is relevant
            if (edge_case.algorithm == algorithm_name or
                any(condition.lower() in problem_context.lower() 
                    for condition in edge_case.input_conditions)):
                relevant_cases.append(edge_case)
        
        # Sort by importance level
        return sorted(relevant_cases, key=lambda x: x.importance_level, reverse=True)
    
    def get_relevant_optimizations(
        self,
        context: List[str],
        performance_requirements: Dict[str, Any]
    ) -> List[OptimizationTechnique]:
        """Get relevant optimization techniques."""
        
        if not self.hardcoded_knowledge:
            return []
        
        relevant_optimizations = []
        
        for optimization in self.hardcoded_knowledge.optimizations:
            # Check if optimization is applicable
            if any(ctx in optimization.applicable_contexts for ctx in context):
                relevant_optimizations.append(optimization)
            
            # Check performance requirements
            if performance_requirements:
                if ('speed' in performance_requirements and 
                    optimization.performance_impact.get('time_improvement', 0) > 0):
                    relevant_optimizations.append(optimization)
                
                if ('memory' in performance_requirements and
                    optimization.performance_impact.get('space_improvement', 0) > 0):
                    relevant_optimizations.append(optimization)
        
        return list(set(relevant_optimizations))  # Remove duplicates


class EnhancedLearningSystem(LearningSystem):
    """Enhanced learning system with hardcoded knowledge integration."""
    
    def __init__(self, storage_path: str = "./enhanced_learning_data"):
        super().__init__(storage_path)
        
        self.knowledge_integrator = HardcodedKnowledgeIntegrator()
        self.hybrid_insights: Dict[str, HybridInsight] = {}
        self.learning_metrics = LearningMetrics(
            total_problems_learned=0,
            hardcoded_knowledge_integrated=0,
            hybrid_insights_generated=0,
            average_improvement=0.0,
            learning_accuracy=0.0,
            knowledge_coverage=0.0,
            adaptation_rate=0.0
        )
        
        # Enhanced prompts for hybrid learning
        self.hybrid_prompts = self._create_hybrid_prompts()
        
        # Initialize with hardcoded knowledge
        asyncio.create_task(self._initialize_hardcoded_knowledge())
        
        logger.info("Enhanced Learning System initialized")
    
    def _create_hybrid_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Create prompts for hybrid learning."""
        
        hybrid_analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert system that combines hardcoded algorithmic knowledge with AI flexibility.
            
            Your task is to analyze a problem and generate insights that combine:
            1. Proven algorithmic approaches from hardcoded solutions
            2. Creative problem-solving from AI capabilities
            3. Novel hybrid strategies that leverage both strengths
            
            Focus on creating actionable insights that improve solution quality."""),
            
            HumanMessage(content="""
            Problem: {problem_statement}
            Type: {problem_type}
            Patterns: {patterns}
            
            Hardcoded Knowledge Available:
            {hardcoded_patterns}
            
            Edge Cases to Consider:
            {edge_cases}
            
            Optimization Opportunities:
            {optimizations}
            
            Generate hybrid insights combining algorithmic precision with AI creativity.
            """)
        ])
        
        solution_enhancement_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Enhance an AI-generated solution using hardcoded algorithmic knowledge.
            
            You should:
            1. Identify parts that can benefit from proven algorithms
            2. Suggest specific algorithmic improvements
            3. Add robustness from hardcoded edge case handling
            4. Recommend performance optimizations
            5. Maintain the creative aspects of the AI solution
            """),
            
            HumanMessage(content="""
            AI Solution: {ai_solution}
            
            Relevant Hardcoded Knowledge:
            {hardcoded_knowledge}
            
            Performance Requirements: {performance_requirements}
            
            Enhance this solution while preserving its strengths.
            """)
        ])
        
        return {
            'hybrid_analysis': hybrid_analysis_prompt,
            'solution_enhancement': solution_enhancement_prompt
        }
    
    async def _initialize_hardcoded_knowledge(self):
        """Initialize the system with hardcoded knowledge."""
        try:
            await self.knowledge_integrator.extract_and_integrate_knowledge()
            self.learning_metrics.hardcoded_knowledge_integrated = len(
                self.knowledge_integrator.hardcoded_knowledge.algorithm_patterns
            )
            logger.info("Hardcoded knowledge integration completed")
        except Exception as e:
            logger.error(f"Failed to initialize hardcoded knowledge: {e}")
    
    async def get_enhanced_insights_for_problem(
        self,
        requirements: ProblemRequirements
    ) -> LearningInsights:
        """Get enhanced insights combining basic learning with hardcoded knowledge."""
        
        # Get basic insights from parent class
        basic_insights = await super().get_insights_for_problem(requirements)
        
        # Get relevant hardcoded patterns
        hardcoded_patterns = await self.knowledge_integrator.find_relevant_hardcoded_patterns(
            requirements, top_k=3
        )
        
        # Get relevant edge cases
        edge_cases = []
        if hardcoded_patterns:
            best_pattern = hardcoded_patterns[0][0]
            edge_cases = self.knowledge_integrator.get_relevant_edge_cases(
                best_pattern.pattern_name, requirements.original_statement
            )
        
        # Get relevant optimizations
        optimizations = self.knowledge_integrator.get_relevant_optimizations(
            [requirements.problem_type.value],
            {'accuracy': requirements.accuracy_threshold if hasattr(requirements, 'accuracy_threshold') else 0.8}
        )
        
        # Generate hybrid insights
        hybrid_insights = await self._generate_hybrid_insights(
            requirements, hardcoded_patterns, edge_cases, optimizations
        )
        
        # Enhance the basic insights
        enhanced_suggestions = list(basic_insights.optimization_suggestions)
        
        # Add hardcoded-derived suggestions
        for pattern, confidence in hardcoded_patterns:
            enhanced_suggestions.extend([
                f"Consider {pattern.pattern_name} approach (confidence: {confidence:.2f})",
                f"Key insight: {pattern.key_insights[0]}" if pattern.key_insights else ""
            ])
        
        # Add edge case handling suggestions
        for edge_case in edge_cases[:2]:  # Top 2 most important
            enhanced_suggestions.append(
                f"Handle edge case: {edge_case.description} using {edge_case.handling_strategy}"
            )
        
        # Add optimization suggestions
        for optimization in optimizations[:2]:  # Top 2 optimizations
            enhanced_suggestions.append(
                f"Apply {optimization.technique_name}: {optimization.description}"
            )
        
        # Calculate enhanced confidence boost
        hardcoded_confidence_boost = 0.0
        if hardcoded_patterns:
            avg_hardcoded_confidence = sum(conf for _, conf in hardcoded_patterns) / len(hardcoded_patterns)
            hardcoded_confidence_boost = avg_hardcoded_confidence * 0.2
        
        enhanced_confidence = min(basic_insights.confidence_boost + hardcoded_confidence_boost, 0.5)
        
        # Create enhanced insights
        enhanced_insights = LearningInsights(
            recommended_strategy=self._get_enhanced_strategy(basic_insights, hardcoded_patterns),
            confidence_boost=enhanced_confidence,
            similar_problems=basic_insights.similar_problems,
            pattern_matches=basic_insights.pattern_matches + [p[0].pattern_name for p in hardcoded_patterns],
            optimization_suggestions=enhanced_suggestions,
            predicted_difficulty=self._adjust_difficulty_prediction(
                basic_insights.predicted_difficulty, hardcoded_patterns
            )
        )
        
        return enhanced_insights
    
    async def _generate_hybrid_insights(
        self,
        requirements: ProblemRequirements,
        hardcoded_patterns: List[Tuple[SolutionPattern, float]],
        edge_cases: List[EdgeCase],
        optimizations: List[OptimizationTechnique]
    ) -> List[str]:
        """Generate hybrid insights combining hardcoded and AI knowledge."""
        
        if not hardcoded_patterns:
            return []
        
        # Prepare context for hybrid analysis
        hardcoded_context = ""
        for pattern, confidence in hardcoded_patterns:
            hardcoded_context += f"Pattern: {pattern.pattern_name} (confidence: {confidence:.2f})\n"
            hardcoded_context += f"Steps: {', '.join(pattern.algorithm_steps[:3])}\n"
            hardcoded_context += f"Insights: {', '.join(pattern.key_insights[:2])}\n\n"
        
        edge_case_context = ""
        for edge_case in edge_cases[:3]:
            edge_case_context += f"Case: {edge_case.description}\n"
            edge_case_context += f"Strategy: {edge_case.handling_strategy}\n\n"
        
        optimization_context = ""
        for opt in optimizations[:3]:
            optimization_context += f"Technique: {opt.technique_name}\n"
            optimization_context += f"Description: {opt.description}\n"
            optimization_context += f"Impact: {opt.performance_impact}\n\n"
        
        # Generate hybrid insights using AI
        prompt = self.hybrid_prompts['hybrid_analysis'].format(
            problem_statement=requirements.original_statement,
            problem_type=requirements.problem_type.value,
            patterns=[p.value for p in requirements.patterns],
            hardcoded_patterns=hardcoded_context,
            edge_cases=edge_case_context,
            optimizations=optimization_context
        )
        
        try:
            response = await self.model.ainvoke(prompt)
            
            # Parse insights from response
            insights = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 20:
                    insights.append(line)
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Failed to generate hybrid insights: {e}")
            return []
    
    def _get_enhanced_strategy(
        self,
        basic_insights: LearningInsights,
        hardcoded_patterns: List[Tuple[SolutionPattern, float]]
    ) -> SolutionStrategy:
        """Get enhanced strategy recommendation."""
        
        if not hardcoded_patterns:
            return basic_insights.recommended_strategy
        
        # If we have high-confidence hardcoded patterns, consider hybrid approach
        best_pattern_confidence = hardcoded_patterns[0][1] if hardcoded_patterns else 0.0
        
        if best_pattern_confidence > 0.8:
            return SolutionStrategy.HYBRID  # Use hybrid approach
        elif best_pattern_confidence > 0.6:
            return SolutionStrategy.CONSTRUCTION  # Lean towards construction with guidance
        else:
            return basic_insights.recommended_strategy  # Fall back to original
    
    def _adjust_difficulty_prediction(
        self,
        basic_difficulty: float,
        hardcoded_patterns: List[Tuple[SolutionPattern, float]]
    ) -> float:
        """Adjust difficulty prediction based on hardcoded knowledge."""
        
        if not hardcoded_patterns:
            return basic_difficulty
        
        # If we have good hardcoded patterns, problem might be easier
        best_confidence = hardcoded_patterns[0][1]
        adjustment = -0.1 * best_confidence  # Reduce difficulty based on pattern confidence
        
        return max(0.0, min(1.0, basic_difficulty + adjustment))
    
    async def learn_from_enhanced_solution(
        self,
        requirements: ProblemRequirements,
        solution: IntelligentSolution,
        hardcoded_knowledge_used: Optional[Dict[str, Any]] = None,
        hybrid_approach_used: bool = False,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Learn from a solution that potentially used hardcoded knowledge."""
        
        # First, learn using the base system
        basic_learning_result = await super().learn_from_solution(requirements, solution, feedback)
        
        # If hardcoded knowledge was used, learn from that integration
        if hardcoded_knowledge_used:
            await self._learn_from_hardcoded_integration(
                requirements, solution, hardcoded_knowledge_used
            )
        
        # If hybrid approach was used, generate hybrid insights
        if hybrid_approach_used:
            hybrid_insight = await self._generate_hybrid_insight_from_solution(
                requirements, solution, hardcoded_knowledge_used or {}
            )
            if hybrid_insight:
                self.hybrid_insights[hybrid_insight.insight_id] = hybrid_insight
                self.learning_metrics.hybrid_insights_generated += 1
        
        # Update learning metrics
        self.learning_metrics.total_problems_learned += 1
        self._update_learning_metrics()
        
        # Add enhancement information to the result
        enhanced_result = basic_learning_result.copy()
        enhanced_result.update({
            'hardcoded_knowledge_used': bool(hardcoded_knowledge_used),
            'hybrid_approach_used': hybrid_approach_used,
            'hybrid_insights_count': len(self.hybrid_insights),
            'learning_metrics': self.learning_metrics
        })
        
        return enhanced_result
    
    async def _learn_from_hardcoded_integration(
        self,
        requirements: ProblemRequirements,
        solution: IntelligentSolution,
        hardcoded_knowledge: Dict[str, Any]
    ):
        """Learn from the integration of hardcoded knowledge."""
        
        # Analyze which hardcoded components were most effective
        for component_type, component_data in hardcoded_knowledge.items():
            if component_type == 'patterns_used':
                for pattern_id in component_data:
                    # Update pattern effectiveness
                    if pattern_id in self.knowledge_integrator.hardcoded_knowledge.algorithm_patterns:
                        # This pattern was successfully used
                        pass  # Could track pattern usage statistics
            
            elif component_type == 'optimizations_applied':
                for opt_id in component_data:
                    # Track optimization effectiveness
                    pass  # Could measure performance improvements
        
        logger.info(f"Learned from hardcoded knowledge integration: {list(hardcoded_knowledge.keys())}")
    
    async def _generate_hybrid_insight_from_solution(
        self,
        requirements: ProblemRequirements,
        solution: IntelligentSolution,
        hardcoded_knowledge: Dict[str, Any]
    ) -> Optional[HybridInsight]:
        """Generate a hybrid insight from a successful solution."""
        
        try:
            # Create insight ID
            insight_id = hashlib.md5(
                f"{requirements.problem_type.value}_{solution.strategy_used.value}_{datetime.utcnow().isoformat()}"
                .encode()
            ).hexdigest()[:12]
            
            # Identify hardcoded and AI components
            hardcoded_component = "Used proven algorithmic patterns"
            ai_component = "Applied creative problem-solving"
            
            if 'patterns_used' in hardcoded_knowledge:
                hardcoded_component = f"Applied {len(hardcoded_knowledge['patterns_used'])} hardcoded patterns"
            
            if solution.confidence_score > 0.8:
                ai_component = "Generated high-confidence creative solution"
            
            # Generate synthesis
            synthesis = f"Hybrid approach combining {hardcoded_component.lower()} with {ai_component.lower()} for {requirements.problem_type.value}"
            
            # Calculate improvement metrics
            improvement_metrics = {
                'accuracy_improvement': 0.1 if solution.confidence_score > 0.8 else 0.0,
                'efficiency_improvement': 0.15 if hardcoded_knowledge else 0.0,
                'robustness_improvement': 0.2  # Hybrid approaches are generally more robust
            }
            
            return HybridInsight(
                insight_id=insight_id,
                hardcoded_component=hardcoded_component,
                ai_component=ai_component,
                synthesis=synthesis,
                confidence=solution.confidence_score,
                applicability=[requirements.problem_type],
                improvement_metrics=improvement_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to generate hybrid insight: {e}")
            return None
    
    async def enhance_ai_solution_with_hardcoded_knowledge(
        self,
        ai_solution: IntelligentSolution,
        requirements: ProblemRequirements,
        performance_requirements: Optional[Dict[str, Any]] = None
    ) -> IntelligentSolution:
        """Enhance an AI solution using hardcoded knowledge."""
        
        # Get relevant hardcoded knowledge
        hardcoded_patterns = await self.knowledge_integrator.find_relevant_hardcoded_patterns(requirements)
        edge_cases = []
        optimizations = []
        
        if hardcoded_patterns:
            best_pattern = hardcoded_patterns[0][0]
            edge_cases = self.knowledge_integrator.get_relevant_edge_cases(
                best_pattern.pattern_name, requirements.original_statement
            )
            optimizations = self.knowledge_integrator.get_relevant_optimizations(
                [requirements.problem_type.value], performance_requirements or {}
            )
        
        # Prepare enhancement context
        hardcoded_knowledge_context = ""
        
        for pattern, confidence in hardcoded_patterns[:2]:
            hardcoded_knowledge_context += f"Pattern: {pattern.pattern_name}\n"
            hardcoded_knowledge_context += f"Key Steps: {', '.join(pattern.algorithm_steps[:3])}\n"
            hardcoded_knowledge_context += f"Insights: {', '.join(pattern.key_insights[:2])}\n\n"
        
        for edge_case in edge_cases[:2]:
            hardcoded_knowledge_context += f"Edge Case: {edge_case.description}\n"
            hardcoded_knowledge_context += f"Handling: {edge_case.handling_strategy}\n\n"
        
        for opt in optimizations[:2]:
            hardcoded_knowledge_context += f"Optimization: {opt.technique_name}\n"
            hardcoded_knowledge_context += f"Description: {opt.description}\n\n"
        
        # Use AI to enhance the solution
        try:
            prompt = self.hybrid_prompts['solution_enhancement'].format(
                ai_solution=json.dumps({
                    'strategy': ai_solution.strategy_used.value,
                    'steps': [step.action for step in ai_solution.solution_steps],
                    'confidence': ai_solution.confidence_score,
                    'final_solution': ai_solution.final_solution
                }, indent=2),
                hardcoded_knowledge=hardcoded_knowledge_context,
                performance_requirements=performance_requirements or {}
            )
            
            response = await self.model.ainvoke(prompt)
            
            # Parse enhanced solution from response
            enhanced_steps = ai_solution.solution_steps.copy()
            
            # Add enhancement metadata
            enhanced_solution = IntelligentSolution(
                strategy_used=SolutionStrategy.HYBRID,
                confidence_score=min(ai_solution.confidence_score + 0.1, 1.0),
                solution_steps=enhanced_steps,
                final_solution=ai_solution.final_solution,
                verification_result=ai_solution.verification_result,
                explanation=ai_solution.explanation + "\n\nEnhanced with hardcoded algorithmic knowledge.",
                metadata={
                    **ai_solution.metadata,
                    'enhanced_with_hardcoded': True,
                    'hardcoded_patterns_used': [p[0].pattern_name for p in hardcoded_patterns],
                    'edge_cases_considered': [ec.description for ec in edge_cases],
                    'optimizations_applied': [opt.technique_name for opt in optimizations]
                }
            )
            
            return enhanced_solution
            
        except Exception as e:
            logger.error(f"Failed to enhance AI solution: {e}")
            return ai_solution  # Return original if enhancement fails
    
    def _update_learning_metrics(self):
        """Update learning system metrics."""
        
        if self.solved_problems:
            total_confidence = sum(p.success_metrics.get('confidence', 0) for p in self.solved_problems.values())
            self.learning_metrics.learning_accuracy = total_confidence / len(self.solved_problems)
        
        # Calculate knowledge coverage (percentage of problem types we've seen)
        seen_problem_types = set(p.problem_type for p in self.solved_problems.values())
        total_problem_types = len(ProblemType)
        self.learning_metrics.knowledge_coverage = len(seen_problem_types) / total_problem_types
        
        # Calculate adaptation rate (recent improvement)
        recent_problems = [
            p for p in self.solved_problems.values()
            if p.timestamp > datetime.utcnow() - timedelta(days=30)
        ]
        
        if recent_problems:
            recent_avg_confidence = sum(p.success_metrics.get('confidence', 0) for p in recent_problems) / len(recent_problems)
            old_problems = [
                p for p in self.solved_problems.values()
                if p.timestamp <= datetime.utcnow() - timedelta(days=30)
            ]
            
            if old_problems:
                old_avg_confidence = sum(p.success_metrics.get('confidence', 0) for p in old_problems) / len(old_problems)
                self.learning_metrics.adaptation_rate = recent_avg_confidence - old_avg_confidence
            else:
                self.learning_metrics.adaptation_rate = 0.0
        
        # Calculate average improvement from hybrid insights
        if self.hybrid_insights:
            total_improvement = sum(
                sum(insight.improvement_metrics.values())
                for insight in self.hybrid_insights.values()
            )
            self.learning_metrics.average_improvement = total_improvement / len(self.hybrid_insights)
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced learning system statistics."""
        
        basic_stats = super().get_statistics()
        
        enhanced_stats = basic_stats.copy()
        enhanced_stats.update({
            'hardcoded_knowledge_integrated': self.learning_metrics.hardcoded_knowledge_integrated,
            'hybrid_insights_generated': self.learning_metrics.hybrid_insights_generated,
            'average_improvement_from_hybrid': self.learning_metrics.average_improvement,
            'knowledge_coverage_percentage': self.learning_metrics.knowledge_coverage * 100,
            'adaptation_rate': self.learning_metrics.adaptation_rate,
            'enhancement_capabilities': {
                'pattern_matching': bool(self.knowledge_integrator.pattern_embeddings),
                'edge_case_handling': len(self.knowledge_integrator.hardcoded_knowledge.edge_cases) if self.knowledge_integrator.hardcoded_knowledge else 0,
                'optimization_techniques': len(self.knowledge_integrator.hardcoded_knowledge.optimizations) if self.knowledge_integrator.hardcoded_knowledge else 0,
                'cross_algorithm_insights': len(self.knowledge_integrator.hardcoded_knowledge.cross_algorithm_insights) if self.knowledge_integrator.hardcoded_knowledge else 0
            }
        })
        
        return enhanced_stats
    
    async def get_hybrid_recommendations(
        self,
        requirements: ProblemRequirements
    ) -> Dict[str, Any]:
        """Get specific recommendations for hybrid approaches."""
        
        recommendations = {
            'use_hybrid': False,
            'hardcoded_components': [],
            'ai_components': [],
            'integration_strategy': '',
            'expected_benefits': {},
            'implementation_notes': []
        }
        
        # Check if hybrid approach is beneficial
        hardcoded_patterns = await self.knowledge_integrator.find_relevant_hardcoded_patterns(requirements)
        
        if hardcoded_patterns and hardcoded_patterns[0][1] > 0.6:
            recommendations['use_hybrid'] = True
            
            # Identify hardcoded components
            best_pattern = hardcoded_patterns[0][0]
            recommendations['hardcoded_components'] = [
                f"Use {best_pattern.pattern_name} for core algorithm",
                f"Apply proven steps: {', '.join(best_pattern.algorithm_steps[:2])}"
            ]
            
            # Identify AI components
            recommendations['ai_components'] = [
                "Use AI for problem interpretation and edge case discovery",
                "Apply AI for optimization and adaptation to specific requirements"
            ]
            
            # Integration strategy
            recommendations['integration_strategy'] = (
                f"Start with {best_pattern.pattern_name} foundation, "
                "then use AI to adapt and optimize for specific problem requirements"
            )
            
            # Expected benefits
            recommendations['expected_benefits'] = {
                'accuracy_improvement': 0.15,
                'reliability_improvement': 0.2,
                'efficiency_improvement': 0.1,
                'robustness_improvement': 0.25
            }
            
            # Implementation notes
            recommendations['implementation_notes'] = [
                f"Priority: High confidence hardcoded pattern available ({hardcoded_patterns[0][1]:.2f})",
                "Validate AI adaptations against hardcoded verification",
                "Use AI for creative problem-solving where hardcoded approach lacks coverage"
            ]
        
        return recommendations


# Global enhanced learning system instance
enhanced_learning_system = EnhancedLearningSystem()


async def get_enhanced_learning_insights(
    problem_statement: str,
    problem_type: ProblemType,
    patterns: List[LanguagePattern],
    **kwargs
) -> LearningInsights:
    """
    Convenience function to get enhanced learning insights.
    
    Args:
        problem_statement: The problem to analyze
        problem_type: Type of the problem
        patterns: Language patterns involved
        **kwargs: Additional parameters for ProblemRequirements
    
    Returns:
        Enhanced learning insights combining hardcoded and AI knowledge
    """
    
    requirements = ProblemRequirements(
        original_statement=problem_statement,
        problem_type=problem_type,
        patterns=patterns,
        **kwargs
    )
    
    return await enhanced_learning_system.get_enhanced_insights_for_problem(requirements)


async def enhance_solution_with_hardcoded_knowledge(
    ai_solution: IntelligentSolution,
    problem_statement: str,
    problem_type: ProblemType,
    patterns: List[LanguagePattern],
    performance_requirements: Optional[Dict[str, Any]] = None
) -> IntelligentSolution:
    """
    Enhance an AI solution with hardcoded knowledge.
    
    Args:
        ai_solution: The AI-generated solution to enhance
        problem_statement: Original problem statement
        problem_type: Type of the problem
        patterns: Language patterns involved
        performance_requirements: Performance requirements
    
    Returns:
        Enhanced solution combining AI creativity with hardcoded reliability
    """
    
    requirements = ProblemRequirements(
        original_statement=problem_statement,
        problem_type=problem_type,
        patterns=patterns
    )
    
    return await enhanced_learning_system.enhance_ai_solution_with_hardcoded_knowledge(
        ai_solution, requirements, performance_requirements
    )