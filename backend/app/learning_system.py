"""
Learning System for Theory of Computation Problem Solving
Learns from solved problems to improve future solutions.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import hashlib

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationSummaryMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .ai_config import AIConfig, ModelType
from .problem_understanding import ProblemRequirements, ProblemType, LanguagePattern
from .intelligent_solver import IntelligentSolution, SolutionStrategy

logger = logging.getLogger(__name__)


@dataclass
class SolvedProblem:
    """Represents a solved problem for learning."""
    problem_id: str
    problem_statement: str
    problem_type: ProblemType
    requirements: ProblemRequirements
    solution: IntelligentSolution
    patterns: List[LanguagePattern]
    strategy_used: SolutionStrategy
    success_metrics: Dict[str, float]
    timestamp: datetime
    feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternKnowledge(BaseModel):
    """Knowledge about a specific pattern."""
    pattern_type: LanguagePattern
    successful_strategies: List[SolutionStrategy]
    common_states_count: Dict[str, int]  # pattern -> typical state count
    common_transitions: List[Dict[str, Any]]
    success_rate: float
    example_problems: List[str]
    insights: List[str]


class StrategyPerformance(BaseModel):
    """Performance metrics for a solution strategy."""
    strategy: SolutionStrategy
    problem_types: Dict[ProblemType, float]  # problem type -> success rate
    average_confidence: float
    total_uses: int
    successful_uses: int
    average_solution_time: float
    best_for_patterns: List[LanguagePattern]


class LearningInsights(BaseModel):
    """Insights derived from learning."""
    recommended_strategy: SolutionStrategy
    confidence_boost: float
    similar_problems: List[str]
    pattern_matches: List[str]
    optimization_suggestions: List[str]
    predicted_difficulty: float


class LearningSystem:
    """
    Learning system that improves problem-solving over time by learning from past solutions.
    """
    
    def __init__(self, storage_path: str = "./learning_data"):
        self.storage_path = storage_path
        self.config = AIConfig()
        self.model = self.config.get_model(ModelType.GENERAL)
        
        # Knowledge bases
        self.solved_problems: Dict[str, SolvedProblem] = {}
        self.pattern_knowledge: Dict[LanguagePattern, PatternKnowledge] = {}
        self.strategy_performance: Dict[SolutionStrategy, StrategyPerformance] = {}
        
        # Embeddings for similarity search
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.problem_vectors = None  # Will be FAISS index
        
        # Learning prompts
        self.analysis_prompt = self._create_analysis_prompt()
        self.insight_prompt = self._create_insight_prompt()
        
        # Load existing knowledge
        self._load_knowledge()
        
        logger.info("Learning System initialized")
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for analyzing solved problems."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Analyze this solved problem to extract learning insights.
            
            Identify:
            1. Key patterns that led to success
            2. Why this strategy worked
            3. Potential improvements
            4. Similar problem characteristics
            5. Reusable solution components
            
            Provide actionable insights for future problems."""),
            HumanMessage(content="{problem_data}")
        ])
    
    def _create_insight_prompt(self) -> ChatPromptTemplate:
        """Create prompt for generating insights."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Based on similar solved problems, generate insights for this new problem.
            
            Consider:
            1. What strategies worked for similar problems
            2. Common pitfalls to avoid
            3. Optimization opportunities
            4. Pattern-specific approaches
            
            Provide specific, actionable recommendations."""),
            HumanMessage(content="{context}")
        ])
    
    async def learn_from_solution(
        self,
        requirements: ProblemRequirements,
        solution: IntelligentSolution,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Learn from a successfully solved problem.
        """
        
        # Create solved problem record
        problem_id = self._generate_problem_id(requirements.original_statement)
        
        solved_problem = SolvedProblem(
            problem_id=problem_id,
            problem_statement=requirements.original_statement,
            problem_type=requirements.problem_type,
            requirements=requirements,
            solution=solution,
            patterns=requirements.patterns,
            strategy_used=solution.strategy_used,
            success_metrics={
                "confidence": solution.confidence_score,
                "verification_score": solution.verification_result.get("correctness", 0),
                "steps_count": len(solution.solution_steps)
            },
            timestamp=datetime.utcnow(),
            feedback=feedback
        )
        
        # Store the solved problem
        self.solved_problems[problem_id] = solved_problem
        
        # Update pattern knowledge
        await self._update_pattern_knowledge(solved_problem)
        
        # Update strategy performance
        self._update_strategy_performance(solved_problem)
        
        # Update vector database
        await self._update_vector_database(solved_problem)
        
        # Generate learning insights
        insights = await self._generate_learning_insights(solved_problem)
        
        # Persist knowledge
        self._save_knowledge()
        
        logger.info(f"Learned from problem {problem_id}")
        
        return {
            "problem_id": problem_id,
            "patterns_learned": [p.value for p in requirements.patterns],
            "strategy_effectiveness": solution.confidence_score,
            "insights": insights,
            "knowledge_updated": True
        }
    
    async def get_insights_for_problem(
        self,
        requirements: ProblemRequirements
    ) -> LearningInsights:
        """
        Get learning insights for a new problem based on past experience.
        """
        
        # Find similar problems
        similar_problems = await self._find_similar_problems(requirements)
        
        # Analyze patterns
        pattern_insights = self._analyze_patterns(requirements.patterns)
        
        # Recommend strategy
        recommended_strategy = self._recommend_strategy(
            requirements.problem_type,
            requirements.patterns
        )
        
        # Calculate confidence boost
        confidence_boost = self._calculate_confidence_boost(
            similar_problems,
            pattern_insights
        )
        
        # Generate optimization suggestions
        suggestions = await self._generate_suggestions(
            requirements,
            similar_problems
        )
        
        # Predict difficulty
        difficulty = self._predict_difficulty(requirements, similar_problems)
        
        return LearningInsights(
            recommended_strategy=recommended_strategy,
            confidence_boost=confidence_boost,
            similar_problems=[p.problem_statement for p in similar_problems[:3]],
            pattern_matches=[p.value for p in requirements.patterns],
            optimization_suggestions=suggestions,
            predicted_difficulty=difficulty
        )
    
    async def _update_pattern_knowledge(self, solved_problem: SolvedProblem):
        """Update knowledge about patterns."""
        
        for pattern in solved_problem.patterns:
            if pattern not in self.pattern_knowledge:
                self.pattern_knowledge[pattern] = PatternKnowledge(
                    pattern_type=pattern,
                    successful_strategies=[],
                    common_states_count={},
                    common_transitions=[],
                    success_rate=0.0,
                    example_problems=[],
                    insights=[]
                )
            
            knowledge = self.pattern_knowledge[pattern]
            
            # Update successful strategies
            if solved_problem.strategy_used not in knowledge.successful_strategies:
                knowledge.successful_strategies.append(solved_problem.strategy_used)
            
            # Update state count statistics
            state_count = len(solved_problem.solution.final_solution.get("states", []))
            pattern_key = pattern.value
            if pattern_key not in knowledge.common_states_count:
                knowledge.common_states_count[pattern_key] = 0
            knowledge.common_states_count[pattern_key] += 1
            
            # Add example problem
            if len(knowledge.example_problems) < 10:
                knowledge.example_problems.append(solved_problem.problem_statement[:100])
            
            # Update success rate
            total_problems = len([
                p for p in self.solved_problems.values()
                if pattern in p.patterns
            ])
            successful_problems = len([
                p for p in self.solved_problems.values()
                if pattern in p.patterns and p.success_metrics["confidence"] > 0.7
            ])
            knowledge.success_rate = successful_problems / total_problems if total_problems > 0 else 0
            
            # Generate pattern-specific insights
            insight = await self._generate_pattern_insight(pattern, solved_problem)
            if insight and len(knowledge.insights) < 20:
                knowledge.insights.append(insight)
    
    def _update_strategy_performance(self, solved_problem: SolvedProblem):
        """Update performance metrics for strategies."""
        
        strategy = solved_problem.strategy_used
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = StrategyPerformance(
                strategy=strategy,
                problem_types={},
                average_confidence=0.0,
                total_uses=0,
                successful_uses=0,
                average_solution_time=0.0,
                best_for_patterns=[]
            )
        
        performance = self.strategy_performance[strategy]
        
        # Update problem type success rates
        problem_type = solved_problem.problem_type
        if problem_type not in performance.problem_types:
            performance.problem_types[problem_type] = 0.0
        
        # Update metrics
        performance.total_uses += 1
        if solved_problem.success_metrics["confidence"] > 0.7:
            performance.successful_uses += 1
        
        # Update average confidence
        performance.average_confidence = (
            (performance.average_confidence * (performance.total_uses - 1) +
             solved_problem.success_metrics["confidence"]) /
            performance.total_uses
        )
        
        # Update problem type success rate
        type_problems = [
            p for p in self.solved_problems.values()
            if p.problem_type == problem_type and p.strategy_used == strategy
        ]
        type_successful = [
            p for p in type_problems
            if p.success_metrics["confidence"] > 0.7
        ]
        performance.problem_types[problem_type] = (
            len(type_successful) / len(type_problems) if type_problems else 0
        )
        
        # Identify best patterns for this strategy
        pattern_success = defaultdict(list)
        for problem in self.solved_problems.values():
            if problem.strategy_used == strategy:
                for pattern in problem.patterns:
                    pattern_success[pattern].append(
                        problem.success_metrics["confidence"]
                    )
        
        best_patterns = []
        for pattern, confidences in pattern_success.items():
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence > 0.8:
                best_patterns.append(pattern)
        
        performance.best_for_patterns = best_patterns
    
    async def _update_vector_database(self, solved_problem: SolvedProblem):
        """Update vector database with new problem."""
        
        # Create embedding for the problem
        problem_text = f"{solved_problem.problem_statement} {solved_problem.problem_type.value}"
        
        if self.problem_vectors is None:
            # Initialize FAISS index
            texts = [problem_text]
            self.problem_vectors = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"problem_id": solved_problem.problem_id}]
            )
        else:
            # Add to existing index
            self.problem_vectors.add_texts(
                [problem_text],
                metadatas=[{"problem_id": solved_problem.problem_id}]
            )
    
    async def _find_similar_problems(
        self,
        requirements: ProblemRequirements,
        top_k: int = 5
    ) -> List[SolvedProblem]:
        """Find similar solved problems."""
        
        if not self.solved_problems or self.problem_vectors is None:
            return []
        
        # Create query embedding
        query_text = f"{requirements.original_statement} {requirements.problem_type.value}"
        
        # Search for similar problems
        similar_docs = self.problem_vectors.similarity_search(
            query_text,
            k=min(top_k, len(self.solved_problems))
        )
        
        # Get solved problems
        similar_problems = []
        for doc in similar_docs:
            problem_id = doc.metadata.get("problem_id")
            if problem_id in self.solved_problems:
                similar_problems.append(self.solved_problems[problem_id])
        
        return similar_problems
    
    def _analyze_patterns(
        self,
        patterns: List[LanguagePattern]
    ) -> Dict[str, Any]:
        """Analyze patterns based on knowledge base."""
        
        insights = {}
        
        for pattern in patterns:
            if pattern in self.pattern_knowledge:
                knowledge = self.pattern_knowledge[pattern]
                insights[pattern.value] = {
                    "success_rate": knowledge.success_rate,
                    "best_strategies": knowledge.successful_strategies[:3],
                    "typical_complexity": knowledge.common_states_count,
                    "key_insights": knowledge.insights[:3]
                }
        
        return insights
    
    def _recommend_strategy(
        self,
        problem_type: ProblemType,
        patterns: List[LanguagePattern]
    ) -> SolutionStrategy:
        """Recommend best strategy based on learning."""
        
        # Score each strategy
        strategy_scores = {}
        
        for strategy, performance in self.strategy_performance.items():
            score = 0.0
            
            # Score based on problem type performance
            if problem_type in performance.problem_types:
                score += performance.problem_types[problem_type] * 2
            
            # Score based on pattern compatibility
            pattern_match = len(set(patterns) & set(performance.best_for_patterns))
            score += pattern_match * 0.5
            
            # Score based on overall performance
            score += performance.average_confidence
            
            strategy_scores[strategy] = score
        
        # Return best strategy or default
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)
        else:
            return SolutionStrategy.CONSTRUCTION
    
    def _calculate_confidence_boost(
        self,
        similar_problems: List[SolvedProblem],
        pattern_insights: Dict[str, Any]
    ) -> float:
        """Calculate confidence boost from prior knowledge."""
        
        boost = 0.0
        
        # Boost from similar problems
        if similar_problems:
            avg_confidence = sum(
                p.success_metrics["confidence"] for p in similar_problems
            ) / len(similar_problems)
            boost += avg_confidence * 0.2
        
        # Boost from pattern knowledge
        if pattern_insights:
            avg_success = sum(
                insights.get("success_rate", 0)
                for insights in pattern_insights.values()
            ) / len(pattern_insights)
            boost += avg_success * 0.1
        
        return min(boost, 0.3)  # Cap at 30% boost
    
    async def _generate_suggestions(
        self,
        requirements: ProblemRequirements,
        similar_problems: List[SolvedProblem]
    ) -> List[str]:
        """Generate optimization suggestions."""
        
        suggestions = []
        
        # Analyze similar problems for common optimizations
        if similar_problems:
            # Find common successful approaches
            common_approaches = defaultdict(int)
            for problem in similar_problems:
                for step in problem.solution.solution_steps:
                    common_approaches[step.action] += 1
            
            # Suggest most common approaches
            top_approaches = sorted(
                common_approaches.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for approach, count in top_approaches:
                if count > len(similar_problems) / 2:
                    suggestions.append(f"Consider using {approach} (worked in {count} similar problems)")
        
        # Pattern-specific suggestions
        for pattern in requirements.patterns:
            if pattern in self.pattern_knowledge:
                knowledge = self.pattern_knowledge[pattern]
                if knowledge.insights:
                    suggestions.append(knowledge.insights[0])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _predict_difficulty(
        self,
        requirements: ProblemRequirements,
        similar_problems: List[SolvedProblem]
    ) -> float:
        """Predict problem difficulty."""
        
        difficulty = 0.5  # Base difficulty
        
        # Adjust based on problem type
        complex_types = [
            ProblemType.TM_CONSTRUCTION,
            ProblemType.PUMPING_LEMMA_PROOF,
            ProblemType.DECIDABILITY
        ]
        if requirements.problem_type in complex_types:
            difficulty += 0.2
        
        # Adjust based on pattern count
        difficulty += len(requirements.patterns) * 0.1
        
        # Adjust based on similar problems
        if similar_problems:
            avg_steps = sum(
                len(p.solution.solution_steps) for p in similar_problems
            ) / len(similar_problems)
            difficulty += (avg_steps - 5) * 0.02  # More steps = harder
        
        return min(max(difficulty, 0.0), 1.0)
    
    async def _generate_pattern_insight(
        self,
        pattern: LanguagePattern,
        solved_problem: SolvedProblem
    ) -> str:
        """Generate insight for a specific pattern."""
        
        prompt = f"""
        Generate a concise insight about solving {pattern.value} problems:
        
        Problem: {solved_problem.problem_statement[:200]}
        Strategy used: {solved_problem.strategy_used.value}
        Success: {solved_problem.success_metrics['confidence']:.2f}
        
        What key insight would help solve similar problems?
        Keep it under 100 characters.
        """
        
        response = await self.model.ainvoke(prompt)
        return response.content[:100]
    
    async def _generate_learning_insights(
        self,
        solved_problem: SolvedProblem
    ) -> List[str]:
        """Generate learning insights from a solved problem."""
        
        prompt = self.analysis_prompt.format(
            problem_data=json.dumps({
                "problem": solved_problem.problem_statement,
                "type": solved_problem.problem_type.value,
                "patterns": [p.value for p in solved_problem.patterns],
                "strategy": solved_problem.strategy_used.value,
                "confidence": solved_problem.success_metrics["confidence"],
                "steps": len(solved_problem.solution.solution_steps)
            })
        )
        
        response = await self.model.ainvoke(prompt)
        
        # Extract insights from response
        insights = []
        for line in response.content.split("\n"):
            if line.strip() and not line.startswith("#"):
                insights.append(line.strip())
        
        return insights[:5]
    
    def _generate_problem_id(self, statement: str) -> str:
        """Generate unique ID for a problem."""
        return hashlib.md5(statement.encode()).hexdigest()[:12]
    
    def _save_knowledge(self):
        """Persist knowledge to storage."""
        try:
            # Save solved problems
            with open(f"{self.storage_path}/solved_problems.pkl", "wb") as f:
                pickle.dump(self.solved_problems, f)
            
            # Save pattern knowledge
            with open(f"{self.storage_path}/pattern_knowledge.pkl", "wb") as f:
                pickle.dump(self.pattern_knowledge, f)
            
            # Save strategy performance
            with open(f"{self.storage_path}/strategy_performance.pkl", "wb") as f:
                pickle.dump(self.strategy_performance, f)
            
            # Save vector database
            if self.problem_vectors:
                self.problem_vectors.save_local(f"{self.storage_path}/vectors")
            
            logger.info("Knowledge saved successfully")
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
    
    def _load_knowledge(self):
        """Load knowledge from storage."""
        try:
            # Load solved problems
            with open(f"{self.storage_path}/solved_problems.pkl", "rb") as f:
                self.solved_problems = pickle.load(f)
            
            # Load pattern knowledge
            with open(f"{self.storage_path}/pattern_knowledge.pkl", "rb") as f:
                self.pattern_knowledge = pickle.load(f)
            
            # Load strategy performance
            with open(f"{self.storage_path}/strategy_performance.pkl", "rb") as f:
                self.strategy_performance = pickle.load(f)
            
            # Load vector database
            self.problem_vectors = FAISS.load_local(
                f"{self.storage_path}/vectors",
                self.embeddings
            )
            
            logger.info(f"Loaded knowledge: {len(self.solved_problems)} problems")
        except Exception as e:
            logger.info(f"No existing knowledge found: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        
        return {
            "total_problems_solved": len(self.solved_problems),
            "patterns_learned": len(self.pattern_knowledge),
            "strategies_evaluated": len(self.strategy_performance),
            "average_confidence": sum(
                p.success_metrics["confidence"] for p in self.solved_problems.values()
            ) / len(self.solved_problems) if self.solved_problems else 0,
            "most_successful_strategy": max(
                self.strategy_performance.items(),
                key=lambda x: x[1].average_confidence
            )[0].value if self.strategy_performance else None,
            "most_common_pattern": max(
                [(p, len([sp for sp in self.solved_problems.values() if p in sp.patterns]))
                 for p in self.pattern_knowledge.keys()],
                key=lambda x: x[1]
            )[0].value if self.pattern_knowledge else None
        }