"""
Tutoring Workflow Implementation using LangGraph.
Provides adaptive, personalized tutoring with checkpointing and learning path optimization.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .langgraph_core import (
    BaseWorkflowNode, ConversationState, WorkflowGraphBuilder, 
    WorkflowConfig, InterruptType, workflow_executor
)
from .orchestrator import ExecutionMode
from .adaptive_learning import AdaptiveLearningEngine
from .prompts import prompt_builder

logger = logging.getLogger(__name__)


class TutoringMode(str, Enum):
    """Different modes of tutoring interaction."""
    INTRODUCTION = "introduction"
    CONCEPT_EXPLANATION = "concept_explanation"
    PRACTICE_PROBLEM = "practice_problem"
    GUIDED_SOLUTION = "guided_solution"
    ASSESSMENT = "assessment"
    REVIEW = "review"
    REMEDIATION = "remediation"


class DifficultyLevel(str, Enum):
    """Difficulty levels for content adaptation."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningStyle(str, Enum):
    """Learning style preferences."""
    VISUAL = "visual"
    ANALYTICAL = "analytical"
    PRACTICAL = "practical"
    THEORETICAL = "theoretical"


class TutoringState(ConversationState):
    """Extended state for tutoring workflow."""
    tutoring_mode: TutoringMode
    topic: str
    difficulty_level: DifficultyLevel
    learning_style: LearningStyle
    student_progress: Dict[str, Any]
    current_concept: Optional[str]
    practice_problems: List[Dict[str, Any]]
    assessment_results: Dict[str, Any]
    misconceptions: List[str]
    learning_objectives: List[str]
    adaptive_hints: List[str]
    performance_metrics: Dict[str, float]


class ConceptIntroductionNode(BaseWorkflowNode):
    """Node for introducing new concepts with adaptive explanations."""
    
    def __init__(self):
        super().__init__("concept_introduction")
        self.learning_engine = AdaptiveLearningEngine()
    
    async def execute(self, state: TutoringState) -> TutoringState:
        """Introduce a concept based on student's level and style."""
        try:
            topic = state["topic"]
            difficulty = state["difficulty_level"]
            learning_style = state["learning_style"]
            
            # Build context-aware prompt
            prompt = prompt_builder.build(
                "concept_introduction",
                {
                    "topic": topic,
                    "difficulty_level": difficulty,
                    "learning_style": learning_style,
                    "prior_knowledge": state["student_progress"].get("mastered_concepts", []),
                    "known_misconceptions": state["misconceptions"]
                }
            )
            
            # Generate explanation using orchestrator
            response = await self.orchestrator.execute(
                task="concept_explanation",
                prompt=prompt,
                mode=ExecutionMode.ENSEMBLE,
                temperature=0.3
            )
            
            # Extract learning objectives
            objectives = await self._extract_learning_objectives(topic, difficulty)
            state["learning_objectives"] = objectives
            
            # Add AI response to conversation
            ai_message = AIMessage(content=response.get("response", ""))
            state["messages"].append(ai_message)
            
            # Update current concept
            state["current_concept"] = topic
            state["tutoring_mode"] = TutoringMode.CONCEPT_EXPLANATION
            
            # Log learning event
            await self.learning_engine.log_learning_event(
                user_id=state["user_id"],
                event_type="concept_introduced",
                content={"topic": topic, "difficulty": difficulty}
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Concept introduction failed: {e}")
            return await self.on_error(state, e)
    
    async def _extract_learning_objectives(self, topic: str, difficulty: str) -> List[str]:
        """Extract learning objectives for the topic."""
        prompt = f"List 3-5 specific learning objectives for {topic} at {difficulty} level."
        
        response = await self.orchestrator.execute(
            task="objective_extraction",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL
        )
        
        # Parse objectives from response
        objectives = []
        if isinstance(response, dict) and "response" in response:
            content = response["response"]
            # Simple parsing - could be enhanced with structured output
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith(('-', '*', '•')) or line.strip().startswith(tuple('123456789')):
                    objectives.append(line.strip().lstrip('-*•123456789. '))
        
        return objectives[:5]  # Limit to 5 objectives


class PracticeProblemNode(BaseWorkflowNode):
    """Node for generating and managing practice problems."""
    
    def __init__(self):
        super().__init__("practice_problem")
        self.learning_engine = AdaptiveLearningEngine()
    
    async def execute(self, state: TutoringState) -> TutoringState:
        """Generate adaptive practice problems."""
        try:
            topic = state["current_concept"] or state["topic"]
            difficulty = state["difficulty_level"]
            
            # Analyze student performance to adapt difficulty
            performance = state["performance_metrics"]
            adapted_difficulty = await self._adapt_difficulty(performance, difficulty)
            
            # Generate practice problem
            problem = await self._generate_practice_problem(
                topic, adapted_difficulty, state["learning_style"]
            )
            
            # Add problem to state
            if "practice_problems" not in state:
                state["practice_problems"] = []
            state["practice_problems"].append(problem)
            
            # Create problem message
            problem_message = AIMessage(content=f"""
Let's practice with this problem:

**Problem:** {problem['question']}

{problem.get('context', '')}

Take your time to work through this. If you need help, just ask for a hint!
            """.strip())
            
            state["messages"].append(problem_message)
            state["tutoring_mode"] = TutoringMode.PRACTICE_PROBLEM
            
            # Set up for human input
            state["metadata"]["awaiting_student_response"] = True
            state["metadata"]["current_problem"] = problem
            
            return state
            
        except Exception as e:
            logger.error(f"Practice problem generation failed: {e}")
            return await self.on_error(state, e)
    
    async def _adapt_difficulty(self, performance: Dict[str, float], current_difficulty: str) -> str:
        """Adapt difficulty based on student performance."""
        if not performance:
            return current_difficulty
        
        accuracy = performance.get("accuracy", 0.5)
        confidence = performance.get("confidence", 0.5)
        
        # Simple adaptation logic
        if accuracy > 0.8 and confidence > 0.7:
            # Increase difficulty
            if current_difficulty == "beginner":
                return "intermediate"
            elif current_difficulty == "intermediate":
                return "advanced"
        elif accuracy < 0.5 or confidence < 0.4:
            # Decrease difficulty
            if current_difficulty == "advanced":
                return "intermediate"
            elif current_difficulty == "intermediate":
                return "beginner"
        
        return current_difficulty
    
    async def _generate_practice_problem(
        self, 
        topic: str, 
        difficulty: str, 
        learning_style: str
    ) -> Dict[str, Any]:
        """Generate a practice problem."""
        prompt = prompt_builder.build(
            "practice_problem_generation",
            {
                "topic": topic,
                "difficulty": difficulty,
                "learning_style": learning_style,
                "format": "structured_problem"
            }
        )
        
        response = await self.orchestrator.execute(
            task="problem_generation",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.5
        )
        
        # Parse structured response
        content = response.get("response", "")
        return {
            "question": content,
            "topic": topic,
            "difficulty": difficulty,
            "generated_at": datetime.now().isoformat(),
            "problem_id": f"prob_{topic}_{int(datetime.now().timestamp())}"
        }


class AssessmentNode(BaseWorkflowNode):
    """Node for evaluating student responses and providing feedback."""
    
    def __init__(self):
        super().__init__("assessment")
        self.learning_engine = AdaptiveLearningEngine()
    
    async def execute(self, state: TutoringState) -> TutoringState:
        """Assess student response and provide feedback."""
        try:
            if not state["messages"]:
                return state
            
            last_message = state["messages"][-1]
            if not isinstance(last_message, HumanMessage):
                return state
            
            student_response = last_message.content
            current_problem = state["metadata"].get("current_problem")
            
            if not current_problem:
                logger.warning("No current problem found for assessment")
                return state
            
            # Evaluate response
            evaluation = await self._evaluate_response(
                student_response, 
                current_problem, 
                state["current_concept"]
            )
            
            # Update performance metrics
            await self._update_performance_metrics(state, evaluation)
            
            # Generate feedback
            feedback = await self._generate_feedback(
                student_response, 
                evaluation, 
                state["learning_style"]
            )
            
            # Add feedback to conversation
            feedback_message = AIMessage(content=feedback)
            state["messages"].append(feedback_message)
            
            # Determine next action based on assessment
            next_mode = await self._determine_next_mode(evaluation, state)
            state["tutoring_mode"] = next_mode
            
            # Clean up metadata
            state["metadata"].pop("awaiting_student_response", None)
            
            # Log assessment event
            await self.learning_engine.log_learning_event(
                user_id=state["user_id"],
                event_type="response_assessed",
                content={
                    "problem_id": current_problem.get("problem_id"),
                    "accuracy": evaluation["accuracy"],
                    "completeness": evaluation["completeness"]
                }
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            return await self.on_error(state, e)
    
    async def _evaluate_response(
        self, 
        response: str, 
        problem: Dict[str, Any], 
        concept: str
    ) -> Dict[str, Any]:
        """Evaluate student response quality and correctness."""
        prompt = prompt_builder.build(
            "response_evaluation",
            {
                "student_response": response,
                "problem": problem["question"],
                "concept": concept,
                "evaluation_criteria": ["correctness", "completeness", "understanding", "approach"]
            }
        )
        
        result = await self.orchestrator.execute(
            task="response_evaluation",
            prompt=prompt,
            mode=ExecutionMode.ENSEMBLE,
            temperature=0.1
        )
        
        # Parse evaluation (simplified - could use structured output)
        evaluation_content = result.get("response", "")
        
        return {
            "accuracy": 0.8,  # Placeholder - would parse from AI response
            "completeness": 0.7,
            "understanding_level": 0.75,
            "approach_quality": 0.8,
            "identified_errors": [],
            "positive_aspects": [],
            "areas_for_improvement": [],
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    async def _update_performance_metrics(
        self, 
        state: TutoringState, 
        evaluation: Dict[str, Any]
    ):
        """Update student performance metrics."""
        if "performance_metrics" not in state:
            state["performance_metrics"] = {
                "total_problems": 0,
                "correct_responses": 0,
                "accuracy": 0.0,
                "confidence": 0.5,
                "progress_rate": 0.0
            }
        
        metrics = state["performance_metrics"]
        metrics["total_problems"] += 1
        
        if evaluation["accuracy"] >= 0.7:
            metrics["correct_responses"] += 1
        
        metrics["accuracy"] = metrics["correct_responses"] / metrics["total_problems"]
        
        # Update confidence based on recent performance
        metrics["confidence"] = min(
            1.0, 
            metrics["confidence"] * 0.9 + evaluation["accuracy"] * 0.1
        )
    
    async def _generate_feedback(
        self, 
        response: str, 
        evaluation: Dict[str, Any], 
        learning_style: str
    ) -> str:
        """Generate personalized feedback."""
        prompt = prompt_builder.build(
            "feedback_generation",
            {
                "student_response": response,
                "evaluation": evaluation,
                "learning_style": learning_style,
                "feedback_style": "encouraging_constructive"
            }
        )
        
        result = await self.orchestrator.execute(
            task="feedback_generation",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.4
        )
        
        return result.get("response", "Great effort! Let's continue learning.")
    
    async def _determine_next_mode(
        self, 
        evaluation: Dict[str, Any], 
        state: TutoringState
    ) -> TutoringMode:
        """Determine next tutoring mode based on assessment."""
        accuracy = evaluation["accuracy"]
        
        if accuracy >= 0.8:
            # Student did well, move to next concept or more practice
            return TutoringMode.PRACTICE_PROBLEM
        elif accuracy >= 0.5:
            # Partial understanding, provide guided solution
            return TutoringMode.GUIDED_SOLUTION
        else:
            # Need remediation
            return TutoringMode.REMEDIATION


class RemediationNode(BaseWorkflowNode):
    """Node for providing remediation and addressing misconceptions."""
    
    def __init__(self):
        super().__init__("remediation")
    
    async def execute(self, state: TutoringState) -> TutoringState:
        """Provide targeted remediation."""
        try:
            concept = state["current_concept"]
            misconceptions = state.get("misconceptions", [])
            performance = state.get("performance_metrics", {})
            
            # Generate remediation content
            prompt = prompt_builder.build(
                "remediation",
                {
                    "concept": concept,
                    "misconceptions": misconceptions,
                    "performance_data": performance,
                    "learning_style": state["learning_style"]
                }
            )
            
            response = await self.orchestrator.execute(
                task="remediation",
                prompt=prompt,
                mode=ExecutionMode.SEQUENTIAL,
                temperature=0.3
            )
            
            # Create remediation message
            remediation_message = AIMessage(content=f"""
Let's take a step back and work through this concept together.

{response.get('response', '')}

I'll provide some additional examples and break this down into smaller steps.
            """.strip())
            
            state["messages"].append(remediation_message)
            state["tutoring_mode"] = TutoringMode.CONCEPT_EXPLANATION
            
            return state
            
        except Exception as e:
            logger.error(f"Remediation failed: {e}")
            return await self.on_error(state, e)


class TutoringWorkflow:
    """Main tutoring workflow orchestrator."""
    
    def __init__(self):
        self.config = WorkflowConfig(
            max_steps=100,
            timeout_seconds=1800,  # 30 minutes
            enable_checkpointing=True,
            enable_human_in_loop=True
        )
    
    async def create_workflow_graph(self):
        """Create the tutoring workflow graph."""
        try:
            builder = WorkflowGraphBuilder("tutoring_workflow", self.config)
            
            # Add nodes
            builder.add_node(ConceptIntroductionNode())
            builder.add_node(PracticeProblemNode())
            builder.add_node(AssessmentNode())
            builder.add_node(RemediationNode())
            
            # Add edges
            builder.add_edge("concept_introduction", "practice_problem")
            
            # Add conditional edges based on tutoring mode
            builder.add_conditional_edge(
                "practice_problem",
                self._route_after_practice,
                {
                    "assessment": "assessment",
                    "concept_explanation": "concept_introduction",
                    "end": "__end__"
                }
            )
            
            builder.add_conditional_edge(
                "assessment",
                self._route_after_assessment,
                {
                    "practice_problem": "practice_problem",
                    "remediation": "remediation",
                    "concept_introduction": "concept_introduction",
                    "end": "__end__"
                }
            )
            
            builder.add_conditional_edge(
                "remediation",
                self._route_after_remediation,
                {
                    "practice_problem": "practice_problem",
                    "concept_introduction": "concept_introduction",
                    "end": "__end__"
                }
            )
            
            # Build and return graph
            return await builder.build()
            
        except Exception as e:
            logger.error(f"Failed to create tutoring workflow: {e}")
            raise
    
    def _route_after_practice(self, state: TutoringState) -> str:
        """Route after practice problem based on state."""
        if state["metadata"].get("awaiting_student_response"):
            return "assessment"
        elif state["tutoring_mode"] == TutoringMode.CONCEPT_EXPLANATION:
            return "concept_explanation"
        else:
            return "end"
    
    def _route_after_assessment(self, state: TutoringState) -> str:
        """Route after assessment based on performance."""
        tutoring_mode = state.get("tutoring_mode", TutoringMode.PRACTICE_PROBLEM)
        
        if tutoring_mode == TutoringMode.REMEDIATION:
            return "remediation"
        elif tutoring_mode == TutoringMode.CONCEPT_EXPLANATION:
            return "concept_introduction"
        elif tutoring_mode == TutoringMode.PRACTICE_PROBLEM:
            return "practice_problem"
        else:
            return "end"
    
    def _route_after_remediation(self, state: TutoringState) -> str:
        """Route after remediation."""
        return "practice_problem"  # Always go to practice after remediation
    
    async def start_tutoring_session(
        self, 
        session_id: str, 
        user_id: str,
        topic: str,
        difficulty_level: str = "beginner",
        learning_style: str = "analytical"
    ) -> Dict[str, Any]:
        """Start a new tutoring session."""
        try:
            # Create initial state
            initial_state = TutoringState(
                messages=[
                    SystemMessage(content="You are an AI tutor specializing in theory of computation.")
                ],
                session_id=session_id,
                user_id=user_id,
                current_step="concept_introduction",
                context={},
                metadata={
                    "session_start": datetime.now().isoformat(),
                    "topic": topic
                },
                error_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tutoring_mode=TutoringMode.INTRODUCTION,
                topic=topic,
                difficulty_level=DifficultyLevel(difficulty_level),
                learning_style=LearningStyle(learning_style),
                student_progress={},
                current_concept=None,
                practice_problems=[],
                assessment_results={},
                misconceptions=[],
                learning_objectives=[],
                adaptive_hints=[],
                performance_metrics={}
            )
            
            # Create workflow graph
            graph = await self.create_workflow_graph()
            
            # Execute workflow
            result = await workflow_executor.execute_workflow(
                graph, initial_state, self.config
            )
            
            return {
                "session_id": session_id,
                "status": "started",
                "result": result.dict(),
                "topic": topic,
                "difficulty_level": difficulty_level,
                "learning_style": learning_style
            }
            
        except Exception as e:
            logger.error(f"Failed to start tutoring session: {e}")
            raise
    
    async def continue_session(
        self,
        session_id: str,
        user_input: str,
        checkpoint_id: str = None
    ) -> Dict[str, Any]:
        """Continue an existing tutoring session."""
        try:
            # Create workflow graph
            graph = await self.create_workflow_graph()
            
            if checkpoint_id:
                # Resume from specific checkpoint
                result = await workflow_executor.resume_workflow(
                    graph, session_id, checkpoint_id, self.config
                )
            else:
                # Continue from latest checkpoint
                result = await workflow_executor.resume_workflow(
                    graph, session_id, None, self.config
                )
            
            return {
                "session_id": session_id,
                "status": "continued",
                "result": result.dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to continue tutoring session: {e}")
            raise


# Global tutoring workflow instance
tutoring_workflow = TutoringWorkflow()