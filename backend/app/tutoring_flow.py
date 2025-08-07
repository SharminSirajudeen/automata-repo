"""
Tutoring Workflow using LangGraph.
(This file is named tutoring_flow.py to work around a file writing issue.)
"""

import logging
from typing import TypedDict, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from langgraph.graph import StateGraph, END

from .langgraph_core import (
    BaseWorkflowNode,
    ConversationState,
    WorkflowGraphBuilder,
    workflow_executor
)
from .agents import AutomataExplainer
from .cache_integration import valkey_checkpoint_store

logger = logging.getLogger(__name__)

# --- Enums for Tutoring Configuration ---

class TutoringMode(str, Enum):
    CONCEPT_LEARNING = "concept_learning"
    PROBLEM_SOLVING = "problem_solving"

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class LearningStyle(str, Enum):
    VISUAL = "visual"
    ANALYTICAL = "analytical"
    PRACTICAL = "practical"

# --- Tutoring State ---

class TutoringState(ConversationState):
    """The state for the tutoring workflow."""
    topic: str
    difficulty: DifficultyLevel
    learning_style: LearningStyle
    mode: TutoringMode
    current_question: Optional[str] = None
    user_answer: Optional[str] = None

# --- Workflow Nodes ---

class IntroductionNode(BaseWorkflowNode):
    """Introduces the topic to the user."""
    async def execute(self, state: TutoringState) -> TutoringState:
        logger.info(f"Executing IntroductionNode for topic: {state['topic']}")
        explainer = AutomataExplainer()
        explanation = await explainer.explain_automaton(
            task=f"Explain {state['topic']}",
            automaton_data={"type": state['topic']}
        )
        state["messages"].append({"role": "assistant", "content": explanation.get("explanation")})
        return state

class QuestionNode(BaseWorkflowNode):
    """Asks the user a question."""
    async def execute(self, state: TutoringState) -> TutoringState:
        logger.info(f"Executing QuestionNode for topic: {state['topic']}")
        question = f"What is a key component of a {state['topic']}? (Type 'exit' to end)"
        state["current_question"] = question
        state["messages"].append({"role": "assistant", "content": question})
        return state

# --- Workflow Manager ---

class TutoringWorkflowManager:
    """Manages the tutoring workflow graph."""
    def __init__(self):
        self._graph = None

    async def get_graph(self) -> StateGraph:
        """Build and compile the workflow graph if it doesn't exist."""
        if self._graph is None:
            builder = WorkflowGraphBuilder("tutoring_workflow")
            builder.add_node(IntroductionNode(name="introduction"))
            builder.add_node(QuestionNode(name="ask_question"))
            builder.add_edge("introduction", "ask_question")
            builder.add_edge("ask_question", END) # Simple flow: introduce, ask, then end.

            graph = await builder.build()
            graph.set_entry_point("introduction")
            self._graph = graph
        return self._graph

    async def start_tutoring_session(
        self, session_id: str, user_id: str, topic: str,
        difficulty_level: str, learning_style: str
    ) -> Dict[str, Any]:
        """Starts a new tutoring session and runs the workflow."""
        logger.info(f"Starting tutoring session {session_id} for user {user_id}")
        graph = await self.get_graph()

        initial_state = TutoringState(
            session_id=session_id,
            user_id=user_id,
            topic=topic,
            difficulty=DifficultyLevel(difficulty_level),
            learning_style=LearningStyle(learning_style),
            mode=TutoringMode.CONCEPT_LEARNING,
            messages=[],
            context={},
            metadata={},
            error_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            current_step="initial"
        )

        final_state = None
        async for output in graph.astream(initial_state, {"configurable": {"thread_id": session_id}}):
            final_state = output

        return final_state if final_state else {}

    async def continue_session(self, session_id: str, user_input: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Continues a session. This is a placeholder for now as the simple graph
        runs to completion in one go. A more complex graph with wait states
        would require this to be fully implemented.
        """
        logger.warning("continue_session is not fully implemented for this simple workflow.")
        return {"status": "completed", "message": "Workflow finished."}

# --- Singleton Instance ---
tutoring_workflow = TutoringWorkflowManager()
