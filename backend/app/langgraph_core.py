"""
LangGraph Core Infrastructure for Stateful AI Conversations.
Provides base classes, state management, and graph orchestration for complex AI workflows.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, TypedDict, Annotated, Literal
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, Graph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import ToolNode
import redis.asyncio as redis

from .config import settings
from .orchestrator import ModelOrchestrator, ExecutionMode

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """Base conversation state for all LangGraph workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    user_id: Optional[str]
    current_step: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    error_count: int
    created_at: datetime
    updated_at: datetime


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class InterruptType(str, Enum):
    """Types of workflow interrupts."""
    HUMAN_INPUT = "human_input"
    CONFIRMATION = "confirmation"
    ERROR_RECOVERY = "error_recovery"
    TIMEOUT = "timeout"
    CUSTOM = "custom"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    max_steps: int = 50
    timeout_seconds: int = 300
    enable_checkpointing: bool = True
    enable_human_in_loop: bool = True
    retry_attempts: int = 3
    error_recovery_strategy: str = "retry"
    interrupt_on_error: bool = True
    save_intermediate_results: bool = True


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    result: Any
    status: WorkflowStatus
    steps_executed: int
    execution_time: float
    error: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)
    interrupts: List[Dict[str, Any]] = field(default_factory=list)


class BaseWorkflowNode(ABC):
    """Abstract base class for workflow nodes."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.orchestrator = ModelOrchestrator()
    
    @abstractmethod
    async def execute(self, state: ConversationState) -> ConversationState:
        """Execute the node logic."""
        pass
    
    async def on_error(self, state: ConversationState, error: Exception) -> ConversationState:
        """Handle node execution errors."""
        logger.error(f"Node {self.name} error: {error}")
        state["error_count"] += 1
        state["metadata"]["last_error"] = str(error)
        state["metadata"]["last_error_time"] = datetime.now().isoformat()
        return state
    
    async def should_interrupt(self, state: ConversationState) -> Optional[InterruptType]:
        """Check if workflow should be interrupted."""
        if state["error_count"] >= 3:
            return InterruptType.ERROR_RECOVERY
        return None


class RedisCheckpointManager:
    """Redis-based checkpoint manager for LangGraph workflows."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis_client = None
        self._saver = None
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client instance."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url)
        return self._redis_client
    
    async def get_checkpoint_saver(self) -> RedisSaver:
        """Get Redis checkpoint saver."""
        if self._saver is None:
            redis_client = await self.get_redis_client()
            self._saver = RedisSaver(redis_client)
        return self._saver
    
    async def save_checkpoint(self, session_id: str, state: ConversationState) -> str:
        """Save workflow checkpoint."""
        try:
            checkpoint_id = f"checkpoint_{session_id}_{int(time.time())}"
            redis_client = await self.get_redis_client()
            
            checkpoint_data = {
                "state": json.dumps(state, default=str),
                "timestamp": datetime.now().isoformat(),
                "checkpoint_id": checkpoint_id
            }
            
            await redis_client.hset(
                f"checkpoints:{session_id}",
                checkpoint_id,
                json.dumps(checkpoint_data)
            )
            
            # Keep only last 10 checkpoints per session
            await redis_client.expire(f"checkpoints:{session_id}", 3600)
            
            logger.info(f"Checkpoint saved: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    async def load_checkpoint(self, session_id: str, checkpoint_id: str = None) -> Optional[ConversationState]:
        """Load workflow checkpoint."""
        try:
            redis_client = await self.get_redis_client()
            
            if checkpoint_id:
                # Load specific checkpoint
                checkpoint_data = await redis_client.hget(
                    f"checkpoints:{session_id}",
                    checkpoint_id
                )
            else:
                # Load latest checkpoint
                checkpoints = await redis_client.hgetall(f"checkpoints:{session_id}")
                if not checkpoints:
                    return None
                
                latest_key = max(checkpoints.keys(), key=lambda k: k.split('_')[-1])
                checkpoint_data = checkpoints[latest_key]
            
            if checkpoint_data:
                data = json.loads(checkpoint_data)
                state = json.loads(data["state"])
                logger.info(f"Checkpoint loaded for session: {session_id}")
                return state
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    async def list_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a session."""
        try:
            redis_client = await self.get_redis_client()
            checkpoints = await redis_client.hgetall(f"checkpoints:{session_id}")
            
            checkpoint_list = []
            for checkpoint_id, data in checkpoints.items():
                checkpoint_info = json.loads(data)
                checkpoint_list.append({
                    "checkpoint_id": checkpoint_id,
                    "timestamp": checkpoint_info["timestamp"],
                    "session_id": session_id
                })
            
            # Sort by timestamp
            checkpoint_list.sort(key=lambda x: x["timestamp"], reverse=True)
            return checkpoint_list
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def cleanup_old_checkpoints(self, hours: int = 24):
        """Clean up checkpoints older than specified hours."""
        try:
            redis_client = await self.get_redis_client()
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Get all checkpoint keys
            pattern = "checkpoints:*"
            keys = await redis_client.keys(pattern)
            
            cleaned_count = 0
            for key in keys:
                checkpoints = await redis_client.hgetall(key)
                for checkpoint_id, data in checkpoints.items():
                    checkpoint_info = json.loads(data)
                    checkpoint_time = datetime.fromisoformat(checkpoint_info["timestamp"])
                    
                    if checkpoint_time < cutoff_time:
                        await redis_client.hdel(key, checkpoint_id)
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old checkpoints")
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")


class WorkflowGraphBuilder:
    """Builder for creating LangGraph workflows with stateful conversation management."""
    
    def __init__(self, workflow_name: str, config: WorkflowConfig = None):
        self.workflow_name = workflow_name
        self.config = config or WorkflowConfig()
        self.nodes: Dict[str, BaseWorkflowNode] = {}
        self.edges: List[tuple] = []
        self.conditional_edges: List[Dict[str, Any]] = []
        self.checkpoint_manager = RedisCheckpointManager()
        
    def add_node(self, node: BaseWorkflowNode) -> 'WorkflowGraphBuilder':
        """Add a node to the workflow."""
        self.nodes[node.name] = node
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> 'WorkflowGraphBuilder':
        """Add an edge between nodes."""
        self.edges.append((from_node, to_node))
        return self
    
    def add_conditional_edge(
        self,
        from_node: str,
        condition_func: callable,
        mapping: Dict[str, str]
    ) -> 'WorkflowGraphBuilder':
        """Add a conditional edge."""
        self.conditional_edges.append({
            "from_node": from_node,
            "condition": condition_func,
            "mapping": mapping
        })
        return self
    
    async def build(self) -> Graph:
        """Build the LangGraph workflow."""
        try:
            # Create state graph
            graph = StateGraph(ConversationState)
            
            # Add nodes
            for node_name, node in self.nodes.items():
                graph.add_node(node_name, self._create_node_wrapper(node))
            
            # Add edges
            for from_node, to_node in self.edges:
                graph.add_edge(from_node, to_node)
            
            # Add conditional edges
            for edge_config in self.conditional_edges:
                graph.add_conditional_edges(
                    edge_config["from_node"],
                    edge_config["condition"],
                    edge_config["mapping"]
                )
            
            # Set up checkpointing if enabled
            if self.config.enable_checkpointing:
                checkpoint_saver = await self.checkpoint_manager.get_checkpoint_saver()
                compiled_graph = graph.compile(checkpointer=checkpoint_saver)
            else:
                compiled_graph = graph.compile()
            
            logger.info(f"Workflow graph '{self.workflow_name}' built successfully")
            return compiled_graph
            
        except Exception as e:
            logger.error(f"Failed to build workflow graph: {e}")
            raise
    
    def _create_node_wrapper(self, node: BaseWorkflowNode):
        """Create a wrapper function for node execution with error handling."""
        async def node_wrapper(state: ConversationState) -> ConversationState:
            try:
                # Update state metadata
                state["current_step"] = node.name
                state["updated_at"] = datetime.now()
                
                # Check for interrupts
                interrupt_type = await node.should_interrupt(state)
                if interrupt_type:
                    state["metadata"]["interrupt"] = {
                        "type": interrupt_type.value,
                        "node": node.name,
                        "timestamp": datetime.now().isoformat()
                    }
                    return state
                
                # Execute node
                result = await node.execute(state)
                
                # Save checkpoint if enabled
                if self.config.enable_checkpointing:
                    checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                        state["session_id"], result
                    )
                    if "checkpoints" not in result["metadata"]:
                        result["metadata"]["checkpoints"] = []
                    result["metadata"]["checkpoints"].append(checkpoint_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Node execution error in {node.name}: {e}")
                return await node.on_error(state, e)
        
        return node_wrapper


class HumanInLoopManager:
    """Manager for human-in-the-loop interactions in workflows."""
    
    def __init__(self, redis_manager: RedisCheckpointManager):
        self.redis_manager = redis_manager
    
    async def request_human_input(
        self,
        session_id: str,
        prompt: str,
        context: Dict[str, Any] = None,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """Request input from human user."""
        try:
            request_id = f"human_input_{session_id}_{int(time.time())}"
            redis_client = await self.redis_manager.get_redis_client()
            
            request_data = {
                "request_id": request_id,
                "session_id": session_id,
                "prompt": prompt,
                "context": context or {},
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat()
            }
            
            # Store request
            await redis_client.setex(
                f"human_input:{request_id}",
                timeout_seconds,
                json.dumps(request_data)
            )
            
            # Add to session queue
            await redis_client.lpush(f"human_input_queue:{session_id}", request_id)
            
            logger.info(f"Human input requested: {request_id}")
            return {"request_id": request_id, "status": "pending"}
            
        except Exception as e:
            logger.error(f"Failed to request human input: {e}")
            raise
    
    async def submit_human_response(
        self,
        request_id: str,
        response: Any,
        user_id: str = None
    ) -> bool:
        """Submit human response to a pending request."""
        try:
            redis_client = await self.redis_manager.get_redis_client()
            
            # Get original request
            request_data = await redis_client.get(f"human_input:{request_id}")
            if not request_data:
                logger.warning(f"Human input request not found: {request_id}")
                return False
            
            request_info = json.loads(request_data)
            
            # Update with response
            request_info.update({
                "response": response,
                "status": "completed",
                "responded_by": user_id,
                "responded_at": datetime.now().isoformat()
            })
            
            # Save updated request
            await redis_client.setex(
                f"human_input:{request_id}",
                3600,  # Keep for 1 hour after completion
                json.dumps(request_info)
            )
            
            # Remove from pending queue
            await redis_client.lrem(
                f"human_input_queue:{request_info['session_id']}",
                1,
                request_id
            )
            
            logger.info(f"Human response submitted: {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit human response: {e}")
            return False
    
    async def get_pending_requests(self, session_id: str) -> List[Dict[str, Any]]:
        """Get pending human input requests for a session."""
        try:
            redis_client = await self.redis_manager.get_redis_client()
            
            # Get request IDs from queue
            request_ids = await redis_client.lrange(f"human_input_queue:{session_id}", 0, -1)
            
            requests = []
            for request_id in request_ids:
                request_data = await redis_client.get(f"human_input:{request_id}")
                if request_data:
                    request_info = json.loads(request_data)
                    
                    # Check if expired
                    expires_at = datetime.fromisoformat(request_info["expires_at"])
                    if expires_at < datetime.now():
                        # Remove expired request
                        await redis_client.delete(f"human_input:{request_id}")
                        await redis_client.lrem(f"human_input_queue:{session_id}", 1, request_id)
                        continue
                    
                    requests.append(request_info)
            
            return requests
            
        except Exception as e:
            logger.error(f"Failed to get pending requests: {e}")
            return []


class WorkflowExecutor:
    """Executor for running LangGraph workflows with enhanced monitoring and control."""
    
    def __init__(self):
        self.checkpoint_manager = RedisCheckpointManager()
        self.human_loop_manager = HumanInLoopManager(self.checkpoint_manager)
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def execute_workflow(
        self,
        graph: Graph,
        initial_state: ConversationState,
        config: WorkflowConfig = None
    ) -> WorkflowResult:
        """Execute a workflow with monitoring and checkpointing."""
        start_time = time.time()
        config = config or WorkflowConfig()
        session_id = initial_state["session_id"]
        
        try:
            # Register active workflow
            self.active_workflows[session_id] = {
                "status": WorkflowStatus.IN_PROGRESS,
                "start_time": start_time,
                "config": config,
                "steps_executed": 0
            }
            
            # Execute workflow
            result_state = None
            steps_executed = 0
            
            # Use streaming execution for better control
            async for event in graph.astream(
                initial_state,
                config={"configurable": {"thread_id": session_id}},
                stream_mode="values"
            ):
                steps_executed += 1
                result_state = event
                
                # Update active workflow tracking
                self.active_workflows[session_id]["steps_executed"] = steps_executed
                
                # Check for timeout
                if time.time() - start_time > config.timeout_seconds:
                    raise TimeoutError(f"Workflow timeout after {config.timeout_seconds} seconds")
                
                # Check for max steps
                if steps_executed >= config.max_steps:
                    logger.warning(f"Workflow reached max steps: {config.max_steps}")
                    break
                
                # Check for interrupts
                if result_state and result_state.get("metadata", {}).get("interrupt"):
                    interrupt_info = result_state["metadata"]["interrupt"]
                    logger.info(f"Workflow interrupted: {interrupt_info}")
                    
                    if interrupt_info["type"] == InterruptType.HUMAN_INPUT.value:
                        # Handle human input interrupt
                        await self._handle_human_input_interrupt(session_id, result_state)
            
            execution_time = time.time() - start_time
            
            # Mark workflow as completed
            self.active_workflows[session_id]["status"] = WorkflowStatus.COMPLETED
            
            return WorkflowResult(
                success=True,
                result=result_state,
                status=WorkflowStatus.COMPLETED,
                steps_executed=steps_executed,
                execution_time=execution_time,
                checkpoints=result_state.get("metadata", {}).get("checkpoints", [])
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Workflow execution failed: {e}")
            
            # Mark workflow as error
            if session_id in self.active_workflows:
                self.active_workflows[session_id]["status"] = WorkflowStatus.ERROR
            
            return WorkflowResult(
                success=False,
                result=None,
                status=WorkflowStatus.ERROR,
                steps_executed=steps_executed,
                execution_time=execution_time,
                error=str(e)
            )
        
        finally:
            # Clean up active workflow tracking
            if session_id in self.active_workflows:
                del self.active_workflows[session_id]
    
    async def resume_workflow(
        self,
        graph: Graph,
        session_id: str,
        checkpoint_id: str = None,
        config: WorkflowConfig = None
    ) -> WorkflowResult:
        """Resume a workflow from a checkpoint."""
        try:
            # Load checkpoint
            checkpoint_state = await self.checkpoint_manager.load_checkpoint(
                session_id, checkpoint_id
            )
            
            if not checkpoint_state:
                raise ValueError(f"No checkpoint found for session: {session_id}")
            
            logger.info(f"Resuming workflow from checkpoint: {checkpoint_id or 'latest'}")
            
            # Resume execution
            return await self.execute_workflow(graph, checkpoint_state, config)
            
        except Exception as e:
            logger.error(f"Failed to resume workflow: {e}")
            raise
    
    async def pause_workflow(self, session_id: str) -> bool:
        """Pause an active workflow."""
        try:
            if session_id in self.active_workflows:
                self.active_workflows[session_id]["status"] = WorkflowStatus.PAUSED
                logger.info(f"Workflow paused: {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to pause workflow: {e}")
            return False
    
    async def cancel_workflow(self, session_id: str) -> bool:
        """Cancel an active workflow."""
        try:
            if session_id in self.active_workflows:
                self.active_workflows[session_id]["status"] = WorkflowStatus.CANCELLED
                del self.active_workflows[session_id]
                logger.info(f"Workflow cancelled: {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")
            return False
    
    async def get_workflow_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active workflow."""
        return self.active_workflows.get(session_id)
    
    async def _handle_human_input_interrupt(
        self,
        session_id: str,
        state: ConversationState
    ):
        """Handle human input interrupt during workflow execution."""
        interrupt_info = state["metadata"]["interrupt"]
        
        # Request human input
        await self.human_loop_manager.request_human_input(
            session_id=session_id,
            prompt=interrupt_info.get("prompt", "Input required to continue workflow"),
            context=interrupt_info.get("context", {}),
            timeout_seconds=300
        )
        
        # Workflow will pause here until human input is provided


# Global instances
checkpoint_manager = RedisCheckpointManager()
workflow_executor = WorkflowExecutor()
human_loop_manager = HumanInLoopManager(checkpoint_manager)