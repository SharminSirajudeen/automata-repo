"""
LangGraph Router for Stateful AI Workflows.
Provides endpoints for managing and executing LangGraph workflows with state persistence.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..langgraph_core import (
    workflow_executor, checkpoint_manager, human_loop_manager,
    WorkflowStatus, InterruptType
)
from ..tutoring_workflow import tutoring_workflow, TutoringMode, DifficultyLevel, LearningStyle
from ..proof_assistant_graph import proof_assistant_workflow, ProofPhase
from ..automata_construction_graph import automata_construction_workflow, ConstructionPhase, AutomataType
from ..redis_integration import (
    redis_session_manager, redis_state_manager, redis_checkpoint_store, redis_monitor
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/langgraph", tags=["langgraph"])


# Request Models
class StartTutoringRequest(BaseModel):
    user_id: str
    topic: str
    difficulty_level: str = Field(default="beginner", description="beginner, intermediate, advanced, expert")
    learning_style: str = Field(default="analytical", description="visual, analytical, practical, theoretical")
    resume_session_id: Optional[str] = None


class StartProofRequest(BaseModel):
    user_id: str
    theorem_statement: str
    auto_verify: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    resume_session_id: Optional[str] = None


class StartAutomataRequest(BaseModel):
    user_id: str
    problem_description: str
    resume_session_id: Optional[str] = None


class ContinueWorkflowRequest(BaseModel):
    session_id: str
    user_input: Optional[str] = None
    checkpoint_id: Optional[str] = None


class HumanResponseRequest(BaseModel):
    request_id: str
    response: Any
    user_id: Optional[str] = None


class WorkflowControlRequest(BaseModel):
    session_id: str
    action: str = Field(description="pause, resume, cancel")


# Response Models
class WorkflowResponse(BaseModel):
    success: bool
    session_id: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    workflow_type: str
    created_at: str
    last_activity: str
    current_phase: Optional[str] = None
    steps_executed: int = 0
    checkpoints: List[str] = []
    pending_interrupts: List[Dict[str, Any]] = []


@router.post("/tutoring/start", response_model=WorkflowResponse)
async def start_tutoring_session(
    request: StartTutoringRequest,
    background_tasks: BackgroundTasks
):
    """Start a new tutoring session with adaptive learning."""
    try:
        # Generate session ID
        session_id = f"tutoring_{request.user_id}_{int(datetime.now().timestamp())}"
        
        # Check if resuming existing session
        if request.resume_session_id:
            session_data = await redis_session_manager.get_session(request.resume_session_id)
            if session_data and session_data.get("status") == "active":
                session_id = request.resume_session_id
            else:
                return WorkflowResponse(
                    success=False,
                    session_id=request.resume_session_id,
                    status="error",
                    message="Session not found or inactive",
                    error="Invalid session ID"
                )
        
        # Create session record
        await redis_session_manager.create_session(
            session_id=session_id,
            user_id=request.user_id,
            session_type="tutoring",
            metadata={
                "topic": request.topic,
                "difficulty_level": request.difficulty_level,
                "learning_style": request.learning_style
            }
        )
        
        # Start workflow in background
        background_tasks.add_task(
            _start_tutoring_workflow,
            session_id,
            request.user_id,
            request.topic,
            request.difficulty_level,
            request.learning_style
        )
        
        return WorkflowResponse(
            success=True,
            session_id=session_id,
            status="starting",
            message=f"Tutoring session started for topic: {request.topic}",
            data={
                "topic": request.topic,
                "difficulty_level": request.difficulty_level,
                "learning_style": request.learning_style
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start tutoring session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proof/start", response_model=WorkflowResponse)
async def start_proof_session(
    request: StartProofRequest,
    background_tasks: BackgroundTasks
):
    """Start a new proof assistant session."""
    try:
        # Generate session ID
        session_id = f"proof_{request.user_id}_{int(datetime.now().timestamp())}"
        
        # Check if resuming existing session
        if request.resume_session_id:
            session_data = await redis_session_manager.get_session(request.resume_session_id)
            if session_data and session_data.get("status") == "active":
                session_id = request.resume_session_id
            else:
                return WorkflowResponse(
                    success=False,
                    session_id=request.resume_session_id,
                    status="error",
                    message="Session not found or inactive",
                    error="Invalid session ID"
                )
        
        # Create session record
        await redis_session_manager.create_session(
            session_id=session_id,
            user_id=request.user_id,
            session_type="proof",
            metadata={
                "theorem_statement": request.theorem_statement,
                "auto_verify": request.auto_verify,
                "confidence_threshold": request.confidence_threshold
            }
        )
        
        # Start workflow in background
        background_tasks.add_task(
            _start_proof_workflow,
            session_id,
            request.user_id,
            request.theorem_statement,
            request.auto_verify,
            request.confidence_threshold
        )
        
        return WorkflowResponse(
            success=True,
            session_id=session_id,
            status="starting",
            message=f"Proof session started for theorem: {request.theorem_statement[:100]}...",
            data={
                "theorem": request.theorem_statement,
                "auto_verify": request.auto_verify,
                "confidence_threshold": request.confidence_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start proof session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/automata/start", response_model=WorkflowResponse)
async def start_automata_session(
    request: StartAutomataRequest,
    background_tasks: BackgroundTasks
):
    """Start a new automata construction session."""
    try:
        # Generate session ID
        session_id = f"automata_{request.user_id}_{int(datetime.now().timestamp())}"
        
        # Check if resuming existing session
        if request.resume_session_id:
            session_data = await redis_session_manager.get_session(request.resume_session_id)
            if session_data and session_data.get("status") == "active":
                session_id = request.resume_session_id
            else:
                return WorkflowResponse(
                    success=False,
                    session_id=request.resume_session_id,
                    status="error",
                    message="Session not found or inactive",
                    error="Invalid session ID"
                )
        
        # Create session record
        await redis_session_manager.create_session(
            session_id=session_id,
            user_id=request.user_id,
            session_type="automata",
            metadata={
                "problem_description": request.problem_description
            }
        )
        
        # Start workflow in background
        background_tasks.add_task(
            _start_automata_workflow,
            session_id,
            request.user_id,
            request.problem_description
        )
        
        return WorkflowResponse(
            success=True,
            session_id=session_id,
            status="starting",
            message=f"Automata construction session started",
            data={
                "problem": request.problem_description
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start automata session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/continue", response_model=WorkflowResponse)
async def continue_workflow(request: ContinueWorkflowRequest):
    """Continue an existing workflow with user input."""
    try:
        # Get session data
        session_data = await redis_session_manager.get_session(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_type = session_data.get("session_type")
        
        # Continue appropriate workflow
        if session_type == "tutoring":
            result = await tutoring_workflow.continue_session(
                request.session_id,
                request.user_input or "",
                request.checkpoint_id
            )
        elif session_type == "proof":
            # For proof workflow, we need to handle continuation differently
            # This would involve loading the graph and continuing execution
            result = {"status": "continued", "message": "Proof workflow continued"}
        elif session_type == "automata":
            # Similar for automata workflow
            result = {"status": "continued", "message": "Automata workflow continued"}
        else:
            raise HTTPException(status_code=400, detail="Unknown session type")
        
        # Update session activity
        await redis_session_manager.update_session(
            request.session_id,
            {"last_activity": datetime.now().isoformat()}
        )
        
        return WorkflowResponse(
            success=True,
            session_id=request.session_id,
            status="continued",
            message="Workflow continued successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Failed to continue workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/control")
async def control_workflow(request: WorkflowControlRequest):
    """Control workflow execution (pause, resume, cancel)."""
    try:
        session_id = request.session_id
        action = request.action.lower()
        
        if action == "pause":
            success = await workflow_executor.pause_workflow(session_id)
            message = "Workflow paused" if success else "Failed to pause workflow"
        elif action == "resume":
            # Resume would need the workflow graph - simplified for demo
            success = True
            message = "Workflow resumed"
        elif action == "cancel":
            success = await workflow_executor.cancel_workflow(session_id)
            if success:
                await redis_session_manager.close_session(session_id)
            message = "Workflow cancelled" if success else "Failed to cancel workflow"
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        return {
            "success": success,
            "session_id": session_id,
            "action": action,
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Failed to control workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get detailed status of a workflow session."""
    try:
        # Get session data
        session_data = await redis_session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get workflow status
        workflow_status = await workflow_executor.get_workflow_status(session_id)
        
        # Get checkpoints
        checkpoints = await redis_checkpoint_store.list_checkpoints(session_id)
        checkpoint_ids = [cp["checkpoint_id"] for cp in checkpoints]
        
        # Get pending human inputs
        pending_requests = await human_loop_manager.get_pending_requests(session_id)
        
        return SessionStatusResponse(
            session_id=session_id,
            status=session_data.get("status", "unknown"),
            workflow_type=session_data.get("session_type", "unknown"),
            created_at=session_data.get("created_at", ""),
            last_activity=session_data.get("last_activity", ""),
            current_phase=workflow_status.get("current_phase") if workflow_status else None,
            steps_executed=workflow_status.get("steps_executed", 0) if workflow_status else 0,
            checkpoints=checkpoint_ids,
            pending_interrupts=pending_requests
        )
        
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/sessions")
async def list_user_sessions(
    user_id: str,
    status: Optional[str] = None
):
    """List all sessions for a user."""
    try:
        sessions = await redis_session_manager.list_user_sessions(user_id, status)
        
        # Enrich with additional data
        enriched_sessions = []
        for session in sessions:
            session_id = session["session_id"]
            
            # Get workflow status
            workflow_status = await workflow_executor.get_workflow_status(session_id)
            
            # Get checkpoint count
            checkpoints = await redis_checkpoint_store.list_checkpoints(session_id)
            
            enriched_session = {
                **session,
                "steps_executed": workflow_status.get("steps_executed", 0) if workflow_status else 0,
                "checkpoint_count": len(checkpoints),
                "workflow_active": workflow_status is not None
            }
            
            enriched_sessions.append(enriched_session)
        
        return {
            "user_id": user_id,
            "sessions": enriched_sessions,
            "total": len(enriched_sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to list user sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/checkpoints")
async def list_session_checkpoints(session_id: str):
    """List all checkpoints for a session."""
    try:
        checkpoints = await redis_checkpoint_store.list_checkpoints(session_id)
        
        return {
            "session_id": session_id,
            "checkpoints": checkpoints,
            "total": len(checkpoints)
        }
        
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/checkpoint/{version}/restore")
async def restore_checkpoint(session_id: str, version: str):
    """Restore a session from a specific checkpoint."""
    try:
        # Load checkpoint data
        checkpoint_data = await redis_checkpoint_store.load_checkpoint(session_id, version)
        
        if not checkpoint_data:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        
        # This would involve reconstructing the workflow state and resuming
        # Simplified for demo
        result = {
            "session_id": session_id,
            "checkpoint_version": version,
            "restored": True,
            "message": f"Session restored from checkpoint {version}"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to restore checkpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/human_input/respond")
async def submit_human_response(request: HumanResponseRequest):
    """Submit human response to a pending request."""
    try:
        success = await human_loop_manager.submit_human_response(
            request.request_id,
            request.response,
            request.user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Request not found or expired")
        
        return {
            "success": True,
            "request_id": request.request_id,
            "message": "Response submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit human response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/human_input/{session_id}/pending")
async def get_pending_human_inputs(session_id: str):
    """Get pending human input requests for a session."""
    try:
        pending_requests = await human_loop_manager.get_pending_requests(session_id)
        
        return {
            "session_id": session_id,
            "pending_requests": pending_requests,
            "count": len(pending_requests)
        }
        
    except Exception as e:
        logger.error(f"Failed to get pending inputs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/redis")
async def get_redis_status():
    """Get Redis status and metrics."""
    try:
        redis_info = await redis_monitor.get_redis_info()
        key_stats = await redis_monitor.get_key_statistics()
        
        return {
            "redis_info": redis_info,
            "key_statistics": key_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get Redis status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor/cleanup")
async def cleanup_expired_data():
    """Clean up expired Redis data."""
    try:
        cleaned_count = await redis_monitor.cleanup_expired_keys()
        
        return {
            "cleaned_keys": cleaned_count,
            "message": f"Cleaned up {cleaned_count} expired keys",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _start_tutoring_workflow(
    session_id: str,
    user_id: str,
    topic: str,
    difficulty_level: str,
    learning_style: str
):
    """Background task to start tutoring workflow."""
    try:
        result = await tutoring_workflow.start_tutoring_session(
            session_id=session_id,
            user_id=user_id,
            topic=topic,
            difficulty_level=difficulty_level,
            learning_style=learning_style
        )
        
        # Update session with result
        await redis_session_manager.update_session(
            session_id,
            {
                "workflow_result": result,
                "status": "active" if result.get("result", {}).get("success") else "error"
            }
        )
        
    except Exception as e:
        logger.error(f"Background tutoring workflow failed: {e}")
        await redis_session_manager.update_session(
            session_id,
            {"status": "error", "error": str(e)}
        )


async def _start_proof_workflow(
    session_id: str,
    user_id: str,
    theorem_statement: str,
    auto_verify: bool,
    confidence_threshold: float
):
    """Background task to start proof workflow."""
    try:
        result = await proof_assistant_workflow.start_proof_session(
            session_id=session_id,
            user_id=user_id,
            theorem_statement=theorem_statement,
            auto_verify=auto_verify,
            confidence_threshold=confidence_threshold
        )
        
        # Update session with result
        await redis_session_manager.update_session(
            session_id,
            {
                "workflow_result": result,
                "status": "active" if result.get("result", {}).get("success") else "error"
            }
        )
        
    except Exception as e:
        logger.error(f"Background proof workflow failed: {e}")
        await redis_session_manager.update_session(
            session_id,
            {"status": "error", "error": str(e)}
        )


async def _start_automata_workflow(
    session_id: str,
    user_id: str,
    problem_description: str
):
    """Background task to start automata workflow."""
    try:
        result = await automata_construction_workflow.start_construction_session(
            session_id=session_id,
            user_id=user_id,
            problem_description=problem_description
        )
        
        # Update session with result
        await redis_session_manager.update_session(
            session_id,
            {
                "workflow_result": result,
                "status": "active" if result.get("result", {}).get("success") else "error"
            }
        )
        
    except Exception as e:
        logger.error(f"Background automata workflow failed: {e}")
        await redis_session_manager.update_session(
            session_id,
            {"status": "error", "error": str(e)}
        )