"""
Learning router for the Automata Learning Platform.
Handles adaptive learning, performance tracking, and personalized recommendations.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging

from ..database import get_db, User
from ..auth import get_current_active_user
from ..adaptive_learning import (
    adaptive_engine as adaptive_learning_engine,
    StudentAction as PerformanceData,
    LearningSession,
    DifficultyLevel as DifficultyAdjustment
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning", tags=["learning"])


class PerformanceUpdate(BaseModel):
    student_id: str
    problem_id: str
    problem_type: str
    score: float
    time_spent: int  # in seconds
    hints_used: int
    attempts: int
    difficulty_level: str


class SessionStart(BaseModel):
    student_id: str
    session_type: str
    learning_objectives: List[str]


class SessionEnd(BaseModel):
    session_id: str
    problems_solved: int
    total_score: float
    session_duration: int  # in seconds
    feedback: Optional[str] = None


@router.post("/update-performance")
async def update_performance(
    performance: PerformanceUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update student performance data for adaptive learning"""
    try:
        # Verify user can update this student's performance
        if str(current_user.id) != performance.student_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        performance_data = PerformanceData(
            student_id=performance.student_id,
            problem_id=performance.problem_id,
            problem_type=performance.problem_type,
            score=performance.score,
            time_spent=performance.time_spent,
            hints_used=performance.hints_used,
            attempts=performance.attempts,
            difficulty_level=performance.difficulty_level
        )
        
        # Update the adaptive learning model
        await adaptive_learning_engine.update_performance(performance_data)
        
        # Get updated difficulty recommendation
        new_difficulty = await adaptive_learning_engine.get_difficulty_recommendation(
            performance.student_id,
            performance.problem_type
        )
        
        logger.info(f"Performance updated for student {performance.student_id}")
        
        return {
            "status": "updated",
            "student_id": performance.student_id,
            "current_score": performance.score,
            "recommended_difficulty": new_difficulty,
            "performance_trend": "improving"  # This would be calculated
        }
        
    except Exception as e:
        logger.error(f"Performance update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{student_id}")
async def get_recommendations(
    student_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get personalized learning recommendations for a student"""
    try:
        # Verify access permissions
        if str(current_user.id) != student_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        recommendations = await adaptive_learning_engine.get_recommendations(student_id)
        
        return {
            "student_id": student_id,
            "recommendations": recommendations,
            "generated_at": "2025-08-05T16:27:32Z",
            "recommendation_count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Recommendations error for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{student_id}")
async def get_learning_analytics(
    student_id: str,
    time_period: str = "week",  # week, month, all
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get learning analytics and progress data for a student"""
    try:
        # Verify access permissions
        if str(current_user.id) != student_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analytics = await adaptive_learning_engine.get_analytics(
            student_id, 
            time_period
        )
        
        return {
            "student_id": student_id,
            "time_period": time_period,
            "analytics": {
                "total_problems_solved": analytics.get("problems_solved", 0),
                "average_score": analytics.get("average_score", 0.0),
                "time_spent_learning": analytics.get("total_time", 0),
                "concept_mastery": analytics.get("concept_mastery", {}),
                "difficulty_progression": analytics.get("difficulty_progression", []),
                "learning_velocity": analytics.get("learning_velocity", 0.0),
                "strength_areas": analytics.get("strengths", []),
                "improvement_areas": analytics.get("weaknesses", [])
            },
            "performance_trends": analytics.get("trends", {}),
            "generated_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Analytics error for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/start")
async def start_learning_session(
    session_data: SessionStart,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start a new learning session"""
    try:
        # Verify user can start session for this student
        if str(current_user.id) != session_data.student_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        session = LearningSession(
            student_id=session_data.student_id,
            session_type=session_data.session_type,
            learning_objectives=session_data.learning_objectives
        )
        
        session_id = await adaptive_learning_engine.start_session(session)
        
        # Get initial problem recommendations for the session
        initial_problems = await adaptive_learning_engine.get_session_problems(
            session_id,
            count=5
        )
        
        logger.info(f"Learning session started for student {session_data.student_id}")
        
        return {
            "session_id": session_id,
            "student_id": session_data.student_id,
            "session_type": session_data.session_type,
            "learning_objectives": session_data.learning_objectives,
            "initial_problems": initial_problems,
            "estimated_duration": 30,  # minutes
            "started_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Session start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/end")
async def end_learning_session(
    session_data: SessionEnd,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """End a learning session and process results"""
    try:
        session_result = await adaptive_learning_engine.end_session(
            session_data.session_id,
            problems_solved=session_data.problems_solved,
            total_score=session_data.total_score,
            duration=session_data.session_duration,
            feedback=session_data.feedback
        )
        
        # Calculate learning outcomes and achievements
        achievements = await adaptive_learning_engine.calculate_achievements(
            session_data.session_id
        )
        
        logger.info(f"Learning session {session_data.session_id} ended")
        
        return {
            "session_id": session_data.session_id,
            "session_summary": {
                "problems_solved": session_data.problems_solved,
                "total_score": session_data.total_score,
                "duration_minutes": session_data.session_duration // 60,
                "average_score": session_data.total_score / max(session_data.problems_solved, 1)
            },
            "learning_outcomes": session_result.get("outcomes", []),
            "achievements_earned": achievements,
            "next_recommendations": session_result.get("next_steps", []),
            "session_rating": session_result.get("rating", "Good"),
            "ended_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Session end error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/difficulty/{student_id}/{problem_type}")
async def get_difficulty_recommendation(
    student_id: str,
    problem_type: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get difficulty recommendation for a specific problem type"""
    try:
        # Verify access permissions
        if str(current_user.id) != student_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        difficulty_data = await adaptive_learning_engine.get_difficulty_recommendation(
            student_id,
            problem_type
        )
        
        return {
            "student_id": student_id,
            "problem_type": problem_type,
            "recommended_difficulty": difficulty_data.level,
            "confidence": difficulty_data.confidence,
            "reasoning": difficulty_data.reasoning,
            "adjustment_factors": {
                "recent_performance": difficulty_data.performance_factor,
                "time_factor": difficulty_data.time_factor,
                "consistency": difficulty_data.consistency_factor
            },
            "alternative_difficulties": difficulty_data.alternatives,
            "generated_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Difficulty recommendation error for {student_id}/{problem_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{student_id}")
async def get_learning_progress(
    student_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get overall learning progress for a student"""
    try:
        # Verify access permissions
        if str(current_user.id) != student_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        progress = await adaptive_learning_engine.get_learning_progress(student_id)
        
        return {
            "student_id": student_id,
            "overall_progress": {
                "completion_percentage": progress.get("completion", 0.0),
                "mastery_level": progress.get("mastery", "beginner"),
                "learning_streak": progress.get("streak", 0),
                "total_study_time": progress.get("study_time", 0)
            },
            "concept_progress": progress.get("concepts", {}),
            "skill_development": progress.get("skills", {}),
            "milestone_achievements": progress.get("milestones", []),
            "projected_completion": progress.get("completion_date", None),
            "last_updated": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Progress tracking error for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/{student_id}")
async def submit_learning_feedback(
    student_id: str,
    feedback: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit feedback about the learning experience"""
    try:
        # Verify access permissions
        if str(current_user.id) != student_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        feedback_id = await adaptive_learning_engine.submit_feedback(
            student_id,
            feedback
        )
        
        logger.info(f"Feedback submitted for student {student_id}")
        
        return {
            "feedback_id": feedback_id,
            "student_id": student_id,
            "status": "received",
            "will_improve_recommendations": True,
            "submitted_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))