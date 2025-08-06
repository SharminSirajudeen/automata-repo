"""
Automated grading router for assignment management and grading.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..database import get_db
from ..automated_grading import (
    automated_grader, AssignmentRequest, SubmissionRequest,
    Assignment, AssignmentSubmission, create_assignment,
    submit_assignment, get_assignment_grades, export_assignment_grades,
    AssignmentType, GradingCriteria, SubmissionStatus
)
from ..api_platform import get_current_client, require_scope, APIScope
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/grading", tags=["Automated Grading"])

@router.post("/assignments", response_model=Dict[str, Any])
async def create_new_assignment(
    request: AssignmentRequest,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.WRITE_PROBLEMS))
):
    """Create a new assignment."""
    try:
        client, _ = client_info
        instructor_id = str(client.id)  # Use client as instructor
        
        assignment = await create_assignment(request, instructor_id, db)
        
        return {
            "id": str(assignment.id),
            "title": assignment.title,
            "type": assignment.type,
            "problem_ids": assignment.problem_ids,
            "total_points": assignment.total_points,
            "start_time": assignment.start_time,
            "due_time": assignment.due_time,
            "created_at": assignment.created_at,
            "is_published": assignment.is_published
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating assignment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create assignment"
        )

@router.get("/assignments")
async def list_assignments(
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.READ_PROBLEMS))
):
    """List all assignments."""
    try:
        assignments = db.query(Assignment).filter(
            Assignment.is_published == True
        ).all()
        
        return {
            "assignments": [
                {
                    "id": str(assignment.id),
                    "title": assignment.title,
                    "description": assignment.description,
                    "type": assignment.type,
                    "total_points": assignment.total_points,
                    "start_time": assignment.start_time,
                    "due_time": assignment.due_time,
                    "problem_count": len(assignment.problem_ids),
                    "allow_multiple_submissions": assignment.allow_multiple_submissions,
                    "max_submissions": assignment.max_submissions
                }
                for assignment in assignments
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing assignments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list assignments"
        )

@router.get("/assignments/{assignment_id}")
async def get_assignment_details(
    assignment_id: str,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.READ_PROBLEMS))
):
    """Get detailed assignment information."""
    try:
        assignment = db.query(Assignment).filter(
            Assignment.id == assignment_id,
            Assignment.is_published == True
        ).first()
        
        if not assignment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assignment not found"
            )
        
        return {
            "id": str(assignment.id),
            "title": assignment.title,
            "description": assignment.description,
            "type": assignment.type,
            "problem_ids": assignment.problem_ids,
            "total_points": assignment.total_points,
            "start_time": assignment.start_time,
            "due_time": assignment.due_time,
            "late_penalty_per_day": assignment.late_penalty_per_day,
            "grading_criteria": assignment.grading_criteria,
            "auto_grade": assignment.auto_grade,
            "allow_multiple_submissions": assignment.allow_multiple_submissions,
            "max_submissions": assignment.max_submissions,
            "plagiarism_check_enabled": assignment.plagiarism_check_enabled,
            "created_at": assignment.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assignment details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get assignment details"
        )

@router.post("/assignments/{assignment_id}/submissions")
async def submit_assignment_solution(
    assignment_id: str,
    submission_data: SubmissionRequest,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.WRITE_SOLUTIONS))
):
    """Submit solutions for an assignment."""
    try:
        client, _ = client_info
        user_id = str(client.id)  # Use client as user
        
        submission = await submit_assignment(
            assignment_id, user_id, submission_data, db
        )
        
        return {
            "id": str(submission.id),
            "assignment_id": str(submission.assignment_id),
            "user_id": str(submission.user_id),
            "submission_number": submission.submission_number,
            "submitted_at": submission.submitted_at,
            "status": submission.status,
            "is_late": submission.is_late,
            "late_penalty": submission.late_penalty,
            "total_score": submission.total_score,
            "percentage_score": submission.percentage_score,
            "plagiarism_level": submission.plagiarism_level
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error submitting assignment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit assignment"
        )

@router.get("/assignments/{assignment_id}/submissions")
async def list_assignment_submissions(
    assignment_id: str,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.READ_SOLUTIONS))
):
    """List submissions for an assignment."""
    try:
        client, _ = client_info
        
        # Check if client has admin scope to see all submissions
        is_admin = APIScope.ADMIN.value in client.scopes
        
        if is_admin:
            # Show all submissions
            submissions = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.assignment_id == assignment_id
            ).all()
        else:
            # Show only client's submissions
            submissions = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.assignment_id == assignment_id,
                AssignmentSubmission.user_id == client.id
            ).all()
        
        return {
            "submissions": [
                {
                    "id": str(submission.id),
                    "user_id": str(submission.user_id) if is_admin else "self",
                    "submission_number": submission.submission_number,
                    "submitted_at": submission.submitted_at,
                    "status": submission.status,
                    "total_score": submission.total_score,
                    "percentage_score": submission.percentage_score,
                    "is_late": submission.is_late,
                    "late_penalty": submission.late_penalty,
                    "plagiarism_level": submission.plagiarism_level,
                    "needs_manual_review": submission.needs_manual_review
                }
                for submission in submissions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing submissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list submissions"
        )

@router.get("/assignments/{assignment_id}/submissions/{submission_id}")
async def get_submission_details(
    assignment_id: str,
    submission_id: str,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.READ_SOLUTIONS))
):
    """Get detailed submission information."""
    try:
        client, _ = client_info
        
        submission = db.query(AssignmentSubmission).filter(
            AssignmentSubmission.id == submission_id,
            AssignmentSubmission.assignment_id == assignment_id
        ).first()
        
        if not submission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found"
            )
        
        # Check access permissions
        is_admin = APIScope.ADMIN.value in client.scopes
        if not is_admin and str(submission.user_id) != str(client.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this submission"
            )
        
        return {
            "id": str(submission.id),
            "assignment_id": str(submission.assignment_id),
            "user_id": str(submission.user_id) if is_admin else "self",
            "submission_number": submission.submission_number,
            "submitted_at": submission.submitted_at,
            "status": submission.status,
            "total_score": submission.total_score,
            "max_score": submission.max_score,
            "percentage_score": submission.percentage_score,
            "is_late": submission.is_late,
            "late_penalty": submission.late_penalty,
            "problem_scores": submission.problem_scores,
            "grading_details": submission.grading_details,
            "feedback": submission.feedback,
            "plagiarism_level": submission.plagiarism_level,
            "plagiarism_score": submission.plagiarism_score,
            "plagiarism_matches": submission.plagiarism_matches if is_admin else [],
            "needs_manual_review": submission.needs_manual_review,
            "reviewed_at": submission.reviewed_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting submission details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get submission details"
        )

@router.get("/assignments/{assignment_id}/grades")
async def get_assignment_grade_summary(
    assignment_id: str,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.READ_SOLUTIONS))
):
    """Get grade summary for an assignment."""
    try:
        client, _ = client_info
        is_admin = APIScope.ADMIN.value in client.scopes
        
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for grade summaries"
            )
        
        grades = await get_assignment_grades(assignment_id, db)
        
        # Calculate statistics
        if grades:
            scores = [g["percentage"] for g in grades]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            # Grade distribution
            grade_ranges = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
            for score in scores:
                if score >= 90:
                    grade_ranges["A"] += 1
                elif score >= 80:
                    grade_ranges["B"] += 1
                elif score >= 70:
                    grade_ranges["C"] += 1
                elif score >= 60:
                    grade_ranges["D"] += 1
                else:
                    grade_ranges["F"] += 1
        else:
            avg_score = max_score = min_score = 0
            grade_ranges = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        
        return {
            "assignment_id": assignment_id,
            "total_submissions": len(grades),
            "statistics": {
                "average_score": avg_score,
                "maximum_score": max_score,
                "minimum_score": min_score,
                "grade_distribution": grade_ranges
            },
            "grades": grades
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting grade summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get grade summary"
        )

@router.get("/assignments/{assignment_id}/export")
async def export_grades(
    assignment_id: str,
    format: str = "csv",
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.ADMIN))
):
    """Export assignment grades."""
    try:
        if format not in ["csv", "json"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supported formats: csv, json"
            )
        
        export_data = await export_assignment_grades(assignment_id, format, db)
        
        content_type = "text/csv" if format == "csv" else "application/json"
        filename = f"assignment_{assignment_id}_grades.{format}"
        
        return Response(
            content=export_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error exporting grades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export grades"
        )

@router.post("/assignments/{assignment_id}/publish")
async def publish_assignment(
    assignment_id: str,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.ADMIN))
):
    """Publish an assignment to make it available to students."""
    try:
        assignment = db.query(Assignment).filter(
            Assignment.id == assignment_id
        ).first()
        
        if not assignment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assignment not found"
            )
        
        assignment.is_published = True
        db.commit()
        
        return {
            "message": "Assignment published successfully",
            "assignment_id": str(assignment.id),
            "title": assignment.title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing assignment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish assignment"
        )

@router.post("/assignments/{assignment_id}/submissions/{submission_id}/review")
async def review_submission(
    assignment_id: str,
    submission_id: str,
    review_data: Dict[str, Any],
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.ADMIN))
):
    """Manually review a submission."""
    try:
        client, _ = client_info
        
        submission = db.query(AssignmentSubmission).filter(
            AssignmentSubmission.id == submission_id,
            AssignmentSubmission.assignment_id == assignment_id
        ).first()
        
        if not submission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found"
            )
        
        # Update submission with review
        if "score_override" in review_data:
            submission.total_score = review_data["score_override"]
            submission.percentage_score = (submission.total_score / submission.max_score) * 100
        
        if "feedback" in review_data:
            submission.feedback = review_data["feedback"]
        
        if "status" in review_data and review_data["status"] in SubmissionStatus:
            submission.status = review_data["status"]
        
        submission.reviewed_by = client.id
        submission.reviewed_at = datetime.utcnow()
        submission.needs_manual_review = False
        
        db.commit()
        
        return {
            "message": "Submission reviewed successfully",
            "submission_id": str(submission.id),
            "final_score": submission.total_score,
            "final_percentage": submission.percentage_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to review submission"
        )

@router.get("/stats/overview")
async def get_grading_overview(
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.READ_SOLUTIONS))
):
    """Get overview statistics for grading system."""
    try:
        client, _ = client_info
        is_admin = APIScope.ADMIN.value in client.scopes
        
        if is_admin:
            # Admin sees all statistics
            total_assignments = db.query(Assignment).count()
            total_submissions = db.query(AssignmentSubmission).count()
            pending_reviews = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.needs_manual_review == True
            ).count()
            
            return {
                "role": "admin",
                "statistics": {
                    "total_assignments": total_assignments,
                    "total_submissions": total_submissions,
                    "pending_reviews": pending_reviews
                }
            }
        else:
            # Regular client sees their own statistics
            user_submissions = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.user_id == client.id
            ).count()
            
            graded_submissions = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.user_id == client.id,
                AssignmentSubmission.status == SubmissionStatus.GRADED.value
            ).count()
            
            return {
                "role": "student",
                "statistics": {
                    "total_submissions": user_submissions,
                    "graded_submissions": graded_submissions,
                    "pending_submissions": user_submissions - graded_submissions
                }
            }
        
    except Exception as e:
        logger.error(f"Error getting grading overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get grading overview"
        )