"""
Automated grading system for the automata learning platform.
Supports assignment submission, automatic correctness checking, partial credit,
plagiarism detection, and LMS integration.
"""

import json
import hashlib
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from .database import Base, get_db, User, Problem, Solution
from .jflap_complete import JFLAPProcessor
from .verification import AutomataVerifier
import logging

logger = logging.getLogger(__name__)

class AssignmentType(str, Enum):
    """Types of assignments."""
    HOMEWORK = "homework"
    QUIZ = "quiz"
    EXAM = "exam"
    PROJECT = "project"
    LAB = "lab"

class GradingCriteria(str, Enum):
    """Grading criteria types."""
    CORRECTNESS = "correctness"
    EFFICIENCY = "efficiency"
    COMPLETENESS = "completeness"
    STYLE = "style"
    DOCUMENTATION = "documentation"

class SubmissionStatus(str, Enum):
    """Submission status."""
    SUBMITTED = "submitted"
    GRADING = "grading"
    GRADED = "graded"
    NEEDS_REVIEW = "needs_review"
    RETURNED = "returned"

class PlagiarismLevel(str, Enum):
    """Plagiarism detection levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

# Database Models
class Assignment(Base):
    """Assignment model."""
    __tablename__ = "assignments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(String(50), nullable=False)
    
    # Problem configuration
    problem_ids = Column(JSON, default=list)  # List of problem IDs
    total_points = Column(Float, default=100.0)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime)
    due_time = Column(DateTime)
    late_penalty_per_day = Column(Float, default=10.0)  # Percentage
    
    # Grading configuration
    grading_criteria = Column(JSON, default=list)
    auto_grade = Column(Boolean, default=True)
    allow_multiple_submissions = Column(Boolean, default=True)
    max_submissions = Column(Integer, default=5)
    
    # Course integration
    course_id = Column(String(255))
    instructor_id = Column(UUID(as_uuid=True))
    
    # Settings
    is_published = Column(Boolean, default=False)
    plagiarism_check_enabled = Column(Boolean, default=True)
    
    # Relationships
    submissions = relationship("AssignmentSubmission", back_populates="assignment")
    
    __table_args__ = (
        Index('idx_assignment_course', 'course_id'),
        Index('idx_assignment_due', 'due_time'),
    )

class AssignmentSubmission(Base):
    """Assignment submission model."""
    __tablename__ = "assignment_submissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assignment_id = Column(UUID(as_uuid=True), ForeignKey("assignments.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Submission data
    solutions = Column(JSON, default=dict)  # {problem_id: solution_data}
    submission_number = Column(Integer, default=1)
    
    # Timing
    submitted_at = Column(DateTime, default=datetime.utcnow)
    is_late = Column(Boolean, default=False)
    late_penalty = Column(Float, default=0.0)
    
    # Grading
    status = Column(String(50), default=SubmissionStatus.SUBMITTED.value)
    total_score = Column(Float, default=0.0)
    max_score = Column(Float, default=100.0)
    percentage_score = Column(Float, default=0.0)
    
    # Detailed grading
    problem_scores = Column(JSON, default=dict)  # {problem_id: score_details}
    grading_details = Column(JSON, default=dict)
    feedback = Column(Text)
    
    # Plagiarism detection
    plagiarism_level = Column(String(50), default=PlagiarismLevel.NONE.value)
    plagiarism_score = Column(Float, default=0.0)
    plagiarism_matches = Column(JSON, default=list)
    
    # Review
    needs_manual_review = Column(Boolean, default=False)
    reviewed_by = Column(UUID(as_uuid=True))
    reviewed_at = Column(DateTime)
    
    # Relationships
    assignment = relationship("Assignment", back_populates="submissions")
    user = relationship("User")

class GradingRubric(Base):
    """Grading rubric model."""
    __tablename__ = "grading_rubrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assignment_id = Column(UUID(as_uuid=True), ForeignKey("assignments.id"))
    problem_id = Column(String(100))
    
    # Rubric details
    criteria = Column(JSON, nullable=False)  # List of grading criteria
    total_points = Column(Float, nullable=False)
    
    # Automated scoring weights
    correctness_weight = Column(Float, default=0.7)
    efficiency_weight = Column(Float, default=0.2)
    style_weight = Column(Float, default=0.1)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class PlagiarismCase(Base):
    """Plagiarism detection case model."""
    __tablename__ = "plagiarism_cases"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assignment_id = Column(UUID(as_uuid=True), ForeignKey("assignments.id"))
    
    # Involved submissions
    submission1_id = Column(UUID(as_uuid=True), ForeignKey("assignment_submissions.id"))
    submission2_id = Column(UUID(as_uuid=True), ForeignKey("assignment_submissions.id"))
    
    # Similarity metrics
    similarity_score = Column(Float, nullable=False)
    similarity_type = Column(String(100))  # "structure", "solution", "combined"
    
    # Analysis details
    matching_elements = Column(JSON, default=list)
    analysis_details = Column(JSON, default=dict)
    
    # Review status
    is_reviewed = Column(Boolean, default=False)
    reviewer_decision = Column(String(100))  # "plagiarism", "collaboration", "coincidence"
    reviewer_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class AssignmentRequest(BaseModel):
    """Request model for creating assignments."""
    title: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    type: AssignmentType
    problem_ids: List[str]
    total_points: float = Field(default=100.0, ge=0)
    start_time: Optional[datetime] = None
    due_time: Optional[datetime] = None
    late_penalty_per_day: float = Field(default=10.0, ge=0, le=100)
    auto_grade: bool = True
    allow_multiple_submissions: bool = True
    max_submissions: int = Field(default=5, ge=1, le=20)
    course_id: Optional[str] = None
    plagiarism_check_enabled: bool = True
    
    @validator('due_time')
    def validate_due_time(cls, v, values):
        if v and 'start_time' in values and values['start_time'] and v <= values['start_time']:
            raise ValueError('Due time must be after start time')
        return v

class SubmissionRequest(BaseModel):
    """Request model for assignment submissions."""
    solutions: Dict[str, Dict[str, Any]]  # {problem_id: solution_data}

class GradingResult(BaseModel):
    """Grading result model."""
    problem_id: str
    max_score: float
    earned_score: float
    percentage: float
    correctness_score: float
    efficiency_score: float
    style_score: float
    feedback: str
    test_results: List[Dict[str, Any]]

@dataclass
class TestCase:
    """Test case for automated grading."""
    input_string: str
    expected_result: bool
    points: float
    description: str = ""

@dataclass
class GradingCriterion:
    """Individual grading criterion."""
    name: str
    description: str
    max_points: float
    weight: float

class AutomatedGrader:
    """Main automated grading system."""
    
    def __init__(self):
        self.jflap_processor = JFLAPProcessor()
        self.verifier = AutomataVerifier()
        self.similarity_threshold = 0.85  # For plagiarism detection
        
    async def create_assignment(self, request: AssignmentRequest, 
                              instructor_id: str, db: Session) -> Assignment:
        """Create a new assignment."""
        try:
            # Validate problem IDs exist
            problems = db.query(Problem).filter(
                Problem.id.in_(request.problem_ids)
            ).all()
            
            if len(problems) != len(request.problem_ids):
                raise ValueError("Some problem IDs are invalid")
            
            # Create assignment
            assignment = Assignment(
                title=request.title,
                description=request.description,
                type=request.type.value,
                problem_ids=request.problem_ids,
                total_points=request.total_points,
                start_time=request.start_time,
                due_time=request.due_time,
                late_penalty_per_day=request.late_penalty_per_day,
                auto_grade=request.auto_grade,
                allow_multiple_submissions=request.allow_multiple_submissions,
                max_submissions=request.max_submissions,
                course_id=request.course_id,
                instructor_id=instructor_id,
                plagiarism_check_enabled=request.plagiarism_check_enabled
            )
            
            db.add(assignment)
            db.commit()
            db.refresh(assignment)
            
            # Create default grading rubrics
            await self._create_default_rubrics(assignment, problems, db)
            
            logger.info(f"Created assignment: {assignment.title}")
            return assignment
            
        except Exception as e:
            logger.error(f"Error creating assignment: {e}")
            db.rollback()
            raise
    
    async def submit_assignment(self, assignment_id: str, user_id: str,
                              submission_data: SubmissionRequest,
                              db: Session) -> AssignmentSubmission:
        """Submit assignment solutions."""
        try:
            # Get assignment
            assignment = db.query(Assignment).filter(
                Assignment.id == assignment_id,
                Assignment.is_published == True
            ).first()
            
            if not assignment:
                raise ValueError("Assignment not found or not published")
            
            # Check if submissions are allowed
            if not assignment.allow_multiple_submissions:
                existing = db.query(AssignmentSubmission).filter(
                    AssignmentSubmission.assignment_id == assignment_id,
                    AssignmentSubmission.user_id == user_id
                ).first()
                
                if existing:
                    raise ValueError("Multiple submissions not allowed")
            
            # Check submission limit
            submission_count = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.assignment_id == assignment_id,
                AssignmentSubmission.user_id == user_id
            ).count()
            
            if submission_count >= assignment.max_submissions:
                raise ValueError("Maximum submissions exceeded")
            
            # Check if late
            is_late = False
            late_penalty = 0.0
            
            if assignment.due_time and datetime.utcnow() > assignment.due_time:
                is_late = True
                days_late = (datetime.utcnow() - assignment.due_time).days + 1
                late_penalty = min(days_late * assignment.late_penalty_per_day, 100.0)
            
            # Create submission
            submission = AssignmentSubmission(
                assignment_id=assignment.id,
                user_id=user_id,
                solutions=submission_data.solutions,
                submission_number=submission_count + 1,
                is_late=is_late,
                late_penalty=late_penalty,
                max_score=assignment.total_points
            )
            
            db.add(submission)
            db.commit()
            db.refresh(submission)
            
            # Auto-grade if enabled
            if assignment.auto_grade:
                await self._grade_submission(submission, db)
            
            # Check for plagiarism if enabled
            if assignment.plagiarism_check_enabled:
                await self._check_plagiarism(submission, db)
            
            logger.info(f"Submission created for assignment {assignment_id} by user {user_id}")
            return submission
            
        except Exception as e:
            logger.error(f"Error submitting assignment: {e}")
            db.rollback()
            raise
    
    async def _grade_submission(self, submission: AssignmentSubmission, db: Session):
        """Grade a submission automatically."""
        try:
            assignment = db.query(Assignment).filter(
                Assignment.id == submission.assignment_id
            ).first()
            
            if not assignment:
                return
            
            submission.status = SubmissionStatus.GRADING.value
            db.commit()
            
            total_score = 0.0
            problem_scores = {}
            grading_details = {}
            
            # Grade each problem
            for problem_id in assignment.problem_ids:
                if problem_id not in submission.solutions:
                    # Missing solution
                    problem_scores[problem_id] = {
                        "earned_score": 0.0,
                        "max_score": assignment.total_points / len(assignment.problem_ids),
                        "feedback": "No solution submitted"
                    }
                    continue
                
                # Get problem
                problem = db.query(Problem).filter(Problem.id == problem_id).first()
                if not problem:
                    continue
                
                # Get grading rubric
                rubric = db.query(GradingRubric).filter(
                    GradingRubric.assignment_id == assignment.id,
                    GradingRubric.problem_id == problem_id
                ).first()
                
                # Grade the solution
                result = await self._grade_problem_solution(
                    problem, submission.solutions[problem_id], rubric
                )
                
                problem_scores[problem_id] = {
                    "earned_score": result.earned_score,
                    "max_score": result.max_score,
                    "percentage": result.percentage,
                    "correctness_score": result.correctness_score,
                    "efficiency_score": result.efficiency_score,
                    "style_score": result.style_score,
                    "feedback": result.feedback,
                    "test_results": result.test_results
                }
                
                total_score += result.earned_score
            
            # Apply late penalty
            if submission.is_late:
                penalty_amount = total_score * (submission.late_penalty / 100.0)
                total_score = max(0, total_score - penalty_amount)
                grading_details["late_penalty_applied"] = penalty_amount
            
            # Update submission
            submission.total_score = total_score
            submission.percentage_score = (total_score / assignment.total_points) * 100
            submission.problem_scores = problem_scores
            submission.grading_details = grading_details
            submission.status = SubmissionStatus.GRADED.value
            
            # Check if manual review needed
            if submission.percentage_score < 50 or submission.plagiarism_level != PlagiarismLevel.NONE.value:
                submission.needs_manual_review = True
                submission.status = SubmissionStatus.NEEDS_REVIEW.value
            
            db.commit()
            
            logger.info(f"Graded submission {submission.id}: {submission.percentage_score:.1f}%")
            
        except Exception as e:
            logger.error(f"Error grading submission: {e}")
            submission.status = SubmissionStatus.NEEDS_REVIEW.value
            submission.needs_manual_review = True
            db.commit()
    
    async def _grade_problem_solution(self, problem: Problem, solution_data: Dict[str, Any],
                                    rubric: Optional[GradingRubric]) -> GradingResult:
        """Grade a single problem solution."""
        try:
            max_score = rubric.total_points if rubric else 10.0
            
            # Extract solution
            automaton_data = solution_data.get("automaton")
            if not automaton_data:
                return GradingResult(
                    problem_id=problem.id,
                    max_score=max_score,
                    earned_score=0.0,
                    percentage=0.0,
                    correctness_score=0.0,
                    efficiency_score=0.0,
                    style_score=0.0,
                    feedback="No automaton provided",
                    test_results=[]
                )
            
            # Test correctness
            test_results = []
            correct_tests = 0
            total_tests = len(problem.test_strings)
            
            for test_case in problem.test_strings:
                test_string = test_case.get("string", "")
                expected = test_case.get("should_accept", False)
                
                try:
                    # Use JFLAP processor to test
                    result = await self._test_string_acceptance(
                        automaton_data, test_string, problem.type
                    )
                    
                    is_correct = result == expected
                    if is_correct:
                        correct_tests += 1
                    
                    test_results.append({
                        "input": test_string,
                        "expected": expected,
                        "actual": result,
                        "correct": is_correct,
                        "points": 1.0 if is_correct else 0.0
                    })
                    
                except Exception as e:
                    test_results.append({
                        "input": test_string,
                        "expected": expected,
                        "actual": None,
                        "correct": False,
                        "error": str(e),
                        "points": 0.0
                    })
            
            # Calculate scores
            correctness_percentage = correct_tests / total_tests if total_tests > 0 else 0.0
            
            # Efficiency score (based on state count vs reference)
            efficiency_score = self._calculate_efficiency_score(
                automaton_data, problem.reference_solution
            )
            
            # Style score (based on naming, organization)
            style_score = self._calculate_style_score(automaton_data)
            
            # Apply rubric weights if available
            if rubric:
                correctness_weight = rubric.correctness_weight
                efficiency_weight = rubric.efficiency_weight
                style_weight = rubric.style_weight
            else:
                correctness_weight = 0.8
                efficiency_weight = 0.15
                style_weight = 0.05
            
            # Calculate final score
            weighted_score = (
                correctness_percentage * correctness_weight +
                efficiency_score * efficiency_weight +
                style_score * style_weight
            )
            
            earned_score = weighted_score * max_score
            
            # Generate feedback
            feedback = self._generate_feedback(
                correctness_percentage, efficiency_score, style_score, test_results
            )
            
            return GradingResult(
                problem_id=problem.id,
                max_score=max_score,
                earned_score=earned_score,
                percentage=weighted_score * 100,
                correctness_score=correctness_percentage,
                efficiency_score=efficiency_score,
                style_score=style_score,
                feedback=feedback,
                test_results=test_results
            )
            
        except Exception as e:
            logger.error(f"Error grading problem solution: {e}")
            return GradingResult(
                problem_id=problem.id,
                max_score=max_score,
                earned_score=0.0,
                percentage=0.0,
                correctness_score=0.0,
                efficiency_score=0.0,
                style_score=0.0,
                feedback=f"Grading error: {str(e)}",
                test_results=[]
            )
    
    async def _test_string_acceptance(self, automaton_data: Dict[str, Any],
                                    test_string: str, problem_type: str) -> bool:
        """Test if automaton accepts a string."""
        try:
            if problem_type in ["dfa", "nfa"]:
                return await self._test_finite_automaton(automaton_data, test_string)
            elif problem_type == "pda":
                return await self._test_pushdown_automaton(automaton_data, test_string)
            elif problem_type == "tm":
                return await self._test_turing_machine(automaton_data, test_string)
            else:
                return False
        except Exception:
            return False
    
    async def _test_finite_automaton(self, automaton_data: Dict[str, Any], 
                                   test_string: str) -> bool:
        """Test string on finite automaton."""
        states = set(automaton_data.get("states", []))
        alphabet = set(automaton_data.get("alphabet", []))
        transitions = automaton_data.get("transitions", [])
        start_state = automaton_data.get("start_state")
        accept_states = set(automaton_data.get("accept_states", []))
        
        if not start_state or start_state not in states:
            return False
        
        # Build transition function
        delta = {}
        for trans in transitions:
            from_state = trans.get("from")
            symbol = trans.get("symbol")
            to_state = trans.get("to")
            
            if from_state and to_state and symbol:
                if from_state not in delta:
                    delta[from_state] = {}
                if symbol not in delta[from_state]:
                    delta[from_state][symbol] = []
                delta[from_state][symbol].append(to_state)
        
        # Simulate automaton
        current_states = {start_state}
        
        for symbol in test_string:
            next_states = set()
            for state in current_states:
                if state in delta and symbol in delta[state]:
                    next_states.update(delta[state][symbol])
            current_states = next_states
            
            if not current_states:
                return False
        
        return bool(current_states & accept_states)
    
    async def _test_pushdown_automaton(self, automaton_data: Dict[str, Any],
                                     test_string: str) -> bool:
        """Test string on pushdown automaton (simplified)."""
        # Simplified PDA simulation
        # In a full implementation, this would handle stack operations
        return True  # Placeholder
    
    async def _test_turing_machine(self, automaton_data: Dict[str, Any],
                                 test_string: str) -> bool:
        """Test string on Turing machine (simplified)."""
        # Simplified TM simulation with step limit
        max_steps = 1000
        # Implementation would simulate TM execution
        return True  # Placeholder
    
    def _calculate_efficiency_score(self, solution: Dict[str, Any],
                                  reference: Optional[Dict[str, Any]]) -> float:
        """Calculate efficiency score based on state count."""
        try:
            solution_states = len(solution.get("states", []))
            
            if not reference:
                # Without reference, score based on reasonable state count
                if solution_states <= 5:
                    return 1.0
                elif solution_states <= 10:
                    return 0.8
                elif solution_states <= 20:
                    return 0.6
                else:
                    return 0.4
            
            reference_states = len(reference.get("states", []))
            if reference_states == 0:
                return 1.0
            
            ratio = reference_states / solution_states
            return min(1.0, ratio)
            
        except Exception:
            return 0.5
    
    def _calculate_style_score(self, automaton_data: Dict[str, Any]) -> float:
        """Calculate style score based on naming and organization."""
        try:
            score = 1.0
            states = automaton_data.get("states", [])
            
            # Check state naming
            has_meaningful_names = False
            for state in states:
                state_id = state.get("id", "")
                if len(state_id) > 2 and not state_id.startswith("q"):
                    has_meaningful_names = True
                    break
            
            if not has_meaningful_names:
                score -= 0.2
            
            # Check for proper start state
            start_state = automaton_data.get("start_state")
            if not start_state:
                score -= 0.3
            
            # Check for accept states
            accept_states = automaton_data.get("accept_states", [])
            if not accept_states:
                score -= 0.2
            
            return max(0.0, score)
            
        except Exception:
            return 0.5
    
    def _generate_feedback(self, correctness: float, efficiency: float,
                         style: float, test_results: List[Dict]) -> str:
        """Generate grading feedback."""
        feedback_parts = []
        
        # Correctness feedback
        if correctness >= 0.9:
            feedback_parts.append("Excellent correctness! All or nearly all test cases passed.")
        elif correctness >= 0.7:
            feedback_parts.append("Good correctness, but some test cases failed.")
        elif correctness >= 0.5:
            feedback_parts.append("Partial correctness. Review failed test cases.")
        else:
            feedback_parts.append("Poor correctness. Most test cases failed.")
        
        # Efficiency feedback
        if efficiency >= 0.9:
            feedback_parts.append("Very efficient solution!")
        elif efficiency >= 0.7:
            feedback_parts.append("Reasonably efficient solution.")
        else:
            feedback_parts.append("Solution could be more efficient with fewer states.")
        
        # Style feedback
        if style < 0.8:
            feedback_parts.append("Consider using more descriptive state names and proper organization.")
        
        # Specific test case feedback
        failed_tests = [t for t in test_results if not t.get("correct", False)]
        if failed_tests:
            feedback_parts.append(f"Failed test cases: {len(failed_tests)}")
            for test in failed_tests[:3]:  # Show first 3 failures
                input_str = test.get("input", "")
                expected = test.get("expected", False)
                actual = test.get("actual", "unknown")
                feedback_parts.append(f"  - Input '{input_str}': expected {expected}, got {actual}")
        
        return " ".join(feedback_parts)
    
    async def _check_plagiarism(self, submission: AssignmentSubmission, db: Session):
        """Check submission for plagiarism."""
        try:
            assignment = db.query(Assignment).filter(
                Assignment.id == submission.assignment_id
            ).first()
            
            if not assignment or not assignment.plagiarism_check_enabled:
                return
            
            # Get other submissions for comparison
            other_submissions = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.assignment_id == assignment.id,
                AssignmentSubmission.id != submission.id
            ).all()
            
            plagiarism_cases = []
            max_similarity = 0.0
            
            for other_submission in other_submissions:
                similarity = await self._calculate_similarity(
                    submission.solutions, other_submission.solutions
                )
                
                if similarity > self.similarity_threshold:
                    # Record plagiarism case
                    case = PlagiarismCase(
                        assignment_id=assignment.id,
                        submission1_id=submission.id,
                        submission2_id=other_submission.id,
                        similarity_score=similarity,
                        similarity_type="combined",
                        matching_elements=[]  # Would contain specific matches
                    )
                    
                    db.add(case)
                    plagiarism_cases.append({
                        "submission_id": str(other_submission.id),
                        "similarity": similarity,
                        "user_id": str(other_submission.user_id)
                    })
                
                max_similarity = max(max_similarity, similarity)
            
            # Determine plagiarism level
            if max_similarity >= 0.95:
                level = PlagiarismLevel.SEVERE
            elif max_similarity >= 0.9:
                level = PlagiarismLevel.HIGH
            elif max_similarity >= 0.8:
                level = PlagiarismLevel.MODERATE
            elif max_similarity >= 0.6:
                level = PlagiarismLevel.LOW
            else:
                level = PlagiarismLevel.NONE
            
            submission.plagiarism_level = level.value
            submission.plagiarism_score = max_similarity
            submission.plagiarism_matches = plagiarism_cases
            
            if level != PlagiarismLevel.NONE:
                submission.needs_manual_review = True
                logger.warning(f"Plagiarism detected in submission {submission.id}: {level.value}")
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error checking plagiarism: {e}")
    
    async def _calculate_similarity(self, solutions1: Dict, solutions2: Dict) -> float:
        """Calculate similarity between two solution sets."""
        try:
            if not solutions1 or not solutions2:
                return 0.0
            
            similarities = []
            
            # Compare solutions for common problems
            common_problems = set(solutions1.keys()) & set(solutions2.keys())
            
            for problem_id in common_problems:
                sol1 = solutions1[problem_id]
                sol2 = solutions2[problem_id]
                
                # Calculate structural similarity
                struct_sim = self._calculate_structural_similarity(sol1, sol2)
                similarities.append(struct_sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_structural_similarity(self, sol1: Dict, sol2: Dict) -> float:
        """Calculate structural similarity between two automaton solutions."""
        try:
            # Extract automaton structures
            auto1 = sol1.get("automaton", {})
            auto2 = sol2.get("automaton", {})
            
            if not auto1 or not auto2:
                return 0.0
            
            # Compare state counts
            states1 = set(s.get("id", "") for s in auto1.get("states", []))
            states2 = set(s.get("id", "") for s in auto2.get("states", []))
            
            state_similarity = len(states1 & states2) / len(states1 | states2) if states1 | states2 else 0.0
            
            # Compare transition structures
            trans1 = set()
            for t in auto1.get("transitions", []):
                trans1.add((t.get("from"), t.get("to"), t.get("symbol")))
            
            trans2 = set()
            for t in auto2.get("transitions", []):
                trans2.add((t.get("from"), t.get("to"), t.get("symbol")))
            
            trans_similarity = len(trans1 & trans2) / len(trans1 | trans2) if trans1 | trans2 else 0.0
            
            # Weighted average
            return 0.4 * state_similarity + 0.6 * trans_similarity
            
        except Exception:
            return 0.0
    
    async def _create_default_rubrics(self, assignment: Assignment,
                                    problems: List[Problem], db: Session):
        """Create default grading rubrics for assignment problems."""
        try:
            points_per_problem = assignment.total_points / len(problems)
            
            for problem in problems:
                rubric = GradingRubric(
                    assignment_id=assignment.id,
                    problem_id=problem.id,
                    criteria=[
                        {
                            "name": "Correctness",
                            "description": "Solution correctly accepts/rejects test strings",
                            "weight": 0.7
                        },
                        {
                            "name": "Efficiency",
                            "description": "Solution uses minimal states",
                            "weight": 0.2
                        },
                        {
                            "name": "Style",
                            "description": "Proper naming and organization",
                            "weight": 0.1
                        }
                    ],
                    total_points=points_per_problem,
                    correctness_weight=0.7,
                    efficiency_weight=0.2,
                    style_weight=0.1
                )
                
                db.add(rubric)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error creating default rubrics: {e}")
    
    async def export_grades(self, assignment_id: str, db: Session, 
                          format: str = "csv") -> str:
        """Export grades for an assignment."""
        try:
            assignment = db.query(Assignment).filter(
                Assignment.id == assignment_id
            ).first()
            
            if not assignment:
                raise ValueError("Assignment not found")
            
            submissions = db.query(AssignmentSubmission).filter(
                AssignmentSubmission.assignment_id == assignment_id
            ).all()
            
            if format == "csv":
                return await self._export_grades_csv(assignment, submissions)
            elif format == "json":
                return await self._export_grades_json(assignment, submissions)
            else:
                raise ValueError("Unsupported export format")
                
        except Exception as e:
            logger.error(f"Error exporting grades: {e}")
            raise
    
    async def _export_grades_csv(self, assignment: Assignment,
                               submissions: List[AssignmentSubmission]) -> str:
        """Export grades as CSV format."""
        lines = [
            "Student ID,Name,Submission Time,Score,Percentage,Late Penalty,Status,Plagiarism Level"
        ]
        
        for submission in submissions:
            user = submission.user
            lines.append(
                f"{user.id},{user.full_name or user.email},"
                f"{submission.submitted_at.isoformat()},"
                f"{submission.total_score:.2f},{submission.percentage_score:.1f}%,"
                f"{submission.late_penalty:.1f}%,{submission.status},"
                f"{submission.plagiarism_level}"
            )
        
        return "\n".join(lines)
    
    async def _export_grades_json(self, assignment: Assignment,
                                submissions: List[AssignmentSubmission]) -> str:
        """Export grades as JSON format."""
        data = {
            "assignment": {
                "id": str(assignment.id),
                "title": assignment.title,
                "total_points": assignment.total_points,
                "due_time": assignment.due_time.isoformat() if assignment.due_time else None
            },
            "submissions": []
        }
        
        for submission in submissions:
            user = submission.user
            data["submissions"].append({
                "user_id": str(user.id),
                "user_name": user.full_name or user.email,
                "submitted_at": submission.submitted_at.isoformat(),
                "score": submission.total_score,
                "percentage": submission.percentage_score,
                "is_late": submission.is_late,
                "late_penalty": submission.late_penalty,
                "status": submission.status,
                "plagiarism_level": submission.plagiarism_level,
                "plagiarism_score": submission.plagiarism_score,
                "needs_review": submission.needs_manual_review,
                "problem_scores": submission.problem_scores
            })
        
        return json.dumps(data, indent=2)

# Global grader instance
automated_grader = AutomatedGrader()

# Utility functions
async def create_assignment(request: AssignmentRequest, instructor_id: str,
                          db: Session) -> Assignment:
    """Create a new assignment."""
    return await automated_grader.create_assignment(request, instructor_id, db)

async def submit_assignment(assignment_id: str, user_id: str,
                          submission_data: SubmissionRequest,
                          db: Session) -> AssignmentSubmission:
    """Submit assignment solutions."""
    return await automated_grader.submit_assignment(
        assignment_id, user_id, submission_data, db
    )

async def get_assignment_grades(assignment_id: str, db: Session) -> List[Dict]:
    """Get grades for an assignment."""
    submissions = db.query(AssignmentSubmission).filter(
        AssignmentSubmission.assignment_id == assignment_id
    ).all()
    
    return [
        {
            "user_id": str(sub.user_id),
            "score": sub.total_score,
            "percentage": sub.percentage_score,
            "status": sub.status,
            "submitted_at": sub.submitted_at.isoformat()
        }
        for sub in submissions
    ]

async def export_assignment_grades(assignment_id: str, format: str,
                                 db: Session) -> str:
    """Export assignment grades."""
    return await automated_grader.export_grades(assignment_id, db, format)