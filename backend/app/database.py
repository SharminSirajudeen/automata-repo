"""
Database configuration and models for the automata-repo application.
"""
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from .config import settings
from typing import Generator

# Create database engine
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database Models
class User(Base):
    """User model for authentication and progress tracking."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Progress tracking
    progress_data = Column(JSON, default=dict)
    skill_level = Column(String(50), default="beginner")  # beginner, intermediate, advanced
    
    # Relationships
    solutions = relationship("Solution", back_populates="user")
    learning_paths = relationship("LearningPath", back_populates="user")


class Problem(Base):
    """Problem model for storing automata problems."""
    __tablename__ = "problems"
    
    id = Column(String(100), primary_key=True)
    type = Column(String(50), nullable=False, index=True)  # dfa, nfa, pda, cfg, tm, regex, pumping_lemma
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    difficulty = Column(String(50), default="intermediate")  # beginner, intermediate, advanced, expert
    category = Column(String(100))  # e.g., "regular_languages", "context_free", "decidability"
    
    # Problem specification
    language_description = Column(Text, nullable=False)
    alphabet = Column(JSON, nullable=False)  # List of symbols
    test_strings = Column(JSON, nullable=False)  # List of test cases
    hints = Column(JSON, default=list)  # Progressive hints
    
    # Reference solution
    reference_solution = Column(JSON)  # Automaton structure
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    prerequisites = Column(JSON, default=list)  # List of problem IDs
    concepts = Column(JSON, default=list)  # List of concepts covered
    
    # Relationships
    solutions = relationship("Solution", back_populates="problem")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_problem_type_difficulty', 'type', 'difficulty'),
    )


class Solution(Base):
    """Solution model for storing user attempts."""
    __tablename__ = "solutions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    problem_id = Column(String(100), ForeignKey("problems.id"), nullable=False)
    
    # Solution data
    automaton_data = Column(JSON, nullable=False)  # User's automaton
    validation_result = Column(JSON)  # Validation details
    score = Column(Float, default=0.0)
    is_correct = Column(Boolean, default=False)
    time_spent = Column(Integer)  # Seconds spent on problem
    
    # AI assistance tracking
    hints_used = Column(Integer, default=0)
    ai_interactions = Column(JSON, default=list)  # Track AI help requests
    
    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="solutions")
    problem = relationship("Problem", back_populates="solutions")
    
    # Indexes
    __table_args__ = (
        Index('idx_solution_user_problem', 'user_id', 'problem_id'),
        Index('idx_solution_submitted', 'submitted_at'),
    )


class LearningPath(Base):
    """Learning path model for adaptive learning."""
    __tablename__ = "learning_paths"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Current progress
    current_concept = Column(String(100), nullable=False)
    current_problem_id = Column(String(100))
    completed_problems = Column(JSON, default=list)  # List of problem IDs
    
    # Mastery tracking
    concept_mastery = Column(JSON, default=dict)  # {concept: mastery_level}
    skill_progression = Column(JSON, default=list)  # History of skill changes
    
    # Personalization
    learning_style = Column(String(50))  # visual, practice, theory
    pace_preference = Column(String(50), default="normal")  # slow, normal, fast
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="learning_paths")


class AIInteraction(Base):
    """AI interaction model for tracking AI usage and improving responses."""
    __tablename__ = "ai_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    problem_id = Column(String(100))
    
    # Interaction data
    interaction_type = Column(String(50))  # hint, explanation, generation, validation
    user_input = Column(Text)
    ai_response = Column(Text)
    model_used = Column(String(50))
    
    # Quality metrics
    response_time = Column(Float)  # Seconds
    tokens_used = Column(Integer)
    user_feedback = Column(String(50))  # helpful, not_helpful, unclear
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_ai_interaction_user', 'user_id'),
        Index('idx_ai_interaction_type', 'interaction_type'),
    )


# Create all tables
def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


# Database utilities
def get_or_create_user(db: Session, email: str) -> User:
    """Get existing user or create a new one."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, hashed_password="temp")  # In production, proper auth
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def save_solution(db: Session, user_id: str, problem_id: str, solution_data: dict) -> Solution:
    """Save a user's solution attempt."""
    solution = Solution(
        user_id=user_id,
        problem_id=problem_id,
        automaton_data=solution_data.get("automaton"),
        validation_result=solution_data.get("validation_result"),
        score=solution_data.get("score", 0.0),
        is_correct=solution_data.get("is_correct", False),
        hints_used=solution_data.get("hints_used", 0)
    )
    db.add(solution)
    db.commit()
    db.refresh(solution)
    return solution