"""
Adaptive Learning System for Automata Theory
Provides performance tracking, difficulty adjustment, and personalized problem recommendations.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from enum import Enum
import logging
import math
import statistics
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class DifficultyLevel(str, Enum):
    """Difficulty levels for problems"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"

class ProblemType(str, Enum):
    """Types of problems in the system"""
    DFA_CONSTRUCTION = "dfa_construction"
    NFA_CONSTRUCTION = "nfa_construction"
    REGEX_TO_AUTOMATON = "regex_to_automaton"
    AUTOMATON_TO_REGEX = "automaton_to_regex"
    PUMPING_LEMMA = "pumping_lemma"
    MINIMIZATION = "minimization"
    EQUIVALENCE = "equivalence"
    PROOF_WRITING = "proof_writing"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    REDUCTION = "reduction"

class LearningObjective(str, Enum):
    """Learning objectives"""
    UNDERSTAND_CONCEPTS = "understand_concepts"
    APPLY_ALGORITHMS = "apply_algorithms"
    PROVE_THEOREMS = "prove_theorems"
    SOLVE_PROBLEMS = "solve_problems"
    ANALYZE_COMPLEXITY = "analyze_complexity"

class StudentAction(BaseModel):
    """Single action taken by a student"""
    timestamp: datetime
    problem_id: str
    problem_type: ProblemType
    difficulty: DifficultyLevel
    action_type: str  # attempt, hint_request, solution_view, etc.
    success: bool
    time_spent: int  # seconds
    hints_used: int = 0
    attempts: int = 1
    score: float = Field(ge=0.0, le=100.0)

class StudentProfile(BaseModel):
    """Student's learning profile and performance data"""
    student_id: str
    name: Optional[str] = None
    current_level: DifficultyLevel = DifficultyLevel.BEGINNER
    learning_objectives: List[LearningObjective] = Field(default_factory=list)
    strengths: List[ProblemType] = Field(default_factory=list)
    weaknesses: List[ProblemType] = Field(default_factory=list)
    total_problems_attempted: int = 0
    total_problems_solved: int = 0
    average_score: float = 0.0
    learning_velocity: float = 1.0  # Rate of improvement
    attention_span: int = 1800  # seconds (30 minutes default)
    preferred_learning_style: str = "mixed"  # visual, textual, interactive, mixed

class LearningSession(BaseModel):
    """A learning session"""
    session_id: str
    student_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    actions: List[StudentAction] = Field(default_factory=list)
    session_score: float = 0.0
    topics_covered: List[str] = Field(default_factory=list)
    goals_achieved: List[str] = Field(default_factory=list)

class ProblemRecommendation(BaseModel):
    """Recommended problem for student"""
    problem_id: str
    problem_type: ProblemType
    difficulty: DifficultyLevel
    title: str
    description: str
    estimated_time: int  # minutes
    learning_objectives: List[LearningObjective]
    prerequisite_topics: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)  # How confident we are this is good
    reasoning: str  # Why this problem was recommended

class AdaptiveLearningEngine:
    """Core adaptive learning engine"""
    
    def __init__(self):
        self.student_profiles: Dict[str, StudentProfile] = {}
        self.learning_sessions: Dict[str, List[LearningSession]] = defaultdict(list)
        self.problem_difficulty_map = self._initialize_problem_difficulties()
        self.topic_prerequisites = self._initialize_prerequisites()
        
    def _initialize_problem_difficulties(self) -> Dict[ProblemType, Dict[DifficultyLevel, List[str]]]:
        """Initialize problem difficulties for each type"""
        return {
            ProblemType.DFA_CONSTRUCTION: {
                DifficultyLevel.BEGINNER: [
                    "Simple pattern recognition (a*, ab*)",
                    "Basic alternation (a|b)*",
                    "Fixed-length strings"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "Complex patterns with multiple conditions",
                    "Substring matching",
                    "Modular arithmetic conditions"
                ],
                DifficultyLevel.ADVANCED: [
                    "Multi-state complex conditions",
                    "Optimization problems",
                    "Error-tolerant recognition"
                ],
                DifficultyLevel.EXPERT: [
                    "Research-level problems",
                    "Novel applications",
                    "Theoretical edge cases"
                ]
            },
            ProblemType.PUMPING_LEMMA: {
                DifficultyLevel.BEGINNER: [
                    "Basic non-regular languages (a^n b^n)",
                    "Simple palindromes",
                    "Equal count problems"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "Complex balance conditions",
                    "Context-free pumping lemma",
                    "Multiple pumping conditions"
                ],
                DifficultyLevel.ADVANCED: [
                    "Subtle non-regularity proofs",
                    "Advanced CFL non-membership",
                    "Complex language constructions"
                ],
                DifficultyLevel.EXPERT: [
                    "Research-level separations",
                    "Novel non-regularity techniques",
                    "Advanced hierarchy theorems"
                ]
            },
            ProblemType.COMPLEXITY_ANALYSIS: {
                DifficultyLevel.BEGINNER: [
                    "Basic Big-O analysis",
                    "Simple loop counting",
                    "Elementary complexity classes"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "Recursive algorithm analysis",
                    "Amortized analysis",
                    "P vs NP basics"
                ],
                DifficultyLevel.ADVANCED: [
                    "Advanced complexity classes",
                    "Reduction constructions",
                    "Hierarchy theorems"
                ],
                DifficultyLevel.EXPERT: [
                    "Research-level complexity",
                    "Novel separation techniques",
                    "Advanced computational models"
                ]
            }
        }
    
    def _initialize_prerequisites(self) -> Dict[str, List[str]]:
        """Initialize topic prerequisites"""
        return {
            "dfa_construction": [],
            "nfa_construction": ["dfa_construction"],
            "regex_to_automaton": ["dfa_construction", "nfa_construction"],
            "pumping_lemma": ["dfa_construction", "regular_languages"],
            "minimization": ["dfa_construction", "equivalence_relations"],
            "complexity_analysis": ["algorithms_basics", "big_o_notation"],
            "np_completeness": ["complexity_analysis", "reductions"],
            "proof_writing": ["mathematical_logic", "proof_techniques"]
        }
    
    def update_student_performance(self, student_id: str, action: StudentAction) -> StudentProfile:
        """Update student profile based on new action"""
        if student_id not in self.student_profiles:
            self.student_profiles[student_id] = StudentProfile(student_id=student_id)
        
        profile = self.student_profiles[student_id]
        
        # Update basic statistics
        profile.total_problems_attempted += 1
        if action.success:
            profile.total_problems_solved += 1
        
        # Update average score
        old_avg = profile.average_score
        n = profile.total_problems_attempted
        profile.average_score = (old_avg * (n - 1) + action.score) / n
        
        # Update learning velocity (rate of improvement)
        profile.learning_velocity = self._calculate_learning_velocity(student_id, action)
        
        # Update strengths and weaknesses
        self._update_strengths_weaknesses(profile, action)
        
        # Adjust difficulty level
        profile.current_level = self._adjust_difficulty_level(profile, action)
        
        # Update attention span estimate
        profile.attention_span = self._estimate_attention_span(student_id, action)
        
        return profile
    
    def _calculate_learning_velocity(self, student_id: str, action: StudentAction) -> float:
        """Calculate student's learning velocity (improvement rate)"""
        sessions = self.learning_sessions[student_id]
        if len(sessions) < 2:
            return 1.0
        
        # Get recent performance trend
        recent_scores = []
        for session in sessions[-5:]:  # Last 5 sessions
            if session.actions:
                session_avg = statistics.mean([a.score for a in session.actions])
                recent_scores.append(session_avg)
        
        if len(recent_scores) < 2:
            return 1.0
        
        # Calculate improvement rate
        x = list(range(len(recent_scores)))
        y = recent_scores
        
        # Simple linear regression slope
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2)
        
        # Normalize to reasonable range
        return max(0.1, min(3.0, 1.0 + slope / 10))
    
    def _update_strengths_weaknesses(self, profile: StudentProfile, action: StudentAction):
        """Update student's strengths and weaknesses"""
        problem_type = action.problem_type
        
        # Use a sliding window approach for recent performance
        type_performance = self._get_recent_type_performance(profile.student_id, problem_type)
        
        avg_performance = statistics.mean(type_performance) if type_performance else action.score
        
        # Update strengths (score > 80 and consistent)
        if avg_performance > 80 and len(type_performance) >= 3:
            if problem_type not in profile.strengths:
                profile.strengths.append(problem_type)
            if problem_type in profile.weaknesses:
                profile.weaknesses.remove(problem_type)
        
        # Update weaknesses (score < 60 and consistent)
        elif avg_performance < 60 and len(type_performance) >= 3:
            if problem_type not in profile.weaknesses:
                profile.weaknesses.append(problem_type)
            if problem_type in profile.strengths:
                profile.strengths.remove(problem_type)
    
    def _get_recent_type_performance(self, student_id: str, problem_type: ProblemType, window_size: int = 5) -> List[float]:
        """Get recent performance for a specific problem type"""
        recent_actions = []
        sessions = self.learning_sessions[student_id]
        
        for session in reversed(sessions):
            for action in reversed(session.actions):
                if action.problem_type == problem_type:
                    recent_actions.append(action.score)
                    if len(recent_actions) >= window_size:
                        break
            if len(recent_actions) >= window_size:
                break
        
        return recent_actions
    
    def _adjust_difficulty_level(self, profile: StudentProfile, action: StudentAction) -> DifficultyLevel:
        """Adjust difficulty level based on performance"""
        current_level = profile.current_level
        recent_performance = self._get_recent_overall_performance(profile.student_id)
        
        if not recent_performance:
            return current_level
        
        avg_recent = statistics.mean(recent_performance)
        
        # Promotion criteria
        if avg_recent > 85 and len(recent_performance) >= 5:
            if current_level == DifficultyLevel.BEGINNER:
                return DifficultyLevel.INTERMEDIATE
            elif current_level == DifficultyLevel.INTERMEDIATE:
                return DifficultyLevel.ADVANCED
            elif current_level == DifficultyLevel.ADVANCED:
                return DifficultyLevel.EXPERT
        
        # Demotion criteria (be more conservative)
        elif avg_recent < 50 and len(recent_performance) >= 8:
            if current_level == DifficultyLevel.EXPERT:
                return DifficultyLevel.ADVANCED
            elif current_level == DifficultyLevel.ADVANCED:
                return DifficultyLevel.INTERMEDIATE
            elif current_level == DifficultyLevel.INTERMEDIATE:
                return DifficultyLevel.BEGINNER
        
        return current_level
    
    def _get_recent_overall_performance(self, student_id: str, window_size: int = 10) -> List[float]:
        """Get recent overall performance scores"""
        recent_scores = []
        sessions = self.learning_sessions[student_id]
        
        for session in reversed(sessions):
            for action in reversed(session.actions):
                recent_scores.append(action.score)
                if len(recent_scores) >= window_size:
                    break
            if len(recent_scores) >= window_size:
                break
        
        return recent_scores
    
    def _estimate_attention_span(self, student_id: str, action: StudentAction) -> int:
        """Estimate student's attention span based on session patterns"""
        sessions = self.learning_sessions[student_id]
        if not sessions:
            return 1800  # Default 30 minutes
        
        # Analyze recent session durations and performance decay
        recent_sessions = sessions[-5:]
        durations = []
        
        for session in recent_sessions:
            if session.end_time and session.start_time:
                duration = (session.end_time - session.start_time).total_seconds()
                if 300 <= duration <= 7200:  # Between 5 minutes and 2 hours
                    durations.append(duration)
        
        if durations:
            # Use median as robust estimate
            return int(statistics.median(durations))
        
        return 1800
    
    def recommend_problems(self, student_id: str, num_recommendations: int = 5) -> List[ProblemRecommendation]:
        """Generate personalized problem recommendations"""
        if student_id not in self.student_profiles:
            return self._get_default_recommendations()
        
        profile = self.student_profiles[student_id]
        recommendations = []
        
        # Strategy 1: Address weaknesses
        weakness_recs = self._recommend_for_weaknesses(profile, num_recommendations // 2)
        recommendations.extend(weakness_recs)
        
        # Strategy 2: Build on strengths with harder problems
        strength_recs = self._recommend_for_strengths(profile, num_recommendations // 3)
        recommendations.extend(strength_recs)
        
        # Strategy 3: Introduce new topics (if ready)
        new_topic_recs = self._recommend_new_topics(profile, num_recommendations - len(recommendations))
        recommendations.extend(new_topic_recs)
        
        # Sort by confidence and return top recommendations
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        return recommendations[:num_recommendations]
    
    def _recommend_for_weaknesses(self, profile: StudentProfile, count: int) -> List[ProblemRecommendation]:
        """Recommend problems to address weaknesses"""
        recommendations = []
        
        for weakness in profile.weaknesses[:count]:
            # Choose slightly easier problems for weaknesses
            target_difficulty = self._get_lower_difficulty(profile.current_level)
            
            rec = ProblemRecommendation(
                problem_id=f"weakness_{weakness}_{target_difficulty}",
                problem_type=weakness,
                difficulty=target_difficulty,
                title=f"Practice {weakness.replace('_', ' ').title()}",
                description=f"Targeted practice for {weakness.replace('_', ' ')} at {target_difficulty} level",
                estimated_time=self._estimate_problem_time(weakness, target_difficulty),
                learning_objectives=[LearningObjective.SOLVE_PROBLEMS],
                confidence_score=0.8,
                reasoning=f"Addressing identified weakness in {weakness}"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _recommend_for_strengths(self, profile: StudentProfile, count: int) -> List[ProblemRecommendation]:
        """Recommend challenging problems in strength areas"""
        recommendations = []
        
        for strength in profile.strengths[:count]:
            # Choose harder problems for strengths
            target_difficulty = self._get_higher_difficulty(profile.current_level)
            
            rec = ProblemRecommendation(
                problem_id=f"strength_{strength}_{target_difficulty}",
                problem_type=strength,
                difficulty=target_difficulty,
                title=f"Advanced {strength.replace('_', ' ').title()}",
                description=f"Challenging {strength.replace('_', ' ')} problems at {target_difficulty} level",
                estimated_time=self._estimate_problem_time(strength, target_difficulty),
                learning_objectives=[LearningObjective.APPLY_ALGORITHMS],
                confidence_score=0.7,
                reasoning=f"Building on strength in {strength} with harder problems"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _recommend_new_topics(self, profile: StudentProfile, count: int) -> List[ProblemRecommendation]:
        """Recommend new topics student is ready for"""
        recommendations = []
        
        # Get topics student hasn't tried much
        attempted_types = set()
        for session in self.learning_sessions[profile.student_id]:
            for action in session.actions:
                attempted_types.add(action.problem_type)
        
        available_types = set(ProblemType) - attempted_types
        
        for problem_type in list(available_types)[:count]:
            rec = ProblemRecommendation(
                problem_id=f"new_{problem_type}_{profile.current_level}",
                problem_type=problem_type,
                difficulty=profile.current_level,
                title=f"Introduction to {problem_type.replace('_', ' ').title()}",
                description=f"Learn {problem_type.replace('_', ' ')} at {profile.current_level} level",
                estimated_time=self._estimate_problem_time(problem_type, profile.current_level),
                learning_objectives=[LearningObjective.UNDERSTAND_CONCEPTS],
                confidence_score=0.6,
                reasoning=f"Ready to explore new topic: {problem_type}"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _get_lower_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get one level lower difficulty"""
        if current == DifficultyLevel.EXPERT:
            return DifficultyLevel.ADVANCED
        elif current == DifficultyLevel.ADVANCED:
            return DifficultyLevel.INTERMEDIATE
        elif current == DifficultyLevel.INTERMEDIATE:
            return DifficultyLevel.BEGINNER
        else:
            return DifficultyLevel.BEGINNER
    
    def _get_higher_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get one level higher difficulty"""
        if current == DifficultyLevel.BEGINNER:
            return DifficultyLevel.INTERMEDIATE
        elif current == DifficultyLevel.INTERMEDIATE:
            return DifficultyLevel.ADVANCED
        elif current == DifficultyLevel.ADVANCED:
            return DifficultyLevel.EXPERT
        else:
            return DifficultyLevel.EXPERT
    
    def _estimate_problem_time(self, problem_type: ProblemType, difficulty: DifficultyLevel) -> int:
        """Estimate problem completion time in minutes"""
        base_times = {
            ProblemType.DFA_CONSTRUCTION: 15,
            ProblemType.NFA_CONSTRUCTION: 20,
            ProblemType.PUMPING_LEMMA: 25,
            ProblemType.PROOF_WRITING: 30,
            ProblemType.COMPLEXITY_ANALYSIS: 20,
            ProblemType.REDUCTION: 35
        }
        
        difficulty_multipliers = {
            DifficultyLevel.BEGINNER: 1.0,
            DifficultyLevel.INTERMEDIATE: 1.5,
            DifficultyLevel.ADVANCED: 2.0,
            DifficultyLevel.EXPERT: 3.0
        }
        
        base_time = base_times.get(problem_type, 20)
        multiplier = difficulty_multipliers[difficulty]
        
        return int(base_time * multiplier)
    
    def _get_default_recommendations(self) -> List[ProblemRecommendation]:
        """Get default recommendations for new students"""
        return [
            ProblemRecommendation(
                problem_id="intro_dfa_1",
                problem_type=ProblemType.DFA_CONSTRUCTION,
                difficulty=DifficultyLevel.BEGINNER,
                title="Introduction to DFA Construction",
                description="Learn to build deterministic finite automata",
                estimated_time=15,
                learning_objectives=[LearningObjective.UNDERSTAND_CONCEPTS],
                confidence_score=0.9,
                reasoning="Essential starting point for automata theory"
            )
        ]
    
    def get_learning_analytics(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive learning analytics for a student"""
        if student_id not in self.student_profiles:
            return {"error": "Student not found"}
        
        profile = self.student_profiles[student_id]
        sessions = self.learning_sessions[student_id]
        
        # Performance over time
        performance_trend = self._get_performance_trend(student_id)
        
        # Problem type analysis
        type_analysis = self._analyze_problem_types(student_id)
        
        # Learning patterns
        learning_patterns = self._analyze_learning_patterns(student_id)
        
        return {
            "profile": profile,
            "performance_trend": performance_trend,
            "problem_type_analysis": type_analysis,
            "learning_patterns": learning_patterns,
            "recommendations": self.recommend_problems(student_id, 3),
            "achievements": self._get_achievements(student_id),
            "next_goals": self._suggest_learning_goals(student_id)
        }
    
    def _get_performance_trend(self, student_id: str) -> List[Dict[str, Any]]:
        """Get performance trend over time"""
        sessions = self.learning_sessions[student_id]
        trend = []
        
        for session in sessions:
            if session.actions:
                avg_score = statistics.mean([a.score for a in session.actions])
                trend.append({
                    "date": session.start_time.isoformat(),
                    "average_score": avg_score,
                    "problems_attempted": len(session.actions),
                    "success_rate": sum(1 for a in session.actions if a.success) / len(session.actions)
                })
        
        return trend
    
    def _analyze_problem_types(self, student_id: str) -> Dict[str, Any]:
        """Analyze performance by problem type"""
        type_stats = defaultdict(list)
        
        for session in self.learning_sessions[student_id]:
            for action in session.actions:
                type_stats[action.problem_type].append(action.score)
        
        analysis = {}
        for problem_type, scores in type_stats.items():
            analysis[problem_type] = {
                "attempts": len(scores),
                "average_score": statistics.mean(scores),
                "success_rate": sum(1 for s in scores if s >= 70) / len(scores),
                "improvement": self._calculate_improvement(scores)
            }
        
        return analysis
    
    def _calculate_improvement(self, scores: List[float]) -> float:
        """Calculate improvement rate from score sequence"""
        if len(scores) < 2:
            return 0.0
        
        # Compare first half to second half
        mid = len(scores) // 2
        first_half_avg = statistics.mean(scores[:mid]) if mid > 0 else scores[0]
        second_half_avg = statistics.mean(scores[mid:])
        
        return second_half_avg - first_half_avg
    
    def _analyze_learning_patterns(self, student_id: str) -> Dict[str, Any]:
        """Analyze student's learning patterns"""
        sessions = self.learning_sessions[student_id]
        
        # Time-based patterns
        session_times = []
        session_durations = []
        
        for session in sessions:
            if session.start_time:
                session_times.append(session.start_time.hour)
            if session.end_time and session.start_time:
                duration = (session.end_time - session.start_time).total_seconds() / 60
                session_durations.append(duration)
        
        return {
            "preferred_study_hours": self._find_peak_hours(session_times),
            "average_session_duration": statistics.mean(session_durations) if session_durations else 0,
            "most_productive_duration": self._find_optimal_duration(student_id),
            "learning_consistency": self._calculate_consistency(sessions)
        }
    
    def _find_peak_hours(self, hours: List[int]) -> List[int]:
        """Find peak study hours"""
        if not hours:
            return []
        
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        max_count = max(hour_counts.values())
        return [hour for hour, count in hour_counts.items() if count == max_count]
    
    def _find_optimal_duration(self, student_id: str) -> int:
        """Find optimal session duration for student"""
        sessions = self.learning_sessions[student_id]
        duration_performance = []
        
        for session in sessions:
            if session.end_time and session.start_time and session.actions:
                duration = (session.end_time - session.start_time).total_seconds() / 60
                avg_score = statistics.mean([a.score for a in session.actions])
                duration_performance.append((duration, avg_score))
        
        if not duration_performance:
            return 30  # Default
        
        # Find duration with best performance
        best_duration = max(duration_performance, key=lambda x: x[1])[0]
        return int(best_duration)
    
    def _calculate_consistency(self, sessions: List[LearningSession]) -> float:
        """Calculate learning consistency score"""
        if len(sessions) < 2:
            return 0.0
        
        # Measure regularity of study sessions
        intervals = []
        for i in range(1, len(sessions)):
            interval = (sessions[i].start_time - sessions[i-1].start_time).days
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        # Lower variance means higher consistency
        variance = statistics.variance(intervals) if len(intervals) > 1 else 0
        consistency = 1.0 / (1.0 + variance)  # Normalize to 0-1
        
        return min(1.0, consistency)
    
    def _get_achievements(self, student_id: str) -> List[Dict[str, Any]]:
        """Get student achievements"""
        profile = self.student_profiles[student_id]
        achievements = []
        
        # Problem-solving achievements
        if profile.total_problems_solved >= 10:
            achievements.append({
                "title": "Problem Solver",
                "description": "Solved 10+ problems",
                "category": "milestone"
            })
        
        if profile.average_score >= 85:
            achievements.append({
                "title": "High Achiever",
                "description": "Maintaining 85+ average score",
                "category": "performance"
            })
        
        # Strength-based achievements
        for strength in profile.strengths:
            achievements.append({
                "title": f"{strength.replace('_', ' ').title()} Expert",
                "description": f"Demonstrated expertise in {strength.replace('_', ' ')}",
                "category": "expertise"
            })
        
        return achievements
    
    def _suggest_learning_goals(self, student_id: str) -> List[str]:
        """Suggest next learning goals"""
        profile = self.student_profiles[student_id]
        goals = []
        
        # Address weaknesses
        if profile.weaknesses:
            weakness = profile.weaknesses[0]
            goals.append(f"Improve performance in {weakness.replace('_', ' ')}")
        
        # Level progression
        if profile.current_level != DifficultyLevel.EXPERT:
            next_level = self._get_higher_difficulty(profile.current_level)
            goals.append(f"Progress to {next_level} level problems")
        
        # Skill diversification
        attempted_types = set()
        for session in self.learning_sessions[student_id]:
            for action in session.actions:
                attempted_types.add(action.problem_type)
        
        remaining_types = set(ProblemType) - attempted_types
        if remaining_types:
            next_type = list(remaining_types)[0]
            goals.append(f"Learn {next_type.replace('_', ' ')}")
        
        return goals

# Global adaptive learning engine
adaptive_engine = AdaptiveLearningEngine()

def update_student_performance(student_id: str, action: StudentAction) -> StudentProfile:
    """Update student performance based on action"""
    return adaptive_engine.update_student_performance(student_id, action)

def get_problem_recommendations(student_id: str, count: int = 5) -> List[ProblemRecommendation]:
    """Get personalized problem recommendations"""
    return adaptive_engine.recommend_problems(student_id, count)

def get_student_analytics(student_id: str) -> Dict[str, Any]:
    """Get comprehensive learning analytics"""
    return adaptive_engine.get_learning_analytics(student_id)

def start_learning_session(student_id: str, session_id: str) -> LearningSession:
    """Start a new learning session"""
    session = LearningSession(
        session_id=session_id,
        student_id=student_id,
        start_time=datetime.now()
    )
    adaptive_engine.learning_sessions[student_id].append(session)
    return session

def end_learning_session(student_id: str, session_id: str) -> Optional[LearningSession]:
    """End a learning session"""
    sessions = adaptive_engine.learning_sessions[student_id]
    for session in reversed(sessions):
        if session.session_id == session_id and session.end_time is None:
            session.end_time = datetime.now()
            if session.actions:
                session.session_score = statistics.mean([a.score for a in session.actions])
            return session
    return None

def get_difficulty_adjustment(student_id: str, problem_type: ProblemType) -> DifficultyLevel:
    """Get recommended difficulty level for a problem type"""
    if student_id not in adaptive_engine.student_profiles:
        return DifficultyLevel.BEGINNER
    
    profile = adaptive_engine.student_profiles[student_id]
    
    # If it's a strength, can handle higher difficulty
    if problem_type in profile.strengths:
        return adaptive_engine._get_higher_difficulty(profile.current_level)
    # If it's a weakness, use lower difficulty
    elif problem_type in profile.weaknesses:
        return adaptive_engine._get_lower_difficulty(profile.current_level)
    else:
        return profile.current_level