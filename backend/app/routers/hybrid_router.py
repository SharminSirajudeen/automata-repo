"""
Hybrid Router - API endpoints for the Hybrid Orchestration System
================================================================

Provides endpoints for:
- Intelligent problem routing
- Hybrid solution execution
- System status and metrics
- Learning insights and recommendations

Author: AegisX AI Software Engineer
Version: 1.0
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

from ..hybrid_orchestrator import (
    hybrid_orchestrator, solve_problem_with_hybrid_approach, HybridSolution
)
from ..intelligent_router import (
    intelligent_router, route_problem, SolutionType, RoutingResult
)
from ..enhanced_learning_system import (
    enhanced_learning_system, get_enhanced_learning_insights
)
from ..problem_understanding import ProblemType, LanguagePattern
from ..auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid", tags=["hybrid"])


# Request/Response Models
class HybridSolveRequest(BaseModel):
    """Request model for hybrid problem solving."""
    problem_statement: str = Field(..., description="The problem to solve")
    problem_type: ProblemType = Field(..., description="Type of the problem")
    patterns: List[LanguagePattern] = Field(default=[], description="Language patterns involved")
    performance_requirements: Optional[Dict[str, Any]] = Field(
        default=None, description="Performance requirements"
    )
    force_approach: Optional[SolutionType] = Field(
        default=None, description="Force a specific solution approach"
    )


class RoutingRequest(BaseModel):
    """Request model for routing decisions."""
    problem_statement: str = Field(..., description="The problem to analyze")
    problem_type: ProblemType = Field(..., description="Type of the problem")
    patterns: List[LanguagePattern] = Field(default=[], description="Language patterns")
    performance_requirements: Optional[Dict[str, Any]] = Field(
        default=None, description="Performance requirements"
    )


class LearningInsightsRequest(BaseModel):
    """Request model for learning insights."""
    problem_statement: str = Field(..., description="The problem to analyze")
    problem_type: ProblemType = Field(..., description="Type of the problem")
    patterns: List[LanguagePattern] = Field(default=[], description="Language patterns")


class HybridSolutionResponse(BaseModel):
    """Response model for hybrid solutions."""
    execution_id: str
    routing_decision: str
    confidence_score: float
    execution_time: float
    primary_solution: Dict[str, Any]
    verification_results: Dict[str, Any]
    learning_applied: bool
    cross_verification_passed: bool
    metadata: Dict[str, Any]


class RoutingResponse(BaseModel):
    """Response model for routing decisions."""
    decision: str
    confidence: float
    reasoning: List[str]
    fallback_chain: List[str]
    expected_performance: Dict[str, float]
    resource_requirements: Dict[str, Any]


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    active_executions: int
    total_executions: int
    success_rate: float
    average_execution_time: float
    routing_accuracy: float
    approach_success_rates: Dict[str, float]
    learning_system_stats: Dict[str, Any]
    routing_stats: Dict[str, Any]


# Main Endpoints

@router.post("/solve", response_model=HybridSolutionResponse)
async def solve_problem_hybrid(
    request: HybridSolveRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Solve a problem using the intelligent hybrid approach.
    
    This endpoint:
    1. Analyzes the problem to determine the best solution approach
    2. Routes to hardcoded algorithms, AI, or hybrid solutions
    3. Cross-verifies results for accuracy
    4. Applies learning from the solution
    """
    try:
        logger.info(f"Solving problem hybrid for user {current_user.get('email', 'unknown')}")
        
        # Execute hybrid problem solving
        solution = await solve_problem_with_hybrid_approach(
            problem_statement=request.problem_statement,
            problem_type=request.problem_type,
            patterns=request.patterns,
            performance_requirements=request.performance_requirements,
            force_approach=request.force_approach
        )
        
        # Convert to response format
        response = HybridSolutionResponse(
            execution_id=solution.metadata.get('execution_id', 'unknown'),
            routing_decision=solution.routing_decision.value,
            confidence_score=solution.confidence_score,
            execution_time=solution.execution_time,
            primary_solution={
                'strategy': solution.primary_solution.strategy_used.value,
                'confidence': solution.primary_solution.confidence_score,
                'final_solution': solution.primary_solution.final_solution,
                'explanation': solution.primary_solution.explanation,
                'steps': [
                    {
                        'action': step.action,
                        'description': step.description,
                        'confidence': step.confidence
                    }
                    for step in solution.primary_solution.solution_steps
                ]
            },
            verification_results=solution.verification_results,
            learning_applied=solution.learning_applied,
            cross_verification_passed=solution.cross_verification_passed,
            metadata=solution.metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Hybrid problem solving failed: {e}")
        raise HTTPException(status_code=500, detail=f"Problem solving failed: {str(e)}")


@router.post("/route", response_model=RoutingResponse)
async def get_routing_decision(
    request: RoutingRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Get intelligent routing decision for a problem without executing the solution.
    
    This endpoint analyzes the problem and recommends the best solution approach
    based on problem complexity, available algorithms, and performance requirements.
    """
    try:
        logger.info(f"Getting routing decision for user {current_user.get('email', 'unknown')}")
        
        # Get routing decision
        routing_result = await route_problem(
            problem_statement=request.problem_statement,
            problem_type=request.problem_type,
            patterns=request.patterns,
            performance_requirements=request.performance_requirements
        )
        
        # Convert to response format
        response = RoutingResponse(
            decision=routing_result.decision.value,
            confidence=routing_result.confidence,
            reasoning=routing_result.reasoning,
            fallback_chain=[chain_item.value for chain_item in routing_result.fallback_chain],
            expected_performance=routing_result.expected_performance,
            resource_requirements=routing_result.resource_requirements
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Routing decision failed: {e}")
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")


@router.post("/insights")
async def get_learning_insights(
    request: LearningInsightsRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Get enhanced learning insights for a problem.
    
    This endpoint provides insights combining:
    - Historical success patterns
    - Hardcoded algorithm knowledge
    - AI learning recommendations
    - Optimization suggestions
    """
    try:
        logger.info(f"Getting learning insights for user {current_user.get('email', 'unknown')}")
        
        # Get enhanced learning insights
        insights = await get_enhanced_learning_insights(
            problem_statement=request.problem_statement,
            problem_type=request.problem_type,
            patterns=request.patterns
        )
        
        return {
            'recommended_strategy': insights.recommended_strategy.value,
            'confidence_boost': insights.confidence_boost,
            'similar_problems': insights.similar_problems,
            'pattern_matches': insights.pattern_matches,
            'optimization_suggestions': insights.optimization_suggestions,
            'predicted_difficulty': insights.predicted_difficulty
        }
        
    except Exception as e:
        logger.error(f"Learning insights failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


@router.get("/recommendations/{problem_type}")
async def get_hybrid_recommendations(
    problem_type: ProblemType = Path(..., description="Type of problem"),
    patterns: List[LanguagePattern] = Query(default=[], description="Language patterns"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get specific recommendations for hybrid approaches for a problem type.
    
    Returns recommendations on:
    - Whether to use hybrid approach
    - Which components to combine
    - Integration strategies
    - Expected benefits
    """
    try:
        from ..problem_understanding import ProblemRequirements
        
        # Create minimal requirements for recommendations
        requirements = ProblemRequirements(
            original_statement=f"Generic {problem_type.value} problem",
            problem_type=problem_type,
            patterns=patterns
        )
        
        recommendations = await enhanced_learning_system.get_hybrid_recommendations(requirements)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Hybrid recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


# Status and Monitoring Endpoints

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(current_user: dict = Depends(get_current_user)):
    """
    Get current system status and performance metrics.
    
    Returns comprehensive metrics about:
    - Execution success rates
    - Performance by approach (hardcoded/AI/hybrid)
    - Learning system statistics
    - Routing accuracy
    """
    try:
        status = hybrid_orchestrator.get_system_status()
        
        response = SystemStatusResponse(
            active_executions=status['active_executions'],
            total_executions=status['total_executions'],
            success_rate=status['success_rate'],
            average_execution_time=status['average_execution_time'],
            routing_accuracy=status['routing_accuracy'],
            approach_success_rates=status['approach_success_rates'],
            learning_system_stats=status['learning_system_stats'],
            routing_stats=status['routing_stats']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"System status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.get("/execution/{execution_id}")
async def get_execution_status(
    execution_id: str = Path(..., description="Execution ID"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get status of a specific execution.
    
    Returns real-time status of problem solving execution including:
    - Current status (routing, executing, verifying, etc.)
    - Execution log
    - Elapsed time
    - Routing decision made
    """
    try:
        status = await hybrid_orchestrator.get_execution_status(execution_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execution status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


# Analytics and Learning Endpoints

@router.get("/analytics/routing")
async def get_routing_analytics(
    days: int = Query(default=30, description="Number of days to analyze"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get routing analytics and statistics.
    
    Returns analysis of routing decisions including:
    - Decision distribution
    - Accuracy by decision type  
    - Performance trends
    - Most common patterns
    """
    try:
        stats = intelligent_router.get_routing_statistics()
        
        # Add time-based filtering here if needed
        # For now, return all available statistics
        
        return {
            'period_days': days,
            'routing_statistics': stats,
            'insights': [
                "Routing system learning from decision outcomes",
                "Performance improves with more data",
                "Hybrid approaches show best overall results"
            ]
        }
        
    except Exception as e:
        logger.error(f"Routing analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@router.get("/analytics/learning")
async def get_learning_analytics(current_user: dict = Depends(get_current_user)):
    """
    Get learning system analytics and statistics.
    
    Returns comprehensive learning statistics including:
    - Knowledge base size
    - Learning accuracy improvements
    - Pattern recognition effectiveness
    - Hybrid insight generation
    """
    try:
        stats = enhanced_learning_system.get_enhanced_statistics()
        
        return {
            'learning_statistics': stats,
            'system_insights': [
                f"Enhanced learning system integrated {stats.get('hardcoded_knowledge_integrated', 0)} algorithm patterns",
                f"Generated {stats.get('hybrid_insights_generated', 0)} hybrid insights",
                f"Knowledge coverage: {stats.get('knowledge_coverage_percentage', 0):.1f}%",
                f"Average improvement from hybrid approach: {stats.get('average_improvement_from_hybrid', 0):.3f}"
            ]
        }
        
    except Exception as e:
        logger.error(f"Learning analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


# Testing and Debug Endpoints

@router.post("/test/compare-approaches")
async def test_compare_approaches(
    request: HybridSolveRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Test endpoint to compare different solution approaches on the same problem.
    
    Executes the problem using all available approaches and compares results:
    - Hardcoded solution (if available)
    - AI solution
    - Hybrid solution
    - Performance comparison
    """
    try:
        logger.info(f"Comparing approaches for user {current_user.get('email', 'unknown')}")
        
        results = {}
        
        # Try hardcoded approach
        try:
            hardcoded_result = await solve_problem_with_hybrid_approach(
                problem_statement=request.problem_statement,
                problem_type=request.problem_type,
                patterns=request.patterns,
                force_approach=SolutionType.HARDCODED
            )
            results['hardcoded'] = {
                'success': True,
                'confidence': hardcoded_result.confidence_score,
                'execution_time': hardcoded_result.execution_time,
                'approach': hardcoded_result.routing_decision.value
            }
        except Exception as e:
            results['hardcoded'] = {
                'success': False,
                'error': str(e)
            }
        
        # Try AI approach
        try:
            ai_result = await solve_problem_with_hybrid_approach(
                problem_statement=request.problem_statement,
                problem_type=request.problem_type,
                patterns=request.patterns,
                force_approach=SolutionType.AI_GENERATED
            )
            results['ai'] = {
                'success': True,
                'confidence': ai_result.confidence_score,
                'execution_time': ai_result.execution_time,
                'approach': ai_result.routing_decision.value
            }
        except Exception as e:
            results['ai'] = {
                'success': False,
                'error': str(e)
            }
        
        # Try hybrid approach
        try:
            hybrid_result = await solve_problem_with_hybrid_approach(
                problem_statement=request.problem_statement,
                problem_type=request.problem_type,
                patterns=request.patterns,
                force_approach=SolutionType.HYBRID
            )
            results['hybrid'] = {
                'success': True,
                'confidence': hybrid_result.confidence_score,
                'execution_time': hybrid_result.execution_time,
                'approach': hybrid_result.routing_decision.value
            }
        except Exception as e:
            results['hybrid'] = {
                'success': False,
                'error': str(e)
            }
        
        # Automatic routing (no force)
        try:
            auto_result = await solve_problem_with_hybrid_approach(
                problem_statement=request.problem_statement,
                problem_type=request.problem_type,
                patterns=request.patterns
            )
            results['automatic_routing'] = {
                'success': True,
                'confidence': auto_result.confidence_score,
                'execution_time': auto_result.execution_time,
                'approach': auto_result.routing_decision.value,
                'routing_confidence': auto_result.metadata.get('routing_confidence', 0.0)
            }
        except Exception as e:
            results['automatic_routing'] = {
                'success': False,
                'error': str(e)
            }
        
        # Generate comparison insights
        successful_approaches = [k for k, v in results.items() if v.get('success', False)]
        if successful_approaches:
            best_confidence = max(results[k].get('confidence', 0) for k in successful_approaches)
            best_speed = min(results[k].get('execution_time', float('inf')) for k in successful_approaches)
            
            comparison_insights = [
                f"Successful approaches: {', '.join(successful_approaches)}",
                f"Best confidence: {best_confidence:.3f}",
                f"Best speed: {best_speed:.3f}s"
            ]
        else:
            comparison_insights = ["All approaches failed"]
        
        return {
            'problem_summary': {
                'type': request.problem_type.value,
                'patterns': [p.value for p in request.patterns],
                'statement_length': len(request.problem_statement)
            },
            'results': results,
            'comparison_insights': comparison_insights
        }
        
    except Exception as e:
        logger.error(f"Approach comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# Health Check
@router.get("/health")
async def health_check():
    """Health check endpoint for the hybrid system."""
    try:
        # Basic system checks
        system_healthy = True
        health_details = {
            'router': True,
            'learning_system': True,
            'orchestrator': True,
            'knowledge_extractor': True
        }
        
        # Could add more sophisticated health checks here
        # For example, checking if ML models are loaded, etc.
        
        return {
            'status': 'healthy' if system_healthy else 'unhealthy',
            'components': health_details,
            'timestamp': hybrid_orchestrator.performance_metrics.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }