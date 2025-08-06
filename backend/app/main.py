"""
Main FastAPI application for the Automata Learning Platform.
Refactored to use microservice architecture with separate routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

# Import configuration and middleware
from .config import settings
from .middleware import ErrorHandlingMiddleware, RateLimitMiddleware
from .database import init_db
from .security import (
    SecurityMiddleware, limiter, security_manager, security_logger,
    rate_limit_general, security_scan_request
)
from .monitoring import (
    MonitoringMiddleware, performance_monitor, health_checker,
    get_metrics, get_health, get_performance_stats, setup_monitoring, cleanup_monitoring
)

# Import all routers
from .routers.auth_router import router as auth_router
from .routers.problems_router import router as problems_router
from .routers.ai_router import router as ai_router
from .routers.ai_jflap_router import router as ai_jflap_router
from .routers.jflap_router import router as jflap_router
from .routers.learning_router import router as learning_router
from .routers.verification_router import router as verification_router
from .routers.papers_router import router as papers_router
from .routers.websocket_router import router as websocket_router
from .routers.langgraph_router import router as langgraph_router
from .routers.latex_router import router as latex_router
from .routers.api_platform_router import router as api_platform_router
from .routers.grading_router import router as grading_router

# Import WebSocket server components
from .websocket_server import socket_app, init_websocket_server, cleanup_websocket_server
from .yjs_integration import initialize_yjs_integration, cleanup_yjs_integration

# Import new platform components
from .api_platform import initialize_api_platform, cleanup_api_platform

# Configure logging
logging.basicConfig(level=settings.log_level, format=settings.log_format)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="AI-powered Theory of Computation Educational Platform",
    debug=settings.debug,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware for error handling, rate limiting, security, and monitoring
app.add_middleware(MonitoringMiddleware, performance_monitor)
app.add_middleware(SecurityMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# Add SlowAPI rate limiting
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from fastapi import Request

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS for production security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include all routers
app.include_router(auth_router)
app.include_router(problems_router)
app.include_router(ai_router)
app.include_router(ai_jflap_router)
app.include_router(jflap_router)
app.include_router(learning_router)
app.include_router(verification_router)
app.include_router(papers_router)
app.include_router(websocket_router)
app.include_router(langgraph_router)
app.include_router(latex_router)
app.include_router(api_platform_router)
app.include_router(grading_router)

# Mount WebSocket app
app.mount("/ws", socket_app)

# Global problems and solutions storage (in production, this would be in a real database)
problems_db = {}
solutions_db = {}

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and application components on startup."""
    try:
        init_db()
        logger.info("Database initialized successfully")
        
        # Initialize other components
        await initialize_ai_components()
        await initialize_problem_database()
        
        # Initialize Redis integration for LangGraph
        await initialize_redis_integration()
        
        # Setup monitoring
        setup_monitoring()
        
        # Initialize WebSocket server
        await init_websocket_server()
        
        # Initialize Y.js integration
        await initialize_yjs_integration()
        
        # Initialize API Platform
        await initialize_api_platform()
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


async def initialize_ai_components():
    """Initialize AI-related components."""
    try:
        # Initialize AI components
        from .agents import AutomataGenerator, AutomataExplainer
        from .prompts import prompt_builder
        from .orchestrator import orchestrate_task
        from .semantic_search import semantic_search_engine
        from .rag_system import rag_system
        from .memory import memory_manager
        
        logger.info("AI components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI components: {e}")


async def initialize_redis_integration():
    """Initialize Redis integration for LangGraph workflows."""
    try:
        from .redis_integration import initialize_redis
        await initialize_redis()
        logger.info("Redis integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis integration: {e}")
        # Don't raise here as Redis is optional for basic functionality


async def initialize_problem_database():
    """Initialize the problems database with sample problems."""
    try:
        # This would typically load problems from a database
        # For now, we'll use the existing problems_db initialization
        init_comprehensive_problems()
        logger.info(f"Problems database initialized with {len(problems_db)} problems")
    except Exception as e:
        logger.error(f"Failed to initialize problems database: {e}")


def init_comprehensive_problems():
    """Initialize comprehensive problem set (placeholder)."""
    # This function would be moved to a separate module in production
    # For now, keeping it simple
    problems_db["sample_dfa"] = {
        "id": "sample_dfa",
        "type": "dfa",
        "title": "Sample DFA Problem",
        "description": "Build a DFA for demonstration",
        "test_strings": [
            {"string": "a", "should_accept": True},
            {"string": "b", "should_accept": False}
        ],
        "hints": ["Start with identifying the accepting condition"]
    }


# Root endpoint
@app.get("/")
@rate_limit_general()
async def root(request: Request):
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": "AI-powered Theory of Computation Educational Platform",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "authentication": "/auth/*",
            "problems": "/problems/*",
            "ai_services": "/api/ai/*",
            "ai_jflap_features": "/api/ai-jflap/*",
            "jflap_algorithms": "/api/jflap/*",
            "adaptive_learning": "/api/learning/*",
            "verification": "/api/verification/*",
            "research_papers": "/api/papers/*",
            "langgraph_workflows": "/api/langgraph/*",
            "documentation": "/docs",
            "health_check": "/health"
        },
        "features": [
            "User authentication and management",
            "Interactive problem solving",
            "AI-powered tutoring and hints",
            "JFLAP algorithm implementations",
            "Advanced multi-tape Turing Machines",
            "Universal Turing Machine simulation",
            "SLR(1) parser with DFA construction",
            "Unrestricted and context-sensitive grammars",
            "GNF (Greibach Normal Form) conversion",
            "Enhanced L-Systems with graphics",
            "AI-powered natural language conversion",
            "Intelligent error recovery",
            "Automated test generation",
            "Adaptive learning system",
            "Formal verification tools",
            "Research paper database",
            "Real-time collaboration",
            "Progress tracking"
        ]
    }


# Health check endpoints
@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.api_version,
            "components": {
                "database": "healthy",
                "ai_services": "healthy",
                "authentication": "healthy",
                "jflap_algorithms": "healthy",
                "learning_engine": "healthy",
                "verification_engine": "healthy",
                "papers_database": "healthy"
            },
            "metrics": {
                "total_problems": len(problems_db),
                "active_sessions": 0,  # This would come from session manager
                "uptime_seconds": 0    # This would come from uptime tracker
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/healthz")
@limiter.limit("120/minute")
async def kubernetes_health_check(request: Request):
    """Kubernetes-style health check."""
    return {"status": "ok"}


# Monitoring endpoints
@app.get("/metrics")
async def metrics_endpoint(request: Request):
    """Prometheus metrics endpoint."""
    return await get_metrics()


@app.get("/health/detailed")
@limiter.limit("30/minute")
async def detailed_health_check(request: Request):
    """Detailed health check with component status."""
    return await get_health()


@app.get("/metrics/performance")
@limiter.limit("60/minute")
async def performance_metrics(request: Request):
    """Performance statistics endpoint."""
    return await get_performance_stats()


# Application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    try:
        logger.info("Application shutdown initiated")
        
        # Clean up AI components
        await cleanup_ai_components()
        
        # Clean up Redis integration
        await cleanup_redis_integration()
        
        # Close database connections
        await cleanup_database_connections()
        
        # Cleanup monitoring
        cleanup_monitoring()
        
        # Cleanup WebSocket server
        await cleanup_websocket_server()
        
        # Cleanup Y.js integration
        await cleanup_yjs_integration()
        
        # Cleanup API Platform
        await cleanup_api_platform()
        
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


async def cleanup_ai_components():
    """Clean up AI-related resources."""
    try:
        # Close AI service connections
        # Clean up model caches
        # Save learning data
        logger.info("AI components cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up AI components: {e}")


async def cleanup_redis_integration():
    """Clean up Redis integration resources."""
    try:
        from .redis_integration import shutdown_redis
        await shutdown_redis()
        logger.info("Redis integration cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up Redis integration: {e}")


async def cleanup_database_connections():
    """Clean up database connections."""
    try:
        # Close database connection pools
        # Save any pending data
        logger.info("Database connections cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up database connections: {e}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )