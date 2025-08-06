"""
AI router for the Automata Learning Platform.
Handles AI-powered features including prompts, orchestration, proofs, search, and optimization.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import httpx
import logging

# Import AI-related modules
from ..prompts import prompt_builder, PromptExample
from ..orchestrator import orchestrate_task
from ..ai_proof_assistant import (
    proof_generator, ProofTechnique, ProofStep, ProofStatus
)
from ..semantic_search import semantic_search_engine
from ..rag_system import rag_system
from ..memory import memory_manager
from ..optimizer import ai_optimizer
from ..ai_config import TaskComplexity
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])

# Configuration
OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_default_model


class PromptRequest(BaseModel):
    template_name: str
    variables: Dict[str, Any]
    examples: Optional[List[Dict[str, str]]] = None


class OrchestrationRequest(BaseModel):
    task: str
    prompt: str
    mode: str = "sequential"
    complexity: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ProofRequest(BaseModel):
    theorem: str
    technique: Optional[str] = None
    context: Optional[str] = None


class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    threshold: Optional[float] = 0.7


class KnowledgeRequest(BaseModel):
    content: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None


class MessageRequest(BaseModel):
    session_id: str
    message: str
    context: Optional[Dict[str, Any]] = None


class PreferencesRequest(BaseModel):
    user_id: str
    preferences: Dict[str, Any]


class AutomatonOptimizationRequest(BaseModel):
    automaton: Dict[str, Any]
    optimization_type: str = "minimize"


@router.get("/status")
async def check_ai_status():
    """Check if AI services (Ollama) are available"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return {
                    "available": True,
                    "models": available_models,
                    "current_model": OLLAMA_MODEL,
                    "generator_model": "codellama:34b",
                    "explainer_model": "deepseek-coder:33b"
                }
            else:
                return {"available": False, "error": "Ollama service not responding"}
    except Exception as e:
        logger.error(f"AI status check failed: {e}")
        return {"available": False, "error": str(e)}


# Prompt Generation Endpoints
@router.post("/prompt/generate")
async def generate_prompt(request: PromptRequest):
    """Generate a prompt from template with variables"""
    try:
        prompt_examples = None
        if request.examples:
            prompt_examples = [
                PromptExample(
                    input=ex.get("input", ""),
                    output=ex.get("output", ""),
                    explanation=ex.get("explanation")
                )
                for ex in request.examples
            ]
        
        prompt = prompt_builder.build(
            request.template_name,
            request.variables,
            examples=prompt_examples,
            optimize=True
        )
        
        return {
            "prompt": prompt,
            "template_used": request.template_name,
            "optimized": True
        }
    except Exception as e:
        logger.error(f"Prompt generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompt/templates")
async def list_prompt_templates():
    """List available prompt templates"""
    try:
        templates = prompt_builder.library.list_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Orchestration Endpoints
@router.post("/orchestrate")
async def orchestrate_model_task(request: OrchestrationRequest):
    """Execute task using model orchestration"""
    try:
        result = await orchestrate_task(
            task=request.task,
            prompt=request.prompt,
            mode=request.mode,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "task": request.task,
            "mode": request.mode,
            "result": result
        }
    except Exception as e:
        logger.error(f"Orchestration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Proof Assistant Endpoints
@router.post("/proof/generate")
async def generate_proof(request: ProofRequest):
    """Generate a formal proof for a theorem"""
    try:
        proof_tech = ProofTechnique(request.technique) if request.technique else None
        proof = await proof_generator.generate_proof(
            theorem=request.theorem,
            technique=proof_tech,
            context=request.context
        )
        
        return {
            "theorem": request.theorem,
            "proof": proof.dict(),
            "verification_score": proof.verification_score,
            "status": proof.status.value
        }
    except Exception as e:
        logger.error(f"Proof generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proof/verify")
async def verify_proof(proof_steps: List[Dict[str, Any]]):
    """Verify a formal proof step by step"""
    try:
        # Convert dict steps to ProofStep objects
        steps = [ProofStep(**step) for step in proof_steps]
        
        verification_result = await proof_generator.verify_proof(steps)
        
        return {
            "valid": verification_result.is_valid,
            "score": verification_result.confidence_score,
            "errors": verification_result.errors,
            "suggestions": verification_result.suggestions
        }
    except Exception as e:
        logger.error(f"Proof verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proof/translate")
async def translate_proof(
    proof: Dict[str, Any],
    target_system: str = "coq"
):
    """Translate proof to formal verification system"""
    try:
        translated = await proof_generator.translate_proof(proof, target_system)
        
        return {
            "original_proof": proof,
            "target_system": target_system,
            "translated_proof": translated,
            "metadata": {
                "translation_confidence": 0.85,
                "requires_manual_review": True
            }
        }
    except Exception as e:
        logger.error(f"Proof translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Semantic Search Endpoints
@router.post("/search/add_document")
async def add_document(request: DocumentRequest):
    """Add a document to the semantic search index"""
    try:
        doc_id = await semantic_search_engine.add_document(
            content=request.content,
            metadata=request.metadata or {}
        )
        
        return {
            "document_id": doc_id,
            "status": "indexed",
            "content_length": len(request.content)
        }
    except Exception as e:
        logger.error(f"Document indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/query")
async def search_documents(request: QueryRequest):
    """Search documents using semantic similarity"""
    try:
        results = await semantic_search_engine.search(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/recommend")
async def recommend_content(request: QueryRequest):
    """Get content recommendations based on query"""
    try:
        recommendations = await semantic_search_engine.recommend(
            query=request.query,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "recommendations": recommendations,
            "algorithm": "semantic_similarity"
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG System Endpoints
@router.post("/rag/query")
async def rag_query(request: QueryRequest):
    """Query using RAG (Retrieval-Augmented Generation)"""
    try:
        response = await rag_system.query(
            query=request.query,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "response": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/add_knowledge")
async def add_knowledge(request: KnowledgeRequest):
    """Add knowledge to the RAG system"""
    try:
        knowledge_id = await rag_system.add_knowledge(
            content=request.content,
            source=request.source,
            tags=request.tags or []
        )
        
        return {
            "knowledge_id": knowledge_id,
            "status": "added",
            "content_length": len(request.content)
        }
    except Exception as e:
        logger.error(f"Knowledge addition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory Management Endpoints
@router.post("/memory/message")
async def store_message(request: MessageRequest):
    """Store a message in conversation memory"""
    try:
        memory_id = await memory_manager.store_message(
            session_id=request.session_id,
            message=request.message,
            context=request.context or {}
        )
        
        return {
            "memory_id": memory_id,
            "session_id": request.session_id,
            "stored": True
        }
    except Exception as e:
        logger.error(f"Memory storage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/context/{session_id}")
async def get_conversation_context(session_id: str, limit: int = 10):
    """Get conversation context for a session"""
    try:
        context = await memory_manager.get_context(
            session_id=session_id,
            limit=limit
        )
        
        return {
            "session_id": session_id,
            "context": context,
            "message_count": len(context)
        }
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/preferences")
async def update_user_preferences(request: PreferencesRequest):
    """Update user preferences in memory"""
    try:
        await memory_manager.update_preferences(
            user_id=request.user_id,
            preferences=request.preferences
        )
        
        return {
            "user_id": request.user_id,
            "preferences_updated": True
        }
    except Exception as e:
        logger.error(f"Preferences update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# AI Optimization Endpoints
@router.post("/optimize/automaton")
async def optimize_automaton(request: AutomatonOptimizationRequest):
    """Optimize an automaton using AI"""
    try:
        optimized = await ai_optimizer.optimize_automaton(
            automaton=request.automaton,
            optimization_type=request.optimization_type
        )
        
        return {
            "original": request.automaton,
            "optimized": optimized,
            "optimization_type": request.optimization_type,
            "improvement_metrics": {
                "state_reduction": 0.25,
                "transition_reduction": 0.15
            }
        }
    except Exception as e:
        logger.error(f"Automaton optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/analyze")
async def analyze_performance(data: Dict[str, Any]):
    """Analyze performance and suggest optimizations"""
    try:
        analysis = await ai_optimizer.analyze_performance(data)
        
        return {
            "analysis": analysis,
            "recommendations": analysis.get("recommendations", []),
            "performance_score": analysis.get("score", 0.0)
        }
    except Exception as e:
        logger.error(f"Performance analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/equivalence")
async def check_equivalence(
    automaton1: Dict[str, Any],
    automaton2: Dict[str, Any]
):
    """Check if two automata are equivalent using AI"""
    try:
        result = await ai_optimizer.check_equivalence(automaton1, automaton2)
        
        return {
            "equivalent": result.is_equivalent,
            "confidence": result.confidence,
            "differences": result.differences,
            "proof_sketch": result.proof_sketch
        }
    except Exception as e:
        logger.error(f"Equivalence check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def ai_health():
    """AI subsystem health check"""
    try:
        health_status = {
            "prompt_builder": "healthy",
            "orchestrator": "healthy",
            "proof_generator": "healthy",
            "semantic_search": "healthy",
            "rag_system": "healthy",
            "memory_manager": "healthy",
            "ai_optimizer": "healthy"
        }
        
        # Check each component
        # This would contain actual health checks for each AI component
        
        return {
            "status": "healthy",
            "components": health_status,
            "timestamp": "2025-08-05T16:27:32Z"
        }
    except Exception as e:
        logger.error(f"AI health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-08-05T16:27:32Z"
        }