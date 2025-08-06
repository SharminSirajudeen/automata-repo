"""
Centralized AI configuration for the Automata Learning Platform.
Manages model selection, embeddings, and AI service configurations.
"""
import os
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Enumeration of available model types."""
    GENERATOR = "generator"
    EXPLAINER = "explainer"
    PROOF = "proof"
    OPTIMIZER = "optimizer"
    EMBEDDER = "embedder"
    VISION = "vision"
    GENERAL = "general"


class TaskComplexity(str, Enum):
    """Task complexity levels for model routing."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RESEARCH = "research"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str
    provider: str = "ollama"
    endpoint: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    context_window: int = 8192
    cost_per_token: float = 0.0
    capabilities: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""
    max_examples: int = 5
    chain_of_thought: bool = True
    system_prompt: Optional[str] = None
    output_format: str = "structured"
    include_reasoning: bool = True


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    provider: str = "chromadb"
    collection_name: str = "automata_knowledge"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    persist_directory: str = "./chroma_db"
    similarity_threshold: float = 0.7
    top_k: int = 5


class RAGConfig(BaseModel):
    """Configuration for RAG system."""
    retrieval_mode: str = "hybrid"  # semantic, keyword, hybrid
    rerank: bool = True
    context_window: int = 4096
    source_attribution: bool = True
    query_expansion: bool = True
    max_sources: int = 5
    min_relevance_score: float = 0.6


class MemoryConfig(BaseModel):
    """Configuration for conversation memory."""
    provider: str = "redis"
    ttl: int = 3600  # Time to live in seconds
    max_messages: int = 50
    summarization_threshold: int = 20
    enable_long_term: bool = True
    user_preference_tracking: bool = True


class AISystemConfig(BaseModel):
    """Main AI system configuration."""
    # Model configurations for different tasks
    models: Dict[ModelType, ModelConfig] = {
        ModelType.GENERATOR: ModelConfig(
            name="codellama:34b",
            provider="ollama",
            temperature=0.8,
            max_tokens=4096,
            capabilities=["code_generation", "automata_creation", "algorithm_design"]
        ),
        ModelType.EXPLAINER: ModelConfig(
            name="deepseek-coder:33b",
            provider="ollama",
            temperature=0.5,
            max_tokens=2048,
            capabilities=["explanation", "documentation", "teaching"]
        ),
        ModelType.PROOF: ModelConfig(
            name="llama3.1:8b",
            provider="ollama",
            temperature=0.3,
            max_tokens=3072,
            capabilities=["formal_proofs", "logical_reasoning", "verification"]
        ),
        ModelType.OPTIMIZER: ModelConfig(
            name="codellama:34b",
            provider="ollama",
            temperature=0.4,
            max_tokens=2048,
            capabilities=["optimization", "minimization", "performance_analysis"]
        ),
        ModelType.EMBEDDER: ModelConfig(
            name="nomic-embed-text",
            provider="ollama",
            temperature=0.0,
            max_tokens=512,
            capabilities=["embedding", "similarity"]
        ),
        ModelType.VISION: ModelConfig(
            name="llava:34b",
            provider="ollama",
            temperature=0.5,
            max_tokens=2048,
            capabilities=["image_analysis", "diagram_understanding"]
        ),
        ModelType.GENERAL: ModelConfig(
            name="llama3.1:8b",
            provider="ollama",
            temperature=0.7,
            max_tokens=2048,
            capabilities=["general_tasks", "conversation", "qa"]
        )
    }
    
    # Task routing configuration
    task_routing: Dict[str, ModelType] = {
        "generate_dfa": ModelType.GENERATOR,
        "generate_nfa": ModelType.GENERATOR,
        "generate_pda": ModelType.GENERATOR,
        "generate_tm": ModelType.GENERATOR,
        "explain_concept": ModelType.EXPLAINER,
        "explain_solution": ModelType.EXPLAINER,
        "prove_theorem": ModelType.PROOF,
        "verify_proof": ModelType.PROOF,
        "optimize_automaton": ModelType.OPTIMIZER,
        "minimize_dfa": ModelType.OPTIMIZER,
        "embed_document": ModelType.EMBEDDER,
        "analyze_diagram": ModelType.VISION,
        "general_query": ModelType.GENERAL
    }
    
    # Complexity-based model selection
    complexity_routing: Dict[TaskComplexity, List[ModelType]] = {
        TaskComplexity.SIMPLE: [ModelType.GENERAL],
        TaskComplexity.MODERATE: [ModelType.EXPLAINER, ModelType.GENERAL],
        TaskComplexity.COMPLEX: [ModelType.GENERATOR, ModelType.PROOF],
        TaskComplexity.RESEARCH: [ModelType.GENERATOR, ModelType.PROOF, ModelType.OPTIMIZER]
    }
    
    # Prompt configuration
    prompt_config: PromptConfig = PromptConfig()
    
    # Vector database configuration
    vector_db: VectorDBConfig = VectorDBConfig()
    
    # RAG configuration
    rag: RAGConfig = RAGConfig()
    
    # Memory configuration
    memory: MemoryConfig = MemoryConfig()
    
    # Ollama configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_streaming: bool = True
    batch_size: int = 10
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Monitoring and logging
    enable_monitoring: bool = True
    log_prompts: bool = True
    log_responses: bool = True
    metrics_enabled: bool = True
    
    # Cost optimization
    enable_cost_optimization: bool = True
    max_cost_per_request: float = 0.1
    fallback_on_error: bool = True
    
    # Security settings
    enable_content_filtering: bool = True
    max_input_length: int = 10000
    rate_limit_per_user: int = 100
    
    class Config:
        use_enum_values = True


@lru_cache()
def get_ai_config() -> AISystemConfig:
    """Get cached AI configuration instance."""
    config = AISystemConfig()
    logger.info("AI configuration loaded successfully")
    return config


def get_model_for_task(task: str, complexity: Optional[TaskComplexity] = None) -> ModelConfig:
    """
    Get the appropriate model configuration for a specific task.
    
    Args:
        task: The task identifier
        complexity: Optional complexity level for advanced routing
    
    Returns:
        ModelConfig for the selected model
    """
    config = get_ai_config()
    
    # First try task-specific routing
    if task in config.task_routing:
        model_type = config.task_routing[task]
        return config.models[model_type]
    
    # Fall back to complexity-based routing
    if complexity and complexity in config.complexity_routing:
        model_types = config.complexity_routing[complexity]
        if model_types:
            return config.models[model_types[0]]
    
    # Default to general model
    return config.models[ModelType.GENERAL]


def estimate_task_complexity(task_description: str, input_size: int = 0) -> TaskComplexity:
    """
    Estimate the complexity of a task based on description and input size.
    
    Args:
        task_description: Description of the task
        input_size: Size of the input data
    
    Returns:
        Estimated TaskComplexity level
    """
    # Simple heuristic-based complexity estimation
    complex_keywords = ["prove", "optimize", "minimize", "research", "advanced", "formal"]
    moderate_keywords = ["explain", "analyze", "compare", "evaluate", "design"]
    
    task_lower = task_description.lower()
    
    if any(keyword in task_lower for keyword in complex_keywords):
        return TaskComplexity.COMPLEX if input_size < 1000 else TaskComplexity.RESEARCH
    elif any(keyword in task_lower for keyword in moderate_keywords):
        return TaskComplexity.MODERATE
    else:
        return TaskComplexity.SIMPLE


# Export main configuration
ai_config = get_ai_config()