"""
AI JFLAP Configuration and Optimization Settings
================================================

Configuration for AI-enhanced JFLAP features including:
- Model selection strategies
- Caching policies
- Performance optimization
- Token management
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
from functools import lru_cache

# ============================
# MODEL SELECTION STRATEGY
# ============================

@dataclass
class ModelConfig:
    """Configuration for specific model usage"""
    name: str
    provider: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    timeout: int = 30
    retry_attempts: int = 3
    cost_per_token: float = 0.0
    capabilities: List[str] = field(default_factory=list)


class TaskType(Enum):
    """AI task types for JFLAP features"""
    TM_GENERATION = "tm_generation"
    TM_OPTIMIZATION = "tm_optimization"
    GRAMMAR_ANALYSIS = "grammar_analysis"
    GRAMMAR_CONVERSION = "grammar_conversion"
    ERROR_RECOVERY = "error_recovery"
    TEST_GENERATION = "test_generation"
    NL_TO_FORMAL = "nl_to_formal"
    FORMAL_TO_NL = "formal_to_nl"
    TUTORING = "tutoring"
    HINT_GENERATION = "hint_generation"
    PROOF_GENERATION = "proof_generation"


# Model configurations for different tasks
MODEL_CONFIGS = {
    # Code generation tasks - use CodeLlama
    "codellama:34b": ModelConfig(
        name="codellama:34b",
        temperature=0.3,
        max_tokens=4096,
        capabilities=["code_generation", "formal_specification", "algorithm_implementation"]
    ),
    
    # Reasoning and explanation - use DeepSeek
    "deepseek-coder:33b": ModelConfig(
        name="deepseek-coder:33b",
        temperature=0.5,
        max_tokens=3072,
        capabilities=["reasoning", "explanation", "optimization", "tutoring"]
    ),
    
    # General purpose - use Llama
    "llama3.1:8b": ModelConfig(
        name="llama3.1:8b",
        temperature=0.7,
        max_tokens=2048,
        capabilities=["general", "quick_response", "validation"]
    )
}


# Task to model mapping
TASK_MODEL_MAPPING = {
    TaskType.TM_GENERATION: "codellama:34b",
    TaskType.TM_OPTIMIZATION: "deepseek-coder:33b",
    TaskType.GRAMMAR_ANALYSIS: "deepseek-coder:33b",
    TaskType.GRAMMAR_CONVERSION: "codellama:34b",
    TaskType.ERROR_RECOVERY: "deepseek-coder:33b",
    TaskType.TEST_GENERATION: "codellama:34b",
    TaskType.NL_TO_FORMAL: "codellama:34b",
    TaskType.FORMAL_TO_NL: "llama3.1:8b",
    TaskType.TUTORING: "deepseek-coder:33b",
    TaskType.HINT_GENERATION: "llama3.1:8b",
    TaskType.PROOF_GENERATION: "deepseek-coder:33b"
}


# ============================
# PROMPT OPTIMIZATION SETTINGS
# ============================

@dataclass
class PromptOptimizationConfig:
    """Settings for prompt optimization"""
    max_prompt_length: int = 8192
    include_examples: bool = True
    max_examples: int = 3
    use_chain_of_thought: bool = True
    use_structured_output: bool = True
    compress_whitespace: bool = True
    remove_comments: bool = False


# Task-specific prompt settings
PROMPT_SETTINGS = {
    TaskType.TM_GENERATION: PromptOptimizationConfig(
        max_prompt_length=6144,
        include_examples=True,
        use_chain_of_thought=True
    ),
    TaskType.GRAMMAR_ANALYSIS: PromptOptimizationConfig(
        max_prompt_length=4096,
        use_structured_output=True
    ),
    TaskType.ERROR_RECOVERY: PromptOptimizationConfig(
        max_prompt_length=3072,
        include_examples=False,
        use_chain_of_thought=False
    ),
    TaskType.TEST_GENERATION: PromptOptimizationConfig(
        max_prompt_length=4096,
        include_examples=True,
        max_examples=5
    ),
    TaskType.TUTORING: PromptOptimizationConfig(
        max_prompt_length=8192,
        include_examples=True,
        use_chain_of_thought=True
    )
}


# ============================
# CACHING CONFIGURATION
# ============================

@dataclass
class CacheConfig:
    """Cache configuration for AI responses"""
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default
    max_size: int = 1000
    cache_type: str = "memory"  # memory, redis, disk
    compression: bool = True
    
    # Cache key strategies
    use_semantic_hashing: bool = True
    normalize_input: bool = True


# Task-specific cache settings
CACHE_SETTINGS = {
    TaskType.TM_GENERATION: CacheConfig(
        ttl_seconds=7200,  # 2 hours
        max_size=500
    ),
    TaskType.GRAMMAR_ANALYSIS: CacheConfig(
        ttl_seconds=3600,
        max_size=1000
    ),
    TaskType.ERROR_RECOVERY: CacheConfig(
        ttl_seconds=1800,  # 30 minutes
        max_size=2000
    ),
    TaskType.TEST_GENERATION: CacheConfig(
        ttl_seconds=3600,
        max_size=500
    ),
    TaskType.NL_TO_FORMAL: CacheConfig(
        ttl_seconds=7200,
        max_size=1000
    ),
    TaskType.TUTORING: CacheConfig(
        ttl_seconds=86400,  # 24 hours
        max_size=200
    ),
    TaskType.HINT_GENERATION: CacheConfig(
        enabled=False  # Don't cache hints - they should be contextual
    )
}


# ============================
# PERFORMANCE OPTIMIZATION
# ============================

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    # Parallel execution
    enable_parallel: bool = True
    max_parallel_requests: int = 4
    
    # Streaming
    enable_streaming: bool = True
    stream_chunk_size: int = 512
    
    # Timeouts
    global_timeout: int = 60
    connection_timeout: int = 10
    read_timeout: int = 50
    
    # Retries
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Resource limits
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    
    # Batching
    enable_batching: bool = True
    batch_size: int = 10
    batch_wait_ms: int = 100


# ============================
# FALLBACK STRATEGIES
# ============================

@dataclass
class FallbackConfig:
    """Fallback configuration for failures"""
    enable_fallback: bool = True
    fallback_models: List[str] = field(default_factory=list)
    use_cached_similar: bool = True
    similarity_threshold: float = 0.85
    provide_partial_results: bool = True
    
    # Degraded mode settings
    degraded_mode_threshold: int = 5  # failures before degraded mode
    degraded_mode_duration: int = 300  # 5 minutes


# Task-specific fallback configurations
FALLBACK_SETTINGS = {
    TaskType.TM_GENERATION: FallbackConfig(
        fallback_models=["llama3.1:8b"],
        provide_partial_results=True
    ),
    TaskType.ERROR_RECOVERY: FallbackConfig(
        use_cached_similar=True,
        similarity_threshold=0.9
    ),
    TaskType.TUTORING: FallbackConfig(
        fallback_models=["llama3.1:8b"],
        provide_partial_results=False  # Don't provide partial tutorials
    )
}


# ============================
# MONITORING & METRICS
# ============================

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # What to track
    track_latency: bool = True
    track_token_usage: bool = True
    track_errors: bool = True
    track_cache_hits: bool = True
    
    # Alerting
    alert_on_high_latency: bool = True
    latency_threshold_ms: int = 5000
    alert_on_error_rate: bool = True
    error_rate_threshold: float = 0.1
    
    # Logging
    log_requests: bool = True
    log_responses: bool = False  # Can be verbose
    log_errors: bool = True
    
    # Sampling
    sampling_rate: float = 1.0  # Log all requests by default


# ============================
# EDUCATIONAL SETTINGS
# ============================

@dataclass
class EducationalConfig:
    """Settings for educational features"""
    # Difficulty adaptation
    adapt_difficulty: bool = True
    initial_difficulty: str = "intermediate"
    
    # Explanation depth
    default_explanation_level: str = "medium"
    include_visuals: bool = True
    include_examples: bool = True
    
    # Progress tracking
    track_progress: bool = True
    store_attempts: bool = True
    max_stored_attempts: int = 100
    
    # Feedback
    provide_immediate_feedback: bool = True
    explain_errors: bool = True
    suggest_improvements: bool = True


# ============================
# GLOBAL CONFIGURATION
# ============================

class AIJFLAPConfig:
    """Global configuration for AI JFLAP features"""
    
    def __init__(self):
        # Base configurations
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.enable_ai = os.getenv("ENABLE_AI_FEATURES", "true").lower() == "true"
        
        # Model configs
        self.model_configs = MODEL_CONFIGS
        self.task_model_mapping = TASK_MODEL_MAPPING
        
        # Optimization settings
        self.prompt_settings = PROMPT_SETTINGS
        self.cache_settings = CACHE_SETTINGS
        self.fallback_settings = FALLBACK_SETTINGS
        
        # Performance
        self.performance = PerformanceConfig()
        
        # Monitoring
        self.monitoring = MonitoringConfig()
        
        # Educational
        self.educational = EducationalConfig()
        
        # Token limits
        self.max_total_tokens = 8192
        self.reserve_tokens = 500  # Reserve for response
        
        # Cost management
        self.enable_cost_tracking = True
        self.max_cost_per_request = 0.10  # $0.10 max per request
        self.daily_cost_limit = 10.00  # $10 daily limit
    
    @lru_cache(maxsize=128)
    def get_model_for_task(self, task: TaskType) -> ModelConfig:
        """Get optimal model configuration for task"""
        model_name = self.task_model_mapping.get(task, "llama3.1:8b")
        return self.model_configs.get(model_name, self.model_configs["llama3.1:8b"])
    
    def get_prompt_settings(self, task: TaskType) -> PromptOptimizationConfig:
        """Get prompt optimization settings for task"""
        return self.prompt_settings.get(
            task,
            PromptOptimizationConfig()  # Default settings
        )
    
    def get_cache_config(self, task: TaskType) -> CacheConfig:
        """Get cache configuration for task"""
        return self.cache_settings.get(
            task,
            CacheConfig()  # Default cache settings
        )
    
    def get_fallback_config(self, task: TaskType) -> FallbackConfig:
        """Get fallback configuration for task"""
        return self.fallback_settings.get(
            task,
            FallbackConfig()  # Default fallback
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def should_use_cache(self, task: TaskType) -> bool:
        """Determine if caching should be used for task"""
        cache_config = self.get_cache_config(task)
        return cache_config.enabled and self.performance.enable_parallel
    
    def get_timeout_for_task(self, task: TaskType) -> int:
        """Get appropriate timeout for task"""
        model_config = self.get_model_for_task(task)
        return min(model_config.timeout, self.performance.global_timeout)


# ============================
# OPTIMIZATION STRATEGIES
# ============================

class OptimizationStrategy:
    """Token and performance optimization strategies"""
    
    @staticmethod
    def compress_prompt(prompt: str, max_length: int) -> str:
        """Compress prompt to fit token limits"""
        if len(prompt) <= max_length:
            return prompt
        
        # Priority sections to keep
        priority_markers = ["Task:", "Problem:", "Required:", "Output:"]
        
        lines = prompt.split('\n')
        priority_lines = []
        other_lines = []
        
        for line in lines:
            if any(marker in line for marker in priority_markers):
                priority_lines.append(line)
            else:
                other_lines.append(line)
        
        # Build compressed prompt
        compressed = '\n'.join(priority_lines)
        
        # Add other lines until limit
        for line in other_lines:
            if len(compressed) + len(line) + 1 < max_length:
                compressed += '\n' + line
            else:
                break
        
        return compressed
    
    @staticmethod
    def batch_similar_requests(
        requests: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> List[List[Dict[str, Any]]]:
        """Group similar requests for batch processing"""
        # Simple batching by task type
        batches = {}
        for request in requests:
            task = request.get("task", "unknown")
            if task not in batches:
                batches[task] = []
            batches[task].append(request)
        
        return list(batches.values())
    
    @staticmethod
    def select_execution_mode(
        task: TaskType,
        priority: str = "balanced"
    ) -> str:
        """Select optimal execution mode for task"""
        if priority == "speed":
            return "parallel"
        elif priority == "accuracy":
            return "ensemble"
        elif priority == "cost":
            return "sequential"
        else:  # balanced
            # Task-specific selection
            if task in [TaskType.TM_GENERATION, TaskType.GRAMMAR_CONVERSION]:
                return "cascade"  # Best quality with fallback
            elif task in [TaskType.ERROR_RECOVERY, TaskType.TEST_GENERATION]:
                return "ensemble"  # Multiple perspectives
            else:
                return "sequential"  # Simple and efficient


# ============================
# SINGLETON INSTANCE
# ============================

# Global configuration instance
ai_jflap_config = AIJFLAPConfig()

# Export commonly used functions
def get_model_for_task(task: TaskType) -> ModelConfig:
    """Get model configuration for task"""
    return ai_jflap_config.get_model_for_task(task)

def get_cache_config(task: TaskType) -> CacheConfig:
    """Get cache configuration for task"""
    return ai_jflap_config.get_cache_config(task)

def optimize_prompt(prompt: str, task: TaskType) -> str:
    """Optimize prompt for task"""
    settings = ai_jflap_config.get_prompt_settings(task)
    if settings.compress_whitespace:
        prompt = ' '.join(prompt.split())
    if len(prompt) > settings.max_prompt_length:
        prompt = OptimizationStrategy.compress_prompt(prompt, settings.max_prompt_length)
    return prompt