"""
Multi-Model Orchestration System for the Automata Learning Platform.
Manages model routing, parallel execution, and response fusion.
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import ollama
from langchain_community.llms import Ollama as LangchainOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from cachetools import TTLCache

from .ai_config import (
    get_ai_config, ModelType, TaskComplexity,
    get_model_for_task, estimate_task_complexity
)
from .prompts import prompt_builder, PromptOptimizer

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Execution modes for model orchestration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ENSEMBLE = "ensemble"
    CASCADE = "cascade"
    FALLBACK = "fallback"


class ModelRequest(BaseModel):
    """Request structure for model execution."""
    task: str
    prompt: str
    model_type: Optional[ModelType] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Response structure from model execution."""
    task: str
    model_name: str
    response: Any
    execution_time: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ModelRouter:
    """Routes tasks to appropriate models based on complexity and requirements."""
    
    def __init__(self):
        self.config = get_ai_config()
        self.routing_cache = TTLCache(maxsize=100, ttl=3600)
    
    def route(
        self,
        task: str,
        complexity: Optional[TaskComplexity] = None,
        requirements: Optional[List[str]] = None
    ) -> List[ModelType]:
        """
        Route a task to appropriate models.
        
        Args:
            task: Task identifier
            complexity: Task complexity level
            requirements: Specific requirements (e.g., "fast", "accurate")
        
        Returns:
            List of ModelType in order of preference
        """
        cache_key = f"{task}:{complexity}:{requirements}"
        if cache_key in self.routing_cache:
            return self.routing_cache[cache_key]
        
        models = []
        
        # Primary routing based on task
        if task in self.config.task_routing:
            models.append(self.config.task_routing[task])
        
        # Add complexity-based models
        if complexity:
            complexity_models = self.config.complexity_routing.get(complexity, [])
            for model in complexity_models:
                if model not in models:
                    models.append(model)
        
        # Handle specific requirements
        if requirements:
            if "fast" in requirements and ModelType.GENERAL not in models:
                models.insert(0, ModelType.GENERAL)
            if "accurate" in requirements and ModelType.PROOF not in models:
                models.append(ModelType.PROOF)
            if "creative" in requirements and ModelType.GENERATOR not in models:
                models.append(ModelType.GENERATOR)
        
        # Default fallback
        if not models:
            models = [ModelType.GENERAL]
        
        self.routing_cache[cache_key] = models
        return models


class ModelExecutor:
    """Executes requests on specific models with retry logic."""
    
    def __init__(self):
        self.config = get_ai_config()
        self.client = ollama.Client(host=self.config.ollama_base_url)
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def execute_ollama(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a prompt on an Ollama model.
        
        Args:
            model_name: Name of the Ollama model
            prompt: Prompt to execute
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        
        Returns:
            Model response dictionary
        """
        try:
            response = self.client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.95,
                    'seed': 42
                },
                stream=stream
            )
            
            if stream:
                return {"response": response, "streaming": True}
            else:
                return {
                    "response": response['response'],
                    "tokens": response.get('eval_count', 0),
                    "model": model_name
                }
        except Exception as e:
            logger.error(f"Ollama execution error: {e}")
            raise
    
    def execute_langchain(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Execute using LangChain for structured output."""
        try:
            llm = LangchainOllama(
                model=model_name,
                base_url=self.config.ollama_base_url,
                temperature=temperature,
                num_predict=max_tokens
            )
            
            # Use structured output if JSON format expected
            if "json" in prompt.lower() or "format" in prompt.lower():
                parser = JsonOutputParser()
                chain = llm | parser
                response = chain.invoke(prompt)
            else:
                response = llm.invoke(prompt)
            
            return {
                "response": response,
                "model": model_name,
                "structured": isinstance(response, dict)
            }
        except Exception as e:
            logger.error(f"LangChain execution error: {e}")
            raise
    
    async def execute_async(
        self,
        request: ModelRequest
    ) -> ModelResponse:
        """Execute a model request asynchronously."""
        start_time = time.time()
        
        try:
            # Get model configuration
            model_config = get_model_for_task(
                request.task,
                estimate_task_complexity(request.task)
            )
            
            # Optimize prompt for model
            optimized_prompt = PromptOptimizer.optimize_for_model(
                request.prompt,
                model_config.name,
                request.max_tokens or model_config.max_tokens
            )
            
            # Execute based on provider
            if model_config.provider == "ollama":
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.execute_ollama,
                    model_config.name,
                    optimized_prompt,
                    request.temperature or model_config.temperature,
                    request.max_tokens or model_config.max_tokens
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.execute_langchain,
                    model_config.name,
                    optimized_prompt,
                    request.temperature or model_config.temperature,
                    request.max_tokens or model_config.max_tokens
                )
            
            execution_time = time.time() - start_time
            
            return ModelResponse(
                task=request.task,
                model_name=model_config.name,
                response=result.get("response"),
                execution_time=execution_time,
                tokens_used=result.get("tokens"),
                cost=self._calculate_cost(
                    result.get("tokens", 0),
                    model_config.cost_per_token
                ),
                metadata={
                    **request.metadata,
                    "optimized": True,
                    "structured": result.get("structured", False)
                }
            )
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            return ModelResponse(
                task=request.task,
                model_name="unknown",
                response=None,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _calculate_cost(self, tokens: int, cost_per_token: float) -> float:
        """Calculate execution cost based on tokens."""
        return tokens * cost_per_token if cost_per_token > 0 else 0.0


class ResponseFusion:
    """Fuses responses from multiple models."""
    
    @staticmethod
    def majority_vote(responses: List[ModelResponse]) -> Any:
        """Perform majority voting on responses."""
        valid_responses = [r.response for r in responses if not r.error]
        if not valid_responses:
            return None
        
        # For structured responses, convert to string for comparison
        response_counts = {}
        for response in valid_responses:
            key = json.dumps(response) if isinstance(response, dict) else str(response)
            response_counts[key] = response_counts.get(key, 0) + 1
        
        # Return most common response
        best_response = max(response_counts, key=response_counts.get)
        try:
            return json.loads(best_response)
        except:
            return best_response
    
    @staticmethod
    def weighted_average(
        responses: List[ModelResponse],
        weights: Optional[List[float]] = None
    ) -> Any:
        """Compute weighted average of numerical responses."""
        valid_responses = [r for r in responses if not r.error]
        if not valid_responses:
            return None
        
        if weights is None:
            # Use execution time as inverse weight (faster = higher weight)
            times = [r.execution_time for r in valid_responses]
            weights = [1.0 / t for t in times]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # For numerical responses
        try:
            values = [float(r.response) for r in valid_responses]
            return sum(v * w for v, w in zip(values, weights))
        except:
            # Fall back to majority vote for non-numerical
            return ResponseFusion.majority_vote(responses)
    
    @staticmethod
    def best_of_n(
        responses: List[ModelResponse],
        scorer: Optional[Callable] = None
    ) -> ModelResponse:
        """Select best response based on scoring function."""
        valid_responses = [r for r in responses if not r.error]
        if not valid_responses:
            return responses[0] if responses else None
        
        if scorer is None:
            # Default scorer: prefer structured, fast, low-cost responses
            def default_scorer(r: ModelResponse) -> float:
                score = 100.0
                score -= r.execution_time * 10  # Penalize slow responses
                score -= (r.cost or 0) * 100    # Penalize expensive responses
                if r.metadata.get("structured"):
                    score += 20  # Bonus for structured output
                return score
            scorer = default_scorer
        
        return max(valid_responses, key=scorer)
    
    @staticmethod
    def merge_responses(
        responses: List[ModelResponse],
        strategy: str = "combine"
    ) -> Dict[str, Any]:
        """
        Merge multiple responses into a single comprehensive response.
        
        Args:
            responses: List of model responses
            strategy: Merging strategy (combine, intersect, union)
        
        Returns:
            Merged response dictionary
        """
        valid_responses = [r for r in responses if not r.error]
        if not valid_responses:
            return {"error": "No valid responses"}
        
        merged = {
            "models_used": [r.model_name for r in valid_responses],
            "total_time": sum(r.execution_time for r in valid_responses),
            "total_cost": sum(r.cost or 0 for r in valid_responses)
        }
        
        if strategy == "combine":
            # Combine all unique information
            combined = {}
            for r in valid_responses:
                if isinstance(r.response, dict):
                    combined.update(r.response)
                else:
                    combined[r.model_name] = r.response
            merged["response"] = combined
        
        elif strategy == "intersect":
            # Only keep information present in all responses
            if all(isinstance(r.response, dict) for r in valid_responses):
                keys = set(valid_responses[0].response.keys())
                for r in valid_responses[1:]:
                    keys &= set(r.response.keys())
                merged["response"] = {
                    k: valid_responses[0].response[k] for k in keys
                }
            else:
                merged["response"] = ResponseFusion.majority_vote(valid_responses)
        
        else:  # union
            # Collect all information
            merged["response"] = [r.response for r in valid_responses]
        
        return merged


class ModelOrchestrator:
    """
    Main orchestration system for coordinating multiple models.
    """
    
    def __init__(self):
        self.router = ModelRouter()
        self.executor = ModelExecutor()
        self.config = get_ai_config()
        self.response_cache = TTLCache(
            maxsize=100,
            ttl=self.config.cache_ttl if self.config.enable_caching else 1
        )
    
    async def execute(
        self,
        task: str,
        prompt: str,
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        **kwargs
    ) -> Union[ModelResponse, List[ModelResponse], Dict[str, Any]]:
        """
        Execute a task using the specified orchestration mode.
        
        Args:
            task: Task identifier
            prompt: Prompt to execute
            mode: Execution mode
            **kwargs: Additional parameters
        
        Returns:
            Response based on execution mode
        """
        # Check cache
        cache_key = f"{task}:{mode}:{hash(prompt)}"
        if self.config.enable_caching and cache_key in self.response_cache:
            logger.info(f"Cache hit for task: {task}")
            return self.response_cache[cache_key]
        
        # Route to appropriate models
        complexity = estimate_task_complexity(prompt)
        model_types = self.router.route(task, complexity)
        
        # Create requests for each model
        requests = [
            ModelRequest(
                task=task,
                prompt=prompt,
                model_type=model_type,
                **kwargs
            )
            for model_type in model_types
        ]
        
        # Execute based on mode
        if mode == ExecutionMode.SEQUENTIAL:
            result = await self._execute_sequential(requests)
        elif mode == ExecutionMode.PARALLEL:
            result = await self._execute_parallel(requests)
        elif mode == ExecutionMode.ENSEMBLE:
            result = await self._execute_ensemble(requests)
        elif mode == ExecutionMode.CASCADE:
            result = await self._execute_cascade(requests)
        elif mode == ExecutionMode.FALLBACK:
            result = await self._execute_fallback(requests)
        else:
            raise ValueError(f"Unknown execution mode: {mode}")
        
        # Cache result
        if self.config.enable_caching:
            self.response_cache[cache_key] = result
        
        return result
    
    async def _execute_sequential(
        self,
        requests: List[ModelRequest]
    ) -> List[ModelResponse]:
        """Execute requests sequentially."""
        responses = []
        for request in requests:
            response = await self.executor.execute_async(request)
            responses.append(response)
            
            # Use response in next request if chaining
            if not response.error and len(requests) > 1:
                # Inject previous response into next prompt
                next_idx = requests.index(request) + 1
                if next_idx < len(requests):
                    requests[next_idx].prompt += f"\n\nPrevious analysis:\n{response.response}"
        
        return responses
    
    async def _execute_parallel(
        self,
        requests: List[ModelRequest]
    ) -> List[ModelResponse]:
        """Execute requests in parallel."""
        tasks = [
            self.executor.execute_async(request)
            for request in requests
        ]
        responses = await asyncio.gather(*tasks)
        return responses
    
    async def _execute_ensemble(
        self,
        requests: List[ModelRequest]
    ) -> Dict[str, Any]:
        """Execute ensemble with response fusion."""
        # Execute all models in parallel
        responses = await self._execute_parallel(requests)
        
        # Fuse responses
        fusion_result = ResponseFusion.merge_responses(
            responses,
            strategy="combine"
        )
        
        # Add ensemble metadata
        fusion_result["ensemble"] = {
            "models": len(responses),
            "consensus": ResponseFusion.majority_vote(responses),
            "best": ResponseFusion.best_of_n(responses).response
        }
        
        return fusion_result
    
    async def _execute_cascade(
        self,
        requests: List[ModelRequest]
    ) -> ModelResponse:
        """Execute cascade - stop when good response found."""
        for request in requests:
            response = await self.executor.execute_async(request)
            
            # Check if response is satisfactory
            if not response.error and self._is_satisfactory(response):
                logger.info(f"Cascade stopped at model: {response.model_name}")
                return response
            
            # Add context for next model
            if not response.error and len(requests) > 1:
                next_idx = requests.index(request) + 1
                if next_idx < len(requests):
                    requests[next_idx].prompt += f"\n\nImprove upon:\n{response.response}"
        
        # Return last response if none satisfactory
        return response
    
    async def _execute_fallback(
        self,
        requests: List[ModelRequest]
    ) -> ModelResponse:
        """Execute with fallback - try next model on failure."""
        for request in requests:
            response = await self.executor.execute_async(request)
            
            if not response.error:
                return response
            
            logger.warning(f"Model {response.model_name} failed, trying fallback")
        
        # All models failed
        return ModelResponse(
            task=requests[0].task,
            model_name="all",
            response=None,
            execution_time=0,
            error="All models failed"
        )
    
    def _is_satisfactory(self, response: ModelResponse) -> bool:
        """Check if a response meets quality criteria."""
        if response.error:
            return False
        
        # Check response content
        if not response.response:
            return False
        
        # Check if structured response is valid
        if response.metadata.get("structured"):
            if isinstance(response.response, dict) and response.response:
                return True
        
        # Check minimum length for text responses
        if isinstance(response.response, str):
            return len(response.response) > 50
        
        return True


# Global orchestrator instance
orchestrator = ModelOrchestrator()


async def orchestrate_task(
    task: str,
    prompt: str,
    mode: str = "sequential",
    **kwargs
) -> Any:
    """
    Convenience function for task orchestration.
    
    Args:
        task: Task identifier
        prompt: Prompt to execute
        mode: Execution mode
        **kwargs: Additional parameters
    
    Returns:
        Orchestrated response
    """
    execution_mode = ExecutionMode(mode.lower())
    return await orchestrator.execute(task, prompt, execution_mode, **kwargs)