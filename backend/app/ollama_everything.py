"""
OLLAMA EVERYTHING - The Ultimate Ollama Integration
=================================================

This module pushes Ollama to its absolute limits by using it for EVERY possible task:
- ALL text processing
- Code generation and analysis
- Mathematical computations
- Pattern recognition  
- Data validation
- Error message generation
- Documentation generation
- Test case generation
- Query optimization
- Configuration suggestions
- Decision making
- Content transformation
- Semantic understanding
- Natural language processing
- Logical reasoning

EVERYTHING that involves text, reasoning, or intelligence uses Ollama.
"""

import asyncio
import json
import logging
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import settings
from .ai_config import get_ai_config, ModelType, TaskComplexity
from .ollama_cost_tracker import cost_tracker, track_ollama_request
from .valkey_integration import valkey_connection_manager

logger = logging.getLogger(__name__)


class OllamaTaskType(str, Enum):
    """Every possible task type that Ollama can handle."""
    # Text Processing
    TEXT_ANALYSIS = "text_analysis"
    TEXT_GENERATION = "text_generation"
    TEXT_TRANSFORMATION = "text_transformation"
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_EXTRACTION = "text_extraction"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_VALIDATION = "text_validation"
    TEXT_ENHANCEMENT = "text_enhancement"
    
    # Code Tasks
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_REVIEW = "code_review"
    CODE_OPTIMIZATION = "code_optimization"
    CODE_DOCUMENTATION = "code_documentation"
    CODE_TESTING = "code_testing"
    CODE_DEBUGGING = "code_debugging"
    CODE_REFACTORING = "code_refactoring"
    
    # Mathematical & Logical
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    LOGICAL_REASONING = "logical_reasoning"
    PATTERN_RECOGNITION = "pattern_recognition"
    DATA_ANALYSIS = "data_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ALGORITHMIC_THINKING = "algorithmic_thinking"
    
    # Security & Validation
    SECURITY_ANALYSIS = "security_analysis"
    THREAT_DETECTION = "threat_detection"
    INPUT_VALIDATION = "input_validation"
    CONTENT_FILTERING = "content_filtering"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    
    # System & DevOps
    SYSTEM_ANALYSIS = "system_analysis"
    CONFIGURATION_GENERATION = "configuration_generation"
    DEPLOYMENT_PLANNING = "deployment_planning"
    MONITORING_INTERPRETATION = "monitoring_interpretation"
    LOG_ANALYSIS = "log_analysis"
    ERROR_DIAGNOSIS = "error_diagnosis"
    
    # Educational & Learning
    EXPLANATION_GENERATION = "explanation_generation"
    TUTORIAL_CREATION = "tutorial_creation"
    EXERCISE_GENERATION = "exercise_generation"
    ASSESSMENT_CREATION = "assessment_creation"
    HINT_GENERATION = "hint_generation"
    FEEDBACK_GENERATION = "feedback_generation"
    
    # Database & Query
    SQL_GENERATION = "sql_generation"
    QUERY_OPTIMIZATION = "query_optimization"
    SCHEMA_DESIGN = "schema_design"
    DATA_MIGRATION = "data_migration"
    
    # Communication & Language
    LANGUAGE_TRANSLATION = "language_translation"
    TONE_ADJUSTMENT = "tone_adjustment"
    STYLE_ADAPTATION = "style_adaptation"
    GRAMMAR_CORRECTION = "grammar_correction"
    SEMANTIC_UNDERSTANDING = "semantic_understanding"
    
    # Decision Making & Strategy
    DECISION_ANALYSIS = "decision_analysis"
    STRATEGY_FORMULATION = "strategy_formulation"
    RECOMMENDATION_ENGINE = "recommendation_engine"
    PRIORITY_RANKING = "priority_ranking"
    OPTION_EVALUATION = "option_evaluation"
    
    # Creative & Content
    CONTENT_CREATION = "content_creation"
    CREATIVE_WRITING = "creative_writing"
    MARKETING_COPY = "marketing_copy"
    TECHNICAL_WRITING = "technical_writing"
    
    # Specialized Tasks
    AUTOMATA_DESIGN = "automata_design"
    FORMAL_VERIFICATION = "formal_verification"
    PROOF_GENERATION = "proof_generation"
    THEOREM_PROVING = "theorem_proving"
    ALGORITHM_DESIGN = "algorithm_design"


@dataclass
class OllamaTask:
    """Represents a task to be processed by Ollama."""
    task_type: OllamaTaskType
    input_data: Any
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher is more important
    model_preference: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    streaming: bool = False
    cache_enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OllamaResult:
    """Result from an Ollama task."""
    task_id: str
    task_type: OllamaTaskType
    result: Any
    confidence_score: float
    processing_time: float
    model_used: str
    tokens_used: int
    cached: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OllamaEverything:
    """The ultimate Ollama integration that uses Ollama for EVERYTHING."""
    
    def __init__(self):
        self.config = get_ai_config()
        self.ollama_base_url = self.config.ollama_base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Task queue and processing
        self.task_queue: deque = deque()
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, OllamaResult] = {}
        
        # Performance optimization
        self.task_executor = ThreadPoolExecutor(max_workers=8)
        self.model_cache: Dict[str, Dict] = {}
        self.prompt_templates: Dict[OllamaTaskType, str] = {}
        
        # Monitoring and analytics
        self.task_stats: Dict[str, Any] = defaultdict(int)
        self.performance_metrics: Dict[str, List] = defaultdict(list)
        
        # Initialize prompt templates
        self._initialize_prompt_templates()
        
        logger.info("OllamaEverything initialized - ready to use Ollama for EVERYTHING!")
    
    def _initialize_prompt_templates(self):
        """Initialize comprehensive prompt templates for every task type."""
        
        # Text Processing Templates
        self.prompt_templates[OllamaTaskType.TEXT_ANALYSIS] = """
        Analyze the following text comprehensively:
        
        Text: {input_data}
        
        Provide analysis including:
        - Sentiment and tone
        - Key themes and topics
        - Writing quality and clarity
        - Potential improvements
        - Statistical insights
        
        Context: {context}
        
        Analysis:"""
        
        self.prompt_templates[OllamaTaskType.TEXT_GENERATION] = """
        Generate high-quality text based on the following requirements:
        
        Requirements: {input_data}
        Style: {style}
        Length: {length}
        Tone: {tone}
        
        Context: {context}
        
        Generated Text:"""
        
        self.prompt_templates[OllamaTaskType.CODE_GENERATION] = """
        Generate production-ready code with the following specifications:
        
        Requirements: {input_data}
        Language: {language}
        Framework: {framework}
        Best Practices: Include error handling, documentation, type hints, and tests
        
        Context: {context}
        
        Generated Code:"""
        
        self.prompt_templates[OllamaTaskType.CODE_ANALYSIS] = """
        Perform comprehensive code analysis:
        
        Code: {input_data}
        
        Analyze for:
        - Code quality and style
        - Performance implications
        - Security vulnerabilities
        - Best practices adherence
        - Optimization opportunities
        - Maintainability issues
        
        Context: {context}
        
        Analysis:"""
        
        self.prompt_templates[OllamaTaskType.SECURITY_ANALYSIS] = """
        Perform thorough security analysis:
        
        Input: {input_data}
        
        Check for:
        - Injection vulnerabilities (SQL, XSS, etc.)
        - Authentication bypasses
        - Authorization flaws
        - Data exposure risks
        - Input validation issues
        - Known attack patterns
        
        Context: {context}
        
        Security Assessment:"""
        
        self.prompt_templates[OllamaTaskType.SQL_GENERATION] = """
        Generate optimized SQL query:
        
        Requirements: {input_data}
        Database Type: {database_type}
        Schema: {schema}
        
        Generate:
        - Efficient SQL query
        - Proper indexing suggestions
        - Performance optimization notes
        - Security considerations
        
        Context: {context}
        
        SQL Query:"""
        
        self.prompt_templates[OllamaTaskType.ERROR_DIAGNOSIS] = """
        Diagnose and provide solutions for the following error:
        
        Error: {input_data}
        Stack Trace: {stack_trace}
        System Info: {system_info}
        
        Provide:
        - Root cause analysis
        - Step-by-step solution
        - Prevention strategies
        - Related documentation
        
        Context: {context}
        
        Diagnosis:"""
        
        self.prompt_templates[OllamaTaskType.LOG_ANALYSIS] = """
        Analyze the following logs for insights and issues:
        
        Logs: {input_data}
        Time Range: {time_range}
        System: {system}
        
        Identify:
        - Error patterns
        - Performance bottlenecks
        - Security incidents
        - Anomalies
        - Optimization opportunities
        
        Context: {context}
        
        Log Analysis:"""
        
        self.prompt_templates[OllamaTaskType.DECISION_ANALYSIS] = """
        Analyze the decision scenario and provide recommendations:
        
        Scenario: {input_data}
        Options: {options}
        Criteria: {criteria}
        Constraints: {constraints}
        
        Provide:
        - Pros and cons analysis
        - Risk assessment
        - Recommendation with reasoning
        - Implementation considerations
        
        Context: {context}
        
        Decision Analysis:"""
        
        # Add more templates for all task types...
        self._add_remaining_templates()
    
    def _add_remaining_templates(self):
        """Add templates for remaining task types."""
        
        # Mathematical & Logical
        self.prompt_templates[OllamaTaskType.MATHEMATICAL_COMPUTATION] = """
        Solve the mathematical problem step-by-step:
        
        Problem: {input_data}
        
        Provide:
        - Step-by-step solution
        - Alternative methods
        - Verification of answer
        - Explanation of concepts
        
        Context: {context}
        
        Solution:"""
        
        # Educational Tasks
        self.prompt_templates[OllamaTaskType.EXPLANATION_GENERATION] = """
        Generate a clear, comprehensive explanation:
        
        Topic: {input_data}
        Audience Level: {level}
        Learning Objectives: {objectives}
        
        Include:
        - Clear conceptual explanation
        - Practical examples
        - Common misconceptions
        - Further reading suggestions
        
        Context: {context}
        
        Explanation:"""
        
        # System Analysis
        self.prompt_templates[OllamaTaskType.SYSTEM_ANALYSIS] = """
        Analyze the system for optimization opportunities:
        
        System Data: {input_data}
        Metrics: {metrics}
        Goals: {goals}
        
        Provide:
        - Performance bottlenecks
        - Optimization strategies
        - Resource utilization analysis
        - Scalability recommendations
        
        Context: {context}
        
        System Analysis:"""
        
        # Pattern Recognition
        self.prompt_templates[OllamaTaskType.PATTERN_RECOGNITION] = """
        Identify patterns in the provided data:
        
        Data: {input_data}
        Pattern Types: {pattern_types}
        
        Find:
        - Recurring patterns
        - Anomalies
        - Trends and correlations
        - Predictive insights
        
        Context: {context}
        
        Pattern Analysis:"""
    
    async def process_task(self, task: OllamaTask) -> OllamaResult:
        """Process a single task using the most appropriate Ollama model."""
        task_id = self._generate_task_id(task)
        start_time = time.time()
        
        try:
            # Check cache first
            if task.cache_enabled:
                cached_result = await self._check_cache(task)
                if cached_result:
                    logger.info(f"Cache hit for task {task_id}")
                    return cached_result
            
            # Select optimal model for the task
            model_name = self._select_model_for_task(task)
            
            # Build prompt from template
            prompt = self._build_prompt(task)
            
            # Execute the task
            response = await self._execute_ollama_request(
                model_name=model_name,
                prompt=prompt,
                task=task
            )
            
            # Process and validate response
            result = await self._process_response(response, task)
            
            # Calculate confidence score
            confidence = await self._calculate_confidence(result, task)
            
            # Create result object
            processing_time = time.time() - start_time
            ollama_result = OllamaResult(
                task_id=task_id,
                task_type=task.task_type,
                result=result,
                confidence_score=confidence,
                processing_time=processing_time,
                model_used=model_name,
                tokens_used=len(str(response)) // 4,  # Rough estimate
                cached=False
            )
            
            # Cache the result
            if task.cache_enabled:
                await self._cache_result(task, ollama_result)
            
            # Update statistics
            self._update_stats(task, ollama_result)
            
            logger.info(f"Task {task_id} completed in {processing_time:.2f}s with {confidence:.2f} confidence")
            return ollama_result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return OllamaResult(
                task_id=task_id,
                task_type=task.task_type,
                result=None,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                model_used="unknown",
                tokens_used=0,
                error=str(e)
            )
    
    async def batch_process_tasks(self, tasks: List[OllamaTask]) -> List[OllamaResult]:
        """Process multiple tasks efficiently in parallel."""
        logger.info(f"Processing batch of {len(tasks)} tasks")
        
        # Group tasks by priority and model requirements
        task_groups = self._group_tasks_optimally(tasks)
        
        results = []
        for group in task_groups:
            # Process each group in parallel
            group_tasks = [self.process_task(task) for task in group]
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            
            for result in group_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch task failed: {result}")
                else:
                    results.append(result)
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def _select_model_for_task(self, task: OllamaTask) -> str:
        """Select the most appropriate Ollama model for the task."""
        if task.model_preference:
            return task.model_preference
        
        # Task-specific model selection
        task_model_map = {
            OllamaTaskType.CODE_GENERATION: "codellama:34b",
            OllamaTaskType.CODE_ANALYSIS: "deepseek-coder:33b",
            OllamaTaskType.MATHEMATICAL_COMPUTATION: "llama3.1:8b",
            OllamaTaskType.SECURITY_ANALYSIS: "deepseek-coder:33b",
            OllamaTaskType.SQL_GENERATION: "codellama:34b",
            OllamaTaskType.TEXT_ANALYSIS: "llama3.1:8b",
            OllamaTaskType.EXPLANATION_GENERATION: "llama3.1:8b",
            OllamaTaskType.DECISION_ANALYSIS: "llama3.1:8b",
            OllamaTaskType.PATTERN_RECOGNITION: "llama3.1:8b",
            OllamaTaskType.LOG_ANALYSIS: "deepseek-coder:33b",
            OllamaTaskType.ERROR_DIAGNOSIS: "deepseek-coder:33b"
        }
        
        return task_model_map.get(task.task_type, "llama3.1:8b")
    
    def _build_prompt(self, task: OllamaTask) -> str:
        """Build a comprehensive prompt from the task."""
        template = self.prompt_templates.get(task.task_type)
        if not template:
            # Fallback to generic template
            template = """
            Task: {task_type}
            Input: {input_data}
            Context: {context}
            
            Please provide a comprehensive and accurate response:"""
        
        # Format the template with task data
        try:
            formatted_prompt = template.format(
                input_data=task.input_data,
                context=json.dumps(task.context, indent=2) if task.context else "None",
                task_type=task.task_type.value,
                **task.context
            )
        except KeyError as e:
            # Handle missing context variables
            logger.warning(f"Missing context variable {e}, using basic template")
            formatted_prompt = f"""
            Task Type: {task.task_type.value}
            Input: {task.input_data}
            Context: {json.dumps(task.context, indent=2)}
            
            Please provide a comprehensive response:"""
        
        return formatted_prompt
    
    async def _execute_ollama_request(
        self,
        model_name: str,
        prompt: str,
        task: OllamaTask
    ) -> str:
        """Execute the actual request to Ollama."""
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": task.streaming,
            "options": {
                "temperature": task.temperature,
                "num_predict": task.max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        for attempt in range(task.retry_attempts):
            try:
                response = await self.client.post(
                    url,
                    json=payload,
                    timeout=task.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.warning(f"Ollama request failed with status {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Ollama request attempt {attempt + 1} failed: {e}")
                if attempt == task.retry_attempts - 1:
                    raise
                
                # Wait before retry with exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("All Ollama request attempts failed")
    
    async def _process_response(self, response: str, task: OllamaTask) -> Any:
        """Process and enhance the raw Ollama response."""
        # Basic processing based on task type
        if task.task_type in [
            OllamaTaskType.CODE_GENERATION,
            OllamaTaskType.SQL_GENERATION,
            OllamaTaskType.CONFIGURATION_GENERATION
        ]:
            # Extract code blocks
            code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', response, re.DOTALL)
            if code_blocks:
                return {"code": code_blocks[0].strip(), "explanation": response}
            return {"code": response, "explanation": ""}
        
        elif task.task_type in [
            OllamaTaskType.DECISION_ANALYSIS,
            OllamaTaskType.SYSTEM_ANALYSIS,
            OllamaTaskType.SECURITY_ANALYSIS
        ]:
            # Try to extract structured analysis
            return {"analysis": response, "recommendations": self._extract_recommendations(response)}
        
        elif task.task_type == OllamaTaskType.MATHEMATICAL_COMPUTATION:
            # Extract numerical results
            numbers = re.findall(r'-?\d+\.?\d*', response)
            return {"solution": response, "numerical_results": numbers}
        
        # For most tasks, return the raw response with metadata
        return {
            "response": response,
            "processed_at": datetime.utcnow().isoformat(),
            "task_type": task.task_type.value
        }
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract actionable recommendations from text."""
        recommendations = []
        
        # Look for common recommendation patterns
        patterns = [
            r'(?:recommend|suggest|should|advise).*?(?:\.|$)',
            r'(?:consider|try|implement).*?(?:\.|$)',
            r'\d+\.\s*(.*?)(?:\.|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            recommendations.extend(matches)
        
        # Clean and deduplicate
        recommendations = [r.strip() for r in recommendations if len(r.strip()) > 10]
        return list(set(recommendations))[:10]  # Top 10 unique recommendations
    
    async def _calculate_confidence(self, result: Any, task: OllamaTask) -> float:
        """Calculate confidence score for the result using Ollama."""
        # Use Ollama to evaluate the quality of its own output
        confidence_task = OllamaTask(
            task_type=OllamaTaskType.DECISION_ANALYSIS,
            input_data=f"Evaluate the quality and confidence of this response: {str(result)[:1000]}",
            context={
                "original_task": task.task_type.value,
                "evaluation_criteria": [
                    "Accuracy and correctness",
                    "Completeness and thoroughness",
                    "Clarity and coherence",
                    "Relevance to the original task",
                    "Practical usefulness"
                ]
            },
            temperature=0.3,
            max_tokens=500
        )
        
        try:
            confidence_result = await self.process_task(confidence_task)
            confidence_text = str(confidence_result.result)
            
            # Extract confidence score from the evaluation
            score_match = re.search(r'(?:score|confidence|rating).*?(\d+(?:\.\d+)?)', confidence_text.lower())
            if score_match:
                score = float(score_match.group(1))
                return min(score / 10.0, 1.0) if score > 1 else score
            
            # Fallback: analyze positive/negative indicators
            positive_indicators = len(re.findall(r'\b(?:good|excellent|accurate|correct|complete)\b', confidence_text.lower()))
            negative_indicators = len(re.findall(r'\b(?:poor|incorrect|incomplete|unclear|wrong)\b', confidence_text.lower()))
            
            base_confidence = 0.7
            confidence_adjustment = (positive_indicators - negative_indicators) * 0.1
            return max(0.1, min(1.0, base_confidence + confidence_adjustment))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    def _generate_task_id(self, task: OllamaTask) -> str:
        """Generate a unique task ID."""
        task_data = f"{task.task_type.value}:{str(task.input_data)[:100]}:{time.time()}"
        return hashlib.md5(task_data.encode()).hexdigest()[:12]
    
    async def _check_cache(self, task: OllamaTask) -> Optional[OllamaResult]:
        """Check if task result exists in cache."""
        try:
            cache_key = self._get_cache_key(task)
            
            async with valkey_connection_manager.get_client() as client:
                cached_data = await client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    result = OllamaResult(**data)
                    result.cached = True
                    return result
                    
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None
    
    async def _cache_result(self, task: OllamaTask, result: OllamaResult):
        """Cache the task result."""
        try:
            cache_key = self._get_cache_key(task)
            cache_data = {
                "task_id": result.task_id,
                "task_type": result.task_type.value,
                "result": result.result,
                "confidence_score": result.confidence_score,
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "cached": True,
                "metadata": result.metadata
            }
            
            # Cache for 6 hours
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    cache_key,
                    21600,  # 6 hours
                    json.dumps(cache_data, default=str)
                )
                
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _get_cache_key(self, task: OllamaTask) -> str:
        """Generate cache key for a task."""
        cache_data = f"{task.task_type.value}:{str(task.input_data)[:200]}:{task.temperature}:{task.max_tokens}"
        return f"ollama_everything:{hashlib.md5(cache_data.encode()).hexdigest()}"
    
    def _group_tasks_optimally(self, tasks: List[OllamaTask]) -> List[List[OllamaTask]]:
        """Group tasks for optimal batch processing."""
        # Sort by priority first
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Group by model requirements and complexity
        groups = []
        current_group = []
        current_model = None
        
        for task in tasks:
            task_model = self._select_model_for_task(task)
            
            if current_model != task_model and current_group:
                groups.append(current_group)
                current_group = []
            
            current_group.append(task)
            current_model = task_model
            
            # Limit group size for optimal performance
            if len(current_group) >= 5:
                groups.append(current_group)
                current_group = []
                current_model = None
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _update_stats(self, task: OllamaTask, result: OllamaResult):
        """Update performance statistics."""
        self.task_stats["total_tasks"] += 1
        self.task_stats[f"tasks_{task.task_type.value}"] += 1
        self.task_stats[f"model_{result.model_used}"] += 1
        
        if result.cached:
            self.task_stats["cache_hits"] += 1
        else:
            self.task_stats["cache_misses"] += 1
        
        # Store performance metrics
        self.performance_metrics["processing_times"].append(result.processing_time)
        self.performance_metrics["confidence_scores"].append(result.confidence_score)
        self.performance_metrics["token_usage"].append(result.tokens_used)
        
        # Keep only recent metrics (last 1000)
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]
    
    # Convenience methods for common tasks
    
    async def analyze_text(self, text: str, context: Dict[str, Any] = None) -> OllamaResult:
        """Analyze any text using Ollama's intelligence."""
        task = OllamaTask(
            task_type=OllamaTaskType.TEXT_ANALYSIS,
            input_data=text,
            context=context or {}
        )
        return await self.process_task(task)
    
    async def generate_code(
        self,
        requirements: str,
        language: str = "python",
        framework: str = None,
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Generate code for any requirements."""
        task = OllamaTask(
            task_type=OllamaTaskType.CODE_GENERATION,
            input_data=requirements,
            context={
                "language": language,
                "framework": framework,
                **(context or {})
            }
        )
        return await self.process_task(task)
    
    async def analyze_security(self, content: str, context: Dict[str, Any] = None) -> OllamaResult:
        """Perform comprehensive security analysis."""
        task = OllamaTask(
            task_type=OllamaTaskType.SECURITY_ANALYSIS,
            input_data=content,
            context=context or {},
            model_preference="deepseek-coder:33b",
            temperature=0.3
        )
        return await self.process_task(task)
    
    async def generate_sql(
        self,
        requirements: str,
        database_type: str = "postgresql",
        schema: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Generate optimized SQL queries."""
        task = OllamaTask(
            task_type=OllamaTaskType.SQL_GENERATION,
            input_data=requirements,
            context={
                "database_type": database_type,
                "schema": schema,
                **(context or {})
            }
        )
        return await self.process_task(task)
    
    async def diagnose_error(
        self,
        error: str,
        stack_trace: str = "",
        system_info: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Diagnose errors and provide solutions."""
        task = OllamaTask(
            task_type=OllamaTaskType.ERROR_DIAGNOSIS,
            input_data=error,
            context={
                "stack_trace": stack_trace,
                "system_info": system_info,
                **(context or {})
            },
            temperature=0.3
        )
        return await self.process_task(task)
    
    async def analyze_logs(
        self,
        logs: str,
        time_range: str = "1 hour",
        system: str = "unknown",
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Analyze logs for patterns, errors, and insights."""
        task = OllamaTask(
            task_type=OllamaTaskType.LOG_ANALYSIS,
            input_data=logs,
            context={
                "time_range": time_range,
                "system": system,
                **(context or {})
            }
        )
        return await self.process_task(task)
    
    async def make_decision(
        self,
        scenario: str,
        options: List[str] = None,
        criteria: List[str] = None,
        constraints: List[str] = None,
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Analyze decisions and provide recommendations."""
        task = OllamaTask(
            task_type=OllamaTaskType.DECISION_ANALYSIS,
            input_data=scenario,
            context={
                "options": options or [],
                "criteria": criteria or [],
                "constraints": constraints or [],
                **(context or {})
            },
            temperature=0.4
        )
        return await self.process_task(task)
    
    async def recognize_patterns(
        self,
        data: Union[str, List, Dict],
        pattern_types: List[str] = None,
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Identify patterns in any type of data."""
        task = OllamaTask(
            task_type=OllamaTaskType.PATTERN_RECOGNITION,
            input_data=str(data),
            context={
                "pattern_types": pattern_types or ["trends", "anomalies", "correlations"],
                **(context or {})
            }
        )
        return await self.process_task(task)
    
    async def explain_concept(
        self,
        topic: str,
        audience_level: str = "intermediate",
        learning_objectives: List[str] = None,
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Generate comprehensive explanations for any concept."""
        task = OllamaTask(
            task_type=OllamaTaskType.EXPLANATION_GENERATION,
            input_data=topic,
            context={
                "level": audience_level,
                "objectives": learning_objectives or [],
                **(context or {})
            }
        )
        return await self.process_task(task)
    
    async def compute_mathematics(
        self,
        problem: str,
        show_steps: bool = True,
        context: Dict[str, Any] = None
    ) -> OllamaResult:
        """Solve mathematical problems step by step."""
        task = OllamaTask(
            task_type=OllamaTaskType.MATHEMATICAL_COMPUTATION,
            input_data=problem,
            context={
                "show_steps": show_steps,
                **(context or {})
            },
            temperature=0.2
        )
        return await self.process_task(task)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_hit_rate = (
            self.task_stats.get("cache_hits", 0) /
            max(self.task_stats.get("total_tasks", 1), 1)
        )
        
        avg_processing_time = (
            sum(self.performance_metrics["processing_times"]) /
            max(len(self.performance_metrics["processing_times"]), 1)
        )
        
        avg_confidence = (
            sum(self.performance_metrics["confidence_scores"]) /
            max(len(self.performance_metrics["confidence_scores"]), 1)
        )
        
        avg_tokens = (
            sum(self.performance_metrics["token_usage"]) /
            max(len(self.performance_metrics["token_usage"]), 1)
        )
        
        return {
            "total_tasks_processed": self.task_stats.get("total_tasks", 0),
            "cache_hit_rate": cache_hit_rate,
            "average_processing_time": avg_processing_time,
            "average_confidence_score": avg_confidence,
            "average_token_usage": avg_tokens,
            "task_type_breakdown": {
                k: v for k, v in self.task_stats.items() 
                if k.startswith("tasks_")
            },
            "model_usage_breakdown": {
                k: v for k, v in self.task_stats.items() 
                if k.startswith("model_")
            }
        }
    
    async def shutdown(self):
        """Clean shutdown of the OllamaEverything system."""
        try:
            await self.client.aclose()
            self.task_executor.shutdown(wait=True)
            logger.info("OllamaEverything shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global instance
ollama_everything = OllamaEverything()


# Convenience functions for easy access
async def ollama_analyze_text(text: str, **kwargs) -> OllamaResult:
    """Analyze any text with Ollama."""
    return await ollama_everything.analyze_text(text, **kwargs)


async def ollama_generate_code(requirements: str, **kwargs) -> OllamaResult:
    """Generate code with Ollama."""
    return await ollama_everything.generate_code(requirements, **kwargs)


async def ollama_security_analysis(content: str, **kwargs) -> OllamaResult:
    """Perform security analysis with Ollama."""
    return await ollama_everything.analyze_security(content, **kwargs)


async def ollama_generate_sql(requirements: str, **kwargs) -> OllamaResult:
    """Generate SQL with Ollama."""
    return await ollama_everything.generate_sql(requirements, **kwargs)


async def ollama_diagnose_error(error: str, **kwargs) -> OllamaResult:
    """Diagnose errors with Ollama."""
    return await ollama_everything.diagnose_error(error, **kwargs)


async def ollama_analyze_logs(logs: str, **kwargs) -> OllamaResult:
    """Analyze logs with Ollama."""
    return await ollama_everything.analyze_logs(logs, **kwargs)


async def ollama_make_decision(scenario: str, **kwargs) -> OllamaResult:
    """Make decisions with Ollama."""
    return await ollama_everything.make_decision(scenario, **kwargs)


async def ollama_explain_concept(topic: str, **kwargs) -> OllamaResult:
    """Explain concepts with Ollama."""
    return await ollama_everything.explain_concept(topic, **kwargs)


async def ollama_compute_math(problem: str, **kwargs) -> OllamaResult:
    """Solve math problems with Ollama."""
    return await ollama_everything.compute_mathematics(problem, **kwargs)


# Initialize and shutdown functions
async def initialize_ollama_everything():
    """Initialize the OllamaEverything system."""
    try:
        # Warm up the system with a test task
        test_result = await ollama_everything.analyze_text(
            "This is a test to verify Ollama integration is working properly.",
            {"test": True}
        )
        
        if test_result.error:
            raise Exception(f"Ollama test failed: {test_result.error}")
        
        logger.info("OllamaEverything system initialized and tested successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OllamaEverything: {e}")
        raise


async def shutdown_ollama_everything():
    """Shutdown the OllamaEverything system."""
    try:
        await ollama_everything.shutdown()
        logger.info("OllamaEverything system shut down successfully")
        
    except Exception as e:
        logger.error(f"Error shutting down OllamaEverything: {e}")