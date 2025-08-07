"""
Ollama Agent Implementation for Local LLM Inference
====================================================

This module provides the core agent implementation using Ollama for local
LLM inference. Each agent has a specific role and capabilities.

Author: APEX AI System
Version: 2.0
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import httpx
import uuid

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 30.0


class AgentRole(Enum):
    """Roles that agents can play in the system."""
    ANALYZER = "analyzer"
    ARCHITECT = "architect"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    OPTIMIZER = "optimizer"
    TEACHER = "teacher"
    LEARNER = "learner"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class AgentCapability(Enum):
    """Capabilities that agents can have."""
    PROBLEM_UNDERSTANDING = "problem_understanding"
    PATTERN_RECOGNITION = "pattern_recognition"
    SOLUTION_DESIGN = "solution_design"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    CODE_GENERATION = "code_generation"
    VERIFICATION = "verification"
    PROOF_GENERATION = "proof_generation"
    OPTIMIZATION = "optimization"
    EXPLANATION = "explanation"
    LEARNING = "learning"
    COORDINATION = "coordination"


@dataclass
class AgentMemory:
    """Memory structure for agents to maintain context."""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: List[Dict[str, Any]] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    max_short_term: int = 10
    max_long_term: int = 100
    
    def add_to_short_term(self, item: Dict[str, Any]):
        """Add item to short-term memory with size limit."""
        self.short_term.append(item)
        if len(self.short_term) > self.max_short_term:
            # Move oldest to long-term
            self.long_term.append(self.short_term.pop(0))
            if len(self.long_term) > self.max_long_term:
                self.long_term.pop(0)
    
    def recall(self, query: str) -> List[Dict[str, Any]]:
        """Recall relevant memories based on query."""
        relevant = []
        
        # Search in working memory
        for key, value in self.working_memory.items():
            if query.lower() in key.lower():
                relevant.append({"source": "working", "data": value})
        
        # Search in short-term memory
        for item in self.short_term:
            if query.lower() in str(item).lower():
                relevant.append({"source": "short_term", "data": item})
        
        # Search in long-term memory (limited to recent)
        for item in self.long_term[-20:]:
            if query.lower() in str(item).lower():
                relevant.append({"source": "long_term", "data": item})
        
        return relevant


@dataclass
class ThoughtProcess:
    """Represents an agent's thought process."""
    thought_id: str
    agent_id: str
    prompt: str
    reasoning_steps: List[str]
    conclusions: Dict[str, Any]
    confidence: float
    tools_considered: List[str]
    tools_selected: List[str]
    timestamp: datetime


class OllamaAgent:
    """
    Individual agent that uses Ollama for reasoning and decision making.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        model_name: str = "codellama:latest",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Agent memory
        self.memory = AgentMemory()
        
        # Thought history
        self.thought_history: List[ThoughtProcess] = []
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_confidence": 0.0,
            "total_thinking_time": 0.0
        }
        
        logger.info(f"Initialized OllamaAgent {agent_id} with role {role.value}")
    
    async def think(
        self,
        prompt: str,
        context: Optional[Any] = None,
        tools_available: Optional[List[str]] = None
    ) -> "AgentDecision":
        """
        Main thinking process of the agent.
        Uses Chain of Thought (CoT) reasoning.
        """
        
        start_time = datetime.utcnow()
        thought_id = str(uuid.uuid4())
        
        # Recall relevant memories
        memories = self.memory.recall(prompt[:50])
        
        # Build enhanced prompt with context
        enhanced_prompt = self._build_enhanced_prompt(prompt, context, memories, tools_available)
        
        try:
            # Get response from Ollama
            response = await self._query_ollama(enhanced_prompt)
            
            # Parse response and extract reasoning
            reasoning_steps, conclusions, confidence = self._parse_response(response)
            
            # Determine tool selection
            tools_to_use = self._select_tools(conclusions, tools_available)
            
            # Create thought process record
            thought = ThoughtProcess(
                thought_id=thought_id,
                agent_id=self.agent_id,
                prompt=prompt,
                reasoning_steps=reasoning_steps,
                conclusions=conclusions,
                confidence=confidence,
                tools_considered=tools_available or [],
                tools_selected=tools_to_use,
                timestamp=datetime.utcnow()
            )
            
            # Store in memory
            self.thought_history.append(thought)
            self.memory.add_to_short_term({
                "thought_id": thought_id,
                "prompt": prompt[:100],
                "conclusion": conclusions.get("main_conclusion", ""),
                "confidence": confidence
            })
            
            # Update metrics
            thinking_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(True, confidence, thinking_time)
            
            # Create decision
            from .agent_system import AgentDecision
            decision = AgentDecision(
                agent_id=self.agent_id,
                agent_role=self.role,
                decision_type=conclusions.get("decision_type", "general"),
                selected_action=conclusions.get("action", ""),
                reasoning=" ".join(reasoning_steps),
                confidence=confidence,
                tools_to_use=tools_to_use,
                parameters=conclusions
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in agent thinking: {e}")
            self._update_metrics(False, 0.0, 0.0)
            
            # Return low-confidence decision
            from .agent_system import AgentDecision
            return AgentDecision(
                agent_id=self.agent_id,
                agent_role=self.role,
                decision_type="error",
                selected_action="fallback",
                reasoning=f"Error occurred: {str(e)}",
                confidence=0.1,
                tools_to_use=[],
                parameters={"error": str(e)}
            )
    
    def _build_enhanced_prompt(
        self,
        prompt: str,
        context: Any,
        memories: List[Dict[str, Any]],
        tools_available: Optional[List[str]]
    ) -> str:
        """Build an enhanced prompt with all context."""
        
        enhanced = f"""
{self.system_prompt}

Current Role: {self.role.value}
Capabilities: {[cap.value for cap in self.capabilities]}

"""
        
        # Add memories if relevant
        if memories:
            enhanced += "Relevant Past Experience:\n"
            for memory in memories[:3]:  # Limit to 3 most relevant
                enhanced += f"- {memory['data']}\n"
            enhanced += "\n"
        
        # Add context if available
        if context:
            enhanced += f"Current Context:\n{self._serialize_context(context)}\n\n"
        
        # Add available tools
        if tools_available:
            enhanced += f"Available Tools: {tools_available}\n\n"
        
        # Add the main prompt
        enhanced += f"""
Task: {prompt}

Please think step by step using Chain of Thought reasoning:

1. First, understand what is being asked
2. Break down the problem into components
3. Consider different approaches
4. Evaluate trade-offs
5. Select the best approach
6. Determine which tools to use (if any)
7. Provide your conclusion with confidence level (0-1)

Format your response as:

THINKING:
[Your step-by-step reasoning]

CONCLUSION:
{{
    "main_conclusion": "...",
    "action": "...",
    "decision_type": "...",
    "parameters": {{...}},
    "confidence": 0.0-1.0,
    "alternatives": [...]
}}

TOOLS_TO_USE:
[List of tool IDs if any]
"""
        
        return enhanced
    
    def _serialize_context(self, context: Any) -> str:
        """Serialize context object to string."""
        if hasattr(context, '__dict__'):
            return json.dumps({
                k: v for k, v in context.__dict__.items()
                if not k.startswith('_') and not callable(v)
            }, indent=2, default=str)
        elif isinstance(context, dict):
            return json.dumps(context, indent=2, default=str)
        else:
            return str(context)
    
    async def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API for response."""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "stream": False,
                        "options": {
                            "num_predict": self.max_tokens
                        }
                    },
                    timeout=DEFAULT_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return ""
    
    def _parse_response(self, response: str) -> Tuple[List[str], Dict[str, Any], float]:
        """Parse the agent's response to extract reasoning and conclusions."""
        
        reasoning_steps = []
        conclusions = {}
        confidence = 0.5
        
        # Extract sections
        sections = {
            "thinking": "",
            "conclusion": "",
            "tools": ""
        }
        
        current_section = None
        for line in response.split('\n'):
            line_upper = line.upper()
            if "THINKING:" in line_upper:
                current_section = "thinking"
            elif "CONCLUSION:" in line_upper:
                current_section = "conclusion"
            elif "TOOLS_TO_USE:" in line_upper:
                current_section = "tools"
            elif current_section:
                sections[current_section] += line + "\n"
        
        # Parse thinking section
        if sections["thinking"]:
            reasoning_steps = [
                step.strip() 
                for step in sections["thinking"].split('\n') 
                if step.strip() and not step.strip().startswith('#')
            ]
        
        # Parse conclusion section
        if sections["conclusion"]:
            try:
                # Try to parse as JSON
                json_start = sections["conclusion"].find('{')
                json_end = sections["conclusion"].rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = sections["conclusion"][json_start:json_end]
                    conclusions = json.loads(json_str)
                    confidence = float(conclusions.get("confidence", 0.5))
            except:
                # Fallback to text parsing
                conclusions = {
                    "main_conclusion": sections["conclusion"].strip(),
                    "action": "process",
                    "confidence": 0.5
                }
        
        # Ensure we have some reasoning
        if not reasoning_steps:
            reasoning_steps = ["Processed the request"]
        
        # Ensure we have conclusions
        if not conclusions:
            conclusions = {
                "main_conclusion": "Completed analysis",
                "action": "continue",
                "decision_type": "default",
                "parameters": {}
            }
        
        return reasoning_steps, conclusions, confidence
    
    def _select_tools(
        self,
        conclusions: Dict[str, Any],
        tools_available: Optional[List[str]]
    ) -> List[str]:
        """Select tools based on conclusions and availability."""
        
        selected_tools = []
        
        # Check if tools are mentioned in conclusions
        if "tools" in conclusions:
            requested_tools = conclusions["tools"]
            if isinstance(requested_tools, list):
                selected_tools.extend(requested_tools)
            elif isinstance(requested_tools, str):
                selected_tools.append(requested_tools)
        
        # Check for specific action mappings
        action = conclusions.get("action", "")
        action_tool_map = {
            "minimize": ["dfa_minimizer"],
            "convert": ["nfa_to_dfa_converter", "regex_converter"],
            "verify": ["automaton_verifier"],
            "optimize": ["state_optimizer"],
            "generate": ["automaton_generator"],
            "prove": ["proof_generator"]
        }
        
        for key, tools in action_tool_map.items():
            if key in action.lower():
                selected_tools.extend(tools)
        
        # Filter by available tools
        if tools_available:
            selected_tools = [t for t in selected_tools if t in tools_available]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in selected_tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)
        
        return unique_tools
    
    def _update_metrics(self, success: bool, confidence: float, thinking_time: float):
        """Update agent performance metrics."""
        
        if success:
            self.metrics["tasks_completed"] += 1
        else:
            self.metrics["tasks_failed"] += 1
        
        # Update average confidence
        total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
        if total_tasks > 0:
            current_avg = self.metrics["avg_confidence"]
            self.metrics["avg_confidence"] = (
                (current_avg * (total_tasks - 1) + confidence) / total_tasks
            )
        
        self.metrics["total_thinking_time"] += thinking_time
    
    async def execute_task(
        self,
        task: str,
        context: Any,
        tool_results: List[Any]
    ) -> Dict[str, Any]:
        """Execute a specific task with tool results."""
        
        prompt = f"""
Execute this task: {task}

Tool Results Available:
{json.dumps([str(r) for r in tool_results], indent=2)}

Synthesize the tool results and complete the task.
Provide the final output.
"""
        
        decision = await self.think(prompt, context)
        
        return {
            "task": task,
            "result": decision.parameters,
            "confidence": decision.confidence,
            "agent_id": self.agent_id
        }
    
    async def collaborate(
        self,
        other_agent: "OllamaAgent",
        task: str,
        context: Any
    ) -> Dict[str, Any]:
        """Collaborate with another agent on a task."""
        
        # Get other agent's perspective
        other_decision = await other_agent.think(
            f"Provide your perspective on: {task}",
            context
        )
        
        # Incorporate other agent's input
        prompt = f"""
Collaborate on this task: {task}

Other agent ({other_agent.role.value}) suggests:
{other_decision.reasoning}

Confidence: {other_decision.confidence}
Action: {other_decision.selected_action}

Synthesize both perspectives and provide the best solution.
"""
        
        final_decision = await self.think(prompt, context)
        
        return {
            "task": task,
            "collaboration": {
                "agent1": self.agent_id,
                "agent2": other_agent.agent_id,
                "combined_confidence": (final_decision.confidence + other_decision.confidence) / 2
            },
            "result": final_decision.parameters
        }
    
    def learn_from_feedback(
        self,
        task: str,
        feedback: Dict[str, Any],
        success: bool
    ):
        """Learn from feedback on completed tasks."""
        
        learning_entry = {
            "task": task[:100],
            "feedback": feedback,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in long-term memory
        self.memory.long_term.append(learning_entry)
        
        # Update working memory with patterns
        if success:
            pattern_key = f"successful_pattern_{task[:20]}"
            self.memory.working_memory[pattern_key] = {
                "approach": feedback.get("approach"),
                "tools_used": feedback.get("tools_used", []),
                "confidence": feedback.get("confidence", 0.0)
            }
        else:
            pattern_key = f"failed_pattern_{task[:20]}"
            self.memory.working_memory[pattern_key] = {
                "issue": feedback.get("issue"),
                "avoid": feedback.get("avoid", [])
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "model": self.model_name,
            "metrics": self.metrics,
            "memory_stats": {
                "short_term_items": len(self.memory.short_term),
                "long_term_items": len(self.memory.long_term),
                "working_memory_keys": list(self.memory.working_memory.keys())
            },
            "recent_thoughts": len(self.thought_history)
        }
    
    def reset_memory(self, preserve_long_term: bool = True):
        """Reset agent memory."""
        
        self.memory.short_term.clear()
        self.memory.working_memory.clear()
        
        if not preserve_long_term:
            self.memory.long_term.clear()
        
        logger.info(f"Agent {self.agent_id} memory reset")


class AgentPool:
    """
    Pool of agents that can be dynamically allocated for tasks.
    """
    
    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self.agents: Dict[str, OllamaAgent] = {}
        self.available_agents: List[str] = []
        self.busy_agents: Set[str] = set()
    
    async def get_agent(
        self,
        role: AgentRole,
        capabilities: List[AgentCapability]
    ) -> Optional[OllamaAgent]:
        """Get an available agent with required capabilities."""
        
        # Check for available agent with matching role
        for agent_id in self.available_agents:
            agent = self.agents[agent_id]
            if agent.role == role and all(cap in agent.capabilities for cap in capabilities):
                self.available_agents.remove(agent_id)
                self.busy_agents.add(agent_id)
                return agent
        
        # Create new agent if under limit
        if len(self.agents) < self.max_agents:
            agent_id = f"{role.value}_{len(self.agents)}"
            agent = OllamaAgent(
                agent_id=agent_id,
                role=role,
                capabilities=capabilities,
                model_name=self._select_model_for_role(role)
            )
            self.agents[agent_id] = agent
            self.busy_agents.add(agent_id)
            return agent
        
        return None
    
    def release_agent(self, agent_id: str):
        """Release an agent back to the pool."""
        
        if agent_id in self.busy_agents:
            self.busy_agents.remove(agent_id)
            self.available_agents.append(agent_id)
    
    def _select_model_for_role(self, role: AgentRole) -> str:
        """Select appropriate Ollama model for role."""
        
        model_map = {
            AgentRole.ANALYZER: "codellama:latest",
            AgentRole.ARCHITECT: "codellama:latest",
            AgentRole.EXECUTOR: "codellama:latest",
            AgentRole.VERIFIER: "deepseek-coder:latest",
            AgentRole.OPTIMIZER: "codellama:latest",
            AgentRole.TEACHER: "llama2:latest",
            AgentRole.LEARNER: "codellama:latest",
            AgentRole.COORDINATOR: "llama2:latest",
            AgentRole.SPECIALIST: "codellama:latest"
        }
        
        return model_map.get(role, "codellama:latest")