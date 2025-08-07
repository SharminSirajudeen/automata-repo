"""
Dynamic Agent System for Automata Learning Platform
===================================================

This system implements a modular agent architecture using Ollama for local LLM inference.
Agents can dynamically discover tools, create new tools on the fly, and learn from
successful solutions to improve over time.

Key Features:
- Tool discovery and registration
- Dynamic tool creation based on problem types
- Agent orchestration with specialized roles
- Learning from successful solutions
- Integration with existing intelligent solver

Author: APEX AI System
Version: 2.0
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import inspect
import uuid

from .ollama_agent import OllamaAgent, AgentRole, AgentCapability
from .tool_registry import ToolRegistry, Tool, ToolCategory, ToolResult
from .automata_tools import AutomataToolkit
from .intelligent_solver import IntelligentSolver, SolutionStrategy
from .problem_understanding import ProblemType, ProblemRequirements
from .knowledge_extractor import extract_hardcoded_knowledge

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States an agent can be in during execution."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    LEARNING = "learning"
    ERROR = "error"


@dataclass
class AgentContext:
    """Context shared among agents during problem solving."""
    problem_statement: str
    problem_type: Optional[ProblemType] = None
    requirements: Optional[ProblemRequirements] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_execution_history: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    learned_patterns: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentDecision:
    """Represents a decision made by an agent."""
    agent_id: str
    agent_role: AgentRole
    decision_type: str
    selected_action: str
    reasoning: str
    confidence: float
    tools_to_use: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentPlan:
    """Execution plan created by the orchestrator."""
    plan_id: str
    steps: List[Dict[str, Any]]
    agent_assignments: Dict[str, List[str]]  # step_id -> [agent_ids]
    tool_requirements: Dict[str, List[str]]  # step_id -> [tool_ids]
    dependencies: Dict[str, List[str]]  # step_id -> [dependent_step_ids]
    estimated_duration: float
    created_at: datetime


class AgentOrchestrator:
    """
    Main orchestrator that coordinates multiple specialized agents to solve
    Theory of Computation problems dynamically.
    """
    
    def __init__(self):
        # Initialize components
        self.tool_registry = ToolRegistry()
        self.automata_toolkit = AutomataToolkit()
        self.intelligent_solver = IntelligentSolver()
        
        # Register tools
        self._register_core_tools()
        
        # Initialize agents
        self.agents: Dict[str, OllamaAgent] = {}
        self._initialize_agents()
        
        # Knowledge base from hardcoded algorithms
        self.knowledge_base = extract_hardcoded_knowledge()
        
        # Agent state tracking
        self.agent_states: Dict[str, AgentState] = {}
        
        # Learning memory
        self.solution_memory: List[Dict[str, Any]] = []
        
        logger.info("Agent Orchestrator initialized with dynamic capabilities")
    
    def _register_core_tools(self):
        """Register core tools from the automata toolkit."""
        # Get all tools from the toolkit
        tools = self.automata_toolkit.get_all_tools()
        
        for tool in tools:
            self.tool_registry.register_tool(tool)
        
        # Register intelligent solver as a tool
        self.tool_registry.register_tool(Tool(
            tool_id="intelligent_solver",
            name="Intelligent Problem Solver",
            category=ToolCategory.SOLVER,
            description="AI-powered solver that can handle any TOC problem",
            function=self._intelligent_solver_tool,
            parameters={
                "problem_statement": "str",
                "problem_type": "Optional[str]",
                "hints": "Optional[List[str]]"
            },
            examples=["Solve DFA construction", "Minimize automaton", "Prove non-regularity"],
            capabilities=["dynamic_solving", "proof_generation", "optimization"]
        ))
        
        logger.info(f"Registered {len(self.tool_registry.list_tools())} core tools")
    
    def _initialize_agents(self):
        """Initialize specialized agents with different roles."""
        
        # Problem Analyzer Agent
        self.agents["analyzer"] = OllamaAgent(
            agent_id="analyzer",
            role=AgentRole.ANALYZER,
            capabilities=[
                AgentCapability.PROBLEM_UNDERSTANDING,
                AgentCapability.PATTERN_RECOGNITION
            ],
            model_name="codellama:latest",
            system_prompt="""You are a Theory of Computation problem analyzer.
            Your role is to understand problems, identify patterns, and determine
            the best approach for solving them. Extract requirements, constraints,
            and identify the problem type."""
        )
        
        # Solution Architect Agent
        self.agents["architect"] = OllamaAgent(
            agent_id="architect",
            role=AgentRole.ARCHITECT,
            capabilities=[
                AgentCapability.SOLUTION_DESIGN,
                AgentCapability.TOOL_SELECTION
            ],
            model_name="codellama:latest",
            system_prompt="""You are a solution architect for automata problems.
            Design high-level solution strategies, select appropriate tools,
            and create execution plans. Consider multiple approaches and
            optimize for efficiency."""
        )
        
        # Executor Agent
        self.agents["executor"] = OllamaAgent(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
            capabilities=[
                AgentCapability.TOOL_EXECUTION,
                AgentCapability.CODE_GENERATION
            ],
            model_name="codellama:latest",
            system_prompt="""You are an execution specialist for automata algorithms.
            Execute tools, generate code, and implement solutions. Handle edge cases
            and ensure correctness of implementations."""
        )
        
        # Verifier Agent
        self.agents["verifier"] = OllamaAgent(
            agent_id="verifier",
            role=AgentRole.VERIFIER,
            capabilities=[
                AgentCapability.VERIFICATION,
                AgentCapability.PROOF_GENERATION
            ],
            model_name="deepseek-coder:latest",
            system_prompt="""You are a formal verification expert.
            Verify solutions, generate proofs, and ensure correctness.
            Test edge cases and validate against requirements."""
        )
        
        # Optimizer Agent
        self.agents["optimizer"] = OllamaAgent(
            agent_id="optimizer",
            role=AgentRole.OPTIMIZER,
            capabilities=[
                AgentCapability.OPTIMIZATION
            ],
            model_name="codellama:latest",
            system_prompt="""You are an optimization specialist.
            Optimize automata for minimal states, improve algorithms,
            and enhance performance. Apply advanced techniques."""
        )
        
        # Teacher Agent
        self.agents["teacher"] = OllamaAgent(
            agent_id="teacher",
            role=AgentRole.TEACHER,
            capabilities=[
                AgentCapability.EXPLANATION
            ],
            model_name="llama2:latest",
            system_prompt="""You are an educational expert in Theory of Computation.
            Explain solutions clearly, provide intuitive understanding,
            and create educational content. Make complex concepts accessible."""
        )
        
        # Learning Agent
        self.agents["learner"] = OllamaAgent(
            agent_id="learner",
            role=AgentRole.LEARNER,
            capabilities=[
                AgentCapability.LEARNING
            ],
            model_name="codellama:latest",
            system_prompt="""You are a learning specialist.
            Learn from successful solutions, identify patterns,
            and improve the system's capabilities over time."""
        )
        
        # Initialize agent states
        for agent_id in self.agents:
            self.agent_states[agent_id] = AgentState.IDLE
        
        logger.info(f"Initialized {len(self.agents)} specialized agents")
    
    async def solve_problem(
        self,
        problem_statement: str,
        problem_type: Optional[ProblemType] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for solving a problem using the agent system.
        
        This orchestrates multiple agents to:
        1. Understand the problem
        2. Design a solution strategy
        3. Execute the solution
        4. Verify correctness
        5. Optimize if needed
        6. Generate explanation
        7. Learn from the solution
        """
        
        logger.info(f"Starting agent-based problem solving for: {problem_statement[:100]}...")
        
        # Create context
        context = AgentContext(
            problem_statement=problem_statement,
            problem_type=problem_type,
            metadata=user_preferences or {}
        )
        
        try:
            # Phase 1: Problem Analysis
            analysis_result = await self._analyze_problem(context)
            context.requirements = analysis_result.get("requirements")
            context.problem_type = analysis_result.get("problem_type")
            
            # Phase 2: Solution Planning
            solution_plan = await self._plan_solution(context)
            
            # Phase 3: Execute Plan
            execution_result = await self._execute_plan(solution_plan, context)
            
            # Phase 4: Verify Solution
            verification_result = await self._verify_solution(execution_result, context)
            
            # Phase 5: Optimize if needed
            if verification_result.get("needs_optimization", False):
                execution_result = await self._optimize_solution(execution_result, context)
            
            # Phase 6: Generate Explanation
            explanation = await self._generate_explanation(execution_result, context)
            
            # Phase 7: Learn from Solution
            learning_result = await self._learn_from_solution(execution_result, context)
            
            # Compile final result
            final_result = {
                "success": True,
                "problem_statement": problem_statement,
                "problem_type": context.problem_type,
                "analysis": analysis_result,
                "solution": execution_result,
                "verification": verification_result,
                "explanation": explanation,
                "learning_insights": learning_result,
                "context": {
                    "session_id": context.session_id,
                    "tools_used": [t["tool_id"] for t in context.tool_execution_history],
                    "agents_involved": list(self.agents.keys()),
                    "execution_time": (datetime.utcnow() - context.created_at).total_seconds()
                }
            }
            
            # Store in solution memory
            self.solution_memory.append(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in agent-based problem solving: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": context.__dict__
            }
    
    async def _analyze_problem(self, context: AgentContext) -> Dict[str, Any]:
        """Phase 1: Analyze the problem using the analyzer agent."""
        
        self.agent_states["analyzer"] = AgentState.THINKING
        
        # Get available tools for analysis
        analysis_tools = self.tool_registry.get_tools_by_category(ToolCategory.ANALYZER)
        
        # Create analysis prompt
        prompt = f"""
        Analyze this Theory of Computation problem:
        
        Problem: {context.problem_statement}
        
        Available analysis tools: {[t.name for t in analysis_tools]}
        
        Provide:
        1. Problem type classification
        2. Key requirements and constraints
        3. Input/output specifications
        4. Complexity indicators
        5. Suggested approach
        
        Return as structured JSON.
        """
        
        # Get analysis from agent
        analysis = await self.agents["analyzer"].think(prompt, context)
        
        # Execute any analysis tools if needed
        if analysis.tools_to_use:
            for tool_id in analysis.tools_to_use:
                tool_result = await self._execute_tool(tool_id, analysis.parameters, context)
                context.intermediate_results[f"analysis_{tool_id}"] = tool_result
        
        self.agent_states["analyzer"] = AgentState.IDLE
        
        return {
            "problem_type": analysis.parameters.get("problem_type"),
            "requirements": analysis.parameters.get("requirements"),
            "constraints": analysis.parameters.get("constraints"),
            "complexity": analysis.parameters.get("complexity"),
            "suggested_approach": analysis.selected_action
        }
    
    async def _plan_solution(self, context: AgentContext) -> AgentPlan:
        """Phase 2: Create a solution plan using the architect agent."""
        
        self.agent_states["architect"] = AgentState.THINKING
        
        # Get all available tools
        all_tools = self.tool_registry.list_tools()
        
        # Get knowledge base insights
        relevant_patterns = self._find_relevant_patterns(context)
        
        prompt = f"""
        Design a solution plan for this problem:
        
        Problem Type: {context.problem_type}
        Requirements: {json.dumps(context.requirements.__dict__ if context.requirements else {})}
        
        Available Tools: {[t.name for t in all_tools]}
        Relevant Patterns: {relevant_patterns}
        
        Create a step-by-step execution plan with:
        1. Ordered steps
        2. Tool assignments for each step
        3. Agent assignments
        4. Dependencies between steps
        
        Return as structured JSON.
        """
        
        # Get plan from architect
        plan_decision = await self.agents["architect"].think(prompt, context)
        
        # Create execution plan
        plan = AgentPlan(
            plan_id=str(uuid.uuid4()),
            steps=plan_decision.parameters.get("steps", []),
            agent_assignments=plan_decision.parameters.get("agent_assignments", {}),
            tool_requirements=plan_decision.parameters.get("tool_requirements", {}),
            dependencies=plan_decision.parameters.get("dependencies", {}),
            estimated_duration=plan_decision.parameters.get("estimated_duration", 10.0),
            created_at=datetime.utcnow()
        )
        
        self.agent_states["architect"] = AgentState.IDLE
        
        return plan
    
    async def _execute_plan(self, plan: AgentPlan, context: AgentContext) -> Dict[str, Any]:
        """Phase 3: Execute the solution plan."""
        
        self.agent_states["executor"] = AgentState.EXECUTING
        
        execution_results = {}
        
        # Execute steps in order, respecting dependencies
        for step in plan.steps:
            step_id = step["id"]
            
            # Wait for dependencies
            if step_id in plan.dependencies:
                for dep_id in plan.dependencies[step_id]:
                    while dep_id not in execution_results:
                        await asyncio.sleep(0.1)
            
            # Get assigned agents and tools
            assigned_agents = plan.agent_assignments.get(step_id, ["executor"])
            required_tools = plan.tool_requirements.get(step_id, [])
            
            # Execute step
            step_result = await self._execute_step(
                step,
                assigned_agents,
                required_tools,
                context
            )
            
            execution_results[step_id] = step_result
            context.intermediate_results[step_id] = step_result
        
        self.agent_states["executor"] = AgentState.IDLE
        
        # Compile execution result
        return {
            "plan_id": plan.plan_id,
            "steps_completed": len(execution_results),
            "results": execution_results,
            "final_output": self._extract_final_output(execution_results)
        }
    
    async def _execute_step(
        self,
        step: Dict[str, Any],
        agent_ids: List[str],
        tool_ids: List[str],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute a single step of the plan."""
        
        step_results = {
            "step_id": step["id"],
            "description": step.get("description", ""),
            "tool_results": [],
            "agent_outputs": []
        }
        
        # Execute tools
        for tool_id in tool_ids:
            try:
                # Check if we need to create a new tool dynamically
                if tool_id.startswith("dynamic_"):
                    tool = await self._create_dynamic_tool(tool_id, step, context)
                    self.tool_registry.register_tool(tool)
                
                # Execute tool
                tool_result = await self._execute_tool(
                    tool_id,
                    step.get("parameters", {}),
                    context
                )
                step_results["tool_results"].append(tool_result)
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_id}: {e}")
                step_results["tool_results"].append({
                    "tool_id": tool_id,
                    "error": str(e)
                })
        
        # Get agent outputs if needed
        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent_output = await self.agents[agent_id].execute_task(
                    step.get("task", ""),
                    context,
                    step_results["tool_results"]
                )
                step_results["agent_outputs"].append(agent_output)
        
        return step_results
    
    async def _verify_solution(
        self,
        solution: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Phase 4: Verify the solution using the verifier agent."""
        
        self.agent_states["verifier"] = AgentState.THINKING
        
        # Get verification tools
        verification_tools = self.tool_registry.get_tools_by_category(ToolCategory.VERIFIER)
        
        prompt = f"""
        Verify this solution for correctness:
        
        Problem: {context.problem_statement}
        Solution: {json.dumps(solution, indent=2)}
        Requirements: {json.dumps(context.requirements.__dict__ if context.requirements else {})}
        
        Available verification tools: {[t.name for t in verification_tools]}
        
        Perform:
        1. Correctness verification
        2. Completeness check
        3. Edge case testing
        4. Performance analysis
        
        Return verification results with confidence score.
        """
        
        verification = await self.agents["verifier"].think(prompt, context)
        
        # Execute verification tools
        verification_results = {
            "is_correct": verification.parameters.get("is_correct", False),
            "completeness": verification.parameters.get("completeness", 0.0),
            "confidence": verification.confidence,
            "issues": verification.parameters.get("issues", []),
            "test_results": [],
            "needs_optimization": verification.parameters.get("needs_optimization", False)
        }
        
        # Run test cases
        if verification.tools_to_use:
            for tool_id in verification.tools_to_use:
                test_result = await self._execute_tool(tool_id, solution, context)
                verification_results["test_results"].append(test_result)
        
        self.agent_states["verifier"] = AgentState.IDLE
        
        return verification_results
    
    async def _optimize_solution(
        self,
        solution: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Phase 5: Optimize the solution if needed."""
        
        self.agent_states["optimizer"] = AgentState.THINKING
        
        # Get optimization tools
        optimization_tools = self.tool_registry.get_tools_by_category(ToolCategory.OPTIMIZER)
        
        prompt = f"""
        Optimize this automata solution:
        
        Current solution: {json.dumps(solution, indent=2)}
        
        Available optimization tools: {[t.name for t in optimization_tools]}
        
        Apply optimizations for:
        1. State minimization
        2. Transition optimization
        3. Performance improvement
        4. Memory efficiency
        
        Return optimized solution.
        """
        
        optimization = await self.agents["optimizer"].think(prompt, context)
        
        # Apply optimizations
        optimized_solution = solution.copy()
        
        if optimization.tools_to_use:
            for tool_id in optimization.tools_to_use:
                opt_result = await self._execute_tool(
                    tool_id,
                    optimized_solution,
                    context
                )
                if opt_result.success:
                    optimized_solution = opt_result.data
        
        self.agent_states["optimizer"] = AgentState.IDLE
        
        return optimized_solution
    
    async def _generate_explanation(
        self,
        solution: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """Phase 6: Generate educational explanation."""
        
        self.agent_states["teacher"] = AgentState.THINKING
        
        prompt = f"""
        Create an educational explanation for this solution:
        
        Problem: {context.problem_statement}
        Solution: {json.dumps(solution, indent=2)}
        
        Include:
        1. Intuitive explanation of the approach
        2. Step-by-step walkthrough
        3. Key insights and patterns
        4. Common mistakes to avoid
        5. Related concepts and extensions
        
        Make it clear and accessible for students.
        """
        
        explanation = await self.agents["teacher"].think(prompt, context)
        
        self.agent_states["teacher"] = AgentState.IDLE
        
        return explanation.parameters.get("explanation", explanation.reasoning)
    
    async def _learn_from_solution(
        self,
        solution: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Phase 7: Learn from the successful solution."""
        
        self.agent_states["learner"] = AgentState.LEARNING
        
        prompt = f"""
        Extract learning insights from this successful solution:
        
        Problem Type: {context.problem_type}
        Solution: {json.dumps(solution, indent=2)}
        Tools Used: {[t["tool_id"] for t in context.tool_execution_history]}
        
        Identify:
        1. Successful patterns and strategies
        2. Tool combinations that worked well
        3. Optimization opportunities discovered
        4. Edge cases handled
        5. Reusable components
        
        Generate learning data for future problems.
        """
        
        learning = await self.agents["learner"].think(prompt, context)
        
        # Store learned patterns
        learned_pattern = {
            "problem_type": context.problem_type,
            "successful_strategy": learning.selected_action,
            "tool_sequence": [t["tool_id"] for t in context.tool_execution_history],
            "key_insights": learning.parameters.get("insights", []),
            "reusable_components": learning.parameters.get("components", []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        context.learned_patterns.append(learned_pattern)
        
        self.agent_states["learner"] = AgentState.IDLE
        
        return learned_pattern
    
    def _find_relevant_patterns(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Find relevant patterns from knowledge base and solution memory."""
        
        relevant_patterns = []
        
        # Search knowledge base
        if context.problem_type:
            for pattern_id, pattern in self.knowledge_base.get("patterns", {}).items():
                if context.problem_type.value in [pt for pt in pattern.get("problem_types", [])]:
                    relevant_patterns.append({
                        "source": "knowledge_base",
                        "pattern": pattern
                    })
        
        # Search solution memory
        for solution in self.solution_memory[-10:]:  # Last 10 solutions
            if solution.get("problem_type") == context.problem_type:
                relevant_patterns.append({
                    "source": "solution_memory",
                    "pattern": solution.get("learning_insights", {})
                })
        
        return relevant_patterns
    
    async def _execute_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        context: AgentContext
    ) -> ToolResult:
        """Execute a tool and record the execution."""
        
        tool = self.tool_registry.get_tool(tool_id)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool {tool_id} not found"
            )
        
        # Execute tool
        result = await tool.execute(parameters)
        
        # Record execution
        context.tool_execution_history.append({
            "tool_id": tool_id,
            "parameters": parameters,
            "result": result.__dict__,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return result
    
    async def _create_dynamic_tool(
        self,
        tool_id: str,
        step: Dict[str, Any],
        context: AgentContext
    ) -> Tool:
        """Create a new tool dynamically based on the problem requirements."""
        
        logger.info(f"Creating dynamic tool: {tool_id}")
        
        # Use executor agent to generate tool code
        prompt = f"""
        Create a new tool for this specific task:
        
        Tool ID: {tool_id}
        Task: {step.get("description", "")}
        Requirements: {step.get("requirements", {})}
        
        Generate:
        1. Tool implementation as a Python function
        2. Parameter specifications
        3. Return type and format
        
        The function should be self-contained and handle the specific task.
        """
        
        tool_spec = await self.agents["executor"].think(prompt, context)
        
        # Create tool function dynamically
        tool_code = tool_spec.parameters.get("code", "")
        
        # Create a simple wrapper function
        async def dynamic_tool_function(params: Dict[str, Any]) -> ToolResult:
            try:
                # Execute the generated code (in production, use sandboxing)
                local_vars = {"params": params}
                exec(tool_code, {}, local_vars)
                result = local_vars.get("result", {})
                
                return ToolResult(
                    success=True,
                    data=result
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e)
                )
        
        # Create and return the tool
        return Tool(
            tool_id=tool_id,
            name=step.get("description", f"Dynamic Tool {tool_id}"),
            category=ToolCategory.DYNAMIC,
            description=f"Dynamically created tool for: {step.get('description', '')}",
            function=dynamic_tool_function,
            parameters=tool_spec.parameters.get("parameters", {}),
            examples=[],
            capabilities=["dynamic_execution"],
            metadata={
                "created_at": datetime.utcnow().isoformat(),
                "created_for": context.session_id
            }
        )
    
    def _extract_final_output(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the final output from execution results."""
        
        # Find the last step's output
        if execution_results:
            last_step = list(execution_results.values())[-1]
            
            # Extract automaton if present
            for tool_result in last_step.get("tool_results", []):
                if isinstance(tool_result, ToolResult) and tool_result.success:
                    if "automaton" in tool_result.data:
                        return tool_result.data
            
            # Extract from agent outputs
            for agent_output in last_step.get("agent_outputs", []):
                if isinstance(agent_output, dict) and "solution" in agent_output:
                    return agent_output["solution"]
        
        return execution_results
    
    async def _intelligent_solver_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Wrapper for intelligent solver as a tool."""
        try:
            solution = await self.intelligent_solver.solve_problem(
                problem_statement=params.get("problem_statement", ""),
                problem_type=params.get("problem_type"),
                hints=params.get("hints")
            )
            
            return ToolResult(
                success=True,
                data=solution.dict() if hasattr(solution, 'dict') else solution
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def create_custom_agent(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        system_prompt: str,
        model_name: str = "codellama:latest"
    ) -> OllamaAgent:
        """Create a custom agent with specific capabilities."""
        
        agent = OllamaAgent(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
            model_name=model_name,
            system_prompt=system_prompt
        )
        
        self.agents[agent_id] = agent
        self.agent_states[agent_id] = AgentState.IDLE
        
        logger.info(f"Created custom agent: {agent_id} with role {role}")
        
        return agent
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get the current status of all agents."""
        return {
            "agents": {
                agent_id: {
                    "role": agent.role.value,
                    "state": self.agent_states[agent_id].value,
                    "capabilities": [cap.value for cap in agent.capabilities]
                }
                for agent_id, agent in self.agents.items()
            },
            "tools_available": len(self.tool_registry.list_tools()),
            "solutions_in_memory": len(self.solution_memory),
            "knowledge_patterns": len(self.knowledge_base.get("patterns", {}))
        }


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()