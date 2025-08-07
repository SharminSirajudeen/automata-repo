"""
Ollama-Powered Agent System with LangChain-Inspired Architecture
================================================================

This integrates the morning's agent system with Ollama for EVERYTHING.
Inspired by LangChain's design patterns but using only Ollama.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

from .ollama_everything import OllamaEverything
from .agent_system import AgentRole, AgentContext, AgentOrchestrator
from .tool_registry import ToolRegistry
from .automata_tools import AutomataToolkit


class OllamaAgentOrchestrator:
    """
    Complete integration of morning's agent system with Ollama.
    Every agent uses Ollama for ALL operations.
    """
    
    def __init__(self):
        self.ollama = OllamaEverything()
        self.tool_registry = ToolRegistry()
        self.toolkit = AutomataToolkit()
        
        # Initialize all agents with Ollama
        self.agents = self._initialize_ollama_agents()
        
        # LangChain-inspired components
        self.memory = OllamaMemory()
        self.chains = OllamaChainBuilder()
        self.hub = OllamaPromptHub()
        
    def _initialize_ollama_agents(self) -> Dict[AgentRole, 'OllamaAgent']:
        """Initialize all 7 specialized agents using Ollama"""
        
        agents = {}
        
        # Each agent uses a specific Ollama model for optimal performance
        model_mapping = {
            AgentRole.ANALYZER: "llama3.2:latest",      # Best for understanding
            AgentRole.ARCHITECT: "qwen2.5-coder:latest", # Best for design
            AgentRole.EXECUTOR: "deepseek-coder-v2:latest", # Best for code
            AgentRole.VERIFIER: "codellama:13b",        # Best for verification
            AgentRole.OPTIMIZER: "mistral:latest",      # Best for optimization
            AgentRole.TEACHER: "llama3.2:latest",       # Best for explanation
            AgentRole.LEARNER: "phi3:medium",           # Efficient for learning
        }
        
        for role, model in model_mapping.items():
            agents[role] = OllamaAgent(
                role=role,
                model=model,
                ollama_client=self.ollama,
                tools=self.tool_registry
            )
        
        return agents
    
    async def solve_problem(
        self,
        problem_statement: str,
        problem_type: str = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Solve any automata theory problem using Ollama-powered agents.
        
        This is the COMPLETE implementation from this morning!
        """
        
        # Phase 1: Analysis (Analyzer Agent with Ollama)
        analysis = await self.agents[AgentRole.ANALYZER].analyze(
            problem_statement,
            problem_type
        )
        
        # Phase 2: Architecture (Architect Agent with Ollama)
        plan = await self.agents[AgentRole.ARCHITECT].design_solution(
            analysis,
            available_tools=self.tool_registry.list_tools()
        )
        
        # Phase 3: Execution (Executor Agent with Ollama)
        solution = await self.agents[AgentRole.EXECUTOR].execute_plan(
            plan,
            tools=self.toolkit
        )
        
        # Phase 4: Verification (Verifier Agent with Ollama)
        verification = await self.agents[AgentRole.VERIFIER].verify_solution(
            solution,
            analysis["requirements"]
        )
        
        # Phase 5: Optimization (Optimizer Agent with Ollama)
        if verification["can_optimize"]:
            solution = await self.agents[AgentRole.OPTIMIZER].optimize(
                solution,
                verification["optimization_suggestions"]
            )
        
        # Phase 6: Teaching (Teacher Agent with Ollama)
        explanation = await self.agents[AgentRole.TEACHER].explain(
            problem_statement,
            solution,
            target_level="undergraduate"
        )
        
        # Phase 7: Learning (Learner Agent with Ollama)
        await self.agents[AgentRole.LEARNER].learn_from_solution(
            problem_statement,
            solution,
            verification["metrics"]
        )
        
        return {
            "analysis": analysis,
            "plan": plan,
            "solution": solution,
            "verification": verification,
            "explanation": explanation,
            "metadata": {
                "agents_used": [role.value for role in AgentRole],
                "tools_used": plan.get("tools", []),
                "model_calls": await self._count_ollama_calls(),
                "total_tokens": await self._count_tokens()
            }
        }
    
    async def _count_ollama_calls(self) -> int:
        """Count total Ollama API calls made"""
        total = 0
        for agent in self.agents.values():
            total += agent.call_count
        return total
    
    async def _count_tokens(self) -> int:
        """Count total tokens used"""
        total = 0
        for agent in self.agents.values():
            total += agent.token_count
        return total


class OllamaAgent:
    """
    Individual agent that uses Ollama for ALL operations.
    Replaces any local ML processing with Ollama calls.
    """
    
    def __init__(self, role: AgentRole, model: str, ollama_client: OllamaEverything, tools: ToolRegistry):
        self.role = role
        self.model = model
        self.ollama = ollama_client
        self.tools = tools
        self.call_count = 0
        self.token_count = 0
        
        # Agent-specific prompts (LangChain Hub inspired)
        self.system_prompt = self._get_system_prompt()
        
    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt"""
        prompts = {
            AgentRole.ANALYZER: """You are an expert at analyzing automata theory problems.
                Extract requirements, identify patterns, and determine problem type.
                Always identify: alphabet, language constraints, automaton type needed.""",
            
            AgentRole.ARCHITECT: """You are a solution architect for automata problems.
                Design step-by-step solution plans using available tools.
                Create efficient, minimal solutions.""",
            
            AgentRole.EXECUTOR: """You are a code executor for automata construction.
                Generate and execute code to build automata.
                Use tools efficiently and handle errors gracefully.""",
            
            AgentRole.VERIFIER: """You are a verification specialist.
                Test solutions thoroughly with edge cases.
                Identify correctness issues and optimization opportunities.""",
            
            AgentRole.OPTIMIZER: """You are an optimization expert.
                Minimize automata, reduce states, optimize performance.
                Apply formal minimization algorithms.""",
            
            AgentRole.TEACHER: """You are an educator in automata theory.
                Explain solutions clearly with examples.
                Adapt explanations to student level.""",
            
            AgentRole.LEARNER: """You are a learning agent.
                Extract patterns from solutions for future use.
                Build knowledge base of problem-solving strategies."""
        }
        return prompts.get(self.role, "You are a helpful assistant.")
    
    async def analyze(self, problem: str, problem_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyzer agent: Understand the problem using Ollama"""
        
        prompt = f"""
        {self.system_prompt}
        
        Analyze this automata theory problem:
        {problem}
        
        {"Problem type hint: " + problem_type if problem_type else ""}
        
        Extract and return as JSON:
        1. Problem type (DFA, NFA, PDA, TM, CFG, etc.)
        2. Alphabet symbols
        3. Language constraints
        4. Required automaton properties
        5. Test cases to verify solution
        6. Complexity estimate
        """
        
        response = await self.ollama.process("analyze_problem", {
            "prompt": prompt,
            "model": self.model
        })
        
        self.call_count += 1
        self.token_count += response.get("token_count", 0)
        
        return response["result"]
    
    async def design_solution(self, analysis: Dict, available_tools: List[str]) -> Dict[str, Any]:
        """Architect agent: Design solution plan using Ollama"""
        
        prompt = f"""
        {self.system_prompt}
        
        Design a solution plan for this analyzed problem:
        {json.dumps(analysis, indent=2)}
        
        Available tools:
        {json.dumps(available_tools, indent=2)}
        
        Create a step-by-step plan with:
        1. Tools to use in order
        2. Parameters for each tool
        3. Expected intermediate results
        4. Fallback strategies
        """
        
        response = await self.ollama.process("design_solution", {
            "prompt": prompt,
            "model": self.model
        })
        
        self.call_count += 1
        self.token_count += response.get("token_count", 0)
        
        return response["result"]
    
    async def execute_plan(self, plan: Dict, tools: AutomataToolkit) -> Dict[str, Any]:
        """Executor agent: Execute the plan using tools and Ollama"""
        
        solution = {}
        
        for step in plan.get("steps", []):
            tool_name = step["tool"]
            params = step["parameters"]
            
            # Use Ollama to generate code for tool execution
            code_prompt = f"""
            {self.system_prompt}
            
            Generate Python code to execute this tool:
            Tool: {tool_name}
            Parameters: {json.dumps(params)}
            
            The code should:
            1. Call the appropriate tool from the toolkit
            2. Handle any errors
            3. Return the result
            """
            
            code_response = await self.ollama.process("generate_code", {
                "prompt": code_prompt,
                "model": self.model
            })
            
            self.call_count += 1
            self.token_count += code_response.get("token_count", 0)
            
            # Execute the generated code (simplified for demonstration)
            # In production, use exec() with proper sandboxing
            tool_result = await tools.execute(tool_name, params)
            solution[step["name"]] = tool_result
        
        return solution
    
    async def verify_solution(self, solution: Dict, requirements: Dict) -> Dict[str, Any]:
        """Verifier agent: Verify solution correctness using Ollama"""
        
        prompt = f"""
        {self.system_prompt}
        
        Verify this automaton solution:
        Solution: {json.dumps(solution, indent=2)}
        Requirements: {json.dumps(requirements, indent=2)}
        
        Perform these checks:
        1. Does it accept all strings in the language?
        2. Does it reject all strings not in the language?
        3. Is it minimal?
        4. Are there edge cases not handled?
        5. Can it be optimized?
        
        Return verification report as JSON.
        """
        
        response = await self.ollama.process("verify_solution", {
            "prompt": prompt,
            "model": self.model
        })
        
        self.call_count += 1
        self.token_count += response.get("token_count", 0)
        
        return response["result"]
    
    async def optimize(self, solution: Dict, suggestions: List[str]) -> Dict[str, Any]:
        """Optimizer agent: Optimize solution using Ollama"""
        
        prompt = f"""
        {self.system_prompt}
        
        Optimize this automaton:
        Current solution: {json.dumps(solution, indent=2)}
        Optimization suggestions: {json.dumps(suggestions, indent=2)}
        
        Apply these optimizations:
        1. State minimization
        2. Transition reduction
        3. Remove unreachable states
        4. Merge equivalent states
        
        Return optimized automaton as JSON.
        """
        
        response = await self.ollama.process("optimize_automaton", {
            "prompt": prompt,
            "model": self.model
        })
        
        self.call_count += 1
        self.token_count += response.get("token_count", 0)
        
        return response["result"]
    
    async def explain(self, problem: str, solution: Dict, target_level: str) -> str:
        """Teacher agent: Generate explanation using Ollama"""
        
        prompt = f"""
        {self.system_prompt}
        
        Explain this automaton solution for {target_level} students:
        
        Problem: {problem}
        Solution: {json.dumps(solution, indent=2)}
        
        Include:
        1. Intuitive explanation of the approach
        2. Step-by-step construction
        3. Why this solution works
        4. Common mistakes to avoid
        5. Practice exercises
        """
        
        response = await self.ollama.process("generate_explanation", {
            "prompt": prompt,
            "model": self.model
        })
        
        self.call_count += 1
        self.token_count += response.get("token_count", 0)
        
        return response["result"]
    
    async def learn_from_solution(self, problem: str, solution: Dict, metrics: Dict) -> None:
        """Learner agent: Extract patterns and learn using Ollama"""
        
        prompt = f"""
        {self.system_prompt}
        
        Learn from this successful solution:
        Problem: {problem}
        Solution: {json.dumps(solution, indent=2)}
        Performance metrics: {json.dumps(metrics, indent=2)}
        
        Extract:
        1. Problem patterns
        2. Solution strategies
        3. Optimization techniques
        4. Common structures
        5. Reusable components
        
        Store these insights for future problems.
        """
        
        response = await self.ollama.process("extract_patterns", {
            "prompt": prompt,
            "model": self.model
        })
        
        self.call_count += 1
        self.token_count += response.get("token_count", 0)
        
        # Store learned patterns (would integrate with vector DB)
        await self._store_knowledge(response["result"])
    
    async def _store_knowledge(self, knowledge: Dict) -> None:
        """Store learned knowledge for future use"""
        # This would integrate with ChromaDB/vector store
        # For now, just log it
        print(f"Learned: {knowledge}")


class OllamaMemory:
    """
    LangChain-inspired memory system using Ollama for summarization.
    No local ML models - everything through Ollama!
    """
    
    def __init__(self, ollama_client: OllamaEverything = None):
        self.ollama = ollama_client or OllamaEverything()
        self.short_term = []  # Recent interactions
        self.long_term = {}   # Summarized knowledge
        
    async def add_interaction(self, input: str, output: str) -> None:
        """Add interaction to memory"""
        self.short_term.append({
            "timestamp": datetime.now().isoformat(),
            "input": input,
            "output": output
        })
        
        # Summarize if short-term memory is getting large
        if len(self.short_term) > 10:
            await self._consolidate_memory()
    
    async def _consolidate_memory(self) -> None:
        """Use Ollama to summarize and consolidate memory"""
        
        prompt = f"""
        Summarize these interactions into key insights:
        {json.dumps(self.short_term[-10:], indent=2)}
        
        Extract:
        1. Common patterns
        2. Important facts
        3. User preferences
        4. Successful strategies
        """
        
        summary = await self.ollama.process("summarize_memory", {
            "prompt": prompt,
            "model": "llama3.2:latest"
        })
        
        # Store in long-term memory
        key = hashlib.md5(str(self.short_term[-10:]).encode()).hexdigest()
        self.long_term[key] = summary["result"]
        
        # Clear old short-term memories
        self.short_term = self.short_term[-5:]
    
    async def retrieve_relevant(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories using Ollama for similarity"""
        
        # Use Ollama to find relevant memories
        prompt = f"""
        Query: {query}
        
        Which of these memories are most relevant?
        {json.dumps(list(self.long_term.values())[:20], indent=2)}
        
        Return the {k} most relevant memories.
        """
        
        relevant = await self.ollama.process("find_relevant_memories", {
            "prompt": prompt,
            "model": "llama3.2:latest"
        })
        
        return relevant["result"]


class OllamaChainBuilder:
    """
    LangChain LCEL-inspired chain builder using only Ollama.
    Build complex chains without heavy dependencies!
    """
    
    def __init__(self, ollama_client: OllamaEverything = None):
        self.ollama = ollama_client or OllamaEverything()
        self.chain = []
    
    def add_step(self, step_type: str, config: Dict) -> 'OllamaChainBuilder':
        """Add a step to the chain"""
        self.chain.append({
            "type": step_type,
            "config": config
        })
        return self
    
    async def run(self, input: Any) -> Any:
        """Execute the chain using Ollama"""
        result = input
        
        for step in self.chain:
            if step["type"] == "prompt":
                result = await self.ollama.process("generate", {
                    "prompt": step["config"]["template"].format(input=result),
                    "model": step["config"].get("model", "llama3.2:latest")
                })
            elif step["type"] == "parse":
                result = await self.ollama.process("parse", {
                    "text": result,
                    "format": step["config"]["format"]
                })
            elif step["type"] == "transform":
                result = await self.ollama.process("transform", {
                    "data": result,
                    "transformation": step["config"]["transformation"]
                })
        
        return result


class OllamaPromptHub:
    """
    LangChain Hub-inspired prompt management using Ollama.
    Store and version prompts, test them with Ollama.
    """
    
    def __init__(self, ollama_client: OllamaEverything = None):
        self.ollama = ollama_client or OllamaEverything()
        self.prompts = {}
        self._load_default_prompts()
    
    def _load_default_prompts(self):
        """Load default prompts for automata problems"""
        self.prompts = {
            "dfa_construction": """
                Construct a DFA for the following language:
                {language_description}
                
                Alphabet: {alphabet}
                
                Return the DFA as JSON with states, transitions, start_state, and accept_states.
            """,
            
            "nfa_to_dfa": """
                Convert this NFA to an equivalent DFA:
                {nfa_definition}
                
                Use the subset construction algorithm.
                Return the resulting DFA.
            """,
            
            "proof_pumping_lemma": """
                Prove using the pumping lemma that this language is not regular:
                {language}
                
                Show all steps clearly.
            """,
            
            "cfg_to_cnf": """
                Convert this context-free grammar to Chomsky Normal Form:
                {grammar}
                
                Show each transformation step.
            """
        }
    
    async def test_prompt(self, prompt_name: str, test_inputs: Dict) -> Dict:
        """Test a prompt with Ollama to ensure it works"""
        
        prompt = self.prompts.get(prompt_name)
        if not prompt:
            return {"error": "Prompt not found"}
        
        # Format prompt with test inputs
        formatted = prompt.format(**test_inputs)
        
        # Test with Ollama
        result = await self.ollama.process("test_prompt", {
            "prompt": formatted,
            "model": "llama3.2:latest"
        })
        
        # Validate result
        validation = await self.ollama.process("validate_output", {
            "output": result,
            "expected_format": "automaton_json"
        })
        
        return {
            "prompt": formatted,
            "result": result,
            "validation": validation
        }
    
    def add_prompt(self, name: str, template: str, metadata: Dict = None):
        """Add a new prompt to the hub"""
        self.prompts[name] = {
            "template": template,
            "metadata": metadata or {},
            "version": 1,
            "created": datetime.now().isoformat()
        }
    
    def get_prompt(self, name: str, version: int = None) -> str:
        """Get a prompt by name and optional version"""
        return self.prompts.get(name, {}).get("template", "")


# Integration with FastAPI
async def setup_ollama_agents(app):
    """Setup function to integrate with FastAPI app"""
    
    # Initialize the orchestrator
    orchestrator = OllamaAgentOrchestrator()
    
    # Add to app state
    app.state.agent_orchestrator = orchestrator
    
    # Add routes
    @app.post("/api/agent/solve")
    async def solve_with_agents(problem: str, problem_type: Optional[str] = None):
        """Solve problem using Ollama-powered agents"""
        result = await orchestrator.solve_problem(problem, problem_type)
        return result
    
    @app.get("/api/agent/status")
    async def get_agent_status():
        """Get status of all agents"""
        status = {}
        for role, agent in orchestrator.agents.items():
            status[role.value] = {
                "model": agent.model,
                "calls": agent.call_count,
                "tokens": agent.token_count
            }
        return status
    
    @app.post("/api/agent/memory/add")
    async def add_to_memory(input: str, output: str):
        """Add interaction to agent memory"""
        await orchestrator.memory.add_interaction(input, output)
        return {"status": "added"}
    
    @app.post("/api/agent/chain/build")
    async def build_and_run_chain(steps: List[Dict], input: Any):
        """Build and run a chain of operations"""
        chain = OllamaChainBuilder(orchestrator.ollama)
        for step in steps:
            chain.add_step(step["type"], step["config"])
        result = await chain.run(input)
        return result
    
    return orchestrator


# Example usage
async def example_usage():
    """Example of using the complete Ollama-powered agent system"""
    
    # Initialize orchestrator
    orchestrator = OllamaAgentOrchestrator()
    
    # Solve a problem using all agents
    problem = "Create a DFA that accepts strings with even number of 0s and odd number of 1s"
    
    result = await orchestrator.solve_problem(problem)
    
    print(f"Analysis: {result['analysis']}")
    print(f"Solution: {result['solution']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Ollama calls: {result['metadata']['model_calls']}")
    print(f"Total tokens: {result['metadata']['total_tokens']}")
    
    # Build a chain (LangChain LCEL style but with Ollama)
    chain = OllamaChainBuilder(orchestrator.ollama)
    chain_result = await (
        chain
        .add_step("prompt", {
            "template": "Explain this concept: {input}",
            "model": "llama3.2:latest"
        })
        .add_step("parse", {
            "format": "markdown"
        })
        .add_step("transform", {
            "transformation": "add_examples"
        })
        .run("Finite Automata")
    )
    
    print(f"Chain result: {chain_result}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())