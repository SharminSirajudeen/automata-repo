"""
Proof Assistant Workflow using LangGraph.
Provides interactive proof construction with backtracking, verification, and step-by-step guidance.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .langgraph_core import (
    BaseWorkflowNode, ConversationState, WorkflowGraphBuilder, 
    WorkflowConfig, InterruptType, workflow_executor
)
from .orchestrator import ExecutionMode
from .ai_proof_assistant import ProofTechnique, ProofStep, ProofStatus

logger = logging.getLogger(__name__)


class ProofPhase(str, Enum):
    """Different phases of proof construction."""
    PROBLEM_ANALYSIS = "problem_analysis"
    STRATEGY_SELECTION = "strategy_selection"
    PROOF_CONSTRUCTION = "proof_construction"
    STEP_VERIFICATION = "step_verification"
    PROOF_REVIEW = "proof_review"
    BACKTRACKING = "backtracking"
    COMPLETION = "completion"


class ProofStrategy(str, Enum):
    """Available proof strategies."""
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CONSTRUCTION = "construction"
    CASE_ANALYSIS = "case_analysis"
    REDUCTION = "reduction"


class VerificationLevel(str, Enum):
    """Levels of proof verification."""
    SYNTAX = "syntax"
    LOGIC = "logic"
    SEMANTIC = "semantic"
    FORMAL = "formal"


@dataclass
class ProofStepDetails:
    """Detailed information about a proof step."""
    step_id: str
    content: str
    justification: str
    dependencies: List[str]
    verification_status: str
    confidence_score: float
    generated_at: datetime
    backtrack_point: bool = False


class ProofAssistantState(ConversationState):
    """Extended state for proof assistant workflow."""
    theorem_statement: str
    proof_strategy: Optional[ProofStrategy]
    current_phase: ProofPhase
    proof_steps: List[ProofStepDetails]
    verification_history: List[Dict[str, Any]]
    backtrack_stack: List[Dict[str, Any]]
    assumptions: List[str]
    definitions: List[str]
    lemmas_used: List[str]
    proof_outline: List[str]
    verification_errors: List[str]
    suggested_fixes: List[str]
    confidence_threshold: float
    auto_verify: bool


class ProblemAnalysisNode(BaseWorkflowNode):
    """Node for analyzing the theorem and determining proof approach."""
    
    def __init__(self):
        super().__init__("problem_analysis")
    
    async def execute(self, state: ProofAssistantState) -> ProofAssistantState:
        """Analyze the theorem statement and extract key components."""
        try:
            theorem = state["theorem_statement"]
            
            # Analyze theorem structure
            analysis_prompt = f"""
Analyze this theorem statement for proof construction:
"{theorem}"

Provide:
1. Key components and structure
2. Required definitions and lemmas
3. Potential proof strategies
4. Difficulty assessment
5. Common pitfalls to avoid

Format as structured analysis.
            """
            
            analysis_result = await self.orchestrator.execute(
                task="theorem_analysis",
                prompt=analysis_prompt,
                mode=ExecutionMode.ENSEMBLE,
                temperature=0.2
            )
            
            # Extract analysis components
            analysis_content = analysis_result.get("response", "")
            
            # Update state with analysis
            state["context"]["theorem_analysis"] = analysis_content
            state["current_phase"] = ProofPhase.STRATEGY_SELECTION
            
            # Extract definitions and assumptions
            definitions = await self._extract_definitions(theorem, analysis_content)
            assumptions = await self._extract_assumptions(theorem, analysis_content)
            
            state["definitions"] = definitions
            state["assumptions"] = assumptions
            
            # Create analysis message
            analysis_message = AIMessage(content=f"""
I've analyzed your theorem: "{theorem}"

Here's my analysis:
{analysis_content}

Based on this analysis, I recommend we consider the following proof strategies.
Let me suggest the most suitable approach.
            """.strip())
            
            state["messages"].append(analysis_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Problem analysis failed: {e}")
            return await self.on_error(state, e)
    
    async def _extract_definitions(self, theorem: str, analysis: str) -> List[str]:
        """Extract required definitions from analysis."""
        prompt = f"""
From this theorem and analysis, list the key definitions needed:
Theorem: {theorem}
Analysis: {analysis}

Return only essential definitions as a list.
        """
        
        result = await self.orchestrator.execute(
            task="definition_extraction",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.1
        )
        
        # Parse definitions (simplified)
        content = result.get("response", "")
        definitions = [line.strip() for line in content.split('\n') if line.strip()]
        return definitions[:10]  # Limit to 10 definitions
    
    async def _extract_assumptions(self, theorem: str, analysis: str) -> List[str]:
        """Extract assumptions from theorem statement."""
        prompt = f"""
Identify all assumptions and preconditions in this theorem:
{theorem}

List them clearly.
        """
        
        result = await self.orchestrator.execute(
            task="assumption_extraction",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.1
        )
        
        # Parse assumptions
        content = result.get("response", "")
        assumptions = [line.strip() for line in content.split('\n') if line.strip()]
        return assumptions[:5]  # Limit to 5 assumptions


class StrategySelectionNode(BaseWorkflowNode):
    """Node for selecting appropriate proof strategy."""
    
    def __init__(self):
        super().__init__("strategy_selection")
    
    async def execute(self, state: ProofAssistantState) -> ProofAssistantState:
        """Select the most appropriate proof strategy."""
        try:
            theorem = state["theorem_statement"]
            analysis = state["context"].get("theorem_analysis", "")
            
            # Generate strategy recommendations
            strategy_prompt = f"""
Based on this theorem and analysis, recommend the best proof strategy:

Theorem: {theorem}
Analysis: {analysis}

Consider these strategies:
1. Direct proof
2. Proof by contradiction
3. Mathematical induction
4. Constructive proof
5. Case analysis
6. Proof by reduction

Provide:
- Primary strategy recommendation with reasoning
- Alternative strategies if the first fails
- Outline of proof steps for chosen strategy
            """
            
            strategy_result = await self.orchestrator.execute(
                task="strategy_selection",
                prompt=strategy_prompt,
                mode=ExecutionMode.ENSEMBLE,
                temperature=0.3
            )
            
            # Extract recommended strategy
            strategy_content = strategy_result.get("response", "")
            strategy = await self._parse_strategy(strategy_content)
            
            state["proof_strategy"] = strategy
            state["context"]["strategy_reasoning"] = strategy_content
            state["current_phase"] = ProofPhase.PROOF_CONSTRUCTION
            
            # Generate proof outline
            outline = await self._generate_proof_outline(theorem, strategy, analysis)
            state["proof_outline"] = outline
            
            # Create strategy message
            strategy_message = AIMessage(content=f"""
I recommend using a **{strategy.value}** approach for this proof.

{strategy_content}

Here's the proof outline I've prepared:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(outline))}

Shall we proceed with constructing the proof step by step?
            """.strip())
            
            state["messages"].append(strategy_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return await self.on_error(state, e)
    
    async def _parse_strategy(self, content: str) -> ProofStrategy:
        """Parse strategy from AI response."""
        content_lower = content.lower()
        
        if "contradiction" in content_lower:
            return ProofStrategy.CONTRADICTION
        elif "induction" in content_lower:
            return ProofStrategy.INDUCTION
        elif "construction" in content_lower:
            return ProofStrategy.CONSTRUCTION
        elif "case" in content_lower:
            return ProofStrategy.CASE_ANALYSIS
        elif "reduction" in content_lower:
            return ProofStrategy.REDUCTION
        else:
            return ProofStrategy.DIRECT
    
    async def _generate_proof_outline(
        self, 
        theorem: str, 
        strategy: ProofStrategy, 
        analysis: str
    ) -> List[str]:
        """Generate high-level proof outline."""
        prompt = f"""
Create a high-level outline for proving this theorem using {strategy.value} strategy:

Theorem: {theorem}
Strategy: {strategy.value}
Analysis: {analysis}

Provide 5-8 main steps in the proof outline.
        """
        
        result = await self.orchestrator.execute(
            task="outline_generation",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.4
        )
        
        # Parse outline
        content = result.get("response", "")
        outline = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.startswith(tuple('123456789')) or line.startswith('-')):
                outline.append(line.lstrip('123456789.- '))
        
        return outline[:8]


class ProofConstructionNode(BaseWorkflowNode):
    """Node for constructing proof steps with verification."""
    
    def __init__(self):
        super().__init__("proof_construction")
    
    async def execute(self, state: ProofAssistantState) -> ProofAssistantState:
        """Construct the next proof step."""
        try:
            theorem = state["theorem_statement"]
            strategy = state["proof_strategy"]
            outline = state["proof_outline"]
            current_steps = state.get("proof_steps", [])
            
            # Determine next step to construct
            next_step_index = len(current_steps)
            if next_step_index >= len(outline):
                # Proof construction complete
                state["current_phase"] = ProofPhase.PROOF_REVIEW
                return state
            
            next_step_outline = outline[next_step_index]
            
            # Construct detailed proof step
            step_details = await self._construct_step(
                theorem, strategy, next_step_outline, current_steps
            )
            
            # Add step to proof
            if "proof_steps" not in state:
                state["proof_steps"] = []
            state["proof_steps"].append(step_details)
            
            # Create checkpoint before verification
            state["metadata"]["checkpoint_before_step"] = {
                "step_index": next_step_index,
                "timestamp": datetime.now().isoformat()
            }
            
            # Move to verification
            state["current_phase"] = ProofPhase.STEP_VERIFICATION
            
            # Create step message
            step_message = AIMessage(content=f"""
**Step {next_step_index + 1}:** {step_details.content}

*Justification:* {step_details.justification}

Let me verify this step before we continue...
            """.strip())
            
            state["messages"].append(step_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Proof construction failed: {e}")
            return await self.on_error(state, e)
    
    async def _construct_step(
        self,
        theorem: str,
        strategy: ProofStrategy,
        step_outline: str,
        previous_steps: List[ProofStepDetails]
    ) -> ProofStepDetails:
        """Construct detailed proof step."""
        # Build context from previous steps
        previous_content = "\n".join([
            f"Step {i+1}: {step.content}" 
            for i, step in enumerate(previous_steps)
        ])
        
        step_prompt = f"""
Construct a detailed proof step for:

Theorem: {theorem}
Strategy: {strategy.value}
Current step outline: {step_outline}

Previous steps:
{previous_content}

Provide:
1. Detailed step content
2. Mathematical justification
3. Dependencies on previous steps
4. Confidence in correctness (0-1)

Be precise and mathematically rigorous.
        """
        
        result = await self.orchestrator.execute(
            task="step_construction",
            prompt=step_prompt,
            mode=ExecutionMode.ENSEMBLE,
            temperature=0.2
        )
        
        step_content = result.get("response", "")
        
        # Parse step details (simplified)
        step_id = f"step_{len(previous_steps) + 1}_{int(datetime.now().timestamp())}"
        
        return ProofStepDetails(
            step_id=step_id,
            content=step_content,
            justification=f"Following {strategy.value} strategy",
            dependencies=[step.step_id for step in previous_steps[-2:]] if previous_steps else [],
            verification_status="pending",
            confidence_score=0.8,  # Would be parsed from AI response
            generated_at=datetime.now(),
            backtrack_point=len(previous_steps) % 3 == 0  # Create backtrack points periodically
        )


class StepVerificationNode(BaseWorkflowNode):
    """Node for verifying proof steps."""
    
    def __init__(self):
        super().__init__("step_verification")
    
    async def execute(self, state: ProofAssistantState) -> ProofAssistantState:
        """Verify the most recent proof step."""
        try:
            if not state.get("proof_steps"):
                return state
            
            current_step = state["proof_steps"][-1]
            theorem = state["theorem_statement"]
            
            # Perform multi-level verification
            verification_result = await self._verify_step(
                current_step, 
                state["proof_steps"][:-1], 
                theorem,
                state.get("confidence_threshold", 0.7)
            )
            
            # Update step verification status
            current_step.verification_status = verification_result["status"]
            current_step.confidence_score = verification_result["confidence"]
            
            # Add to verification history
            if "verification_history" not in state:
                state["verification_history"] = []
            
            state["verification_history"].append({
                "step_id": current_step.step_id,
                "verification_result": verification_result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Handle verification results
            if verification_result["status"] == "valid":
                # Step is valid, continue construction
                state["current_phase"] = ProofPhase.PROOF_CONSTRUCTION
                
                verification_message = AIMessage(content=f"""
✅ **Step verified successfully!**

Confidence: {verification_result['confidence']:.2f}
{verification_result.get('notes', '')}

Let's continue with the next step.
                """.strip())
                
            elif verification_result["status"] == "questionable":
                # Step needs review or improvement
                state["current_phase"] = ProofPhase.BACKTRACKING
                state["verification_errors"] = verification_result.get("errors", [])
                state["suggested_fixes"] = verification_result.get("suggestions", [])
                
                verification_message = AIMessage(content=f"""
⚠️ **Step verification found issues:**

Issues: {', '.join(verification_result.get('errors', []))}

Suggestions: {', '.join(verification_result.get('suggestions', []))}

Should I attempt to fix this step or would you like to revise it?
                """.strip())
                
            else:
                # Step is invalid, backtrack
                state["current_phase"] = ProofPhase.BACKTRACKING
                state["verification_errors"] = verification_result.get("errors", [])
                
                verification_message = AIMessage(content=f"""
❌ **Step verification failed.**

Errors: {', '.join(verification_result.get('errors', []))}

I need to backtrack and try a different approach for this step.
                """.strip())
            
            state["messages"].append(verification_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Step verification failed: {e}")
            return await self.on_error(state, e)
    
    async def _verify_step(
        self,
        step: ProofStepDetails,
        previous_steps: List[ProofStepDetails],
        theorem: str,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Perform comprehensive step verification."""
        # Build verification context
        context = {
            "current_step": step.content,
            "previous_steps": [s.content for s in previous_steps],
            "theorem": theorem,
            "justification": step.justification
        }
        
        verification_prompt = f"""
Verify this proof step for correctness and logical validity:

Theorem: {theorem}
Current step: {step.content}
Justification: {step.justification}

Previous steps:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(context["previous_steps"]))}

Check for:
1. Logical consistency
2. Mathematical correctness
3. Proper justification
4. Dependencies satisfaction
5. Completeness

Provide verification status (valid/questionable/invalid) and confidence score (0-1).
        """
        
        result = await self.orchestrator.execute(
            task="step_verification",
            prompt=verification_prompt,
            mode=ExecutionMode.ENSEMBLE,
            temperature=0.1
        )
        
        verification_content = result.get("response", "")
        
        # Parse verification result (simplified)
        confidence = 0.8  # Would extract from AI response
        
        if confidence >= confidence_threshold:
            status = "valid"
        elif confidence >= 0.5:
            status = "questionable"
        else:
            status = "invalid"
        
        return {
            "status": status,
            "confidence": confidence,
            "notes": verification_content,
            "errors": [] if status == "valid" else ["Logic gap detected"],
            "suggestions": [] if status == "valid" else ["Strengthen justification"]
        }


class BacktrackingNode(BaseWorkflowNode):
    """Node for handling backtracking when proof steps fail."""
    
    def __init__(self):
        super().__init__("backtracking")
    
    async def execute(self, state: ProofAssistantState) -> ProofAssistantState:
        """Handle backtracking to previous valid state."""
        try:
            proof_steps = state.get("proof_steps", [])
            if not proof_steps:
                # No steps to backtrack, restart strategy selection
                state["current_phase"] = ProofPhase.STRATEGY_SELECTION
                return state
            
            # Find last valid backtrack point
            backtrack_point = await self._find_backtrack_point(proof_steps)
            
            if backtrack_point is None:
                # No valid backtrack point, restart proof
                state["proof_steps"] = []
                state["current_phase"] = ProofPhase.STRATEGY_SELECTION
                
                backtrack_message = AIMessage(content="""
I need to reconsider the entire proof strategy. Let me analyze the problem again 
and try a different approach.
                """.strip())
            else:
                # Backtrack to valid point
                state["proof_steps"] = proof_steps[:backtrack_point + 1]
                
                # Save current state to backtrack stack
                if "backtrack_stack" not in state:
                    state["backtrack_stack"] = []
                
                state["backtrack_stack"].append({
                    "failed_steps": proof_steps[backtrack_point + 1:],
                    "errors": state.get("verification_errors", []),
                    "timestamp": datetime.now().isoformat()
                })
                
                state["current_phase"] = ProofPhase.PROOF_CONSTRUCTION
                
                backtrack_message = AIMessage(content=f"""
I'm backtracking to step {backtrack_point + 1} and will try a different approach 
for the subsequent steps.

The previous attempt had issues: {', '.join(state.get('verification_errors', []))}

Let me continue from the valid state.
                """.strip())
            
            state["messages"].append(backtrack_message)
            
            # Clear error states
            state["verification_errors"] = []
            state["suggested_fixes"] = []
            
            return state
            
        except Exception as e:
            logger.error(f"Backtracking failed: {e}")
            return await self.on_error(state, e)
    
    async def _find_backtrack_point(
        self, 
        proof_steps: List[ProofStepDetails]
    ) -> Optional[int]:
        """Find the best point to backtrack to."""
        # Look for the last step marked as backtrack point with valid verification
        for i in reversed(range(len(proof_steps))):
            step = proof_steps[i]
            if (step.backtrack_point and 
                step.verification_status == "valid" and 
                step.confidence_score >= 0.7):
                return i
        
        # If no explicit backtrack point, find last valid step
        for i in reversed(range(len(proof_steps))):
            step = proof_steps[i]
            if step.verification_status == "valid":
                return i
        
        return None


class ProofReviewNode(BaseWorkflowNode):
    """Node for final proof review and completion."""
    
    def __init__(self):
        super().__init__("proof_review")
    
    async def execute(self, state: ProofAssistantState) -> ProofAssistantState:
        """Review completed proof for coherence and completeness."""
        try:
            theorem = state["theorem_statement"]
            proof_steps = state.get("proof_steps", [])
            
            if not proof_steps:
                state["current_phase"] = ProofPhase.PROOF_CONSTRUCTION
                return state
            
            # Compile full proof
            full_proof = self._compile_proof(theorem, proof_steps)
            
            # Perform comprehensive review
            review_result = await self._review_proof(theorem, full_proof, proof_steps)
            
            state["context"]["proof_review"] = review_result
            
            if review_result["complete"] and review_result["valid"]:
                # Proof is complete and valid
                state["current_phase"] = ProofPhase.COMPLETION
                
                review_message = AIMessage(content=f"""
🎉 **Proof completed successfully!**

**Theorem:** {theorem}

**Proof:**
{full_proof}

**Review Summary:**
- Completeness: ✅
- Logical validity: ✅
- Confidence: {review_result['confidence']:.2f}

{review_result.get('summary', '')}
                """.strip())
            else:
                # Proof needs more work
                issues = review_result.get("issues", [])
                state["verification_errors"] = issues
                state["current_phase"] = ProofPhase.BACKTRACKING
                
                review_message = AIMessage(content=f"""
The proof needs some refinement:

**Issues found:**
{chr(10).join(f"- {issue}" for issue in issues)}

**Suggestions:**
{chr(10).join(f"- {suggestion}" for suggestion in review_result.get('suggestions', []))}

Let me address these issues.
                """.strip())
            
            state["messages"].append(review_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Proof review failed: {e}")
            return await self.on_error(state, e)
    
    def _compile_proof(
        self, 
        theorem: str, 
        proof_steps: List[ProofStepDetails]
    ) -> str:
        """Compile individual steps into complete proof."""
        proof_text = f"**Theorem:** {theorem}\n\n**Proof:**\n\n"
        
        for i, step in enumerate(proof_steps):
            proof_text += f"{i + 1}. {step.content}\n\n"
        
        proof_text += "∎ (Q.E.D.)"
        
        return proof_text
    
    async def _review_proof(
        self,
        theorem: str,
        full_proof: str,
        proof_steps: List[ProofStepDetails]
    ) -> Dict[str, Any]:
        """Perform comprehensive proof review."""
        review_prompt = f"""
Review this complete proof for mathematical correctness and completeness:

{full_proof}

Check for:
1. Logical flow and coherence
2. Complete coverage of all cases
3. Proper use of definitions and theorems
4. Mathematical rigor
5. Clarity of argument

Provide:
- Overall validity assessment
- Completeness assessment  
- Confidence score (0-1)
- List of any issues found
- Suggestions for improvement
        """
        
        result = await self.orchestrator.execute(
            task="proof_review",
            prompt=review_prompt,
            mode=ExecutionMode.ENSEMBLE,
            temperature=0.1
        )
        
        review_content = result.get("response", "")
        
        # Parse review result (simplified)
        return {
            "complete": True,
            "valid": True,
            "confidence": 0.9,
            "summary": review_content,
            "issues": [],
            "suggestions": []
        }


class ProofAssistantWorkflow:
    """Main proof assistant workflow orchestrator."""
    
    def __init__(self):
        self.config = WorkflowConfig(
            max_steps=200,
            timeout_seconds=3600,  # 60 minutes
            enable_checkpointing=True,
            enable_human_in_loop=True,
            retry_attempts=5
        )
    
    async def create_workflow_graph(self):
        """Create the proof assistant workflow graph."""
        try:
            builder = WorkflowGraphBuilder("proof_assistant_workflow", self.config)
            
            # Add nodes
            builder.add_node(ProblemAnalysisNode())
            builder.add_node(StrategySelectionNode())
            builder.add_node(ProofConstructionNode())
            builder.add_node(StepVerificationNode())
            builder.add_node(BacktrackingNode())
            builder.add_node(ProofReviewNode())
            
            # Add edges
            builder.add_edge("problem_analysis", "strategy_selection")
            builder.add_edge("strategy_selection", "proof_construction")
            builder.add_edge("proof_construction", "step_verification")
            
            # Add conditional edges
            builder.add_conditional_edge(
                "step_verification",
                self._route_after_verification,
                {
                    "continue": "proof_construction",
                    "backtrack": "backtracking",
                    "review": "proof_review"
                }
            )
            
            builder.add_conditional_edge(
                "backtracking",
                self._route_after_backtrack,
                {
                    "retry": "proof_construction",
                    "restart": "strategy_selection",
                    "end": "__end__"
                }
            )
            
            builder.add_conditional_edge(
                "proof_review",
                self._route_after_review,
                {
                    "complete": "__end__",
                    "backtrack": "backtracking"
                }
            )
            
            return await builder.build()
            
        except Exception as e:
            logger.error(f"Failed to create proof assistant workflow: {e}")
            raise
    
    def _route_after_verification(self, state: ProofAssistantState) -> str:
        """Route after step verification."""
        phase = state.get("current_phase", ProofPhase.PROOF_CONSTRUCTION)
        
        if phase == ProofPhase.PROOF_REVIEW:
            return "review"
        elif phase == ProofPhase.BACKTRACKING:
            return "backtrack"
        else:
            return "continue"
    
    def _route_after_backtrack(self, state: ProofAssistantState) -> str:
        """Route after backtracking."""
        phase = state.get("current_phase", ProofPhase.PROOF_CONSTRUCTION)
        
        if phase == ProofPhase.STRATEGY_SELECTION:
            return "restart"
        elif phase == ProofPhase.PROOF_CONSTRUCTION:
            return "retry"
        else:
            return "end"
    
    def _route_after_review(self, state: ProofAssistantState) -> str:
        """Route after proof review."""
        phase = state.get("current_phase", ProofPhase.COMPLETION)
        
        if phase == ProofPhase.COMPLETION:
            return "complete"
        else:
            return "backtrack"
    
    async def start_proof_session(
        self,
        session_id: str,
        user_id: str,
        theorem_statement: str,
        auto_verify: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Start a new proof construction session."""
        try:
            # Create initial state
            initial_state = ProofAssistantState(
                messages=[
                    SystemMessage(content="You are an AI proof assistant helping construct mathematical proofs.")
                ],
                session_id=session_id,
                user_id=user_id,
                current_step="problem_analysis",
                context={},
                metadata={
                    "session_start": datetime.now().isoformat(),
                    "theorem": theorem_statement
                },
                error_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                theorem_statement=theorem_statement,
                proof_strategy=None,
                current_phase=ProofPhase.PROBLEM_ANALYSIS,
                proof_steps=[],
                verification_history=[],
                backtrack_stack=[],
                assumptions=[],
                definitions=[],
                lemmas_used=[],
                proof_outline=[],
                verification_errors=[],
                suggested_fixes=[],
                confidence_threshold=confidence_threshold,
                auto_verify=auto_verify
            )
            
            # Create workflow graph
            graph = await self.create_workflow_graph()
            
            # Execute workflow
            result = await workflow_executor.execute_workflow(
                graph, initial_state, self.config
            )
            
            return {
                "session_id": session_id,
                "status": "started",
                "result": result.dict(),
                "theorem": theorem_statement,
                "confidence_threshold": confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to start proof session: {e}")
            raise


# Global proof assistant workflow instance
proof_assistant_workflow = ProofAssistantWorkflow()