"""
Structured Prompt Templating System for the Automata Learning Platform.
Provides reusable templates, variable injection, and prompt optimization.
"""
import json
import re
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from jinja2 import Template, Environment, meta
import logging
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Types of prompt templates available."""
    GENERATION = "generation"
    EXPLANATION = "explanation"
    PROOF = "proof"
    OPTIMIZATION = "optimization"
    VERIFICATION = "verification"
    TUTORING = "tutoring"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"
    REASONING = "reasoning"


class OutputFormat(str, Enum):
    """Expected output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    NATURAL_LANGUAGE = "natural_language"
    STRUCTURED = "structured"
    STEP_BY_STEP = "step_by_step"


class PromptExample(BaseModel):
    """Structure for few-shot learning examples."""
    input: str
    output: str
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptTemplate(BaseModel):
    """Base prompt template structure."""
    name: str
    type: PromptType
    template: str
    system_prompt: Optional[str] = None
    variables: List[str] = Field(default_factory=list)
    examples: List[PromptExample] = Field(default_factory=list)
    output_format: OutputFormat = OutputFormat.NATURAL_LANGUAGE
    constraints: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('variables', pre=True, always=True)
    def extract_variables(cls, v, values):
        """Extract variables from template if not provided."""
        if 'template' in values:
            env = Environment()
            ast = env.parse(values['template'])
            return list(meta.find_undeclared_variables(ast))
        return v


class PromptLibrary:
    """Library of reusable prompt templates."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize with default prompt templates."""
        
        # DFA Generation Template
        self.templates["dfa_generation"] = PromptTemplate(
            name="DFA Generation",
            type=PromptType.GENERATION,
            template="""Generate a Deterministic Finite Automaton (DFA) for the following specification:

**Language Description**: {{ language_description }}
{% if alphabet %}**Alphabet**: {{ alphabet }}{% endif %}
{% if constraints %}**Constraints**:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

Please provide the DFA in the following format:
1. States: List all states with meaningful names
2. Start State: Identify the initial state
3. Accept States: List all accepting states
4. Transitions: Provide transition function as (state, symbol) -> state
5. Explanation: Brief explanation of how the DFA works

{% if examples %}
**Example strings that should be accepted**:
{% for example in examples %}
- {{ example }}
{% endfor %}
{% endif %}

Think step by step:
1. Identify the pattern in the language
2. Determine minimum number of states needed
3. Design state transitions
4. Verify with test cases""",
            system_prompt="You are an expert in formal language theory and automata design. Generate precise and minimal DFAs.",
            output_format=OutputFormat.STRUCTURED
        )
        
        # Proof Assistant Template
        self.templates["proof_assistant"] = PromptTemplate(
            name="Proof Assistant",
            type=PromptType.PROOF,
            template="""Prove the following theorem using formal methods:

**Theorem**: {{ theorem }}
**Given**: {{ given_conditions }}
{% if proof_technique %}**Suggested Technique**: {{ proof_technique }}{% endif %}

Provide a formal proof with the following structure:

## Proof
{% if proof_technique == "induction" %}
### Base Case
[Prove for n = {{ base_case | default(0) }}]

### Inductive Hypothesis
[State what we assume for n = k]

### Inductive Step
[Prove for n = k + 1]
{% elif proof_technique == "contradiction" %}
### Assumption
[State the negation of what we want to prove]

### Derivation
[Show this leads to a contradiction]

### Conclusion
[Therefore, our original statement must be true]
{% else %}
### Direct Proof
[Step-by-step logical derivation]
{% endif %}

### Conclusion
[Final statement of what was proved]

**Verification**: Check the proof for logical consistency and completeness.""",
            system_prompt="You are a formal methods expert specializing in mathematical proofs and verification.",
            output_format=OutputFormat.STEP_BY_STEP
        )
        
        # Explanation Template
        self.templates["concept_explanation"] = PromptTemplate(
            name="Concept Explanation",
            type=PromptType.EXPLANATION,
            template="""Explain the following concept in Theory of Computation:

**Concept**: {{ concept_name }}
{% if student_level %}**Student Level**: {{ student_level }}{% endif %}
{% if context %}**Context**: {{ context }}{% endif %}

Structure your explanation as follows:

1. **Definition**: Clear, precise definition
2. **Intuition**: Informal explanation with analogies
3. **Key Properties**: Important characteristics
4. **Examples**: 
   - Simple example
   - Complex example
5. **Common Misconceptions**: What students often get wrong
6. **Applications**: Real-world uses
{% if related_concepts %}7. **Related Concepts**: {{ related_concepts | join(", ") }}{% endif %}

Use clear language appropriate for {{ student_level | default("undergraduate") }} level.""",
            system_prompt="You are an experienced computer science educator specializing in theory of computation.",
            output_format=OutputFormat.MARKDOWN
        )
        
        # Optimization Template
        self.templates["automaton_optimization"] = PromptTemplate(
            name="Automaton Optimization",
            type=PromptType.OPTIMIZATION,
            template="""Optimize the following automaton:

**Type**: {{ automaton_type }}
**Current States**: {{ num_states }}
**Transitions**: 
{{ transitions }}

**Optimization Goals**:
{% for goal in optimization_goals %}
- {{ goal }}
{% endfor %}

Perform the following optimizations:

1. **State Minimization**:
   - Identify equivalent states
   - Apply minimization algorithm
   - Show equivalence classes

2. **Transition Optimization**:
   - Remove unreachable states
   - Eliminate redundant transitions
   - Simplify transition conditions

3. **Performance Analysis**:
   - Original complexity: O(?)
   - Optimized complexity: O(?)
   - Space savings: X%
   - Time savings: Y%

4. **Verification**:
   - Prove equivalence with original
   - Test with sample inputs

Output the optimized automaton in the same format as input.""",
            system_prompt="You are an expert in automata theory optimization and algorithm efficiency.",
            output_format=OutputFormat.STRUCTURED
        )
        
        # Chain of Thought Template
        self.templates["chain_of_thought"] = PromptTemplate(
            name="Chain of Thought Reasoning",
            type=PromptType.CHAIN_OF_THOUGHT,
            template="""Let's solve this step-by-step.

**Problem**: {{ problem }}

## Thinking Process

### Step 1: Understanding the Problem
- What are we asked to find?
- What information is given?
- What are the constraints?

### Step 2: Planning the Approach
- What technique should we use?
- Why is this approach suitable?
- What are the key insights?

### Step 3: Detailed Solution
{% if solution_steps %}
{% for step in solution_steps %}
{{ loop.index }}. {{ step }}
{% endfor %}
{% else %}
[Work through the solution methodically]
{% endif %}

### Step 4: Verification
- Check the solution against requirements
- Test with examples
- Identify edge cases

### Step 5: Conclusion
- Final answer: 
- Key takeaways:
- Alternative approaches:""",
            system_prompt="Think step by step through complex problems. Show your reasoning clearly.",
            output_format=OutputFormat.STEP_BY_STEP
        )
        
        # Few-Shot Learning Template
        self.templates["few_shot_learning"] = PromptTemplate(
            name="Few-Shot Learning",
            type=PromptType.FEW_SHOT,
            template="""Learn from these examples and solve the new problem:

{% for example in examples %}
### Example {{ loop.index }}
**Input**: {{ example.input }}
**Output**: {{ example.output }}
{% if example.explanation %}**Explanation**: {{ example.explanation }}{% endif %}

{% endfor %}

### New Problem
**Input**: {{ problem_input }}
**Output**: [Apply the pattern learned from examples]

Reasoning:
1. Pattern identified from examples:
2. How it applies to new problem:
3. Step-by-step solution:
4. Final answer:""",
            system_prompt="Learn patterns from examples and apply them to new problems.",
            output_format=OutputFormat.STRUCTURED
        )
        
        logger.info(f"Initialized {len(self.templates)} default prompt templates")
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add a new template to the library."""
        self.templates[template.name] = template
        logger.info(f"Added template: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Retrieve a template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())


class PromptBuilder:
    """Builder for constructing and optimizing prompts."""
    
    def __init__(self, library: Optional[PromptLibrary] = None):
        self.library = library or PromptLibrary()
        self.env = Environment()
    
    def build(
        self,
        template_name: str,
        variables: Dict[str, Any],
        examples: Optional[List[PromptExample]] = None,
        optimize: bool = True
    ) -> str:
        """
        Build a prompt from a template with variable injection.
        
        Args:
            template_name: Name of the template to use
            variables: Variables to inject into the template
            examples: Optional examples for few-shot learning
            optimize: Whether to optimize the prompt
        
        Returns:
            Constructed prompt string
        """
        template_obj = self.library.get_template(template_name)
        if not template_obj:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Add examples if provided
        if examples:
            variables['examples'] = examples
        elif template_obj.examples:
            variables['examples'] = template_obj.examples
        
        # Render the template
        template = self.env.from_string(template_obj.template)
        prompt = template.render(**variables)
        
        if optimize:
            prompt = self._optimize_prompt(prompt, template_obj)
        
        # Add system prompt if available
        if template_obj.system_prompt:
            prompt = f"System: {template_obj.system_prompt}\n\n{prompt}"
        
        return prompt
    
    def _optimize_prompt(self, prompt: str, template: PromptTemplate) -> str:
        """
        Optimize prompt for better model performance.
        
        Args:
            prompt: Raw prompt string
            template: Template object with metadata
        
        Returns:
            Optimized prompt string
        """
        # Remove excessive whitespace
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)
        prompt = re.sub(r' {2,}', ' ', prompt)
        
        # Add output format instructions if needed
        if template.output_format == OutputFormat.JSON:
            prompt += "\n\nProvide your response in valid JSON format."
        elif template.output_format == OutputFormat.CODE:
            prompt += "\n\nProvide executable code with proper syntax."
        elif template.output_format == OutputFormat.STEP_BY_STEP:
            prompt += "\n\nNumber each step clearly and provide detailed explanations."
        
        # Add constraints if specified
        if template.constraints:
            constraints_text = "\n".join(f"- {c}" for c in template.constraints)
            prompt += f"\n\nConstraints:\n{constraints_text}"
        
        return prompt.strip()
    
    def create_chain_of_thought(
        self,
        problem: str,
        steps: Optional[List[str]] = None
    ) -> str:
        """Create a chain-of-thought prompt for complex reasoning."""
        return self.build(
            "chain_of_thought",
            {"problem": problem, "solution_steps": steps}
        )
    
    def create_few_shot(
        self,
        problem: str,
        examples: List[PromptExample]
    ) -> str:
        """Create a few-shot learning prompt."""
        return self.build(
            "few_shot_learning",
            {"problem_input": problem},
            examples=examples
        )


class PromptOptimizer:
    """Optimize prompts for specific models and tasks."""
    
    @staticmethod
    def optimize_for_model(
        prompt: str,
        model_name: str,
        max_tokens: int = 2048
    ) -> str:
        """
        Optimize prompt for a specific model.
        
        Args:
            prompt: Original prompt
            model_name: Target model name
            max_tokens: Maximum token limit
        
        Returns:
            Optimized prompt string
        """
        # Model-specific optimizations
        if "codellama" in model_name.lower():
            # CodeLlama prefers code-style formatting
            prompt = prompt.replace("**", "").replace("*", "")
            prompt = f"# Task\n{prompt}"
        elif "llama" in model_name.lower():
            # Llama models work well with clear sections
            prompt = prompt.replace("##", "\n###")
        elif "deepseek" in model_name.lower():
            # DeepSeek prefers structured format
            prompt = f"### Instruction ###\n{prompt}\n### Response ###"
        
        # Truncate if too long (rough estimation)
        estimated_tokens = len(prompt.split()) * 1.3
        if estimated_tokens > max_tokens * 0.8:
            # Keep most important parts
            lines = prompt.split('\n')
            important_lines = [l for l in lines if any(
                keyword in l.lower() for keyword in 
                ['task', 'problem', 'input', 'output', 'goal']
            )]
            prompt = '\n'.join(important_lines[:int(max_tokens * 0.6 / 1.3)])
        
        return prompt
    
    @staticmethod
    def add_reasoning_structure(prompt: str) -> str:
        """Add reasoning structure to improve output quality."""
        reasoning_suffix = """
Before providing your final answer:
1. Identify key requirements
2. Consider edge cases
3. Verify your solution
4. Explain your reasoning
"""
        return f"{prompt}\n{reasoning_suffix}"


# Global instance
prompt_library = PromptLibrary()
prompt_builder = PromptBuilder(prompt_library)


def get_prompt(
    template_name: str,
    **variables
) -> str:
    """
    Convenience function to get a formatted prompt.
    
    Args:
        template_name: Name of the template
        **variables: Variables to inject
    
    Returns:
        Formatted prompt string
    """
    return prompt_builder.build(template_name, variables)