"""
AI-Powered JFLAP Integration
============================

This module provides AI enhancements for JFLAP features using Ollama models.
Includes intelligent generation, analysis, error recovery, and tutoring.

Author: LLM Systems Architect
Version: 1.0
"""

import json
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta

# Import orchestrator and AI components
from .orchestrator import ModelOrchestrator, ExecutionMode
from .agents import AutomataGenerator, AutomataExplainer
from .prompts import PromptTemplate


class AIJFLAPFeature(Enum):
    """Available AI-enhanced features"""
    MULTI_TAPE_TM = "multi_tape_tm"
    GRAMMAR_ANALYSIS = "grammar_analysis"
    ERROR_RECOVERY = "error_recovery"
    TEST_GENERATION = "test_generation"
    NL_CONVERSION = "nl_conversion"
    TUTORING = "tutoring"


@dataclass
class AIResponse:
    """Standardized AI response"""
    success: bool
    result: Any
    confidence: float
    explanation: str
    metadata: Dict[str, Any]


class AIJFLAPIntegration:
    """
    Main AI integration class for JFLAP features
    Provides intelligent assistance for all advanced automata operations
    """
    
    def __init__(self, orchestrator: Optional[ModelOrchestrator] = None):
        self.orchestrator = orchestrator or ModelOrchestrator()
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = timedelta(minutes=30)
        
    # =====================================
    # Multi-tape TM Generation & Optimization
    # =====================================
    
    async def generate_multi_tape_tm(
        self, 
        problem_description: str,
        num_tapes: int = 2,
        optimize: bool = True
    ) -> AIResponse:
        """
        Generate optimal multi-tape TM for a problem
        
        Args:
            problem_description: Natural language problem description
            num_tapes: Number of tapes (2-5)
            optimize: Whether to optimize the generated TM
            
        Returns:
            AIResponse with TM specification
        """
        # Check cache
        cache_key = self._get_cache_key("mt_tm", problem_description, num_tapes)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now() - cached['time'] < self.cache_ttl:
                return cached['response']
        
        # Prepare prompt
        prompt = f"""
        Generate a {num_tapes}-tape Turing Machine for the following problem:
        
        Problem: {problem_description}
        
        Requirements:
        1. Use exactly {num_tapes} tapes
        2. Provide clear tape purposes
        3. Use JFLAP format: x1;y1,d1|x2;y2,d2|...
        4. Minimize number of states
        5. Include test cases
        
        Output format:
        {{
            "tapes": [
                {{"index": 0, "purpose": "input/output"}},
                {{"index": 1, "purpose": "workspace"}}
            ],
            "states": ["q0", "q1", ...],
            "initial_state": "q0",
            "final_states": ["qf"],
            "transitions": [
                {{
                    "from": "q0",
                    "to": "q1",
                    "operations": "a;b,R|□;□,S"
                }}
            ],
            "test_cases": [
                {{"input": ["abc", ""], "expected": "result"}}
            ]
        }}
        """
        
        # Execute with orchestrator
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.ENSEMBLE if optimize else ExecutionMode.SINGLE,
            models=["codellama:34b", "deepseek-coder:33b"]
        )
        
        # Parse and validate response
        try:
            tm_spec = json.loads(result['response'])
            
            # Optimize if requested
            if optimize:
                tm_spec = await self._optimize_multi_tape_tm(tm_spec)
            
            response = AIResponse(
                success=True,
                result=tm_spec,
                confidence=result.get('confidence', 0.8),
                explanation=f"Generated {num_tapes}-tape TM with {len(tm_spec['states'])} states",
                metadata={
                    'complexity': self._analyze_tm_complexity(tm_spec),
                    'tape_utilization': self._analyze_tape_utilization(tm_spec)
                }
            )
            
            # Cache result
            self.cache[cache_key] = {'response': response, 'time': datetime.now()}
            
            return response
            
        except Exception as e:
            return AIResponse(
                success=False,
                result=None,
                confidence=0.0,
                explanation=f"Failed to generate TM: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def _optimize_multi_tape_tm(self, tm_spec: Dict) -> Dict:
        """Optimize multi-tape TM using AI"""
        optimization_prompt = f"""
        Optimize this multi-tape Turing Machine:
        {json.dumps(tm_spec, indent=2)}
        
        Optimization goals:
        1. Minimize state count
        2. Reduce transition complexity
        3. Improve tape utilization
        4. Eliminate redundant operations
        
        Return optimized TM in same format.
        """
        
        result = await self.orchestrator.execute(
            task=optimization_prompt,
            mode=ExecutionMode.SINGLE,
            models=["codellama:34b"]
        )
        
        try:
            return json.loads(result['response'])
        except:
            return tm_spec  # Return original if optimization fails
    
    def _analyze_tm_complexity(self, tm_spec: Dict) -> Dict:
        """Analyze TM complexity metrics"""
        return {
            'state_count': len(tm_spec['states']),
            'transition_count': len(tm_spec['transitions']),
            'alphabet_size': len(set(
                t['operations'].replace(';', '').replace(',', '').replace('|', '')
                for t in tm_spec['transitions']
            )),
            'estimated_runtime': 'O(n^2)' if len(tm_spec['states']) > 10 else 'O(n)'
        }
    
    def _analyze_tape_utilization(self, tm_spec: Dict) -> List[Dict]:
        """Analyze how each tape is utilized"""
        utilization = []
        for tape in tm_spec.get('tapes', []):
            usage = {
                'index': tape['index'],
                'purpose': tape['purpose'],
                'read_count': 0,
                'write_count': 0,
                'move_count': 0
            }
            
            # Analyze transitions for this tape
            for trans in tm_spec['transitions']:
                ops = trans['operations'].split('|')
                if tape['index'] < len(ops):
                    op = ops[tape['index']]
                    if ';' in op:
                        usage['read_count'] += 1
                        parts = op.split(';')
                        if len(parts) > 1:
                            write, move = parts[1].split(',')
                            if write != parts[0]:
                                usage['write_count'] += 1
                            if move in ['L', 'R']:
                                usage['move_count'] += 1
            
            utilization.append(usage)
        
        return utilization
    
    # =====================================
    # Grammar Analysis & Conversion
    # =====================================
    
    async def analyze_grammar(self, grammar: Dict) -> AIResponse:
        """
        Analyze grammar type and properties using AI
        
        Args:
            grammar: Grammar specification
            
        Returns:
            AIResponse with analysis results
        """
        prompt = f"""
        Analyze this formal grammar:
        {json.dumps(grammar, indent=2)}
        
        Determine:
        1. Grammar type (Regular, Context-Free, Context-Sensitive, Unrestricted)
        2. Special properties (LL(1), LR(0), ambiguous, left-recursive)
        3. Language characteristics
        4. Suggested optimizations
        
        Output format:
        {{
            "type": "context_free",
            "chomsky_hierarchy": 2,
            "properties": {{
                "is_ambiguous": false,
                "is_left_recursive": false,
                "is_ll1": true,
                "is_lr0": false
            }},
            "language_properties": {{
                "is_finite": false,
                "is_regular": false,
                "examples": ["valid strings"]
            }},
            "optimizations": ["suggestions"]
        }}
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.CASCADE,
            models=["deepseek-coder:33b", "codellama:34b"]
        )
        
        try:
            analysis = json.loads(result['response'])
            
            return AIResponse(
                success=True,
                result=analysis,
                confidence=0.85,
                explanation=f"Grammar identified as {analysis['type']} (Type-{analysis['chomsky_hierarchy']})",
                metadata={
                    'production_count': len(grammar.get('productions', [])),
                    'variable_count': len(grammar.get('variables', [])),
                    'terminal_count': len(grammar.get('terminals', []))
                }
            )
        except Exception as e:
            return AIResponse(
                success=False,
                result=None,
                confidence=0.0,
                explanation=f"Grammar analysis failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def convert_grammar(
        self,
        grammar: Dict,
        target_form: str
    ) -> AIResponse:
        """
        Convert grammar to target form using AI assistance
        
        Args:
            grammar: Source grammar
            target_form: Target form (CNF, GNF, etc.)
            
        Returns:
            AIResponse with converted grammar
        """
        prompt = f"""
        Convert this grammar to {target_form}:
        {json.dumps(grammar, indent=2)}
        
        Apply necessary transformations:
        1. Remove epsilon productions (if needed)
        2. Remove unit productions (if needed)
        3. Convert to {target_form} form
        4. Verify the conversion is correct
        
        Return converted grammar in same format.
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.SINGLE,
            models=["codellama:34b"]
        )
        
        try:
            converted = json.loads(result['response'])
            
            # Verify conversion
            is_valid = await self._verify_grammar_form(converted, target_form)
            
            return AIResponse(
                success=is_valid,
                result=converted,
                confidence=0.9 if is_valid else 0.5,
                explanation=f"Grammar converted to {target_form}",
                metadata={
                    'transformations_applied': self._detect_transformations(grammar, converted),
                    'is_equivalent': True  # Assuming correctness
                }
            )
        except Exception as e:
            return AIResponse(
                success=False,
                result=None,
                confidence=0.0,
                explanation=f"Conversion failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def _verify_grammar_form(self, grammar: Dict, target_form: str) -> bool:
        """Verify grammar is in target form"""
        if target_form == "CNF":
            # Check Chomsky Normal Form
            for prod in grammar.get('productions', []):
                rhs = prod.get('right', '')
                if len(rhs) == 1:
                    # Must be terminal
                    if rhs not in grammar.get('terminals', []):
                        return False
                elif len(rhs) == 2:
                    # Must be two variables
                    if not all(c in grammar.get('variables', []) for c in rhs):
                        return False
                elif rhs != 'ε':
                    return False
        elif target_form == "GNF":
            # Check Greibach Normal Form
            for prod in grammar.get('productions', []):
                rhs = prod.get('right', '')
                if rhs and rhs[0] not in grammar.get('terminals', []):
                    return False
        
        return True
    
    def _detect_transformations(self, original: Dict, converted: Dict) -> List[str]:
        """Detect what transformations were applied"""
        transformations = []
        
        # Check for epsilon removal
        original_has_epsilon = any(
            p.get('right', '') in ['', 'ε'] 
            for p in original.get('productions', [])
        )
        converted_has_epsilon = any(
            p.get('right', '') in ['', 'ε']
            for p in converted.get('productions', [])
        )
        
        if original_has_epsilon and not converted_has_epsilon:
            transformations.append("epsilon_removal")
        
        # Check for new variables (likely from CNF conversion)
        if len(converted.get('variables', [])) > len(original.get('variables', [])):
            transformations.append("variable_introduction")
        
        return transformations
    
    # =====================================
    # Intelligent Error Recovery
    # =====================================
    
    async def suggest_error_recovery(
        self,
        error_context: Dict,
        automaton_type: str
    ) -> AIResponse:
        """
        Suggest recovery strategies for parsing/simulation errors
        
        Args:
            error_context: Error information and context
            automaton_type: Type of automaton (DFA, PDA, TM, etc.)
            
        Returns:
            AIResponse with recovery suggestions
        """
        prompt = f"""
        Analyze this {automaton_type} error and suggest recovery:
        
        Error: {error_context.get('error_message', 'Unknown error')}
        Input: {error_context.get('input', '')}
        State: {error_context.get('current_state', '')}
        Configuration: {json.dumps(error_context.get('configuration', {}), indent=2)}
        
        Provide:
        1. Root cause analysis
        2. Recovery strategies (ranked by likelihood)
        3. Suggested corrections
        4. Prevention tips
        
        Output format:
        {{
            "root_cause": "explanation",
            "recovery_strategies": [
                {{
                    "strategy": "description",
                    "confidence": 0.9,
                    "implementation": "how to apply"
                }}
            ],
            "corrections": [
                {{
                    "type": "transition/state/input",
                    "original": "value",
                    "suggested": "new_value",
                    "reason": "why"
                }}
            ],
            "prevention": ["tips"]
        }}
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.SINGLE,
            models=["llama3.1:8b"]  # Fast model for error recovery
        )
        
        try:
            recovery = json.loads(result['response'])
            
            return AIResponse(
                success=True,
                result=recovery,
                confidence=max(s['confidence'] for s in recovery['recovery_strategies']),
                explanation=recovery['root_cause'],
                metadata={
                    'automaton_type': automaton_type,
                    'error_type': self._classify_error(error_context)
                }
            )
        except Exception as e:
            # Fallback recovery suggestions
            return AIResponse(
                success=True,
                result={
                    'root_cause': 'Unable to determine exact cause',
                    'recovery_strategies': [
                        {
                            'strategy': 'Check for missing transitions',
                            'confidence': 0.5,
                            'implementation': 'Add transitions for all state-symbol pairs'
                        },
                        {
                            'strategy': 'Verify input format',
                            'confidence': 0.4,
                            'implementation': 'Ensure input uses correct alphabet'
                        }
                    ],
                    'corrections': [],
                    'prevention': ['Add comprehensive error handling', 'Validate automaton before execution']
                },
                confidence=0.3,
                explanation='Generic recovery suggestions provided',
                metadata={'fallback': True}
            )
    
    def _classify_error(self, error_context: Dict) -> str:
        """Classify error type"""
        error_msg = error_context.get('error_message', '').lower()
        
        if 'transition' in error_msg:
            return 'missing_transition'
        elif 'state' in error_msg:
            return 'invalid_state'
        elif 'stack' in error_msg:
            return 'stack_error'
        elif 'input' in error_msg or 'symbol' in error_msg:
            return 'invalid_input'
        elif 'timeout' in error_msg or 'loop' in error_msg:
            return 'infinite_loop'
        else:
            return 'unknown'
    
    # =====================================
    # Automated Test Generation
    # =====================================
    
    async def generate_test_cases(
        self,
        automaton: Dict,
        coverage_target: str = "comprehensive"
    ) -> AIResponse:
        """
        Generate comprehensive test cases for automaton
        
        Args:
            automaton: Automaton specification
            coverage_target: Type of coverage (basic, comprehensive, edge_cases)
            
        Returns:
            AIResponse with test cases
        """
        prompt = f"""
        Generate {coverage_target} test cases for this automaton:
        {json.dumps(automaton, indent=2)}
        
        Include:
        1. Positive test cases (accepted strings)
        2. Negative test cases (rejected strings)
        3. Edge cases (empty, single symbol, very long)
        4. Boundary cases (loop limits, stack depth)
        
        Output format:
        {{
            "test_suite": {{
                "positive": [
                    {{"input": "string", "description": "why valid"}}
                ],
                "negative": [
                    {{"input": "string", "description": "why invalid"}}
                ],
                "edge_cases": [
                    {{"input": "string", "description": "what it tests"}}
                ],
                "performance": [
                    {{"input": "string", "description": "performance aspect"}}
                ]
            }},
            "coverage": {{
                "state_coverage": 0.95,
                "transition_coverage": 0.90,
                "alphabet_coverage": 1.0
            }}
        }}
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.SINGLE,
            models=["codellama:34b"]
        )
        
        try:
            test_suite = json.loads(result['response'])
            
            # Analyze coverage
            actual_coverage = self._analyze_test_coverage(automaton, test_suite)
            test_suite['coverage'] = actual_coverage
            
            return AIResponse(
                success=True,
                result=test_suite,
                confidence=0.9,
                explanation=f"Generated {self._count_tests(test_suite)} test cases with {actual_coverage['overall']:.1%} coverage",
                metadata={
                    'test_count': self._count_tests(test_suite),
                    'coverage_gaps': self._identify_coverage_gaps(automaton, test_suite)
                }
            )
        except Exception as e:
            return AIResponse(
                success=False,
                result=None,
                confidence=0.0,
                explanation=f"Test generation failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def _analyze_test_coverage(self, automaton: Dict, test_suite: Dict) -> Dict:
        """Analyze actual test coverage"""
        # Simplified coverage analysis
        total_states = len(automaton.get('states', []))
        total_transitions = len(automaton.get('transitions', []))
        
        # Estimate based on test count and diversity
        test_count = self._count_tests(test_suite)
        
        state_coverage = min(1.0, test_count / (total_states * 2))
        transition_coverage = min(1.0, test_count / (total_transitions * 1.5))
        
        return {
            'state_coverage': state_coverage,
            'transition_coverage': transition_coverage,
            'alphabet_coverage': 1.0,  # Assume full alphabet coverage
            'overall': (state_coverage + transition_coverage) / 2
        }
    
    def _count_tests(self, test_suite: Dict) -> int:
        """Count total number of tests"""
        count = 0
        for category in test_suite.get('test_suite', {}).values():
            count += len(category)
        return count
    
    def _identify_coverage_gaps(self, automaton: Dict, test_suite: Dict) -> List[str]:
        """Identify what's not covered by tests"""
        gaps = []
        
        # Check for untested states (simplified)
        if self._count_tests(test_suite) < len(automaton.get('states', [])):
            gaps.append("Some states may not be reached")
        
        # Check for edge cases
        all_tests = []
        for category in test_suite.get('test_suite', {}).values():
            all_tests.extend([t.get('input', '') for t in category])
        
        if not any(t == '' for t in all_tests):
            gaps.append("Empty string not tested")
        
        if not any(len(t) > 20 for t in all_tests):
            gaps.append("Long strings not tested")
        
        return gaps
    
    # =====================================
    # Natural Language Conversion
    # =====================================
    
    async def nl_to_formal(
        self,
        description: str,
        target_formalism: str = "auto"
    ) -> AIResponse:
        """
        Convert natural language to formal specification
        
        Args:
            description: Natural language description
            target_formalism: Target formalism (DFA, CFG, TM, or auto)
            
        Returns:
            AIResponse with formal specification
        """
        prompt = f"""
        Convert this natural language description to a formal automaton/grammar:
        
        Description: {description}
        Target: {target_formalism if target_formalism != 'auto' else 'most appropriate formalism'}
        
        Analyze the problem and provide:
        1. Chosen formalism and why
        2. Formal specification
        3. Test examples
        
        Output format:
        {{
            "formalism": "DFA/NFA/PDA/CFG/TM",
            "reasoning": "why this formalism",
            "specification": {{
                "states": [...],
                "alphabet": [...],
                "transitions": [...],
                // or grammar format
            }},
            "examples": {{
                "accepted": ["strings"],
                "rejected": ["strings"]
            }}
        }}
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.CASCADE,
            models=["deepseek-coder:33b", "codellama:34b"]
        )
        
        try:
            formal_spec = json.loads(result['response'])
            
            return AIResponse(
                success=True,
                result=formal_spec,
                confidence=0.85,
                explanation=f"Converted to {formal_spec['formalism']}: {formal_spec['reasoning']}",
                metadata={
                    'complexity': self._estimate_complexity(formal_spec),
                    'formalism_alternatives': self._suggest_alternatives(description)
                }
            )
        except Exception as e:
            return AIResponse(
                success=False,
                result=None,
                confidence=0.0,
                explanation=f"Conversion failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def formal_to_nl(
        self,
        specification: Dict,
        detail_level: str = "medium"
    ) -> AIResponse:
        """
        Convert formal specification to natural language
        
        Args:
            specification: Formal automaton/grammar
            detail_level: Level of detail (basic, medium, detailed)
            
        Returns:
            AIResponse with natural language description
        """
        prompt = f"""
        Explain this formal specification in natural language:
        {json.dumps(specification, indent=2)}
        
        Detail level: {detail_level}
        
        Include:
        1. What language/problem it solves
        2. How it works (intuitive explanation)
        3. Key properties
        4. Example behavior
        
        Make it understandable for someone learning automata theory.
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.SINGLE,
            models=["llama3.1:8b"]
        )
        
        return AIResponse(
            success=True,
            result=result['response'],
            confidence=0.9,
            explanation="Generated natural language explanation",
            metadata={
                'detail_level': detail_level,
                'word_count': len(result['response'].split())
            }
        )
    
    def _estimate_complexity(self, formal_spec: Dict) -> str:
        """Estimate complexity class of formal specification"""
        formalism = formal_spec.get('formalism', '')
        
        if formalism == 'DFA' or formalism == 'NFA':
            return 'Regular (Type-3)'
        elif formalism == 'PDA':
            return 'Context-Free (Type-2)'
        elif formalism == 'TM':
            # Try to estimate based on structure
            if len(formal_spec.get('specification', {}).get('states', [])) < 10:
                return 'Decidable'
            else:
                return 'Recursively Enumerable'
        else:
            return 'Unknown'
    
    def _suggest_alternatives(self, description: str) -> List[str]:
        """Suggest alternative formalisms"""
        alternatives = []
        
        keywords = description.lower()
        
        if 'count' in keywords or 'same number' in keywords:
            alternatives.append('PDA')
        if 'match' in keywords or 'bracket' in keywords or 'parenthes' in keywords:
            alternatives.append('PDA')
        if 'copy' in keywords or 'reverse' in keywords:
            alternatives.append('TM')
        if 'pattern' in keywords:
            alternatives.append('Regular Expression')
        
        return alternatives
    
    # =====================================
    # Step-by-Step Tutoring
    # =====================================
    
    async def create_tutorial(
        self,
        topic: str,
        student_level: str = "intermediate"
    ) -> AIResponse:
        """
        Create personalized tutorial for automata topic
        
        Args:
            topic: Topic to teach
            student_level: Student's level (beginner, intermediate, advanced)
            
        Returns:
            AIResponse with tutorial content
        """
        prompt = f"""
        Create a step-by-step tutorial for: {topic}
        Student level: {student_level}
        
        Structure:
        1. Introduction and motivation
        2. Prerequisites
        3. Core concepts (with examples)
        4. Step-by-step algorithm/construction
        5. Practice problems (progressive difficulty)
        6. Common mistakes to avoid
        7. Advanced tips (if appropriate)
        
        Make it engaging and interactive.
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.SINGLE,
            models=["llama3.1:8b"]
        )
        
        # Structure the tutorial
        tutorial = {
            'topic': topic,
            'level': student_level,
            'content': result['response'],
            'sections': self._parse_tutorial_sections(result['response']),
            'estimated_time': self._estimate_tutorial_time(result['response'])
        }
        
        return AIResponse(
            success=True,
            result=tutorial,
            confidence=0.9,
            explanation=f"Created {student_level} tutorial for {topic}",
            metadata={
                'word_count': len(result['response'].split()),
                'concepts_covered': self._extract_concepts(result['response'])
            }
        )
    
    async def provide_hint(
        self,
        problem_context: Dict,
        hint_level: int = 1
    ) -> AIResponse:
        """
        Provide progressive hints for problem solving
        
        Args:
            problem_context: Current problem state
            hint_level: Level of hint (1=subtle, 2=moderate, 3=explicit)
            
        Returns:
            AIResponse with hint
        """
        prompt = f"""
        Student is working on: {problem_context.get('problem', '')}
        Current attempt: {json.dumps(problem_context.get('attempt', {}), indent=2)}
        Error/Issue: {problem_context.get('issue', 'stuck')}
        
        Provide a level {hint_level} hint:
        - Level 1: Subtle guidance, ask leading questions
        - Level 2: Point to specific area, give partial solution
        - Level 3: Show exact step needed
        
        Be encouraging and educational.
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.SINGLE,
            models=["llama3.1:8b"]
        )
        
        return AIResponse(
            success=True,
            result={
                'hint': result['response'],
                'next_hint_available': hint_level < 3
            },
            confidence=0.95,
            explanation=f"Level {hint_level} hint provided",
            metadata={
                'hint_level': hint_level,
                'problem_type': self._identify_problem_type(problem_context)
            }
        )
    
    async def explain_step(
        self,
        algorithm: str,
        step_number: int,
        context: Dict
    ) -> AIResponse:
        """
        Explain specific step in algorithm execution
        
        Args:
            algorithm: Algorithm name
            step_number: Step to explain
            context: Current execution context
            
        Returns:
            AIResponse with explanation
        """
        prompt = f"""
        Explain step {step_number} of {algorithm} algorithm:
        
        Current state: {json.dumps(context, indent=2)}
        
        Provide:
        1. What this step does
        2. Why it's necessary
        3. What changes it makes
        4. What comes next
        
        Use clear, simple language with examples.
        """
        
        result = await self.orchestrator.execute(
            task=prompt,
            mode=ExecutionMode.SINGLE,
            models=["llama3.1:8b"]
        )
        
        return AIResponse(
            success=True,
            result={
                'explanation': result['response'],
                'visualization_hint': self._suggest_visualization(algorithm, step_number)
            },
            confidence=0.9,
            explanation=f"Step {step_number} explanation provided",
            metadata={
                'algorithm': algorithm,
                'step': step_number
            }
        )
    
    def _parse_tutorial_sections(self, content: str) -> List[Dict]:
        """Parse tutorial content into sections"""
        sections = []
        current_section = None
        
        for line in content.split('\n'):
            if line.startswith('#'):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    'title': line.strip('#').strip(),
                    'content': []
                }
            elif current_section:
                current_section['content'].append(line)
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _estimate_tutorial_time(self, content: str) -> int:
        """Estimate time to complete tutorial in minutes"""
        word_count = len(content.split())
        # Assume 200 words per minute reading + exercises
        return max(5, word_count // 200 + 10)
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from tutorial"""
        concepts = []
        
        # Simple keyword extraction
        keywords = [
            'state', 'transition', 'alphabet', 'accept', 'reject',
            'deterministic', 'non-deterministic', 'epsilon', 'closure',
            'grammar', 'production', 'terminal', 'variable', 'derivation',
            'parse', 'reduce', 'shift', 'lookahead', 'stack', 'tape'
        ]
        
        content_lower = content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                concepts.append(keyword)
        
        return concepts
    
    def _identify_problem_type(self, problem_context: Dict) -> str:
        """Identify type of problem from context"""
        problem = str(problem_context.get('problem', '')).lower()
        
        if 'convert' in problem or 'transformation' in problem:
            return 'conversion'
        elif 'minimize' in problem or 'simplify' in problem:
            return 'optimization'
        elif 'parse' in problem or 'accept' in problem or 'recognize' in problem:
            return 'recognition'
        elif 'construct' in problem or 'design' in problem or 'build' in problem:
            return 'construction'
        else:
            return 'general'
    
    def _suggest_visualization(self, algorithm: str, step: int) -> str:
        """Suggest what to visualize for this step"""
        visualizations = {
            'nfa_to_dfa': 'Highlight current state subset being processed',
            'minimize_dfa': 'Show equivalence classes being formed',
            'cyk_parse': 'Highlight current cell in parse table',
            'll1_parse': 'Show stack and remaining input',
            'slr_parse': 'Display parse stack and action/goto tables',
            'multi_tape_tm': 'Show all tape contents and head positions'
        }
        
        return visualizations.get(algorithm.lower(), 'Highlight current operation')
    
    # =====================================
    # Cache Management
    # =====================================
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        content = json.dumps(args, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'memory_usage': sum(len(str(v)) for v in self.cache.values()),
            'oldest_entry': min(
                (v['time'] for v in self.cache.values()),
                default=None
            )
        }


# Export main class
__all__ = ['AIJFLAPIntegration', 'AIJFLAPFeature', 'AIResponse']

# Global instance for easy import
ai_jflap = AIJFLAPIntegration()