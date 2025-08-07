"""
Natural Language Problem Understanding System
Parses and understands Theory of Computation problems from natural language.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field

from .ai_config import AIConfig, ModelType

logger = logging.getLogger(__name__)


class ProblemType(str, Enum):
    """Types of Theory of Computation problems."""
    DFA_CONSTRUCTION = "dfa_construction"
    NFA_CONSTRUCTION = "nfa_construction"
    REGEX_TO_AUTOMATON = "regex_to_automaton"
    AUTOMATON_TO_REGEX = "automaton_to_regex"
    DFA_MINIMIZATION = "dfa_minimization"
    NFA_TO_DFA = "nfa_to_dfa"
    CFG_CONSTRUCTION = "cfg_construction"
    PDA_CONSTRUCTION = "pda_construction"
    TM_CONSTRUCTION = "tm_construction"
    PUMPING_LEMMA_PROOF = "pumping_lemma_proof"
    CLOSURE_PROOF = "closure_proof"
    DECIDABILITY = "decidability"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    LANGUAGE_EQUIVALENCE = "language_equivalence"
    SYNTHESIS = "synthesis_from_examples"
    OPTIMIZATION = "optimization"
    VERIFICATION = "verification"
    TRANSFORMATION = "transformation"
    PROOF = "general_proof"
    UNKNOWN = "unknown"


class LanguagePattern(str, Enum):
    """Common language patterns in TOC."""
    EVEN_ODD = "even_odd_count"
    DIVISIBILITY = "divisibility"
    SUBSTRING = "substring_pattern"
    PREFIX_SUFFIX = "prefix_suffix"
    BALANCED = "balanced_symbols"
    PALINDROME = "palindrome"
    ARITHMETIC = "arithmetic_expression"
    CONTEXT_SENSITIVE = "context_sensitive"
    RECURSIVE = "recursive_pattern"


@dataclass
class ProblemRequirements:
    """Extracted requirements from a problem statement."""
    problem_type: ProblemType
    alphabet: Set[str]
    constraints: List[str]
    requirements: Dict[str, Any]
    examples: Dict[str, List[str]]  # positive and negative examples
    formal_language: Optional[str] = None
    complexity_class: Optional[str] = None
    original_statement: str = ""
    keywords: List[str] = field(default_factory=list)
    patterns: List[LanguagePattern] = field(default_factory=list)
    confidence: float = 0.0


class PatternExtractor(BaseModel):
    """Extracts patterns from problem statements."""
    pattern: str
    description: str
    examples: List[str]
    regex: Optional[str] = None


class ProblemAnalyzer:
    """
    Analyzes natural language TOC problems to extract requirements and understand intent.
    Uses NLP and pattern matching to parse problems.
    """
    
    def __init__(self):
        self.config = AIConfig()
        self.model = self.config.get_model(ModelType.GENERAL)
        
        # Pattern matchers for common problem types
        self.pattern_matchers = self._init_pattern_matchers()
        
        # Keyword mappings
        self.keyword_map = self._init_keyword_map()
        
        # NLP prompt for problem understanding
        self.understanding_prompt = self._create_understanding_prompt()
        
        logger.info("Problem Analyzer initialized with NLP capabilities")
    
    def _init_pattern_matchers(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for common problem structures."""
        return {
            "even_odd": re.compile(
                r"(even|odd)\s+(number|count)\s+of\s+(\w+)",
                re.IGNORECASE
            ),
            "divisible": re.compile(
                r"(divisible|multiple)\s+by\s+(\d+)",
                re.IGNORECASE
            ),
            "starts_with": re.compile(
                r"(starts?|begins?)\s+with\s+(.+)",
                re.IGNORECASE
            ),
            "ends_with": re.compile(
                r"ends?\s+with\s+(.+)",
                re.IGNORECASE
            ),
            "contains": re.compile(
                r"contains?\s+(.+?)(\s+as\s+substring)?",
                re.IGNORECASE
            ),
            "not_contains": re.compile(
                r"(does\s+not|doesn't)\s+contain\s+(.+)",
                re.IGNORECASE
            ),
            "at_least": re.compile(
                r"at\s+least\s+(\d+)\s+(.+)",
                re.IGNORECASE
            ),
            "at_most": re.compile(
                r"at\s+most\s+(\d+)\s+(.+)",
                re.IGNORECASE
            ),
            "exactly": re.compile(
                r"exactly\s+(\d+)\s+(.+)",
                re.IGNORECASE
            ),
            "balanced": re.compile(
                r"balanced\s+(\w+)\s+and\s+(\w+)",
                re.IGNORECASE
            ),
            "palindrome": re.compile(
                r"palindrome",
                re.IGNORECASE
            ),
            "alphabet": re.compile(
                r"(alphabet|symbols?|over)\s*[{=:]\s*([^}]+)[}]?",
                re.IGNORECASE
            )
        }
    
    def _init_keyword_map(self) -> Dict[str, ProblemType]:
        """Map keywords to problem types."""
        return {
            "dfa": ProblemType.DFA_CONSTRUCTION,
            "nfa": ProblemType.NFA_CONSTRUCTION,
            "regular expression": ProblemType.REGEX_TO_AUTOMATON,
            "regex": ProblemType.REGEX_TO_AUTOMATON,
            "minimize": ProblemType.DFA_MINIMIZATION,
            "minimization": ProblemType.DFA_MINIMIZATION,
            "convert": ProblemType.TRANSFORMATION,
            "cfg": ProblemType.CFG_CONSTRUCTION,
            "context-free": ProblemType.CFG_CONSTRUCTION,
            "pda": ProblemType.PDA_CONSTRUCTION,
            "pushdown": ProblemType.PDA_CONSTRUCTION,
            "turing machine": ProblemType.TM_CONSTRUCTION,
            "tm": ProblemType.TM_CONSTRUCTION,
            "pumping lemma": ProblemType.PUMPING_LEMMA_PROOF,
            "prove": ProblemType.PROOF,
            "decidable": ProblemType.DECIDABILITY,
            "complexity": ProblemType.COMPLEXITY_ANALYSIS,
            "equivalent": ProblemType.LANGUAGE_EQUIVALENCE,
            "verify": ProblemType.VERIFICATION
        }
    
    def _create_understanding_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for understanding problems."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in Theory of Computation.
            Analyze the given problem statement and extract:
            
            1. Problem Type (DFA, NFA, CFG, PDA, TM, proof, etc.)
            2. Alphabet/Symbols used
            3. Language constraints and requirements
            4. Positive examples (strings to accept)
            5. Negative examples (strings to reject)
            6. Formal language description if possible
            7. Any special patterns (even/odd, divisibility, etc.)
            
            Return your analysis as structured JSON."""),
            HumanMessage(content="{problem_statement}")
        ])
    
    async def analyze(
        self,
        problem_statement: str,
        problem_type: Optional[ProblemType] = None
    ) -> ProblemRequirements:
        """
        Analyze a natural language problem statement to extract requirements.
        
        This method:
        1. Uses pattern matching to identify common structures
        2. Applies NLP to understand the problem intent
        3. Extracts alphabet, constraints, and examples
        4. Determines the problem type and patterns
        """
        
        logger.info(f"Analyzing problem: {problem_statement[:100]}...")
        
        # Step 1: Quick pattern matching
        patterns = self._extract_patterns(problem_statement)
        
        # Step 2: Extract alphabet
        alphabet = self._extract_alphabet(problem_statement)
        
        # Step 3: Identify problem type
        if problem_type is None:
            problem_type = self._identify_problem_type(problem_statement)
        
        # Step 4: Use AI for deep understanding
        ai_analysis = await self._ai_understand(problem_statement)
        
        # Step 5: Extract constraints
        constraints = self._extract_constraints(problem_statement, ai_analysis)
        
        # Step 6: Generate examples
        examples = await self._generate_examples(problem_statement, constraints)
        
        # Step 7: Build requirements
        requirements = self._build_requirements(
            problem_statement,
            patterns,
            ai_analysis,
            constraints
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            patterns, ai_analysis, problem_type != ProblemType.UNKNOWN
        )
        
        return ProblemRequirements(
            problem_type=problem_type,
            alphabet=alphabet,
            constraints=constraints,
            requirements=requirements,
            examples=examples,
            formal_language=ai_analysis.get("formal_language"),
            complexity_class=ai_analysis.get("complexity_class"),
            original_statement=problem_statement,
            keywords=self._extract_keywords(problem_statement),
            patterns=patterns,
            confidence=confidence
        )
    
    def _extract_patterns(self, statement: str) -> List[LanguagePattern]:
        """Extract language patterns from the problem statement."""
        patterns = []
        
        for pattern_name, regex in self.pattern_matchers.items():
            if regex.search(statement):
                if "even_odd" in pattern_name:
                    patterns.append(LanguagePattern.EVEN_ODD)
                elif "divisible" in pattern_name:
                    patterns.append(LanguagePattern.DIVISIBILITY)
                elif "balanced" in pattern_name:
                    patterns.append(LanguagePattern.BALANCED)
                elif "palindrome" in pattern_name:
                    patterns.append(LanguagePattern.PALINDROME)
                elif any(x in pattern_name for x in ["starts", "ends", "contains"]):
                    patterns.append(LanguagePattern.SUBSTRING)
        
        return patterns
    
    def _extract_alphabet(self, statement: str) -> Set[str]:
        """Extract the alphabet from the problem statement."""
        alphabet = set()
        
        # Look for explicit alphabet definition
        alphabet_match = self.pattern_matchers["alphabet"].search(statement)
        if alphabet_match:
            symbols = alphabet_match.group(2)
            # Extract individual symbols
            for char in symbols:
                if char.isalnum():
                    alphabet.add(char)
        
        # Look for common alphabets
        if "binary" in statement.lower():
            alphabet = {"0", "1"}
        elif "ternary" in statement.lower():
            alphabet = {"0", "1", "2"}
        elif re.search(r"\ba\s+and\s+b\b", statement, re.IGNORECASE):
            alphabet = {"a", "b"}
        
        # Extract from examples in the statement
        example_pattern = re.compile(r'"([^"]+)"|\'([^\']+)\'')
        for match in example_pattern.finditer(statement):
            example = match.group(1) or match.group(2)
            for char in example:
                if char.isalnum():
                    alphabet.add(char)
        
        # Default to binary if no alphabet found
        if not alphabet:
            alphabet = {"0", "1"}
        
        return alphabet
    
    def _identify_problem_type(self, statement: str) -> ProblemType:
        """Identify the type of problem from the statement."""
        statement_lower = statement.lower()
        
        # Check for explicit problem type keywords
        for keyword, problem_type in self.keyword_map.items():
            if keyword in statement_lower:
                return problem_type
        
        # Infer from context
        if "design" in statement_lower or "construct" in statement_lower:
            if "accept" in statement_lower:
                return ProblemType.DFA_CONSTRUCTION
        elif "minimize" in statement_lower:
            return ProblemType.DFA_MINIMIZATION
        elif "convert" in statement_lower or "transform" in statement_lower:
            return ProblemType.TRANSFORMATION
        elif "prove" in statement_lower:
            return ProblemType.PROOF
        elif "verify" in statement_lower:
            return ProblemType.VERIFICATION
        
        return ProblemType.UNKNOWN
    
    async def _ai_understand(self, statement: str) -> Dict[str, Any]:
        """Use AI to deeply understand the problem."""
        
        prompt = self.understanding_prompt.format(problem_statement=statement)
        
        try:
            response = await self.model.ainvoke(prompt)
            # Parse JSON response
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"AI understanding failed: {e}")
            # Return basic analysis
            return {
                "problem_type": "unknown",
                "constraints": [],
                "examples": {"positive": [], "negative": []},
                "patterns": []
            }
    
    def _extract_constraints(
        self,
        statement: str,
        ai_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract constraints from the problem statement."""
        constraints = []
        
        # From AI analysis
        if "constraints" in ai_analysis:
            constraints.extend(ai_analysis["constraints"])
        
        # Pattern-based extraction
        patterns_to_constraints = {
            "even_odd": self._extract_even_odd_constraint,
            "divisible": self._extract_divisibility_constraint,
            "starts_with": self._extract_prefix_constraint,
            "ends_with": self._extract_suffix_constraint,
            "contains": self._extract_substring_constraint,
            "at_least": self._extract_counting_constraint,
            "at_most": self._extract_counting_constraint,
            "exactly": self._extract_counting_constraint
        }
        
        for pattern_name, extractor in patterns_to_constraints.items():
            if pattern_name in self.pattern_matchers:
                match = self.pattern_matchers[pattern_name].search(statement)
                if match:
                    constraint = extractor(match)
                    if constraint:
                        constraints.append(constraint)
        
        return list(set(constraints))  # Remove duplicates
    
    def _extract_even_odd_constraint(self, match: re.Match) -> str:
        """Extract even/odd constraint."""
        parity = match.group(1).lower()
        symbol = match.group(3)
        return f"{parity}_count_{symbol}"
    
    def _extract_divisibility_constraint(self, match: re.Match) -> str:
        """Extract divisibility constraint."""
        divisor = match.group(2)
        return f"length_divisible_by_{divisor}"
    
    def _extract_prefix_constraint(self, match: re.Match) -> str:
        """Extract prefix constraint."""
        prefix = match.group(2).strip()
        return f"starts_with_{prefix}"
    
    def _extract_suffix_constraint(self, match: re.Match) -> str:
        """Extract suffix constraint."""
        suffix = match.group(1).strip()
        return f"ends_with_{suffix}"
    
    def _extract_substring_constraint(self, match: re.Match) -> str:
        """Extract substring constraint."""
        substring = match.group(1).strip()
        return f"contains_{substring}"
    
    def _extract_counting_constraint(self, match: re.Match) -> str:
        """Extract counting constraint."""
        quantity = match.group(1)
        item = match.group(2).strip()
        constraint_type = match.group(0).split()[0].lower()
        return f"{constraint_type}_{quantity}_{item}"
    
    async def _generate_examples(
        self,
        statement: str,
        constraints: List[str]
    ) -> Dict[str, List[str]]:
        """Generate positive and negative examples."""
        
        prompt = f"""
        Generate test examples for this automaton problem:
        
        Problem: {statement}
        Constraints: {constraints}
        
        Generate:
        - 5 strings that should be ACCEPTED (positive examples)
        - 5 strings that should be REJECTED (negative examples)
        
        Return as JSON: {{"positive": [...], "negative": [...]}}
        """
        
        try:
            response = await self.model.ainvoke(prompt)
            examples = json.loads(response.content)
            return examples
        except:
            # Fallback examples
            return {
                "positive": ["", "0", "00", "000", "0000"],
                "negative": ["1", "01", "001", "0001", "00001"]
            }
    
    def _build_requirements(
        self,
        statement: str,
        patterns: List[LanguagePattern],
        ai_analysis: Dict[str, Any],
        constraints: List[str]
    ) -> Dict[str, Any]:
        """Build comprehensive requirements dictionary."""
        
        requirements = {
            "description": statement,
            "patterns": [p.value for p in patterns],
            "constraints": constraints,
            "ai_insights": ai_analysis
        }
        
        # Add pattern-specific requirements
        for pattern in patterns:
            if pattern == LanguagePattern.EVEN_ODD:
                requirements["parity_requirements"] = self._extract_parity_requirements(statement)
            elif pattern == LanguagePattern.DIVISIBILITY:
                requirements["divisibility_requirements"] = self._extract_divisibility_requirements(statement)
            elif pattern == LanguagePattern.SUBSTRING:
                requirements["substring_requirements"] = self._extract_substring_requirements(statement)
        
        return requirements
    
    def _extract_parity_requirements(self, statement: str) -> Dict[str, str]:
        """Extract parity (even/odd) requirements."""
        requirements = {}
        
        for match in self.pattern_matchers["even_odd"].finditer(statement):
            parity = match.group(1).lower()
            symbol = match.group(3)
            requirements[symbol] = parity
        
        return requirements
    
    def _extract_divisibility_requirements(self, statement: str) -> Dict[str, int]:
        """Extract divisibility requirements."""
        requirements = {}
        
        for match in self.pattern_matchers["divisible"].finditer(statement):
            divisor = int(match.group(2))
            requirements["length_divisor"] = divisor
        
        return requirements
    
    def _extract_substring_requirements(self, statement: str) -> Dict[str, List[str]]:
        """Extract substring requirements."""
        requirements = {
            "required_substrings": [],
            "forbidden_substrings": [],
            "prefixes": [],
            "suffixes": []
        }
        
        # Required substrings
        for match in self.pattern_matchers["contains"].finditer(statement):
            if "not" not in statement[max(0, match.start()-10):match.start()]:
                requirements["required_substrings"].append(match.group(1).strip())
        
        # Forbidden substrings
        for match in self.pattern_matchers["not_contains"].finditer(statement):
            requirements["forbidden_substrings"].append(match.group(2).strip())
        
        # Prefixes
        for match in self.pattern_matchers["starts_with"].finditer(statement):
            requirements["prefixes"].append(match.group(2).strip())
        
        # Suffixes
        for match in self.pattern_matchers["ends_with"].finditer(statement):
            requirements["suffixes"].append(match.group(1).strip())
        
        return requirements
    
    def _extract_keywords(self, statement: str) -> List[str]:
        """Extract important keywords from the statement."""
        keywords = []
        
        # TOC-specific keywords
        toc_keywords = [
            "accept", "reject", "recognize", "language",
            "state", "transition", "alphabet", "string",
            "regular", "context-free", "decidable",
            "deterministic", "nondeterministic",
            "minimal", "equivalent", "closure"
        ]
        
        statement_lower = statement.lower()
        for keyword in toc_keywords:
            if keyword in statement_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _calculate_confidence(
        self,
        patterns: List[LanguagePattern],
        ai_analysis: Dict[str, Any],
        has_known_type: bool
    ) -> float:
        """Calculate confidence in the analysis."""
        
        confidence = 0.5  # Base confidence
        
        # Increase for identified patterns
        confidence += len(patterns) * 0.1
        
        # Increase for AI analysis
        if ai_analysis and "problem_type" in ai_analysis:
            confidence += 0.2
        
        # Increase for known problem type
        if has_known_type:
            confidence += 0.2
        
        return min(confidence, 1.0)