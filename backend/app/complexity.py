"""
Complexity Theory Module for Automata Theory
Provides complexity class definitions, reduction verification, NP-completeness proofs, and complexity analysis.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
import re
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class ComplexityClass(str, Enum):
    """Standard complexity classes"""
    P = "P"
    NP = "NP"
    PSPACE = "PSPACE"
    EXPTIME = "EXPTIME"
    LOGSPACE = "LOGSPACE"
    NPSPACE = "NPSPACE"
    NEXPTIME = "NEXPTIME"
    # Decision problems
    DECIDABLE = "DECIDABLE"
    UNDECIDABLE = "UNDECIDABLE"
    # Language classes
    REGULAR = "REGULAR"
    CONTEXT_FREE = "CONTEXT_FREE"
    CONTEXT_SENSITIVE = "CONTEXT_SENSITIVE"
    RECURSIVELY_ENUMERABLE = "RECURSIVELY_ENUMERABLE"

class ReductionType(str, Enum):
    """Types of reductions"""
    POLYNOMIAL_TIME = "polynomial_time"  # Karp reduction
    LOG_SPACE = "log_space"
    POLYNOMIAL_SPACE = "polynomial_space"
    MANY_ONE = "many_one"
    TURING = "turing"

class ComplexityRelation(str, Enum):
    """Relationships between complexity classes"""
    SUBSET = "subset"
    EQUAL = "equal"
    SUPERSET = "superset"
    INCOMPARABLE = "incomparable"
    UNKNOWN = "unknown"

class Problem(BaseModel):
    """A computational problem"""
    name: str
    description: str
    complexity_class: ComplexityClass
    is_complete: bool = False
    input_format: str
    output_format: str

class Reduction(BaseModel):
    """A reduction between problems"""
    from_problem: str
    to_problem: str
    reduction_type: ReductionType
    time_complexity: str
    space_complexity: str
    description: str

class ComplexityAnalysisRequest(BaseModel):
    """Request for complexity analysis"""
    algorithm_description: str
    input_size_variable: str = "n"
    time_analysis: Optional[str] = None
    space_analysis: Optional[str] = None

class ReductionVerificationRequest(BaseModel):
    """Request to verify a reduction"""
    reduction: Reduction
    reduction_function: Optional[str] = None
    correctness_proof: Optional[str] = None

class ComplexityClassHierarchy:
    """Manages complexity class relationships and hierarchies"""
    
    def __init__(self):
        self.class_definitions = self._load_class_definitions()
        self.known_inclusions = self._load_known_inclusions()
        self.complete_problems = self._load_complete_problems()
    
    def _load_class_definitions(self) -> Dict[ComplexityClass, Dict[str, Any]]:
        """Load definitions of complexity classes"""
        return {
            ComplexityClass.P: {
                "name": "Polynomial Time",
                "definition": "Problems solvable by deterministic TM in polynomial time",
                "formal": "DTIME(n^O(1))",
                "machine_model": "Deterministic Turing Machine",
                "resource": "Time",
                "bound": "Polynomial"
            },
            ComplexityClass.NP: {
                "name": "Nondeterministic Polynomial Time",
                "definition": "Problems solvable by nondeterministic TM in polynomial time",
                "formal": "NTIME(n^O(1))",
                "machine_model": "Nondeterministic Turing Machine",
                "resource": "Time",
                "bound": "Polynomial",
                "alternative": "Problems with polynomial-time verifiable certificates"
            },
            ComplexityClass.PSPACE: {
                "name": "Polynomial Space",
                "definition": "Problems solvable using polynomial space",
                "formal": "DSPACE(n^O(1))",
                "machine_model": "Deterministic Turing Machine",
                "resource": "Space",
                "bound": "Polynomial"
            },
            ComplexityClass.EXPTIME: {
                "name": "Exponential Time",
                "definition": "Problems solvable in exponential time",
                "formal": "DTIME(2^n^O(1))",
                "machine_model": "Deterministic Turing Machine",
                "resource": "Time",
                "bound": "Exponential"
            },
            ComplexityClass.LOGSPACE: {
                "name": "Logarithmic Space",
                "definition": "Problems solvable using logarithmic space",
                "formal": "DSPACE(log n)",
                "machine_model": "Deterministic Turing Machine",
                "resource": "Space",
                "bound": "Logarithmic"
            },
            ComplexityClass.REGULAR: {
                "name": "Regular Languages",
                "definition": "Languages recognizable by finite automata",
                "formal": "Languages accepted by DFA/NFA",
                "machine_model": "Finite Automaton",
                "resource": "States",
                "bound": "Finite"
            },
            ComplexityClass.CONTEXT_FREE: {
                "name": "Context-Free Languages",
                "definition": "Languages recognizable by pushdown automata",
                "formal": "Languages accepted by PDA",
                "machine_model": "Pushdown Automaton",
                "resource": "Stack + States",
                "bound": "Finite states, unbounded stack"
            }
        }
    
    def _load_known_inclusions(self) -> Dict[Tuple[ComplexityClass, ComplexityClass], ComplexityRelation]:
        """Load known relationships between complexity classes"""
        return {
            (ComplexityClass.REGULAR, ComplexityClass.CONTEXT_FREE): ComplexityRelation.SUBSET,
            (ComplexityClass.CONTEXT_FREE, ComplexityClass.CONTEXT_SENSITIVE): ComplexityRelation.SUBSET,
            (ComplexityClass.CONTEXT_SENSITIVE, ComplexityClass.RECURSIVELY_ENUMERABLE): ComplexityRelation.SUBSET,
            (ComplexityClass.LOGSPACE, ComplexityClass.P): ComplexityRelation.SUBSET,
            (ComplexityClass.P, ComplexityClass.NP): ComplexityRelation.SUBSET,
            (ComplexityClass.NP, ComplexityClass.PSPACE): ComplexityRelation.SUBSET,
            (ComplexityClass.PSPACE, ComplexityClass.EXPTIME): ComplexityRelation.SUBSET,
            (ComplexityClass.P, ComplexityClass.PSPACE): ComplexityRelation.SUBSET,
            # Some known equalities (by Savitch's theorem)
            (ComplexityClass.PSPACE, ComplexityClass.NPSPACE): ComplexityRelation.EQUAL,
        }
    
    def _load_complete_problems(self) -> Dict[ComplexityClass, List[Problem]]:
        """Load complete problems for each complexity class"""
        return {
            ComplexityClass.P: [
                Problem(
                    name="2-SAT",
                    description="Satisfiability of 2-CNF formulas",
                    complexity_class=ComplexityClass.P,
                    is_complete=True,
                    input_format="2-CNF formula",
                    output_format="Boolean (satisfiable or not)"
                )
            ],
            ComplexityClass.NP: [
                Problem(
                    name="3-SAT",
                    description="Satisfiability of 3-CNF formulas",
                    complexity_class=ComplexityClass.NP,
                    is_complete=True,
                    input_format="3-CNF formula",
                    output_format="Boolean (satisfiable or not)"
                ),
                Problem(
                    name="HAMILTONIAN-CYCLE",
                    description="Existence of Hamiltonian cycle in graph",
                    complexity_class=ComplexityClass.NP,
                    is_complete=True,
                    input_format="Undirected graph",
                    output_format="Boolean (has cycle or not)"
                ),
                Problem(
                    name="CLIQUE",
                    description="Existence of k-clique in graph",
                    complexity_class=ComplexityClass.NP,
                    is_complete=True,
                    input_format="Graph G and integer k",
                    output_format="Boolean (has k-clique or not)"
                ),
                Problem(
                    name="VERTEX-COVER",
                    description="Existence of vertex cover of size k",
                    complexity_class=ComplexityClass.NP,
                    is_complete=True,
                    input_format="Graph G and integer k",
                    output_format="Boolean (has vertex cover of size ≤ k)"
                ),
                Problem(
                    name="SUBSET-SUM",
                    description="Subset with given sum",
                    complexity_class=ComplexityClass.NP,
                    is_complete=True,
                    input_format="Set of integers and target sum",
                    output_format="Boolean (subset exists or not)"
                )
            ],
            ComplexityClass.PSPACE: [
                Problem(
                    name="TQBF",
                    description="True Quantified Boolean Formula",
                    complexity_class=ComplexityClass.PSPACE,
                    is_complete=True,
                    input_format="Quantified boolean formula",
                    output_format="Boolean (true or false)"
                ),
                Problem(
                    name="GEOGRAPHY",
                    description="Generalized Geography game",
                    complexity_class=ComplexityClass.PSPACE,
                    is_complete=True,
                    input_format="Directed graph and starting vertex",
                    output_format="Boolean (first player has winning strategy)"
                )
            ]
        }

class ComplexityAnalyzer:
    """Analyzes computational complexity of algorithms and problems"""
    
    def __init__(self):
        self.hierarchy = ComplexityClassHierarchy()
        
    def analyze_algorithm_complexity(self, request: ComplexityAnalysisRequest) -> Dict[str, Any]:
        """Analyze the complexity of an algorithm"""
        try:
            algorithm = request.algorithm_description
            n = request.input_size_variable
            
            # Extract complexity patterns
            time_complexity = self._extract_time_complexity(algorithm, n)
            space_complexity = self._extract_space_complexity(algorithm, n)
            
            # Classify into complexity class
            complexity_class = self._classify_complexity(time_complexity, space_complexity)
            
            return {
                "algorithm": algorithm,
                "analysis": {
                    "time_complexity": time_complexity,
                    "space_complexity": space_complexity,
                    "complexity_class": complexity_class,
                    "big_o_time": self._to_big_o(time_complexity),
                    "big_o_space": self._to_big_o(space_complexity)
                },
                "explanation": self._generate_complexity_explanation(time_complexity, space_complexity),
                "classification": self._get_class_info(complexity_class)
            }
            
        except Exception as e:
            logger.error(f"Error in complexity analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def verify_reduction(self, request: ReductionVerificationRequest) -> Dict[str, Any]:
        """Verify a reduction between problems"""
        try:
            reduction = request.reduction
            
            # Check if reduction type is appropriate for complexity classes
            verification = self._verify_reduction_correctness(reduction)
            
            # Analyze reduction complexity
            complexity_analysis = self._analyze_reduction_complexity(reduction)
            
            return {
                "reduction": {
                    "from": reduction.from_problem,
                    "to": reduction.to_problem,
                    "type": reduction.reduction_type,
                    "description": reduction.description
                },
                "verification": verification,
                "complexity_analysis": complexity_analysis,
                "implications": self._get_reduction_implications(reduction)
            }
            
        except Exception as e:
            logger.error(f"Error in reduction verification: {e}")
            return {"error": f"Verification failed: {str(e)}"}
    
    def get_class_relationship(self, class1: ComplexityClass, class2: ComplexityClass) -> Dict[str, Any]:
        """Get relationship between two complexity classes"""
        pair = (class1, class2)
        reverse_pair = (class2, class1)
        
        if pair in self.hierarchy.known_inclusions:
            relation = self.hierarchy.known_inclusions[pair]
        elif reverse_pair in self.hierarchy.known_inclusions:
            reverse_relation = self.hierarchy.known_inclusions[reverse_pair]
            relation = self._reverse_relation(reverse_relation)
        else:
            relation = ComplexityRelation.UNKNOWN
        
        return {
            "class1": class1,
            "class2": class2,
            "relationship": relation,
            "explanation": self._explain_relationship(class1, class2, relation),
            "known_separations": self._get_known_separations(class1, class2),
            "open_questions": self._get_open_questions(class1, class2)
        }
    
    def _extract_time_complexity(self, algorithm: str, n: str) -> str:
        """Extract time complexity from algorithm description"""
        algorithm_lower = algorithm.lower()
        
        # Look for explicit complexity mentions
        complexity_patterns = [
            (r"o\(([^)]+)\)", r"\1"),
            (r"time complexity:?\s*([^\n]+)", r"\1"),
            (r"running time:?\s*([^\n]+)", r"\1"),
            (r"(\d*)\s*loops?.*nested", lambda m: f"{n}^{len(m.group(0).split('nested')) + 1}"),
        ]
        
        for pattern, replacement in complexity_patterns:
            match = re.search(pattern, algorithm_lower)
            if match:
                if callable(replacement):
                    return replacement(match)
                else:
                    return match.group(1) if hasattr(match, 'group') else replacement
        
        # Analyze algorithm structure
        if "nested" in algorithm_lower and "loop" in algorithm_lower:
            nesting_level = algorithm_lower.count("nested") + 1
            return f"{n}^{nesting_level}"
        elif "for" in algorithm_lower or "while" in algorithm_lower:
            loop_count = algorithm_lower.count("for") + algorithm_lower.count("while")
            if loop_count >= 2:
                return f"{n}^{loop_count}"
            else:
                return n
        elif "recursive" in algorithm_lower:
            if "divide" in algorithm_lower and "conquer" in algorithm_lower:
                return f"{n} log {n}"
            else:
                return f"2^{n}"  # Assume exponential if not divide-and-conquer
        elif "dynamic programming" in algorithm_lower or "memoization" in algorithm_lower:
            return f"{n}^2"  # Common DP complexity
        else:
            return n  # Default to linear
    
    def _extract_space_complexity(self, algorithm: str, n: str) -> str:
        """Extract space complexity from algorithm description"""
        algorithm_lower = algorithm.lower()
        
        if "constant space" in algorithm_lower or "in-place" in algorithm_lower:
            return "1"
        elif "logarithmic space" in algorithm_lower:
            return f"log {n}"
        elif "recursive" in algorithm_lower:
            if "tail recursive" in algorithm_lower:
                return "1"
            else:
                return f"log {n}"  # For recursion depth
        elif "dynamic programming" in algorithm_lower:
            return f"{n}^2"  # Common DP space usage
        elif "stack" in algorithm_lower or "queue" in algorithm_lower:
            return n
        else:
            return n  # Default assumption
    
    def _classify_complexity(self, time_complexity: str, space_complexity: str) -> ComplexityClass:
        """Classify algorithm into complexity class"""
        time_lower = time_complexity.lower()
        
        if "2^" in time_lower or "exponential" in time_lower:
            return ComplexityClass.EXPTIME
        elif "^" in time_lower and not "log" in time_lower:
            return ComplexityClass.PSPACE  # Assume polynomial space for polynomial time
        elif "log" in time_lower or time_lower == "n" or "polynomial" in time_lower:
            return ComplexityClass.P
        else:
            return ComplexityClass.P  # Default conservative assumption
    
    def _to_big_o(self, complexity: str) -> str:
        """Convert complexity expression to Big O notation"""
        if not complexity or complexity == "1":
            return "O(1)"
        elif complexity.startswith("O("):
            return complexity
        else:
            return f"O({complexity})"
    
    def _generate_complexity_explanation(self, time_complexity: str, space_complexity: str) -> str:
        """Generate explanation of complexity analysis"""
        explanations = []
        
        if "^" in time_complexity:
            explanations.append(f"Time complexity is polynomial: {self._to_big_o(time_complexity)}")
        elif "2^" in time_complexity:
            explanations.append(f"Time complexity is exponential: {self._to_big_o(time_complexity)}")
        elif "log" in time_complexity:
            explanations.append(f"Time complexity is efficient: {self._to_big_o(time_complexity)}")
        
        if space_complexity == "1":
            explanations.append("Uses constant space")
        elif "log" in space_complexity:
            explanations.append("Uses logarithmic space")
        elif space_complexity == "n":
            explanations.append("Uses linear space")
        
        return ". ".join(explanations) + "."
    
    def _get_class_info(self, complexity_class: ComplexityClass) -> Dict[str, Any]:
        """Get information about a complexity class"""
        if complexity_class in self.hierarchy.class_definitions:
            return self.hierarchy.class_definitions[complexity_class]
        else:
            return {"name": str(complexity_class), "definition": "Custom complexity class"}
    
    def _verify_reduction_correctness(self, reduction: Reduction) -> Dict[str, Any]:
        """Verify if a reduction is correctly defined"""
        verification = {"is_valid": True, "issues": []}
        
        # Check if reduction type makes sense for complexity classes
        if reduction.reduction_type == ReductionType.POLYNOMIAL_TIME:
            if "exponential" in reduction.time_complexity.lower():
                verification["is_valid"] = False
                verification["issues"].append("Polynomial-time reduction cannot have exponential time complexity")
        
        # Check if reduction preserves the right properties
        if reduction.reduction_type == ReductionType.MANY_ONE:
            verification["preserves"] = "Yes/No instances"
        elif reduction.reduction_type == ReductionType.TURING:
            verification["preserves"] = "Solvability"
        
        return verification
    
    def _analyze_reduction_complexity(self, reduction: Reduction) -> Dict[str, Any]:
        """Analyze the complexity of a reduction"""
        return {
            "time_complexity": reduction.time_complexity,
            "space_complexity": reduction.space_complexity,
            "type": reduction.reduction_type,
            "efficiency": self._assess_reduction_efficiency(reduction)
        }
    
    def _assess_reduction_efficiency(self, reduction: Reduction) -> str:
        """Assess if reduction is efficient enough for its purpose"""
        time_comp = reduction.time_complexity.lower()
        
        if reduction.reduction_type == ReductionType.POLYNOMIAL_TIME:
            if "polynomial" in time_comp or "^" in time_comp:
                return "Efficient for polynomial-time reduction"
            elif "exponential" in time_comp or "2^" in time_comp:
                return "Too expensive for polynomial-time reduction"
            else:
                return "Efficiency unclear"
        
        return "Standard efficiency for reduction type"
    
    def _get_reduction_implications(self, reduction: Reduction) -> List[str]:
        """Get implications of a reduction"""
        implications = []
        
        from_prob = reduction.from_problem
        to_prob = reduction.to_problem
        
        if reduction.reduction_type == ReductionType.POLYNOMIAL_TIME:
            implications.append(f"If {to_prob} is polynomial-time solvable, then so is {from_prob}")
            implications.append(f"If {from_prob} is not polynomial-time solvable, then neither is {to_prob}")
        
        if "NP-complete" in reduction.description:
            implications.append(f"If P = NP, both problems are in P")
            implications.append(f"If P ≠ NP, both problems are not in P")
        
        return implications
    
    def _reverse_relation(self, relation: ComplexityRelation) -> ComplexityRelation:
        """Reverse a complexity relation"""
        if relation == ComplexityRelation.SUBSET:
            return ComplexityRelation.SUPERSET
        elif relation == ComplexityRelation.SUPERSET:
            return ComplexityRelation.SUBSET
        else:
            return relation
    
    def _explain_relationship(self, class1: ComplexityClass, class2: ComplexityClass, 
                             relation: ComplexityRelation) -> str:
        """Explain relationship between complexity classes"""
        if relation == ComplexityRelation.SUBSET:
            return f"{class1} ⊆ {class2}: Every problem in {class1} is also in {class2}"
        elif relation == ComplexityRelation.SUPERSET:
            return f"{class1} ⊇ {class2}: {class1} contains all problems in {class2}"
        elif relation == ComplexityRelation.EQUAL:
            return f"{class1} = {class2}: These complexity classes are equal"
        elif relation == ComplexityRelation.UNKNOWN:
            return f"Relationship between {class1} and {class2} is unknown or open"
        else:
            return f"{class1} and {class2} are incomparable"
    
    def _get_known_separations(self, class1: ComplexityClass, class2: ComplexityClass) -> List[str]:
        """Get known separations between complexity classes"""
        separations = []
        
        # Some known results
        if (class1, class2) in [(ComplexityClass.REGULAR, ComplexityClass.CONTEXT_FREE)]:
            separations.append("Regular languages are properly contained in context-free languages")
        
        return separations
    
    def _get_open_questions(self, class1: ComplexityClass, class2: ComplexClass) -> List[str]:
        """Get open questions related to complexity classes"""
        open_questions = []
        
        if (class1, class2) in [(ComplexityClass.P, ComplexityClass.NP), (ComplexityClass.NP, ComplexityClass.P)]:
            open_questions.append("P vs NP: Is P = NP? This is the most famous open problem in computer science")
        
        if (class1, class2) in [(ComplexityClass.NP, ComplexityClass.PSPACE), (ComplexityClass.PSPACE, ComplexityClass.NP)]:
            open_questions.append("NP vs PSPACE: Is NP = PSPACE? This is another major open problem")
        
        return open_questions

# Global analyzer instance
complexity_analyzer = ComplexityAnalyzer()

def analyze_complexity(request: ComplexityAnalysisRequest) -> Dict[str, Any]:
    """Analyze algorithm complexity"""
    return complexity_analyzer.analyze_algorithm_complexity(request)

def verify_reduction(request: ReductionVerificationRequest) -> Dict[str, Any]:
    """Verify a reduction between problems"""
    return complexity_analyzer.verify_reduction(request)

def get_complexity_classes() -> Dict[str, Any]:
    """Get information about complexity classes"""
    return {
        "classes": complexity_analyzer.hierarchy.class_definitions,
        "relationships": complexity_analyzer.hierarchy.known_inclusions,
        "complete_problems": complexity_analyzer.hierarchy.complete_problems
    }

def get_class_relationship(class1: str, class2: str) -> Dict[str, Any]:
    """Get relationship between two complexity classes"""
    try:
        c1 = ComplexityClass(class1.upper())
        c2 = ComplexityClass(class2.upper())
        return complexity_analyzer.get_class_relationship(c1, c2)
    except ValueError:
        return {"error": f"Unknown complexity class: {class1} or {class2}"}

def get_complexity_templates() -> Dict[str, Any]:
    """Get templates for complexity theory proofs and reductions"""
    return {
        "np_completeness_proof": {
            "steps": [
                "Show the problem is in NP",
                "Choose a known NP-complete problem to reduce from",
                "Construct a polynomial-time reduction",
                "Prove the reduction is correct",
                "Conclude the problem is NP-complete"
            ],
            "reduction_techniques": [
                "Local replacement: Replace components of source with target components",
                "Component design: Design target components to simulate source behavior",
                "Global structure: Preserve overall structure while changing components"
            ]
        },
        "complexity_class_separation": {
            "steps": [
                "Define the two complexity classes precisely",
                "Construct a problem that separates them",
                "Prove the problem is in the larger class",
                "Prove the problem is not in the smaller class",
                "Use diagonalization or other advanced techniques if needed"
            ]
        },
        "time_hierarchy_theorem": {
            "statement": "DTIME(f(n)) ⊊ DTIME(f(n) log²f(n)) for time-constructible f",
            "proof_technique": "Diagonalization against all machines with smaller time bound"
        },
        "space_hierarchy_theorem": {
            "statement": "DSPACE(f(n)) ⊊ DSPACE(f(n) log f(n)) for space-constructible f ≥ log n",
            "proof_technique": "Diagonalization with space-bounded computation"
        }
    }