"""
Pumping Lemma Module for Automata Theory
Provides string decomposition, pumping validation, and non-regularity proof support.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from enum import Enum
import re
import logging
from itertools import combinations_with_replacement

logger = logging.getLogger(__name__)

class LanguageType(str, Enum):
    """Types of languages for pumping lemma"""
    REGULAR = "regular"
    CONTEXT_FREE = "context_free"

class PumpingType(str, Enum):
    """Types of pumping lemma applications"""
    PROOF_NON_REGULAR = "proof_non_regular"
    PROOF_NON_CFL = "proof_non_cfl"
    VALIDATE_REGULAR = "validate_regular"
    VALIDATE_CFL = "validate_cfl"

class RegularDecomposition(BaseModel):
    """xyz decomposition for regular pumping lemma"""
    x: str = Field(..., description="First part (can be empty)")
    y: str = Field(..., min_length=1, description="Middle part (must be non-empty)")
    z: str = Field(..., description="Last part (can be empty)")
    
    def get_full_string(self) -> str:
        return self.x + self.y + self.z
    
    def pump(self, i: int) -> str:
        """Generate xy^i z"""
        if i < 0:
            raise ValueError("Pumping parameter i must be non-negative")
        return self.x + (self.y * i) + self.z

class CFLDecomposition(BaseModel):
    """uvxyz decomposition for context-free pumping lemma"""
    u: str = Field(..., description="First part")
    v: str = Field(..., description="Second part")
    x: str = Field(..., description="Middle part")
    y: str = Field(..., description="Fourth part")
    z: str = Field(..., description="Last part")
    
    def get_full_string(self) -> str:
        return self.u + self.v + self.x + self.y + self.z
    
    def pump(self, i: int) -> str:
        """Generate uv^i xy^i z"""
        if i < 0:
            raise ValueError("Pumping parameter i must be non-negative")
        return self.u + (self.v * i) + self.x + (self.y * i) + self.z
    
    def is_valid_cfl_decomposition(self, original_string: str, pumping_length: int) -> Tuple[bool, str]:
        """Check if decomposition satisfies CFL pumping lemma conditions"""
        # Condition 1: uvxyz = original string
        if self.get_full_string() != original_string:
            return False, "Decomposition doesn't reconstruct original string"
        
        # Condition 2: |vxy| ≤ pumping_length
        vxy_length = len(self.v) + len(self.x) + len(self.y)
        if vxy_length > pumping_length:
            return False, f"|vxy| = {vxy_length} > {pumping_length} (pumping length)"
        
        # Condition 3: |vy| ≥ 1 (at least one of v or y is non-empty)
        if len(self.v) == 0 and len(self.y) == 0:
            return False, "|vy| = 0, but must be ≥ 1"
        
        return True, "Valid CFL decomposition"

class PumpingRequest(BaseModel):
    """Request for pumping lemma operations"""
    language_type: LanguageType
    pumping_type: PumpingType
    string: str = Field(..., min_length=1, max_length=1000)
    pumping_length: int = Field(..., ge=1, le=100)
    decomposition: Optional[Dict[str, str]] = None
    test_values: List[int] = Field(default=[0, 1, 2], description="Values of i to test")

class PumpingValidationRequest(BaseModel):
    """Request to validate a pumping lemma application"""
    language_description: str
    candidate_string: str
    decomposition: Dict[str, str]
    language_type: LanguageType
    claimed_pumping_length: int

class NonRegularityProofRequest(BaseModel):
    """Request for non-regularity proof assistance"""
    language_description: str
    candidate_strings: List[str]
    suspected_pumping_length: Optional[int] = None

class PumpingResult(BaseModel):
    """Result of pumping lemma operation"""
    is_valid: bool
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    decompositions: List[Dict[str, str]] = Field(default_factory=list)
    counter_examples: List[str] = Field(default_factory=list)

class PumpingLemmaEngine:
    """Core engine for pumping lemma operations"""
    
    def __init__(self):
        self.max_decompositions = 50  # Limit number of decompositions to try
        self.max_string_length = 1000
    
    def validate_pumping(self, request: PumpingRequest) -> PumpingResult:
        """Validate a pumping lemma application"""
        try:
            if request.language_type == LanguageType.REGULAR:
                return self._validate_regular_pumping(request)
            elif request.language_type == LanguageType.CONTEXT_FREE:
                return self._validate_cfl_pumping(request)
            else:
                return PumpingResult(
                    is_valid=False,
                    message="Unsupported language type"
                )
        except Exception as e:
            logger.error(f"Error in pumping validation: {e}")
            return PumpingResult(
                is_valid=False,
                message=f"Validation error: {str(e)}"
            )
    
    def _validate_regular_pumping(self, request: PumpingRequest) -> PumpingResult:
        """Validate regular pumping lemma application"""
        string = request.string
        p = request.pumping_length
        
        if len(string) < p:
            return PumpingResult(
                is_valid=False,
                message=f"String length {len(string)} < pumping length {p}",
                details={"requirement": "String must be at least as long as pumping length"}
            )
        
        # If decomposition is provided, validate it
        if request.decomposition:
            return self._validate_provided_regular_decomposition(request)
        
        # Generate all possible decompositions
        valid_decompositions = self._generate_regular_decompositions(string, p)
        
        if request.pumping_type == PumpingType.PROOF_NON_REGULAR:
            # For non-regularity proof, we need to show no valid decomposition exists
            return self._check_non_regularity_proof(string, p, valid_decompositions, request.test_values)
        else:
            # For validation, show valid decompositions exist
            return PumpingResult(
                is_valid=len(valid_decompositions) > 0,
                message=f"Found {len(valid_decompositions)} valid decompositions",
                decompositions=[{
                    "x": d.x, "y": d.y, "z": d.z,
                    "constraints": f"|xy| ≤ {p}, |y| ≥ 1"
                } for d in valid_decompositions[:10]]  # Limit output
            )
    
    def _validate_cfl_pumping(self, request: PumpingRequest) -> PumpingResult:
        """Validate context-free pumping lemma application"""
        string = request.string
        p = request.pumping_length
        
        if len(string) < p:
            return PumpingResult(
                is_valid=False,
                message=f"String length {len(string)} < pumping length {p}"
            )
        
        if request.decomposition:
            return self._validate_provided_cfl_decomposition(request)
        
        # Generate possible decompositions
        valid_decompositions = self._generate_cfl_decompositions(string, p)
        
        if request.pumping_type == PumpingType.PROOF_NON_CFL:
            return self._check_non_cfl_proof(string, p, valid_decompositions, request.test_values)
        else:
            return PumpingResult(
                is_valid=len(valid_decompositions) > 0,
                message=f"Found {len(valid_decompositions)} valid decompositions",
                decompositions=[{
                    "u": d.u, "v": d.v, "x": d.x, "y": d.y, "z": d.z,
                    "constraints": f"|vxy| ≤ {p}, |vy| ≥ 1"
                } for d in valid_decompositions[:10]]
            )
    
    def _validate_provided_regular_decomposition(self, request: PumpingRequest) -> PumpingResult:
        """Validate a specific regular decomposition"""
        decomp_dict = request.decomposition
        if not all(key in decomp_dict for key in ['x', 'y', 'z']):
            return PumpingResult(
                is_valid=False,
                message="Regular decomposition must have x, y, z parts"
            )
        
        decomp = RegularDecomposition(
            x=decomp_dict['x'],
            y=decomp_dict['y'],
            z=decomp_dict['z']
        )
        
        # Check constraints
        if decomp.get_full_string() != request.string:
            return PumpingResult(
                is_valid=False,
                message="Decomposition doesn't reconstruct original string"
            )
        
        if len(decomp.x + decomp.y) > request.pumping_length:
            return PumpingResult(
                is_valid=False,
                message=f"|xy| = {len(decomp.x + decomp.y)} > {request.pumping_length}"
            )
        
        if len(decomp.y) == 0:
            return PumpingResult(
                is_valid=False,
                message="|y| = 0, must be ≥ 1"
            )
        
        # Test pumping
        test_results = []
        for i in request.test_values:
            pumped = decomp.pump(i)
            test_results.append({"i": i, "pumped_string": pumped})
        
        return PumpingResult(
            is_valid=True,
            message="Valid regular decomposition",
            details={"test_results": test_results}
        )
    
    def _validate_provided_cfl_decomposition(self, request: PumpingRequest) -> PumpingResult:
        """Validate a specific CFL decomposition"""
        decomp_dict = request.decomposition
        required_keys = ['u', 'v', 'x', 'y', 'z']
        if not all(key in decomp_dict for key in required_keys):
            return PumpingResult(
                is_valid=False,
                message=f"CFL decomposition must have {required_keys} parts"
            )
        
        decomp = CFLDecomposition(**decomp_dict)
        
        is_valid, error_msg = decomp.is_valid_cfl_decomposition(request.string, request.pumping_length)
        if not is_valid:
            return PumpingResult(is_valid=False, message=error_msg)
        
        # Test pumping
        test_results = []
        for i in request.test_values:
            pumped = decomp.pump(i)
            test_results.append({"i": i, "pumped_string": pumped})
        
        return PumpingResult(
            is_valid=True,
            message="Valid CFL decomposition",
            details={"test_results": test_results}
        )
    
    def _generate_regular_decompositions(self, string: str, p: int) -> List[RegularDecomposition]:
        """Generate all valid regular decompositions"""
        decompositions = []
        n = len(string)
        
        # xy must have length ≤ p, y must be non-empty
        for xy_len in range(1, min(p + 1, n + 1)):  # xy length from 1 to min(p, n)
            for y_len in range(1, xy_len + 1):  # y length from 1 to xy_len
                x_len = xy_len - y_len
                
                x = string[:x_len]
                y = string[x_len:x_len + y_len]
                z = string[x_len + y_len:]
                
                decompositions.append(RegularDecomposition(x=x, y=y, z=z))
        
        return decompositions
    
    def _generate_cfl_decompositions(self, string: str, p: int) -> List[CFLDecomposition]:
        """Generate valid CFL decompositions"""
        decompositions = []
        n = len(string)
        
        # Limit search to reasonable number of decompositions
        count = 0
        
        # vxy must have length ≤ p, vy must be non-empty
        for vxy_start in range(n):
            for vxy_len in range(1, min(p + 1, n - vxy_start + 1)):
                vxy_end = vxy_start + vxy_len
                
                # Try different v, x, y splits within vxy
                for v_len in range(vxy_len + 1):
                    for y_len in range(vxy_len - v_len + 1):
                        if v_len == 0 and y_len == 0:
                            continue  # vy must be non-empty
                        
                        x_len = vxy_len - v_len - y_len
                        
                        u = string[:vxy_start]
                        v = string[vxy_start:vxy_start + v_len]
                        x = string[vxy_start + v_len:vxy_start + v_len + x_len]
                        y = string[vxy_start + v_len + x_len:vxy_end]
                        z = string[vxy_end:]
                        
                        decomp = CFLDecomposition(u=u, v=v, x=x, y=y, z=z)
                        decompositions.append(decomp)
                        
                        count += 1
                        if count >= self.max_decompositions:
                            return decompositions
        
        return decompositions
    
    def _check_non_regularity_proof(self, string: str, p: int, 
                                   decompositions: List[RegularDecomposition], 
                                   test_values: List[int]) -> PumpingResult:
        """Check if string can be used to prove non-regularity"""
        failed_decompositions = []
        
        for decomp in decompositions:
            # Test if this decomposition fails for some i
            failed_for_some_i = False
            failure_details = []
            
            for i in test_values:
                pumped = decomp.pump(i)
                # Here we'd need language-specific validation
                # For now, we'll demonstrate the process
                failure_details.append({
                    "i": i,
                    "pumped_string": pumped,
                    "length": len(pumped),
                    "note": "Verify if this string is in the language"
                })
            
            failed_decompositions.append({
                "decomposition": {"x": decomp.x, "y": decomp.y, "z": decomp.z},
                "tests": failure_details
            })
        
        return PumpingResult(
            is_valid=True,
            message=f"Generated {len(decompositions)} decompositions to test",
            details={
                "instruction": "For each decomposition, verify if pumped strings are in the language",
                "decompositions_to_test": failed_decompositions
            }
        )
    
    def _check_non_cfl_proof(self, string: str, p: int,
                            decompositions: List[CFLDecomposition],
                            test_values: List[int]) -> PumpingResult:
        """Check if string can be used to prove non-CFL"""
        failed_decompositions = []
        
        for decomp in decompositions[:10]:  # Limit for output
            failure_details = []
            
            for i in test_values:
                pumped = decomp.pump(i)
                failure_details.append({
                    "i": i,
                    "pumped_string": pumped,
                    "length": len(pumped),
                    "note": "Verify if this string is in the language"
                })
            
            failed_decompositions.append({
                "decomposition": {
                    "u": decomp.u, "v": decomp.v, "x": decomp.x,
                    "y": decomp.y, "z": decomp.z
                },
                "tests": failure_details
            })
        
        return PumpingResult(
            is_valid=True,
            message=f"Generated {len(decompositions)} decompositions to test",
            details={
                "instruction": "For each decomposition, verify if pumped strings are in the language",
                "decompositions_to_test": failed_decompositions
            }
        )
    
    def suggest_non_regular_candidates(self, language_description: str) -> List[Dict[str, Any]]:
        """Suggest candidate strings for non-regularity proofs"""
        suggestions = []
        
        # Common patterns that suggest non-regularity
        if "a^n b^n" in language_description.lower() or "equal" in language_description.lower():
            suggestions.append({
                "pattern": "a^n b^n",
                "example_strings": ["ab", "aabb", "aaabbb", "aaaabbbb"],
                "reason": "Equal counts typically require context-free power",
                "pumping_strategy": "Show that pumping disrupts the balance"
            })
        
        if "palindrome" in language_description.lower():
            suggestions.append({
                "pattern": "Palindromes",
                "example_strings": ["aba", "abba", "abcba", "abccba"],
                "reason": "Palindromes require memory of entire first half",
                "pumping_strategy": "Show pumping breaks palindrome property"
            })
        
        if "a^n b^n c^n" in language_description.lower():
            suggestions.append({
                "pattern": "a^n b^n c^n",
                "example_strings": ["abc", "aabbcc", "aaabbbccc"],
                "reason": "Three-way equality requires context-sensitive power",
                "pumping_strategy": "Neither regular nor CFL pumping works"
            })
        
        if "perfect square" in language_description.lower() or "n^2" in language_description.lower():
            suggestions.append({
                "pattern": "Perfect squares",
                "example_strings": ["a", "aaaa", "aaaaaaaaa", "aaaaaaaaaaaaaaaa"],
                "reason": "Quadratic growth is beyond regular languages",
                "pumping_strategy": "Show pumping doesn't preserve square property"
            })
        
        if not suggestions:
            # Default suggestions
            suggestions.append({
                "pattern": "Balanced parentheses",
                "example_strings": ["()", "(())", "((()))", "(()())"],
                "reason": "Common non-regular language example",
                "pumping_strategy": "Show pumping breaks balance"
            })
        
        return suggestions

# Global engine instance
pumping_engine = PumpingLemmaEngine()

def validate_pumping_lemma(request: PumpingRequest) -> PumpingResult:
    """Validate a pumping lemma application"""
    return pumping_engine.validate_pumping(request)

def get_non_regular_candidates(language_description: str) -> List[Dict[str, Any]]:
    """Get candidate strings for non-regularity proofs"""
    return pumping_engine.suggest_non_regular_candidates(language_description)

def decompose_string_regular(string: str, pumping_length: int) -> List[Dict[str, str]]:
    """Get all valid regular decompositions of a string"""
    decompositions = pumping_engine._generate_regular_decompositions(string, pumping_length)
    return [{"x": d.x, "y": d.y, "z": d.z} for d in decompositions]

def decompose_string_cfl(string: str, pumping_length: int) -> List[Dict[str, str]]:
    """Get valid CFL decompositions of a string"""
    decompositions = pumping_engine._generate_cfl_decompositions(string, pumping_length)
    return [{"u": d.u, "v": d.v, "x": d.x, "y": d.y, "z": d.z} for d in decompositions[:20]]

def get_pumping_templates() -> Dict[str, Any]:
    """Get templates for pumping lemma proofs"""
    return {
        "regular_non_regularity_proof": {
            "steps": [
                "Assume L is regular",
                "Let p be the pumping length given by the pumping lemma",
                "Choose a string w in L with |w| ≥ p (choose strategically)",
                "By pumping lemma, w = xyz where |xy| ≤ p, |y| ≥ 1",
                "Show that for some i ≥ 0, xy^i z is not in L",
                "This contradicts the pumping lemma",
                "Therefore L is not regular"
            ],
            "common_strategies": [
                "For a^n b^n: choose w = a^p b^p, show pumping y (all a's) breaks balance",
                "For palindromes: choose long palindrome, show pumping breaks symmetry",
                "For equal counts: choose string with equal symbols, show pumping disrupts equality"
            ]
        },
        "cfl_non_cfl_proof": {
            "steps": [
                "Assume L is context-free",
                "Let p be the pumping length for context-free languages",
                "Choose a string w in L with |w| ≥ p",
                "By CFL pumping lemma, w = uvxyz where |vxy| ≤ p, |vy| ≥ 1",
                "Show that for some i ≥ 0, uv^i xy^i z is not in L",
                "This contradicts the CFL pumping lemma",
                "Therefore L is not context-free"
            ],
            "common_strategies": [
                "For a^n b^n c^n: choose w = a^p b^p c^p, show no valid decomposition exists",
                "For a^i b^j where i,j have complex relationship: exploit the constraint"
            ]
        }
    }