"""
MIT/Oxford-level course structure for Theory of Computation
"""
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ConceptType(str, Enum):
    DFA = "dfa"
    NFA = "nfa"
    REGEX = "regex"
    PDA = "pda"
    CFG = "cfg"
    TM = "turing_machine"
    PUMPING_LEMMA = "pumping_lemma"
    COMPLEXITY = "complexity"


class Module(BaseModel):
    """Course module with prerequisites"""
    id: str
    title: str
    description: str
    concepts: List[ConceptType]
    prerequisites: List[str]  # Module IDs
    difficulty: Difficulty
    estimated_hours: float
    learning_objectives: List[str]
    

class CourseStructure:
    """MIT/Oxford-level course structure"""
    
    @staticmethod
    def get_modules() -> List[Module]:
        return [
            # Foundation Module
            Module(
                id="mod-1",
                title="Mathematical Foundations",
                description="Essential mathematical concepts for automata theory",
                concepts=[],
                prerequisites=[],
                difficulty=Difficulty.BEGINNER,
                estimated_hours=4,
                learning_objectives=[
                    "Understand sets, relations, and functions",
                    "Master proof techniques (induction, contradiction)",
                    "Work with formal languages and alphabets"
                ]
            ),
            
            # Module 2: Regular Languages
            Module(
                id="mod-2",
                title="Finite Automata and Regular Languages",
                description="Introduction to DFA, NFA, and regular expressions",
                concepts=[ConceptType.DFA, ConceptType.NFA, ConceptType.REGEX],
                prerequisites=["mod-1"],
                difficulty=Difficulty.BEGINNER,
                estimated_hours=8,
                learning_objectives=[
                    "Design and analyze DFAs for pattern recognition",
                    "Convert between DFA, NFA, and regular expressions",
                    "Apply closure properties of regular languages",
                    "Use the pumping lemma for regular languages"
                ]
            ),
            
            # Module 3: Context-Free Languages
            Module(
                id="mod-3",
                title="Context-Free Languages and Pushdown Automata",
                description="CFGs, PDAs, and parsing techniques",
                concepts=[ConceptType.CFG, ConceptType.PDA],
                prerequisites=["mod-2"],
                difficulty=Difficulty.INTERMEDIATE,
                estimated_hours=10,
                learning_objectives=[
                    "Design context-free grammars for languages",
                    "Construct pushdown automata",
                    "Apply CYK parsing algorithm",
                    "Prove languages are not context-free"
                ]
            ),
            
            # Module 4: Turing Machines
            Module(
                id="mod-4",
                title="Turing Machines and Computability",
                description="Universal computation and decidability",
                concepts=[ConceptType.TM],
                prerequisites=["mod-3"],
                difficulty=Difficulty.ADVANCED,
                estimated_hours=12,
                learning_objectives=[
                    "Design Turing machines for complex problems",
                    "Understand Church-Turing thesis",
                    "Prove undecidability using reductions",
                    "Analyze the halting problem"
                ]
            ),
            
            # Module 5: Complexity Theory
            Module(
                id="mod-5",
                title="Computational Complexity Theory",
                description="P, NP, and complexity classes",
                concepts=[ConceptType.COMPLEXITY],
                prerequisites=["mod-4"],
                difficulty=Difficulty.EXPERT,
                estimated_hours=15,
                learning_objectives=[
                    "Classify problems into complexity classes",
                    "Understand P vs NP problem",
                    "Perform polynomial-time reductions",
                    "Analyze space complexity"
                ]
            ),
            
            # Module 6: Advanced Topics
            Module(
                id="mod-6",
                title="Advanced Pumping Lemmas and Proofs",
                description="Deep dive into pumping lemmas for all language classes",
                concepts=[ConceptType.PUMPING_LEMMA],
                prerequisites=["mod-3"],
                difficulty=Difficulty.ADVANCED,
                estimated_hours=6,
                learning_objectives=[
                    "Master pumping lemma for regular languages",
                    "Apply pumping lemma for context-free languages",
                    "Construct adversarial examples",
                    "Develop intuition for non-regular patterns"
                ]
            )
        ]
    
    @staticmethod
    def get_learning_path(target_concept: ConceptType) -> List[str]:
        """Get ordered module IDs to learn a concept"""
        concept_to_modules = {
            ConceptType.DFA: ["mod-1", "mod-2"],
            ConceptType.NFA: ["mod-1", "mod-2"],
            ConceptType.REGEX: ["mod-1", "mod-2"],
            ConceptType.CFG: ["mod-1", "mod-2", "mod-3"],
            ConceptType.PDA: ["mod-1", "mod-2", "mod-3"],
            ConceptType.TM: ["mod-1", "mod-2", "mod-3", "mod-4"],
            ConceptType.PUMPING_LEMMA: ["mod-1", "mod-2", "mod-6"],
            ConceptType.COMPLEXITY: ["mod-1", "mod-2", "mod-3", "mod-4", "mod-5"]
        }
        return concept_to_modules.get(target_concept, [])
    
    @staticmethod
    def check_prerequisites(module_id: str, completed_modules: List[str]) -> bool:
        """Check if prerequisites are met"""
        modules = {m.id: m for m in CourseStructure.get_modules()}
        if module_id not in modules:
            return False
        
        module = modules[module_id]
        return all(prereq in completed_modules for prereq in module.prerequisites)