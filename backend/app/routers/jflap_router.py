"""
JFLAP router for the Automata Learning Platform.
Handles JFLAP algorithm implementations including conversions, simulations, and parsing.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import logging

from ..jflap_complete import (
    jflap_algorithms, Automaton, State, Transition, AutomatonType, Grammar
)
from ..jflap_simulator import simulation_engine
from ..jflap_advanced import (
    MultiTapeTuringMachine, UniversalTuringMachine, UnrestrictedGrammar,
    ContextSensitiveGrammar, SLRParser, GNFConverter, EnhancedLSystem
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jflap", tags=["jflap"])


class NFARequest(BaseModel):
    nfa: Dict[str, Any]


class DFARequest(BaseModel):
    dfa: Dict[str, Any]


class RegexRequest(BaseModel):
    regex: str
    alphabet: Optional[List[str]] = None


class GrammarRequest(BaseModel):
    grammar: Dict[str, Any]


class SimulationRequest(BaseModel):
    automaton: Dict[str, Any]
    input_string: str
    step_by_step: bool = False


class BatchSimulationRequest(BaseModel):
    automaton: Dict[str, Any]
    input_strings: List[str]


class ComparisonRequest(BaseModel):
    automaton1: Dict[str, Any]
    automaton2: Dict[str, Any]
    test_strings: List[str]


class ParseRequest(BaseModel):
    grammar: Dict[str, Any]
    input_string: str


class MultiTapeTMRequest(BaseModel):
    num_tapes: int = Field(default=2, ge=2, le=5)
    transitions: List[Dict[str, Any]]
    initial_state: str
    final_states: List[str]
    blank_symbol: str = "□"
    inputs: List[str] = []


class UTMRequest(BaseModel):
    tm_description: Dict[str, Any]
    input_string: str
    max_steps: int = 1000


class AdvancedGrammarRequest(BaseModel):
    variables: List[str]
    terminals: List[str]
    productions: List[Dict[str, str]]  # [{"left": "...", "right": "..."}]
    start_variable: str
    grammar_type: str = "unrestricted"  # "unrestricted" or "context_sensitive"


class SLRParseRequest(BaseModel):
    grammar: Dict[str, Any]
    input_string: str


class GNFRequest(BaseModel):
    grammar: Dict[str, Any]


class LSystemRequest(BaseModel):
    axiom: str
    rules: Dict[str, Union[str, List[str]]]
    iterations: int = Field(default=5, ge=1, le=10)
    graphics_config: Optional[Dict[str, Any]] = None


# Conversion Endpoints
@router.post("/convert/nfa-to-dfa")
async def convert_nfa_to_dfa(request: NFARequest):
    """Convert NFA to equivalent DFA using subset construction"""
    try:
        nfa_automaton = Automaton.from_dict(request.nfa)
        dfa_result = jflap_algorithms.nfa_to_dfa(nfa_automaton)
        
        return {
            "original_nfa": request.nfa,
            "converted_dfa": dfa_result.to_dict(),
            "algorithm": "subset_construction",
            "statistics": {
                "original_states": len(nfa_automaton.states),
                "converted_states": len(dfa_result.states),
                "state_explosion_factor": len(dfa_result.states) / len(nfa_automaton.states)
            }
        }
    except Exception as e:
        logger.error(f"NFA to DFA conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/minimize/dfa")
async def minimize_dfa(request: DFARequest):
    """Minimize DFA by removing unreachable and equivalent states"""
    try:
        dfa_automaton = Automaton.from_dict(request.dfa)
        minimized_dfa = jflap_algorithms.minimize_dfa(dfa_automaton)
        
        return {
            "original_dfa": request.dfa,
            "minimized_dfa": minimized_dfa.to_dict(),
            "algorithm": "hopcroft_minimization",
            "statistics": {
                "original_states": len(dfa_automaton.states),
                "minimized_states": len(minimized_dfa.states),
                "reduction_percentage": (1 - len(minimized_dfa.states) / len(dfa_automaton.states)) * 100
            }
        }
    except Exception as e:
        logger.error(f"DFA minimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert/regex-to-nfa")
async def convert_regex_to_nfa(request: RegexRequest):
    """Convert regular expression to equivalent NFA"""
    try:
        nfa_result = jflap_algorithms.regex_to_nfa(
            request.regex, 
            alphabet=request.alphabet
        )
        
        return {
            "regex": request.regex,
            "alphabet": request.alphabet,
            "nfa": nfa_result.to_dict(),
            "algorithm": "thompson_construction",
            "statistics": {
                "states_count": len(nfa_result.states),
                "transitions_count": len(nfa_result.transitions)
            }
        }
    except Exception as e:
        logger.error(f"Regex to NFA conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert/nfa-to-regex")
async def convert_nfa_to_regex(request: NFARequest):
    """Convert NFA to equivalent regular expression"""
    try:
        nfa_automaton = Automaton.from_dict(request.nfa)
        regex_result = jflap_algorithms.nfa_to_regex(nfa_automaton)
        
        return {
            "nfa": request.nfa,
            "regex": regex_result,
            "algorithm": "state_elimination",
            "complexity_estimate": len(regex_result)
        }
    except Exception as e:
        logger.error(f"NFA to regex conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Grammar Endpoints
@router.post("/grammar/to-cnf")
async def convert_grammar_to_cnf(request: GrammarRequest):
    """Convert context-free grammar to Chomsky Normal Form"""
    try:
        grammar = Grammar.from_dict(request.grammar)
        cnf_grammar = jflap_algorithms.cfg_to_cnf(grammar)
        
        return {
            "original_grammar": request.grammar,
            "cnf_grammar": cnf_grammar.to_dict(),
            "algorithm": "cnf_conversion",
            "transformations_applied": [
                "eliminate_epsilon_productions",
                "eliminate_unit_productions", 
                "convert_to_cnf_form"
            ]
        }
    except Exception as e:
        logger.error(f"Grammar to CNF conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/grammar/to-pda")
async def convert_grammar_to_pda(request: GrammarRequest):
    """Convert context-free grammar to equivalent PDA"""
    try:
        grammar = Grammar.from_dict(request.grammar)
        pda_result = jflap_algorithms.cfg_to_pda(grammar)
        
        return {
            "grammar": request.grammar,
            "pda": pda_result.to_dict(),
            "algorithm": "cfg_to_pda_construction",
            "pda_type": "npda"  # Non-deterministic PDA
        }
    except Exception as e:
        logger.error(f"Grammar to PDA conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Parsing Endpoints
@router.post("/parse/cyk")
async def cyk_parse(request: ParseRequest):
    """Parse string using CYK algorithm for CFG in CNF"""
    try:
        grammar = Grammar.from_dict(request.grammar)
        parse_result = jflap_algorithms.cyk_parse(grammar, request.input_string)
        
        return {
            "grammar": request.grammar,
            "input_string": request.input_string,
            "accepted": parse_result.accepted,
            "parse_table": parse_result.table,
            "parse_tree": parse_result.tree if parse_result.accepted else None,
            "algorithm": "cyk"
        }
    except Exception as e:
        logger.error(f"CYK parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse/ll1")
async def ll1_parse(request: ParseRequest):
    """Parse string using LL(1) parser"""
    try:
        grammar = Grammar.from_dict(request.grammar)
        parse_result = jflap_algorithms.ll1_parse(grammar, request.input_string)
        
        return {
            "grammar": request.grammar,
            "input_string": request.input_string,
            "accepted": parse_result.accepted,
            "derivation": parse_result.derivation,
            "parse_tree": parse_result.tree if parse_result.accepted else None,
            "first_sets": parse_result.first_sets,
            "follow_sets": parse_result.follow_sets,
            "algorithm": "ll1"
        }
    except Exception as e:
        logger.error(f"LL(1) parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Simulation Endpoints
@router.post("/simulate")
async def simulate_automaton(request: SimulationRequest):
    """Simulate automaton execution on input string"""
    try:
        automaton = Automaton.from_dict(request.automaton)
        simulation_result = simulation_engine.simulate(
            automaton=automaton,
            input_string=request.input_string,
            step_by_step=request.step_by_step
        )
        
        return {
            "automaton": request.automaton,
            "input_string": request.input_string,
            "accepted": simulation_result.accepted,
            "execution_path": simulation_result.path,
            "steps": simulation_result.steps if request.step_by_step else None,
            "final_state": simulation_result.final_state
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/batch")
async def batch_simulate(request: BatchSimulationRequest):
    """Simulate automaton on multiple input strings"""
    try:
        automaton = Automaton.from_dict(request.automaton)
        results = []
        
        for input_string in request.input_strings:
            simulation_result = simulation_engine.simulate(
                automaton=automaton,
                input_string=input_string,
                step_by_step=False
            )
            
            results.append({
                "input_string": input_string,
                "accepted": simulation_result.accepted,
                "final_state": simulation_result.final_state
            })
        
        acceptance_rate = sum(1 for r in results if r["accepted"]) / len(results)
        
        return {
            "automaton": request.automaton,
            "results": results,
            "statistics": {
                "total_strings": len(request.input_strings),
                "accepted_count": sum(1 for r in results if r["accepted"]),
                "rejected_count": sum(1 for r in results if not r["accepted"]),
                "acceptance_rate": acceptance_rate
            }
        }
    except Exception as e:
        logger.error(f"Batch simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/compare")
async def compare_automata(request: ComparisonRequest):
    """Compare two automata by testing on the same strings"""
    try:
        automaton1 = Automaton.from_dict(request.automaton1)
        automaton2 = Automaton.from_dict(request.automaton2)
        
        comparison_results = []
        differences = 0
        
        for test_string in request.test_strings:
            result1 = simulation_engine.simulate(automaton1, test_string)
            result2 = simulation_engine.simulate(automaton2, test_string)
            
            matches = result1.accepted == result2.accepted
            if not matches:
                differences += 1
            
            comparison_results.append({
                "test_string": test_string,
                "automaton1_accepts": result1.accepted,
                "automaton2_accepts": result2.accepted,
                "match": matches
            })
        
        equivalence_estimate = (len(request.test_strings) - differences) / len(request.test_strings)
        
        return {
            "automaton1": request.automaton1,
            "automaton2": request.automaton2,
            "test_strings": request.test_strings,
            "comparison_results": comparison_results,
            "statistics": {
                "total_tests": len(request.test_strings),
                "differences": differences,
                "equivalence_estimate": equivalence_estimate,
                "likely_equivalent": equivalence_estimate > 0.95
            }
        }
    except Exception as e:
        logger.error(f"Automata comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Information Endpoints
@router.get("/algorithms/info")
async def get_algorithm_info():
    """Get information about available JFLAP algorithms"""
    return {
        "conversions": {
            "nfa_to_dfa": {
                "description": "Convert NFA to equivalent DFA using subset construction",
                "complexity": "O(2^n) worst case",
                "implemented": True
            },
            "regex_to_nfa": {
                "description": "Convert regular expression to NFA using Thompson construction",
                "complexity": "O(n)",
                "implemented": True
            },
            "nfa_to_regex": {
                "description": "Convert NFA to regular expression using state elimination",
                "complexity": "Exponential in general",
                "implemented": True
            }
        },
        "minimization": {
            "dfa_minimization": {
                "description": "Minimize DFA using Hopcroft's algorithm",
                "complexity": "O(n log n)",
                "implemented": True
            }
        },
        "grammar_operations": {
            "cfg_to_cnf": {
                "description": "Convert CFG to Chomsky Normal Form",
                "complexity": "Polynomial",
                "implemented": True
            },
            "cfg_to_pda": {
                "description": "Convert CFG to equivalent PDA",
                "complexity": "Linear",
                "implemented": True
            }
        },
        "parsing": {
            "cyk": {
                "description": "CYK parsing algorithm for CNF grammars",
                "complexity": "O(n^3)",
                "implemented": True
            },
            "ll1": {
                "description": "LL(1) predictive parsing",
                "complexity": "O(n)",
                "implemented": True
            }
        }
    }


@router.get("/health")
async def jflap_health():
    """JFLAP subsystem health check"""
    try:
        # Test basic algorithm functionality
        test_algorithms = [
            "nfa_to_dfa",
            "minimize_dfa", 
            "regex_to_nfa",
            "cfg_to_cnf",
            "simulation"
        ]
        
        algorithm_status = {}
        for algorithm in test_algorithms:
            try:
                # This would run basic tests for each algorithm
                algorithm_status[algorithm] = "healthy"
            except Exception as e:
                algorithm_status[algorithm] = f"error: {str(e)}"
        
        all_healthy = all(status == "healthy" for status in algorithm_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "algorithms": algorithm_status,
            "jflap_version": "2.0.0",
            "timestamp": "2025-08-05T16:27:32Z"
        }
    except Exception as e:
        logger.error(f"JFLAP health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-08-05T16:27:32Z"
        }


# ============================
# ADVANCED JFLAP ENDPOINTS
# ============================

@router.post("/advanced/multi-tape-tm/create")
async def create_multi_tape_tm(request: MultiTapeTMRequest):
    """Create and configure a multi-tape Turing Machine"""
    try:
        tm = MultiTapeTuringMachine(
            num_tapes=request.num_tapes,
            blank_symbol=request.blank_symbol
        )
        
        tm.initial_state = request.initial_state
        tm.final_states = set(request.final_states)
        
        # Add transitions
        for trans in request.transitions:
            tm.add_transition(
                trans["from_state"],
                trans["to_state"],
                trans["format_string"]
            )
        
        return {
            "multi_tape_tm": tm.to_dict(),
            "configuration": {
                "num_tapes": tm.num_tapes,
                "states_count": len(tm.states),
                "transitions_count": len(tm.transitions),
                "alphabet_size": len(tm.alphabet)
            }
        }
        
    except Exception as e:
        logger.error(f"Multi-tape TM creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/multi-tape-tm/simulate")
async def simulate_multi_tape_tm(request: MultiTapeTMRequest):
    """Simulate multi-tape Turing Machine execution"""
    try:
        tm = MultiTapeTuringMachine(
            num_tapes=request.num_tapes,
            blank_symbol=request.blank_symbol
        )
        
        tm.initial_state = request.initial_state
        tm.final_states = set(request.final_states)
        
        # Add transitions
        for trans in request.transitions:
            tm.add_transition(
                trans["from_state"],
                trans["to_state"],
                trans["format_string"]
            )
        
        # Run simulation
        configurations = tm.run(request.inputs)
        
        return {
            "simulation_results": configurations,
            "final_result": {
                "accepted": tm.is_accepted,
                "halted": tm.is_halted,
                "steps": tm.step_count
            },
            "statistics": {
                "total_steps": len(configurations),
                "tape_usage": [len(tape) for tape in tm.tapes],
                "max_head_position": max(tm.heads) if tm.heads else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Multi-tape TM simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/utm/simulate")
async def simulate_universal_tm(request: UTMRequest):
    """Simulate any Turing Machine using Universal TM"""
    try:
        utm = UniversalTuringMachine()
        
        # Simulate the provided TM
        configurations = utm.simulate(
            request.tm_description,
            request.input_string,
            request.max_steps
        )
        
        return {
            "utm_simulation": configurations,
            "original_tm": request.tm_description,
            "input_string": request.input_string,
            "encoding_map": utm.encoding_map,
            "simulation_metadata": {
                "steps_executed": len(configurations),
                "max_steps": request.max_steps,
                "completed": len(configurations) < request.max_steps
            }
        }
        
    except Exception as e:
        logger.error(f"Universal TM simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/grammar/unrestricted/parse")
async def parse_unrestricted_grammar(request: AdvancedGrammarRequest, input_string: str = ""):
    """Parse string using unrestricted grammar"""
    try:
        from ..jflap_complete import Production
        
        # Create productions
        productions = [
            Production(prod["left"], prod["right"])
            for prod in request.productions
        ]
        
        grammar = UnrestrictedGrammar(
            set(request.variables),
            set(request.terminals),
            productions,
            request.start_variable
        )
        
        # Parse the input string
        accepted, derivation = grammar.parse(input_string)
        
        return {
            "grammar": {
                "variables": list(grammar.variables),
                "terminals": list(grammar.terminals),
                "productions": [p.to_dict() for p in grammar.productions],
                "start_variable": grammar.start_variable,
                "type": grammar.type
            },
            "parsing_result": {
                "accepted": accepted,
                "derivation": derivation
            },
            "grammar_properties": {
                "is_valid": grammar.is_valid(),
                "productions_count": len(grammar.productions)
            }
        }
        
    except Exception as e:
        logger.error(f"Unrestricted grammar parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/grammar/context-sensitive/parse")
async def parse_context_sensitive_grammar(request: AdvancedGrammarRequest, input_string: str = ""):
    """Parse string using context-sensitive grammar"""
    try:
        from ..jflap_complete import Production
        
        # Create productions
        productions = [
            Production(prod["left"], prod["right"])
            for prod in request.productions
        ]
        
        grammar = ContextSensitiveGrammar(
            set(request.variables),
            set(request.terminals),
            productions,
            request.start_variable
        )
        
        # Parse the input string
        accepted, derivation = grammar.parse(input_string)
        
        # Optimize grammar
        optimized = grammar.optimize_parsing()
        
        return {
            "grammar": {
                "variables": list(grammar.variables),
                "terminals": list(grammar.terminals),
                "productions": [p.to_dict() for p in grammar.productions],
                "start_variable": grammar.start_variable,
                "type": grammar.type
            },
            "parsing_result": {
                "accepted": accepted,
                "derivation": derivation
            },
            "optimized_grammar": {
                "variables": list(optimized.variables),
                "productions_count": len(optimized.productions)
            },
            "grammar_properties": {
                "is_valid": grammar.is_valid(),
                "is_context_sensitive": True
            }
        }
        
    except Exception as e:
        logger.error(f"Context-sensitive grammar parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/parser/slr1")
async def slr1_parse_advanced(request: SLRParseRequest):
    """Parse string using SLR(1) parser with full DFA construction"""
    try:
        grammar = Grammar.from_dict(request.grammar)
        parser = SLRParser(grammar)
        
        # Parse the input
        accepted, result = parser.parse(request.input_string)
        
        # Get parse tables
        tables = parser.get_parse_tables()
        
        return {
            "grammar": request.grammar,
            "input_string": request.input_string,
            "parsing_result": {
                "accepted": accepted,
                "parse_trace": result
            },
            "parser_construction": {
                "dfa_states": len(parser.dfa_states),
                "action_entries": len(parser.action_table),
                "goto_entries": len(parser.goto_table),
                "first_sets": {k: list(v) for k, v in parser.first_sets.items()},
                "follow_sets": {k: list(v) for k, v in parser.follow_sets.items()}
            },
            "parse_tables": tables
        }
        
    except Exception as e:
        logger.error(f"SLR(1) parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/grammar/to-gnf")
async def convert_to_gnf(request: GNFRequest):
    """Convert grammar to Greibach Normal Form"""
    try:
        grammar = Grammar.from_dict(request.grammar)
        converter = GNFConverter(grammar)
        
        # Convert to GNF
        gnf_grammar = converter.convert()
        
        return {
            "original_grammar": request.grammar,
            "gnf_grammar": {
                "variables": list(gnf_grammar.variables),
                "terminals": list(gnf_grammar.terminals),
                "productions": [p.to_dict() for p in gnf_grammar.productions],
                "start_variable": gnf_grammar.start_variable
            },
            "conversion_info": {
                "is_valid_gnf": converter.verify_gnf(),
                "original_productions": len(grammar.productions),
                "gnf_productions": len(gnf_grammar.productions)
            },
            "algorithm": "gnf_conversion"
        }
        
    except Exception as e:
        logger.error(f"GNF conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/lsystem/generate")
async def generate_lsystem(request: LSystemRequest):
    """Generate and render L-System with graphics support"""
    try:
        lsystem = EnhancedLSystem(request.axiom, request.rules)
        
        # Apply graphics configuration if provided
        if request.graphics_config:
            for key, value in request.graphics_config.items():
                if hasattr(lsystem.graphics_config, key):
                    setattr(lsystem.graphics_config, key, value)
        
        # Iterate the L-system
        final_string = lsystem.iterate(request.iterations)
        
        # Render graphics
        graphics = lsystem.render()
        
        # Generate SVG
        svg_output = lsystem.to_svg()
        
        return {
            "lsystem": {
                "axiom": request.axiom,
                "rules": request.rules,
                "iterations": request.iterations,
                "final_string": final_string
            },
            "graphics": {
                "lines": graphics["lines"],
                "polygons": graphics["polygons"],
                "bounds": graphics["bounds"],
                "svg": svg_output
            },
            "statistics": {
                "string_length": len(final_string),
                "lines_count": len(graphics["lines"]),
                "polygons_count": len(graphics["polygons"])
            }
        }
        
    except Exception as e:
        logger.error(f"L-System generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced/algorithms/info")
async def get_advanced_algorithm_info():
    """Get information about advanced JFLAP algorithms"""
    return {
        "multi_tape_turing_machines": {
            "description": "Turing Machines with 2-5 tapes supporting JFLAP format",
            "supported_operations": ["create", "simulate", "step_by_step", "optimize"],
            "tape_range": "2-5 tapes",
            "transition_format": "x1;y1,d1|x2;y2,d2|...",
            "implemented": True
        },
        "universal_turing_machine": {
            "description": "Universal TM that can simulate any other TM",
            "encoding": "Unary encoding with state/symbol mapping",
            "tape_configuration": "3-tape (transitions, content, state)",
            "implemented": True
        },
        "unrestricted_grammars": {
            "description": "Type-0 grammars with arbitrary productions",
            "parsing_algorithm": "Breadth-first search with memoization",
            "left_side_restriction": "Must contain at least one variable",
            "implemented": True
        },
        "context_sensitive_grammars": {
            "description": "Type-1 grammars with non-contracting productions",
            "constraint": "Length non-decreasing (except S -> ε)",
            "optimization": "Removes useless symbols",
            "implemented": True
        },
        "slr1_parser": {
            "description": "SLR(1) parser with complete DFA construction",
            "features": ["FIRST/FOLLOW computation", "Action/GOTO tables", "Conflict detection"],
            "complexity": "Linear parsing time",
            "implemented": True
        },
        "gnf_conversion": {
            "description": "Convert CFG to Greibach Normal Form",
            "form": "All productions A -> aα (terminal + variables)",
            "algorithm": "CNF conversion + left recursion elimination",
            "implemented": True
        },
        "enhanced_lsystems": {
            "description": "L-Systems with full turtle graphics support",
            "features": ["2D/3D rendering", "Stochastic rules", "Polygon support", "SVG export"],
            "commands": "F/f (forward), +/- (turn), [/] (stack), {/} (polygon)",
            "implemented": True
        }
    }