"""
JFLAP Integration Module
Provides import/export compatibility with JFLAP file formats and conventions
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class AutomatonType(Enum):
    DFA = "dfa"
    NFA = "nfa"
    PDA = "pda"
    TM = "tm"
    CFG = "cfg"

@dataclass
class JFLAPState:
    """JFLAP state representation"""
    id: str
    name: str
    x: float
    y: float
    is_initial: bool = False
    is_final: bool = False

@dataclass
class JFLAPTransition:
    """JFLAP transition representation"""
    from_state: str
    to_state: str
    input_symbol: str
    output_symbol: Optional[str] = None  # For Mealy machines
    stack_pop: Optional[str] = None      # For PDA
    stack_push: Optional[str] = None     # For PDA
    tape_write: Optional[str] = None     # For TM
    tape_move: Optional[str] = None      # For TM (L, R, S)

class JFLAPConverter:
    """Convert between JFLAP format and our internal format"""
    
    def __init__(self):
        # Load JFLAP knowledge base
        with open('jflap_extracted_content.json', 'r') as f:
            self.jflap_knowledge = json.load(f)
    
    def import_jff(self, jff_content: str) -> Dict[str, Any]:
        """
        Import JFLAP .jff file format (XML-based)
        
        Args:
            jff_content: XML content of .jff file
            
        Returns:
            Internal automaton representation
        """
        root = ET.fromstring(jff_content)
        automaton_type = root.find('type').text if root.find('type') is not None else 'fa'
        
        # Parse states
        states = []
        state_map = {}
        for state_elem in root.findall('.//state'):
            state_id = state_elem.get('id')
            state_name = state_elem.get('name', f'q{state_id}')
            
            # Get position
            x = float(state_elem.find('x').text) if state_elem.find('x') is not None else 0
            y = float(state_elem.find('y').text) if state_elem.find('y') is not None else 0
            
            # Check if initial or final
            is_initial = state_elem.find('initial') is not None
            is_final = state_elem.find('final') is not None
            
            state = JFLAPState(
                id=state_id,
                name=state_name,
                x=x,
                y=y,
                is_initial=is_initial,
                is_final=is_final
            )
            states.append(state)
            state_map[state_id] = state
        
        # Parse transitions
        transitions = []
        for trans_elem in root.findall('.//transition'):
            from_state = trans_elem.find('from').text
            to_state = trans_elem.find('to').text
            
            # Get input symbol (handle epsilon)
            read_elem = trans_elem.find('read')
            input_symbol = read_elem.text if read_elem is not None and read_elem.text else 'ε'
            
            transition = JFLAPTransition(
                from_state=from_state,
                to_state=to_state,
                input_symbol=input_symbol
            )
            
            # PDA specific
            if automaton_type == 'pda':
                pop_elem = trans_elem.find('pop')
                push_elem = trans_elem.find('push')
                transition.stack_pop = pop_elem.text if pop_elem is not None else None
                transition.stack_push = push_elem.text if push_elem is not None else None
            
            # TM specific
            if automaton_type == 'turing':
                write_elem = trans_elem.find('write')
                move_elem = trans_elem.find('move')
                transition.tape_write = write_elem.text if write_elem is not None else None
                transition.tape_move = move_elem.text if move_elem is not None else None
            
            transitions.append(transition)
        
        # Convert to internal format
        return self._to_internal_format(states, transitions, automaton_type)
    
    def export_jff(self, automaton: Dict[str, Any]) -> str:
        """
        Export automaton to JFLAP .jff format
        
        Args:
            automaton: Internal automaton representation
            
        Returns:
            XML string in .jff format
        """
        # Create root element
        root = ET.Element("structure")
        
        # Add type
        type_elem = ET.SubElement(root, "type")
        type_elem.text = self._get_jflap_type(automaton.get('type', 'dfa'))
        
        # Create automaton element
        automaton_elem = ET.SubElement(root, "automaton")
        
        # Add states
        state_id_map = {}
        for i, state in enumerate(automaton.get('states', [])):
            state_elem = ET.SubElement(automaton_elem, "state")
            state_elem.set("id", str(i))
            state_elem.set("name", state.get('name', f'q{i}'))
            
            # Position
            x_elem = ET.SubElement(state_elem, "x")
            x_elem.text = str(state.get('x', i * 100))
            y_elem = ET.SubElement(state_elem, "y")
            y_elem.text = str(state.get('y', 100))
            
            # Initial/final markers
            if state.get('is_initial', False):
                ET.SubElement(state_elem, "initial")
            if state.get('is_final', False):
                ET.SubElement(state_elem, "final")
            
            state_id_map[state.get('name', f'q{i}')] = str(i)
        
        # Add transitions
        for transition in automaton.get('transitions', []):
            trans_elem = ET.SubElement(automaton_elem, "transition")
            
            # From/to states
            from_elem = ET.SubElement(trans_elem, "from")
            from_elem.text = state_id_map.get(transition['from'], '0')
            to_elem = ET.SubElement(trans_elem, "to")
            to_elem.text = state_id_map.get(transition['to'], '0')
            
            # Input symbol
            read_elem = ET.SubElement(trans_elem, "read")
            read_elem.text = transition.get('symbol', '')
            
            # PDA specific
            if automaton.get('type') == 'pda':
                if 'stack_pop' in transition:
                    pop_elem = ET.SubElement(trans_elem, "pop")
                    pop_elem.text = transition['stack_pop']
                if 'stack_push' in transition:
                    push_elem = ET.SubElement(trans_elem, "push")
                    push_elem.text = transition['stack_push']
            
            # TM specific
            if automaton.get('type') == 'tm':
                if 'tape_write' in transition:
                    write_elem = ET.SubElement(trans_elem, "write")
                    write_elem.text = transition['tape_write']
                if 'tape_move' in transition:
                    move_elem = ET.SubElement(trans_elem, "move")
                    move_elem.text = transition['tape_move']
        
        return ET.tostring(root, encoding='unicode')
    
    def validate_jflap_format(self, content: str) -> Tuple[bool, str]:
        """
        Validate if content is valid JFLAP format
        
        Args:
            content: File content to validate
            
        Returns:
            (is_valid, error_message)
        """
        try:
            root = ET.fromstring(content)
            
            # Check for required elements
            if root.tag != 'structure':
                return False, "Root element must be 'structure'"
            
            automaton = root.find('automaton')
            if automaton is None:
                return False, "Missing 'automaton' element"
            
            # Check for at least one state
            states = automaton.findall('state')
            if not states:
                return False, "No states found"
            
            # Check for initial state
            has_initial = any(state.find('initial') is not None for state in states)
            if not has_initial:
                return False, "No initial state defined"
            
            return True, "Valid JFLAP format"
            
        except ET.ParseError as e:
            return False, f"XML parsing error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _to_internal_format(self, states: List[JFLAPState], 
                           transitions: List[JFLAPTransition],
                           automaton_type: str) -> Dict[str, Any]:
        """Convert JFLAP structures to internal format"""
        
        # Find initial and final states
        initial_states = [s for s in states if s.is_initial]
        final_states = [s for s in states if s.is_final]
        
        # Build alphabet from transitions
        alphabet = set()
        for trans in transitions:
            if trans.input_symbol and trans.input_symbol != 'ε':
                alphabet.add(trans.input_symbol)
        
        return {
            'type': self._map_jflap_type(automaton_type),
            'states': [
                {
                    'name': s.name,
                    'x': s.x,
                    'y': s.y,
                    'is_initial': s.is_initial,
                    'is_final': s.is_final
                }
                for s in states
            ],
            'transitions': [
                {
                    'from': t.from_state,
                    'to': t.to_state,
                    'symbol': t.input_symbol,
                    'stack_pop': t.stack_pop,
                    'stack_push': t.stack_push,
                    'tape_write': t.tape_write,
                    'tape_move': t.tape_move
                }
                for t in transitions
            ],
            'alphabet': list(alphabet),
            'initial_state': initial_states[0].name if initial_states else None,
            'final_states': [s.name for s in final_states]
        }
    
    def _get_jflap_type(self, internal_type: str) -> str:
        """Map internal type to JFLAP type"""
        mapping = {
            'dfa': 'fa',
            'nfa': 'fa',
            'pda': 'pda',
            'tm': 'turing',
            'cfg': 'grammar'
        }
        return mapping.get(internal_type, 'fa')
    
    def _map_jflap_type(self, jflap_type: str) -> str:
        """Map JFLAP type to internal type"""
        mapping = {
            'fa': 'nfa',  # JFLAP doesn't distinguish DFA/NFA in type
            'pda': 'pda',
            'turing': 'tm',
            'grammar': 'cfg'
        }
        return mapping.get(jflap_type, 'nfa')

class JFLAPSimulator:
    """Simulate JFLAP-style step-by-step execution"""
    
    def __init__(self, automaton: Dict[str, Any]):
        self.automaton = automaton
        self.current_states = set()
        self.stack = []  # For PDA
        self.tape = []   # For TM
        self.head_position = 0  # For TM
        
    def reset(self):
        """Reset simulator to initial state"""
        self.current_states = {self.automaton['initial_state']}
        self.stack = []
        self.tape = []
        self.head_position = 0
    
    def step(self, input_symbol: str) -> Dict[str, Any]:
        """
        Execute one step of simulation
        
        Returns:
            Dict with current configuration and possible next moves
        """
        next_states = set()
        moves = []
        
        for state in self.current_states:
            # Find applicable transitions
            for transition in self.automaton['transitions']:
                if transition['from'] == state:
                    if transition['symbol'] == input_symbol or transition['symbol'] == 'ε':
                        next_states.add(transition['to'])
                        moves.append({
                            'from': state,
                            'to': transition['to'],
                            'symbol': transition['symbol'],
                            'type': 'transition'
                        })
        
        self.current_states = next_states
        
        return {
            'current_states': list(self.current_states),
            'possible_moves': moves,
            'is_accepting': any(s in self.automaton['final_states'] 
                               for s in self.current_states),
            'stack': self.stack.copy() if self.automaton['type'] == 'pda' else None,
            'tape': self.tape.copy() if self.automaton['type'] == 'tm' else None
        }
    
    def run(self, input_string: str) -> List[Dict[str, Any]]:
        """
        Run complete simulation on input string
        
        Returns:
            List of configurations at each step
        """
        self.reset()
        configurations = []
        
        for symbol in input_string:
            config = self.step(symbol)
            configurations.append(config)
        
        return configurations

# API Integration functions
async def import_jflap_file(file_content: str) -> Dict[str, Any]:
    """Import JFLAP file and convert to internal format"""
    converter = JFLAPConverter()
    
    # Validate format
    is_valid, error_msg = converter.validate_jflap_format(file_content)
    if not is_valid:
        raise ValueError(f"Invalid JFLAP file: {error_msg}")
    
    # Convert to internal format
    automaton = converter.import_jff(file_content)
    
    return {
        "success": True,
        "automaton": automaton,
        "type": automaton['type'],
        "states_count": len(automaton['states']),
        "transitions_count": len(automaton['transitions'])
    }

async def export_to_jflap(automaton: Dict[str, Any]) -> str:
    """Export internal automaton to JFLAP format"""
    converter = JFLAPConverter()
    jff_content = converter.export_jff(automaton)
    return jff_content

async def simulate_jflap_style(automaton: Dict[str, Any], 
                               input_string: str) -> List[Dict[str, Any]]:
    """Run JFLAP-style step-by-step simulation"""
    simulator = JFLAPSimulator(automaton)
    configurations = simulator.run(input_string)
    return configurations