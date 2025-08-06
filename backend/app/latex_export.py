"""
LaTeX export functionality for the automata learning platform.
Supports exporting automata, grammars, and proofs to LaTeX format with TikZ diagrams.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from .database import get_db, User, Problem, Solution
import logging

logger = logging.getLogger(__name__)

class ExportFormat(str, Enum):
    """Supported LaTeX export formats."""
    TIKZ = "tikz"
    AUTOMATA = "automata"
    ALGORITHM = "algorithm"
    PROOF = "proof"
    GRAMMAR = "grammar"
    COMPLETE = "complete"

class DocumentStyle(str, Enum):
    """LaTeX document styles."""
    ARTICLE = "article"
    REPORT = "report"
    BOOK = "book"
    BEAMER = "beamer"
    STANDALONE = "standalone"

class LaTeXTemplate(BaseModel):
    """LaTeX template configuration."""
    name: str
    style: DocumentStyle
    packages: List[str] = Field(default_factory=list)
    custom_commands: List[str] = Field(default_factory=list)
    tikz_libraries: List[str] = Field(default_factory=list)

class ExportRequest(BaseModel):
    """Request model for LaTeX export."""
    format: ExportFormat
    template: Optional[str] = "default"
    include_solutions: bool = True
    include_proofs: bool = True
    custom_style: Optional[Dict[str, Any]] = None
    problem_ids: Optional[List[str]] = None
    user_id: Optional[str] = None

class LaTeXExporter:
    """Main LaTeX export class."""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.tikz_styles = self._load_tikz_styles()
    
    def _load_templates(self) -> Dict[str, LaTeXTemplate]:
        """Load predefined LaTeX templates."""
        templates = {
            "default": LaTeXTemplate(
                name="Default Article",
                style=DocumentStyle.ARTICLE,
                packages=[
                    "amsmath", "amssymb", "amsfonts", "amsthm",
                    "tikz", "pgfplots", "algorithm", "algorithmic",
                    "geometry", "fancyhdr", "hyperref", "xcolor"
                ],
                tikz_libraries=[
                    "automata", "positioning", "arrows", "shapes",
                    "backgrounds", "calc", "decorations.pathmorphing"
                ],
                custom_commands=[
                    r"\newtheorem{theorem}{Theorem}",
                    r"\newtheorem{lemma}{Lemma}",
                    r"\newtheorem{definition}{Definition}",
                    r"\newtheorem{proof}{Proof}",
                ]
            ),
            "academic": LaTeXTemplate(
                name="Academic Paper",
                style=DocumentStyle.ARTICLE,
                packages=[
                    "amsmath", "amssymb", "amsthm", "tikz", "algorithm",
                    "algorithmic", "cite", "url", "graphicx", "subcaption"
                ],
                tikz_libraries=["automata", "positioning", "arrows"],
                custom_commands=[
                    r"\theoremstyle{definition}",
                    r"\newtheorem{definition}{Definition}",
                    r"\theoremstyle{plain}",
                    r"\newtheorem{theorem}{Theorem}",
                    r"\newtheorem{lemma}[theorem]{Lemma}",
                    r"\theoremstyle{remark}",
                    r"\newtheorem{remark}{Remark}",
                ]
            ),
            "presentation": LaTeXTemplate(
                name="Beamer Presentation",
                style=DocumentStyle.BEAMER,
                packages=["tikz", "algorithm", "algorithmic"],
                tikz_libraries=["automata", "positioning"],
                custom_commands=[]
            ),
            "homework": LaTeXTemplate(
                name="Homework Template",
                style=DocumentStyle.ARTICLE,
                packages=[
                    "amsmath", "amssymb", "tikz", "enumerate",
                    "fancyhdr", "lastpage"
                ],
                tikz_libraries=["automata", "positioning"],
                custom_commands=[
                    r"\pagestyle{fancy}",
                    r"\lhead{Theory of Computation}",
                    r"\rhead{\today}",
                    r"\cfoot{Page \thepage\ of \pageref{LastPage}}"
                ]
            )
        }
        return templates
    
    def _load_tikz_styles(self) -> Dict[str, str]:
        """Load TikZ style definitions."""
        return {
            "state": """
                state/.style={
                    circle,
                    draw=black,
                    minimum size=0.8cm,
                    inner sep=0pt
                }
            """,
            "accept_state": """
                accept/.style={
                    state,
                    double,
                    double distance=2pt
                }
            """,
            "initial_state": """
                initial/.style={
                    state,
                    fill=lightgray
                }
            """,
            "transition": """
                transition/.style={
                    ->,
                    >=stealth,
                    bend right=15,
                    auto,
                    node distance=2.5cm
                }
            """,
            "self_loop": """
                loop/.style={
                    ->,
                    >=stealth,
                    loop above,
                    min distance=15mm
                }
            """
        }
    
    async def export_automaton(self, automaton_data: Dict[str, Any], 
                             template: str = "default") -> str:
        """Export automaton to LaTeX TikZ format."""
        try:
            template_config = self.templates[template]
            
            # Extract automaton components
            states = automaton_data.get("states", [])
            transitions = automaton_data.get("transitions", [])
            alphabet = automaton_data.get("alphabet", [])
            start_state = automaton_data.get("start_state")
            accept_states = automaton_data.get("accept_states", [])
            
            # Generate TikZ code
            tikz_code = self._generate_automaton_tikz(
                states, transitions, start_state, accept_states
            )
            
            # Wrap in document if needed
            if template_config.style == DocumentStyle.STANDALONE:
                return self._wrap_standalone_tikz(tikz_code, template_config)
            else:
                return tikz_code
                
        except Exception as e:
            logger.error(f"Error exporting automaton: {e}")
            raise
    
    def _generate_automaton_tikz(self, states: List[Dict], transitions: List[Dict],
                                start_state: str, accept_states: List[str]) -> str:
        """Generate TikZ code for automaton diagram."""
        lines = [
            "\\begin{tikzpicture}[>=stealth, auto, node distance=2.5cm]",
            "  % Define styles"
        ]
        
        # Add style definitions
        for style_name, style_def in self.tikz_styles.items():
            lines.append(f"  {style_def.strip()}")
        
        lines.append("\n  % States")
        
        # Generate states with positions
        positions = self._calculate_state_positions(states)
        
        for i, state in enumerate(states):
            state_id = state.get("id", f"q{i}")
            label = state.get("label", state_id)
            x, y = positions.get(state_id, (i * 3, 0))
            
            # Determine state style
            style_parts = ["state"]
            if state_id == start_state:
                style_parts.append("initial")
            if state_id in accept_states:
                style_parts.append("accept")
            
            style = ",".join(style_parts)
            lines.append(f"  \\node[{style}] ({state_id}) at ({x}, {y}) {{{label}}};")
        
        lines.append("\n  % Transitions")
        
        # Generate transitions
        transition_groups = self._group_transitions(transitions)
        
        for (from_state, to_state), labels in transition_groups.items():
            labels_str = ",".join(labels)
            
            if from_state == to_state:
                # Self-loop
                lines.append(f"  \\path ({from_state}) edge[loop above] node {{{labels_str}}} ({to_state});")
            else:
                # Regular transition
                lines.append(f"  \\path ({from_state}) edge[->] node {{{labels_str}}} ({to_state});")
        
        # Add initial state arrow
        if start_state:
            lines.append(f"  \\path[->] (-1, 0) edge ({start_state});")
        
        lines.append("\\end{tikzpicture}")
        
        return "\n".join(lines)
    
    def _calculate_state_positions(self, states: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Calculate optimal positions for states in TikZ diagram."""
        positions = {}
        n = len(states)
        
        if n == 0:
            return positions
        
        if n == 1:
            state_id = states[0].get("id", "q0")
            positions[state_id] = (0, 0)
        elif n <= 6:
            # Arrange in a circle
            import math
            radius = 3
            angle_step = 2 * math.pi / n
            
            for i, state in enumerate(states):
                state_id = state.get("id", f"q{i}")
                angle = i * angle_step
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions[state_id] = (round(x, 2), round(y, 2))
        else:
            # Arrange in a grid
            cols = int(math.ceil(math.sqrt(n)))
            for i, state in enumerate(states):
                state_id = state.get("id", f"q{i}")
                row = i // cols
                col = i % cols
                x = col * 3
                y = -row * 2.5
                positions[state_id] = (x, y)
        
        return positions
    
    def _group_transitions(self, transitions: List[Dict]) -> Dict[Tuple[str, str], List[str]]:
        """Group transitions by source and target states."""
        groups = {}
        
        for transition in transitions:
            from_state = transition.get("from")
            to_state = transition.get("to")
            symbol = transition.get("symbol", "ε")
            
            key = (from_state, to_state)
            if key not in groups:
                groups[key] = []
            groups[key].append(symbol)
        
        return groups
    
    async def export_grammar(self, grammar_data: Dict[str, Any], 
                           template: str = "default") -> str:
        """Export context-free grammar to LaTeX format."""
        try:
            variables = grammar_data.get("variables", [])
            terminals = grammar_data.get("terminals", [])
            productions = grammar_data.get("productions", [])
            start_variable = grammar_data.get("start_variable")
            
            lines = [
                "\\begin{align*}",
                f"G &= (V, T, P, {start_variable}) \\\\",
                f"V &= \\{{{', '.join(variables)}\\}} \\\\",
                f"T &= \\{{{', '.join(terminals)}\\}} \\\\",
                "P &= \\begin{cases}"
            ]
            
            # Format productions
            for production in productions:
                left = production.get("left", "")
                right = production.get("right", "")
                if isinstance(right, list):
                    right = " | ".join(right)
                lines.append(f"    {left} &\\to {right} \\\\")
            
            lines.extend([
                "\\end{cases}",
                "\\end{align*}"
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error exporting grammar: {e}")
            raise
    
    async def export_proof(self, proof_data: Dict[str, Any], 
                         template: str = "default") -> str:
        """Export formal proof to LaTeX theorem environment."""
        try:
            theorem_type = proof_data.get("type", "theorem")
            statement = proof_data.get("statement", "")
            proof_steps = proof_data.get("steps", [])
            
            lines = [
                f"\\begin{{{theorem_type}}}",
                statement,
                f"\\end{{{theorem_type}}}",
                "",
                "\\begin{proof}"
            ]
            
            for i, step in enumerate(proof_steps, 1):
                step_text = step.get("text", "")
                justification = step.get("justification", "")
                
                if justification:
                    lines.append(f"({i}) {step_text} \\quad \\text{{({justification})}}")
                else:
                    lines.append(f"({i}) {step_text}")
            
            lines.append("\\end{proof}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error exporting proof: {e}")
            raise
    
    async def export_complete_document(self, export_request: ExportRequest,
                                     db: Session) -> str:
        """Export complete LaTeX document with multiple components."""
        try:
            template_config = self.templates[export_request.template or "default"]
            
            # Document header
            lines = [
                f"\\documentclass{{{template_config.style.value}}}",
                ""
            ]
            
            # Packages
            for package in template_config.packages:
                lines.append(f"\\usepackage{{{package}}}")
            
            # TikZ libraries
            if template_config.tikz_libraries:
                tikz_libs = ",".join(template_config.tikz_libraries)
                lines.append(f"\\usetikzlibrary{{{tikz_libs}}}")
            
            lines.append("")
            
            # Custom commands
            for command in template_config.custom_commands:
                lines.append(command)
            
            lines.extend([
                "",
                "\\title{Theory of Computation - Exported Solutions}",
                "\\author{Generated by Automata Learning Platform}",
                f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}",
                "",
                "\\begin{document}",
                "\\maketitle",
                ""
            ])
            
            # Export problems and solutions
            if export_request.problem_ids:
                problems = db.query(Problem).filter(
                    Problem.id.in_(export_request.problem_ids)
                ).all()
                
                for problem in problems:
                    lines.extend(await self._export_problem_section(
                        problem, export_request, db
                    ))
            
            lines.extend([
                "",
                "\\end{document}"
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error exporting complete document: {e}")
            raise
    
    async def _export_problem_section(self, problem: Problem, 
                                    export_request: ExportRequest,
                                    db: Session) -> List[str]:
        """Export a single problem section."""
        lines = [
            f"\\section{{{problem.title}}}",
            "",
            f"\\textbf{{Type:}} {problem.type.upper()}",
            "",
            f"\\textbf{{Description:}} {problem.description}",
            "",
            f"\\textbf{{Language:}} {problem.language_description}",
            ""
        ]
        
        # Add alphabet
        if problem.alphabet:
            alphabet_str = ", ".join(problem.alphabet)
            lines.extend([
                f"\\textbf{{Alphabet:}} $\\Sigma = \\{{{alphabet_str}\\}}$",
                ""
            ])
        
        # Add test strings
        if problem.test_strings:
            lines.extend([
                "\\textbf{Test Strings:}",
                "\\begin{itemize}"
            ])
            
            for test in problem.test_strings[:10]:  # Limit to first 10
                string = test.get("string", "")
                should_accept = test.get("should_accept", False)
                result = "accept" if should_accept else "reject"
                lines.append(f"    \\item $w = {string}$ (should {result})")
            
            lines.extend([
                "\\end{itemize}",
                ""
            ])
        
        # Add reference solution if available
        if export_request.include_solutions and problem.reference_solution:
            lines.extend([
                "\\subsection{Reference Solution}",
                ""
            ])
            
            # Export automaton diagram
            try:
                tikz_code = await self.export_automaton(problem.reference_solution)
                lines.extend([
                    "\\begin{center}",
                    tikz_code,
                    "\\end{center}",
                    ""
                ])
            except Exception as e:
                logger.warning(f"Could not export reference solution diagram: {e}")
        
        # Add user solutions if requested
        if export_request.include_solutions and export_request.user_id:
            user_solutions = db.query(Solution).filter(
                Solution.problem_id == problem.id,
                Solution.user_id == export_request.user_id,
                Solution.is_correct == True
            ).order_by(Solution.submitted_at.desc()).limit(1).all()
            
            if user_solutions:
                lines.extend([
                    "\\subsection{User Solution}",
                    ""
                ])
                
                solution = user_solutions[0]
                try:
                    tikz_code = await self.export_automaton(solution.automaton_data)
                    lines.extend([
                        "\\begin{center}",
                        tikz_code,
                        "\\end{center}",
                        f"\\textbf{{Score:}} {solution.score:.1f}\\%",
                        f"\\textbf{{Submitted:}} {solution.submitted_at.strftime('%B %d, %Y at %I:%M %p')}",
                        ""
                    ])
                except Exception as e:
                    logger.warning(f"Could not export user solution diagram: {e}")
        
        lines.append("\\clearpage")
        
        return lines
    
    def _wrap_standalone_tikz(self, tikz_code: str, template_config: LaTeXTemplate) -> str:
        """Wrap TikZ code in standalone document."""
        lines = [
            "\\documentclass[tikz,border=10pt]{standalone}",
            ""
        ]
        
        # TikZ libraries
        if template_config.tikz_libraries:
            tikz_libs = ",".join(template_config.tikz_libraries)
            lines.append(f"\\usetikzlibrary{{{tikz_libs}}}")
        
        lines.extend([
            "",
            "\\begin{document}",
            tikz_code,
            "\\end{document}"
        ])
        
        return "\n".join(lines)
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get list of available templates."""
        return {name: template.name for name, template in self.templates.items()}
    
    async def validate_export_request(self, request: ExportRequest) -> List[str]:
        """Validate export request and return any errors."""
        errors = []
        
        if request.template and request.template not in self.templates:
            errors.append(f"Unknown template: {request.template}")
        
        if request.format not in ExportFormat.__members__.values():
            errors.append(f"Invalid export format: {request.format}")
        
        return errors


# Global exporter instance
latex_exporter = LaTeXExporter()


# Utility functions for router integration
async def export_automaton_to_latex(automaton_data: Dict[str, Any], 
                                  template: str = "default") -> str:
    """Export automaton to LaTeX format."""
    return await latex_exporter.export_automaton(automaton_data, template)


async def export_grammar_to_latex(grammar_data: Dict[str, Any], 
                                template: str = "default") -> str:
    """Export grammar to LaTeX format."""
    return await latex_exporter.export_grammar(grammar_data, template)


async def export_proof_to_latex(proof_data: Dict[str, Any], 
                               template: str = "default") -> str:
    """Export proof to LaTeX format."""
    return await latex_exporter.export_proof(proof_data, template)


async def export_complete_document_to_latex(export_request: ExportRequest,
                                          db: Session) -> str:
    """Export complete document to LaTeX format."""
    return await latex_exporter.export_complete_document(export_request, db)


def get_latex_templates() -> Dict[str, str]:
    """Get available LaTeX templates."""
    return latex_exporter.get_available_templates()