"""
LaTeX export router for the automata learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from ..database import get_db
from ..latex_export import (
    latex_exporter, ExportRequest, ExportFormat, DocumentStyle,
    export_automaton_to_latex, export_grammar_to_latex,
    export_proof_to_latex, export_complete_document_to_latex,
    get_latex_templates
)
from ..api_platform import get_current_client, require_scope, APIScope
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["LaTeX Export"])

@router.get("/templates")
async def get_available_templates():
    """Get list of available LaTeX templates."""
    try:
        templates = get_latex_templates()
        return {
            "templates": templates,
            "formats": list(ExportFormat.__members__.keys()),
            "styles": list(DocumentStyle.__members__.keys())
        }
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve templates"
        )

@router.post("/automaton")
async def export_automaton(
    automaton_data: Dict[str, Any],
    template: str = "default",
    client_info = Depends(require_scope(APIScope.EXPORT))
):
    """Export automaton to LaTeX TikZ format."""
    try:
        latex_code = await export_automaton_to_latex(automaton_data, template)
        
        return Response(
            content=latex_code,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=automaton_{template}.tex"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error exporting automaton: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export automaton"
        )

@router.post("/grammar")
async def export_grammar(
    grammar_data: Dict[str, Any],
    template: str = "default",
    client_info = Depends(require_scope(APIScope.EXPORT))
):
    """Export grammar to LaTeX format."""
    try:
        latex_code = await export_grammar_to_latex(grammar_data, template)
        
        return Response(
            content=latex_code,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=grammar_{template}.tex"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error exporting grammar: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export grammar"
        )

@router.post("/proof")
async def export_proof(
    proof_data: Dict[str, Any],
    template: str = "default",
    client_info = Depends(require_scope(APIScope.EXPORT))
):
    """Export proof to LaTeX theorem environment."""
    try:
        latex_code = await export_proof_to_latex(proof_data, template)
        
        return Response(
            content=latex_code,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=proof_{template}.tex"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error exporting proof: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export proof"
        )

@router.post("/document")
async def export_complete_document(
    export_request: ExportRequest,
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.EXPORT))
):
    """Export complete LaTeX document with multiple components."""
    try:
        # Validate export request
        errors = await latex_exporter.validate_export_request(export_request)
        if errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid export request: {'; '.join(errors)}"
            )
        
        latex_code = await export_complete_document_to_latex(export_request, db)
        
        filename = f"automata_export_{export_request.template or 'default'}.tex"
        
        return Response(
            content=latex_code,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error exporting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export document"
        )

@router.post("/batch")
async def batch_export(
    export_requests: List[ExportRequest],
    db: Session = Depends(get_db),
    client_info = Depends(require_scope(APIScope.EXPORT))
):
    """Export multiple documents in batch."""
    try:
        if len(export_requests) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 exports per batch request"
            )
        
        results = []
        
        for i, request in enumerate(export_requests):
            try:
                # Validate request
                errors = await latex_exporter.validate_export_request(request)
                if errors:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": "; ".join(errors)
                    })
                    continue
                
                latex_code = await export_complete_document_to_latex(request, db)
                
                results.append({
                    "index": i,
                    "status": "success",
                    "latex_code": latex_code,
                    "template": request.template or "default"
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": str(e)
                })
        
        return {"batch_results": results}
        
    except Exception as e:
        logger.error(f"Error in batch export: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch export"
        )

@router.get("/formats")
async def get_export_formats():
    """Get available export formats and their descriptions."""
    return {
        "formats": {
            "tikz": {
                "name": "TikZ Diagram",
                "description": "Standalone TikZ diagram for automata",
                "file_extension": ".tex"
            },
            "automata": {
                "name": "Automata Document",
                "description": "Complete document with automata diagrams",
                "file_extension": ".tex"
            },
            "algorithm": {
                "name": "Algorithm Document",
                "description": "Document with algorithms and pseudocode",
                "file_extension": ".tex"
            },
            "proof": {
                "name": "Proof Document",
                "description": "Document with formal proofs",
                "file_extension": ".tex"
            },
            "grammar": {
                "name": "Grammar Document",
                "description": "Document with grammar specifications",
                "file_extension": ".tex"
            },
            "complete": {
                "name": "Complete Export",
                "description": "Full document with all components",
                "file_extension": ".tex"
            }
        }
    }

@router.get("/preview/{format}")
async def preview_export_format(
    format: ExportFormat,
    template: str = "default"
):
    """Get preview of export format with sample data."""
    try:
        sample_data = {
            "states": [
                {"id": "q0", "label": "q0", "x": 0, "y": 0},
                {"id": "q1", "label": "q1", "x": 3, "y": 0},
                {"id": "q2", "label": "q2", "x": 6, "y": 0}
            ],
            "transitions": [
                {"from": "q0", "to": "q1", "symbol": "a"},
                {"from": "q1", "to": "q2", "symbol": "b"},
                {"from": "q2", "to": "q2", "symbol": "a,b"}
            ],
            "alphabet": ["a", "b"],
            "start_state": "q0",
            "accept_states": ["q2"]
        }
        
        if format == ExportFormat.TIKZ:
            preview = await latex_exporter.export_automaton(sample_data, template)
        elif format == ExportFormat.GRAMMAR:
            grammar_sample = {
                "variables": ["S", "A", "B"],
                "terminals": ["a", "b"],
                "productions": [
                    {"left": "S", "right": "aA | bB"},
                    {"left": "A", "right": "aS | b"},
                    {"left": "B", "right": "bS | a"}
                ],
                "start_variable": "S"
            }
            preview = await latex_exporter.export_grammar(grammar_sample, template)
        elif format == ExportFormat.PROOF:
            proof_sample = {
                "type": "theorem",
                "statement": "The language L = {a^n b^n | n ≥ 0} is context-free.",
                "steps": [
                    {
                        "text": "We construct a PDA that recognizes L.",
                        "justification": "Construction"
                    },
                    {
                        "text": "The PDA uses a stack to count a's and match them with b's.",
                        "justification": "Stack operation"
                    },
                    {
                        "text": "Therefore, L is context-free.",
                        "justification": "Definition of context-free languages"
                    }
                ]
            }
            preview = await latex_exporter.export_proof(proof_sample, template)
        else:
            preview = "% Sample LaTeX code for format: " + format.value
        
        return {
            "format": format.value,
            "template": template,
            "preview": preview
        }
        
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate preview"
        )