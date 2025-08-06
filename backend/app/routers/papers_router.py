"""
Papers router for the Automata Learning Platform.
Handles research paper search, recommendations, and citation management.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging

from ..database import get_db, User
from ..auth import get_current_active_user
from ..papers import (
    papers_database,
    PaperSearchRequest,
    CitationRequest,
    RecommendationEngine
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/papers", tags=["papers"])


class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    offset: int = 0


class RecommendRequest(BaseModel):
    user_interests: List[str]
    recent_papers: Optional[List[str]] = None
    limit: int = 5


class CitationGenerationRequest(BaseModel):
    paper_id: str
    citation_style: str = "apa"  # apa, mla, ieee, bibtex


@router.post("/search")
async def search_papers(
    request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Search research papers using various criteria"""
    try:
        search_request = PaperSearchRequest(
            query=request.query,
            filters=request.filters or {},
            limit=request.limit,
            offset=request.offset
        )
        
        results = await papers_database.search(search_request)
        
        logger.info(f"Paper search performed by user {current_user.id}")
        
        return {
            "query": request.query,
            "results": results.papers,
            "total_found": results.total_count,
            "search_metadata": {
                "search_time_ms": results.search_time,
                "filters_applied": request.filters,
                "relevance_threshold": results.relevance_threshold
            },
            "facets": results.facets,  # Categories, years, authors, etc.
            "suggestions": results.query_suggestions
        }
        
    except Exception as e:
        logger.error(f"Paper search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend")
async def recommend_papers(
    request: RecommendRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get personalized paper recommendations"""
    try:
        recommendation_engine = RecommendationEngine()
        
        recommendations = await recommendation_engine.get_recommendations(
            user_id=str(current_user.id),
            interests=request.user_interests,
            recent_papers=request.recent_papers or [],
            limit=request.limit
        )
        
        logger.info(f"Paper recommendations generated for user {current_user.id}")
        
        return {
            "user_id": str(current_user.id),
            "recommendations": recommendations.papers,
            "recommendation_reasons": recommendations.explanations,
            "algorithm_used": recommendations.algorithm,
            "confidence_scores": recommendations.confidence_scores,
            "diversity_score": recommendations.diversity_score,
            "generated_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Paper recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/citation")
async def generate_citation(
    request: CitationGenerationRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate citation for a research paper in specified format"""
    try:
        paper = await papers_database.get_paper(request.paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        citation_request = CitationRequest(
            paper_id=request.paper_id,
            style=request.citation_style
        )
        
        citation = await papers_database.generate_citation(citation_request)
        
        return {
            "paper_id": request.paper_id,
            "citation_style": request.citation_style,
            "citation": citation.formatted_citation,
            "bibtex_entry": citation.bibtex if request.citation_style == "bibtex" else None,
            "paper_metadata": {
                "title": paper.title,
                "authors": paper.authors,
                "venue": paper.venue,
                "year": paper.year,
                "doi": paper.doi
            }
        }
        
    except Exception as e:
        logger.error(f"Citation generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topic/{topic}")
async def get_papers_by_topic(
    topic: str,
    limit: int = 20,
    sort_by: str = "relevance",  # relevance, date, citations
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get papers related to a specific topic"""
    try:
        papers = await papers_database.get_by_topic(
            topic=topic,
            limit=limit,
            sort_by=sort_by
        )
        
        return {
            "topic": topic,
            "papers": papers.papers,
            "total_count": papers.total_count,
            "topic_metadata": {
                "related_topics": papers.related_topics,
                "trending_papers": papers.trending_papers,
                "key_researchers": papers.key_researchers
            },
            "sort_by": sort_by,
            "last_updated": papers.last_updated
        }
        
    except Exception as e:
        logger.error(f"Topic papers retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{paper_id}")
async def get_paper_details(
    paper_id: str,
    include_citations: bool = True,
    include_references: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific paper"""
    try:
        paper = await papers_database.get_paper_details(
            paper_id=paper_id,
            include_citations=include_citations,
            include_references=include_references
        )
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Track paper view for recommendations
        await papers_database.track_paper_view(
            user_id=str(current_user.id),
            paper_id=paper_id
        )
        
        return {
            "paper": {
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "venue": paper.venue,
                "year": paper.year,
                "doi": paper.doi,
                "pdf_url": paper.pdf_url,
                "keywords": paper.keywords,
                "categories": paper.categories
            },
            "metrics": {
                "citation_count": paper.citation_count,
                "h_index_contribution": paper.h_index_contribution,
                "altmetric_score": paper.altmetric_score,
                "view_count": paper.view_count
            },
            "citations": paper.citations if include_citations else None,
            "references": paper.references if include_references else None,
            "related_papers": paper.related_papers,
            "impact_analysis": paper.impact_analysis
        }
        
    except Exception as e:
        logger.error(f"Paper details retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_database_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get statistics about the papers database"""
    try:
        stats = await papers_database.get_statistics()
        
        return {
            "database_stats": {
                "total_papers": stats.total_papers,
                "total_authors": stats.total_authors,
                "total_venues": stats.total_venues,
                "date_range": {
                    "earliest_year": stats.earliest_year,
                    "latest_year": stats.latest_year
                }
            },
            "content_distribution": {
                "by_category": stats.category_distribution,
                "by_year": stats.year_distribution,
                "by_venue": stats.top_venues
            },
            "quality_metrics": {
                "papers_with_abstracts": stats.papers_with_abstracts,
                "papers_with_pdfs": stats.papers_with_pdfs,
                "papers_with_citations": stats.papers_with_citations,
                "average_citation_count": stats.avg_citation_count
            },
            "recent_activity": {
                "papers_added_last_week": stats.recent_additions,
                "most_viewed_papers": stats.trending_papers,
                "most_cited_recent": stats.most_cited_recent
            },
            "last_updated": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bookmark/{paper_id}")
async def bookmark_paper(
    paper_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Bookmark a paper for later reference"""
    try:
        bookmark_id = await papers_database.add_bookmark(
            user_id=str(current_user.id),
            paper_id=paper_id
        )
        
        return {
            "bookmark_id": bookmark_id,
            "paper_id": paper_id,
            "user_id": str(current_user.id),
            "bookmarked_at": "2025-08-05T16:27:32Z",
            "status": "bookmarked"
        }
        
    except Exception as e:
        logger.error(f"Paper bookmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/bookmark/{paper_id}")
async def remove_bookmark(
    paper_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Remove a paper bookmark"""
    try:
        success = await papers_database.remove_bookmark(
            user_id=str(current_user.id),
            paper_id=paper_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Bookmark not found")
        
        return {
            "paper_id": paper_id,
            "user_id": str(current_user.id),
            "status": "bookmark_removed"
        }
        
    except Exception as e:
        logger.error(f"Bookmark removal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bookmarks/list")
async def list_bookmarks(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    """List user's bookmarked papers"""
    try:
        bookmarks = await papers_database.get_user_bookmarks(
            user_id=str(current_user.id),
            limit=limit,
            offset=offset
        )
        
        return {
            "user_id": str(current_user.id),
            "bookmarks": bookmarks.papers,
            "total_bookmarks": bookmarks.total_count,
            "bookmark_categories": bookmarks.categories,
            "last_updated": bookmarks.last_updated
        }
        
    except Exception as e:
        logger.error(f"Bookmarks list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review/{paper_id}")
async def submit_paper_review(
    paper_id: str,
    review: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit a review or rating for a paper"""
    try:
        review_id = await papers_database.add_review(
            user_id=str(current_user.id),
            paper_id=paper_id,
            review_data=review
        )
        
        return {
            "review_id": review_id,
            "paper_id": paper_id,
            "user_id": str(current_user.id),
            "review_submitted": True,
            "moderation_status": "pending",
            "submitted_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Paper review submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bibliography")
async def export_bibliography(
    paper_ids: List[str],
    format: str = "bibtex",  # bibtex, ris, endnote
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export a bibliography in the specified format"""
    try:
        bibliography = await papers_database.export_bibliography(
            paper_ids=paper_ids,
            format=format
        )
        
        return {
            "format": format,
            "paper_count": len(paper_ids),
            "bibliography": bibliography.content,
            "filename": bibliography.suggested_filename,
            "exported_at": "2025-08-05T16:27:32Z"
        }
        
    except Exception as e:
        logger.error(f"Bibliography export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))