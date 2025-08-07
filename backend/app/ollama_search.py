"""
OLLAMA SEARCH - AI-Powered Search and Retrieval System
====================================================

This module uses Ollama for comprehensive search and retrieval:
- Natural language search understanding
- Query expansion and semantic enhancement
- Result ranking using AI intelligence
- Relevance scoring with context awareness
- Semantic similarity without separate embedding models
- Multi-modal search capabilities
- Context-aware result filtering
- Intelligent query suggestions
"""

import asyncio
import json
import logging
import re
import time
import hashlib
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import difflib

from .ollama_everything import ollama_everything, OllamaTask, OllamaTaskType, OllamaResult
from .valkey_integration import valkey_connection_manager
from .config import settings

logger = logging.getLogger(__name__)


class SearchType(str, Enum):
    """Types of search operations."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FUZZY = "fuzzy"
    CONTEXTUAL = "contextual"
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"


class ContentType(str, Enum):
    """Types of content being searched."""
    TEXT = "text"
    CODE = "code"
    DOCUMENTATION = "documentation"
    LOGS = "logs"
    RESEARCH_PAPERS = "research_papers"
    EDUCATIONAL_CONTENT = "educational_content"
    API_RESPONSES = "api_responses"
    USER_GENERATED = "user_generated"
    SYSTEM_DATA = "system_data"


@dataclass
class SearchQuery:
    """Structured search query."""
    original_query: str
    expanded_query: Optional[str] = None
    search_type: SearchType = SearchType.HYBRID
    content_types: List[ContentType] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 20
    min_relevance_score: float = 0.5
    boost_recent: bool = True
    boost_popular: bool = True
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Individual search result."""
    id: str
    title: str
    content: str
    content_type: ContentType
    relevance_score: float
    semantic_score: float
    keyword_score: float
    popularity_score: float
    recency_score: float
    final_score: float
    matched_terms: List[str] = field(default_factory=list)
    matched_concepts: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class SearchResponse:
    """Complete search response."""
    query: SearchQuery
    results: List[SearchResult]
    total_found: int
    search_time: float
    suggestions: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    facets: Dict[str, Dict] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)


class ContentIndexItem:
    """Represents an indexed content item."""
    
    def __init__(
        self,
        id: str,
        title: str,
        content: str,
        content_type: ContentType,
        metadata: Dict[str, Any] = None,
        created_at: datetime = None,
        updated_at: datetime = None
    ):
        self.id = id
        self.title = title
        self.content = content
        self.content_type = content_type
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        
        # Computed fields
        self.word_count = len(content.split())
        self.keywords = self._extract_keywords()
        self.concepts = []
        self.embedding_vector = None  # Would be computed by Ollama
        self.popularity_score = 0.0
        self.quality_score = 0.0
    
    def _extract_keywords(self) -> List[str]:
        """Extract keywords from content using simple NLP."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'was',
            'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may',
            'might', 'must', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her',
            'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them',
            'their', 'theirs'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', self.content.lower())
        
        # Filter out stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get most frequent keywords
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(50)]


class OllamaSearch:
    """AI-powered search engine using Ollama for all intelligence."""
    
    def __init__(self):
        # Content index
        self.content_index: Dict[str, ContentIndexItem] = {}
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Search intelligence
        self.query_cache: Dict[str, SearchResponse] = {}
        self.query_history: deque = deque(maxlen=10000)
        self.user_preferences: Dict[str, Dict] = defaultdict(dict)
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time": 0.0,
            "popular_queries": Counter()
        }
        
        # Configuration
        self.cache_ttl = 3600  # 1 hour
        self.max_query_expansion_terms = 10
        self.semantic_similarity_threshold = 0.7
        
        logger.info("OllamaSearch initialized with AI-powered intelligence")
    
    async def search(self, query: Union[str, SearchQuery], user_id: str = None) -> SearchResponse:
        """Perform intelligent search using Ollama."""
        start_time = time.time()
        
        # Normalize query
        if isinstance(query, str):
            query = SearchQuery(original_query=query)
        
        # Check cache first
        cache_key = self._generate_cache_key(query, user_id)
        cached_response = await self._check_search_cache(cache_key)
        if cached_response:
            self.search_stats["cache_hits"] += 1
            return cached_response
        
        try:
            # Enhance query using AI
            enhanced_query = await self._enhance_query_with_ai(query, user_id)
            
            # Perform multi-stage search
            preliminary_results = await self._perform_preliminary_search(enhanced_query)
            
            # Rank results using AI
            ranked_results = await self._rank_results_with_ai(preliminary_results, enhanced_query)
            
            # Generate search suggestions
            suggestions = await self._generate_suggestions(enhanced_query)
            
            # Generate related queries
            related_queries = await self._generate_related_queries(enhanced_query)
            
            # Build response
            search_time = time.time() - start_time
            response = SearchResponse(
                query=enhanced_query,
                results=ranked_results,
                total_found=len(preliminary_results),
                search_time=search_time,
                suggestions=suggestions,
                related_queries=related_queries,
                debug_info={
                    "preliminary_results_count": len(preliminary_results),
                    "enhancement_applied": enhanced_query.expanded_query is not None
                }
            )
            
            # Cache the response
            await self._cache_search_response(cache_key, response)
            
            # Update statistics
            self._update_search_stats(query, search_time)
            
            # Learn from search
            await self._learn_from_search(query, response, user_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty results rather than failing
            return SearchResponse(
                query=query,
                results=[],
                total_found=0,
                search_time=time.time() - start_time,
                debug_info={"error": str(e)}
            )
    
    async def _enhance_query_with_ai(self, query: SearchQuery, user_id: str = None) -> SearchQuery:
        """Enhance search query using Ollama intelligence."""
        
        # Build context for query enhancement
        context = {
            "original_query": query.original_query,
            "search_type": query.search_type.value,
            "content_types": [ct.value for ct in query.content_types],
            "user_context": self.user_preferences.get(user_id, {}) if user_id else {},
            "recent_queries": [q["query"] for q in list(self.query_history)[-10:]]
        }
        
        # Ask Ollama to enhance the query
        enhancement_task = OllamaTask(
            task_type=OllamaTaskType.TEXT_ENHANCEMENT,
            input_data=f"""
            Enhance this search query to improve search results:
            
            Original Query: {query.original_query}
            Search Context: {json.dumps(context, indent=2)}
            
            Please provide:
            1. Expanded query with synonyms and related terms
            2. Key concepts to search for
            3. Potential search refinements
            4. Alternative phrasings
            5. Technical terms if applicable
            
            Return a JSON response with:
            {{
                "expanded_query": "enhanced search terms",
                "key_concepts": ["concept1", "concept2"],
                "synonyms": ["synonym1", "synonym2"],
                "refinements": ["refinement1", "refinement2"],
                "search_intent": "what user is trying to find"
            }}
            """,
            context=context,
            temperature=0.4,
            max_tokens=1000
        )
        
        try:
            ai_result = await ollama_everything.process_task(enhancement_task)
            
            if ai_result.error:
                logger.warning(f"Query enhancement failed: {ai_result.error}")
                return query
            
            # Parse AI response
            enhancement_data = self._parse_query_enhancement(ai_result.result)
            
            if enhancement_data:
                # Create enhanced query
                enhanced_query = SearchQuery(
                    original_query=query.original_query,
                    expanded_query=enhancement_data.get("expanded_query"),
                    search_type=query.search_type,
                    content_types=query.content_types,
                    filters=query.filters,
                    context={
                        **query.context,
                        "key_concepts": enhancement_data.get("key_concepts", []),
                        "synonyms": enhancement_data.get("synonyms", []),
                        "search_intent": enhancement_data.get("search_intent"),
                        "ai_enhanced": True
                    },
                    max_results=query.max_results,
                    min_relevance_score=query.min_relevance_score,
                    boost_recent=query.boost_recent,
                    boost_popular=query.boost_popular,
                    user_preferences=query.user_preferences
                )
                
                return enhanced_query
            
        except Exception as e:
            logger.error(f"Query enhancement error: {e}")
        
        return query
    
    def _parse_query_enhancement(self, ai_result: Any) -> Optional[Dict[str, Any]]:
        """Parse AI query enhancement response."""
        try:
            result_str = str(ai_result)
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: parse structured response
            enhancement = {}
            
            # Extract expanded query
            expanded_match = re.search(r'expanded[_\s]*query[:\s]*["\']?(.*?)["\']?(?:\n|$)', result_str, re.IGNORECASE)
            if expanded_match:
                enhancement["expanded_query"] = expanded_match.group(1).strip()
            
            # Extract key concepts
            concepts_match = re.search(r'key[_\s]*concepts?[:\s]*\[(.*?)\]', result_str, re.IGNORECASE | re.DOTALL)
            if concepts_match:
                concepts_str = concepts_match.group(1)
                concepts = [c.strip().strip('"\'') for c in concepts_str.split(',')]
                enhancement["key_concepts"] = [c for c in concepts if c]
            
            return enhancement if enhancement else None
            
        except Exception as e:
            logger.error(f"Failed to parse query enhancement: {e}")
            return None
    
    async def _perform_preliminary_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform preliminary search across all indexed content."""
        results = []
        
        # Get search terms
        search_terms = self._extract_search_terms(query)
        
        # Search through indexed content
        for item_id, item in self.content_index.items():
            # Calculate different relevance scores
            keyword_score = self._calculate_keyword_relevance(search_terms, item)
            semantic_score = await self._calculate_semantic_relevance(query, item)
            popularity_score = item.popularity_score
            recency_score = self._calculate_recency_score(item)
            
            # Apply content type filtering
            if query.content_types and item.content_type not in query.content_types:
                continue
            
            # Apply other filters
            if not self._passes_filters(item, query.filters):
                continue
            
            # Calculate combined relevance score
            relevance_score = self._combine_relevance_scores(
                keyword_score, semantic_score, popularity_score, recency_score, query
            )
            
            # Only include if meets minimum relevance
            if relevance_score >= query.min_relevance_score:
                result = SearchResult(
                    id=item.id,
                    title=item.title,
                    content=item.content[:500] + "..." if len(item.content) > 500 else item.content,
                    content_type=item.content_type,
                    relevance_score=relevance_score,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    popularity_score=popularity_score,
                    recency_score=recency_score,
                    final_score=relevance_score,
                    matched_terms=self._find_matched_terms(search_terms, item),
                    metadata=item.metadata,
                    created_at=item.created_at,
                    updated_at=item.updated_at
                )
                
                results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda r: r.final_score, reverse=True)
        
        # Limit results
        return results[:query.max_results * 2]  # Get more for re-ranking
    
    def _extract_search_terms(self, query: SearchQuery) -> List[str]:
        """Extract search terms from query."""
        terms = []
        
        # Add original query terms
        original_terms = re.findall(r'\b\w+\b', query.original_query.lower())
        terms.extend(original_terms)
        
        # Add expanded query terms
        if query.expanded_query:
            expanded_terms = re.findall(r'\b\w+\b', query.expanded_query.lower())
            terms.extend(expanded_terms)
        
        # Add context terms
        if query.context.get("key_concepts"):
            terms.extend([c.lower() for c in query.context["key_concepts"]])
        
        if query.context.get("synonyms"):
            terms.extend([s.lower() for s in query.context["synonyms"]])
        
        # Deduplicate and filter
        return list(set([t for t in terms if len(t) > 2]))
    
    def _calculate_keyword_relevance(self, search_terms: List[str], item: ContentIndexItem) -> float:
        """Calculate keyword-based relevance score."""
        if not search_terms:
            return 0.0
        
        # Combine title and content for searching
        searchable_text = (item.title + " " + item.content).lower()
        
        # Count matches
        matches = 0
        total_terms = len(search_terms)
        
        for term in search_terms:
            if term in searchable_text:
                # Weight title matches higher
                title_matches = item.title.lower().count(term)
                content_matches = item.content.lower().count(term)
                
                term_score = (title_matches * 2.0) + (content_matches * 1.0)
                matches += min(term_score, 5.0)  # Cap individual term contribution
        
        # Calculate score as percentage of terms matched
        base_score = matches / total_terms if total_terms > 0 else 0.0
        
        # Apply length normalization (shorter documents get slight boost)
        length_factor = 1.0 + (1000.0 / max(item.word_count, 100))
        
        return min(base_score * length_factor, 1.0)
    
    async def _calculate_semantic_relevance(self, query: SearchQuery, item: ContentIndexItem) -> float:
        """Calculate semantic relevance using Ollama."""
        
        # For performance, cache semantic scores
        cache_key = f"semantic:{hashlib.md5((query.original_query + item.id).encode()).hexdigest()[:16]}"
        
        try:
            # Check cache first
            async with valkey_connection_manager.get_client() as client:
                cached_score = await client.get(cache_key)
                if cached_score:
                    return float(cached_score)
            
            # Use Ollama to calculate semantic similarity
            similarity_task = OllamaTask(
                task_type=OllamaTaskType.SEMANTIC_UNDERSTANDING,
                input_data=f"""
                Calculate semantic similarity between the search query and content:
                
                Search Query: {query.original_query}
                Content Title: {item.title}
                Content Preview: {item.content[:300]}...
                
                Rate the semantic similarity on a scale of 0.0 to 1.0:
                - 1.0: Perfect match, content directly answers the query
                - 0.8: High relevance, content strongly related
                - 0.6: Good relevance, content somewhat related
                - 0.4: Low relevance, content tangentially related
                - 0.2: Poor relevance, content barely related
                - 0.0: No relevance, content unrelated
                
                Respond with just the numerical score (e.g., 0.75):
                """,
                context={
                    "task": "semantic_similarity",
                    "content_type": item.content_type.value
                },
                temperature=0.2,
                max_tokens=50
            )
            
            ai_result = await ollama_everything.process_task(similarity_task)
            
            if ai_result.error:
                return 0.5  # Neutral score on error
            
            # Extract numerical score
            result_str = str(ai_result.result)
            score_match = re.search(r'(\d+\.?\d*)', result_str)
            
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                
                # Cache the score
                async with valkey_connection_manager.get_client() as client:
                    await client.setex(cache_key, 3600, str(score))  # Cache for 1 hour
                
                return score
            
            return 0.5  # Default score if parsing fails
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.5  # Neutral score on error
    
    def _calculate_recency_score(self, item: ContentIndexItem) -> float:
        """Calculate recency score based on content age."""
        if not item.updated_at:
            return 0.5
        
        now = datetime.utcnow()
        age_days = (now - item.updated_at).total_seconds() / 86400
        
        # Exponential decay with 30-day half-life
        return math.exp(-age_days / 30.0)
    
    def _combine_relevance_scores(
        self,
        keyword_score: float,
        semantic_score: float,
        popularity_score: float,
        recency_score: float,
        query: SearchQuery
    ) -> float:
        """Combine different relevance scores into final score."""
        
        # Base weights
        weights = {
            "keyword": 0.4,
            "semantic": 0.4,
            "popularity": 0.1,
            "recency": 0.1
        }
        
        # Adjust weights based on search type
        if query.search_type == SearchType.SEMANTIC:
            weights = {"keyword": 0.2, "semantic": 0.6, "popularity": 0.1, "recency": 0.1}
        elif query.search_type == SearchType.KEYWORD:
            weights = {"keyword": 0.8, "semantic": 0.1, "popularity": 0.05, "recency": 0.05}
        
        # Apply user preferences for boosting
        if query.boost_recent:
            weights["recency"] *= 1.5
        if query.boost_popular:
            weights["popularity"] *= 1.5
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        final_score = (
            keyword_score * weights["keyword"] +
            semantic_score * weights["semantic"] +
            popularity_score * weights["popularity"] +
            recency_score * weights["recency"]
        )
        
        return max(0.0, min(1.0, final_score))
    
    def _passes_filters(self, item: ContentIndexItem, filters: Dict[str, Any]) -> bool:
        """Check if item passes all filters."""
        
        for filter_key, filter_value in filters.items():
            if filter_key == "date_range":
                start_date = filter_value.get("start")
                end_date = filter_value.get("end")
                
                if start_date and item.created_at < start_date:
                    return False
                if end_date and item.created_at > end_date:
                    return False
            
            elif filter_key == "min_length":
                if item.word_count < filter_value:
                    return False
            
            elif filter_key == "max_length":
                if item.word_count > filter_value:
                    return False
            
            elif filter_key in item.metadata:
                if item.metadata[filter_key] != filter_value:
                    return False
        
        return True
    
    def _find_matched_terms(self, search_terms: List[str], item: ContentIndexItem) -> List[str]:
        """Find which search terms matched in the content."""
        matched = []
        searchable_text = (item.title + " " + item.content).lower()
        
        for term in search_terms:
            if term in searchable_text:
                matched.append(term)
        
        return matched
    
    async def _rank_results_with_ai(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Re-rank results using AI intelligence."""
        
        if not results or len(results) <= 1:
            return results
        
        # Prepare results for AI ranking
        results_data = []
        for i, result in enumerate(results[:20]):  # Limit for AI processing
            results_data.append({
                "index": i,
                "title": result.title,
                "content": result.content[:200],  # First 200 chars
                "content_type": result.content_type.value,
                "relevance_score": result.relevance_score,
                "matched_terms": result.matched_terms
            })
        
        # Ask AI to re-rank results
        ranking_task = OllamaTask(
            task_type=OllamaTaskType.DECISION_ANALYSIS,
            input_data=f"""
            Re-rank these search results for optimal relevance to the user query:
            
            Query: {query.original_query}
            Search Intent: {query.context.get('search_intent', 'Unknown')}
            
            Results to rank:
            {json.dumps(results_data, indent=2)}
            
            Please provide a ranking that considers:
            1. Direct relevance to the query
            2. Quality and completeness of content
            3. User intent and context
            4. Content authority and reliability
            
            Return a JSON array with the reordered indices:
            [0, 3, 1, 7, 2, ...] (indices in new order)
            """,
            context={
                "task": "result_ranking",
                "result_count": len(results_data)
            },
            temperature=0.3,
            max_tokens=500
        )
        
        try:
            ai_result = await ollama_everything.process_task(ranking_task)
            
            if not ai_result.error:
                # Parse ranking
                new_order = self._parse_ranking_response(ai_result.result, len(results_data))
                
                if new_order:
                    # Apply new ranking
                    ranked_results = []
                    for idx in new_order:
                        if 0 <= idx < len(results):
                            ranked_results.append(results[idx])
                    
                    # Add any remaining results
                    added_indices = set(new_order)
                    for i, result in enumerate(results):
                        if i not in added_indices:
                            ranked_results.append(result)
                    
                    return ranked_results[:query.max_results]
        
        except Exception as e:
            logger.error(f"AI ranking failed: {e}")
        
        # Fallback: return original ranking
        return results[:query.max_results]
    
    def _parse_ranking_response(self, ai_result: Any, max_index: int) -> Optional[List[int]]:
        """Parse AI ranking response."""
        try:
            result_str = str(ai_result)
            
            # Look for JSON array
            array_match = re.search(r'\[([0-9,\s]+)\]', result_str)
            if array_match:
                indices_str = array_match.group(1)
                indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
                
                # Validate indices
                valid_indices = [i for i in indices if 0 <= i < max_index]
                return valid_indices if valid_indices else None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse ranking response: {e}")
            return None
    
    async def _generate_suggestions(self, query: SearchQuery) -> List[str]:
        """Generate search suggestions using AI."""
        
        suggestion_task = OllamaTask(
            task_type=OllamaTaskType.TEXT_GENERATION,
            input_data=f"""
            Generate helpful search suggestions based on this query:
            
            Original Query: {query.original_query}
            Search Context: {json.dumps(query.context, indent=2)}
            
            Provide 5 improved search suggestions that might help the user find better results:
            1. More specific variations
            2. Broader search terms
            3. Related concepts
            4. Different phrasings
            5. Alternative approaches
            
            Format as a simple list:
            - suggestion 1
            - suggestion 2
            - suggestion 3
            - suggestion 4
            - suggestion 5
            """,
            context={"task": "search_suggestions"},
            temperature=0.5,
            max_tokens=300
        )
        
        try:
            ai_result = await ollama_everything.process_task(suggestion_task)
            
            if not ai_result.error:
                return self._parse_suggestions(ai_result.result)
        
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
        
        return []
    
    def _parse_suggestions(self, ai_result: Any) -> List[str]:
        """Parse AI-generated suggestions."""
        suggestions = []
        result_str = str(ai_result)
        
        # Look for bulleted lists
        bullet_matches = re.findall(r'[-*•]\s*(.+)', result_str)
        suggestions.extend([s.strip() for s in bullet_matches if len(s.strip()) > 5])
        
        # Look for numbered lists
        number_matches = re.findall(r'\d+\.\s*(.+)', result_str)
        suggestions.extend([s.strip() for s in number_matches if len(s.strip()) > 5])
        
        # Deduplicate and limit
        return list(set(suggestions))[:5]
    
    async def _generate_related_queries(self, query: SearchQuery) -> List[str]:
        """Generate related queries using AI."""
        
        related_task = OllamaTask(
            task_type=OllamaTaskType.TEXT_GENERATION,
            input_data=f"""
            Generate related search queries for:
            
            Original Query: {query.original_query}
            
            Provide 3-4 related queries that explore:
            1. Related concepts
            2. Broader categories
            3. Specific sub-topics
            4. Practical applications
            
            Format as simple phrases separated by newlines.
            """,
            context={"task": "related_queries"},
            temperature=0.6,
            max_tokens=200
        )
        
        try:
            ai_result = await ollama_everything.process_task(related_task)
            
            if not ai_result.error:
                result_str = str(ai_result.result)
                queries = [line.strip() for line in result_str.split('\n') if len(line.strip()) > 5]
                return queries[:4]
        
        except Exception as e:
            logger.error(f"Related query generation failed: {e}")
        
        return []
    
    async def _learn_from_search(self, query: SearchQuery, response: SearchResponse, user_id: str = None):
        """Learn from search patterns to improve future searches."""
        
        # Record search in history
        search_record = {
            "query": query.original_query,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results_count": len(response.results),
            "search_time": response.search_time,
            "search_type": query.search_type.value
        }
        
        self.query_history.append(search_record)
        
        # Update user preferences if user identified
        if user_id and response.results:
            user_prefs = self.user_preferences[user_id]
            
            # Track preferred content types
            content_types = [r.content_type.value for r in response.results]
            user_prefs["preferred_content_types"] = user_prefs.get("preferred_content_types", Counter())
            user_prefs["preferred_content_types"].update(content_types)
            
            # Track search patterns
            user_prefs["search_patterns"] = user_prefs.get("search_patterns", [])
            user_prefs["search_patterns"].append(query.original_query.lower())
            user_prefs["search_patterns"] = user_prefs["search_patterns"][-50:]  # Keep last 50
    
    def _generate_cache_key(self, query: SearchQuery, user_id: str = None) -> str:
        """Generate cache key for search query."""
        cache_data = {
            "query": query.original_query,
            "search_type": query.search_type.value,
            "content_types": [ct.value for ct in query.content_types],
            "filters": query.filters,
            "user_id": user_id
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def _check_search_cache(self, cache_key: str) -> Optional[SearchResponse]:
        """Check if search result is cached."""
        try:
            async with valkey_connection_manager.get_client() as client:
                cached_data = await client.get(f"ollama_search:{cache_key}")
                
                if cached_data:
                    data = json.loads(cached_data)
                    # Reconstruct SearchResponse (simplified)
                    return SearchResponse(
                        query=SearchQuery(original_query=data["query"]["original_query"]),
                        results=[SearchResult(**r) for r in data["results"]],
                        total_found=data["total_found"],
                        search_time=data["search_time"],
                        suggestions=data.get("suggestions", []),
                        related_queries=data.get("related_queries", [])
                    )
        
        except Exception as e:
            logger.error(f"Cache check failed: {e}")
        
        return None
    
    async def _cache_search_response(self, cache_key: str, response: SearchResponse):
        """Cache search response."""
        try:
            cache_data = {
                "query": {"original_query": response.query.original_query},
                "results": [r.__dict__ for r in response.results],
                "total_found": response.total_found,
                "search_time": response.search_time,
                "suggestions": response.suggestions,
                "related_queries": response.related_queries,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    f"ollama_search:{cache_key}",
                    self.cache_ttl,
                    json.dumps(cache_data, default=str)
                )
        
        except Exception as e:
            logger.error(f"Cache store failed: {e}")
    
    def _update_search_stats(self, query: SearchQuery, search_time: float):
        """Update search statistics."""
        self.search_stats["total_searches"] += 1
        
        # Update average search time
        n = self.search_stats["total_searches"]
        old_avg = self.search_stats["avg_search_time"]
        self.search_stats["avg_search_time"] = ((n - 1) * old_avg + search_time) / n
        
        # Track popular queries
        self.search_stats["popular_queries"][query.original_query.lower()] += 1
    
    # Content management methods
    
    async def index_content(
        self,
        id: str,
        title: str,
        content: str,
        content_type: ContentType,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Index new content for searching."""
        try:
            # Create content item
            item = ContentIndexItem(
                id=id,
                title=title,
                content=content,
                content_type=content_type,
                metadata=metadata or {}
            )
            
            # Use AI to extract concepts
            await self._extract_concepts_with_ai(item)
            
            # Calculate quality score
            item.quality_score = await self._calculate_quality_score(item)
            
            # Add to index
            self.content_index[id] = item
            
            # Update inverted index
            self._update_inverted_index(item)
            
            logger.info(f"Indexed content: {id} ({content_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index content {id}: {e}")
            return False
    
    async def _extract_concepts_with_ai(self, item: ContentIndexItem):
        """Extract key concepts from content using AI."""
        
        concept_task = OllamaTask(
            task_type=OllamaTaskType.TEXT_ANALYSIS,
            input_data=f"""
            Extract key concepts from this content:
            
            Title: {item.title}
            Content: {item.content[:1000]}...
            Content Type: {item.content_type.value}
            
            Identify:
            1. Main topics and themes
            2. Technical terms and concepts
            3. Key entities and proper nouns
            4. Subject matter categories
            
            Return a JSON list of concepts:
            ["concept1", "concept2", "concept3", ...]
            """,
            context={"task": "concept_extraction"},
            temperature=0.3,
            max_tokens=300
        )
        
        try:
            ai_result = await ollama_everything.process_task(concept_task)
            
            if not ai_result.error:
                concepts = self._parse_concepts(ai_result.result)
                item.concepts = concepts[:20]  # Limit to 20 concepts
        
        except Exception as e:
            logger.error(f"Concept extraction failed for {item.id}: {e}")
    
    def _parse_concepts(self, ai_result: Any) -> List[str]:
        """Parse concepts from AI response."""
        concepts = []
        result_str = str(ai_result)
        
        # Look for JSON array
        array_match = re.search(r'\[([^\]]+)\]', result_str)
        if array_match:
            try:
                concepts_data = json.loads(array_match.group(0))
                if isinstance(concepts_data, list):
                    concepts = [str(c).strip() for c in concepts_data if c]
            except:
                pass
        
        # Fallback: extract quoted terms
        if not concepts:
            quoted_terms = re.findall(r'"([^"]+)"', result_str)
            concepts.extend([t for t in quoted_terms if len(t) > 2])
        
        return concepts
    
    async def _calculate_quality_score(self, item: ContentIndexItem) -> float:
        """Calculate content quality score using AI."""
        
        quality_task = OllamaTask(
            task_type=OllamaTaskType.TEXT_ANALYSIS,
            input_data=f"""
            Assess the quality of this content on a scale of 0.0 to 1.0:
            
            Title: {item.title}
            Content: {item.content[:500]}...
            Word Count: {item.word_count}
            
            Consider:
            1. Clarity and coherence
            2. Completeness and depth
            3. Accuracy and reliability
            4. Usefulness and relevance
            5. Writing quality
            
            Respond with just a numerical score (e.g., 0.78):
            """,
            context={"task": "quality_assessment"},
            temperature=0.2,
            max_tokens=50
        )
        
        try:
            ai_result = await ollama_everything.process_task(quality_task)
            
            if not ai_result.error:
                result_str = str(ai_result.result)
                score_match = re.search(r'(\d+\.?\d*)', result_str)
                
                if score_match:
                    score = float(score_match.group(1))
                    return max(0.0, min(1.0, score))
        
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
        
        return 0.7  # Default quality score
    
    def _update_inverted_index(self, item: ContentIndexItem):
        """Update inverted index with new content."""
        
        # Index keywords
        for keyword in item.keywords:
            self.inverted_index[keyword].add(item.id)
        
        # Index concepts
        for concept in item.concepts:
            self.inverted_index[concept.lower()].add(item.id)
        
        # Index title words
        title_words = re.findall(r'\b\w+\b', item.title.lower())
        for word in title_words:
            if len(word) > 2:
                self.inverted_index[word].add(item.id)
    
    async def remove_content(self, id: str) -> bool:
        """Remove content from search index."""
        if id in self.content_index:
            item = self.content_index[id]
            
            # Remove from inverted index
            for keyword in item.keywords + item.concepts:
                if keyword.lower() in self.inverted_index:
                    self.inverted_index[keyword.lower()].discard(id)
            
            # Remove from main index
            del self.content_index[id]
            
            logger.info(f"Removed content from index: {id}")
            return True
        
        return False
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics."""
        
        total_searches = self.search_stats["total_searches"]
        
        analytics = {
            "total_searches": total_searches,
            "cache_hit_rate": self.search_stats["cache_hits"] / max(total_searches, 1),
            "avg_search_time": self.search_stats["avg_search_time"],
            "indexed_content_count": len(self.content_index),
            "content_type_distribution": Counter(item.content_type.value for item in self.content_index.values()),
            "popular_queries": dict(self.search_stats["popular_queries"].most_common(10)),
            "recent_query_count": len(self.query_history),
            "unique_users": len(self.user_preferences),
            "avg_results_per_search": sum(len(r["results"]) for r in self.query_history) / max(len(self.query_history), 1) if self.query_history else 0
        }
        
        return analytics
    
    async def shutdown(self):
        """Shutdown the search system."""
        try:
            # Save search analytics
            await self._save_search_state()
            logger.info("OllamaSearch shutdown completed")
        
        except Exception as e:
            logger.error(f"Error during search shutdown: {e}")
    
    async def _save_search_state(self):
        """Save search state to persistent storage."""
        try:
            state_data = {
                "search_stats": dict(self.search_stats),
                "user_preferences": dict(self.user_preferences),
                "query_history": list(self.query_history)[-1000:],  # Save last 1000 queries
                "last_updated": datetime.utcnow().isoformat()
            }
            
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    "ollama_search_state",
                    86400 * 7,  # 7 days
                    json.dumps(state_data, default=str)
                )
        
        except Exception as e:
            logger.warning(f"Failed to save search state: {e}")


# Global search instance
ollama_search = OllamaSearch()


# Convenience functions
async def search_with_ollama(
    query: Union[str, SearchQuery],
    user_id: str = None,
    content_types: List[ContentType] = None,
    max_results: int = 20
) -> SearchResponse:
    """Perform AI-powered search."""
    
    if isinstance(query, str):
        search_query = SearchQuery(
            original_query=query,
            content_types=content_types or [],
            max_results=max_results
        )
    else:
        search_query = query
    
    return await ollama_search.search(search_query, user_id)


async def index_content_for_search(
    id: str,
    title: str,
    content: str,
    content_type: ContentType,
    metadata: Dict[str, Any] = None
) -> bool:
    """Index content for AI search."""
    return await ollama_search.index_content(id, title, content, content_type, metadata)


async def semantic_search(query: str, user_id: str = None) -> SearchResponse:
    """Perform semantic search using AI."""
    search_query = SearchQuery(
        original_query=query,
        search_type=SearchType.SEMANTIC
    )
    return await ollama_search.search(search_query, user_id)


async def get_search_suggestions(query: str) -> List[str]:
    """Get AI-powered search suggestions."""
    search_query = SearchQuery(original_query=query)
    enhanced_query = await ollama_search._enhance_query_with_ai(search_query)
    return await ollama_search._generate_suggestions(enhanced_query)


# Initialize and shutdown functions
async def initialize_ollama_search():
    """Initialize the Ollama search system."""
    try:
        # Load existing search state
        async with valkey_connection_manager.get_client() as client:
            state_data = await client.get("ollama_search_state")
            
            if state_data:
                data = json.loads(state_data)
                ollama_search.search_stats.update(data.get("search_stats", {}))
                ollama_search.user_preferences.update(data.get("user_preferences", {}))
                
                # Restore query history
                query_history = data.get("query_history", [])
                ollama_search.query_history.extend(query_history)
                
                logger.info(f"Loaded search state with {len(query_history)} query records")
        
        # Test the search system
        test_result = await ollama_search.search("test query")
        if not test_result:
            raise Exception("Search system test failed")
        
        logger.info("OllamaSearch initialized and tested successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OllamaSearch: {e}")
        raise


async def shutdown_ollama_search():
    """Shutdown the Ollama search system."""
    try:
        await ollama_search.shutdown()
        logger.info("OllamaSearch shutdown completed")
        
    except Exception as e:
        logger.error(f"Error shutting down OllamaSearch: {e}")