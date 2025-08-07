"""
Advanced Semantic Caching System for Ollama Responses.
Implements similarity-based caching, intelligent eviction, and cost optimization
to maximize performance while minimizing resource usage for educational AI applications.
"""

import asyncio
import json
import logging
import hashlib
import time
import pickle
import gzip
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, NamedTuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import threading
from pathlib import Path

# For semantic similarity
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from .config import settings
from .valkey_integration import valkey_connection_manager

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different content types."""
    EXACT_MATCH = "exact_match"              # Perfect string match
    FUZZY_MATCH = "fuzzy_match"              # Approximate string matching
    SEMANTIC_SIMILARITY = "semantic_similarity"  # Vector-based similarity
    HYBRID = "hybrid"                        # Combination of strategies


class ContentType(Enum):
    """Types of content for specialized caching."""
    GENERAL_TEXT = "general_text"
    CODE_GENERATION = "code_generation"
    MATHEMATICAL_PROOF = "mathematical_proof"
    JFLAP_EXPLANATION = "jflap_explanation"
    AUTOMATA_THEORY = "automata_theory"
    EDUCATIONAL_CONTENT = "educational_content"


@dataclass
class CacheEntry:
    """Cached response entry with metadata."""
    key: str
    prompt_hash: str
    model_name: str
    content_type: ContentType
    prompt: str
    response: str
    token_count: int
    compute_cost: float
    similarity_vector: Optional[List[float]] = None
    
    # Cache metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    hit_count: int = 0
    
    # Quality metrics
    confidence_score: float = 1.0
    user_feedback: Optional[float] = None  # User rating 0-5
    effectiveness_score: float = 1.0  # How effective this cache has been
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def record_hit(self):
        """Record a cache hit."""
        self.hit_count += 1
        self.update_access()
        # Update effectiveness based on hit rate
        self.effectiveness_score = min(2.0, self.hit_count / max(1, self.access_count))
    
    @property
    def age_hours(self) -> float:
        """Get age of cache entry in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    @property
    def staleness_factor(self) -> float:
        """Calculate staleness factor (0-1, higher = more stale)."""
        # Content gets stale over time, but educational content ages slower
        if self.content_type == ContentType.EDUCATIONAL_CONTENT:
            half_life_hours = 168  # 7 days
        elif self.content_type == ContentType.MATHEMATICAL_PROOF:
            half_life_hours = 72   # 3 days
        elif self.content_type == ContentType.CODE_GENERATION:
            half_life_hours = 24   # 1 day
        else:
            half_life_hours = 48   # 2 days
        
        return min(1.0, self.age_hours / half_life_hours)
    
    @property
    def cache_score(self) -> float:
        """Calculate overall cache score for eviction decisions."""
        # Higher score = more valuable to keep
        base_score = self.confidence_score * self.effectiveness_score
        
        # Bonus for recent access
        recency_bonus = max(0, 1.0 - (datetime.now() - self.last_accessed).total_seconds() / 86400)
        
        # Penalty for staleness
        staleness_penalty = self.staleness_factor * 0.5
        
        # Bonus for high access count
        popularity_bonus = min(0.5, self.access_count / 100)
        
        # Bonus for user feedback
        feedback_bonus = (self.user_feedback or 0) / 10
        
        return max(0.1, base_score + recency_bonus + popularity_bonus + feedback_bonus - staleness_penalty)


class SemanticSimilarityEngine:
    """Engine for calculating semantic similarity between prompts."""
    
    def __init__(self):
        self.vectorizers = {}
        self.svd_models = {}
        self.vector_cache = {}  # Cache for prompt vectors
        
        # Initialize TF-IDF vectorizers for different content types
        self.initialize_vectorizers()
    
    def initialize_vectorizers(self):
        """Initialize TF-IDF vectorizers for different content types."""
        base_config = {
            'max_features': 5000,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8
        }
        
        # General text vectorizer
        self.vectorizers[ContentType.GENERAL_TEXT] = TfidfVectorizer(**base_config)
        
        # Code generation (preserve technical terms)
        code_config = base_config.copy()
        code_config.update({
            'stop_words': None,  # Don't remove code keywords
            'token_pattern': r'\b\w+\b|[(){}[\];,.]'  # Include punctuation
        })
        self.vectorizers[ContentType.CODE_GENERATION] = TfidfVectorizer(**code_config)
        
        # Mathematical proofs (preserve mathematical notation)
        math_config = base_config.copy()
        math_config.update({
            'token_pattern': r'\b\w+\b|[=<>≤≥∈∀∃∧∨¬→↔]',  # Include math symbols
            'max_features': 3000
        })
        self.vectorizers[ContentType.MATHEMATICAL_PROOF] = TfidfVectorizer(**math_config)
        
        # Educational content (focus on concepts)
        edu_config = base_config.copy()
        edu_config.update({
            'ngram_range': (1, 3),  # Longer phrases for concepts
            'max_features': 7000
        })
        self.vectorizers[ContentType.EDUCATIONAL_CONTENT] = TfidfVectorizer(**edu_config)
    
    def preprocess_prompt(self, prompt: str, content_type: ContentType) -> str:
        """Preprocess prompt for better similarity matching."""
        # Basic cleanup
        processed = prompt.lower().strip()
        
        # Content-type specific preprocessing
        if content_type == ContentType.CODE_GENERATION:
            # Normalize code-like elements
            processed = re.sub(r'\s+', ' ', processed)  # Normalize whitespace
            processed = re.sub(r'["\']([^"\']*)["\']', r'STRING', processed)  # Normalize strings
            processed = re.sub(r'\b\d+\b', 'NUMBER', processed)  # Normalize numbers
        
        elif content_type == ContentType.MATHEMATICAL_PROOF:
            # Normalize mathematical expressions
            processed = re.sub(r'\b[a-z]\b', 'VAR', processed)  # Variables
            processed = re.sub(r'\b\d+\b', 'NUM', processed)  # Numbers
        
        elif content_type == ContentType.JFLAP_EXPLANATION:
            # Focus on automata concepts
            automata_terms = {
                'dfa': 'deterministic_finite_automaton',
                'nfa': 'nondeterministic_finite_automaton',
                'tm': 'turing_machine',
                'pda': 'pushdown_automaton'
            }
            for abbrev, full in automata_terms.items():
                processed = processed.replace(abbrev, full)
        
        return processed
    
    def get_prompt_vector(self, prompt: str, content_type: ContentType) -> np.ndarray:
        """Get vector representation of a prompt."""
        # Check cache first
        cache_key = hashlib.md5(f"{prompt}:{content_type.value}".encode()).hexdigest()
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        # Preprocess prompt
        processed_prompt = self.preprocess_prompt(prompt, content_type)
        
        # Get appropriate vectorizer
        vectorizer = self.vectorizers.get(content_type, self.vectorizers[ContentType.GENERAL_TEXT])
        
        try:
            # Transform prompt to vector
            vector = vectorizer.transform([processed_prompt]).toarray()[0]
            
            # Apply dimensionality reduction if available
            if content_type in self.svd_models:
                vector = self.svd_models[content_type].transform([vector])[0]
            
            # Cache the result
            self.vector_cache[cache_key] = vector
            
            return vector
            
        except Exception as e:
            logger.warning(f"Failed to vectorize prompt: {e}")
            # Return zero vector as fallback
            return np.zeros(100)
    
    def calculate_similarity(
        self,
        prompt1: str,
        prompt2: str,
        content_type: ContentType,
        strategy: CacheStrategy = CacheStrategy.HYBRID
    ) -> float:
        """Calculate similarity between two prompts."""
        if strategy == CacheStrategy.EXACT_MATCH:
            return 1.0 if prompt1.strip() == prompt2.strip() else 0.0
        
        elif strategy == CacheStrategy.FUZZY_MATCH:
            return SequenceMatcher(None, prompt1, prompt2).ratio()
        
        elif strategy == CacheStrategy.SEMANTIC_SIMILARITY:
            vector1 = self.get_prompt_vector(prompt1, content_type)
            vector2 = self.get_prompt_vector(prompt2, content_type)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        
        elif strategy == CacheStrategy.HYBRID:
            # Combine multiple strategies
            exact_score = 1.0 if prompt1.strip() == prompt2.strip() else 0.0
            fuzzy_score = SequenceMatcher(None, prompt1, prompt2).ratio()
            
            # If exact match, return immediately
            if exact_score == 1.0:
                return 1.0
            
            # If very similar strings, return fuzzy score
            if fuzzy_score > 0.9:
                return fuzzy_score
            
            # Otherwise, use semantic similarity
            try:
                semantic_score = self.calculate_similarity(
                    prompt1, prompt2, content_type, CacheStrategy.SEMANTIC_SIMILARITY
                )
                
                # Weight the scores
                return 0.3 * fuzzy_score + 0.7 * semantic_score
                
            except Exception as e:
                logger.warning(f"Semantic similarity failed, using fuzzy: {e}")
                return fuzzy_score
    
    def train_vectorizers(self, prompts_by_type: Dict[ContentType, List[str]]):
        """Train vectorizers on existing prompts."""
        for content_type, prompts in prompts_by_type.items():
            if len(prompts) < 10:  # Need minimum data
                continue
            
            try:
                # Preprocess prompts
                processed_prompts = [
                    self.preprocess_prompt(prompt, content_type)
                    for prompt in prompts
                ]
                
                # Fit vectorizer
                vectorizer = self.vectorizers[content_type]
                vectorizer.fit(processed_prompts)
                
                # Train SVD for dimensionality reduction
                if len(prompts) > 50:
                    vectors = vectorizer.transform(processed_prompts)
                    svd = TruncatedSVD(n_components=min(100, len(prompts) // 2))
                    svd.fit(vectors)
                    self.svd_models[content_type] = svd
                
                logger.info(f"Trained vectorizer for {content_type.value} with {len(prompts)} prompts")
                
            except Exception as e:
                logger.error(f"Failed to train vectorizer for {content_type.value}: {e}")


class SemanticCache:
    """Advanced semantic cache for Ollama responses."""
    
    def __init__(
        self,
        max_size: int = 10000,
        similarity_threshold: float = 0.85,
        max_memory_mb: int = 500
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.max_memory_mb = max_memory_mb
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.prompt_index: Dict[str, List[str]] = defaultdict(list)  # For faster lookup
        self.similarity_engine = SemanticSimilarityEngine()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'bytes_saved': 0,
            'compute_cost_saved': 0.0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.training_task = None
        
        # Configuration
        self.cache_strategies = {
            ContentType.GENERAL_TEXT: CacheStrategy.HYBRID,
            ContentType.CODE_GENERATION: CacheStrategy.SEMANTIC_SIMILARITY,
            ContentType.MATHEMATICAL_PROOF: CacheStrategy.SEMANTIC_SIMILARITY,
            ContentType.JFLAP_EXPLANATION: CacheStrategy.HYBRID,
            ContentType.AUTOMATA_THEORY: CacheStrategy.HYBRID,
            ContentType.EDUCATIONAL_CONTENT: CacheStrategy.HYBRID
        }
        
        # Similarity thresholds by content type
        self.similarity_thresholds = {
            ContentType.GENERAL_TEXT: 0.80,
            ContentType.CODE_GENERATION: 0.90,  # Code needs higher similarity
            ContentType.MATHEMATICAL_PROOF: 0.85,
            ContentType.JFLAP_EXPLANATION: 0.82,
            ContentType.AUTOMATA_THEORY: 0.83,
            ContentType.EDUCATIONAL_CONTENT: 0.78  # Educational content can be more flexible
        }
        
        logger.info("Semantic cache initialized with advanced similarity matching")
    
    async def initialize(self):
        """Initialize the semantic cache."""
        try:
            # Load cache from persistent storage
            await self._load_cache_from_storage()
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.training_task = asyncio.create_task(self._training_loop())
            
            logger.info("Semantic cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the semantic cache."""
        try:
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.training_task:
                self.training_task.cancel()
            
            # Wait for tasks to complete
            for task in [self.cleanup_task, self.training_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Save cache to persistent storage
            await self._save_cache_to_storage()
            
            logger.info("Semantic cache shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during semantic cache shutdown: {e}")
    
    def _classify_content_type(self, prompt: str, model_name: str) -> ContentType:
        """Classify prompt content type for optimal caching strategy."""
        prompt_lower = prompt.lower()
        
        # Code generation indicators
        if any(keyword in prompt_lower for keyword in [
            'code', 'function', 'class', 'implement', 'programming',
            'python', 'javascript', 'java', 'algorithm', 'debug'
        ]):
            return ContentType.CODE_GENERATION
        
        # Mathematical proof indicators
        elif any(keyword in prompt_lower for keyword in [
            'prove', 'theorem', 'lemma', 'proof', 'mathematical',
            'induction', 'contradiction', 'qed', 'therefore'
        ]):
            return ContentType.MATHEMATICAL_PROOF
        
        # JFLAP/Automata indicators
        elif any(keyword in prompt_lower for keyword in [
            'jflap', 'automaton', 'dfa', 'nfa', 'turing machine',
            'finite automaton', 'state', 'transition', 'acceptance'
        ]):
            return ContentType.JFLAP_EXPLANATION
        
        # Automata theory indicators
        elif any(keyword in prompt_lower for keyword in [
            'automata theory', 'formal language', 'grammar',
            'pumping lemma', 'closure', 'decidable', 'complexity'
        ]):
            return ContentType.AUTOMATA_THEORY
        
        # Educational content indicators
        elif any(keyword in prompt_lower for keyword in [
            'explain', 'what is', 'how does', 'difference between',
            'example', 'tutorial', 'learn', 'understand', 'concept'
        ]):
            return ContentType.EDUCATIONAL_CONTENT
        
        # Default
        return ContentType.GENERAL_TEXT
    
    def _generate_cache_key(
        self,
        prompt: str,
        model_name: str,
        content_type: ContentType
    ) -> str:
        """Generate cache key with content-aware hashing."""
        # Create a normalized representation
        normalized_prompt = self.similarity_engine.preprocess_prompt(prompt, content_type)
        
        # Include model and content type in key
        key_components = [normalized_prompt, model_name, content_type.value]
        combined = "|".join(key_components)
        
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def get(
        self,
        prompt: str,
        model_name: str,
        max_age_hours: float = 168,  # 7 days default
        min_confidence: float = 0.8
    ) -> Optional[Tuple[str, CacheEntry]]:
        """Get cached response with semantic similarity matching."""
        self.stats['total_requests'] += 1
        
        try:
            # Classify content type
            content_type = self._classify_content_type(prompt, model_name)
            
            # Get cache strategy and similarity threshold
            strategy = self.cache_strategies.get(content_type, CacheStrategy.HYBRID)
            similarity_threshold = self.similarity_thresholds.get(content_type, self.similarity_threshold)
            
            # First try exact key lookup
            exact_key = self._generate_cache_key(prompt, model_name, content_type)
            if exact_key in self.cache:
                entry = self.cache[exact_key]
                if (entry.age_hours <= max_age_hours and 
                    entry.confidence_score >= min_confidence):
                    entry.record_hit()
                    self.stats['hits'] += 1
                    self.stats['bytes_saved'] += len(entry.response.encode())
                    self.stats['compute_cost_saved'] += entry.compute_cost
                    
                    # Move to end (LRU)
                    self.cache.move_to_end(exact_key)
                    
                    logger.debug(f"Cache hit (exact): {exact_key[:8]}...")
                    return entry.response, entry
            
            # Try semantic similarity search
            best_match = None
            best_similarity = 0.0
            best_key = None
            
            # Search through relevant entries
            candidate_keys = self._get_candidate_keys(model_name, content_type)
            
            for key in candidate_keys:
                if key not in self.cache:
                    continue
                
                entry = self.cache[key]
                
                # Skip if too old or low confidence
                if (entry.age_hours > max_age_hours or 
                    entry.confidence_score < min_confidence):
                    continue
                
                # Calculate similarity
                similarity = self.similarity_engine.calculate_similarity(
                    prompt, entry.prompt, content_type, strategy
                )
                
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match = entry
                    best_key = key
            
            if best_match:
                best_match.record_hit()
                self.stats['hits'] += 1
                self.stats['bytes_saved'] += len(best_match.response.encode())
                self.stats['compute_cost_saved'] += best_match.compute_cost
                
                # Move to end (LRU)
                self.cache.move_to_end(best_key)
                
                logger.debug(f"Cache hit (semantic): {best_key[:8]}... similarity={best_similarity:.3f}")
                return best_match.response, best_match
            
            # No match found
            self.stats['misses'] += 1
            logger.debug("Cache miss: no suitable match found")
            return None
            
        except Exception as e:
            logger.error(f"Error in cache get: {e}")
            self.stats['misses'] += 1
            return None
    
    async def put(
        self,
        prompt: str,
        response: str,
        model_name: str,
        token_count: int,
        compute_cost: float,
        confidence_score: float = 1.0
    ) -> str:
        """Store response in cache with semantic indexing."""
        try:
            # Classify content type
            content_type = self._classify_content_type(prompt, model_name)
            
            # Generate cache key
            cache_key = self._generate_cache_key(prompt, model_name, content_type)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
                model_name=model_name,
                content_type=content_type,
                prompt=prompt,
                response=response,
                token_count=token_count,
                compute_cost=compute_cost,
                confidence_score=confidence_score
            )
            
            # Generate similarity vector for semantic search
            try:
                entry.similarity_vector = self.similarity_engine.get_prompt_vector(
                    prompt, content_type
                ).tolist()
            except Exception as e:
                logger.warning(f"Failed to generate similarity vector: {e}")
            
            # Check if we need to evict entries
            await self._ensure_capacity()
            
            # Store in cache
            self.cache[cache_key] = entry
            
            # Update prompt index for faster lookup
            index_key = f"{model_name}:{content_type.value}"
            self.prompt_index[index_key].append(cache_key)
            
            # Keep index manageable
            if len(self.prompt_index[index_key]) > 1000:
                self.prompt_index[index_key] = self.prompt_index[index_key][-1000:]
            
            logger.debug(f"Cache stored: {cache_key[:8]}... ({content_type.value})")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error in cache put: {e}")
            raise
    
    def _get_candidate_keys(self, model_name: str, content_type: ContentType) -> List[str]:
        """Get candidate keys for similarity search."""
        index_key = f"{model_name}:{content_type.value}"
        candidates = self.prompt_index.get(index_key, [])
        
        # Also include general text candidates if specific type has few entries
        if len(candidates) < 100 and content_type != ContentType.GENERAL_TEXT:
            general_key = f"{model_name}:{ContentType.GENERAL_TEXT.value}"
            candidates.extend(self.prompt_index.get(general_key, [])[-50:])  # Last 50
        
        return candidates[-500:]  # Limit search scope for performance
    
    async def _ensure_capacity(self):
        """Ensure cache doesn't exceed capacity limits."""
        # Check size limit
        while len(self.cache) >= self.max_size:
            await self._evict_entry()
        
        # Check memory limit (approximate)
        estimated_memory_mb = self._estimate_memory_usage()
        while estimated_memory_mb > self.max_memory_mb:
            await self._evict_entry()
            estimated_memory_mb = self._estimate_memory_usage()
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        if not self.cache:
            return 0.0
        
        # Sample a few entries to estimate average size
        sample_size = min(10, len(self.cache))
        sample_entries = list(self.cache.values())[:sample_size]
        
        avg_entry_size = sum(
            len(entry.prompt.encode()) + 
            len(entry.response.encode()) +
            len(entry.similarity_vector or []) * 8 +  # 8 bytes per float
            200  # Overhead
            for entry in sample_entries
        ) / sample_size
        
        total_bytes = avg_entry_size * len(self.cache)
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    async def _evict_entry(self):
        """Evict least valuable cache entry."""
        if not self.cache:
            return
        
        # Find entry with lowest cache score
        worst_key = None
        worst_score = float('inf')
        
        # Sample entries for performance (don't check all entries every time)
        sample_size = min(100, len(self.cache))
        sample_keys = list(self.cache.keys())[-sample_size:]  # Check recent entries first
        
        for key in sample_keys:
            entry = self.cache[key]
            score = entry.cache_score
            
            if score < worst_score:
                worst_score = score
                worst_key = key
        
        if worst_key:
            # Remove from cache
            evicted_entry = self.cache.pop(worst_key)
            
            # Remove from prompt index
            index_key = f"{evicted_entry.model_name}:{evicted_entry.content_type.value}"
            if index_key in self.prompt_index:
                try:
                    self.prompt_index[index_key].remove(worst_key)
                except ValueError:
                    pass  # Key not in list
            
            self.stats['evictions'] += 1
            logger.debug(f"Evicted cache entry: {worst_key[:8]}... (score={worst_score:.3f})")
    
    async def invalidate_by_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        to_remove = []
        
        for key, entry in self.cache.items():
            if (pattern in entry.prompt.lower() or 
                pattern in entry.response.lower()):
                to_remove.append(key)
        
        for key in to_remove:
            del self.cache[key]
        
        logger.info(f"Invalidated {len(to_remove)} cache entries matching pattern: {pattern}")
    
    async def update_user_feedback(self, cache_key: str, rating: float):
        """Update cache entry with user feedback."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            entry.user_feedback = max(0.0, min(5.0, rating))
            logger.debug(f"Updated feedback for {cache_key[:8]}...: {rating}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.stats['total_requests']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        # Content type distribution
        content_type_dist = defaultdict(int)
        for entry in self.cache.values():
            content_type_dist[entry.content_type.value] += 1
        
        # Age distribution
        now = datetime.now()
        age_buckets = {
            '< 1 hour': 0,
            '1-6 hours': 0,
            '6-24 hours': 0,
            '1-7 days': 0,
            '> 7 days': 0
        }
        
        for entry in self.cache.values():
            age_hours = (now - entry.created_at).total_seconds() / 3600
            if age_hours < 1:
                age_buckets['< 1 hour'] += 1
            elif age_hours < 6:
                age_buckets['1-6 hours'] += 1
            elif age_hours < 24:
                age_buckets['6-24 hours'] += 1
            elif age_hours < 168:  # 7 days
                age_buckets['1-7 days'] += 1
            else:
                age_buckets['> 7 days'] += 1
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'bytes_saved': self.stats['bytes_saved'],
            'compute_cost_saved': self.stats['compute_cost_saved'],
            'estimated_memory_mb': self._estimate_memory_usage(),
            'content_type_distribution': dict(content_type_dist),
            'age_distribution': age_buckets,
            'average_confidence': sum(e.confidence_score for e in self.cache.values()) / len(self.cache) if self.cache else 0,
            'average_effectiveness': sum(e.effectiveness_score for e in self.cache.values()) / len(self.cache) if self.cache else 0
        }
    
    async def _cleanup_loop(self):
        """Periodic cleanup of stale cache entries."""
        while True:
            try:
                await asyncio.sleep(1800)  # Clean up every 30 minutes
                
                now = datetime.now()
                stale_keys = []
                
                for key, entry in self.cache.items():
                    # Remove very old entries or entries with very low scores
                    if (entry.age_hours > 240 or  # 10 days
                        (entry.age_hours > 24 and entry.cache_score < 0.1)):
                        stale_keys.append(key)
                
                # Remove stale entries
                for key in stale_keys:
                    if key in self.cache:
                        del self.cache[key]
                
                if stale_keys:
                    logger.info(f"Cleaned up {len(stale_keys)} stale cache entries")
                
                # Periodic save to storage
                await self._save_cache_to_storage()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _training_loop(self):
        """Periodic training of similarity models."""
        while True:
            try:
                await asyncio.sleep(7200)  # Train every 2 hours
                
                if len(self.cache) < 100:  # Need minimum data
                    continue
                
                # Collect prompts by content type
                prompts_by_type = defaultdict(list)
                for entry in self.cache.values():
                    prompts_by_type[entry.content_type].append(entry.prompt)
                
                # Train vectorizers
                self.similarity_engine.train_vectorizers(dict(prompts_by_type))
                
                # Clear vector cache to use new models
                self.similarity_engine.vector_cache.clear()
                
                logger.info("Completed periodic similarity model training")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
    
    async def _save_cache_to_storage(self):
        """Save cache to persistent storage (Valkey)."""
        try:
            if not self.cache:
                return
            
            # Sample recent entries for saving (don't save entire cache)
            recent_entries = list(self.cache.values())[-1000:]  # Last 1000 entries
            
            cache_data = {
                'entries': [
                    {
                        'key': entry.key,
                        'prompt_hash': entry.prompt_hash,
                        'model_name': entry.model_name,
                        'content_type': entry.content_type.value,
                        'prompt': entry.prompt,
                        'response': entry.response,
                        'token_count': entry.token_count,
                        'compute_cost': entry.compute_cost,
                        'created_at': entry.created_at.isoformat(),
                        'last_accessed': entry.last_accessed.isoformat(),
                        'access_count': entry.access_count,
                        'hit_count': entry.hit_count,
                        'confidence_score': entry.confidence_score,
                        'effectiveness_score': entry.effectiveness_score,
                        'user_feedback': entry.user_feedback
                    }
                    for entry in recent_entries
                ],
                'stats': self.stats,
                'saved_at': datetime.now().isoformat()
            }
            
            # Compress and save
            compressed_data = gzip.compress(
                json.dumps(cache_data, separators=(',', ':')).encode()
            )
            
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    "semantic_cache:backup",
                    86400,  # 24 hours TTL
                    compressed_data
                )
            
            logger.debug(f"Saved {len(recent_entries)} cache entries to storage")
            
        except Exception as e:
            logger.warning(f"Failed to save cache to storage: {e}")
    
    async def _load_cache_from_storage(self):
        """Load cache from persistent storage (Valkey)."""
        try:
            async with valkey_connection_manager.get_client() as client:
                compressed_data = await client.get("semantic_cache:backup")
                
                if not compressed_data:
                    return
                
                # Decompress and parse
                cache_data = json.loads(gzip.decompress(compressed_data).decode())
                
                # Restore entries
                for entry_data in cache_data.get('entries', []):
                    entry = CacheEntry(
                        key=entry_data['key'],
                        prompt_hash=entry_data['prompt_hash'],
                        model_name=entry_data['model_name'],
                        content_type=ContentType(entry_data['content_type']),
                        prompt=entry_data['prompt'],
                        response=entry_data['response'],
                        token_count=entry_data['token_count'],
                        compute_cost=entry_data['compute_cost'],
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                        access_count=entry_data['access_count'],
                        hit_count=entry_data['hit_count'],
                        confidence_score=entry_data['confidence_score'],
                        effectiveness_score=entry_data['effectiveness_score'],
                        user_feedback=entry_data.get('user_feedback')
                    )
                    
                    # Only restore recent entries
                    if entry.age_hours <= 168:  # 7 days
                        self.cache[entry.key] = entry
                        
                        # Update prompt index
                        index_key = f"{entry.model_name}:{entry.content_type.value}"
                        self.prompt_index[index_key].append(entry.key)
                
                # Restore statistics
                if 'stats' in cache_data:
                    self.stats.update(cache_data['stats'])
                
                logger.info(f"Loaded {len(self.cache)} cache entries from storage")
                
        except Exception as e:
            logger.warning(f"Failed to load cache from storage: {e}")


# Global semantic cache instance
semantic_cache = SemanticCache()


async def initialize_semantic_cache():
    """Initialize the semantic cache."""
    await semantic_cache.initialize()


async def shutdown_semantic_cache():
    """Shutdown the semantic cache."""
    await semantic_cache.shutdown()


# Convenience functions
async def get_cached_response(
    prompt: str,
    model_name: str,
    max_age_hours: float = 168
) -> Optional[str]:
    """Quick function to get cached response."""
    result = await semantic_cache.get(prompt, model_name, max_age_hours)
    return result[0] if result else None


async def cache_response(
    prompt: str,
    response: str,
    model_name: str,
    token_count: int = 0,
    compute_cost: float = 0.0
) -> str:
    """Quick function to cache a response."""
    return await semantic_cache.put(
        prompt, response, model_name, token_count, compute_cost
    )