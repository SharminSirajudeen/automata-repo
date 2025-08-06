"""
Semantic Search System with ChromaDB for the Automata Learning Platform.
Provides vector storage, semantic search, and knowledge graph capabilities.
"""
import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import logging
import numpy as np
from datetime import datetime
import asyncio

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import ollama

from .ai_config import get_ai_config
from .orchestrator import orchestrator, ExecutionMode

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Types of documents in the knowledge base."""
    PROBLEM = "problem"
    SOLUTION = "solution"
    CONCEPT = "concept"
    THEOREM = "theorem"
    PROOF = "proof"
    EXAMPLE = "example"
    LECTURE = "lecture"
    EXERCISE = "exercise"
    RESEARCH_PAPER = "research_paper"


class SearchMode(str, Enum):
    """Search modes available."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SIMILARITY = "similarity"
    CONCEPTUAL = "conceptual"


class Document(BaseModel):
    """Document structure for storage."""
    id: str
    content: str
    type: DocumentType
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def generate_id(self) -> str:
        """Generate unique ID based on content."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.type.value}_{content_hash}_{int(self.created_at.timestamp())}"


class SearchResult(BaseModel):
    """Search result structure."""
    document_id: str
    content: str
    score: float
    type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    highlight: Optional[str] = None


class KnowledgeNode(BaseModel):
    """Node in the knowledge graph."""
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeEdge(BaseModel):
    """Edge in the knowledge graph."""
    source: str
    target: str
    relationship: str
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class ChromaDBManager:
    """Manages ChromaDB collections and operations."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.config = get_ai_config()
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            self.config.vector_db.embedding_model
        )
        
        # Initialize Ollama embedding function
        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=self.config.ollama_base_url,
            model_name="nomic-embed-text"
        )
        
        # Initialize collections
        self._initialize_collections()
        
        logger.info(f"ChromaDB initialized at {persist_directory}")
    
    def _initialize_collections(self):
        """Initialize or get existing collections."""
        # Main knowledge collection
        self.knowledge_collection = self.client.get_or_create_collection(
            name=self.config.vector_db.collection_name,
            embedding_function=self.ollama_ef,
            metadata={"description": "Automata theory knowledge base"}
        )
        
        # Problem-solution collection
        self.problems_collection = self.client.get_or_create_collection(
            name="automata_problems",
            embedding_function=self.ollama_ef,
            metadata={"description": "Problems and solutions"}
        )
        
        # Concepts collection
        self.concepts_collection = self.client.get_or_create_collection(
            name="automata_concepts",
            embedding_function=self.ollama_ef,
            metadata={"description": "Theoretical concepts"}
        )
    
    def add_document(
        self,
        document: Document,
        collection_name: Optional[str] = None
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            document: Document to add
            collection_name: Specific collection to use
        
        Returns:
            Document ID
        """
        if not document.id:
            document.id = document.generate_id()
        
        # Select collection
        if collection_name:
            collection = self.client.get_collection(collection_name)
        elif document.type in [DocumentType.PROBLEM, DocumentType.SOLUTION]:
            collection = self.problems_collection
        elif document.type in [DocumentType.CONCEPT, DocumentType.THEOREM]:
            collection = self.concepts_collection
        else:
            collection = self.knowledge_collection
        
        # Prepare metadata
        metadata = {
            "type": document.type.value,
            "title": document.title or "",
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            **document.metadata
        }
        
        # Add to collection
        collection.add(
            documents=[document.content],
            metadatas=[metadata],
            ids=[document.id]
        )
        
        logger.info(f"Added document {document.id} to {collection.name}")
        return document.id
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Metadata filters
            collection_name: Specific collection to search
        
        Returns:
            List of search results
        """
        # Select collection
        collection = (
            self.client.get_collection(collection_name)
            if collection_name
            else self.knowledge_collection
        )
        
        # Perform search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        
        # Parse results
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                search_results.append(SearchResult(
                    document_id=results['ids'][0][i],
                    content=doc,
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    type=DocumentType(results['metadatas'][0][i].get('type', 'concept')),
                    metadata=results['metadatas'][0][i],
                    highlight=self._generate_highlight(doc, query)
                ))
        
        return search_results
    
    def _generate_highlight(self, content: str, query: str, max_length: int = 200) -> str:
        """Generate highlighted snippet from content."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find best matching position
        pos = content_lower.find(query_lower)
        if pos == -1:
            # No exact match, return beginning
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # Extract context around match
        start = max(0, pos - 50)
        end = min(len(content), pos + len(query) + 150)
        
        highlight = content[start:end]
        if start > 0:
            highlight = "..." + highlight
        if end < len(content):
            highlight = highlight + "..."
        
        return highlight
    
    def update_document(
        self,
        document_id: str,
        new_content: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> bool:
        """Update an existing document."""
        collection = (
            self.client.get_collection(collection_name)
            if collection_name
            else self.knowledge_collection
        )
        
        try:
            if new_content:
                collection.update(
                    ids=[document_id],
                    documents=[new_content]
                )
            
            if new_metadata:
                collection.update(
                    ids=[document_id],
                    metadatas=[new_metadata]
                )
            
            logger.info(f"Updated document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def delete_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete a document from the vector store."""
        collection = (
            self.client.get_collection(collection_name)
            if collection_name
            else self.knowledge_collection
        )
        
        try:
            collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False


class SemanticSearchEngine:
    """Advanced semantic search with multiple strategies."""
    
    def __init__(self):
        self.db_manager = ChromaDBManager()
        self.config = get_ai_config()
    
    async def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining keyword and semantic search.
        
        Args:
            query: Search query
            n_results: Number of results
            keyword_weight: Weight for keyword matching
            semantic_weight: Weight for semantic similarity
        
        Returns:
            Combined search results
        """
        # Semantic search
        semantic_results = self.db_manager.search(query, n_results * 2)
        
        # Keyword search (using ChromaDB's built-in text search)
        keyword_results = await self._keyword_search(query, n_results * 2)
        
        # Combine and re-rank
        combined_results = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        return combined_results[:n_results]
    
    async def _keyword_search(
        self,
        query: str,
        n_results: int
    ) -> List[SearchResult]:
        """Perform keyword-based search."""
        # Use ChromaDB's where clause for keyword matching
        keywords = query.lower().split()
        
        results = []
        for collection in [
            self.db_manager.knowledge_collection,
            self.db_manager.problems_collection,
            self.db_manager.concepts_collection
        ]:
            # Simple keyword matching in content
            all_docs = collection.get(limit=1000)  # Get sample of documents
            
            if all_docs['documents']:
                for i, doc in enumerate(all_docs['documents']):
                    score = sum(1 for kw in keywords if kw in doc.lower())
                    if score > 0:
                        results.append(SearchResult(
                            document_id=all_docs['ids'][i],
                            content=doc,
                            score=score / len(keywords),
                            type=DocumentType(all_docs['metadatas'][i].get('type', 'concept')),
                            metadata=all_docs['metadatas'][i]
                        ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:n_results]
    
    def _combine_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[SearchResult]:
        """Combine and re-rank search results."""
        # Create score map
        combined_scores = {}
        
        for result in semantic_results:
            combined_scores[result.document_id] = {
                'result': result,
                'score': result.score * semantic_weight
            }
        
        for result in keyword_results:
            if result.document_id in combined_scores:
                combined_scores[result.document_id]['score'] += result.score * keyword_weight
            else:
                combined_scores[result.document_id] = {
                    'result': result,
                    'score': result.score * keyword_weight
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Update scores and return
        final_results = []
        for item in sorted_results:
            result = item['result']
            result.score = item['score']
            final_results.append(result)
        
        return final_results
    
    async def similarity_search(
        self,
        document_id: str,
        n_results: int = 5,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Find similar documents to a given document.
        
        Args:
            document_id: ID of the reference document
            n_results: Number of similar documents
            threshold: Minimum similarity threshold
        
        Returns:
            Similar documents
        """
        # Get the document
        doc = self.db_manager.knowledge_collection.get(ids=[document_id])
        
        if not doc['documents']:
            return []
        
        # Search for similar documents
        results = self.db_manager.search(
            doc['documents'][0],
            n_results + 1  # +1 to exclude self
        )
        
        # Filter out self and apply threshold
        filtered_results = [
            r for r in results
            if r.document_id != document_id and r.score >= threshold
        ]
        
        return filtered_results[:n_results]
    
    async def conceptual_search(
        self,
        concept: str,
        related_concepts: List[str] = None,
        n_results: int = 10
    ) -> List[SearchResult]:
        """
        Search based on conceptual understanding.
        
        Args:
            concept: Main concept to search
            related_concepts: Related concepts to consider
            n_results: Number of results
        
        Returns:
            Conceptually related documents
        """
        # Expand query with related concepts
        expanded_query = concept
        if related_concepts:
            expanded_query += " " + " ".join(related_concepts)
        
        # Generate conceptual embedding using AI
        prompt = f"""Generate a comprehensive search query for finding documents about:
Main concept: {concept}
Related concepts: {', '.join(related_concepts) if related_concepts else 'None'}

Include key terms, synonyms, and related technical vocabulary."""
        
        response = await orchestrator.execute(
            task="query_expansion",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.7
        )
        
        expanded_query = response[0].response if isinstance(response, list) else response.response
        
        # Perform search with expanded query
        results = await self.hybrid_search(expanded_query, n_results)
        
        return results


class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from documents."""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.search_engine = SemanticSearchEngine()
    
    async def build_graph(
        self,
        documents: List[Document],
        relationship_threshold: float = 0.6
    ) -> Tuple[List[KnowledgeNode], List[KnowledgeEdge]]:
        """
        Build knowledge graph from documents.
        
        Args:
            documents: List of documents
            relationship_threshold: Minimum similarity for creating edges
        
        Returns:
            Tuple of (nodes, edges)
        """
        # Create nodes from documents
        for doc in documents:
            node = KnowledgeNode(
                id=doc.id,
                label=doc.title or f"{doc.type.value}_{doc.id[:8]}",
                type=doc.type.value,
                properties={
                    "content_preview": doc.content[:200],
                    "created_at": doc.created_at.isoformat()
                }
            )
            self.nodes[doc.id] = node
        
        # Find relationships between documents
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                similarity = await self._calculate_similarity(doc1, doc2)
                
                if similarity >= relationship_threshold:
                    relationship = await self._determine_relationship(doc1, doc2)
                    
                    edge = KnowledgeEdge(
                        source=doc1.id,
                        target=doc2.id,
                        relationship=relationship,
                        weight=similarity
                    )
                    self.edges.append(edge)
        
        return list(self.nodes.values()), self.edges
    
    async def _calculate_similarity(
        self,
        doc1: Document,
        doc2: Document
    ) -> float:
        """Calculate similarity between two documents."""
        # Use embedding similarity
        embeddings = self.search_engine.db_manager.embedding_model.encode(
            [doc1.content, doc2.content]
        )
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    async def _determine_relationship(
        self,
        doc1: Document,
        doc2: Document
    ) -> str:
        """Determine the type of relationship between documents."""
        # Rule-based relationship detection
        if doc1.type == DocumentType.PROBLEM and doc2.type == DocumentType.SOLUTION:
            return "has_solution"
        elif doc1.type == DocumentType.THEOREM and doc2.type == DocumentType.PROOF:
            return "has_proof"
        elif doc1.type == DocumentType.CONCEPT and doc2.type == DocumentType.EXAMPLE:
            return "has_example"
        elif doc1.type == doc2.type:
            return "related_to"
        else:
            return "references"
    
    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find path between two nodes in the knowledge graph.
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_depth: Maximum path depth
        
        Returns:
            Path as list of node IDs, or None if no path exists
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            return None
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(start_node, [start_node])])
        visited = {start_node}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end_node:
                return path
            
            if len(path) >= max_depth:
                continue
            
            # Find neighbors
            for edge in self.edges:
                neighbor = None
                if edge.source == current:
                    neighbor = edge.target
                elif edge.target == current:
                    neighbor = edge.source
                
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_related_concepts(
        self,
        node_id: str,
        depth: int = 2,
        min_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Get related concepts from the knowledge graph.
        
        Args:
            node_id: Starting node
            depth: How many hops to explore
            min_weight: Minimum edge weight to consider
        
        Returns:
            List of (node_id, relevance_score) tuples
        """
        if node_id not in self.nodes:
            return []
        
        related = {}
        to_explore = [(node_id, 1.0, 0)]
        explored = set()
        
        while to_explore:
            current, score, current_depth = to_explore.pop(0)
            
            if current in explored or current_depth >= depth:
                continue
            
            explored.add(current)
            
            # Find connected nodes
            for edge in self.edges:
                if edge.weight < min_weight:
                    continue
                
                neighbor = None
                if edge.source == current:
                    neighbor = edge.target
                elif edge.target == current:
                    neighbor = edge.source
                
                if neighbor and neighbor != node_id:
                    # Calculate decaying score based on depth
                    neighbor_score = score * edge.weight * (0.8 ** current_depth)
                    
                    if neighbor not in related or related[neighbor] < neighbor_score:
                        related[neighbor] = neighbor_score
                        to_explore.append((neighbor, neighbor_score, current_depth + 1))
        
        # Sort by relevance score
        return sorted(related.items(), key=lambda x: x[1], reverse=True)


class RecommendationEngine:
    """Provides content recommendations based on semantic search."""
    
    def __init__(self):
        self.search_engine = SemanticSearchEngine()
        self.graph_builder = KnowledgeGraphBuilder()
    
    async def recommend_next_topics(
        self,
        current_topic: str,
        user_history: List[str] = None,
        n_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recommend next topics to study.
        
        Args:
            current_topic: Current topic being studied
            user_history: Previous topics studied
            n_recommendations: Number of recommendations
        
        Returns:
            List of recommended topics with reasoning
        """
        # Search for related content
        related = await self.search_engine.conceptual_search(
            current_topic,
            related_concepts=user_history[-3:] if user_history else None,
            n_results=n_recommendations * 2
        )
        
        recommendations = []
        seen_topics = set(user_history) if user_history else set()
        
        for result in related:
            # Skip if already studied
            if result.metadata.get('title') in seen_topics:
                continue
            
            # Generate recommendation reasoning
            reasoning = await self._generate_reasoning(
                current_topic,
                result.metadata.get('title', result.content[:50])
            )
            
            recommendations.append({
                "document_id": result.document_id,
                "title": result.metadata.get('title', 'Untitled'),
                "type": result.type.value,
                "relevance_score": result.score,
                "reasoning": reasoning,
                "preview": result.highlight or result.content[:200]
            })
            
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    async def _generate_reasoning(
        self,
        current_topic: str,
        recommended_topic: str
    ) -> str:
        """Generate reasoning for why a topic is recommended."""
        prompt = f"""Explain in one sentence why someone studying "{current_topic}" 
should next study "{recommended_topic}" in the context of automata theory."""
        
        response = await orchestrator.execute(
            task="recommendation_reasoning",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.6,
            max_tokens=100
        )
        
        return response[0].response if isinstance(response, list) else response.response
    
    async def find_prerequisites(
        self,
        topic: str,
        max_prerequisites: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find prerequisite topics for a given topic.
        
        Args:
            topic: Target topic
            max_prerequisites: Maximum number of prerequisites
        
        Returns:
            List of prerequisite topics
        """
        prompt = f"""List the prerequisite topics needed to understand "{topic}" 
in automata theory and formal languages. Include only essential prerequisites."""
        
        response = await orchestrator.execute(
            task="find_prerequisites",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.3
        )
        
        prereq_text = response[0].response if isinstance(response, list) else response.response
        
        # Search for each prerequisite
        prerequisites = []
        for line in prereq_text.split('\n'):
            if line.strip():
                # Search for this prerequisite
                results = await self.search_engine.hybrid_search(
                    line.strip(),
                    n_results=1
                )
                
                if results:
                    prerequisites.append({
                        "topic": line.strip(),
                        "document_id": results[0].document_id,
                        "content": results[0].content[:300],
                        "importance": "essential"
                    })
        
        return prerequisites[:max_prerequisites]


# Global instances
db_manager = ChromaDBManager()
search_engine = SemanticSearchEngine()
knowledge_graph = KnowledgeGraphBuilder()
recommendation_engine = RecommendationEngine()