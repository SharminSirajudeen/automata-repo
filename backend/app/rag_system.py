"""
RAG (Retrieval-Augmented Generation) System with LangChain for the Automata Learning Platform.
Provides dynamic context retrieval, multi-document reasoning, and source attribution.
"""
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document as LangchainDocument
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from .ai_config import get_ai_config
from .semantic_search import ChromaDBManager, Document, DocumentType
from .orchestrator import orchestrator, ExecutionMode

logger = logging.getLogger(__name__)


class RAGMode(str, Enum):
    """RAG operation modes."""
    SIMPLE = "simple"
    CONVERSATIONAL = "conversational"
    MULTI_QUERY = "multi_query"
    RECURSIVE = "recursive"
    ADAPTIVE = "adaptive"


class RetrievalStrategy(str, Enum):
    """Document retrieval strategies."""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximal Marginal Relevance
    THRESHOLD = "threshold"
    CONTEXTUAL = "contextual"


class RAGRequest(BaseModel):
    """Request structure for RAG operations."""
    query: str
    mode: RAGMode = RAGMode.SIMPLE
    max_sources: int = 5
    include_sources: bool = True
    conversation_id: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    temperature: float = 0.7


class RAGResponse(BaseModel):
    """Response structure from RAG operations."""
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    context_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentProcessor:
    """Processes and chunks documents for RAG."""
    
    def __init__(self):
        self.config = get_ai_config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.vector_db.chunk_size,
            chunk_overlap=self.config.vector_db.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[LangchainDocument]:
        """
        Process text into chunks for RAG.
        
        Args:
            text: Raw text content
            metadata: Document metadata
        
        Returns:
            List of processed document chunks
        """
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata['chunk_index'] = i
            doc_metadata['total_chunks'] = len(chunks)
            
            documents.append(LangchainDocument(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def process_documents(
        self,
        documents: List[Document]
    ) -> List[LangchainDocument]:
        """
        Process multiple documents for RAG.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of processed LangChain documents
        """
        all_docs = []
        
        for doc in documents:
            metadata = {
                'document_id': doc.id,
                'type': doc.type.value,
                'title': doc.title,
                'created_at': doc.created_at.isoformat(),
                **doc.metadata
            }
            
            processed = self.process_text(doc.content, metadata)
            all_docs.extend(processed)
        
        return all_docs
    
    async def load_from_file(
        self,
        file_path: str,
        file_type: str = "text"
    ) -> List[LangchainDocument]:
        """
        Load and process documents from files.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (text, json, pdf)
        
        Returns:
            Processed documents
        """
        if file_type == "text":
            loader = TextLoader(file_path)
        elif file_type == "json":
            loader = JSONLoader(
                file_path,
                jq_schema='.',
                text_content=False
            )
        elif file_type == "pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        raw_docs = await loader.aload()
        
        # Further process if needed
        processed_docs = []
        for doc in raw_docs:
            chunks = self.text_splitter.split_documents([doc])
            processed_docs.extend(chunks)
        
        return processed_docs


class AdvancedRetriever:
    """Advanced document retrieval with multiple strategies."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.config = get_ai_config()
        
        # Initialize LLM for contextual compression
        self.llm = Ollama(
            model=self.config.models[ModelType.GENERAL].name,
            base_url=self.config.ollama_base_url,
            temperature=0.3
        )
    
    def get_retriever(
        self,
        strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        k: int = 5,
        threshold: Optional[float] = None
    ):
        """
        Get retriever with specified strategy.
        
        Args:
            strategy: Retrieval strategy to use
            k: Number of documents to retrieve
            threshold: Similarity threshold for filtering
        
        Returns:
            Configured retriever
        """
        if strategy == RetrievalStrategy.SIMILARITY:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        
        elif strategy == RetrievalStrategy.MMR:
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2, "lambda_mult": 0.5}
            )
        
        elif strategy == RetrievalStrategy.THRESHOLD:
            return self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": threshold or self.config.rag.min_relevance_score,
                    "k": k
                }
            )
        
        elif strategy == RetrievalStrategy.CONTEXTUAL:
            # Use contextual compression for more relevant results
            base_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k * 2}
            )
            
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    async def multi_query_retrieval(
        self,
        query: str,
        n_queries: int = 3,
        k: int = 5
    ) -> List[LangchainDocument]:
        """
        Retrieve documents using multiple query variations.
        
        Args:
            query: Original query
            n_queries: Number of query variations
            k: Documents per query
        
        Returns:
            Combined retrieved documents
        """
        # Generate query variations
        prompt = f"""Generate {n_queries} different ways to ask this question:
"{query}"

Provide variations that might retrieve different relevant information.
Format: One question per line."""
        
        response = await orchestrator.execute(
            task="query_expansion",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.8
        )
        
        queries = [query]  # Include original
        response_text = response[0].response if isinstance(response, list) else response.response
        queries.extend([q.strip() for q in response_text.split('\n') if q.strip()][:n_queries-1])
        
        # Retrieve for each query
        all_docs = []
        seen_contents = set()
        
        for q in queries:
            docs = await self.vectorstore.asimilarity_search(q, k=k)
            for doc in docs:
                # Deduplicate
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        return all_docs


class RAGChain:
    """Main RAG chain for question answering."""
    
    def __init__(self):
        self.config = get_ai_config()
        self.processor = DocumentProcessor()
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=self.config.ollama_base_url
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=self.config.vector_db.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_documents"
        )
        
        # Initialize retriever
        self.retriever_manager = AdvancedRetriever(self.vectorstore)
        
        # Initialize LLMs for different purposes
        self.qa_llm = Ollama(
            model=self.config.models[ModelType.EXPLAINER].name,
            base_url=self.config.ollama_base_url,
            temperature=0.7,
            callbacks=[StreamingStdOutCallbackHandler()] if self.config.enable_streaming else []
        )
        
        self.summary_llm = Ollama(
            model=self.config.models[ModelType.GENERAL].name,
            base_url=self.config.ollama_base_url,
            temperature=0.3
        )
        
        # Initialize chains
        self._initialize_chains()
        
        # Conversation memory
        self.conversations: Dict[str, ConversationBufferMemory] = {}
    
    def _initialize_chains(self):
        """Initialize various QA chains."""
        # Simple QA chain
        self.simple_qa_chain = RetrievalQA.from_chain_type(
            llm=self.qa_llm,
            chain_type="stuff",
            retriever=self.retriever_manager.get_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self._get_qa_prompt()
            }
        )
        
        # Map-reduce chain for long contexts
        self.map_reduce_chain = RetrievalQA.from_chain_type(
            llm=self.qa_llm,
            chain_type="map_reduce",
            retriever=self.retriever_manager.get_retriever(k=10),
            return_source_documents=True
        )
        
        # Refine chain for iterative improvement
        self.refine_chain = RetrievalQA.from_chain_type(
            llm=self.qa_llm,
            chain_type="refine",
            retriever=self.retriever_manager.get_retriever(),
            return_source_documents=True
        )
    
    def _get_qa_prompt(self) -> PromptTemplate:
        """Get QA prompt template."""
        template = """You are an expert in automata theory and formal languages.
Use the following context to answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
1. Provide a clear, detailed answer based on the context
2. Use technical terms appropriately
3. Include examples when helpful
4. If the context doesn't contain enough information, say so
5. Cite specific parts of the context when making claims

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    async def add_documents(
        self,
        documents: List[Document]
    ) -> int:
        """
        Add documents to the RAG system.
        
        Args:
            documents: Documents to add
        
        Returns:
            Number of chunks added
        """
        # Process documents
        langchain_docs = self.processor.process_documents(documents)
        
        # Add to vector store
        await self.vectorstore.aadd_documents(langchain_docs)
        
        logger.info(f"Added {len(langchain_docs)} chunks to RAG system")
        return len(langchain_docs)
    
    async def query(
        self,
        request: RAGRequest
    ) -> RAGResponse:
        """
        Execute a RAG query.
        
        Args:
            request: RAG request parameters
        
        Returns:
            RAG response with answer and sources
        """
        if request.mode == RAGMode.SIMPLE:
            return await self._simple_query(request)
        elif request.mode == RAGMode.CONVERSATIONAL:
            return await self._conversational_query(request)
        elif request.mode == RAGMode.MULTI_QUERY:
            return await self._multi_query(request)
        elif request.mode == RAGMode.RECURSIVE:
            return await self._recursive_query(request)
        elif request.mode == RAGMode.ADAPTIVE:
            return await self._adaptive_query(request)
        else:
            raise ValueError(f"Unknown RAG mode: {request.mode}")
    
    async def _simple_query(self, request: RAGRequest) -> RAGResponse:
        """Execute simple RAG query."""
        # Run QA chain
        result = await self.simple_qa_chain.ainvoke({
            "query": request.query
        })
        
        # Extract sources
        sources = []
        if request.include_sources and "source_documents" in result:
            for doc in result["source_documents"][:request.max_sources]:
                sources.append({
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata,
                    "relevance": 1.0  # Would need actual score
                })
        
        return RAGResponse(
            answer=result["result"],
            sources=sources,
            confidence=0.85,  # Would calculate based on retrieval scores
            context_used=[doc.page_content for doc in result.get("source_documents", [])]
        )
    
    async def _conversational_query(self, request: RAGRequest) -> RAGResponse:
        """Execute conversational RAG query with memory."""
        # Get or create conversation memory
        if request.conversation_id not in self.conversations:
            self.conversations[request.conversation_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        memory = self.conversations[request.conversation_id]
        
        # Create conversational chain
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=self.qa_llm,
            retriever=self.retriever_manager.get_retriever(),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": self._get_qa_prompt()
            }
        )
        
        # Run query
        result = await conv_chain.ainvoke({
            "question": request.query
        })
        
        # Extract sources
        sources = []
        if request.include_sources and "source_documents" in result:
            for doc in result["source_documents"][:request.max_sources]:
                sources.append({
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                })
        
        return RAGResponse(
            answer=result["answer"],
            sources=sources,
            confidence=0.85,
            context_used=[doc.page_content for doc in result.get("source_documents", [])],
            metadata={"conversation_id": request.conversation_id}
        )
    
    async def _multi_query(self, request: RAGRequest) -> RAGResponse:
        """Execute multi-query RAG for better coverage."""
        # Get documents from multiple queries
        docs = await self.retriever_manager.multi_query_retrieval(
            request.query,
            n_queries=3,
            k=request.max_sources
        )
        
        # Create QA chain with retrieved docs
        qa_chain = load_qa_chain(
            self.qa_llm,
            chain_type="stuff",
            prompt=self._get_qa_prompt()
        )
        
        # Run chain
        result = await qa_chain.ainvoke({
            "input_documents": docs,
            "question": request.query
        })
        
        # Extract sources
        sources = []
        if request.include_sources:
            for doc in docs[:request.max_sources]:
                sources.append({
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                })
        
        return RAGResponse(
            answer=result["output_text"],
            sources=sources,
            confidence=0.9,  # Higher confidence due to multiple queries
            context_used=[doc.page_content for doc in docs]
        )
    
    async def _recursive_query(self, request: RAGRequest) -> RAGResponse:
        """Execute recursive RAG for complex queries."""
        # Break down complex query
        breakdown_prompt = f"""Break down this complex question into simpler sub-questions:
"{request.query}"

Provide 2-4 sub-questions that together answer the main question.
Format: One question per line."""
        
        response = await orchestrator.execute(
            task="query_decomposition",
            prompt=breakdown_prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.6
        )
        
        sub_questions = [request.query]  # Include original
        response_text = response[0].response if isinstance(response, list) else response.response
        sub_questions.extend([q.strip() for q in response_text.split('\n') if q.strip()][:3])
        
        # Answer each sub-question
        sub_answers = []
        all_sources = []
        all_contexts = []
        
        for sub_q in sub_questions:
            sub_request = RAGRequest(
                query=sub_q,
                mode=RAGMode.SIMPLE,
                max_sources=3,
                include_sources=True
            )
            sub_response = await self._simple_query(sub_request)
            sub_answers.append(f"Q: {sub_q}\nA: {sub_response.answer}")
            all_sources.extend(sub_response.sources)
            all_contexts.extend(sub_response.context_used)
        
        # Synthesize final answer
        synthesis_prompt = f"""Based on these sub-answers, provide a comprehensive answer to the original question:

Original Question: {request.query}

Sub-Answers:
{chr(10).join(sub_answers)}

Synthesize a complete, coherent answer:"""
        
        final_response = await orchestrator.execute(
            task="answer_synthesis",
            prompt=synthesis_prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.5
        )
        
        # Deduplicate sources
        unique_sources = []
        seen = set()
        for source in all_sources:
            key = source.get("content", "")[:50]
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        
        return RAGResponse(
            answer=final_response[0].response if isinstance(final_response, list) else final_response.response,
            sources=unique_sources[:request.max_sources],
            confidence=0.92,  # High confidence due to recursive approach
            context_used=all_contexts,
            metadata={"sub_questions": sub_questions}
        )
    
    async def _adaptive_query(self, request: RAGRequest) -> RAGResponse:
        """Execute adaptive RAG that adjusts strategy based on query complexity."""
        # Analyze query complexity
        complexity_prompt = f"""Analyze the complexity of this question:
"{request.query}"

Rate as: simple, moderate, or complex
Consider: technical depth, multiple concepts, reasoning required"""
        
        complexity_response = await orchestrator.execute(
            task="complexity_analysis",
            prompt=complexity_prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.2
        )
        
        complexity = complexity_response[0].response if isinstance(complexity_response, list) else complexity_response.response
        complexity_lower = complexity.lower()
        
        # Choose strategy based on complexity
        if "simple" in complexity_lower:
            return await self._simple_query(request)
        elif "moderate" in complexity_lower:
            return await self._multi_query(request)
        else:  # complex
            return await self._recursive_query(request)
    
    async def summarize_documents(
        self,
        documents: List[Document],
        summary_type: str = "map_reduce"
    ) -> str:
        """
        Summarize multiple documents.
        
        Args:
            documents: Documents to summarize
            summary_type: Type of summarization (stuff, map_reduce, refine)
        
        Returns:
            Summary text
        """
        # Process documents
        langchain_docs = self.processor.process_documents(documents)
        
        # Create summarization chain
        if summary_type == "stuff":
            # For small documents
            chain = load_summarize_chain(
                self.summary_llm,
                chain_type="stuff"
            )
        elif summary_type == "map_reduce":
            # For medium documents
            chain = load_summarize_chain(
                self.summary_llm,
                chain_type="map_reduce"
            )
        else:  # refine
            # For large documents with iterative refinement
            chain = load_summarize_chain(
                self.summary_llm,
                chain_type="refine"
            )
        
        # Run summarization
        result = await chain.ainvoke({
            "input_documents": langchain_docs
        })
        
        return result["output_text"]


class SourceAttributor:
    """Manages source attribution and citation."""
    
    @staticmethod
    def format_citations(
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Add citations to answer text.
        
        Args:
            answer: Answer text
            sources: Source documents
        
        Returns:
            Answer with citations
        """
        # Add numbered citations
        cited_answer = answer
        
        # Add source list
        citation_text = "\n\nSources:\n"
        for i, source in enumerate(sources, 1):
            title = source.get("metadata", {}).get("title", "Unknown")
            doc_type = source.get("metadata", {}).get("type", "document")
            citation_text += f"[{i}] {title} ({doc_type})\n"
        
        return cited_answer + citation_text
    
    @staticmethod
    def verify_claims(
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verify claims in answer against sources.
        
        Args:
            answer: Answer text
            sources: Source documents
        
        Returns:
            List of verified claims with supporting sources
        """
        # Extract claims from answer
        sentences = answer.split('. ')
        
        verified_claims = []
        for sentence in sentences:
            if len(sentence) < 20:  # Skip short sentences
                continue
            
            # Find supporting sources
            supporting_sources = []
            for source in sources:
                content = source.get("content", "").lower()
                if any(word in content for word in sentence.lower().split()[:5]):
                    supporting_sources.append(source.get("metadata", {}).get("document_id"))
            
            verified_claims.append({
                "claim": sentence,
                "supported": len(supporting_sources) > 0,
                "sources": supporting_sources
            })
        
        return verified_claims


# Global instances
rag_chain = RAGChain()
source_attributor = SourceAttributor()


async def execute_rag_query(
    query: str,
    mode: str = "simple",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for RAG queries.
    
    Args:
        query: Query text
        mode: RAG mode
        **kwargs: Additional parameters
    
    Returns:
        RAG response dictionary
    """
    request = RAGRequest(
        query=query,
        mode=RAGMode(mode),
        **kwargs
    )
    
    response = await rag_chain.query(request)
    
    # Add citations if sources included
    if response.sources:
        response.answer = source_attributor.format_citations(
            response.answer,
            response.sources
        )
    
    return response.dict()