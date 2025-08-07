"""
Conversation Memory System for the Automata Learning Platform.
Manages session-based conversations, long-term memory, and user preferences.
"""
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as redis
from collections import deque
import pickle

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms import Ollama

from .ai_config import get_ai_config, ModelType
from .orchestrator import orchestrator, ExecutionMode

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PREFERENCE = "preference"


class ConversationRole(str, Enum):
    """Roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Individual message in conversation."""
    role: ConversationRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class ConversationSession(BaseModel):
    """Conversation session structure."""
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    summary: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Message):
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get n most recent messages."""
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def get_context_window(self, max_tokens: int = 2048) -> List[Message]:
        """Get messages that fit within token limit."""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = 0
        max_chars = max_tokens * 4
        context_messages = []
        
        for message in reversed(self.messages):
            msg_chars = len(message.content)
            if total_chars + msg_chars > max_chars:
                break
            context_messages.insert(0, message)
            total_chars += msg_chars
        
        return context_messages


class UserPreference(BaseModel):
    """User learning preferences."""
    user_id: str
    learning_style: Optional[str] = None  # visual, textual, interactive
    difficulty_level: Optional[str] = None  # beginner, intermediate, advanced
    preferred_topics: List[str] = Field(default_factory=list)
    avoided_topics: List[str] = Field(default_factory=list)
    pace: Optional[str] = None  # slow, normal, fast
    example_preference: Optional[str] = None  # minimal, moderate, extensive
    notation_preference: Optional[str] = None  # formal, informal, mixed
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.now)


class MemoryStore:
    """Base memory storage interface."""
    
    def __init__(self):
        self.config = get_ai_config()
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value with optional TTL."""
        raise NotImplementedError
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError


class RedisMemoryStore(MemoryStore):
    """Redis-based memory storage."""
    
    def __init__(self):
        super().__init__()
        self.redis_client = None
        self.connection_pool = None
    
    async def connect(self):
        """Establish Redis connection."""
        if not self.redis_client:
            self.connection_pool = redis.ConnectionPool.from_url(
                self.config.memory.provider,
                decode_responses=False
            )
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            try:
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            await self.connection_pool.disconnect()
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in Redis."""
        await self.connect()
        
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            
            if ttl:
                await self.redis_client.setex(key, ttl, serialized)
            else:
                await self.redis_client.set(key, serialized)
            
            return True
        except Exception as e:
            logger.error(f"Redis store error: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value from Redis."""
        await self.connect()
        
        try:
            serialized = await self.redis_client.get(key)
            if serialized:
                return pickle.loads(serialized)
            return None
        except Exception as e:
            logger.error(f"Redis retrieve error: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        await self.connect()
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        await self.connect()
        
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False


class InMemoryStore(MemoryStore):
    """In-memory storage for development/testing."""
    
    def __init__(self):
        super().__init__()
        self.storage: Dict[str, Any] = {}
        self.expiry: Dict[str, datetime] = {}
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in memory."""
        self.storage[key] = value
        if ttl:
            self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
        return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value from memory."""
        # Check expiry
        if key in self.expiry:
            if datetime.now() > self.expiry[key]:
                del self.storage[key]
                del self.expiry[key]
                return None
        
        return self.storage.get(key)
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory."""
        if key in self.storage:
            del self.storage[key]
            if key in self.expiry:
                del self.expiry[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        return key in self.storage


class ConversationMemoryManager:
    """Manages conversation memory and context."""
    
    def __init__(self, store: Optional[MemoryStore] = None):
        self.config = get_ai_config()
        self.store = store or InMemoryStore()
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Initialize summarization LLM
        self.summary_llm = Ollama(
            model=self.config.models[ModelType.GENERAL].name,
            base_url=self.config.ollama_base_url,
            temperature=0.3
        )
    
    async def create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id
        )
        
        self.active_sessions[session_id] = session
        await self.store.store(
            f"session:{session_id}",
            session,
            ttl=self.config.memory.ttl
        )
        
        logger.info(f"Created session: {session_id}")
        return session
    
    async def get_session(
        self,
        session_id: str
    ) -> Optional[ConversationSession]:
        """Get an existing session."""
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to retrieve from store
        session = await self.store.retrieve(f"session:{session_id}")
        if session:
            self.active_sessions[session_id] = session
            return session
        
        return None
    
    async def add_message(
        self,
        session_id: str,
        role: ConversationRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a message to a session."""
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id)
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        session.add_message(message)
        
        # Check if summarization needed
        if len(session.messages) >= self.config.memory.summarization_threshold:
            await self._summarize_old_messages(session)
        
        # Update storage
        await self.store.store(
            f"session:{session_id}",
            session,
            ttl=self.config.memory.ttl
        )
        
        return True
    
    async def _summarize_old_messages(
        self,
        session: ConversationSession
    ):
        """Summarize old messages to save space."""
        # Get messages to summarize (keep recent ones)
        keep_recent = self.config.memory.max_messages // 2
        to_summarize = session.messages[:-keep_recent]
        
        if len(to_summarize) < 5:
            return  # Not enough to summarize
        
        # Create summary
        conversation_text = "\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in to_summarize
        ])
        
        summary_prompt = f"""Summarize this conversation concisely:

{conversation_text}

Provide a brief summary capturing key points and context:"""
        
        response = await orchestrator.execute(
            task="conversation_summary",
            prompt=summary_prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.3,
            max_tokens=500
        )
        
        summary = response[0].response if isinstance(response, list) else response.response
        
        # Update session
        session.summary = summary
        session.messages = session.messages[-keep_recent:]
        
        logger.info(f"Summarized {len(to_summarize)} messages for session {session.session_id}")
    
    async def get_context(
        self,
        session_id: str,
        max_tokens: int = 2048
    ) -> str:
        """Get conversation context for AI models."""
        session = await self.get_session(session_id)
        if not session:
            return ""
        
        context_parts = []
        
        # Add summary if exists
        if session.summary:
            context_parts.append(f"Previous conversation summary: {session.summary}")
        
        # Add recent messages
        recent_messages = session.get_context_window(max_tokens)
        for msg in recent_messages:
            context_parts.append(f"{msg.role.value}: {msg.content}")
        
        return "\n".join(context_parts)
    
    async def extract_topics(
        self,
        session_id: str
    ) -> List[str]:
        """Extract topics discussed in session."""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        # Get conversation text
        conversation_text = "\n".join([
            msg.content for msg in session.messages
            if msg.role != ConversationRole.SYSTEM
        ])
        
        # Extract topics using AI
        prompt = f"""Extract the main topics discussed in this conversation about automata theory:

{conversation_text[:2000]}

List only topic names, one per line:"""
        
        response = await orchestrator.execute(
            task="topic_extraction",
            prompt=prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.3
        )
        
        topics_text = response[0].response if isinstance(response, list) else response.response
        topics = [t.strip() for t in topics_text.split('\n') if t.strip()]
        
        # Update session
        session.topics = topics
        await self.store.store(
            f"session:{session_id}",
            session,
            ttl=self.config.memory.ttl
        )
        
        return topics


class LongTermMemory:
    """Manages long-term memory and knowledge retention."""
    
    def __init__(self, store: Optional[MemoryStore] = None):
        self.config = get_ai_config()
        self.store = store or InMemoryStore()
    
    async def store_knowledge(
        self,
        user_id: str,
        knowledge_type: str,
        content: Dict[str, Any],
        importance: float = 0.5
    ) -> bool:
        """Store knowledge in long-term memory."""
        key = f"knowledge:{user_id}:{knowledge_type}:{hashlib.md5(str(content).encode()).hexdigest()[:8]}"
        
        knowledge = {
            "type": knowledge_type,
            "content": content,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": datetime.now().isoformat()
        }
        
        return await self.store.store(key, knowledge)
    
    async def retrieve_knowledge(
        self,
        user_id: str,
        knowledge_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve knowledge from long-term memory."""
        # In a real implementation, would use pattern matching
        # For now, simple retrieval
        knowledge_items = []
        
        # This is a simplified implementation
        # In production, would use Redis SCAN or similar
        pattern = f"knowledge:{user_id}:*"
        
        # Retrieve and filter
        # (Implementation would depend on actual storage backend)
        
        return knowledge_items[:limit]
    
    async def consolidate_memories(
        self,
        user_id: str,
        session_ids: List[str]
    ) -> Dict[str, Any]:
        """Consolidate multiple sessions into long-term memory."""
        memory_manager = ConversationMemoryManager(self.store)
        
        all_topics = []
        all_concepts = []
        key_insights = []
        
        for session_id in session_ids:
            session = await memory_manager.get_session(session_id)
            if session:
                all_topics.extend(session.topics)
                
                # Extract key concepts and insights
                for msg in session.messages:
                    if msg.role == ConversationRole.ASSISTANT:
                        # Simple extraction (could be more sophisticated)
                        if "important" in msg.content.lower() or "key" in msg.content.lower():
                            key_insights.append(msg.content[:200])
        
        # Store consolidated knowledge
        consolidated = {
            "user_id": user_id,
            "topics": list(set(all_topics)),
            "insights": key_insights[:10],
            "session_count": len(session_ids),
            "consolidated_at": datetime.now().isoformat()
        }
        
        await self.store_knowledge(
            user_id,
            "consolidated",
            consolidated,
            importance=0.8
        )
        
        return consolidated


class UserPreferenceManager:
    """Manages user learning preferences."""
    
    def __init__(self, store: Optional[MemoryStore] = None):
        self.config = get_ai_config()
        self.store = store or InMemoryStore()
    
    async def update_preference(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> UserPreference:
        """Update user preferences."""
        # Get existing or create new
        existing = await self.get_preferences(user_id)
        
        if existing:
            # Update existing
            for key, value in preferences.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            existing.updated_at = datetime.now()
            pref = existing
        else:
            # Create new
            pref = UserPreference(
                user_id=user_id,
                **preferences
            )
        
        # Store
        await self.store.store(
            f"preferences:{user_id}",
            pref
        )
        
        return pref
    
    async def get_preferences(
        self,
        user_id: str
    ) -> Optional[UserPreference]:
        """Get user preferences."""
        return await self.store.retrieve(f"preferences:{user_id}")
    
    async def learn_preferences(
        self,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Learn preferences from user interactions."""
        memory_manager = ConversationMemoryManager(self.store)
        session = await memory_manager.get_session(session_id)
        
        if not session:
            return {}
        
        # Analyze conversation for preferences
        learned = {}
        
        # Analyze message patterns
        user_messages = [m for m in session.messages if m.role == ConversationRole.USER]
        
        # Detect difficulty preference
        beginner_keywords = ["simple", "basic", "explain", "what is"]
        advanced_keywords = ["prove", "formal", "theorem", "complexity"]
        
        beginner_count = sum(
            1 for msg in user_messages
            if any(kw in msg.content.lower() for kw in beginner_keywords)
        )
        advanced_count = sum(
            1 for msg in user_messages
            if any(kw in msg.content.lower() for kw in advanced_keywords)
        )
        
        if beginner_count > advanced_count * 2:
            learned["difficulty_level"] = "beginner"
        elif advanced_count > beginner_count * 2:
            learned["difficulty_level"] = "advanced"
        else:
            learned["difficulty_level"] = "intermediate"
        
        # Detect example preference
        example_requests = sum(
            1 for msg in user_messages
            if "example" in msg.content.lower()
        )
        
        if example_requests > len(user_messages) * 0.3:
            learned["example_preference"] = "extensive"
        elif example_requests > len(user_messages) * 0.1:
            learned["example_preference"] = "moderate"
        else:
            learned["example_preference"] = "minimal"
        
        # Update preferences
        if learned:
            await self.update_preference(user_id, learned)
        
        return learned
    
    async def apply_preferences(
        self,
        user_id: str,
        content: str
    ) -> str:
        """Apply user preferences to content."""
        prefs = await self.get_preferences(user_id)
        
        if not prefs:
            return content
        
        # Adjust content based on preferences
        adjusted_prompt = f"""Adjust this content for a user with these preferences:
- Difficulty: {prefs.difficulty_level or 'intermediate'}
- Examples: {prefs.example_preference or 'moderate'}
- Style: {prefs.learning_style or 'mixed'}

Original content:
{content}

Adjusted content:"""
        
        response = await orchestrator.execute(
            task="content_personalization",
            prompt=adjusted_prompt,
            mode=ExecutionMode.SEQUENTIAL,
            temperature=0.5
        )
        
        return response[0].response if isinstance(response, list) else response.response


class WorkingMemory:
    """Short-term working memory for current task context."""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        self.focus: Optional[Any] = None
    
    def add(self, item: Any):
        """Add item to working memory."""
        self.items.append(item)
    
    def set_focus(self, item: Any):
        """Set current focus item."""
        self.focus = item
        self.add(item)  # Also add to items
    
    def get_context(self) -> List[Any]:
        """Get current working memory context."""
        return list(self.items)
    
    def clear(self):
        """Clear working memory."""
        self.items.clear()
        self.focus = None
    
    def find_related(self, query: str) -> List[Any]:
        """Find related items in working memory."""
        related = []
        query_lower = query.lower()
        
        for item in self.items:
            if isinstance(item, str) and query_lower in item.lower():
                related.append(item)
            elif isinstance(item, dict):
                # Check dict values
                for value in item.values():
                    if isinstance(value, str) and query_lower in value.lower():
                        related.append(item)
                        break
        
        return related


# Global instances
memory_store = InMemoryStore()  # Use Redis in production
memory_manager = ConversationMemoryManager(memory_store)
long_term_memory = LongTermMemory(memory_store)
preference_manager = UserPreferenceManager(memory_store)
working_memory = WorkingMemory()