"""
Advanced Database Optimization Module
Provides connection pooling, query optimization, indexing, and performance monitoring
"""
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
import psycopg
from sqlalchemy import create_engine, text, event, Index, MetaData, Table
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select
import redis.asyncio as redis
from datetime import datetime, timedelta
from .config import settings
from .database import Base, User, Problem, Solution, LearningPath, AIInteraction

logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    
@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_type: QueryType
    execution_time: float
    rows_affected: int
    query_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
@dataclass
class ConnectionPoolConfig:
    """Database connection pool configuration"""
    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    
    # Connection settings
    connect_timeout: int = 10
    command_timeout: int = 60
    
    # Async pool settings (for asyncpg)
    min_pool_size: int = 10
    max_pool_size: int = 100
    
class DatabaseOptimizer:
    """Main database optimization class"""
    
    def __init__(self, database_url: str, redis_url: Optional[str] = None):
        self.database_url = database_url
        self.redis_url = redis_url
        self.config = ConnectionPoolConfig()
        
        # Connection pools
        self.sync_engine: Optional[Engine] = None
        self.async_engine = None
        self.async_session_factory = None
        self.asyncpg_pool = None
        self.redis_pool: Optional[redis.Redis] = None
        
        # Metrics
        self.query_metrics: List[QueryMetrics] = []
        self.slow_query_threshold = 1.0  # seconds
        
        # Cache
        self.query_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize all database connections and pools"""
        await self._setup_sqlalchemy_pools()
        await self._setup_asyncpg_pool()
        await self._setup_redis()
        await self._create_indexes()
        await self._setup_query_monitoring()
    
    async def _setup_sqlalchemy_pools(self):
        """Setup SQLAlchemy connection pools"""
        # Sync engine with connection pooling
        self.sync_engine = create_engine(
            self.database_url.replace('+asyncpg', ''),
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            echo=False,  # Set to True for query logging in development
            connect_args={
                "connect_timeout": self.config.connect_timeout,
                "command_timeout": self.config.command_timeout,
            }
        )
        
        # Async engine for high-performance operations
        async_database_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
        self.async_engine = create_async_engine(
            async_database_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            echo=False
        )
        
        self.async_session_factory = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def _setup_asyncpg_pool(self):
        """Setup native asyncpg connection pool for raw SQL operations"""
        try:
            # Parse database URL for asyncpg
            from urllib.parse import urlparse
            parsed = urlparse(self.database_url)
            
            self.asyncpg_pool = await asyncpg.create_pool(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:] if parsed.path else 'postgres',
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                server_settings={
                    'jit': 'off',  # Disable JIT for faster cold starts
                    'application_name': 'automata-app'
                }
            )
            logger.info(f"AsyncPG pool created with {self.config.min_pool_size}-{self.config.max_pool_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to create AsyncPG pool: {e}")
            self.asyncpg_pool = None
    
    async def _setup_redis(self):
        """Setup Redis connection for caching"""
        if not self.redis_url:
            return
            
        try:
            self.redis_pool = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
            )
            await self.redis_pool.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_pool = None
    
    async def _create_indexes(self):
        """Create optimized database indexes"""
        indexes = [
            # User indexes
            Index('idx_users_email_active', User.email, User.is_active),
            Index('idx_users_skill_level', User.skill_level),
            Index('idx_users_created_at', User.created_at),
            
            # Problem indexes  
            Index('idx_problems_type_difficulty_category', Problem.type, Problem.difficulty, Problem.category),
            Index('idx_problems_concepts', Problem.concepts, postgresql_using='gin'),
            Index('idx_problems_updated_at', Problem.updated_at),
            
            # Solution indexes
            Index('idx_solutions_user_problem_submitted', Solution.user_id, Solution.problem_id, Solution.submitted_at),
            Index('idx_solutions_correct_score', Solution.is_correct, Solution.score),
            Index('idx_solutions_submitted_at', Solution.submitted_at),
            
            # Learning path indexes
            Index('idx_learning_paths_user_concept', LearningPath.user_id, LearningPath.current_concept),
            Index('idx_learning_paths_updated_at', LearningPath.updated_at),
            
            # AI interaction indexes
            Index('idx_ai_interactions_user_type_created', AIInteraction.user_id, AIInteraction.interaction_type, AIInteraction.created_at),
            Index('idx_ai_interactions_response_time', AIInteraction.response_time),
            Index('idx_ai_interactions_feedback', AIInteraction.user_feedback),
        ]
        
        try:
            if self.sync_engine:
                with self.sync_engine.connect() as conn:
                    for index in indexes:
                        try:
                            index.create(conn, checkfirst=True)
                        except Exception as e:
                            logger.warning(f"Failed to create index {index.name}: {e}")
                    conn.commit()
                logger.info(f"Created {len(indexes)} database indexes")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    async def _setup_query_monitoring(self):
        """Setup query performance monitoring"""
        if not self.sync_engine:
            return
            
        @event.listens_for(self.sync_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.sync_engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - context._query_start_time
            
            # Determine query type
            query_type = QueryType.SELECT
            statement_upper = statement.strip().upper()
            if statement_upper.startswith('INSERT'):
                query_type = QueryType.INSERT
            elif statement_upper.startswith('UPDATE'):
                query_type = QueryType.UPDATE
            elif statement_upper.startswith('DELETE'):
                query_type = QueryType.DELETE
            
            # Create metrics
            metrics = QueryMetrics(
                query_type=query_type,
                execution_time=total_time,
                rows_affected=cursor.rowcount,
                query_hash=hash(statement)
            )
            
            self.query_metrics.append(metrics)
            
            # Log slow queries
            if total_time > self.slow_query_threshold:
                logger.warning(f"Slow query ({total_time:.2f}s): {statement[:200]}...")
    
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session with proper cleanup"""
        if not self.async_session_factory:
            raise RuntimeError("Async session factory not initialized")
            
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    @asynccontextmanager  
    async def get_asyncpg_connection(self):
        """Get raw asyncpg connection for high-performance queries"""
        if not self.asyncpg_pool:
            raise RuntimeError("AsyncPG pool not available")
            
        async with self.asyncpg_pool.acquire() as conn:
            yield conn
    
    async def execute_cached_query(self, query: str, params: tuple = (), cache_key: Optional[str] = None, ttl: int = None) -> List[Dict]:
        """Execute query with Redis caching"""
        if not self.redis_pool:
            return await self._execute_raw_query(query, params)
        
        # Generate cache key
        if not cache_key:
            cache_key = f"query:{hash(query + str(params))}"
        
        # Try cache first
        try:
            cached_result = await self.redis_pool.get(cache_key)
            if cached_result:
                import json
                return json.loads(cached_result)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        # Execute query
        result = await self._execute_raw_query(query, params)
        
        # Cache result
        try:
            import json
            await self.redis_pool.setex(
                cache_key,
                ttl or self.cache_ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return result
    
    async def _execute_raw_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute raw SQL query using asyncpg"""
        async with self.get_asyncpg_connection() as conn:
            start_time = time.time()
            rows = await conn.fetch(query, *params)
            execution_time = time.time() - start_time
            
            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow raw query ({execution_time:.2f}s): {query[:200]}...")
            
            return [dict(row) for row in rows]
    
    async def get_user_solutions_optimized(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Optimized query for user solutions with joins"""
        query = """
        SELECT 
            s.id,
            s.problem_id,
            s.score,
            s.is_correct,
            s.submitted_at,
            p.title as problem_title,
            p.difficulty as problem_difficulty,
            p.type as problem_type
        FROM solutions s
        JOIN problems p ON s.problem_id = p.id
        WHERE s.user_id = $1
        ORDER BY s.submitted_at DESC
        LIMIT $2
        """
        
        cache_key = f"user_solutions:{user_id}:{limit}"
        return await self.execute_cached_query(query, (user_id, limit), cache_key, ttl=180)
    
    async def get_problem_statistics(self, problem_id: str) -> Dict:
        """Get aggregated problem statistics"""
        query = """
        SELECT 
            COUNT(*) as total_attempts,
            COUNT(CASE WHEN is_correct THEN 1 END) as correct_attempts,
            AVG(score) as average_score,
            AVG(time_spent) as average_time,
            AVG(hints_used) as average_hints
        FROM solutions 
        WHERE problem_id = $1
        """
        
        cache_key = f"problem_stats:{problem_id}"
        result = await self.execute_cached_query(query, (problem_id,), cache_key, ttl=600)
        return result[0] if result else {}
    
    async def get_learning_progress(self, user_id: str) -> Dict:
        """Get comprehensive learning progress data"""
        query = """
        SELECT 
            concept,
            COUNT(*) as total_problems,
            COUNT(CASE WHEN s.is_correct THEN 1 END) as solved_problems,
            AVG(s.score) as average_score
        FROM problems p
        LEFT JOIN solutions s ON p.id = s.problem_id AND s.user_id = $1
        WHERE p.concepts IS NOT NULL
        GROUP BY concept
        ORDER BY average_score DESC NULLS LAST
        """
        
        cache_key = f"learning_progress:{user_id}"
        return await self.execute_cached_query(query, (user_id,), cache_key, ttl=300)
    
    async def bulk_insert_solutions(self, solutions: List[Dict]) -> None:
        """Optimized bulk insert for solutions"""
        if not solutions:
            return
            
        async with self.get_asyncpg_connection() as conn:
            await conn.executemany(
                """
                INSERT INTO solutions (id, user_id, problem_id, automaton_data, score, is_correct, submitted_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO NOTHING
                """,
                [(
                    sol['id'], sol['user_id'], sol['problem_id'], 
                    sol['automaton_data'], sol['score'], sol['is_correct'], 
                    sol['submitted_at']
                ) for sol in solutions]
            )
    
    async def invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache keys matching pattern"""
        if not self.redis_pool:
            return
            
        try:
            keys = await self.redis_pool.keys(pattern)
            if keys:
                await self.redis_pool.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache keys for pattern: {pattern}")
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get database performance metrics"""
        if not self.query_metrics:
            return {}
        
        total_queries = len(self.query_metrics)
        slow_queries = [m for m in self.query_metrics if m.execution_time > self.slow_query_threshold]
        
        by_type = {}
        for query_type in QueryType:
            type_metrics = [m for m in self.query_metrics if m.query_type == query_type]
            if type_metrics:
                by_type[query_type.value] = {
                    'count': len(type_metrics),
                    'avg_time': sum(m.execution_time for m in type_metrics) / len(type_metrics),
                    'max_time': max(m.execution_time for m in type_metrics)
                }
        
        return {
            'total_queries': total_queries,
            'slow_queries': len(slow_queries),
            'slow_query_percentage': len(slow_queries) / total_queries * 100 if total_queries > 0 else 0,
            'by_type': by_type,
            'pool_info': {
                'sync_pool_size': self.sync_engine.pool.size() if self.sync_engine else 0,
                'sync_pool_checked_out': self.sync_engine.pool.checkedout() if self.sync_engine else 0,
                'asyncpg_pool_size': self.asyncpg_pool._queue.qsize() if self.asyncpg_pool else 0,
            }
        }
    
    async def cleanup(self):
        """Cleanup all database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.sync_engine:
            self.sync_engine.dispose()
        if self.asyncpg_pool:
            await self.asyncpg_pool.close()
        if self.redis_pool:
            await self.redis_pool.close()

# Global optimizer instance
db_optimizer: Optional[DatabaseOptimizer] = None

async def get_db_optimizer() -> DatabaseOptimizer:
    """Get or create database optimizer instance"""
    global db_optimizer
    if db_optimizer is None:
        db_optimizer = DatabaseOptimizer(
            database_url=settings.database_url,
            redis_url=getattr(settings, 'redis_url', None)
        )
        await db_optimizer.initialize()
    return db_optimizer

# Dependency for FastAPI
async def get_optimized_db_session():
    """FastAPI dependency for optimized database session"""
    optimizer = await get_db_optimizer()
    async with optimizer.get_async_session() as session:
        yield session

# Health check functions
async def check_database_health() -> Dict[str, Any]:
    """Check database connection health"""
    optimizer = await get_db_optimizer()
    
    try:
        async with optimizer.get_asyncpg_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            
        return {
            "status": "healthy",
            "database": "connected",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health"""
    optimizer = await get_db_optimizer()
    
    if not optimizer.redis_pool:
        return {
            "status": "disabled",
            "redis": "not_configured"
        }
    
    try:
        await optimizer.redis_pool.ping()
        return {
            "status": "healthy",
            "redis": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "redis": "disconnected", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }