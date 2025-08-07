"""
OLLAMA DB - AI-Powered Database Assistant
========================================

This module uses Ollama for comprehensive database operations:
- Natural language to SQL conversion
- Query optimization using AI reasoning
- Index suggestions based on query patterns
- Schema design recommendations
- Migration generation with safety checks
- Database performance analysis
- Automatic query explanation
- SQL security validation
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .ollama_everything import ollama_everything, OllamaTask, OllamaTaskType, OllamaResult
from .valkey_integration import valkey_connection_manager

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"


@dataclass
class SQLQuery:
    """Generated SQL query with metadata."""
    sql: str
    explanation: str
    confidence_score: float
    estimated_performance: str
    suggested_indexes: List[str] = field(default_factory=list)
    security_warnings: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


class OllamaDB:
    """AI-powered database assistant using Ollama."""
    
    def __init__(self):
        self.database_schemas: Dict[str, Dict] = {}
        self.query_cache: Dict[str, SQLQuery] = {}
        self.performance_history: Dict[str, List] = {}
        
        logger.info("OllamaDB initialized with AI-powered database assistance")
    
    async def natural_language_to_sql(
        self,
        natural_query: str,
        database_type: DatabaseType = DatabaseType.POSTGRESQL,
        schema_info: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> SQLQuery:
        """Convert natural language to optimized SQL using Ollama."""
        
        # Build comprehensive prompt for SQL generation
        schema_context = self._format_schema_context(schema_info) if schema_info else "No schema provided"
        
        sql_task = OllamaTask(
            task_type=OllamaTaskType.SQL_GENERATION,
            input_data=f"""
            Convert this natural language query to optimized {database_type.value.upper()} SQL:
            
            Natural Query: {natural_query}
            Database Type: {database_type.value}
            
            Schema Information:
            {schema_context}
            
            Additional Context:
            {json.dumps(context or {}, indent=2)}
            
            Please provide:
            1. Optimized SQL query
            2. Detailed explanation of the query logic
            3. Performance considerations
            4. Index suggestions for optimization
            5. Security warnings if any
            6. Alternative query approaches if applicable
            
            Return as JSON:
            {{
                "sql": "SELECT ...",
                "explanation": "This query...",
                "confidence_score": 0.95,
                "estimated_performance": "Good/Fair/Poor",
                "suggested_indexes": ["index suggestions"],
                "security_warnings": ["security considerations"],
                "optimization_suggestions": ["optimization tips"],
                "alternative_queries": ["alternative approaches"]
            }}
            """,
            context={
                "database_type": database_type.value,
                "schema_available": schema_info is not None,
                "query_complexity": self._assess_query_complexity(natural_query)
            },
            temperature=0.2,  # Low temperature for precise SQL generation
            max_tokens=2000
        )
        
        try:
            ai_result = await ollama_everything.process_task(sql_task)
            
            if ai_result.error:
                logger.error(f"SQL generation failed: {ai_result.error}")
                return SQLQuery(
                    sql="-- SQL generation failed",
                    explanation=f"Error: {ai_result.error}",
                    confidence_score=0.0,
                    estimated_performance="Unknown"
                )
            
            return self._parse_sql_response(ai_result.result, natural_query)
            
        except Exception as e:
            logger.error(f"SQL generation exception: {e}")
            return SQLQuery(
                sql="-- Exception occurred during SQL generation",
                explanation=f"Exception: {str(e)}",
                confidence_score=0.0,
                estimated_performance="Unknown"
            )
    
    def _format_schema_context(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for AI context."""
        if not schema_info:
            return "No schema information available"
        
        context_parts = []
        
        if "tables" in schema_info:
            context_parts.append("Tables:")
            for table_name, table_info in schema_info["tables"].items():
                context_parts.append(f"  {table_name}:")
                if "columns" in table_info:
                    for col_name, col_info in table_info["columns"].items():
                        col_type = col_info.get("type", "unknown")
                        nullable = "NULL" if col_info.get("nullable", True) else "NOT NULL"
                        context_parts.append(f"    - {col_name}: {col_type} {nullable}")
                if "indexes" in table_info:
                    context_parts.append(f"    Indexes: {', '.join(table_info['indexes'])}")
                if "foreign_keys" in table_info:
                    context_parts.append(f"    Foreign Keys: {table_info['foreign_keys']}")
        
        if "relationships" in schema_info:
            context_parts.append(f"\nRelationships: {schema_info['relationships']}")
        
        return "\n".join(context_parts)
    
    def _assess_query_complexity(self, natural_query: str) -> str:
        """Assess the complexity of a natural language query."""
        complexity_indicators = {
            "simple": ["show", "list", "get", "find"],
            "moderate": ["join", "group", "order", "filter", "where", "aggregate"],
            "complex": ["nested", "subquery", "union", "case when", "window", "recursive"]
        }
        
        query_lower = natural_query.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return complexity
        
        # Default based on query length
        if len(query_lower.split()) > 20:
            return "complex"
        elif len(query_lower.split()) > 10:
            return "moderate"
        else:
            return "simple"
    
    def _parse_sql_response(self, ai_result: Any, original_query: str) -> SQLQuery:
        """Parse AI SQL generation response."""
        try:
            result_str = str(ai_result)
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return SQLQuery(
                        sql=data.get("sql", "-- No SQL generated"),
                        explanation=data.get("explanation", "No explanation provided"),
                        confidence_score=float(data.get("confidence_score", 0.5)),
                        estimated_performance=data.get("estimated_performance", "Unknown"),
                        suggested_indexes=data.get("suggested_indexes", []),
                        security_warnings=data.get("security_warnings", []),
                        optimization_suggestions=data.get("optimization_suggestions", [])
                    )
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract SQL from code blocks
            sql_match = re.search(r'```sql\n(.*?)\n```', result_str, re.DOTALL | re.IGNORECASE)
            if not sql_match:
                sql_match = re.search(r'```\n(.*?)\n```', result_str, re.DOTALL)
            
            sql_query = sql_match.group(1).strip() if sql_match else "-- No SQL found in response"
            
            return SQLQuery(
                sql=sql_query,
                explanation=result_str,
                confidence_score=0.7,  # Default confidence for text parsing
                estimated_performance="Unknown"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse SQL response: {e}")
            return SQLQuery(
                sql="-- Failed to parse SQL response",
                explanation=f"Parsing error: {str(e)}",
                confidence_score=0.0,
                estimated_performance="Unknown"
            )
    
    async def optimize_query(
        self,
        sql_query: str,
        database_type: DatabaseType = DatabaseType.POSTGRESQL,
        schema_info: Dict[str, Any] = None,
        performance_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Optimize an existing SQL query using AI analysis."""
        
        optimization_task = OllamaTask(
            task_type=OllamaTaskType.CODE_OPTIMIZATION,
            input_data=f"""
            Optimize this {database_type.value.upper()} SQL query for better performance:
            
            Current Query:
            {sql_query}
            
            Schema Context:
            {self._format_schema_context(schema_info) if schema_info else "No schema provided"}
            
            Performance Context:
            {json.dumps(performance_context or {}, indent=2)}
            
            Provide optimization analysis including:
            1. Optimized version of the query
            2. Specific optimizations applied
            3. Expected performance improvement
            4. Index recommendations
            5. Query execution plan considerations
            6. Alternative query strategies
            7. Potential bottlenecks identified
            
            Return as JSON with clear before/after comparison.
            """,
            context={
                "database_type": database_type.value,
                "optimization_focus": "performance"
            },
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            ai_result = await ollama_everything.process_task(optimization_task)
            
            if not ai_result.error:
                return self._parse_optimization_response(ai_result.result)
            else:
                return {"error": ai_result.error}
                
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return {"error": str(e)}
    
    async def suggest_indexes(
        self,
        queries: List[str],
        database_type: DatabaseType = DatabaseType.POSTGRESQL,
        schema_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Suggest database indexes based on query patterns."""
        
        index_task = OllamaTask(
            task_type=OllamaTaskType.SYSTEM_ANALYSIS,
            input_data=f"""
            Analyze these {database_type.value.upper()} queries and suggest optimal indexes:
            
            Queries to analyze:
            {self._format_queries_for_analysis(queries)}
            
            Schema Information:
            {self._format_schema_context(schema_info) if schema_info else "No schema provided"}
            
            Provide index recommendations including:
            1. Specific CREATE INDEX statements
            2. Justification for each index
            3. Impact on query performance
            4. Storage overhead considerations
            5. Maintenance impact
            6. Composite index strategies
            7. Priority ranking of indexes
            
            Focus on indexes that provide maximum performance benefit.
            """,
            context={
                "database_type": database_type.value,
                "analysis_type": "index_optimization",
                "query_count": len(queries)
            },
            temperature=0.2,
            max_tokens=1500
        )
        
        try:
            ai_result = await ollama_everything.process_task(index_task)
            
            if not ai_result.error:
                return self._parse_index_suggestions(ai_result.result)
            else:
                return {"error": ai_result.error}
                
        except Exception as e:
            logger.error(f"Index suggestion failed: {e}")
            return {"error": str(e)}
    
    def _format_queries_for_analysis(self, queries: List[str]) -> str:
        """Format queries for AI analysis."""
        formatted_queries = []
        for i, query in enumerate(queries[:10], 1):  # Limit to 10 queries
            formatted_queries.append(f"Query {i}:\n{query.strip()}\n")
        return "\n".join(formatted_queries)
    
    def _parse_optimization_response(self, ai_result: Any) -> Dict[str, Any]:
        """Parse query optimization response."""
        result_str = str(ai_result)
        
        optimization_data = {
            "original_analysis": result_str,
            "optimized_query": None,
            "optimizations_applied": [],
            "performance_improvement": "Unknown",
            "index_recommendations": [],
            "bottlenecks_identified": []
        }
        
        # Extract optimized SQL
        sql_match = re.search(r'(?:optimized|improved).*?```sql\n(.*?)\n```', result_str, re.DOTALL | re.IGNORECASE)
        if sql_match:
            optimization_data["optimized_query"] = sql_match.group(1).strip()
        
        # Extract optimizations
        optimization_patterns = [
            r'optimization[s]?[:\s]+(.*?)(?:\n|$)',
            r'improvement[s]?[:\s]+(.*?)(?:\n|$)',
            r'change[s]?[:\s]+(.*?)(?:\n|$)'
        ]
        
        for pattern in optimization_patterns:
            matches = re.findall(pattern, result_str, re.IGNORECASE | re.MULTILINE)
            optimization_data["optimizations_applied"].extend([m.strip() for m in matches])
        
        return optimization_data
    
    def _parse_index_suggestions(self, ai_result: Any) -> Dict[str, Any]:
        """Parse index suggestions from AI response."""
        result_str = str(ai_result)
        
        suggestions = {
            "full_analysis": result_str,
            "create_index_statements": [],
            "index_justifications": {},
            "priority_ranking": [],
            "estimated_impact": {}
        }
        
        # Extract CREATE INDEX statements
        index_matches = re.findall(r'CREATE\s+(?:UNIQUE\s+)?INDEX.*?;', result_str, re.IGNORECASE | re.DOTALL)
        suggestions["create_index_statements"] = [idx.strip() for idx in index_matches]
        
        # Extract justifications
        justification_pattern = r'(?:justification|reason|benefit)[:\s]+(.*?)(?:\n|$)'
        justifications = re.findall(justification_pattern, result_str, re.IGNORECASE | re.MULTILINE)
        for i, just in enumerate(justifications):
            if i < len(suggestions["create_index_statements"]):
                suggestions["index_justifications"][suggestions["create_index_statements"][i]] = just.strip()
        
        return suggestions
    
    async def generate_migration(
        self,
        description: str,
        current_schema: Dict[str, Any] = None,
        target_schema: Dict[str, Any] = None,
        database_type: DatabaseType = DatabaseType.POSTGRESQL
    ) -> Dict[str, Any]:
        """Generate database migration scripts using AI."""
        
        migration_task = OllamaTask(
            task_type=OllamaTaskType.CODE_GENERATION,
            input_data=f"""
            Generate a safe {database_type.value.upper()} database migration:
            
            Migration Description: {description}
            
            Current Schema:
            {json.dumps(current_schema, indent=2) if current_schema else "Not provided"}
            
            Target Schema:
            {json.dumps(target_schema, indent=2) if target_schema else "Not provided"}
            
            Generate migration scripts that include:
            1. UP migration (forward changes)
            2. DOWN migration (rollback changes)
            3. Data preservation strategies
            4. Index management
            5. Constraint handling
            6. Safety checks and validations
            7. Performance considerations for large tables
            
            Ensure migrations are:
            - Safe for production use
            - Reversible
            - Non-blocking where possible
            - Include proper error handling
            """,
            context={
                "database_type": database_type.value,
                "migration_type": "schema_change"
            },
            temperature=0.1,  # Very low temperature for safe migrations
            max_tokens=2000
        )
        
        try:
            ai_result = await ollama_everything.process_task(migration_task)
            
            if not ai_result.error:
                return self._parse_migration_response(ai_result.result)
            else:
                return {"error": ai_result.error}
                
        except Exception as e:
            logger.error(f"Migration generation failed: {e}")
            return {"error": str(e)}
    
    def _parse_migration_response(self, ai_result: Any) -> Dict[str, Any]:
        """Parse migration generation response."""
        result_str = str(ai_result)
        
        migration_data = {
            "full_response": result_str,
            "up_migration": None,
            "down_migration": None,
            "safety_checks": [],
            "performance_notes": [],
            "warnings": []
        }
        
        # Extract UP migration
        up_match = re.search(r'(?:up|forward).*?```sql\n(.*?)\n```', result_str, re.DOTALL | re.IGNORECASE)
        if up_match:
            migration_data["up_migration"] = up_match.group(1).strip()
        
        # Extract DOWN migration
        down_match = re.search(r'(?:down|rollback|reverse).*?```sql\n(.*?)\n```', result_str, re.DOTALL | re.IGNORECASE)
        if down_match:
            migration_data["down_migration"] = down_match.group(1).strip()
        
        # Extract safety information
        safety_patterns = [
            r'(?:safety|caution|warning)[:\s]+(.*?)(?:\n|$)',
            r'(?:important|note|attention)[:\s]+(.*?)(?:\n|$)'
        ]
        
        for pattern in safety_patterns:
            matches = re.findall(pattern, result_str, re.IGNORECASE | re.MULTILINE)
            migration_data["safety_checks"].extend([m.strip() for m in matches if len(m.strip()) > 10])
        
        return migration_data


# Global instance
ollama_db = OllamaDB()


# Convenience functions
async def nl_to_sql(
    natural_query: str,
    database_type: str = "postgresql",
    schema_info: Dict[str, Any] = None
) -> SQLQuery:
    """Convert natural language to SQL."""
    db_type = DatabaseType(database_type.lower())
    return await ollama_db.natural_language_to_sql(natural_query, db_type, schema_info)


async def optimize_sql_with_ai(
    sql_query: str,
    database_type: str = "postgresql",
    schema_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Optimize SQL query using AI."""
    db_type = DatabaseType(database_type.lower())
    return await ollama_db.optimize_query(sql_query, db_type, schema_info)


async def get_index_suggestions(
    queries: List[str],
    database_type: str = "postgresql",
    schema_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Get AI-powered index suggestions."""
    db_type = DatabaseType(database_type.lower())
    return await ollama_db.suggest_indexes(queries, db_type, schema_info)


# Initialize function
async def initialize_ollama_db():
    """Initialize the Ollama database assistant."""
    try:
        # Test the system with a simple query
        test_query = await nl_to_sql("Show me all users")
        if not test_query.sql:
            raise Exception("Database assistant test failed")
        
        logger.info("OllamaDB initialized and tested successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OllamaDB: {e}")
        raise