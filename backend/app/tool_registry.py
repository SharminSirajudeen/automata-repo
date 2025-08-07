"""
Dynamic Tool Registry and Management System
============================================

This module provides a registry for tools that agents can discover and use.
Tools can be registered statically or created dynamically based on needs.

Author: APEX AI System
Version: 2.0
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import inspect
import uuid

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools available in the system."""
    GENERATOR = "generator"
    CONVERTER = "converter"
    MINIMIZER = "minimizer"
    ANALYZER = "analyzer"
    VERIFIER = "verifier"
    OPTIMIZER = "optimizer"
    SIMULATOR = "simulator"
    VISUALIZER = "visualizer"
    SOLVER = "solver"
    PROVER = "prover"
    DYNAMIC = "dynamic"
    UTILITY = "utility"


@dataclass
class ToolMetadata:
    """Metadata about a tool."""
    author: str = "system"
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """Represents a tool that agents can use."""
    tool_id: str
    name: str
    category: ToolCategory
    description: str
    function: Callable
    parameters: Dict[str, str]  # parameter_name -> type_hint
    examples: List[str]
    capabilities: List[str]
    metadata: ToolMetadata = field(default_factory=ToolMetadata)
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        
        start_time = datetime.utcnow()
        
        try:
            # Validate parameters
            self._validate_parameters(params)
            
            # Execute function
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(params)
            else:
                result = self.function(params)
            
            # Update metadata
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.metadata.last_used = datetime.utcnow()
            self.metadata.usage_count += 1
            self._update_avg_execution_time(execution_time)
            
            # Return result
            if isinstance(result, ToolResult):
                result.execution_time = execution_time
                return result
            else:
                return ToolResult(
                    success=True,
                    data=result,
                    execution_time=execution_time
                )
                
        except Exception as e:
            logger.error(f"Error executing tool {self.tool_id}: {e}")
            
            # Update failure metrics
            self._update_success_rate(False)
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _validate_parameters(self, params: Dict[str, Any]):
        """Validate that required parameters are provided."""
        
        required_params = [p for p in self.parameters if not p.startswith("optional_")]
        
        for param in required_params:
            if param not in params:
                raise ValueError(f"Required parameter '{param}' not provided")
    
    def _update_avg_execution_time(self, new_time: float):
        """Update average execution time."""
        
        count = self.metadata.usage_count
        if count == 1:
            self.metadata.avg_execution_time = new_time
        else:
            current_avg = self.metadata.avg_execution_time
            self.metadata.avg_execution_time = (
                (current_avg * (count - 1) + new_time) / count
            )
    
    def _update_success_rate(self, success: bool):
        """Update success rate."""
        
        count = self.metadata.usage_count
        if count == 0:
            self.metadata.success_rate = 1.0 if success else 0.0
        else:
            current_rate = self.metadata.success_rate
            successes = current_rate * count
            if success:
                successes += 1
            self.metadata.success_rate = successes / (count + 1)
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information."""
        
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "parameters": self.parameters,
            "examples": self.examples,
            "capabilities": self.capabilities,
            "metadata": {
                "usage_count": self.metadata.usage_count,
                "success_rate": self.metadata.success_rate,
                "avg_execution_time": self.metadata.avg_execution_time,
                "tags": self.metadata.tags
            }
        }


class ToolRegistry:
    """
    Registry for managing and discovering tools.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
        self.capability_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        
        logger.info("Tool Registry initialized")
    
    def register_tool(self, tool: Tool) -> bool:
        """Register a new tool in the registry."""
        
        if tool.tool_id in self.tools:
            logger.warning(f"Tool {tool.tool_id} already registered")
            return False
        
        # Add to main registry
        self.tools[tool.tool_id] = tool
        
        # Add to category index
        self.categories[tool.category].append(tool.tool_id)
        
        # Add to capability index
        for capability in tool.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(tool.tool_id)
        
        # Add to tag index
        for tag in tool.metadata.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(tool.tool_id)
        
        logger.info(f"Registered tool: {tool.name} ({tool.tool_id})")
        return True
    
    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool from the registry."""
        
        if tool_id not in self.tools:
            return False
        
        tool = self.tools[tool_id]
        
        # Remove from main registry
        del self.tools[tool_id]
        
        # Remove from category index
        self.categories[tool.category].remove(tool_id)
        
        # Remove from capability index
        for capability in tool.capabilities:
            if capability in self.capability_index:
                self.capability_index[capability].discard(tool_id)
        
        # Remove from tag index
        for tag in tool.metadata.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(tool_id)
        
        logger.info(f"Unregistered tool: {tool_id}")
        return True
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID."""
        return self.tools.get(tool_id)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self.tools.values())
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category."""
        
        tool_ids = self.categories.get(category, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def get_tools_by_capability(self, capability: str) -> List[Tool]:
        """Get all tools with a specific capability."""
        
        tool_ids = self.capability_index.get(capability, set())
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def get_tools_by_tag(self, tag: str) -> List[Tool]:
        """Get all tools with a specific tag."""
        
        tool_ids = self.tag_index.get(tag, set())
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def search_tools(
        self,
        query: str,
        category: Optional[ToolCategory] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[Tool]:
        """Search for tools based on various criteria."""
        
        results = set(self.tools.keys())
        
        # Filter by category
        if category:
            category_tools = set(self.categories.get(category, []))
            results &= category_tools
        
        # Filter by capabilities
        if capabilities:
            for capability in capabilities:
                capability_tools = self.capability_index.get(capability, set())
                results &= capability_tools
        
        # Filter by tags
        if tags:
            for tag in tags:
                tag_tools = self.tag_index.get(tag, set())
                results &= tag_tools
        
        # Filter by query string
        if query:
            query_lower = query.lower()
            filtered = set()
            for tool_id in results:
                tool = self.tools[tool_id]
                if (query_lower in tool.name.lower() or
                    query_lower in tool.description.lower()):
                    filtered.add(tool_id)
            results = filtered
        
        return [self.tools[tid] for tid in results]
    
    def recommend_tools(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tool]:
        """Recommend tools based on task description."""
        
        recommendations = []
        task_lower = task_description.lower()
        
        # Keyword-based recommendations
        keyword_map = {
            "minimize": [ToolCategory.MINIMIZER, ToolCategory.OPTIMIZER],
            "convert": [ToolCategory.CONVERTER],
            "generate": [ToolCategory.GENERATOR],
            "verify": [ToolCategory.VERIFIER],
            "prove": [ToolCategory.PROVER],
            "simulate": [ToolCategory.SIMULATOR],
            "visualize": [ToolCategory.VISUALIZER],
            "solve": [ToolCategory.SOLVER],
            "analyze": [ToolCategory.ANALYZER]
        }
        
        categories_to_check = []
        for keyword, categories in keyword_map.items():
            if keyword in task_lower:
                categories_to_check.extend(categories)
        
        # Get tools from relevant categories
        for category in set(categories_to_check):
            recommendations.extend(self.get_tools_by_category(category))
        
        # Sort by success rate and usage
        recommendations.sort(
            key=lambda t: (t.metadata.success_rate, t.metadata.usage_count),
            reverse=True
        )
        
        # Return top recommendations
        return recommendations[:5]
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered tools."""
        
        total_tools = len(self.tools)
        
        category_stats = {
            cat.value: len(self.categories[cat])
            for cat in ToolCategory
        }
        
        # Calculate usage statistics
        total_usage = sum(t.metadata.usage_count for t in self.tools.values())
        avg_success_rate = (
            sum(t.metadata.success_rate for t in self.tools.values()) / total_tools
            if total_tools > 0 else 0
        )
        
        most_used = sorted(
            self.tools.values(),
            key=lambda t: t.metadata.usage_count,
            reverse=True
        )[:5]
        
        return {
            "total_tools": total_tools,
            "categories": category_stats,
            "total_usage": total_usage,
            "avg_success_rate": avg_success_rate,
            "most_used_tools": [
                {
                    "name": t.name,
                    "usage_count": t.metadata.usage_count,
                    "success_rate": t.metadata.success_rate
                }
                for t in most_used
            ],
            "capabilities": list(self.capability_index.keys()),
            "tags": list(self.tag_index.keys())
        }
    
    def create_composite_tool(
        self,
        name: str,
        tool_ids: List[str],
        orchestration: str = "sequential"
    ) -> Tool:
        """Create a composite tool from multiple existing tools."""
        
        component_tools = [self.get_tool(tid) for tid in tool_ids if tid in self.tools]
        
        if not component_tools:
            raise ValueError("No valid component tools provided")
        
        async def composite_function(params: Dict[str, Any]) -> ToolResult:
            """Execute composite tool."""
            
            results = []
            current_input = params
            
            if orchestration == "sequential":
                # Execute tools in sequence
                for tool in component_tools:
                    result = await tool.execute(current_input)
                    results.append(result)
                    
                    if not result.success:
                        return ToolResult(
                            success=False,
                            error=f"Failed at tool {tool.name}: {result.error}",
                            data={"partial_results": results}
                        )
                    
                    # Use output as input for next tool
                    if result.data:
                        current_input = result.data if isinstance(result.data, dict) else {"data": result.data}
                
                return ToolResult(
                    success=True,
                    data={
                        "final_output": results[-1].data if results else None,
                        "all_results": [r.data for r in results]
                    }
                )
                
            elif orchestration == "parallel":
                # Execute tools in parallel
                tasks = [tool.execute(params) for tool in component_tools]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures
                failures = [r for r in results if isinstance(r, Exception) or (isinstance(r, ToolResult) and not r.success)]
                
                if failures:
                    return ToolResult(
                        success=False,
                        error=f"Some tools failed: {failures}",
                        data={"results": results}
                    )
                
                return ToolResult(
                    success=True,
                    data={"parallel_results": [r.data if isinstance(r, ToolResult) else r for r in results]}
                )
            
            else:
                raise ValueError(f"Unknown orchestration type: {orchestration}")
        
        # Create composite tool
        composite_tool = Tool(
            tool_id=f"composite_{uuid.uuid4().hex[:8]}",
            name=name,
            category=ToolCategory.DYNAMIC,
            description=f"Composite tool combining: {', '.join(t.name for t in component_tools)}",
            function=composite_function,
            parameters={
                **component_tools[0].parameters  # Use first tool's parameters as base
            },
            examples=[],
            capabilities=list(set(cap for t in component_tools for cap in t.capabilities)),
            metadata=ToolMetadata(
                tags=["composite", orchestration],
                dependencies=tool_ids
            )
        )
        
        # Register the composite tool
        self.register_tool(composite_tool)
        
        return composite_tool
    
    def export_registry(self) -> Dict[str, Any]:
        """Export the tool registry for persistence."""
        
        return {
            "tools": {
                tool_id: {
                    "name": tool.name,
                    "category": tool.category.value,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "examples": tool.examples,
                    "capabilities": tool.capabilities,
                    "metadata": {
                        "author": tool.metadata.author,
                        "version": tool.metadata.version,
                        "tags": tool.metadata.tags,
                        "usage_count": tool.metadata.usage_count,
                        "success_rate": tool.metadata.success_rate
                    }
                }
                for tool_id, tool in self.tools.items()
            },
            "statistics": self.get_tool_statistics()
        }
    
    def import_registry(self, data: Dict[str, Any]):
        """Import tools from exported data."""
        
        # Note: This would need actual function implementations
        # In practice, you'd have a function factory or module loader
        
        logger.info(f"Would import {len(data.get('tools', {}))} tools")
        # Implementation would go here


# Global registry instance
tool_registry = ToolRegistry()