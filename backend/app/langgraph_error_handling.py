"""
Comprehensive Error Handling and Recovery for LangGraph Workflows.
Provides error detection, classification, recovery strategies, and monitoring.
"""

import asyncio
import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .langgraph_core import ConversationState, WorkflowStatus, InterruptType
from .redis_integration import redis_state_manager, redis_checkpoint_store

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Categories of errors in workflow execution."""
    NETWORK_ERROR = "network_error"
    MODEL_ERROR = "model_error"
    STATE_ERROR = "state_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    DATA_ERROR = "data_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    LOGIC_ERROR = "logic_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    RESTART = "restart"
    MANUAL_INTERVENTION = "manual_intervention"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    session_id: str
    node_name: str
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: str
    state_snapshot: Optional[Dict[str, Any]] = None
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""
    strategy: RecoveryStrategy
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    timeout_seconds: int = 30
    prerequisites: List[str] = field(default_factory=list)


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
    
    def classify_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Classify an error and create error context."""
        error_id = f"error_{int(datetime.now().timestamp() * 1000)}"
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        
        # Determine category
        category = self._determine_category(exception, error_message)
        
        # Determine severity
        severity = self._determine_severity(exception, category, context)
        
        return ErrorContext(
            error_id=error_id,
            session_id=context.get("session_id", "unknown"),
            node_name=context.get("node_name", "unknown"),
            error_type=error_type,
            error_message=error_message,
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            stack_trace=stack_trace,
            state_snapshot=context.get("state"),
            metadata=context.get("metadata", {})
        )
    
    def determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryAction:
        """Determine the best recovery strategy for an error."""
        category = error_context.category
        severity = error_context.severity
        attempts = error_context.recovery_attempts
        
        # Apply classification rules
        for rule in self.classification_rules:
            if rule["condition"](error_context):
                return RecoveryAction(
                    strategy=rule["strategy"],
                    description=rule["description"],
                    parameters=rule.get("parameters", {}),
                    max_attempts=rule.get("max_attempts", 3),
                    timeout_seconds=rule.get("timeout_seconds", 30)
                )
        
        # Default strategy
        return RecoveryAction(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            description="Manual intervention required - no automatic recovery available",
            max_attempts=1
        )
    
    def _determine_category(self, exception: Exception, message: str) -> ErrorCategory:
        """Determine error category based on exception type and message."""
        error_type = type(exception).__name__.lower()
        message_lower = message.lower()
        
        # Network-related errors
        if any(keyword in error_type for keyword in ["connection", "timeout", "network"]):
            return ErrorCategory.NETWORK_ERROR
        if any(keyword in message_lower for keyword in ["connection", "timeout", "network"]):
            return ErrorCategory.NETWORK_ERROR
        
        # Model-related errors
        if any(keyword in error_type for keyword in ["model", "ollama", "openai"]):
            return ErrorCategory.MODEL_ERROR
        if any(keyword in message_lower for keyword in ["model", "generation", "completion"]):
            return ErrorCategory.MODEL_ERROR
        
        # State-related errors
        if any(keyword in error_type for keyword in ["state", "checkpoint", "redis"]):
            return ErrorCategory.STATE_ERROR
        if any(keyword in message_lower for keyword in ["state", "checkpoint", "session"]):
            return ErrorCategory.STATE_ERROR
        
        # Validation errors
        if any(keyword in error_type for keyword in ["validation", "value", "schema"]):
            return ErrorCategory.VALIDATION_ERROR
        
        # Timeout errors
        if "timeout" in error_type or "timeout" in message_lower:
            return ErrorCategory.TIMEOUT_ERROR
        
        # Resource errors
        if any(keyword in error_type for keyword in ["memory", "disk", "resource"]):
            return ErrorCategory.RESOURCE_ERROR
        
        # Authentication/Permission errors
        if any(keyword in error_type for keyword in ["auth", "permission", "forbidden"]):
            return ErrorCategory.AUTHENTICATION_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _determine_severity(
        self, 
        exception: Exception, 
        category: ErrorCategory, 
        context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Determine error severity."""
        # Critical errors that require immediate attention
        if category in [ErrorCategory.STATE_ERROR, ErrorCategory.RESOURCE_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.MODEL_ERROR, ErrorCategory.AUTHENTICATION_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.TIMEOUT_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def _build_classification_rules(self) -> List[Dict[str, Any]]:
        """Build classification rules for error recovery."""
        return [
            {
                "condition": lambda ctx: ctx.category == ErrorCategory.NETWORK_ERROR and ctx.recovery_attempts < 3,
                "strategy": RecoveryStrategy.RETRY,
                "description": "Retry network operation with exponential backoff",
                "parameters": {"backoff_factor": 2, "max_delay": 60},
                "max_attempts": 3,
                "timeout_seconds": 30
            },
            {
                "condition": lambda ctx: ctx.category == ErrorCategory.MODEL_ERROR and ctx.recovery_attempts < 2,
                "strategy": RecoveryStrategy.FALLBACK,
                "description": "Use fallback model or simplified prompt",
                "parameters": {"fallback_model": "simple"},
                "max_attempts": 2,
                "timeout_seconds": 45
            },
            {
                "condition": lambda ctx: ctx.category == ErrorCategory.TIMEOUT_ERROR,
                "strategy": RecoveryStrategy.RETRY,
                "description": "Retry with increased timeout",
                "parameters": {"timeout_multiplier": 2},
                "max_attempts": 2
            },
            {
                "condition": lambda ctx: ctx.category == ErrorCategory.STATE_ERROR,
                "strategy": RecoveryStrategy.RESTART,
                "description": "Restart workflow from last valid checkpoint",
                "max_attempts": 1
            },
            {
                "condition": lambda ctx: ctx.severity == ErrorSeverity.CRITICAL,
                "strategy": RecoveryStrategy.MANUAL_INTERVENTION,
                "description": "Critical error requires manual intervention",
                "max_attempts": 1
            },
            {
                "condition": lambda ctx: ctx.recovery_attempts >= 3,
                "strategy": RecoveryStrategy.GRACEFUL_DEGRADATION,
                "description": "Too many recovery attempts - graceful degradation",
                "max_attempts": 1
            }
        ]


class ErrorRecoveryEngine:
    """Executes recovery strategies for workflow errors."""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.recovery_handlers = self._build_recovery_handlers()
    
    async def handle_error(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle an error with appropriate recovery strategy."""
        try:
            # Classify the error
            error_context = self.classifier.classify_error(exception, context)
            
            # Log error
            await self._log_error(error_context)
            
            # Determine recovery strategy
            recovery_action = self.classifier.determine_recovery_strategy(error_context)
            
            # Execute recovery
            recovery_result = await self._execute_recovery(error_context, recovery_action)
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.error(f"Error in recovery process: {recovery_error}")
            return {
                "success": False,
                "strategy": "none",
                "message": "Recovery process failed",
                "error": str(recovery_error)
            }
    
    async def _execute_recovery(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Execute a specific recovery action."""
        strategy = recovery_action.strategy
        
        if strategy not in self.recovery_handlers:
            logger.error(f"No handler for recovery strategy: {strategy}")
            return {
                "success": False,
                "strategy": strategy.value,
                "message": f"No handler available for strategy: {strategy.value}"
            }
        
        handler = self.recovery_handlers[strategy]
        
        try:
            # Update recovery attempt count
            error_context.recovery_attempts += 1
            
            # Execute recovery with timeout
            result = await asyncio.wait_for(
                handler(error_context, recovery_action),
                timeout=recovery_action.timeout_seconds
            )
            
            # Log successful recovery
            logger.info(f"Recovery successful: {strategy.value} for error {error_context.error_id}")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Recovery timeout: {strategy.value} for error {error_context.error_id}")
            return {
                "success": False,
                "strategy": strategy.value,
                "message": "Recovery operation timed out"
            }
        except Exception as e:
            logger.error(f"Recovery failed: {strategy.value} for error {error_context.error_id}: {e}")
            return {
                "success": False,
                "strategy": strategy.value,
                "message": f"Recovery operation failed: {str(e)}"
            }
    
    def _build_recovery_handlers(self) -> Dict[RecoveryStrategy, Callable]:
        """Build recovery handler functions."""
        return {
            RecoveryStrategy.RETRY: self._handle_retry,
            RecoveryStrategy.FALLBACK: self._handle_fallback,
            RecoveryStrategy.SKIP: self._handle_skip,
            RecoveryStrategy.RESTART: self._handle_restart,
            RecoveryStrategy.MANUAL_INTERVENTION: self._handle_manual_intervention,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._handle_graceful_degradation,
            RecoveryStrategy.ABORT: self._handle_abort
        }
    
    async def _handle_retry(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Handle retry recovery strategy."""
        parameters = recovery_action.parameters
        
        # Implement exponential backoff
        backoff_factor = parameters.get("backoff_factor", 2)
        max_delay = parameters.get("max_delay", 60)
        
        delay = min(backoff_factor ** error_context.recovery_attempts, max_delay)
        
        logger.info(f"Retrying in {delay} seconds for error {error_context.error_id}")
        await asyncio.sleep(delay)
        
        return {
            "success": True,
            "strategy": "retry",
            "message": f"Retry scheduled after {delay} seconds",
            "delay": delay,
            "attempt": error_context.recovery_attempts
        }
    
    async def _handle_fallback(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Handle fallback recovery strategy."""
        parameters = recovery_action.parameters
        fallback_model = parameters.get("fallback_model", "simple")
        
        # Update state to use fallback configuration
        if error_context.state_snapshot:
            error_context.state_snapshot["metadata"]["fallback_mode"] = True
            error_context.state_snapshot["metadata"]["fallback_model"] = fallback_model
        
        return {
            "success": True,
            "strategy": "fallback",
            "message": f"Switched to fallback mode: {fallback_model}",
            "fallback_model": fallback_model
        }
    
    async def _handle_skip(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Handle skip recovery strategy."""
        return {
            "success": True,
            "strategy": "skip",
            "message": f"Skipped node: {error_context.node_name}",
            "skipped_node": error_context.node_name
        }
    
    async def _handle_restart(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Handle restart recovery strategy."""
        session_id = error_context.session_id
        
        try:
            # Find last valid checkpoint
            checkpoints = await redis_checkpoint_store.list_checkpoints(session_id)
            
            if checkpoints:
                latest_checkpoint = checkpoints[0]  # Most recent
                checkpoint_data = await redis_checkpoint_store.load_checkpoint(
                    session_id, latest_checkpoint["version"]
                )
                
                if checkpoint_data:
                    # Save current state as failed state
                    await redis_state_manager.save_state(
                        session_id, 
                        error_context.state_snapshot or {},
                        state_type="failed_state"
                    )
                    
                    return {
                        "success": True,
                        "strategy": "restart",
                        "message": f"Restarted from checkpoint: {latest_checkpoint['version']}",
                        "checkpoint_version": latest_checkpoint["version"],
                        "restored_state": checkpoint_data
                    }
            
            # No checkpoint available - restart from beginning
            return {
                "success": True,
                "strategy": "restart",
                "message": "Restarted from beginning - no checkpoint available",
                "full_restart": True
            }
            
        except Exception as e:
            logger.error(f"Failed to restart from checkpoint: {e}")
            return {
                "success": False,
                "strategy": "restart",
                "message": f"Restart failed: {str(e)}"
            }
    
    async def _handle_manual_intervention(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Handle manual intervention recovery strategy."""
        # Create intervention request
        intervention_request = {
            "error_id": error_context.error_id,
            "session_id": error_context.session_id,
            "error_details": {
                "category": error_context.category.value,
                "severity": error_context.severity.value,
                "message": error_context.error_message,
                "node": error_context.node_name
            },
            "requested_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Store intervention request (would typically notify administrators)
        await redis_state_manager.save_state(
            error_context.session_id,
            intervention_request,
            state_type="intervention_request"
        )
        
        return {
            "success": True,
            "strategy": "manual_intervention",
            "message": "Manual intervention requested - workflow paused",
            "intervention_id": error_context.error_id,
            "requires_admin": True
        }
    
    async def _handle_graceful_degradation(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Handle graceful degradation recovery strategy."""
        # Enable degraded mode - simplified functionality
        degraded_state = {
            "degraded_mode": True,
            "degradation_reason": error_context.error_message,
            "degraded_since": datetime.now().isoformat(),
            "original_error_id": error_context.error_id
        }
        
        # Update state
        if error_context.state_snapshot:
            error_context.state_snapshot["metadata"]["degraded_mode"] = degraded_state
        
        return {
            "success": True,
            "strategy": "graceful_degradation",
            "message": "Workflow degraded to simplified mode",
            "degradation_details": degraded_state
        }
    
    async def _handle_abort(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> Dict[str, Any]:
        """Handle abort recovery strategy."""
        # Clean up resources and terminate workflow
        session_id = error_context.session_id
        
        # Save error state for analysis
        error_summary = {
            "aborted_at": datetime.now().isoformat(),
            "error_context": {
                "error_id": error_context.error_id,
                "category": error_context.category.value,
                "severity": error_context.severity.value,
                "message": error_context.error_message
            },
            "final_state": error_context.state_snapshot
        }
        
        await redis_state_manager.save_state(
            session_id,
            error_summary,
            state_type="aborted_workflow"
        )
        
        return {
            "success": True,
            "strategy": "abort",
            "message": "Workflow aborted due to unrecoverable error",
            "abort_reason": error_context.error_message,
            "final_state_saved": True
        }
    
    async def _log_error(self, error_context: ErrorContext):
        """Log error details for monitoring and analysis."""
        log_entry = {
            "timestamp": error_context.timestamp.isoformat(),
            "error_id": error_context.error_id,
            "session_id": error_context.session_id,
            "node_name": error_context.node_name,
            "error_type": error_context.error_type,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "message": error_context.error_message,
            "recovery_attempts": error_context.recovery_attempts,
            "metadata": error_context.metadata
        }
        
        # Log based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {json.dumps(log_entry, indent=2)}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {json.dumps(log_entry, indent=2)}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {json.dumps(log_entry, indent=2)}")
        else:
            logger.info(f"LOW SEVERITY ERROR: {json.dumps(log_entry, indent=2)}")
        
        # Store error log for analysis
        await redis_state_manager.save_state(
            error_context.session_id,
            log_entry,
            state_type="error_log"
        )


class ErrorMonitor:
    """Monitors errors and provides analytics."""
    
    def __init__(self):
        self.error_patterns = {}
        self.recovery_success_rates = {}
    
    async def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time window."""
        try:
            # This would typically query the error logs from Redis
            # Simplified implementation for demo
            stats = {
                "total_errors": 0,
                "errors_by_category": {},
                "errors_by_severity": {},
                "recovery_success_rate": 0.0,
                "most_common_errors": [],
                "error_trends": {},
                "time_window_hours": time_window_hours,
                "generated_at": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {"error": str(e)}
    
    async def detect_error_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in errors that might indicate systemic issues."""
        try:
            patterns = []
            
            # Pattern detection logic would go here
            # For example: frequent timeouts, specific model failures, etc.
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect error patterns: {e}")
            return []
    
    async def suggest_preventive_measures(self, error_patterns: List[Dict[str, Any]]) -> List[str]:
        """Suggest preventive measures based on error patterns."""
        suggestions = []
        
        for pattern in error_patterns:
            if pattern.get("type") == "frequent_timeouts":
                suggestions.append("Consider increasing timeout values or optimizing slow operations")
            elif pattern.get("type") == "model_failures":
                suggestions.append("Review model configurations and consider fallback models")
            elif pattern.get("type") == "memory_errors":
                suggestions.append("Implement memory management and consider resource limits")
        
        return suggestions


# Global error recovery engine
error_recovery_engine = ErrorRecoveryEngine()
error_monitor = ErrorMonitor()