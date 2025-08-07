"""
Graceful Shutdown Handler System.
Handles signals, saves state before shutdown, closes connections properly,
and manages cleanup tasks for safe application termination.
"""

import asyncio
import signal
import logging
import time
import threading
from typing import List, Callable, Optional, Any, Dict
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of graceful shutdown."""
    RUNNING = "running"
    SHUTDOWN_INITIATED = "shutdown_initiated"
    STOPPING_NEW_REQUESTS = "stopping_new_requests"
    DRAINING_CONNECTIONS = "draining_connections"
    SAVING_STATE = "saving_state"
    CLEANUP_RESOURCES = "cleanup_resources"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    FORCE_SHUTDOWN = "force_shutdown"


@dataclass
class ShutdownTask:
    """A task to run during shutdown."""
    name: str
    callback: Callable
    priority: int = 0  # Higher priority runs first
    timeout_seconds: float = 30.0
    phase: ShutdownPhase = ShutdownPhase.CLEANUP_RESOURCES
    is_critical: bool = False  # Critical tasks must complete
    
    async def execute(self) -> bool:
        """Execute the shutdown task."""
        try:
            logger.info(f"Executing shutdown task: {self.name}")
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(self.callback):
                await asyncio.wait_for(self.callback(), timeout=self.timeout_seconds)
            else:
                # Run sync function in executor with timeout
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.callback),
                    timeout=self.timeout_seconds
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Shutdown task '{self.name}' completed in {elapsed:.2f}s")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Shutdown task '{self.name}' timed out after {self.timeout_seconds}s")
            return not self.is_critical
        except Exception as e:
            logger.error(f"Shutdown task '{self.name}' failed: {e}")
            return not self.is_critical


@dataclass
class ShutdownStats:
    """Statistics about the shutdown process."""
    shutdown_initiated_at: datetime
    shutdown_completed_at: Optional[datetime] = None
    shutdown_reason: str = "unknown"
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    total_duration_seconds: float = 0.0
    forced_shutdown: bool = False
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of shutdown tasks."""
        if self.total_tasks == 0:
            return 1.0
        return self.completed_tasks / self.total_tasks


class GracefulShutdownHandler:
    """Handles graceful shutdown of the application."""
    
    def __init__(
        self,
        shutdown_timeout: float = 30.0,
        force_shutdown_timeout: float = 60.0
    ):
        self.shutdown_timeout = shutdown_timeout
        self.force_shutdown_timeout = force_shutdown_timeout
        
        # Shutdown state
        self.current_phase = ShutdownPhase.RUNNING
        self.shutdown_initiated = False
        self.shutdown_event = asyncio.Event()
        self.force_shutdown_event = asyncio.Event()
        
        # Tasks and callbacks
        self.shutdown_tasks: List[ShutdownTask] = []
        self.phase_tasks: Dict[ShutdownPhase, List[ShutdownTask]] = {phase: [] for phase in ShutdownPhase}
        
        # Statistics
        self.stats = ShutdownStats(datetime.now())
        
        # Active connections and resources to track
        self.active_connections = weakref.WeakSet()
        self.resource_managers = []
        
        # Signal handlers
        self.original_handlers = {}
        self.shutdown_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        logger.info("Graceful shutdown handler initialized")
    
    def register_shutdown_task(
        self,
        name: str,
        callback: Callable,
        priority: int = 0,
        timeout_seconds: float = 30.0,
        phase: ShutdownPhase = ShutdownPhase.CLEANUP_RESOURCES,
        is_critical: bool = False
    ):
        """Register a shutdown task."""
        task = ShutdownTask(
            name=name,
            callback=callback,
            priority=priority,
            timeout_seconds=timeout_seconds,
            phase=phase,
            is_critical=is_critical
        )
        
        self.shutdown_tasks.append(task)
        self.phase_tasks[phase].append(task)
        
        # Sort by priority (highest first)
        self.shutdown_tasks.sort(key=lambda t: t.priority, reverse=True)
        self.phase_tasks[phase].sort(key=lambda t: t.priority, reverse=True)
        
        logger.debug(f"Registered shutdown task: {name} (priority: {priority})")
    
    def register_resource_manager(self, manager: Any):
        """Register a resource manager that needs cleanup."""
        if hasattr(manager, 'shutdown') or hasattr(manager, 'close'):
            self.resource_managers.append(manager)
            logger.debug(f"Registered resource manager: {type(manager).__name__}")
    
    def track_connection(self, connection: Any):
        """Track an active connection."""
        self.active_connections.add(connection)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        # Only setup signal handlers in the main thread
        if threading.current_thread() != threading.main_thread():
            logger.warning("Signal handlers can only be set up in the main thread")
            return
        
        # Signals to handle
        signals_to_handle = [
            signal.SIGTERM,  # Termination request
            signal.SIGINT,   # Interrupt (Ctrl+C)
        ]
        
        # Add SIGQUIT on Unix systems
        if hasattr(signal, 'SIGQUIT'):
            signals_to_handle.append(signal.SIGQUIT)
        
        for sig in signals_to_handle:
            try:
                # Store original handler
                self.original_handlers[sig] = signal.signal(sig, signal.SIG_DFL)
                
                # Set new handler
                signal.signal(sig, self._signal_handler)
                
                logger.info(f"Registered signal handler for {sig.name}")
                
            except (OSError, ValueError) as e:
                logger.warning(f"Could not register handler for {sig.name}: {e}")
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received shutdown signal: {signal_name}")
        
        self.stats.shutdown_reason = f"signal_{signal_name.lower()}"
        
        # Schedule graceful shutdown
        if not self.shutdown_initiated:
            # Use asyncio.create_task if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                if not self.shutdown_task or self.shutdown_task.done():
                    self.shutdown_task = loop.create_task(self.initiate_shutdown())
            except RuntimeError:
                # Not in async context, set event for manual checking
                logger.warning("Signal received outside async context")
                self.shutdown_initiated = True
        else:
            # Second signal = force shutdown
            logger.warning(f"Second {signal_name} signal - forcing immediate shutdown")
            self.force_shutdown_event.set()
    
    async def initiate_shutdown(self, reason: str = "manual"):
        """Initiate graceful shutdown process."""
        async with self.lock:
            if self.shutdown_initiated:
                logger.warning("Shutdown already initiated")
                return
            
            self.shutdown_initiated = True
            self.stats.shutdown_initiated_at = datetime.now()
            self.stats.shutdown_reason = reason
            
            logger.info(f"Initiating graceful shutdown: {reason}")
            self.shutdown_event.set()
        
        try:
            # Start shutdown process
            await self._execute_shutdown_sequence()
        except Exception as e:
            logger.error(f"Error during shutdown sequence: {e}")
            await self._force_shutdown()
    
    async def _execute_shutdown_sequence(self):
        """Execute the complete shutdown sequence."""
        start_time = time.time()
        
        try:
            # Phase 1: Stop accepting new requests
            await self._execute_phase(ShutdownPhase.STOPPING_NEW_REQUESTS)
            
            # Phase 2: Drain existing connections
            await self._execute_phase(ShutdownPhase.DRAINING_CONNECTIONS)
            await self._wait_for_connections_to_drain()
            
            # Phase 3: Save critical state
            await self._execute_phase(ShutdownPhase.SAVING_STATE)
            
            # Phase 4: Cleanup resources
            await self._execute_phase(ShutdownPhase.CLEANUP_RESOURCES)
            
            # Complete shutdown
            self.current_phase = ShutdownPhase.SHUTDOWN_COMPLETE
            self.stats.shutdown_completed_at = datetime.now()
            self.stats.total_duration_seconds = time.time() - start_time
            
            logger.info(
                f"Graceful shutdown completed in {self.stats.total_duration_seconds:.2f}s "
                f"({self.stats.completed_tasks}/{self.stats.total_tasks} tasks successful)"
            )
            
        except asyncio.TimeoutError:
            logger.error("Shutdown sequence timed out - forcing shutdown")
            await self._force_shutdown()
        except Exception as e:
            logger.error(f"Shutdown sequence failed: {e}")
            await self._force_shutdown()
    
    async def _execute_phase(self, phase: ShutdownPhase):
        """Execute all tasks for a specific phase."""
        self.current_phase = phase
        logger.info(f"Entering shutdown phase: {phase.value}")
        
        tasks = self.phase_tasks.get(phase, [])
        if not tasks:
            return
        
        # Execute tasks with timeout
        phase_timeout = min(self.shutdown_timeout / 4, 15.0)  # Max 15s per phase
        
        try:
            await asyncio.wait_for(
                self._execute_tasks(tasks),
                timeout=phase_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Phase {phase.value} timed out after {phase_timeout}s")
            raise
    
    async def _execute_tasks(self, tasks: List[ShutdownTask]):
        """Execute a list of shutdown tasks."""
        for task in tasks:
            # Check for force shutdown
            if self.force_shutdown_event.is_set():
                logger.warning("Force shutdown requested - stopping task execution")
                break
            
            self.stats.total_tasks += 1
            
            success = await task.execute()
            
            if success:
                self.stats.completed_tasks += 1
            else:
                self.stats.failed_tasks += 1
                
                # Stop if critical task failed
                if task.is_critical:
                    logger.error(f"Critical shutdown task failed: {task.name}")
                    raise Exception(f"Critical shutdown task failed: {task.name}")
    
    async def _wait_for_connections_to_drain(self, timeout: float = 10.0):
        """Wait for active connections to drain."""
        start_time = time.time()
        
        while len(self.active_connections) > 0 and time.time() - start_time < timeout:
            if self.force_shutdown_event.is_set():
                break
            
            logger.info(f"Waiting for {len(self.active_connections)} connections to close...")
            await asyncio.sleep(1.0)
        
        remaining = len(self.active_connections)
        if remaining > 0:
            logger.warning(f"{remaining} connections did not close gracefully")
    
    async def _force_shutdown(self):
        """Force immediate shutdown."""
        logger.warning("Forcing immediate shutdown")
        
        self.current_phase = ShutdownPhase.FORCE_SHUTDOWN
        self.stats.forced_shutdown = True
        self.stats.shutdown_completed_at = datetime.now()
        
        # Close all resource managers immediately
        for manager in self.resource_managers:
            try:
                if hasattr(manager, 'shutdown'):
                    if asyncio.iscoroutinefunction(manager.shutdown):
                        await asyncio.wait_for(manager.shutdown(), timeout=2.0)
                    else:
                        manager.shutdown()
                elif hasattr(manager, 'close'):
                    if asyncio.iscoroutinefunction(manager.close):
                        await asyncio.wait_for(manager.close(), timeout=2.0)
                    else:
                        manager.close()
            except Exception as e:
                logger.error(f"Error during force shutdown of {type(manager).__name__}: {e}")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete."""
        await self.shutdown_event.wait()
    
    def is_shutdown_initiated(self) -> bool:
        """Check if shutdown has been initiated."""
        return self.shutdown_initiated
    
    def get_shutdown_stats(self) -> Dict[str, Any]:
        """Get shutdown statistics."""
        return {
            'shutdown_initiated': self.shutdown_initiated,
            'current_phase': self.current_phase.value,
            'shutdown_reason': self.stats.shutdown_reason,
            'total_tasks': self.stats.total_tasks,
            'completed_tasks': self.stats.completed_tasks,
            'failed_tasks': self.stats.failed_tasks,
            'success_rate': self.stats.success_rate,
            'total_duration_seconds': self.stats.total_duration_seconds,
            'forced_shutdown': self.stats.forced_shutdown,
            'active_connections': len(self.active_connections),
            'registered_tasks': len(self.shutdown_tasks)
        }
    
    def restore_signal_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self.original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass


class ShutdownManager:
    """Manager for coordinating application shutdown across modules."""
    
    def __init__(self):
        self.shutdown_handler = GracefulShutdownHandler()
        self.modules_registered = False
        
        # Register core shutdown tasks
        self._register_core_tasks()
    
    def _register_core_tasks(self):
        """Register core application shutdown tasks."""
        
        # Valkey shutdown
        self.shutdown_handler.register_shutdown_task(
            name="shutdown_valkey",
            callback=self._shutdown_valkey,
            priority=90,
            phase=ShutdownPhase.CLEANUP_RESOURCES,
            is_critical=True,
            timeout_seconds=10.0
        )
        
        # Memory manager shutdown
        self.shutdown_handler.register_shutdown_task(
            name="shutdown_memory_manager",
            callback=self._shutdown_memory_manager,
            priority=80,
            phase=ShutdownPhase.SAVING_STATE,
            is_critical=False,
            timeout_seconds=15.0
        )
        
        # Checkpoint manager shutdown
        self.shutdown_handler.register_shutdown_task(
            name="shutdown_checkpoint_manager",
            callback=self._shutdown_checkpoint_manager,
            priority=85,
            phase=ShutdownPhase.SAVING_STATE,
            is_critical=True,
            timeout_seconds=20.0
        )
        
        # Semantic cache shutdown
        self.shutdown_handler.register_shutdown_task(
            name="shutdown_semantic_cache",
            callback=self._shutdown_semantic_cache,
            priority=70,
            phase=ShutdownPhase.SAVING_STATE,
            is_critical=False,
            timeout_seconds=10.0
        )
        
        # Rate limiter shutdown
        self.shutdown_handler.register_shutdown_task(
            name="shutdown_rate_limiter",
            callback=self._shutdown_rate_limiter,
            priority=75,
            phase=ShutdownPhase.SAVING_STATE,
            is_critical=False,
            timeout_seconds=5.0
        )
        
        # Cost tracker shutdown
        self.shutdown_handler.register_shutdown_task(
            name="shutdown_cost_tracker",
            callback=self._shutdown_cost_tracker,
            priority=60,
            phase=ShutdownPhase.SAVING_STATE,
            is_critical=False,
            timeout_seconds=5.0
        )
        
        # Health monitor shutdown
        self.shutdown_handler.register_shutdown_task(
            name="shutdown_health_monitor",
            callback=self._shutdown_health_monitor,
            priority=50,
            phase=ShutdownPhase.CLEANUP_RESOURCES,
            is_critical=False,
            timeout_seconds=5.0
        )
    
    async def _shutdown_valkey(self):
        """Shutdown Valkey connections."""
        try:
            from .valkey_integration import shutdown_valkey
            await shutdown_valkey()
        except ImportError:
            logger.warning("Valkey module not available for shutdown")
        except Exception as e:
            logger.error(f"Error shutting down Valkey: {e}")
            raise
    
    async def _shutdown_memory_manager(self):
        """Shutdown memory manager."""
        try:
            from .memory_manager import shutdown_memory_manager
            await shutdown_memory_manager()
        except ImportError:
            logger.warning("Memory manager module not available for shutdown")
        except Exception as e:
            logger.error(f"Error shutting down memory manager: {e}")
    
    async def _shutdown_checkpoint_manager(self):
        """Shutdown checkpoint manager."""
        try:
            from .checkpoint_manager import shutdown_checkpoint_manager
            await shutdown_checkpoint_manager()
        except ImportError:
            logger.warning("Checkpoint manager module not available for shutdown")
        except Exception as e:
            logger.error(f"Error shutting down checkpoint manager: {e}")
            raise
    
    async def _shutdown_semantic_cache(self):
        """Shutdown semantic cache."""
        try:
            from .semantic_cache import shutdown_semantic_cache
            await shutdown_semantic_cache()
        except ImportError:
            logger.warning("Semantic cache module not available for shutdown")
        except Exception as e:
            logger.error(f"Error shutting down semantic cache: {e}")
    
    async def _shutdown_rate_limiter(self):
        """Shutdown rate limiter."""
        try:
            from .middleware.rate_limiter import shutdown_rate_limiter
            await shutdown_rate_limiter()
        except ImportError:
            logger.warning("Rate limiter module not available for shutdown")
        except Exception as e:
            logger.error(f"Error shutting down rate limiter: {e}")
    
    async def _shutdown_cost_tracker(self):
        """Shutdown cost tracker."""
        try:
            from .ollama_cost_tracker import shutdown_cost_tracker
            await shutdown_cost_tracker()
        except ImportError:
            logger.warning("Cost tracker module not available for shutdown")
        except Exception as e:
            logger.error(f"Error shutting down cost tracker: {e}")
    
    async def _shutdown_health_monitor(self):
        """Shutdown health monitor."""
        try:
            from .health_checks import shutdown_health_monitor
            await shutdown_health_monitor()
        except ImportError:
            logger.warning("Health monitor module not available for shutdown")
        except Exception as e:
            logger.error(f"Error shutting down health monitor: {e}")
    
    def setup(self):
        """Setup signal handlers and initialize shutdown system."""
        self.shutdown_handler.setup_signal_handlers()
        logger.info("Shutdown manager setup complete")
    
    async def initiate_shutdown(self, reason: str = "manual"):
        """Initiate graceful shutdown."""
        await self.shutdown_handler.initiate_shutdown(reason)
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete."""
        await self.shutdown_handler.wait_for_shutdown()
    
    def is_shutdown_initiated(self) -> bool:
        """Check if shutdown is in progress."""
        return self.shutdown_handler.is_shutdown_initiated()
    
    def register_custom_task(
        self,
        name: str,
        callback: Callable,
        priority: int = 0,
        phase: ShutdownPhase = ShutdownPhase.CLEANUP_RESOURCES,
        is_critical: bool = False
    ):
        """Register a custom shutdown task."""
        self.shutdown_handler.register_shutdown_task(
            name=name,
            callback=callback,
            priority=priority,
            phase=phase,
            is_critical=is_critical
        )
    
    def track_connection(self, connection: Any):
        """Track an active connection."""
        self.shutdown_handler.track_connection(connection)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get shutdown statistics."""
        return self.shutdown_handler.get_shutdown_stats()


# Global shutdown manager instance
shutdown_manager = ShutdownManager()


# Context manager for tracking connections
@asynccontextmanager
async def tracked_connection(connection: Any):
    """Context manager for tracking connections during shutdown."""
    shutdown_manager.track_connection(connection)
    try:
        yield connection
    finally:
        # Connection will be automatically removed from WeakSet when deleted
        pass


# Decorators
def shutdown_task(
    priority: int = 0,
    phase: ShutdownPhase = ShutdownPhase.CLEANUP_RESOURCES,
    is_critical: bool = False,
    timeout_seconds: float = 30.0
):
    """Decorator to register a function as a shutdown task."""
    def decorator(func):
        task_name = f"{func.__module__}.{func.__name__}"
        shutdown_manager.register_custom_task(
            name=task_name,
            callback=func,
            priority=priority,
            phase=phase,
            is_critical=is_critical
        )
        return func
    return decorator


# Application lifecycle management
class ApplicationLifecycle:
    """Manages complete application lifecycle including startup and shutdown."""
    
    def __init__(self):
        self.startup_complete = False
        self.shutdown_initiated = False
    
    async def startup(self):
        """Perform application startup."""
        try:
            logger.info("Starting application lifecycle...")
            
            # Initialize all components
            from .valkey_integration import initialize_valkey
            from .memory_manager import initialize_memory_manager
            from .checkpoint_manager import initialize_checkpoint_manager
            from .semantic_cache import initialize_semantic_cache
            from .middleware.rate_limiter import initialize_rate_limiter
            from .ollama_cost_tracker import initialize_cost_tracker
            from .health_checks import initialize_health_monitor
            
            # Initialize components in order
            await initialize_valkey()
            await initialize_memory_manager()
            await initialize_checkpoint_manager()
            await initialize_semantic_cache()
            await initialize_rate_limiter()
            await initialize_cost_tracker()
            await initialize_health_monitor()
            
            # Setup shutdown handling
            shutdown_manager.setup()
            
            self.startup_complete = True
            logger.info("Application startup completed successfully")
            
        except Exception as e:
            logger.error(f"Application startup failed: {e}")
            raise
    
    async def shutdown(self, reason: str = "manual"):
        """Perform application shutdown."""
        if self.shutdown_initiated:
            return
        
        self.shutdown_initiated = True
        logger.info(f"Starting application shutdown: {reason}")
        
        await shutdown_manager.initiate_shutdown(reason)
        
        logger.info("Application shutdown completed")
    
    def is_healthy(self) -> bool:
        """Check if application is healthy."""
        return self.startup_complete and not self.shutdown_initiated


# Global application lifecycle instance
app_lifecycle = ApplicationLifecycle()


# Convenience functions
async def startup_application():
    """Start the application."""
    await app_lifecycle.startup()


async def shutdown_application(reason: str = "manual"):
    """Shutdown the application."""
    await app_lifecycle.shutdown(reason)


def is_application_healthy() -> bool:
    """Check if application is healthy."""
    return app_lifecycle.is_healthy()


def register_shutdown_task(
    name: str,
    callback: Callable,
    priority: int = 0,
    is_critical: bool = False
):
    """Register a custom shutdown task."""
    shutdown_manager.register_custom_task(name, callback, priority, is_critical)


def is_shutdown_initiated() -> bool:
    """Check if shutdown has been initiated."""
    return shutdown_manager.is_shutdown_initiated()


# Signal handling for standalone usage
def setup_signal_handling():
    """Setup signal handling for graceful shutdown."""
    shutdown_manager.setup()


# Main application runner with graceful shutdown
async def run_application_with_shutdown(main_coroutine: Callable):
    """Run application with proper shutdown handling."""
    try:
        # Setup shutdown handling
        setup_signal_handling()
        
        # Start application
        await startup_application()
        
        # Run main application logic
        app_task = asyncio.create_task(main_coroutine())
        shutdown_task = asyncio.create_task(shutdown_manager.wait_for_shutdown())
        
        # Wait for either app completion or shutdown signal
        done, pending = await asyncio.wait(
            [app_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Shutdown application
        if not shutdown_manager.is_shutdown_initiated():
            await shutdown_application("application_completed")
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await shutdown_application("keyboard_interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        await shutdown_application("application_error")
    finally:
        logger.info("Application runner finished")