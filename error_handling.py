#!/usr/bin/env python3
"""
Comprehensive error handling and robustness framework for OCaml documentation processing.
Provides centralized error handling, retry mechanisms, validation, and recovery procedures.
"""

import os
import json
import time
import logging
import traceback
import threading
import hashlib
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import signal
import sys
from contextlib import contextmanager
import queue
import random


class ErrorSeverity(Enum):
    """Error severity levels for proper categorization and handling."""
    CRITICAL = "critical"    # System-breaking errors requiring immediate attention
    HIGH = "high"           # Major functionality failures
    MEDIUM = "medium"       # Recoverable errors with degraded functionality
    LOW = "low"            # Minor issues, warnings
    INFO = "info"          # Informational messages


class ErrorCategory(Enum):
    """Categories of errors for better organization and handling strategies."""
    NETWORK = "network"               # API failures, timeouts, connection issues
    FILESYSTEM = "filesystem"         # File I/O, permissions, disk space
    DATA_VALIDATION = "data_validation" # Malformed data, schema violations
    PROCESSING = "processing"         # Business logic errors, computation failures
    CONFIGURATION = "configuration"   # Setup, configuration issues
    RESOURCE = "resource"            # Memory, CPU, disk exhaustion
    EXTERNAL_DEPENDENCY = "external_dependency" # Third-party service failures


@dataclass
class ErrorContext:
    """Rich context information for errors to aid in debugging and recovery."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    component: str
    message: str
    details: Dict[str, Any]
    traceback_info: Optional[str] = None
    system_state: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    recoverable: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['severity'] = self.severity.value
        result['category'] = self.category.value
        return result


class RetryStrategy:
    """Configurable retry strategies for different types of operations."""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True,
                 exponential: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.exponential = exponential
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.exponential:
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        else:
            delay = self.base_delay * attempt
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time < self.timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = "half-open"
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e


class RobustLogger:
    """Enhanced logging with structured output, rotation, and error handling."""
    
    def __init__(self, name: str, log_dir: Path, level: str = "INFO"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create multiple loggers for different purposes
        self.main_logger = self._setup_logger(f"{name}_main", "main.log", level)
        self.error_logger = self._setup_logger(f"{name}_errors", "errors.log", "ERROR")
        self.performance_logger = self._setup_logger(f"{name}_perf", "performance.log", "INFO")
        self.audit_logger = self._setup_logger(f"{name}_audit", "audit.log", "INFO")
        
        # Error tracking
        self.error_counts = {}
        self.last_errors = []
        self.max_recent_errors = 100
        self._lock = threading.Lock()
    
    def _setup_logger(self, name: str, filename: str, level: str) -> logging.Logger:
        """Setup individual logger with file rotation."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter with structured data
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - '
            '[PID:%(process)d] [Thread:%(thread)d]'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_error(self, error_context: ErrorContext):
        """Log error with full context and tracking."""
        with self._lock:
            # Track error frequency
            error_key = f"{error_context.category.value}:{error_context.operation}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Keep recent errors for analysis
            self.last_errors.append(error_context)
            if len(self.last_errors) > self.max_recent_errors:
                self.last_errors.pop(0)
        
        # Log to appropriate channels
        error_msg = (
            f"ERROR [{error_context.error_id}] {error_context.operation}: "
            f"{error_context.message}"
        )
        
        if error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.error_logger.error(error_msg)
            self.main_logger.error(error_msg)
        else:
            self.main_logger.warning(error_msg)
        
        # Log full context as JSON for analysis
        self.audit_logger.info(f"ERROR_CONTEXT: {json.dumps(error_context.to_dict())}")
    
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any]):
        """Log performance metrics."""
        perf_data = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": time.time(),
            **details
        }
        self.performance_logger.info(f"PERFORMANCE: {json.dumps(perf_data)}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors for monitoring."""
        with self._lock:
            return {
                "error_counts": self.error_counts.copy(),
                "recent_error_count": len(self.last_errors),
                "critical_errors": len([e for e in self.last_errors 
                                      if e.severity == ErrorSeverity.CRITICAL]),
                "most_common_errors": sorted(self.error_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]
            }


class StateManager:
    """Manages application state with checkpointing and recovery capabilities."""
    
    def __init__(self, checkpoint_dir: Path, checkpoint_interval: int = 300):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = 0
        self.state_data = {}
        self._lock = threading.Lock()
    
    def update_state(self, key: str, value: Any):
        """Update state with automatic checkpointing."""
        with self._lock:
            self.state_data[key] = value
            
            # Auto-checkpoint if interval elapsed
            if time.time() - self.last_checkpoint > self.checkpoint_interval:
                self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save current state to checkpoint file."""
        try:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{int(time.time())}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "state": self.state_data
                }, f, indent=2)
            
            self.last_checkpoint = time.time()
            
            # Cleanup old checkpoints (keep last 10)
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            for old_checkpoint in checkpoints[:-10]:
                old_checkpoint.unlink()
                
        except Exception as e:
            # Don't let checkpoint failures break the main process
            print(f"Warning: Failed to save checkpoint: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                return None
            
            latest = checkpoints[-1]
            with open(latest) as f:
                data = json.load(f)
            
            self.state_data = data.get("state", {})
            return data
            
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        with self._lock:
            return self.state_data.get(key, default)
    
    def force_checkpoint(self):
        """Force immediate checkpoint."""
        with self._lock:
            self._save_checkpoint()


class DataValidator:
    """Validates data integrity and structure."""
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate JSON has required structure."""
        try:
            for field in required_fields:
                if '.' in field:
                    # Nested field check
                    parts = field.split('.')
                    current = data
                    for part in parts:
                        if not isinstance(current, dict) or part not in current:
                            return False
                        current = current[part]
                else:
                    if field not in data:
                        return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_file_integrity(file_path: Path) -> bool:
        """Validate file exists and is readable."""
        try:
            return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0
        except Exception:
            return False
    
    @staticmethod
    def calculate_checksum(file_path: Path) -> Optional[str]:
        """Calculate SHA-256 checksum of file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return None
    
    @staticmethod
    def validate_embeddings(embeddings: List[float], expected_dim: int) -> bool:
        """Validate embedding vectors."""
        try:
            if not isinstance(embeddings, list):
                return False
            if len(embeddings) != expected_dim:
                return False
            return all(isinstance(x, (int, float)) and not (
                x != x or x == float('inf') or x == float('-inf')  # Check for NaN/inf
            ) for x in embeddings)
        except Exception:
            return False


class ResourceMonitor:
    """Monitors system resources and enforces limits."""
    
    def __init__(self, 
                 max_memory_mb: int = 8000,
                 max_cpu_percent: float = 90.0,
                 max_disk_usage_percent: float = 90.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_disk_usage_percent = max_disk_usage_percent
        self.process = psutil.Process()
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage."""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            disk_usage = psutil.disk_usage('/').percent
            
            return {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "disk_usage_percent": disk_usage,
                "memory_ok": memory_mb < self.max_memory_mb,
                "cpu_ok": cpu_percent < self.max_cpu_percent,
                "disk_ok": disk_usage < self.max_disk_usage_percent,
                "overall_ok": (memory_mb < self.max_memory_mb and 
                             cpu_percent < self.max_cpu_percent and
                             disk_usage < self.max_disk_usage_percent)
            }
        except Exception as e:
            return {"error": str(e), "overall_ok": False}
    
    def enforce_limits(self) -> bool:
        """Enforce resource limits, return False if limits exceeded."""
        status = self.check_resources()
        return status.get("overall_ok", False)


class RobustErrorHandler:
    """Main error handling coordinator that ties all components together."""
    
    def __init__(self, 
                 log_dir: Path,
                 checkpoint_dir: Path,
                 component_name: str = "ocaml_processor"):
        self.logger = RobustLogger(component_name, log_dir)
        self.state_manager = StateManager(checkpoint_dir)
        self.resource_monitor = ResourceMonitor()
        self.circuit_breakers = {}
        self.retry_strategies = {}
        self.component_name = component_name
        
        # Setup default retry strategies
        self.retry_strategies.update({
            "api_call": RetryStrategy(max_attempts=3, base_delay=2.0, max_delay=30.0),
            "file_io": RetryStrategy(max_attempts=5, base_delay=0.5, max_delay=10.0),
            "network": RetryStrategy(max_attempts=4, base_delay=1.0, max_delay=60.0),
            "processing": RetryStrategy(max_attempts=2, base_delay=5.0, max_delay=120.0)
        })
        
        # Install global exception handler
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._global_exception_handler
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _global_exception_handler(self, exc_type, exc_value, exc_traceback):
        """Global exception handler to catch unhandled exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            self._original_excepthook(exc_type, exc_value, exc_traceback)
            return
        
        error_ctx = ErrorContext(
            error_id=self._generate_error_id(),
            timestamp=time.time(),
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.PROCESSING,
            operation="global_exception",
            component=self.component_name,
            message=f"Unhandled exception: {exc_type.__name__}: {exc_value}",
            details={"exception_type": exc_type.__name__},
            traceback_info=traceback.format_exception(exc_type, exc_value, exc_traceback),
            system_state=self.resource_monitor.check_resources(),
            recoverable=False
        )
        
        self.logger.log_error(error_ctx)
        self.state_manager.force_checkpoint()
        
        # Call original handler
        self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals gracefully."""
        self.logger.main_logger.warning(f"Received signal {signum}, initiating graceful shutdown")
        self.state_manager.force_checkpoint()
        sys.exit(1)
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        return f"ERR_{int(time.time())}_{random.randint(1000, 9999)}"
    
    @contextmanager
    def error_context(self, operation: str, component: str = None):
        """Context manager for error handling around operations."""
        component = component or self.component_name
        start_time = time.time()
        
        try:
            yield
            
            # Log successful operation
            duration = time.time() - start_time
            self.logger.log_performance(operation, duration, {
                "component": component,
                "status": "success"
            })
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Determine error category and severity
            category = self._categorize_error(e)
            severity = self._determine_severity(e, category)
            
            error_ctx = ErrorContext(
                error_id=self._generate_error_id(),
                timestamp=time.time(),
                severity=severity,
                category=category,
                operation=operation,
                component=component,
                message=str(e),
                details={
                    "exception_type": type(e).__name__,
                    "duration_seconds": duration
                },
                traceback_info=traceback.format_exc(),
                system_state=self.resource_monitor.check_resources()
            )
            
            self.logger.log_error(error_ctx)
            raise
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize exception by type."""
        error_type = type(exception).__name__.lower()
        
        if any(term in error_type for term in ['connection', 'timeout', 'network', 'http']):
            return ErrorCategory.NETWORK
        elif any(term in error_type for term in ['file', 'permission', 'disk', 'io']):
            return ErrorCategory.FILESYSTEM
        elif any(term in error_type for term in ['json', 'parse', 'decode', 'validation']):
            return ErrorCategory.DATA_VALIDATION
        elif any(term in error_type for term in ['memory', 'resource']):
            return ErrorCategory.RESOURCE
        elif any(term in error_type for term in ['config', 'argument']):
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.PROCESSING
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on exception and category."""
        error_type = type(exception).__name__.lower()
        
        # Critical errors that should stop processing
        if any(term in error_type for term in ['memory', 'system', 'fatal']):
            return ErrorSeverity.CRITICAL
        
        # High severity for network/external dependencies
        if category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_DEPENDENCY]:
            return ErrorSeverity.HIGH
        
        # Medium for data issues (recoverable)
        if category == ErrorCategory.DATA_VALIDATION:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.HIGH
    
    def retry_with_backoff(self, 
                          func: Callable, 
                          strategy_name: str = "default",
                          *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        strategy = self.retry_strategies.get(strategy_name, self.retry_strategies["processing"])
        
        for attempt in range(1, strategy.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == strategy.max_attempts:
                    # Final attempt failed
                    error_ctx = ErrorContext(
                        error_id=self._generate_error_id(),
                        timestamp=time.time(),
                        severity=ErrorSeverity.HIGH,
                        category=self._categorize_error(e),
                        operation=f"retry_{func.__name__}",
                        component=self.component_name,
                        message=f"Failed after {strategy.max_attempts} attempts: {str(e)}",
                        details={"final_attempt": attempt, "strategy": strategy_name},
                        retry_count=attempt - 1,
                        recoverable=False
                    )
                    self.logger.log_error(error_ctx)
                    raise
                
                # Wait before retry
                delay = strategy.get_delay(attempt)
                self.logger.main_logger.warning(
                    f"Attempt {attempt} failed for {func.__name__}: {str(e)}. "
                    f"Retrying in {delay:.1f}s"
                )
                time.sleep(delay)
        
        # Should never reach here
        raise Exception("Retry logic error")
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for given operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        return {
            "timestamp": time.time(),
            "component": self.component_name,
            "resources": self.resource_monitor.check_resources(),
            "errors": self.logger.get_error_summary(),
            "circuit_breakers": {
                name: {"state": cb.state, "failures": cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            },
            "checkpoints": len(list(self.state_manager.checkpoint_dir.glob("checkpoint_*.json")))
        }


# Convenience decorators for common error handling patterns
def robust_operation(error_handler: RobustErrorHandler, 
                    operation_name: str,
                    retry_strategy: str = "default"):
    """Decorator to wrap functions with comprehensive error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with error_handler.error_context(operation_name):
                return error_handler.retry_with_backoff(
                    func, retry_strategy, *args, **kwargs
                )
        return wrapper
    return decorator


def circuit_breaker_protected(error_handler: RobustErrorHandler, 
                             breaker_name: str,
                             **breaker_kwargs):
    """Decorator to protect functions with circuit breaker."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb = error_handler.get_circuit_breaker(breaker_name, **breaker_kwargs)
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator