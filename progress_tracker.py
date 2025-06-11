#!/usr/bin/env python3
"""
Progress tracking and resumption system for long-running OCaml documentation processing tasks.
Provides checkpoint-based resumption, progress persistence, and recovery capabilities.
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import uuid
from contextlib import contextmanager
import fcntl
import tempfile
import shutil


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ProcessingPhase(Enum):
    """Processing phases for different operations."""
    INITIALIZATION = "initialization"
    EXTRACTION = "extraction"  
    VALIDATION = "validation"
    DESCRIPTION_GENERATION = "description_generation"
    POST_PROCESSING = "post_processing"
    FINALIZATION = "finalization"


@dataclass
class TaskInfo:
    """Information about a processing task."""
    task_id: str
    package_name: str
    phase: ProcessingPhase
    status: TaskStatus
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['phase'] = self.phase.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskInfo':
        """Create from dictionary."""
        data['phase'] = ProcessingPhase(data['phase'])
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


class ProgressDatabase:
    """SQLite-based progress tracking database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL,
                    end_time REAL,
                    total_tasks INTEGER,
                    completed_tasks INTEGER,
                    failed_tasks INTEGER,
                    configuration TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    package_name TEXT,
                    phase TEXT,
                    status TEXT,
                    input_file TEXT,
                    output_file TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    error_message TEXT,
                    retry_count INTEGER,
                    metadata TEXT,
                    checksum TEXT,
                    created_at REAL,
                    updated_at REAL,
                    FOREIGN KEY (session_id) REFERENCES processing_sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_session_status 
                ON tasks (session_id, status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_package_phase 
                ON tasks (package_name, phase)
            """)
            
            conn.commit()
    
    def create_session(self, session_id: str, total_tasks: int, 
                      configuration: Dict[str, Any]) -> str:
        """Create new processing session."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO processing_sessions 
                (session_id, start_time, total_tasks, completed_tasks, failed_tasks, 
                 configuration, metadata)
                VALUES (?, ?, ?, 0, 0, ?, ?)
            """, (session_id, time.time(), total_tasks, 
                  json.dumps(configuration), json.dumps({})))
            conn.commit()
        return session_id
    
    def update_session(self, session_id: str, completed: int, failed: int):
        """Update session progress."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE processing_sessions 
                SET completed_tasks = ?, failed_tasks = ?, end_time = ?
                WHERE session_id = ?
            """, (completed, failed, time.time(), session_id))
            conn.commit()
    
    def add_task(self, session_id: str, task: TaskInfo):
        """Add task to tracking."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            now = time.time()
            conn.execute("""
                INSERT OR REPLACE INTO tasks 
                (task_id, session_id, package_name, phase, status, input_file, 
                 output_file, start_time, end_time, duration, error_message, 
                 retry_count, metadata, checksum, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task.task_id, session_id, task.package_name, task.phase.value,
                  task.status.value, task.input_file, task.output_file,
                  task.start_time, task.end_time, task.duration,
                  task.error_message, task.retry_count, 
                  json.dumps(task.metadata), task.checksum, now, now))
            conn.commit()
    
    def update_task(self, task: TaskInfo):
        """Update existing task."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks 
                SET status = ?, end_time = ?, duration = ?, error_message = ?,
                    retry_count = ?, metadata = ?, checksum = ?, updated_at = ?
                WHERE task_id = ?
            """, (task.status.value, task.end_time, task.duration,
                  task.error_message, task.retry_count, json.dumps(task.metadata),
                  task.checksum, time.time(), task.task_id))
            conn.commit()
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task by ID."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM tasks WHERE task_id = ?
            """, (task_id,))
            row = cursor.fetchone()
            
            if row:
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'] or '{}')
                return TaskInfo.from_dict(data)
            return None
    
    def get_session_tasks(self, session_id: str, 
                         status: Optional[TaskStatus] = None) -> List[TaskInfo]:
        """Get all tasks for a session, optionally filtered by status."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status:
                cursor = conn.execute("""
                    SELECT * FROM tasks 
                    WHERE session_id = ? AND status = ?
                    ORDER BY created_at
                """, (session_id, status.value))
            else:
                cursor = conn.execute("""
                    SELECT * FROM tasks 
                    WHERE session_id = ?
                    ORDER BY created_at
                """, (session_id,))
            
            tasks = []
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'] or '{}')
                tasks.append(TaskInfo.from_dict(data))
            
            return tasks
    
    def get_resumable_tasks(self, session_id: str) -> List[TaskInfo]:
        """Get tasks that can be resumed (pending or failed with retries)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM tasks 
                WHERE session_id = ? 
                AND (status = 'pending' OR (status = 'failed' AND retry_count < 3))
                ORDER BY created_at
            """, (session_id,))
            
            tasks = []
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'] or '{}')
                tasks.append(TaskInfo.from_dict(data))
            
            return tasks
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM tasks 
                WHERE session_id = ? 
                GROUP BY status
            """, (session_id,))
            
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor = conn.execute("""
                SELECT phase, COUNT(*) as count 
                FROM tasks 
                WHERE session_id = ? 
                GROUP BY phase
            """, (session_id,))
            
            phase_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor = conn.execute("""
                SELECT 
                    MIN(start_time) as earliest_start,
                    MAX(end_time) as latest_end,
                    AVG(duration) as avg_duration,
                    SUM(CASE WHEN duration IS NOT NULL THEN duration ELSE 0 END) as total_duration
                FROM tasks 
                WHERE session_id = ?
            """, (session_id,))
            
            timing_stats = cursor.fetchone()
            
            return {
                "status_counts": status_counts,
                "phase_counts": phase_counts,
                "timing": {
                    "earliest_start": timing_stats[0],
                    "latest_end": timing_stats[1],
                    "average_duration": timing_stats[2],
                    "total_duration": timing_stats[3]
                }
            }


class FileCheckpoint:
    """File-based checkpoint system for additional reliability."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
    
    def save_checkpoint(self, session_id: str, data: Dict[str, Any]):
        """Save checkpoint data to file."""
        with self._lock:
            checkpoint_file = self.checkpoint_dir / f"{session_id}_checkpoint.json"
            temp_file = checkpoint_file.with_suffix('.tmp')
            
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump({
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "data": data
                }, f, indent=2)
            
            # Atomic move
            shutil.move(str(temp_file), str(checkpoint_file))
    
    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file."""
        with self._lock:
            checkpoint_file = self.checkpoint_dir / f"{session_id}_checkpoint.json"
            
            if not checkpoint_file.exists():
                return None
            
            try:
                with open(checkpoint_file) as f:
                    checkpoint = json.load(f)
                return checkpoint.get("data")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {checkpoint_file}: {e}")
                return None
    
    def list_sessions(self) -> List[str]:
        """List all available checkpoint sessions."""
        sessions = []
        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            session_id = checkpoint_file.stem.replace("_checkpoint", "")
            sessions.append(session_id)
        return sessions


class ProgressTracker:
    """Main progress tracking coordinator."""
    
    def __init__(self, 
                 progress_dir: Path,
                 session_id: Optional[str] = None,
                 auto_checkpoint_interval: int = 30):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(exist_ok=True)
        
        self.session_id = session_id or str(uuid.uuid4())
        self.db = ProgressDatabase(self.progress_dir / "progress.db")
        self.file_checkpoint = FileCheckpoint(self.progress_dir / "checkpoints")
        
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.last_checkpoint = time.time()
        
        self._active_tasks = {}  # task_id -> TaskInfo
        self._lock = threading.Lock()
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
    
    def start_session(self, total_tasks: int, configuration: Dict[str, Any]):
        """Start new processing session."""
        self.total_tasks = total_tasks
        self.start_time = time.time()
        
        self.db.create_session(self.session_id, total_tasks, configuration)
        
        # Save initial checkpoint
        self.file_checkpoint.save_checkpoint(self.session_id, {
            "session_id": self.session_id,
            "total_tasks": total_tasks,
            "configuration": configuration,
            "start_time": self.start_time
        })
    
    def resume_session(self, session_id: str) -> bool:
        """Resume existing session."""
        # Load from checkpoint
        checkpoint_data = self.file_checkpoint.load_checkpoint(session_id)
        if not checkpoint_data:
            return False
        
        self.session_id = session_id
        self.total_tasks = checkpoint_data.get("total_tasks", 0)
        self.start_time = checkpoint_data.get("start_time", time.time())
        
        # Load existing tasks
        all_tasks = self.db.get_session_tasks(session_id)
        for task in all_tasks:
            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks += 1
            elif task.status == TaskStatus.FAILED:
                self.failed_tasks += 1
        
        return True
    
    def create_task(self, 
                   package_name: str,
                   phase: ProcessingPhase,
                   input_file: Optional[str] = None,
                   output_file: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new task and return its ID."""
        task_id = f"{package_name}_{phase.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        task = TaskInfo(
            task_id=task_id,
            package_name=package_name,
            phase=phase,
            status=TaskStatus.PENDING,
            input_file=input_file,
            output_file=output_file,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._active_tasks[task_id] = task
            self.db.add_task(self.session_id, task)
        
        return task_id
    
    def start_task(self, task_id: str):
        """Mark task as started."""
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                task.status = TaskStatus.IN_PROGRESS
                task.start_time = time.time()
                self.db.update_task(task)
                self._maybe_checkpoint()
    
    def complete_task(self, task_id: str, 
                     output_file: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     checksum: Optional[str] = None):
        """Mark task as completed."""
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.end_time = time.time()
                task.duration = task.end_time - (task.start_time or task.end_time)
                
                if output_file:
                    task.output_file = output_file
                if metadata:
                    task.metadata.update(metadata)
                if checksum:
                    task.checksum = checksum
                
                self.db.update_task(task)
                self.completed_tasks += 1
                self._maybe_checkpoint()
    
    def fail_task(self, task_id: str, error_message: str, 
                 can_retry: bool = True):
        """Mark task as failed."""
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                task.status = TaskStatus.FAILED if not can_retry else TaskStatus.RETRYING
                task.end_time = time.time()
                task.duration = task.end_time - (task.start_time or task.end_time)
                task.error_message = error_message
                task.retry_count += 1
                
                self.db.update_task(task)
                
                if not can_retry or task.retry_count >= 3:
                    self.failed_tasks += 1
                
                self._maybe_checkpoint()
    
    def skip_task(self, task_id: str, reason: str):
        """Mark task as skipped."""
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                task.status = TaskStatus.SKIPPED
                task.end_time = time.time()
                task.error_message = f"Skipped: {reason}"
                
                self.db.update_task(task)
                self._maybe_checkpoint()
    
    def get_resumable_tasks(self) -> List[TaskInfo]:
        """Get tasks that can be resumed."""
        return self.db.get_resumable_tasks(self.session_id)
    
    def get_completed_packages(self) -> Set[str]:
        """Get set of packages that have been successfully completed."""
        completed_tasks = self.db.get_session_tasks(self.session_id, TaskStatus.COMPLETED)
        return {task.package_name for task in completed_tasks}
    
    def should_skip_package(self, package_name: str, phase: ProcessingPhase) -> bool:
        """Check if package should be skipped (already completed)."""
        completed_packages = self.get_completed_packages()
        return package_name in completed_packages
    
    def _maybe_checkpoint(self):
        """Save checkpoint if interval has elapsed."""
        if time.time() - self.last_checkpoint > self.auto_checkpoint_interval:
            self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save current state to checkpoint."""
        checkpoint_data = {
            "session_id": self.session_id,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "start_time": self.start_time,
            "last_update": time.time(),
            "progress_percent": (self.completed_tasks / max(1, self.total_tasks)) * 100
        }
        
        self.file_checkpoint.save_checkpoint(self.session_id, checkpoint_data)
        self.db.update_session(self.session_id, self.completed_tasks, self.failed_tasks)
        self.last_checkpoint = time.time()
    
    def force_checkpoint(self):
        """Force immediate checkpoint."""
        self._save_checkpoint()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        stats = self.db.get_session_stats(self.session_id)
        
        elapsed_time = time.time() - (self.start_time or time.time())
        progress_percent = (self.completed_tasks / max(1, self.total_tasks)) * 100
        
        # Estimate remaining time
        if self.completed_tasks > 0:
            avg_time_per_task = elapsed_time / self.completed_tasks
            remaining_tasks = self.total_tasks - self.completed_tasks - self.failed_tasks
            estimated_remaining = avg_time_per_task * remaining_tasks
        else:
            estimated_remaining = None
        
        return {
            "session_id": self.session_id,
            "progress": {
                "total_tasks": self.total_tasks,
                "completed": self.completed_tasks,
                "failed": self.failed_tasks,
                "remaining": self.total_tasks - self.completed_tasks - self.failed_tasks,
                "percent_complete": progress_percent
            },
            "timing": {
                "elapsed_seconds": elapsed_time,
                "estimated_remaining_seconds": estimated_remaining,
                "start_time": self.start_time
            },
            "detailed_stats": stats
        }
    
    @contextmanager
    def track_task(self, 
                   package_name: str,
                   phase: ProcessingPhase,
                   input_file: Optional[str] = None,
                   output_file: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking task execution."""
        task_id = self.create_task(package_name, phase, input_file, output_file, metadata)
        
        try:
            self.start_task(task_id)
            yield task_id
            self.complete_task(task_id, output_file, metadata)
        
        except Exception as e:
            # Determine if this error is recoverable
            can_retry = not isinstance(e, (MemoryError, SystemExit, KeyboardInterrupt))
            self.fail_task(task_id, str(e), can_retry)
            raise
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old checkpoint data."""
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        # Clean up old checkpoint files
        for checkpoint_file in self.file_checkpoint.checkpoint_dir.glob("*_checkpoint.json"):
            try:
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors
    
    def export_session_report(self, output_file: Path):
        """Export detailed session report."""
        summary = self.get_progress_summary()
        all_tasks = self.db.get_session_tasks(self.session_id)
        
        report = {
            "session_summary": summary,
            "tasks": [task.to_dict() for task in all_tasks],
            "export_time": time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)


# Utility functions for common progress tracking patterns
def create_progress_tracker(progress_dir: str, session_id: str = None) -> ProgressTracker:
    """Create and initialize progress tracker."""
    return ProgressTracker(Path(progress_dir), session_id)


def resume_or_create_session(progress_dir: str, 
                           session_id: str = None,
                           total_tasks: int = 0,
                           configuration: Dict[str, Any] = None) -> ProgressTracker:
    """Resume existing session or create new one."""
    tracker = ProgressTracker(Path(progress_dir), session_id)
    
    if session_id and tracker.resume_session(session_id):
        print(f"Resumed existing session: {session_id}")
    else:
        tracker.start_session(total_tasks, configuration or {})
        print(f"Started new session: {tracker.session_id}")
    
    return tracker