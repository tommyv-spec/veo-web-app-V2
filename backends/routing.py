# -*- coding: utf-8 -*-
"""
Job Routing Module for Veo Web App

This module provides helpers to integrate Flow backend with the main application:
- Automatic backend selection for new jobs
- Queue management for Flow jobs
- Status updates and monitoring

Usage in main.py or worker.py:
    from backends.routing import route_new_job, is_flow_job
"""

import json
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from backends.selector import BackendType, choose_backend_for_job
from models import Job, Clip, add_job_log


def route_new_job(
    db: Session,
    job: Job,
    api_keys: Optional[List[str]] = None
) -> BackendType:
    """
    Determine and set the backend for a new job.
    
    This should be called when creating a new job to set the backend field.
    
    Args:
        db: Database session
        job: The job being created
        api_keys: Optional list of API keys from the job request
        
    Returns:
        The selected backend type
    """
    # Determine backend based on available API keys
    backend = choose_backend_for_job(db, job.user_id, api_keys)
    
    # Set backend on job
    job.backend = backend.value
    db.commit()
    
    # Log the routing decision
    add_job_log(
        db, job.id,
        f"Job routed to {backend.value.upper()} backend",
        "INFO", "routing"
    )
    
    print(f"[Routing] Job {job.id[:8]} → {backend.value.upper()} backend", flush=True)
    
    return backend


def is_flow_job(job: Job) -> bool:
    """
    Check if a job uses the Flow backend.
    
    Args:
        job: Job to check
        
    Returns:
        True if Flow backend, False if API backend
    """
    return job.backend == BackendType.FLOW.value


def enqueue_if_flow(
    db: Session,
    job: Job,
    priority: int = 0
) -> bool:
    """
    Enqueue a job to the Flow queue if it uses Flow backend.
    
    NOTE: If flow_worker module is not available (local Flow worker mode),
    this function just returns True - the local worker polls the database directly.
    
    Args:
        db: Database session
        job: Job to potentially enqueue
        priority: Job priority (higher = sooner)
        
    Returns:
        True if job was enqueued or Flow backend is ready, False if API backend
    """
    if not is_flow_job(job):
        return False
    
    # Try to import flow_worker - if not available, job is already "queued" in database
    # The local Flow worker polls the database directly via HTTP API
    try:
        from flow_worker import enqueue_flow_job
        success = enqueue_flow_job(job.id, priority)
    except ImportError:
        # Local Flow worker mode - jobs are already available via /api/local-worker endpoints
        print(f"[Routing] flow_worker not available - using local worker mode for job {job.id[:8]}", flush=True)
        success = True
    
    if success:
        add_job_log(
            db, job.id,
            "Job queued for Flow processing",
            "INFO", "routing"
        )
    else:
        add_job_log(
            db, job.id,
            "Failed to queue job for Flow processing",
            "ERROR", "routing"
        )
    
    return success


def get_backend_for_user(db: Session, user_id: str) -> BackendType:
    """
    Get the backend that would be used for a user's jobs.
    
    Useful for UI hints (e.g., showing different messaging).
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        Expected backend type
    """
    return choose_backend_for_job(db, user_id)


def get_backend_stats() -> Dict[str, Any]:
    """
    Get statistics about backend usage.
    
    Returns:
        Dict with backend statistics
    """
    from models import get_db
    from backends.selector import get_backend_status
    
    stats = get_backend_status()
    
    # Add job counts by backend
    with get_db() as db:
        api_jobs = db.query(Job).filter(Job.backend == "api").count()
        flow_jobs = db.query(Job).filter(Job.backend == "flow").count()
        
        # Pending Flow jobs
        pending_flow = db.query(Job).filter(
            Job.backend == "flow",
            Job.status == "pending"
        ).count()
        
        stats["job_counts"] = {
            "api_total": api_jobs,
            "flow_total": flow_jobs,
            "flow_pending": pending_flow,
        }
    
    # Add queue status
    try:
        from flow_worker import get_queue_status
        stats["flow_queue"] = get_queue_status()
    except ImportError:
        # Local Flow worker mode - no server-side queue
        stats["flow_queue"] = {"mode": "local_worker", "status": "Jobs polled via HTTP API"}
    except Exception:
        stats["flow_queue"] = {"error": "Queue not available"}
    
    return stats


def handle_flow_auth_required(db: Session, job_id: str):
    """
    Handle a job that needs Flow authentication.
    
    This should be called when the Flow worker detects auth is needed.
    
    Args:
        db: Database session
        job_id: Job ID
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return
    
    job.flow_needs_auth = True
    job.status = "paused"
    db.commit()
    
    add_job_log(
        db, job_id,
        "Job paused: Flow authentication required. Please re-authenticate.",
        "WARNING", "flow"
    )
    
    # TODO: Send notification to admin


def resume_flow_jobs_after_auth(db: Session):
    """
    Resume paused Flow jobs after authentication is restored.
    
    Args:
        db: Database session
    """
    paused_jobs = db.query(Job).filter(
        Job.backend == "flow",
        Job.flow_needs_auth == True,
        Job.status == "paused"
    ).all()
    
    try:
        from flow_worker import enqueue_flow_job
        use_queue = True
    except ImportError:
        # Local Flow worker mode - just update status, worker will poll
        use_queue = False
    
    for job in paused_jobs:
        job.flow_needs_auth = False
        job.status = "pending"
        db.commit()
        
        if use_queue:
            enqueue_flow_job(job.id)
        
        add_job_log(
            db, job.id,
            "Job resumed after Flow authentication restored",
            "INFO", "flow"
        )
    
    print(f"[Routing] Resumed {len(paused_jobs)} paused Flow jobs", flush=True)


# === Export convenience functions ===

__all__ = [
    'route_new_job',
    'is_flow_job',
    'enqueue_if_flow',
    'get_backend_for_user',
    'get_backend_stats',
    'handle_flow_auth_required',
    'resume_flow_jobs_after_auth',
]
