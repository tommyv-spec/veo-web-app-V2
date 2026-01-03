# -*- coding: utf-8 -*-
"""
Backend Selector for Veo Web App

Determines which backend to use for a job:
- API: User has valid Gemini API keys
- FLOW: User has no API keys (uses browser automation)

This is internal routing logic - NOT a user-facing setting.
"""

from enum import Enum
from typing import Optional, List
from sqlalchemy.orm import Session


class BackendType(str, Enum):
    """Available backends for video generation"""
    API = "api"      # Direct API calls (requires user API keys)
    FLOW = "flow"    # Browser automation via Google Flow UI


def has_valid_api_keys(
    db: Session,
    user_id: str,
    job_api_keys: Optional[List[str]] = None
) -> bool:
    """
    Check if user has at least one valid API key.
    NOTE: Only checks USER's keys, not environment/server keys.
    Server keys are NOT used for routing decision.
    
    Args:
        db: Database session
        user_id: User ID to check
        job_api_keys: Optional list of API keys provided with the job (from UI)
        
    Returns:
        True if user has valid keys, False otherwise
    """
    # Check 1: Job-provided keys from UI (takes priority)
    if job_api_keys and len(job_api_keys) > 0:
        valid_keys = [k for k in job_api_keys if k and k.strip() and not k.startswith("your-")]
        if valid_keys:
            return True
    
    # Check 2: User's stored API keys in database
    if user_id:
        from models import UserAPIKey
        
        user_keys = db.query(UserAPIKey).filter(
            UserAPIKey.user_id == user_id,
            UserAPIKey.is_active == True,
            UserAPIKey.is_valid == True
        ).all()
        
        if user_keys and len(user_keys) > 0:
            return True
    
    # NOTE: Do NOT check environment/server keys here!
    # Server keys should NOT influence routing decision.
    # Users without their own keys should go to FLOW backend.
    
    return False


def choose_backend_for_job(
    db: Session,
    user_id: Optional[str] = None,
    job_api_keys: Optional[List[str]] = None
) -> BackendType:
    """
    Choose the appropriate backend for a job.
    
    Logic:
    - If user has valid API keys (personal or env) → API backend
    - Otherwise → Flow backend (browser automation)
    
    Args:
        db: Database session
        user_id: User ID (optional)
        job_api_keys: API keys provided with the job (optional)
        
    Returns:
        BackendType.API or BackendType.FLOW
    """
    if has_valid_api_keys(db, user_id, job_api_keys):
        return BackendType.API
    
    return BackendType.FLOW


def choose_backend_for_user(db: Session, user_id: str) -> BackendType:
    """
    Choose the appropriate backend for a user (used for UI hints).
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        BackendType.API or BackendType.FLOW
    """
    return choose_backend_for_job(db, user_id)


def is_flow_enabled() -> bool:
    """
    Check if Flow backend is available/enabled.
    
    Returns:
        True if Flow backend can accept jobs
    """
    import os
    
    # Check if Flow worker is configured
    flow_enabled = os.environ.get("FLOW_BACKEND_ENABLED", "true").lower() == "true"
    
    # Check if we have auth state configured
    flow_auth_url = os.environ.get("FLOW_STORAGE_STATE_URL", "")
    
    # Flow is enabled if explicitly enabled AND we have auth state
    # OR if we have Redis queue configured (worker will handle auth)
    redis_url = os.environ.get("KEYVALUE_URL", "") or os.environ.get("REDIS_URL", "")
    
    return flow_enabled and (flow_auth_url or redis_url)


def get_backend_status() -> dict:
    """
    Get status of all backends (for admin/debug).
    
    Returns:
        Dictionary with backend availability status
    """
    import os
    from config import api_keys_config, key_pool
    
    # API backend status
    api_status = {
        "enabled": True,
        "available_keys": 0,
        "total_keys": 0,
    }
    
    if api_keys_config:
        api_status["total_keys"] = len(api_keys_config.gemini_api_keys)
        pool_status = key_pool.get_pool_status_summary(api_keys_config)
        api_status["available_keys"] = pool_status.get("available", 0)
    
    # Flow backend status
    flow_status = {
        "enabled": is_flow_enabled(),
        "auth_configured": bool(os.environ.get("FLOW_STORAGE_STATE_URL")),
        "queue_configured": bool(os.environ.get("KEYVALUE_URL") or os.environ.get("REDIS_URL")),
    }
    
    return {
        "api": api_status,
        "flow": flow_status,
    }
