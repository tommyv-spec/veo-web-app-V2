# -*- coding: utf-8 -*-
"""
Backend routing module for Veo Web App

Provides automatic routing between:
- API backend (users with their own API keys)
- Flow backend (browser automation for users without keys)
"""

from .selector import BackendType, choose_backend_for_job, choose_backend_for_user
from .storage import ObjectStorage, get_storage, is_storage_configured
from .routing import (
    route_new_job,
    is_flow_job,
    enqueue_if_flow,
    get_backend_for_user,
    get_backend_stats,
)

__all__ = [
    # Types
    'BackendType',
    
    # Selection
    'choose_backend_for_job',
    'choose_backend_for_user',
    
    # Storage
    'ObjectStorage',
    'get_storage',
    'is_storage_configured',
    
    # Routing helpers
    'route_new_job',
    'is_flow_job',
    'enqueue_if_flow',
    'get_backend_for_user',
    'get_backend_stats',
]
