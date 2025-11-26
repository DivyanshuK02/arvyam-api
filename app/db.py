# app/db.py
"""
Supabase client setup for Phase 3.1.

This module provides:
- Singleton Supabase client instance
- Feature flag checks for memory/auth endpoints
- Connection helpers with error handling

Infrastructure Decision (Locked):
- Database Client: supabase-py directly (no SQLAlchemy/ORM)
- Rationale: Simpler for MVP, dashboard debugging, no abstraction overhead

Environment Variables Required:
- SUPABASE_URL: Project URL (https://xxx.supabase.co)
- SUPABASE_KEY: Service role key or anon key

Feature Flags (all OFF by default):
- MEMORY_CONTEXT_ENABLED: Enable memory context building
- MEMORY_RERANK_ENABLED: Enable post-selection reranking
- MEMORY_ENDPOINTS_ENABLED: Enable /forget-me, /export-data
- AUTH_ENDPOINTS_ENABLED: Enable /auth/request-otp, /auth/verify-otp
"""

from __future__ import annotations

import os
import logging
from typing import Optional
from functools import lru_cache

log = logging.getLogger("arvyam.db")

# ============================================================
# Feature Flags (Phase 3.1)
# ============================================================

def _flag_on(name: str) -> bool:
    """Check if a feature flag is enabled. Default: off."""
    val = os.getenv(name, "off").lower()
    return val in ("on", "true", "1", "yes")


def is_memory_context_enabled() -> bool:
    """Check if memory context building is enabled."""
    return _flag_on("MEMORY_CONTEXT_ENABLED")


def is_memory_rerank_enabled() -> bool:
    """Check if post-selection reranking is enabled."""
    return _flag_on("MEMORY_RERANK_ENABLED")


def is_memory_endpoints_enabled() -> bool:
    """Check if privacy endpoints (/forget-me, /export-data) are enabled."""
    return _flag_on("MEMORY_ENDPOINTS_ENABLED")


def is_auth_endpoints_enabled() -> bool:
    """Check if auth endpoints (/auth/request-otp, /auth/verify-otp) are enabled."""
    return _flag_on("AUTH_ENDPOINTS_ENABLED")


# ============================================================
# Supabase Client
# ============================================================

# Lazy import to avoid startup errors if supabase not installed
_supabase_client = None


def _get_supabase_url() -> str:
    """Get Supabase URL from environment."""
    url = os.getenv("SUPABASE_URL", "").strip()
    if not url:
        log.warning("SUPABASE_URL not set - database operations will fail")
    return url


def _get_supabase_key() -> str:
    """Get Supabase key from environment."""
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not key:
        log.warning("SUPABASE_KEY not set - database operations will fail")
    return key


@lru_cache(maxsize=1)
def get_supabase_client():
    """
    Get singleton Supabase client instance.
    
    Returns:
        Supabase client or None if not configured.
        
    Note:
        Uses lru_cache for singleton pattern.
        Returns None if SUPABASE_URL or SUPABASE_KEY not set.
    """
    global _supabase_client
    
    url = _get_supabase_url()
    key = _get_supabase_key()
    
    if not url or not key:
        log.error("Supabase credentials not configured")
        return None
    
    try:
        from supabase import create_client, Client
        _supabase_client = create_client(url, key)
        log.info("Supabase client initialized successfully")
        return _supabase_client
    except ImportError:
        log.error("supabase-py not installed. Run: pip install supabase")
        return None
    except Exception as e:
        log.error(f"Failed to initialize Supabase client: {e}")
        return None


def get_db():
    """
    Dependency injection helper for FastAPI routes.
    
    Usage:
        @router.post("/example")
        def example(db = Depends(get_db)):
            if db is None:
                raise HTTPException(503, "Database unavailable")
            ...
    
    Returns:
        Supabase client instance or None.
    """
    return get_supabase_client()


# ============================================================
# Health Check
# ============================================================

def check_db_health() -> dict:
    """
    Check database connectivity for /health endpoint.
    
    Returns:
        Dict with status and optional error message.
    """
    client = get_supabase_client()
    
    if client is None:
        return {
            "database": "unavailable",
            "error": "Supabase client not initialized"
        }
    
    try:
        # Simple connectivity check - query users table (will be empty initially)
        result = client.table("users").select("id").limit(1).execute()
        return {
            "database": "ok",
            "tables_accessible": True
        }
    except Exception as e:
        error_msg = str(e)[:100]  # Truncate for safety
        log.warning(f"Database health check failed: {error_msg}")
        return {
            "database": "degraded",
            "error": error_msg
        }


# ============================================================
# Table Names (Constants)
# ============================================================

# Table names as constants for type safety and refactoring
TABLE_USERS = "users"
TABLE_ORDERS = "orders"
TABLE_RECIPIENT_PROFILES = "recipient_profiles"
TABLE_OTP_CODES = "otp_codes"


# ============================================================
# Query Helpers (No Raw String Interpolation)
# ============================================================

def query_user_by_email(email: str) -> Optional[dict]:
    """
    Query user by email (case-insensitive).
    
    Args:
        email: User email address.
        
    Returns:
        User dict or None if not found.
    """
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        result = client.table(TABLE_USERS)\
            .select("*")\
            .ilike("email", email)\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        log.error(f"Failed to query user by email: {e}")
        return None


def query_user_by_id(user_id: str) -> Optional[dict]:
    """
    Query user by UUID.
    
    Args:
        user_id: User UUID string.
        
    Returns:
        User dict or None if not found.
    """
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        result = client.table(TABLE_USERS)\
            .select("*")\
            .eq("id", user_id)\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        log.error(f"Failed to query user by id: {e}")
        return None


# ============================================================
# Module Initialization
# ============================================================

def init_db() -> bool:
    """
    Initialize database connection on app startup.
    
    Call this from main.py startup event.
    
    Returns:
        True if connection successful, False otherwise.
    """
    client = get_supabase_client()
    if client is None:
        log.warning("Database initialization skipped - credentials not configured")
        return False
    
    health = check_db_health()
    if health.get("database") == "ok":
        log.info("Database connection verified")
        return True
    else:
        log.warning(f"Database health check: {health}")
        return False
