# app/accounts/auth.py
"""
Authentication module for Phase 3.1.

This module provides:
- JWT token creation and verification (PyJWT)
- User creation and authentication after OTP verification
- Historical guest order linkage on opt-in
- FastAPI dependencies for authenticated routes

Infrastructure Decision (Locked):
- Authentication: Signed tokens via PyJWT
- After OTP verification, issue 24h token
- Simpler than full JWT refresh flow

Environment Variables Required:
- JWT_SECRET: Random 32+ character string (REQUIRED)
- JWT_EXPIRY_HOURS: Token expiry in hours (default: 24)

Token Payload:
- user_id: User UUID
- email: User email
- exp: Expiry timestamp
- iat: Issued at timestamp
"""

from __future__ import annotations

import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from functools import lru_cache

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.db import get_supabase_client, TABLE_USERS, TABLE_ORDERS

log = logging.getLogger("arvyam.auth")

# ============================================================
# Configuration
# ============================================================

def _get_jwt_secret() -> str:
    """Get JWT secret from environment."""
    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        log.error("JWT_SECRET not set - authentication will fail")
    elif len(secret) < 32:
        log.warning("JWT_SECRET should be at least 32 characters")
    return secret


def _get_jwt_expiry_hours() -> int:
    """Get JWT expiry in hours from environment."""
    try:
        return int(os.getenv("JWT_EXPIRY_HOURS", "24"))
    except ValueError:
        return 24


# ============================================================
# Token Creation
# ============================================================

def create_token(user_id: str, email: str) -> Tuple[str, datetime]:
    """
    Create signed JWT token for authenticated user.
    
    Args:
        user_id: User UUID string.
        email: User email address.
        
    Returns:
        Tuple of (token: str, expires_at: datetime).
        
    Raises:
        ValueError: If JWT_SECRET not configured.
    """
    import jwt
    
    secret = _get_jwt_secret()
    if not secret:
        raise ValueError("JWT_SECRET not configured")
    
    now = datetime.now(timezone.utc)
    expiry_hours = _get_jwt_expiry_hours()
    expires_at = now + timedelta(hours=expiry_hours)
    
    payload = {
        "user_id": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    
    token = jwt.encode(payload, secret, algorithm="HS256")
    
    log.info("Token created for user %s*** (expires in %dh)", user_id[:8], expiry_hours)
    return token, expires_at


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string.
        
    Returns:
        Decoded payload dict with user_id, email, iat, exp.
        None if token is invalid or expired.
    """
    import jwt
    
    secret = _get_jwt_secret()
    if not secret:
        log.error("Cannot verify token: JWT_SECRET not configured")
        return None
    
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        log.info("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        log.warning("Invalid token: %s", str(e)[:50])
        return None


# ============================================================
# User Operations
# ============================================================

def create_user(email: str, phone: Optional[str] = None, memory_opt_in: bool = False) -> Optional[dict]:
    """
    Create new user in database.
    
    Args:
        email: User email (will be lowercased).
        phone: Optional phone in E.164 format.
        memory_opt_in: Whether user opts into memory (default: false).
        
    Returns:
        Created user dict with id, email, memory_opt_in, created_at.
        None if creation fails.
    """
    client = get_supabase_client()
    if not client:
        log.error("Database unavailable for user creation")
        return None
    
    email = email.strip().lower()
    
    try:
        # Check if user already exists
        existing = client.table(TABLE_USERS)\
            .select("id")\
            .ilike("email", email)\
            .limit(1)\
            .execute()
        
        if existing.data:
            log.info("User already exists: %s***", email[:3])
            return None
        
        # Create user
        user_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        result = client.table(TABLE_USERS).insert({
            "id": user_id,
            "email": email,
            "phone": phone,
            "memory_opt_in": memory_opt_in,
            "created_at": now,
        }).execute()
        
        if result.data:
            log.info("User created: %s*** (memory_opt_in=%s)", email[:3], memory_opt_in)
            return result.data[0]
        return None
        
    except Exception as e:
        log.error("User creation failed: %s", str(e)[:100])
        return None


def get_user_by_email(email: str) -> Optional[dict]:
    """
    Get user by email address.
    
    Args:
        email: User email (case-insensitive).
        
    Returns:
        User dict or None if not found.
    """
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        result = client.table(TABLE_USERS)\
            .select("*")\
            .ilike("email", email.strip().lower())\
            .limit(1)\
            .execute()
        
        if result.data:
            return result.data[0]
        return None
        
    except Exception as e:
        log.error("User lookup failed: %s", str(e)[:100])
        return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    """
    Get user by UUID.
    
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
        
        if result.data:
            return result.data[0]
        return None
        
    except Exception as e:
        log.error("User lookup by ID failed: %s", str(e)[:100])
        return None


def update_user_memory_opt_in(user_id: str, memory_opt_in: bool) -> bool:
    """
    Update user's memory opt-in status.
    
    Args:
        user_id: User UUID string.
        memory_opt_in: New opt-in status.
        
    Returns:
        True if updated successfully.
    """
    client = get_supabase_client()
    if not client:
        return False
    
    try:
        result = client.table(TABLE_USERS)\
            .update({"memory_opt_in": memory_opt_in})\
            .eq("id", user_id)\
            .execute()
        
        if result.data:
            log.info("User %s*** memory_opt_in updated to %s", user_id[:8], memory_opt_in)
            return True
        return False
        
    except Exception as e:
        log.error("User update failed: %s", str(e)[:100])
        return False


# ============================================================
# Historical Guest Order Linkage
# ============================================================

def link_historical_orders(user_id: str, email: str) -> int:
    """
    Link historical guest orders to user account.
    
    On opt-in, links orders from last 90 days where:
    - email matches
    - user_id is NULL (guest orders)
    
    This is idempotent — safe to call multiple times.
    
    Args:
        user_id: User UUID string.
        email: User email address.
        
    Returns:
        Number of orders linked.
        
    Note on Supabase API:
        We use `.is_("user_id", "null")` to filter for NULL values.
        This is the correct supabase-py syntax for IS NULL checks.
        The string "null" is interpreted by the PostgREST API as SQL NULL.
    """
    client = get_supabase_client()
    if not client:
        return 0
    
    email = email.strip().lower()
    cutoff = (datetime.utcnow() - timedelta(days=90)).isoformat()
    
    try:
        # Find guest orders from last 90 days with matching email
        # NIT: Using .is_("user_id", "null") - this is supabase-py's syntax for IS NULL
        # The string "null" is converted to SQL NULL by PostgREST
        result = client.table(TABLE_ORDERS)\
            .update({"user_id": user_id})\
            .ilike("email", email)\
            .is_("user_id", "null")\
            .gte("created_at", cutoff)\
            .execute()
        
        linked = len(result.data) if result.data else 0
        
        if linked > 0:
            log.info("Linked %d historical orders to user %s***", linked, user_id[:8])
        
        return linked
        
    except Exception as e:
        log.error("Historical order linkage failed: %s", str(e)[:100])
        return 0


# ============================================================
# Authentication After OTP Verification
# ============================================================

def authenticate_or_create_user(email: str) -> Tuple[bool, str, Optional[dict]]:
    """
    Authenticate existing user or create new user after OTP verification.
    
    This is called after successful OTP verification to:
    1. Find or create user
    2. Issue JWT token
    3. Link historical guest orders (for new users)
    
    Args:
        email: Verified email address.
        
    Returns:
        Tuple of (success: bool, message: str, response_data: Optional[dict]).
        response_data contains: token, expires_at, is_new_user, user_id
    """
    email = email.strip().lower()
    
    # Check if user exists
    user = get_user_by_email(email)
    is_new_user = False
    
    if user:
        user_id = user["id"]
        log.info("Existing user authenticated: %s***", email[:3])
    else:
        # Create new user (memory_opt_in defaults to false)
        created = create_user(email, memory_opt_in=False)
        if not created:
            return False, "Failed to create account", None
        
        user_id = created["id"]
        is_new_user = True
        
        # Link historical guest orders for new users
        linked = link_historical_orders(user_id, email)
        if linked > 0:
            log.info("Linked %d historical orders for new user %s***", linked, email[:3])
    
    # Create token
    try:
        token, expires_at = create_token(user_id, email)
    except ValueError as e:
        log.error("Token creation failed: %s", str(e))
        return False, "Authentication configuration error", None
    
    return True, "Authentication successful", {
        "token": token,
        "expires_at": expires_at.isoformat(),
        "is_new_user": is_new_user,
        "user_id": user_id,
    }


# ============================================================
# FastAPI Dependencies
# ============================================================

# HTTP Bearer token extractor
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[dict]:
    """
    Get current user from JWT token (optional).
    
    Use this dependency when authentication is optional.
    Returns None if no valid token provided.
    
    Usage:
        @router.get("/endpoint")
        def endpoint(user: Optional[dict] = Depends(get_current_user_optional)):
            if user:
                # Authenticated
            else:
                # Guest
    """
    if not credentials:
        return None
    
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    
    # Optionally verify user still exists in database
    user_id = payload.get("user_id")
    if not user_id:
        return None
    
    user = get_user_by_id(user_id)
    return user


async def get_current_user_required(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """
    Get current user from JWT token (required).
    
    Use this dependency when authentication is required.
    Raises HTTPException 401 if not authenticated.
    
    Usage:
        @router.get("/protected")
        def protected(user: dict = Depends(get_current_user_required)):
            # Guaranteed to have valid user
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "AUTH_REQUIRED", "message": "Authentication required"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "TOKEN_INVALID", "message": "Invalid or expired token"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "TOKEN_MALFORMED", "message": "Token missing user_id"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "USER_NOT_FOUND", "message": "User account not found"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


# ============================================================
# Convenience function to extract user_id from request
# ============================================================

def get_user_id_from_token(request: Request) -> Optional[str]:
    """
    Extract user_id from Authorization header without raising exceptions.
    
    Useful for optional user context in non-protected routes.
    
    Args:
        request: FastAPI Request object.
        
    Returns:
        User ID string or None.
    """
    auth_header = request.headers.get("Authorization", "")
    
    if not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    payload = verify_token(token)
    
    if payload:
        return payload.get("user_id")
    return None
