# app/accounts/__init__.py
"""
ARVYAM Accounts Package — Phase 3.1

This package provides:
- User account management (soft accounts, opt-in memory)
- OTP-based authentication (proactive signup)
- Privacy endpoints (/forget-me, /export-data)
- Recipient profile management

Constitutional Rails:
- Guest-first: No signup wall anywhere
- Privacy-safe: No raw PII in logs, ≤90d retention
- Selection invariance: Memory affects order/copy only, NOT SKU IDs

Feature Flags (controlled in app/db.py):
- AUTH_ENDPOINTS_ENABLED: /auth/request-otp, /auth/verify-otp
- MEMORY_ENDPOINTS_ENABLED: /forget-me, /export-data, /profile

Submodules:
- models: Pydantic models for Users, Orders, RecipientProfiles, OtpCodes
- otp: OTP generation, sending, verification
- auth: JWT token management, user authentication
- auth_routes: FastAPI routes for /auth/*
- routes: FastAPI routes for /forget-me, /export-data, /profile
"""

from __future__ import annotations

# Explicit exports for clean imports
__all__ = [
    # Models
    "User",
    "Order",
    "RecipientProfile",
    "OtpCode",
    # Schemas
    "UserCreate",
    "UserResponse",
    "OrderCreate",
    "RecipientProfileCreate",
    "RecipientProfileResponse",
    "PreferencesSchema",
    "ExportDataResponse",
    # OTP
    "OTPService",
    "OTPManager",
    "generate_otp",
    "hash_otp",
    "verify_otp_hash",
    "get_otp_service",
    "get_otp_manager",
    # Auth
    "create_token",
    "verify_token",
    "create_user",
    "get_user_by_email",
    "get_user_by_id",
    "authenticate_or_create_user",
    "get_current_user_optional",
    "get_current_user_required",
    # Routes
    "auth_router",
    "privacy_router",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import pattern for clean module loading."""
    
    # Models
    if name in ("User", "Order", "RecipientProfile", "OtpCode",
                "UserCreate", "UserResponse", "OrderCreate",
                "RecipientProfileCreate", "RecipientProfileResponse",
                "PreferencesSchema", "ExportDataResponse"):
        from . import models
        return getattr(models, name)
    
    # OTP
    if name in ("OTPService", "OTPManager", "generate_otp", "hash_otp",
                "verify_otp_hash", "get_otp_service", "get_otp_manager"):
        from . import otp
        return getattr(otp, name)
    
    # Auth
    if name in ("create_token", "verify_token", "create_user",
                "get_user_by_email", "get_user_by_id",
                "authenticate_or_create_user",
                "get_current_user_optional", "get_current_user_required"):
        from . import auth
        return getattr(auth, name)
    
    # Routes
    if name == "auth_router":
        from .auth_routes import router
        return router
    
    if name == "privacy_router":
        from .routes import router
        return router
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
