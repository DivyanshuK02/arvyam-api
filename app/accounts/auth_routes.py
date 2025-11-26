# app/accounts/auth_routes.py
"""
Authentication routes for Phase 3.1.

Endpoints:
- POST /auth/request-otp — Request OTP for email
- POST /auth/verify-otp — Verify OTP and get token

Why Proactive Signup:
Premium users may WANT accounts before ordering to save recipient preferences
and track future orders. Guest-first is preserved — signup is completely optional.

Feature Flag:
- AUTH_ENDPOINTS_ENABLED: Must be "on" for routes to function

Rate Limits:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Endpoint            │ IP Limit      │ Per-Email Limit                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ /auth/request-otp   │ 5/min (route) │ 60s cooldown + 20/day (OTPManager)    │
│ /auth/verify-otp    │ 10/min (route)│ 5 attempts/OTP (OTPManager)           │
└─────────────────────────────────────────────────────────────────────────────┘

Security:
- Anti-enumeration: Same response whether email exists or not
- OTP hashed before storage (SHA-256)
- Constant-time hash comparison
- No raw PII in logs
"""

from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr, field_validator

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.db import is_auth_endpoints_enabled
from app.accounts.otp import request_otp as otp_request, verify_otp as otp_verify
from app.accounts.auth import authenticate_or_create_user

log = logging.getLogger("arvyam.auth_routes")

# ============================================================
# Router Setup
# ============================================================

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Rate limiter (uses same limiter as main app)
# NOTE: This provides IP-based rate limiting at the route level.
# Per-email limits are enforced in OTPManager (otp.py).
limiter = Limiter(key_func=get_remote_address)


# ============================================================
# Request/Response Schemas
# ============================================================

class RequestOTPInput(BaseModel):
    """Input schema for /auth/request-otp."""
    
    email: EmailStr = Field(..., description="Email address to send OTP to")
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


class VerifyOTPInput(BaseModel):
    """Input schema for /auth/verify-otp."""
    
    email: EmailStr = Field(..., description="Email address")
    otp: str = Field(..., description="6-digit verification code", min_length=6, max_length=6)
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v
    
    @field_validator("otp", mode="before")
    @classmethod
    def normalize_otp(cls, v):
        if isinstance(v, str):
            # Strip whitespace and validate format
            cleaned = v.strip()
            if not cleaned.isdigit():
                raise ValueError("OTP must contain only digits")
            return cleaned
        return v


class RequestOTPResponse(BaseModel):
    """Response schema for /auth/request-otp."""
    
    success: bool = Field(..., description="Whether request was processed")
    message: str = Field(..., description="User-safe message (anti-enumeration)")


class VerifyOTPResponse(BaseModel):
    """Response schema for /auth/verify-otp."""
    
    token: str = Field(..., description="JWT token for authenticated requests")
    expires_at: str = Field(..., description="Token expiry timestamp (ISO 8601)")
    is_new_user: bool = Field(..., description="Whether this is a newly created account")
    user_id: str = Field(..., description="User UUID")


class ErrorResponse(BaseModel):
    """Standard error response (Phase 1.7 format)."""
    
    error: dict = Field(..., description="Error details with code and message")
    persona: str = Field(default="ARVY", description="API persona")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")


# ============================================================
# Feature Flag Guard
# ============================================================

def check_auth_enabled():
    """
    Dependency to check if auth endpoints are enabled.
    
    Raises HTTPException 503 if AUTH_ENDPOINTS_ENABLED != "on".
    """
    if not is_auth_endpoints_enabled():
        log.info("Auth endpoints disabled (AUTH_ENDPOINTS_ENABLED=off)")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "AUTH_DISABLED",
                    "message": "Authentication endpoints are currently disabled"
                },
                "persona": "ARVY"
            }
        )


# ============================================================
# Endpoints
# ============================================================

@router.post(
    "/request-otp",
    response_model=RequestOTPResponse,
    responses={
        200: {"description": "OTP request processed (anti-enumeration response)"},
        400: {"description": "Invalid email format"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Auth endpoints disabled"},
    },
    summary="Request OTP",
    description="""
    Request a verification code for the given email address.
    
    **Anti-Enumeration**: Response is always the same whether the email exists or not.
    
    **Rate Limits**:
    - IP-based: 5 requests per minute (enforced at route level)
    - Per-email cooldown: 60 seconds between requests
    - Per-email daily cap: 20 OTPs per 24 hours
    
    **OTP Details**: 6-digit code, expires in 10 minutes, single-use.
    """,
)
@limiter.limit("5/minute")  # SHOULD-FIX verified: IP-based rate limit present
async def request_otp_endpoint(
    body: RequestOTPInput,
    request: Request,
    _: None = Depends(check_auth_enabled),
) -> RequestOTPResponse:
    """
    Request OTP for email authentication.
    
    Send a 6-digit verification code to the provided email address.
    The code expires in 10 minutes and can only be used once.
    
    Rate Limits Enforced:
    - This route: 5/min per IP (via slowapi decorator)
    - OTPManager: 60s cooldown per email
    - OTPManager: 20/day cap per email (MUST-FIX implemented)
    """
    email = body.email
    
    # Mask email for logging
    masked_email = email[:3] + "***" if len(email) > 3 else "***"
    log.info("OTP request for %s", masked_email)
    
    # Request OTP (per-email limits enforced in OTPManager)
    success, message = otp_request(email)
    
    # Anti-enumeration: Always return success=True for valid email format
    # The message is deliberately vague about whether email exists
    # Even if daily cap is hit, we return the same response
    return RequestOTPResponse(
        success=True,  # Always true for valid format (anti-enumeration)
        message="If this email is valid, a verification code was sent"
    )


@router.post(
    "/verify-otp",
    response_model=VerifyOTPResponse,
    responses={
        200: {"description": "OTP verified, token issued"},
        400: {"description": "Invalid OTP format or verification failed"},
        429: {"description": "Too many attempts"},
        503: {"description": "Auth endpoints disabled"},
    },
    summary="Verify OTP",
    description="""
    Verify the OTP and receive a JWT token for authenticated requests.
    
    **New Users**: If the email doesn't have an account, one is created automatically
    with `memory_opt_in=false` (guest-first philosophy).
    
    **Existing Users**: Simply authenticates and returns a token.
    
    **Historical Orders**: For new users, guest orders from the last 90 days with
    the same email are automatically linked to the new account.
    
    **Token Usage**: Include in Authorization header as `Bearer <token>`.
    
    **Rate Limits**:
    - IP-based: 10 requests per minute
    - Per-OTP: 5 verification attempts maximum
    """,
)
@limiter.limit("10/minute")  # IP-based rate limit
async def verify_otp_endpoint(
    body: VerifyOTPInput,
    request: Request,
    _: None = Depends(check_auth_enabled),
) -> VerifyOTPResponse:
    """
    Verify OTP and issue authentication token.
    
    After successful verification:
    1. Creates user if doesn't exist (memory_opt_in=false by default)
    2. Links historical guest orders (90-day lookback)
    3. Issues JWT token (24h expiry)
    
    Rate Limits Enforced:
    - This route: 10/min per IP (via slowapi decorator)
    - OTPManager: 5 attempts per OTP code
    """
    email = body.email
    otp = body.otp
    
    # Mask email for logging
    masked_email = email[:3] + "***" if len(email) > 3 else "***"
    
    # Verify OTP
    success, message, data = otp_verify(email, otp)
    
    if not success:
        log.info("OTP verification failed for %s: %s", masked_email, message)
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "OTP_INVALID",
                    "message": message
                },
                "persona": "ARVY"
            }
        )
    
    # Authenticate or create user
    auth_success, auth_message, auth_data = authenticate_or_create_user(email)
    
    if not auth_success:
        log.error("Authentication failed for %s: %s", masked_email, auth_message)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "AUTH_FAILED",
                    "message": "Authentication failed. Please try again."
                },
                "persona": "ARVY"
            }
        )
    
    log.info(
        "User authenticated: %s (is_new=%s, user_id=%s***)",
        masked_email,
        auth_data["is_new_user"],
        auth_data["user_id"][:8]
    )
    
    return VerifyOTPResponse(
        token=auth_data["token"],
        expires_at=auth_data["expires_at"],
        is_new_user=auth_data["is_new_user"],
        user_id=auth_data["user_id"],
    )


# ============================================================
# Health Check (for internal monitoring)
# ============================================================

@router.get(
    "/status",
    response_model=dict,
    include_in_schema=False,  # Internal endpoint
)
async def auth_status():
    """
    Internal endpoint to check auth system status.
    
    Not included in OpenAPI schema.
    """
    return {
        "auth_enabled": is_auth_endpoints_enabled(),
        "timestamp": datetime.utcnow().isoformat(),
    }
