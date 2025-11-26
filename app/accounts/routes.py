# app/accounts/routes.py
"""
Privacy and Profile routes for Phase 3.1.

Endpoints:
- POST /forget-me — OTP-verified cascade delete (DSAR compliance)
- POST /export-data — OTP-verified data export (DSAR compliance)
- GET /profile — List recipient profiles (auth required)
- POST /profile — Create/update recipient profile (auth required)

Feature Flags:
- MEMORY_ENDPOINTS_ENABLED: Must be "on" for /forget-me, /export-data, /profile

Rate Limits:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Endpoint        │ IP Limit      │ Per-Email Limit                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ /forget-me      │ 5/min (route) │ OTP verification (same limits as auth)    │
│ /export-data    │ 5/min (route) │ OTP verification (same limits as auth)    │
│ /profile        │ 10/min (route)│ JWT auth required                         │
└─────────────────────────────────────────────────────────────────────────────┘

Security:
- Anti-enumeration: Same response whether email exists or not
- OTP verification required for /forget-me and /export-data
- JWT auth required for /profile
- No raw PII in logs

DSAR Compliance:
- /forget-me: Complete within 30 days (per Phase 2.7)
- /export-data: Complete within 30 days (per Phase 2.7)
- Response headers: Cache-Control: no-store
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr, field_validator

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.db import (
    get_supabase_client,
    is_memory_endpoints_enabled,
    TABLE_USERS,
    TABLE_ORDERS,
    TABLE_RECIPIENT_PROFILES,
)
from app.accounts.otp import verify_otp as otp_verify
from app.accounts.auth import get_current_user_required
from app.accounts.models import (
    RecipientProfileCreate,
    RecipientProfileResponse,
    RecipientProfile,
    PreferencesSchema,
    ExportDataResponse,
)
from app.privacy_utils import mask_email, hash_user_id

log = logging.getLogger("arvyam.privacy_routes")

# ============================================================
# Router Setup
# ============================================================

router = APIRouter(tags=["Privacy & Profiles"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ============================================================
# Request/Response Schemas
# ============================================================

class ForgetMeInput(BaseModel):
    """Input schema for /forget-me."""
    
    email: EmailStr = Field(..., description="Email address to delete")
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
            cleaned = v.strip()
            if not cleaned.isdigit():
                raise ValueError("OTP must contain only digits")
            return cleaned
        return v


class ForgetMeResponse(BaseModel):
    """Response schema for /forget-me."""
    
    success: bool = Field(..., description="Whether deletion was successful")
    deleted_at: str = Field(..., description="ISO timestamp of deletion")
    message: str = Field(default="Your data has been deleted", description="User message")


class ExportDataInput(BaseModel):
    """Input schema for /export-data."""
    
    email: EmailStr = Field(..., description="Email address to export")
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
            cleaned = v.strip()
            if not cleaned.isdigit():
                raise ValueError("OTP must contain only digits")
            return cleaned
        return v


class ProfileInput(BaseModel):
    """Input schema for POST /profile."""
    
    name: str = Field(..., description="Recipient name", min_length=1, max_length=50)
    preferences: Optional[PreferencesSchema] = Field(
        default=None,
        description="Recipient preferences (flowers, tier, palette_pref)"
    )
    
    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class ProfileUpdateInput(BaseModel):
    """Input schema for updating a profile."""
    
    profile_id: str = Field(..., description="Profile UUID to update")
    name: Optional[str] = Field(None, description="New recipient name", min_length=1, max_length=50)
    preferences: Optional[PreferencesSchema] = Field(
        None,
        description="New preferences (replaces existing)"
    )


# ============================================================
# Feature Flag Guard
# ============================================================

def check_memory_endpoints_enabled():
    """
    Dependency to check if memory/privacy endpoints are enabled.
    
    Raises HTTPException 503 if MEMORY_ENDPOINTS_ENABLED != "on".
    
    Note: In production, this flag should normally stay ON for DSAR compliance.
    Only disable for emergency maintenance with manual DSAR fallback.
    """
    if not is_memory_endpoints_enabled():
        log.info("Memory endpoints disabled (MEMORY_ENDPOINTS_ENABLED=off)")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "ENDPOINTS_DISABLED",
                    "message": "Privacy endpoints are currently disabled. Please contact support."
                },
                "persona": "ARVY"
            }
        )


# ============================================================
# Privacy Endpoints (OTP-verified)
# ============================================================

@router.post(
    "/forget-me",
    response_model=ForgetMeResponse,
    responses={
        200: {"description": "Data deleted successfully"},
        400: {"description": "Invalid OTP or verification failed"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Endpoints disabled"},
    },
    summary="Delete User Data (DSAR)",
    description="""
    Delete all user data after OTP verification.
    
    **Cascade Delete**: Removes Users + Orders + RecipientProfiles for the verified email.
    
    **Security**: Requires valid OTP (request via /auth/request-otp first).
    
    **Anti-Enumeration**: Response format is consistent whether email exists or not.
    
    **DSAR Compliance**: Request processed immediately (within 30-day SLA).
    """,
)
@limiter.limit("5/minute")
async def forget_me(
    body: ForgetMeInput,
    request: Request,
    response: Response,
    _: None = Depends(check_memory_endpoints_enabled),
) -> ForgetMeResponse:
    """
    Delete all user data after OTP verification.
    
    Cascade delete order:
    1. RecipientProfiles (FK to Users)
    2. Orders (FK to Users, SET NULL on delete)
    3. Users (primary)
    """
    email = body.email
    otp = body.otp
    
    # Mask email for logging
    masked_email = mask_email(email)
    log.info("Forget-me request for %s", masked_email)
    
    # Verify OTP first
    otp_success, otp_message, _ = otp_verify(email, otp)
    
    if not otp_success:
        log.info("Forget-me OTP failed for %s: %s", masked_email, otp_message)
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "OTP_INVALID",
                    "message": otp_message
                },
                "persona": "ARVY"
            }
        )
    
    # OTP verified - proceed with deletion
    client = get_supabase_client()
    if not client:
        log.error("Database unavailable for forget-me")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Service temporarily unavailable"
                },
                "persona": "ARVY"
            }
        )
    
    try:
        # Find user by email
        user_result = client.table(TABLE_USERS)\
            .select("id")\
            .ilike("email", email)\
            .limit(1)\
            .execute()
        
        deleted_at = datetime.utcnow().isoformat()
        
        if not user_result.data:
            # Anti-enumeration: Return success even if user doesn't exist
            log.info("Forget-me: no user found for %s (anti-enum success)", masked_email)
            return ForgetMeResponse(
                success=True,
                deleted_at=deleted_at,
                message="Your data has been deleted"
            )
        
        user_id = user_result.data[0]["id"]
        hashed_user_id = hash_user_id(user_id)
        
        # CASCADE DELETE
        # 1. Delete RecipientProfiles (FK constraint)
        profiles_result = client.table(TABLE_RECIPIENT_PROFILES)\
            .delete()\
            .eq("user_id", user_id)\
            .execute()
        profiles_deleted = len(profiles_result.data) if profiles_result.data else 0
        
        # 2. De-link Orders (SET user_id = NULL, preserve for accounting)
        # Note: We de-link rather than delete to preserve order records
        orders_result = client.table(TABLE_ORDERS)\
            .update({"user_id": None, "email": None})\
            .eq("user_id", user_id)\
            .execute()
        orders_delinked = len(orders_result.data) if orders_result.data else 0
        
        # 3. Delete User
        client.table(TABLE_USERS)\
            .delete()\
            .eq("id", user_id)\
            .execute()
        
        log.info(
            "Forget-me completed for %s (user=%s): profiles=%d, orders_delinked=%d",
            masked_email, hashed_user_id, profiles_deleted, orders_delinked
        )
        
        # Set cache control header
        response.headers["Cache-Control"] = "no-store"
        
        return ForgetMeResponse(
            success=True,
            deleted_at=deleted_at,
            message="Your data has been deleted"
        )
        
    except Exception as e:
        log.error("Forget-me failed for %s: %s", masked_email, str(e)[:100])
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "DELETE_FAILED",
                    "message": "Failed to delete data. Please try again or contact support."
                },
                "persona": "ARVY"
            }
        )


@router.post(
    "/export-data",
    response_model=ExportDataResponse,
    responses={
        200: {"description": "Data exported successfully"},
        400: {"description": "Invalid OTP or verification failed"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Endpoints disabled"},
    },
    summary="Export User Data (DSAR)",
    description="""
    Export all user data after OTP verification.
    
    **Data Included**: User profile, order history, recipient profiles.
    
    **Security**: Requires valid OTP (request via /auth/request-otp first).
    
    **Privacy**: Sensitive data is included in export; response should not be cached.
    
    **DSAR Compliance**: Request processed immediately (within 30-day SLA).
    """,
)
@limiter.limit("5/minute")
async def export_data(
    body: ExportDataInput,
    request: Request,
    response: Response,
    _: None = Depends(check_memory_endpoints_enabled),
) -> ExportDataResponse:
    """
    Export all user data after OTP verification.
    
    Returns:
        JSON with schema_version, user, orders[], recipients[].
    """
    email = body.email
    otp = body.otp
    
    # Mask email for logging
    masked_email = mask_email(email)
    log.info("Export-data request for %s", masked_email)
    
    # Verify OTP first
    otp_success, otp_message, _ = otp_verify(email, otp)
    
    if not otp_success:
        log.info("Export-data OTP failed for %s: %s", masked_email, otp_message)
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "OTP_INVALID",
                    "message": otp_message
                },
                "persona": "ARVY"
            }
        )
    
    # OTP verified - proceed with export
    client = get_supabase_client()
    if not client:
        log.error("Database unavailable for export-data")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Service temporarily unavailable"
                },
                "persona": "ARVY"
            }
        )
    
    try:
        # Find user by email
        user_result = client.table(TABLE_USERS)\
            .select("*")\
            .ilike("email", email)\
            .limit(1)\
            .execute()
        
        exported_at = datetime.utcnow().isoformat()
        
        if not user_result.data:
            # Anti-enumeration: Return empty export even if user doesn't exist
            log.info("Export-data: no user found for %s (anti-enum empty)", masked_email)
            response.headers["Cache-Control"] = "no-store"
            return ExportDataResponse(
                schema_version=1,
                exported_at=exported_at,
                user=None,
                orders=[],
                recipients=[]
            )
        
        user = user_result.data[0]
        user_id = user["id"]
        hashed_user_id = hash_user_id(user_id)
        
        # Fetch orders
        orders_result = client.table(TABLE_ORDERS)\
            .select("id, sku_id, emotion, created_at")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        orders = orders_result.data or []
        
        # Fetch recipient profiles
        profiles_result = client.table(TABLE_RECIPIENT_PROFILES)\
            .select("id, name, preferences, created_at")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        recipients = profiles_result.data or []
        
        log.info(
            "Export-data completed for %s (user=%s): orders=%d, recipients=%d",
            masked_email, hashed_user_id, len(orders), len(recipients)
        )
        
        # Prepare user data (exclude internal fields)
        user_export = {
            "id": user["id"],
            "email": user["email"],
            "phone": user.get("phone"),
            "memory_opt_in": user.get("memory_opt_in", False),
            "created_at": user["created_at"],
        }
        
        # Set cache control header
        response.headers["Cache-Control"] = "no-store"
        
        return ExportDataResponse(
            schema_version=1,
            exported_at=exported_at,
            user=user_export,
            orders=orders,
            recipients=recipients
        )
        
    except Exception as e:
        log.error("Export-data failed for %s: %s", masked_email, str(e)[:100])
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "EXPORT_FAILED",
                    "message": "Failed to export data. Please try again or contact support."
                },
                "persona": "ARVY"
            }
        )


# ============================================================
# Profile Endpoints (JWT-authenticated)
# ============================================================

@router.get(
    "/profile",
    response_model=List[RecipientProfileResponse],
    responses={
        200: {"description": "List of recipient profiles"},
        401: {"description": "Authentication required"},
        503: {"description": "Endpoints disabled"},
    },
    summary="List Recipient Profiles",
    description="""
    List all recipient profiles for the authenticated user.
    
    **Auth Required**: JWT token in Authorization header.
    """,
)
@limiter.limit("10/minute")
async def list_profiles(
    request: Request,
    user: dict = Depends(get_current_user_required),
    _: None = Depends(check_memory_endpoints_enabled),
) -> List[RecipientProfileResponse]:
    """
    List recipient profiles for authenticated user.
    """
    user_id = user["id"]
    hashed_user_id = hash_user_id(user_id)
    
    client = get_supabase_client()
    if not client:
        log.error("Database unavailable for profile list")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Service temporarily unavailable"
                },
                "persona": "ARVY"
            }
        )
    
    try:
        result = client.table(TABLE_RECIPIENT_PROFILES)\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        
        profiles = result.data or []
        
        log.info("Profile list for user=%s: count=%d", hashed_user_id, len(profiles))
        
        return [
            RecipientProfileResponse.from_profile(RecipientProfile.from_db_row(p))
            for p in profiles
        ]
        
    except Exception as e:
        log.error("Profile list failed for user=%s: %s", hashed_user_id, str(e)[:100])
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "LIST_FAILED",
                    "message": "Failed to list profiles"
                },
                "persona": "ARVY"
            }
        )


@router.post(
    "/profile",
    response_model=RecipientProfileResponse,
    responses={
        200: {"description": "Profile created/updated"},
        400: {"description": "Invalid input"},
        401: {"description": "Authentication required"},
        503: {"description": "Endpoints disabled"},
    },
    summary="Create Recipient Profile",
    description="""
    Create a new recipient profile for the authenticated user.
    
    **Auth Required**: JWT token in Authorization header.
    
    **Preferences**: Optional, with whitelisted keys only (flowers, tier, palette_pref).
    """,
)
@limiter.limit("10/minute")
async def create_profile(
    body: ProfileInput,
    request: Request,
    user: dict = Depends(get_current_user_required),
    _: None = Depends(check_memory_endpoints_enabled),
) -> RecipientProfileResponse:
    """
    Create a new recipient profile.
    """
    user_id = user["id"]
    hashed_user_id = hash_user_id(user_id)
    
    client = get_supabase_client()
    if not client:
        log.error("Database unavailable for profile create")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Service temporarily unavailable"
                },
                "persona": "ARVY"
            }
        )
    
    try:
        profile_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        # Prepare preferences as dict (or None)
        prefs_dict = body.preferences.model_dump() if body.preferences else None
        
        result = client.table(TABLE_RECIPIENT_PROFILES).insert({
            "id": profile_id,
            "user_id": user_id,
            "name": body.name,
            "preferences": prefs_dict,
            "schema_version": 1,
            "created_at": now,
        }).execute()
        
        if not result.data:
            raise Exception("Insert returned no data")
        
        profile = RecipientProfile.from_db_row(result.data[0])
        
        log.info(
            "Profile created for user=%s: profile_id=%s, name=%s",
            hashed_user_id, profile_id[:8], body.name
        )
        
        return RecipientProfileResponse.from_profile(profile)
        
    except Exception as e:
        log.error("Profile create failed for user=%s: %s", hashed_user_id, str(e)[:100])
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "CREATE_FAILED",
                    "message": "Failed to create profile"
                },
                "persona": "ARVY"
            }
        )


@router.put(
    "/profile/{profile_id}",
    response_model=RecipientProfileResponse,
    responses={
        200: {"description": "Profile updated"},
        400: {"description": "Invalid input"},
        401: {"description": "Authentication required"},
        404: {"description": "Profile not found"},
        503: {"description": "Endpoints disabled"},
    },
    summary="Update Recipient Profile",
    description="""
    Update an existing recipient profile.
    
    **Auth Required**: JWT token in Authorization header.
    
    **Security**: Can only update profiles owned by the authenticated user.
    """,
)
@limiter.limit("10/minute")
async def update_profile(
    profile_id: str,
    body: ProfileInput,
    request: Request,
    user: dict = Depends(get_current_user_required),
    _: None = Depends(check_memory_endpoints_enabled),
) -> RecipientProfileResponse:
    """
    Update an existing recipient profile.
    """
    user_id = user["id"]
    hashed_user_id = hash_user_id(user_id)
    
    client = get_supabase_client()
    if not client:
        log.error("Database unavailable for profile update")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Service temporarily unavailable"
                },
                "persona": "ARVY"
            }
        )
    
    try:
        # Verify profile exists and belongs to user
        existing = client.table(TABLE_RECIPIENT_PROFILES)\
            .select("id")\
            .eq("id", profile_id)\
            .eq("user_id", user_id)\
            .limit(1)\
            .execute()
        
        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Profile not found"
                    },
                    "persona": "ARVY"
                }
            )
        
        # Prepare update data
        update_data = {"name": body.name}
        if body.preferences is not None:
            update_data["preferences"] = body.preferences.model_dump()
        
        result = client.table(TABLE_RECIPIENT_PROFILES)\
            .update(update_data)\
            .eq("id", profile_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not result.data:
            raise Exception("Update returned no data")
        
        profile = RecipientProfile.from_db_row(result.data[0])
        
        log.info(
            "Profile updated for user=%s: profile_id=%s",
            hashed_user_id, profile_id[:8]
        )
        
        return RecipientProfileResponse.from_profile(profile)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error("Profile update failed for user=%s: %s", hashed_user_id, str(e)[:100])
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "UPDATE_FAILED",
                    "message": "Failed to update profile"
                },
                "persona": "ARVY"
            }
        )


@router.delete(
    "/profile/{profile_id}",
    responses={
        200: {"description": "Profile deleted"},
        401: {"description": "Authentication required"},
        404: {"description": "Profile not found"},
        503: {"description": "Endpoints disabled"},
    },
    summary="Delete Recipient Profile",
    description="""
    Delete a recipient profile.
    
    **Auth Required**: JWT token in Authorization header.
    
    **Security**: Can only delete profiles owned by the authenticated user.
    """,
)
@limiter.limit("10/minute")
async def delete_profile(
    profile_id: str,
    request: Request,
    user: dict = Depends(get_current_user_required),
    _: None = Depends(check_memory_endpoints_enabled),
) -> dict:
    """
    Delete a recipient profile.
    """
    user_id = user["id"]
    hashed_user_id = hash_user_id(user_id)
    
    client = get_supabase_client()
    if not client:
        log.error("Database unavailable for profile delete")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Service temporarily unavailable"
                },
                "persona": "ARVY"
            }
        )
    
    try:
        # Verify profile exists and belongs to user, then delete
        result = client.table(TABLE_RECIPIENT_PROFILES)\
            .delete()\
            .eq("id", profile_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Profile not found"
                    },
                    "persona": "ARVY"
                }
            )
        
        log.info(
            "Profile deleted for user=%s: profile_id=%s",
            hashed_user_id, profile_id[:8]
        )
        
        return {
            "success": True,
            "deleted_id": profile_id,
            "message": "Profile deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error("Profile delete failed for user=%s: %s", hashed_user_id, str(e)[:100])
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "DELETE_FAILED",
                    "message": "Failed to delete profile"
                },
                "persona": "ARVY"
            }
        )


# ============================================================
# Health Check (for internal monitoring)
# ============================================================

@router.get(
    "/privacy/status",
    response_model=dict,
    include_in_schema=False,  # Internal endpoint
)
async def privacy_status():
    """
    Internal endpoint to check privacy endpoints status.
    
    Not included in OpenAPI schema.
    """
    return {
        "memory_endpoints_enabled": is_memory_endpoints_enabled(),
        "timestamp": datetime.utcnow().isoformat(),
    }
