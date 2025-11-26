# app/accounts/models.py
"""
Pydantic models for Phase 3.1 database tables.

Tables (4):
1. Users — User accounts with optional memory opt-in
2. Orders — Order history (user_id nullable for guests)
3. RecipientProfiles — Saved recipient preferences (user_id required)
4. OtpCodes — OTP storage for auth and privacy verification

Design Decisions:
- Pydantic v2 syntax (model_validator, ConfigDict)
- Strict validation with explicit error messages
- Phone normalization to E.164 format
- Preferences schema with whitelisted keys only

Privacy Rails:
- No PII in preferences (addresses, birthdays blocked)
- Email validated but never logged raw
- Phone stored in E.164 format only

Batch 4 Update:
- Issue #1 FIXED: Strict emotion validation against frozen anchors
"""

from __future__ import annotations

import re
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Literal, Any, Annotated
from pydantic import (
    BaseModel,
    Field,
    EmailStr,
    field_validator,
    model_validator,
    ConfigDict,
    conlist,
)

log = logging.getLogger("arvyam.models")

# ============================================================
# Constants
# ============================================================

# Allowed emotion anchors (frozen per Phase 1.6 catalog)
# These are the EXACT values used in catalog.json
EMOTION_ANCHORS = frozenset({
    # Primary anchors from catalog
    "Affection/Support",
    "Loyalty/Dependability",
    "Encouragement/Positivity",
    "Strength/Resilience",
    "Intellect/Wisdom",
    "Adventurous/Creativity",
    "Selflessness/Generosity",
    "Fun/Humor",
    # Fallback anchor
    "general",
})

# Allowed tier values
TIER_VALUES = frozenset({"Classic", "Signature", "Luxury"})

# Allowed palette preferences
PALETTE_PREFS = frozenset({"pastels", "vibrant", "classic", "muted"})

# Flowers whitelist for preferences (common flowers only)
FLOWERS_WHITELIST = frozenset({
    "roses", "lilies", "orchids", "tulips", "sunflowers",
    "carnations", "chrysanthemums", "hydrangeas", "peonies",
    "gerberas", "mixed", "seasonal",
})

# OTP settings
OTP_TTL_MINUTES = 10
OTP_MAX_ATTEMPTS = 5
OTP_COOLDOWN_SECONDS = 60


# ============================================================
# Phone Normalization
# ============================================================

def normalize_phone(phone: str | None) -> str | None:
    """
    Normalize phone to E.164 format.
    
    Args:
        phone: Raw phone input (may include spaces, dashes, etc.)
        
    Returns:
        E.164 formatted phone (e.g., +911234567890) or None if invalid.
        
    Behavior:
        - If input starts with '+', validates as E.164 (+ followed by 7-15 digits)
        - If input is exactly 10 digits, auto-prepends +91 (India default)
        - All other inputs return None
        
    Examples:
        normalize_phone("+91 12345 67890") → "+911234567890"
        normalize_phone("1234567890") → "+911234567890"  # Auto +91 for 10 digits
        normalize_phone("+14155551234") → "+14155551234"  # Valid E.164 preserved
        normalize_phone("12345") → None  # Too short
        normalize_phone("invalid") → None
    """
    if not phone:
        return None
    
    # Remove all non-digit characters except leading +
    cleaned = re.sub(r'[^\d+]', '', phone.strip())
    
    # Must start with + for E.164
    if not cleaned.startswith('+'):
        # Try to add India country code if 10 digits
        digits_only = re.sub(r'\D', '', phone)
        if len(digits_only) == 10:
            cleaned = f"+91{digits_only}"
        else:
            return None
    
    # E.164: + followed by 7-15 digits
    if re.match(r'^\+\d{7,15}$', cleaned):
        return cleaned
    
    return None


# ============================================================
# Emotion Validation Helper
# ============================================================

def validate_emotion_anchor(emotion: str, strict: bool = True) -> str:
    """
    Validate and normalize emotion anchor value.
    
    Args:
        emotion: Raw emotion string.
        strict: If True, raise ValueError for invalid values.
                If False, log warning but allow value.
                
    Returns:
        Normalized emotion string.
        
    Raises:
        ValueError: If strict=True and emotion not in EMOTION_ANCHORS.
    """
    if not isinstance(emotion, str):
        raise ValueError("emotion must be a string")
    
    normalized = emotion.strip()
    
    if not normalized:
        raise ValueError("emotion cannot be empty")
    
    if normalized not in EMOTION_ANCHORS:
        if strict:
            raise ValueError(
                f"emotion must be one of: {', '.join(sorted(EMOTION_ANCHORS))}. "
                f"Got: '{normalized}'"
            )
        else:
            # Log warning for monitoring but allow value
            log.warning("Unknown emotion anchor: %s (not in frozen set)", normalized)
    
    return normalized


# ============================================================
# Preferences Schema (Whitelisted Keys)
# ============================================================

class PreferencesSchema(BaseModel):
    """
    Validated preferences for recipient profiles.
    
    Whitelisted keys only — unknown keys rejected.
    No PII allowed (addresses, birthdays, phone numbers blocked).
    """
    
    model_config = ConfigDict(extra="forbid")  # Reject unknown keys
    
    # FIX #2: Use conlist for proper max_length validation on list itself
    flowers: Optional[conlist(str, max_length=5)] = Field(
        default=None,
        description="Preferred flower types (max 5)",
    )
    tier: Optional[str] = Field(
        default=None,
        description="Preferred tier: Classic, Signature, or Luxury",
    )
    palette_pref: Optional[str] = Field(
        default=None,
        description="Preferred palette: pastels, vibrant, classic, muted",
    )
    
    @field_validator("flowers", mode="before")
    @classmethod
    def validate_flowers(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("flowers must be a list")
        # Normalize and filter to whitelist
        normalized = []
        for f in v:
            if isinstance(f, str):
                flower = f.strip().lower()
                if flower in FLOWERS_WHITELIST:
                    normalized.append(flower)
        return normalized if normalized else None
    
    @field_validator("tier", mode="before")
    @classmethod
    def validate_tier(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("tier must be a string")
        # Normalize capitalization
        tier_map = {t.lower(): t for t in TIER_VALUES}
        normalized = v.strip().lower()
        if normalized not in tier_map:
            raise ValueError(f"tier must be one of: {', '.join(TIER_VALUES)}")
        return tier_map[normalized]
    
    @field_validator("palette_pref", mode="before")
    @classmethod
    def validate_palette_pref(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("palette_pref must be a string")
        normalized = v.strip().lower()
        if normalized not in PALETTE_PREFS:
            raise ValueError(f"palette_pref must be one of: {', '.join(PALETTE_PREFS)}")
        return normalized


# ============================================================
# User Model
# ============================================================

class UserBase(BaseModel):
    """Base user fields shared across create/response."""
    
    email: EmailStr = Field(..., description="User email (unique, primary identifier)")
    phone: Optional[str] = Field(
        default=None,
        description="Phone in E.164 format (e.g., +911234567890)",
    )
    memory_opt_in: bool = Field(
        default=False,
        description="Whether user consents to memory (default: false)",
    )


class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone_field(cls, v):
        return normalize_phone(v)
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


class User(UserBase):
    """Full user model with database fields."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID = Field(..., description="User UUID (primary key)")
    created_at: datetime = Field(..., description="Account creation timestamp")
    
    @classmethod
    def from_db_row(cls, row: dict) -> "User":
        """Create User from database row."""
        return cls(
            id=uuid.UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            email=row["email"],
            phone=row.get("phone"),
            memory_opt_in=row.get("memory_opt_in", False),
            created_at=row["created_at"] if isinstance(row["created_at"], datetime) 
                       else datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
        )


class UserResponse(BaseModel):
    """Public-safe user response (no sensitive internal fields)."""
    
    user_id: str = Field(..., description="User UUID as string")
    email: str = Field(..., description="User email")
    memory_opt_in: bool = Field(..., description="Memory consent status")
    created_at: str = Field(..., description="ISO timestamp")
    
    @classmethod
    def from_user(cls, user: User) -> "UserResponse":
        return cls(
            user_id=str(user.id),
            email=user.email,
            memory_opt_in=user.memory_opt_in,
            created_at=user.created_at.isoformat(),
        )


# ============================================================
# Order Model
# ============================================================

class OrderBase(BaseModel):
    """Base order fields."""
    
    sku_id: str = Field(..., description="Product SKU from catalog", min_length=1)
    emotion: str = Field(..., description="Emotion anchor (frozen enum)")
    
    @field_validator("emotion", mode="before")
    @classmethod
    def validate_emotion(cls, v):
        """
        Strict emotion validation (Issue #1 FIX - Batch 4).
        
        Validates against frozen EMOTION_ANCHORS set from Phase 1.6 catalog.
        Rejects invalid values to maintain data integrity.
        """
        return validate_emotion_anchor(v, strict=True)


class OrderCreate(OrderBase):
    """Schema for creating a new order."""
    
    user_id: Optional[uuid.UUID] = Field(
        default=None,
        description="User UUID (nullable for guest checkout)",
    )
    email: Optional[EmailStr] = Field(
        default=None,
        description="Email for guest order linkage (90-day historical)",
    )
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


class Order(OrderBase):
    """Full order model with database fields."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID = Field(..., description="Order UUID (primary key)")
    user_id: Optional[uuid.UUID] = Field(
        default=None,
        description="User UUID (nullable for guests)",
    )
    email: Optional[str] = Field(
        default=None,
        description="Email for historical guest linkage",
    )
    created_at: datetime = Field(..., description="Order timestamp")
    
    @classmethod
    def from_db_row(cls, row: dict) -> "Order":
        """Create Order from database row."""
        return cls(
            id=uuid.UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            user_id=uuid.UUID(row["user_id"]) if row.get("user_id") else None,
            email=row.get("email"),
            sku_id=row["sku_id"],
            emotion=row["emotion"],
            created_at=row["created_at"] if isinstance(row["created_at"], datetime)
                       else datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
        )


# ============================================================
# RecipientProfile Model
# ============================================================

class RecipientProfileBase(BaseModel):
    """Base recipient profile fields."""
    
    name: str = Field(
        ...,
        description="Recipient name (e.g., 'Mom', 'Dad')",
        min_length=1,
        max_length=50,
    )
    preferences: Optional[PreferencesSchema] = Field(
        default=None,
        description="Validated preferences (whitelisted keys only)",
    )
    
    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class RecipientProfileCreate(RecipientProfileBase):
    """Schema for creating a recipient profile."""
    
    # user_id is NOT in create schema — derived from authenticated session
    pass


class RecipientProfile(RecipientProfileBase):
    """Full recipient profile model with database fields."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID = Field(..., description="Profile UUID (primary key)")
    user_id: uuid.UUID = Field(..., description="Owner user UUID")
    schema_version: int = Field(default=1, description="Preferences schema version")
    created_at: datetime = Field(..., description="Profile creation timestamp")
    
    @classmethod
    def from_db_row(cls, row: dict) -> "RecipientProfile":
        """Create RecipientProfile from database row."""
        prefs_raw = row.get("preferences")
        prefs = None
        if prefs_raw:
            if isinstance(prefs_raw, dict):
                prefs = PreferencesSchema(**prefs_raw)
            elif isinstance(prefs_raw, str):
                import json
                prefs = PreferencesSchema(**json.loads(prefs_raw))
        
        return cls(
            id=uuid.UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            user_id=uuid.UUID(row["user_id"]) if isinstance(row["user_id"], str) else row["user_id"],
            name=row["name"],
            preferences=prefs,
            schema_version=row.get("schema_version", 1),
            created_at=row["created_at"] if isinstance(row["created_at"], datetime)
                       else datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
        )


class RecipientProfileResponse(BaseModel):
    """Public response for recipient profile."""
    
    id: str = Field(..., description="Profile UUID as string")
    name: str = Field(..., description="Recipient name")
    preferences: Optional[dict] = Field(default=None, description="Preferences dict")
    created_at: str = Field(..., description="ISO timestamp")
    
    @classmethod
    def from_profile(cls, profile: RecipientProfile) -> "RecipientProfileResponse":
        return cls(
            id=str(profile.id),
            name=profile.name,
            preferences=profile.preferences.model_dump() if profile.preferences else None,
            created_at=profile.created_at.isoformat(),
        )


# ============================================================
# OtpCode Model
# ============================================================

class OtpCode(BaseModel):
    """
    OTP code for authentication and privacy verification.
    
    Security:
    - OTP stored as SHA-256 hash (never plaintext)
    - TTL: 10 minutes
    - Max attempts: 5
    - Single-use: Delete after successful verification
    - Cooldown: 60s between requests per email
    """
    
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID = Field(..., description="OTP record UUID")
    email: str = Field(..., description="Email address (lowercase)")
    otp_hash: str = Field(..., description="SHA-256 hash of 6-digit OTP")
    expires_at: datetime = Field(..., description="Expiry timestamp (created_at + 10min)")
    attempts: int = Field(default=0, description="Verification attempts (max 5)")
    created_at: datetime = Field(..., description="OTP creation timestamp")
    
    @property
    def is_expired(self) -> bool:
        """Check if OTP has expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_locked(self) -> bool:
        """Check if max attempts exceeded."""
        return self.attempts >= OTP_MAX_ATTEMPTS
    
    @classmethod
    def from_db_row(cls, row: dict) -> "OtpCode":
        """Create OtpCode from database row."""
        return cls(
            id=uuid.UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            email=row["email"],
            otp_hash=row["otp_hash"],
            expires_at=row["expires_at"] if isinstance(row["expires_at"], datetime)
                       else datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00")),
            attempts=row.get("attempts", 0),
            created_at=row["created_at"] if isinstance(row["created_at"], datetime)
                       else datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
        )
    
    @staticmethod
    def compute_expiry(created_at: datetime | None = None) -> datetime:
        """Compute expiry timestamp (created_at + 10 minutes)."""
        base = created_at or datetime.utcnow()
        return base + timedelta(minutes=OTP_TTL_MINUTES)
    
    @staticmethod
    def is_cooldown_active(last_created_at: datetime) -> bool:
        """Check if cooldown period (60s) is still active."""
        elapsed = (datetime.utcnow() - last_created_at).total_seconds()
        return elapsed < OTP_COOLDOWN_SECONDS


# ============================================================
# Export Data Schema (for /export-data response)
# ============================================================

class ExportDataResponse(BaseModel):
    """Response schema for /export-data endpoint."""
    
    schema_version: int = Field(default=1, description="Export schema version")
    exported_at: str = Field(..., description="ISO timestamp of export")
    user: Optional[dict] = Field(default=None, description="User data")
    orders: List[dict] = Field(default_factory=list, description="Order history")
    recipients: List[dict] = Field(default_factory=list, description="Recipient profiles")
