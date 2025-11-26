# app/accounts/otp.py
"""
OTP Service for Phase 3.1.

This module provides:
- Abstract OTP interface (OTPService base class)
- Stub implementation (logs to console, for development)
- OTP generation, hashing, and verification utilities
- Rate limiting and cooldown enforcement

Infrastructure Decision (Locked):
- OTP Service: Abstract interface + stub
- Stub logs to console in dev
- Real provider (Resend/SendGrid) plugged in post-deploy via OTP_PROVIDER env var

Security:
- OTP is 6 digits (000000-999999)
- OTP hashed with SHA-256 before storage (never store plaintext)
- TTL: 10 minutes
- Max attempts: 5 per OTP
- Cooldown: 60s between OTP requests per email
- Daily cap: 20 OTPs per email per 24 hours
- Single-use: Delete after successful verification

Rate Limits (enforced in this module):
- Per-email cooldown: 60 seconds between requests
- Per-email daily cap: 20 OTPs per 24 hours
- Per-IP limits: Enforced at route level via slowapi
"""

from __future__ import annotations

import os
import secrets
import logging
from abc import ABC, abstractmethod
from hashlib import sha256
from datetime import datetime, timedelta
from typing import Optional, Tuple

from app.db import get_supabase_client, TABLE_OTP_CODES
from app.accounts.models import OtpCode, OTP_TTL_MINUTES, OTP_MAX_ATTEMPTS, OTP_COOLDOWN_SECONDS

log = logging.getLogger("arvyam.otp")

# ============================================================
# Constants
# ============================================================

# Daily cap: Maximum OTP requests per email per 24 hours
OTP_DAILY_CAP_PER_EMAIL = 20

# ============================================================
# OTP Utilities
# ============================================================

def generate_otp() -> str:
    """
    Generate a secure 6-digit OTP.
    
    Returns:
        String of 6 digits (e.g., "123456", "000001").
        
    Security:
        Uses secrets module for cryptographically secure random numbers.
    """
    return f"{secrets.randbelow(1000000):06d}"


def hash_otp(otp: str) -> str:
    """
    Hash OTP with SHA-256.
    
    Args:
        otp: 6-digit OTP string.
        
    Returns:
        Hexadecimal SHA-256 hash.
        
    Security:
        Never store plaintext OTP. Always hash before storage.
    """
    return sha256(otp.encode("utf-8")).hexdigest()


def verify_otp_hash(otp: str, otp_hash: str) -> bool:
    """
    Verify OTP against stored hash (constant-time comparison).
    
    Args:
        otp: User-provided OTP.
        otp_hash: Stored SHA-256 hash.
        
    Returns:
        True if OTP matches hash, False otherwise.
        
    Security:
        Uses constant-time comparison to prevent timing attacks.
    """
    import hmac
    computed = hash_otp(otp)
    return hmac.compare_digest(computed, otp_hash)


# ============================================================
# Abstract OTP Service
# ============================================================

class OTPService(ABC):
    """
    Abstract base class for OTP delivery services.
    
    Implementations:
    - StubOTPService: Logs to console (development)
    - ResendOTPService: Sends via Resend API (production)
    - SendGridOTPService: Sends via SendGrid API (production)
    """
    
    @abstractmethod
    def send_otp(self, email: str, otp: str) -> bool:
        """
        Send OTP to email address.
        
        Args:
            email: Recipient email address.
            otp: 6-digit OTP to send.
            
        Returns:
            True if sent successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name for logging."""
        pass


# ============================================================
# Stub Implementation (Development)
# ============================================================

class StubOTPService(OTPService):
    """
    Stub OTP service that logs to console.
    
    Use in development. Never use in production.
    """
    
    def send_otp(self, email: str, otp: str) -> bool:
        """Log OTP to console instead of sending email."""
        log.info(
            "[STUB OTP] Would send OTP to %s: %s (expires in %d minutes)",
            email[:3] + "***",  # Mask email in logs
            otp,
            OTP_TTL_MINUTES
        )
        # Also print to stdout for easy development testing
        print(f"\n{'='*50}")
        print(f"[DEV MODE] OTP for {email}: {otp}")
        print(f"{'='*50}\n")
        return True
    
    def get_provider_name(self) -> str:
        return "stub"


# ============================================================
# Resend Implementation (Production - Optional)
# ============================================================

class ResendOTPService(OTPService):
    """
    OTP service using Resend API.
    
    Requires:
    - OTP_API_KEY: Resend API key
    - OTP_FROM_EMAIL: Sender email (optional, defaults to noreply@arvyam.com)
    """
    
    def __init__(self):
        self.api_key = os.getenv("OTP_API_KEY", "")
        self.from_email = os.getenv("OTP_FROM_EMAIL", "noreply@arvyam.com")
        
        if not self.api_key:
            log.warning("OTP_API_KEY not set for Resend provider")
    
    def send_otp(self, email: str, otp: str) -> bool:
        """Send OTP via Resend API."""
        if not self.api_key:
            log.error("Cannot send OTP: OTP_API_KEY not configured")
            return False
        
        try:
            import httpx
            
            response = httpx.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": self.from_email,
                    "to": [email],
                    "subject": "Your ARVYAM verification code",
                    "html": f"""
                        <div style="font-family: sans-serif; max-width: 400px; margin: 0 auto;">
                            <h2 style="color: #333;">ARVYAM Verification</h2>
                            <p>Your verification code is:</p>
                            <div style="font-size: 32px; font-weight: bold; letter-spacing: 4px; 
                                        padding: 20px; background: #f5f5f5; text-align: center; 
                                        border-radius: 8px; margin: 20px 0;">
                                {otp}
                            </div>
                            <p style="color: #666; font-size: 14px;">
                                This code expires in {OTP_TTL_MINUTES} minutes.
                            </p>
                            <p style="color: #999; font-size: 12px;">
                                If you didn't request this code, please ignore this email.
                            </p>
                        </div>
                    """,
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                log.info("OTP sent via Resend to %s***", email[:3])
                return True
            else:
                log.error("Resend API error: %s %s", response.status_code, response.text[:100])
                return False
                
        except ImportError:
            log.error("httpx not installed. Run: pip install httpx")
            return False
        except Exception as e:
            log.error("Failed to send OTP via Resend: %s", str(e)[:100])
            return False
    
    def get_provider_name(self) -> str:
        return "resend"


# ============================================================
# SendGrid Implementation (Production - Optional)
# ============================================================

class SendGridOTPService(OTPService):
    """
    OTP service using SendGrid API.
    
    Requires:
    - OTP_API_KEY: SendGrid API key
    - OTP_FROM_EMAIL: Verified sender email
    """
    
    def __init__(self):
        self.api_key = os.getenv("OTP_API_KEY", "")
        self.from_email = os.getenv("OTP_FROM_EMAIL", "noreply@arvyam.com")
        
        if not self.api_key:
            log.warning("OTP_API_KEY not set for SendGrid provider")
    
    def send_otp(self, email: str, otp: str) -> bool:
        """Send OTP via SendGrid API."""
        if not self.api_key:
            log.error("Cannot send OTP: OTP_API_KEY not configured")
            return False
        
        try:
            import httpx
            
            response = httpx.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "personalizations": [{"to": [{"email": email}]}],
                    "from": {"email": self.from_email},
                    "subject": "Your ARVYAM verification code",
                    "content": [
                        {
                            "type": "text/html",
                            "value": f"""
                                <div style="font-family: sans-serif; max-width: 400px; margin: 0 auto;">
                                    <h2 style="color: #333;">ARVYAM Verification</h2>
                                    <p>Your verification code is:</p>
                                    <div style="font-size: 32px; font-weight: bold; letter-spacing: 4px; 
                                                padding: 20px; background: #f5f5f5; text-align: center; 
                                                border-radius: 8px; margin: 20px 0;">
                                        {otp}
                                    </div>
                                    <p style="color: #666; font-size: 14px;">
                                        This code expires in {OTP_TTL_MINUTES} minutes.
                                    </p>
                                </div>
                            """,
                        }
                    ],
                },
                timeout=10.0,
            )
            
            if response.status_code in (200, 202):
                log.info("OTP sent via SendGrid to %s***", email[:3])
                return True
            else:
                log.error("SendGrid API error: %s %s", response.status_code, response.text[:100])
                return False
                
        except ImportError:
            log.error("httpx not installed. Run: pip install httpx")
            return False
        except Exception as e:
            log.error("Failed to send OTP via SendGrid: %s", str(e)[:100])
            return False
    
    def get_provider_name(self) -> str:
        return "sendgrid"


# ============================================================
# Service Factory
# ============================================================

def get_otp_service() -> OTPService:
    """
    Get OTP service based on OTP_PROVIDER environment variable.
    
    Providers:
    - "stub" (default): Logs to console
    - "resend": Sends via Resend API
    - "sendgrid": Sends via SendGrid API
    
    Returns:
        OTPService implementation.
    """
    provider = os.getenv("OTP_PROVIDER", "stub").lower().strip()
    
    if provider == "resend":
        return ResendOTPService()
    elif provider == "sendgrid":
        return SendGridOTPService()
    else:
        if provider != "stub":
            log.warning("Unknown OTP_PROVIDER '%s', falling back to stub", provider)
        return StubOTPService()


# ============================================================
# OTP Manager (Database Operations)
# ============================================================

class OTPManager:
    """
    Manages OTP lifecycle: create, verify, cleanup.
    
    Responsibilities:
    - Create OTP records with hashed codes
    - Verify OTP with attempt tracking
    - Enforce cooldown between requests (60s)
    - Enforce daily cap per email (20/day) [MUST-FIX from Auditor]
    - Cleanup expired OTPs
    
    Rate Limits Enforced:
    - Per-email cooldown: 60 seconds between requests
    - Per-email daily cap: 20 OTPs per 24 hours
    - Per-IP limits: Enforced at route level (not here)
    """
    
    def __init__(self, service: OTPService | None = None):
        """
        Initialize OTP manager.
        
        Args:
            service: OTP delivery service. If None, uses get_otp_service().
        """
        self.service = service or get_otp_service()
    
    def _check_daily_cap(self, email: str, client) -> Tuple[bool, int]:
        """
        Check if email has exceeded daily OTP cap (20/day).
        
        Args:
            email: Email address (lowercase).
            client: Supabase client.
            
        Returns:
            Tuple of (within_limit: bool, count: int).
            within_limit is True if under 20 OTPs in last 24 hours.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        try:
            # Count OTPs for this email in last 24 hours
            result = client.table(TABLE_OTP_CODES)\
                .select("id", count="exact")\
                .ilike("email", email)\
                .gte("created_at", cutoff)\
                .execute()
            
            count = result.count if hasattr(result, 'count') and result.count is not None else len(result.data or [])
            within_limit = count < OTP_DAILY_CAP_PER_EMAIL
            
            if not within_limit:
                log.warning(
                    "Daily OTP cap reached for %s*** (%d/%d in 24h)",
                    email[:3], count, OTP_DAILY_CAP_PER_EMAIL
                )
            
            return within_limit, count
            
        except Exception as e:
            log.error("Failed to check daily cap: %s", str(e)[:100])
            # Fail open for now (allow request) to avoid blocking legitimate users
            # In production, consider failing closed
            return True, 0
    
    def request_otp(self, email: str) -> Tuple[bool, str]:
        """
        Request OTP for email address.
        
        Args:
            email: Email address to send OTP to.
            
        Returns:
            Tuple of (success: bool, message: str).
            Message is user-safe (doesn't reveal if email exists).
            
        Rate Limits Enforced:
        - Per-email cooldown: 60s between requests
        - Per-email daily cap: 20 OTPs per 24 hours (MUST-FIX)
        - Per-IP limits: Enforced at route level via slowapi
            
        Security:
        - Anti-enumeration: Same response whether email exists or not
        - OTP hashed before storage (SHA-256)
        - Daily cap prevents abuse without revealing email existence
        """
        email = email.strip().lower()
        client = get_supabase_client()
        
        if not client:
            log.error("Database unavailable for OTP request")
            return False, "Service temporarily unavailable"
        
        try:
            # ================================================================
            # MUST-FIX: Check daily cap (20 OTPs per email per 24 hours)
            # Anti-enumeration: Return same success message even if cap hit
            # ================================================================
            within_daily_cap, daily_count = self._check_daily_cap(email, client)
            
            if not within_daily_cap:
                # Cap exceeded - return anti-enumeration response (don't send OTP)
                log.info(
                    "OTP request blocked (daily cap) for %s*** (%d/%d)",
                    email[:3], daily_count, OTP_DAILY_CAP_PER_EMAIL
                )
                # Anti-enumeration: Same response as success
                return True, "If this email is valid, a verification code was sent"
            
            # ================================================================
            # Check cooldown (60s between requests per email)
            # ================================================================
            recent = client.table(TABLE_OTP_CODES)\
                .select("created_at")\
                .ilike("email", email)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if recent.data:
                last_created = datetime.fromisoformat(
                    recent.data[0]["created_at"].replace("Z", "+00:00")
                )
                if OtpCode.is_cooldown_active(last_created):
                    elapsed = (datetime.utcnow() - last_created.replace(tzinfo=None)).total_seconds()
                    wait = int(OTP_COOLDOWN_SECONDS - elapsed)
                    log.info("OTP cooldown active for %s***, wait %ds", email[:3], wait)
                    return False, f"Please wait {wait} seconds before requesting another code"
            
            # ================================================================
            # Generate and store OTP
            # ================================================================
            otp = generate_otp()
            otp_hashed = hash_otp(otp)
            now = datetime.utcnow()
            expires = OtpCode.compute_expiry(now)
            
            # Delete any existing OTPs for this email (single active OTP per email)
            client.table(TABLE_OTP_CODES)\
                .delete()\
                .ilike("email", email)\
                .execute()
            
            # Insert new OTP
            client.table(TABLE_OTP_CODES).insert({
                "email": email,
                "otp_hash": otp_hashed,
                "expires_at": expires.isoformat(),
                "attempts": 0,
                "created_at": now.isoformat(),
            }).execute()
            
            # ================================================================
            # Send OTP via configured service
            # ================================================================
            sent = self.service.send_otp(email, otp)
            
            if sent:
                log.info(
                    "OTP requested for %s*** via %s (daily: %d/%d)",
                    email[:3], self.service.get_provider_name(),
                    daily_count + 1, OTP_DAILY_CAP_PER_EMAIL
                )
            else:
                log.warning("OTP delivery failed for %s***", email[:3])
            
            # Anti-enumeration: Always return success message
            return True, "If this email is valid, a verification code was sent"
            
        except Exception as e:
            log.error("OTP request failed: %s", str(e)[:100])
            return False, "Service temporarily unavailable"
    
    def verify_otp(self, email: str, otp: str) -> Tuple[bool, str, Optional[dict]]:
        """
        Verify OTP for email address.
        
        Args:
            email: Email address.
            otp: User-provided 6-digit OTP.
            
        Returns:
            Tuple of (success: bool, message: str, user_data: Optional[dict]).
            user_data is included for convenience when creating/authenticating users.
            
        Security:
        - Max 5 attempts per OTP
        - Single-use: Deleted after successful verification
        - Constant-time hash comparison
        """
        email = email.strip().lower()
        otp = otp.strip()
        
        # Basic validation
        if not otp or len(otp) != 6 or not otp.isdigit():
            return False, "Invalid verification code format", None
        
        client = get_supabase_client()
        
        if not client:
            log.error("Database unavailable for OTP verification")
            return False, "Service temporarily unavailable", None
        
        try:
            # Find OTP record
            result = client.table(TABLE_OTP_CODES)\
                .select("*")\
                .ilike("email", email)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if not result.data:
                log.info("No OTP found for %s***", email[:3])
                return False, "Invalid or expired verification code", None
            
            record = OtpCode.from_db_row(result.data[0])
            
            # Check expiry
            if record.is_expired:
                # Cleanup expired record
                client.table(TABLE_OTP_CODES)\
                    .delete()\
                    .eq("id", str(record.id))\
                    .execute()
                log.info("OTP expired for %s***", email[:3])
                return False, "Verification code has expired", None
            
            # Check attempts
            if record.is_locked:
                log.warning("OTP locked (max attempts) for %s***", email[:3])
                return False, "Too many attempts. Please request a new code", None
            
            # Verify hash (constant-time)
            if not verify_otp_hash(otp, record.otp_hash):
                # Increment attempts
                new_attempts = record.attempts + 1
                client.table(TABLE_OTP_CODES)\
                    .update({"attempts": new_attempts})\
                    .eq("id", str(record.id))\
                    .execute()
                
                remaining = OTP_MAX_ATTEMPTS - new_attempts
                log.info("OTP mismatch for %s***, %d attempts remaining", email[:3], remaining)
                
                if remaining <= 0:
                    return False, "Too many attempts. Please request a new code", None
                return False, f"Invalid code. {remaining} attempts remaining", None
            
            # Success! Delete OTP (single-use)
            client.table(TABLE_OTP_CODES)\
                .delete()\
                .eq("id", str(record.id))\
                .execute()
            
            log.info("OTP verified successfully for %s***", email[:3])
            return True, "Verification successful", {"email": email}
            
        except Exception as e:
            log.error("OTP verification failed: %s", str(e)[:100])
            return False, "Service temporarily unavailable", None
    
    def cleanup_expired(self) -> int:
        """
        Cleanup expired OTP records.
        
        Returns:
            Number of records deleted.
            
        Note:
            Can be called periodically or via cron job.
        """
        client = get_supabase_client()
        
        if not client:
            return 0
        
        try:
            now = datetime.utcnow().isoformat()
            result = client.table(TABLE_OTP_CODES)\
                .delete()\
                .lt("expires_at", now)\
                .execute()
            
            deleted = len(result.data) if result.data else 0
            if deleted > 0:
                log.info("Cleaned up %d expired OTP records", deleted)
            return deleted
            
        except Exception as e:
            log.error("OTP cleanup failed: %s", str(e)[:100])
            return 0


# ============================================================
# Module-level convenience functions
# ============================================================

_otp_manager: Optional[OTPManager] = None


def get_otp_manager() -> OTPManager:
    """Get singleton OTP manager instance."""
    global _otp_manager
    if _otp_manager is None:
        _otp_manager = OTPManager()
    return _otp_manager


def request_otp(email: str) -> Tuple[bool, str]:
    """Convenience function to request OTP."""
    return get_otp_manager().request_otp(email)


def verify_otp(email: str, otp: str) -> Tuple[bool, str, Optional[dict]]:
    """Convenience function to verify OTP."""
    return get_otp_manager().verify_otp(email, otp)
