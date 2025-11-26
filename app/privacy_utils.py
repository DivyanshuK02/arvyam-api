# app/privacy_utils.py
"""
Privacy utilities for Phase 3.1.

This module provides centralized PII masking and hashing for all logging.
These utilities ensure no raw PII appears in logs while preserving operational value.

Functions:
- hash_prompt_transient(prompt) — Hash prompt for dedup detection
- hash_for_log(full_hash) — Truncate hash for log readability
- mask_email(email) — Mask email for logs
- mask_phone(phone) — Mask phone for logs
- hash_user_id(user_id) — Hash user ID for logs

Usage in logs:
```python
from app.privacy_utils import mask_email, hash_user_id, hash_for_log

log.info({
    "email": mask_email(user_email),           # jo**@example.com
    "user_id": hash_user_id(user_id),          # a1b2c3d4
    "prompt_hash": hash_for_log(prompt_hash),  # e3b0c44298fc (12 chars)
    "prompt_len": len(prompt)                  # 42
})
```

Privacy Rails:
- Never log raw prompts (only prompt_len + hash)
- Never log raw emails (use mask_email)
- Never log raw user IDs (use hash_user_id)
- Never log OTPs (not even hashed)
"""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Optional

# ============================================================
# Prompt Hashing
# ============================================================

def hash_prompt_transient(prompt: str) -> str:
    """
    Hash prompt for dedup detection. Discard plaintext immediately after call.
    
    Args:
        prompt: Raw user prompt text.
        
    Returns:
        Full SHA-256 hash (64 characters) for internal dedupe/QA.
        
    Usage:
        Use this for dedup detection and passing to selection engine.
        For logging, use hash_for_log() to get truncated version.
        
    Security:
        - Never store the raw prompt
        - Discard plaintext immediately after hashing
        - Full hash is for internal use only
    """
    if not prompt:
        return sha256(b"").hexdigest()
    return sha256(prompt.encode("utf-8")).hexdigest()


def hash_for_log(full_hash: str) -> str:
    """
    Truncate hash to 12 chars for log readability.
    
    Args:
        full_hash: Full SHA-256 hash (64 characters).
        
    Returns:
        First 12 characters of hash.
        
    Usage:
        prompt_hash = hash_prompt_transient(prompt)
        log.info({"prompt_hash": hash_for_log(prompt_hash)})
    """
    if not full_hash:
        return ""
    return full_hash[:12]


# ============================================================
# Email Masking
# ============================================================

def mask_email(email: str) -> str:
    """
    Mask email for logs: john@example.com → jo**@example.com
    
    Args:
        email: User email address.
        
    Returns:
        Masked email string.
        
    Examples:
        mask_email("john@example.com") → "jo**@example.com"
        mask_email("ab@example.com") → "**@example.com"
        mask_email("a@example.com") → "**@example.com"
        mask_email(None) → "***"
        mask_email("invalid") → "***"
    """
    if not email or not isinstance(email, str):
        return "***"
    
    email = email.strip()
    
    if "@" not in email:
        return "***"
    
    try:
        local, domain = email.rsplit("@", 1)
    except ValueError:
        return "***"
    
    if len(local) <= 2:
        return f"**@{domain}"
    
    return f"{local[:2]}**@{domain}"


# ============================================================
# Phone Masking
# ============================================================

def mask_phone(phone: str) -> str:
    """
    Mask phone for logs: +911234567890 → +91****7890
    
    Args:
        phone: Phone number (preferably E.164 format).
        
    Returns:
        Masked phone string.
        
    Examples:
        mask_phone("+911234567890") → "+91****7890"
        mask_phone("+14155551234") → "+14****1234"
        mask_phone("1234567890") → "******7890"
        mask_phone("12345") → "****"
        mask_phone(None) → "****"
    """
    if not phone or not isinstance(phone, str):
        return "****"
    
    phone = phone.strip()
    
    # Extract digits only
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) < 6:
        return "****"
    
    # Check if original had + prefix
    has_plus = phone.startswith('+')
    
    if has_plus:
        # E.164 format: show country code + last 4
        # +91****7890
        return f"+{digits[:2]}****{digits[-4:]}"
    else:
        # No country code: show last 4 only
        # ******7890
        return f"******{digits[-4:]}"


# ============================================================
# User ID Hashing
# ============================================================

def hash_user_id(user_id: str) -> str:
    """
    Hash user ID for logs: full UUID → first 8 chars of SHA-256.
    
    Args:
        user_id: User UUID string.
        
    Returns:
        First 8 characters of SHA-256 hash, or "anon" if no user_id.
        
    Examples:
        hash_user_id("550e8400-e29b-41d4-a716-446655440000") → "a1b2c3d4"
        hash_user_id(None) → "anon"
        hash_user_id("") → "anon"
    """
    if not user_id or not isinstance(user_id, str):
        return "anon"
    
    user_id = user_id.strip()
    
    if not user_id:
        return "anon"
    
    return sha256(user_id.encode("utf-8")).hexdigest()[:8]


# ============================================================
# Request ID (Not PII, but useful)
# ============================================================

def truncate_request_id(request_id: str, length: int = 8) -> str:
    """
    Truncate request ID for log readability.
    
    Args:
        request_id: Full request UUID.
        length: Number of characters to keep (default: 8).
        
    Returns:
        Truncated request ID.
    """
    if not request_id or not isinstance(request_id, str):
        return "unknown"
    return request_id[:length]


# ============================================================
# SKU ID (Not PII, safe to log)
# ============================================================

def format_sku_list(sku_ids: list) -> str:
    """
    Format list of SKU IDs for logging.
    
    Args:
        sku_ids: List of SKU ID strings.
        
    Returns:
        Comma-separated string of SKU IDs.
    """
    if not sku_ids or not isinstance(sku_ids, list):
        return "[]"
    return ",".join(str(s) for s in sku_ids)


# ============================================================
# Composite Log Record Builder
# ============================================================

def build_privacy_safe_log(
    *,
    email: Optional[str] = None,
    user_id: Optional[str] = None,
    phone: Optional[str] = None,
    prompt_hash: Optional[str] = None,
    prompt_len: Optional[int] = None,
    request_id: Optional[str] = None,
    sku_ids: Optional[list] = None,
    **extra
) -> dict:
    """
    Build a privacy-safe log record with all PII masked.
    
    Args:
        email: User email (will be masked).
        user_id: User UUID (will be hashed).
        phone: User phone (will be masked).
        prompt_hash: Full prompt hash (will be truncated).
        prompt_len: Prompt length (safe to log).
        request_id: Request UUID (will be truncated).
        sku_ids: List of SKU IDs (safe to log).
        **extra: Additional fields (passed through as-is).
        
    Returns:
        Dict with privacy-safe values ready for logging.
        
    Example:
        log_record = build_privacy_safe_log(
            email="john@example.com",
            user_id="550e8400-...",
            prompt_hash="abc123...",
            prompt_len=42,
            action="curate"
        )
        log.info("Request processed", extra=log_record)
    """
    record = {}
    
    if email is not None:
        record["email"] = mask_email(email)
    
    if user_id is not None:
        record["user_id"] = hash_user_id(user_id)
    
    if phone is not None:
        record["phone"] = mask_phone(phone)
    
    if prompt_hash is not None:
        record["prompt_hash"] = hash_for_log(prompt_hash)
    
    if prompt_len is not None:
        record["prompt_len"] = prompt_len
    
    if request_id is not None:
        record["request_id"] = truncate_request_id(request_id)
    
    if sku_ids is not None:
        record["sku_ids"] = format_sku_list(sku_ids)
    
    # Add any extra fields (caller's responsibility to ensure privacy)
    record.update(extra)
    
    return record


# ============================================================
# Validation Helpers (for testing)
# ============================================================

def is_pii_masked(value: str) -> bool:
    """
    Check if a value appears to be masked (contains ** or ****).
    
    Useful for testing that masking is applied correctly.
    """
    if not value or not isinstance(value, str):
        return False
    return "**" in value or value == "anon" or value == "***" or value == "****"


def contains_raw_email(log_string: str) -> bool:
    """
    Check if a log string contains what looks like a raw email.
    
    Useful for testing that emails are masked in logs.
    
    Note: This is a heuristic, not foolproof.
    """
    if not log_string or not isinstance(log_string, str):
        return False
    
    # Look for patterns like word@word.word that don't have ** masking
    email_pattern = r'\b[A-Za-z0-9._%+-]{3,}@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, log_string)
    
    # Filter out masked emails (those containing **)
    raw_emails = [m for m in matches if '**' not in m]
    
    return len(raw_emails) > 0
