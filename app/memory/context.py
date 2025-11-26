# app/memory/context.py
"""
Memory Context Builder for Phase 3.1.

Builds read-only context from user's order history and recipient profiles.
Used by selection engine for post-selection reranking and copy-tone nudges.

Privacy Rails:
- No raw prompts in context (only SKU IDs + frozen enums)
- Context is read-only (cannot modify user data)
- Respects memory_opt_in flag (no context for non-opted users)

Usage:
```python
from app.memory.context import build_context

# Build context for authenticated user
ctx = build_context(user_id="uuid-string")

# Context structure:
{
    "recent_emotions": ["gratitude", "romance"],  # last 3 distinct
    "recent_skus": ["SKU-001", "SKU-002", "SKU-003"],  # last 3 orders
    "recipient_prefs": [
        {"name": "Mom", "flowers": ["lilies"], "tier": "Signature"}
    ]
}
```
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

log = logging.getLogger("arvyam.memory.context")


# ============================================================
# Type Definitions
# ============================================================

class RecipientPref(TypedDict, total=False):
    """Recipient preference from profile."""
    name: str
    flowers: Optional[List[str]]
    tier: Optional[str]
    palette_pref: Optional[str]


class MemoryContext(TypedDict, total=False):
    """
    Read-only memory context for post-selection processing.
    
    Attributes:
        recent_emotions: Last 3 distinct emotion anchors from orders
        recent_skus: Last 3 SKU IDs from orders
        recipient_prefs: Preferences from recipient profiles
        has_history: Whether user has any order history
    """
    recent_emotions: List[str]
    recent_skus: List[str]
    recipient_prefs: List[RecipientPref]
    has_history: bool


# ============================================================
# Constants
# ============================================================

# Maximum lookback for order history (90 days per spec)
HISTORY_LOOKBACK_DAYS = 90

# Maximum items to include in context
MAX_RECENT_EMOTIONS = 3
MAX_RECENT_SKUS = 3
MAX_RECIPIENT_PREFS = 5

# Cache TTL (optional in-memory caching)
CACHE_TTL_SECONDS = 300  # 5 minutes


# ============================================================
# Context Builder
# ============================================================

def build_context(
    user_id: str,
    email: Optional[str] = None,
    *,
    include_recipients: bool = True,
) -> MemoryContext:
    """
    Build read-only memory context for a user.
    
    Args:
        user_id: User UUID string.
        email: Optional email for additional lookups (not currently used).
        include_recipients: Whether to include recipient preferences.
        
    Returns:
        MemoryContext dict with recent_emotions, recent_skus, recipient_prefs.
        
    Notes:
        - Only includes data from users with memory_opt_in=True
        - Returns empty context if user not found or not opted in
        - Only looks back 90 days for orders
        - No raw prompts included (privacy rail)
        
    Example:
        ctx = build_context("550e8400-e29b-41d4-a716-446655440000")
        if ctx.get("has_history"):
            # User has order history, can personalize
            ...
    """
    # Import here to avoid circular dependencies
    try:
        from app.db import get_supabase_client, TABLE_USERS, TABLE_ORDERS, TABLE_RECIPIENT_PROFILES
    except ImportError:
        log.warning("Database not available for memory context")
        return _empty_context()
    
    if not user_id:
        return _empty_context()
    
    client = get_supabase_client()
    if not client:
        log.warning("Database client unavailable")
        return _empty_context()
    
    try:
        # 1. Check user exists and has opted in
        user_result = client.table(TABLE_USERS)\
            .select("id, memory_opt_in")\
            .eq("id", user_id)\
            .limit(1)\
            .execute()
        
        if not user_result.data:
            log.debug("User not found for memory context: %s", user_id[:8])
            return _empty_context()
        
        user = user_result.data[0]
        
        # Respect memory_opt_in flag
        if not user.get("memory_opt_in", False):
            log.debug("User not opted in for memory: %s", user_id[:8])
            return _empty_context()
        
        # 2. Fetch recent orders (last 90 days)
        cutoff = (datetime.utcnow() - timedelta(days=HISTORY_LOOKBACK_DAYS)).isoformat()
        orders_result = client.table(TABLE_ORDERS)\
            .select("sku_id, emotion, created_at")\
            .eq("user_id", user_id)\
            .gte("created_at", cutoff)\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
        
        orders = orders_result.data or []
        
        # Extract recent SKUs (last 3)
        recent_skus = []
        for order in orders:
            sku = order.get("sku_id")
            if sku and sku not in recent_skus:
                recent_skus.append(sku)
                if len(recent_skus) >= MAX_RECENT_SKUS:
                    break
        
        # Extract recent emotions (last 3 distinct)
        recent_emotions = []
        for order in orders:
            emotion = order.get("emotion")
            if emotion and emotion not in recent_emotions:
                recent_emotions.append(emotion)
                if len(recent_emotions) >= MAX_RECENT_EMOTIONS:
                    break
        
        # 3. Fetch recipient profiles (if requested)
        recipient_prefs: List[RecipientPref] = []
        if include_recipients:
            profiles_result = client.table(TABLE_RECIPIENT_PROFILES)\
                .select("name, preferences")\
                .eq("user_id", user_id)\
                .limit(MAX_RECIPIENT_PREFS)\
                .execute()
            
            for profile in (profiles_result.data or []):
                pref: RecipientPref = {"name": profile.get("name", "Unknown")}
                prefs = profile.get("preferences") or {}
                
                if prefs.get("flowers"):
                    pref["flowers"] = prefs["flowers"]
                if prefs.get("tier"):
                    pref["tier"] = prefs["tier"]
                if prefs.get("palette_pref"):
                    pref["palette_pref"] = prefs["palette_pref"]
                
                recipient_prefs.append(pref)
        
        has_history = len(recent_skus) > 0 or len(recipient_prefs) > 0
        
        log.debug(
            "Memory context built for %s: emotions=%d, skus=%d, prefs=%d",
            user_id[:8], len(recent_emotions), len(recent_skus), len(recipient_prefs)
        )
        
        return MemoryContext(
            recent_emotions=recent_emotions,
            recent_skus=recent_skus,
            recipient_prefs=recipient_prefs,
            has_history=has_history,
        )
        
    except Exception as e:
        log.error("Failed to build memory context for %s: %s", user_id[:8], str(e)[:100])
        return _empty_context()


def _empty_context() -> MemoryContext:
    """Return empty memory context."""
    return MemoryContext(
        recent_emotions=[],
        recent_skus=[],
        recipient_prefs=[],
        has_history=False,
    )


# ============================================================
# Context Utilities
# ============================================================

def has_preference_match(
    context: MemoryContext,
    sku_flowers: Optional[List[str]] = None,
    sku_tier: Optional[str] = None,
) -> bool:
    """
    Check if SKU matches any recipient preferences.
    
    Args:
        context: Memory context from build_context().
        sku_flowers: List of flowers in the SKU.
        sku_tier: Tier of the SKU.
        
    Returns:
        True if SKU matches at least one recipient preference.
    """
    if not context.get("recipient_prefs"):
        return False
    
    sku_flowers_set = set(f.lower() for f in (sku_flowers or []))
    sku_tier_lower = (sku_tier or "").lower()
    
    for pref in context["recipient_prefs"]:
        # Check flower match
        pref_flowers = set(f.lower() for f in (pref.get("flowers") or []))
        if pref_flowers and sku_flowers_set & pref_flowers:
            return True
        
        # Check tier match
        pref_tier = (pref.get("tier") or "").lower()
        if pref_tier and pref_tier == sku_tier_lower:
            return True
    
    return False


def get_nudge_text(context: MemoryContext, matched_flower: Optional[str] = None) -> Optional[str]:
    """
    Generate optional micro-copy nudge based on memory context.
    
    Args:
        context: Memory context from build_context().
        matched_flower: Optional flower that matched a preference.
        
    Returns:
        Nudge text string or None if no nudge appropriate.
        
    Note:
        This is for UI-only hints, does NOT affect SKU selection.
    """
    if not context.get("has_history"):
        return None
    
    if matched_flower:
        return f"We noticed you loved {matched_flower} before"
    
    recent_skus = context.get("recent_skus", [])
    if recent_skus:
        return "Based on your previous selections"
    
    return None
