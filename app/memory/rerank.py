# app/memory/rerank.py
"""
Post-Selection Reranker for Phase 3.1.

Applies optional reranking to the 3-item triad AFTER selection engine completes.
This is the ONLY place memory can affect output — and only ordering, not SKU IDs.

Constitutional Rails (CRITICAL):
- Selection Invariance: set(input_ids) == set(output_ids) ALWAYS
- Bounded Weights: All boosts capped at MAX_WEIGHT (0.3)
- Deterministic: Same (triad, context) → identical output
- Tie-Break Order: (weight DESC, original_index ASC, sku_id ASC)
- Composition Preserved: 2 MIX + 1 MONO structure unchanged

Usage:
```python
from app.memory.rerank import rerank

# After selection engine produces triad
reranked = rerank(triad, memory_context)

# Result has same SKUs, possibly different order
assert set(item["id"] for item in triad) == set(item["id"] for item in reranked.items)
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .context import MemoryContext

log = logging.getLogger("arvyam.memory.rerank")


# ============================================================
# Constants
# ============================================================

# Maximum weight boost (bounded per spec)
MAX_WEIGHT = 0.3

# Weight values for different match types
WEIGHT_FLOWER_MATCH = 0.25  # SKU contains flower from recipient pref
WEIGHT_TIER_MATCH = 0.15    # SKU tier matches recipient pref
WEIGHT_RECENT_EMOTION = 0.10  # SKU emotion matches recent order emotion

# Minimum weight to apply (filter out negligible boosts)
MIN_WEIGHT_THRESHOLD = 0.05


# ============================================================
# Result Types
# ============================================================

@dataclass
class RerankedTriad:
    """
    Result of reranking operation.
    
    Attributes:
        items: The reranked triad (same 3 SKUs, possibly different order)
        was_reranked: Whether any reordering occurred
        weights: Debug info showing weights applied
        nudge_text: Optional micro-copy nudge for UI
    """
    items: List[Dict[str, Any]]
    was_reranked: bool
    weights: List[Tuple[str, float]]  # [(sku_id, weight), ...]
    nudge_text: Optional[str] = None
    
    def __post_init__(self):
        """Validate invariants."""
        if len(self.items) != 3:
            raise ValueError(f"Reranked triad must have exactly 3 items, got {len(self.items)}")


# ============================================================
# Main Rerank Function
# ============================================================

def rerank(
    triad: List[Dict[str, Any]],
    context: MemoryContext,
) -> RerankedTriad:
    """
    Apply post-selection reranking based on memory context.
    
    Args:
        triad: 3-item list from selection engine (2 MIX + 1 MONO).
        context: Memory context from build_context().
        
    Returns:
        RerankedTriad with same SKUs, possibly different order.
        
    Selection Invariance Guarantee:
        set(item["id"] for item in triad) == set(item["id"] for item in result.items)
        
    Determinism Guarantee:
        Same (triad, context) inputs always produce identical output.
        Tie-break: (weight DESC, original_index ASC, sku_id ASC)
        
    Note:
        If context has no history or no preferences match, returns original order.
    """
    # Validate input
    if not triad or len(triad) != 3:
        log.warning("Invalid triad passed to rerank: len=%s", len(triad) if triad else 0)
        return RerankedTriad(
            items=list(triad) if triad else [],
            was_reranked=False,
            weights=[],
        )
    
    # If no history, return as-is
    if not context.get("has_history", False):
        return RerankedTriad(
            items=list(triad),
            was_reranked=False,
            weights=[(item.get("id", ""), 0.0) for item in triad],
        )
    
    # Calculate weights for each item
    weighted_items: List[Tuple[int, str, float, Dict[str, Any]]] = []
    matched_flower: Optional[str] = None
    
    for idx, item in enumerate(triad):
        sku_id = item.get("id", "")
        weight = _calculate_weight(item, context)
        
        # Track matched flower for nudge
        if weight > MIN_WEIGHT_THRESHOLD and not matched_flower:
            matched_flower = _get_matched_flower(item, context)
        
        # Tuple: (original_index, sku_id, weight, item)
        weighted_items.append((idx, sku_id, weight, item))
    
    # Sort by: weight DESC, original_index ASC, sku_id ASC (deterministic tie-break)
    sorted_items = sorted(
        weighted_items,
        key=lambda x: (-x[2], x[0], x[1])  # -weight, orig_idx, sku_id
    )
    
    # Extract reranked items
    reranked = [item for _, _, _, item in sorted_items]
    weights = [(sku_id, weight) for _, sku_id, weight, _ in sorted_items]
    
    # Check if order actually changed
    original_ids = [item.get("id") for item in triad]
    reranked_ids = [item.get("id") for item in reranked]
    was_reranked = original_ids != reranked_ids
    
    # INVARIANT CHECK: SKU set must be identical
    if set(original_ids) != set(reranked_ids):
        log.error(
            "SELECTION_INVARIANCE_VIOLATION: original=%s reranked=%s",
            original_ids, reranked_ids
        )
        # Fail-safe: return original order
        return RerankedTriad(
            items=list(triad),
            was_reranked=False,
            weights=[],
        )
    
    # Generate nudge text if reranking occurred
    nudge_text = None
    if was_reranked:
        from .context import get_nudge_text
        nudge_text = get_nudge_text(context, matched_flower)
    
    if was_reranked:
        log.debug(
            "Reranked triad: %s → %s (weights: %s)",
            original_ids, reranked_ids, weights
        )
    
    return RerankedTriad(
        items=reranked,
        was_reranked=was_reranked,
        weights=weights,
        nudge_text=nudge_text,
    )


# ============================================================
# Weight Calculation
# ============================================================

def _calculate_weight(item: Dict[str, Any], context: MemoryContext) -> float:
    """
    Calculate rerank weight for a single item.
    
    Weight is sum of:
    - Flower match with recipient prefs (0.25)
    - Tier match with recipient prefs (0.15)
    - Emotion match with recent orders (0.10)
    
    Total capped at MAX_WEIGHT (0.3).
    
    Args:
        item: SKU item from triad.
        context: Memory context.
        
    Returns:
        Weight value between 0.0 and MAX_WEIGHT.
    """
    weight = 0.0
    
    # Get item attributes (handle catalog vs transformed format)
    item_flowers = _get_item_flowers(item)
    item_tier = (item.get("tier") or "").lower()
    item_emotion = (item.get("emotion") or "").lower()
    
    recipient_prefs = context.get("recipient_prefs", [])
    recent_emotions = context.get("recent_emotions", [])
    
    # 1. Flower match with recipient preferences
    for pref in recipient_prefs:
        pref_flowers = set(f.lower() for f in (pref.get("flowers") or []))
        if pref_flowers and item_flowers & pref_flowers:
            weight += WEIGHT_FLOWER_MATCH
            break  # Only count once per item
    
    # 2. Tier match with recipient preferences
    for pref in recipient_prefs:
        pref_tier = (pref.get("tier") or "").lower()
        if pref_tier and pref_tier == item_tier:
            weight += WEIGHT_TIER_MATCH
            break  # Only count once per item
    
    # 3. Emotion match with recent orders
    recent_emotions_lower = [e.lower() for e in recent_emotions]
    if item_emotion and item_emotion in recent_emotions_lower:
        weight += WEIGHT_RECENT_EMOTION
    
    # Cap at MAX_WEIGHT
    return min(weight, MAX_WEIGHT)


def _get_item_flowers(item: Dict[str, Any]) -> set:
    """
    Extract flower names from item.
    
    Handles both catalog format (flowers: []) and transformed format.
    """
    # Direct flowers field
    flowers = item.get("flowers") or []
    if flowers:
        return set(f.lower() for f in flowers if isinstance(f, str))
    
    # Try to extract from title/desc (fallback)
    # Common flower names to look for
    common_flowers = {"rose", "lily", "orchid", "tulip", "sunflower", "carnation", "hydrangea", "peony"}
    
    title = (item.get("title") or "").lower()
    desc = (item.get("desc") or "").lower()
    text = f"{title} {desc}"
    
    found = set()
    for flower in common_flowers:
        if flower in text:
            found.add(flower)
    
    return found


def _get_matched_flower(item: Dict[str, Any], context: MemoryContext) -> Optional[str]:
    """
    Get the specific flower that matched recipient preferences.
    
    Used for micro-copy nudge text.
    """
    item_flowers = _get_item_flowers(item)
    
    for pref in context.get("recipient_prefs", []):
        pref_flowers = set(f.lower() for f in (pref.get("flowers") or []))
        matched = item_flowers & pref_flowers
        if matched:
            # Return first match, capitalized
            return next(iter(matched)).capitalize()
    
    return None


# ============================================================
# Invariance Testing Helpers
# ============================================================

def verify_invariance(
    original: List[Dict[str, Any]],
    reranked: List[Dict[str, Any]],
) -> bool:
    """
    Verify selection invariance: SKU sets must be identical.
    
    Args:
        original: Original triad from selection engine.
        reranked: Reranked triad from rerank().
        
    Returns:
        True if invariant holds (SKU sets match).
        
    Raises:
        ValueError: If invariant is violated.
        
    Usage (in tests):
        assert verify_invariance(original, result.items)
    """
    original_ids = set(item.get("id") for item in original)
    reranked_ids = set(item.get("id") for item in reranked)
    
    if original_ids != reranked_ids:
        raise ValueError(
            f"SELECTION_INVARIANCE_VIOLATION: "
            f"original={sorted(original_ids)} reranked={sorted(reranked_ids)}"
        )
    
    return True


def verify_composition(triad: List[Dict[str, Any]]) -> bool:
    """
    Verify triad composition: must be 2 MIX + 1 MONO.
    
    Args:
        triad: Triad to verify.
        
    Returns:
        True if composition is valid.
        
    Raises:
        ValueError: If composition is invalid.
    """
    if len(triad) != 3:
        raise ValueError(f"Triad must have 3 items, got {len(triad)}")
    
    mono_count = sum(1 for item in triad if item.get("mono"))
    
    if mono_count != 1:
        raise ValueError(f"Triad must have exactly 1 mono item, got {mono_count}")
    
    return True
