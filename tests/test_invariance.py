# tests/test_invariance.py
"""
Phase 3.1 Selection Invariance Tests — AC3 PROOF

CRITICAL: These tests prove that memory/reranking CANNOT change SKU IDs.

Tests for:
- SKU set unchanged after rerank (set(input_ids) == set(output_ids) ALWAYS)
- Deterministic tie-break verification
- Bounded weights (<0.3) enforcement
- Fail-closed behavior
- Composition preserved (2 MIX + 1 MONO)

Run with: pytest tests/test_invariance.py -v

NOTE: This test file does NOT require database integration.
"""

import pytest
import os
import sys
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Helper Functions
# ============================================================

def verify_invariance(input_items: List[Dict], output_items: List[Dict]) -> bool:
    """
    Verify that SKU IDs are unchanged between input and output.
    
    This is the CORE invariance check for AC3.
    
    Args:
        input_items: Original triad from selection engine.
        output_items: Reranked triad from memory module.
        
    Returns:
        True if set(input_ids) == set(output_ids), False otherwise.
    """
    input_ids = {item.get("id") or item.get("sku_id") for item in input_items}
    output_ids = {item.get("id") or item.get("sku_id") for item in output_items}
    
    return input_ids == output_ids


def verify_composition(items: List[Dict]) -> bool:
    """
    Verify 2 MIX + 1 MONO composition.
    
    Args:
        items: List of 3 product items.
        
    Returns:
        True if exactly 2 MIX and 1 MONO.
    """
    if len(items) != 3:
        return False
    
    mono_count = sum(1 for item in items if item.get("mono", False))
    mix_count = len(items) - mono_count
    
    return mix_count == 2 and mono_count == 1


# ============================================================
# Core Invariance Tests
# ============================================================

class TestSelectionInvariance:
    """Core tests for AC3: Selection Invariance."""
    
    def test_rerank_preserves_sku_set(self):
        """CRITICAL: Rerank must not change SKU IDs."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        # Input triad
        input_items = [
            {"id": "SKU-001", "title": "Rose Bouquet", "tier": "Classic", "mono": False, "flowers": ["roses"]},
            {"id": "SKU-002", "title": "Lily Arrangement", "tier": "Signature", "mono": False, "flowers": ["lilies"]},
            {"id": "SKU-003", "title": "Orchid Collection", "tier": "Luxury", "mono": True, "flowers": ["orchids"]},
        ]
        
        # Memory context with preferences
        context = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=["SKU-OLD-001"],
            recipient_prefs=[
                {"name": "Mom", "flowers": ["lilies"], "tier": "Signature"}
            ],
            has_history=True,
        )
        
        # Rerank
        output_items, _ = rerank(input_items, context)
        
        # INVARIANCE CHECK
        assert verify_invariance(input_items, output_items), \
            "INVARIANT VIOLATION: SKU IDs changed after rerank!"
    
    def test_rerank_with_empty_context(self):
        """Rerank with empty context should preserve order."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        input_items = [
            {"id": "SKU-001", "title": "A", "mono": False},
            {"id": "SKU-002", "title": "B", "mono": False},
            {"id": "SKU-003", "title": "C", "mono": True},
        ]
        
        # Empty context
        context = MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[],
            has_history=False,
        )
        
        output_items, _ = rerank(input_items, context)
        
        # Invariance must still hold
        assert verify_invariance(input_items, output_items)
        
        # With no context, order should be preserved
        assert [item["id"] for item in output_items] == [item["id"] for item in input_items]
    
    def test_rerank_never_adds_items(self):
        """Rerank must never add items to the triad."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        input_items = [
            {"id": "SKU-001", "mono": False},
            {"id": "SKU-002", "mono": False},
            {"id": "SKU-003", "mono": True},
        ]
        
        context = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=[],
            recipient_prefs=[{"name": "Test", "flowers": ["roses"]}],
            has_history=True,
        )
        
        output_items, _ = rerank(input_items, context)
        
        assert len(output_items) == len(input_items), \
            "Rerank must not change item count!"
    
    def test_rerank_never_removes_items(self):
        """Rerank must never remove items from the triad."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        input_items = [
            {"id": "SKU-001", "mono": False},
            {"id": "SKU-002", "mono": False},
            {"id": "SKU-003", "mono": True},
        ]
        
        context = MemoryContext(
            recent_emotions=["sympathy"],
            recent_skus=["SKU-001", "SKU-002"],  # All items in recent
            recipient_prefs=[],
            has_history=True,
        )
        
        output_items, _ = rerank(input_items, context)
        
        assert len(output_items) == 3, \
            "Rerank must not remove items even if in recent history!"


# ============================================================
# Bounded Weights Tests
# ============================================================

class TestBoundedWeights:
    """Tests for weight bounding (<0.3)."""
    
    def test_max_weight_is_bounded(self):
        """Maximum rerank weight should be ≤0.30."""
        from app.memory.rerank import MAX_RERANK_WEIGHT
        
        assert MAX_RERANK_WEIGHT <= 0.30, \
            f"Max weight should be ≤0.30, got {MAX_RERANK_WEIGHT}"
    
    def test_weights_never_exceed_bound(self):
        """Computed weights should never exceed MAX_RERANK_WEIGHT."""
        from app.memory.rerank import rerank, MAX_RERANK_WEIGHT
        from app.memory.context import MemoryContext
        
        input_items = [
            {"id": "SKU-001", "mono": False, "flowers": ["roses"], "tier": "Classic"},
            {"id": "SKU-002", "mono": False, "flowers": ["lilies"], "tier": "Signature"},
            {"id": "SKU-003", "mono": True, "flowers": ["orchids"], "tier": "Luxury"},
        ]
        
        # Context that matches everything (maximum possible boost)
        context = MemoryContext(
            recent_emotions=["gratitude", "romance", "celebration"],
            recent_skus=[],
            recipient_prefs=[
                {"name": "Test", "flowers": ["roses", "lilies", "orchids"], "tier": "Signature"}
            ],
            has_history=True,
        )
        
        _, debug_info = rerank(input_items, context)
        
        # Check all weights are bounded
        if "weights" in debug_info:
            for weight in debug_info["weights"]:
                assert weight <= MAX_RERANK_WEIGHT, \
                    f"Weight {weight} exceeds bound {MAX_RERANK_WEIGHT}"


# ============================================================
# Determinism Tests
# ============================================================

class TestDeterminism:
    """Tests for deterministic reranking."""
    
    def test_rerank_is_deterministic(self):
        """Same input should always produce same output."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        input_items = [
            {"id": "SKU-001", "mono": False, "flowers": ["roses"]},
            {"id": "SKU-002", "mono": False, "flowers": ["lilies"]},
            {"id": "SKU-003", "mono": True, "flowers": ["orchids"]},
        ]
        
        context = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=[],
            recipient_prefs=[{"name": "Mom", "flowers": ["lilies"]}],
            has_history=True,
        )
        
        results = []
        for _ in range(10):
            output, _ = rerank(input_items.copy(), context)
            results.append([item["id"] for item in output])
        
        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results):
            assert result == first_result, \
                f"Run {i+1} produced different result: {result} vs {first_result}"
    
    def test_tiebreak_is_stable(self):
        """Tie-break rule should be deterministic."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        # Items with same potential weight (no preference matches)
        input_items = [
            {"id": "SKU-003", "mono": False, "flowers": ["tulips"]},
            {"id": "SKU-001", "mono": False, "flowers": ["daisies"]},
            {"id": "SKU-002", "mono": True, "flowers": ["carnations"]},
        ]
        
        context = MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[],
            has_history=False,
        )
        
        # Run multiple times
        results = []
        for _ in range(10):
            output, _ = rerank(input_items.copy(), context)
            results.append([item["id"] for item in output])
        
        # All results should be identical
        assert all(r == results[0] for r in results), \
            "Tie-break is not deterministic!"


# ============================================================
# Composition Preservation Tests
# ============================================================

class TestCompositionPreservation:
    """Tests for 2 MIX + 1 MONO composition."""
    
    def test_composition_preserved_after_rerank(self):
        """Rerank must preserve 2 MIX + 1 MONO composition."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        input_items = [
            {"id": "SKU-001", "mono": False},
            {"id": "SKU-002", "mono": False},
            {"id": "SKU-003", "mono": True},
        ]
        
        context = MemoryContext(
            recent_emotions=["celebration"],
            recent_skus=[],
            recipient_prefs=[{"name": "Test", "flowers": ["roses"]}],
            has_history=True,
        )
        
        output_items, _ = rerank(input_items, context)
        
        assert verify_composition(output_items), \
            "Composition must be 2 MIX + 1 MONO after rerank!"
    
    def test_mono_item_position_flexible(self):
        """MONO item position may change, but count must be 1."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        input_items = [
            {"id": "SKU-001", "mono": False, "flowers": ["tulips"]},
            {"id": "SKU-002", "mono": True, "flowers": ["roses"]},  # MONO in middle
            {"id": "SKU-003", "mono": False, "flowers": ["lilies"]},
        ]
        
        context = MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[{"name": "Test", "flowers": ["roses"]}],  # Matches MONO
            has_history=True,
        )
        
        output_items, _ = rerank(input_items, context)
        
        mono_count = sum(1 for item in output_items if item.get("mono"))
        assert mono_count == 1, "Must have exactly 1 MONO item"


# ============================================================
# Fail-Closed Behavior Tests
# ============================================================

class TestFailClosed:
    """Tests for fail-closed behavior on errors."""
    
    def test_returns_input_on_invalid_context(self):
        """Should return input unchanged if context is invalid."""
        from app.memory.rerank import rerank
        
        input_items = [
            {"id": "SKU-001", "mono": False},
            {"id": "SKU-002", "mono": False},
            {"id": "SKU-003", "mono": True},
        ]
        
        # Invalid context (None or malformed)
        output_items, _ = rerank(input_items, None)
        
        # Should return input unchanged
        assert [item["id"] for item in output_items] == [item["id"] for item in input_items]
    
    def test_returns_input_on_empty_items(self):
        """Should return empty list if input is empty."""
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        context = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=[],
            recipient_prefs=[],
            has_history=True,
        )
        
        output_items, _ = rerank([], context)
        
        assert output_items == []


# ============================================================
# Integration with Selection Engine
# ============================================================

class TestSelectionEngineIntegration:
    """Tests for selection engine + rerank integration."""
    
    def test_full_pipeline_preserves_invariance(self):
        """Full selection + rerank pipeline must preserve SKU invariance."""
        # This test verifies the complete flow:
        # 1. Selection engine selects 3 items
        # 2. Rerank reorders them
        # 3. SKU IDs are unchanged
        
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        # Simulate selection engine output
        selection_output = [
            {"id": "SKU-CLASSIC-001", "tier": "Classic", "mono": False, "emotion": "gratitude"},
            {"id": "SKU-SIGNATURE-001", "tier": "Signature", "mono": False, "emotion": "romance"},
            {"id": "SKU-LUXURY-001", "tier": "Luxury", "mono": True, "emotion": "celebration"},
        ]
        
        context = MemoryContext(
            recent_emotions=["gratitude", "romance"],
            recent_skus=["SKU-OLD-001"],
            recipient_prefs=[
                {"name": "Partner", "flowers": ["roses"], "tier": "Signature"}
            ],
            has_history=True,
        )
        
        # Rerank
        reranked_output, _ = rerank(selection_output, context)
        
        # Verify invariance
        assert verify_invariance(selection_output, reranked_output), \
            "Full pipeline violated SKU invariance!"


# ============================================================
# Evidence Generation
# ============================================================

def generate_invariance_proof() -> dict:
    """
    Generate evidence for AC3 invariance compliance.
    
    Returns:
        Dict with test results and proof data.
    """
    from app.memory.rerank import rerank, MAX_RERANK_WEIGHT
    from app.memory.context import MemoryContext
    
    results = {
        "test_cases": [],
        "violations": 0,
        "max_weight_observed": 0.0,
        "determinism_checks": 0,
        "timestamp": None,
    }
    
    # Test case 1: Standard rerank
    input_items = [
        {"id": "SKU-001", "mono": False, "flowers": ["roses"]},
        {"id": "SKU-002", "mono": False, "flowers": ["lilies"]},
        {"id": "SKU-003", "mono": True, "flowers": ["orchids"]},
    ]
    
    context = MemoryContext(
        recent_emotions=["gratitude"],
        recent_skus=[],
        recipient_prefs=[{"name": "Mom", "flowers": ["lilies"]}],
        has_history=True,
    )
    
    output, debug = rerank(input_items, context)
    
    invariant_held = verify_invariance(input_items, output)
    
    results["test_cases"].append({
        "name": "standard_rerank",
        "input_ids": [item["id"] for item in input_items],
        "output_ids": [item["id"] for item in output],
        "invariant_held": invariant_held,
    })
    
    if not invariant_held:
        results["violations"] += 1
    
    return results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
