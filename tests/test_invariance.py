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
    """
    input_ids = {item.get("id") or item.get("sku_id") for item in input_items}
    output_ids = {item.get("id") or item.get("sku_id") for item in output_items}
    
    return input_ids == output_ids


def verify_composition(items: List[Dict]) -> bool:
    """Verify 2 MIX + 1 MONO composition."""
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
        
        input_items = [
            {"id": "SKU-001", "title": "Rose Bouquet", "tier": "Classic", "mono": False, "flowers": ["roses"]},
            {"id": "SKU-002", "title": "Lily Arrangement", "tier": "Signature", "mono": False, "flowers": ["lilies"]},
            {"id": "SKU-003", "title": "Orchid Collection", "tier": "Luxury", "mono": True, "flowers": ["orchids"]},
        ]
        
        context = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=["SKU-OLD-001"],
            recipient_prefs=[
                {"name": "Mom", "flowers": ["lilies"], "tier": "Signature"}
            ],
            has_history=True,
        )
        
        # rerank() returns RerankedTriad object with .items attribute
        result = rerank(input_items, context)
        
        # INVARIANCE CHECK
        assert verify_invariance(input_items, result.items), \
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
        
        context = MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[],
            has_history=False,
        )
        
        result = rerank(input_items, context)
        
        # Invariance must still hold
        assert verify_invariance(input_items, result.items)
        
        # With no context, order should be preserved
        assert [item["id"] for item in result.items] == [item["id"] for item in input_items]
        assert result.was_reranked is False
    
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
        
        result = rerank(input_items, context)
        
        assert len(result.items) == len(input_items), \
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
            recent_skus=["SKU-001", "SKU-002"],
            recipient_prefs=[],
            has_history=True,
        )
        
        result = rerank(input_items, context)
        
        assert len(result.items) == 3, \
            "Rerank must not remove items even if in recent history!"


# ============================================================
# Bounded Weights Tests
# ============================================================

class TestBoundedWeights:
    """Tests for weight bounding (<0.3)."""
    
    def test_max_weight_is_bounded(self):
        """Maximum rerank weight should be ≤0.30."""
        # FIX: Actual constant name is MAX_WEIGHT (not MAX_RERANK_WEIGHT)
        from app.memory.rerank import MAX_WEIGHT
        
        assert MAX_WEIGHT <= 0.30, \
            f"Max weight should be ≤0.30, got {MAX_WEIGHT}"
    
    def test_weights_never_exceed_bound(self):
        """Computed weights should never exceed MAX_WEIGHT."""
        from app.memory.rerank import rerank, MAX_WEIGHT
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
        
        result = rerank(input_items, context)
        
        # Check all weights are bounded
        for sku_id, weight in result.weights:
            assert weight <= MAX_WEIGHT, \
                f"Weight {weight} for {sku_id} exceeds bound {MAX_WEIGHT}"


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
            result = rerank(list(input_items), context)
            results.append([item["id"] for item in result.items])
        
        # All results should be identical
        first_result = results[0]
        for i, r in enumerate(results):
            assert r == first_result, \
                f"Run {i+1} produced different result: {r} vs {first_result}"
    
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
        
        results = []
        for _ in range(10):
            result = rerank(list(input_items), context)
            results.append([item["id"] for item in result.items])
        
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
        
        result = rerank(input_items, context)
        
        assert verify_composition(result.items), \
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
        
        result = rerank(input_items, context)
        
        mono_count = sum(1 for item in result.items if item.get("mono"))
        assert mono_count == 1, "Must have exactly 1 MONO item"


# ============================================================
# Fail-Closed Behavior Tests
# ============================================================

class TestFailClosed:
    """Tests for fail-closed behavior on errors."""
    
    def test_raises_on_invalid_triad_length(self):
        """Should raise ValueError if triad has wrong length.
        
        Note: Implementation attempts graceful return but RerankedTriad
        dataclass enforces 3-item invariant in __post_init__.
        """
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        # Only 2 items (invalid)
        input_items = [
            {"id": "SKU-001", "mono": False},
            {"id": "SKU-002", "mono": True},
        ]
        
        context = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=[],
            recipient_prefs=[],
            has_history=True,
        )
        
        # Implementation raises ValueError due to RerankedTriad invariant
        with pytest.raises(ValueError, match="must have exactly 3 items"):
            rerank(input_items, context)
    
    def test_raises_on_empty_triad(self):
        """Should raise ValueError for empty triad.
        
        Note: Implementation attempts graceful return but RerankedTriad
        dataclass enforces 3-item invariant in __post_init__.
        """
        from app.memory.rerank import rerank
        from app.memory.context import MemoryContext
        
        context = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=[],
            recipient_prefs=[],
            has_history=True,
        )
        
        # Implementation raises ValueError due to RerankedTriad invariant
        with pytest.raises(ValueError, match="must have exactly 3 items"):
            rerank([], context)


# ============================================================
# Integration with Selection Engine
# ============================================================

class TestSelectionEngineIntegration:
    """Tests for selection engine + rerank integration."""
    
    def test_full_pipeline_preserves_invariance(self):
        """Full selection + rerank pipeline must preserve SKU invariance."""
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
        
        result = rerank(selection_output, context)
        
        # Verify invariance
        assert verify_invariance(selection_output, result.items), \
            "Full pipeline violated SKU invariance!"


# ============================================================
# Verify Helper Functions
# ============================================================

class TestVerifyHelpers:
    """Tests for the verify_* helper functions in rerank module."""
    
    def test_verify_invariance_function(self):
        """Test the module's verify_invariance function."""
        from app.memory.rerank import verify_invariance as module_verify
        
        original = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        reordered = [{"id": "C"}, {"id": "A"}, {"id": "B"}]
        
        # Same set, different order - should pass
        assert module_verify(original, reordered) is True
    
    def test_verify_composition_function(self):
        """Test the module's verify_composition function."""
        from app.memory.rerank import verify_composition as module_verify
        
        valid = [{"mono": False}, {"mono": False}, {"mono": True}]
        assert module_verify(valid) is True
        
        invalid = [{"mono": False}, {"mono": False}, {"mono": False}]
        with pytest.raises(ValueError):
            module_verify(invalid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
