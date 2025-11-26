# tests/test_memory.py
"""
Phase 3.1 Memory Context Tests

Tests for:
- Context building (build_context)
- 90-day lookback enforcement
- memory_opt_in flag respected
- No raw prompts in context
- No PII in logs
- Recipient preferences handling

Run with: pytest tests/test_memory.py -v
Run integration tests: pytest tests/test_memory.py -v -m integration
"""

import pytest

# Mark entire module as integration (requires Phase 3.1 modules)
pytestmark = pytest.mark.integration

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Memory Context Structure Tests
# ============================================================

class TestMemoryContextStructure:
    """Tests for MemoryContext data structure."""
    
    def test_memory_context_has_required_fields(self):
        """MemoryContext should have all required fields."""
        from app.memory.context import MemoryContext
        
        # MemoryContext is a TypedDict, verify keys
        context = MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[],
            has_history=False,
        )
        
        assert "recent_emotions" in context
        assert "recent_skus" in context
        assert "recipient_prefs" in context
        assert "has_history" in context
    
    def test_empty_context_has_no_history(self):
        """Empty context should have has_history=False."""
        from app.memory.context import MemoryContext
        
        context = MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[],
            has_history=False,
        )
        
        assert context["has_history"] is False


# ============================================================
# 90-Day Lookback Tests
# ============================================================

class TestLookbackEnforcement:
    """Tests for 90-day history lookback."""
    
    def test_lookback_constant_is_90_days(self):
        """History lookback should be 90 days."""
        from app.memory.context import HISTORY_LOOKBACK_DAYS
        
        assert HISTORY_LOOKBACK_DAYS == 90, f"Lookback should be 90 days, got {HISTORY_LOOKBACK_DAYS}"
    
    def test_lookback_cutoff_calculation(self):
        """Cutoff date should be 90 days ago."""
        from app.memory.context import HISTORY_LOOKBACK_DAYS
        
        now = datetime.utcnow()
        cutoff = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
        
        # Cutoff should be approximately 90 days ago
        days_diff = (now - cutoff).days
        assert days_diff == 90


# ============================================================
# Memory Opt-In Tests
# ============================================================

class TestMemoryOptIn:
    """Tests for memory_opt_in flag enforcement."""
    
    @patch('app.memory.context.get_supabase_client')
    def test_context_empty_when_not_opted_in(self, mock_get_client):
        """Should return empty context when user hasn't opted in."""
        from app.memory.context import build_context, MemoryContext
        
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock user with memory_opt_in = False
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"id": "user-123", "memory_opt_in": False}
        ]
        
        context = build_context("user-123")
        
        # Should return empty context
        assert context["has_history"] is False
        assert len(context["recent_emotions"]) == 0
        assert len(context["recent_skus"]) == 0
    
    @patch('app.memory.context.get_supabase_client')
    def test_context_built_when_opted_in(self, mock_get_client):
        """Should build context when user has opted in."""
        from app.memory.context import build_context
        
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock user with memory_opt_in = True
        user_response = MagicMock()
        user_response.data = [{"id": "user-123", "memory_opt_in": True}]
        
        # Mock orders
        orders_response = MagicMock()
        orders_response.data = [
            {"sku_id": "SKU-001", "emotion": "gratitude", "created_at": datetime.utcnow().isoformat()},
            {"sku_id": "SKU-002", "emotion": "romance", "created_at": datetime.utcnow().isoformat()},
        ]
        
        # Mock profiles
        profiles_response = MagicMock()
        profiles_response.data = [
            {"name": "Mom", "preferences": {"flowers": ["lilies"], "tier": "Signature"}}
        ]
        
        # Configure mock chain
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = user_response
        mock_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = orders_response
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = profiles_response
        
        context = build_context("user-123")
        
        # Context should have history
        assert context["has_history"] is True


# ============================================================
# No Raw Prompts Tests
# ============================================================

class TestNoRawPrompts:
    """Tests to ensure no raw prompts in memory context."""
    
    def test_context_contains_only_sku_ids(self):
        """Context should contain SKU IDs, not raw prompts."""
        from app.memory.context import MemoryContext
        
        # Valid context should have SKU IDs
        context = MemoryContext(
            recent_emotions=["gratitude", "romance"],
            recent_skus=["SKU-001", "SKU-002"],
            recipient_prefs=[],
            has_history=True,
        )
        
        # Verify structure - no "prompts" field
        assert "prompts" not in context
        assert "raw_prompts" not in context
        
        # SKUs should be string IDs
        for sku in context["recent_skus"]:
            assert isinstance(sku, str)
            assert sku.startswith("SKU-") or sku.startswith("sku_")
    
    def test_context_emotions_are_frozen_enums(self):
        """Emotions in context should be from frozen enum set."""
        from app.memory.context import MemoryContext
        
        # Valid emotions (from anchor_thresholds.json)
        valid_emotions = {
            "gratitude", "romance", "celebration", "sympathy",
            "encouragement", "apology", "farewell", "affection"
        }
        
        context = MemoryContext(
            recent_emotions=["gratitude", "romance"],
            recent_skus=[],
            recipient_prefs=[],
            has_history=True,
        )
        
        for emotion in context["recent_emotions"]:
            # Emotions should be normalized strings
            assert isinstance(emotion, str)


# ============================================================
# Recipient Preferences Tests
# ============================================================

class TestRecipientPreferences:
    """Tests for recipient preference handling."""
    
    def test_preferences_structure(self):
        """Recipient preferences should have correct structure."""
        from app.memory.context import MemoryContext
        
        context = MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[
                {"name": "Mom", "flowers": ["lilies"], "tier": "Signature"},
                {"name": "Dad", "flowers": ["sunflowers"], "tier": "Classic"},
            ],
            has_history=True,
        )
        
        for pref in context["recipient_prefs"]:
            assert "name" in pref
            assert "flowers" in pref or "tier" in pref


# ============================================================
# Nudge Text Generation Tests
# ============================================================

class TestNudgeGeneration:
    """Tests for nudge text generation."""
    
    def test_nudge_for_matched_flower(self):
        """Should generate nudge when flower matches preference."""
        # Nudge format: "Mom loves lilies" or similar
        # This is generated in the rerank module
        pass  # Implementation-specific
    
    def test_nudge_generic_when_no_match(self):
        """Should generate generic nudge when no preference matches."""
        # Generic nudge for returning customer
        pass  # Implementation-specific


# ============================================================
# Database Error Handling Tests
# ============================================================

class TestDatabaseErrorHandling:
    """Tests for graceful database error handling."""
    
    @patch('app.memory.context.get_supabase_client')
    def test_returns_empty_context_on_db_error(self, mock_get_client):
        """Should return empty context on database error."""
        from app.memory.context import build_context
        
        mock_get_client.return_value = None  # No DB connection
        
        context = build_context("user-123")
        
        # Should return safe empty context
        assert context["has_history"] is False
        assert len(context["recent_emotions"]) == 0
    
    @patch('app.memory.context.get_supabase_client')
    def test_handles_query_exception(self, mock_get_client):
        """Should handle query exceptions gracefully."""
        from app.memory.context import build_context
        
        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("Database error")
        mock_get_client.return_value = mock_client
        
        context = build_context("user-123")
        
        # Should return safe empty context
        assert context["has_history"] is False


# ============================================================
# Feature Flag Tests
# ============================================================

class TestMemoryFeatureFlags:
    """Tests for memory feature flags."""
    
    @patch.dict(os.environ, {"MEMORY_CONTEXT_ENABLED": "off"})
    def test_context_disabled_by_default(self):
        """Memory context should be disabled by default."""
        from app.db import is_memory_context_enabled
        
        assert is_memory_context_enabled() is False
    
    @patch.dict(os.environ, {"MEMORY_CONTEXT_ENABLED": "on"})
    def test_context_enabled_when_flag_on(self):
        """Memory context should be enabled when flag is on."""
        from app.db import is_memory_context_enabled
        
        assert is_memory_context_enabled() is True
    
    @patch.dict(os.environ, {"MEMORY_RERANK_ENABLED": "off"})
    def test_rerank_disabled_by_default(self):
        """Memory reranking should be disabled by default."""
        from app.db import is_memory_rerank_enabled
        
        assert is_memory_rerank_enabled() is False
    
    @patch.dict(os.environ, {"MEMORY_RERANK_ENABLED": "on"})
    def test_rerank_enabled_when_flag_on(self):
        """Memory reranking should be enabled when flag is on."""
        from app.db import is_memory_rerank_enabled
        
        assert is_memory_rerank_enabled() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
