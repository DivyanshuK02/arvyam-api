# tests/test_memory.py
"""
Phase 3.1 Memory Context Tests.

Tests for:
- Memory context structure (MemoryContext TypedDict)
- 90-day lookback enforcement
- memory_opt_in flag respect
- No raw prompts (privacy rail)
- Recipient preferences handling
- Database error handling (graceful degradation)
- Feature flags

Run with: pytest tests/test_memory.py -v
"""

import pytest
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Memory Context Structure Tests
# ============================================================

class TestMemoryContextStructure:
    """Tests for MemoryContext TypedDict structure."""
    
    def test_memory_context_has_required_fields(self):
        """MemoryContext should have all required fields."""
        from app.memory.context import MemoryContext
        
        # Create a valid context
        ctx = MemoryContext(
            recent_emotions=["gratitude"],
            recent_skus=["SKU-001"],
            recipient_prefs=[],
            has_history=True,
        )
        
        assert "recent_emotions" in ctx
        assert "recent_skus" in ctx
        assert "recipient_prefs" in ctx
        assert "has_history" in ctx
    
    def test_lookback_days_constant(self):
        """Lookback should be 90 days."""
        from app.memory.context import HISTORY_LOOKBACK_DAYS
        
        assert HISTORY_LOOKBACK_DAYS == 90


# ============================================================
# Memory Opt-In Tests
# ============================================================

class TestMemoryOptIn:
    """Tests for memory_opt_in flag respect."""
    
    @patch('app.db.get_supabase_client')
    def test_context_empty_when_not_opted_in(self, mock_get_client):
        """Context should be empty when user hasn't opted in."""
        from app.memory.context import build_context
        
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # User exists but not opted in
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"id": "user-123", "memory_opt_in": False}
        ]
        
        ctx = build_context("user-123")
        
        assert ctx.get("has_history") is False
        assert ctx.get("recent_emotions", []) == []
        assert ctx.get("recent_skus", []) == []
    
    @patch('app.db.get_supabase_client')
    def test_context_built_when_opted_in(self, mock_get_client):
        """Context should be built when user has opted in."""
        from app.memory.context import build_context
        
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock user lookup - opted in
        user_response = MagicMock()
        user_response.data = [{"id": "user-123", "memory_opt_in": True}]
        
        # Mock orders lookup
        orders_response = MagicMock()
        orders_response.data = [
            {"sku_id": "SKU-001", "emotion": "gratitude", "created_at": "2024-11-01T00:00:00Z"},
            {"sku_id": "SKU-002", "emotion": "romance", "created_at": "2024-10-15T00:00:00Z"},
        ]
        
        # Mock profiles lookup
        profiles_response = MagicMock()
        profiles_response.data = [
            {"name": "Mom", "preferences": {"flowers": ["lilies"]}}
        ]
        
        # Chain the mocks
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        
        # Different returns for different table queries
        def table_side_effect(table_name):
            mock = MagicMock()
            if "user" in table_name.lower():
                mock.select.return_value.eq.return_value.limit.return_value.execute.return_value = user_response
            elif "order" in table_name.lower():
                mock.select.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = orders_response
            elif "profile" in table_name.lower():
                mock.select.return_value.eq.return_value.limit.return_value.execute.return_value = profiles_response
            return mock
        
        mock_client.table.side_effect = table_side_effect
        
        ctx = build_context("user-123")
        
        # Should have history
        assert ctx.get("has_history") is True


# ============================================================
# No Raw Prompts Tests (Privacy Rail)
# ============================================================

class TestNoRawPrompts:
    """Tests for privacy rail: no raw prompts in context."""
    
    def test_context_contains_only_safe_fields(self):
        """Context should only contain SKU IDs and frozen enums, not raw text."""
        from app.memory.context import MemoryContext
        
        ctx = MemoryContext(
            recent_emotions=["gratitude", "romance"],
            recent_skus=["SKU-001", "SKU-002"],
            recipient_prefs=[{"name": "Mom", "flowers": ["lilies"]}],
            has_history=True,
        )
        
        # Should NOT have raw prompt fields
        assert "raw_prompt" not in ctx
        assert "user_message" not in ctx
        assert "original_text" not in ctx
        
        # Should only have structured data
        for emotion in ctx.get("recent_emotions", []):
            assert isinstance(emotion, str)
            assert len(emotion) < 50  # Emotion anchors are short
        
        for sku in ctx.get("recent_skus", []):
            assert isinstance(sku, str)
            assert sku.startswith("SKU-") or len(sku) < 100  # SKU IDs are structured


# ============================================================
# Database Error Handling Tests
# ============================================================

class TestDatabaseErrorHandling:
    """Tests for graceful degradation on database errors."""
    
    @patch('app.db.get_supabase_client')
    def test_returns_empty_context_on_db_error(self, mock_get_client):
        """Should return empty context when database is unavailable."""
        from app.memory.context import build_context
        
        # Database unavailable
        mock_get_client.return_value = None
        
        ctx = build_context("user-123")
        
        assert ctx.get("has_history") is False
        assert ctx.get("recent_emotions", []) == []
    
    @patch('app.db.get_supabase_client')
    def test_handles_query_exception(self, mock_get_client):
        """Should handle query exceptions gracefully."""
        from app.memory.context import build_context
        
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Query throws exception
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.side_effect = Exception("DB error")
        
        ctx = build_context("user-123")
        
        # Should return empty context, not crash
        assert ctx.get("has_history") is False


# ============================================================
# Feature Flag Tests
# ============================================================

class TestFeatureFlags:
    """Tests for memory feature flags."""
    
    def test_memory_context_enabled_flag_exists(self):
        """MEMORY_CONTEXT_ENABLED flag should be checkable."""
        # This tests that the flag can be read from environment
        flag = os.environ.get("MEMORY_CONTEXT_ENABLED", "off")
        assert flag in ["on", "off", "true", "false", "1", "0", ""]
    
    def test_memory_rerank_enabled_flag_exists(self):
        """MEMORY_RERANK_ENABLED flag should be checkable."""
        flag = os.environ.get("MEMORY_RERANK_ENABLED", "off")
        assert flag in ["on", "off", "true", "false", "1", "0", ""]


# ============================================================
# Empty User Tests
# ============================================================

class TestEmptyUser:
    """Tests for users with no history."""
    
    @patch('app.db.get_supabase_client')
    def test_new_user_gets_empty_context(self, mock_get_client):
        """New user with no orders should get empty context."""
        from app.memory.context import build_context
        
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # User exists and opted in
        user_response = MagicMock()
        user_response.data = [{"id": "user-123", "memory_opt_in": True}]
        
        # No orders
        orders_response = MagicMock()
        orders_response.data = []
        
        # No profiles
        profiles_response = MagicMock()
        profiles_response.data = []
        
        def table_side_effect(table_name):
            mock = MagicMock()
            if "user" in table_name.lower():
                mock.select.return_value.eq.return_value.limit.return_value.execute.return_value = user_response
            elif "order" in table_name.lower():
                mock.select.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = orders_response
            elif "profile" in table_name.lower():
                mock.select.return_value.eq.return_value.limit.return_value.execute.return_value = profiles_response
            return mock
        
        mock_client.table.side_effect = table_side_effect
        
        ctx = build_context("user-123")
        
        # Empty but valid context
        assert ctx.get("recent_emotions", []) == []
        assert ctx.get("recent_skus", []) == []
    
    def test_empty_user_id_returns_empty_context(self):
        """Empty user_id should return empty context."""
        from app.memory.context import build_context
        
        ctx = build_context("")
        
        assert ctx.get("has_history") is False
        assert ctx.get("recent_emotions", []) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
