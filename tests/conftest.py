# tests/conftest.py
"""
Pytest configuration and shared fixtures.

Phase 1.x: Core API testing (rate limiter, evidence directory)
Phase 3.1: Auth, privacy, memory testing (mocks, sample data)
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Make "from app.main import app" work when tests run from CI/workdir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


# ============================================================
# Rate Limiter Disabling (Phase 1.x - PRESERVED)
# ============================================================
# Disable rate limiting in tests only (prevents 429 when many POSTs run)
# Works for common SlowAPI setups: either app.state.limiter or a module-level "limiter".

try:
    if hasattr(app.state, "limiter") and getattr(app.state.limiter, "enabled", True):
        app.state.limiter.enabled = False  # type: ignore[attr-defined]
except Exception:
    pass

try:
    # Fallback: some apps expose a module-level limiter in app.main
    from app.main import limiter as _limiter  # type: ignore
    if getattr(_limiter, "enabled", True):
        _limiter.enabled = False
except Exception:
    pass


# ============================================================
# Core Test Client (Phase 1.x - PRESERVED)
# ============================================================

@pytest.fixture(scope="session")
def client():
    """TestClient against the FastAPI app with test-only limiter disabled."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def evidence_dir():
    """A unique directory for golden harness artifacts."""
    d = Path("evidence") / f"p1_4a_harness_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# Phase 3.1 Environment Setup
# ============================================================

@pytest.fixture
def phase_3_1_env():
    """Set up Phase 3.1 test environment variables."""
    test_env = {
        "JWT_SECRET": "test-secret-for-testing-only-not-production",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
        "OTP_PROVIDER": "stub",
        "AUTH_ENDPOINTS_ENABLED": "off",
        "MEMORY_ENDPOINTS_ENABLED": "off",
        "MEMORY_CONTEXT_ENABLED": "off",
        "MEMORY_RERANK_ENABLED": "off",
    }
    
    with patch.dict(os.environ, test_env):
        yield


# ============================================================
# Phase 3.1 Test Clients
# ============================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client (function-scoped for isolation)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def test_client_with_auth():
    """Create test client with auth endpoints enabled."""
    with patch.dict(os.environ, {"AUTH_ENDPOINTS_ENABLED": "on"}):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def test_client_with_memory():
    """Create test client with memory endpoints enabled."""
    with patch.dict(os.environ, {
        "MEMORY_ENDPOINTS_ENABLED": "on",
        "MEMORY_CONTEXT_ENABLED": "on",
        "MEMORY_RERANK_ENABLED": "on",
    }):
        with TestClient(app) as c:
            yield c


# ============================================================
# Database Mock Fixtures (Phase 3.1)
# ============================================================

@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    mock_client = MagicMock()
    
    # Default empty responses
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    mock_client.table.return_value.insert.return_value.execute.return_value.data = []
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = []
    mock_client.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = []
    
    with patch('app.db.get_supabase_client', return_value=mock_client):
        yield mock_client


# ============================================================
# Sample Data Fixtures (Phase 3.1)
# ============================================================

@pytest.fixture
def sample_user():
    """Sample user data."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "email": "test@example.com",
        "phone": "+911234567890",
        "memory_opt_in": True,
        "created_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_order():
    """Sample order data."""
    return {
        "id": "order-uuid-001",
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "sku_id": "SKU-CLASSIC-001",
        "emotion": "gratitude",
        "email": "test@example.com",
        "created_at": "2024-11-01T00:00:00Z",
    }


@pytest.fixture
def sample_profile():
    """Sample recipient profile data."""
    return {
        "id": "profile-uuid-001",
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Mom",
        "preferences": {
            "flowers": ["lilies", "roses"],
            "tier": "Signature",
        },
        "schema_version": 1,
        "created_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_triad():
    """Sample 3-item triad for selection tests."""
    return [
        {
            "id": "SKU-CLASSIC-001",
            "title": "Classic Rose Bouquet",
            "tier": "Classic",
            "emotion": "gratitude",
            "mono": False,
            "flowers": ["roses"],
            "price": 1500,
            "currency": "INR",
        },
        {
            "id": "SKU-SIGNATURE-001",
            "title": "Signature Lily Arrangement",
            "tier": "Signature",
            "emotion": "romance",
            "mono": False,
            "flowers": ["lilies"],
            "price": 2500,
            "currency": "INR",
        },
        {
            "id": "SKU-LUXURY-001",
            "title": "Luxury Orchid Collection",
            "tier": "Luxury",
            "emotion": "celebration",
            "mono": True,
            "flowers": ["orchids"],
            "price": 5000,
            "currency": "INR",
        },
    ]


# ============================================================
# Memory Context Fixtures (Phase 3.1)
# ============================================================

@pytest.fixture
def empty_memory_context():
    """Empty memory context (no history)."""
    try:
        from app.memory.context import MemoryContext
        return MemoryContext(
            recent_emotions=[],
            recent_skus=[],
            recipient_prefs=[],
            has_history=False,
        )
    except ImportError:
        # Phase 3.1 not deployed yet
        return {
            "recent_emotions": [],
            "recent_skus": [],
            "recipient_prefs": [],
            "has_history": False,
        }


@pytest.fixture
def memory_context_with_history():
    """Memory context with order history and preferences."""
    try:
        from app.memory.context import MemoryContext
        return MemoryContext(
            recent_emotions=["gratitude", "romance"],
            recent_skus=["SKU-OLD-001", "SKU-OLD-002"],
            recipient_prefs=[
                {"name": "Mom", "flowers": ["lilies"], "tier": "Signature"},
                {"name": "Dad", "flowers": ["sunflowers"], "tier": "Classic"},
            ],
            has_history=True,
        )
    except ImportError:
        # Phase 3.1 not deployed yet
        return {
            "recent_emotions": ["gratitude", "romance"],
            "recent_skus": ["SKU-OLD-001", "SKU-OLD-002"],
            "recipient_prefs": [
                {"name": "Mom", "flowers": ["lilies"], "tier": "Signature"},
                {"name": "Dad", "flowers": ["sunflowers"], "tier": "Classic"},
            ],
            "has_history": True,
        }


# ============================================================
# JWT Fixtures (Phase 3.1)
# ============================================================

@pytest.fixture
def valid_jwt_token(sample_user):
    """Generate a valid JWT token for testing."""
    try:
        from app.accounts.auth import create_token
        return create_token(sample_user["id"], sample_user["email"])
    except ImportError:
        # Phase 3.1 not deployed yet - return mock token
        return "mock-jwt-token-for-testing"


@pytest.fixture
def expired_jwt_token(sample_user):
    """Generate an expired JWT token for testing."""
    try:
        import jwt
        secret = os.getenv("JWT_SECRET", "test-secret")
        payload = {
            "user_id": sample_user["id"],
            "email": sample_user["email"],
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=25),
        }
        return jwt.encode(payload, secret, algorithm="HS256")
    except ImportError:
        return "expired-mock-token"


# ============================================================
# Markers
# ============================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require database)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "phase3: marks tests specific to Phase 3.1"
    )
