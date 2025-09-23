# tests/conftest.py
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make "from app.main import app" work when tests run from CI/workdir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

# --- Disable rate limiting in tests only (prevents 429 when many POSTs run) ---
# Works for common SlowAPI setups: either app.state.limiter or a module-level "limiter".
try:
    # Recent SlowAPI pattern: limiter stored on app.state
    if hasattr(app.state, "limiter"):
        app.state.limiter.enabled = False  # type: ignore[attr-defined]
except Exception:
    pass

try:
    # Fallback: some apps expose a module-level limiter in app.main
    from app.main import limiter as _limiter  # type: ignore
    try:
        _limiter.enabled = False  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    pass

# Optional: if your limiter keys by client IP via slowapi.util.get_remote_address,
# we pin a harmless constant so tests are deterministic. Not required when disabled,
# but harmless to keep.
try:
    from slowapi.util import get_remote_address  # type: ignore

    def _test_ip_override():
        return "127.0.0.1"

    app.dependency_overrides[get_remote_address] = _test_ip_override  # type: ignore[index]
except Exception:
    pass
# --- End limiter handling ---


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
