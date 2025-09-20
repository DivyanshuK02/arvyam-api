import os
import json
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from slowapi.util import get_remote_address

# Import the main FastAPI app instance
from app.main import app

# --- ADD THIS SECTION TO DISABLE RATE LIMITING IN TESTS ---
# This dummy function will replace the real rate limiter
def override_get_remote_address():
    return "127.0.0.1"

# Replace the real dependency with our dummy one for all tests
app.dependency_overrides[get_remote_address] = override_get_remote_address
# --- END OF ADDED SECTION ---

@pytest.fixture(scope="session")
def client():
    """A TestClient that has rate limiting disabled."""
    return TestClient(app)

@pytest.fixture(scope="session")
def evidence_dir():
    """A unique directory for golden harness test artifacts."""
    d = Path("evidence") / f"p1.4a_harness_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d
