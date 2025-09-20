# tests/conftest.py
import pytest
import uuid
from pathlib import Path
from fastapi.testclient import TestClient
from slowapi.util import get_remote_address

# Import the main FastAPI app instance
from app.main import app

# This dummy function will replace the real rate limiter during tests
def override_get_remote_address():
    return "127.0.0.1"

@pytest.fixture(scope="session")
def client():
    """A TestClient that has rate limiting reliably disabled."""
    # Apply the override right before creating the TestClient
    app.dependency_overrides[get_remote_address] = override_get_remote_address
    
    with TestClient(app) as c:
        yield c

    # Clean up the override after all tests are done
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def evidence_dir():
    """A unique directory for golden harness test artifacts."""
    d = Path("evidence") / f"p1.4a_harness_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d
