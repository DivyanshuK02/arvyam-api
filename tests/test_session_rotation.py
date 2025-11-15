# tests/test_session_rotation.py
from fastapi.testclient import TestClient
from app.main import app, SESSION_STORE

# Module-level client (reused across tests)
client = TestClient(app)

def _ids(resp):
    """Extract SKU IDs from response"""
    return [x["id"] for x in resp.json()]

def test_rotation_changes_within_session_same_anchor():
    """
    Verify that repeated curates with same prompt return different SKUs.
    Tests core rotation bug fix: suppression within session.
    """
    r1 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r1.status_code == 200
    ids1 = _ids(r1)
    assert len(ids1) == 3

    r2 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r2.status_code == 200
    ids2 = _ids(r2)
    assert len(ids2) == 3

    r3 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r3.status_code == 200
    ids3 = _ids(r3)
    assert len(ids3) == 3

    # Each triad should be disjoint (no overlapping SKUs)
    assert set(ids1).isdisjoint(set(ids2)), "Triad 1 and 2 should not overlap"
    assert set(ids2).isdisjoint(set(ids3)), "Triad 2 and 3 should not overlap"

def test_determinism_across_users_same_prompt():
    """
    Verify that different users (different sessions) get same first triad
    for same prompt. Tests Phase 1.6 determinism requirement.
    """
    # Two independent clients (no shared cookies/session)
    cA = TestClient(app)
    cB = TestClient(app)

    ra = cA.post("/api/curate", json={"prompt": "romantic anniversary"})
    rb = cB.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert ra.status_code == 200 and rb.status_code == 200

    # Same prompt → same first triad (determinism)
    assert _ids(ra) == _ids(rb), "Same prompt should yield identical first triad across users"

def test_session_expiry_resets_rotation():
    """
    Verify that expired sessions reset rotation to first triad.
    Isolates from previous tests by clearing store before baseline.
    """
    # ISOLATE: Clear store to avoid contamination from previous tests
    SESSION_STORE._store.clear()
    
    # BASELINE: Capture true first triad (fresh session, no suppression)
    r0 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r0.status_code == 200
    ids0 = _ids(r0)
    assert len(ids0) == 3
    
    # ROTATE: Induce rotation within session (suppression should trigger)
    r1 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r1.status_code == 200
    ids1 = _ids(r1)
    assert set(ids0).isdisjoint(set(ids1)), "Baseline and rotated triads should differ"
    
    # PURGE: Simulate session expiry by clearing store
    SESSION_STORE._store.clear()
    
    # VERIFY: After expiry, should return to baseline first triad
    r2 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r2.status_code == 200
    ids2 = _ids(r2)
    assert ids2 == ids0, "After expiry, rotation must reset to baseline first triad"
