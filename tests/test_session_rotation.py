# tests/test_session_rotation.py
from fastapi.testclient import TestClient
from app.main import app, SESSION_STORE

client = TestClient(app)

def _ids(resp):
    return [x["id"] for x in resp.json()]

def test_rotation_changes_within_session_same_anchor():
    # same prompt, same session (cookie jar preserved) → different triads via suppression
    r1 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r1.status_code == 200
    ids1 = _ids(r1); assert len(ids1) == 3

    r2 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r2.status_code == 200
    ids2 = _ids(r2); assert len(ids2) == 3

    r3 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r3.status_code == 200
    ids3 = _ids(r3); assert len(ids3) == 3

    assert set(ids1).isdisjoint(set(ids2)), "Triad 1 and 2 should not overlap"
    assert set(ids2).isdisjoint(set(ids3)), "Triad 2 and 3 should not overlap"

def test_determinism_across_users_same_prompt():
    # two independent clients (different sessions) → same first triad for same prompt
    cA = TestClient(app)
    cB = TestClient(app)

    ra = cA.post("/api/curate", json={"prompt": "romantic anniversary"})
    rb = cB.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert ra.status_code == 200 and rb.status_code == 200

    assert _ids(ra) == _ids(rb), "Same prompt should yield identical first triad across users"

def test_session_expiry_resets_rotation():
    # first curate to seed rotation
    r1 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r1.status_code == 200
    ids1 = _ids(r1)

    # simulate expiry by clearing in-memory session store (fast, dev-safe)
    SESSION_STORE._store.clear()

    # next curate → back to first triad pattern
    r2 = client.post("/api/curate", json={"prompt": "romantic anniversary"})
    assert r2.status_code == 200
    ids2 = _ids(r2)

    assert ids1 == ids2, "After session expiry, rotation resets to first triad"
