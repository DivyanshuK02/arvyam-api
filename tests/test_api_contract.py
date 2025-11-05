# tests/test_api_contract.py
from fastapi.testclient import TestClient
from app.main import app

# Standalone client (keeps this test independent of any fixture variations)
client = TestClient(app)

PUBLIC_KEYS = {
    "id","title","desc","image","price","currency","emotion",
    "tier","mono","palette","note","edge_case","edge_type"
}
FORBIDDEN_KEYS = {
    "packaging","luxury_grand","image_url","price_inr","flowers","weight","tags"
}

def test_curate_contract():
    r = client.post("/api/curate", json={"prompt": "I’m so sorry for your loss"})
    assert r.status_code == 200

    # Response is the list itself (not wrapped)
    items = r.json()
    assert isinstance(items, list)
    assert len(items) == 3

    # Exactly one MONO
    assert sum(1 for it in items if it.get("mono") is True) == 1

    # Each item must conform to the v1 public schema, no leaks
    for it in items:
        # required minimal keys
        for k in ("id","image","price","currency","emotion","tier","mono","palette","desc","title"):
            assert k in it, f"missing key {k} in item: {it}"

        # palette shape
        assert isinstance(it["palette"], list) and len(it["palette"]) > 0

        # currency fixed to INR in v1
        assert it["currency"] == "INR"

        # public keys subset only
        assert set(it.keys()) <= PUBLIC_KEYS, f"unexpected public key(s): {set(it.keys()) - PUBLIC_KEYS}"

        # strictly forbid internal/legacy keys
        for fk in FORBIDDEN_KEYS:
            assert fk not in it, f"forbidden key leaked: {fk}"

        # forbid any underscored/private keys
        assert not any(str(k).startswith("_") for k in it.keys()), "underscored key leaked in public payload"

def test_health_ok_persona():
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    # Minimal health contract for P1.7+
    assert payload.get("status") == "ok"
    assert payload.get("persona") == "ARVY"  # persona must be present
    assert "version" in payload               # optional but useful for UI/debug

def test_checkout_stub_url():
    # Minimal accepted payload (MVP): product_id only
    body = {"product_id": "SKU_TEST"}
    r = client.post("/api/checkout", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    # P1.7 keeps the stub URL shape (Handbook allows stub URL or accepted:true)
    url = payload.get("checkout_url")
    assert isinstance(url, str) and len(url) > 0
    # helpful invariant check for our stub implementation
    assert "pid=SKU_TEST" in url
