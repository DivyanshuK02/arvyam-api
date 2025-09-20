# tests/test_apology_context.py
ROMANTIC_PROMPT = "I’m sorry, my love — I want to make it right"
PROFESSIONAL_PROMPT = "I let my team down and I’m truly sorry"

def _emotions(items: list[dict]) -> set[str]:
    return { (it.get("emotion") or "").strip() for it in items }

def test_apology_romantic_routes_to_affection_support(client):
    r = client.post("/api/curate", json={"prompt": ROMANTIC_PROMPT})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 3

    # at least one card should carry the romantic lane’s anchor
    emos = _emotions(items)
    assert "Affection/Support" in emos, f"emotions picked: {emos}"

def test_apology_professional_routes_to_reconciliation_lanes(client):
    r = client.post("/api/curate", json={"prompt": PROFESSIONAL_PROMPT})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 3

    # reconciliation (non-romantic) is anchored on Encouragement/Positivity
    emos = _emotions(items)
    assert "Encouragement/Positivity" in emos, f"emotions picked: {emos}"
