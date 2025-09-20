# tests/test_family_boundaries.py
from typing import Iterable

# Palettes that clearly read as "celebration/romance" we want to block for grief/farewell
CELEBRATION_TOKENS: set[str] = {
    "deep-red", "red", "crimson", "gold", "neon", "bright", "hot-pink"
}

def _flatten_palettes(items: list[dict]) -> set[str]:
    seen: set[str] = set()
    for it in items:
        pal = it.get("palette") or []
        for p in pal:
            if isinstance(p, str):
                seen.add(p.strip().lower())
    return seen

def _has_any(tokens: Iterable[str], universe: set[str]) -> bool:
    return any(t in universe for t in tokens)

def test_sympathy_never_uses_celebration_palettes(client):
    r = client.post("/api/curate", json={"prompt": "I’m so sorry for your loss"})
    assert r.status_code == 200
    
    # FIX: The response from r.json() is the list of items directly
    items = r.json()
    assert isinstance(items, list) and len(items) == 3

    # Public schema validation (restored from original)
    for it in items:
        for k in ("id", "title", "desc", "image", "price", "currency"):
            assert k in it and it[k] not in (None, "")
        assert "image_url" not in it and "price_inr" not in it

    palettes = _flatten_palettes(items)
    assert not _has_any(CELEBRATION_TOKENS, palettes), f"drifted palettes: {palettes}"

def test_farewell_never_uses_celebration_palettes(client):
    r = client.post("/api/curate", json={"prompt": "Farewell and good luck in your next role"})
    assert r.status_code == 200
    
    # FIX: The response from r.json() is the list of items directly
    items = r.json()
    assert isinstance(items, list) and len(items) == 3

    palettes = _flatten_palettes(items)
    assert not _has_any(CELEBRATION_TOKENS, palettes), f"drifted palettes: {palettes}"
