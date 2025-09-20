# tests/test_sympathy_palette_guard.py
from typing import Iterable

# tokens we consider "celebration/romance" and want to *block* in grief/farewell
CELEBRATION_TOKENS = {
    "deep-red", "crimson", "gold", "neon", "bright", "hot-pink"
}

def _flatten_palettes(items) -> set[str]:
    seen = set()
    for it in items:
        for p in (it.get("palette") or []):
            if isinstance(p, str):
                seen.add(p.strip().lower())
    return seen

def _has_any(tokens: Iterable[str], universe: set[str]) -> bool:
    return any(t in universe for t in tokens)

def test_sympathy_never_uses_celebration_palettes(client):
    r = client.post("/api/curate", json={"prompt": "I’m so sorry for your loss"})
    assert r.status_code == 200
    
    # FIX: The response is the list of items directly
    items = r.json()
    assert isinstance(items, list) and len(items) == 3

    palettes = _flatten_palettes(items)
    assert not _has_any(CELEBRATION_TOKENS, palettes), (
        f"Found celebration tokens {CELEBRATION_TOKENS & palettes} in sympathy triad"
    )
