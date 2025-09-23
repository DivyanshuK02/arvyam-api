# tests/test_palette_allowlist.py
from pathlib import Path
import json

# Built from current catalog.json ∪ cheat table (Phase 1.4 Option A)
PALETTE_ALLOWLIST: dict[str, set[str]] = {
    "Affection/Support": set({
        "blush", "cream", "crimson", "deep-red", "gold", "pearl", "pink", "red", "rose-gold", "soft-pink",
        "soft-rose"
    }),
    "Loyalty/Dependability": set({
        "blue", "cream", "navy", "peach", "pearl", "soft-green", "soft-grey", "steel", "white", "yellow"
    }),
    "Encouragement/Positivity": set({
        "apricot", "citrus", "cream", "gold", "golden", "lavender", "peach", "pearl", "soft-green", "soft-orange",
        "soft-purple", "white", "yellow"
    }),
    "Strength/Resilience": set({
        "cream", "deep-green", "eucalyptus", "ivory", "lavender", "peach", "pearl", "purple", "sage", "soft-green",
        "white"
    }),
    "Intellect/Wisdom": set({
        "cream", "eucalyptus", "ivory", "lavender", "linen", "pearl", "soft-beige", "soft-green", "soft-purple", "white"
    }),
    "Adventurous/Creativity": set({
        "accent", "blush", "contrast", "coral", "cream", "crimson", "gold", "multicolor", "peach", "pearl",
        "soft-green", "soft-pink", "vibrant", "white"
    }),
    "Selflessness/Generosity": set({
        "amber", "caramel", "cream", "honey", "peach", "pearl", "soft-gold", "soft-green", "warm", "white"
    }),
    "Fun/Humor": set({
        "bright-yellow", "citrus", "cream", "gold", "green", "marigold", "peach", "pearl", "sunny", "white",
        "yellow"
    }),
}

# Global celebration block forbidden in sympathy/farewell lanes
CELEBRATION_BLOCK = {"deep-red","crimson","gold","neon","bright","hot-pink"}

def _load_catalog() -> list[dict]:
    p = Path('app/catalog.json')
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def test_catalog_palettes_match_allowlist():
    catalog = _load_catalog()
    anchors = set(PALETTE_ALLOWLIST.keys())
    for it in catalog:
        assert it.get('emotion') in anchors, f"{it.get('id')} has unknown emotion: {it.get('emotion')}"
        pal = it.get('palette') or []
        assert isinstance(pal, list) and pal, f"{it.get('id')} has empty or invalid palette"
        allowed = PALETTE_ALLOWLIST[it['emotion']]
        for token in pal:
            if isinstance(token, str):
                t = token.strip().lower()
                assert t in allowed, f"{it['id']}: palette token '{t}' not allowed for {it['emotion']}"

def test_sympathy_blocklist_redundant_guard(client):
    # Redundant, but helpful: ensures sympathy/farewell response contains no celebratory tokens
    r = client.post('/api/curate', json={'prompt': 'I’m so sorry for your loss'})
    assert r.status_code == 200
    items = r.json()
    chips = { (p or '').strip().lower() for it in items for p in (it.get('palette') or []) if isinstance(p, str) }
    assert CELEBRATION_BLOCK.isdisjoint(chips), f'found blocked colors: {CELEBRATION_BLOCK & chips}'