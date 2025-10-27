# tests/test_palette_allowlist.py
import json, re
from pathlib import Path

ANCHORS = {
    "Affection/Support","Loyalty/Dependability","Encouragement/Positivity",
    "Strength/Resilience","Intellect/Wisdom","Adventurous/Creativity",
    "Selflessness/Generosity","Fun/Humor"
}

SAFE_CHARS = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")  # kebab-case
MAX_TOKENS = 6
GLOBAL_BLOCK = {"neon"}  # obvious anti-premium outlier; keep catalog flexible

def _load_catalog():
    with Path("app/catalog.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def test_palette_shape_and_sanity():
    for row in _load_catalog():
        assert row.get("emotion") in ANCHORS, f"{row.get('id')} has unknown emotion"
        pal = row.get("palette")
        assert isinstance(pal, list) and pal, f"{row.get('id')} has empty/invalid palette"
        assert len(pal) <= MAX_TOKENS, f"{row.get('id')} palette too long"
        seen = set()
        for p in pal:
            assert isinstance(p, str), f"{row.get('id')} palette token must be str"
            t = p.strip().lower()
            assert t, f"{row.get('id')} has empty token"
            assert SAFE_CHARS.match(t), f"{row.get('id')} bad token format: {t}"
            assert t not in GLOBAL_BLOCK, f"{row.get('id')} uses globally blocked token: {t}"
            assert t not in seen, f"{row.get('id')} duplicate token: {t}"
            seen.add(t)
