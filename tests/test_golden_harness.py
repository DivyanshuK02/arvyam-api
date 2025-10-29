# tests/test_golden_harness.py
import json, uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from app.selection_engine import selection_engine  # for context evidence
from test_family_boundaries import CELEBRATION_BLOCK  # single source of truth
# client + evidence_dir fixtures come from tests/conftest.py

# Golden Harness v2: store expectations with each prompt
GOLDEN_CASES: List[Dict[str, Any]] = [
    {
        "name": "valentine_romantic",
        "prompt": "valentine surprise for my love",
        "expect_anchor": "Affection/Support",
        "expect_edge": "valentine",
        "require_mono_rose": True
    },
    {
        "name": "apology_romantic",
        "prompt": "I'm sorry my love, I hurt you",
        "expect_anchor": "Affection/Support",
        "expect_edge": "apology"
    },
    {
        "name": "apology_professional",
        "prompt": "I sincerely apologize to my team for the mistake",
        "expect_anchor": "Encouragement/Positivity",
        "expect_edge": "apology"
    },
    {
        "name": "apology_friendship",
        "prompt": "hey friend, i was wrong – sorry",
        "expect_anchor": "Encouragement/Positivity",
        "expect_edge": "apology"
    },
    {
        "name": "sympathy_loss",
        "prompt": "i'm so sorry for your loss",
        "expect_anchor": "Strength/Resilience",
        "expect_edge": "sympathy",
        "sobriety_required": True
    },
    {
        "name": "farewell_respect",
        "prompt": "farewell and best wishes for your next journey",
        "expect_anchor": "Loyalty/Dependability",
        "expect_edge": "farewell",
        "sobriety_required": True
    },
    {
        "name": "obscure_fallback",
        "prompt": "starlight river pebble whisper",
        "expect_anchor": None,   # don't lock this down; only verify fallback_reason appears
        "expect_edge": None,
        "expect_fallback": True
    },
    # Optional Polish: Add a celebratory case to ensure sobriety guard isn't over-applied
    {
        "name": "celebration_party",
        "prompt": "big birthday party celebration!",
        "expect_anchor": "Fun/Humor",
        "expect_edge": None,
    },
]

def _load_catalog():
    try:
        return json.loads(Path("app/catalog.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return json.loads(Path("../app/catalog.json").read_text(encoding="utf-8"))

CAT = {row["id"]: row for row in _load_catalog()}

REQUIRED_PUBLIC_KEYS = (
    "id", "title", "desc", "image", "price", "currency",
    "emotion", "tier", "mono", "palette"
)
FORBIDDEN_KEYS = {
    "packaging","luxury_grand","image_url","price_inr","flowers","weight","tags"
}

# Editorial mapping (internal packaging stays hidden; UI label is derived from tier)
DISPLAY_LABEL = {"Classic": "Box", "Signature": "Vase", "Luxury": "Premium Box"}


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def test_golden_harness_v2(client, evidence_dir):
    """
    Golden Harness v2 - Integration Test & Evidence Collection

    PURPOSE:
      Validates core selection engine behavior across edge cases and
      generates audit-ready artifacts for business stakeholders.

    VALIDATES:
      • Emotion resolution accuracy (anchor + edge_type)
      • Triad structure (3 items: 1 MONO + 2 MIX)
      • Edge case rules (valentine roses, sympathy sobriety)
      • Fallback transparency (fallback_reason exposed)
      • API contract (public fields only, no internal leakage)

    ARTIFACTS:
      • response.json: API output with UI labels
      • context.json: Engine decision-making context
      • manifest.json: Index of all test runs for regression tracking
      
    USAGE:
      Artifacts stored in evidence_dir (injected by conftest.py).
      Review artifacts after releases to verify no regressions.
    """
    manifest = []

    for case in GOLDEN_CASES:
        # 1) Call the PUBLIC API (source of truth for user output)
        r = client.post("/curate", json={"prompt": case["prompt"]})
        assert r.status_code == 200, f"{case['name']} -> HTTP {r.status_code}"
        items: List[Dict[str, Any]] = r.json()
        assert isinstance(items, list), "Public response must be a list"
        assert len(items) == 3, f"{case['name']}: Expected 3 items, got {len(items)}"

        # 2) Call the ENGINE LOCALLY once to obtain meta/ctx for assertions & artifacts
        eng_items, ctx, meta = selection_engine(prompt=case["prompt"], context={})

        # === V2 ASSERTIONS ===

        # Determinism check (IDs should match between API and local engine)
        assert [it["id"] for it in items] == [it["id"] for it in eng_items], case["name"]

        # Anchor / edge expectations (when provided)
        if case.get("expect_anchor") is not None:
            assert meta.get("resolved_anchor") == case["expect_anchor"], (case['name'], meta)
        if case.get("expect_edge") is not None:
            assert meta.get("edge_type") == case["expect_edge"], (case['name'], meta)

        # All API items must match the resolved anchor (when meta provides it)
        if meta.get("resolved_anchor"):
            for it in items:
                assert it.get("emotion") == meta["resolved_anchor"], (case['name'], it)

        # FIX #1: Valentine mono must include a rose (use catalog as source of truth)
        if case.get("require_mono_rose"):
            mono = next((it for it in items if it.get("mono") is True), None)
            assert mono, (case["name"], "mono item missing")
            # flowers field is in FORBIDDEN_KEYS, so always use catalog
            cat_item = CAT.get(mono["id"])
            assert cat_item, f"{case['name']}: Mono item {mono['id']} not in catalog"
            flowers = cat_item.get("flowers", [])
            assert any((f or "").lower() == "rose" for f in flowers), \
                f"{case['name']}: Expected rose in mono, got flowers={flowers}"

        # Sobriety for sympathy/farewell (validates public API palette)
        if case.get("sobriety_required"):
            for it in items:
                pal = { (p or "").lower().strip() for p in (it.get("palette") or []) if isinstance(p, str) }
                assert pal.isdisjoint(CELEBRATION_BLOCK), (case["name"], it["id"], pal & CELEBRATION_BLOCK)

        # Fallback visibility
        fr = (ctx.get("fallback_reason") or "none").lower()
        if case.get("expect_fallback"):
            assert fr != "none", (case["name"], "fallback_reason should be present")
        else:
            # Note: "none" means no fallback *logic* triggered.
            # "in_family" or "duplicate_tier" are acceptable "soft" fallbacks, not errors.
            assert fr in {"none","in_family","general_in_family","duplicate_tier","cross_family_last_resort"}, (case["name"], fr)

        # === ARTIFACT & MANIFEST LOGIC (Restored) ===

        # Guard: public fields only, no raw catalog or private fields
        mono_count = 0
        seen_tiers = set()
        ui_labels = []

        for it in items:
            # must-have keys
            for k in REQUIRED_PUBLIC_KEYS:
                assert k in it, f"{case['name']}: Missing public field '{k}'"
            # no forbidden or underscored keys
            for fk in FORBIDDEN_KEYS:
                assert fk not in it, f"{case['name']}: forbidden key leaked: {fk}"
            assert not any(str(k).startswith("_") for k in it.keys()), f"{case['name']}: underscored key leaked"
            # palette shape + currency
            assert isinstance(it["palette"], list) and len(it["palette"]) > 0, f"{case['name']}: palette[] must be non-empty"
            assert it["currency"] == "INR"

            # editorial label (derived from tier)
            tier = it["tier"]
            assert tier in DISPLAY_LABEL, f"{case['name']}: unexpected tier value: {tier}"
            ui_labels.append(f"{tier} {DISPLAY_LABEL[tier]}")
            seen_tiers.add(tier)

            if it.get("mono") is True:
                mono_count += 1
        assert mono_count == 1, f"{case['name']}: There must be exactly 1 MONO card"
        # Optional Polish: Explicitly assert 2 MIX cards
        assert sum(1 for it in items if it.get("mono") is False) == 2, f"{case['name']}: There must be exactly 2 MIX cards"

        # If any Luxury appears, its label must be 'Premium Box' (with a space)
        if "Luxury" in seen_tiers:
            assert "Luxury Premium Box" in ui_labels

        # FIX #2: Use meta consistently for resolved fields
        ctx_light = {
            "resolved_anchor": meta.get("resolved_anchor"),
            "edge_type": meta.get("edge_type"),
            "relationship_context": ctx.get("relationship_context"),
            "sentiment_family": ctx.get("sentiment_family"),
            "pool_size": {
                "pre_suppress": ctx.get("pool_sizes", {}).get("pre_suppress"),
                "post_suppress": ctx.get("pool_sizes", {}).get("post_suppress"),
            },
            "fallback_reason": ctx.get("fallback_reason"),
        }

        # FIX #3: Add pool size validation
        pool_sizes = ctx.get("pool_sizes", {})
        pre = pool_sizes.get("pre_suppress", {})
        post = pool_sizes.get("post_suppress", {})
        
        # Sanity check: post-suppress should not exceed pre-suppress
        for tier in ["classic", "signature", "luxury"]:
            pre_count = pre.get(tier, 0)
            post_count = post.get(tier, 0)
            assert post_count <= pre_count, \
                f"{case['name']}: Post-suppress ({post_count}) > pre-suppress ({pre_count}) for {tier}"

        # 4) Save artifacts per case
        case_id = f"{case['name']}_{uuid.uuid4().hex[:6]}"
        ts = datetime.utcnow().isoformat() + "Z"

        _save_json(evidence_dir / f"{case_id}.response.json", {
            "prompt": case['prompt'],
            "timestamp_utc": ts,
            "items": items,
            "ui_labels": ui_labels,  # editorial labels for auditors
        })
        _save_json(evidence_dir / f"{case_id}.context.json", {
            "prompt": case['prompt'],
            "timestamp_utc": ts,
            "context": ctx_light,
        })

        manifest.append({
            "case": case['name'],
            "prompt": case['prompt'],
            "timestamp_seq": ts, # Use 'ts' for consistency
            "files": {
                "response": str((evidence_dir / f"{case_id}.response.json").as_posix()),
                "context": str((evidence_dir / f"{case_id}.context.json").as_posix()),
            },
            "counts": {
                "items": len(items),
                "mono_cards": mono_count,
            },
        })

    # 5) Write manifest last
    _save_json(evidence_dir / "manifest.json", manifest)
