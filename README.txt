ARVYAM — API (Beginner-Safe, Zero-Drift Rails) — Phase 1.6 Snapshot

Deterministic curation API that turns a short user prompt into exactly three products (2 MIX + 1 MONO) from the catalog, each mapped to one of 8 emotional anchors with an explicit palette[].
Stabilized rails (Phase 1.4a + 1.4)
- Single public schema (one return path)
- Family boundaries ON
- Apology relationship_context routing
- Sympathy/Farewell palette sobriety guard
- Enriched rulebook (synonyms, misspellings, combos, India-aware variants)
- Golden harness + contract tests in CI
- Offline LLM feeder (docs + tools) — no runtime LLM

What's new in Phase 1.6A (PR-1 → PR-8)
- PR-1: Hotfix for NameError in unclear-prompt path
- PR-2R: Rules-first detector (Edges → Exact → Combos → Disambig → Keywords)
- PR-3R: JSON-driven edge filters restored (edge_registers.json) with gold-neutral nuance for grief/farewell
- PR-4: Deterministic rotation (prompt-hash seed + tier salts) with recent-ID suppression
- PR-5: Pool sizes surfaced for observability
- PR-6: Catalog & rulebook touch-ups (e.g., Affection MONOs → roses), docs/tests aligned
- PR-7: Observability: prompt_hash, pool_sizes, resolved_anchor, edge_type, fallback_reason, final_ids, suppressed_recent_count
- PR-8: Tests & docs refresh (Golden Harness v2, boundary fallback tests, sobriety via single source of truth)

Cross-family fallback (rare): items keep their catalog emotions; we do not forcibly stamp them to the resolved anchor. Tests reflect this contract.
Phase snapshot
1.4a — Stabilization: ✅ complete
1.4 — Emotions & Palettes (Enrichment): ✅ complete
1.5 — Packaging Tiers (internal-only): ✅ complete
1.6 — Catalog Schema Freeze (JSON Schema + CI check): ✅ complete
1.6A — Selection Engine Corrections & Observability (PRs 1–8): ✅ complete
1.7 - BACKEND API: ✅ complete
1.8 - Evidence & Observability (Evidence 2.0) frozen : ✅ complete
Evidence source of truth: stored on device at …/Evidence/ — see Phase-1 §1.8 for structure & manifest rules

Quickstart (local)
# Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt

# Run API
uvicorn app.main:app --reload
# Open docs: http://127.0.0.1:8000/docs

Public API (contract is frozen by tests)
POST /api/curate
{ "prompt": "Happy birthday to you!"
}

Response (always an array of 3 public items):
[
  {
    "id": "sku_123",
    "title": "Signature Lilies",
    "desc": "Elegant white lilies for timeless wishes",
    "image": "https://.../lilies.jpg",
    "price": 1799,
    "currency": "INR",
    "emotion": "Encouragement/Positivity",
    "palette": ["yellow","orange"],
    "mono": false
  },
  { "... second mix ..." },
  { "... mono ..." }
]

Never returns internal fields (e.g., raw image URLs, packaging, luxury_grand, internal costs).
Evidence & Observability (PR-7)
Each request writes a single JSON-line evidence record.
Error JSON shape & rate-limit policy: see docs/deploy.md.
Keys (superset across context+meta):
- request_id (UUID4), prompt_hash
- resolved_anchor (one of 8), edge_type (e.g., valentine, apology, sympathy…)
- relationship_context = romantic|familial|friendship|professional|unknown
- fallback_reason = none|in_family|general_in_family|duplicate_tier|cross_family_last_resort
- pool_sizes = { pre_suppress:{classic,signature,luxury}, post_suppress:{...} }
- suppressed_recent_count
- final_ids (the 3 SKU ids returned)

Zero-drift acceptance floor
- Exactly 3 items; never 2 or 4
- Public fields only
- Family boundaries enforced; apology context routed
- Grief/Farewell palette sobriety: no celebratory tokens (gold is allowed)
- Deterministic rotation per prompt; recent-ID suppression
- Evidence line stable; CI must be green before merge
- Catalog validates against docs/catalog.schema.json in CI before tests

Repo layout
.
├─ app/
│  ├─ main.py                # FastAPI routes; public transform; evidence writer
│  ├─ selection_engine.py    # Pure logic: detector, edges, boundaries, rotation, triad contract
│  └─ rules/
│     ├─ emotion_keywords.json
│     ├─ edge_registers.json
│     ├─ sentiment_families.json
│     └─ tier_policy.json
├─ tests/
│  ├─ test_api_contract.py         # Public shape + 3-card invariant
│  ├─ test_golden_harness.py       # v2: anchors/edges, valentine mono-rose, sobriety, artifacts
│  ├─ test_family_boundaries.py    # CELEBRATION_BLOCK (gold allowed) — exported constant
│  ├─ test_boundary_fallbacks.py   # fallback reasons, pool sizes; 1 MONO invariant; post<=pre
│  ├─ test_apology_context.py
│  ├─ test_apply_enums.py
│  ├─ test_miner_filter.py
│  └─ test_packaging_tiers.py
├─ tools/
│  ├─ mine_unknowns.py         # Feeder: mine unknown phrases from logs (offline)
│  ├─ make_review_sheet.py     # Feeder: candidates/proposals → review.csv
│  ├─ apply_tokens.py          # Feeder: add-only to emotion_keywords.json (+ .bak)
│  ├─ validate_rulebook.py     # Schema/enum validator for rulebook
│  ├─ validate_catalog.py      # Validates app/catalog.json against docs/catalog.schema.json
│  └─ rotate_backups.py        # Keep newest N .bak files (local hygiene)
├─ docs/
│  ├─ llm_feeder.md            # Offline feeder guide (no runtime LLM)
│  ├─ phase_status.json        # Live phase ticks
│  └─ catalog.schema.json      # Frozen catalog JSON Schema (P1.6)
├─ evidence/                   # Golden harness artifacts (gitignored)
├─ requirements.txt
└─ dev-requirements.txt

Tests & CI
Local:
pytest -q

CI runs (order):
- Rulebook schema check (tools/validate_rulebook.py)
- Feeder smoke dry-run (tools/apply_tokens.py on tests/data/review_smoke.csv)
- Catalog JSON Schema check (tools/validate_catalog.py against docs/catalog.schema.json)
- Pytest suite:
  • Contract + public-shape (3-card rule)  
  • Golden Harness v2 (artifacts under evidence/)  
  • Boundary fallbacks (reasons, pool size monotonicity, 1 MONO invariant)  
  • Palette sobriety using CELEBRATION_BLOCK  
  • Enums & feeder smokes (no runtime LLM)
- Artifacts uploaded: evidence-bundle, review_smoke_csv

Removed legacy test_palette_allowlist.py and redundant sympathy guard file; sobriety is enforced via test_family_boundaries.py and used by Golden Harness v2.

Behavior notes (for reviewers)
- Detector is rules-first; keywords are a last resort.
- Edges (valentine/apology/sympathy/…) read from edge_registers.json (gold-neutral grief/farewell).
- Rotation is deterministic per prompt (CRC32 prompt hash + tier salts).
- Cross-family fallback: when anchor pools are exhausted, we prefer anchor-coherent items; if still insufficient, we may loosen—items keep catalog emotions (we don't restamp to the resolved anchor). Tests assert the contract.
Offline feeder (docs + tools)
Guide: docs/llm_feeder.md (beginner-proof; add-only; no runtime LLM).
Reproduce CI's smoke locally (sanity-check headers/enums):
python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run

Backups: tools/apply_tokens.py creates timestamped .bak next to the rulebook.
Local hygiene (optional):
python tools/rotate_backups.py --keep 10

Troubleshooting
- "Cannot import app in tests": run from repo root with venv active.
- Golden Harness mismatch on valentine MONO: ensure the MONO SKU contains rose (catalog).
- Sobriety failures in grief/farewell: verify palettes vs CELEBRATION_BLOCK.
- Unexpected fallback: check fallback_reason, anchor_filter_loosened, and pool_sizes in evidence.
- Import error in tests (cannot find app): run from repo root with an active venv.
- CI fails on feeder smoke: verify tests/data/review_smoke.csv headers and anchor enums.
- Engine behavior: confirm edge overrides (e.g., romantic apology → Affection/Support) are honored in selection_engine.py per edge_registers.json.
Contribution notes
- Keep changes additive in app/rules/emotion_keywords.json (Phase 1.4 policy).
- Do not weaken the rails (public schema, 3 cards, boundaries, apology context).
- If you touch rules or tests, run the feeder smoke command above before pushing.
Release tags
v1.6 — Catalog Schema Freeze (p1.6-schema-freeze)
v1.6A — Selection Engine Corrections & Observability (PRs 1–8)
