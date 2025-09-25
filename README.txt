ARVYAM — API (Beginner-Safe, Zero-Drift Rails)

Deterministic curation API that turns a short user prompt into exactly three products
(2 MIX + 1 MONO) chosen from the catalog, each mapped to one of 8 emotional anchors
with an explicit palette[].

Stabilized by Phase 1.4a rails and Phase 1.4 enrichment:
• Single public schema (one return path)
• Family boundaries ON by default
• Apology relationship_context routing
• Sympathy/Farewell palette guard
• Rulebook enrichment (synonyms, misspellings, combos, India-aware variants)
• Golden harness + contract tests in CI
• Offline LLM feeder (docs + tools) — no runtime LLM

Phase snapshot
• 1.4a — Stabilization: ✅ complete
• 1.4  — Emotions & Palettes (Enrichment): ✅ complete
• 1.5  — Packaging Tiers (internal-only): ✅ complete
See: docs/phase_status.json for live ticks.

Quickstart (local)
# 1) Create venv (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
# 2) Install deps
pip install -r requirements.txt
pip install -r dev-requirements.txt
# 3) Run the API
uvicorn app.main:app --reload
# 4) Open docs
# http://127.0.0.1:8000/docs

Public API (contract is frozen by tests)
POST /api/curate
{ "prompt": "Happy birthday to you!" }

Response (always an array of 3 items; public fields only):
[
  {
    "id": "sku_123",
    "title": "Signature Lilies",
    "desc": "Elegant white lilies for timeless wishes",
    "image": "https://.../lilies.jpg",
    "price": 1799,
    "currency": "INR",
    "emotion": "Encouragement/Positivity",
    "tier": "Signature",
    "mono": false,
    "palette": ["yellow","orange"]
    // optional: note | edge_case | edge_type
  },
  { "... second mix ..." },
  { "... third mono ..." }
]

NEVER returned: internal fields (packaging, luxury_grand, image_url, price_inr, flowers, weight, tags).

Rails (v1 freeze)
• Exactly 3 items; never 2 or 4.
• Public fields only; any key starting with “_” is dropped.
• Family boundaries enforced; apology relationship_context logged & routed.
• Sympathy/Farewell palette guard (no celebration contamination).
• Stable evidence line per request.

Evidence logging (one JSON line per request)
- request_id (uuid4)
- resolved_anchor (one of 8 anchors:
  Affection/Support, Loyalty/Dependability, Encouragement/Positivity, Strength/Resilience,
  Intellect/Wisdom, Adventurous/Creativity, Selflessness/Generosity, Fun/Humor)
- relationship_context ∈ {romantic, familial, friendship, professional, unknown}
- fallback_reason ∈ {in_family, general_in_family, duplicate_tier, cross_family_last_resort}
- pool_size = { pre_suppress:{classic,signature,luxury}, post_suppress:{...} }

Phase 1.5 — Packaging Tiers (internal-only; non-public)
• Enum file: app/rules/tiers.json = ["Classic","Signature","Luxury"].
• Frozen mapping in catalog:
  Classic → Box, Signature → Vase, Luxury → PremiumBox.
• Luxury Grand (LG) is a boolean flag INSIDE Luxury; it is not a tier.
• Banlist: no “Keepsake”, and no spaced “Premium Box” in packaging (must be “PremiumBox”).
• Public API is packaging-blind (no packaging / luxury_grand keys ever leave the server).

Repo layout
.
├─ app/
│  ├─ main.py                  # FastAPI app & routes; public transform; request_id injection
│  ├─ selection_engine.py      # Pure logic: families, apology context, triad ritual
│  └─ rules/
│     ├─ emotion_keywords.json
│     ├─ edge_registers.json
│     ├─ sentiment_families.json
│     ├─ tier_policy.json
│     ├─ tiers.json            # Phase 1.5 enum: ["Classic","Signature","Luxury"]
│     └─ (other policy files…)
├─ tests/
│  ├─ test_api_contract.py       # Public schema + 3-card rule; forbid internals
│  ├─ test_golden_harness.py     # Canonical prompts; saves evidence/ artifacts
│  ├─ test_apply_enums.py        # Feeder smoke + tiers enum freeze
│  ├─ test_packaging_tiers.py    # Catalog mapping (Classic/Box, Signature/Vase, Luxury/PremiumBox) + LG rule
│  ├─ test_sympathy_palette_guard.py
│  ├─ test_palette_allowlist.py
│  ├─ test_apology_context.py
│  └─ test_family_boundaries.py
├─ tools/
│  ├─ apply_tokens.py          # Feeder: add-only to emotion_keywords.json (+ .bak)
│  ├─ make_review_sheet.py     # Feeder: candidates/proposals → review.csv
│  ├─ mine_unknowns.py         # Feeder: mine unknown phrases from logs (offline)
│  ├─ validate_rulebook.py     # Rulebook schema/enum validator
│  └─ rotate_backups.py        # Keep newest N .bak files (local hygiene)
├─ docs/
│  ├─ llm_feeder.md            # Offline feeder guide (no runtime LLM)
│  └─ phase_status.json        # Live phase ticks
├─ evidence/                   # Golden harness & phase bundles (gitignored)
├─ requirements.txt
└─ dev-requirements.txt

Tests & CI
Local:
  pytest -q

GitHub Actions (CI):
• Rulebook schema check:
  python tools/validate_rulebook.py app/rules/emotion_keywords.json
• Feeder smoke (no mutations):
  python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run
• Run pytest (contract + guards + golden harness).
• Upload artifacts from evidence/ and the smoke CSV.

Offline feeder (docs + tools)
Guide: docs/llm_feeder.md (beginner-proof; add-only; no runtime LLM).
Reproduce CI’s smoke locally:
python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run
Backups: tools/apply_tokens.py creates timestamped .bak alongside the rulebook.

Troubleshooting
• Import error in tests: run from repo root with an active venv.
• CI fails on feeder smoke: verify review_smoke.csv headers and anchor enums.
• Unexpected palettes in grief/farewell: see test_sympathy_palette_guard.py & test_palette_allowlist.py.
• Engine behavior: confirm edge overrides (e.g., romantic apology → Affection/Support) are honored.

Contribution notes
• Do not weaken the rails (public schema, 3 cards, boundaries, apology context).
• Do not invent enums or expose internal fields (packaging, luxury_grand) publicly.
• Keep changes additive in app/rules/emotion_keywords.json (Phase 1.4 policy).
• If you touch rules or tests, run the feeder smoke command above before pushing.
