ARVYAM — API (Beginner-Safe, Zero-Drift Rails)

Deterministic curation API that turns a short user prompt into exactly three products (2 MIX + 1 MONO) chosen from the catalog, each mapped to one of 8 emotional anchors with an explicit palette[].

Stabilized by Phase 1.4a rails and Phase 1.4 enrichment:
Single public schema (one return path)
Family boundaries default ON
Apology relationship_context routing
Sympathy/Farewell palette guard
Rulebook enrichment (synonyms, misspellings, combos, India-aware variants)
Golden harness + contract tests in CI
Offline LLM feeder (docs + tools) — no runtime LLM

Phase snapshot
1.4a — Stabilization: ✅ complete
1.4 — Emotions & Palettes (Enrichment): ✅ complete
1.5 — Kickoff: 🚀 starting (track in docs/phase_status.json)
See: docs/phase_status.json for live ticks.

Quickstart (local)
# 1) Create venv (any Python 3.11+)
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
    "palette": ["yellow","orange"],
    "mono": false
  },
  { "... second mix ..." },
  { "... third mono ..." }
]

Never returns internal fields such as image_url or price_inr.
Evidence logging (single JSON line per request)
request_id (UUID4)
resolved_anchor (one of the 8 anchors)
relationship_context = romantic|familial|friendship|professional|unknown
fallback_reason = in_family|general_in_family|duplicate_tier|cross_family_last_resort
pool_size = { pre_suppress:{classic,signature,luxury}, post_suppress:{...} }
(Used for analytics and catalog health.)

Zero-drift acceptance floor (frozen by tests)
Exactly 3 items; never 2 or 4
Public fields only (no internals)
Family boundaries enforced; apology relationship_context logged & routed
Sympathy/Farewell palette guard (no celebration tokens)
Stable evidence line with the keys above
CI must be green before merge

Repo layout
.
├─ app/
│  ├─ main.py                  # FastAPI app & routes; public transform; request_id injection
│  ├─ selection_engine.py      # Pure logic: family boundaries, apology context, triad contract
│  └─ rules/
│     ├─ emotion_keywords.json # Enriched rulebook (synonyms, misspellings, combos)
│     ├─ edge_registers.json   # Edges (apology/sympathy/…); relationship overrides
│     ├─ sentiment_families.json
│     └─ tier_policy.json
├─ tests/
│  ├─ test_public_schema.py    # Public shape + 3-card rule
│  ├─ test_golden_harness.py   # Canonical prompts; artifacts saved under evidence/
│  ├─ test_sympathy_palette_guard.py
│  ├─ test_apology_context.py
│  └─ test_palette_allowlist.py # Anchor → palette allowlist guard
├─ tools/
│  ├─ mine_unknowns.py         # Feeder: mine unknown phrases from logs (offline)
│  ├─ make_review_sheet.py     # Feeder: candidates/proposals → review.csv
│  ├─ apply_tokens.py          # Feeder: add-only to emotion_keywords.json (+ .bak)
│  ├─ validate_rulebook.py     # Schema/enum validator for rulebook
│  └─ rotate_backups.py        # Keep newest N .bak files (local hygiene)
├─ docs/
│  ├─ llm_feeder.md            # Offline feeder guide (no runtime LLM)
│  └─ phase_status.json        # Live phase ticks
├─ evidence/                   # Golden harness artifacts (gitignored)
├─ requirements.txt
└─ dev-requirements.txt

Tests & CI
Local:
pytest -q

What CI runs (GitHub Actions):
Unit + contract tests (public schema, 3 cards, palette guard, apology context)
Palette allowlist gate (anchor → allowed palette tokens)
Rulebook validator (tools/validate_rulebook.py)
Feeder smoke (see below)
Upload of evidence artifacts (golden harness outputs) on every run
Offline feeder (docs + tools)
Guide: docs/llm_feeder.md (beginner-proof; add-only; no runtime LLM).
Reproduce CI’s smoke locally (sanity-check headers/enums):
python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run

Backups: tools/apply_tokens.py creates timestamped .bak next to the rulebook.
Local hygiene (optional):
python tools/rotate_backups.py --keep 10

Release tags
We tag stable snapshots so you can diff/rollback confidently.
v1.4-enrichment — palette allowlist guard, rulebook enrichment, feeder docs/tools, CI smoke/validator.

Troubleshooting
Import error in tests (cannot find app): run from repo root with an active venv.
CI fails on feeder smoke: verify tests/data/review_smoke.csv headers and anchor enums.
Unexpected palettes in grief/farewell: see test_sympathy_palette_guard.py and test_palette_allowlist.py.
Engine behavior: confirm edge overrides (e.g., romantic apology → Affection/Support) are honored in selection_engine.py per edge_registers.json.

Contribution notes
Keep changes additive in app/rules/emotion_keywords.json (Phase 1.4 policy).
Do not weaken the rails (public schema, 3 cards, boundaries, apology context).
If you touch rules or tests, run the feeder smoke command above before pushing.
