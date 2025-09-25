ARVYAM — API (Beginner-Safe, Zero-Drift Rails)

Deterministic curation API that turns a short user prompt into exactly three products
(2 MIX + 1 MONO) chosen from the catalog, each mapped to one of 8 emotional anchors
with an explicit palette[].

Stabilized by Phase 1.4a rails and 1.4 enrichment:
• Single public schema (one return path)
• Family boundaries ON by default (no grief→celebration contamination)
• Apology relationship_context routing
• Sympathy / Farewell palette guard
• Rulebook enrichment (synonyms, misspellings, combos, India-aware variants)
• Golden harness + contract tests in CI
• Offline LLM feeder (docs + tools) — no runtime LLM

Phase snapshot
• 1.4a — Stabilization: ✅ complete
• 1.4  — Emotions & Palettes (Enrichment): ✅ complete
• 1.5  — Packaging Tiers (internal rails): ✅ complete
  - Frozen mapping (internal only): Classic→Box, Signature→Vase, Luxury→PremiumBox
  - Luxury Grand is a boolean inside Luxury (not a tier)
  - Public API remains packaging-blind
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
    "image": "https://…/lilies.jpg",
    "price": 1799,
    "currency": "INR",
    "emotion": "Encouragement/Positivity",
    "tier": "Signature",
    "mono": false,
    "palette": ["yellow","orange"]
    // optional: note, edge_case, edge_type
  },
  { "... second mix ..." },
  { "... third mono ..." }
]

Never returns internal fields such as: packaging, luxury_grand, image_url, price_inr,
flowers, weight, tags, or any key starting with “_”.

Evidence logging (one JSON line per request)
• request_id (UUID4)
• resolved_anchor (one of the 8 anchors)
• relationship_context ∈ {romantic|familial|friendship|professional|unknown}
• fallback_reason ∈ {in_family|general_in_family|duplicate_tier|cross_family_last_resort}
• pool_size = { pre_suppress:{classic,signature,luxury}, post_suppress:{…} }

Zero-drift acceptance floor (frozen by tests)
• Exactly 3 items; never 2 or 4
• Public fields only (no internals)
• Family boundaries enforced; apology relationship_context logged & routed
• Sympathy/Farewell palette guard (no celebration tokens)
• Stable evidence line with the keys above
• CI must be green before merge

Repo layout (key files)
.
├─ app/
│  ├─ main.py                  # FastAPI app & routes; public transform; request_id injection
│  ├─ selection_engine.py      # Pure logic: family boundaries, apology context, triad ritual
│  └─ rules/
│     ├─ emotion_keywords.json
│     ├─ edge_registers.json
│     ├─ sentiment_families.json
│     ├─ tier_policy.json
│     ├─ tiers.json            # Phase 1.5: ["Classic","Signature","Luxury"]
│     └─ (other rulebook JSONs as applicable)
├─ tests/
│  ├─ test_api_contract.py       # Public shape + 3-card rule; forbids internal fields
│  ├─ test_golden_harness.py     # Canonical prompts; saves artifacts under evidence/
│  ├─ test_apply_enums.py        # Enum freezes incl. tiers.json
│  ├─ test_packaging_tiers.py    # Catalog mapping (Classic/Signature/Luxury) + LG rule
│  ├─ test_sympathy_palette_guard.py
│  ├─ test_family_boundaries.py
│  ├─ test_apology_context.py
│  └─ test_palette_allowlist.py
├─ tools/
│  ├─ mine_unknowns.py         # Feeder: mine unknown phrases from logs (offline)
│  ├─ make_review_sheet.py     # Feeder: candidates/proposals → review.csv
│  ├─ apply_tokens.py          # Feeder: add-only to emotion_keywords.json (+ .bak)
│  ├─ validate_rulebook.py     # Schema/enum validator for rulebook
│  └─ rotate_backups.py        # Keep newest N .bak files (local hygiene)
├─ docs/
│  ├─ llm_feeder.md            # Offline feeder guide (no runtime LLM)
│  └─ phase_status.json        # Live phase ticks
├─ evidence/                   # Golden harness outputs (gitignored)
├─ requirements.txt
└─ dev-requirements.txt

Tests & CI
Local:
pytest -q

GitHub Actions (summary):
• Install prod + dev deps
• Rulebook schema check: tools/validate_rulebook.py app/rules/emotion_keywords.json
• Feeder smoke dry-run:
  python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run
• Unit + contract tests (public schema, 3 cards, palette guard, apology context, tiers freeze, packaging tiers)
• Upload evidence bundle under artifacts

Offline feeder (docs + tools)
Guide: docs/llm_feeder.md (beginner-proof; add-only; no runtime LLM)
Reproduce CI’s smoke locally (sanity-check headers/enums):
python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run
Backups: apply_tokens.py creates timestamped .bak next to the rulebook.
Local hygiene (optional):
python tools/rotate_backups.py --keep 10

Release tags
We tag stable snapshots so you can diff/rollback confidently.
• v1.4-enrichment — palette allowlist guard, rulebook enrichment, feeder docs/tools, CI smoke/validator
• v1.5-packaging-tiers — internal packaging map + LG flag; public contract unchanged

Troubleshooting
• Import error in tests: run from repo root with an active venv.
• CI fails on feeder smoke: verify tests/data/review_smoke.csv headers and anchor enums.
• Unexpected palettes in grief/farewell: see test_sympathy_palette_guard.py and test_palette_allowlist.py.
• Engine behavior: confirm edge overrides (e.g., romantic apology → Affection/Support) per edge_registers.json.
• Public payload showing internal fields: check main.py sanitizer and ItemOut schema, and test_api_contract.py.
