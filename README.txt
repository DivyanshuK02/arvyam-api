ARVYAM — API (Beginner-Safe, Zero-Drift Rails)

PURPOSE
Deterministic curation API. A short user prompt returns exactly THREE products (2 MIX + 1 MONO), each mapped to one of EIGHT emotional anchors with an explicit palette[].

Rails enforced by Phase 1.4a: single public schema, family boundaries, apology relationship context, pool-size telemetry, golden harness + CI.

PERSONA
Persona name: ARVY (single source of truth)
Env var: PERSONA_NAME=ARVY

PUBLIC API (V1)
GET /health
-> 200 { "status":"ok", "persona":"ARVY", "version":"v1" }

POST /api/curate
Request: { "prompt": "string (1–500)" }
Response: ALWAYS an array of 3 items, each with public-only fields:
id, title, desc, image, price, currency, emotion, tier, packaging, mono, palette[], luxury_grand

Notes:
Only public fields. Internals like image_url, price_inr never leak (transformed in app/main.py).
Family boundaries ON; apology relationship context logged.

ZERO-DRIFT ACCEPTANCE FLOOR (FROZEN BY TESTS)
Exactly 3 items in every response.
Public schema only (no internal fields).
Family boundaries enforced; apology relationship_context logged.
Sympathy/Farewell palette guard (no celebration tokens).
Evidence line per request with: request_id, resolved_anchor, relationship_context, fallback_reason ∈ {in_family, general_in_family, duplicate_tier, cross_family_last_resort}, pool_size (pre/post by tier).

REPO LAYOUT
.
├─ app/
│ ├─ main.py (routes; public transform; request_id)
│ ├─ selection_engine.py (pure logic; boundaries; apology context; telemetry)
│ ├─ catalog.json (id/title/desc/image/price/currency/tier/…)
│ └─ rules/
│ ├─ emotion_keywords.json (enriched in Phase 1.4: synonyms, misspellings, combos)
│ ├─ sentiment_families.json
│ ├─ edge_registers.json
│ └─ tier_policy.json
├─ tests/
│ ├─ conftest.py
│ ├─ test_api_contract.py
│ ├─ test_golden_harness.py
│ ├─ test_sympathy_palette_guard.py
│ ├─ test_apology_context.py
│ ├─ test_palette_allowlist.py (per-anchor allowlists + grief/farewell block)
│ ├─ test_miner_filter.py (offline feeder miner)
│ └─ test_apply_enums.py (offline feeder enums/dupes)
├─ tools/
│ ├─ mine_unknowns.py
│ ├─ make_review_sheet.py
│ └─ apply_tokens.py (add-only; .bak; --dry-run)
├─ docs/
│ ├─ phase_status.json (phase checklist)
│ └─ llm_feeder.md (OFFLINE feeder guide; no runtime LLM)
├─ evidence/ (golden artifacts; gitignored)
├─ .github/workflows/ci.yml
├─ requirements.txt
└─ dev-requirements.txt

SETUP (LOCAL)
Python 3.11
pip install -r requirements.txt
uvicorn app.main:app --reload
Visit http://127.0.0.1:8000/health

ENV (EXAMPLES)
PERSONA_NAME=ARVY
ALLOWED_ORIGINS=https://arvyam.com
RATE_LIMIT_PER_MIN=10
API_VERSION=v1

TESTS (LOCAL)
pip install -r dev-requirements.txt
pytest -q
Golden harness outputs to ./evidence/ (CI uploads artifacts)

PALETTES & ANCHORS (ALLOWLIST CHEAT TABLE)
Use only these palette tokens per anchor in app/catalog.json. If you add a token, update tests first.
Affection/Support → pink, blush, soft-rose, pearl, rose-gold
Loyalty/Dependability → white, blue, navy, soft-grey, steel
Encouragement/Positivity → yellow, golden, soft-orange, apricot, citrus
Strength/Resilience → purple, lavender, deep-green, eucalyptus, sage
Intellect/Wisdom → cream, ivory, white, soft-beige, linen
Adventurous/Creativity → multicolor, vibrant, contrast, accent
Selflessness/Generosity → warm, amber, soft-gold, honey, caramel
Fun/Humor → bright-yellow, sunny, citrus, marigold
Celebration tokens (BLOCKED in sympathy/farewell): deep-red, crimson, gold, neon, bright, hot-pink

EVIDENCE LOG (ONE LINE / REQUEST)
event=SELECTION_EVIDENCE
request_id=UUIDv4
resolved_anchor=<one of 8>
relationship_context=romantic|familial|friendship|professional|unknown
fallback_reason ∈ {in_family, general_in_family, duplicate_tier, cross_family_last_resort}
pool_size = { pre_suppress:{classic,signature,luxury}, post_suppress:{...} }

OFFLINE FEEDER (ADD-ONLY; NO RUNTIME LLM)
Read: docs/llm_feeder.md (beginner-proof, copy-paste flow)

Reproduce the CI smoke locally (sanity check headers/enums):
python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run

PHASE STATUS (SNAPSHOT)
1.4a — Stabilization: COMPLETE
1.4 — Emotions & Palettes (Enrichment): COMPLETE
1.5 — Next phase: STARTING (see docs/phase_status.json)

CONTRIBUTING
Small PRs targeted at a single change (e.g., “feat(rulebook): add synonyms for Positivity”).
Do not change public schema or the 3-card contract without updating tests and handbook.
Update docs/phase_status.json when a task completes.

ROLLBACK (FAST PATH)
Rulebook: revert to previous .bak produced by tools/apply_tokens.py (or git revert).
Re-run pytest; confirm contract tests + palette guard + goldens.
Re-deploy.

LICENSE
Proprietary © Arvyam
