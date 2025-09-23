ARVYAM — API (Beginner-Safe, Zero-Drift Rails)

Project purpose
---------------
A deterministic curation API that turns a short user prompt into exactly
THREE products (2 MIX + 1 MONO) chosen from the catalog, mapped to one of
EIGHT emotional anchors with an explicit palette[].

This repo is stabilized by Phase 1.4a rails:
- Single public schema, one return path
- Family boundaries ON + apology relationship context
- Pool-size telemetry in evidence logs
- Golden harness + contract tests in CI

Live shape (public API)
-----------------------
GET  /health
→ 200 { "status":"ok", "persona":"ARVY", "version":"v1" }

POST /api/curate
Request:
  { "prompt":"string (1–500)", "context"?{ ... } }
Response (ALWAYS length = 3):
  [
    {
      "id":"sku_x",
      "title":"Rose Bouquet • Signature",
      "desc":"A poised arrangement with soft blush notes.",
      "image":"https://cdn/...",
      "price":1699, "currency":"INR",
      "emotion":"Affection/Support",
      "tier":"Signature", "packaging":"Box",
      "mono":false,
      "palette":["blush","soft-pink"],
      "luxury_grand":false
    },
    { ... }, { ... }
  ]

POST /api/checkout   (stub)
Request:  { "product_id":"sku_x", "quantity"?1 }
Response: { "checkout_url":"https://example/...", "expires_at":"...", "test":true }

Zero-drift acceptance floor (frozen by tests)
---------------------------------------------
- Exactly 3 items; never 2 or 4
- Public schema only (no internal fields like image_url/price_inr)
- Family boundaries enforced; apology relationship_context logged
- Sympathy/farewell palette guard (no celebration tokens)
- Evidence line per request (request_id, resolved_anchor, fallback_reason, pool_size)

Repo layout (single source of truth)
------------------------------------
.
├─ app/
│  ├─ __init__.py
│  ├─ main.py                  # FastAPI app & routes; injects request_id; schema transform
│  ├─ selection_engine.py      # pure logic: boundaries, relationship context, telemetry
│  ├─ catalog.json             # public catalog (id/title/desc/image/price/currency/…)
│  └─ rules/
│     ├─ emotion_keywords.json # NOW being enriched in Phase 1.4
│     ├─ sentiment_families.json
│     ├─ edge_registers.json
│     └─ tier_policy.json
├─ tests/
│  ├─ conftest.py
│  ├─ test_api_contract.py
│  ├─ test_golden_harness.py
│  ├─ test_sympathy_palette_guard.py
│  └─ test_apology_context.py
├─ evidence/                   # golden harness output (gitignored)
├─ .github/workflows/ci.yml
├─ requirements.txt
├─ dev-requirements.txt
└─ docs/
   ├─ phase_status.json        # phase checklist & progress
   └─ llm_feeder.md           # Docs: see docs/llm_feeder.md (offline, add-only; no runtime LLM).

Environment (Render / local)
----------------------------
Python: 3.11
Start:  uvicorn app.main:app --host 0.0.0.0 --port $PORT

ENV:
- PERSONA_NAME=ARVY
- ALLOWED_ORIGINS=https://arvyam.com
- RATE_LIMIT_PER_MIN=10
- API_VERSION=v1
(Optionally) SUPABASE_URL, SUPABASE_KEY

Local run
---------
pip install -r requirements.txt
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/health

Tests & evidence
----------------
pip install -r dev-requirements.txt
pytest -q
# Golden harness artifacts saved under ./evidence/
# CI always uploads ./evidence/ as an artifact for audits

Palettes & anchors (cheat table)
--------------------------------
Use only these palette tokens per anchor in catalog.json.
(If you add a token, update tests first.)

1) Affection/Support         → pink, blush, soft-rose, pearl, rose-gold
2) Loyalty/Dependability     → white, blue, navy, soft-grey, steel
3) Encouragement/Positivity  → yellow, golden, soft-orange, apricot, citrus
4) Strength/Resilience       → purple, lavender, deep-green, eucalyptus, sage
5) Intellect/Wisdom          → cream, ivory, white, soft-beige, linen
6) Adventurous/Creativity    → multicolor, vibrant, contrast, accent
7) Selflessness/Generosity   → warm, amber, soft-gold, honey, caramel
8) Fun/Humor                 → bright-yellow, sunny, citrus, marigold

Celebration tokens (BLOCKED for sympathy/farewell lanes):
deep-red, crimson, gold, neon, bright, hot-pink

Evidence line (per request)
---------------------------
event: "SELECTION_EVIDENCE"
request_id: UUIDv4
resolved_anchor: one of 8 anchors
relationship_context: romantic|familial|friendship|professional|unknown
fallback_reason: in_family|general_in_family|duplicate_tier|cross_family_last_resort
pool_size: { pre_suppress:{classic,signature,luxury}, post_suppress:{...} }

Contributing (web-only workflow)
--------------------------------
- Single main branch, small PRs
- Commit style:  feat(rulebook): add synonyms for Positivity
- Never change tests or contracts casually — they are the rails
- Update docs/phase_status.json when a phase task changes state

Current phase status (snapshot)
-------------------------------
See docs/phase_status.json
- 1.4a: complete (routes unification, transform everywhere, policy sync, golden harness)
- Phase 2 rails (B1–B4): complete (boundaries default, apology context, pool telemetry, schema guard)
- 1.4 Enrichment: emotion_keywords (in_progress); llm_feeder.md (planned); golden prompts refresh (planned)

Rollback (fast path)
--------------------
1) Revert rules JSON or selection_engine.py to last green commit
2) Re-run pytest; confirm contract + palette guard + golden harness
3) Re-deploy Render
(Do not touch app/main.py unless public schema breaks)

License
-------
Proprietary © Arvyam

