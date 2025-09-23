BEGIN;

-- Replace whole README in one shot
DELETE FROM doc WHERE page = 'README';

INSERT INTO doc (page, section, body) VALUES
('README','Title', $$ARVYAM — API (Beginner-Safe, Zero-Drift Rails)$$),

('README','Project purpose', $$
A deterministic curation API that turns a short user prompt into exactly THREE products (2 MIX + 1 MONO)
chosen from the catalog, mapped to one of EIGHT emotional anchors with an explicit palette[].

Stabilized by Phase 1.4a rails:
• Single public schema, one return path
• Family boundaries ON + apology relationship context
• Pool-size telemetry in evidence logs
• Golden harness + contract tests in CI
$$),

('README','Live shape (public API)', $$
GET /health
→ 200 { "status":"ok", "persona":"ARVY", "version":"v1" }

POST /api/curate
Request:
{ "prompt": "string (1–500)", "context"?: { ... } }

Response (ALWAYS 3 items):
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

POST /api/checkout (stub)
Request:  { "product_id":"sku_x", "quantity"?:1 }
Response: { "checkout_url":"https://example/...", "expires_at":"...", "test":true }
$$),

('README','Zero-drift acceptance floor (frozen by tests)', $$
• Exactly 3 items; never 2 or 4
• Public fields only (no image_url, price_inr, or internals)
• Family boundaries enforced; apology relationship_context logged
• Sympathy/farewell palette guard (no celebration tokens)
• One evidence log per request: request_id, resolved_anchor, fallback_reason, pool_size
$$),

('README','Repo layout', $$
.
├─ app/
│  ├─ __init__.py
│  ├─ main.py                  # FastAPI app & routes; injects request_id; public transform
│  ├─ selection_engine.py      # pure logic: boundaries, apology context, telemetry
│  ├─ catalog.json             # public catalog
│  └─ rules/
│     ├─ emotion_keywords.json # enriched in Phase 1.4 (synonyms, misspellings, combos…)
│     ├─ sentiment_families.json
│     ├─ edge_registers.json
│     └─ tier_policy.json
├─ tests/
│  ├─ conftest.py
│  ├─ test_api_contract.py
│  ├─ test_golden_harness.py
│  ├─ test_sympathy_palette_guard.py
│  ├─ test_apology_context.py
│  ├─ test_palette_allowlist.py       # per-anchor allowlists + grief/farewell block
│  ├─ test_miner_filter.py            # feeder miner filters
│  └─ test_apply_enums.py             # feeder enum/duplicate behavior
├─ tools/
│  ├─ mine_unknowns.py
│  ├─ make_review_sheet.py
│  └─ apply_tokens.py
├─ docs/
│  ├─ phase_status.json
│  └─ llm_feeder.md                   # Offline feeder guide (no runtime LLM)
├─ evidence/                          # golden harness output (gitignored)
├─ .github/workflows/ci.yml
├─ requirements.txt
└─ dev-requirements.txt
$$),

('README','Environment (Render / local)', $$
Python: 3.11
Start local: uvicorn app.main:app --reload
Start prod:  uvicorn app.main:app --host 0.0.0.0 --port $PORT

ENV
• PERSONA_NAME=ARVY
• ALLOWED_ORIGINS=https://arvyam.com
• RATE_LIMIT_PER_MIN=10
• API_VERSION=v1
• (Optional) SUPABASE_URL, SUPABASE_KEY
$$),

('README','Quickstart', $$
pip install -r requirements.txt
uvicorn app.main:app --reload
# Visit http://127.0.0.1:8000/health
$$),

('README','Tests & evidence', $$
pip install -r dev-requirements.txt
pytest -q
# Golden harness artifacts are written to ./evidence/ and CI uploads them as artifacts
$$),

('README','Palettes & anchors (cheat table)', $$
Use only these palette tokens per anchor in app/catalog.json.
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
$$),

('README','Evidence line (per request)', $$
event: "SELECTION_EVIDENCE"
request_id: UUIDv4
resolved_anchor: one of the 8 anchors
relationship_context: romantic|familial|friendship|professional|unknown
fallback_reason: in_family|general_in_family|duplicate_tier|cross_family_last_resort
pool_size: { pre_suppress:{classic,signature,luxury}, post_suppress:{...} }
$$),

('README','Offline feeder (docs + reproduce CI smoke locally)', $$
Guide: docs/llm_feeder.md  (beginner-proof; add-only; no runtime LLM)

CI “smoke” step (run locally to sanity-check headers/enums):
python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run
$$),

('README','Phase progress (snapshot)', $$
• 1.4a — Stabilization: COMPLETE
• 1.4 — Emotions & Palettes (Enrichment): COMPLETE
• 1.5 — Kickoff: STARTING (track tasks in docs/phase_status.json)
$$),

('README','Contributing', $$
• Small, focused PRs to main
• Commit style: feat(rulebook): add synonyms for Positivity
• Never change public schema/tests casually — they are the rails
• Update docs/phase_status.json whenever a phase task changes state
$$),

('README','Rollback', $$
1) Revert rules JSON (or use the timestamped .bak created by tools/apply_tokens.py)
2) Re-run pytest; confirm contract, palette guard, golden harness
3) Re-deploy Render
$$),

('README','License', $$Proprietary © Arvyam$$);

COMMIT;
