ARVYAM — Deployment & Rails (Beginner-Safe)

Persona (Canonical)
- Persona name: ARVY (single source of truth)
- Environment variable: PERSONA_NAME=ARVY
- UI caps: 240 chars (short input), 500 chars (long input in future)
- Error JSON must include: "persona":"ARVY"

How to Run (Render / No CLI)
1) Add these Environment Variables in Render → Settings → Environment:
   - PERSONA_NAME=ARVY
   - ALLOWED_ORIGINS=<your frontend origin>
   - PUBLIC_ASSET_BASE=/assets
2) Click “Save changes” → “Deploy latest commit”.

Build
- pip install -r requirements.txt

Start (Render uses this)
- uvicorn app.main:app --host 0.0.0.0 --port $PORT

API Contracts (Zero-Drift)
- /api/curate (POST) body: { "prompt": "<text ≤240 chars>" }
  returns: [ { id,title,desc,image,price,currency } x3 ]
- Error shape (always structured):
  {
    "error": { "code":"INPUT_TOO_LONG", "message":"Max 240 chars" },
    "persona":"ARVY",
    "request_id":"<uuid>"
  }

Evidence (Phase 1.1)
- Env screenshot: PERSONA_NAME=ARVY
- UI screenshot: placeholder shows ARVY + 240 cap
- Commit link: “chore: Full propagation ARVY persona — UI/README”
