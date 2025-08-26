# Arvyam API — Step 1 (tiny API)

## Run locally
python -m venv .venv
# Windows: .venv\Scripts\activate
# mac/linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

Visit:
http://127.0.0.1:8000/health  -> {"ok": true}

## Deploy on Render
Build: pip install -r requirements.txt
Start: uvicorn main:app --host 0.0.0.0 --port $PORT
Env vars:
- ALLOWED_ORIGINS = https://arvyam.com
- PUBLIC_ASSET_BASE = https://arvyam.com
