Build: pip install -r requirements.txt
Start: uvicorn app.main:app --host 0.0.0.0 --port $PORT
Env: ALLOWED_ORIGINS, PUBLIC_ASSET_BASE
