import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://arvyam.com")
ASSET_BASE = os.getenv("PUBLIC_ASSET_BASE", "https://arvyam.com").rstrip("/")

def img(path):
    return f"{ASSET_BASE}{path}"

app = FastAPI(title="Arvyam API (Step 1)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CurateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=350)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/curate")
def curate(body: CurateIn):
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail={"error":{"code":"EMPTY_PROMPT","message":"Please write a short line."}})
    items = [
        {"id":"sku_x","title":"For encouragement","desc":"A bright, hopeful pick-me-up.","image":img("/assets/card-1.jpg"),"price":1299,"currency":"INR"},
        {"id":"sku_y","title":"For lasting love","desc":"Softly romantic and timeless.","image":img("/assets/card-2.jpg"),"price":1699,"currency":"INR"},
        {"id":"sku_z","title":"For gratitude","desc":"Warm, appreciative blooms.","image":img("/assets/card-3.jpg"),"price":1499,"currency":"INR"},
    ]
    return items

from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError

@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"error":{"code":"VALIDATION_ERROR","message":"Invalid input."}})