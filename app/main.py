import os, json
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError

ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://arvyam.com")
ASSET_BASE = os.getenv("PUBLIC_ASSET_BASE", "https://arvyam.com").rstrip("/")

app = FastAPI(title="Arvyam API (Step 3: Excel-driven curation)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = os.path.dirname(__file__)

def load_json(relpath: str):
    with open(os.path.join(HERE, relpath), "r", encoding="utf-8") as f:
        return json.load(f)

RULES: List[Dict[str, Any]] = load_json("rules.json")
CATALOG: List[Dict[str, Any]] = load_json("catalog.json")

def to_public_image(url_or_path: str) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    return f"{ASSET_BASE}{url_or_path}" if ASSET_BASE else url_or_path

def sanitize_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def classify_category(prompt: str) -> str:
    p = prompt.lower()
    best_cat = None
    best_hits = 0
    for r in RULES:
        kws = [k.lower() for k in (r.get("keywords") or [])]
        hits = sum(1 for k in kws if k and k in p)
        if hits > best_hits:
            best_cat = r["category"]
            best_hits = hits
    if best_cat:
        return best_cat
    for fallback in ["Love", "Encouragement", "Gratitude"]:
        if any(r["category"].lower() == fallback.lower() for r in RULES):
            return fallback
    return RULES[0]["category"]

def flower_priority_for_category(category: str) -> List[str]:
    for r in RULES:
        if r["category"].lower() == category.lower():
            pr = [x.title() for x in (r.get("priority") or [])]
            def boost(arr, word):
                arr2 = [x for x in arr if x.lower() != word]
                if any(x.lower() == word for x in arr):
                    arr2 = [word.title()] + arr2
                return arr2
            pr = boost(pr, "rose"); pr = boost(pr, "roses")
            pr = boost(pr, "lily"); pr = boost(pr, "lilies")
            return pr
    return []

def shape_item(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": r["id"],
        "title": r["title"],
        "desc": r["desc"],
        "image": to_public_image(r.get("image_url") or r.get("image","")),
        "price": int(r["price_inr"]),
        "currency": "INR",
    }

def pick_three(category: str) -> List[Dict[str, Any]]:
    candidates = [c for c in CATALOG if category in (c.get("tags") or [])]

    prio = flower_priority_for_category(category)
    def prio_index(flower: str) -> int:
        fl = (flower or "").title()
        return prio.index(fl) if fl in prio else 9999
    candidates.sort(key=lambda x: (prio_index(x.get("flower")), x.get("price_inr", 999999)))

    if not candidates:
        base = sorted(CATALOG, key=lambda x: x.get("price_inr", 999999))
        return [shape_item(x) for x in base[:3]]

    prices = sorted(sorted({c["price_inr"] for c in candidates}))
    if len(prices) >= 3:
        low_price, mid_price, high_price = prices[0], prices[len(prices)//2], prices[-1]
    elif len(prices) == 2:
        low_price, mid_price, high_price = prices[0], prices[0], prices[1]
    else:
        low_price = mid_price = high_price = prices[0]

    def first_at(price):
        for c in candidates:
            if c["price_inr"] == price:
                return c
        return None

    chosen = []
    for p in [low_price, mid_price, high_price]:
        item = first_at(p)
        if item and item not in chosen:
            chosen.append(item)

    for c in candidates:
        if len(chosen) >= 3: break
        if c not in chosen:
            chosen.append(c)

    return [shape_item(x) for x in chosen[:3]]

class CurateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=350)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/curate")
def curate(body: CurateIn, request):
    prompt = sanitize_text(body.prompt)
    if not prompt:
        raise HTTPException(status_code=422, detail={"error":{"code":"EMPTY_PROMPT","message":"Please write a short line."}})
    category = classify_category(prompt)
    items = pick_three(category)
    return items

@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"error":{"code":"VALIDATION_ERROR","message":"Invalid input."}})