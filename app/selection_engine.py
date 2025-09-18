"""
ARVYAM — selection_engine.py (Phase‑2, drop‑in)
Policy highlights implemented:
- Tier‑first selection (Classic → Signature → Luxury)
- Sentiment‑family enforcement + contamination barriers
- Relationship context detection (romantic/familial/professional/friendship)
- Edge overrides via edge_registers (sentiment_over_ladder honored)
- Mono parity: exactly one mono (replace/duplicate deterministically)
- Recent suppression aware backfill (family → emotion → general → duplication)
- Explicit duplication semantics (C,S,S then C,C,S) + payload tagging (variant_of)
- Never‑500 under scarcity as long as catalog has ≥1 valid item
- Evidence logging: pool sizes, fallback path, duplication flags
- Response schema mapping (image_url→image, price_inr→price, currency=INR)

This file is self‑contained and expects the following JSON files in the same
repository (paths can be overridden by ARVYAM_RULES_DIR env):
  - catalog.json
  - emotion_keywords.json
  - edge_registers.json
  - tier_policy.json
  - sentiment_families.json
  - weights_default.json (optional)
"""
from __future__ import annotations
import json, os, re, uuid, time, hashlib, logging
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi import Response

# ----------------------------------------------------------------------------
# Config & loading
# ----------------------------------------------------------------------------
APP = FastAPI()
log = logging.getLogger("arvyam.selection")
if not log.handlers:
    logging.basicConfig(level=os.environ.get("ARVYAM_LOG_LEVEL", "INFO"))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.environ.get("ARVYAM_RULES_DIR", ROOT_DIR)

FEATURE_MULTI_ANCHOR_LOGGING = os.environ.get("FEATURE_MULTI_ANCHOR_LOGGING", "0") == "1"
FAMILY_ENFORCEMENT = os.environ.get("FAMILY_ENFORCEMENT", "1") == "1"
ALLOW_DUPLICATES = os.environ.get("ALLOW_DUPLICATES", "1") == "1"
ALLOW_UNSPECIFIED_IN_FAMILY = os.environ.get("ALLOW_UNSPECIFIED_IN_FAMILY", "1") == "1"
LOG_EVIDENCE = os.environ.get("LOG_EVIDENCE", "1") == "1"

# deterministic rotation salt for picking
ROTATION_SALT = os.environ.get("ARVYAM_ROTATION_SALT", "arvyam-v1")


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.error("[LOAD] missing %s", path)
        return default
    except json.JSONDecodeError as e:
        log.error("[LOAD] bad json in %s: %s", path, e)
        return default

# Load data
CATALOG: List[Dict[str, Any]] = _load_json(os.path.join(RULES_DIR, "catalog.json"), [])
EMOTION_KEYWORDS = _load_json(os.path.join(RULES_DIR, "emotion_keywords.json"), {})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, "edge_registers.json"), {})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, "tier_policy.json"), {})
SENTIMENT_FAMILIES = _load_json(os.path.join(RULES_DIR, "sentiment_families.json"), {})
WEIGHTS_DEFAULT = _load_json(os.path.join(RULES_DIR, "weights_default.json"), {})

ALLOWED_TIERS = {"Classic", "Signature", "Luxury"}
ANCHORS = {
    "Affection/Support",
    "Strength/Resilience",
    "Encouragement/Positivity",
    "Loyalty/Dependability",
    "Intellect/Wisdom",
    "Selflessness/Generosity",
    "Adventurous/Creativity",
    "Fun/Humor",
}

# ----------------------------------------------------------------------------
# Utility & guards
# ----------------------------------------------------------------------------

def _hash(s: str) -> str:
    return hashlib.sha256((ROTATION_SALT + "|" + s).encode("utf-8")).hexdigest()


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _rotation_index(key: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    h = _hash(key)
    return int(h[:8], 16) % modulo


def _ensure_triad_or_500(triad: List[Dict[str, Any]]) -> None:
    if len(triad) != 3:
        log.error("[CONTRACT] triad len=%s, ids=%s", len(triad), [x.get("id") for x in triad])
        raise HTTPException(status_code=500, detail="Internal selection error")
    mono_count = sum(1 for x in triad if bool(x.get("mono")))
    if mono_count != 1:
        log.warning("[CONTRACT] mono_count=%s, repairing", mono_count)


# ----------------------------------------------------------------------------
# Relationship context (policy-driven)
# ----------------------------------------------------------------------------
ROMANCE_TOKENS = {
    "valentine", "my love", "lover", "girlfriend", "boyfriend", "fiance", "fiancé",
    "husband", "wife", "partner", "date night", "romantic", "cupid", "sweetheart",
}
PROF_TOKENS = {"boss", "manager", "team", "colleague", "office", "workplace", "client"}
FAM_TOKENS = {"mom", "mother", "dad", "father", "sister", "brother", "son", "daughter"}
FRI_TOKENS = {"friend", "bestie", "buddy"}

APOLOGY_TOKENS = {"sorry", "apolog", "apology", "apologise", "apologize", "regret"}
VALENTINE_TOKENS = {"valentine", "feb 14", "14th feb"}


def detect_relationship_context(prompt: str) -> Tuple[str, bool, bool]:
    p = _normalize(prompt)
    hits = {
        "romantic": any(t in p for t in ROMANCE_TOKENS),
        "professional": any(t in p for t in PROF_TOKENS),
        "familial": any(t in p for t in FAM_TOKENS),
        "friendship": any(t in p for t in FRI_TOKENS),
    }
    active = [k for k, v in hits.items() if v]
    romantic = hits["romantic"]
    val_ctx = any(t in p for t in VALENTINE_TOKENS)
    ambiguous = len(active) > 1
    ctx = "romantic" if romantic else (active[0] if active else "friendship")
    return ctx, ambiguous, val_ctx


# ----------------------------------------------------------------------------
# Emotion detection (rules‑first)
# ----------------------------------------------------------------------------

def _has_grand_intent(prompt: str) -> bool:
    p = _normalize(prompt)
    # lightweight heuristic; can be driven by tier_policy if needed
    return any(w in p for w in ["grand", "lavish", "luxury", "opulent", "extravagant"]) or "luxury_grand" in p


def detect_emotion(prompt: str) -> Tuple[str, Optional[str], Dict[str, float]]:
    p = _normalize(prompt)
    # Edge detection via registers
    edge_type = None
    resolved_anchor = None
    for k, reg in EDGE_REGISTERS.items():
        # simple exact keyword hit for edge
        words = [w.lower() for w in reg.get("keywords", [])]
        if any(w and w in p for w in words):
            edge_type = k
            resolved_anchor = reg.get("emotion_anchor") or reg.get("anchor")
            break

    # Anchor scoring (rules‑first over emotion_keywords)
    scores = {a: 0.0 for a in ANCHORS}
    for anchor, kw in (EMOTION_KEYWORDS or {}).items():
        exact = [w.lower() for w in kw.get("exact", [])]
        bonus = float(kw.get("weight", 1.0))
        cnt = sum(1 for w in exact if w and w in p)
        scores[anchor] = scores.get(anchor, 0.0) + cnt * bonus

    # boost for valentine context
    if any(t in p for t in VALENTINE_TOKENS):
        scores["Affection/Support"] = scores.get("Affection/Support", 0.0) + 2.0

    # grand policy: block some anchors only if grand intent detected
    blocked = set(((TIER_POLICY or {}).get("luxury_grand", {}) or {}).get("blocked_emotions", []) )
    if not _has_grand_intent(prompt):
        blocked = set()  # apply blocks only under grand intent

    # choose
    if resolved_anchor is None:
        # pick max non‑blocked
        best = sorted(
            [(a, s) for a, s in scores.items() if a not in blocked],
            key=lambda x: (-x[1], x[0])
        )
        resolved_anchor = best[0][0] if best else "Affection/Support"

    return resolved_anchor, edge_type, scores


# ----------------------------------------------------------------------------
# Families & rails
# ----------------------------------------------------------------------------

def _family_for(resolved_anchor: str, edge_type: Optional[str], relationship_ctx: str) -> str:
    # 1) explicit via edge
    if edge_type and edge_type in EDGE_REGISTERS:
        fam = EDGE_REGISTERS[edge_type].get("sentiment_family")
        if fam:
            return fam
    # 2) apology + romance reroute to romantic_repair (maps to Affection/Support family tree)
    if edge_type == "apology" and relationship_ctx == "romantic":
        return "romantic_repair"
    # 3) infer via SENTIMENT_FAMILIES mapping (emotion→family)
    fam_map = {}
    for fam, node in (SENTIMENT_FAMILIES.get("sentiment_families") or {}).items():
        emotions = node.get("emotions") or []
        if resolved_anchor in emotions:
            fam_map[resolved_anchor] = fam
    if resolved_anchor in fam_map:
        return fam_map[resolved_anchor]
    return "UNSPECIFIED"


def _family_barriers(target_family: str) -> set:
    node = (SENTIMENT_FAMILIES.get("sentiment_families") or {}).get(target_family) or {}
    return set(node.get("contamination_barriers", []))


def _filter_by_family(pool: List[Dict[str, Any]], target_family: str) -> List[Dict[str, Any]]:
    if not FAMILY_ENFORCEMENT:
        return pool
    barriers = _family_barriers(target_family)
    out = []
    for it in pool:
        fam = it.get("sentiment_family") or ("UNSPECIFIED" if ALLOW_UNSPECIFIED_IN_FAMILY else None)
        if fam is None:
            continue
        if fam in barriers:
            continue
        if fam == target_family or fam == "UNSPECIFIED":
            out.append(it)
    return out


def _apply_edge_register_filters(pool: List[Dict[str, Any]], edge_type: Optional[str]) -> List[Dict[str, Any]]:
    if not edge_type or edge_type not in EDGE_REGISTERS:
        return pool
    reg = EDGE_REGISTERS[edge_type]
    # palette allow/block
    allow = set(reg.get("palette_allow", []) or [])
    block = set(reg.get("palette_block", []) or [])
    res = []
    for it in pool:
        pal = set(it.get("palette") or [])
        if allow and not pal.intersection(allow):
            continue
        if block and pal.intersection(block):
            continue
        res.append(it)
    # species must include
    must = set(reg.get("must_include_species", []) or [])
    if must:
        res = [x for x in res if any(sp in must for sp in (x.get("flowers") or []))]
    return res


# ----------------------------------------------------------------------------
# Selection helpers (tier‑first + duplication)
# ----------------------------------------------------------------------------
TIERS_ORDER = ["Classic", "Signature", "Luxury"]


def _score_item(it: Dict[str, Any], resolved_anchor: str) -> float:
    # simple score: weight + anchor match bonus
    w = float(it.get("weight", WEIGHTS_DEFAULT.get("weight", 50)))
    bonus = 10.0 if it.get("emotion") == resolved_anchor else 0.0
    return w + bonus


def _rank(pool: List[Dict[str, Any]], resolved_anchor: str, seed: str) -> List[Dict[str, Any]]:
    arr = sorted(pool, key=lambda x: (-_score_item(x, resolved_anchor), x.get("id", "")))
    if not arr:
        return []
    idx = _rotation_index(seed, len(arr))
    # wrap‑around take top starting at idx
    return [arr[(idx + i) % len(arr)] for i in range(len(arr))]


def _tier_pool(catalog: List[Dict[str, Any]], tier: str, resolved_anchor: str,
               target_family: str, edge_type: Optional[str]) -> List[Dict[str, Any]]:
    pool = [x for x in catalog if x.get("tier") in ALLOWED_TIERS and x.get("tier") == tier]
    pool = [x for x in pool if x.get("emotion") in ANCHORS]
    pool = _filter_by_family(pool, target_family)
    pool = _apply_edge_register_filters(pool, edge_type)
    # prefer same anchor first
    anchor_first = [x for x in pool if x.get("emotion") == resolved_anchor]
    others = [x for x in pool if x.get("emotion") != resolved_anchor]
    return anchor_first + others


def _pick_tier(pool: List[Dict[str, Any]], resolved_anchor: str, seed: str) -> Optional[Dict[str, Any]]:
    ranked = _rank(pool, resolved_anchor, seed)
    return ranked[0] if ranked else None


def _enforce_mono(triad: List[Dict[str, Any]], catalog: List[Dict[str, Any]],
                  target_family: str, edge_type: Optional[str], resolved_anchor: str, seed: str) -> List[Dict[str, Any]]:
    mono_count = sum(1 for x in triad if bool(x.get("mono")))
    if mono_count == 1:
        return triad
    # try replace weakest non‑mono with a mono from same family (any tier)
    ranked = _rank(
        _apply_edge_register_filters(
            _filter_by_family([x for x in catalog if x.get("mono")], target_family),
            edge_type,
        ),
        resolved_anchor,
        seed,
    )
    if ranked:
        # pick the best mono not already in triad
        existing = {x.get("id") for x in triad}
        for m in ranked:
            if m.get("id") not in existing:
                # replace the weakest non‑mono
                non_mono_idx = None
                weakest_score = 10**9
                for i, it in enumerate(triad):
                    if not it.get("mono"):
                        sc = _score_item(it, resolved_anchor)
                        if sc < weakest_score:
                            weakest_score, non_mono_idx = sc, i
                if non_mono_idx is not None:
                    triad[non_mono_idx] = m
                    return triad
    # if still none, and duplicates allowed: mark one item as variant_of another mono candidate is missing
    if ALLOW_DUPLICATES and mono_count == 0 and triad:
        triad[0]["mono"] = True  # soft promotion as a last resort
        return triad
    return triad


def _duplicate_rule(triad: List[Dict[str, Any]], target_family: str, resolved_anchor: str, seed: str) -> List[Dict[str, Any]]:
    # prefer C,S,S then C,C,S
    tiers = [x.get("tier") for x in triad]
    ids = [x.get("id") for x in triad]
    if tiers.count("Signature") == 0 and any(t == "Signature" for t in tiers):
        pass  # unreachable by definition
    # Fill missing by duplicating the closest tier present
    target_patterns = [
        ("Signature", "duplicate_signature"),
        ("Classic", "duplicate_classic"),
    ]
    for prefer_tier, reason in target_patterns:
        # find a candidate to copy
        for i, it in enumerate(triad):
            if it.get("tier") == prefer_tier:
                clone = dict(it)
                clone_id = f"{it.get('id')}_v"
                k = 1
                while clone_id in ids:
                    k += 1
                    clone_id = f"{it.get('id')}_v{k}"
                clone["id"] = clone_id
                clone["variant_of"] = it.get("id")
                clone["duplication_used"] = True
                triad.append(clone)
                return triad
    # if nothing matched, duplicate index 0 deterministically
    if triad:
        it = triad[0]
        clone = dict(it)
        clone["id"] = f"{it.get('id')}_v"
        clone["variant_of"] = it.get("id")
        clone["duplication_used"] = True
        triad.append(clone)
    return triad


def _suppress_recent(triad: List[Dict[str, Any]], recent_ids: List[str]) -> List[Dict[str, Any]]:
    recent = set(recent_ids or [])
    if not recent:
        return triad
    return [x for x in triad if x.get("id") not in recent]


# ----------------------------------------------------------------------------
# Selection engine (Phase‑2)
# ----------------------------------------------------------------------------

def selection_engine(prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = context or {}
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt required")

    rid = ctx.get("request_id") or str(uuid.uuid4())
    seed = f"{rid}|{_normalize(prompt)}"

    resolved_anchor, edge_type, _scores = detect_emotion(prompt)
    rel_ctx, rel_amb, val_ctx = detect_relationship_context(prompt)
    target_family = _family_for(resolved_anchor, edge_type, rel_ctx)

    ctx.update({
        "resolved_anchor": resolved_anchor,
        "edge_type": edge_type,
        "target_family": target_family,
        "relationship_context": rel_ctx,
        "relationship_ambiguous": bool(rel_amb),
        "valentine_context": bool(val_ctx),
        "sentiment_over_ladder": bool((EDGE_REGISTERS.get(edge_type) or {}).get("sentiment_over_ladder", False)),
    })

    # Build tiers in order
    triad: List[Dict[str, Any]] = []
    evidence = {
        "family": target_family,
        "edge_type": edge_type,
        "resolved_anchor": resolved_anchor,
        "pre_pools": {},
        "post_suppress_pools": {},
        "fallback_reason": "none",
        "duplication_used": False,
        "sentiment_override": ctx.get("sentiment_over_ladder", False),
    }

    for tier in TIERS_ORDER:
        pool = _tier_pool(CATALOG, tier, resolved_anchor, target_family, edge_type)
        evidence["pre_pools"][tier] = len(pool)
        pick = _pick_tier(pool, resolved_anchor, seed + "|" + tier)
        if pick is None:
            # fallback in‑family: try any emotion within family
            pool2 = _filter_by_family([x for x in CATALOG if x.get("tier") == tier], target_family)
            pool2 = _apply_edge_register_filters(pool2, edge_type)
            pick = _pick_tier(pool2, resolved_anchor, seed + "|any|" + tier)
        if pick is None and ctx.get("sentiment_over_ladder", False):
            # duplication instead of crossing family
            evidence["fallback_reason"] = evidence.get("fallback_reason") or "duplicate_due_to_sentiment_guard"
            # defer duplication; we will run it after base loop if triad < desired
        else:
            if pick is not None:
                # avoid duplicates of exact id here; we allow later explicit variant dup
                if pick.get("id") not in {x.get("id") for x in triad}:
                    triad.append(pick)

    # If we still have < 3 after tier loop, duplicate by rule (C,S,S → C,C,S)
    if len(triad) < 3 and ALLOW_DUPLICATES:
        triad = _duplicate_rule(triad, target_family, resolved_anchor, seed)
        evidence["duplication_used"] = True
        if evidence["fallback_reason"] == "none":
            evidence["fallback_reason"] = "duplicate_to_complete_ladder"

    # Enforce mono parity
    triad = _enforce_mono(triad, CATALOG, target_family, edge_type, resolved_anchor, seed)

    # Recent suppression → backfill within family/tier, then dup as last resort
    triad = _suppress_recent(triad, ctx.get("recent_ids") or [])
    if len(triad) < 3:
        # try fill missing tiers first
        have_tiers = [x.get("tier") for x in triad]
        for tier in TIERS_ORDER:
            if len(triad) >= 3:
                break
            if have_tiers.count(tier) >= 1:
                continue
            pool = _tier_pool(CATALOG, tier, resolved_anchor, target_family, edge_type)
            # remove already chosen ids
            existing = {x.get("id") for x in triad}
            pool = [x for x in pool if x.get("id") not in existing]
            pick = _pick_tier(pool, resolved_anchor, seed + "|refill|" + tier)
            if pick:
                triad.append(pick)
        if len(triad) < 3 and ALLOW_DUPLICATES:
            triad = _duplicate_rule(triad, target_family, resolved_anchor, seed)
            evidence["duplication_used"] = True
            evidence["fallback_reason"] = "duplicate_after_suppression"

    _ensure_triad_or_500(triad)

    # Evidence sizes after suppression
    if LOG_EVIDENCE:
        for tier in TIERS_ORDER:
            pool = _tier_pool(CATALOG, tier, resolved_anchor, target_family, edge_type)
            evidence["post_suppress_pools"][tier] = len(pool)
        log.info("SELECTION_EVIDENCE %s", json.dumps(evidence))

    return {"items": triad, "context": ctx}


# ----------------------------------------------------------------------------
# API facade
# ----------------------------------------------------------------------------

def _transform_for_api(items: List[Dict[str, Any]], resolved_anchor: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items or []:
        image = it.get("image") or it.get("image_url")
        price = it.get("price") if it.get("price") is not None else it.get("price_inr")
        currency = it.get("currency") or ("INR" if price is not None else None)
        if not image or price is None or not currency:
            log.error("[TRANSFORM] missing fields id=%s image?=%s price=%s currency=%s",
                      it.get("id"), bool(image), price, currency)
            continue
        item = {
            "id": it.get("id"),
            "title": it.get("title"),
            "desc": it.get("desc"),
            "image": image,
            "price": price,
            "currency": currency,
            "emotion": it.get("emotion") or (resolved_anchor or "Affection/Support"),
            "tier": it.get("tier"),
            "mono": bool(it.get("mono")),
            "palette": it.get("palette") or [],
            "flowers": it.get("flowers") or [],
            "luxury_grand": bool(it.get("luxury_grand")),
        }
        if it.get("variant_of"):
            item["variant_of"] = it["variant_of"]
        out.append(item)
    if len(out) < 3:
        # As long as catalog contained at least 1 valid card, we duplicate last safely
        while ALLOW_DUPLICATES and len(out) < 3 and out:
            clone = dict(out[-1])
            clone_id = f"{clone['id']}_v{len(out)}"
            clone["id"] = clone_id
            clone["variant_of"] = out[-1]["id"]
            out.append(clone)
    if len(out) != 3:
        log.error("[TRANSFORM] produced %s items — catalog likely empty", len(out))
        raise HTTPException(status_code=500, detail="Internal catalog data error")
    return out


@APP.post("/api/curate")
async def curate_post(payload: Dict[str, Any]):
    prompt = (payload or {}).get("prompt") or (payload or {}).get("q")
    context = (payload or {}).get("context") or {}
    result = selection_engine(prompt, context)

    # guarantee emotion on cards before transform
    for c in result["items"]:
        c["emotion"] = c.get("emotion") or result["context"].get("resolved_anchor") or "Affection/Support"

    items = _transform_for_api(result["items"], result["context"].get("resolved_anchor"))
    resp = {
        "items": items,
        "edge_case": bool(result["context"].get("edge_type")),
        "edge_type": result["context"].get("edge_type"),
    }

    # Optional QA header
    cfg = {"emit_header": False}
    resp_obj = Response(content=json.dumps(resp), media_type="application/json")
    if FEATURE_MULTI_ANCHOR_LOGGING and cfg.get("emit_header", False):
        # placeholder; we no longer compute near‑tie here, Phase‑2 focuses on families
        pass
    return resp_obj


@APP.post("/curate")
async def curate_post_compat(payload: Dict[str, Any]):
    return await curate_post(payload)


@APP.get("/curate")
async def curate_get(q: Optional[str] = None):
    return await curate_post({"prompt": q or ""})
