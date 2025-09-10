# app/selection_engine.py
# Ultra-beginner-safe Selection Engine
# - Rules-first (no AI classification): keywords, substitutions, weights
# - Deterministic weighted picks
# - Always returns exactly 3 items: 2 MIX + 1 MONO (MONO = iconic only by default)
# - Palette-aware; includes adjacent-emotion MIX fallback with palette matching
# - Respects pricing floors and Luxury-Grand (LG) policy (LG is a flag inside Luxury)
# - Gentle context boosts (budget, packaging); never hard-filters valid choices

from __future__ import annotations
import json, os, re, hashlib
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

ROOT = os.path.dirname(__file__)
def _p(*parts): return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

# -----------------------------------------------------------------------------
# Load rule tables (these are the ONLY sources of truth for behavior)
# -----------------------------------------------------------------------------

EMO = _load_json(_p("rules","emotion_keywords.json"), {"anchors": [], "exact": {}, "keywords": {}})
SUBS = _load_json(_p("rules","substitution_map.json"), {})
WEI  = _load_json(_p("rules","weights_default.json"), {"flower_weights":{}, "emotion_bias":{}})
PRICE= _load_json(_p("rules","pricing_policy.json"), {"currency":"INR","floors":{},"luxury_grand_floor":4999})
TIER = _load_json(_p("rules","tier_policy.json"), {"luxury_grand":{"allowed_emotions":[],"blocked_emotions":[],"max_per_triad":1,"allow_in_mix":True,"allow_in_mono":True}})

# Catalog: you must provide app/catalog.json with required fields
CATALOG: List[Dict[str,Any]] = _load_json(_p("catalog.json"), [])

# -----------------------------------------------------------------------------
# Constants & simple guides
# -----------------------------------------------------------------------------

ICONIC = {"rose","lily","orchid"}  # Only these are allowed for MONO by default
CURRENCY = PRICE.get("currency","INR")

# Adjacent emotion lanes to borrow MIX from when the primary lane is thin
ADJACENT_EMOTIONS: Dict[str, List[str]] = {
    "Romance":       ["Gratitude", "Birthday", "Friendship"],
    "Sympathy":      ["GetWell", "Encouragement"],
    "Celebration":   ["Birthday", "Gratitude"],
    "Encouragement": ["Friendship", "GetWell"],
    "Gratitude":     ["Romance", "Friendship"],
    "Friendship":    ["Encouragement", "Birthday"],
    "GetWell":       ["Sympathy", "Encouragement"],
    "Birthday":      ["Celebration", "Friendship"]
}

# Palette guidance per emotion. These are soft filters for adjacent-emotion fallback.
EMOTION_PALETTE_GUIDE: Dict[str, List[str]] = {
    "Romance":      ["blush","soft-pink","red","rose","cream","ivory"],
    "Sympathy":     ["white","ivory","cream","soft-green","pastel"],
    "Celebration":  ["bright","yellow","orange","fuchsia","vibrant","multicolor"],
    "Encouragement":["warm","yellow","orange","peach","sunny"],
    "Gratitude":    ["soft-pink","peach","cream","pastel"],
    "Friendship":   ["yellow","sunny","bright","cheerful"],
    "GetWell":      ["white","green","soft","pastel","peach"],
    "Birthday":     ["bright","pink","purple","yellow","fun"]
}

# -----------------------------------------------------------------------------
# Text & weight helpers
# -----------------------------------------------------------------------------

def normalize(text: str) -> str:
    if not text: return ""
    t = text.strip().lower()
    t = re.sub(r"https?://\\S+","", t)                         # strip URLs
    t = re.sub(r"[\\w.+-]+@[\\w-]+\\.[\\w.-]+","", t)          # strip emails
    t = re.sub(r"\\s+"," ", t)                                 # collapse spaces
    return t

def _hash01(seed: str) -> float:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    n = int(h[:8], 16) % 10000
    return n / 10000.0

def detect_emotion(norm: str, context: Optional[Dict[str,Any]]) -> str:
    # exact phrase beats keywords
    for phrase, emo in (EMO.get("exact") or {}).items():
        if phrase in norm: return emo
    counts = {a:0 for a in (EMO.get("anchors") or [])}
    for kw, emo in (EMO.get("keywords") or {}).items():
        if kw in norm: counts[emo] = counts.get(emo,0) + 1
    if counts:
        best = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0]
        if best[1] > 0: return best[0]
    if context and context.get("emotion_hint") in (EMO.get("anchors") or []):
        return context["emotion_hint"]
    return "Encouragement"  # default support lane

def weight_for_item(item: Dict[str,Any], emotion: str) -> int:
    flowers = item.get("flowers") or []
    primary = flowers[0] if flowers else "rose"
    base = WEI.get("flower_weights",{}).get(primary, 50)
    bias = WEI.get("emotion_bias",{}).get(emotion,{}).get(primary, base)
    return max(1, round((base + bias)/2))

def pick_deterministic(pool: List[Dict[str,Any]], k: int, seed: str, emotion: str) -> List[Dict[str,Any]]:
    scored = []
    for it in pool:
        w = weight_for_item(it, emotion)
        rnd = _hash01(f"{seed}::{it['id']}")
        score = rnd / (w/100.0)  # lower is better
        scored.append((score, it))
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, it in scored:
        if len(out) >= k: break
        if it["id"] in seen: continue
        out.append(it); seen.add(it["id"])
    return out

# -----------------------------------------------------------------------------
# Policy helpers (pricing floors, LG policy, palettes, context boosts)
# -----------------------------------------------------------------------------

def passes_pricing_floor(item: Dict[str,Any]) -> bool:
    floors = PRICE.get("floors", {})
    lg_floor = int(PRICE.get("luxury_grand_floor", 4999))
    tier = item.get("tier","Classic")
    if item.get("luxury_grand"):
        return int(item.get("price_inr", 0)) >= lg_floor
    return int(item.get("price_inr", 0)) >= int(floors.get(tier, 0))

def lg_allowed_for_emotion(item: Dict[str,Any], emotion: str) -> bool:
    if not item.get("luxury_grand"): return True
    lg = TIER.get("luxury_grand", {})
    if emotion in lg.get("blocked_emotions", []): return False
    allowed = lg.get("allowed_emotions", [])
    return (emotion in allowed) if allowed else True

def _has_palette(item: Dict[str,Any]) -> bool:
    pal = item.get("palette")
    return isinstance(pal, list) and len(pal) > 0

def _palette_match(tokens: List[str], desired: List[str]) -> bool:
    if not tokens or not desired: return False
    tset = {t.lower() for t in tokens}
    dset = {d.lower() for d in desired}
    return len(tset.intersection(dset)) > 0

def apply_context_preferences(pool: List[Dict[str,Any]], context: Optional[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not context: return pool[:]
    budget = context.get("budget_inr")
    pref_pack = (context.get("packaging_pref") or "").lower()
    def score(x):
        s = 0
        if pref_pack and (x.get("packaging","").lower()==pref_pack): s += 10
        if isinstance(budget,(int,float)):
            s += 6 if x.get("price_inr",10**9) <= budget else -2
        return s
    return sorted(pool, key=score, reverse=True)

# -----------------------------------------------------------------------------
# Intent & redirection helpers
# -----------------------------------------------------------------------------

def strong_intent_override(norm: str) -> Dict[str,Any]:
    out = {"forceIconic": None, "forbidMono": None, "note": None}
    m = re.search(r"only\\s+(roses?|lilies|orchids?)", norm)
    if m:
        w = m.group(1)
        if w.startswith("rose"): out["forceIconic"] = "rose"
        elif w.startswith("lil"): out["forceIconic"] = "lily"
        elif w.startswith("orchid"): out["forceIconic"] = "orchid"
    m2 = re.search(r"mono\\s+([a-z]+)", norm)
    if m2:
        flower = m2.group(1)
        if flower not in ICONIC:
            out["forbidMono"] = flower
            out["note"] = f"Requested mono {flower} is unavailable; offering nearest alternatives."
    return out

def editorial_redirection(norm: str) -> Optional[Dict[str,Any]]:
    for k, subs in (SUBS or {}).items():
        if k in norm:
            return {
                "redirectedFrom": k,
                "substitutes": subs,
                "note": f"“{k}” is unavailable/seasonal. Redirected to: {', '.join(subs)}."
            }
    return None

# -----------------------------------------------------------------------------
# Candidate pools + adjacent-emotion fallback
# -----------------------------------------------------------------------------

def _base_filter(item: Dict[str,Any], emotion: str) -> bool:
    return (
        item.get("emotion")==emotion
        and passes_pricing_floor(item)
        and lg_allowed_for_emotion(item, emotion)
        and _has_palette(item)
    )

def _adjacent_mix_candidates(emotion: str) -> List[Dict[str,Any]]:
    desired = EMOTION_PALETTE_GUIDE.get(emotion, [])
    cands: List[Dict[str,Any]] = []
    for adj in ADJACENT_EMOTIONS.get(emotion, []):
        for x in CATALOG:
            if x.get("mono"): 
                continue
            if not _base_filter(x, adj):
                continue
            if not _palette_match(x.get("palette") or [], desired):
                continue
            cands.append(x)
    return cands

def candidate_pools(emotion: str, context: Optional[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    lgp = TIER.get("luxury_grand", {})
    allow_mix  = lgp.get("allow_in_mix",  True)
    allow_mono = lgp.get("allow_in_mono", True)

    base = [x for x in CATALOG if _base_filter(x, emotion)]

    # MIX and MONO (MONO = iconic only)
    mix  = [x for x in base if not x.get("mono")]
    mono = [x for x in base if x.get("mono") and any(f in ICONIC for f in (x.get("flowers") or []))]

    # Enforce LG lane settings on each pool
    if not allow_mix:
        mix  = [x for x in mix  if not x.get("luxury_grand")]
    if not allow_mono:
        mono = [x for x in mono if not x.get("luxury_grand")]

    # Adjacent-emotion fallback for MIX when thin
    if len(mix) < 2:
        adj = _adjacent_mix_candidates(emotion)
        # de-dup by id while preserving order
        seen = {m["id"] for m in mix}
        for it in adj:
            if it["id"] not in seen:
                mix.append(it); seen.add(it["id"])

    # Standard fallbacks
    if not mono:
        mono = [x for x in CATALOG
                if x.get("mono")
                and any(f in ICONIC for f in (x.get("flowers") or []))
                and passes_pricing_floor(x)
                and lg_allowed_for_emotion(x, emotion)
                and _has_palette(x)]
        if not allow_mono:
            mono = [x for x in mono if not x.get("luxury_grand")]

    if not mix:
        mix = [x for x in CATALOG
               if not x.get("mono")
               and any(f in ICONIC for f in (x.get("flowers") or []))
               and passes_pricing_floor(x)
               and lg_allowed_for_emotion(x, emotion)
               and _has_palette(x)]
        if not allow_mix:
            mix = [x for x in mix if not x.get("luxury_grand")]

    mix  = apply_context_preferences(mix,  context)
    mono = apply_context_preferences(mono, context)
    return mix, mono

# -----------------------------------------------------------------------------
# Output mapping, variety & caps
# -----------------------------------------------------------------------------

def _map_out(it: Dict[str,Any], note: Optional[str]=None) -> Dict[str,Any]:
    out = {
        "id": it["id"],
        "title": it["title"],
        "desc": it["desc"],
        "image": it["image_url"],
        "price": it["price_inr"],
        "currency": CURRENCY,
        "emotion": it["emotion"],
        "tier": it["tier"],
        "packaging": it["packaging"],
        "mono": bool(it.get("mono")),
        "palette": it.get("palette") or [],
        "luxury_grand": bool(it.get("luxury_grand"))
    }
    if note and not out["mono"]:
        out["note"] = note
    return out

def try_tier_variety(triad: List[Dict[str,Any]], mix_pool: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    tiers = {t["tier"] for t in triad}
    if len(tiers) >= 2: return triad
    current = triad[0]["tier"]
    used = {t["id"] for t in triad}
    alt = next((x for x in mix_pool if x["tier"] != current and x["id"] not in used), None)
    if alt:
        triad[1] = _map_out(alt, note=triad[1].get("note"))
    return triad

def enforce_lg_cap(triad: List[Dict[str,Any]], mix_pool: List[Dict[str,Any]], mono_pool: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    lgp = TIER.get("luxury_grand", {})
    cap = int(lgp.get("max_per_triad", 1))
    idxs = [i for i,t in enumerate(triad) if t.get("luxury_grand")]
    if len(idxs) <= cap: return triad

    used = {t["id"] for t in triad}
    need = len(idxs) - cap
    order = [i for i in idxs if not triad[i]["mono"]] + [i for i in idxs if triad[i]["mono"]]
    for i in order:
        if need <= 0: break
        pool = mix_pool if not triad[i]["mono"] else mono_pool
        rep = next((x for x in pool if not x.get("luxury_grand") and x["id"] not in used), None)
        if rep:
            triad[i] = _map_out(rep)
            used.add(rep["id"])
            need -= 1
    if sum(1 for t in triad if t.get("luxury_grand")) > cap:
        raise ValueError("Luxury-Grand cap exceeded; add non-LG alternatives or relax policy.")
    return triad

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_output(items: List[Dict[str,Any]]) -> None:
    if not isinstance(items, list) or len(items)!=3:
        raise ValueError("Output must contain exactly 3 items.")
    if sum(1 for x in items if x["mono"]) != 1:
        raise ValueError("Exactly one MONO item is required.")
    ids=set()
    req = ["id","title","desc","image","price","currency","emotion","tier","packaging","mono","palette","luxury_grand"]
    for it in items:
        for k in req:
            v = it.get(k, None)
            if v is None or (k=="palette" and not v):
                raise ValueError(f"Missing/empty field {k} on {it.get('id','unknown')}")
        if it["id"] in ids: raise ValueError("Duplicate SKU in triad.")
        ids.add(it["id"])

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def selection_engine(prompt: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
    """
    Input:
      - prompt: string (1–500 chars)
      - context (optional): { emotion_hint?, budget_inr?, packaging_pref?, locale? }
    Output:
      - list of 3 curated items: exactly 2 MIX + 1 MONO, palette[] present on each
    """
    context = context or {}
    norm = normalize(prompt)
    emotion = detect_emotion(norm, context)

    mix_pool, mono_pool = candidate_pools(emotion, context)

    intent = strong_intent_override(norm)
    redir  = editorial_redirection(norm)
    note = redir["note"] if redir else intent.get("note")

    # Picks
    mix_picks = pick_deterministic(mix_pool, 2, seed=f"{emotion}/{norm}/mix", emotion=emotion)

    mono_pick = None
    if intent.get("forceIconic"):
        mono_pick = next((x for x in mono_pool if intent["forceIconic"] in (x.get("flowers") or [])), None)

    if not mono_pick and mono_pool:
        mono_pick = pick_deterministic(mono_pool, 1, seed=f"{emotion}/{norm}/mono", emotion=emotion)[0]
    elif not mono_pick:
        # Last-resort: borrow any iconic MONO from wider catalog (policy-safe)
        alt_mono = [x for x in CATALOG
                    if x.get("mono")
                    and any(f in ICONIC for f in (x.get("flowers") or []))
                    and passes_pricing_floor(x)
                    and lg_allowed_for_emotion(x, emotion)
                    and _has_palette(x)]
        if alt_mono:
            mono_pick = pick_deterministic(alt_mono, 1, seed=f"{emotion}/{norm}/mono2", emotion=emotion)[0]

    raw: List[Dict[str,Any]] = []
    for it in mix_picks:
        raw.append(_map_out(it, note=note))
    if mono_pick:
        raw.append(_map_out(mono_pick))  # mono never gets note

    if len(raw) != 3:
        raise ValueError("Could not assemble 3 items; check catalog coverage and rules.")

    raw = try_tier_variety(raw, mix_pool)
    raw = enforce_lg_cap(raw, mix_pool, mono_pool)
    validate_output(raw)
    return raw
