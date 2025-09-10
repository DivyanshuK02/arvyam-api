# app/selection_engine.py
# Ultra-beginner-safe, rules-first, 2 MIX + 1 MONO, palette-aware.

from __future__ import annotations
import json, os, re, hashlib
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(__file__)
def _p(*parts): return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

# ---- Load rules (already in your repo) --------------------------------------
EMO = _load_json(_p("rules","emotion_keywords.json"), {"anchors": [], "exact": {}, "keywords": {}})
SUBS = _load_json(_p("rules","substitution_map.json"), {})
WEI  = _load_json(_p("rules","weights_default.json"), {"flower_weights":{}, "emotion_bias":{}})
PRICE= _load_json(_p("rules","pricing_policy.json"), {"currency":"INR","floors":{},"luxury_grand_floor":4999})
TIER = _load_json(_p("rules","tier_policy.json"), {"luxury_grand":{"allowed_emotions":[],"blocked_emotions":[],"max_per_triad":1,"allow_in_mix":True,"allow_in_mono":True}})

# ---- Load catalog -----------------------------------------------------------
CATALOG: List[Dict[str,Any]] = _load_json(_p("catalog.json"), [])

ICONIC = {"rose","lily","orchid"}  # R/L/O anchors
CURRENCY = PRICE.get("currency","INR")

# ---- Helpers ----------------------------------------------------------------
def normalize(text: str) -> str:
    if not text: return ""
    t = text.strip().lower()
    t = re.sub(r"https?://\S+","", t)
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+","", t)
    t = re.sub(r"\s+"," ", t)
    return t

def _hash01(seed: str) -> float:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    n = int(h[:8], 16) % 10000
    return n / 10000.0  # 0..1

def detect_emotion(norm: str, context: Optional[Dict[str,Any]]) -> str:
    # 1) exact phrase
    for phrase, emo in (EMO.get("exact") or {}).items():
        if phrase in norm: return emo
    # 2) keyword vote
    anchors = EMO.get("anchors") or []
    counts = {a:0 for a in anchors}
    for kw, emo in (EMO.get("keywords") or {}).items():
        if kw in norm: counts[emo] = counts.get(emo,0) + 1
    if counts:
        best = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0]
        if best[1] > 0: return best[0]
    # 3) context hint
    if context and context.get("emotion_hint") in anchors:
        return context["emotion_hint"]
    # 4) default lane
    return "Encouragement"

def passes_pricing_floor(item: Dict[str,Any]) -> bool:
    floors = PRICE.get("floors", {})
    lg_floor = int(PRICE.get("luxury_grand_floor", 4999))
    tier = item.get("tier","Classic")
    if item.get("luxury_grand"):  # LG lives under Luxury but with its own floor
        return int(item.get("price_inr", 0)) >= lg_floor
    min_floor = int(floors.get(tier, 0))
    return int(item.get("price_inr", 0)) >= min_floor

def lg_allowed_for_emotion(item: Dict[str,Any], emotion: str) -> bool:
    if not item.get("luxury_grand"): return True
    lg = TIER.get("luxury_grand", {})
    if emotion in lg.get("blocked_emotions", []): return False
    allow = lg.get("allowed_emotions", [])
    return (emotion in allow) if allow else True

def _has_palette(item: Dict[str,Any]) -> bool:
    pal = item.get("palette")
    return isinstance(pal, list) and len(pal) > 0

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

def weight_for_item(item: Dict[str,Any], emotion: str) -> int:
    flowers = item.get("flowers") or []
    primary = flowers[0] if flowers else "rose"
    base = WEI.get("flower_weights",{}).get(primary, 50)
    emo_bias = WEI.get("emotion_bias",{}).get(emotion,{}).get(primary, base)
    return max(1, round((base + emo_bias)/2))

def pick_deterministic(pool: List[Dict[str,Any]], k: int, seed: str, emotion: str) -> List[Dict[str,Any]]:
    scored = []
    for it in pool:
        w = weight_for_item(it, emotion)
        rnd = _hash01(f"{seed}::{it['id']}")
        score = rnd / (w/100.0)     # lower is better
        scored.append((score, it))
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _,it in scored:
        if len(out) >= k: break
        if it["id"] in seen: continue
        out.append(it); seen.add(it["id"])
    return out

def strong_intent_override(norm: str) -> Dict[str,Any]:
    out = {"forceIconic": None, "forbidMono": None, "note": None}
    m = re.search(r"only\s+(roses?|lilies|orchids?)", norm)
    if m:
        word = m.group(1)
        if word.startswith("rose"):   out["forceIconic"] = "rose"
        elif word.startswith("lil"):  out["forceIconic"] = "lily"
        elif word.startswith("orchid"): out["forceIconic"] = "orchid"
    m2 = re.search(r"mono\s+([a-z]+)", norm)
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

def candidate_pools(emotion: str, context: Optional[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    lgp = TIER.get("luxury_grand", {})
    allow_mix  = lgp.get("allow_in_mix",  True)
    allow_mono = lgp.get("allow_in_mono", True)

    base = [x for x in CATALOG
            if x.get("emotion")==emotion
            and passes_pricing_floor(x)
            and lg_allowed_for_emotion(x, emotion)
            and _has_palette(x)]

    mix  = [x for x in base if not x.get("mono")]
    mono = [x for x in base if     x.get("mono")]

    if not allow_mix:
        mix  = [x for x in mix if not x.get("luxury_grand")]
    if not allow_mono:
        mono = [x for x in mono if not x.get("luxury_grand")]

    # Fallbacks if pools are thin
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

    # replace MIX LG first, then MONO if needed
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

def selection_engine(prompt: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
    """
    Input:  prompt (string), context (optional dict: emotion_hint, budget_inr, packaging_pref, locale)
    Output: list[3] => exactly 2 MIX + 1 MONO, each with palette[]
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
    if not mono_pick:
        base = mix_pool if intent.get("forbidMono") else mono_pool
        if base:
            mono_pick = pick_deterministic(base, 1, seed=f"{emotion}/{norm}/mono", emotion=emotion)[0]

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
