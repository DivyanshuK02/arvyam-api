# app/selection_engine.py
# Ultra-beginner-safe Selection Engine — Phase 1.2 + 1.3 (Edge-Case Playbooks)
# Deterministic, rules-first; strict tier scaffold (Classic → Signature → Luxury).
# Always returns exactly 3 items (2 MIX + 1 MONO) with palette[].
# Edge registers (sympathy/apology/farewell/valentine) apply tone/palette/species rules and copy ≤ N words.
# LG policy: emotion block-list from rules/tier_policy.json and soft multipliers in registers;
#            no numeric budgets in code (LG boost only with intent signal or explicit budget presence).
#
# External contract:
# - Public entrypoint: curate(prompt: str, context: dict) -> list[dict] of 3 items
# - Context may include: {"budget_inr": int|None, "emotion_hint": str|None, "packaging_pref": str|None, "locale": str|None}
# - No network calls. Only reads JSON files from /app and /app/rules.

from __future__ import annotations
import os, re, json, hashlib
from typing import Any, Dict, List, Tuple, Optional

# ------------------------------
# File helpers
# ------------------------------
ROOT = os.path.dirname(__file__)

def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

# ------------------------------
# Load rule tables & catalog
# ------------------------------
CATALOG: List[Dict[str, Any]] = _load_json(_p("catalog.json"), [])
EMO_KEYWORDS: Dict[str, List[str]] = _load_json(_p("rules", "emotion_keywords.json"), {
    "Romance": ["romantic","anniversary","valentine","love","crush","date"],
    "Celebration": ["congrats","congratulations","celebration","party","proud","promotion"],
    "Gratitude": ["thank you","thanks","grateful","appreciate"],
    "Friendship": ["friend","buddy","bestie","pal"],
    "GetWell": ["get well","recover","healing","feel better"],
    "Encouragement": ["you got this","good luck","exams","cheer up","encourage"],
    "Birthday": ["birthday","bday"],
    "Sympathy": ["sorry for your loss","condolence","grief","passing"],
})
SUBS_MAP: Dict[str, str] = _load_json(_p("rules","substitution_map.json"), {
    "hydrangea": "rose"
})
EDGE_KWS: Dict[str, List[str]] = _load_json(_p("rules","edge_keywords.json"), {
    "sympathy": ["sorry for your loss","condolence","grief","passing","bereavement"],
    "apology": ["sorry","apologize","forgive","forgiveness","my mistake","i was wrong"],
    "farewell": ["farewell","goodbye","last day","send off","parting","moving on","retiring","retirement"],
    "valentine": ["valentine","valentines","love","romantic","anniversary","be my valentine","proposal"],
    "grand_intent_keywords": ["grand","bigger","large","lavish","extravagant","50+","hundred","massive","impressive","showstopper","statement"],
    "relationship_grandeur_cues": ["boss","manager","director","ceo","founder","mentor","teacher","professor","parents","grandparents","board"]
})
EDGE_REGS: Dict[str, Any] = _load_json(_p("rules","edge_registers.json"), {
    "copy_max_words": 20,
    "sympathy": {
        "edge_case": True, "tone":"muted",
        "palette_targets": ["white","cream","soft-green","eucalyptus","ivory"],
        "palette_avoid": ["neon","neon-mixed","electric","bright-yellow"],
        "palette_target_boost": 1.15, "palette_avoid_penalty": 0.90,
        "species_prefer": ["lily","orchid","chrysanthemum"],
        "species_avoid": ["sunflower","gerbera"],
        "mono_must_include": ["lily"],
        "lg_policy": "block", "lg_weight_multiplier": 0.0,
        "min_budget_for_lg": 999999,
        "allow_lg_in_mix": False, "allow_lg_in_mono": False
    },
    "apology": {
        "edge_case": True, "tone":"gentle",
        "palette_targets": ["white","soft-purple","lavender","blush","soft-green"],
        "palette_avoid": ["neon","neon-mixed","electric"],
        "palette_target_boost": 1.15, "palette_avoid_penalty": 0.90,
        "species_prefer": ["orchid","lily","rose"],
        "species_avoid": [],
        "lg_policy": "allow_soft", "lg_weight_multiplier": 0.8,
        "min_budget_for_lg": 4999,
        "allow_lg_in_mix": True, "allow_lg_in_mono": True
    },
    "farewell": {
        "edge_case": True, "tone":"respectful",
        "palette_targets": ["white","cream","peach","soft-green"],
        "palette_avoid": ["neon","neon-mixed","electric","bright-yellow"],
        "palette_target_boost": 1.15, "palette_avoid_penalty": 0.90,
        "species_prefer": ["rose","lily","orchid"],
        "species_avoid": ["sunflower","gerbera"],
        "lg_policy": "allow_soft", "lg_weight_multiplier": 0.8,
        "min_budget_for_lg": 4999,
        "allow_lg_in_mix": True, "allow_lg_in_mono": True
    },
    "valentine": {
        "edge_case": True, "tone":"bold-elegant",
        "palette_targets": ["red","crimson","burgundy","blush","soft-pink","cream"],
        "palette_avoid": ["neon","neon-mixed","electric"],
        "palette_target_boost": 1.15, "palette_avoid_penalty": 0.90,
        "species_prefer": ["rose","lily"],
        "species_avoid": [],
        "lg_policy": "allow_soft", "lg_weight_multiplier": 1.5,
        "min_budget_for_lg": 4999,
        "allow_lg_in_mix": True, "allow_lg_in_mono": True
    }
})
TIER_POLICY: Dict[str, Any] = _load_json(_p("rules","tier_policy.json"), {
    "luxury_grand": {
        "allowed_emotions": [],
        "blocked_emotions": ["Sympathy","GetWell","Encouragement"],
        "max_per_triad": 1,
        "allow_in_mix": True,
        "allow_in_mono": True
    }
})

# ------------------------------
# Normalization & helpers
# ------------------------------
TIER_RANK = {"Classic":0, "Signature":1, "Luxury":2}
ICONIC = {"rose":"rose", "roses":"rose", "lily":"lily", "lilies":"lily", "orchid":"orchid", "orchids":"orchid"}

def _norm(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _contains_any(text: str, kws: List[str]) -> bool:
    t = _norm(text)
    return any(kw in t for kw in kws)

def _only_iconic_intent(prompt: str) -> Optional[str]:
    t = _norm(prompt)
    m = re.search(r"\bonly (roses?|lil(?:y|ies)|orchids?)\b", t)
    if not m: 
        return None
    w = m.group(1)
    return ICONIC.get(w, None)

def _detect_edge_case(prompt: str) -> Optional[str]:
    t = _norm(prompt)
    for k in ["sympathy","apology","farewell","valentine"]:
        if _contains_any(t, EDGE_KWS.get(k, [])):
            return k
    return None

def _detect_emotion(prompt: str, context: Dict[str, Any]) -> str:
    # context hint wins
    hint = (context or {}).get("emotion_hint")
    if isinstance(hint, str) and hint:
        return hint
    t = _norm(prompt)
    for emo, kws in EMO_KEYWORDS.items():
        if _contains_any(t, kws):
            return emo
    # fallbacks: align with edge-case to a canonical emotion
    edge = _detect_edge_case(prompt)
    if edge == "sympathy": return "Sympathy"
    if edge == "apology": return "Gratitude"  # apology adjacent; gentle
    if edge == "farewell": return "Friendship"
    if edge == "valentine": return "Romance"
    return "Romance"

def _truncate_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(",.;:") + "…"

def _score_item(it: Dict[str, Any], edge: Optional[str], prompt: str, context: Dict[str, Any]) -> float:
    score = float(it.get("weight", 50))
    flowers = [f.lower() for f in it.get("flowers", [])]
    palette = [p.lower() for p in it.get("palette", [])]
    is_lg = bool(it.get("luxury_grand"))

    # Edge-register adjustments
    if edge and edge in EDGE_REGS:
        reg = EDGE_REGS[edge]
        # palette boosts / penalties
        pt = set([p.lower() for p in reg.get("palette_targets", [])])
        pa = set([p.lower() for p in reg.get("palette_avoid", [])])
        if pt and any(p in pt for p in palette):
            score *= float(reg.get("palette_target_boost", 1.0))
        if pa and any(p in pa for p in palette):
            score *= float(reg.get("palette_avoid_penalty", 1.0))
        # species prefer / avoid
        spref = set([s.lower() for s in reg.get("species_prefer", [])])
        sav = set([s.lower() for s in reg.get("species_avoid", [])])
        if spref and any(f in spref for f in flowers):
            score *= 1.10
        if sav and any(f in sav for f in flowers):
            score *= 0.90
        # LG policy (soft multiplier only with signal; block handled separately)
        if is_lg:
            # signal detection
            t = _norm(prompt)
            grand_signal = any(kw in t for kw in EDGE_KWS.get("grand_intent_keywords", [])) \
                           or any(kw in t for kw in EDGE_KWS.get("relationship_grandeur_cues", [])) \
                           or (isinstance(context.get("budget_inr", None), (int, float)))
            if grand_signal:
                score *= float(reg.get("lg_weight_multiplier", 1.0))
            # else: neutral (no extra boost)

    # slight freshness variance to break ties deterministically
    # hash by id for deterministic ordering
    h = int(hashlib.md5((it.get("id","") + it.get("title","")).encode("utf-8")).hexdigest(), 16)
    score += (h % 7) * 0.01
    return score

def _allow_lg_for_emotion(emotion: str) -> bool:
    blocked = set(TIER_POLICY.get("luxury_grand", {}).get("blocked_emotions", []))
    return emotion not in blocked

# ------------------------------
# Pooling & picking
# ------------------------------
def _filter_pool(emotion: str, edge: Optional[str]) -> List[Dict[str, Any]]:
    # prefer same emotion; if too few for a tier, we will fallback at pick-time.
    pool = [x for x in CATALOG if x.get("image_url") and x.get("palette")]
    return pool

def _split_by_tier(pool: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"Classic": [], "Signature": [], "Luxury": []}
    for x in pool:
        t = x.get("tier")
        if t in out:
            out[t].append(x)
    return out

def _best_candidate(cands: List[Dict[str, Any]], edge: Optional[str], prompt: str, ctx: Dict[str, Any],
                    want_mono: Optional[bool], allow_lg: bool) -> Optional[Dict[str, Any]]:
    filtered = []
    for it in cands:
        if want_mono is not None and bool(it.get("mono")) != want_mono:
            continue
        if it.get("luxury_grand") and not allow_lg:
            continue
        filtered.append(it)
    if not filtered:
        return None
    scored = [( _score_item(it, edge, prompt, ctx), it ) for it in filtered]
    scored.sort(key=lambda z: z[0], reverse=True)
    return scored[0][1]

def _gather_candidates_by_tier(tier_pool: Dict[str, List[Dict[str, Any]]],
                               edge: Optional[str], prompt: str, ctx: Dict[str, Any],
                               allow_lg: bool) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
    per = {}
    for tier in ["Classic","Signature","Luxury"]:
        cands = tier_pool.get(tier, [])
        per[tier] = {
            "mix": _best_candidate(cands, edge, prompt, ctx, want_mono=False, allow_lg=allow_lg),
            "mono": _best_candidate(cands, edge, prompt, ctx, want_mono=True, allow_lg=allow_lg),
        }
    return per

def _choose_triad(per: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
                  edge: Optional[str], prompt: str, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Build three combinations (mono on Classic / Signature / Luxury), score each, pick the best available.
    combos = []
    for mono_tier in ["Classic","Signature","Luxury"]:
        triad = []
        ok = True
        for tier in ["Classic","Signature","Luxury"]:
            pick = per[tier]["mono"] if tier == mono_tier else per[tier]["mix"]
            if pick is None:
                ok = False; break
            triad.append(pick)
        if ok:
            s = sum(_score_item(it, edge, prompt, ctx) for it in triad)
            combos.append((s, mono_tier, triad))
    if combos:
        combos.sort(key=lambda z: z[0], reverse=True)
        return combos[0][2]

    # Fallbacks: if some tiers are missing candidates of requested mix/mono, relax mono constraint
    # 1) pick best available (mix first) per tier
    triad = []
    for tier in ["Classic","Signature","Luxury"]:
        pick = per[tier]["mix"] or per[tier]["mono"]
        if pick is None:
            # last-resort: grab anything in catalog of this tier
            triad.append({"__missing__": True, "tier": tier})
        else:
            triad.append(pick)
    # replace any missing with highest-score any-item from that tier across catalog
    fixed = []
    for it in triad:
        if "__missing__" in it:
            t = it["tier"]
            all_tier = [x for x in CATALOG if x.get("tier")==t]
            best = None
            bests = sorted([( _score_item(x, edge, prompt, ctx), x) for x in all_tier], key=lambda z:z[0], reverse=True)
            best = bests[0][1] if bests else None
            if best: fixed.append(best)
        else:
            fixed.append(it)
    triad = fixed

    # Ensure exactly one MONO: if not, try to swap one tier item to the nearest mono same tier
    mono_count = sum(1 for it in triad if it.get("mono"))
    if mono_count == 0:
        # try switch Luxury, then Signature, then Classic to mono
        for t in ["Luxury","Signature","Classic"]:
            cands = [x for x in CATALOG if x.get("tier")==t and x.get("mono")]
            if cands:
                best = sorted([( _score_item(x, edge, prompt, ctx), x) for x in cands], key=lambda z:z[0], reverse=True)[0][1]
                for i, it in enumerate(triad):
                    if it.get("tier")==t:
                        triad[i]=best; mono_count=1; break
            if mono_count==1: break
    elif mono_count > 1:
        # reduce to one mono by converting the lowest-score mono to mix candidate of the same tier if possible
        monos = [( _score_item(it, edge, prompt, ctx), i) for i,it in enumerate(triad) if it.get("mono")]
        monos.sort()  # lowest first
        for _, idx in monos[1:]:
            tier = triad[idx].get("tier")
            cands = [x for x in CATALOG if x.get("tier")==tier and not x.get("mono")]
            if cands:
                best = sorted([( _score_item(x, edge, prompt, ctx), x) for x in cands], key=lambda z:z[0], reverse=True)[0][1]
                triad[idx] = best
        # re-check
        mono_count = sum(1 for it in triad if it.get("mono"))

    return triad

def _ensure_sympathy_lily(triad: List[Dict[str, Any]], edge: Optional[str], prompt: str, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    if edge != "sympathy":
        return triad
    has_lily = any("lily" in [f.lower() for f in it.get("flowers",[])] for it in triad)
    if has_lily:
        return triad
    # try to swap MONO item to a lily mono within the same tier first, else any tier (while preserving one-per-tier)
    mono_idx = next((i for i,it in enumerate(triad) if it.get("mono")), None)
    tiers = ["Classic","Signature","Luxury"]
    search_order = [mono_idx] + [i for i in range(3) if i!=mono_idx] if mono_idx is not None else list(range(3))
    for idx in search_order:
        tier = triad[idx].get("tier")
        cands = [x for x in CATALOG if x.get("tier")==tier and x.get("mono") and "lily" in [f.lower() for f in x.get("flowers",[])]]
        if cands:
            best = sorted([( _score_item(x, edge, prompt, ctx), x) for x in cands], key=lambda z:z[0], reverse=True)[0][1]
            triad[idx] = best
            return triad
    # if nothing, just return triad as-is
    return triad

def _map_out(it: Dict[str, Any], emotion: str, edge: Optional[str], max_words: int) -> Dict[str, Any]:
    out = {
        "id": it.get("id"),
        "title": it.get("title"),
        "desc": _truncate_words(it.get("desc","").strip(), max_words),
        "image": it.get("image_url"),
        "price": it.get("price_inr"),
        "currency": "INR",
        "emotion": emotion,
        "tier": it.get("tier"),
        "packaging": it.get("packaging"),
        "mono": bool(it.get("mono")),
        "palette": it.get("palette", []),
        "luxury_grand": bool(it.get("luxury_grand")),
    }
    if edge in {"sympathy","apology","farewell","valentine"}:
        out["edge_case"] = True
    note = it.get("__note__")
    if note:
        out["note"] = note
    return out

def curate(prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    context = context or {}
    norm_prompt = _norm(prompt)
    emotion = _detect_emotion(prompt, context)
    edge = _detect_edge_case(prompt)

    # base pool (we filter tiers later). Keep all catalog to allow safe fallbacks with limited data.
    pool = _filter_pool(emotion, edge)
    allow_lg = _allow_lg_for_emotion(emotion)

    # substitute editorial (note) if explicit unavailable item is asked
    note_text = None
    for unavailable, sub in SUBS_MAP.items():
        if unavailable in norm_prompt:
            note_text = f"{unavailable.capitalize()} is seasonal; showing nearest alternative."
            break

    # build candidates per tier
    tiers = _split_by_tier(pool)
    per = _gather_candidates_by_tier(tiers, edge, prompt, context, allow_lg=allow_lg)
    triad = _choose_triad(per, edge, prompt, context)

    # Strong-intent iconic override (only roses/lilies/orchids) → force the MONO at its tier (choose the tier where mono candidate exists with best score)
    iconic = _only_iconic_intent(prompt)
    if iconic:
        # find mono candidates with that flower by tier
        best_slot = None
        best_score = -1e9
        best_cand = None
        for t in ["Classic","Signature","Luxury"]:
            cands = [x for x in CATALOG if x.get("tier")==t and x.get("mono") and iconic in [f.lower() for f in x.get("flowers",[])]]
            if not cands: 
                continue
            cand = sorted([( _score_item(x, edge, prompt, context), x) for x in cands], key=lambda z:z[0], reverse=True)[0][1]
            sc = _score_item(cand, edge, prompt, context)
            if sc > best_score:
                best_score = sc; best_slot = t; best_cand = cand
        if best_cand and best_slot:
            for i,it in enumerate(triad):
                if it.get("tier")==best_slot:
                    triad[i] = dict(best_cand)  # copy
                    triad[i]["__iconic__"] = True
                    break

    # Edge-case lily guarantee for Sympathy
    triad = _ensure_sympathy_lily(triad, edge, prompt, context)

    # Attach editorial note to first MIX item when redirection happened
    if note_text:
        for it in triad:
            if not it.get("mono"):
                it["__note__"] = note_text
                break

    # Final ordering by tier
    triad.sort(key=lambda it: (TIER_RANK.get(it.get("tier"), 99), 1 if it.get("luxury_grand") else 0))

    # Map out with copy trimming
    max_words = int(EDGE_REGS.get("copy_max_words", 20))
    out = [_map_out(it, emotion, edge, max_words) for it in triad]

    # Ensure exactly 2 MIX + 1 MONO (hard check at the end)
    mono_count = sum(1 for x in out if x["mono"])
    if mono_count != 1:
        # try to repair: keep best mix in two tiers and mono in one tier; if impossible, flip the lowest scoring to mix
        # (given limited catalog, we avoid over-engineering: enforce by toggling the Classic item if needed)
        # In worst case (no mono exists in catalog), we simply mark the Signature as mix.
        if mono_count == 0:
            # attempt to convert Luxury to mono if a mono exists
            luxury_mono = [x for x in CATALOG if x.get("tier")=="Luxury" and x.get("mono")]
            if luxury_mono:
                best = sorted([( _score_item(x, edge, prompt, context), x) for x in luxury_mono], key=lambda z:z[0], reverse=True)[0][1]
                for i,oi in enumerate(out):
                    if oi["tier"]=="Luxury":
                        out[i] = _map_out(best, emotion, edge, max_words)
                        out[i]["mono"]=True
                        break
        elif mono_count > 1:
            # set Classic to mix if possible
            for i,oi in enumerate(out):
                if oi["tier"]=="Classic" and oi["mono"]:
                    mix_cands = [x for x in CATALOG if x.get("tier")=="Classic" and not x.get("mono")]
                    if mix_cands:
                        best = sorted([( _score_item(x, edge, prompt, context), x) for x in mix_cands], key=lambda z:z[0], reverse=True)[0][1]
                        out[i] = _map_out(best, emotion, edge, max_words)
                        out[i]["mono"]=False
                    break

    return out
