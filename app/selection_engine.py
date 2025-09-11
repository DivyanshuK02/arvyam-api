
# app/selection_engine.py
# Ultra-beginner-safe Selection Engine — Phase 1.2 + 1.3 (Edge-Case Playbooks)
# Deterministic, rules-first; strict tier scaffold (Classic → Signature → Luxury).
# Always returns exactly 3 items (2 MIX + 1 MONO) with palette[].
# Edge registers (sympathy/apology/farewell/valentine) apply tone/palette/species rules and copy ≤ N words.
# LG policy: emotion block-list from tier_policy and soft/encouraging multipliers in registers;
#            *no numeric budgets in code* (LG boost only with intent signal or explicit budget presence).

from __future__ import annotations
import json, os, re, hashlib
from typing import Any, Dict, List, Tuple, Optional

# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------

ROOT = os.path.dirname(__file__)

def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

# ------------------------------------------------------------
# Load rule tables & catalog (lazy)
# ------------------------------------------------------------

def _rules() -> Dict[str, Any]:
    return {
        "emotion_keywords": _load_json(_p("rules", "emotion_keywords.json"), {}),
        "edge_keywords": _load_json(_p("rules", "edge_keywords.json"), {}),
        "edge_registers": _load_json(_p("rules", "edge_registers.json"), {}),
        "pricing_policy": _load_json(_p("rules", "pricing_policy.json"), {}),
        "tier_policy": _load_json(_p("rules", "tier_policy.json"), {}),
        "weights_default": _load_json(_p("rules", "weights_default.json"), {}),
        "substitution_map": _load_json(_p("rules", "substitution_map.json"), {}),
    }

def _catalog() -> List[Dict[str, Any]]:
    cat = _load_json(_p("catalog.json"), [])
    # Normalize some fields
    for it in cat:
        it.setdefault("mono", False)
        it.setdefault("palette", [])
        it.setdefault("flowers", [])
        it.setdefault("luxury_grand", False)
        it.setdefault("weight", 1.0)
        # Normalize tier case
        if "tier" in it and isinstance(it["tier"], str):
            t = it["tier"].title()
            if t not in ("Classic","Signature","Luxury"):
                t = "Classic"
            it["tier"] = t
        else:
            it["tier"] = "Classic"
    return cat

# ------------------------------------------------------------
# Public helpers (imported by main.py)
# ------------------------------------------------------------

_WORD_RE = re.compile(r"[\w']+")

def normalize(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"https?://\S+|\S+@\S+", " ", text, flags=re.I)  # strip URLs/emails
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def detect_emotion(prompt: str, context: Optional[Dict[str, Any]]=None) -> str:
    """Rules-first emotion detection using rules/emotion_keywords.json.
       Fallback to Romance if no signal."""
    r = _rules()
    kw: Dict[str, List[str]] = r.get("emotion_keywords", {})
    p = normalize(prompt)
    # exact group match priority
    for emo, words in kw.items():
        for w in words:
            if w and w in p:
                return emo
    # hint from context
    if context and isinstance(context.get("emotion_hint"), str):
        return context["emotion_hint"]
    # fallback
    return "Romance"

# ------------------------------------------------------------
# Edge-case routing (1.3)
# ------------------------------------------------------------

def detect_edge_case(prompt: str) -> Optional[str]:
    r = _rules().get("edge_keywords", {})
    p = normalize(prompt)
    for tag in ("sympathy","apology","farewell","valentine"):
        words = r.get(tag, [])
        for w in words:
            if w and w in p:
                return tag
    return None

def has_grand_intent(prompt: str) -> bool:
    r = _rules().get("edge_keywords", {})
    p = normalize(prompt)
    for key in ("grand_intent_keywords","relationship_grandeur_cues"):
        for w in r.get(key, []):
            if w and w in p:
                return True
    return False

# ------------------------------------------------------------
# Weighting utilities (deterministic)
# ------------------------------------------------------------

TIER_ORDER = ("Classic","Signature","Luxury")
ICONIC_FLOWERS = ("rose","lily","orchid")
ICONIC_VARIANTS = {
    "rose": ["rose","roses"],
    "lily": ["lily","lilies"],
    "orchid": ["orchid","orchids"]
}

def _hash_f(s: str) -> float:
    # Deterministic tiny jitter to break ties: [0.0, 0.01)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) % 100) / 10000.0

def _base_weight(it: Dict[str,Any], weights_default: Dict[str,Any], emotion: str) -> float:
    # Start from catalog row weight
    w = float(it.get("weight", 1.0))
    # Apply optional global species weights (flower_weights) if present
    fw = weights_default.get("flower_weights", {})
    for sp in it.get("flowers", []):
        w *= float(fw.get(str(sp).lower(), 1.0))
    # Apply optional emotion-specific bias per species (emotion_bias[emotion][species])
    eb_emotion = weights_default.get("emotion_bias", {}).get(emotion, {})
    for sp in it.get("flowers", []):
        w *= float(eb_emotion.get(str(sp).lower(), 1.0)) / 60.0  # normalize to keep scale reasonable
    return max(w, 0.0001)

def _palette_boost(it: Dict[str,Any], reg: Dict[str,Any]) -> float:
    """Multiplicative boost/penalty based on palette matches from edge_registers.json."""
    pal = set([p.lower() for p in it.get("palette", [])])
    tgt = set([p.lower() for p in reg.get("palette_targets", [])])
    avoid = set([p.lower() for p in reg.get("palette_avoid", [])])
    boost = float(reg.get("palette_target_boost", 1.0))
    penalty = float(reg.get("palette_avoid_penalty", 1.0))
    f = 1.0
    if pal & tgt:
        f *= boost
    if pal & avoid:
        f *= penalty
    return f

def _species_factor(it: Dict[str,Any], reg: Dict[str,Any]) -> float:
    flowers = set([f.lower() for f in it.get("flowers", [])])
    prefer = set([f.lower() for f in reg.get("species_prefer", [])])
    avoid = set([f.lower() for f in reg.get("species_avoid", [])])
    f = 1.0
    if flowers & prefer:
        f *= 1.35
    if flowers & avoid:
        f *= 0.7
    return f

def _lg_allowed_for_emotion(emotion: str) -> bool:
    tp = _rules().get("tier_policy", {}).get("luxury_grand", {})
    blocked = set(tp.get("blocked_emotions", []))
    return emotion not in blocked

def _lg_multiplier(it: Dict[str,Any], emotion: str, reg: Optional[Dict[str,Any]], budget_inr: Optional[int], grand_intent: bool) -> float:
    # Non-LG items unaffected
    if not it.get("luxury_grand", False):
        return 1.0
    # Respect emotion block-list
    if not _lg_allowed_for_emotion(emotion):
        return 0.0
    # No register → allowed but unshaped
    if not reg:
        return 1.0
    # Base multiplier from register (e.g., 0.8 for apology/farewell, 1.5 for valentine)
    mul = float(reg.get("lg_weight_multiplier", 1.0))
    # SIGNAL-ONLY: LG gets a boost only if there's intent signal (grand keywords) OR any budget provided.
    has_signal = bool(grand_intent or (budget_inr is not None))
    if not has_signal and mul > 1.0:
        mul = 1.0  # eligible, but no extra lift without signal
    return max(mul, 0.0)

def _score_item(it: Dict[str,Any], emotion: str, reg: Optional[Dict[str,Any]], weights_default: Dict[str,Any],
                budget_inr: Optional[int], grand_intent: bool, prompt: str) -> float:
    w = _base_weight(it, weights_default, emotion)
    if reg:
        w *= _palette_boost(it, reg)
        w *= _species_factor(it, reg)
        w *= _lg_multiplier(it, emotion, reg, budget_inr, grand_intent)
    else:
        # still respect LG block-list even if no register is active
        if it.get("luxury_grand", False) and not _lg_allowed_for_emotion(emotion):
            return 0.0
    # tiny deterministic jitter
    w += _hash_f(it.get("id","") + prompt)
    return w

# ------------------------------------------------------------
# Build candidate pools
# ------------------------------------------------------------

def _filter_by_emotion(cat: List[Dict[str,Any]], emotion: str) -> List[Dict[str,Any]]:
    out = []
    for it in cat:
        e = it.get("emotion")
        if not e or e == emotion:
            out.append(it)
    return out

def _top_by_tier(items: List[Dict[str,Any]], want_mono: Optional[bool]) -> Dict[str, List[Dict[str,Any]]]:
    tiers: Dict[str, List[Dict[str,Any]]] = {t: [] for t in TIER_ORDER}
    for it in items:
        if want_mono is not None and bool(it.get("mono")) != bool(want_mono):
            continue
        t = it.get("tier", "Classic").title()
        if t in tiers:
            tiers[t].append(it)
    # unique by id, keep original order
    for t in tiers:
        seen = set()
        uniq = []
        for it in tiers[t]:
            if it.get("id") in seen:
                continue
            seen.add(it.get("id"))
            uniq.append(it)
        tiers[t] = uniq
    return tiers

# ------------------------------------------------------------
# Strong intent override (only lilies/roses/orchids)
# ------------------------------------------------------------

def _strong_intent_iconic(prompt: str) -> Optional[str]:
    p = normalize(prompt)
    # handle plurals: "only lilies/roses/orchids"
    for base, variants in ICONIC_VARIANTS.items():
        for v in variants:
            if f"only {v}" in p:
                return base
    # also allow singular phrasing without "only", e.g., "mono lilies" (light support)
    if "mono " in p:
        for base, variants in ICONIC_VARIANTS.items():
            for v in variants:
                if v in p:
                    return base
    return None

def _find_iconic_mono(cat: List[Dict[str,Any]], flower: str) -> Optional[Dict[str,Any]]:
    for it in cat:
        if it.get("mono") and flower.lower() in [x.lower() for x in it.get("flowers", [])]:
            return it
    return None

# ------------------------------------------------------------
# Editorial redirection (substitutions) — note on first MIX
# ------------------------------------------------------------

def _needs_redirection(prompt: str) -> Optional[Tuple[str, List[str]]]:
    p = normalize(prompt)
    m = _rules().get("substitution_map", {})
    for src, alts in m.items():
        if src.lower() in p:
            return (src.lower(), alts)
    return None

# ------------------------------------------------------------
# Triad assembly (one per tier; exactly 2 MIX + 1 MONO)
# ------------------------------------------------------------

def _order_by_tier(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    def key(it):
        tier = it.get("tier","Classic").title()
        tier_rank = TIER_ORDER.index(tier) if tier in TIER_ORDER else 99
        lg_bias = 1 if it.get("luxury_grand") else 0  # inside Luxury, LG appears after non-LG
        return (tier_rank, lg_bias)
    return sorted(items, key=key)

def _truncate_words(s: str, max_words: int) -> str:
    if not s: return ""
    words = _WORD_RE.findall(s)
    if len(words) <= max_words:
        return s.strip()
    trimmed = " ".join(words[:max_words])
    return trimmed.strip()

def selection_engine(prompt: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
    context = context or {}
    prompt_n = normalize(prompt)
    cat = _catalog()
    rules = _rules()

    emotion = detect_emotion(prompt, context)
    edge = detect_edge_case(prompt)
    reg = rules.get("edge_registers", {}).get(edge, None) if edge else None
    budget_inr = context.get("budget_inr")
    grand_intent = has_grand_intent(prompt)

    # strong-intent iconic override?
    iconic_only = _strong_intent_iconic(prompt)

    # base filter
    base = _filter_by_emotion(cat, emotion)

    # score items (deterministic)
    for it in base:
        it["_score"] = _score_item(it, emotion, reg, rules.get("weights_default", {}), budget_inr, grand_intent, prompt_n)

    # pools by mono/mix
    mix_by_tier = _top_by_tier([x for x in base if not x.get("mono") and x["_score"]>0], want_mono=False)
    mono_by_tier = _top_by_tier([x for x in base if x.get("mono") and x["_score"]>0], want_mono=True)

    # if no mono available, borrow iconic mono from catalog (R/L/O neutral copy)
    if all(len(v)==0 for v in mono_by_tier.values()):
        for flower in ICONIC_FLOWERS:
            cand = _find_iconic_mono(cat, flower)
            if cand:
                mono_by_tier[cand.get("tier","Classic").title()].append(cand)
                break

    # Choose the tier that will host MONO: the tier whose top mono score is highest
    mono_scores = {}
    for t in TIER_ORDER:
        cand = mono_by_tier[t][0] if mono_by_tier[t] else None
        mono_scores[t] = cand["_score"] if cand and "_score" in cand else 0.0
    tier_for_mono = max(TIER_ORDER, key=lambda k: mono_scores[k])

    # Build triad: one per tier
    triad: List[Dict[str,Any]] = []
    for t in TIER_ORDER:
        chosen = None
        if t == tier_for_mono and mono_by_tier[t]:
            chosen = mono_by_tier[t][0]
        else:
            chosen = mix_by_tier[t][0] if mix_by_tier[t] else (mono_by_tier[t][0] if mono_by_tier[t] else None)
        if chosen:
            triad.append(chosen)

    # If we somehow did not collect 3 (missing a tier), fill from any tier (fallback)
    if len(triad) < 3:
        missing = [t for t in TIER_ORDER if t not in [x.get("tier","Classic").title() for x in triad]]
        rest = [x for x in base if x["_score"]>0 and x.get("id") not in {y.get("id") for y in triad}]
        rest.sort(key=lambda x: (-x["_score"]))
        for t in missing:
            for x in rest:
                if x.get("tier","Classic").title() == t:
                    triad.append(x)
                    break
        if len(triad) < 3 and rest:
            triad = _order_by_tier(triad + rest[: 3-len(triad)])[:3]

    # Enforce exactly 2 MIX + 1 MONO
    mix_count = sum(1 for x in triad if not x.get("mono"))
    if mix_count < 2:
        # ensure at least two mixes: replace lowest-scored mono with best mix from the same tier
        monos = [(i,x) for i,x in enumerate(triad) if x.get("mono")]
        if monos:
            i_mono, mono_item = min(monos, key=lambda p: p[1].get("_score",0.0))
            tier = mono_item.get("tier","Classic").title()
            mixes = [it for it in base if it.get("tier","Classic").title()==tier and not it.get("mono") and it["_score"]>0]
            if mixes:
                triad[i_mono] = sorted(mixes, key=lambda z: -z["_score"])[0]

    # Strong-intent iconic override (“only lilies/roses/orchids” → force MONO iconic)
    if iconic_only:
        ic = _find_iconic_mono(cat, iconic_only)
        if ic:
            tier = ic.get("tier","Classic").title()
            for i,x in enumerate(triad):
                if x.get("tier","Classic").title() == tier:
                    triad[i] = ic
                    break
            else:
                idx, _ = min(list(enumerate(triad)), key=lambda p: p[1].get("_score",0.0))
                triad[idx] = ic

    # Edge: Sympathy lily guarantee — if no lily in the triad, swap in a lily mono if available
    if edge == "sympathy":
        has_lily = any("lily" in [f.lower() for f in it.get("flowers", [])] for it in triad)
        if not has_lily:
            lily_mono = _find_iconic_mono(cat, "lily")
            if lily_mono:
                mono_idxs = [i for i,x in enumerate(triad) if x.get("mono")]
                if mono_idxs:
                    triad[mono_idxs[0]] = lily_mono
                else:
                    idx, _ = min(list(enumerate(triad)), key=lambda p: p[1].get("_score",0.0))
                    triad[idx] = lily_mono

    # Redirection note (first MIX if a redirect happened)
    redirect = _needs_redirection(prompt)
    redir_note = None
    if redirect:
        src, _ = redirect
        redir_note = f"Nearest alternative to '{src}' has been used."

    # Map output, truncate copy, attach edge flag + note
    out = []
    max_words = int(rules.get("edge_registers", {}).get("copy_max_words", 20))
    for it in _order_by_tier(triad):
        o = {
            "id": it.get("id"),
            "title": it.get("title"),
            "desc": _truncate_words(it.get("desc",""), max_words) if edge else it.get("desc","").strip(),
            "image": it.get("image_url") or it.get("image"),
            "price": it.get("price_inr"),
            "currency": it.get("currency","INR"),
            "emotion": emotion,
            "tier": it.get("tier","Classic").title(),
            "packaging": it.get("packaging","Box"),
            "mono": bool(it.get("mono")),
            "palette": it.get("palette", []),
            "luxury_grand": bool(it.get("luxury_grand", False))
        }
        if edge:
            o["edge_case"] = True
            o["edge_type"] = edge
        out.append(o)

    # attach note to the FIRST MIX only (after mapping), preserve on remaps
    if redir_note:
        for i,oo in enumerate(out):
            if not oo.get("mono"):
                if "note" not in oo:
                    out[i]["note"] = redir_note
                break

    # Guarantee one item per tier in the final mapped output
    per_tier: Dict[str, Dict[str,Any]] = {}
    backing: Dict[str, Dict[str,Any]] = {x.get("id"): x for x in triad}
    for mapped in out:
        t = mapped.get("tier","Classic").title()
        src = backing.get(mapped["id"], {})
        if t not in per_tier or src.get("_score",0) > backing.get(per_tier[t]["id"],{}).get("_score",0):
            per_tier[t] = mapped
    out2 = [per_tier.get("Classic"), per_tier.get("Signature"), per_tier.get("Luxury")]
    out = [x for x in out2 if x]

    # Final strict order & 2 MIX + 1 MONO re-check
    out = _order_by_tier(out)
    mix_cnt = sum(1 for x in out if not x.get("mono"))
    if mix_cnt < 2 and len(out) == 3:
        # Flip one mono to best mix in same tier if available
        tier_mono = None
        for x in out:
            if x.get("mono"):
                tier_mono = x["tier"]
                break
        if tier_mono:
            mixes = [it for it in base if it.get("tier","Classic").title()==tier_mono and not it.get("mono") and it["_score"]>0]
            if mixes:
                best_mix = sorted(mixes, key=lambda z: -z["_score"])[0]
                for i,oo in enumerate(out):
                    if oo.get("mono") and oo["tier"]==tier_mono:
                        out[i] = {
                            "id": best_mix.get("id"),
                            "title": best_mix.get("title"),
                            "desc": _truncate_words(best_mix.get("desc",""), max_words) if edge else best_mix.get("desc","").strip(),
                            "image": best_mix.get("image_url") or best_mix.get("image"),
                            "price": best_mix.get("price_inr"),
                            "currency": best_mix.get("currency","INR"),
                            "emotion": emotion,
                            "tier": best_mix.get("tier","Classic").title(),
                            "packaging": best_mix.get("packaging","Box"),
                            "mono": False,
                            "palette": best_mix.get("palette", []),
                            "luxury_grand": bool(best_mix.get("luxury_grand", False))
                        }
                        break

    return out
