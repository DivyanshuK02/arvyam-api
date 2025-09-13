# app/selection_engine.py
# Ultra-beginner-safe Selection Engine — Phase 1.2 + 1.3 (Edge-Case Playbooks)
# - Deterministic, rules-first; strict tier scaffold (Classic → Signature → Luxury).
# - Always returns exactly 3 items (2 MIX + 1 MONO) with palette[].
# - Edge registers (sympathy/apology/farewell/valentine): tone/palette/species rules and copy ≤ N words.
# - LG policy: emotion block-list from rules/tier_policy.json and soft multipliers in registers (intent-only; no numeric budgets in code).
# - External contract: export curate(), selection_engine(), normalize(), detect_emotion().
#
# Notes
# -----
# * No network calls. Reads JSON files from /app and /app/rules only.
# * This module avoids hard-coded prices or numeric budgets (per project direction).

from __future__ import annotations

import json, os, re, hashlib
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["curate", "selection_engine", "normalize", "detect_emotion"]

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
# Data sources (all project-owned JSON files)
# ------------------------------------------------------------

CATALOG = _load_json(_p("catalog.json"), [])
RULES_DIR = _p("rules")
EMOTION_KEYWORDS = _load_json(os.path.join(RULES_DIR, "emotion_keywords.json"), {})
EDGE_KEYWORDS = _load_json(os.path.join(RULES_DIR, "edge_keywords.json"), {})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, "edge_registers.json"), {})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, "tier_policy.json"), {"luxury_grand": {"blocked_emotions": []}})
# Edge register keys (strict): only these four are considered "edge cases".
EDGE_REGISTERS = {"sympathy","apology","farewell","valentine"}

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

TIER_ORDER = ["Classic", "Signature", "Luxury"]
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

ICONIC_SPECIES = {
    "lily": ["lily", "lilies"],
    "rose": ["rose", "roses"],
    "orchid": ["orchid", "orchids"]
}

def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s[:500]

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z]+", normalize(s))

def _any_match(text: str, phrases: List[str]) -> bool:
    t = normalize(text)
    for p in phrases:
        if p and normalize(p) in t:
            return True
    return False

def detect_emotion(prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Rules-first emotion detection against Phase-1 schema in rules/emotion_keywords.json.
    Evaluation order (from file notes): edge registers handled elsewhere; then exact phrase,
    combos/context, then keyword buckets in anchor order. Falls back to first anchor.
    """
    p = normalize(prompt)

    # Explicit hint wins if it matches a known anchor
    if context and isinstance(context.get("emotion_hint"), str):
        hint = context["emotion_hint"].strip()
        anchors = EMOTION_KEYWORDS.get("anchors", [])
        return hint if hint in anchors else hint.title()

    anchors: List[str] = EMOTION_KEYWORDS.get("anchors", [])
    exact: Dict[str, str] = EMOTION_KEYWORDS.get("exact", {})
    combos: List[Dict[str, Any]] = EMOTION_KEYWORDS.get("combos", [])
    keywords: Dict[str, List[str]] = EMOTION_KEYWORDS.get("keywords", {})

    # 1) Exact phrases
    for phrase, anchor in exact.items():
        if phrase and normalize(phrase) in p:
            return anchor

    # 2) Combo/context rules
    for rule in combos:
        all_terms = rule.get("all", [])
        anchor = rule.get("anchor")
        if all_terms and anchor and all(normalize(t) in p for t in all_terms):
            return anchor

    # 3) Keyword buckets (in anchor order)
    for anchor in anchors:
        bucket = keywords.get(anchor, [])
        if bucket and any(normalize(k) in p for k in bucket):
            return anchor

    # 4) Default to first configured anchor or Affection/Support
    return anchors[0] if anchors else "Affection/Support"

def detect_edge_case(prompt: str) -> Optional[str]:
    p = normalize(prompt)
    for label, phrases in EDGE_KEYWORDS.items():
        if _any_match(p, phrases):
            return label  # sympathy | apology | farewell | valentine
    return None

def _has_species(item: Dict[str, Any], species: str) -> bool:
    flowers = [f.lower() for f in item.get("flowers", [])]
    return species.lower() in flowers

def _is_lg(item: Dict[str, Any]) -> bool:
    return bool(item.get("luxury_grand"))

def _is_mono(item: Dict[str, Any]) -> bool:
    return bool(item.get("mono", False))

def _tier_rank(item: Dict[str, Any]) -> int:
    return TIER_RANK.get(item.get("tier"), 99)

def _truncate_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words])

def _order_by_tier(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(it):
        r = _tier_rank(it)
        # LG should not jump outside Luxury slot; within Luxury, keep LG after non-LG
        lg_bias = 1 if (it.get("tier") == "Luxury" and _is_lg(it)) else 0
        return (r, lg_bias)
    return sorted(items, key=key)

# ------------------------------------------------------------
# Scoring (palette/species + registers, no numeric budgets)
# ------------------------------------------------------------

def _compute_weight(item: Dict[str, Any],
                    base_int: int,
                    register: Dict[str, Any],
                    edge_case: Optional[str],
                    iconic_intent: Optional[str]) -> float:
    w = float(base_int)

    palette = [p.lower() for p in item.get("palette", [])]
    flowers = [f.lower() for f in item.get("flowers", [])]

    # Edge register adjustments
    if edge_case and register:
        ptb = float(register.get("palette_target_boost", 1.0))
        pap = float(register.get("palette_avoid_penalty", 1.0))

        targets = [x.lower() for x in register.get("palette_targets", [])]
        avoids  = [x.lower() for x in register.get("palette_avoid", [])]

        if targets and any(t in palette for t in targets):
            w *= ptb
        if avoids and any(a in palette for a in avoids):
            w *= pap

        spref = [x.lower() for x in register.get("species_prefer", [])]
        savoid = [x.lower() for x in register.get("species_avoid", [])]

        if spref and any(s in flowers for s in spref):
            w *= 1.10
        if savoid and any(s in flowers for s in savoid):
            w *= 0.90

        # Sympathy lily guard encourages lilies strongly
        if edge_case == "sympathy" and "lily" in flowers:
            w *= 1.25

        # Soft LG encouragement/penalty via multiplier (applied later per-item).

    # Iconic "only X" intent: massive preference for mono of that species
    if iconic_intent:
        species = iconic_intent
        if _is_mono(item) and species in flowers:
            w *= 10.0
        elif species in flowers:
            w *= 1.5
        else:
            w *= 0.6

    # Item's own editorial weight
    w *= max(0.1, float(item.get("weight", 50)) / 50.0)

    return w

def _apply_lg_policy(item: Dict[str, Any], emotion: str, edge_case: Optional[str]) -> float:
    """Return a multiplier (0..1.5) based on LG policies and edge registers."""
    if not _is_lg(item):
        return 1.0

    # Global block-list from tier policy
    blocked = [x.lower() for x in TIER_POLICY.get("luxury_grand", {}).get("blocked_emotions", [])]
    if emotion.lower() in blocked:
        return 0.0

    # Edge register soft multiplier
    if edge_case and edge_case in EDGE_REGISTERS:
        reg = EDGE_REGISTERS[edge_case]
        if str(reg.get("lg_policy", "")).lower() == "block":
            return 0.0
        mul = float(reg.get("lg_weight_multiplier", 1.0))
        return mul

    return 1.0

# ------------------------------------------------------------
# Candidate building & picking
# ------------------------------------------------------------

def _candidates_for_tier(items: List[Dict[str, Any]], tier: str) -> List[Dict[str, Any]]:
    return [it for it in items if it.get("tier") == tier]

def _pick_best(cands: List[Dict[str, Any]], want_mono: Optional[bool]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    filtered = cands
    if want_mono is True:
        filtered = [c for c in cands if _is_mono(c)]
    elif want_mono is False:
        filtered = [c for c in cands if not _is_mono(c)]
    if not filtered:
        filtered = cands
    return max(filtered, key=lambda x: x.get("_score", 0.0))

def _mark_note_if_redirect(triad: List[Dict[str, Any]], prompt: str) -> None:
    p = normalize(prompt)
    if "hydrangea" in p:
        # If none of the chosen items actually contains hydrangea, attach note to first MIX
        has_hyd = any("hydrangea" in [f.lower() for f in it.get("flowers", [])] for it in triad)
        if not has_hyd:
            for it in triad:
                if not _is_mono(it):
                    it["note"] = "Hydrangea is seasonal; showing the closest palette match."
                    break

def _enforce_copy_limit(triad: List[Dict[str, Any]], edge_case: Optional[str]) -> None:
    if not edge_case:
        return
    max_words = int(EDGE_REGISTERS.get("copy_max_words", 20))
    for it in triad:
        it["desc"] = _truncate_words(it.get("desc", ""), max_words)

def _ensure_two_mix_one_mono(triad: List[Dict[str, Any]], all_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Guarantee exactly 2 MIX + 1 MONO while keeping tier order."""
    mixes = [i for i in triad if not _is_mono(i)]
    monos = [i for i in triad if _is_mono(i)]
    if len(mixes) == 2 and len(monos) == 1:
        return triad

    # Too many monos → downgrade extras to MIX by swapping with next best MIX from same tier pool (if exists)
    if len(monos) > 1:
        # Replace highest-rank mono (except keep at least one) with best MIX of the same tier
        to_replace = monos[1:]
        for m in to_replace:
            tier = m.get("tier")
            pool = [x for x in all_items if x.get("tier") == tier and not _is_mono(x)]
            pool = sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True)
            if pool:
                idx = triad.index(m)
                triad[idx] = pool[0]
        # recompute
        mixes = [i for i in triad if not _is_mono(i)]
        monos = [i for i in triad if _is_mono(i)]

    # If still wrong (e.g., 3 MIX), try to upgrade Luxury to MONO if available, else Signature
    if len(monos) == 0:
        for pref_tier in ["Luxury", "Signature", "Classic"]:
            pool = [x for x in all_items if x.get("tier") == pref_tier and _is_mono(x)]
            pool = sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True)
            if pool:
                # replace current item of that tier
                for i, it in enumerate(triad):
                    if it.get("tier") == pref_tier:
                        triad[i] = pool[0]
                        return triad
    return triad

# ------------------------------------------------------------
# Public entrypoint
# ------------------------------------------------------------

def selection_engine(prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Returns exactly 3 curated items.
    - One per tier in order Classic → Signature → Luxury (LG sits within Luxury).
    - Exactly 2 MIX + 1 MONO.
    - Edge registers & notes applied.
    """
    context = context or {}
    p = normalize(prompt)

    # Emotion + edge-case
    emotion = detect_emotion(p, context)
    edge_case = detect_edge_case(p)

    # Iconic override intent (only lilies/roses/orchids)
    iconic_intent = None
    m = re.search(r"only\s+(lil(y|ies)|roses?|orchids?)", p)
    if m:
        if "lil" in m.group(0):
            iconic_intent = "lily"
        elif "rose" in m.group(0):
            iconic_intent = "rose"
        elif "orchid" in m.group(0):
            iconic_intent = "orchid"

    # Register for edge-case
    register = EDGE_REGISTERS.get(edge_case or "", {})

    # Score all catalog items
    scored: List[Dict[str, Any]] = []
    for item in CATALOG:
        # Skip items missing required fields
        if item.get("tier") not in TIER_ORDER or not item.get("palette"):
            continue

        base = int(item.get("weight", 50))
        w = _compute_weight(item, base, register, edge_case, iconic_intent)
        w *= _apply_lg_policy(item, emotion, edge_case)

        # If LG blocked multiplier is 0 → effectively remove
        if w <= 0:
            continue

        # Add a tiny deterministic tie-break so identical inputs stay stable:
        if item.get("id"):
            w += (int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % 997) / 1e9

        candidate = dict(item)
        candidate["_score"] = w
        # Strict edge gating + proper emotion (anchor) assignment
        edge_type = edge_case if edge_case in EDGE_REGISTERS else None
        candidate["emotion"] = emotion               # resolved anchor from detect_emotion(...)
        candidate["edge_case"] = bool(edge_type)
        candidate["_edge_type"] = edge_type          # optional, useful for logs/UI
        scored.append(candidate)

    if not scored:
        return []

    # Pick per tier with preference: MIX for Classic/Signature, any for Luxury (mono often chosen last)
    triad: List[Dict[str, Any]] = []
    for tier in TIER_ORDER:
        pool = _candidates_for_tier(scored, tier)
        want_mono = None
        if tier in ("Classic", "Signature"):
            want_mono = False
        best = _pick_best(pool, want_mono)
        if best:
            triad.append(best)

    # If we failed to get three (catalog scarcity), fill from next best irrespective of tier uniqueness
    # But keep only unique tiers (strong invariant)
    tiers_present = {it.get("tier") for it in triad}
    for tier in TIER_ORDER:
        if tier not in tiers_present:
            pool = _candidates_for_tier(scored, tier)
            if pool:
                triad.append(pool[0])
                tiers_present.add(tier)

    # Ensure we have exactly one per tier (drop extras if any weirdness)
    triad = _order_by_tier({t: max([i for i in triad if i.get("tier")==t], key=lambda x: x.get("_score", 0.0)) for t in TIER_ORDER if any(i.get("tier")==t for i in triad)}.values())  # type: ignore

    # Strong iconic override: try to replace the mono slot with the iconic species mono when present
    if iconic_intent:
        # find best mono of species; prefer Luxury, then Signature, then Classic
        for pref_tier in ["Luxury", "Signature", "Classic"]:
            pool = [x for x in scored if x.get("tier")==pref_tier and _is_mono(x) and _has_species(x, iconic_intent)]
            pool = sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True)
            if pool:
                # replace the current item for that pref_tier
                for i, it in enumerate(triad):
                    if it.get("tier")==pref_tier:
                        triad[i] = pool[0]
                        break
                break

    # Last-mile ritual: exactly 2 MIX + 1 MONO
    triad = _ensure_two_mix_one_mono(triad, scored)

    # Editorial redirection notes
    _mark_note_if_redirect(triad, p)

    # Edge-case copy limit
    _enforce_copy_limit(triad, edge_case)

    # Map fields for API output
    out = []
    for it in _order_by_tier(triad):
        out.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "desc": it.get("desc"),
            "image": it.get("image_url") or it.get("image"),
            "price": it.get("price_inr"),
            "currency": "INR",
            "emotion": it.get("emotion"),
            "tier": it.get("tier"),
            "packaging": it.get("packaging"),
            "mono": bool(it.get("mono")),
            "palette": list(it.get("palette") or []),
            "luxury_grand": bool(it.get("luxury_grand")),
            "note": it.get("note"),
            "edge_case": bool(it.get("edge_case")),
            "edge_type": it.get("_edge_type"), # optional exposure
        })
    return out

# Back-compat thin wrapper
def curate(prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return selection_engine(prompt, context or {})
