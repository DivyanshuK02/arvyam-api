# app/selection_engine.py
# Ultra-beginner-safe Selection Engine – Phase 1.2 + 1.3 (Edge-Case Playbooks)
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

import json, os, re, hashlib, logging
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Query
from enum import Enum
from pydantic import BaseModel
from hashlib import sha256
from datetime import datetime, timezone
import uuid
from fastapi import Response

__all__ = ["curate", "selection_engine", "normalize", "detect_emotion"]

# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------

ROOT = os.path.dirname(__file__)
log = logging.getLogger("arvyam.engine")

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
# Single source of truth for edge rules (Phase 1.4a-B):
EDGES = (EMOTION_KEYWORDS.get("edges", {}) or {})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, "edge_registers.json"), {})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, "tier_policy.json"), {"luxury_grand": {"blocked_emotions": []}})
SUB_NOTES = _load_json(os.path.join(RULES_DIR, "substitution_notes.json"), {"default": "Requested {from} is seasonal/unavailable; offering {alt} as the nearest alternative."})
ANCHOR_THRESHOLDS = _load_json(os.path.join(RULES_DIR, "anchor_thresholds.json"), {})
# Edge register keys (strict): only these four are considered "edge cases".
EDGE_CASE_KEYS = {"sympathy", "apology", "farewell", "valentine"}
FEATURE_MULTI_ANCHOR_LOGGING = os.getenv("FEATURE_MULTI_ANCHOR_LOGGING", "0") == "1" # Off by default

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

ONLY_ICONIC_RE = re.compile(r"\bonly\s+(lil(?:y|ies)|roses?|orchids?)\b", re.I)

VAGUE_TERMS = {"nice", "beautiful", "pretty", "some", "any", "simple", "best", "good", "flowers", "flower"}
GRAND_INTENT = {"grand","bigger","large","lavish","extravagant","50+","hundred","massive","most beautiful"}

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

def _intent_clarity(prompt: str, matched_keywords: int) -> float:
    """Return 0.0 when prompt is very generic/unclear; else 1.0."""
    p = normalize(prompt)
    words = re.findall(r"[a-z]+", p)
    if matched_keywords == 0 and (len(words) <= 4 or all(w in VAGUE_TERMS for w in words)):
        return 0.0
    return 1.0

def _has_grand_intent(prompt: str) -> bool:
    p = normalize(prompt)
    return any(w in p for w in GRAND_INTENT)

def _add_unclear_mix_note(triad: list) -> None:
    """Attach a single gentle clarifying note to the first MIX card, if none already."""
    for it in triad:
        if not it.get("mono") and not it.get("note"):
            it["note"] = "Versatile picks while you decide – tell us the occasion for a more personal curation."
            break

def _contains_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any((term or "").lower() in t for term in (terms or []))

def _matches_regex_list(text: str, patterns: list[str]) -> bool:
    if not patterns: return False
    for pat in patterns:
        try:
            if re.search(pat, text, flags=re.I):
                return True
        except re.error:
            continue
    return False

def _has_proximity(text: str, a: str, b: str, window: int) -> bool:
    if not a or not b or window <= 0: return False
    tokens = re.findall(r"\w+", text.lower())
    pos_a = [i for i,w in enumerate(tokens) if w == a.lower()]
    pos_b = [i for i,w in enumerate(tokens) if w == b.lower()]
    return any(abs(i-j) <= window for i in pos_a for j in pos_b)

def _is_false_friend(text: str, phrases: list[str]) -> bool:
    return _contains_any(text, phrases or [])

def _stable_hash_u32(s: str) -> int:
    h = hashlib.sha256((s or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _rotation_index(seed: str, k: int) -> int:
    if k <= 1: return 0
    return _stable_hash_u32(seed) % k

def _truncate_words(text: str, max_words: int) -> str:
    words = [w for w in (text or "").split()]
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]).rstrip() + "…"

def _suppress_recent(items: list[dict], recent_set: set[str]) -> list[dict]:
    if not recent_set:
        return items
    # Keep order, drop anything recently shown
    return [it for it in items if _stable_id(it) not in recent_set]

def _has_species_match(item: Dict[str, Any], species_name: str) -> bool:
    s = (species_name or "").lower()
    # primary: catalog["flowers"] is a list
    for f in item.get("flowers", []) or []:
        if (f or "").lower() == s:
            return True
    # fallback: legacy text field
    raw = (item.get("species_raw") or "").lower()
    return s in raw if raw else False

def _has_emotion_match(item: Dict[str, Any], emotion: str) -> bool:
    target = (emotion or "").lower()
    v = item.get("emotion")
    if isinstance(v, str) and target in v.lower():
        return True
    v = item.get("emotions")
    if isinstance(v, str) and target in v.lower():
        return True
    if isinstance(v, list) and any(target == (x or "").lower() for x in v):
        return True
    return False

# ---------------- Phase-1.4a helpers (drop-in) ----------------
def _stable_id(it: dict) -> str:
    return str(it.get("id", ""))

def _apply_edge_register_filters(pool: list[dict], edge_type: str) -> list[dict]:
    """Score+filter pool by edge register (palette/species/LG). Non-destructive."""
    regs = EDGE_REGISTERS.get(edge_type, {}) or {}  # Fixed: was missing {}
    # 1) Block LG where required
    lg_policy = regs.get("lg_policy", "allow")
    allow_lg_mix = bool(regs.get("allow_lg_in_mix", True))
    allow_lg_mono = bool(regs.get("allow_lg_in_mono", True))
    def _lg_ok(it, is_mono=False):
        if lg_policy == "block": return False
        return allow_lg_mono if is_mono else allow_lg_mix

    # 2) Light scoring for palette/species steering
    targets = set((regs.get("palette_targets") or []))
    avoid   = set((regs.get("palette_avoid") or []))
    prefer_species = set((regs.get("species_prefer") or []))
    avoid_species  = set((regs.get("species_avoid") or []))
    t_boost = float(regs.get("palette_target_boost", 1.0))
    a_pen   = float(regs.get("palette_avoid_penalty", 1.0))

    scored = []
    for it in pool:
        score = 1.0
        pal = set(it.get("palette", []) or [])
        if pal & targets: score *= t_boost
        if pal & avoid:   score *= a_pen
        flw = set((it.get("flowers") or []))
        if flw & prefer_species: score *= 1.05
        if flw & avoid_species:  score *= 0.95
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored if _lg_ok(it, False)]

def _pick_mono_slot(available: list[dict], selected_species: str|None, edge_type: str|None) -> dict|None:
    """Choose deterministic MONO item, honoring species + edge mono_must_include when possible."""
    pool = list(available)
    # Edge mono_must_include steering
    if edge_type:
        must = set(EDGE_REGISTERS.get(edge_type, {}).get("mono_must_include", []) or [])
        if must:
            pool = [it for it in pool if set(it.get("flowers") or []) & must] or pool
    # Requested species
    if selected_species:
        sp = [it for it in pool if _has_species_match(it, selected_species)]
        if sp: pool = sp

    # Respect LG policy for mono under the active edge (minimal: allow/deny)
    if edge_type:
        regs = EDGE_REGISTERS.get(edge_type, {}) or {}
        lg_policy = regs.get("lg_policy", "allow")
        allow_lg_mono = bool(regs.get("allow_lg_in_mono", True))
        if lg_policy == "block" or not allow_lg_mono:
            pool = [it for it in pool if not it.get("luxury_grand")]

    if not pool: return None
    idx = _rotation_index("mono:"+_stable_id(pool[0]), len(pool))
    mono_item = pool[idx]
    mono_item = dict(mono_item); mono_item["mono"] = True
    return mono_item

def _pick_mix_slots(available: list[dict], need: int, seed: str, seen: set[str]) -> list[dict]:
    """Fill remaining slots with MIX items deterministically, no RNG."""
    out = []
    pool = [it for it in available if _stable_id(it) not in seen]
    for i in range(need):
        if not pool: break
        idx = _rotation_index(f"{seed}:mix:{i}", len(pool))
        it = dict(pool.pop(idx))
        it["mono"] = False
        seen.add(_stable_id(it))
        out.append(it)
    return out

def _ensure_triad_or_500(items: list[dict]) -> None:
    """Production-safe contract check: exactly 3 items; exactly 1 mono=True."""
    triad_ok = isinstance(items, list) and len(items) == 3
    mono_count = sum(1 for it in items if it.get("mono") is True)
    mono_ok = (mono_count == 1)

    if triad_ok and mono_ok:
        return

    # Server-side detail, client-safe surface
    log.error(
        "[TRIAD_CONTRACT] invalid triad: len=%s mono_count=%s ids=%s",
        (len(items) if isinstance(items, list) else None),
        mono_count,
        [getattr(it, "get", lambda *_: None)("id") for it in (items or [])],
    )
    # Keep external message generic (no schema internals)
    raise HTTPException(status_code=500, detail="Internal catalog data error")

def _fallback_fill_triad(items: list[dict], catalog: list[dict]) -> list[dict]:
    # Ensure exactly 3 and exactly one mono
    out = items[:3] if len(items) >= 3 else items + catalog[: 3 - len(items)]
    mono_count = sum(1 for it in out if it.get("mono") is True)
    if mono_count == 0:
        # Promote first mono candidate from catalog if available
        for it in catalog:
            if it.get("mono") is True and it not in out:
                out[-1] = it; break
    elif mono_count > 1:
        # Reduce to one mono: demote extras to mix items from catalog
        mix_fill = (it for it in catalog if it.get("mono") is False and it not in out)
        i = 0
        for idx, it in enumerate(out):
            if it.get("mono") is True and i > 0:
                out[idx] = next(mix_fill, it)
            if it.get("mono") is True: i += 1
    return out

def _compute_detected_emotions(scores: dict[str,float]) -> list[dict]:
    """Return ≤2 entries based on JSON thresholds; omit when no near-tie."""
    cfg = (ANCHOR_THRESHOLDS.get("multi_anchor_logging") or {})
    if not cfg.get("enabled", True): return []
    max_entries = int(cfg.get("max_entries", 2))
    score2_min = float(cfg.get("score2_min", 0.25))
    delta_max  = float(cfg.get("delta_max", 0.15))
    # order anchors by score desc
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered: return []
    top1 = ordered[0]
    out = [{"anchor": top1[0], "score": round(float(top1[1]), 2)}]
    if len(ordered) > 1:
        top2 = ordered[1]
        if float(top2[1]) >= score2_min and (float(top1[1]) - float(top2[1])) <= delta_max:
            out.append({"anchor": top2[0], "score": round(float(top2[1]), 2)})
    return out[:max_entries]

def detect_emotion(prompt: str, context: dict | None) -> Tuple[str, Optional[str], Dict[str, float]]:
    """
    Core emotion and intent detection logic. Returns (resolved_anchor, edge_type, scores)
    """
    p = normalize(prompt)

    # Edge registers short-circuit routing; item stamping uses this edge_type exactly once.
    # Canary tests (CI): one exact/regex/proximity sample per register; fail CI if any canary stops matching.
    # 0) Edge case check (highest priority) from emotion_keywords["edges"]
    # Check for canonical edge cases first (highest priority)
    for edge_type, rules in (EDGES or {}).items():
        p = p  # already normalized above
        # exact
        for phrase in (rules.get("exact") or []):
            if phrase and phrase.lower() in p:
                anchor = (EDGE_REGISTERS.get(edge_type, {}) or {}).get("emotion_anchor", "general")
                return anchor, edge_type, {}
        # contains_any
        if any(tok and tok.lower() in p for tok in (rules.get("contains_any") or [])):
            anchor = (EDGE_REGISTERS.get(edge_type, {}) or {}).get("emotion_anchor", "general")
            return anchor, edge_type, {}
        # regex
        patterns = []
        for r in (rules.get("regex") or []):
            patterns.append(r if isinstance(r, str) else r.get("pattern", ""))
        if patterns and _matches_regex_list(p, patterns):
            anchor = (EDGE_REGISTERS.get(edge_type, {}) or {}).get("emotion_anchor", "general")
            return anchor, edge_type, {}
        # proximity_pairs
        for spec in (rules.get("proximity_pairs") or []):
            a = spec.get("a",""); b = spec.get("b",""); w = int(spec.get("window",2))
            if a and b and _has_proximity(p, a, b, window=w):
                anchor = (EDGE_REGISTERS.get(edge_type, {}) or {}).get("emotion_anchor", "general")
                return anchor, edge_type, {}
    
    # --- KEYWORD BUCKET SCORING (fix) ---
    scores: Dict[str, float] = {}
    tokens = _tokenize(prompt)
    # Buckets are per-anchor keyword lists in JSON
    buckets = EMOTION_KEYWORDS.get("keywords", {}) or {}
    
    # Scope LG block list policy (only when grand intent is present)
    blocked = (TIER_POLICY.get("luxury_grand", {}) or {}).get("blocked_emotions") or []
    if _has_grand_intent(prompt) and blocked:
        scores = {k: v for k, v in scores.items() if k not in blocked}
    
    for anchor, words in buckets.items():          # <-- iterate buckets, not anchors[]
        hits = 0
        for w in (words or []):
            if (w or "").lower() in p:
                hits += 1
        if hits > 0:
            scores[anchor] = float(hits)

    # Optional LG hint
    if _has_grand_intent(prompt):
        scores["luxury_grand"] = scores.get("luxury_grand", 0.0) + 1.0

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    resolved_anchor = sorted_scores[0][0] if sorted_scores else "general"

    # If 'general' but there is another scored anchor, prefer that next best
    if resolved_anchor == "general" and len(sorted_scores) > 1:
        resolved_anchor = sorted_scores[1][0]

    # At this point no edge was triggered, so edge_type stays None
    edge_type = None
    return resolved_anchor, edge_type, scores

def find_iconic_species(prompt_norm: str) -> Optional[str]:
    """Find a single iconic species mentioned in the prompt."""
    if ONLY_ICONIC_RE.search(prompt_norm):
        for species, terms in ICONIC_SPECIES.items():
            if any(term in prompt_norm for term in terms):
                return species
    return None

def assign_tiers(triad: List[Dict], context: Dict) -> List[Dict]:
    # Count monos (should be 1, but handle 0/3 for robustness)
    monos = [it for it in triad if it.get("mono")]
    mixes = [it for it in triad if not it.get("mono")]
    mono_count = len(monos)
    if mono_count == 0:
        # All mix; assign sequentially
        tiers = ["Classic", "Signature", "Luxury"]
        for i, it in enumerate(triad):
            it["tier"] = tiers[i % 3]
        return triad
    elif mono_count == 1:
        # Standard: Classic/Signature to mixes, Luxury to mono (if grand)
        mixes[0]["tier"] = "Classic"
        mixes[1]["tier"] = "Signature"
        if context.get("resolved_anchor") == "luxury_grand" or _has_grand_intent(context["prompt"]):
            monos[0]["tier"] = "Luxury"
        else:
            monos[0]["tier"] = "Luxury"  # Or Signature if non-grand
        return triad
    else:
        # Invalid; fallback sequential
        tiers = ["Classic", "Signature", "Luxury"]
        for i, it in enumerate(triad):
            it["tier"] = tiers[i % 3]
        return triad

def find_and_assign_note(triad: list, selected_species: Optional[str],
                         selected_emotion: Optional[str], prompt_text: str = "") -> None:
    # Only handles "species not found" for now
    found_species = any(selected_species in it.get("flowers", []) for it in triad) if selected_species else False
    if selected_species and not found_species:
        note = SUB_NOTES.get(
            "species_not_found",
            "We couldn't find a {species} bouquet at the moment; offering a similar style."
        ).replace("{species}", selected_species)

        # Prefer attaching to a MIX (non-mono) card; fall back to Classic → Luxury → any
        target = next((it for it in triad if not it.get("mono")), None)
        if target is None:
            # all mono? unlikely, but be safe
            target = triad[0] if triad else None

        if target is not None:
            target["note"] = note

def _transform_for_api(items: List[Dict], resolved_anchor: Optional[str]) -> List[Dict]:
    out: List[Dict] = []
    for it in items or []:
        # Map fields with fallbacks
        image = it.get("image") or it.get("image_url") or ""
        price = it.get("price") if it.get("price") is not None else it.get("price_inr")
        currency = it.get("currency") or ("INR" if price is not None else None)
        if not image or price is None or not currency:
            # Log error, skip to avoid invalid item
            log.warning(f"Invalid item in triad: {it.get('id')}")
            continue
        item = dict(it)
        item["image"] = image
        item["price"] = price
        item["currency"] = currency
        item["emotion"] = item.get("emotion") or (resolved_anchor or "general")
        # Preserve edge stamps
        if "edge_case" in it:
            item["edge_case"] = it["edge_case"]
            item["edge_type"] = it["edge_type"]
        if "note" in it:
            item["note"] = it["note"]
        out.append(item)
    if len(out) != 3:
        raise HTTPException(status_code=500, detail="Internal catalog data error")
    return out

def _take_n_wrapped(sorted_pool: list[dict], start_idx: int, n: int) -> list[dict]:
    L = len(sorted_pool)
    if L == 0:
        return []
    return [sorted_pool[(start_idx + i) % L] for i in range(min(n, L))]
    
def _backfill_to_three(items: list[dict], catalog: list[dict], context: dict) -> list[dict]:
    """Backfills the triad to exactly 3 items after suppression."""
    current_ids = {_stable_id(it) for it in items}
    fill_needed = 3 - len(items)
    
    if fill_needed <= 0:
        return items
    
    # Emotion pool refill
    emotion_pool = [it for it in catalog if _has_emotion_match(it, context.get("resolved_anchor", "general")) and _stable_id(it) not in current_ids]
    emotion_fill = _pick_mix_slots(emotion_pool, fill_needed, context.get("prompt_hash", "seed") + ":backfill_emotion", current_ids)
    items.extend(emotion_fill)
    current_ids.update({_stable_id(it) for it in emotion_fill})
    fill_needed = 3 - len(items)
    
    if fill_needed <= 0:
        return items

    # General pool refill
    general_pool = [it for it in catalog if _has_emotion_match(it, "general") and _stable_id(it) not in current_ids]
    general_fill = _pick_mix_slots(general_pool, fill_needed, context.get("prompt_hash", "seed") + ":backfill_general", current_ids)
    items.extend(general_fill)
    current_ids.update({_stable_id(it) for it in general_fill})
    fill_needed = 3 - len(items)
    
    if fill_needed <= 0:
        return items

    # Any pool refill
    any_pool = [it for it in catalog if _stable_id(it) not in current_ids]
    any_fill = _pick_mix_slots(any_pool, fill_needed, context.get("prompt_hash", "seed") + ":backfill_any", current_ids)
    items.extend(any_fill)
    
    return items


def _pick_mono(catalog: list[dict], context: dict) -> dict | None:
    """Helper for picking the mono item with edge-case filters."""
    edge_type = context.get("edge_type")
    
    pool = list(catalog)
    if edge_type:
        pool = _apply_edge_register_filters(pool, edge_type)

    must = set(context.get("edge_must_include_species", []))
    if must:
        pool = [x for x in pool if any(sp in must for sp in (x.get("flowers") or []))]
    
    # Apply LG mono policy from edge registers
    if edge_type:
        regs = EDGE_REGISTERS.get(edge_type, {}) or {}
        lg_policy = regs.get("lg_policy", "allow")
        allow_lg_mono = bool(regs.get("allow_lg_in_mono", True))
        if lg_policy == "block" or not allow_lg_mono:
            pool = [it for it in pool if not it.get("luxury_grand")]

    if not pool: return None
    
    idx = _rotation_index("mono:" + context.get("prompt_hash", "seed"), len(pool))
    mono_item = pool[idx]
    mono_item = dict(mono_item); mono_item["mono"] = True
    return mono_item


def selection_engine(prompt: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Main function for selecting a curated triad of items.
    """
    available_catalog = list(CATALOG)
    prompt_norm = normalize(prompt)
    resolved_anchor, edge_type, _scores = detect_emotion(prompt, context or {})
    is_edge = edge_type is not None
    selected_species = find_iconic_species(prompt_norm)
    
    # Carry the resolved anchor
    context["resolved_anchor"] = resolved_anchor
    context["edge_type"] = edge_type
    
    ph = (context or {}).get("prompt_hash", "")
    seed = ph or "seed"

    # 1. Build candidate pools
    emotion_pool = [it for it in available_catalog if _has_emotion_match(it, resolved_anchor)]
    if is_edge:
        # steer by palette/species/LG from registers
        emotion_pool = _apply_edge_register_filters(emotion_pool, edge_type)

    # 2. Build the triad as 1 MONO + 2 MIX (deterministic)
    triad: list[dict] = []
    seen: set[str] = set()

    # MONO slot first (species/edge aware)
    mono_item = _pick_mono_slot(available_catalog, selected_species, edge_type)

    if mono_item:
        triad.append(mono_item); seen.add(_stable_id(mono_item))

    # MIX slots from emotion_pool (fall back to 'general' if needed)
    mix_needed = 2
    mix_filled = _take_n_wrapped(emotion_pool, _rotation_index(seed, len(emotion_pool)), mix_needed)
    triad.extend(mix_filled)
    seen.update({_stable_id(it) for it in mix_filled})

    # Backfill if still short
    if len(triad) < 3:
        general_pool = [it for it in available_catalog if _has_emotion_match(it, "general") and _stable_id(it) not in seen]
        triad.extend(_take_n_wrapped(general_pool, _rotation_index(seed + ":general", len(general_pool)), 3 - len(triad)))

    # Final safety: if still short, pad from any available
    if len(triad) < 3:
        any_pool = [it for it in available_catalog if _stable_id(it) not in seen]
        triad.extend(_take_n_wrapped(any_pool, _rotation_index(seed + ":any", len(any_pool)), 3 - len(triad)))


    _ensure_triad_or_500(triad)

    final_triad = assign_tiers(triad, context)  # preserves 3 cards; keep 'mono' flags on items

    # A) Stamp edge metadata on items
    if is_edge:
        for it in final_triad:
            it["edge_case"] = True
            it["edge_type"] = edge_type

    # B) Handle substitutions & notes (copy, not price)
    find_and_assign_note(final_triad, selected_species, resolved_anchor, prompt_text=context.get("prompt",""))
    if not _intent_clarity(prompt, len(_scores)):
        _add_unclear_mix_note(final_triad)

    # 4. Filter by suppress list
    if context.get("recent_ids"):
        final_triad = _suppress_recent(final_triad, set(context["recent_ids"]))
        if len(final_triad) < 3:
            final_triad = _backfill_to_three(final_triad, CATALOG, context)

    _ensure_triad_or_500(final_triad)

    # 5. Add meta info + logging
    # ---- Phase-1.4a meta + logs (flag-gated) ----
    meta_detected = []
    if FEATURE_MULTI_ANCHOR_LOGGING:
        meta_detected = _compute_detected_emotions(_scores or {})
        # compact, privacy-safe log
        near_tie_compact = [{"a": m["anchor"], "s": m["score"]} for m in meta_detected][:2]
        ph = (context or {}).get("prompt_hash", "")
        log_record = {
            "ts": (context or {}).get("ts",""),
            "rid": (context or {}).get("request_id",""),
            "prompt_hash": "sha256:" + (ph[:8] if ph else ""),
            "resolved": resolved_anchor,
            "edges": {"case": is_edge, "type": edge_type},
            "near_tie": near_tie_compact
        }
        log.info(json.dumps(log_record, ensure_ascii=False))  # example sink

    # Guarantee emotion on each card for API schema; fallback to detected anchor
    for c in final_triad:
        c["emotion"] = c.get("emotion") or context.get("resolved_anchor") or "general"
        
    # 6. Transform and return
    # No transform here, handled by the calling endpoint to make the endpoint logic more clear
    if FEATURE_MULTI_ANCHOR_LOGGING and meta_detected:
        # attach via context for the endpoint to place into response/headers
        context["__meta_detected_emotions"] = meta_detected

    return final_triad


# FastAPI endpoints
app = FastAPI()

class CurateRequest(BaseModel):
    prompt: Optional[str] = None
    q: Optional[str] = None
    recent_ids: Optional[List[str]] = None
    run_count: Optional[int] = 0

@app.post("/curate", tags=["public"])
async def curate_post(req: CurateRequest):
    prompt = (req.prompt or req.q or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Invalid input")

    # Always seed context with a hash to avoid KeyError downstream
    norm = normalize(prompt)
    prompt_hash = sha256(norm.encode("utf-8")).hexdigest()

    context = {
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "recent_ids": req.recent_ids,
        "run_count": req.run_count or 0,
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
    }

    triad = selection_engine(prompt, context)
    payload_items = _transform_for_api(triad, context.get("resolved_anchor"))

    payload = {
        "items": payload_items,
        "edge_case": any(i.get("edge_case") for i in triad),
    }

    resp = Response(content=json.dumps(payload), media_type="application/json")

    # Optional QA-only header (feature-flag + config-gated)
    cfg = (ANCHOR_THRESHOLDS.get("multi_anchor_logging") or {})
    if FEATURE_MULTI_ANCHOR_LOGGING and cfg.get("emit_header", False):
        meta = context.get("__meta_detected_emotions") or []
        if meta:
            resp.headers["X-Detected-Emotions"] = ",".join(
                f"{m['anchor']}:{m['score']}" for m in meta[:2]
            )
    return resp

@app.post("/api/curate", tags=["public"])
async def curate_post_api(q: Optional[str] = Query(None),
                          prompt: Optional[str] = Query(None),
                          recent_ids: Optional[List[str]] = Query(None)):
    return await curate_post(CurateRequest(q=q, prompt=prompt, recent_ids=recent_ids))

@app.get("/curate", tags=["public"])
async def curate_get(q: Optional[str] = Query(None),
                     prompt: Optional[str] = Query(None),
                     recent_ids: Optional[List[str]] = Query(None)):
    return await curate_post(CurateRequest(q=q, prompt=prompt, recent_ids=recent_ids))

# For backward compatibility
def curate(*args, **kwargs):
    """Legacy function for backward compatibility"""
    return selection_engine(*args, **kwargs)
