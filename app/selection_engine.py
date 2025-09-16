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
from fastapi import FastAPI, HTTPException, Query # Added Query import

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

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # lower, collapse whitespace, unify quotes, strip punctuation we don’t match on
    t = text.lower().replace("’", "'")
    return " ".join(t.replace(",", " ").split())

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
            it["note"] = "Versatile picks while you decide — tell us the occasion for a more personal curation."
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

def detect_edge_register(prompt: str) -> Optional[str]:
    p = normalize(prompt)
    if not p:
        return None

    # Priority order from JSON (sympathy > apology > farewell > valentine)
    # The keys in `EDGE_KEYWORDS` have the `-enriched` suffix
    priority_keys = EDGE_KEYWORDS.get("priority", {})
    sorted_keys = sorted(priority_keys.keys(), key=lambda k: priority_keys[k], reverse=True)

    for key in sorted_keys:
        enr_key = f"{key}_enriched"
        enr = EDGE_KEYWORDS.get(enr_key) or {}

        # 0) Block false-friends first
        if _is_false_friend(p, enr.get("false_friends")):
            continue

        # 1) Exact match (highest precision)
        if _contains_any(p, enr.get("exact") or []):
            return key

        # 2) Contains any
        if _contains_any(p, enr.get("contains_any") or []):
            return key

        # 3) Regex match
        if _matches_regex_list(p, enr.get("regex") or []):
            return key

        # 4) Proximity match
        prox = enr.get("proximity_pairs")
        if isinstance(prox, list):
            for pair in prox:
                a, b = pair.get("a"), pair.get("b")
                window = int(pair.get("window", 0) or 0)
                if a and b and window and _has_proximity(p, a, b, window):
                    return key

        # 5) Legacy fallback
        base = EDGE_KEYWORDS.get(key)
        if isinstance(base, list) and base and _contains_any(p, base):
            return key

    return None

def detect_emotion(prompt: str, context: dict | None) -> Tuple[str, Optional[str], Dict[str, float]]:
    p = normalize(prompt)

    # Edge registers short-circuit routing; item stamping uses this edge_type exactly once.
    # Canary tests (CI): one exact/regex/proximity sample per register; fail CI if any canary stops matching.
    edge_type = detect_edge_register(p)
    if edge_type:
        resolved_anchor = EDGE_REGISTERS.get(edge_type, {}).get("emotion_anchor", None)
        if resolved_anchor:
            return resolved_anchor, edge_type, {}

    # 1) Safe context hint (only accept known anchors)
    anchors: List[str] = EMOTION_KEYWORDS.get("anchors", []) or []
    hint = (context or {}).get("emotion_hint", "") or ""
    hint = hint.strip()
    if hint and hint in anchors:
        return hint, None, {}
    
    # Tables
    exact_map = EMOTION_KEYWORDS.get("exact", {}) or {}
    combos    = EMOTION_KEYWORDS.get("combos", []) or {}
    buckets   = EMOTION_KEYWORDS.get("keywords", {}) or {}
    disamb    = EMOTION_KEYWORDS.get("disambiguation", []) or {}

    # 2) Disambiguation rules (highest precision)
    for rule in disamb:
        when_any = rule.get("when_any", [])
        if not when_any or not any(w in p for w in map(str.lower, when_any)):
            continue
        for branch in rule.get("route", []):
            if_any = branch.get("if_any", [])
            if if_any and any(t in p for t in map(str.lower, if_any)):
                return branch.get("anchor"), None, {}
            if "else" in branch:
                return branch["else"], None, {}

    # 3) Exact phrase map (case-insensitive contains)
    for phrase, anchor in exact_map.items():
        if phrase.lower() in p:
            return anchor, None, {}

    # 4) Combos: all terms present
    for combo in combos:
        terms = [t.lower() for t in combo.get("all", [])]
        if terms and all(t in p for t in terms):
            return combo.get("anchor"), None, {}

    # 5) Enriched (optional) — regex & proximity_pairs
    enr = EMOTION_KEYWORDS.get("enriched", {}) or {}
    if enr.get("false_friends") and _is_false_friend(p, enr["false_friends"]):
        # fall through to buckets without enriched scoring
        pass
    else:
        # regex with anchor
        for spec in enr.get("regex", []):
            if _matches_regex_list(p, [spec.get("pattern","")]):
                return spec.get("anchor"), None, {}
        # proximity pairs with anchor
        for spec in enr.get("proximity_pairs", []):
            if _has_proximity(p, spec.get("a",""), spec.get("b",""), int(spec.get("window",2))):
                return spec.get("anchor"), None, {}

    # 6) Buckets: score by keyword hits; tie-break by anchors[] order
    scores = {a: 0 for a in anchors}
    for a, words in (buckets or {}).items():
        scores[a] = sum(1 for w in (words or []) if w.lower() in p)

    # pick best; if tie, prefer earlier in anchors[]
    best = max(anchors, key=lambda a: (scores.get(a,0), -anchors.index(a))) if anchors else None
    if best and scores.get(best, 0) > 0:
        # Normalize scores for logging
        max_score = scores.get(best) or 1
        scores_norm = {a: s / max_score for a, s in scores.items()}
        return best, None, scores_norm

    # 7) Fallback default
    return anchors[0] if anchors else "Affection/Support", None, {}

def _is_canonical_edge(resolved_emotion: str, prompt_norm: str) -> tuple[bool, Optional[str]]:
    """
    Returns (edge_case_bool, edge_type).
    Step 1: The resolved emotion must be configured as a potential edge in EDGE_REGISTERS.
    Step 2: The prompt must contain at least one canonical phrase from EDGE_KEYWORDS[emotion].
    """
    reg = EDGE_REGISTERS.get(resolved_emotion) or {}
    if not reg.get("edge_case"):
        return False, None
    phrases = EDGE_KEYWORDS.get(resolved_emotion) or []
    for p in phrases:
        if p and p.lower() in prompt_norm:
            return True, resolved_emotion # edge confirmed for that emotion
    return False, None

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
    return " ".join(words[:max_words]).rstrip(",.;:!—-") + "…"


def _enforce_copy_limit(text: str, edge_type: Optional[str]) -> str:
    """
    Truncates `text` to the copy_max_words for the given edge_type.
    Pass the register key (e.g., "sympathy", "apology", "farewell", "valentine") or None.
    Caps to Phase-1 hard fence (≤20 words). Called only when an edge case is active.
    """
    if not edge_type:
        return text
    reg = EDGE_REGISTERS.get(edge_type) or {}
    max_words = int(reg.get("copy_max_words", 0) or 0)
    if max_words <= 0:
        return text

    words = (text or "").strip().split()
    if len(words) <= max_words:
        return text
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

        spref  = [x.lower() for x in register.get("species_prefer", [])]
        savoid = [x.lower() for x in register.get("species_avoid", [])]
        if spref and any(s in flowers for s in spref):
            w *= 1.10
        if savoid and any(s in flowers for s in savoid):
            w *= 0.90

        # Sympathy lily guard
        if edge_case == "sympathy" and "lily" in flowers:
            w *= 1.25

    # Iconic "only X" override intent
    if iconic_intent:
        if _is_mono(item) and iconic_intent in flowers:
            w *= 10.0
        elif iconic_intent in flowers:
            w *= 1.5
        else:
            w *= 0.6

    # Editorial base weight
    w *= max(0.1, float(item.get("weight", 50)) / 50.0)
    return w

def _apply_lg_policy(item: Dict[str, Any], emotion: str, edge_case: Optional[str]) -> float:
    if not item.get("luxury_grand"):
        return 1.0

    # Global block-list
    blocked = [x.lower() for x in TIER_POLICY.get("luxury_grand", {}).get("blocked_emotions", [])]
    if emotion.lower() in blocked:
        return 0.0

    # Edge soft policy
    if edge_case and edge_case in EDGE_CASE_KEYS:
        reg = EDGE_REGISTERS.get(edge_case, {})
        if str(reg.get("lg_policy", "")).lower() == "block":
            return 0.0
        return float(reg.get("lg_weight_multiplier", 1.0))

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

def _apply_substitution_notes(triad: List[Dict[str, Any]], redirect_from: Optional[str], alts: List[str]) -> None:
    """
    If user asked for an unavailable species (redirect_from), attach a note to each
    triad item whose flowers include one of the chosen substitutes in `alts`.
    Fallback: if nothing matches, put the note on the first MIX item only.
    """
    if not redirect_from or not alts:
        return
    note_tpl = SUB_NOTES.get(redirect_from.lower(), SUB_NOTES.get("default", "Requested {from} is seasonal/unavailable; offering {alt} as the nearest alternative."))
    matched = 0
    for it in triad:
        flowers = [f.lower() for f in (it.get("flowers") or [])]
        hit = next((a for a in alts if a.lower() in flowers), None)
        if hit:
            it["note"] = note_tpl.format(**{"from": redirect_from, "alt": hit})
            matched += 1
    if matched == 0:
        # fallback: first MIX only (keeps UI clean)
        for it in triad:
            if not it.get("mono"):
                it["note"] = note_tpl.format(**{"from": redirect_from, "alt": alts[0]})
                break

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
def _apply_recent_filter(cands: List[Dict[str, Any]], recent_set: set) -> List[Dict[str, Any]]:
    if not cands: return cands
    filtered = [c for c in cands if c.get("id") not in recent_set]
    return filtered if filtered else cands

def _pick_rotated(cands: List[Dict[str, Any]], tier_name: str, session_id: str, run_count: int, prompt_norm: str) -> Optional[Dict[str, Any]]:
    if not cands: return None
    seed = f"{session_id}|{run_count}|{prompt_norm}|{tier_name}"
    idx = _rotation_index(seed, len(cands))
    return cands[idx]

def log_near_tie_compact(request_id: str, prompt_hash: str, resolved_anchor: str, edge_type: str, near_tie: List[Dict[str, Any]]):
    """
    Compact logger for near-tie emotion detection.
    (This is a placeholder for a real logging implementation)
    """
    log_data = {
        "req_id": request_id,
        "prompt_hash": prompt_hash,
        "anchor": resolved_anchor,
        "edge": edge_type,
        "near_tie": near_tie,
    }
    # In a real system, you'd send this to a dedicated logging stream
    # print(json.dumps(log_data))
    pass

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

    # Dedupe window (Phase-1): UI may supply `recent_ids` (list[str]) to avoid repeat cards
    # across a short session. Engine should SKIP any candidate whose `id` is in this set,
    # then fall back deterministically (rotation index) if a tier empties.
    recent_ids = (context or {}).get("recent_ids") or []
    recent_set = {rid for rid in recent_ids if isinstance(rid, str)}
    original_catalog = list(CATALOG)
    
    # single source of truth; detect_emotion() already handles edge-first
    resolved_anchor, edge_type, anchor_scores = detect_emotion(p, context)
    is_edge = bool(edge_type) and edge_type in EDGE_CASE_KEYS
    
    # Optional: Log near-tie emotions
    if FEATURE_MULTI_ANCHOR_LOGGING and anchor_scores:
        thresholds = ANCHOR_THRESHOLDS.get("multi_anchor_logging", {})
        top = sorted(anchor_scores.items(), key=lambda kv: kv[1], reverse=True)
        if top:
            near_tie = [{"a": top[0][0], "s": round(top[0][1], 2)}]
            if len(top) > 1:
                s1, s2 = top[0][1], top[1][1]
                if s2 >= thresholds.get("score2_min", 0.25) and (s1 - s2) <= thresholds.get("delta_max", 0.20):
                    near_tie.append({"a": top[1][0], "s": round(s2, 2)})
            
            # Placeholder for logging function
            log_near_tie_compact(
                request_id=context.get("request_id", ""),
                prompt_hash=hashlib.sha256(p.encode()).hexdigest(),
                resolved_anchor=resolved_anchor,
                edge_type=edge_type or "none",
                near_tie=near_tie
            )
    
    # --- intent clarity & LG dampener ---
    bucket = (EMOTION_KEYWORDS.get("keywords", {}) or {}).get(resolved_anchor, [])
    matched_keywords = sum(1 for k in bucket if normalize(k) in normalize(prompt))
    clarity = _intent_clarity(prompt, matched_keywords)
    soft_lg_multiplier = 1.0
    if clarity == 0.0 and not _has_grand_intent(prompt):
        try:
            budget = int((context or {}).get("budget_inr", 0) or 0)
        except Exception:
            budget = 0
        if budget < 4999:
            soft_lg_multiplier = 0.8

    redirect_from: Optional[str] = None
    redirect_alts: List[str] = []

    # Check for specific substitution case (e.g., hydrangea)
    if "hydrangea" in p:
        redirect_from = "hydrangea"
        # Dummy alternatives for now; a real implementation would find these
        redirect_alts = ["lily", "chrysanthemum"]

    # Iconic override intent (only lilies/roses/orchids)
    iconic_intent = None
    m = ONLY_ICONIC_RE.search(p)
    if m:
        g = m.group(1).lower()
        if g.startswith("lil"):
            iconic_intent = "lily"
        elif g.startswith("rose"):
            iconic_intent = "rose"
        elif g.startswith("orchid"):
            iconic_intent = "orchid"

    # Score all catalog items
    scored: List[Dict[str, Any]] = []
    
    # Filter candidates based on recent_ids
    candidates = [c for c in original_catalog if c["id"] not in recent_set]
    
    # if filtering leaves too few to satisfy 2 MIX + 1 MONO, fall back to the full set to keep guarantees
    if len(candidates) < 3:
        candidates = original_catalog

    # The rest of the logic for scoring and selection will remain the same.
    for item in candidates:
        # Skip items missing required fields
        if item.get("tier") not in TIER_ORDER or not item.get("palette"):
            continue

        base = int(item.get("weight", 50))
        register = EDGE_REGISTERS.get(edge_type or "", {})
        w = _compute_weight(item, base, register, edge_type, iconic_intent)
        w *= _apply_lg_policy(item, resolved_anchor, edge_type)

        # dampen LG softly for unclear intent (unless 'grand' or budget qualifies)
        if item.get("luxury_grand"):
            w *= soft_lg_multiplier

        # If LG blocked multiplier is 0 → effectively remove
        if w <= 0:
            continue

        # Add a tiny deterministic tie-break so identical inputs stay stable:
        if item.get("id"):
            w += (int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % 997) / 1e9

        candidate = dict(item)
        candidate["_score"] = w
        scored.append(candidate)

    if not scored:
        return []

    scored_classic = sorted(_candidates_for_tier(scored, "Classic"), key=lambda x: x.get("_score", 0.0), reverse=True)
    scored_signature = sorted(_candidates_for_tier(scored, "Signature"), key=lambda x: x.get("_score", 0.0), reverse=True)
    scored_luxury = sorted(_candidates_for_tier(scored, "Luxury"), key=lambda x: x.get("_score", 0.0), reverse=True)

    # Per-session dedupe window
    filtered_classic = _apply_recent_filter(scored_classic, recent_set)
    filtered_signature = _apply_recent_filter(scored_signature, recent_set)
    filtered_luxury = _apply_recent_filter(scored_luxury, recent_set)
    
    # Deterministic dedupe fallback
    pick_classic = filtered_classic if filtered_classic else scored_classic
    pick_signature = filtered_signature if filtered_signature else scored_signature
    pick_luxury = filtered_luxury if filtered_luxury else scored_luxury
    
    # Deterministic top-K rotation (K=5)
    session_id = (context or {}).get("session_id") or ""
    run_count = int((context or {}).get("run_count") or 0)

    win_classic   = _pick_rotated(pick_classic,   "Classic",   session_id, run_count, p)
    win_signature = _pick_rotated(pick_signature, "Signature", session_id, run_count, p)
    win_luxury    = _pick_rotated(pick_luxury,    "Luxury",    session_id, run_count, p)

    triad: List[Dict[str, Any]] = []
    if win_classic:
        triad.append(win_classic)
    if win_signature:
        triad.append(win_signature)
    if win_luxury:
        triad.append(win_luxury)
    
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
    triad = list({t: max([i for i in triad if i.get("tier")==t], key=lambda x: x.get("_score", 0.0)) for t in TIER_ORDER if any(i.get("tier")==t for i in triad)}.values())  # type: ignore

    # Strong iconic override: try to replace the mono slot with the iconic species mono when present
    if iconic_intent:
        # find best mono of species; prefer Luxury, then Signature, then Classic
        for pref_tier in ["Luxury", "Signature", "Classic"]:
            pool = [x for x in scored if x.get("tier")==pref_tier and _is_mono(x) and _has_species(x, iconic_intent)]
            pool = sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True)
            if pool:
                # replace the current item for that pref_tier
                for i, it in enumerate(triad):
                    if it.get("tier") == pref_tier:
                        triad[i] = pool[0]
                        break
                break

    # Last-mile ritual: exactly 2 MIX + 1 MONO
    triad = _ensure_two_mix_one_mono(triad, scored)

    # Apply notes for substituted species
    _apply_substitution_notes(triad, redirect_from, redirect_alts)

    # if the user intent is very generic, add a single clarifying note on first MIX
    if clarity == 0.0:
        _add_unclear_mix_note(triad)

    # Apply emotional/edge-case stamping and copy limits
    for it in triad:
        it["emotion"] = resolved_anchor
        it["edge_case"] = is_edge
        it["edge_type"] = edge_type if is_edge else None
        it["desc"] = _enforce_copy_limit(it.get("desc", ""), edge_type)

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
            "emotion": resolved_anchor, # ✅ single source of truth
            "tier": it.get("tier"),
            "packaging": it.get("packaging"),
            "mono": bool(it.get("mono")),
            "palette": list(it.get("palette") or []),
            "luxury_grand": bool(it.get("luxury_grand", False)),
            "note": it.get("note"),
            "edge_case": is_edge,
            "edge_type": edge_type if is_edge else None,
        })
    return out

# Back-compat thin wrapper
def curate(prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return selection_engine(prompt, context or {})

# FastAPI endpoints
app = FastAPI()

@app.post("/curate", tags=["public"])
async def curate_post(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    text = (q or prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Invalid input")
    context = {"recent_ids": recent_ids}
    items = selection_engine(text, context)
    return {"items": items, "edge_case": any(i.get("edge_case") for i in items)}

# Alias the existing endpoints to also work with the /api prefix
@app.post("/api/curate", tags=["public"])
async def curate_post_api(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    return await curate_post(q, prompt, recent_ids)

@app.get("/curate", tags=["public"])
async def curate_get(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    text = (q or prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Invalid input")
    context = {"recent_ids": recent_ids}
    items = selection_engine(text, context)
    return {"items": items, "edge_case": any(i.get("edge_case") for i in items)}

@app.get("/api/curate", tags=["public"])
async def curate_get_api(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    return await curate_get(q, prompt, recent_ids)
