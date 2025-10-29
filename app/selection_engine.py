# app/selection_engine.py
# P1.4a/PR-2: This file has been refactored to be a pure logic module.
# All FastAPI routes and web-server code have been removed.
# The main selection_engine function now returns a (items, context, meta) tuple.
# The core selection algorithm has been preserved.
# P1.6: Rotation is now deterministic based on prompt hash, applies recent_ids
# suppression, and uses tier-specific salts for variety.

from __future__ import annotations

import json, os, re, zlib, logging
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
from fastapi import HTTPException # Kept for internal error signaling
from collections import defaultdict

__all__ = ["selection_engine", "normalize", "detect_emotion", "_transform_for_api"]

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
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning(f"Failed to load {path}: {e}")
        return default

# ------------------------------------------------------------
# Data sources (all project-owned JSON files)
# ------------------------------------------------------------

CATALOG = _load_json(_p("catalog.json"), [])
CAT_BY_ID = {item['id']: item for item in CATALOG if 'id' in item}

# ---- P1.5 Packaging rails (internal-only) ----
PACKAGING_BY_TIER = {"Classic": "Box", "Signature": "Vase", "Luxury": "PremiumBox"}
# Build a defensive id→LG map once
LG_INDEX = {row["id"]: bool(row.get("luxury_grand", False))
            for row in CATALOG if isinstance(row, dict) and "id" in row}

def _lg(item: dict) -> bool:
    # Preferred: read from the item; fallback: catalog index
    return bool(item.get("luxury_grand", LG_INDEX.get(item.get("id"), False)))
# ----------------------------------------------

RULES_DIR = _p("rules")
EMOTION_KEYWORDS = _load_json(os.path.join(RULES_DIR, "emotion_keywords.json"), {})
EDGES = (EMOTION_KEYWORDS.get("edges", {}) or {})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, "edge_registers.json"), {})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, "tier_policy.json"), {"luxury_grand": {"blocked_emotions": []}})
SUB_NOTES = _load_json(os.path.join(RULES_DIR, "substitution_notes.json"), {"default": "Requested {from} is seasonal/unavailable; offering {alt} as the nearest alternative."})
ANCHOR_THRESHOLDS = _load_json(os.path.join(RULES_DIR, "anchor_thresholds.json"), {})
SENTIMENT_FAMILIES = _load_json(os.path.join(RULES_DIR, "sentiment_families.json"), {})

# --- Telemetry enums (single source of truth) ---
ALLOWED_FALLBACK_REASONS = frozenset({
    "in_family",
    "general_in_family",
    "duplicate_tier",
    "cross_family_last_resort",
})

def _set_fallback_reason(context: dict, reason: str) -> None:
    if reason not in ALLOWED_FALLBACK_REASONS:
        # Keep it strict so dashboards never get typos
        raise ValueError(f"Invalid fallback_reason: {reason}")
    context["fallback_reason"] = reason


def _rel_terms(key: str, fallback: set[str]) -> set[str]:
    src = (EMOTION_KEYWORDS.get("relationship_signals") or {}).get(key) or []
    # accept both lists and dict-likes; coerce to lowercase strings
    out = { (str(t or "")).strip().lower() for t in src if str(t or "").strip() }
    return out or { s.lower() for s in fallback }

# Relationship context tokens (from emotion_keywords.json with fallback)
ROMANTIC_TOKENS    = _rel_terms("romantic",    {
    "love","romance","romantic","blush","sweetheart","darling","my love","crush",
    "miss you","thinking of you","together","couple","partner","wife","husband",
    "girlfriend","boyfriend","fiancé","anniversary","hearts","date night","soulmate",
    "adoration","forever","valentine","my valentine","sweetheart"
})
PROFESSIONAL_TOKENS = _rel_terms("professional", {
    "colleague","coworker","teammate","work","office","team","project","deadline",
    "years of service","milestone","new role","promotion","manager","client"
})
FAMILIAL_TOKENS     = _rel_terms("familial",    {
    "mother","mom","mum","father","dad","parent","parents","son","daughter","brother",
    "sister","grandma","grandmother","granny","grandpa","grandfather","aunt","uncle",
    "cousin","niece","nephew","in-law","mother-in-law","father-in-law","family","familial"
})
FRIENDSHIP_TOKENS   = _rel_terms("friendship",  {
    "friend","friends","best friend","bestie","bff","buddy","pal","mate","bros","homie","friendship"
})

EDGE_CASE_KEYS = {"sympathy", "apology", "farewell", "valentine"}
FEATURE_MULTI_ANCHOR_LOGGING = os.getenv("FEATURE_MULTI_ANCHOR_LOGGING", "0") == "1" # Off by default

def _validate_catalog_on_startup():
    """Validate catalog data at startup to prevent runtime failures"""
    if not CATALOG:
        log.critical("CATALOG is empty - service cannot start")
        raise RuntimeError("Invalid catalog configuration")
    
    if not isinstance(CATALOG, list):
        log.critical("CATALOG must be a list")
        raise RuntimeError("Invalid catalog format")
    
    for i, item in enumerate(CATALOG):
        if not isinstance(item, dict):
            log.critical(f"Catalog item {i} is not a dictionary")
            raise RuntimeError("Invalid catalog item format")
        
        has_image = bool(item.get("image") or item.get("image_url"))
        has_price = bool(item.get("price") is not None or item.get("price_inr") is not None)
        
        if not item.get("id"):
            log.critical(f"Catalog item {i} missing 'id' field")
            raise RuntimeError("Invalid catalog schema: missing id")
            
        if not has_image:
            log.warning(f"Catalog item {item.get('id')} missing image field")
            
        if not has_price:
            log.warning(f"Catalog item {item.get('id')} missing price field")

    log.info(f"Catalog validation passed: {len(CATALOG)} items loaded")

def _validate_catalog_families_on_startup():
    """PHASE-2: Validate catalog items against sentiment families config."""
    if not SENTIMENT_FAMILIES.get("sentiment_families"):
        log.warning("Sentiment families config is missing or empty.")
        return

    all_families = SENTIMENT_FAMILIES["sentiment_families"]
    emotion_to_family = {}
    for fname, fconfig in all_families.items():
        if "subfamilies" in fconfig:
            for sub_name, sub_config in fconfig["subfamilies"].items():
                for emotion in sub_config.get("emotions", []):
                    emotion_to_family[emotion] = sub_name
        else:
            for emotion in fconfig.get("emotions", []):
                emotion_to_family[emotion] = fname
    
    for item in CATALOG:
        item_id = item.get("id", "Unknown")
        family = item.get("sentiment_family")
        emotion = item.get("emotion")
        if not family:
            log.warning(f"Catalog item {item_id} is missing 'sentiment_family'. Treating as UNSPECIFIED.")
        if emotion and emotion not in emotion_to_family:
            log.warning(f"Catalog item {item_id} has emotion '{emotion}' which does not map to any sentiment family.")

# Validate on module import
_validate_catalog_on_startup()
_validate_catalog_families_on_startup()

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

TIER_ORDER = ["Classic", "Signature", "Luxury"]
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

# ---- P1.6 START: Updated Constants ----
FAMILY_SCARCITY_THRESHOLD = 3  # 6 → 3 for MVP variety

TIER_SALTS = {                 # per-tier variety on same prompt
    "Classic":   0xA11CE,      # Kept original valid hex values
    "Signature": 0xBEEEF,
    "Luxury":    0xCAFE5,
}
# ---- celebration pops (gold NOT included) ----
# (no hard-coded palette blocks; use JSON registers)
# ---- P1.6 END: Updated Constants ----


ICONIC_SPECIES = { "lily": ["lily", "lilies"], "rose": ["rose", "roses"], "orchid": ["orchid", "orchids"] }
ONLY_ICONIC_RE = re.compile(r"\bonly\s+(lil(?:y|ies)|roses?|orchids?)\b", re.I)

VAGUE_TERMS = {"nice", "beautiful", "pretty", "some", "any", "simple", "best", "good", "flowers", "flower"}
GRAND_INTENT = {"grand","bigger","large","lavish","extravagagnt","50+","hundred","massive","most beautiful"}

def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s[:500]

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z]+", normalize(s))

def _contains_any(text: str, toks: Iterable[str]) -> bool:
    t = text.lower()
    return any((w or "").lower() in t for w in (toks or []))

def _intent_clarity(prompt: str, matched_keywords: int) -> float:
    p = normalize(prompt)
    words = re.findall(r"[a-z]+", p)
    if matched_keywords == 0 and (len(words) <= 4 or all(w in VAGUE_TERMS for w in words)):
        return 0.0
    return 1.0

def _has_grand_intent(prompt: str) -> bool:
    p = normalize(prompt)
    return any(w in p for w in GRAND_INTENT)

def _add_unclear_mix_note(triad: list) -> None:
    for it in triad:
        if not it.get("mono") and not it.get("note"):
            it["note"] = "Versatile picks while you decide - tell us the occasion for a more personal curation."
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

# ---- P1.6 START: Updated Helpers ----

def _stable_hash_u32(s: str) -> int:
    """simple, fast, repeatable (murmur/xxh would be fine too, this keeps deps zero)"""
    return zlib.crc32((s or "").encode("utf-8")) & 0xFFFFFFFF

def _mix(seed: int, salt: int) -> int:
    return (seed ^ salt) & 0xFFFFFFFF

def _rotation_index(seed: int, salt: int, n: int) -> int:
    """Get a deterministic rotation index based on an int seed and salt"""
    if n <= 0:
        return 0
    mix = (seed ^ salt) * 2654435761 & 0xFFFFFFFF
    return mix % n

def _suppress_recent(pool: List[Dict[str, Any]], recent_ids: set) -> List[Dict[str, Any]]:
    """Filters list, falling back to original if all items would be suppressed"""
    if not pool or not recent_ids:
        return pool
    filtered = [it for it in pool if it.get("id") not in recent_ids]
    return filtered or pool  # fallback if suppression wipes the pool

# ---- P1.6 END: Updated Helpers ----

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

def _match_edge(text: str, rules: dict) -> bool:
    if not rules: return False
    for phrase in (rules.get("exact") or []):
        if phrase and phrase.lower() in text: return True
    if any(tok and tok.lower() in text for tok in (rules.get("contains_any") or [])): return True
    patterns = [r if isinstance(r, str) else r.get("pattern", "") for r in (rules.get("regex") or [])]
    if patterns and _matches_regex_list(text, patterns): return True
    for spec in (rules.get("proximity_pairs") or []):
        if _has_proximity(text, spec.get("a",""), spec.get("b",""), int(spec.get("window",2))): return True
    return False

def _match_exact_map(text: str, exact: Dict[str, str]) -> Optional[str]:
    """If any exact phrase matches, return mapped anchor."""
    if not isinstance(exact, dict):
        return None
    t = text.lower()
    for phrase, anchor in exact.items():
        if phrase and phrase.lower() in t:
            return anchor
    return None

def _match_combos(text: str, combos: List[Dict[str, Any]]) -> Optional[str]:
    """
    Each combo entry may have:
      - {"all": ["romantic", "anniversary"], "anchor": "Affection/Support"}
      - {"any": ["proposal", "engagement"], "anchor": "..."}
      - {"proximity": {"a":"thank", "b":"you", "window":2}, "anchor":"..."}
    Return first matching anchor.
    """
    if not isinstance(combos, list):
        return None
    t = text.lower()
    for spec in combos:
        if not isinstance(spec, dict):
            continue
        anchor = spec.get("anchor")
        ok = False
        if spec.get("all"):
            ok = all((w or "").lower() in t for w in spec["all"])
        elif spec.get("any"):
            ok = _contains_any(t, spec["any"])
        elif spec.get("proximity"):
            px = spec["proximity"] or {}
            ok = _has_proximity(t, px.get("a",""), px.get("b",""), int(px.get("window",2)))
        if ok and anchor:
            return anchor
    return None

def _apply_disambiguation(text: str, disambig: Dict[str, str], current: Optional[str]) -> Optional[str]:
    """If a phrase is present in disambiguation map, override anchor."""
    if not isinstance(disambig, dict):
        return current
    t = text.lower()
    for phrase, anchor in disambig.items():
        if phrase and phrase.lower() in t:
            return anchor
    return current

def _stable_id(it: dict) -> str:
    return str(it.get("id", "unknown"))

def _get_target_family(resolved_anchor: str, edge_type: Optional[str], relationship_context: Optional[str] = None) -> str:
    if edge_type:
        reg = EDGE_REGISTERS.get(edge_type)
        if reg:
            if edge_type == "apology" and relationship_context:
                subfamily_map = { "romantic": "romantic_repair", "familial": "familial_repair", "professional": "professional_repair", "friendship": "friendship_repair", }
                return subfamily_map.get(relationship_context, "friendship_repair")
            return reg.get("sentiment_family", "UNSPECIFIED")
    all_families = SENTIMENT_FAMILIES.get("sentiment_families", {})
    for fname, fconfig in all_families.items():
        if "subfamilies" in fconfig:
            for sub_name, sub_config in fconfig["subfamilies"].items():
                if resolved_anchor in sub_config.get("emotions", []):
                    return sub_name
        elif resolved_anchor in fconfig.get("emotions", []):
            return fname
    return "UNSPECIFIED"

def _get_family_config(target_family: str) -> dict:
    all_families = SENTIMENT_FAMILIES.get("sentiment_families", {})
    for fname, fconfig in all_families.items():
        if fname == target_family:
            return fconfig
        if "subfamilies" in fconfig:
            if target_family in fconfig["subfamilies"]:
                return fconfig["subfamilies"][target_family]
    return {}

def _filter_by_family(pool: list[dict], target_family: str, context: dict) -> list[dict]:
    if not target_family or target_family == "UNSPECIFIED":
        return pool
    family_config = _get_family_config(target_family)
    barriers = set(family_config.get("contamination_barriers", []))
    context["barriers_triggered"] = list(barriers)
    def is_allowed(item: dict) -> bool:
        item_family = item.get("sentiment_family") or "UNSPECIFIED"
        if item_family in barriers: return False
        return item_family == target_family or item_family == "UNSPECIFIED"
    return [it for it in pool if is_allowed(it)]

def _ensure_one_mono_in_triad(triad: list[dict], context: dict) -> list[dict]:
    if not triad or len(triad) != 3:
        return triad
    mono_items = [it for it in triad if it.get("mono")]
    if len(mono_items) == 1:
        return triad
    if len(mono_items) > 1:
        first_mono_found = False
        for item in triad:
            if item.get("mono"):
                if not first_mono_found:
                    first_mono_found = True
                else:
                    item["mono"] = False
        return triad
    context["fallback_reason"] = "converted_to_mono"
    mono_candidates = [it for it in triad if CAT_BY_ID.get(it["id"], {}).get("mono")]
    if mono_candidates:
        mono_candidates.sort(key=lambda x: TIER_RANK.get(x.get("tier"), 0), reverse=True)
        mono_id_to_set = mono_candidates[0]["id"]
        for item in triad:
            item["mono"] = (item["id"] == mono_id_to_set)
    else:
        classic_item_found = False
        for item in triad:
            if item.get("tier") == "Classic":
                item["mono"] = True
                classic_item_found = True
                break
        if not classic_item_found:
            triad[0]["mono"] = True
    return triad

# (removed) old _apply_edge_guards implementation – JSON-driven registers are authoritative


def _apply_edge_register_filters(pool: List[Dict[str, Any]], edge_type: str) -> List[Dict[str, Any]]:
    """
    JSON-driven filters/biases restored. Reads EDGE_REGISTERS[edge_type] and
    applies:
      - palette_forbid (with *gold-neutral* for sympathy/farewell)
      - avoid_species / prefer_species
      - mono_must_include (if item is mono)
      - lg_policy: "block" removes LG items
    Never over-prunes: if a step empties the pool, it falls back to previous.
    """
    regs = EDGE_REGISTERS.get(edge_type) or {}
    if not regs or not pool:
        return pool

    # keep a working copy
    working = list(pool)
    previous = list(working)

    # 1) palette forbid (with gold-neutral)
    forbid: Set[str] = {str(p).lower() for p in (regs.get("palette_forbid") or []) if p}
    if edge_type in ("sympathy", "farewell") and "gold" in forbid:
        forbid.discard("gold")
    if forbid:
        pruned = []
        for it in working:
            pal = {(p or "").strip().lower() for p in (it.get("palette") or []) if isinstance(p, str)}
            if pal.isdisjoint(forbid):
                pruned.append(it)
        if pruned:
            previous, working = working, pruned

    # 2) avoid species
    avoid: Set[str] = {str(s).lower() for s in (regs.get("avoid_species") or []) if s}
    if avoid:
        pruned = [it for it in working if avoid.isdisjoint({f.lower() for f in (it.get("flowers") or [])})]
        if pruned:
            previous, working = working, pruned

    # 3) mono must include
    must: Set[str] = {str(s).lower() for s in (regs.get("mono_must_include") or []) if s}
    if must:
        pruned = []
        for it in working:
            if it.get("mono"):
                fl = {f.lower() for f in (it.get("flowers") or []) if isinstance(f, str)}
                if fl & must:
                    pruned.append(it)
            else:
                pruned.append(it)
        if pruned:
            previous, working = working, pruned

    # 4) lg policy
    if regs.get("lg_policy") == "block":
        pruned = [it for it in working if not bool(it.get("luxury_grand"))]
        if pruned:
            previous, working = working, pruned

    return working or previous


def _ensure_triad_or_500(items: list[dict]) -> None:
    triad_ok = isinstance(items, list) and len(items) == 3
    mono_count = sum(1 for it in items if it.get("mono") is True)
    mono_ok = (mono_count == 1)
    if triad_ok and mono_ok:
        return
    log.error("[TRIAD_CONTRACT] invalid triad: len=%s mono_count=%s ids=%s",
        (len(items) if isinstance(items, list) else None), mono_count,
        [_stable_id(it) for it in (items or [])])
    raise HTTPException(status_code=500, detail="Internal catalog data error")

def _compute_detected_emotions(scores: dict[str,float]) -> list[dict]:
    cfg = (ANCHOR_THRESHOLDS.get("multi_anchor_logging") or {})
    if not cfg.get("enabled", True): return []
    max_entries = int(cfg.get("max_entries", 2))
    score2_min = float(cfg.get("score2_min", 0.25))
    delta_max  = float(cfg.get("delta_max", 0.15))
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
    Resolution order:
      1) EDGE short-circuit (from emotion_keywords.json -> edges)
      2) EXACT phrase map
      3) COMBOS (all/any/proximity)
      4) DISAMBIGUATION override
      5) KEYWORDS scoring (fallback)
      6) Grand-intent & LG policy mask
    """
    p = normalize(prompt)
    scores: Dict[str, float] = {}
    edge_type: Optional[str] = None

    ek = EMOTION_KEYWORDS or {}

    # (1) edges
    for edge, rules in (ek.get("edges") or {}).items():
        if _match_edge(p, rules):
            edge_type = edge
            resolved_anchor = (EDGE_REGISTERS.get(edge) or {}).get("emotion_anchor") or "general"
            return resolved_anchor, edge_type, scores

    # (2) exact
    exact_anchor = _match_exact_map(p, ek.get("exact") or {})
    if exact_anchor:
        return exact_anchor, None, scores

    # (3) combos
    combo_anchor = _match_combos(p, ek.get("combos") or [])
    if combo_anchor:
        return combo_anchor, None, scores

    # (4) disambiguation (override later decision)
    disambig = ek.get("disambiguation") or {}

    # (5) keywords fallback
    for anchor, words in (ek.get("keywords") or {}).items():
        hits = sum(1 for w in (words or []) if (w or "").lower() in p)
        if hits:
            scores[anchor] = float(hits)

    # (6) grand-intent & LG policy
    has_grand = _has_grand_intent(prompt)
    if has_grand:
        scores["luxury_grand"] = (scores.get("luxury_grand", 0.0) + 20.0) * 1.1
        blocked = (TIER_POLICY.get("luxury_grand", {}) or {}).get("blocked_emotions") or []
        if blocked:
            scores = {k: v for k, v in scores.items() if k not in blocked}

    if not scores:
        # last resort
        resolved = _apply_disambiguation(p, disambig, "general")
        return resolved or "general", None, {}

    top = max(scores.items(), key=lambda kv: kv[1])[0]
    top = _apply_disambiguation(p, disambig, top) or top
    return top, edge_type, scores

def detect_relationship_context(prompt_norm: str) -> tuple[str, bool]:
    p = prompt_norm
    hits = {
        "romantic": any(tok in p for tok in ROMANTIC_TOKENS),
        "familial": any(tok in p for tok in FAMILIAL_TOKENS),
        "friendship": any(tok in p for tok in FRIENDSHIP_TOKENS),
        "professional": any(tok in p for tok in PROFESSIONAL_TOKENS),
    }
    positives = [k for k,v in hits.items() if v]
    if not positives:
        return "unknown", True
    # priority: romantic > professional > familial > friendship
    for lane in ("romantic","professional","familial","friendship"):
        if lane in positives:
            return lane, (len(positives) > 1)
    return "unknown", True

def find_iconic_species(prompt_norm: str) -> Optional[str]:
    if ONLY_ICONIC_RE.search(prompt_norm):
        for species, terms in ICONIC_SPECIES.items():
            if any(term in prompt_norm for term in terms):
                return species
    return None

def find_and_assign_note(triad: list, selected_species: Optional[str], selected_emotion: Optional[str], prompt_text: str = "") -> None:
    found_species = any(selected_species in it.get("flowers", []) for it in triad) if selected_species else False
    if selected_species and not found_species:
        note = SUB_NOTES.get("species_not_found", "We couldn't find a {species} bouquet...").replace("{species}", selected_species)
        target = next((it for it in triad if not it.get("mono")), triad[0] if triad else None)
        if target is not None: target["note"] = note

def _transform_for_api(items: List[Dict], resolved_anchor: Optional[str]) -> List[Dict]:
    _ensure_triad_or_500(items)
    out: List[Dict] = []
    for it in items:
        item = dict(it)
        item["image"] = item.get("image") or item.get("image_url") or "/static/default-flower.jpg"
        item["price"] = item.get("price") if item.get("price") is not None else item.get("price_inr", 1000)
        item["currency"] = "INR"
        item["emotion"] = item.get("emotion") or resolved_anchor or "general"
        if "image_url" in item: del item["image_url"]
        if "price_inr" in item: del item["price_inr"]
        out.append(item)
    return out

def _backfill_to_three(items: list[dict], catalog: list[dict], context: dict) -> list[dict]:
    if len(items) >= 3:
        return items[:3]
    final_triad = [None, None, None]
    existing_ids = set()
    for item in items:
        tier_idx = TIER_RANK.get(item.get("tier"))
        if tier_idx is not None and final_triad[tier_idx] is None:
            final_triad[tier_idx] = item
            existing_ids.add(_stable_id(item))
            
    target_family = context.get("target_family")
    family_pool = [it for it in catalog if (it.get("sentiment_family") == target_family or it.get("sentiment_family") == "UNSPECIFIED") and _stable_id(it) not in existing_ids]

    if context.get("edge_type"):
        # ---- P1.6 MODIFICATION ----
        family_pool = _apply_edge_register_filters(family_pool, context["edge_type"]) # NEW
        # ---- P1.6 END ----
        
    missing_tiers = [TIER_ORDER[i] for i, item in enumerate(final_triad) if item is None]

    for tier in missing_tiers:
        tier_pool = [it for it in family_pool if it.get("tier") == tier]
        if tier_pool:
            selected = tier_pool[0]
            final_triad[TIER_RANK[tier]] = selected
            existing_ids.add(_stable_id(selected))
            family_pool = [it for it in family_pool if _stable_id(it) != _stable_id(selected)]
            
    if None in final_triad and context.get("sentiment_over_ladder"):
        filled_items = [it for it in final_triad if it is not None]
        if filled_items:
            filler = filled_items[-1]
            for i in range(3):
                if final_triad[i] is None:
                    final_triad[i] = dict(filler)
                    final_triad[i]['tier'] = TIER_ORDER[i]
                    
    if None in final_triad:
        _set_fallback_reason(context, "cross_family_last_resort")
        any_pool = [it for it in catalog if _stable_id(it) not in existing_ids]
        for i in range(3):
            if final_triad[i] is None:
                tier_to_fill = TIER_ORDER[i]
                candidate = next((it for it in any_pool if it.get("tier") == tier_to_fill), None)
                if candidate:
                    final_triad[i] = candidate
                    existing_ids.add(_stable_id(candidate))
                    
    final_triad = [it for it in final_triad if it is not None]
    while len(final_triad) < 3:
        final_triad.append(final_triad[-1] if final_triad else catalog[0])
        
    return final_triad[:3]

def _family_for_anchor(anchor: str, families_json: dict) -> str:
    families = families_json.get("sentiment_families", {})
    for fam, spec in families.items():
        emos = set(spec.get("emotions") or [])
        if anchor in emos:
            return fam
        if isinstance(spec, dict) and "subfamilies" in spec:
            for subname, sub in spec["subfamilies"].items():
                if isinstance(sub, dict) and anchor in set(sub.get("emotions") or []):
                    return subname
    return "UNSPECIFIED"

def _filter_in_family(catalog: list[dict], fam: str, families_json: dict) -> list[dict]:
    families = families_json.get("sentiment_families", {})
    fam_emos = set()
    spec = families.get(fam)
    if not spec:
        for main_fam in families.values():
            if isinstance(main_fam, dict) and "subfamilies" in main_fam and fam in main_fam["subfamilies"]:
                spec = main_fam["subfamilies"][fam]
                break
    
    if isinstance(spec, dict):
        fam_emos.update(spec.get("emotions") or [])

    return [x for x in catalog if (x.get("emotion") in fam_emos)]

def _pool_general_in_family(catalog: list[dict], fam: str) -> list[dict]:
    return [x for x in catalog if (x.get("emotion") == "general" and x.get("sentiment_family") == fam)]

def _fallback_boundary_path(available_catalog: list[dict], target_family: str, context: dict,
                            families_json: dict, need_count: int) -> list[dict]:
    if not context.get("fallback_reason"):
        context["fallback_reason"] = "none" # Default before logic runs

    # Build a barrier-aware base pool once
    base_pool = _filter_by_family(available_catalog, target_family, context)
    
    # 1) in-family (emotion-specific)
    p1 = [x for x in base_pool if x.get("emotion") in set((_get_family_config(target_family) or {}).get("emotions") or [])]
    if len(p1) >= need_count:
        _set_fallback_reason(context, "in_family")
        return p1

    # 2) general-in-family
    p2_candidates = [x for x in base_pool if x.get("emotion") == "general"]
    p2 = p1 + [item for item in p2_candidates if item["id"] not in {x["id"] for x in p1}]
    if len(p2) >= need_count:
        _set_fallback_reason(context, "general_in_family")
        return p2

    # 3) duplicate-tier
    if context.get("sentiment_over_ladder"):
        _set_fallback_reason(context, "duplicate_tier")
        return p2

    # 4) last resort cross-family
    _set_fallback_reason(context, "cross_family_last_resort")
    return available_catalog

def selection_engine(prompt: str, context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not CATALOG:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    # --- Robustness: initialize defaults ---
    context.setdefault("fallback_reason", "none")
    context.setdefault("pool_size", {"pre_suppress": {}, "post_suppress": {}})
    
    prompt_norm = normalize(prompt)
    
    # --- P1.6 Rotation Seed (from prompt) ---
    norm_prompt_hash = _stable_hash_u32(prompt_norm)
    # Use provided hash if present (for CI/testing), else use prompt's hash
    seed = context.get("prompt_hash") 
    if not isinstance(seed, int): # Check if it's missing or not an int
        seed = norm_prompt_hash
        context["prompt_hash"] = seed # Store the computed hash for logging
    # --- End P1.6 ---
    
    resolved_anchor, edge_type, _scores = detect_emotion(prompt, context or {})
    selected_species = find_iconic_species(prompt_norm)
    
    context["resolved_anchor"] = resolved_anchor
    context["edge_type"] = edge_type
    
    ctx, amb = detect_relationship_context(prompt_norm)
    context["relationship_context"] = ctx
    context["relationship_ambiguous"] = amb

    if context.get("edge_type") == "apology":
        rc = context.get("relationship_context")
        if rc == "romantic":
            context["sentiment_family"] = "romantic_repair"
        elif rc in {"familial","friendship","professional"}:
            context["sentiment_family"] = f"{rc}_repair"
        else:
            context["sentiment_family"] = "friendship_repair"
            
        # Honor anchor override for relationship-specific apology subfamilies (rails: romantic → Affection/Support)
        try:
            overrides = EDGE_REGISTERS.get("apology", {}).get("relationship_overrides", {})
            key = f"{rc}_repair" if rc else None
            if key and key in overrides:
                anchor_override = overrides.get(key, {}).get("emotion_anchor")
                if anchor_override:
                    resolved_anchor = anchor_override
                    context["resolved_anchor"] = anchor_override
        except Exception:
            # Never fail selection due to misconfigured overrides
            pass

    target_family = context.get("sentiment_family") or _family_for_anchor(resolved_anchor, SENTIMENT_FAMILIES)
    context["target_family"] = target_family
    sentiment_over_ladder = bool(EDGE_REGISTERS.get(edge_type, {}).get("sentiment_over_ladder", False))
    context["sentiment_over_ladder"] = sentiment_over_ladder
    
    available_catalog = list(CATALOG)
    emotion_pool = _fallback_boundary_path(available_catalog, target_family, context, SENTIMENT_FAMILIES, need_count=3)

    # B3: Pool-size telemetry (pre-suppression)
    pool_sizes = {
        "pre_suppress": {
            "classic": sum(1 for x in emotion_pool if x.get("tier")=="Classic"),
            "signature": sum(1 for x in emotion_pool if x.get("tier")=="Signature"),
            "luxury": sum(1 for x in emotion_pool if x.get("tier")=="Luxury"),
        }
    }
    # expose for tests/telemetry
    context["pool_size"] = pool_sizes        # legacy (kept for tests)
    context["pool_sizes"] = pool_sizes       # new (used by logging/artifacts)

    context["duplication_used"] = False
    context["barriers_triggered"] = []
    
    recent_set = set(context.get("recent_ids") or []) # Get recent IDs once
    triad = []
    seen_ids = set()

    for tier in TIER_ORDER:
        tier_pool = [it for it in emotion_pool if it.get("tier") == tier and _stable_id(it) not in seen_ids]
        
        # ---- P1.6 MODIFICATIONS START ----
        
        # 1. Apply new edge guards first
        if edge_type:
            # tier_pool = _apply_edge_register_filters(tier_pool, edge_type) # OLD
            tier_pool = _apply_edge_register_filters(tier_pool, edge_type) # NEW
        
        # 2. Apply suppression using the new helper function
        tier_pool = _suppress_recent(tier_pool, recent_set)
        
        # ---- P1.6 MODIFICATIONS END ----

        selected_item = None
        if tier_pool:
            # Use deterministic, tier-specific rotation (using new P1.6 helpers)
            salt = TIER_SALTS.get(tier, 0xABCDE) # Get salt for the tier
            idx = _rotation_index(seed, salt, len(tier_pool)) # Use new rotation function
            selected_item = dict(tier_pool[idx])
        else:
            if context.get("fallback_reason") == "duplicate_tier" and triad:
                context["duplication_used"] = True
                if tier == "Signature" and triad[0]: selected_item = dict(triad[0])
                elif tier == "Luxury" and len(triad) > 1 and triad[1]: selected_item = dict(triad[1])
                elif triad[0]: selected_item = dict(triad[0])
        
        if selected_item:
            selected_item["tier"] = tier
            triad.append(selected_item)
            seen_ids.add(_stable_id(selected_item))
        else:
            triad.append(None)
            
    if any(item is None for item in triad):
        triad_no_none = [item for item in triad if item is not None]
        final_triad = _backfill_to_three(triad_no_none, available_catalog, context)
    else:
        final_triad = triad[:3]

    # compute a post-suppress view of the pool (exclude recent_ids)
    # Note: recent_set was already defined before the loop
    post_pool = [x for x in emotion_pool if _stable_id(x) not in recent_set]

    pool_sizes["post_suppress"] = {
        "classic":   sum(1 for x in post_pool if x.get("tier") == "Classic"),
        "signature": sum(1 for x in post_pool if x.get("tier") == "Signature"),
        "luxury":    sum(1 for x in post_pool if x.get("tier") == "Luxury"),
    }
    # keep legacy + new aliases in sync after post-suppress counts
    context["pool_size"] = pool_sizes
    context["pool_sizes"] = pool_sizes
            
    final_triad = _ensure_one_mono_in_triad(final_triad, context)
    _ensure_triad_or_500(final_triad)
    
    if edge_type:
        for it in final_triad:
            it["edge_case"] = True
            it["edge_type"] = edge_type
            
    find_and_assign_note(final_triad, selected_species, resolved_anchor, prompt)
    if not _intent_clarity(prompt, len(_scores)):
        _add_unclear_mix_note(final_triad) # FIX: final_ad was a typo, corrected to final_triad
    
    meta_detected = []
    if FEATURE_MULTI_ANCHOR_LOGGING:
        meta_detected = _compute_detected_emotions(_scores or {})
    
    # compute extra observability fields
    _pre_total  = sum(pool_sizes["pre_suppress"].values())
    _post_total = sum(pool_sizes["post_suppress"].values())
    suppressed_recent_count = max(0, _pre_total - _post_total)
    final_ids = [_stable_id(it) for it in final_triad]

    log_record = {
        "request_id": context.get("request_id"),
        "resolved_anchor": context.get("resolved_anchor"),
        "edge_case": bool(context.get("edge_type")),
        "edge_type": context.get("edge_type"),
        "fallback_reason": context.get("fallback_reason", "none"),
        "duplication_used": context.get("duplication_used", False),
        "sentiment_override_used": context.get("sentiment_over_ladder", False),
        "relationship_context": context.get("relationship_context"),
        "relationship_ambiguous": bool(context.get("relationship_ambiguous")),
        "pool_size": pool_sizes,
        "prompt_hash": seed,                    # deterministic seed used
        "suppressed_recent_count": suppressed_recent_count,
        "final_ids": final_ids
    }
    log.info("SELECTION_EVIDENCE: %s", json.dumps(log_record, ensure_ascii=False))

    # Attach internal fields for P1.5 (do NOT expose publicly)
    for it in final_triad:
        it["_packaging"] = PACKAGING_BY_TIER.get(it.get("tier"))
        it["_luxury_grand"] = _lg(it)
        
    for c in final_triad:
        c["emotion"] = c.get("emotion") or context.get("resolved_anchor") or "general"
        
    meta = {
        "detected_emotions": meta_detected,
        "prompt_hash": context.get("prompt_hash"),
        "pool_sizes": pool_sizes,
        "resolved_anchor": context.get("resolved_anchor"),
        "edge_type": context.get("edge_type"),
        "fallback_reason": context.get("fallback_reason"),
    }

    return final_triad, context, meta
