
# app/selection_engine.py
# Ultra-beginner-safe Selection Engine — Phase 1.2 + 1.3 (Edge-Case Playbooks)
# Deterministic, rules-first; strict tier scaffold (Classic → Signature → Luxury).
# Always returns exactly 3 items (2 MIX + 1 MONO) with palette[].
# Edge registers (sympathy/apology/farewell/valentine) apply tone/palette/species rules and copy ≤ N words.
# LG (Luxury-Grand) policy enforced via rules/tier_policy.json, with soft multipliers in registers.

from __future__ import annotations
import json, os, re, hashlib
from typing import Any, Dict, List, Tuple, Optional

# -----------------------------
# File helpers
# -----------------------------

ROOT = os.path.dirname(__file__)
def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Graceful fallback if the file is missing or temporarily malformed
        return default

# -----------------------------
# Load rule tables
# -----------------------------

EMOTION_KEYWORDS: Dict[str, List[str]] = _load_json(_p('rules','emotion_keywords.json'), {})
SUBSTITUTIONS: Dict[str, List[str]] = _load_json(_p('rules','substitution_map.json'), {})
WEIGHTS_DEFAULT: Dict[str, Dict[str, int]] = _load_json(_p('rules','weights_default.json'), {})
PRICING_POLICY: Dict[str, Any] = _load_json(_p('rules','pricing_policy.json'), {
    "floors_inr": {"Classic": 1499, "Signature": 2499, "Luxury": 3999, "LG": 4999}
})
TIER_POLICY: Dict[str, Any] = _load_json(_p('rules','tier_policy.json'), {
    "luxury_grand": {
        "allowed_emotions": [],
        "blocked_emotions": ["Sympathy","GetWell","Encouragement"],
        "max_per_triad": 1,
        "allow_in_mix": True,
        "allow_in_mono": True
    }
})
EDGE_KEYWORDS: Dict[str, List[str]] = _load_json(_p('rules','edge_keywords.json'), {})
EDGE_REGISTERS: Dict[str, Any] = _load_json(_p('rules','edge_registers.json'), {})
CATALOG: List[Dict[str, Any]] = _load_json(_p('catalog.json'), [])

ICONIC = {"rose","lily","orchid"}
TIER_RANK = {"Classic": 0, "Signature": 1, "Luxury": 2}

# -----------------------------
# Utilities
# -----------------------------

def normalize(text: str) -> str:
    text = text or ''
    text = text.lower()
    text = re.sub(r'(https?://\S+|\S+@\S+)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _hash_seed(prompt: str) -> int:
    return int(hashlib.sha1(prompt.encode('utf-8')).hexdigest(), 16) % (10**8)

def _score(item: Dict[str, Any], seed: int, base_weight: int) -> float:
    # Deterministic jitter based on id + seed — keeps stable order for same prompt
    salt = str(item.get('id','')) + str(seed)
    h = int(hashlib.md5(salt.encode('utf-8')).hexdigest(), 16) % 1000
    return base_weight * 1000 + h

def _truncate_words(s: str, max_words: int) -> str:
    words = (s or '').split()
    if len(words) <= max_words:
        return s or ''
    return ' '.join(words[:max_words]).rstrip(' ,.;:!?') + '…'

def _order_by_tier(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    def key(it):
        r = TIER_RANK.get(it.get('tier'), 99)
        lg = 1 if it.get('luxury_grand') else 0
        return (r, lg)
    return sorted(items, key=key)

def _meets_floor(item: Dict[str,Any]) -> bool:
    floors = PRICING_POLICY.get('floors_inr', {})
    tier = item.get('tier', 'Classic')
    price = int(item.get('price', item.get('price_inr', 0)) or 0)
    if tier == 'Luxury' and item.get('luxury_grand'):
        return price >= int(floors.get('LG', 4999))
    return price >= int(floors.get(tier, 0))

def _lg_allowed(emotion: str) -> bool:
    pol = TIER_POLICY.get('luxury_grand', {})
    blk = set(pol.get('blocked_emotions', []))
    if emotion in blk:
        return False
    allow = pol.get('allowed_emotions', [])
    # If allow list empty → allow unless blocked
    return True if not allow else (emotion in set(allow))

def _is_iconic_flowers(flowers: List[str]) -> bool:
    if not flowers: return False
    return any(f.lower() in ICONIC for f in flowers)

def _has_grand_intent(prompt: str) -> bool:
    keys = set(EDGE_KEYWORDS.get('grand_intent_keywords', []))
    cues = set(EDGE_KEYWORDS.get('relationship_grandeur_cues', []))
    p = normalize(prompt)
    return any(k in p for k in keys) or any(c in p for c in cues)

# -----------------------------
# Detection (edge-case then emotion)
# -----------------------------

def detect_edge_case(prompt: str) -> Optional[str]:
    p = normalize(prompt)
    for kind, keys in EDGE_KEYWORDS.items():
        if kind in ('grand_intent_keywords','relationship_grandeur_cues'):
            continue
        for k in keys:
            if k in p:
                return kind  # sympathy|apology|farewell|valentine
    return None

def detect_emotion(prompt: str, context: Optional[Dict[str,Any]]=None) -> str:
    p = normalize(prompt)
    # Exact keyword match table first
    for emo, keys in EMOTION_KEYWORDS.items():
        for k in keys:
            if k in p:
                return emo
    # Context hint next
    emo_hint = (context or {}).get('emotion_hint')
    if emo_hint:
        return emo_hint
    # Default
    return 'Romance'  # Affection/Support default

# -----------------------------
# Pools & weighting
# -----------------------------

def _base_candidates(emotion: str) -> List[Dict[str,Any]]:
    # Pick catalog items with matching emotion (string compare, fallback neutral "Any")
    matches = [x for x in CATALOG if x.get('emotion') == emotion]
    if not matches:
        matches = [x for x in CATALOG if x.get('emotion') in ('Any','General','Romance')]
    return matches

def _split_mix_mono(items: List[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    mix = [x for x in items if not bool(x.get('mono'))]
    # Iconic-only MONO
    mono = [x for x in items if bool(x.get('mono')) and _is_iconic_flowers([*(x.get('flowers') or [])])]
    return mix, mono

def _apply_palette_pref(score: int, palette: List[str], targets: List[str]) -> int:
    if not targets: return score
    pal = set([*(palette or [])])
    hit = len(pal.intersection(set(targets)))
    return score + 20*hit  # gentle boost

def _apply_species_prefs(score: int, flowers: List[str], prefer: List[str], avoid: List[str]) -> int:
    fset = set([*(flowers or [])])
    if prefer and any(p in (s.lower() for s in fset) for p in (w.lower() for w in prefer)):
        score += 50
    if avoid and any(a in (s.lower() for s in fset) for a in (w.lower() for w in avoid)):
        score -= 100
    return score

def _apply_lg_policy(item: Dict[str,Any], emotion: str, register: Dict[str,Any], budget: Optional[int], prompt: str) -> int:
    # Start with weight field (1..100), fallback per-flower weights_default, else 10
    w = int(item.get('weight') or 0)
    if not w:
        # fallback by first flower/emotion if present
        fls = item.get('flowers') or []
        if fls:
            w = int(WEIGHTS_DEFAULT.get(fls[0].lower(), {}).get(emotion, 10))
        else:
            w = 10

    # LG adjustments
    if item.get('tier') == 'Luxury' and item.get('luxury_grand'):
        if not _lg_allowed(emotion):
            return 0  # disallowed
        lg_mult = float(register.get('lg_weight_multiplier', 1.0))
        min_b = int(register.get('min_budget_for_lg', 0) or 0)
        has_grand = _has_grand_intent(prompt)
        # Enable if budget passes or grand-intent, otherwise multiplier still applies
        if budget and budget >= max(min_b, int(PRICING_POLICY.get('floors_inr',{}).get('LG', 4999))):
            pass  # ok
        elif has_grand:
            pass
        else:
            # keep multiplier (e.g., 0.8) or if register says block, zero it
            if register.get('lg_policy') == 'block':
                return 0
        w = int(w * lg_mult)
    return max(w, 0)

def _score_pool(pool: List[Dict[str,Any]], seed: int, emotion: str, register: Dict[str,Any], budget: Optional[int], prompt: str) -> List[Tuple[float, Dict[str,Any]]]:
    sc = []
    for it in pool:
        w = _apply_lg_policy(it, emotion, register, budget, prompt)
        if w <= 0:
            continue
        w = _apply_palette_pref(w, it.get('palette') or [], register.get('palette_targets', []))
        w = _apply_species_prefs(w, it.get('flowers') or [], register.get('species_prefer', []), register.get('species_avoid', []))
        sc.append((_score(it, seed, w), it))
    sc.sort(key=lambda x: x[0], reverse=True)
    return sc

# -----------------------------
# Strong intent & redirection helpers
# -----------------------------

def _strong_intent_iconic(prompt: str) -> Optional[str]:
    p = normalize(prompt)
    for s in ICONIC:
        if re.search(rf'\bonly\s+{s}', p):
            return s
    return None

def _requested_unavailable(prompt: str) -> Optional[str]:
    p = normalize(prompt)
    # Find any substitution key that appears in prompt
    for flower in SUBSTITUTIONS.keys():
        if flower in p:
            return flower
    return None

# -----------------------------
# Assembly
# -----------------------------

def selection_engine(prompt: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
        # Normalize prompt
        p_norm = normalize(prompt)
        ctx = context or {}
        budget = ctx.get('budget_inr')

        # Edge-case routing (sympathy/apology/farewell/valentine)
        edge = detect_edge_case(p_norm)
        emotion = detect_emotion(p_norm, ctx)
        register = EDGE_REGISTERS.get(edge or '', {})
        copy_cap = int(EDGE_REGISTERS.get('copy_max_words', 20 if edge else 9999))

        # Base candidates and split
        base = _base_candidates(emotion)
        mix_all, mono_all = _split_mix_mono(base)

        # Strong-intent iconic mono?
        iconic_req = _strong_intent_iconic(p_norm)
        if iconic_req:
            # force MONO of requested iconic species if available
            mono_all = [m for m in mono_all if any(iconic_req == (f or '').lower() for f in (m.get('flowers') or []))] or mono_all

        # Redirection note for unavailable request
        redir_note = None
        req_unavail = _requested_unavailable(p_norm)
        subs_used: List[str] = []
        if req_unavail:
            alts = SUBSTITUTIONS.get(req_unavail, [])
            subs_used = alts
            redir_note = f"Nearest alternative to requested {req_unavail}."
            # Softly bias MIX items that contain the substitution targets
            def boost_alt(it: Dict[str,Any]) -> int:
                fls = [*(it.get('flowers') or [])]
                return 30 if any((a.lower() in (f.lower() for f in fls)) for a in alts) else 0
        else:
            def boost_alt(it: Dict[str,Any]) -> int: return 0  # no-op

        # Score pools with register rules & deterministic seed
        seed = _hash_seed(p_norm)
        mix_scored = _score_pool(mix_all, seed, emotion, register, budget, p_norm)
        # apply any alt boost
        if req_unavail:
            mix_scored = [ (s + boost_alt(it), it) for (s,it) in mix_scored ]
            mix_scored.sort(key=lambda x: x[0], reverse=True)
        mono_scored = _score_pool(mono_all, seed, emotion, register, budget, p_norm)

        # Build strict one-per-tier triad: Classic + Signature + Luxury
        def first_by_tier(scored: List[Tuple[float,Dict[str,Any]]], tier: str, allow_mono: bool) -> Optional[Dict[str,Any]]:
            for _, it in scored:
                if it.get('tier') != tier: 
                    continue
                if allow_mono is False and it.get('mono'):
                    continue
                if it.get('tier') == 'Luxury' and it.get('luxury_grand') and not _lg_allowed(emotion):
                    continue
                if not _meets_floor(it):
                    continue
                return it
            return None

        # Choose which tier holds MONO (greedy: pick the highest-scoring mono across tiers)
        mono_candidates = [it for _, it in mono_scored if it.get('tier') in ('Classic','Signature','Luxury') and _meets_floor(it)]
        mono_pick: Optional[Dict[str,Any]] = mono_candidates[0] if mono_candidates else None

        # Fill tiers
        out: List[Dict[str,Any]] = []
        used_ids = set()

        def pick_for_tier(tier: str, prefer_mono: bool=False) -> Optional[Dict[str,Any]]:
            nonlocal mono_pick
            # Mono first (if designated tier matches and not used)
            if prefer_mono and mono_pick and mono_pick.get('tier') == tier and mono_pick.get('id') not in used_ids:
                it = mono_pick
                mono_pick = None  # consume
                return it
            # Else pick mix
            it = first_by_tier(mix_scored, tier, allow_mono=False)
            if it and it.get('id') not in used_ids:
                return it
            # Fallback: if still empty and mono in that tier exists (but mono will count as mono slot)
            if not prefer_mono and mono_pick and mono_pick.get('tier') == tier and mono_pick.get('id') not in used_ids:
                it = mono_pick
                mono_pick = None
                return it
            return None

        # Decide where mono sits: use its own tier if any; otherwise keep Signature
        mono_tier = mono_pick.get('tier') if mono_pick else 'Signature'
        for tier in ['Classic','Signature','Luxury']:
            prefer_mono = (tier == mono_tier)
            it = pick_for_tier(tier, prefer_mono=prefer_mono)
            if not it:
    # Palette-aware tier fallback: prefer items whose palette intersects the register palette targets
    targets = set(register.get('palette_targets', [])) if register else set()
    def good(x):
        return (
            x.get('tier') == tier
            and x.get('id') not in used_ids
            and _meets_floor(x)
        )
    # 1) prefer palette-aligned
    any_tier_item = next(
        (x for x in CATALOG if good(x) and targets and targets.intersection(set(x.get('palette') or []))),
        None
    )
    # 2) else take anything valid in the tier
    if not any_tier_item:
        any_tier_item = next((x for x in CATALOG if good(x)), None)
    if any_tier_item:
        it = any_tier_item
            if it:
                used_ids.add(it.get('id'))
                out.append(it)

        # If after all we didn't get 3, backfill from top mix/mono regardless of tier uniqueness (last resort)
        while len(out) < 3:
            for scored in (mix_scored, mono_scored):
                for _, it in scored:
                    if it.get('id') in used_ids: 
                        continue
                    if not _meets_floor(it): 
                        continue
                    used_ids.add(it.get('id'))
                    out.append(it)
                    break
                if len(out) >= 3:
                    break

        # Map output fields + apply edge copy truncation and notes
        triad = []
        for it in out[:3]:
            rec = {
                "id": it.get("id"),
                "title": it.get("title"),
                "desc": _truncate_words(it.get("desc",""), copy_cap) if edge else (it.get("desc") or ""),
                "image": it.get("image_url") or it.get("image"),
                "price": int(it.get("price", it.get("price_inr", 0)) or 0),
                "currency": "INR",
                "emotion": it.get("emotion"),
                "tier": it.get("tier"),
                "packaging": it.get("packaging"),
                "mono": bool(it.get("mono")),
                "palette": it.get("palette") or [],
                "luxury_grand": bool(it.get("luxury_grand")),
            }
            if req_unavail and not rec["mono"] and redir_note:
                rec["note"] = redir_note
            if edge:
                rec["edge_case"] = True
                rec["edge_register"] = edge
            triad.append(rec)

        # Ensure exactly one mono in final 3 (if we accidentally pulled two, demote the lower-scored mono to MIX alt if possible)
        mono_count = sum(1 for x in triad if x["mono"])
        if mono_count == 0 and mono_candidates:
            # promote a mono by replacing the lowest-ranked mix
            lowest_mix_idx = None
            for i, x in enumerate(triad):
                if not x["mono"]:
                    lowest_mix_idx = i
                    break
            if lowest_mix_idx is not None:
                m = mono_candidates[0]
                triad[lowest_mix_idx] = {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "desc": _truncate_words(m.get("desc",""), copy_cap) if edge else (m.get("desc") or ""),
                    "image": m.get("image_url") or m.get("image"),
                    "price": int(m.get("price", m.get("price_inr", 0)) or 0),
                    "currency": "INR",
                    "emotion": m.get("emotion"),
                    "tier": m.get("tier"),
                    "packaging": m.get("packaging"),
                    "mono": True,
                    "palette": m.get("palette") or [],
                    "luxury_grand": bool(m.get("luxury_grand")),
                }
                if edge:
                    triad[lowest_mix_idx]["edge_case"] = True
                    triad[lowest_mix_idx]["edge_register"] = edge

        # Order final by tier
        triad = _order_by_tier(triad)

        # Validate
        _validate_output(triad)

        return triad

def _validate_output(items: List[Dict[str,Any]]) -> None:
    assert len(items) == 3, "Triad must be length 3"
    ids = [x["id"] for x in items]
    assert len(set(ids)) == 3, "Duplicate SKUs in triad"
    palettes_ok = all(isinstance(x.get("palette"), list) and len(x.get("palette"))>0 for x in items)
    assert palettes_ok, "All items must include non-empty palette[]"
    mono_count = sum(1 for x in items if x.get("mono"))
    assert mono_count == 1, "Exactly one MONO required"
    # One per tier if possible (soft assert — do not raise if catalog thin)
    tiers = [x.get("tier") for x in items]
    if len(set(tiers)) != 3:
        # soft: do nothing; kept to protect thin catalogs
        pass

# Simple local run
if __name__ == "__main__":
    sample = selection_engine("valentine surprise", {"budget_inr": 6000})
    print(json.dumps(sample, indent=2, ensure_ascii=False))
