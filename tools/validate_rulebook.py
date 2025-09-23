#!/usr/bin/env python3
"""
Validate the structure of app/rules/emotion_keywords.json.

Checks:
- JSON loads and is a dict
- Optional top-level sections: exact (dict), combos (list[{all: [str,...]}]), keywords (dict[str, list[str]])
- For each of the 8 anchors (if present):
    synonyms: list[str]
    misspellings: list[str]
    synonyms_detailed: list[ { token: str, intensity?: low|med|high, polarity?: pos|neu|neg } ]
- All strings are non-empty
- exact[...] values (if present) must be valid anchors
- keywords[...] keys (if present) must be valid anchors
Exit codes:
- 0 = OK
- 2 = validation error
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ANCHORS = {
    "Affection/Support",
    "Loyalty/Dependability",
    "Encouragement/Positivity",
    "Strength/Resilience",
    "Intellect/Wisdom",
    "Adventurous/Creativity",
    "Selflessness/Generosity",
    "Fun/Humor",
}

VALID_INTENSITY = {"low", "med", "high"}
VALID_POLARITY = {"pos", "neu", "neg"}


def err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def _must_list_of_str(name: str, arr: Any) -> bool:
    if not isinstance(arr, list):
        err(f"{name} must be a list")
        return False
    for i, v in enumerate(arr):
        if not isinstance(v, str) or not v.strip():
            err(f"{name}[{i}] must be a non-empty string")
            return False
    return True


def _must_list_of_detailed(name: str, arr: Any) -> bool:
    if not isinstance(arr, list):
        err(f"{name} must be a list")
        return False
    for i, d in enumerate(arr):
        if not isinstance(d, dict):
            err(f"{name}[{i}] must be an object")
            return False
        tok = d.get("token")
        if not isinstance(tok, str) or not tok.strip():
            err(f"{name}[{i}].token must be a non-empty string")
            return False
        inten = d.get("intensity")
        if inten is not None and inten != "" and inten not in VALID_INTENSITY:
            err(f"{name}[{i}].intensity invalid: {inten!r}")
            return False
        polar = d.get("polarity")
        if polar is not None and polar != "" and polar not in VALID_POLARITY:
            err(f"{name}[{i}].polarity invalid: {polar!r}")
            return False
    return True


def validate_rulebook(rb: Dict[str, Any]) -> bool:
    ok = True

    # exact (optional)
    exact = rb.get("exact")
    if exact is not None:
        if not isinstance(exact, dict):
            err("exact must be a dict of {phrase: Anchor}")
            ok = False
        else:
            for phrase, anchor in exact.items():
                if not isinstance(phrase, str) or not phrase.strip():
                    err("exact keys must be non-empty strings")
                    ok = False
                if anchor not in ANCHORS:
                    err(f"exact[{phrase!r}] invalid anchor: {anchor!r}")
                    ok = False

    # combos (optional)
    combos = rb.get("combos")
    if combos is not None:
        if not isinstance(combos, list):
            err("combos must be a list")
            ok = False
        else:
            for i, c in enumerate(combos):
                if not isinstance(c, dict):
                    err(f"combos[{i}] must be an object")
                    ok = False
                    continue
                allv = c.get("all", [])
                if not _must_list_of_str(f"combos[{i}].all", allv):
                    ok = False

    # keywords (optional)
    keywords = rb.get("keywords")
    if keywords is not None:
        if not isinstance(keywords, dict):
            err("keywords must be a dict of {Anchor: [str,...]}")
            ok = False
        else:
            for ak, arr in keywords.items():
                if ak not in ANCHORS:
                    err(f"keywords[{ak!r}] invalid anchor")
                    ok = False
                if not _must_list_of_str(f"keywords[{ak}]", arr):
                    ok = False

    # anchor blocks (optional per anchor)
    for a in ANCHORS:
        if a not in rb:
            continue
        block = rb[a]
        if not isinstance(block, dict):
            err(f"anchor block {a!r} must be an object")
            ok = False
            continue
        if "synonyms" in block and not _must_list_of_str(f"{a}.synonyms", block["synonyms"]):
            ok = False
        if "misspellings" in block and not _must_list_of_str(f"{a}.misspellings", block["misspellings"]):
            ok = False
        if "synonyms_detailed" in block and not _must_list_of_detailed(f"{a}.synonyms_detailed", block["synonyms_detailed"]):
            ok = False

    return ok


def main(argv: List[str]) -> int:
    if not argv:
        err("usage: validate_rulebook.py app/rules/emotion_keywords.json")
        return 2
    path = Path(argv[0])
    try:
        rb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        err(f"failed to read/parse JSON: {e}")
        return 2
    if not isinstance(rb, dict):
        err("top-level JSON must be an object (dict)")
        return 2
    if validate_rulebook(rb):
        print(f"[ok] validation passed: {path}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
