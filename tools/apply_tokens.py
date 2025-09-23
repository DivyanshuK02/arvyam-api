#!/usr/bin/env python3
"""
apply_tokens.py — apply human-approved phrases to the rulebook (additive only).

Review CSV columns (header required):
  phrase,decision,anchor,notes[,list_type][,intensity][,polarity]

- decision: approve|reject   (only 'approve' rows are applied)
- anchor:   one of 8 fixed anchors (exact text)
- list_type (optional): synonyms|misspellings  (default: synonyms)
- intensity/polarity (optional): if provided, row is stored in synonyms_detailed[]

Safety:
  - Validates anchor enum
  - Dedupe case-insensitive
  - Creates timestamped .bak backup
  - Supports --dry-run

Typical:
  python tools/apply_tokens.py \
    --review feeder/runs/2025-09-22/review.csv \
    --rules app/rules/emotion_keywords.json \
    --dry-run
"""

from __future__ import annotations
import argparse, csv, datetime as dt, json, shutil, sys
from pathlib import Path

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

VALID_LIST_TYPES = {"synonyms", "misspellings"}
VALID_INTENSITY   = {"low", "med", "high"}
VALID_POLARITY    = {"pos", "neu", "neg"}

def _cf(s: str) -> str:
    return (s or "").casefold()

def load_rulebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def backup(path: Path) -> Path:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_suffix(path.suffix + f".{ts}.bak")
    shutil.copy2(path, bak)
    return bak

def ensure_lists(block: dict):
    block.setdefault("synonyms", [])
    block.setdefault("misspellings", [])
    block.setdefault("synonyms_detailed", [])

def dedupe_lower(strings: list[str]) -> list[str]:
    seen = set()
    out = []
    for s in strings:
        k = _cf(s)
        if k not in seen:
            out.append(s)
            seen.add(k)
    return out

def main():
    ap = argparse.ArgumentParser(description="Apply approved phrases to emotion_keywords.json")
    ap.add_argument("--review", required=True, help="CSV with phrase decisions")
    ap.add_argument("--rules",   default="rules/emotion_keywords.json", help="Rulebook JSON path")
    ap.add_argument("--dry-run", action="store_true", help="Print changes only; don't write")
    args = ap.parse_args()

    rules_path = Path(args.rules)
    rb = load_rulebook(rules_path)

    # Build quick lookup for existing strings (casefold) per anchor + list
    def existing_set(block, list_key):
        return {_cf(s) for s in (block.get(list_key) or []) if isinstance(s, str)}

    # Collect changes staged per anchor
    staged = {a: {"synonyms": [], "misspellings": [], "synonyms_detailed": []} for a in ANCHORS}

    # Read review CSV
    added_rows = 0
    with open(args.review, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        required = {"phrase","decision","anchor"}
        missing = required - set(rdr.fieldnames or [])
        if missing:
            print(f"[error] review CSV missing columns: {sorted(missing)}", file=sys.stderr)
            sys.exit(2)

        for row in rdr:
            if _cf(row.get("decision")) != "approve":
                continue
            phrase = (row.get("phrase") or "").strip()
            anchor = (row.get("anchor") or "").strip()
            if not phrase:
                continue
            if anchor not in ANCHORS:
                print(f"[warn] skip unknown anchor: {anchor!r} for phrase {phrase!r}")
                continue

            list_type = (row.get("list_type") or "synonyms").strip().lower()
            if list_type not in VALID_LIST_TYPES:
                list_type = "synonyms"

            intensity = (row.get("intensity") or "").strip().lower() or None
            polarity  = (row.get("polarity")  or "").strip().lower() or None
            use_detailed = bool(intensity or polarity)

            if use_detailed:
                if intensity and intensity not in VALID_INTENSITY:
                    print(f"[warn] ignore invalid intensity {intensity!r} for {phrase!r}")
                    intensity = None
                if polarity and polarity not in VALID_POLARITY:
                    print(f"[warn] ignore invalid polarity {polarity!r} for {phrase!r}")
                    polarity = None

            staged[anchor]["synonyms_detailed" if use_detailed else list_type].append(
                {"token": phrase, "intensity": intensity, "polarity": polarity}
                if use_detailed else phrase
            )
            added_rows += 1

    if not added_rows:
        print("[info] no approved rows to apply; nothing to do.")
        return

    # Apply to rulebook (additive + dedupe)
    for anchor, adds in staged.items():
        if not any(adds.values()):
            continue
        block = rb.setdefault(anchor, {})
        ensure_lists(block)

        # simple lists
        for list_key in ("synonyms", "misspellings"):
            if not adds[list_key]:
                continue
            existing = existing_set(block, list_key)
            for s in adds[list_key]:
                if _cf(s) not in existing:
                    block[list_key].append(s)
                    existing.add(_cf(s))
            block[list_key] = dedupe_lower([s for s in block[list_key] if isinstance(s, str)])

        # detailed list
        if adds["synonyms_detailed"]:
            # dedupe by token casefold
            existing_det = {_cf(d.get("token","")) for d in (block.get("synonyms_detailed") or []) if isinstance(d, dict)}
            for d in adds["synonyms_detailed"]:
                tok = _cf(d.get("token",""))
                if tok and tok not in existing_det:
                    block["synonyms_detailed"].append(d)
                    existing_det.add(tok)

    # Show diff-ish summary
    print("[plan] staged updates:")
    for a in ANCHORS:
        adds = staged[a]
        if any(adds.values()):
            print(f"  - {a}: +{len(adds['synonyms'])} synonyms,"
                  f" +{len(adds['misspellings'])} misspellings,"
                  f" +{len(adds['synonyms_detailed'])} detailed")

    if args.dry_run:
        print("[dry-run] not writing changes.")
        return

    bak = backup(rules_path)
    with rules_path.open("w", encoding="utf-8") as f:
        json.dump(rb, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[ok] rulebook updated: {rules_path} (backup: {bak.name})")

if __name__ == "__main__":
    main()

