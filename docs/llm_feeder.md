
# Offline Feeder (Phase 1.4) — Add-only updates to `emotion_keywords.json`

**Purpose**  
A simple, weekly, human-in-the-loop process to grow `app/rules/emotion_keywords.json` (synonyms, misspellings, optional intensity/polarity).  
**Out of scope**: no runtime LLM, no code/schema changes, no request-path modifications.

---

## One-time layout

```
repo-root/
  app/
    rules/
      emotion_keywords.json
  tools/
    mine_unknowns.py
    make_review_sheet.py
    apply_tokens.py
  feeder/
    runs/
  docs/
    llm_feeder.md
```

---

## Weekly loop (copy–paste)

### 1) Mine phrases we missed
```bash
RUN="feeder/runs/$(date +%F)"; mkdir -p "$RUN"

python tools/mine_unknowns.py   --logs logs/evidence_*.jsonl   --rules app/rules/emotion_keywords.json   --out "$RUN/candidates.json"   --min-count 3   --fallback-only
```

**Output:** `$RUN/candidates.json`
```json
[
  {"phrase": "ecstatic for you", "count": 5},
  {"phrase": "sorry yaar", "count": 3}
]
```

### 2) (Optional) Add offline proposals
Create `$RUN/llm_proposals.json`:
```json
[
  {
    "phrase": "ecstatic for you",
    "count": 5,
    "suggested_anchor": "Encouragement/Positivity",
    "rationale": "congrats-like; positive"
  }
]
```

### 3) Build the review sheet
```bash
python tools/make_review_sheet.py   --in "$RUN/candidates.json"   --in "$RUN/llm_proposals.json"   --out "$RUN/review.csv"   --min-count 3   --limit 250   --default-list-type synonyms
```

### 4) Dry-run apply (preview only)
```bash
python tools/apply_tokens.py   --review "$RUN/review.csv"   --rules  app/rules/emotion_keywords.json   --dry-run
```

### 5) Apply for real (additive + auto-backup)
```bash
python tools/apply_tokens.py   --review "$RUN/review.csv"   --rules  app/rules/emotion_keywords.json
```

Backup created next to the rules file:
```
emotion_keywords.json.YYYYMMDDTHHMMSSZ.bak
```

---

## Guardrails & conventions

- **Add-only** in Phase 1.4 (no deletions).
- Keep tokens **lowercase** (Latin script).
- Avoid generic words (e.g., “team”, “role”) in flat anchor lists—use combos/proximity/disambiguation instead.
- If a phrase can map to multiple anchors, prefer **disambiguation** over duplicating keywords.
- Unknown enum values (anchor/list_type/intensity/polarity) are ignored by tools with a warning.

---

## Quick format reference

**`candidates.json`**
```json
[{ "phrase": "string", "count": 12 }]
```

**`llm_proposals.json` (optional)**
```json
[{ "phrase": "string", "count": 5, "suggested_anchor": "Anchor/Name", "rationale": "string" }]
```

**`review.csv` columns**
```
phrase,decision,anchor,notes,list_type,intensity,polarity
```

**Roles**
- Analyst → mines unknowns, drafts proposals
- Librarian → reviews CSV, sets anchor/list_type/notes
- Maintainer → applies tokens and commits
