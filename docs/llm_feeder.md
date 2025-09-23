# Offline Feeder (Phase 1.4) — Add‑only updates to `emotion_keywords.json`

**What this is:** A simple, weekly, human‑in‑the‑loop process to safely grow `app/rules/emotion_keywords.json` (synonyms, misspellings, optional intensity/polarity).  
**What it is NOT:** No runtime LLM, no code/schema changes, no request‑path modifications.

---

## One‑time layout

```
repo-root/
  app/
    rules/
      emotion_keywords.json     # rulebook to enrich
  tools/
    mine_unknowns.py            # find uncaught user phrases from evidence logs
    make_review_sheet.py        # build a reviewer-friendly CSV
    apply_tokens.py             # apply APPROVED rows (additive; auto-backup)
    validate_rulebook.py        # quick schema/shape check
    stopwords_indic_en.txt      # small Indic-English noise list (optional)
  feeder/
    runs/                       # per-run artifacts
  docs/
    llm_feeder.md               # this file
```

---

## Weekly loop (copy–paste commands)

> Replace `$(date +%F)` with your run date if needed.

### 0) Quick rulebook sanity check (optional but recommended)
```bash
python tools/validate_rulebook.py app/rules/emotion_keywords.json
```

### 1) Mine phrases we missed
```bash
RUN="feeder/runs/$(date +%F)"; mkdir -p "$RUN"

python tools/mine_unknowns.py \
  --logs logs/evidence_*.jsonl \
  --rules app/rules/emotion_keywords.json \
  --out "$RUN/candidates.json" \
  --min-count 3 \
  --fallback-only \
  --stopwords-file tools/stopwords_indic_en.txt
```
**Output:** `$RUN/candidates.json`

### 2) (Optional) Add offline proposals
Create `$RUN/llm_proposals.json` (hand-made or offline LLM), for example:
```json
[
  {"phrase": "ecstatic for you", "count": 5,
   "suggested_anchor": "Encouragement/Positivity",
   "rationale": "congrats-like; positive"}
]
```

### 3) Build the review sheet
```bash
python tools/make_review_sheet.py \
  --in "$RUN/candidates.json" \
  --in "$RUN/llm_proposals.json" \
  --out "$RUN/review.csv" \
  --min-count 3 \
  --limit 250 \
  --default-list-type synonyms
```
**Reviewers fill only:**
- `decision` → `approve` or `skip`
- `anchor` → one of the 8 anchors (exact spelling)
- optional: `list_type` (`synonyms|misspellings|synonyms_detailed`), `intensity` (`low|med|high`), `polarity` (`pos|neu|neg`)

### 4) Dry‑run apply (safe preview)
```bash
python tools/apply_tokens.py \
  --review "$RUN/review.csv" \
  --rules  app/rules/emotion_keywords.json \
  --dry-run
```

### 5) Apply for real (additive + auto‑backup)
```bash
python tools/apply_tokens.py \
  --review "$RUN/review.csv" \
  --rules  app/rules/emotion_keywords.json
```
**Backup created next to the rules file:** `emotion_keywords.json.YYYYMMDDTHHMMSSZ.bak`

### 6) Open PR
Commit the updated rulebook (and, if desired, the run folder) and open a PR. CI guards will run automatically.

---

## Guardrails (enforced by tools)
- **Add‑only** in Phase 1.4 (no deletions)
- Hard‑fail enums in `apply_tokens.py`:
  - `anchor` ∈ 8 anchors
  - `decision` ∈ `{approve, skip}` (`reject` treated as `skip` with a warning)
  - `list_type` ∈ `{synonyms, misspellings, synonyms_detailed}`
  - `intensity` ∈ `{low, med, high, ""}` ; `polarity` ∈ `{pos, neu, neg, ""}`
- First approved row wins per phrase (duplicates ignored with a warning)
- Deterministic sorting of lists keeps PR diffs tiny

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
- **Analyst** → mines unknowns, drafts proposals
- **Librarian** → reviews CSV, sets final anchor/list_type/notes
- **Maintainer** → applies tokens and opens PR
