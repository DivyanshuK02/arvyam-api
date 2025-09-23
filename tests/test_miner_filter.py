from pathlib import Path
import json

import tools.mine_unknowns as miner


def test_known_tokens_suppressed_and_stopwords(tmp_path):
    # Minimal rulebook with known tokens
    rb = {
        "exact": {"thank you": "Selflessness/Generosity"},
        "keywords": {"Encouragement/Positivity": ["good luck"]},
        "combos": [{"all": ["board", "exam"]}],
    }
    rb_path = tmp_path / "rb.json"
    rb_path.write_text(json.dumps(rb), encoding="utf-8")

    known = miner.load_rulebook_tokens(rb_path)

    prompts = [
        "Good luck for board exam",     # should be suppressed (known)
        "please please sorry yaar",     # 'please' stopword; reduce to 'sorry' (if yaar is in custom stopwords)
        "so so happy",                  # 'so' is stopword; reduce to 'happy'
    ]

    stopwords = set(miner.DEFAULT_STOPWORDS)
    stopwords.update({"yaar"})  # simulate extra Indic-English stopwords

    unknowns = miner.find_unknowns(prompts, known_tokens=known, min_count=1, stopwords=stopwords)

    phrases = {u["phrase"] for u in unknowns}
    assert "happy" in phrases
    assert "sorry" in phrases or "sorry yaar" in phrases
    # the known 'good luck...' should not appear
    assert not any("good luck" in u["phrase"] for u in unknowns)


def test_iter_evidence_prompts_fallback_only(tmp_path):
    log = tmp_path / "evidence.jsonl"
    lines = [
        {"prompt": "kept (has fallback)", "fallback_reason": "in_family"},
        {"prompt": "dropped (no fallback)", "fallback_reason": ""},
        {"prompt": "also dropped (missing key)"},
    ]
    log.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    kept = list(miner.iter_evidence_prompts([str(log)], fallback_only=True))
    assert kept == ["kept (has fallback)"]
