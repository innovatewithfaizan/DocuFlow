"""Scoring functions behind the numbers in the README (evaluation/metrics.py)."""
import json
from pathlib import Path

import pytest

from evaluation.metrics import (
    evidence_rank, faithfulness_score, generation_summary, normalize, parse_judgement, retrieval_summary,
)


def test_normalize_ignores_case_spacing_and_punctuation():
    assert normalize("d model = 512") == normalize("dmodel=512") == "dmodel512"


def test_evidence_rank_finds_first_matching_passage():
    contexts = ["unrelated", "The encoder is a stack of N = 6 identical layers.", "N = 6 again"]
    assert evidence_rank(contexts, ["stack of N=6 identical layers"]) == 2


def test_evidence_rank_accepts_any_of_several_phrases():
    assert evidence_rank(["trained for 3.5 days"], ["300,000 steps", "3.5 days"]) == 1


def test_evidence_rank_is_none_on_miss():
    assert evidence_rank(["nothing relevant"], ["28.4 BLEU"]) is None


def test_retrieval_summary():
    s = retrieval_summary([1, 2, None, 1], [100, 200, 300, 400])
    assert s == {"questions": 4, "hits": 3, "hit_rate": 0.75, "mrr": 0.625, "mean_context_chars": 250}


def test_parse_judgement_tolerates_code_fences():
    raw = '```json\n{"claims": [{"claim": "N is 6", "supported": true}], "abstained": false, "correct": true}\n```'
    assert parse_judgement(raw) == {"claims": [{"claim": "N is 6", "supported": True}],
                                    "abstained": False, "correct": True}


def test_parse_judgement_rejects_non_json():
    with pytest.raises(ValueError):
        parse_judgement("The answer looks fine to me.")


def test_faithfulness_is_supported_over_total_claims():
    j = {"claims": [{"supported": True}, {"supported": False}, {"supported": True}, {"supported": True}],
         "abstained": False, "correct": True}
    assert faithfulness_score(j) == 0.75


def test_abstentions_have_no_faithfulness_score():
    assert faithfulness_score({"claims": [], "abstained": True, "correct": False}) is None


def test_generation_summary():
    judgements = [
        {"claims": [{"supported": True}], "abstained": False, "correct": True},
        {"claims": [{"supported": True}, {"supported": False}], "abstained": False, "correct": False},
        {"claims": [], "abstained": True, "correct": False},
    ]
    assert generation_summary(judgements) == {
        "questions": 3, "answered": 2, "faithfulness": 0.75, "fully_faithful_answers": 1,
        "correctness": 0.3333, "abstention_rate": 0.3333,
    }


def test_dataset_is_well_formed():
    path = Path(__file__).resolve().parent.parent / "evaluation" / "qa_attention.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert 20 <= len(rows) <= 30
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["question"].endswith("?") and r["answer"] and r["evidence"]
