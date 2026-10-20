"""
Scoring for the Groundwork evaluation. Pure functions, unit-tested in tests/test_evaluation_metrics.py.

Retrieval
    A question is a *hit* when any of its evidence phrases (short verbatim spans from
    the source document) appears in a retrieved passage. Matching ignores case,
    whitespace and punctuation, so PDF extraction quirks ("d model = 512" vs
    "dmodel=512") do not count as misses.
    hit_rate = hits / questions;  MRR = mean of 1/rank of the first passage that hits.

Generation (LLM-as-judge)
    faithfulness: the judge splits the answer into atomic claims and marks each as
    supported or not *by the retrieved context only*; per answer, score = supported /
    claims (the RAGAS definition). Averaged over answers that made claims.
    correctness: the judge's verdict on whether the answer states the reference fact.
    abstention: the answer declined ("I don't know") and made no claims.
"""
from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def evidence_rank(contexts: Sequence[str], evidence: Iterable[str]) -> Optional[int]:
    """1-based rank of the first context containing any evidence phrase, else None."""
    needles = [normalize(e) for e in evidence if normalize(e)]
    for rank, ctx in enumerate(contexts, start=1):
        hay = normalize(ctx)
        if any(n in hay for n in needles):
            return rank
    return None


def retrieval_summary(ranks: Sequence[Optional[int]], context_chars: Sequence[int]) -> Dict[str, Any]:
    n = len(ranks)
    if n == 0:
        raise ValueError("no questions scored")
    return {
        "questions": n,
        "hits": sum(r is not None for r in ranks),
        "hit_rate": round(sum(r is not None for r in ranks) / n, 4),
        "mrr": round(mean(1.0 / r if r else 0.0 for r in ranks), 4),
        "mean_context_chars": round(mean(context_chars)),
    }


def parse_judgement(raw: str) -> Dict[str, Any]:
    """Parse the judge's JSON reply (tolerates code fences and surrounding prose)."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"judge reply has no JSON object: {raw[:200]!r}")
    data = json.loads(match.group(0))
    claims = data.get("claims") or []
    if not isinstance(claims, list):
        raise ValueError("'claims' must be a list")
    return {
        "claims": [{"claim": str(c.get("claim", "")), "supported": bool(c.get("supported"))} for c in claims],
        "abstained": bool(data.get("abstained", False)),
        "correct": bool(data.get("correct", False)),
    }


def faithfulness_score(judgement: Dict[str, Any]) -> Optional[float]:
    """Supported claims / claims; None when the answer made no claims (e.g. abstained)."""
    claims = judgement["claims"]
    if judgement["abstained"] or not claims:
        return None
    return sum(c["supported"] for c in claims) / len(claims)


def generation_summary(judgements: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(judgements)
    if n == 0:
        raise ValueError("no answers scored")
    scores = [s for s in (faithfulness_score(j) for j in judgements) if s is not None]
    return {
        "questions": n,
        "answered": len(scores),
        "faithfulness": round(mean(scores), 4) if scores else None,
        "fully_faithful_answers": sum(s == 1.0 for s in scores),
        "correctness": round(sum(j["correct"] for j in judgements) / n, 4),
        "abstention_rate": round(sum(j["abstained"] for j in judgements) / n, 4),
    }
