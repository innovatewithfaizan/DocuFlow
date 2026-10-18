"""
Evaluate DocuFlow's retrieval and answers on "Attention Is All You Need" (arXiv:1706.03762v7).

The document goes through the app's real ingestion path (ChatIngestor: PyPDFLoader ->
RecursiveCharacterTextSplitter -> FAISS), then:

  1. retrieval: every question is run against similarity, MMR and both with compression;
     hit rate, MRR and context size are reported (see evaluation/metrics.py);
  2. generation (unless --retrieval-only): ConversationalRAG answers each question with
     the configured retriever, and a separate judge model scores faithfulness and
     correctness.

Examples
    # retrieval only, local embeddings, no keys
    python -m evaluation.run_eval --embeddings fastembed --retrieval-only

    # full run: production embeddings, gpt-4o answers judged by gpt-4.1 (~US$0.25)
    OPENAI_API_KEY=... python -m evaluation.run_eval --embeddings openai --llm openai \\
        --judge-provider openai --judge-model gpt-4.1
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import (  # noqa: E402
    evidence_rank, generation_summary, faithfulness_score, normalize, parse_judgement, retrieval_summary,
)
from src.document_chat.retrieval import ConversationalRAG  # noqa: E402
from src.document_chat.retriever_factory import RetrieverSettings, build_retriever  # noqa: E402
from src.document_ingestion.data_ingestion import ChatIngestor  # noqa: E402
from utils.model_loader import ModelLoader  # noqa: E402

EVAL_DIR = ROOT / "evaluation"
PDF_URL = "https://arxiv.org/pdf/1706.03762v7"
PDF_SHA256 = "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
PDF_PATH = EVAL_DIR / "data" / "attention_is_all_you_need.pdf"

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict evaluator of answers produced by a retrieval-augmented system. "
     "You judge only against the material provided, never your own knowledge. Reply with JSON only."),
    ("human",
     "QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n"
     "RETRIEVED CONTEXT (the only material the system could use):\n{context}\n\n"
     "ANSWER TO EVALUATE:\n{answer}\n\n"
     "Instructions:\n"
     "1. Split the ANSWER into atomic factual claims. Ignore filler and restatements of the question.\n"
     "2. For each claim set \"supported\" to true only if the RETRIEVED CONTEXT states it or directly implies it.\n"
     "3. Set \"abstained\" to true if the answer declines to answer (for example \"I don't know\") "
     "and makes no factual claim; then \"claims\" is empty.\n"
     "4. Set \"correct\" to true if the answer conveys the key fact of the REFERENCE ANSWER without "
     "contradicting it. Extra correct detail is fine.\n\n"
     "Return exactly: {{\"claims\": [{{\"claim\": \"...\", \"supported\": true}}], "
     "\"abstained\": false, \"correct\": true}}"),
])


class Throttle:
    """Keeps at least `interval` seconds between remote calls, for rate-limited free tiers."""

    def __init__(self, interval: float):
        self.interval, self._last = interval, 0.0

    def wait(self) -> None:
        delay = self._last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


class CachedEmbeddings(Embeddings):
    """Memoises vectors so each text is embedded once across all retriever configs."""

    def __init__(self, base: Embeddings, throttle: Throttle, batch_queries: bool):
        self.base, self.throttle, self.batch_queries = base, throttle, batch_queries
        self.docs: Dict[str, List[float]] = {}
        self.queries: Dict[str, List[float]] = {}
        self.calls = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self.docs))
        if missing:
            self.throttle.wait()
            self.calls += 1
            self.docs.update(zip(missing, self.base.embed_documents(missing)))
        return [self.docs[t] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if text not in self.queries:
            self.throttle.wait()
            self.calls += 1
            self.queries[text] = self.base.embed_query(text)
        return self.queries[text]

    def prime_queries(self, questions: List[str]) -> None:
        """Remote models embed queries and documents identically, so batch them in one call."""
        if self.batch_queries:
            self.throttle.wait()
            self.calls += 1
            self.queries.update(zip(questions, self.base.embed_documents(questions)))


class _Upload:
    def __init__(self, path: Path):
        self.name, self._path = path.name, path

    def getbuffer(self) -> bytes:
        return self._path.read_bytes()


def ensure_pdf() -> str:
    if not PDF_PATH.exists():
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {PDF_URL}")
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    digest = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    if digest != PDF_SHA256:
        print(f"WARNING: PDF sha256 {digest} differs from the pinned {PDF_SHA256}; results may not be comparable")
    return digest


def load_dataset(path: Path, document_text: str) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hay = normalize(document_text)
    for row in rows:
        if not any(normalize(e) in hay for e in row["evidence"]):
            raise SystemExit(f"{row['id']}: no evidence phrase occurs in the document: {row['evidence']}")
    return rows


def build_index(embeddings: CachedEmbeddings, workdir: Path, chunk_size: int, chunk_overlap: int):
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)  # ChatIngestor writes data/, faiss_index/ and logs/ relative to cwd
    ingestor = ChatIngestor(temp_base="data", faiss_base="faiss_index", session_id="attention")
    if (ingestor.faiss_dir / "index.faiss").exists():
        for f in ingestor.faiss_dir.iterdir():
            f.unlink()
    ingestor.model_loader.load_embeddings = lambda: embeddings  # route indexing through the cache
    ingestor.build_retriever([_Upload(PDF_PATH)], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = FAISS.load_local(str(ingestor.faiss_dir), embeddings, allow_dangerous_deserialization=True)
    return store, ingestor.stats


def answerable(store, rows) -> int:
    """Questions whose evidence sits whole inside at least one indexed chunk (the hit-rate ceiling)."""
    chunks = [normalize(d.page_content) for d in store.docstore._dict.values()]
    return sum(any(normalize(e) in c for c in chunks for e in row["evidence"]) for row in rows)


def retrieval_configs(k: int) -> Dict[str, RetrieverSettings]:
    base = RetrieverSettings.from_config(k=k)
    return {
        "similarity": base.with_overrides(search_type="similarity", compress=False),
        "mmr": base.with_overrides(search_type="mmr", compress=False),
        "similarity+compression": base.with_overrides(search_type="similarity", compress=True),
        "mmr+compression": base.with_overrides(search_type="mmr", compress=True),
    }


def evaluate_retrieval(store, embeddings, rows, configs) -> Dict[str, Any]:
    results = {}
    for name, settings in configs.items():
        retriever = build_retriever(store, settings, embeddings=embeddings)
        ranks, chars, per_q = [], [], []
        for row in rows:
            contexts = [d.page_content for d in retriever.invoke(row["question"])]
            rank = evidence_rank(contexts, row["evidence"])
            ranks.append(rank)
            chars.append(sum(len(c) for c in contexts))
            per_q.append({"id": row["id"], "rank": rank, "context_chars": chars[-1]})
        results[name] = {"settings": _settings_dict(settings), "summary": retrieval_summary(ranks, chars),
                         "per_question": per_q}
        print(f"  {name:24s} {results[name]['summary']}")
    return results


def evaluate_generation(store, embeddings, rows, settings, judge, throttle) -> Dict[str, Any]:
    rag = ConversationalRAG(session_id="eval", retriever=build_retriever(store, settings, embeddings=embeddings))
    judgements, per_q = [], []
    for row in rows:
        throttle.wait()
        result = rag.invoke_with_sources(row["question"])
        prompt = JUDGE_PROMPT.invoke({
            "question": row["question"], "reference": row["answer"],
            "context": "\n\n---\n\n".join(result["contexts"]), "answer": result["answer"],
        })
        throttle.wait()
        judgement = parse_judgement(judge.invoke(prompt).content)
        judgements.append(judgement)
        per_q.append({"id": row["id"], "question": row["question"], "answer": result["answer"],
                      "reference": row["answer"], "faithfulness": faithfulness_score(judgement), **judgement})
        print(f"  {row['id']} faithfulness={per_q[-1]['faithfulness']} correct={judgement['correct']}")
    return {"settings": _settings_dict(settings), "generator": _model_name(rag.llm),
            "summary": generation_summary(judgements), "per_question": per_q}


def _settings_dict(s: RetrieverSettings) -> Dict[str, Any]:
    return {"search_type": s.search_type, "k": s.k, "fetch_k": s.fetch_k, "lambda_mult": s.lambda_mult,
            "compression": dataclasses.asdict(s.compression) if s.compression.enabled else None}


def _model_name(obj: Any) -> Optional[str]:
    return getattr(obj, "model_name", None) or getattr(obj, "model", None)


def write_report(out: Path, report: Dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# DocuFlow evaluation: {report['dataset']['document']}", "",
             f"- Embeddings: `{report['models']['embeddings']}`",
             f"- Chunks: {report['index']['chunks']} (size {report['index']['chunk_size']}, "
             f"overlap {report['index']['chunk_overlap']}); questions: {report['dataset']['questions']} "
             f"({report['index']['answerable_questions']} have their evidence inside a single chunk)",
             f"- Run: {report['run_at']}", "",
             "## Retrieval", "", "| Retriever | Hit rate | MRR | Mean context (chars) |", "|---|---|---|---|"]
    for name, r in report["retrieval"].items():
        s = r["summary"]
        lines.append(f"| {name} | {s['hits']}/{s['questions']} ({s['hit_rate']:.0%}) | {s['mrr']:.3f} | "
                     f"{s['mean_context_chars']:,} |")
    gen = report.get("generation")
    if gen:
        s = gen["summary"]
        lines += ["", "## Answers", "",
                  f"- Generator: `{report['models']['llm']}` with the `{gen['settings']['search_type']}` "
                  f"retriever (k={gen['settings']['k']}); judge: `{report['models']['judge']}`", "",
                  "| Faithfulness | Fully faithful answers | Correct | Abstained |", "|---|---|---|---|",
                  f"| {s['faithfulness']:.2f} | {s['fully_faithful_answers']}/{s['answered']} | "
                  f"{s['correctness']:.0%} | {s['abstention_rate']:.0%} |"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qa", type=Path, default=EVAL_DIR / "qa_attention.jsonl")
    p.add_argument("--embeddings", default="fastembed", help="embedding provider (fastembed|openai|google)")
    p.add_argument("--llm", help="LLM provider key in config.yaml for answers (openai|groq|google)")
    p.add_argument("--judge-provider", help="LLM provider key for the judge (defaults to --llm)")
    p.add_argument("--judge-model", help="model name for the judge (defaults to the provider's model)")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--min-interval", type=float, default=0.0, help="seconds between remote API calls")
    p.add_argument("--out", type=Path, help="output path without extension (default evaluation/results/<embeddings>)")
    args = p.parse_args(argv)
    if not args.retrieval_only and not args.llm:
        p.error("--llm is required unless --retrieval-only")

    os.environ["EMBEDDING_PROVIDER"] = args.embeddings
    if args.llm:
        os.environ["LLM_PROVIDER"] = args.llm
    out = (args.out or EVAL_DIR / "results" / args.embeddings).resolve()
    qa_path = args.qa.resolve()

    logging.getLogger().setLevel(logging.WARNING)  # the app logs every call at INFO
    sha = ensure_pdf()
    from langchain_community.document_loaders import PyPDFLoader
    document_text = "\n".join(d.page_content for d in PyPDFLoader(str(PDF_PATH)).load())
    rows = load_dataset(qa_path, document_text)

    throttle = Throttle(args.min_interval)
    loader = ModelLoader()
    embeddings = CachedEmbeddings(loader.load_embeddings(), throttle,
                                  batch_queries=args.embeddings == "openai")
    print(f"Indexing with {args.embeddings} embeddings...")
    store, stats = build_index(embeddings, EVAL_DIR / ".cache" / args.embeddings, args.chunk_size, args.chunk_overlap)
    embeddings.prime_queries([r["question"] for r in rows])

    print("Retrieval:")
    report: Dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": {"document": "Attention Is All You Need (arXiv:1706.03762v7)", "pdf_sha256": sha,
                    "qa_file": qa_path.name, "questions": len(rows)},
        "index": {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, **stats,
                  "answerable_questions": answerable(store, rows)},
        "models": {"embeddings": f"{args.embeddings}:{_model_name(embeddings.base)}"},
        "retrieval": evaluate_retrieval(store, embeddings, rows, retrieval_configs(args.k)),
    }

    if not args.retrieval_only:
        judge = loader.load_llm(args.judge_provider or args.llm, model_name=args.judge_model)
        settings = RetrieverSettings.from_config(k=args.k)
        print(f"Answers ({settings.search_type}):")
        report["generation"] = evaluate_generation(store, embeddings, rows, settings, judge, throttle)
        report["models"]["llm"] = f"{args.llm}:{report['generation']['generator']}"
        report["models"]["judge"] = f"{args.judge_provider or args.llm}:{_model_name(judge)}"

    report["api_calls"] = {"embedding_requests": embeddings.calls}
    write_report(out, report)
    print(f"Wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
