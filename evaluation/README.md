# Evaluation

How well does DocuFlow find the right passage, and do its answers stick to what it found?

## Setup

| | |
|---|---|
| Document | *Attention Is All You Need* ([arXiv:1706.03762v7](https://arxiv.org/abs/1706.03762), 15 pages), downloaded at run time and checked against a pinned SHA-256 |
| Questions | 25 hand-written factual questions in [`qa_attention.jsonl`](qa_attention.jsonl). Each has a reference answer and one or more short **evidence** phrases copied from the paper |
| Pipeline | The app's own ingestion path: `ChatIngestor` → PyPDFLoader → 1,000-char chunks with 200 overlap → FAISS (52 chunks) |
| Retrievers | `similarity` and `mmr` (k=5, fetch_k=20, λ=0.5), each with and without contextual compression |

## Metrics

- **Hit rate:** the share of questions where an evidence phrase appears in one of the k retrieved passages. Matching ignores case, spacing and punctuation. Every evidence phrase sits whole inside at least one chunk, so 100% is reachable.
- **MRR:** the mean of 1/rank of the first passage containing the evidence.
- **Faithfulness:** a judge model splits each answer into atomic claims and marks each one supported or not **by the retrieved context alone**. Each answer scores supported ÷ claims (the RAGAS definition), and the scores are averaged over answers that made claims.
- **Correct:** the judge's verdict on whether the answer states the reference fact.
- **Abstained:** the answer said it didn't know and made no claims.

By default the generator is the app's configured model (`gpt-4o`) and the judge is a different model (`gpt-4.1`), which limits self-preference bias.

## Status

| Part | Status |
|---|---|
| Retrieval, local `bge-small` embeddings | Measured. Runs in CI on every PR that touches `evaluation/`, and results are in [`results/fastembed.md`](results/fastembed.md) |
| Retrieval with the production embeddings, plus answer faithfulness and correctness | Harness ready, **not yet run**. It needs `OPENAI_API_KEY` and costs about US$0.25 per run |

## Running it

```bash
# Retrieval only, local embeddings: no API key needed (~20 s)
pip install -r requirements.txt -r evaluation/requirements.txt
python -m evaluation.run_eval --embeddings fastembed --retrieval-only

# Full run: production embeddings (text-embedding-3-small), gpt-4o answers, gpt-4.1 judge
OPENAI_API_KEY=... python -m evaluation.run_eval --embeddings openai --llm openai \
    --judge-provider openai --judge-model gpt-4.1
```

Each run writes `results/<embeddings>.json`, with every per-question score, retrieved rank, answer and judge verdict, plus a `.md` summary. In CI, [`.github/workflows/evaluation.yml`](../.github/workflows/evaluation.yml) runs the retrieval part on PRs. For the full run, add an `OPENAI_API_KEY` repository secret and start it from **Actions → RAG evaluation → Run workflow**.

## Limitations

- **Small sample.** One document and 25 questions: a single question moves hit rate by 4 points. Read differences of one or two questions as noise.
- **Easy question type.** The questions are single-fact lookups, which favour pure similarity search. MMR and compression are aimed at broad or multi-document questions, which this set doesn't cover.
- **Strict hit metric.** A passage that paraphrases the evidence, or a compressed piece that splits the phrase, counts as a miss.
- **LLM judge.** Faithfulness and correctness come from an LLM judge, not human labels. The per-question verdicts are in the JSON so they can be audited.
