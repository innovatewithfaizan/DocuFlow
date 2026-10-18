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

The judge (`openai/gpt-4o`) is a different, larger model than the generator (`openai/gpt-4o-mini`), which limits self-preference bias.

## Running it

```bash
# Retrieval only, local embeddings: no API key needed (~20 s)
pip install -r requirements.txt -r evaluation/requirements.txt
python -m evaluation.run_eval --embeddings fastembed --retrieval-only

# Full run on GitHub Models (what CI does); any token with models:read works
GITHUB_TOKEN=... python -m evaluation.run_eval --embeddings github --llm github \
    --judge-provider github --judge-model openai/gpt-4o --min-interval 4.5
```

Each run writes `results/<embeddings>.json`, with every per-question score, retrieved rank, answer and judge verdict, plus a `.md` summary. In CI, [`.github/workflows/evaluation.yml`](../.github/workflows/evaluation.yml) runs both commands and uploads the results.

## Limitations

- **Small sample.** One document and 25 questions: a single question moves hit rate by 4 points. Read differences of one or two questions as noise.
- **Easy question type.** The questions are single-fact lookups, which favour pure similarity search. MMR and compression are aimed at broad or multi-document questions, which this set doesn't cover.
- **Strict hit metric.** A passage that paraphrases the evidence, or a compressed piece that splits the phrase, counts as a miss.
- **LLM judge.** Faithfulness and correctness come from an LLM judge, not human labels. The per-question verdicts are in the JSON so they can be audited.
