# DocuFlow evaluation: Attention Is All You Need (arXiv:1706.03762v7)

- Embeddings: `fastembed:BAAI/bge-small-en-v1.5`
- Chunks: 52 (size 1000, overlap 200); questions: 25 (25 have their evidence inside a single chunk)
- Run: 2026-10-18 20:43 UTC

## Retrieval

| Retriever | Hit rate | MRR | Mean context (chars) |
|---|---|---|---|
| similarity | 22/25 (88%) | 0.760 | 4,492 |
| mmr | 19/25 (76%) | 0.708 | 4,411 |
| similarity+compression | 21/25 (84%) | 0.700 | 2,604 |
| mmr+compression | 19/25 (76%) | 0.680 | 2,606 |
