# RAG Retrieval Evaluation

This benchmark is separate from two-tower teammate prediction. Queries are authored from real profile
attributes, and relevance labels come from recorded role-relative statistics or champion history.

- Real profiles: **2,854**
- Controlled queries: **40**
- Embedding model: **sentence-transformers/all-MiniLM-L6-v2**

| Retriever | NDCG@10 (95% CI) | Precision@10 | Hit rate@10 |
|---|---:|---:|---:|
| semantic_chroma | 0.3792 [0.2934, 0.4719] | 0.3675 | 1.0000 |
| hybrid_chroma | 0.8641 [0.7780, 0.9402] | 0.8550 | 1.0000 |
| tfidf | 0.4483 [0.3335, 0.5753] | 0.4400 | 0.7500 |
| random | 0.2076 [0.1617, 0.2542] | 0.2150 | 0.8750 |

## Grounded hybrid retrieval by query type

| Type | NDCG@10 |
|---|---:|
| assists | 1.0000 |
| champion | 0.4563 |
| damage | 1.0000 |
| experience | 1.0000 |
| kda | 1.0000 |
| vision | 1.0000 |
| win_rate | 1.0000 |

## Limitations

- Labels are deterministic judgments derived from real profile fields, not human preference ratings.
- This evaluates retrieval, not the OpenAI-generated scout prose.
- Scout quality still requires a human rubric for grounding, usefulness, and unsupported claims.
