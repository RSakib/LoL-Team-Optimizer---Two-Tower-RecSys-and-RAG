# Trained Two-Tower Evaluation

The query tower encodes the anonymous finder's role/rank plus the requested teammate role and optional
champion. The candidate tower encodes each real player's identity, role/rank, main champion, and normalized
historical metrics. Training uses real same-team pairs with role-matched sampled negatives.

## Leakage controls

- Train matches: **71,290**
- Validation matches: **15,276**
- Test matches: **15,277**
- Split is chronological at match level.
- Early-stopping epoch is selected on validation NDCG@10 only.
- The deployed checkpoint is retrained on train + validation and evaluated once on the held-out test window.

## Held-out successful future teammates with target champion context

Queries: **3,589**

| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |
|---|---:|---:|---:|
| two_tower | 0.3557 [0.3447, 0.3670] | 0.5865 [0.5726, 0.6031] | 0.3007 [0.2899, 0.3117] |
| popularity | 0.1403 [0.1321, 0.1483] | 0.2836 [0.2694, 0.2981] | 0.1204 [0.1133, 0.1268] |
| random | 0.0543 [0.0495, 0.0592] | 0.1243 [0.1144, 0.1339] | 0.0559 [0.0525, 0.0599] |

## Held-out successful future teammates with role/rank context only

Queries: **3,589**

| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |
|---|---:|---:|---:|
| two_tower | 0.1810 [0.1724, 0.1894] | 0.3667 [0.3514, 0.3823] | 0.1476 [0.1398, 0.1551] |
| popularity | 0.1403 [0.1321, 0.1483] | 0.2836 [0.2694, 0.2981] | 0.1204 [0.1133, 0.1268] |
| random | 0.0543 [0.0495, 0.0592] | 0.1243 [0.1144, 0.1339] | 0.0559 [0.0525, 0.0599] |

## Held-out all future teammates with target champion context

Queries: **3,618**

| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |
|---|---:|---:|---:|
| two_tower | 0.3320 [0.3209, 0.3424] | 0.5459 [0.5297, 0.5601] | 0.2841 [0.2738, 0.2943] |
| popularity | 0.1357 [0.1277, 0.1433] | 0.2698 [0.2557, 0.2840] | 0.1186 [0.1118, 0.1248] |
| random | 0.0550 [0.0493, 0.0600] | 0.1224 [0.1110, 0.1338] | 0.0572 [0.0530, 0.0614] |

## Held-out all future teammates with role/rank context only

Queries: **3,618**

| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |
|---|---:|---:|---:|
| two_tower | 0.1692 [0.1606, 0.1766] | 0.3413 [0.3270, 0.3549] | 0.1405 [0.1334, 0.1463] |
| popularity | 0.1357 [0.1277, 0.1433] | 0.2698 [0.2557, 0.2840] | 0.1186 [0.1118, 0.1248] |
| random | 0.0550 [0.0493, 0.0600] | 0.1224 [0.1110, 0.1338] | 0.0572 [0.0530, 0.0614] |

## Training

- Real directed training pairs: **58,850**
- Real directed final-training pairs: **69,378**
- Selected epochs: **30**
- Device: **cuda**
- Objective: pairwise logistic ranking loss with same-role real-player negatives.

## RAG boundary

The two-tower checkpoint is the recommendation model. Chroma/Sentence Transformer retrieval is evaluated
separately and participates only when a user supplies a natural-language preference. Candidate lists are
combined with weighted reciprocal-rank fusion; the LLM scout receives the selected candidate's exact indexed
evidence. The LLM is not the ranker.

## Limitations

- Co-occurrence is implicit feedback, not proof that two players intentionally chose one another.
- Unobserved pairs are sampled negatives, not verified incompatibilities.
- The anonymous finder context limits personalization to role, rank, champion request, and RAG preference.
- A human-rated benchmark is still required for generated scout-report quality.
