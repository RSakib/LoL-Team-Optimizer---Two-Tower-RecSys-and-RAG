# Role Queue Team-Builder Evaluation

The finder occupies one queued role. Each evaluation query asks the system to fill one different role using
profiles computed only from earlier matches. Weights are selected on validation matches; the final test window
is evaluated separately.

- Train matches: **71,290**
- Validation matches: **15,276**
- Test matches: **15,277**
- Successful role-fill test queries: **3,563**
- All observed role-fill test queries: **5,000**

## Successful future lineup retrieval

| Model | NDCG@5 | Recall@5 |
|---|---:|---:|
| team_builder | 0.3724 | 0.5021 |
| champion_affinity | 0.3393 | 0.4681 |
| experience | 0.1070 | 0.1799 |
| performance | 0.0387 | 0.0601 |
| random | 0.0333 | 0.0578 |

## All observed future role fills

| Model | NDCG@5 | Recall@5 |
|---|---:|---:|
| team_builder | 0.3646 | 0.4898 |
| champion_affinity | 0.3357 | 0.4596 |
| experience | 0.1051 | 0.1752 |
| performance | 0.0363 | 0.0554 |
| random | 0.0352 | 0.0598 |

## Paired NDCG@5 comparisons

| Comparison | Delta | 95% CI |
|---|---:|---:|
| team_builder_vs_champion_affinity | 0.0331 | [0.0235, 0.0428] |
| team_builder_vs_experience | 0.2654 | [0.2443, 0.2886] |
| team_builder_vs_random | 0.3391 | [0.3182, 0.3581] |

## Selected validation weights

`{"champion_affinity_score": 0.2767640373484348, "experience_score": 0.12808757860140432, "performance_score": 0.004965775629813836, "preference_score": 0.05, "rank_fit_score": 0.590182608420347}`

## Scope and limitations

- This evaluates ranking real candidates for open role-queue slots.
- Future co-teammates are exposure-based proxy labels; successful-lineup queries additionally require a win.
- The target champion is known query context, while candidate champion history comes only from earlier matches.
- Natural-language preference retrieval requires a separate human-labeled evaluation set.
