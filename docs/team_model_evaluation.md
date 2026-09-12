# Joint Team Model Evaluation

This second-stage model scores all four recommended players together. Training and evaluation use only
fully observed five-player teams that occurred in the local match logs. No shuffled lineups, synthetic
teams, substituted players, imputed members, or fallback scores are used.

## Temporal protocol

- Complete train teams before eligibility filtering: **591**
- Eligible train teams: **463**
- Eligible validation teams: **65**
- Eligible final-training teams: **531**
- Eligible untouched test teams: **74**
- Selected team-model epochs: **1**
- Player profiles and the provisional two-tower used for validation are built from the training window only.
- The deployment team model is fit on train + validation, then evaluated once on the future test window.

## Future complete-team outcome prediction

| Model | ROC-AUC (95% CI) | Log loss | Brier score | Accuracy |
|---|---:|---:|---:|---:|
| team_model | 0.6447 [0.5224, 0.7799] | 0.6990 | 0.2530 | 0.5270 |
| learned_pair_compatibility | 0.6139 [0.4993, 0.7352] | 0.7012 | 0.2539 | 0.5270 |
| mean_historical_win_rate | 0.5040 [0.3744, 0.6440] | 0.6961 | 0.2514 | 0.5405 |
| constant_prior | 0.5000 [0.5000, 0.5000] | 0.7080 | 0.2572 | 0.5270 |

## Same-match winner ranking

For **10** future matches where both complete real teams are available, the model ranked
the recorded winning lineup above the losing lineup **40.0%** of the time.

## What the model learns

- A performance head sees the complete four-player lineup plus the anonymous finder role/rank.
- A pair-interaction head scores every pair among the four recommended players.
- Inputs include frozen two-tower player embeddings, weighted top-champion-pool embeddings, and role-relative
  experience, win rate, KDA, vision, damage, assists, champion-pool depth, entropy, and concentration.
- Runtime optimization evaluates complete cross-role candidate combinations and selects one jointly.

## Limitations

- The complete-team cohort is small; confidence intervals must be reported with the point estimates.
- The outcome score is not well calibrated on this test cohort and must be treated as a ranking score, not a win probability.
- The 10-match same-match comparison underperforms the historical-win-rate baseline and is too small for a firm conclusion.
- Recorded wins are observational outcomes, not proof that the four players caused the win.
- The finder remains anonymous beyond role/rank, so the score cannot model their personal play style.
- Pair interactions inherit the observed team outcome as supervision; there are no direct pair-compatibility labels.
- The model estimates statistical compatibility from observed outcomes; it does not claim social chemistry.
