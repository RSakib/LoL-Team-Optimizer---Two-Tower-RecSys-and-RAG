# Joint Team Model Evaluation

The former single-window evaluation has been removed because it was superseded after that test window was inspected
during development. Keeping its figures beside the current benchmark would present an obsolete generalization estimate.

The current and only reported joint-team evaluation is the four-fold rolling temporal benchmark:

- [Latest joint-team benchmark](team_benchmark.md)
- [Calibration plot](team_calibration.png)

Run `python -m src.evaluation.team_benchmark --device cuda` to regenerate the current report from the compact real-match
history. No synthetic teams, substituted players, imputed members, or fallback scores are used.
