import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.rag_human import score_reviews
from src.evaluation.team_benchmark import expected_calibration_error, rolling_temporal_folds


def test_rolling_temporal_folds_have_non_overlapping_future_tests():
    events = pd.DataFrame({
        "match_id": [f"match-{index:03d}" for index in range(100)],
        "timestamp": np.arange(100, dtype=np.int64),
    })
    folds = rolling_temporal_folds(events, folds=4)
    test_ids = [set(fold.test["match_id"]) for fold in folds]
    assert all(test_ids[left].isdisjoint(test_ids[right]) for left in range(4) for right in range(left + 1, 4))
    for fold in folds:
        assert fold.train["timestamp"].max() < fold.validation["timestamp"].min()
        assert fold.validation["timestamp"].max() < fold.test["timestamp"].min()


def test_expected_calibration_error_is_zero_for_exact_bin_rates():
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(labels, scores, bins=10) == 0.0


def test_human_rag_scores_refuse_blank_generated_reports(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"case_id": "real-001", "generated_report": ""}) + "\n", encoding="utf-8")
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("case_id,reviewer_id,grounding,tactical_usefulness,preference_alignment,limitation_clarity,unsupported_claim_count\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="generated report"):
        score_reviews(cases, ratings)
