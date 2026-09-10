import numpy as np
import pandas as pd

from src.evaluation.events import temporal_three_way_split
from src.evaluation.role_queue import _weight_candidates, ranking_metrics


def test_three_way_temporal_split_is_disjoint_and_ordered():
    events = pd.DataFrame([
        {"match_id": f"m{i}", "timestamp": i, "summoner_id": "a", "team_id": 100, "win": True}
        for i in range(20)
    ])
    train, validation, test, validation_cutoff, test_cutoff = temporal_three_way_split(events, 0.7, 0.15)
    assert train["timestamp"].max() < validation_cutoff <= validation["timestamp"].min()
    assert validation["timestamp"].max() < test_cutoff <= test["timestamp"].min()
    assert set(train["match_id"]).isdisjoint(validation["match_id"])
    assert set(train["match_id"]).isdisjoint(test["match_id"])
    assert set(validation["match_id"]).isdisjoint(test["match_id"])


def test_ranking_metrics_reward_relevant_role_candidate():
    recall, ndcg = ranking_metrics(np.array([0.1, 0.9, 0.2]), ["top-a", "top-b", "top-c"], {"top-b"}, 1)
    assert recall == 1.0
    assert ndcg == 1.0


def test_role_queue_weight_candidates_are_convex():
    weights = _weight_candidates(32, seed=42)
    assert weights.shape == (32, 4)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
