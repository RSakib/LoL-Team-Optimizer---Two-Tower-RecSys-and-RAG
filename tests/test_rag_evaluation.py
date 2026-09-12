import pandas as pd

from src.evaluation.rag import build_retrieval_cases, ranking_metrics


def test_ranking_metrics_reward_relevant_retrieval():
    metrics = ranking_metrics(["a", "b", "c"], {"a", "c"}, k=3)
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_labels_reference_only_real_profile_ids():
    rows = []
    for index in range(10):
        rows.append({
            "summoner_id": f"player-{index}",
            "role": "MID",
            "matches": index + 1,
            "avg_vision_score": index,
            "avg_damage_dealt": index * 100,
            "kda": index / 2,
            "win_rate": index / 10,
            "avg_assists": index,
            "top_champions": {"Ahri": index + 1},
        })
    profiles = pd.DataFrame(rows)
    cases = build_retrieval_cases(profiles)
    known = set(profiles["summoner_id"])
    assert cases
    assert all(case["positives"] <= known for case in cases)
