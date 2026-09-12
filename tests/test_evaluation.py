import pandas as pd

from src.evaluation.events import temporal_three_way_split
from src.recsys.train import build_positive_pairs


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


def test_positive_pairs_only_join_real_same_team_different_role_players():
    events = pd.DataFrame([
        {"match_id": "m1", "team_id": 100, "summoner_id": "mid", "role": "MID", "tier": "GOLD", "rank": "II", "champion_name": "Ahri", "win": True},
        {"match_id": "m1", "team_id": 100, "summoner_id": "jungle", "role": "JUNGLE", "tier": "GOLD", "rank": "II", "champion_name": "Vi", "win": True},
        {"match_id": "m1", "team_id": 200, "summoner_id": "enemy", "role": "TOP", "tier": "GOLD", "rank": "II", "champion_name": "Garen", "win": False},
    ])
    pairs = build_positive_pairs(events, {"mid", "jungle", "enemy"})
    assert set(zip(pairs["finder_id"], pairs["candidate_id"])) == {
        ("mid", "jungle"), ("jungle", "mid"),
    }
