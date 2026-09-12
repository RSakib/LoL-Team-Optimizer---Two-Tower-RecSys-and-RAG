import pandas as pd
import torch

from src.recsys.team_model import TeamLineupModel, build_observed_lineups
from src.recsys.two_tower import ROLES


def _real_team() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "match_id": "real-match",
            "timestamp": 1000,
            "team_id": 100,
            "summoner_id": f"player-{role.lower()}",
            "role": role,
            "tier": "GOLD",
            "rank": "II",
            "champion_name": f"Champion{role}",
            "win": True,
        }
        for role in ROLES
    ])


def test_observed_lineups_only_hide_members_of_the_same_real_team():
    events = _real_team()
    ids = set(events["summoner_id"])
    profile_roles = dict(zip(events["summoner_id"], events["role"]))
    examples = build_observed_lineups(events, profile_roles)
    assert len(examples) == 5
    for row in examples.itertuples(index=False):
        selected = {
            getattr(row, f"player_{role}") for role in ROLES if role != row.finder_role
        }
        assert selected == ids.difference({f"player-{row.finder_role.lower()}"})
        assert row.match_id == "real-match"
        assert row.won is True


def test_incomplete_real_team_is_not_imputed():
    events = _real_team().iloc[:-1].copy()
    assert build_observed_lineups(events, set(events["summoner_id"])).empty


def test_team_model_scores_performance_and_all_pair_interactions():
    metadata = {
        "player_feature_dim": 20,
        "hidden_dim": 16,
        "role_vocab_size": 6,
        "tier_vocab_size": 11,
        "division_vocab_size": 5,
    }
    model = TeamLineupModel(metadata)
    player_features = torch.randn(2, 5, 20)
    present = torch.tensor([
        [False, True, True, True, True],
        [True, True, False, True, True],
    ])
    performance, pairs, pair_mask = model(
        player_features,
        present,
        torch.tensor([1, 3]),
        torch.tensor([4, 4]),
        torch.tensor([2, 2]),
    )
    assert performance.shape == (2,)
    assert pairs.shape == (2, 10)
    assert torch.equal(pair_mask.sum(dim=1), torch.tensor([6, 6]))
