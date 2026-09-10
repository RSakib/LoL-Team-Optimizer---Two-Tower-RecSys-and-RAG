from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.recsys.engine import ROLES, TeamBuilderEngine


def _profiles():
    rows = []
    for role in ROLES:
        rows.append({
            "summoner_id": f"player-{role.lower()}", "puuid": f"puuid-{role.lower()}",
            "player_name": f"Player {role}", "tier": "GOLD", "rank": "I", "role": role,
            "matches": 20, "wins": 12, "win_rate": .6, "kda": 3.0,
            "avg_kills": 5.0, "avg_deaths": 4.0, "avg_assists": 7.0,
            "avg_vision_score": 20.0, "avg_gold_earned": 11000.0,
            "avg_damage_dealt": 18000.0, "top_champions": {"Lulu": 10},
            "rag_document": "real profile",
        })
    return pd.DataFrame(rows)


class _Store:
    def search_teammates(self, query, top_k):
        return []


def test_team_endpoint_returns_four_open_role_slots(monkeypatch):
    profiles = _profiles()
    runtime = SimpleNamespace(profiles=profiles, vector_store=_Store(), recommender=TeamBuilderEngine(profiles))
    monkeypatch.setattr(api_module, "get_runtime", lambda: runtime)
    response = TestClient(api_module.app).post(
        "/team/recommend",
        json={"finder": "player-mid", "primary_role": "MID", "candidates_per_role": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert len(payload["team"]["slots"]) == 4
    assert {slot["role"] for slot in payload["team"]["slots"]} == {"TOP", "JUNGLE", "BOTTOM", "SUPPORT"}


def test_team_endpoint_accepts_fill(monkeypatch):
    profiles = _profiles()
    runtime = SimpleNamespace(profiles=profiles, vector_store=_Store(), recommender=TeamBuilderEngine(profiles))
    monkeypatch.setattr(api_module, "get_runtime", lambda: runtime)
    response = TestClient(api_module.app).post(
        "/team/recommend",
        json={"finder": "player-mid", "primary_role": "FILL", "candidates_per_role": 1},
    )
    assert response.status_code == 200
    assert response.json()["team"]["fill_assignment"] in ROLES
