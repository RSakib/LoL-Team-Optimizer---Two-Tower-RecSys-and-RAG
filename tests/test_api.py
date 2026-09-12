from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.recsys.two_tower import ROLES


def _profiles() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "summoner_id": f"player-{role.lower()}",
            "player_name": f"Player {role}",
            "tier": "GOLD",
            "rank": "I",
            "role": role,
            "rag_document": "real profile",
        }
        for role in ROLES
    ])


class _Store:
    embedding_model_name = "test-local-embedding"

    class collection:
        @staticmethod
        def count():
            return 5

    def search_teammates(self, query, top_k, role=None):
        return []

    def get_profile_document(self, summoner_id):
        return {
            "summoner_id": summoner_id,
            "rag_document": "indexed real profile",
            "metadata": {"role": "MID"},
        }


class _Recommender:
    loaded = SimpleNamespace(device="cpu", metadata={"training": {"development_pairs": 10}})

    def recommend_team(self, finder_primary_role, **kwargs):
        assigned = "MID" if finder_primary_role == "FILL" else finder_primary_role
        roles = [role for role in ROLES if role != assigned]
        slots = [
            {
                "role": role,
                "target_champion": None,
                "candidates": [{"summoner_id": f"player-{role.lower()}", "recommendation_score": 80.0}],
            }
            for role in roles
        ]
        return {
            "finder": {"primary_role": assigned, "tier": kwargs.get("tier"), "rank": kwargs.get("division")},
            "finder_primary_role": assigned,
            "requested_primary_role": finder_primary_role,
            "fill_assignment": assigned if finder_primary_role == "FILL" else None,
            "fill_scenarios": [],
            "slots": slots,
            "suggested_lineup": [slot["candidates"][0] for slot in slots],
            "complete": True,
            "missing_roles": [],
            "team_fit_score": 80.0,
        }


def _runtime():
    return SimpleNamespace(profiles=_profiles(), vector_store=_Store(), recommender=_Recommender())


def test_team_endpoint_returns_four_open_role_slots(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    response = TestClient(api_module.app).post(
        "/team/recommend",
        json={"primary_role": "MID", "candidates_per_role": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert len(payload["team"]["slots"]) == 4
    assert {slot["role"] for slot in payload["team"]["slots"]} == {
        "TOP", "JUNGLE", "BOTTOM", "SUPPORT",
    }


def test_team_endpoint_accepts_fill(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    response = TestClient(api_module.app).post(
        "/team/recommend",
        json={"primary_role": "FILL", "candidates_per_role": 1},
    )
    assert response.status_code == 200
    assert response.json()["team"]["fill_assignment"] in ROLES


def test_scout_uses_exact_chroma_evidence(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    captured = {}

    def fake_report(candidate, preference):
        captured.update(candidate)
        return "grounded report"

    monkeypatch.setattr(api_module, "generate_scout_report", fake_report)
    response = TestClient(api_module.app).post(
        "/scout",
        json={"summoner_id": "player-mid", "slot_role": "MID", "preference": "vision"},
    )
    assert response.status_code == 200
    assert captured["rag_document"] == "indexed real profile"
    assert response.json()["grounded_profile"] == "indexed real profile"
