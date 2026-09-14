from __future__ import annotations

from typing import Any

from src.ui import app as ui


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def test_recommendation_uses_role_queue_contract_and_real_empty_state(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    api_result = {
        "status": "no_matches",
        "message": "No matching real candidates found for any open role",
        "candidate_count": 0,
        "team": {"slots": []},
    }

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> FakeResponse:
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(api_result)

    monkeypatch.setattr(ui.requests, "post", fake_post)
    rendered, state, selector, lineup_report, candidate_report = ui.recommend_team(
        "Mid", "GOLD", "II", 3, 1, "objective control",
        "Ornn", "Vi", "Ahri", "Jinx", "Nautilus",
    )

    assert captured["url"].endswith("/team/recommend")
    assert captured["json"] == {
        "primary_role": "MID",
        "target_champions": {
            "TOP": "Ornn",
            "JUNGLE": "Vi",
            "BOTTOM": "Jinx",
            "SUPPORT": "Nautilus",
        },
        "tier": "GOLD",
        "rank": "II",
        "preference": "objective control",
        "candidates_per_role": 3,
        "max_tier_gap": 1,
    }
    assert "No matching real candidates found" in rendered
    assert state["response"] == api_result
    assert selector["choices"] == []
    assert lineup_report == candidate_report == ""


def test_candidate_scout_forwards_only_selected_real_candidate(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> FakeResponse:
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"report": "Grounded report"})

    monkeypatch.setattr(ui.requests, "post", fake_post)
    state = {
        "preference": "vision control",
        "candidates": {
            "SUPPORT::real-player-id": {
                "summoner_id": "real-player-id",
                "slot_role": "SUPPORT",
                "target_champion": "Nautilus",
            }
        },
    }

    report = ui.scout_candidate("SUPPORT::real-player-id", state)

    assert captured["url"].endswith("/scout")
    assert captured["json"] == {
        "summoner_id": "real-player-id",
        "slot_role": "SUPPORT",
        "target_champion": "Nautilus",
        "preference": "vision control",
    }
    assert "Grounded report" in report


def test_lineup_scout_refuses_partial_lineup_without_api_call(monkeypatch) -> None:
    def unexpected_post(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("partial lineups must not be sent to the scout endpoint")

    monkeypatch.setattr(ui.requests, "post", unexpected_post)
    state = {"response": {"team": {"complete": False, "suggested_lineup": []}}}

    assert "complete real lineup is required" in ui.scout_lineup(state)
