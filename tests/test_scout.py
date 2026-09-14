import requests
import pytest

import src.llm.scout as scout


class _FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {"message": {"content": "grounded local report"}}
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


def test_ollama_generates_from_grounded_facts_without_api_credit(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(scout.requests, "post", fake_post)
    monkeypatch.setattr(scout, "OLLAMA_MODEL", "gemma3:4b")
    monkeypatch.setattr(scout, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(scout, "OLLAMA_TIMEOUT_SECONDS", 180.0)

    report = scout.generate_scout_report(
        {"summoner_id": "real-player", "rag_document": "real retrieved evidence"},
        "vision-focused teammate",
    )

    assert report == "grounded local report"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["json"]["model"] == "gemma3:4b"
    assert captured["json"]["stream"] is False
    assert "real retrieved evidence" in captured["json"]["messages"][1]["content"]


def test_offline_ollama_returns_actionable_configuration_error(monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(scout.requests, "post", offline)
    monkeypatch.setattr(scout, "OLLAMA_MODEL", "gemma3:4b")
    monkeypatch.setattr(scout, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    with pytest.raises(scout.ScoutConfigurationError, match="ollama pull gemma3:4b"):
        scout.generate_lineup_scout_report({"retrieved_real_profiles": []}, "")


def test_ollama_refuses_an_empty_response(monkeypatch):
    monkeypatch.setattr(
        scout.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse({"message": {"content": ""}}),
    )
    with pytest.raises(scout.ScoutConfigurationError, match="no scout-report text"):
        scout.generate_scout_report({"summoner_id": "real-player"}, "")
