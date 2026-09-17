"""No-model contracts for tab-local state and bounded Gradio SSE lifetimes."""
import copy
import json
import socket
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
import gradio as gr
import pytest
import requests
import uvicorn

from src.ui import app as ui
from src.ui.idle import assert_on_demand_ui, valid_snapshot


def test_ui_is_on_demand_without_server_session_state():
    config = ui.demo.get_config_file()
    assert config["connect_heartbeat"] is False
    assert config["analytics_enabled"] is False
    snapshot = next(c for c in config["components"] if c["props"].get("elem_id") == "tab-lineup-snapshot")
    assert snapshot["type"] == "json" and snapshot["props"]["visible"] is False
    assert_on_demand_ui(ui.demo)


@pytest.mark.parametrize("change", ["heartbeat", "state", "timer", "load", "tick", "unload"])
def test_startup_rejects_idle_connection_regressions(change):
    config = copy.deepcopy(ui.demo.get_config_file())
    if change == "heartbeat":
        config["connect_heartbeat"] = True
    elif change in {"state", "timer"}:
        config["components"].append({"type": change})
    else:
        config["dependencies"][0]["targets"] = [[None, change]]
    with pytest.raises(RuntimeError, match="Idle-cost safeguard"):
        assert_on_demand_ui(SimpleNamespace(get_config_file=lambda: config))


@pytest.mark.parametrize("state", [None, [], "bad", {"candidates": []},
    {"response": {"team": []}}, {"response": {"team": {"slots": None}}},
    {"response": {"team": {"slots": ["bad"]}}}, {"candidate_reports": {"id": {}}},
    {"preference": "x" * 1001}, {"response": {"huge": "x" * 1_000_001}}])
def test_malformed_browser_state_never_calls_backend(state, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Malformed display state must not trigger inference")
    monkeypatch.setattr(ui.backend, "post", unexpected)
    assert not valid_snapshot(state)
    assert "Invalid browser" in ui.scout_lineup(state)
    assert "Invalid browser" in ui.scout_candidate("TOP::test-id", state)
    assert list(ui.scout_card(state, gr.EventData(None, {}))) == [(gr.skip(), gr.skip())]


@pytest.fixture
def snapshot():
    # Isolated transport fixtures, not production candidates or training data.
    roles = ["TOP", "JUNGLE", "BOTTOM", "SUPPORT"]
    candidates = [{"summoner_id": "transport-test-" + role, "slot_role": role} for role in roles]
    return {
        "revision": "test-revision", "preference": "vision", "candidate_reports": {},
        "candidates": {c["slot_role"] + "::" + c["summoner_id"]: c for c in candidates},
        "response": {"team": {"complete": True, "finder_primary_role": "MID",
            "suggested_lineup": candidates,
            "slots": [{"role": c["slot_role"], "candidates": [c]} for c in candidates]}},
    }


def test_candidate_payload_never_accepts_browser_supplied_facts(monkeypatch, snapshot):
    calls = []
    key = next(iter(snapshot["candidates"]))
    snapshot["candidates"][key].update({"rag_document": "UNTRUSTED", "win_rate": 1.0, "report": "UNTRUSTED"})
    def post(url, *, json, timeout):
        calls.append(json)
        return SimpleNamespace(ok=True, json=lambda: {"report": "server evidence report"})
    monkeypatch.setattr(ui.backend, "post", post)
    assert "server evidence report" in ui.scout_candidate(key, snapshot)
    assert set(calls[0]) == {"summoner_id", "slot_role", "target_champion", "preference"}
    assert "UNTRUSTED" not in json.dumps(calls)


@pytest.fixture
def live_ui():
    """Actual queued Gradio HTTP server; no recommender/LLM is initialized."""
    demo = ui.build_app().queue(max_size=8, default_concurrency_limit=1)
    app = gr.mount_gradio_app(FastAPI(), demo, path="/")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "Test server failed to start"
    try:
        yield "http://127.0.0.1:" + str(sock.getsockname()[1])
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def invoke(url, config, name, data, event_data=None):
    dependency = next(dep for dep in config["dependencies"] if dep["api_name"] == name)
    session = uuid4().hex  # A new session proves no Python gr.State is required.
    result = requests.post(url + "/gradio_api/queue/join", json={
        "fn_index": dependency["id"], "data": data, "session_hash": session,
        "event_data": event_data,
    }, timeout=10)
    result.raise_for_status()
    messages = []
    started = time.monotonic()
    with requests.get(url + "/gradio_api/queue/data", params={"session_hash": session},
                      stream=True, timeout=(5, 5)) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            assert time.monotonic() - started < 10, "Completed SSE must reach EOF, not stay alive"
            if line.startswith(b"data: "):
                messages.append(json.loads(line[6:]))
    assert messages[-1]["msg"] == "close_stream"
    return next(message for message in reversed(messages) if message["msg"] == "process_completed")


def test_real_queue_closes_for_recommendation_card_lineup_and_error(live_ui, monkeypatch, snapshot):
    calls = []
    def post(url, *, json, timeout):
        calls.append((url, json))
        if url.endswith("/team/recommend"):
            payload = {"status": "no_matches", "message": "No matching real candidates found", "team": {}}
        else:
            payload = {"report": "TEAM report" if url.endswith("/team/scout") else "PLAYER report"}
        return SimpleNamespace(ok=True, json=lambda: payload)
    monkeypatch.setattr(ui.backend, "post", post)
    config = requests.get(live_ui + "/config", timeout=5).json()
    assert config["connect_heartbeat"] is False
    result = invoke(live_ui, config, "recommend_team", ["Fill", "PLATINUM", "IV", 3, 1, "", "", "", "", "", ""])
    assert result["success"], result
    browser_state = result["output"]["data"][1]
    assert browser_state["response"]["status"] == "no_matches"
    key = next(iter(snapshot["candidates"]))
    result = invoke(live_ui, config, "scout_card", [snapshot], {"candidate_key": key, "revision": snapshot["revision"]})
    assert result["success"], result
    saved_state = result["output"]["data"][1]
    assert "PLAYER report" in saved_state["candidate_reports"][key]
    result = invoke(live_ui, config, "scout_lineup_ui", [saved_state])
    assert result["success"] and "TEAM report" in result["output"]["data"][0]
    assert "PLAYER report" not in result["output"]["data"][0]
    assert len(calls) == 3
    # Also check that an unexpected callback exception closes the queue stream.
    def fail(*args, **kwargs):
        raise RuntimeError("Transport test error; no model invoked")
    monkeypatch.setattr(ui.backend, "post", fail)
    result = invoke(live_ui, config, "scout_lineup_ui", [saved_state])
    assert not result["success"]
