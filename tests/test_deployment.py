"""Offline contract tests; no model downloads or raw match loading."""
from types import SimpleNamespace
import zipfile

import pytest
import torch

from scripts.package_space import ROOT, package
from src.deployment import validate_bundle
from src.llm import local_generator as generator
from src.ui import backend
from src.api import main as api


def test_local_backend_uses_validated_shared_handler(monkeypatch):
    monkeypatch.setattr(backend, "UI_BACKEND", "local")
    captured = []
    monkeypatch.setattr(api, "recommend_team", lambda request: captured.append(request) or {"status": "no_matches"})
    result = backend.post("http://unused/team/recommend", json={"primary_role": "FILL", "tier": "PLATINUM"}, timeout=1)
    assert result.ok and result.json() == {"status": "no_matches"}
    assert isinstance(captured[0], api.TeamRecommendRequest)
    invalid = backend.post("/team/recommend", json={"primary_role": "FILL"}, timeout=1)
    assert invalid.status_code == 422


def test_local_backend_returns_explicit_missing_data_error(monkeypatch):
    monkeypatch.setattr(backend, "UI_BACKEND", "local")

    def unavailable(request):
        raise RuntimeError("Missing real profiles")

    monkeypatch.setattr(api, "scout", unavailable)
    response = backend.post("/scout", json={"summoner_id": "not-present"}, timeout=1)
    assert response.status_code == 503
    assert "Missing real profiles" in response.json()["detail"]


def test_public_request_bounds():
    for extra in ({"preference": "x" * 1001}, {"candidates_per_role": 6}):
        with pytest.raises(ValueError):
            api.TeamRecommendRequest(primary_role="FILL", tier="PLATINUM", **extra)


def test_generator_preserves_evidence_and_decodes_only_new_tokens(monkeypatch):
    captured = {}

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return messages[1]["content"]

        def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return {"input_ids": torch.tensor([[10, 11, 12]])}

        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [20, 21]
            return "report"

    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=32768)

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[10, 11, 12, 20, 21]])

    monkeypatch.setattr(generator, "prepare_generator", lambda: (Tokenizer(), Model()))
    assert generator._generate([{"role": "system", "content": "rules"}, {"role": "user", "content": "exact evidence"}]) == "report"
    assert captured["prompt"] == "exact evidence"
    assert captured["do_sample"] is False
    monkeypatch.setattr(generator, "SCOUT_MAX_INPUT_TOKENS", 2)
    with pytest.raises(RuntimeError, match="no evidence was truncated"):
        generator._generate([{}, {"content": "exact evidence"}])


def test_packaging_refuses_missing_artifacts(tmp_path):
    with pytest.raises(ValueError, match="Missing real deployment artifacts"):
        package(tmp_path, tmp_path / "upload.zip")


@pytest.mark.skipif(not (ROOT / "data/processed/player_profiles.jsonl").exists(), reason="local real artifacts required")
def test_real_bundle_has_no_secrets_or_raw_data_and_validates(tmp_path):
    output = tmp_path / "space.zip"
    manifest = package(ROOT, output)
    assert manifest["real_profile_count"] > 0
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "app.py" in names and "deployment/player_profiles.jsonl" in names
        assert "deployment/rag_embeddings.npz" in names
        assert "assets/fonts/BeaufortforLOL-Bold.ttf" in names
        assert "assets/fonts/NOTICE.md" in names
        assert "assets/league/catalog.json" in names
        assert "assets/league/champions/Ahri.png" in names
        assert "assets/league/champions/Yunara.png" in names
        assert "assets/league/ranks/EMERALD.png" in names
        assert "assets/league/divisions/IV.svg" in names
        assert "assets/league/NOTICE.md" in names
        assert manifest["format_version"] == 3
        assert ".env.example" in names and ".env" not in names
        assert not any(name.startswith(("data/", ".venv/", ".git/", ".chroma/", "tests/")) for name in names)
        archive.extractall(tmp_path / "extracted")
    validate_bundle(tmp_path / "extracted")
    with (tmp_path / "extracted/deployment/player_profiles.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_bundle(tmp_path / "extracted")
