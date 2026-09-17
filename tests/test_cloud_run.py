"""Offline Cloud Run contracts; never download models, train, or contact Google."""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from scripts.package_space import ROOT, package
from scripts import preload_models
from src.deployment import validate_bundle


def test_cloud_run_honors_port_without_loading_models(monkeypatch):
    from src import config, deployment, service
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7860")
    monkeypatch.setattr(config, "SCOUT_DEVICE", "cpu")
    monkeypatch.setattr(config, "UI_BACKEND", "local")
    monkeypatch.setattr(deployment, "validate_bundle", lambda root: None)
    monkeypatch.setattr(service, "warm_runtime", lambda: None)
    captured = {}

    class Demo:
        def queue(self, **kwargs):
            captured["queue"] = kwargs
            return self

        def launch(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "src.ui.app", SimpleNamespace(demo=Demo()))
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    assert captured["server_port"] == 8080
    assert captured["queue"]["default_concurrency_limit"] == 1


def test_model_download_is_file_only_and_records_revisions(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        # Files are downloader test doubles, never used as player/model evidence.
        snapshot = tmp_path / kwargs["repo_id"].split("/")[-1] / "snapshots" / "test-commit"
        for filename in kwargs["allow_patterns"]:
            path = snapshot / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"download-contract-test")
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    manifest = preload_models.preload(tmp_path / "cache")
    assert len(calls) == 2
    assert all(call["max_workers"] == 2 for call in calls)
    assert all("*.bin" not in call["allow_patterns"] for call in calls)
    assert manifest[preload_models.SCOUT_REPO]["revision"] == "test-commit"
    assert json.loads((tmp_path / "cache/build-model-manifest.json").read_text()) == manifest


def test_model_download_fails_closed_on_missing_file(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kw: str(tmp_path)))
    with pytest.raises(RuntimeError, match="Incomplete model cache"):
        preload_models.preload(tmp_path / "cache")
    assert not (tmp_path / "cache/build-model-manifest.json").exists()


def test_cloud_run_cost_and_upload_safety_configuration():
    deploy = (ROOT / "deploy/cloudrun/deploy.sh").read_text()
    for flag in ("--min 0", "--max 1", "--min-instances 0", "--max-instances 1",
                 "--cpu-throttling", "--no-allow-unauthenticated", "--concurrency 8"):
        assert flag in deploy
    assert "--dry-run" in deploy and '"DEPLOY"' in deploy
    for name in (".dockerignore", ".gcloudignore"):
        ignore = (ROOT / "deploy/cloudrun" / name).read_text()
        assert "!deployment/**" in ignore and "!artifacts/**" in ignore
        assert "**/.env" in ignore
        assert "#!include:.gitignore" not in ignore
        assert "!data/" not in ignore and "!.venv/" not in ignore
    docker = (ROOT / "deploy/cloudrun/Dockerfile").read_text()
    assert "https://download.pytorch.org/whl/cpu" in docker
    assert "USER appuser" in docker
    assert "HF_HUB_OFFLINE=1" in docker
    assert "COPY . ." not in docker


@pytest.mark.skipif(not (ROOT / "data/processed/player_profiles.jsonl").exists(), reason="real local artifacts required")
def test_cloud_run_bundle_preserves_real_artifacts_and_excludes_secrets(tmp_path):
    output = tmp_path / "cloud.zip"
    manifest = package(ROOT, output, target="cloud-run")
    assert manifest["target"] == "cloud-run" and manifest["real_profile_count"] > 0
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        for name in ("Dockerfile", ".dockerignore", ".gcloudignore", "deploy-cloud-run.sh", "CLOUD_RUN.md", "scripts/preload_models.py"):
            assert name in names
        assert ".env" not in names
        assert not any(name.startswith(("data/", ".venv/", ".git/", "tmp/")) for name in names)
        assert b"\r\n" not in archive.read("deploy-cloud-run.sh")
        archive.extractall(tmp_path / "extracted")
    validate_bundle(tmp_path / "extracted")
    with pytest.raises(FileExistsError):
        package(ROOT, output, target="cloud-run")


def test_unknown_deployment_target_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown deployment target"):
        package(tmp_path, tmp_path / "x.zip", target="unexpected")
