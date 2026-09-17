"""Transport checks use existing real artifacts, never model downloads."""
import json
import zipfile

import pytest

from scripts.package_space import ROOT, package
from src.deployment import validate_bundle


@pytest.fixture
def bundle(tmp_path):
    if not (ROOT / "data/processed/player_profiles.jsonl").exists():
        pytest.skip("local real artifacts required")
    archive_path = tmp_path / "space.zip"
    package(ROOT, archive_path)
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    return destination


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_bundle_accepts_only_newline_transport_changes(bundle, line_ending):
    for name in (
        "artifacts/two_tower/metadata.json", "artifacts/team_model/metadata.json",
        "deployment/player_profiles.jsonl",
    ):
        path = bundle / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", line_ending))
    validate_bundle(bundle)


def test_bundle_rejects_changed_json_values(bundle):
    path = bundle / "artifacts/team_model/metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["hidden_dim"] += 1
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_bundle(bundle)


def test_bundle_rejects_same_size_checkpoint_corruption(bundle):
    path = bundle / "artifacts/team_model/model.pt"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_bundle(bundle)


def test_bundle_identifies_missing_file_separately(bundle):
    path = bundle / "artifacts/two_tower/metadata.json"
    path.rename(path.with_suffix(".omitted"))
    with pytest.raises(RuntimeError, match="Missing deployment artifact: artifacts/two_tower/metadata.json"):
        validate_bundle(bundle)
