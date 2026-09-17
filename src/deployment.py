"""Integrity validation of the portable, real-data-only deployment artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def artifact_digest(path: Path, normalize_newlines: bool = False) -> str:
    """Hash incrementally; optionally ignore CRLF/LF transport differences only."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if normalize_newlines:
            for line in stream:
                digest.update(line.replace(b"\r\n", b"\n"))
        else:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_bundle(root: Path) -> None:
    manifest_path = root / "deployment/manifest.json"
    if not manifest_path.exists():
        return  # Development checkout uses existing configured local artifacts.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") not in {1, 2, 3} or manifest.get("real_profile_count", 0) <= 0:
        raise RuntimeError("Invalid deployment manifest; rebuild the real-data upload bundle")
    required = {
        "deployment/player_profiles.jsonl", "artifacts/two_tower/model.pt",
        "artifacts/two_tower/metadata.json", "artifacts/team_model/model.pt",
        "artifacts/team_model/metadata.json",
    }
    if manifest["format_version"] >= 3:
        required.add("deployment/rag_embeddings.npz")
    for name in sorted(required):
        expected = manifest.get("files", {}).get(name)
        path = root / name
        if not expected:
            raise RuntimeError(f"Deployment manifest lacks {name}; upload the complete matching bundle")
        if not path.is_file():
            raise RuntimeError(f"Missing deployment artifact: {name}; upload this file at the exact path shown")
        actual_size = path.stat().st_size
        if actual_size == expected["bytes"] and artifact_digest(path) == expected["sha256"]:
            continue
        # Git may normalize Windows text on Linux. Checkpoint bytes stay strict.
        if path.suffix in {".json", ".jsonl"} and expected.get("lf_sha256"):
            if artifact_digest(path, normalize_newlines=True) == expected["lf_sha256"]:
                continue
        raise RuntimeError(
            f"Deployment artifact checksum mismatch or incomplete upload: {name} "
            f"(expected {expected['bytes']} bytes, received {actual_size}). "
            "Re-upload this artifact and deployment/manifest.json from the same newly built ZIP."
        )
