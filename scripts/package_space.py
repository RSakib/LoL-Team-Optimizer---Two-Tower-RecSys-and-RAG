"""Build an allowlisted Space upload ZIP, never a copy of the whole workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deployment import artifact_digest as digest

REQUIRED = (
    "app.py", "README.md", "requirements.txt", ".env.example", ".gitattributes", ".gitignore",
    "artifacts/two_tower/model.pt", "artifacts/two_tower/metadata.json",
    "artifacts/team_model/model.pt", "artifacts/team_model/metadata.json",
    "assets/fonts/BeaufortforLOL-Bold.ttf", "assets/fonts/NOTICE.md",
    "assets/league/catalog.json", "assets/league/NOTICE.md",
)

CLOUD_RUN_FILES = {
    "Dockerfile": "deploy/cloudrun/Dockerfile",
    ".dockerignore": "deploy/cloudrun/.dockerignore",
    ".gcloudignore": "deploy/cloudrun/.gcloudignore",
    "deploy-cloud-run.sh": "deploy/cloudrun/deploy.sh",
    "CLOUD_RUN.md": "docs/cloud_run.md",
}


def artifact_info(path: Path) -> dict:
    info = {"sha256": digest(path), "bytes": path.stat().st_size}
    if path.suffix in {".json", ".jsonl"}:
        info["lf_sha256"] = digest(path, normalize_newlines=True)
    return info


def package(root: Path, output: Path, target: str = "space") -> dict:
    if target not in {"space", "cloud-run"}:
        raise ValueError("Unknown deployment target")
    profiles = root / "data/processed/player_profiles.jsonl"
    if not profiles.exists():
        profiles = root / "deployment/player_profiles.jsonl"
    embeddings = profiles.parent / "rag_embeddings.npz"
    required = [root / name for name in REQUIRED] + [profiles, embeddings]
    if target == "cloud-run":
        required.extend(root / name for name in CLOUD_RUN_FILES.values())
        required.append(root / "scripts/preload_models.py")
    missing = [str(path.relative_to(root)) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("Missing real deployment artifacts: " + ", ".join(missing) +
                         ". Export vectors with `python scripts/export_rag_embeddings.py` before packaging.")
    # Refuse LFS pointer files and mismatched trained checkpoint pairs.
    tower = root / "artifacts/two_tower/model.pt"
    team = root / "artifacts/team_model/model.pt"
    for checkpoint in (tower, team):
        with checkpoint.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs"):
                raise ValueError(f"Download the actual Git LFS checkpoint: {checkpoint.name}")
    metadata = json.loads((root / "artifacts/team_model/metadata.json").read_text(encoding="utf-8"))
    if metadata.get("two_tower_model_sha256") != digest(tower):
        raise ValueError("Team checkpoint is not bound to this two-tower checkpoint")
    ids = set()
    with profiles.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            player_id = row.get("summoner_id")
            if not player_id or not row.get("rag_document") or row.get("role") not in {"TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT"}:
                raise ValueError("Profile lacks an ID, evidence document, or valid recorded role")
            if str(player_id) in ids:
                raise ValueError("Duplicate player ID in real profiles")
            ids.add(str(player_id))
    if not ids:
        raise ValueError("No real profiles to deploy")
    import pandas as pd
    from src.config import RAG_EMBEDDING_MODEL
    from src.rag.portable import load_embeddings
    from src.rag.vector_store import RealPlayerVectorStore
    profile_frame = pd.read_json(profiles, lines=True)
    load_embeddings(
        embeddings, profile_frame, RAG_EMBEDDING_MODEL,
        RealPlayerVectorStore.PIPELINE_VERSION, RealPlayerVectorStore._fingerprint(profile_frame),
    )

    files = {name: root / name for name in REQUIRED}
    # Explicit catalog allowlist: don't ship original asset packs or temp downloads.
    artwork = json.loads((root / "assets/league/catalog.json").read_text(encoding="utf-8"))
    asset_names = [entry["file"] for group in ("champions", "ranks") for entry in artwork[group].values()]
    asset_names.extend(f"divisions/{division}.svg" for division in ("I", "II", "III", "IV"))
    asset_root = (root / "assets/league").resolve()
    for name in asset_names:
        path = asset_root / name
        if (not path.resolve().is_relative_to(asset_root) or path.is_symlink()
                or not path.is_file() or path.suffix not in {".png", ".svg"}):
            raise ValueError(f"Missing or unsafe UI artwork: {name}")
        files["assets/league/" + name] = path
    for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("docs", "*.md"), ("artifacts", "*.json")):
        for path in (root / folder).rglob(pattern):
            if not path.is_symlink():
                files[path.relative_to(root).as_posix()] = path
    files["deployment/player_profiles.jsonl"] = profiles
    files["deployment/rag_embeddings.npz"] = embeddings
    if target == "cloud-run":
        files.update({name: root / source for name, source in CLOUD_RUN_FILES.items()})
    manifest = {
        "format_version": 3, "target": target, "real_profile_count": len(ids),
        "files": {name: artifact_info(path) for name, path in sorted(files.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never silently overwrite a previous upload bundle.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("deployment/manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("space", "cloud-run"), default="space")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "dist" / (
        "lol-cloud-run.zip" if args.target == "cloud-run" else "lol-huggingface-space.zip"
    )
    manifest = package(ROOT, output, target=args.target)
    print(f"Created {output.resolve()} with {manifest['real_profile_count']} real profiles.")
    if args.target == "cloud-run":
        print("Upload this ZIP to Cloud Shell, extract into a new folder, and follow CLOUD_RUN.md.")
    else:
        print("Extract and upload the CONTENTS to a Gradio Space repo root. Do not upload the ZIP as the app.")
    print("Review dataset licensing and public player identifiers before publishing. No .env or raw data is included.")


if __name__ == "__main__":
    main()
