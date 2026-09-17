"""Cache public model files in Cloud Build; never load weights into RAM or train."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCOUT_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"
SCOUT_FILES = (
    "LICENSE", "README.md",
    "config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
)
EMBEDDING_FILES = (
    "README.md",
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "modules.json", "sentence_bert_config.json",
    "config_sentence_transformers.json", "1_Pooling/config.json",
)


def preload(cache_root: Path) -> dict:
    from huggingface_hub import snapshot_download

    cache_root.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for repo, files in (
        (SCOUT_REPO, SCOUT_FILES),
        (EMBEDDING_REPO, EMBEDDING_FILES),
    ):
        print(f"[build] Caching {repo}", flush=True)
        snapshot = Path(snapshot_download(
            repo_id=repo, revision="main", cache_dir=str(cache_root / "hub"),
            allow_patterns=list(files), max_workers=2,
        ))
        for name in files:
            path = snapshot / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Incomplete model cache: {repo}/{name}")
        snapshots[repo] = {"revision": snapshot.name, "files": list(files)}
    (cache_root / "build-model-manifest.json").write_text(
        json.dumps(snapshots, indent=2) + "\n", encoding="utf-8",
    )
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cache = os.environ.get("HF_HOME")
    if not cache:
        parser.error("Set HF_HOME to an explicit model cache directory")
    preload(Path(cache))


if __name__ == "__main__":
    main()
