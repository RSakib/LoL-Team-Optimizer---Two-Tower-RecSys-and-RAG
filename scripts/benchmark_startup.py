"""Time cold vector import, startup warmup, and two real requests; no LLM generation."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True, help="A fresh directory for the diagnostic Chroma index")
    args = parser.parse_args()
    if args.index_dir.exists():
        raise SystemExit("Choose a fresh diagnostic index directory; existing indexes will not be overwritten")
    os.environ["LOL_CHROMA_DIR"] = str(args.index_dir.resolve())
    os.environ["LOL_TWO_TOWER_DEVICE"] = "cpu"
    os.environ["RAG_EMBEDDING_DEVICE"] = "cpu"
    os.environ["RAG_EMBEDDING_BATCH_SIZE"] = "16"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    started = perf_counter()
    from src.service import warm_runtime
    from src.api.main import recommend_team, TeamRecommendRequest
    from src.rag.vector_store import RealPlayerVectorStore

    def forbid_encoding(*args, **kwargs):
        raise RuntimeError("Unexpected document encoding: the benchmark requires exported real vectors")

    RealPlayerVectorStore._encode_documents = forbid_encoding
    runtime = warm_runtime()
    startup_seconds = perf_counter() - started
    request = TeamRecommendRequest(primary_role="FILL", tier="PLATINUM", rank="IV", preference="reliable vision and team-oriented setup")
    timings = []
    for _ in range(2):
        started = perf_counter()
        result = recommend_team(request)
        timings.append(perf_counter() - started)
    print(json.dumps({
        "profiles": len(runtime.profiles), "document_encoding_calls": 0,
        "startup_including_imports_seconds": round(startup_seconds, 3),
        "first_request_seconds": round(timings[0], 3), "repeat_request_seconds": round(timings[1], 3),
        "result_status": result["status"], "selected_players": len(result["team"]["suggested_lineup"]),
        "note": "Local CPU measurement, not Hugging Face latency. No scout model loaded.",
    }, indent=2))


if __name__ == "__main__":
    main()
