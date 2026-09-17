"""Export the current real Chroma vectors without downloading or running a model."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CHROMA_DIR, PROFILE_PATH, PROCESSED_DIR
from src.rag.portable import export_existing_index
from src.rag.vector_store import RealPlayerVectorStore
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "rag_embeddings.npz")
    args = parser.parse_args()
    profiles = pd.read_json(PROFILE_PATH, lines=True)
    count = export_existing_index(RealPlayerVectorStore(CHROMA_DIR), profiles, args.output)
    print(f"Exported {count} existing real-profile vectors to {args.output}. No profiles were re-embedded.")


if __name__ == "__main__":
    main()
