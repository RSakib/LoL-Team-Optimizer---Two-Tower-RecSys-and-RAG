from __future__ import annotations

import argparse

from src.config import DATA_DIR
from src.data.preprocess_large import main as streaming_main


def main() -> None:
    streaming_main()


if __name__ == "__main__":
    main()
