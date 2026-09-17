"""Gradio entry point for Hugging Face, Cloud Run, and local development."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "deployment/player_profiles.jsonl").exists():
    os.environ.setdefault("LOL_PROFILE_PATH", str(ROOT / "deployment/player_profiles.jsonl"))

# Keep retrieval and ranking off the temporary ZeroGPU device and bound CPU work.
os.environ.setdefault("LOL_TWO_TOWER_DEVICE", "cpu")
os.environ.setdefault("RAG_EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("RAG_EMBEDDING_BATCH_SIZE", "16")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

if os.getenv("SPACES_ZERO_GPU") == "1":
    import spaces  # Must precede torch and any transitive torch imports.

from src.config import SCOUT_DEVICE, UI_BACKEND
from src.deployment import validate_bundle
from src.llm.local_generator import prepare_generator
from src.ui.app import demo
from src.ui.theme import THEME, GLOBAL_CSS, FONT_HEAD

validate_bundle(ROOT)

if SCOUT_DEVICE == "zero_gpu":
    prepare_generator()

if __name__ == "__main__":
    if UI_BACKEND == "local":
        from src.service import warm_runtime
        warm_runtime()
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        theme=THEME, css=GLOBAL_CSS, head=FONT_HEAD,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))),
        show_error=False,
    )
