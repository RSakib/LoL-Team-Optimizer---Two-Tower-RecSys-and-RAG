"""On-Space inference: no hosted inference API, credentials, or invented fallback."""
from __future__ import annotations

import threading
from time import perf_counter
from typing import Any

from src.config import SCOUT_DEVICE, SCOUT_MODEL, SCOUT_MAX_INPUT_TOKENS, SCOUT_MAX_NEW_TOKENS

_tokenizer: Any = None
_model: Any = None
_lock = threading.RLock()


def prepare_generator() -> tuple[Any, Any]:
    """Lazy on CPU; called during Space startup for ZeroGPU CUDA emulation."""
    global _tokenizer, _model
    with _lock:
        if _model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if SCOUT_DEVICE not in {"cpu", "zero_gpu"}:
                raise RuntimeError("SCOUT_DEVICE must be cpu or zero_gpu")
            if not SCOUT_MODEL:
                raise RuntimeError("SCOUT_MODEL is required")
            tokenizer = AutoTokenizer.from_pretrained(SCOUT_MODEL, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(
                SCOUT_MODEL, trust_remote_code=False,
                dtype=torch.float16 if SCOUT_DEVICE == "zero_gpu" else torch.float32,
            )
            model = model.to("cuda" if SCOUT_DEVICE == "zero_gpu" else "cpu").eval()
            _tokenizer, _model = tokenizer, model
        return _tokenizer, _model


def _generate(messages: list[dict[str, str]]) -> str:
    import torch

    # The Gradio queue serializes calls; the lock also protects standalone API use.
    with _lock:
        started = perf_counter()
        tokenizer, model = prepare_generator()
        load_seconds = perf_counter() - started
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_length = int(inputs["input_ids"].shape[-1])
        context_limit = int(getattr(model.config, "max_position_embeddings", 32768))
        if SCOUT_MAX_NEW_TOKENS < 1 or SCOUT_MAX_INPUT_TOKENS < 1:
            raise RuntimeError("Scout token limits must be positive")
        if input_length > min(SCOUT_MAX_INPUT_TOKENS, context_limit - SCOUT_MAX_NEW_TOKENS):
            raise RuntimeError("Retrieved evidence exceeds the scout context budget; no evidence was truncated")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generation_started = perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=SCOUT_MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(
            f"[scout] device={SCOUT_DEVICE} load={load_seconds:.2f}s "
            f"input_tokens={input_length} max_new_tokens={SCOUT_MAX_NEW_TOKENS} "
            f"generation={perf_counter() - generation_started:.2f}s",
            flush=True,
        )
        return tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()


if SCOUT_DEVICE == "zero_gpu":
    # app.py imports spaces before any torch imports, as required by ZeroGPU.
    import spaces
    generate_text = spaces.GPU(duration=60)(_generate)
else:
    generate_text = _generate
