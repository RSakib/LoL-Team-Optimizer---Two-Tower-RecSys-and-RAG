from __future__ import annotations

import json
from typing import Any

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS


class ScoutConfigurationError(RuntimeError):
    pass


def scout_generation_status() -> dict[str, Any]:
    """Describe the configured generator without making a network request."""
    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "configured": bool(OLLAMA_MODEL and OLLAMA_BASE_URL),
    }


def _generate_grounded_text(instructions: str, prompt: str, model: str | None = None) -> str:
    selected_model = model or OLLAMA_MODEL
    if not selected_model:
        raise ScoutConfigurationError("OLLAMA_MODEL is not configured; no scout report was generated")
    if not OLLAMA_BASE_URL:
        raise ScoutConfigurationError("OLLAMA_BASE_URL is not configured; no scout report was generated")

    request = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=request,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ScoutConfigurationError(
            f"Ollama scout generation failed for model '{selected_model}'. "
            f"Confirm Ollama is running and run `ollama pull {selected_model}` once."
        ) from exc

    message = payload.get("message") if isinstance(payload, dict) else None
    text = str(message.get("content", "") if isinstance(message, dict) else "").strip()
    if not text:
        raise ScoutConfigurationError("Ollama returned no scout-report text")
    return text


def generate_scout_report(candidate: dict[str, Any], preference: str, model: str | None = None) -> str:
    """Generate a report from supplied real facts; never invent a candidate or statistic."""
    facts = {key: candidate.get(key) for key in (
        "summoner_id", "player_name", "tier", "rank", "role", "matches", "win_rate", "kda",
        "avg_kills", "avg_deaths", "avg_assists", "avg_vision_score", "avg_gold_earned",
        "avg_damage_dealt", "top_champions", "slot_role", "target_champion", "rag_document",
        "retrieval_metadata",
    )}
    instructions = (
        "You are a League of Legends tactical scout. Use only the provided JSON facts. "
        "Do not infer unrecorded play style, champion skill, personality, availability, or causality. "
        "Assess the candidate only for the requested role-queue slot and target champion context. "
        "Clearly label limited sample sizes and unavailable facts. Give concise strengths, risks, and team-fit tactics."
    )
    prompt = (
        f"User preference: {preference or 'No additional preference supplied.'}\n"
        f"Candidate facts:\n{json.dumps(facts, ensure_ascii=False)}"
    )
    return _generate_grounded_text(instructions, prompt, model)


def generate_lineup_scout_report(
    lineup_facts: dict[str, Any], preference: str, model: str | None = None
) -> str:
    """Explain a joint lineup using only exact retrieved profiles and labeled model estimates."""
    instructions = (
        "You are a League of Legends lineup analyst. Use only the supplied JSON evidence. "
        "The retrieved profile documents and champion pools are recorded facts. Performance, overall compatibility, "
        "and pair compatibility are uncalibrated ranking scores, not probabilities, facts, or causal guarantees. "
        "Explain concrete complementary strengths across the four roles, champion-pool coverage, and risks. Never "
        "infer personality, communication, availability, unrecorded champion skill, damage type, champion class, or "
        "social chemistry. Do not claim the lineup will win. If evidence does not support a claimed complement, "
        "state that it is unavailable."
    )
    prompt = (
        f"User preference: {preference or 'No additional preference supplied.'}\n"
        f"Joint lineup evidence:\n{json.dumps(lineup_facts, ensure_ascii=False)}"
    )
    return _generate_grounded_text(instructions, prompt, model)
