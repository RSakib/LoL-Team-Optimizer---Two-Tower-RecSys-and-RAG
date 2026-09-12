from __future__ import annotations

import json
import os
from typing import Any

from src.config import OPENAI_MODEL


class ScoutConfigurationError(RuntimeError):
    pass


def generate_scout_report(candidate: dict[str, Any], preference: str, model: str = OPENAI_MODEL) -> str:
    """Generate a report from supplied real facts; never invent a candidate or statistic."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ScoutConfigurationError("OPENAI_API_KEY is not configured; no scout report was generated")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ScoutConfigurationError("The openai package is not installed") from exc
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
    prompt = f"User preference: {preference or 'No additional preference supplied.'}\nCandidate facts:\n{json.dumps(facts, ensure_ascii=False)}"
    response = OpenAI().responses.create(model=model, instructions=instructions, input=prompt, store=False)
    if not response.output_text:
        raise RuntimeError("The LLM returned no scout-report text")
    return response.output_text
