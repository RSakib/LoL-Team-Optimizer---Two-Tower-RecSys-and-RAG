from __future__ import annotations

import html
import os
from typing import Any

import gradio as gr
import requests

from src.config import API_URL


ROLE_LABELS = {
    "Top": "TOP", "Jungle": "JUNGLE", "Mid": "MID",
    "Bottom": "BOTTOM", "Support": "SUPPORT", "Fill": "FILL",
}
DISPLAY_ROLE = {value: key for key, value in ROLE_LABELS.items()}
TIERS = [
    "No rank filter", "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
    "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
]
DIVISIONS = ["Any division", "IV", "III", "II", "I"]


def _safe(value: Any) -> str:
    return html.escape(str(value))


def _role_name(role: str | None) -> str:
    return DISPLAY_ROLE.get(str(role), str(role or "Unknown").title())


def _response_error(response: requests.Response, default: str) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail or default)


def _target_champions(
    primary_role: str,
    top: str,
    jungle: str,
    mid: str,
    bottom: str,
    support: str,
) -> dict[str, str]:
    values = {
        "TOP": top,
        "JUNGLE": jungle,
        "MID": mid,
        "BOTTOM": bottom,
        "SUPPORT": support,
    }
    return {
        role: champion.strip()
        for role, champion in values.items()
        if champion and champion.strip() and (primary_role == "FILL" or role != primary_role)
    }


def _candidate_selector(
    data: dict[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    choices: list[tuple[str, str]] = []
    candidates: dict[str, dict[str, Any]] = {}
    for slot in data.get("team", {}).get("slots", []):
        role = str(slot.get("role", ""))
        for candidate in slot.get("candidates", []):
            player_id = str(candidate.get("summoner_id", ""))
            key = f"{role}::{player_id}"
            name = candidate.get("player_name") or player_id
            selected = " · selected" if candidate.get("selected_for_lineup") else ""
            choices.append((f"{_role_name(role)} — {name}{selected}", key))
            candidates[key] = {
                "summoner_id": player_id,
                "slot_role": role,
                "target_champion": slot.get("target_champion"),
            }
    return choices, candidates


def _render_team(data: dict[str, Any]) -> str:
    status = data.get("status")
    message = data.get("message")
    team = data.get("team") or {}
    lines = ["## Jointly optimized lineup"]

    if status in {"partial", "no_matches"} and message:
        lines.append(f"> **{_safe(message)}**")
    if status == "no_matches":
        return "\n\n".join(lines)

    if team.get("team_fit_score") is not None:
        performance = float(team.get("predicted_performance", 0.0))
        compatibility = float(team.get("compatibility_score", 0.0))
        evaluated = int(team.get("lineups_evaluated", 0))
        lines.extend([
            "| Model performance score¹ | Four-player compatibility² | Complete lineups evaluated |",
            "|---:|---:|---:|",
            f"| **{performance:.1f}%** | **{compatibility:.1f}%** | **{evaluated:,}** |",
            "¹ Learned ranking score, not a calibrated win probability. ² Average learned interaction across the six teammate pairs.",
        ])

    if team.get("requested_primary_role") == "FILL" and team.get("fill_assignment"):
        lines.append(
            f"**Fill assignment:** {_safe(_role_name(team['fill_assignment']))} produced the strongest available lineup."
        )

    finder_role = _role_name(team.get("finder_primary_role"))
    lines.append(
        f"Your reserved role is **{_safe(finder_role)}**. The four first choices below were selected together, not independently."
    )

    selected = team.get("suggested_lineup") or []
    if selected:
        lines.extend([
            "### Selected four-player lineup",
            "| Role | Player | Rank | Recommendation | KDA | Win rate |",
            "|---|---|---|---:|---:|---:|",
        ])
        for candidate in selected:
            rank = " ".join(filter(None, [candidate.get("tier"), candidate.get("rank")])) or "Unavailable"
            name = candidate.get("player_name") or candidate.get("summoner_id")
            lines.append(
                f"| {_safe(_role_name(candidate.get('slot_role')))} | {_safe(name)} | {_safe(rank)} | "
                f"{float(candidate.get('recommendation_score', 0.0)):.1f} | "
                f"{float(candidate.get('kda', 0.0)):.2f} | {float(candidate.get('win_rate', 0.0)):.1%} |"
            )

    pairs = team.get("pair_compatibility") or []
    evidence = team.get("champion_pool_evidence") or {}
    if pairs or evidence:
        lines.append("### Learned team evidence")
        for pair in pairs:
            roles = pair.get("roles") or ["", ""]
            lines.append(
                f"- {_safe(_role_name(roles[0]))} + {_safe(_role_name(roles[1]))}: "
                f"{float(pair.get('model_compatibility', 0.0)):.1f}% learned interaction score"
            )
        champions = evidence.get("distinct_top_champions") or []
        coverage = ", ".join(_safe(champion) for champion in champions) or "Unavailable"
        lines.append(f"- Recorded top-champion coverage: {coverage}")

    lines.append("### Selected teammates and alternatives")
    for slot in team.get("slots", []):
        role_name = _role_name(slot.get("role"))
        target = slot.get("target_champion")
        heading = f"#### {_safe(role_name)}"
        if target:
            heading += f" · requested champion: {_safe(target)}"
        lines.append(heading)
        candidates = slot.get("candidates") or []
        if not candidates:
            lines.append(f"> No matching real candidates found for {_safe(role_name)}.")
            continue
        for index, candidate in enumerate(candidates, start=1):
            name = candidate.get("player_name") or candidate.get("summoner_id")
            selected_label = " — **selected for the joint lineup**" if candidate.get("selected_for_lineup") else ""
            rank = " ".join(filter(None, [candidate.get("tier"), candidate.get("rank")])) or "Unavailable"
            champions = candidate.get("top_champions") or {}
            champion_text = ", ".join(
                f"{_safe(champion)} ({games})" for champion, games in champions.items()
            ) or "Unavailable"
            rag_rank = candidate.get("rag_rank")
            rag_similarity = candidate.get("rag_similarity")
            rag_text = ""
            if rag_rank is not None or rag_similarity is not None:
                rag_text = (
                    f" · RAG rank: **#{rag_rank}**"
                    if rag_rank is not None
                    else " · RAG rank: outside top retrieval"
                )
                if rag_similarity is not None:
                    rag_text += f" · RAG similarity: **{float(rag_similarity):.3f}**"
            lines.extend([
                f"**{index}. {_safe(name)}**{selected_label}",
                f"Rank: **{_safe(rank)}** · Recommendation: **{float(candidate.get('recommendation_score', 0.0)):.1f}** · "
                f"KDA: **{float(candidate.get('kda', 0.0)):.2f}** · Win rate: **{float(candidate.get('win_rate', 0.0)):.1%}** · "
                f"Matches: **{int(candidate.get('matches', 0))}**",
                f"Top champions: {champion_text}",
                f"Two-tower rank: **#{candidate.get('two_tower_rank', 'Unavailable')}** · "
                f"Embedding similarity: **{float(candidate.get('two_tower_similarity', 0.0)):.3f}**{rag_text}",
                f"> {_safe(candidate.get('rag_document', 'No indexed profile document available.'))}",
            ])
    return "\n\n".join(lines)


def recommend_team(
    primary_role_label: str,
    tier_label: str,
    division_label: str,
    candidates_per_role: int,
    max_tier_gap: int,
    preference: str,
    top_champion: str,
    jungle_champion: str,
    mid_champion: str,
    bottom_champion: str,
    support_champion: str,
) -> tuple[str, dict[str, Any], Any, str, str]:
    primary_role = ROLE_LABELS[primary_role_label]
    targets = _target_champions(
        primary_role, top_champion, jungle_champion, mid_champion,
        bottom_champion, support_champion,
    )
    payload = {
        "primary_role": primary_role,
        "target_champions": targets,
        "tier": None if tier_label == "No rank filter" else tier_label,
        "rank": None if division_label == "Any division" else division_label,
        "preference": preference or "",
        "candidates_per_role": int(candidates_per_role),
        "max_tier_gap": int(max_tier_gap),
    }
    try:
        response = requests.post(f"{API_URL}/team/recommend", json=payload, timeout=120)
        if not response.ok:
            return (
                f"## Request failed\n\n{_safe(_response_error(response, response.text or 'The API rejected the request.'))}",
                {}, gr.update(choices=[], value=None), "", "",
            )
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return (
            f"## API connection failed\n\n{_safe(exc)}\n\nConfirm FastAPI is running at `{_safe(API_URL)}`.",
            {}, gr.update(choices=[], value=None), "", "",
        )

    choices, candidates = _candidate_selector(data)
    state = {"response": data, "preference": preference or "", "candidates": candidates}
    return _render_team(data), state, gr.update(choices=choices, value=None), "", ""


def scout_lineup(state: dict[str, Any]) -> str:
    if not state or not state.get("response"):
        return "Build a complete team before requesting a lineup explanation."
    team = state["response"].get("team") or {}
    if not team.get("complete") or not team.get("suggested_lineup"):
        return "A complete real lineup is required before Ollama can explain it."
    lineup = {
        candidate["slot_role"]: candidate["summoner_id"]
        for candidate in team["suggested_lineup"]
    }
    payload = {
        "primary_role": team["finder_primary_role"],
        "lineup": lineup,
        "tier": (team.get("finder") or {}).get("tier"),
        "rank": (team.get("finder") or {}).get("rank"),
        "target_champions": {
            slot["role"]: slot["target_champion"]
            for slot in team.get("slots", []) if slot.get("target_champion")
        },
        "preference": state.get("preference", ""),
    }
    try:
        response = requests.post(f"{API_URL}/team/scout", json=payload, timeout=180)
        if not response.ok:
            return f"**Lineup report unavailable:** {_safe(_response_error(response, response.text))}"
        return "## Grounded lineup explanation\n\n" + response.json()["report"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        return f"**Lineup report failed:** {_safe(exc)}"


def scout_candidate(candidate_key: str | None, state: dict[str, Any]) -> str:
    if not candidate_key or not state:
        return "Choose a real candidate from the recommendation results first."
    candidate = (state.get("candidates") or {}).get(candidate_key)
    if not candidate:
        return "That candidate is no longer present in the current recommendation results."
    payload = {**candidate, "preference": state.get("preference", "")}
    try:
        response = requests.post(f"{API_URL}/scout", json=payload, timeout=180)
        if not response.ok:
            return f"**Candidate report unavailable:** {_safe(_response_error(response, response.text))}"
        return "## Grounded candidate scout report\n\n" + response.json()["report"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        return f"**Candidate report failed:** {_safe(exc)}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LoL joint team recommender") as demo:
        gr.Markdown(
            "# LoL joint team recommender\n"
            "A trained two-tower model retrieves real candidates, then a trained team model scores complete "
            "four-player lineups jointly. Ollama generates RAG-grounded explanations on request."
        )
        result_state = gr.State({})

        with gr.Row():
            primary_role = gr.Dropdown(list(ROLE_LABELS), value="Mid", label="Finder’s primary role")
            tier = gr.Dropdown(TIERS, value="No rank filter", label="Your rank tier")
            division = gr.Dropdown(DIVISIONS, value="Any division", label="Your division")
        with gr.Row():
            candidates_per_role = gr.Slider(1, 5, value=3, step=1, label="Candidates per open role")
            max_tier_gap = gr.Slider(0, 3, value=1, step=1, label="Maximum tier gap")

        preference = gr.Textbox(
            label="Team preference",
            lines=3,
            placeholder="Examples: reliable vision, objective control, experienced frontline players",
        )
        with gr.Accordion("Optional target champions for teammate roles", open=False):
            gr.Markdown("Specify a champion only for a role where recorded champion history is required.")
            with gr.Row():
                top_champion = gr.Textbox(label="Top champion", placeholder="Optional")
                jungle_champion = gr.Textbox(label="Jungle champion", placeholder="Optional")
                mid_champion = gr.Textbox(label="Mid champion", placeholder="Optional")
            with gr.Row():
                bottom_champion = gr.Textbox(label="Bottom champion", placeholder="Optional")
                support_champion = gr.Textbox(label="Support champion", placeholder="Optional")

        build_button = gr.Button("Build my team", variant="primary")
        recommendation_output = gr.Markdown()

        with gr.Tab("Whole-lineup RAG scout"):
            lineup_scout_button = gr.Button("Explain why this lineup complements itself", variant="primary")
            lineup_scout_output = gr.Markdown()
        with gr.Tab("Candidate RAG scout"):
            candidate_selector = gr.Dropdown(label="Recommended candidate", choices=[])
            candidate_scout_button = gr.Button("Generate role-specific scout report")
            candidate_scout_output = gr.Markdown()

        build_button.click(
            recommend_team,
            inputs=[
                primary_role, tier, division, candidates_per_role, max_tier_gap,
                preference, top_champion, jungle_champion, mid_champion,
                bottom_champion, support_champion,
            ],
            outputs=[
                recommendation_output, result_state, candidate_selector,
                lineup_scout_output, candidate_scout_output,
            ],
        )
        lineup_scout_button.click(
            scout_lineup, inputs=[result_state], outputs=[lineup_scout_output]
        )
        candidate_scout_button.click(
            scout_candidate,
            inputs=[candidate_selector, result_state],
            outputs=[candidate_scout_output],
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
