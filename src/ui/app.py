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
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD",
    "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
]
DIVISIONS = ["Any division", "IV", "III", "II", "I"]

APP_CSS = """
.gradio-container {
    max-width: 1480px !important;
}
#team-results {
    margin-top: 1.25rem;
}
.team-results {
    color: var(--body-text-color);
    font-family: var(--font);
}
.results-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
}
.results-kicker, .role-kicker {
    color: #c89b3c;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.results-header h2, .role-heading h3 {
    margin: .2rem 0 0;
    line-height: 1.1;
}
.results-note {
    color: var(--body-text-color-subdued);
    font-size: .9rem;
    margin: 0;
}
.result-alert {
    border: 1px solid #c89b3c66;
    border-radius: 12px;
    background: #c89b3c14;
    padding: .85rem 1rem;
    margin: .75rem 0 1rem;
}
.team-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .75rem;
    margin-bottom: 1rem;
}
.team-metric {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    background: var(--background-fill-secondary);
    padding: 1rem;
}
.team-metric span, .player-metric span {
    color: var(--body-text-color-subdued);
    display: block;
    font-size: .75rem;
    margin-bottom: .2rem;
}
.team-metric strong {
    color: #c89b3c;
    font-size: 1.55rem;
}
.score-disclaimer {
    color: var(--body-text-color-subdued);
    font-size: .78rem;
    margin: -.35rem 0 1.25rem;
}
.selected-lineup, .candidate-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
    gap: .85rem;
}
.selected-lineup {
    margin: .75rem 0 1.4rem;
}
.lineup-chip {
    border: 1px solid #c89b3c55;
    border-radius: 12px;
    background: linear-gradient(135deg, #c89b3c18, transparent);
    padding: .85rem;
}
.lineup-chip-role {
    color: #c89b3c;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
}
.lineup-chip-name {
    display: block;
    font-size: 1rem;
    margin: .22rem 0;
}
.lineup-chip-meta {
    color: var(--body-text-color-subdued);
    font-size: .8rem;
}
.team-evidence {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    margin: 0 0 1.4rem;
    padding: .85rem 1rem;
}
.team-evidence summary, .model-evidence summary {
    cursor: pointer;
    font-weight: 700;
}
.team-evidence ul {
    margin-bottom: 0;
}
.role-section {
    margin: 1.5rem 0 2rem;
}
.role-heading {
    align-items: end;
    display: flex;
    justify-content: space-between;
    margin-bottom: .7rem;
}
.target-chip, .champion-chip, .selected-badge, .rank-badge {
    border-radius: 999px;
    display: inline-block;
    font-size: .72rem;
    font-weight: 700;
    padding: .28rem .55rem;
}
.target-chip, .champion-chip {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
}
.player-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    background: var(--background-fill-secondary);
    box-shadow: 0 5px 18px rgba(0, 0, 0, .08);
    min-width: 0;
    overflow: hidden;
    padding: 1rem;
    transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease;
}
.player-card:hover {
    border-color: #c89b3c88;
    box-shadow: 0 9px 24px rgba(0, 0, 0, .14);
    transform: translateY(-2px);
}
.player-card.selected {
    border: 1px solid #c89b3c;
    box-shadow: 0 0 0 1px #c89b3c33, 0 8px 24px rgba(200, 155, 60, .12);
}
.player-card-top {
    align-items: center;
    display: flex;
    justify-content: space-between;
    margin-bottom: .75rem;
}
.candidate-number {
    color: var(--body-text-color-subdued);
    font-size: .75rem;
    font-weight: 700;
}
.selected-badge {
    background: #c89b3c;
    color: #101318;
}
.player-name {
    font-size: 1.15rem;
    margin: 0;
    overflow-wrap: anywhere;
}
.player-rank {
    align-items: center;
    color: var(--body-text-color-subdued);
    display: flex;
    font-size: .8rem;
    gap: .4rem;
    margin: .35rem 0 .9rem;
}
.rank-badge {
    background: #0f6cbd22;
    border: 1px solid #0f6cbd55;
    color: #56a8f5;
}
.player-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .5rem;
    margin-bottom: .9rem;
}
.player-metric {
    border-radius: 9px;
    background: var(--background-fill-primary);
    padding: .55rem;
}
.player-metric strong {
    font-size: .95rem;
}
.champion-label {
    color: var(--body-text-color-subdued);
    font-size: .72rem;
    font-weight: 700;
    margin-bottom: .38rem;
    text-transform: uppercase;
}
.champion-list {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem;
    min-height: 1.8rem;
}
.model-evidence {
    border-top: 1px solid var(--border-color-primary);
    font-size: .8rem;
    margin-top: .9rem;
    padding-top: .75rem;
}
.evidence-scores {
    color: var(--body-text-color-subdued);
    line-height: 1.65;
    margin-top: .65rem;
}
.profile-document {
    background: var(--background-fill-primary);
    border-radius: 8px;
    line-height: 1.45;
    margin-top: .55rem;
    max-height: 9rem;
    overflow-y: auto;
    padding: .65rem;
}
@media (max-width: 720px) {
    .team-metrics { grid-template-columns: 1fr; }
    .results-header, .role-heading { align-items: flex-start; flex-direction: column; }
}
"""


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


def _error_panel(title: str, detail: Any) -> str:
    return (
        '<section class="team-results"><div class="results-header"><div>'
        '<div class="results-kicker">Request error</div>'
        f'<h2>{_safe(title)}</h2></div></div>'
        f'<div class="result-alert">{_safe(detail)}</div></section>'
    )


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
    alert = ""
    if status in {"partial", "no_matches"} and message:
        alert = f'<div class="result-alert">{_safe(message)}</div>'
    header = (
        '<section class="team-results">'
        '<div class="results-header"><div><div class="results-kicker">Team builder</div>'
        '<h2>Jointly optimized lineup</h2></div>'
        '<p class="results-note">Every player below comes from indexed match history.</p></div>'
    )
    if status == "no_matches":
        return f"{header}{alert}</section>"

    sections = [header, alert]
    if team.get("team_fit_score") is not None:
        performance = float(team.get("predicted_performance", 0.0))
        compatibility = float(team.get("compatibility_score", 0.0))
        evaluated = int(team.get("lineups_evaluated", 0))
        sections.append(
            '<div class="team-metrics">'
            f'<div class="team-metric"><span>Model performance score</span><strong>{performance:.1f}%</strong></div>'
            f'<div class="team-metric"><span>Four-player compatibility</span><strong>{compatibility:.1f}%</strong></div>'
            f'<div class="team-metric"><span>Complete lineups evaluated</span><strong>{evaluated:,}</strong></div>'
            '</div><p class="score-disclaimer">Performance is a learned ranking score, not a calibrated win probability. '
            'Compatibility averages learned interactions across the six teammate pairs.</p>'
        )

    assignment = ""
    if team.get("requested_primary_role") == "FILL" and team.get("fill_assignment"):
        assignment = f" Fill assigned you to <strong>{_safe(_role_name(team['fill_assignment']))}</strong>."
    finder_role = _safe(_role_name(team.get("finder_primary_role")))
    sections.append(
        f'<p class="results-note">Your reserved role is <strong>{finder_role}</strong>. '
        f'The first choices were optimized together, not independently.{assignment}</p>'
    )

    selected = team.get("suggested_lineup") or []
    if selected:
        chips = []
        for candidate in selected:
            rank = " ".join(filter(None, [candidate.get("tier"), candidate.get("rank")])) or "Unavailable"
            name = candidate.get("player_name") or candidate.get("summoner_id")
            chips.append(
                '<div class="lineup-chip">'
                f'<div class="lineup-chip-role">{_safe(_role_name(candidate.get("slot_role")))}</div>'
                f'<strong class="lineup-chip-name">{_safe(name)}</strong>'
                f'<div class="lineup-chip-meta">{_safe(rank)} · {float(candidate.get("recommendation_score", 0.0)):.1f} score</div>'
                '</div>'
            )
        sections.append('<div class="selected-lineup">' + "".join(chips) + '</div>')

    pairs = team.get("pair_compatibility") or []
    evidence = team.get("champion_pool_evidence") or {}
    if pairs or evidence:
        evidence_items = []
        for pair in pairs:
            roles = pair.get("roles") or ["", ""]
            evidence_items.append(
                f'<li>{_safe(_role_name(roles[0]))} + {_safe(_role_name(roles[1]))}: '
                f'{float(pair.get("model_compatibility", 0.0)):.1f}% learned interaction score</li>'
            )
        champions = evidence.get("distinct_top_champions") or []
        coverage = ", ".join(_safe(champion) for champion in champions) or "Unavailable"
        evidence_items.append(f'<li>Recorded top-champion coverage: {coverage}</li>')
        sections.append(
            '<details class="team-evidence"><summary>Inspect learned team evidence</summary>'
            f'<ul>{"".join(evidence_items)}</ul></details>'
        )

    for slot in team.get("slots", []):
        role_name = _role_name(slot.get("role"))
        target = slot.get("target_champion")
        target_badge = f'<span class="target-chip">Target: {_safe(target)}</span>' if target else ""
        candidates = slot.get("candidates") or []
        if not candidates:
            sections.append(
                '<section class="role-section"><div class="role-heading">'
                f'<div><div class="role-kicker">Open role</div><h3>{_safe(role_name)}</h3></div>{target_badge}</div>'
                f'<div class="result-alert">No matching real candidates found for {_safe(role_name)}.</div></section>'
            )
            continue
        cards = []
        for index, candidate in enumerate(candidates, start=1):
            name = candidate.get("player_name") or candidate.get("summoner_id")
            is_selected = bool(candidate.get("selected_for_lineup"))
            selected_badge = '<span class="selected-badge">Selected</span>' if is_selected else ""
            rank = " ".join(filter(None, [candidate.get("tier"), candidate.get("rank")])) or "Unavailable"
            champions = candidate.get("top_champions") or {}
            champion_chips = "".join(
                f'<span class="champion-chip">{_safe(champion)} · {int(games)}</span>'
                for champion, games in champions.items()
            ) or '<span class="champion-chip">Unavailable</span>'
            rag_rank = candidate.get("rag_rank")
            rag_similarity = candidate.get("rag_similarity")
            rag_evidence = ""
            if rag_rank is not None:
                rag_evidence += f'<br>RAG rank: <strong>#{int(rag_rank)}</strong>'
            elif rag_similarity is not None:
                rag_evidence += '<br>RAG rank: <strong>Outside top retrieval</strong>'
            if rag_similarity is not None:
                rag_evidence += f' · similarity <strong>{float(rag_similarity):.3f}</strong>'
            card_class = "player-card selected" if is_selected else "player-card"
            cards.append(
                f'<article class="{card_class}">'
                f'<div class="player-card-top"><span class="candidate-number">OPTION {index}</span>{selected_badge}</div>'
                f'<h4 class="player-name">{_safe(name)}</h4>'
                f'<div class="player-rank"><span class="rank-badge">{_safe(rank)}</span>'
                f'<span>{int(candidate.get("matches", 0))} recorded matches</span></div>'
                '<div class="player-metrics">'
                f'<div class="player-metric"><span>Recommendation</span><strong>{float(candidate.get("recommendation_score", 0.0)):.1f}</strong></div>'
                f'<div class="player-metric"><span>KDA</span><strong>{float(candidate.get("kda", 0.0)):.2f}</strong></div>'
                f'<div class="player-metric"><span>Win rate</span><strong>{float(candidate.get("win_rate", 0.0)):.1%}</strong></div>'
                '</div>'
                '<div class="champion-label">Recorded top champions</div>'
                f'<div class="champion-list">{champion_chips}</div>'
                '<details class="model-evidence"><summary>Why this candidate</summary>'
                '<div class="evidence-scores">'
                f'Two-tower rank: <strong>#{_safe(candidate.get("two_tower_rank", "Unavailable"))}</strong> · '
                f'similarity <strong>{float(candidate.get("two_tower_similarity", 0.0)):.3f}</strong>{rag_evidence}'
                '</div>'
                f'<div class="profile-document">{_safe(candidate.get("rag_document", "No indexed profile document available."))}</div>'
                '</details></article>'
            )
        sections.append(
            '<section class="role-section"><div class="role-heading">'
            f'<div><div class="role-kicker">Open role</div><h3>{_safe(role_name)}</h3></div>{target_badge}</div>'
            f'<div class="candidate-grid">{"".join(cards)}</div></section>'
        )
    sections.append('</section>')
    return "".join(sections)


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
        "tier": tier_label,
        "rank": None if division_label == "Any division" else division_label,
        "preference": preference or "",
        "candidates_per_role": int(candidates_per_role),
        "max_tier_gap": int(max_tier_gap),
    }
    try:
        response = requests.post(f"{API_URL}/team/recommend", json=payload, timeout=120)
        if not response.ok:
            return (
                _error_panel(
                    "Request failed",
                    _response_error(response, response.text or "The API rejected the request."),
                ),
                {}, gr.update(choices=[], value=None), "", "",
            )
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return (
            _error_panel(
                "API connection failed",
                f"{exc} Confirm FastAPI is running at {API_URL}.",
            ),
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
            primary_role = gr.Dropdown(list(ROLE_LABELS), value="Fill", label="Finder’s primary role")
            tier = gr.Dropdown(TIERS, value="PLATINUM", label="Your rank tier")
            division = gr.Dropdown(DIVISIONS, value="IV", label="Your division")
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
        recommendation_output = gr.HTML(
            value='<div class="team-results"></div>',
            container=False,
            elem_id="team-results",
            css_template=APP_CSS,
        )

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
