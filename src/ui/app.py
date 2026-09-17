from __future__ import annotations

import html
import os
import re
from typing import Any
from uuid import uuid4

import gradio as gr
import requests

from src.config import API_URL
from src.ui import backend
from src.ui.idle import assert_on_demand_ui, valid_snapshot
from src.ui.icon_select import icon_select
from src.ui.league_assets import (
    ASSET_ROOT, ICON_CSS, champion_html, champion_label, champion_text_html,
    champion_url, division_url, rank_html, rank_url, recorded_champions,
)
from src.ranks import MATCHMAKING_TIERS
from src.name_safety import mask_name, mask_known_names
from src.ui.theme import TITLE, THEME, GLOBAL_CSS, FONT_HEAD, HERO_HTML, HERO_CSS, SECTION_CSS, EMPTY_HTML


ROLE_LABELS = {
    "Top": "TOP", "Jungle": "JUNGLE", "Mid": "MID",
    "Bottom": "BOTTOM", "Support": "SUPPORT", "Fill": "FILL",
}
DISPLAY_ROLE = {value: key for key, value in ROLE_LABELS.items()}
TIERS = list(MATCHMAKING_TIERS)
DIVISIONS = ["Any division", "IV", "III", "II", "I"]

# Delegation survives HTML updates; only explicit card buttons trigger inference.
CARD_SCOUT_JS = """
element.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-candidate-key]');
    if (!button || !element.contains(button)) return;
    if (button.disabled) return;
    trigger('scout', {candidate_key: button.dataset.candidateKey, revision: button.dataset.revision});
});
"""

APP_CSS = ICON_CSS + """
#team-results {
    margin-top: 1.25rem;
}
.team-results {
    color: var(--body-text-color);
    font-family: var(--font);
}
.empty-lineup { border:1px dashed #37525a; background:linear-gradient(135deg,#0c202c,#07131d); text-align:center; padding:30px 20px; border-radius:6px; }
.empty-mark { font-size:30px; color:#c8aa6e; }
.empty-lineup h2 { margin:8px 0; font:600 18px/1.4 Arial,sans-serif; color:#f0e6d2; }
.empty-lineup p { margin:0 auto; max-width:520px; color:#a3b4bc; font-size:13px; line-height:1.6; }
.empty-slots { display:flex; justify-content:center; gap:10px; margin-top:20px; }
.empty-slots span { border:1px solid #28424b; color:#71939d; padding:10px 20px; font-size:11px; letter-spacing:.1em; }
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
    border-radius: 5px;
    background: linear-gradient(135deg, #102835, #081923);
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
    border-radius: 5px;
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
.team-evidence summary {
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
    border-radius: 3px;
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
    border-radius: 6px;
    background: linear-gradient(145deg, #102835, #07141f);
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
    border: 1px solid #c8aa6e;
    box-shadow: inset 0 3px 0 #c8aa6e, 0 8px 24px #00000030;
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
    background: #0ac8b914;
    border: 1px solid #0ac8b944;
    color: #7dd9db;
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
.candidate-scout-button {
    width: 100%; margin-top: 1rem; padding: .7rem; border-radius: 4px;
    border: 1px solid #c8aa6e; background: #102835; color: #f0e6d2;
    font: 600 .85rem/1.4 Arial, sans-serif; cursor: pointer;
}
.candidate-scout-button:hover { background: #193b49; }
.candidate-scout-button:focus-visible { outline: 2px solid #0ac8b9; outline-offset: 3px; }
.candidate-scout {
    border-top: 1px solid var(--border-color-primary);
    font-size: .8rem;
    margin-top: .9rem;
    padding-top: .75rem;
}
.candidate-report {
    background: var(--background-fill-primary);
    border-radius: 8px;
    line-height: 1.45;
    margin-top: .55rem;
    max-height: 26rem;
    overflow-y: auto;
    padding: .65rem;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.candidate-report:empty { display: none; }
.candidate-scout-button:disabled { opacity: .65; cursor: wait; }
@media (max-width: 720px) {
    .team-metrics { grid-template-columns: 1fr; }
    .results-header, .role-heading { align-items: flex-start; flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
    .player-card { transition: none; }
    .player-card:hover { transform: none; }
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
        f'<div class="result-alert">{champion_text_html(str(detail))}</div></section>'
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
            choices.append((f"{_role_name(role)} — {mask_name(str(name))}{selected}", key))
            candidates[key] = {
                "summoner_id": player_id,
                "slot_role": role,
                "target_champion": slot.get("target_champion"),
            }
    return choices, candidates


def _render_team(
    data: dict[str, Any], reports: dict[str, str] | None = None,
    loading_key: str | None = None, revision: str = "",
) -> str:
    status = data.get("status")
    message = data.get("message")
    team = data.get("team") or {}
    alert = ""
    if status in {"partial", "no_matches"} and message:
        alert = f'<div class="result-alert">{champion_text_html(str(message))}</div>'
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
            rank = rank_html(candidate.get("tier"), candidate.get("rank"))
            name = candidate.get("player_name") or candidate.get("summoner_id")
            chips.append(
                '<div class="lineup-chip">'
                f'<div class="lineup-chip-role">{_safe(_role_name(candidate.get("slot_role")))}</div>'
                f'<strong class="lineup-chip-name">{_safe(mask_name(str(name)))}</strong>'
                f'<div class="lineup-chip-meta">{rank} · {float(candidate.get("recommendation_score", 0.0)):.1f} score</div>'
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
        coverage = ", ".join(champion_html(champion) for champion in champions) or "Unavailable"
        evidence_items.append(f'<li>Recorded top-champion coverage: {coverage}</li>')
        sections.append(
            '<details class="team-evidence"><summary>Inspect learned team evidence</summary>'
            f'<ul>{"".join(evidence_items)}</ul></details>'
        )

    for slot in team.get("slots", []):
        role_name = _role_name(slot.get("role"))
        target = slot.get("target_champion")
        target_badge = f'<span class="target-chip">Target: {champion_html(target)}</span>' if target else ""
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
            rank = rank_html(candidate.get("tier"), candidate.get("rank"))
            champions = candidate.get("top_champions") or {}
            champion_chips = "".join(
                f'<span class="champion-chip">{champion_html(champion)} · {int(games)}</span>'
                for champion, games in champions.items()
            ) or '<span class="champion-chip">Unavailable</span>'
            candidate_key = str(slot.get("role", "")) + "::" + str(candidate.get("summoner_id", ""))
            busy = candidate_key == loading_key
            report = (reports or {}).get(candidate_key, "")
            if busy:
                report = "Retrieving this teammate’s recorded profile and generating their RAG scout…"
            report_html = _inline_report_html(report)
            button_label = "Scouting this candidate…" if busy else "Why this candidate? · RAG scout"
            disabled = ' disabled aria-busy="true"' if busy else ""
            card_class = "player-card selected" if is_selected else "player-card"
            cards.append(
                f'<article class="{card_class}">'
                f'<div class="player-card-top"><span class="candidate-number">OPTION {index}</span>{selected_badge}</div>'
                f'<h4 class="player-name">{_safe(mask_name(str(name)))}</h4>'
                f'<div class="player-rank"><span class="rank-badge">{rank}</span>'
                f'<span>{int(candidate.get("matches", 0))} recorded matches</span></div>'
                '<div class="player-metrics">'
                f'<div class="player-metric"><span>Recommendation</span><strong>{float(candidate.get("recommendation_score", 0.0)):.1f}</strong></div>'
                f'<div class="player-metric"><span>KDA</span><strong>{float(candidate.get("kda", 0.0)):.2f}</strong></div>'
                f'<div class="player-metric"><span>Win rate</span><strong>{float(candidate.get("win_rate", 0.0)):.1%}</strong></div>'
                '</div>'
                '<div class="champion-label">Recorded top champions</div>'
                f'<div class="champion-list">{champion_chips}</div>'
                '<section class="candidate-scout">'
                f'<button type="button" class="candidate-scout-button"{disabled} '
                f'data-candidate-key="{_safe(candidate_key)}" data-revision="{_safe(revision)}" '
                f'aria-label="Scout {_safe(mask_name(str(name)))} for {_safe(role_name)}">{button_label}</button>'
                f'<div class="candidate-report" role="status" aria-live="polite">{report_html}</div>'
                '</section></article>'
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
) -> tuple[str, dict[str, Any], str]:
    if primary_role_label not in ROLE_LABELS or tier_label not in TIERS or division_label not in DIVISIONS:
        return _error_panel("Invalid selection", "Choose one of the listed roles, rank tiers, and divisions."), {}, ""
    allowed = set(recorded_champions())
    for champion in (top_champion, jungle_champion, mid_champion, bottom_champion, support_champion):
        if champion and (not isinstance(champion, str) or champion not in allowed):
            return _error_panel("Invalid champion", "Select a champion from the recorded dataset dropdown."), {}, ""
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
        response = backend.post(f"{API_URL}/team/recommend", json=payload, timeout=120)
        if not response.ok:
            return (
                _error_panel(
                    "Request failed",
                    _response_error(response, response.text or "The API rejected the request."),
                ),
                {}, "",
            )
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return (
            _error_panel(
                "API connection failed",
                f"{exc} Confirm FastAPI is running at {API_URL}.",
            ),
            {}, "",
        )

    _, candidates = _candidate_selector(data)
    state = {
        "response": data, "preference": preference or "", "candidates": candidates,
        "candidate_reports": {}, "revision": uuid4().hex,
    }
    return _render_team(data, revision=state["revision"]), state, ""


def scout_lineup(state: dict[str, Any]) -> str:
    if not valid_snapshot(state):
        return "Invalid browser results. Please rebuild your team."
    if not state or not state.get("response"):
        return "Build a complete team before requesting a lineup explanation."
    team = state["response"].get("team") or {}
    if not team.get("complete") or not team.get("suggested_lineup"):
        return "A complete real lineup is required before the local model can explain it."
    selected = team["suggested_lineup"]
    if (len(selected) != 4 or any(
        not isinstance(candidate.get("summoner_id"), str)
        or not isinstance(candidate.get("slot_role"), str)
        or candidate.get("slot_role") not in set(DISPLAY_ROLE) - {"FILL"}
        for candidate in selected
    ) or not isinstance(team.get("finder_primary_role"), str)
            or team.get("finder_primary_role") not in set(DISPLAY_ROLE) - {"FILL"}):
        return "Invalid browser lineup. Please rebuild your team."
    lineup = {candidate["slot_role"]: candidate["summoner_id"] for candidate in selected}
    payload = {
        "primary_role": team["finder_primary_role"],
        "lineup": lineup,
        "tier": (team.get("finder") or {}).get("tier"),
        "rank": (team.get("finder") or {}).get("rank"),
        "target_champions": {
            slot["role"]: slot["target_champion"]
            for slot in team.get("slots", [])
            if isinstance(slot.get("role"), str) and isinstance(slot.get("target_champion"), str)
        },
        "preference": state.get("preference", ""),
    }
    try:
        response = backend.post(f"{API_URL}/team/scout", json=payload, timeout=900)
        if not response.ok:
            return f"**Lineup report unavailable:** {_safe(_response_error(response, response.text))}"
        return "## Grounded lineup explanation\n\n" + _display_report(response.json()["report"], state)
    except (requests.RequestException, ValueError, KeyError) as exc:
        return f"**Lineup report failed:** {_safe(exc)}"


def scout_candidate(candidate_key: str | None, state: dict[str, Any]) -> str:
    if not valid_snapshot(state):
        return "Invalid browser results. Please rebuild your team."
    if not candidate_key or not state:
        return "Choose a real candidate from the recommendation results first."
    candidate = (state.get("candidates") or {}).get(candidate_key)
    if not candidate:
        return "That candidate is no longer present in the current recommendation results."
    if (not isinstance(candidate.get("summoner_id"), str)
            or not isinstance(candidate.get("slot_role"), str)):
        return "Invalid browser candidate. Please rebuild your team."
    # Browser state supplies identity/context only. Never forward displayed
    # metrics, reports, or rag_document as authoritative scouting evidence.
    payload = {
        "summoner_id": candidate["summoner_id"],
        "slot_role": candidate["slot_role"],
        "target_champion": candidate.get("target_champion"),
        "preference": state.get("preference", ""),
    }
    try:
        response = backend.post(f"{API_URL}/scout", json=payload, timeout=900)
        if not response.ok:
            return f"**Candidate report unavailable:** {_safe(_response_error(response, response.text))}"
        label = _role_name(candidate.get("slot_role"))
        for slot in (state.get("response", {}).get("team") or {}).get("slots", []):
            for player in slot.get("candidates", []):
                if str(player.get("summoner_id")) == candidate["summoner_id"]:
                    label += " — " + mask_name(str(player.get("player_name") or candidate["summoner_id"]))
                    break
        return "## Candidate scout: " + _safe(label).replace("*", "&#42;") + "\n\n" + _display_report(response.json()["report"], state)
    except (requests.RequestException, ValueError, KeyError) as exc:
        return f"**Candidate report failed:** {_safe(exc)}"


def _display_report(report: str, state: dict[str, Any]) -> str:
    names = [str(player.get("player_name") or player.get("summoner_id", ""))
             for slot in (state.get("response", {}).get("team") or {}).get("slots", [])
             for player in slot.get("candidates", [])]
    # Entities render literal asterisks instead of Markdown emphasis/rules.
    return re.sub(r"\*{3,}", lambda match: "&#42;" * len(match.group()), mask_known_names(report, names))


def _inline_report_html(report: str) -> str:
    # Render a small, safe subset of Markdown. Escape everything before adding
    # our own tags; never insert model-produced HTML into the card.
    safe = champion_text_html(html.unescape(report))
    safe = re.sub(r"(?m)^#{1,6}\s+([^\n]+)", r"<strong>\1</strong>", safe)
    return re.sub(r"(?<!\*)\*\*([^*\n]+)\*\*(?!\*)", r"<strong>\1</strong>", safe)


def scout_card(state: dict[str, Any], evt: gr.EventData):
    if not valid_snapshot(state):
        gr.Warning("Invalid browser results. Please rebuild your team.")
        yield gr.skip(), gr.skip()
        return
    key = getattr(evt, "candidate_key", None)
    if (not isinstance(key, str) or key not in (state or {}).get("candidates", {})
            or getattr(evt, "revision", None) != (state or {}).get("revision")):
        gr.Warning("This card is no longer in your current results. Please use the newly built lineup.")
        yield gr.skip(), gr.skip()
        return
    reports = dict(state.get("candidate_reports") or {})
    try:
        loading = _render_team(state["response"], reports, loading_key=key, revision=state["revision"])
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError):
        gr.Warning("Invalid browser results. Please rebuild your team.")
        yield gr.skip(), gr.skip()
        return
    yield loading, gr.skip()
    reports[key] = scout_candidate(key, state)
    updated = {**state, "candidate_reports": reports}
    yield _render_team(state["response"], reports, revision=state["revision"]), updated


def scout_lineup_ui(state: dict[str, Any]):
    yield "Generating a whole-lineup RAG report from the four selected teammates…"
    yield '<div class="lineup-report">' + _inline_report_html(scout_lineup(state)) + '</div>'


def build_app() -> gr.Blocks:
    # Serve only the public artwork folder, never data, model files, or .env.
    gr.set_static_paths(paths=[ASSET_ROOT])
    champions = [{"value": name, "label": champion_label(name), "icon": champion_url(name)}
                 for name in recorded_champions()]
    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.HTML(HERO_HTML, css_template=HERO_CSS, elem_id="league-hero", container=False)
        # gr.State opens a persistent /heartbeat SSE request for every tab,
        # keeping request-billed Cloud Run active. JSON lives only in this page,
        # and is returned to Python only when a scout button is clicked.
        result_state = gr.JSON(value={}, visible=False, elem_id="tab-lineup-snapshot")
        gr.HTML('<div class="section-heading"><span>01</span> Configure your team</div>', css_template=SECTION_CSS)
        with gr.Column(elem_id="finder-panel"):
            with gr.Row():
                primary_role = gr.Dropdown(list(ROLE_LABELS), value="Fill", label="Your Primary Role")
                tier = icon_select(
                    [{"value": name, "label": name.title(), "icon": rank_url(name)} for name in TIERS],
                    value="PLATINUM", label="Your rank tier", elem_id="rank-tier",
                )
                division = icon_select(
                    [{"value": name, "label": name, "icon": division_url(name)} for name in DIVISIONS],
                    value="IV", label="Your division", image_only=True, elem_id="rank-division",
                )
            with gr.Accordion("Advanced search settings", open=False, elem_id="advanced-search-settings"):
                with gr.Row():
                    candidates_per_role = gr.Slider(1, 5, value=3, step=1, label="Candidates per open role")
                    max_tier_gap = gr.Slider(0, 3, value=1, step=1, label="Maximum tier gap")
                preference = gr.Textbox(
                    label="Team preference", lines=3, max_length=1000, elem_id="team-preference",
                    placeholder="What does your team need? Try reliable vision, high assists, or experienced players.",
                )
                gr.Markdown("**Optional target champions** — specify a champion only for a role where recorded champion history is required.")
                with gr.Row():
                    top_champion = icon_select(champions, label="Top champion", optional=True, searchable=True, elem_id="target-top")
                    jungle_champion = icon_select(champions, label="Jungle champion", optional=True, searchable=True, elem_id="target-jungle")
                    mid_champion = icon_select(champions, label="Mid champion", optional=True, searchable=True, elem_id="target-mid")
                with gr.Row():
                    bottom_champion = icon_select(champions, label="Bottom champion", optional=True, searchable=True, elem_id="target-bottom")
                    support_champion = icon_select(champions, label="Support champion", optional=True, searchable=True, elem_id="target-support")
                if not champions:
                    gr.Markdown("No recorded champions available. Preprocess the real data before choosing target champions.")
            build_button = gr.Button("Build my team", variant="primary", elem_id="build-team")
        recommendation_output = gr.HTML(
            value=EMPTY_HTML,
            container=False,
            elem_id="team-results",
            css_template=APP_CSS,
            js_on_load=CARD_SCOUT_JS,
        )

        gr.HTML('<div class="section-heading"><span>02</span> Scout your lineup</div>', css_template=SECTION_CSS)
        with gr.Column(elem_id="lineup-scout-panel"):
            lineup_scout_button = gr.Button("Explain why this lineup complements itself", variant="secondary")
            lineup_scout_output = gr.HTML(elem_id="lineup-scout-report", css_template=ICON_CSS)
        gr.Markdown(
            "Recommendations use recorded match data and trained ranking models. Scores are not win probabilities. "
            "AI scout reports may contain errors; inspect their evidence.\n\n"
            "Unofficial fan-made portfolio project. Not endorsed or sponsored by Riot Games.",
            elem_id="project-note",
        )

        build_button.click(
            recommend_team,
            inputs=[
                primary_role, tier, division, candidates_per_role, max_tier_gap,
                preference, top_champion, jungle_champion, mid_champion,
                bottom_champion, support_champion,
            ],
            outputs=[
                recommendation_output, result_state, lineup_scout_output,
            ],
            concurrency_id="inference", concurrency_limit=1,
        )
        lineup_scout_button.click(
            scout_lineup_ui, inputs=[result_state], outputs=[lineup_scout_output],
            concurrency_id="inference", concurrency_limit=1,
        )
        recommendation_output.scout(
            scout_card, inputs=[result_state],
            outputs=[recommendation_output, result_state],
            concurrency_id="inference", concurrency_limit=1,
            scroll_to_output=False, show_progress="hidden",
        )
    assert_on_demand_ui(demo)
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        theme=THEME, css=GLOBAL_CSS, head=FONT_HEAD,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
