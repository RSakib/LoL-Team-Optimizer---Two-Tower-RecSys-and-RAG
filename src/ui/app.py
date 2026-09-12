from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Streamlit executes this file as a script, so its directory (src/ui) can be the
# only project path available to Python. Add the repository root before importing
# the shared application package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import API_URL


ROLE_LABELS = {
    "Top": "TOP", "Jungle": "JUNGLE", "Mid": "MID",
    "Bottom": "BOTTOM", "Support": "SUPPORT", "Fill": "FILL",
}
DISPLAY_ROLE = {value: key for key, value in ROLE_LABELS.items()}

st.set_page_config(page_title="LoL joint team recommender", page_icon=":material/groups:", layout="wide")
st.session_state.setdefault("team_response", None)
st.session_state.setdefault("submitted_preference", "")
st.session_state.setdefault("team_scout_report", None)

st.title("LoL joint team recommender")
st.caption(
    "A two-tower model retrieves real candidates, then a trained team model scores complete four-player lineups jointly."
)

primary_role_label = st.selectbox(
    "Finder’s primary role",
    list(ROLE_LABELS),
    help="Fill evaluates every possible role assignment and returns the strongest complete lineup.",
    key="primary_role_label",
)
primary_role = ROLE_LABELS[primary_role_label]
open_roles = [role for role in DISPLAY_ROLE if role != primary_role] if primary_role != "FILL" else list(DISPLAY_ROLE)[:-1]

with st.form("team_builder_form"):
    tier_col, division_col, settings_col = st.columns(3, vertical_alignment="bottom")
    with tier_col:
        tier_label = st.selectbox(
            "Your rank tier",
            ["No rank filter", "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"],
            help="Choose a tier to recommend nearby-ranked teammates, or leave rank unrestricted.",
        )
    with division_col:
        division_label = st.selectbox(
            "Your division",
            ["Any division", "IV", "III", "II", "I"],
            help="Optionally require recommended teammates to have the same recorded division.",
        )
    with settings_col:
        candidates_per_role = st.slider("Candidates per open role", 1, 5, 3)
        max_tier_gap = st.slider("Maximum tier gap", 0, 3, 1)

    with st.expander("Optional target champions for teammate roles", expanded=False):
        st.caption("Specify a champion only when you want that role filled by someone with recorded history on it.")
        champion_columns = st.columns(2)
        target_champions: dict[str, str] = {}
        for index, role in enumerate(open_roles):
            with champion_columns[index % 2]:
                champion = st.text_input(
                    f"{DISPLAY_ROLE[role]} champion",
                    key=f"target_champion_{role}",
                    placeholder="Optional",
                )
                if champion.strip():
                    target_champions[role] = champion.strip()

    preference = st.text_area(
        "Team preference",
        placeholder="Examples: reliable vision, objective control, experienced frontline players",
    )
    submitted = st.form_submit_button(
        "Build my team",
        type="primary",
        icon=":material/group_add:",
        width="stretch",
    )

if submitted:
    payload = {
        "finder": None,
        "primary_role": primary_role,
        "target_champions": target_champions,
        "tier": None if tier_label == "No rank filter" else tier_label,
        "rank": None if division_label == "Any division" else division_label,
        "preference": preference,
        "candidates_per_role": candidates_per_role,
        "max_tier_gap": max_tier_gap,
    }
    results_slot = st.container()
    with results_slot.skeleton(height=240):
        try:
            response = requests.post(f"{API_URL}/team/recommend", json=payload, timeout=120)
            response.raise_for_status()
            st.session_state.team_response = response.json()
            st.session_state.submitted_preference = preference
            st.session_state.team_scout_report = None
        except requests.RequestException as exc:
            st.session_state.team_response = None
            st.error(f"The API could not build the team: {exc}")

data = st.session_state.team_response
if data:
    team = data["team"]
    st.divider()
    st.subheader("Jointly optimized lineup")
    if team.get("team_fit_score") is not None:
        with st.container(horizontal=True):
            st.metric(
                "Model performance score",
                f"{team['predicted_performance']:.1f}%",
                border=True,
                help="A ranking score learned from complete real teams. It is not a calibrated win probability or a guarantee.",
            )
            st.metric(
                "Four-player compatibility",
                f"{team['compatibility_score']:.1f}%",
                border=True,
                help="Average learned interaction score across the six pairs among the four recommended players.",
            )
            st.metric("Complete lineups evaluated", f"{team['lineups_evaluated']:,}", border=True)

    if team.get("requested_primary_role") == "FILL":
        st.info(
            f"Fill assignment: **{DISPLAY_ROLE[team['fill_assignment']]}** produced the strongest available lineup."
        )
    if data["status"] == "partial":
        st.warning(data["message"])
    elif data["status"] == "no_matches":
        st.warning(data["message"])

    st.caption(
        f"Your reserved role: **{DISPLAY_ROLE[team['finder_primary_role']]}**. "
        "The four displayed first choices were selected together, not independently."
    )
    if team.get("complete") and team.get("suggested_lineup"):
        selected_lineup = {
            candidate["slot_role"]: candidate["summoner_id"]
            for candidate in team["suggested_lineup"]
        }
        if st.button(
            "Explain why this lineup complements itself",
            type="primary",
            icon=":material/psychology:",
        ):
            try:
                scout_response = requests.post(
                    f"{API_URL}/team/scout",
                    json={
                        "primary_role": team["finder_primary_role"],
                        "lineup": selected_lineup,
                        "tier": team["finder"].get("tier"),
                        "rank": team["finder"].get("rank"),
                        "target_champions": {
                            slot["role"]: slot["target_champion"]
                            for slot in team["slots"] if slot.get("target_champion")
                        },
                        "preference": st.session_state.submitted_preference,
                    },
                    timeout=120,
                )
                if scout_response.status_code == 503:
                    st.info(scout_response.json().get("detail", "Lineup scouting is not configured."))
                else:
                    scout_response.raise_for_status()
                    st.session_state.team_scout_report = scout_response.json()["report"]
            except requests.RequestException as exc:
                st.error(f"Lineup report failed: {exc}")
        if st.session_state.team_scout_report:
            with st.container(border=True):
                st.markdown("#### Grounded lineup explanation")
                st.markdown(st.session_state.team_scout_report)
        with st.expander("Inspect learned pair compatibility and real lineup evidence"):
            for pair in team.get("pair_compatibility", []):
                st.write(
                    f"{DISPLAY_ROLE[pair['roles'][0]]} + {DISPLAY_ROLE[pair['roles'][1]]}: "
                    f"{pair['model_compatibility']:.1f}% learned interaction score"
                )
            evidence = team.get("champion_pool_evidence") or {}
            champions = evidence.get("distinct_top_champions") or []
            st.write(
                "Recorded top-champion coverage: "
                + (", ".join(champions) if champions else "Unavailable")
            )
            st.caption(
                "Interaction values are ranking scores, not probabilities. Champion pools and profile statistics are exact retrieved evidence."
            )

    st.subheader("Selected teammates and alternatives")
    for slot in team["slots"]:
        role_name = DISPLAY_ROLE[slot["role"]]
        target = f" · requested champion: {slot['target_champion']}" if slot.get("target_champion") else ""
        st.markdown(f"## {role_name}{target}")
        if not slot["candidates"]:
            st.warning(f"No matching real candidates found for {role_name}.")
            continue
        for index, candidate in enumerate(slot["candidates"], start=1):
            name = candidate.get("player_name") or candidate["summoner_id"]
            with st.container(border=True):
                selected_label = " — selected for the joint lineup" if candidate.get("selected_for_lineup") else ""
                st.markdown(f"### {index}. {name}{selected_label}")
                metrics = st.columns(4)
                metrics[0].metric("Recommendation", f"{candidate['recommendation_score']:.1f}")
                metrics[1].metric(
                    "Rank",
                    " ".join(filter(None, [candidate.get("tier"), candidate.get("rank")])) or "Unavailable",
                )
                metrics[2].metric("KDA", f"{candidate['kda']:.2f}")
                metrics[3].metric("Win rate", f"{candidate['win_rate']:.1%}")
                st.caption(f"{candidate['matches']} recorded matches · primary role {role_name}")
                champions = candidate.get("top_champions") or {}
                st.write(
                    "Top champions: "
                    + (", ".join(f"{champion} ({games})" for champion, games in champions.items()) or "Unavailable")
                )
                with st.expander("Why this candidate"):
                    with st.container(horizontal=True):
                        st.metric("Two-tower rank", f"#{candidate['two_tower_rank']}", border=True)
                        st.metric("Embedding similarity", f"{candidate['two_tower_similarity']:.3f}", border=True)
                        if st.session_state.submitted_preference.strip():
                            rag_rank = candidate.get("rag_rank")
                            rag_score = candidate.get("rag_similarity")
                            st.metric("RAG rank", f"#{rag_rank}" if rag_rank else "Outside top retrieval", border=True)
                            st.metric("RAG similarity", f"{rag_score:.3f}" if rag_score is not None else "Unavailable", border=True)
                    if st.session_state.submitted_preference.strip():
                        st.caption(
                            "The trained two-tower rank and role-scoped RAG rank are combined with reciprocal-rank fusion."
                        )
                    else:
                        st.caption("Candidate generation comes from the trained two-tower embedding similarity.")
                    st.write(candidate["rag_document"])
                if st.button(
                    "Generate role-specific scout report",
                    key=f"scout_{slot['role']}_{candidate['summoner_id']}",
                    icon=":material/strategy:",
                ):
                    try:
                        scout_response = requests.post(
                            f"{API_URL}/scout",
                            json={
                                "summoner_id": candidate["summoner_id"],
                                "slot_role": slot["role"],
                                "target_champion": slot.get("target_champion"),
                                "preference": st.session_state.submitted_preference,
                            },
                            timeout=120,
                        )
                        if scout_response.status_code == 503:
                            st.info(scout_response.json().get("detail", "LLM scouting is not configured."))
                        else:
                            scout_response.raise_for_status()
                            st.markdown(scout_response.json()["report"])
                    except requests.RequestException as exc:
                        st.error(f"Scout report failed: {exc}")
