from __future__ import annotations

import requests
import streamlit as st

from src.config import API_URL


ROLE_LABELS = {
    "Top": "TOP", "Jungle": "JUNGLE", "Mid": "MID",
    "Bottom": "BOTTOM", "Support": "SUPPORT", "Fill": "FILL",
}
DISPLAY_ROLE = {value: key for key, value in ROLE_LABELS.items()}

st.set_page_config(page_title="LoL role queue team builder", page_icon=":material/groups:", layout="wide")
st.session_state.setdefault("team_response", None)
st.session_state.setdefault("submitted_preference", "")

st.title("LoL role queue team builder")
st.caption(
    "Choose the finder’s primary queue role. The system fills every other role with candidates derived only from local match records."
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
    identity_col, rank_col, settings_col = st.columns(3, vertical_alignment="bottom")
    with identity_col:
        finder = st.text_input(
            "Finder identity",
            placeholder="Summoner ID, PUUID, or Riot game name",
        )
    with rank_col:
        tier_label = st.selectbox(
            "Rank tier",
            ["Use finder profile", "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"],
        )
    with settings_col:
        candidates_per_role = st.slider("Candidates per open role", 1, 5, 3)
        max_tier_gap = st.slider("Maximum tier gap", 0, 3, 1)

    with st.expander("Optional target champions for open roles", expanded=False):
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
        "finder": finder or None,
        "primary_role": primary_role,
        "target_champions": target_champions,
        "tier": None if tier_label == "Use finder profile" else tier_label,
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
        except requests.RequestException as exc:
            st.session_state.team_response = None
            st.error(f"The API could not build the team: {exc}")

data = st.session_state.team_response
if data:
    team = data["team"]
    st.divider()
    header = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
    header.subheader("Recommended lineup")
    if team.get("team_fit_score") is not None:
        header.metric("Average team-fit score", f"{team['team_fit_score']:.1f}")

    if team.get("requested_primary_role") == "FILL":
        st.info(
            f"Fill assignment: **{DISPLAY_ROLE[team['fill_assignment']]}** produced the strongest available lineup."
        )
    if data["status"] == "partial":
        st.warning(data["message"])
    elif data["status"] == "no_matches":
        st.warning(data["message"])

    finder_info = team["finder"]
    with st.container(border=True):
        st.markdown(f"### {DISPLAY_ROLE[team['finder_primary_role']]} — finder")
        finder_name = finder_info.get("player_name") or finder_info.get("summoner_id") or "Unresolved finder"
        st.write(finder_name)
        if finder_info.get("profile_found"):
            st.caption(" ".join(filter(None, [finder_info.get("tier"), finder_info.get("rank")])))

    st.subheader("Open role recommendations")
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
                st.markdown(f"### {index}. {name}")
                metrics = st.columns(4)
                metrics[0].metric("Team fit", f"{candidate['team_fit_score']:.1f}")
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
                    detail_columns = st.columns(4)
                    detail_columns[0].metric("Experience", f"{100 * candidate['experience_score']:.0f}%")
                    detail_columns[1].metric("Performance", f"{100 * candidate['performance_score']:.0f}%")
                    detail_columns[2].metric("Rank fit", f"{100 * candidate['rank_fit_score']:.0f}%")
                    detail_columns[3].metric(
                        "Champion affinity", f"{100 * candidate['champion_affinity_score']:.0f}%"
                    )
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
