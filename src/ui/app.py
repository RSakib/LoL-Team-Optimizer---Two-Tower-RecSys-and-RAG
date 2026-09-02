from __future__ import annotations

import requests
import streamlit as st

from src.config import API_URL

st.set_page_config(page_title="LoL Tactical Matchmaker", page_icon="⚔️", layout="wide")
st.title("LoL Tactical Matchmaker")
st.caption("Every candidate and statistic comes from match records in the local data/ directory.")

with st.form("recommendation_form"):
    left, middle, right = st.columns(3)
    with left:
        summoner = st.text_input("Your Summoner ID, PUUID, or Riot game name")
        role = st.selectbox("Required teammate role", ["Any", "TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT"])
    with middle:
        champion = st.text_input("Target champion (optional)")
        tier = st.selectbox("Rank tier", ["Any", "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"])
    with right:
        rank = st.selectbox("Division", ["Any", "I", "II", "III", "IV"])
        top_k = st.slider("Candidates", 1, 20, 5)
    preference = st.text_area("What kind of teammate are you looking for?", placeholder="Strong vision control and a reliable support champion pool")
    submitted = st.form_submit_button("Find real candidates", type="primary")

if submitted:
    payload = {
        "summoner": summoner or None, "role": None if role == "Any" else role,
        "target_champion": champion or None, "tier": None if tier == "Any" else tier,
        "rank": None if rank == "Any" else rank, "preference": preference, "top_k": top_k,
    }
    try:
        response = requests.post(f"{API_URL}/recommend", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        st.error(f"The API could not complete the request: {exc}")
        st.stop()
    if not data["candidates"]:
        st.warning(data.get("message") or "No matching real candidates found")
    for candidate in data["candidates"]:
        name = candidate.get("player_name") or candidate["summoner_id"]
        st.subheader(name)
        a, b, c, d = st.columns(4)
        a.metric("Match score", f"{candidate['match_score']:.1f}")
        b.metric("Rank", " ".join(filter(None, [candidate.get("tier"), candidate.get("rank")])) or "Unavailable")
        c.metric("KDA", f"{candidate['kda']:.2f}")
        d.metric("Win rate", f"{candidate['win_rate']:.1%}")
        st.write(f"**Role:** {candidate.get('role') or 'Unavailable'} · **Recorded matches:** {candidate['matches']}")
        champions = candidate.get("top_champions") or {}
        st.write("**Top champions:** " + (", ".join(f"{name} ({games})" for name, games in champions.items()) or "Unavailable"))
        with st.expander("Computed profile"):
            st.write(candidate["rag_document"])
        if st.button("Generate tactical scout report", key=f"scout-{candidate['summoner_id']}"):
            try:
                scout_response = requests.post(
                    f"{API_URL}/scout", json={"summoner_id": candidate["summoner_id"], "preference": preference}, timeout=120
                )
                if scout_response.status_code == 503:
                    st.info(scout_response.json().get("detail", "LLM scouting is not configured."))
                else:
                    scout_response.raise_for_status()
                    st.markdown(scout_response.json()["report"])
            except requests.RequestException as exc:
                st.error(f"Scout report failed: {exc}")
        st.divider()
