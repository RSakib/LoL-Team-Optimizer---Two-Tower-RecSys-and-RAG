"""Serving policy only; historical training vocabularies retain every rank."""
from typing import Literal, get_args

MatchmakingTier = Literal["PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
MATCHMAKING_TIERS = get_args(MatchmakingTier)
