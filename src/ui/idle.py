"""Keep idle tabs stateless on the server; model evidence stays server-owned."""
from typing import Any


def assert_on_demand_ui(demo: Any) -> None:
    """Fail at startup if UI changes introduce an always-connected session."""
    config = demo.get_config_file()
    if config.get("connect_heartbeat") is not False:
        raise RuntimeError("Idle-cost safeguard: Gradio session heartbeat must be disabled.")
    if any(component["type"] in {"state", "browserstate", "timer"}
           for component in config["components"]):
        raise RuntimeError("Idle-cost safeguard: use tab-local JSON, not session state or timers.")
    for dependency in config["dependencies"]:
        if not dependency["targets"] or any(
            event not in {"click", "scout"} for _, event in dependency["targets"]
        ):
            raise RuntimeError("Idle-cost safeguard: backend work must require an explicit button click.")


def valid_snapshot(value: Any) -> bool:
    """Bound and check untrusted display state, never authenticate evidence with it.

    Hidden JSON is sent by the browser on each scout request, including after a
    container restart. The API independently resolves IDs and fetches real facts.
    No signing secret or server session is needed to keep the displayed lineup.
    """
    if not isinstance(value, dict):
        return False
    pending = [(value, 0)]
    nodes = chars = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 20000 or depth > 12:
            return False
        if isinstance(item, dict):
            if len(item) > 1000 or not all(isinstance(key, str) for key in item):
                return False
            chars += sum(len(key) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            if len(item) > 1000:
                return False
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            chars += len(item)
        elif item is not None and not isinstance(item, (bool, int, float)):
            return False
        if chars > 1_000_000:
            return False
    if not isinstance(value.get("preference", ""), str) or len(value.get("preference", "")) > 1000:
        return False
    if not isinstance(value.get("revision", ""), str):
        return False
    candidates = value.get("candidates", {})
    reports = value.get("candidate_reports", {})
    response = value.get("response", {})
    if not all(isinstance(item, dict) for item in (candidates, reports, response)):
        return False
    if len(candidates) > 20 or len(reports) > 20:
        return False
    if not all(isinstance(item, dict) for item in candidates.values()):
        return False
    if not all(isinstance(item, str) for item in reports.values()):
        return False
    team = response.get("team", {})
    if not isinstance(team, dict) or not isinstance(team.get("finder") or {}, dict):
        return False
    for field, maximum in (("slots", 4), ("suggested_lineup", 4), ("pair_compatibility", 6)):
        rows = team.get(field, [])
        if not isinstance(rows, list) or len(rows) > maximum or not all(isinstance(row, dict) for row in rows):
            return False
    for slot in team.get("slots", []):
        rows = slot.get("candidates", [])
        if not isinstance(rows, list) or len(rows) > 5 or not all(isinstance(row, dict) for row in rows):
            return False
    return True
