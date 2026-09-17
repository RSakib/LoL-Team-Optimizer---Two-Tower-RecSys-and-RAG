"""Reuse the validated API handlers directly in a single-process Gradio Space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

from src.config import UI_BACKEND


@dataclass
class LocalResponse:
    payload: dict[str, Any]
    status_code: int = 200

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        return str(self.payload)

    def json(self) -> dict[str, Any]:
        return self.payload


def post(url: str, *, json: dict[str, Any], timeout: int) -> Any:
    if UI_BACKEND == "http":
        return requests.post(url, json=json, timeout=timeout)
    if UI_BACKEND != "local":
        return LocalResponse({"detail": "LOL_UI_BACKEND must be local or http"}, 503)
    from fastapi import HTTPException
    from pydantic import ValidationError
    from src.api import main as api

    routes = {
        "/team/recommend": (api.TeamRecommendRequest, api.recommend_team),
        "/scout": (api.ScoutRequest, api.scout),
        "/team/scout": (api.TeamScoutRequest, api.scout_team),
    }
    route = routes.get(urlparse(url).path)
    if route is None:
        return LocalResponse({"detail": "Unknown route"}, 404)
    schema, handler = route
    try:
        return LocalResponse(handler(schema.model_validate(json)))
    except HTTPException as exc:
        return LocalResponse({"detail": exc.detail}, exc.status_code)
    except ValidationError as exc:
        return LocalResponse({"detail": str(exc)}, 422)
    except (RuntimeError, OSError, ValueError) as exc:
        return LocalResponse({"detail": f"Local service unavailable: {exc}"}, 503)
