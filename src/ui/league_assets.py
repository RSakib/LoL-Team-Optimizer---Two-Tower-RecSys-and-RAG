"""Small, local UI assets and a streaming catalog of recorded champions.

Artwork metadata is not a recommendation source. Only champions actually present
in the configured processed profiles become selectable. No raw matches or models
are loaded by this module.
"""
from __future__ import annotations

from functools import lru_cache
import html
import json
from pathlib import Path
import re
from urllib.parse import quote

from src.config import PROFILE_PATH, PROJECT_ROOT

ASSET_ROOT = PROJECT_ROOT / "assets/league"


@lru_cache(maxsize=1)
def artwork() -> dict:
    path = ASSET_ROOT / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@lru_cache(maxsize=2)
def _recorded_champions(path: str, modified: int, size: int) -> tuple[str, ...]:
    names = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                names.update(name for name, games in row.get("top_champions", {}).items()
                             if isinstance(name, str) and games > 0)
    return tuple(sorted(names, key=lambda name: champion_label(name).casefold()))


def recorded_champions() -> tuple[str, ...]:
    try:
        stat = PROFILE_PATH.stat()
    except FileNotFoundError:
        return ()
    return _recorded_champions(str(PROFILE_PATH), stat.st_mtime_ns, stat.st_size)


def asset_url(relative: str) -> str:
    path = (ASSET_ROOT / relative).resolve()
    if not path.is_relative_to(ASSET_ROOT.resolve()) or not path.is_file():
        return ""
    return "./gradio_api/file=" + quote(path.as_posix(), safe="/:")


def champion_label(name: str) -> str:
    return artwork().get("champions", {}).get(name, {}).get("label", name)


def champion_url(name: str) -> str:
    entry = artwork().get("champions", {}).get(name)
    return asset_url(entry["file"]) if entry else ""


def rank_url(tier: str) -> str:
    entry = artwork().get("ranks", {}).get(tier)
    return asset_url(entry["file"]) if entry else ""


def division_url(division: str) -> str:
    return asset_url(f"divisions/{division}.svg") if division in {"I", "II", "III", "IV"} else ""


def image_tag(url: str, css_class: str, alt: str = "") -> str:
    if not url:
        return ""
    return (f'<img class="{css_class}" src="{html.escape(url, quote=True)}" '
            f'alt="{html.escape(alt, quote=True)}" loading="lazy" decoding="async">')


def champion_html(name: str, *, label: str | None = None) -> str:
    return ('<span class="champion-mention">' + image_tag(champion_url(name), "champion-portrait")
            + html.escape(champion_label(name) if label is None else label) + '</span>')


def rank_html(tier: str | None, division: str | None) -> str:
    tier = tier or "Unavailable"
    return ('<span class="rank-mention">' + image_tag(rank_url(tier), "tier-crest")
            + html.escape(tier.title()) + ' '
            + (image_tag(division_url(division), "division-badge", f"Division {division}")
               if division_url(division) else html.escape(division or "")) + '</span>')


def champion_text_html(text: str) -> str:
    """Escape plain/model text, decorating whole champion names only, never HTML.

    Match exact casing to avoid turning ordinary words ('vi', 'brand') into
    champion claims. Display aliases (Wukong) still map to recorded IDs (MonkeyKing).
    """
    names = {alias: name for name in recorded_champions()
             for alias in (name, champion_label(name))}
    if not names:
        return html.escape(text)
    pattern = re.compile(r"(?<![\w])(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?![\w])")
    parts, start = [], 0
    for match in pattern.finditer(text):
        parts.extend((html.escape(text[start:match.start()]),
                      champion_html(names[match.group()], label=match.group())))
        start = match.end()
    parts.append(html.escape(text[start:]))
    return "".join(parts)


ICON_CSS = """
.champion-mention, .rank-mention { display:inline-flex; align-items:center; gap:6px; vertical-align:middle; }
img.champion-portrait { width:24px; height:24px; object-fit:cover; border:1px solid #8b7443; border-radius:3px; display:inline-block; vertical-align:middle; }
img.tier-crest { width:38px; height:38px; object-fit:contain; display:inline-block; vertical-align:middle; }
img.division-badge { width:30px; height:30px; display:inline-block; vertical-align:middle; }
.lineup-report { white-space:pre-wrap; line-height:1.7; color:#f0e6d2; }
"""
