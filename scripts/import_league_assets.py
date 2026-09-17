"""Copy only dataset champion portraits and rank crests; never read raw matches.

Run after downloading Riot's ranked-emblems-latest.zip. Missing newer champion
portraits are fetched individually from the dataset's Data Dragon patch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dragon-root", type=Path, required=True)
    parser.add_argument("--rank-zip", type=Path, required=True)
    args = parser.parse_args()
    names = set()
    with (ROOT / "data/processed/player_profiles.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            names.update(json.loads(line).get("top_champions", {}))
    if not names:
        raise ValueError("No recorded champions found")
    source = args.dragon_root / "13.22.1"
    labels = json.loads((source / "data/en_US/champion.json").read_text(encoding="utf-8"))["data"]
    dest = ROOT / "assets/league"
    (dest / "champions").mkdir(parents=True, exist_ok=True)
    manifest = {"champions": {}, "ranks": {}}
    for name in sorted(names):
        if not name.isalnum():
            raise ValueError(f"Unsafe asset ID: {name}")
        local = source / "img/champion" / f"{name}.png"
        target = dest / "champions" / f"{name}.png"
        origin = f"Data Dragon 13.22.1/img/champion/{name}.png"
        if local.is_file():
            shutil.copyfile(local, target)
        else:
            origin = f"https://ddragon.leagueoflegends.com/cdn/15.14.1/img/champion/{name}.png"
            with urllib.request.urlopen(origin, timeout=30) as response, target.open("wb") as output:
                shutil.copyfileobj(response, output)
        if target.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG: {target}")
        manifest["champions"][name] = {"label": labels.get(name, {}).get("name", name),
                                        "file": f"champions/{name}.png", "source": origin}
    (dest / "ranks").mkdir(exist_ok=True)
    with zipfile.ZipFile(args.rank_zip) as archive:
        for tier in ("Platinum", "Emerald", "Diamond", "Master", "Grandmaster", "Challenger"):
            entry = f"Ranked Emblems Latest/Rank={tier}.png"
            with archive.open(entry) as source_image, (dest / "ranks" / f"{tier.upper()}.png").open("wb") as output:
                shutil.copyfileobj(source_image, output)
            manifest["ranks"][tier.upper()] = {"file": f"ranks/{tier.upper()}.png"}
    (dest / "catalog.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(names)} real champion portraits and 6 rank crests")


if __name__ == "__main__":
    main()
