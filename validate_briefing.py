#!/usr/bin/env python3
"""Check team_news.json before it is committed. Exit code 1 and a list of problems if invalid."""
from __future__ import annotations

import json
import sys

import server


def main(path: str) -> int:
    problems = []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"Cannot read {path}: {exc}")
        return 1
    bootstrap = server.fpl_get("/bootstrap-static/")
    nxt = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    ids = {p["id"]: p["web_name"] for p in bootstrap["elements"]}
    if not nxt or data.get("gameweek") != nxt["id"]:
        problems.append(f"gameweek should be {nxt['id'] if nxt else '?'}, got {data.get('gameweek')}")
    for key in ("generated_at", "summary", "players"):
        if key not in data:
            problems.append(f"missing '{key}'")
    for i, p in enumerate(data.get("players") or []):
        where = f"players[{i}]"
        if p.get("id") not in ids:
            problems.append(f"{where}: unknown FPL id {p.get('id')}")
        elif p.get("name") and p["name"] != ids[p["id"]]:
            problems.append(f"{where}: id {p['id']} is {ids[p['id']]}, not {p['name']}")
        sc = p.get("start_chance")
        if sc is not None and not (isinstance(sc, (int, float)) and 0 <= sc <= 100):
            problems.append(f"{where}: start_chance must be 0-100 or null")
        if p.get("confidence") not in ("high", "medium", "low"):
            problems.append(f"{where}: confidence must be high, medium or low")
        if not p.get("verdict"):
            problems.append(f"{where}: missing verdict")
        for s in p.get("sources") or []:
            if s.get("link") and not str(s["link"]).startswith("http"):
                problems.append(f"{where}: bad source link {s['link']}")
    if problems:
        print("Problems:\n- " + "\n- ".join(problems))
        return 1
    print(f"OK: GW{data['gameweek']}, {len(data.get('players') or [])} players.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "team_news.json"))
