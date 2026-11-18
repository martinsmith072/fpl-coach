#!/usr/bin/env python3
"""Build the hosted (GitHub Pages) version of FPL Coach.

Fetches FPL data for each team, runs the analysis and signals, and writes a static site:
    site/index.html
    site/data/teams.json                    (index of teams)
    site/data/teams/<id>/team.json
    site/data/teams/<id>/signals.json
    site/data/signals_snapshot.json         (used next time to show "what's changed")

Usage:
    python3 build_site.py [--teams 633619,258412] [--out site] [--prev-snapshot-url URL]

Teams come from teams.json unless --teams or FPL_TEAM_IDS is given. Each team is written to
site/data/teams/<id>/, with an index at site/data/teams.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from typing import List

import server
import signals

ROOT = os.path.dirname(os.path.abspath(__file__))


def team_ids(arg: str) -> List[int]:
    """Teams to build: --teams / FPL_TEAM_IDS (comma separated), else teams.json, else FPL_TEAM_ID."""
    raw = arg or os.environ.get("FPL_TEAM_IDS", "")
    if not raw:
        try:
            with open(os.path.join(ROOT, "teams.json"), encoding="utf-8") as fh:
                raw = ",".join(str(t) for t in json.load(fh).get("teams", []))
        except (OSError, ValueError):
            raw = os.environ.get("FPL_TEAM_ID", "")
    ids = []
    for part in raw.replace(" ", ",").split(","):
        if part.strip().isdigit() and int(part) not in ids:
            ids.append(int(part))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", default="", help="comma-separated FPL team IDs")
    ap.add_argument("--out", default=os.path.join(ROOT, "site"))
    ap.add_argument("--prev-snapshot-url", default="")
    args = ap.parse_args()
    ids = team_ids(args.teams)
    if not ids:
        print("No team IDs. Add them to teams.json.", file=sys.stderr)
        return 2

    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)
    snap_path = os.path.join(data_dir, "signals_snapshot.json")  # one file, keyed by team ID
    signals.SNAPSHOT = snap_path

    # Bring back last run's snapshot so "what's changed" works between runs.
    if args.prev_snapshot_url:
        try:
            req = urllib.request.Request(args.prev_snapshot_url, headers={"User-Agent": server.UA})
            with urllib.request.urlopen(req, timeout=20) as resp:
                prev = json.loads(resp.read().decode("utf-8"))
            with open(snap_path, "w", encoding="utf-8") as fh:
                json.dump(prev, fh)
            print("Loaded previous snapshot.")
        except Exception as exc:  # first run, or site not published yet
            print(f"No previous snapshot ({exc}); this run starts fresh.")

    index = []
    for tid in ids:
        try:
            team = server.load_team(tid)
            if team.get("error"):
                raise RuntimeError(team["error"])
            sig = server.load_signals(tid, refresh=False)
        except Exception as exc:  # one bad ID shouldn't break the others
            print(f"Team {tid}: skipped ({exc})", file=sys.stderr)
            index.append({"id": tid, "error": str(exc)})
            continue
        tdir = os.path.join(data_dir, "teams", str(tid))
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "team.json"), "w", encoding="utf-8") as fh:
            json.dump(team, fh, default=str)
        with open(os.path.join(tdir, "signals.json"), "w", encoding="utf-8") as fh:
            json.dump(sig, fh, default=str)
        index.append({"id": tid, "team_name": team["manager"]["team_name"], "manager": team["manager"]["manager"],
                      "overall_rank": team["manager"]["overall_rank"]})
        print(f"Built team {tid} ({team['manager']['team_name']}): GW{team['meta']['next_gw']}, {len(team['actions'])} actions.")

    if not any("error" not in t for t in index):
        print("No teams could be built.", file=sys.stderr)
        return 1
    with open(os.path.join(data_dir, "teams.json"), "w", encoding="utf-8") as fh:
        json.dump({"teams": index, "repo": os.environ.get("GITHUB_REPOSITORY", "")}, fh)
    with open(os.path.join(data_dir, "build.json"), "w", encoding="utf-8") as fh:
        json.dump({"built": datetime.now(timezone.utc).isoformat(), "teams": ids,
                   "trigger": os.environ.get("GITHUB_EVENT_NAME", "manual")}, fh)
    shutil.copy(os.path.join(ROOT, "static", "index.html"), os.path.join(args.out, "index.html"))
    open(os.path.join(args.out, ".nojekyll"), "w").close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
