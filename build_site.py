#!/usr/bin/env python3
"""Build the hosted (GitHub Pages) version of FPL Coach.

Fetches your FPL data, runs the analysis and signals, and writes a static site:
    site/index.html
    site/data/team.json
    site/data/signals.json
    site/data/signals_snapshot.json   (used next time to show "what's changed")

Usage:
    python3 build_site.py --team 633619 [--out site] [--prev-snapshot-url URL]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone

import server
import signals

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=int(os.environ.get("FPL_TEAM_ID") or 0))
    ap.add_argument("--out", default=os.path.join(ROOT, "site"))
    ap.add_argument("--prev-snapshot-url", default="")
    args = ap.parse_args()
    if not args.team:
        print("Set your team ID with --team or the FPL_TEAM_ID variable.", file=sys.stderr)
        return 2

    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)
    snap_path = os.path.join(data_dir, "signals_snapshot.json")
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

    team = server.load_team(args.team)
    if team.get("error"):
        print(team["error"], file=sys.stderr)
        return 1
    sig = server.load_signals(args.team, refresh=False)

    with open(os.path.join(data_dir, "team.json"), "w", encoding="utf-8") as fh:
        json.dump(team, fh, default=str)
    with open(os.path.join(data_dir, "signals.json"), "w", encoding="utf-8") as fh:
        json.dump(sig, fh, default=str)
    with open(os.path.join(data_dir, "build.json"), "w", encoding="utf-8") as fh:
        json.dump({"built": datetime.now(timezone.utc).isoformat(), "team": args.team,
                   "trigger": os.environ.get("GITHUB_EVENT_NAME", "manual")}, fh)
    shutil.copy(os.path.join(ROOT, "static", "index.html"), os.path.join(args.out, "index.html"))
    open(os.path.join(args.out, ".nojekyll"), "w").close()
    print(f"Built site for team {args.team}: GW{team['meta']['next_gw']}, {len(team['actions'])} actions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
