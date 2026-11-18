#!/usr/bin/env python3
"""Collect recent football news for the players and clubs that matter to our teams.

Writes one JSON file with headlines grouped by player and club, for Claude to read and turn
into a team-news briefing (team_news.json). Sources: Google News search feeds (which include
national press, club and fan sites, and press-conference reports) plus BBC Sport and the Guardian.

Usage:
    python3 collect_news.py --out news.json [--teams 633619,258412] [--force]

Without --force it only collects when the next FPL deadline is within NEWS_WINDOW_HOURS.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import server
from build_site import team_ids

UA = "Mozilla/5.0 (compatible; FPL-Coach/1.3; personal use)"
NEWS_WINDOW_HOURS = 36
MAX_PER_QUERY = 8
FEEDS = [
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml"),
    ("The Guardian", "https://www.theguardian.com/football/premierleague/rss"),
]
CLUB_SEARCH = {  # how clubs are usually written in headlines
    "Man Utd": "Manchester United", "Man City": "Manchester City", "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest", "Wolves": "Wolves",
}


def _fetch_rss(url: str, source: Optional[str] = None, timeout: float = 15.0) -> List[Dict[str, Any]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except Exception:
        return []
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = source or (it.findtext("source") or "").strip()
        if not source and " - " in title:  # Google News appends " - Publisher"
            title, _, pub = title.rpartition(" - ")
            src = src or pub
        try:
            when = parsedate_to_datetime(it.findtext("pubDate") or "").astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            when = None
        desc = re.sub(r"<[^>]+>", " ", it.findtext("description") or "")
        out.append({"title": title, "source": src, "link": (it.findtext("link") or "").strip(),
                    "published": when, "snippet": " ".join(desc.split())[:280]})
    return out


def google_news(query: str, days: int = 4) -> List[Dict[str, Any]]:
    q = urllib.parse.quote(f"{query} when:{days}d")
    return _fetch_rss(f"https://news.google.com/rss/search?q={q}&hl=en-GB&gl=GB&ceid=GB:en")[:MAX_PER_QUERY]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="news.json")
    ap.add_argument("--teams", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    bootstrap = server.fpl_get("/bootstrap-static/")
    nxt = next((e for e in bootstrap.get("events") or [] if e.get("is_next")), None)
    if not nxt:
        print("No upcoming gameweek; nothing to collect.")
        return 0
    deadline = datetime.fromisoformat(str(nxt["deadline_time"]).replace("Z", "+00:00"))
    hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
    if not args.force and not (0 < hours <= NEWS_WINDOW_HOURS):
        print(f"Next deadline (GW{nxt['id']}) is {hours:.0f}h away; collecting only within {NEWS_WINDOW_HOURS}h. Skipping.")
        return 0

    elements = {p["id"]: p for p in bootstrap["elements"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}

    # Players that matter: both squads, planned/suggested transfers, and anyone at risk of losing a place.
    focus: Dict[int, str] = {}
    managers = []
    for tid in team_ids(args.teams):
        try:
            adv = server.load_team(tid)
        except Exception as exc:
            print(f"Team {tid}: skipped ({exc})", file=sys.stderr)
            continue
        managers.append(adv["manager"]["team_name"])
        for p in adv["squad"]:
            focus.setdefault(p["id"], "squad")
        mv = adv["transfers"]["plan"].get("moves") or {}
        for o in (mv.get("in") or []):
            focus.setdefault(o["id"], "planned transfer")
        for s in adv["transfers"]["singles"][:6]:
            for o in s["in"]:
                focus.setdefault(o["id"], "transfer option")
        for c in adv.get("competition") or []:
            focus.setdefault(c["id"], "competition for place")
        for c in adv["captaincy"][:3]:
            focus.setdefault(c["id"], "captain option")

    def player_query(pid: int) -> str:
        p = elements[pid]
        club = teams[p["team"]]["name"]
        club = CLUB_SEARCH.get(club, club)
        return f'"{p["web_name"]}" {club}'

    clubs = sorted({elements[pid]["team"] for pid in focus})
    jobs = [("player", pid, player_query(pid)) for pid in focus]
    for cid in clubs:
        club = CLUB_SEARCH.get(teams[cid]["name"], teams[cid]["name"])
        jobs.append(("club", cid, f'"{club}" team news OR "press conference" OR injury'))

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda j: (j, google_news(j[2])), jobs))
    general = []
    for name, url in FEEDS:
        general += _fetch_rss(url, name)[:25]

    out_players, out_clubs = {}, {}
    total = 0
    for (kind, key, query), items in results:
        total += len(items)
        if kind == "player":
            p = elements[key]
            out_players[str(key)] = {
                "name": p["web_name"], "full_name": f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
                "team": teams[p["team"]]["short_name"], "why": focus[key], "fpl_status": p.get("status"),
                "fpl_chance": p.get("chance_of_playing_next_round"), "fpl_news": p.get("news") or "",
                "headlines": items,
            }
        else:
            out_clubs[teams[key]["short_name"]] = {"name": teams[key]["name"], "headlines": items}

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "gameweek": nxt["id"], "deadline": nxt["deadline_time"], "teams": managers,
        "players": out_players, "clubs": out_clubs, "general": general,
        "counts": {"players": len(out_players), "clubs": len(out_clubs), "headlines": total + len(general)},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"Collected {payload['counts']['headlines']} headlines for {len(out_players)} players and {len(out_clubs)} clubs "
          f"(GW{nxt['id']}, deadline in {hours:.0f}h) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
