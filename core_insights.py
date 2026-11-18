"""Match-level data from FPL-Core-Insights (github.com/olbauday/FPL-Core-Insights).

Free, updated twice a day, keyed by official FPL IDs. We use it for:
  * team xG for and against in every league match (better fixture difficulty)
  * each player's recent matches: minutes, starts, xG, xA and defensive actions
    (a true form signal, rather than FPL's points-based "form")

Data source credit: FPL-Core-Insights by olbauday.
"""
from __future__ import annotations

import csv
import io
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional

RAW = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"
UA = "FPL-Coach/1.2 (personal use)"
TTL = 3600.0
RECENT_MATCHES = 6
_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def _f(x: Any) -> float:
    try:
        return float(x) if x not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _get_csv(path: str) -> Optional[List[Dict[str, str]]]:
    url = f"{RAW}/{urllib.parse.quote(path)}"
    now = time.time()
    with _lock:
        hit = _cache.get(url)
        if hit and now - hit[0] < TTL:
            return hit[1]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            rows = list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))
    except Exception:
        rows = None
    with _lock:
        _cache[url] = (now, rows)
    return rows


def season_folder(bootstrap: Dict[str, Any]) -> str:
    events = bootstrap.get("events") or []
    year = int(str(events[0].get("deadline_time", "2026"))[:4]) if events else 2026
    return f"{year}-{year + 1}"


def load(bootstrap: Dict[str, Any], workers: int = 6) -> Dict[str, Any]:
    """Fetch league matches and player match stats for every finished gameweek."""
    from concurrent.futures import ThreadPoolExecutor

    events = bootstrap.get("events") or []
    gws = [e["id"] for e in events if e.get("finished") or e.get("is_current")]
    season = season_folder(bootstrap)
    code_to_id = {t.get("code"): t["id"] for t in bootstrap.get("teams") or []}
    elements = {p["id"]: p for p in bootstrap.get("elements") or []}

    def one(gw: int):
        base = f"{season}/By Gameweek/GW{gw}/"
        return gw, _get_csv(base + "matches.csv"), _get_csv(base + "playermatchstats.csv")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, gws))

    team_matches: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    player_matches: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    kickoff: Dict[str, str] = {}
    loaded = 0
    for gw, matches, pms in results:
        if not matches:
            continue
        loaded += 1
        for m in matches:
            if m.get("tournament") != "prem" or str(m.get("finished")).lower() != "true":
                continue
            try:
                h = code_to_id[int(float(m["home_team"]))]
                a = code_to_id[int(float(m["away_team"]))]
            except (KeyError, ValueError):
                continue
            hx, ax = _f(m.get("home_expected_goals_xg")), _f(m.get("away_expected_goals_xg"))
            hs, as_ = _f(m.get("home_score")), _f(m.get("away_score"))
            ko = m.get("kickoff_time") or ""
            kickoff[m["match_id"]] = ko
            team_matches[h].append({"gw": gw, "kickoff": ko, "home": True, "opp": a, "xgf": hx, "xga": ax, "gf": hs, "ga": as_})
            team_matches[a].append({"gw": gw, "kickoff": ko, "home": False, "opp": h, "xgf": ax, "xga": hx, "gf": as_, "ga": hs})
        for r in pms or []:
            if "-prem-" not in (r.get("match_id") or ""):
                continue
            try:
                pid = int(r["player_id"])
            except (KeyError, ValueError):
                continue
            if pid not in elements:
                continue
            mins = _f(r.get("minutes_played"))
            pos = elements[pid].get("element_type")
            cbit = _f(r.get("clearances")) + _f(r.get("blocks")) + _f(r.get("interceptions")) + _f(r.get("tackles"))
            cbirt = cbit + _f(r.get("recoveries"))
            player_matches[pid].append({
                "gw": gw, "match_id": r["match_id"], "kickoff": kickoff.get(r["match_id"], ""),
                "minutes": mins, "started": mins > 0 and _f(r.get("start_min")) == 0,
                "xg": _f(r.get("xg")), "xa": _f(r.get("xa")), "shots": _f(r.get("total_shots")),
                "box_touches": _f(r.get("touches_opposition_box")),
                "defcon": cbit if pos == 2 else cbirt,
            })
    for lst in list(team_matches.values()) + list(player_matches.values()):
        lst.sort(key=lambda x: (x["gw"], x["kickoff"]))
    return {"team_matches": dict(team_matches), "player_matches": dict(player_matches),
            "gameweeks_loaded": loaded, "gameweeks_expected": len(gws), "season": season}


def recent_summary(core: Optional[Dict[str, Any]], pid: int, n: int = RECENT_MATCHES) -> Optional[Dict[str, Any]]:
    """A player's last n league matches (including ones they missed through selection)."""
    if not core:
        return None
    ms = (core.get("player_matches") or {}).get(pid)
    if not ms:
        return None
    last = ms[-n:]
    mins = sum(m["minutes"] for m in last)
    return {
        "games": len(last),
        "starts": sum(1 for m in last if m["started"]),
        "minutes": [int(m["minutes"]) for m in last],
        "mins_total": mins,
        "xg": sum(m["xg"] for m in last),
        "xa": sum(m["xa"] for m in last),
        "shots": sum(m["shots"] for m in last),
        "defcon": [m["defcon"] for m in last if m["minutes"] >= 60],
    }
