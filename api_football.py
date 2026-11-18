"""Injuries and suspensions from API-Football (api-sports.io).

A second opinion on who will miss the next match, alongside FPL's own flags. Needs a free
API key in the API_FOOTBALL_KEY environment variable (a GitHub secret when hosted).
Each update uses 1-2 requests of the free plan's ~100 a day.
"""
from __future__ import annotations

import difflib
import json
import os
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

BASE = "https://v3.football.api-sports.io"
PREMIER_LEAGUE = 39
TTL = 1800.0
_cache: Dict[str, Any] = {}

TEAM_ALIASES = {
    "manchester united": "man utd", "manchester city": "man city", "tottenham": "spurs",
    "tottenham hotspur": "spurs", "nottingham forest": "nott'm forest", "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves", "brighton & hove albion": "brighton", "brighton and hove albion": "brighton",
    "newcastle united": "newcastle", "west ham united": "west ham", "leeds united": "leeds",
    "ipswich town": "ipswich", "coventry city": "coventry", "hull city": "hull", "leicester city": "leicester",
    "sheffield united": "sheffield utd", "afc bournemouth": "bournemouth", "luton town": "luton",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", " ").split())


def _get(path: str, params: Dict[str, Any], key: str) -> Dict[str, Any]:
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-apisports-key": key, "Accept": "application/json",
                                               "User-Agent": "FPL-Coach/1.2 (personal use)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _errors(payload: Dict[str, Any]) -> str:
    err = payload.get("errors")
    if isinstance(err, dict) and err:
        return "; ".join(f"{k}: {v}" for k, v in err.items())
    if isinstance(err, list) and err:
        return "; ".join(str(e) for e in err)
    return ""


def fetch(bootstrap: Dict[str, Any], key: Optional[str] = None) -> Dict[str, Any]:
    """Return {"by_player": {fpl_id: {...}}, "status": str, "error": str|None, "requests": int}."""
    key = key or os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        return {"by_player": {}, "status": "not configured", "error": None, "requests": 0}
    events = bootstrap.get("events") or []
    season = int(str(events[0].get("deadline_time", "2026"))[:4]) if events else 2026
    ck = f"inj{season}"
    hit = _cache.get(ck)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]

    requests = 0
    rows: List[Dict[str, Any]] = []
    try:
        page_payload = _get("/injuries", {"league": PREMIER_LEAGUE, "season": season}, key)
        requests += 1
        err = _errors(page_payload)
        if err:
            out = {"by_player": {}, "status": "error", "error": err, "requests": requests}
            _cache[ck] = (time.time(), out)
            return out
        rows = page_payload.get("response") or []
    except urllib.error.HTTPError as exc:
        return {"by_player": {}, "status": "error", "error": f"HTTP {exc.code}", "requests": requests}
    except Exception as exc:  # network, JSON, etc.
        return {"by_player": {}, "status": "error", "error": str(exc)[:160], "requests": requests}

    # Only entries for upcoming fixtures matter (the feed also lists past matches).
    now = datetime.now(timezone.utc)
    nxt = next((e for e in events if e.get("is_next")), None)
    try:
        start = datetime.fromisoformat(str(nxt["deadline_time"]).replace("Z", "+00:00")) if nxt else now
    except (TypeError, ValueError):
        start = now
    end = start + timedelta(days=8)

    teams = bootstrap.get("teams") or []
    fpl_team_names = {_norm(t["name"]): t["id"] for t in teams}
    elements = bootstrap.get("elements") or []
    by_team: Dict[int, List[Dict[str, Any]]] = {}
    for p in elements:
        by_team.setdefault(p["team"], []).append(p)

    def team_id(api_name: str) -> Optional[int]:
        n = _norm(api_name)
        n = TEAM_ALIASES.get(n, n)
        if n in fpl_team_names:
            return fpl_team_names[n]
        m = difflib.get_close_matches(n, list(fpl_team_names), n=1, cutoff=0.6)
        return fpl_team_names[m[0]] if m else None

    def player_id(api_name: str, tid: int) -> Optional[int]:
        n = _norm(api_name)
        parts = n.split()
        surname = parts[-1] if parts else n
        best = None
        for p in by_team.get(tid, []):
            full = _norm(f"{p.get('first_name', '')} {p.get('second_name', '')}")
            web = _norm(p.get("web_name", ""))
            if n == full or n == web:
                return p["id"]
            web_last = web.split()[-1] if web else ""
            if surname and (surname in full.split() or surname == web_last):
                initial_ok = len(parts) < 2 or not parts[0] or full.startswith(parts[0][0])
                if initial_ok:
                    best = best or p["id"]
        return best

    by_player: Dict[int, Dict[str, Any]] = {}
    unmatched = 0
    for r in rows:
        fx = r.get("fixture") or {}
        try:
            when = datetime.fromisoformat(str(fx.get("date")).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if not (start - timedelta(hours=12) <= when <= end):
            continue
        tid = team_id((r.get("team") or {}).get("name", ""))
        pl = r.get("player") or {}
        pid = player_id(pl.get("name", ""), tid) if tid else None
        if not pid:
            unmatched += 1
            continue
        by_player[pid] = {"type": pl.get("type") or "", "reason": pl.get("reason") or "",
                          "date": fx.get("date"), "api_name": pl.get("name")}
    out = {"by_player": by_player, "status": "ok", "error": None, "requests": requests,
           "listed": len(rows), "upcoming": len(by_player), "unmatched": unmatched}
    _cache[ck] = (time.time(), out)
    return out
