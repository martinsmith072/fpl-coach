#!/usr/bin/env python3
"""FPL Coach: local dashboard backed by the official Fantasy Premier League API.

Run:  python3 server.py      then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from fpl_engine import build_advice
from signals import build_signals

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
FPL = "https://fantasy.premierleague.com/api"
UA = "Mozilla/5.0 (compatible; FPL-Coach/1.1; personal use)"
CACHE_TTL = 120.0
_cache: Dict[str, Tuple[float, Any]] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def fpl_get(path: str, ttl: float = CACHE_TTL) -> Any:
    url = f"{FPL}{path}"
    now = time.time()
    with _lock:
        hit = _cache.get(url)
        if hit and now - hit[0] < ttl:
            if isinstance(hit[1], Exception):
                raise hit[1]
            return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 503:
            err = RuntimeError("FPL is updating (usually between gameweeks). Try again in a few minutes.")
        else:
            err = RuntimeError(f"FPL {exc.code} for {path}: {body[:180]}")
        if exc.code == 404:  # remember misses briefly so we don't hammer the API
            with _lock:
                _cache[url] = (now, err)
        raise err from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach FPL ({exc.reason}). Check your connection.") from exc
    with _lock:
        _cache[url] = (now, data)
    return data


def _deadline_passed(event: Dict[str, Any]) -> bool:
    try:
        dl = datetime.fromisoformat(str(event.get("deadline_time")).replace("Z", "+00:00"))
    except ValueError:
        return False
    return dl <= datetime.now(timezone.utc)


def load_picks(team_id: int, bootstrap: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    events = bootstrap.get("events") or []
    notes: List[str] = []
    gw_ids: List[int] = []
    for e in events:
        if e.get("is_next") and _deadline_passed(e):
            gw_ids.append(e["id"])
    for e in events:
        if e.get("is_current"):
            gw_ids.append(e["id"])
    for e in reversed(events):
        if e.get("finished") or e.get("is_previous"):
            gw_ids.append(e["id"])
            break
    gw_ids = list(dict.fromkeys(gw_ids))

    picks: Optional[Dict[str, Any]] = None
    last_err: Optional[Exception] = None
    for gid in gw_ids:
        try:
            picks = fpl_get(f"/entry/{team_id}/event/{gid}/picks/", ttl=60)
            if picks and picks.get("picks"):
                break
        except RuntimeError as exc:
            last_err = exc
    if not picks or not picks.get("picks"):
        raise RuntimeError(str(last_err) if last_err else "Could not load your picks. Has the season started, and is the team ID right?")

    # A Free Hit squad reverts afterwards, so advise on the squad you'll actually have.
    if picks.get("active_chip") == "freehit":
        gw = (picks.get("entry_history") or {}).get("event")
        if gw and gw > 1:
            try:
                prev = fpl_get(f"/entry/{team_id}/event/{gw - 1}/picks/", ttl=300)
                if prev.get("picks"):
                    prev = dict(prev)
                    prev["entry_history"] = picks.get("entry_history")  # keep the latest bank
                    picks = prev
                    notes.append(f"You played Free Hit in GW{gw}, so advice uses your GW{gw - 1} squad (the one that comes back).")
            except RuntimeError:
                notes.append("You played Free Hit this week. Advice may show the temporary squad.")
    return picks, notes


def load_team(team_id: int) -> Dict[str, Any]:
    bootstrap = fpl_get("/bootstrap-static/", ttl=300)
    fixtures = fpl_get("/fixtures/", ttl=300)
    entry = fpl_get(f"/entry/{team_id}/", ttl=60)
    history = fpl_get(f"/entry/{team_id}/history/", ttl=60)
    try:
        transfers = fpl_get(f"/entry/{team_id}/transfers/", ttl=300)
    except RuntimeError:
        transfers = None
    picks, notes = load_picks(team_id, bootstrap)
    return build_advice(bootstrap, fixtures, entry, history, picks, transfers=transfers, notes=notes)


def load_signals(team_id: int, refresh: bool) -> Dict[str, Any]:
    if refresh:
        clear_cache()
    bootstrap = fpl_get("/bootstrap-static/", ttl=300)
    picks, _ = load_picks(team_id, bootstrap)
    squad = [p["element"] for p in picks.get("picks") or []]
    return build_signals(team_id, bootstrap, squad, lambda pid: fpl_get(f"/element-summary/{pid}/", ttl=900))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=STATIC, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _team_id(self, rest: str) -> Optional[int]:
        if rest.isascii() and rest.isdigit() and 0 < len(rest) < 12:
            return int(rest)
        self._json({"error": "Team ID must be a number (from your FPL URL)."}, 400)
        return None

    def _run(self, fn: Any, *args: Any) -> None:
        try:
            data = fn(*args)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, 502)
            return
        except Exception as exc:  # pragma: no cover
            self._json({"error": f"Unexpected error: {exc}"}, 500)
            return
        self._json(data, 200 if not (isinstance(data, dict) and data.get("error")) else 502)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        if path == "/api/health":
            self._json({"ok": True})
            return
        if path.startswith("/api/team/"):
            tid = self._team_id(path[len("/api/team/"):])
            if tid is not None:
                if query.get("refresh"):
                    clear_cache()
                self._run(load_team, tid)
            return
        if path.startswith("/api/signals/"):
            tid = self._team_id(path[len("/api/signals/"):])
            if tid is not None:
                self._run(load_signals, tid, bool(query.get("refresh")))
            return
        super().do_GET()


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"FPL Coach → http://127.0.0.1:{port}")
    print("Enter your FPL team ID (the number in fantasy.premierleague.com/entry/XXXXX/)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
