"""Player signals: what has changed since you last looked.

Press "Refresh signals" (for example on Friday after the press conferences) to:
  * re-pull the latest injury flags, chance of playing and news from FPL
  * compare them with your previous refresh (saved to signals_snapshot.json)
  * check your squad's recent minutes for rotation risk
  * scan public football headlines for mentions of your players
  * show transfer momentum and likely price changes
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals_snapshot.json")
FEEDS = [
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml"),
    ("The Guardian", "https://www.theguardian.com/football/premierleague/rss"),
]
STATUS = {"a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended", "u": "Unavailable", "n": "Not available"}
UA = "FPL-Coach/1.0 (local assistant)"


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _chance(p: Dict[str, Any]) -> int:
    c = p.get("chance_of_playing_next_round")
    if c is None:
        return 100 if (p.get("status") or "a") == "a" else 0
    return int(c)


def _load_snapshot(team_id: int) -> Optional[Dict[str, Any]]:
    try:
        with open(SNAPSHOT, "r", encoding="utf-8") as fh:
            return json.load(fh).get(str(team_id))
    except (OSError, ValueError):
        return None


def _save_snapshot(team_id: int, snap: Dict[str, Any]) -> None:
    try:
        with open(SNAPSHOT, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    data[str(team_id)] = snap
    tmp = SNAPSHOT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, SNAPSHOT)


def fetch_headlines(timeout: float = 10.0) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for source, url in FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                root = ET.fromstring(resp.read())
        except Exception:  # feeds are a bonus; never fail the page over them
            continue
        for it in root.iter("item"):
            items.append({
                "source": source,
                "title": (it.findtext("title") or "").strip(),
                "summary": re.sub(r"<[^>]+>", "", it.findtext("description") or "").strip()[:300],
                "link": (it.findtext("link") or "").strip(),
                "published": (it.findtext("pubDate") or "").strip(),
            })
    return items


def _name_patterns(p: Dict[str, Any]) -> List[re.Pattern]:
    names = {p.get("web_name") or "", p.get("second_name") or ""}
    pats = []
    for n in names:
        n = n.strip()
        if len(n) >= 4:
            pats.append(re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE))
    return pats


def minutes_trend(summary: Dict[str, Any]) -> Dict[str, Any]:
    hist = [h for h in summary.get("history") or []]
    last = hist[-5:]
    mins = [int(_f(h.get("minutes"))) for h in last]
    starts = sum(1 for h in last if int(_f(h.get("starts"))) == 1 or (h.get("starts") is None and _f(h.get("minutes")) >= 60))
    out = {"last5_minutes": mins, "starts_last5": starts, "games": len(last), "flag": ""}
    if not last:
        return out
    if mins[-1] == 0 and sum(mins[:-1]) > 0:
        out["flag"] = "Didn't play last match"
    elif 0 < mins[-1] < 60 and len(mins) > 1 and mins[-2] >= 60:
        out["flag"] = f"Only {mins[-1]} mins last match"
    elif len(last) >= 4 and starts <= len(last) - 2:
        out["flag"] = f"Rotation risk: started {starts} of the last {len(last)}"
    return out


def build_signals(team_id: int, bootstrap: Dict[str, Any], squad_ids: List[int],
                  element_summary: Callable[[int], Dict[str, Any]], with_headlines: bool = True) -> Dict[str, Any]:
    elements = {p["id"]: p for p in bootstrap.get("elements") or []}
    teams = {t["id"]: t for t in bootstrap.get("teams") or []}
    squad = set(squad_ids)
    now = datetime.now(timezone.utc)
    prev = _load_snapshot(team_id)
    prev_players = (prev or {}).get("players", {})

    def label(p: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": p["id"], "name": p.get("web_name"), "team": teams.get(p["team"], {}).get("short_name", "?"),
                "pos": {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(p.get("element_type")),
                "price": round(_f(p.get("now_cost")) / 10, 1), "selected": _f(p.get("selected_by_percent")),
                "mine": p["id"] in squad, "status": p.get("status") or "a", "chance": _chance(p), "news": p.get("news") or ""}

    # 1) Changes since last refresh
    changes = []
    if prev_players:
        for pid, p in elements.items():
            old = prev_players.get(str(pid))
            if not old:
                continue
            cur_ch, old_ch = _chance(p), old.get("chance", 100)
            msgs = []
            kind = "info"
            if cur_ch != old_ch:
                if cur_ch < old_ch:
                    msgs.append(f"Chance of playing down {old_ch}% → {cur_ch}%")
                    kind = "bad"
                else:
                    msgs.append(f"Chance of playing up {old_ch}% → {cur_ch}%")
                    kind = "good"
            elif (p.get("status") or "a") != old.get("status"):
                msgs.append(f"{STATUS.get(old.get('status'), old.get('status'))} → {STATUS.get(p.get('status') or 'a')}")
                kind = "bad" if (p.get("status") or "a") != "a" else "good"
            if (p.get("news") or "") != old.get("news", "") and p.get("news"):
                msgs.append(f"News: {p.get('news')}")
            dc = int(_f(p.get("now_cost"))) - int(old.get("cost", p.get("now_cost")))
            if dc:
                msgs.append(f"Price {'+' if dc > 0 else ''}{dc / 10:.1f}m")
                if kind == "info":
                    kind = "good" if dc > 0 else "bad"
            if msgs and (p["id"] in squad or _f(p.get("selected_by_percent")) >= 3 or dc):
                changes.append({**label(p), "kind": kind, "changes": msgs})
        changes.sort(key=lambda c: (not c["mine"], {"bad": 0, "good": 1, "info": 2}[c["kind"]], -c["selected"]))

    # 2) Fresh news (last 7 days) from FPL's own flags
    fresh = []
    for p in elements.values():
        if not p.get("news"):
            continue
        added = p.get("news_added")
        try:
            dt = datetime.fromisoformat(str(added).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if now - dt <= timedelta(days=7) and (p["id"] in squad or _f(p.get("selected_by_percent")) >= 2):
            fresh.append({**label(p), "added": dt.isoformat()})
    fresh.sort(key=lambda x: (not x["mine"], x["added"]), reverse=False)
    fresh.sort(key=lambda x: x["mine"], reverse=True)

    # 3) Your squad: status, minutes trend
    squad_rows = []
    for pid in squad_ids:
        p = elements.get(pid)
        if not p:
            continue
        try:
            trend = minutes_trend(element_summary(pid))
        except Exception:
            trend = {"last5_minutes": [], "flag": "", "starts_last5": None, "games": 0}
        level = "ok"
        if (p.get("status") or "a") in ("i", "s", "u", "n") or _chance(p) <= 25:
            level = "bad"
        elif (p.get("status") or "a") == "d" or trend["flag"]:
            level = "warn"
        squad_rows.append({**label(p), **trend, "level": level})
    squad_rows.sort(key=lambda r: {"bad": 0, "warn": 1, "ok": 2}[r["level"]])

    # 4) Transfer momentum
    by_in = sorted(elements.values(), key=lambda p: _f(p.get("transfers_in_event")), reverse=True)[:10]
    by_out = sorted(elements.values(), key=lambda p: _f(p.get("transfers_out_event")), reverse=True)[:10]
    momentum = {
        "in": [{**label(p), "count": int(_f(p.get("transfers_in_event")))} for p in by_in],
        "out": [{**label(p), "count": int(_f(p.get("transfers_out_event")))} for p in by_out],
    }

    # 5) Headlines mentioning your players
    headlines = []
    feed_ok = False
    if with_headlines:
        items = fetch_headlines()
        feed_ok = bool(items)
        watch = [elements[i] for i in squad_ids if i in elements]
        watch += [p for p in elements.values() if _f(p.get("selected_by_percent")) >= 15 and p["id"] not in squad]
        pats = [(p, _name_patterns(p)) for p in watch]
        for it in items:
            text = f"{it['title']} {it['summary']}"
            hits = [p for p, ps in pats if any(rx.search(text) for rx in ps)]
            if hits:
                headlines.append({**it, "players": [label(p) for p in hits[:4]], "mine": any(p["id"] in squad for p in hits)})
        headlines.sort(key=lambda h: not h["mine"])

    snap = {
        "taken": now.isoformat(),
        "players": {str(pid): {"status": p.get("status") or "a", "chance": _chance(p), "news": p.get("news") or "",
                               "cost": int(_f(p.get("now_cost")))} for pid, p in elements.items()},
    }
    _save_snapshot(team_id, snap)
    return {
        "refreshed": now.isoformat(),
        "previous": (prev or {}).get("taken"),
        "changes": changes[:60],
        "fresh_news": fresh[:40],
        "squad": squad_rows,
        "momentum": momentum,
        "headlines": headlines[:30],
        "feeds_reachable": feed_ok,
    }
