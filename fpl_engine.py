"""FPL Coach engine.

Turns raw Fantasy Premier League API data into advice:
  * expected points (xPts) per player per gameweek, using season xG/xA, minutes,
    availability, form and opponent strength
  * the best starting XI, bench order and captain
  * transfer suggestions (single, double, and with a -4 hit) within budget and club limits
  * fixture difficulty for every team
  * when to play each chip (Wildcard, Free Hit, Bench Boost, Triple Captain)
  * extras: injury and news flags, captaincy shortlist, price-change watch, history

All numbers are estimates. The model is deliberately simple and transparent, not a
crystal ball.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_COUNT = {1: 2, 2: 5, 3: 5, 4: 3}
PLAY_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
PLAY_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3

HORIZON = 6  # gameweeks looked ahead
WEIGHTS = [1.0, 0.92, 0.85, 0.78, 0.72, 0.66]  # later weeks are less certain
TRANSFER_HORIZON = 5  # gameweeks used to judge a transfer
HIT_COST = 4
ROLL_VALUE = 1.5  # rough value of banking a free transfer
FODDER_GK_MAX = 45  # £4.5m: a backup keeper this cheap is assumed to be bench fodder

CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
    "manager": "Assistant Manager",
}

# Position priors (per 90) used to steady small samples.
PRIOR_XG = {1: 0.0, 2: 0.04, 3: 0.11, 4: 0.30}
PRIOR_XA = {1: 0.01, 2: 0.06, 3: 0.12, 4: 0.10}
PRIOR_BONUS = {1: 0.20, 2: 0.20, 3: 0.25, 4: 0.30}
DEFCON_THRESHOLD = {2: 10.0, 3: 12.0, 4: 12.0}


# --------------------------------------------------------------------------- helpers

def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _price(cost: Any) -> float:
    return round(_f(cost) / 10.0, 1)


def _r(x: float, n: int = 2) -> float:
    return round(float(x), n)


# --------------------------------------------------------------------------- scoring

def scoring_rules(bootstrap: Dict[str, Any]) -> Dict[str, Any]:
    """Points per action. Uses the API's own scoring table when present."""
    rules = {
        "goal": {1: 6, 2: 6, 3: 5, 4: 4},
        "cs": {1: 4, 2: 4, 3: 1, 4: 0},
        "assist": 3,
        "defcon": 2,
    }
    short_to_id = {et.get("singular_name_short"): et.get("id") for et in bootstrap.get("element_types") or []}
    scoring = (bootstrap.get("game_config") or {}).get("scoring") or {}

    def apply(key: str, target: Dict[int, float]) -> None:
        table = scoring.get(key)
        if not isinstance(table, dict):
            return
        for k, v in table.items():
            tid = short_to_id.get(k) or (int(k) if str(k).isdigit() else None)
            if tid in target and isinstance(v, (int, float)):
                target[tid] = v

    apply("goals_scored", rules["goal"])
    apply("clean_sheets", rules["cs"])
    a = scoring.get("assists")
    if isinstance(a, (int, float)):
        rules["assist"] = a
    return rules


# --------------------------------------------------------------------------- fixtures & teams

class FixtureIndex:
    def __init__(self, fixtures: Iterable[Dict[str, Any]]):
        self.by_team_event: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        self.games_played: Counter = Counter()
        self.goals_for: Counter = Counter()
        self.goals_against: Counter = Counter()
        for f in fixtures:
            h, a = f.get("team_h"), f.get("team_a")
            done = bool(f.get("finished") or f.get("finished_provisional"))
            if done and f.get("team_h_score") is not None and f.get("team_a_score") is not None:
                hs, as_ = int(f["team_h_score"]), int(f["team_a_score"])
                self.games_played[h] += 1
                self.games_played[a] += 1
                self.goals_for[h] += hs
                self.goals_against[h] += as_
                self.goals_for[a] += as_
                self.goals_against[a] += hs
            ev = f.get("event")
            if ev is None:
                continue
            self.by_team_event[(h, ev)].append(
                {"opp": a, "home": True, "fdr": f.get("team_h_difficulty"), "kickoff": f.get("kickoff_time"), "finished": done}
            )
            self.by_team_event[(a, ev)].append(
                {"opp": h, "home": False, "fdr": f.get("team_a_difficulty"), "kickoff": f.get("kickoff_time"), "finished": done}
            )

    def get(self, team: int, event: int) -> List[Dict[str, Any]]:
        return self.by_team_event.get((team, event), [])


class TeamModel:
    """Attack strength and defensive weakness per team and venue (1.0 = league average)."""

    def __init__(self, teams: Sequence[Dict[str, Any]], fx: FixtureIndex,
                 elements: Optional[Sequence[Dict[str, Any]]] = None):
        total_goals = sum(fx.goals_for.values())
        total_games = sum(fx.games_played.values())
        mu = total_goals / total_games if total_games >= 20 else 1.4
        self.mu = _clip(mu, 1.0, 1.9)

        # Expected goals for and against each team this season. xG is a steadier guide to
        # strength than actual goals, which swing on luck.
        xgf: Dict[int, float] = defaultdict(float)
        keeper: Dict[int, Tuple[float, float]] = {}
        for p in elements or []:
            xgf[p["team"]] += _f(p.get("expected_goals"))
            if p.get("element_type") == 1 and _f(p.get("minutes")) > keeper.get(p["team"], (0.0, 0.0))[0]:
                keeper[p["team"]] = (_f(p.get("minutes")), _f(p.get("expected_goals_conceded")))
        have_xg = bool(elements) and sum(xgf.values()) > 0

        def mean(keys: Sequence[str]) -> float:
            vals = [_f(t.get(k)) for t in teams for k in keys if _f(t.get(k)) > 0]
            return sum(vals) / len(vals) if vals else 1.0

        att_mean = mean(["strength_attack_home", "strength_attack_away"])
        def_mean = mean(["strength_defence_home", "strength_defence_away"])
        k = 4.0  # light shrinkage towards average: real differences between teams are large
        self.A: Dict[int, Dict[str, float]] = {}
        self.D: Dict[int, Dict[str, float]] = {}
        for t in teams:
            tid = t["id"]
            gp = fx.games_played[tid]
            gf, ga = float(fx.goals_for[tid]), float(fx.goals_against[tid])
            if have_xg and gp:
                gk_min, gk_xgc = keeper.get(tid, (0.0, 0.0))
                xga = gk_xgc * (gp * 90.0 / gk_min) if gk_min >= 90 else ga
                gf = 0.35 * gf + 0.65 * xgf[tid]
                ga = 0.35 * ga + 0.65 * xga
            att_res = ((gf + self.mu * k) / (gp + k)) / self.mu
            dw_res = ((ga + self.mu * k) / (gp + k)) / self.mu
            w_res = _clip(gp / (gp + 4.0), 0.0, 0.85)
            self.A[tid], self.D[tid] = {}, {}
            for venue, hf in (("home", 1.08), ("away", 0.92)):
                sa = _f(t.get(f"strength_attack_{venue}")) or att_mean
                sd = _f(t.get(f"strength_defence_{venue}")) or def_mean
                s_att = (sa / att_mean) ** 3
                s_def = (def_mean / sd) ** 3
                self.A[tid][venue] = w_res * att_res * hf + (1 - w_res) * s_att
                self.D[tid][venue] = w_res * dw_res * (2 - hf) + (1 - w_res) * s_def

    def a_avg(self, tid: int) -> float:
        return (self.A[tid]["home"] + self.A[tid]["away"]) / 2

    def d_avg(self, tid: int) -> float:
        return (self.D[tid]["home"] + self.D[tid]["away"]) / 2

    def fixture(self, team: int, opp: int, home: bool) -> Tuple[float, float]:
        """Return (attack multiplier vs average, expected goals against)."""
        v, ov = ("home", "away") if home else ("away", "home")
        att_mult = (self.A[team][v] * self.D[opp][ov]) / max(self.a_avg(team), 1e-6)
        lam_against = self.mu * self.A[opp][ov] * self.D[team][v]
        return att_mult, lam_against


# --------------------------------------------------------------------------- player projections

class PlayerRates:
    __slots__ = (
        "pos", "start_rate", "sub_rate", "mps", "xg90", "xa90", "bonus90", "saves90",
        "yc90", "p_defcon", "form_mult",
    )

    def __init__(self, p: Dict[str, Any], team_games: int):
        self.pos = p["element_type"]
        mins = _f(p.get("minutes"))
        n90 = mins / 90.0
        tg = max(team_games, 0)
        starts = p.get("starts")
        if tg == 0:
            self.start_rate = 0.75 if _f(p.get("ep_next")) >= 2 else 0.2
            self.mps = 80.0
        else:
            if starts is None:
                self.start_rate = _clip(mins / (90.0 * tg), 0.0, 1.0)
            else:
                self.start_rate = _clip(_f(starts) / tg, 0.0, 1.0)
            st = _f(starts) if starts is not None else mins / 85.0
            self.mps = _clip(mins / st, 55.0, 90.0) if st > 0 else 70.0
        self.sub_rate = min(1 - self.start_rate, 0.3) if mins > 0 else 0.0
        w = 4.0
        self.xg90 = (_f(p.get("expected_goals")) + PRIOR_XG[self.pos] * w) / (n90 + w)
        self.xa90 = (_f(p.get("expected_assists")) + PRIOR_XA[self.pos] * w) / (n90 + w)
        self.bonus90 = (_f(p.get("bonus")) + PRIOR_BONUS[self.pos] * w) / (n90 + w)
        self.saves90 = (_f(p.get("saves")) + 3.0 * w) / (n90 + w) if self.pos == 1 else 0.0
        self.yc90 = _f(p.get("yellow_cards")) / n90 if n90 >= 3 else 0.12
        self.p_defcon = 0.0
        if self.pos in DEFCON_THRESHOLD and p.get("defensive_contribution") is not None and n90 >= 1:
            dc90 = _f(p.get("defensive_contribution")) / n90
            p_hit = 1.0 / (1.0 + math.exp(-(dc90 - DEFCON_THRESHOLD[self.pos]) / 2.0))
            trust = _clip(n90 / 6.0, 0.0, 1.0)
            self.p_defcon = p_hit * trust
        form = _f(p.get("form"))
        ppg = _f(p.get("points_per_game"))
        self.form_mult = 1.0 + 0.15 * _clip((form - ppg) / max(ppg, 2.0), -1.0, 1.0)


def availability(p: Dict[str, Any], k: int) -> float:
    """Chance a player is available k gameweeks after the next one."""
    status = p.get("status") or "a"
    if status in ("u", "n"):
        return 0.0
    c = p.get("chance_of_playing_next_round")
    if c is None:
        c = 100 if status == "a" else (0 if status in ("i", "s") else 75)
    c = _f(c) / 100.0
    if k <= 0:
        return c
    return c + (1 - c) * min(1.0, k / 3.0)


def fixture_points(r: PlayerRates, rules: Dict[str, Any], att_mult: float, lam_against: float,
                   lam_against_avg: float, avail: float) -> float:
    ps = r.start_rate * avail
    psub = r.sub_rate * avail
    fmin = r.mps / 90.0
    pos = r.pos
    attack = r.xg90 * rules["goal"][pos] + r.xa90 * rules["assist"]
    pts = ps * 2.0 + psub * 1.0
    pts += ps * fmin * att_mult * attack + psub * 0.25 * att_mult * attack
    p_cs = math.exp(-lam_against)
    if pos in (1, 2):
        cs_ok = 1.0 if r.mps >= 60 else 0.5
        pts += ps * cs_ok * p_cs * rules["cs"][pos]
        exp_pairs = (lam_against - (1 - math.exp(-2 * lam_against)) / 2) / 2
        pts -= ps * fmin * exp_pairs
    elif pos == 3:
        pts += ps * 0.9 * p_cs * rules["cs"][3]
    if pos == 1:
        pts += ps * (r.saves90 / 3.0) * _clip(lam_against / max(lam_against_avg, 0.3), 0.6, 1.6)
    pts += ps * fmin * r.bonus90 * (0.5 + 0.5 * att_mult)
    pts += ps * r.p_defcon * rules["defcon"]
    pts -= ps * fmin * r.yc90
    return max(pts * r.form_mult, 0.0)


class Projector:
    def __init__(self, bootstrap: Dict[str, Any], fx: FixtureIndex, horizon: List[int],
                 recent: Optional[Dict[int, Dict[str, Any]]] = None):
        self.fx = fx
        self.recent = recent or {}
        self.total_players = max(_f(bootstrap.get("total_players"), 10_000_000), 1.0)
        self.tm = TeamModel(bootstrap.get("teams") or [], fx, bootstrap.get("elements") or [])
        self.rules = scoring_rules(bootstrap)
        self.horizon = horizon
        self.xpts: Dict[int, List[float]] = {}
        self.rates: Dict[int, PlayerRates] = {}
        for p in bootstrap.get("elements") or []:
            self.xpts[p["id"]] = self._project(p)

    def _project(self, p: Dict[str, Any]) -> List[float]:
        team = p["team"]
        r = PlayerRates(p, self.fx.games_played[team])
        # Signal 1: recent minutes. Weight the last few matches equally with the whole season,
        # so a player who has been dropped or rotated is caught quickly.
        rec = self.recent.get(p["id"])
        if rec and rec.get("games", 0) >= 3:
            r.start_rate = 0.5 * r.start_rate + 0.5 * rec["start_rate"]
            r.sub_rate = min(1 - r.start_rate, 0.3) if _f(p.get("minutes")) > 0 else 0.0
        self.rates[p["id"]] = r
        lam_avg = self.tm.mu * self.tm.d_avg(team)
        out: List[float] = []
        for k, ev in enumerate(self.horizon):
            av = availability(p, k)
            total = 0.0
            for g in self.fx.get(team, ev):
                if g["opp"] not in self.tm.A:
                    continue
                att_mult, lam_against = self.tm.fixture(team, g["opp"], g["home"])
                total += fixture_points(r, self.rules, att_mult, lam_against, lam_avg, av)
            out.append(total)
        # Signal 2: transfer momentum. The crowd often reacts to team news before the flags do.
        # Capped at +/-4% so it nudges rather than drives the projection.
        net = _f(p.get("transfers_in_event")) - _f(p.get("transfers_out_event"))
        owners = max(_f(p.get("selected_by_percent")) * self.total_players / 100.0, 20000.0)
        mom = 1.0 + 0.04 * math.tanh(5.0 * net / owners)
        out = [x * mom for x in out]
        # Early in the season, lean on the game's own next-week estimate (ep_next). Later it adds
        # nothing: ep_next is essentially recent form, which the model already uses.
        ep = _f(p.get("ep_next"), -1.0)
        if out and out[0] > 0.2 and ep >= 0 and self.fx.games_played[team] < 4:
            w = 0.6
            scale = _clip(ep / out[0], 0.6, 1.6) if ep > 0 else 0.6
            calib = 1 + w * (scale - 1)
            out = [x * calib for x in out]
        return out


# --------------------------------------------------------------------------- lineup optimisation

FORMATIONS = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3]


def best_xi(squad: Sequence[int], pos: Dict[int, int], pts: Dict[int, float]) -> Dict[str, Any]:
    by_pos: Dict[int, List[int]] = defaultdict(list)
    for pid in squad:
        by_pos[pos[pid]].append(pid)
    for k in by_pos:
        by_pos[k].sort(key=lambda i: pts.get(i, 0.0), reverse=True)
    best: Optional[Tuple[float, Tuple[int, int, int]]] = None
    for d, m, f in FORMATIONS:
        if len(by_pos[2]) < d or len(by_pos[3]) < m or len(by_pos[4]) < f or not by_pos[1]:
            continue
        total = sum(pts.get(i, 0.0) for i in by_pos[1][:1] + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f])
        if best is None or total > best[0] + 1e-9:
            best = (total, (d, m, f))
    if best is None:
        return {"xi": list(squad)[:11], "bench": list(squad)[11:], "total": 0.0, "formation": "?", "captain": None,
                "vice": None, "with_captain": 0.0}
    d, m, f = best[1]
    xi = by_pos[1][:1] + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f]
    bench_out = sorted(by_pos[2][d:] + by_pos[3][m:] + by_pos[4][f:], key=lambda i: pts.get(i, 0.0), reverse=True)
    bench = by_pos[1][1:2] + bench_out
    ranked = sorted(xi, key=lambda i: (pos[i] != 1, pts.get(i, 0.0)), reverse=True)  # keepers last
    cap = ranked[0] if ranked else None
    vice = ranked[1] if len(ranked) > 1 else None
    total = best[0]
    return {
        "xi": xi, "bench": bench, "formation": f"{d}-{m}-{f}", "captain": cap, "vice": vice,
        "total": total, "with_captain": total + (pts.get(cap, 0.0) if cap else 0.0),
        "bench_total": sum(pts.get(i, 0.0) for i in bench),
    }


def squad_value(squad: Sequence[int], pos: Dict[int, int], xpts: Dict[int, List[float]], n: int) -> float:
    """Weighted points of the best XI plus captain over the next n weeks, with a little credit for the bench."""
    total = 0.0
    for k in range(n):
        pts = {i: xpts[i][k] if k < len(xpts[i]) else 0.0 for i in squad}
        res = best_xi(squad, pos, pts)
        total += WEIGHTS[k] * (res["with_captain"] + 0.1 * res.get("bench_total", 0.0))
    return total


# --------------------------------------------------------------------------- selling prices & free transfers

def selling_prices(squad: Sequence[int], elements: Dict[int, Dict[str, Any]],
                   transfers: Optional[List[Dict[str, Any]]], fh_events: set) -> Dict[int, int]:
    """FPL selling price: purchase price plus half of any rise (rounded down); full value of any fall."""
    bought: Dict[int, int] = {}
    for t in sorted(transfers or [], key=lambda t: t.get("event") or 0):
        if t.get("event") in fh_events:
            continue
        bought[t.get("element_in")] = int(_f(t.get("element_in_cost")))
    out: Dict[int, int] = {}
    for pid in squad:
        el = elements[pid]
        now = int(_f(el.get("now_cost")))
        purchase = bought.get(pid, now - int(_f(el.get("cost_change_start"))))
        out[pid] = purchase + (now - purchase) // 2 if now > purchase else now
    return out


def estimate_free_transfers(history: Dict[str, Any], chips_by_event: Dict[int, str]) -> int:
    rows = sorted(history.get("current") or [], key=lambda r: r.get("event") or 0)
    if not rows:
        return 1
    ft = 1
    for row in rows[1:]:
        ev = row.get("event")
        made = int(_f(row.get("event_transfers")))
        if chips_by_event.get(ev) in ("wildcard", "freehit"):
            ft = min(5, ft + 1)
        else:
            ft = min(5, max(ft - made, 0) + 1)
    return ft


# --------------------------------------------------------------------------- transfers

def _club_ok(squad: Sequence[int], elements: Dict[int, Dict[str, Any]]) -> bool:
    c = Counter(elements[i]["team"] for i in squad)
    return all(v <= MAX_PER_CLUB for v in c.values())


def transfer_options(squad: List[int], elements: Dict[int, Dict[str, Any]], pos: Dict[int, int],
                     xpts: Dict[int, List[float]], sell: Dict[int, int], bank: int,
                     pools: Dict[int, List[int]], locked: Optional[set] = None) -> Dict[str, Any]:
    locked = locked or set()
    n = min(TRANSFER_HORIZON, len(next(iter(xpts.values()), [])))
    hsum = {i: sum(WEIGHTS[k] * v for k, v in enumerate(xpts[i][:n])) for i in xpts}
    base = squad_value(squad, pos, xpts, n)
    owned = set(squad)
    team_count = Counter(elements[i]["team"] for i in squad)

    # Singles: screen with a quick estimate, then score properly.
    singles_raw = []
    for out in squad:
        if out in locked:
            continue
        budget = bank + sell[out]
        p_out = pos[out]
        for cand in pools[p_out]:
            if cand in owned or elements[cand]["now_cost"] > budget:
                continue
            t_in = elements[cand]["team"]
            if t_in != elements[out]["team"] and team_count[t_in] >= MAX_PER_CLUB:
                continue
            singles_raw.append((hsum[cand] - hsum[out], out, cand))
    singles_raw.sort(reverse=True)
    singles = []
    seen = set()
    for _, out, cand in singles_raw[:90]:
        new = [cand if i == out else i for i in squad]
        gain = squad_value(new, pos, xpts, n) - base
        singles.append({"out": out, "in": [cand], "outs": [out], "gain": gain,
                        "bank_after": bank + sell[out] - elements[cand]["now_cost"]})
        seen.add((out, cand))
    singles.sort(key=lambda s: s["gain"], reverse=True)

    # Doubles: try every pair of outgoing players, so a downgrade can fund an upgrade.
    doubles_raw = []
    for a_i in range(len(squad)):
        for b_i in range(a_i + 1, len(squad)):
            o1, o2 = squad[a_i], squad[b_i]
            if o1 in locked or o2 in locked:
                continue
            budget = bank + sell[o1] + sell[o2]
            base_h = hsum[o1] + hsum[o2]
            tc = team_count.copy()
            tc[elements[o1]["team"]] -= 1
            tc[elements[o2]["team"]] -= 1
            p1 = [c for c in pools[pos[o1]][:35] if c not in owned]
            p2 = [c for c in pools[pos[o2]][:35] if c not in owned]
            for c1 in p1:
                cost1 = elements[c1]["now_cost"]
                if cost1 > budget:
                    continue
                t1 = elements[c1]["team"]
                if tc[t1] + 1 > MAX_PER_CLUB:
                    continue
                for c2 in p2:
                    if c2 == c1 or cost1 + elements[c2]["now_cost"] > budget:
                        continue
                    t2 = elements[c2]["team"]
                    if tc[t2] + 1 + (1 if t2 == t1 else 0) > MAX_PER_CLUB:
                        continue
                    doubles_raw.append((hsum[c1] + hsum[c2] - base_h, o1, o2, c1, c2))
    doubles_raw.sort(reverse=True)
    doubles = []
    seen2 = set()
    for _, o1, o2, c1, c2 in doubles_raw[:400]:
        key = (frozenset((o1, o2)), frozenset((c1, c2)))
        if key in seen2:
            continue
        seen2.add(key)
        new = [c1 if i == o1 else c2 if i == o2 else i for i in squad]
        gain = squad_value(new, pos, xpts, n) - base
        doubles.append({"outs": [o1, o2], "in": [c1, c2], "gain": gain,
                        "bank_after": bank + sell[o1] + sell[o2] - elements[c1]["now_cost"] - elements[c2]["now_cost"]})
        if len(doubles) >= 60:
            break
    doubles.sort(key=lambda s: s["gain"], reverse=True)
    return {"singles": singles[:12], "doubles": doubles[:8], "horizon": n, "hsum": hsum, "base": base}


def pick_plan(opts: Dict[str, Any], ft: int) -> Dict[str, Any]:
    roll = 0.0 if ft >= 5 else ROLL_VALUE
    plans = [{"kind": "roll", "transfers": 0, "hits": 0, "gain": 0.0, "net": roll, "moves": None}]
    if opts["singles"]:
        s = opts["singles"][0]
        hits = max(0, 1 - ft) * HIT_COST
        plans.append({"kind": "single", "transfers": 1, "hits": hits, "gain": s["gain"],
                      "net": s["gain"] - hits + (roll if ft >= 2 else 0.0), "moves": s})
    if opts["doubles"]:
        d = opts["doubles"][0]
        hits = max(0, 2 - ft) * HIT_COST
        plans.append({"kind": "double", "transfers": 2, "hits": hits, "gain": d["gain"],
                      "net": d["gain"] - hits + (roll if ft >= 3 else 0.0), "moves": d})
    best = max(plans, key=lambda p: p["net"])
    return {"best": best, "plans": plans, "roll_value": roll}


# --------------------------------------------------------------------------- squad builder (Wildcard / Free Hit)

def build_squad(elements: Dict[int, Dict[str, Any]], pos: Dict[int, int], pools: Dict[int, List[int]],
                score: Dict[int, float], budget: int) -> Tuple[List[int], float]:
    """Greedy best 15 for a budget: take the best, then make the cheapest downgrades, then upgrade with spare money."""
    pts = score

    def value(sq: List[int]) -> float:
        r = best_xi(sq, pos, pts)
        return r["with_captain"] + 0.1 * r["bench_total"]

    squad: List[int] = []
    clubs: Counter = Counter()
    for p_id in (1, 2, 3, 4):
        for c in sorted(pools[p_id], key=lambda i: pts.get(i, 0.0), reverse=True):
            if sum(1 for i in squad if pos[i] == p_id) >= SQUAD_COUNT[p_id]:
                break
            if clubs[elements[c]["team"]] >= MAX_PER_CLUB:
                continue
            squad.append(c)
            clubs[elements[c]["team"]] += 1

    def cost(sq: List[int]) -> int:
        return sum(elements[i]["now_cost"] for i in sq)

    for _ in range(40):
        over = cost(squad) - budget
        if over <= 0:
            break
        cur = value(squad)
        best_move = None
        for out in squad:
            clubs = Counter(elements[i]["team"] for i in squad if i != out)
            for c in pools[pos[out]]:
                if c in squad or elements[c]["now_cost"] >= elements[out]["now_cost"]:
                    continue
                if clubs[elements[c]["team"]] >= MAX_PER_CLUB:
                    continue
                saved = elements[out]["now_cost"] - elements[c]["now_cost"]
                new = [c if i == out else i for i in squad]
                loss = cur - value(new)
                ratio = loss / min(saved, over)
                if best_move is None or ratio < best_move[0]:
                    best_move = (ratio, new)
        if best_move is None:
            break
        squad = best_move[1]

    for _ in range(25):
        spare = budget - cost(squad)
        cur = value(squad)
        best_move = None
        for out in squad:
            clubs = Counter(elements[i]["team"] for i in squad if i != out)
            for c in pools[pos[out]]:
                if c in squad or elements[c]["now_cost"] - elements[out]["now_cost"] > spare:
                    continue
                if clubs[elements[c]["team"]] >= MAX_PER_CLUB:
                    continue
                new = [c if i == out else i for i in squad]
                gain = value(new) - cur
                if gain > 0.05 and (best_move is None or gain > best_move[0]):
                    best_move = (gain, new)
        if best_move is None:
            break
        squad = best_move[1]
    return squad, value(squad)


# --------------------------------------------------------------------------- main entry point

def build_advice(bootstrap: Dict[str, Any], fixtures: List[Dict[str, Any]], entry: Dict[str, Any],
                 history: Dict[str, Any], picks: Dict[str, Any],
                 transfers: Optional[List[Dict[str, Any]]] = None,
                 notes: Optional[List[str]] = None,
                 recent: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[str, Any]:
    notes = list(notes or [])
    events = bootstrap.get("events") or []
    ev_by_id = {e["id"]: e for e in events}
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt is None:
        return {"error": "There is no upcoming gameweek. The season may be over."}
    next_gw = nxt["id"]
    last_gw = max(ev_by_id)
    horizon = [g for g in range(next_gw, min(next_gw + HORIZON, last_gw + 1))]

    teams = {t["id"]: t for t in bootstrap.get("teams") or []}
    elements = {p["id"]: p for p in bootstrap.get("elements") or []}
    pos = {i: p["element_type"] for i, p in elements.items()}
    fx = FixtureIndex(fixtures)
    recent = recent or {}
    proj = Projector(bootstrap, fx, horizon, recent)
    xpts = proj.xpts
    total_players = int(_f(bootstrap.get("total_players"), 10_000_000))

    # ---- squad
    raw_picks = sorted(picks.get("picks") or [], key=lambda p: p.get("position", 99))
    squad = [p["element"] for p in raw_picks if p["element"] in elements]
    current_xi = [p["element"] for p in raw_picks if p.get("position", 99) <= 11]
    current_cap = next((p["element"] for p in raw_picks if p.get("is_captain")), None)
    eh = picks.get("entry_history") or {}
    hist_rows = sorted(history.get("current") or [], key=lambda r: r.get("event") or 0)
    bank = int(_f(hist_rows[-1].get("bank") if hist_rows else eh.get("bank")))

    chips_used = history.get("chips") or []
    chips_by_event = {c.get("event"): c.get("name") for c in chips_used}
    fh_events = {c.get("event") for c in chips_used if c.get("name") == "freehit"}
    sell = selling_prices(squad, elements, transfers, fh_events)
    if transfers is None:
        notes.append("Selling prices are estimated from start-of-season prices.")
    ft = estimate_free_transfers(history, chips_by_event)
    squad_sell_value = sum(sell.values())

    def fixtures_for(team: int, gws: Sequence[int]) -> List[List[Dict[str, Any]]]:
        out = []
        for g in gws:
            cell = []
            for f in fx.get(team, g):
                o = teams.get(f["opp"], {})
                cell.append({"opp": o.get("short_name", "?"), "home": f["home"], "fdr": f["fdr"]})
            out.append(cell)
        return out

    def player_view(pid: int) -> Dict[str, Any]:
        p = elements[pid]
        t = teams.get(p["team"], {})
        xs = xpts[pid]
        net = int(_f(p.get("transfers_in_event"))) - int(_f(p.get("transfers_out_event")))
        owners = max(_f(p.get("selected_by_percent")) * total_players / 100.0, 1.0)
        return {
            "id": pid,
            "name": p.get("web_name"),
            "full_name": f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
            "team": t.get("short_name", "?"),
            "team_id": p["team"],
            "pos": POS[p["element_type"]],
            "pos_id": p["element_type"],
            "price": _price(p.get("now_cost")),
            "form": _f(p.get("form")),
            "ppg": _f(p.get("points_per_game")),
            "total_points": int(_f(p.get("total_points"))),
            "minutes": int(_f(p.get("minutes"))),
            "xg": _f(p.get("expected_goals")),
            "xa": _f(p.get("expected_assists")),
            "goals": int(_f(p.get("goals_scored"))),
            "assists": int(_f(p.get("assists"))),
            "selected": _f(p.get("selected_by_percent")),
            "status": p.get("status") or "a",
            "chance": p.get("chance_of_playing_next_round"),
            "news": p.get("news") or "",
            "xpts": [_r(x) for x in xs],
            "xpts_next": _r(xs[0]) if xs else 0.0,
            "xpts_h": _r(sum(xs[:TRANSFER_HORIZON])),
            "fixtures": fixtures_for(p["team"], horizon),
            "net_transfers": net,
            "price_trend": "rising" if net / owners > 0.04 and net > 20000 else "falling" if net / owners < -0.04 and net < -20000 else "",
            "cost_change_event": _price(p.get("cost_change_event")),
            "recent_starts": (f"{recent[pid]['starts']}/{recent[pid]['games']}" if pid in recent else None),
            "momentum": net,
        }

    # ---- lineup for next GW
    next_pts = {i: xpts[i][0] if xpts[i] else 0.0 for i in squad}
    lineup = best_xi(squad, pos, next_pts)
    changes = []
    if current_xi:
        bring_in = [i for i in lineup["xi"] if i not in current_xi]
        drop = [i for i in current_xi if i not in lineup["xi"]]
        for i in bring_in:
            changes.append(f"Start {elements[i]['web_name']} ({next_pts[i]:.1f} xPts)")
        for i in drop:
            changes.append(f"Bench {elements[i]['web_name']} ({next_pts[i]:.1f} xPts)")
        if current_cap and lineup["captain"] and current_cap != lineup["captain"]:
            changes.append(f"Captain {elements[lineup['captain']]['web_name']} instead of {elements[current_cap]['web_name']}")

    # ---- candidate pools per position
    playable = [i for i, p in elements.items() if (p.get("status") or "a") not in ("u", "n")]
    hsum5 = {i: sum(WEIGHTS[k] * v for k, v in enumerate(xpts[i][:TRANSFER_HORIZON])) for i in elements}
    pools: Dict[int, List[int]] = {}
    for pid_pos in (1, 2, 3, 4):
        cands = [i for i in playable if pos[i] == pid_pos]
        by_score = sorted(cands, key=lambda i: hsum5[i], reverse=True)[:45]
        by_value = sorted(cands, key=lambda i: hsum5[i] / max(elements[i]["now_cost"], 35), reverse=True)[:20]
        cheap = sorted([i for i in cands if hsum5[i] > 0.5], key=lambda i: (elements[i]["now_cost"], -hsum5[i]))[:6]
        seen_ids: List[int] = []
        for i in by_score + by_value + cheap:
            if i not in seen_ids:
                seen_ids.append(i)
        pools[pid_pos] = sorted(seen_ids, key=lambda i: hsum5[i], reverse=True)

    # ---- transfers
    # A cheap backup keeper who never plays is deliberate "fodder": don't suggest replacing him.
    fodder = {i for i in squad if pos[i] == 1 and i not in lineup["xi"] and int(_f(elements[i].get("now_cost"))) <= FODDER_GK_MAX}
    for i in fodder:
        notes.append(f"{elements[i]['web_name']} is treated as backup-keeper fodder and won't be suggested as a transfer out.")
    opts = transfer_options(squad, elements, pos, xpts, sell, bank, pools, locked=fodder)
    plan = pick_plan(opts, ft)

    def mv_player(i: int, price: int) -> Dict[str, Any]:
        p = elements[i]
        n = TRANSFER_HORIZON
        cells = fixtures_for(p["team"], horizon[:n])
        fd = [f["fdr"] for c in cells for f in c if f.get("fdr")]
        return {"id": i, "name": p["web_name"], "price": _price(price), "team": teams[p["team"]]["short_name"],
                "pos": POS[p["element_type"]], "xpts_h": _r(sum(xpts[i][:n])), "xpts": [_r(x) for x in xpts[i][:n]],
                "fixtures": cells, "avg_fdr": _r(sum(fd) / len(fd), 2) if fd else None, "form": _f(p.get("form")),
                "status": p.get("status") or "a", "chance": p.get("chance_of_playing_next_round"), "news": p.get("news") or "",
                "selected": _f(p.get("selected_by_percent")),
                "recent_starts": (f"{recent[i]['starts']}/{recent[i]['games']}" if i in recent else None)}

    def move_view(m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "out": [mv_player(i, sell[i]) for i in m["outs"]],
            "in": [mv_player(i, elements[i]["now_cost"]) for i in m["in"]],
            "gws": horizon[:TRANSFER_HORIZON],
            "gain": _r(m["gain"], 1),
            "bank_after": _price(m["bank_after"]),
        }

    # Sell watchlist: injury, poor projection for price, bad fixtures, falling price.
    sell_list = []
    for i in squad:
        if i in fodder:
            continue
        p = elements[i]
        reasons = []
        severity = 0.0
        st = p.get("status") or "a"
        if st in ("i", "s", "u", "n"):
            reasons.append({"i": "Injured", "s": "Suspended", "u": "Unavailable", "n": "Not available"}[st] + (f": {p.get('news')}" if p.get("news") else ""))
            severity += 4
        elif st == "d":
            reasons.append(f"Doubtful ({p.get('chance_of_playing_next_round')}%)" + (f": {p.get('news')}" if p.get("news") else ""))
            severity += 2
        same_pos = [c for c in pools[pos[i]] if c not in squad and elements[c]["now_cost"] <= sell[i] + bank]
        if same_pos:
            alt = max(same_pos, key=lambda c: hsum5[c])
            diff = hsum5[alt] - hsum5[i]
            if diff > 3:
                reasons.append(f"{elements[alt]['web_name']} is projected {diff:.1f} pts better over {TRANSFER_HORIZON} GWs for the money")
                severity += diff / 3
        fdrs = [f["fdr"] for g in horizon[:4] for f in fx.get(p["team"], g) if f.get("fdr")]
        blanks = sum(1 for g in horizon[:4] if not fx.get(p["team"], g))
        if fdrs and sum(fdrs) / len(fdrs) >= 3.75:
            reasons.append(f"Tough run: average difficulty {sum(fdrs) / len(fdrs):.1f} over the next 4")
            severity += 1
        if blanks:
            reasons.append(f"Blank in {blanks} of the next 4 GWs")
            severity += blanks
        rec = recent.get(i)
        if rec and rec.get("games", 0) >= 3 and rec["starts"] <= rec["games"] - 2 and st == "a":
            reasons.append(f"Rotation risk: started {rec['starts']} of the last {rec['games']}")
            severity += 1.5
        net_i = int(_f(p.get("transfers_in_event"))) - int(_f(p.get("transfers_out_event")))
        if net_i < -50000:
            reasons.append(f"Heavily sold this week ({net_i:,} net)")
            severity += 0.5
        if _f(p.get("form")) < 2 and _f(p.get("minutes")) > 0 and pos[i] != 1:
            reasons.append(f"Poor form ({p.get('form')})")
            severity += 0.5
        pv = player_view(i)
        if pv["price_trend"] == "falling":
            reasons.append("Price likely to fall")
            severity += 0.5
        if reasons:
            sell_list.append({"id": i, "name": p["web_name"], "pos": POS[pos[i]], "severity": _r(severity, 1), "reasons": reasons,
                              "xpts_h": pv["xpts_h"], "flagged": st != "a"})
    sell_list.sort(key=lambda s: s["severity"], reverse=True)

    max_sell = {pp: max((sell[i] for i in squad if pos[i] == pp), default=0) for pp in (1, 2, 3, 4)}
    buy = {}
    for pp in (1, 2, 3, 4):
        cands = [i for i in pools[pp] if i not in squad]
        buy[POS[pp]] = {
            "top": [player_view(i) for i in sorted(cands, key=lambda i: hsum5[i], reverse=True)[:8]],
            "affordable": [player_view(i) for i in sorted([c for c in cands if elements[c]["now_cost"] <= max_sell[pp] + bank],
                                                          key=lambda i: hsum5[i], reverse=True)[:8]],
            "differentials": [player_view(i) for i in sorted([c for c in cands if _f(elements[c].get("selected_by_percent")) < 7],
                                                             key=lambda i: hsum5[i], reverse=True)[:6]],
        }

    # ---- chips
    chip_defs = bootstrap.get("chips") or []
    if not chip_defs:
        chip_defs = [{"name": n, "start_event": s, "stop_event": e}
                     for n in ("wildcard", "freehit", "bboost", "3xc") for s, e in ((1, 19), (20, 38))]
    chip_rows = []
    budget_total = squad_sell_value + bank
    wc_cache: Dict[str, Any] = {}

    # Chips are judged over their whole window (up to 9 weeks ahead), so project further than usual.
    CHIP_LOOK = 9
    chip_h = list(range(next_gw, min(last_gw, next_gw + CHIP_LOOK - 1 + TRANSFER_HORIZON - 1) + 1))
    lp = Projector(bootstrap, fx, chip_h, recent).xpts
    chip_weeks = chip_h[:CHIP_LOOK]
    mv_best = plan["best"]["moves"]
    after = list(squad)
    if mv_best:
        after = [mv_best["in"][mv_best["outs"].index(i)] if i in mv_best["outs"] else i for i in squad]

    def squad_fdr(sq: List[int], gws: Sequence[int]) -> Optional[float]:
        f = [g["fdr"] for i in sq for w in gws for g in fx.get(elements[i]["team"], w) if g.get("fdr")]
        return sum(f) / len(f) if f else None

    def wildcard_timing(stop: int) -> List[Tuple[int, float, str, List[int]]]:
        """Gain from a Wildcard played in each week of its window, versus your squad after this week's planned transfers."""
        def val(sq: List[int], s0: int, n: int) -> float:
            t = 0.0
            for k in range(n):
                r = best_xi(sq, pos, {i: lp[i][s0 + k] for i in sq})
                t += WEIGHTS[k] * (r["with_captain"] + 0.1 * r["bench_total"])
            return t

        out = []
        for w in chip_weeks:
            if w > stop:
                break
            s0 = w - next_gw
            n = min(TRANSFER_HORIZON, len(chip_h) - s0)
            if n < 3:
                break
            score = {i: sum(WEIGHTS[k] * lp[i][s0 + k] for k in range(n)) for i in elements}
            wc_pools = {pp: list(dict.fromkeys(sorted([i for i in playable if pos[i] == pp], key=lambda i: score[i], reverse=True)[:40]
                                               + [i for i in pools[pp] if elements[i]["now_cost"] <= 50][:6])) for pp in (1, 2, 3, 4)}
            sq, _ = build_squad(elements, pos, wc_pools, score, budget_total)
            base_sq = squad if w == next_gw else after
            gain = val(sq, s0, n) - val(base_sq, s0, n)
            if w == next_gw:  # this week you could make the planned transfers instead
                gain -= max(0.0, plan["best"]["gain"] - plan["best"]["hits"])
            fd = squad_fdr(after, range(w, w + n))
            out.append((w, gain, f"difficulty {fd:.2f} (next {n})" if fd else "", sq))
        return out

    def freehit_eval(k: int) -> Dict[str, Any]:
        key = f"fh{k}"
        if key not in wc_cache:
            score = {i: lp[i][k] for i in elements}
            fh_pools = {pp: sorted([i for i in playable if pos[i] == pp], key=lambda i: score[i], reverse=True)[:30]
                        + [i for i in pools[pp] if elements[i]["now_cost"] <= 50][:6] for pp in (1, 2, 3, 4)}
            for pp in fh_pools:
                fh_pools[pp] = list(dict.fromkeys(fh_pools[pp]))
            sq, val = build_squad(elements, pos, fh_pools, score, budget_total)
            base_sq = after
            mine = best_xi(base_sq, pos, {i: lp[i][k] for i in base_sq})
            wc_cache[key] = {"squad": sq, "gain": val - (mine["with_captain"] + 0.1 * mine["bench_total"])}
        return wc_cache[key]

    per_gw = []
    for k, g in enumerate(chip_weeks):
        sq = after  # assume this week's recommended transfers are made
        pts = {i: lp[i][k] for i in sq}
        r = best_xi(sq, pos, pts)
        playing = sum(1 for i in sq if fx.get(elements[i]["team"], g))
        doubles = sum(1 for i in sq if len(fx.get(elements[i]["team"], g)) > 1)
        cap_fx = ", ".join(f"{teams[f['opp']]['short_name']} ({'H' if f['home'] else 'A'})"
                           for f in fx.get(elements[r["captain"]]["team"], g)) if r["captain"] else ""
        per_gw.append({"gw": g, "k": k, "xi": r, "bench": r["bench_total"], "cap": pts.get(r["captain"], 0.0) if r["captain"] else 0.0,
                       "cap_id": r["captain"], "cap_fx": cap_fx, "playing": playing, "doubles": doubles,
                       "fdr": squad_fdr(sq, [g])})

    def ctx(x: Dict[str, Any]) -> str:
        bits = []
        if x["fdr"]:
            bits.append(f"difficulty {x['fdr']:.2f}")
        if x["doubles"]:
            bits.append(f"{x['doubles']} doubles")
        if x["playing"] < 15:
            bits.append(f"{15 - x['playing']} blank")
        return " · ".join(bits)

    for cdef in chip_defs:
        name = cdef.get("name")
        if name not in CHIP_LABELS:
            continue
        start, stop = int(_f(cdef.get("start_event"), 1)), int(_f(cdef.get("stop_event"), 38))
        used = next((c for c in chips_used if c.get("name") == name and start <= int(_f(c.get("event"))) <= stop), None)
        row: Dict[str, Any] = {"name": name, "label": CHIP_LABELS[name], "window": [start, stop], "values": []}
        if used:
            row.update(status="used", verdict=f"Played in GW{used.get('event')}", score=-1)
            chip_rows.append(row)
            continue
        if stop < next_gw:
            row.update(status="expired", verdict=f"Window ended GW{stop}", score=-1)
            chip_rows.append(row)
            continue
        if start > chip_weeks[-1]:
            row.update(status="later", verdict=f"Available from GW{start}", score=-1)
            chip_rows.append(row)
            continue
        if name == "manager":
            row.update(status="available", verdict="Not modelled. Check the game's rules for this chip.", score=0)
            chip_rows.append(row)
            continue
        window = [x for x in per_gw if start <= x["gw"] <= stop]
        sees_whole_window = bool(window) and window[-1]["gw"] >= stop
        if name == "bboost":
            vals = [(x["gw"], x["bench"], ctx(x)) for x in window]
            threshold = 10.0
            row["explain"] = "Bench Boost: your 4 substitutes score too, for one gameweek. Value = your bench's projected points that week. Best when everyone has a game, ideally a double."
        elif name == "3xc":
            vals = [(x["gw"], x["cap"], f"{elements[x['cap_id']]['web_name']} v {x['cap_fx']}" if x["cap_id"] else "") for x in window]
            threshold = 9.0
            row["explain"] = "Triple Captain: your captain scores triple instead of double, for one gameweek. Value = the extra points from your best captain that week. Best with a premium player in a double gameweek or a very easy fixture."
        elif name == "freehit":
            raw = [(x, freehit_eval(x["k"])["gain"]) for x in window]
            gains = sorted(g for _, g in raw)
            typical = gains[len(gains) // 2] if gains else 0.0
            vals = [(x["gw"], g - typical + (6.0 if x["playing"] <= 9 else 0.0), ctx(x)) for x, g in raw]
            threshold = 10.0
            row["explain"] = ("Free Hit: unlimited transfers for one gameweek, then your squad reverts. Value = how much better a one-week "
                              "dream team would do than your squad, compared with a normal week. It shines in blank gameweeks, when many of your players don't play.")
        else:  # wildcard
            timing = wildcard_timing(stop) if start <= next_gw <= stop else []
            vals = [(w, g, n) for w, g, n, _ in timing]
            threshold = 20.0
            if timing:
                row["draft"] = [player_view(i) for i in sorted(timing[0][3], key=lambda i: (pos[i], -hsum5[i]))]
            row["explain"] = ("Wildcard: unlimited free transfers for one gameweek, and the new squad is permanent. "
                              f"Value = how much better a rebuild would do over the {TRANSFER_HORIZON} GWs from that week than your squad "
                              "(this week: after your planned free transfers). Waiting costs you the improvement in the weeks in between. "
                              "The model is optimistic about its own picks, so treat the gain as an upper bound.")
        weeks_left = stop - next_gw + 1
        if not vals:
            row.update(status="available", verdict="Hold", score=0)
            chip_rows.append(row)
            continue
        best_gw, best_val, _ = max(vals, key=lambda v: v[1])
        if name in ("bboost", "3xc"):
            # Prefer an earlier week when it is almost as good: less risk of injuries or the window closing.
            best_gw, best_val, _ = next(v for v in vals if v[1] >= best_val - 0.5)
        now_val = vals[0][1] if vals[0][0] == next_gw else 0.0
        if name == "wildcard":
            problems = sum(1 for s in sell_list if s["severity"] >= 3)
            per_week = max(now_val, 0.0) / TRANSFER_HORIZON
            eff = [(g, v - (g - next_gw) * per_week) for g, v, _ in vals]  # waiting forfeits the in-between weeks
            rec_gw = max(eff, key=lambda e: e[1])[0]
            rec_val = dict((g, v) for g, v, _ in vals)[rec_gw]
            later_best = max(((g, v) for g, v, _ in vals if g > next_gw), key=lambda e: e[1], default=(None, 0.0))
            if rec_gw != next_gw and rec_val >= threshold:
                status = "plan"
                verdict = f"Your fixtures swing: a rebuild in GW{rec_gw} projects +{rec_val:.1f} (vs +{now_val:.1f} now), enough to justify waiting"
            elif now_val >= threshold or problems >= 4:
                status = "play"
                swing = ("no fixture swing worth waiting for" if later_best[0] is None or later_best[1] - now_val < 5
                         else f"GW{later_best[0]} is slightly better (+{later_best[1]:.1f}) but not by enough to wait")
                verdict = (f"A rebuild projects +{now_val:.1f} pts more than your free transfers over {TRANSFER_HORIZON} GWs, and {swing}. "
                           f"Each week you wait costs about {per_week:.0f} pts")
            elif weeks_left <= 3:
                status = "urgent"
                verdict = f"Use it or lose it: the window closes after GW{stop}"
            else:
                status = "hold"
                rec_gw = best_gw
                verdict = (f"Optional. A rebuild only beats your free transfers by {best_val:+.1f}; it's worth it from about +{threshold:.0f}. "
                           f"Keep it for injuries or a bad run, and use it by GW{stop}")
        else:
            rec_gw = best_gw
            any_blank_or_double = any(x["playing"] < 15 or x["doubles"] for x in window)
            if name == "freehit" and best_val < threshold and not any_blank_or_double:
                status = "hold"
                verdict = (f"No blank or double gameweeks before GW{min(stop, window[-1]['gw'])}, so there's no ideal Free Hit week. "
                           f"Keep it as emergency cover in case injuries or suspensions pile up. If it's still unused, the best week is GW{best_gw}")
            elif sees_whole_window:
                # Unused chips are wasted when the window closes, so always name the best remaining week.
                if best_gw == next_gw:
                    status = "play"
                    verdict = f"This is your best week left before the chip expires after GW{stop} (+{best_val:.1f} pts)"
                else:
                    status = "plan"
                    strength = "a strong week" if best_val >= threshold else "the best of an average bunch, as no double gameweeks are scheduled yet"
                    verdict = f"Hold until GW{best_gw} (+{best_val:.1f} pts): {strength}. Expires after GW{stop}"
            else:
                if best_gw == next_gw and best_val >= threshold:
                    status, verdict = "play", f"Play this GW (+{best_val:.1f} pts expected)"
                elif best_val >= threshold:
                    status, verdict = "plan", f"Plan for GW{best_gw} (+{best_val:.1f} pts)"
                else:
                    status, verdict = "hold", (f"Best week in view is GW{best_gw} (+{best_val:.1f}), but nothing stands out. "
                                               f"Double gameweeks are usually announced later, so keep it for one")
            if name == "bboost":
                dead = [elements[i]["web_name"] for i in after if i not in per_gw[0]["xi"]["xi"]
                        and (elements[i].get("status") in ("u", "n", "i", "s"))]
                if dead:
                    verdict += f". Note: {', '.join(dead)} would score 0 on your bench; a playing backup adds about 3–4 pts"
            if name == "freehit" and rec_gw == next_gw and now_val >= threshold:
                row["draft"] = [player_view(i) for i in sorted(freehit_eval(0)["squad"], key=lambda i: (pos[i], -lp[i][0]))]
        cover = name == "freehit" and status == "hold" and "emergency cover" in verdict
        row["values"] = [{"gw": g, "value": _r(v, 1), "note": n, "rec": g == rec_gw and not cover} for g, v, n in vals]
        row.update(status=status, verdict=verdict, score=best_val / threshold, best_gw=rec_gw,
                   best_value=_r(dict((g, v) for g, v, _ in vals)[rec_gw], 1),
                   rec_label=(f"Emergency cover · use by GW{stop}" if cover else f"GW{rec_gw}" + (" (this week)" if rec_gw == next_gw else "")))
        chip_rows.append(row)

    # Collapse chip rows: keep the one that matters now for each chip type.
    chip_view: Dict[str, Dict[str, Any]] = {}
    order = {"play": 0, "urgent": 1, "plan": 2, "hold": 3, "available": 4, "later": 5, "used": 6, "expired": 7}
    for row in chip_rows:
        cur = chip_view.get(row["name"])
        if cur is None or order[row["status"]] < order[cur["status"]]:
            row["also"] = (cur or {}).get("also", []) + ([{"window": cur["window"], "status": cur["status"], "verdict": cur["verdict"]}] if cur else [])
            chip_view[row["name"]] = row
        else:
            cur.setdefault("also", []).append({"window": row["window"], "status": row["status"], "verdict": row["verdict"]})

    # ---- fixture ticker
    ticker = []
    for tid, t in teams.items():
        cells = fixtures_for(tid, horizon)
        fdrs = [f["fdr"] for c in cells for f in c if f.get("fdr")]
        att = []
        dfn = []
        for g in horizon:
            for f in fx.get(tid, g):
                am, la = proj.tm.fixture(tid, f["opp"], f["home"])
                att.append(am)
                dfn.append(math.exp(-la))
        games = sum(len(c) for c in cells)
        ticker.append({
            "id": tid, "name": t.get("name"), "short": t.get("short_name"), "cells": cells,
            "avg_fdr": _r(sum(fdrs) / len(fdrs), 2) if fdrs else 5.0,
            "attack": _r(sum(att), 2), "cs": _r(sum(dfn), 2), "games": games,
        })
    ticker.sort(key=lambda r: (r["avg_fdr"] - 0.6 * (r["games"] - len(horizon))))

    # ---- players table
    top_ids = sorted([i for i in playable if _f(elements[i].get("minutes")) > 0 or hsum5[i] > 1],
                     key=lambda i: hsum5[i], reverse=True)[:260]
    players_tbl = [player_view(i) for i in top_ids]

    # ---- history
    avg = {e["id"]: e.get("average_entry_score") for e in events}
    hist_out = [{
        "gw": r.get("event"), "points": r.get("points"), "avg": avg.get(r.get("event")), "rank": r.get("rank"),
        "overall_rank": r.get("overall_rank"), "bench": r.get("points_on_bench"), "hits": r.get("event_transfers_cost"),
        "transfers": r.get("event_transfers"), "chip": chips_by_event.get(r.get("event")), "value": _price(r.get("value")),
    } for r in hist_rows]
    hist_tot = {
        "bench_points": sum(int(_f(r.get("points_on_bench"))) for r in hist_rows),
        "hits": sum(int(_f(r.get("event_transfers_cost"))) for r in hist_rows),
        "above_avg": sum(1 for h in hist_out if h["avg"] is not None and (h["points"] or 0) > h["avg"]),
        "weeks": len(hist_out),
        "past": history.get("past") or [],
    }

    # ---- captaincy
    cap_list = sorted(squad, key=lambda i: next_pts[i], reverse=True)[:5]
    captaincy = []
    for i in cap_list:
        pv = player_view(i)
        captaincy.append({**pv, "eo_note": "Template pick: safe" if pv["selected"] >= 30 else "Differential: high risk, high reward" if pv["selected"] < 10 else ""})

    # ---- price watch
    price_watch = [player_view(i) for i in squad if player_view(i)["price_trend"]]

    # ---- action plan
    actions: List[str] = []
    best = plan["best"]
    if best["kind"] == "roll":
        actions.append(f"Transfers: no move clearly beats saving your free transfer. Roll it (you'd have {min(ft + 1, 5)} next week).")
    else:
        mv = move_view(best["moves"])
        outs = ", ".join(o["name"] for o in mv["out"])
        ins = ", ".join(o["name"] for o in mv["in"])
        hit = f" (a -{best['hits']} hit, still worth it)" if best["hits"] else ""
        actions.append(f"Transfers: {outs} → {ins}, +{mv['gain']} xPts over {opts['horizon']} GWs{hit}.")
    if lineup["captain"]:
        cap = elements[lineup["captain"]]["web_name"]
        vice = elements[lineup["vice"]]["web_name"] if lineup["vice"] else "-"
        actions.append(f"Captain {cap}, vice {vice}.")
    leaving = {elements[i]["web_name"] for i in ((best["moves"] or {}).get("outs") or [])}
    for c in changes:
        if not c.startswith("Captain") and not any(c.startswith(f"Bench {n} ") for n in leaving):
            actions.append(f"Lineup: {c}.")
    for row in chip_view.values():
        if row["status"] in ("play", "urgent"):
            actions.append(f"Chip: {row['label']}: {row['verdict']}.")
    for s in sell_list:
        if s["flagged"] and s["id"] in squad:
            actions.append(f"Check news on {s['name']} before the deadline: {s['reasons'][0].rstrip('. ')}.")
    for pv in price_watch:
        if pv["price_trend"] == "falling" and any(m == pv["id"] for m in (best["moves"] or {}).get("outs", [])):
            actions.append(f"Sell {pv['name']} early. Their price looks likely to drop.")

    if best["moves"]:
        for pid_in in best["moves"]["in"]:
            pv = player_view(pid_in)
            if pv["price_trend"] == "rising":
                actions.append(f"Buy {pv['name']} early. They're heavily bought ({pv['momentum']:+,} this week) and their price may rise.")

    deadline = nxt.get("deadline_time")
    try:
        dl = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
        hours = (dl - datetime.now(timezone.utc)).total_seconds() / 3600
    except (TypeError, ValueError):
        hours = None

    squad_view = []
    for i in squad:
        pv = player_view(i)
        pv["selling_price"] = _price(sell[i])
        pv["in_xi"] = i in lineup["xi"]
        pv["is_captain"] = i == lineup["captain"]
        pv["is_vice"] = i == lineup["vice"]
        pv["bench_order"] = lineup["bench"].index(i) + 1 if i in lineup["bench"] else None
        squad_view.append(pv)

    return {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "next_gw": next_gw,
            "deadline": deadline,
            "hours_to_deadline": _r(hours, 1) if hours is not None else None,
            "horizon": horizon,
            "transfer_horizon": opts["horizon"],
            "free_transfers": ft,
            "bank": _price(bank),
            "squad_value": _price(squad_sell_value),
            "team_value": _price(sum(elements[i]["now_cost"] for i in squad)),
            "picks_gw": eh.get("event"),
            "notes": notes + ["Free transfers are estimated from your transfer history.",
                              "Transfers you've already made for the next GW aren't visible until the deadline passes."],
        },
        "manager": {
            "id": entry.get("id"),
            "team_name": entry.get("name"),
            "manager": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
            "overall_points": entry.get("summary_overall_points"),
            "overall_rank": entry.get("summary_overall_rank"),
            "gw_points": entry.get("summary_event_points"),
            "gw_rank": entry.get("summary_event_rank"),
        },
        "actions": actions,
        "squad": squad_view,
        "lineup": {
            "gw": next_gw, "formation": lineup["formation"], "xi": lineup["xi"], "bench": lineup["bench"],
            "captain": lineup["captain"], "vice": lineup["vice"], "xpts": _r(lineup["with_captain"], 1), "changes": changes,
        },
        "captaincy": captaincy,
        "transfers": {
            "free_transfers": ft,
            "bank": _price(bank),
            "plan": {"kind": best["kind"], "hits": best["hits"], "net": _r(best["net"], 1),
                     "moves": move_view(best["moves"]) if best["moves"] else None},
            "roll_value": plan["roll_value"],
            "singles": [move_view(s) for s in opts["singles"][:8]],
            "doubles": [move_view(d) for d in opts["doubles"][:5]],
            "sell": sell_list,
            "buy": buy,
        },
        "chips": sorted(chip_view.values(), key=lambda r: order[r["status"]]),
        "fixtures": {"gws": horizon, "teams": ticker},
        "players": players_tbl,
        "history": {"gws": hist_out, "totals": hist_tot},
        "price_watch": price_watch,
    }
