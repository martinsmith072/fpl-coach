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

    def __init__(self, teams: Sequence[Dict[str, Any]], fx: FixtureIndex):
        total_goals = sum(fx.goals_for.values())
        total_games = sum(fx.games_played.values())
        mu = total_goals / total_games if total_games >= 20 else 1.4
        self.mu = _clip(mu, 1.0, 1.9)

        def mean(keys: Sequence[str]) -> float:
            vals = [_f(t.get(k)) for t in teams for k in keys if _f(t.get(k)) > 0]
            return sum(vals) / len(vals) if vals else 1.0

        att_mean = mean(["strength_attack_home", "strength_attack_away"])
        def_mean = mean(["strength_defence_home", "strength_defence_away"])
        k = 8.0
        self.A: Dict[int, Dict[str, float]] = {}
        self.D: Dict[int, Dict[str, float]] = {}
        for t in teams:
            tid = t["id"]
            gp = fx.games_played[tid]
            att_res = ((fx.goals_for[tid] + self.mu * k) / (gp + k)) / self.mu
            dw_res = ((fx.goals_against[tid] + self.mu * k) / (gp + k)) / self.mu
            w_res = _clip(gp / (gp + 10.0), 0.0, 0.6)
            self.A[tid], self.D[tid] = {}, {}
            for venue, hf in (("home", 1.08), ("away", 0.92)):
                sa = _f(t.get(f"strength_attack_{venue}")) or att_mean
                sd = _f(t.get(f"strength_defence_{venue}")) or def_mean
                s_att = (sa / att_mean) ** 2
                s_def = (def_mean / sd) ** 2
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
    def __init__(self, bootstrap: Dict[str, Any], fx: FixtureIndex, horizon: List[int]):
        self.fx = fx
        self.tm = TeamModel(bootstrap.get("teams") or [], fx)
        self.rules = scoring_rules(bootstrap)
        self.horizon = horizon
        self.xpts: Dict[int, List[float]] = {}
        self.rates: Dict[int, PlayerRates] = {}
        for p in bootstrap.get("elements") or []:
            self.xpts[p["id"]] = self._project(p)

    def _project(self, p: Dict[str, Any]) -> List[float]:
        team = p["team"]
        r = PlayerRates(p, self.fx.games_played[team])
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
                     pools: Dict[int, List[int]]) -> Dict[str, Any]:
    n = min(TRANSFER_HORIZON, len(next(iter(xpts.values()), [])))
    hsum = {i: sum(WEIGHTS[k] * v for k, v in enumerate(xpts[i][:n])) for i in xpts}
    base = squad_value(squad, pos, xpts, n)
    owned = set(squad)
    team_count = Counter(elements[i]["team"] for i in squad)

    # Singles: screen with a quick estimate, then score properly.
    singles_raw = []
    for out in squad:
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
                 notes: Optional[List[str]] = None) -> Dict[str, Any]:
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
    proj = Projector(bootstrap, fx, horizon)
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
    opts = transfer_options(squad, elements, pos, xpts, sell, bank, pools)
    plan = pick_plan(opts, ft)

    def move_view(m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "out": [{"id": i, "name": elements[i]["web_name"], "price": _price(sell[i]), "team": teams[elements[i]["team"]]["short_name"],
                     "xpts_h": _r(sum(xpts[i][:TRANSFER_HORIZON]))} for i in m["outs"]],
            "in": [{"id": i, "name": elements[i]["web_name"], "price": _price(elements[i]["now_cost"]), "team": teams[elements[i]["team"]]["short_name"],
                    "xpts_h": _r(sum(xpts[i][:TRANSFER_HORIZON]))} for i in m["in"]],
            "gain": _r(m["gain"], 1),
            "bank_after": _price(m["bank_after"]),
        }

    # Sell watchlist: injury, poor projection for price, bad fixtures, falling price.
    sell_list = []
    for i in squad:
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

    def wildcard_eval() -> Dict[str, Any]:
        if "wc" not in wc_cache:
            n = min(TRANSFER_HORIZON, len(horizon))
            score = {i: sum(WEIGHTS[k] * xpts[i][k] for k in range(n)) for i in elements}
            sq, _ = build_squad(elements, pos, pools, score, budget_total)
            gain = squad_value(sq, pos, xpts, n) - squad_value(squad, pos, xpts, n)
            wc_cache["wc"] = {"squad": sq, "gain": gain}
        return wc_cache["wc"]

    def freehit_eval(k: int) -> Dict[str, Any]:
        key = f"fh{k}"
        if key not in wc_cache:
            score = {i: xpts[i][k] for i in elements}
            fh_pools = {pp: sorted([i for i in playable if pos[i] == pp], key=lambda i: score[i], reverse=True)[:30]
                        + [i for i in pools[pp] if elements[i]["now_cost"] <= 50][:6] for pp in (1, 2, 3, 4)}
            for pp in fh_pools:
                fh_pools[pp] = list(dict.fromkeys(fh_pools[pp]))
            sq, val = build_squad(elements, pos, fh_pools, score, budget_total)
            mine = best_xi(squad, pos, {i: xpts[i][k] for i in squad})
            wc_cache[key] = {"squad": sq, "gain": val - (mine["with_captain"] + 0.1 * mine["bench_total"])}
        return wc_cache[key]

    per_gw = []
    for k, g in enumerate(horizon):
        pts = {i: xpts[i][k] for i in squad}
        r = best_xi(squad, pos, pts)
        playing = sum(1 for i in squad if fx.get(elements[i]["team"], g))
        doubles = sum(1 for i in squad if len(fx.get(elements[i]["team"], g)) > 1)
        per_gw.append({"gw": g, "k": k, "xi": r, "bench": r["bench_total"], "cap": pts.get(r["captain"], 0.0) if r["captain"] else 0.0,
                       "cap_id": r["captain"], "playing": playing, "doubles": doubles})

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
        if start > horizon[-1]:
            row.update(status="later", verdict=f"Available from GW{start}", score=-1)
            chip_rows.append(row)
            continue
        if name == "manager":
            row.update(status="available", verdict="Not modelled. Check the game's rules for this chip.", score=0)
            chip_rows.append(row)
            continue
        window = [x for x in per_gw if start <= x["gw"] <= stop]
        if name == "bboost":
            vals = [(x["gw"], x["bench"], f"{x['doubles']} doubles" if x["doubles"] else "") for x in window]
            threshold = 10.0
        elif name == "3xc":
            vals = [(x["gw"], x["cap"], f"{elements[x['cap_id']]['web_name']}" if x["cap_id"] else "") for x in window]
            threshold = 9.0
        elif name == "freehit":
            raw = [(x, freehit_eval(x["k"])["gain"]) for x in window]
            gains = sorted(g for _, g in raw)
            typical = gains[len(gains) // 2] if gains else 0.0
            # Value = how much better than a normal week a Free Hit would be (blanks and doubles stand out).
            vals = [(x["gw"], g - typical + (6.0 if x["playing"] <= 9 else 0.0),
                     f"{x['playing']}/15 have a fixture" + (f", {x['doubles']} doubles" if x["doubles"] else "")) for x, g in raw]
            threshold = 10.0
        else:  # wildcard
            wc = wildcard_eval()
            # Compare with what your free transfers could achieve anyway, not with doing nothing.
            ft_gain = max(0.0, plan["best"]["gain"] - plan["best"]["hits"])
            vals = [(next_gw, wc["gain"] - ft_gain,
                     f"rebuild +{wc['gain']:.0f} vs +{ft_gain:.0f} from your free transfers")] if start <= next_gw <= stop else []
            threshold = 20.0
            row["draft"] = [player_view(i) for i in sorted(wc["squad"], key=lambda i: (pos[i], -hsum5[i]))]
        row["values"] = [{"gw": g, "value": _r(v, 1), "note": n} for g, v, n in vals]
        weeks_left = stop - next_gw + 1
        if not vals:
            row.update(status="available", verdict="Hold", score=0)
        else:
            best_gw, best_val, _ = max(vals, key=lambda v: v[1])
            now_val = vals[0][1] if vals[0][0] == next_gw else 0.0
            row["best_gw"] = best_gw
            row["best_value"] = _r(best_val, 1)
            if name == "wildcard":
                problems = sum(1 for s in sell_list if s["severity"] >= 3)
                if best_val >= threshold or problems >= 4:
                    verdict = f"Worth considering now: a rebuild projects +{best_val:.1f} pts more than using your free transfers, over {min(TRANSFER_HORIZON, len(horizon))} GWs"
                    status = "play"
                elif weeks_left <= 3:
                    verdict = f"Use it or lose it: the window closes after GW{stop}. Gain now: +{best_val:.1f}"
                    status = "urgent"
                else:
                    verdict = f"Hold. A rebuild only beats your free transfers by {best_val:+.1f} pts; aim for at least ~{threshold:.0f}"
                    status = "hold"
            else:
                if best_gw == next_gw and best_val >= threshold:
                    verdict, status = f"Play this GW (+{best_val:.1f} pts expected)", "play"
                elif weeks_left <= 3:
                    verdict, status = f"Window closes after GW{stop}. Best remaining week is GW{best_gw} (+{best_val:.1f})", "urgent"
                elif best_val >= threshold:
                    verdict, status = f"Plan for GW{best_gw} (+{best_val:.1f} pts). Hold until then", "plan"
                else:
                    verdict, status = (f"Hold. Best spot in view is GW{best_gw} (+{best_val:.1f}). "
                                       f"Save it for a double gameweek or a stronger week"), "hold"
                if name == "freehit" and now_val >= threshold:
                    row["draft"] = [player_view(i) for i in sorted(freehit_eval(0)["squad"], key=lambda i: (pos[i], -xpts[i][0]))]
            row.update(status=status, verdict=verdict, score=best_val / threshold)
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
    for c in changes:
        if not c.startswith("Captain"):
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
