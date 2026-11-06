from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Courier, Order, Segment, Stop, Variant

ROAD_FACTOR = 1.3
DETOUR_FACTOR = 1.25
CLOSURE_FACTOR = 1.4
SERVICE_MIN = 5
EXACT_LIMIT = 7


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def road_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return haversine_km(a[0], a[1], b[0], b[1]) * ROAD_FACTOR


def _xy(lat: float, lon: float, lat0: float) -> Tuple[float, float]:
    return lon * 111.32 * math.cos(math.radians(lat0)), lat * 110.57


def point_to_leg_km(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat0 = (a[0] + b[0]) / 2
    px, py = _xy(p[0], p[1], lat0)
    ax, ay = _xy(a[0], a[1], lat0)
    bx, by = _xy(b[0], b[1], lat0)
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def touched_segments(a, b, segments: Sequence[Segment]) -> List[Segment]:
    return [g for g in segments
            if g.policy and g.lat is not None
            and point_to_leg_km((g.lat, g.lon), a, b) <= g.radius_km]


class RoutePlanner:
    def __init__(self, segments: Sequence[Segment], urgent: Dict[str, bool],
                 clusters: Dict[str, int]):
        self.segments = list(segments)
        self.urgent = urgent
        self.clusters = clusters

    def simulate(self, courier: Courier, ignores_traffic: bool, start: Tuple[float, float],
                 start_min: int, seq: Sequence[Order], mode: str) -> dict:
        pos, t = start, start_min
        stops: List[Stop] = []
        drive_total, dist_total, used = 0.0, 0.0, []
        for o in seq:
            dest = (o.lat, o.lon)
            km = road_km(pos, dest)
            delay, notes = 0, []
            if not ignores_traffic:
                for g in touched_segments(pos, dest, self.segments):
                    if g.policy == "detour_only":
                        km *= CLOSURE_FACTOR
                        notes.append("объезд: " + g.name)
                    elif mode == "through":
                        delay += g.delay
                        notes.append("через: {} (+{} мин)".format(g.name, g.delay))
                    else:
                        km *= DETOUR_FACTOR
                        notes.append("объезд: " + g.name)
                    if g.name not in used:
                        used.append(g.name)
            drive = km / courier.speed * 60 + delay
            arrival = t + drive
            begin = max(arrival, o.window_start)
            late = max(0, begin - o.window_end)
            stops.append(Stop(order_id=o.id, address=o.address, priority=o.priority,
                              urgent=self.urgent.get(o.id, False), window_start=o.window_start,
                              window_end=o.window_end, arrival=int(round(arrival)),
                              start=int(round(begin)), depart=int(round(begin + SERVICE_MIN)),
                              late_min=int(round(late)), leg_km=round(km, 2), lat=o.lat, lon=o.lon,
                              weight=o.weight, note="; ".join(notes)))
            drive_total += drive
            dist_total += km
            t = begin + SERVICE_MIN
            pos = dest
        late_total = sum(s.late_min for s in stops)
        slack = min((s.window_end - s.start for s in stops), default=0)
        return {"stops": stops, "drive": drive_total, "dist": dist_total, "end": int(round(t)),
                "late": late_total, "slack": slack, "used": used}

    def _units(self, orders: Sequence[Order]) -> List[List[Order]]:
        groups: Dict[int, List[Order]] = {}
        for o in orders:
            groups.setdefault(self.clusters.get(o.id, id(o)), []).append(o)
        return [sorted(g, key=lambda o: (o.window_end, o.id)) for g in groups.values()]

    @staticmethod
    def _flat(units: Sequence[List[Order]]) -> List[Order]:
        return [o for u in units for o in u]

    def order_by_priority(self, courier, ign, start, start_min, orders, mode) -> List[Order]:
        units = self._units(orders)
        urgent = sorted([u for u in units if any(self.urgent.get(o.id) for o in u)],
                        key=lambda u: min(o.window_end for o in u))
        rest = [u for u in units if u not in urgent]
        head = self._flat(urgent)

        def cost(seq_units):
            seq = head + self._flat(seq_units)
            r = self.simulate(courier, ign, start, start_min, seq, mode)
            return r["late"] * 1000 + r["drive"]

        if len(rest) <= EXACT_LIMIT:
            best = min(itertools.permutations(rest), key=cost) if rest else ()
            return head + self._flat(best)
        seq_units = sorted(rest, key=lambda u: min(o.window_end for o in u))
        return head + self._flat(self._two_opt(seq_units, cost))

    def order_by_distance(self, start, orders) -> List[Order]:
        units = self._units(orders)
        pos, seq_units = start, []
        left = list(units)
        while left:
            nxt = min(left, key=lambda u: road_km(pos, (u[0].lat, u[0].lon)))
            seq_units.append(nxt)
            left.remove(nxt)
            pos = (nxt[-1].lat, nxt[-1].lon)

        def length(su):
            p, total = start, 0.0
            for o in self._flat(su):
                total += road_km(p, (o.lat, o.lon))
                p = (o.lat, o.lon)
            return total
        return self._flat(self._two_opt(seq_units, length))

    @staticmethod
    def _two_opt(seq, cost):
        best, best_cost = list(seq), cost(seq)
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    c = cost(cand)
                    if c + 1e-9 < best_cost:
                        best, best_cost, improved = cand, c, True
        return best

    def variants(self, courier: Courier, ignores_traffic: bool, start, start_min,
                 orders: Sequence[Order]) -> List[Variant]:
        if not orders:
            return []
        prio = self.order_by_priority(courier, ignores_traffic, start, start_min, orders, "detour")
        dist = self.order_by_distance(start, orders)
        candidates = [("priority", prio, "detour"), ("through", prio, "through"), ("distance", dist, "detour")]
        seen, out = set(), []
        for kind, seq, mode in candidates:
            r = self.simulate(courier, ignores_traffic, start, start_min, seq, mode)
            sig = (tuple(o.id for o in seq), tuple(s.note for s in r["stops"]))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(self._make_variant(kind, r))
        out.sort(key=lambda v: (-v.score, v.drive_min))
        for i, v in enumerate(out):
            v.label = "ABC"[i]
        return out

    @staticmethod
    def _make_variant(kind: str, r: dict) -> Variant:
        if r["late"] > 0:
            risk = "высокий"
        elif r["slack"] < 15:
            risk = "средний"
        else:
            risk = "низкий"
        penalty = {"низкий": 0, "средний": 10, "высокий": 25}[risk]
        seen_regular, misplaced = False, 0
        for s in r["stops"]:
            if s.urgent and seen_regular:
                misplaced += 1
            seen_regular = seen_regular or not s.urgent
        penalty += 15 * misplaced
        score = int(round(max(0.0, min(100.0, 100 - r["drive"] - 5 * r["late"] - penalty))))
        seq_txt = " → ".join(s.order_id for s in r["stops"])
        notes = sorted({s.note for s in r["stops"] if s.note})
        desc = seq_txt + (" (" + "; ".join(notes) + ")" if notes else "")
        return Variant(label="", kind=kind, stops=r["stops"], distance_km=round(r["dist"], 1),
                       drive_min=int(round(r["drive"])), end_min=r["end"], late_total=r["late"],
                       min_slack=int(r["slack"]), risk=risk, score=score, description=desc,
                       segments_used=r["used"])
