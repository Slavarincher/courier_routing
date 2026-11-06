from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .engine import Firing, InferenceEngine, WorkingMemory
from .kb_loader import KnowledgeBase
from .models import (Courier, CourierPlan, Order, PlanResult, Route, Segment, ShiftState,
                     TraceLine)
from .router import RoutePlanner, haversine_km, road_km

_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class PlanError(Exception):
    pass


def parse_time(text: str) -> Optional[int]:
    m = _TIME.match((text or "").strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def fmt_time(minutes: Optional[int]) -> str:
    if minutes is None:
        return ""
    m = int(round(minutes))
    return "{:02d}:{:02d}".format((m // 60) % 24, m % 60)


def _distance_km(f1: dict, f2: dict) -> float:
    return round(haversine_km(f1["lat"], f1["lon"], f2["lat"], f2["lon"]), 2)


class ExpertSystem:
    def __init__(self, base_dir: Path):
        base = Path(base_dir)
        self.kb = KnowledgeBase(base / "kb" / "CourierRouting.pont", base / "kb" / "CourierRouting.pins")
        self.engine = InferenceEngine.from_json(base / "kb" / "rules.json", {"distance_km": _distance_km})
        self.geo = json.loads((base / "data" / "geo.json").read_text(encoding="utf-8"))
        self.depot = self.geo["depot"]
        self.load_warnings: List[str] = []

    def new_shift(self) -> ShiftState:
        kb, self.load_warnings = self.kb, []
        orders = []
        for inst in kb.instances_of("Order"):
            addr = inst.get("address", "")
            geo = self.geo["addresses"].get(addr)
            if geo is None:
                self.load_warnings.append("Адрес «{}» заказа {} не найден в справочнике - заказ пропущен"
                                          .format(addr, inst.get("orderId")))
                continue
            zone = kb.get(inst.get("belongsToZone"))
            orders.append(Order(
                id=str(inst.get("orderId", inst.name)), address=addr, lat=geo["lat"], lon=geo["lon"],
                priority=str(inst.get("priority", "medium")), weight=float(inst.get("weightKg", 0)),
                window_start=parse_time(inst.get("windowStart", "")) or 0,
                window_end=parse_time(inst.get("windowEnd", "")) or 24 * 60 - 1,
                zone=zone.name if zone else None, zone_name=zone.get("zoneName", "") if zone else "",
                restricted=bool(zone and kb.is_a(zone.cls, "RestrictedZone")), source="БЗ"))
        orders.sort(key=lambda o: o.id)
        couriers = []
        for inst in kb.instances_of("Courier"):
            veh = kb.get(inst.get("usesVehicle"))
            if veh is None:
                self.load_warnings.append("У курьера {} не указано транспортное средство".format(inst.name))
                continue
            couriers.append(Courier(
                id=inst.name, name=inst.get("fullName", inst.name), vehicle_id=veh.name,
                vehicle_class=veh.cls, vehicle_label=kb.class_label(veh.cls),
                capacity=float(veh.get("maxLoadKg", 0)), speed=float(veh.get("avgSpeedKmh", 15)),
                rating=float(inst.get("courierRating", 5)), phone=inst.get("phone", ""),
                lat=self.depot["lat"], lon=self.depot["lon"]))
        segments = []
        for inst in kb.instances_of("TrafficCondition"):
            g = self.geo["segments"].get(inst.name, {})
            segments.append(Segment(id=inst.name, name=inst.get("roadSegment", inst.name), state=inst.cls,
                                    delay=int(inst.get("delayMin", 0)), lat=g.get("lat"), lon=g.get("lon"),
                                    radius_km=float(g.get("radius_km", 1.0))))
        disp = kb.instances_of("Dispatcher")
        dispatcher = ""
        if disp:
            parts = str(disp[0].get("fullName", "")).split()
            dispatcher = "{} {}. {}.".format(parts[0], parts[1][0], parts[2][0]) if len(parts) >= 3 else " ".join(parts)
        return ShiftState(orders=orders, couriers=couriers, segments=segments, dispatcher=dispatcher)

    def traffic_states(self) -> List[Tuple[str, str]]:
        return [(c.name, c.doc) for c in self.kb.classes.values()
                if self.kb.is_a(c.name, "TrafficCondition") and not c.abstract]

    def known_addresses(self) -> List[str]:
        return list(self.geo["addresses"].keys())

    def validate_order(self, form: Dict[str, str], shift: ShiftState) -> Tuple[Optional[Order], Dict[str, str]]:
        err: Dict[str, str] = {}
        oid = (form.get("order_id") or "").strip()
        if not oid:
            nums = [int(o.id) for o in shift.orders if o.id.isdigit()]
            oid = str(max(nums) + 1 if nums else 1001)
        elif shift.order(oid):
            err["order_id"] = "Заказ с таким номером уже есть"

        address = (form.get("address") or "").strip()
        geo = self.geo["addresses"].get(address)
        if not address:
            err["address"] = "Введите адрес"
        elif geo is None:
            err["address"] = "Адрес не найден в справочнике - выберите из списка"

        priority = (form.get("priority") or "").strip()
        e = self.kb.check_value("priority", priority)
        if e:
            err["priority"] = "Приоритет: " + e

        weight_txt = (form.get("weight") or "").strip().replace(",", ".")
        try:
            weight = float(weight_txt)
            e = self.kb.check_value("weightKg", weight)
            if e:
                err["weight"] = "Вес: " + e
            elif weight <= 0:
                err["weight"] = "Вес должен быть больше 0"
            else:
                max_cap = max((c.capacity for c in shift.couriers), default=0)
                if weight > max_cap:
                    err["weight"] = "Вес превышает грузоподъемность всех курьеров (макс. {:g} кг)".format(max_cap)
        except ValueError:
            weight = 0.0
            err["weight"] = "Вес должен быть числом"

        ws, we = parse_time(form.get("window_start", "")), parse_time(form.get("window_end", ""))
        if ws is None:
            err["window_start"] = "Время в формате ЧЧ:ММ"
        if we is None:
            err["window_end"] = "Время в формате ЧЧ:ММ"
        if ws is not None and we is not None and ws >= we:
            err["window_end"] = "Конец окна должен быть позже начала"

        if err:
            return None, err
        zone = self.kb.get(geo.get("zone"))
        late = bool(shift.routes)
        return Order(id=oid, address=address, lat=geo["lat"], lon=geo["lon"], priority=priority,
                     weight=weight, window_start=ws, window_end=we,
                     zone=zone.name if zone else None, zone_name=zone.get("zoneName", "") if zone else "",
                     restricted=bool(zone and self.kb.is_a(zone.cls, "RestrictedZone")),
                     late_added=late, source=form.get("source", "форма")), {}

    def plan(self, shift: ShiftState) -> PlanResult:
        t0 = time.perf_counter()
        orders = [o for o in shift.orders if o.active]
        couriers = [c for c in shift.couriers if c.active and shift.route_of(c.id) is None]
        if not orders:
            raise PlanError("Не введено ни одного заказа для планирования")
        if not couriers:
            raise PlanError("Нет свободных курьеров на смене - отметьте хотя бы одного курьера")

        wm = WorkingMemory()
        for o in orders:
            wm.add("order", id=o.id, priority=o.priority, weight=o.weight, window_start=o.window_start,
                   window_end=o.window_end, window_left_min=o.window_end - shift.now(),
                   restricted=o.restricted, zone_name=o.zone_name, status=o.status,
                   late_added=o.late_added, lat=o.lat, lon=o.lon, urgent=False, needs_eco=False)
        for c in couriers:
            wm.add("courier", id=c.id, name=c.short_name, vehicle_class=c.vehicle_class,
                   vehicle_label=c.vehicle_label, rating=c.rating, capacity=c.capacity,
                   capacity_left=c.capacity, status="free", lat=c.lat, lon=c.lon,
                   eco=False, ignores_traffic=False, can_take_high=True)
        for g in shift.segments:
            wm.add("segment", id=g.id, name=g.name, state=g.state, delay=g.delay, policy=None)
        firings = self.engine.infer(wm, "classify")

        of = {f["id"]: f for f in wm.of_type("order")}
        cf = {f["id"]: f for f in wm.of_type("courier")}
        for g in shift.segments:
            g.policy = wm.facts["segment#" + g.id].get("policy")
        clusters = self._clusters(wm)
        urgent = {oid: bool(f.get("urgent")) for oid, f in of.items()}

        assigned, unassigned = self._assign(orders, couriers, of, cf, clusters)

        planner = RoutePlanner(shift.segments, urgent, clusters)
        plans: List[CourierPlan] = []
        for c in couriers:
            if not assigned[c.id]:
                continue
            start_min = max(shift.start_min, c.clock or 0)
            variants = planner.variants(c, bool(cf[c.id].get("ignores_traffic")), (c.lat, c.lon),
                                        start_min, assigned[c.id])
            plans.append(CourierPlan(courier=c, orders=assigned[c.id], variants=variants))

        wm2 = WorkingMemory()
        for o, reason in unassigned:
            wm2.add("unassigned", id=o.id, order_id=o.id, weight=_num(o.weight), reason=reason)
        for p in plans:
            for v in p.variants:
                for s in v.stops:
                    wm2.add("stop", courier_id=p.courier.id, courier_name=p.courier.short_name,
                            variant=v.label, order_id=s.order_id, start=s.start,
                            window_end=s.window_end, late_min=s.late_min)
            rec = p.recommended
            wm2.add("plan_route", id=p.courier.id, courier_id=p.courier.id, courier_name=p.courier.short_name,
                    end_min=rec.end_min, shift_end_min=p.courier.shift_end,
                    shift_limit_min=p.courier.shift_end - int(self.engine.parameters.get("shift_margin_min", 30)))
        firings2 = self.engine.infer(wm2, "check")

        warnings_global: List[str] = []
        for f in firings2:
            owner = self._owner(wm2, f)
            p = next((x for x in plans if x.courier.id == owner), None)
            (p.warnings if p else warnings_global).extend(f.warnings)
        for p in plans:
            p.explanation = self._explain(p, firings, wm)

        trace = [TraceLine(f.phase, f.rule_no, f.rule_name, f.text) for f in firings + firings2]
        result = PlanResult(plans=plans, unassigned=unassigned, warnings=warnings_global, trace=trace,
                            elapsed_ms=(time.perf_counter() - t0) * 1000, start_min=shift.start_min)
        shift.last_plan = result
        return result

    @staticmethod
    def _clusters(wm: WorkingMemory) -> Dict[str, int]:
        parent: Dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for f in wm.of_type("near"):
            parent[find(f["a"])] = find(f["b"])
        roots: Dict[str, int] = {}
        out: Dict[str, int] = {}
        for oid in list(parent):
            out[oid] = roots.setdefault(find(oid), len(roots) + 1)
        return out

    def _assign(self, orders, couriers, of, cf, clusters):
        groups: Dict[object, List[Order]] = {}
        for o in orders:
            groups.setdefault(clusters.get(o.id, o.id), []).append(o)
        queue = sorted(groups.values(), key=lambda g: (not any(of[o.id]["urgent"] for o in g),
                                                       min(o.window_end for o in g)))
        cap = {c.id: c.capacity for c in couriers}
        last = {c.id: (c.lat, c.lon) for c in couriers}
        count = {c.id: 0 for c in couriers}
        assigned: Dict[str, List[Order]] = {c.id: [] for c in couriers}
        unassigned: List[Tuple[Order, str]] = []

        def fits(c, needs_eco, high, w):
            return (cap[c.id] >= w and (not needs_eco or cf[c.id].get("eco"))
                    and (not high or cf[c.id].get("can_take_high", True)))

        while queue:
            g = queue.pop(0)
            w = sum(o.weight for o in g)
            needs_eco = any(of[o.id].get("needs_eco") for o in g)
            high = any(o.priority == "high" for o in g)
            cands = [c for c in couriers if fits(c, needs_eco, high, w)]
            if not cands and len(g) > 1:
                queue[0:0] = [[o] for o in g]
                continue
            if not cands:
                o = g[0]
                zone_ok = [c for c in couriers if not needs_eco or cf[c.id].get("eco")]
                rating_ok = [c for c in zone_ok if not high or cf[c.id].get("can_take_high", True)]
                reason = "restricted_zone" if not zone_ok else ("rating" if not rating_ok else "capacity")
                unassigned.append((o, reason))
                continue
            preferred = next((of[o.id].get("preferred_courier") for o in g if of[o.id].get("preferred_courier")), None)
            chosen = next((c for c in cands if c.id == preferred), None)
            if chosen is None:
                first = g[0]
                chosen = min(cands, key=lambda c: (road_km(last[c.id], (first.lat, first.lon)) / c.speed * 60
                                                   + 15 * count[c.id], c.id))
            assigned[chosen.id].extend(g)
            cap[chosen.id] -= w
            count[chosen.id] += len(g)
            last[chosen.id] = (g[-1].lat, g[-1].lon)
        return assigned, unassigned

    @staticmethod
    def _owner(wm: WorkingMemory, f: Firing) -> Optional[str]:
        for fid in f.fact_ids:
            cid = wm.facts.get(fid, {}).get("courier_id")
            if cid:
                return cid
        return None

    def _explain(self, p: CourierPlan, firings: List[Firing], wm: WorkingMemory) -> List[str]:
        rec = p.recommended
        mine = {"order#" + o.id for o in p.orders} | {"courier#" + p.courier.id}
        seg_ids = {fid for fid, f in wm.facts.items()
                   if f["type"] == "segment" and f["name"] in rec.segments_used}
        lines = []
        for f in firings:
            ids = set(f.fact_ids)
            if ids & mine or ids & seg_ids:
                lines.append("Правило {} ({}): {}".format(f.rule_no, f.rule_name, f.text))
        why = "Вариант {} выбран по наибольшей оценке {} из 100: в пути {} мин, {:.1f} км".format(
            rec.label, rec.score, rec.drive_min, rec.distance_km).replace(".", ",")
        why += ", без опозданий" if rec.late_total == 0 else ", опоздание {} мин".format(rec.late_total)
        return [why + "."] + lines

    def approve(self, shift: ShiftState, courier_id: str, label: str) -> Route:
        plan = shift.last_plan.plan_for(courier_id) if shift.last_plan else None
        if plan is None or plan.variant(label) is None:
            raise PlanError("Вариант маршрута не найден - постройте маршрут заново")
        if shift.route_of(courier_id):
            raise PlanError("У курьера уже есть утвержденный маршрут")
        v = plan.variant(label)
        c = shift.courier(courier_id)
        rid = "R{}".format(shift.next_route_no)
        shift.next_route_no += 1
        start = max(shift.start_min, c.clock or 0)
        route = Route(id=rid, courier_id=courier_id, variant_label=label, stops=copy.deepcopy(v.stops),
                      start_min=start, clock=start, lat=c.lat, lon=c.lon,
                      notifications=["{} Маршрут утвержден диспетчером (вариант {}).".format(fmt_time(start), label)])
        for s in route.stops:
            shift.order(s.order_id).status = "assigned"
        c.status = "on_route"
        plan.approved_route = rid
        shift.routes[rid] = route
        return route

    def handle_event(self, shift: ShiftState, route_id: str, kind: str, payload: Dict[str, str]) -> List[str]:
        route = shift.routes[route_id]
        courier = shift.courier(route.courier_id)
        if route.status != "in_progress":
            raise PlanError("Маршрут уже завершен")
        current = route.pending[0] if route.pending else None
        messages: List[str] = []
        event = {"kind": kind, "route_id": route.id}

        if kind == "delivered" and current:
            current.status = "delivered"
            route.clock = max(route.clock, current.depart)
            route.lat, route.lon = current.lat, current.lon
            shift.order(current.order_id).status = "delivered"
            messages.append("Заказ {} доставлен в {}.".format(current.order_id, fmt_time(current.start)))
        elif kind == "client_unavailable" and current:
            event["order_id"] = current.order_id
            route.clock = max(route.clock, current.start)
            route.lat, route.lon = current.lat, current.lon
        elif kind == "deviation":
            try:
                event["distance_km"] = float(str(payload.get("distance_km", "0")).replace(",", "."))
            except ValueError:
                raise PlanError("Расстояние отклонения должно быть числом")
        elif kind == "road_closure":
            seg = shift.segment(payload.get("segment", ""))
            if seg is None:
                raise PlanError("Выберите участок дороги")
            seg.state = "RoadClosure"
            event["segment"] = seg.id
        else:
            raise PlanError("Неизвестное событие")

        firings = self._event_inference(shift, route, courier, event)
        wm_route = firings["route"]
        recalc = bool(wm_route.get("needs_recalc")) or kind == "road_closure"
        for f in firings["list"]:
            if f.rule_id in ("R8", "R11", "R12") or (f.rule_id == "R10" and kind == "road_closure"):
                messages.append("Правило {}: {}.".format(f.rule_no, f.text))

        for oid in firings["retry"]:
            for s in route.stops:
                if s.order_id == oid and s.status == "pending":
                    s.status = "retry"
            shift.order(oid).status = "retry"
            recalc = True
        if kind == "deviation":
            if wm_route.get("needs_recalc"):
                route.clock += int(round(event["distance_km"] * 2 / courier.speed * 60))
            else:
                messages.append("Отклонение {} км меньше порога {} км - маршрут не изменен.".format(
                    _dec(event["distance_km"]), _dec(self.engine.parameters.get("deviation_km", 0))))

        if recalc and route.pending:
            late = self._recalculate(shift, route, courier)
            messages.append("Маршрут пересчитан: новое время прибытия обновлено.")
            messages.extend(late)

        if not route.pending and route.status != "completed":
            more = self._event_inference(shift, route, courier, {"kind": "check", "route_id": route.id})
            for f in more["list"]:
                if f.rule_id == "R12":
                    messages.append("Правило {}: {}.".format(f.rule_no, f.text))
            firings["list"] += more["list"]
        if route.status == "completed":
            courier.status = "free"
            courier.lat, courier.lon, courier.clock = route.lat, route.lon, route.clock

        route.trace += [TraceLine(f.phase, f.rule_no, f.rule_name, f.text) for f in firings["list"]]
        stamp = fmt_time(route.clock)
        route.notifications = ["{} {}".format(stamp, m) for m in messages] + route.notifications
        return messages

    def _event_inference(self, shift, route, courier, event) -> dict:
        wm = WorkingMemory()
        wm.add("event", id="e", **event)
        rf = wm.add("route", id=route.id, courier_id=route.courier_id, remaining=len(route.pending),
                    status=route.status, needs_recalc=False)
        cf = wm.add("courier", id=courier.id, name=courier.short_name, status=courier.status)
        for s in route.pending:
            wm.add("order", id=s.order_id, status="assigned")
        for g in shift.segments:
            wm.add("segment", id=g.id, name=g.name, state=g.state, delay=g.delay, policy=None)
        firings = self.engine.infer(wm, "event")
        for g in shift.segments:
            g.policy = wm.facts["segment#" + g.id].get("policy")
        route.status = rf["status"]
        courier.status = cf["status"] if route.status == "completed" else courier.status
        retry = [f["id"] for f in wm.of_type("order") if f.get("status") == "retry"]
        return {"list": firings, "route": rf, "retry": retry}

    def _recalculate(self, shift: ShiftState, route: Route, courier: Courier) -> List[str]:
        pending = [shift.order(s.order_id) for s in route.pending]
        urgent = {s.order_id: s.urgent for s in route.pending}
        planner = RoutePlanner(shift.segments, urgent, {})
        ign = courier.vehicle_class in ("Bicycle", "OnFoot")
        seq = planner.order_by_priority(courier, ign, (route.lat, route.lon), route.clock, pending, "detour")
        sim = planner.simulate(courier, ign, (route.lat, route.lon), route.clock, seq, "detour")
        done = [s for s in route.stops if s.status != "pending"]
        route.stops = done + sim["stops"]
        wm = WorkingMemory()
        for s in sim["stops"]:
            wm.add("stop", courier_id=courier.id, courier_name=courier.short_name, variant=route.variant_label,
                   order_id=s.order_id, start=s.start, window_end=s.window_end, late_min=s.late_min)
        return [w for f in self.engine.infer(wm, "check") for w in f.warnings]


def _dec(x: float) -> str:
    return "{:.1f}".format(float(x)).replace(".", ",")


def _num(x: float):
    return int(x) if float(x).is_integer() else x
