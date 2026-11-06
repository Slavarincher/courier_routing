from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PRIORITY_LABELS = {"high": "высокий", "medium": "средний", "low": "низкий"}
STATE_LABELS = {"FreeFlow": "свободно", "Congestion": "пробка", "RoadClosure": "перекрыто"}


@dataclass
class Order:
    id: str
    address: str
    lat: float
    lon: float
    priority: str
    weight: float
    window_start: int
    window_end: int
    zone: Optional[str] = None
    zone_name: str = ""
    restricted: bool = False
    status: str = "new"
    late_added: bool = False
    source: str = "БЗ"

    @property
    def priority_label(self) -> str:
        return PRIORITY_LABELS.get(self.priority, self.priority)

    @property
    def active(self) -> bool:
        return self.status in ("new", "retry")


@dataclass
class Courier:
    id: str
    name: str
    vehicle_id: str
    vehicle_class: str
    vehicle_label: str
    capacity: float
    speed: float
    rating: float
    lat: float
    lon: float
    phone: str = ""
    active: bool = True
    status: str = "free"
    clock: Optional[int] = None
    shift_end: int = 18 * 60

    @property
    def short_name(self) -> str:
        parts = self.name.split()
        if len(parts) >= 3:
            return "{} {}. {}.".format(parts[0], parts[1][0], parts[2][0])
        return self.name


@dataclass
class Segment:
    id: str
    name: str
    state: str
    delay: int
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_km: float = 1.0
    policy: Optional[str] = None

    @property
    def state_label(self) -> str:
        return STATE_LABELS.get(self.state, self.state)


@dataclass
class Stop:
    order_id: str
    address: str
    priority: str
    urgent: bool
    window_start: int
    window_end: int
    arrival: int
    start: int
    depart: int
    late_min: int
    leg_km: float
    lat: float
    lon: float
    weight: float
    note: str = ""
    status: str = "pending"


@dataclass
class Variant:
    label: str
    kind: str
    stops: List[Stop]
    distance_km: float
    drive_min: int
    end_min: int
    late_total: int
    min_slack: int
    risk: str
    score: int
    description: str
    segments_used: List[str] = field(default_factory=list)


@dataclass
class CourierPlan:
    courier: Courier
    orders: List[Order]
    variants: List[Variant]
    explanation: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    approved_route: Optional[str] = None

    @property
    def recommended(self) -> Optional[Variant]:
        return self.variants[0] if self.variants else None

    def variant(self, label: str) -> Optional[Variant]:
        return next((v for v in self.variants if v.label == label), None)


@dataclass
class TraceLine:
    phase: str
    rule: str
    name: str
    text: str


@dataclass
class PlanResult:
    plans: List[CourierPlan]
    unassigned: List[Tuple[Order, str]]
    warnings: List[str]
    trace: List[TraceLine]
    elapsed_ms: float
    start_min: int

    def plan_for(self, courier_id: str) -> Optional[CourierPlan]:
        return next((p for p in self.plans if p.courier.id == courier_id), None)


@dataclass
class Route:
    id: str
    courier_id: str
    variant_label: str
    stops: List[Stop]
    start_min: int
    clock: int
    lat: float
    lon: float
    status: str = "in_progress"
    notifications: List[str] = field(default_factory=list)
    trace: List[TraceLine] = field(default_factory=list)

    @property
    def pending(self) -> List[Stop]:
        return [s for s in self.stops if s.status == "pending"]

    @property
    def distance_km(self) -> float:
        return round(sum(s.leg_km for s in self.stops), 1)

    @property
    def load(self) -> float:
        return round(sum(s.weight for s in self.stops if s.status == "pending"), 1)

    @property
    def end_min(self) -> int:
        times = [s.depart for s in self.stops if s.status in ("pending", "delivered")]
        return max(times) if times else self.clock


@dataclass
class ShiftState:
    orders: List[Order]
    couriers: List[Courier]
    segments: List[Segment]
    start_min: int = 10 * 60
    dispatcher: str = ""
    routes: Dict[str, Route] = field(default_factory=dict)
    last_plan: Optional[PlanResult] = None
    next_route_no: int = 1

    def now(self) -> int:
        return max([self.start_min] + [r.clock for r in self.routes.values()])

    def order(self, oid: str) -> Optional[Order]:
        return next((o for o in self.orders if o.id == oid), None)

    def courier(self, cid: str) -> Optional[Courier]:
        return next((c for c in self.couriers if c.id == cid), None)

    def segment(self, sid: str) -> Optional[Segment]:
        return next((g for g in self.segments if g.id == sid), None)

    def route_of(self, courier_id: str) -> Optional[Route]:
        return next((r for r in self.routes.values()
                     if r.courier_id == courier_id and r.status == "in_progress"), None)
