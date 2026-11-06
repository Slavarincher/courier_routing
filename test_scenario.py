import time
import unittest
from pathlib import Path

from es.expert_system import ExpertSystem, PlanError

BASE = Path(__file__).resolve().parent.parent


class ControlExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.es = ExpertSystem(BASE)

    def setUp(self):
        self.shift = self.es.new_shift()

    def test_kb_loaded(self):
        self.assertEqual(len(self.es.kb.classes), 16)
        self.assertEqual(len(self.es.kb.slots), 28)
        self.assertEqual(len(self.es.kb.instances), 15)
        self.assertEqual(len({r.no for r in self.es.engine.rules}), 14)

    def test_urgent_orders_first(self):
        res = self.es.plan(self.shift)
        fired = {t.text for t in res.trace if t.rule == "1"}
        self.assertTrue(any("1001" in t for t in fired) and any("1003" in t for t in fired))
        for p in res.plans:
            self.assertTrue(p.recommended.stops[0].urgent)

    def test_detour_for_congestion(self):
        res = self.es.plan(self.shift)
        petrov = res.plan_for("Courier_Petrov")
        self.assertIn("объезд", petrov.recommended.description)
        self.assertTrue(any(v.kind == "through" for v in petrov.variants))

    def test_restricted_zone_to_bicycle(self):
        res = self.es.plan(self.shift)
        sid = res.plan_for("Courier_Sidorov")
        self.assertEqual(sorted(o.id for o in sid.orders), ["1003", "1004"])
        self.assertTrue(any(t.rule == "7" for t in res.trace))

    def test_time_limit(self):
        for i in range(46):
            o, err = self.es.validate_order({"address": list(self.es.geo["addresses"])[i % 16],
                                             "priority": "medium", "weight": "1",
                                             "window_start": "10:00", "window_end": "18:00"}, self.shift)
            self.assertFalse(err)
            self.shift.orders.append(o)
        t0 = time.perf_counter()
        self.es.plan(self.shift)
        self.assertLess(time.perf_counter() - t0, 5.0)

    def test_low_rating_and_capacity(self):
        self.shift.courier("Courier_Sidorov").rating = 3.5
        res = self.es.plan(self.shift)
        self.assertTrue(any(t.rule == "13" for t in res.trace))
        reasons = dict((o.id, r) for o, r in res.unassigned)
        self.assertEqual(reasons.get("1003"), "rating")

    def test_input_validation(self):
        _, err = self.es.validate_order({"address": "нет такого", "priority": "urgent", "weight": "-5",
                                         "window_start": "12:00", "window_end": "11:00"}, self.shift)
        self.assertEqual(set(err), {"address", "priority", "weight", "window_end"})
        self.shift.orders.clear()
        with self.assertRaises(PlanError):
            self.es.plan(self.shift)

    def test_route_events(self):
        self.es.plan(self.shift)
        r = self.es.approve(self.shift, "Courier_Petrov", "A")
        msgs = self.es.handle_event(self.shift, r.id, "deviation", {"distance_km": "2"})
        self.assertTrue(any("Правило 8" in m for m in msgs))
        msgs = self.es.handle_event(self.shift, r.id, "road_closure", {"segment": "Traffic_Nezavisimosti"})
        self.assertTrue(any("Правило 10" in m for m in msgs))
        self.es.handle_event(self.shift, r.id, "delivered", {})
        msgs = self.es.handle_event(self.shift, r.id, "client_unavailable", {})
        self.assertTrue(any("Правило 11" in m for m in msgs))
        self.assertTrue(any("Правило 12" in m for m in msgs))
        self.assertEqual(self.shift.order("1002").status, "retry")
        self.assertEqual(self.shift.courier("Courier_Petrov").status, "free")

    def test_new_urgent_order_to_free_courier(self):
        self.es.plan(self.shift)
        r = self.es.approve(self.shift, "Courier_Petrov", "A")
        self.es.handle_event(self.shift, r.id, "delivered", {})
        self.es.handle_event(self.shift, r.id, "delivered", {})
        o, err = self.es.validate_order({"address": "ул. Ложинская, 5", "priority": "high", "weight": "3",
                                         "window_start": "11:00", "window_end": "12:30"}, self.shift)
        self.shift.orders.append(o)
        res = self.es.plan(self.shift)
        self.assertTrue(any(t.rule == "9" for t in res.trace))
        self.assertIn(o.id, [x.id for x in res.plan_for("Courier_Petrov").orders])


if __name__ == "__main__":
    unittest.main()
