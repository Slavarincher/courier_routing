from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
}
_CALL = re.compile(r"^(\w+)\(([\w,\s]*)\)$")
_PLACEHOLDER = re.compile(r"\{([^{}|]+)(?:\|(\w+))?\}")


class WorkingMemory:
    def __init__(self) -> None:
        self.facts: Dict[str, dict] = {}
        self._seq = 0

    def add(self, type_: str, **attrs: Any) -> dict:
        self._seq += 1
        fid = "{}#{}".format(type_, attrs.get("id", self._seq))
        while fid in self.facts:
            self._seq += 1
            fid = "{}#{}".format(type_, self._seq)
        fact = dict(attrs)
        fact["type"] = type_
        fact["_fid"] = fid
        self.facts[fid] = fact
        return fact

    def of_type(self, type_: str) -> List[dict]:
        return [f for f in self.facts.values() if f["type"] == type_]


@dataclass
class Rule:
    id: str
    no: str
    name: str
    phases: List[str]
    salience: int
    patterns: List[dict]
    actions: List[dict]
    explain: str
    order: int = 0


@dataclass
class Firing:
    rule_id: str
    rule_no: str
    rule_name: str
    phase: str
    text: str
    fact_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Activation:
    rule: Rule
    bindings: Dict[str, dict]

    @property
    def key(self) -> Tuple:
        return (self.rule.id,) + tuple(sorted((v, f["_fid"]) for v, f in self.bindings.items()))


class InferenceEngine:
    def __init__(self, rules: List[Rule], parameters: Dict[str, Any],
                 functions: Optional[Dict[str, Callable]] = None, max_cycles: int = 5000):
        self.rules = rules
        self.parameters = parameters
        self.functions = functions or {}
        self.max_cycles = max_cycles

    @classmethod
    def from_json(cls, path: Path, functions: Optional[Dict[str, Callable]] = None) -> "InferenceEngine":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = []
        for i, r in enumerate(data["rules"]):
            rules.append(Rule(
                id=r["id"], no=str(r.get("no", r["id"])), name=r["name"],
                phases=list(r.get("phase", ["classify"])), salience=int(r.get("salience", 0)),
                patterns=r["if"], actions=r["then"], explain=r.get("explain", r["name"]), order=i))
        return cls(rules, data.get("parameters", {}), functions)

    def run(self, wm: WorkingMemory, phase: str) -> List[Firing]:
        fired_keys = set()
        firings: List[Firing] = []
        self._facts_ref = wm.facts
        rules = [r for r in self.rules if phase in r.phases]
        for _ in range(self.max_cycles):
            agenda = [a for a in self._agenda(wm, rules) if a.key not in fired_keys]
            if not agenda:
                break
            agenda.sort(key=lambda a: (-a.rule.salience, a.rule.order))
            for act in agenda:
                if act.key in fired_keys or not self._still_holds(act):
                    continue
                fired_keys.add(act.key)
                firing, _ = self._fire(act, phase, wm)
                firings.append(firing)
        return firings

    def _still_holds(self, act: Activation) -> bool:
        b = act.bindings
        return all(f["_fid"] in self._facts_ref for f in b.values()) and all(
            self._test(t, b) for pat in act.rule.patterns for t in pat.get("test", []))

    def _agenda(self, wm: WorkingMemory, rules: List[Rule]) -> List[Activation]:
        out: List[Activation] = []
        for rule in rules:
            for b in self._match(wm, rule.patterns, 0, {}):
                out.append(Activation(rule, b))
        return out

    def _match(self, wm: WorkingMemory, patterns: List[dict], i: int, bindings: Dict[str, dict]):
        if i == len(patterns):
            yield dict(bindings)
            return
        pat = patterns[i]
        for fact in wm.of_type(pat["type"]):
            bindings[pat["bind"]] = fact
            if all(self._test(t, bindings) for t in pat.get("test", [])):
                yield from self._match(wm, patterns, i + 1, bindings)
            del bindings[pat["bind"]]

    def _test(self, test: list, b: Dict[str, dict]) -> bool:
        left, op, right = test
        try:
            return bool(OPS[op](self.resolve(left, b), self._value(right, b)))
        except (TypeError, KeyError):
            return False

    def resolve(self, expr: str, b: Dict[str, dict]) -> Any:
        m = _CALL.match(expr)
        if m:
            fn = self.functions[m.group(1)]
            args = [b[a.strip()] for a in m.group(2).split(",") if a.strip()]
            return fn(*args)
        var, _, attr = expr.partition(".")
        fact = b[var]
        return fact.get(attr) if attr else fact

    def _value(self, v: Any, b: Dict[str, dict]) -> Any:
        if isinstance(v, str) and v.startswith("@"):
            return self.resolve(v[1:], b)
        return v

    def render(self, template: str, b: Dict[str, dict]) -> str:
        def repl(m):
            try:
                val = self.resolve(m.group(1).strip(), b)
            except KeyError:
                return m.group(0)
            return _format(val, m.group(2))
        return _PLACEHOLDER.sub(repl, template)

    def _fire(self, act: Activation, phase: str, wm: WorkingMemory) -> Tuple[Firing, bool]:
        changed = False
        warns: List[str] = []
        b = act.bindings
        for action in act.rule.actions:
            if "set" in action:
                var, _, attr = action["set"].partition(".")
                value = self._value(action["value"], b)
                if b[var].get(attr) != value:
                    b[var][attr] = value
                    changed = True
            elif "assert" in action:
                attrs = {k: self._value(v, b) for k, v in action["assert"].items()}
                type_ = attrs.pop("type")
                exists = any(all(f.get(k) == v for k, v in attrs.items()) for f in wm.of_type(type_))
                if not exists:
                    wm.add(type_, **attrs)
                    changed = True
            elif "warn" in action:
                warns.append(self.render(action["warn"], b))
        firing = Firing(act.rule.id, act.rule.no, act.rule.name, phase,
                        self.render(act.rule.explain, b), [f["_fid"] for f in b.values()], warns)
        return firing, changed

    def infer(self, wm: WorkingMemory, phase: str) -> List[Firing]:
        if not wm.of_type("settings"):
            wm.add("settings", id="s", **self.parameters)
        return self.run(wm, phase)


def _format(val: Any, fmt: Optional[str]) -> str:
    if fmt == "time" and isinstance(val, (int, float)):
        m = int(round(val))
        return "{:02d}:{:02d}".format((m // 60) % 24, m % 60)
    if fmt == "km" and isinstance(val, (int, float)):
        return "{:.1f}".format(val).replace(".", ",")
    if isinstance(val, float):
        return ("{:.1f}".format(val) if not val.is_integer() else str(int(val))).replace(".", ",")
    return str(val)
