from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class Sym(str):
    pass


class Str(str):
    pass


class Ref(str):
    pass


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
        elif ch == ";":
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == '"':
            j, buf = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            tokens.append(Str("".join(buf)))
            i = j + 1
        elif ch == "[":
            j = text.index("]", i)
            tokens.append(Ref(text[i + 1:j]))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n()";':
                j += 1
            tokens.append(Sym(text[i:j]))
            i = j
    return tokens


def parse_forms(text: str) -> List[list]:
    stack: List[list] = [[]]
    for t in tokenize(text):
        if type(t) is str and t == "(":
            stack.append([])
        elif type(t) is str and t == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    return stack[0]


def _atom(v: Any) -> Any:
    if isinstance(v, (Str, Ref)):
        return v
    if isinstance(v, Sym):
        for conv in (int, float):
            try:
                return conv(v)
            except ValueError:
                pass
        return str(v)
    return v


@dataclass
class SlotInfo:
    name: str
    multi: bool = False
    type: str = "STRING"
    allowed_values: List[str] = field(default_factory=list)
    range: Optional[tuple] = None
    default: Any = None


@dataclass
class ClassInfo:
    name: str
    doc: str = ""
    parents: List[str] = field(default_factory=list)
    abstract: bool = False
    slots: List[str] = field(default_factory=list)


@dataclass
class Instance:
    name: str
    cls: str
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, slot: str, default: Any = None) -> Any:
        return self.values.get(slot, default)


class KnowledgeBase:

    def __init__(self, pont_path: Path, pins_path: Path):
        self.pont_path, self.pins_path = Path(pont_path), Path(pins_path)
        self.classes: Dict[str, ClassInfo] = {}
        self.slots: Dict[str, SlotInfo] = {}
        self.instances: Dict[str, Instance] = {}
        self._load_pont()
        self._load_pins()

    def _load_pont(self) -> None:
        for form in parse_forms(_read_text(self.pont_path)):
            if not form or form[0] != "defclass":
                continue
            name = str(form[1])
            info = ClassInfo(name=name)
            for part in form[2:]:
                if isinstance(part, Str):
                    info.doc = str(part)
                elif isinstance(part, list) and part:
                    head = part[0]
                    if head == "is-a":
                        info.parents = [str(p) for p in part[1:]]
                    elif head == "role":
                        info.abstract = str(part[1]) == "abstract"
                    elif head in ("single-slot", "multislot"):
                        slot = self._parse_slot(part)
                        info.slots.append(slot.name)
                        known = self.slots.get(slot.name)
                        if known is None or (not known.allowed_values and slot.allowed_values) \
                                or (known.range is None and slot.range is not None):
                            self.slots[slot.name] = slot
            if not name.startswith("%3A"):
                self.classes[name] = info

    @staticmethod
    def _parse_slot(part: list) -> SlotInfo:
        slot = SlotInfo(name=str(part[1]), multi=(part[0] == "multislot"))
        for facet in part[2:]:
            if not isinstance(facet, list) or not facet:
                continue
            key = facet[0]
            if key == "type":
                slot.type = str(facet[1]).upper()
            elif key == "allowed-values":
                slot.allowed_values = [str(v) for v in facet[1:]]
            elif key == "range":
                lo, hi = (_atom(v) for v in facet[1:3])
                slot.range = (None if lo == "?VARIABLE" else lo, None if hi == "?VARIABLE" else hi)
            elif key == "default":
                slot.default = _atom(facet[1]) if len(facet) > 1 else None
        return slot

    def _load_pins(self) -> None:
        for form in parse_forms(_read_text(self.pins_path)):
            if len(form) < 3 or not isinstance(form[0], Ref) or form[1] != "of":
                continue
            inst = Instance(name=str(form[0]), cls=str(form[2]))
            for part in form[3:]:
                if not isinstance(part, list) or not part:
                    continue
                slot = str(part[0])
                values = [_atom(v) for v in part[1:]]
                info = self.slots.get(slot)
                inst.values[slot] = values if (info and info.multi) else (values[0] if values else None)
            self.instances[inst.name] = inst

    def is_a(self, cls: str, ancestor: str) -> bool:
        seen = set()
        stack = [cls]
        while stack:
            c = stack.pop()
            if c == ancestor:
                return True
            if c in seen or c not in self.classes:
                continue
            seen.add(c)
            stack.extend(self.classes[c].parents)
        return False

    def instances_of(self, cls: str) -> List[Instance]:
        return [i for i in self.instances.values() if self.is_a(i.cls, cls)]

    def get(self, name: Optional[str]) -> Optional[Instance]:
        return self.instances.get(str(name)) if name else None

    def class_label(self, cls: str) -> str:
        info = self.classes.get(cls)
        return info.doc if info and info.doc else cls

    def check_value(self, slot_name: str, value: Any) -> Optional[str]:
        slot = self.slots.get(slot_name)
        if slot is None:
            return None
        if slot.allowed_values and str(value) not in slot.allowed_values:
            return "допустимые значения: " + ", ".join(slot.allowed_values)
        if slot.type in ("FLOAT", "INTEGER"):
            try:
                num = float(value)
            except (TypeError, ValueError):
                return "нужно число"
            if slot.range:
                lo, hi = slot.range
                if (lo is not None and num < lo) or (hi is not None and num > hi):
                    return "допустимый диапазон: {} - {}".format(_fmt(lo), _fmt(hi))
        return None

    def summary(self) -> str:
        return "{} классов, {} слотов, {} экземпляров".format(
            len(self.classes), len(self.slots), len(self.instances))


def _fmt(x: Any) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)
