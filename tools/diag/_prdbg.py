# -*- coding: utf-8 -*-
"""临时调试：看 priors.window_for 在 case 1 那组读数上算出什么。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
sys.stdout.reconfigure(encoding="utf-8")

from cvparts.hypo import CaseHypotheses            # noqa: E402
from fusion import priors as PR                    # noqa: E402

P = PR.Priors()


class _P:
    def __init__(self):
        self.hy = CaseHypotheses("t")
        self.meta = {}
        self.notes = []
        self.doc = {}
        self.assigned = {}

    def slot_ids(self):
        return sorted(self.hy.slots)

    def best(self, sid):
        s = self.hy.slots.get(sid)
        return s.best() if s else None

    def alts(self, sid):
        s = self.hy.slots.get(sid)
        return list(s.alts) if s else []

    def add(self, sid, v, score=0.0, source="", evidence=None, question="",
            meta=None, **kw):
        self.hy.add(sid, v, score, source, evidence or {}, question)
        if meta:
            self.meta[sid] = meta
        return self.hy.slots[sid]

    def note(self, *a, **kw):
        self.notes.append((a, kw))


reads = ["C51", "C15", "C5", "C54", "C5", "C15"]
p = _P()
for i, r in enumerate(reads):
    p.add("comp.c%d.key" % i, r, score=1.0 - 0.01 * i, source="ocr",
          meta={"type": "c", "cid": "c%d" % i})

slots = [s for s in p.slot_ids() if s.startswith("comp.") and s.endswith(".key")]
print("slots:", slots)
for s in slots:
    print("  %-16s best=%s alts=%s" % (s, p.best(s).value,
                                       [a.value for a in p.alts(s)]))
    print("      repairs of %s -> %s" % (p.best(s).value,
                                         P.repair_variants(str(p.best(s).value), "c")))

w = P.window_for("C", slots, p)
print("window:", w)
st = P.annotate(p)
print("annotate:", st)
for s in slots:
    print("  %-16s -> %s" % (s, [(a.value, round(a.score, 3), a.source)
                                 for a in p.alts(s)]))
