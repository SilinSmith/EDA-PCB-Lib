# -*- coding: utf-8 -*-
"""临时调试 doc.apply。"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "eda-pcb"))
from fusion import doc as D

doc = {
    "case": "x",
    "components": {
        "p001": {"type": "r", "Name": "p001", "bbox": [0, 0, 10, 10]},
        "p002": {"type": "box", "Name": "RP2350B", "bbox": [20, 0, 60, 40]},
        "C15": {"type": "c", "Name": "C15", "value": "100nF", "bbox": [70, 0, 80, 10]},
    },
    "pins": {"p001": {"pin_1": {"pinname": "1"}}, "p002": {"pin_1": {"pinname": "VDD"}}},
    "nets": {"net_1": {"hyperGraph": "(p001.1,p002.3)"},
             "net_2": {"hyperGraph": "(C15.1)", "label": "GND"}},
}
stage1 = {"components": []}
assigned = {"comps": {"c0000": {"type": "r", "designator": {"text": "p001"}},
                      "c0001": {"type": "box", "designator": {"text": "p002"}},
                      "c0002": {"type": "c", "designator": {"text": "C15"}}}}
sol = {"assign": {"comp.c0000.key": ("R7", 1.15, "vlm:enrich"),
                  "comp.c0001.key": ("U3", 1.15, "vlm:enrich"),
                  "comp.c0002.key": ("C15", 1.15, "vlm:enrich"),
                  "comp.c0001.partname": ("W25Q128JVPIM", 1.15, "vlm:enrich"),
                  "comp.c0002.partname": ("XX", 1.15, "vlm:enrich")}}

D.vlm_keymap = D._fake_vlm_keymap
before = D.invert_keymap(doc, stage1, assigned)
print("before =", before)
dec = D._decided(sol)
print("dec =", dec)
ren = {}
import re
from fusion import priors as PR
from fusion.claims import FKEY_RE
for sid, val in dec.items():
    m = re.match(r"^comp\.([^.]+)\.key$", sid)
    if not m:
        continue
    cid = m.group(1)
    cur = before.get(cid)
    print("  sid=%s cid=%s cur=%s val=%s desig=%s fkey=%s" % (
        sid, cid, cur, val, bool(PR.DESIG_RE.match(val)), bool(FKEY_RE.match(val))))
    if not cur or val == cur:
        continue
    if not PR.DESIG_RE.match(val) or FKEY_RE.match(val):
        continue
    ren[cur] = val
print("ren =", ren)
keep = D.rename_keys(doc, ren)
print("keep =", keep)
print("comps now =", list(doc["components"]))
