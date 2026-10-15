# -*- coding: utf-8 -*-
"""§5.115 取证：队友某次跑的 stage1 文档里「引脚是怎么来的」——按 cid dump 来源证据。

打印：组件表（id/type/bbox）、`pin_color`（彩色实测记录，按边）、`pin_mono`（黑白框
审计）、`pin_evidence`、`passive_pin_report`、`non_ic_pin_structure`、`pins`（最终脚，
含它们的 source/证据字段，若脚本里带的话）。

用法：python -X utf8 work\\_pr3stage.py --case 8 --run pr3b [--comp c0000] [--max 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="8")
ap.add_argument("--run", default="pr3b")
ap.add_argument("--comp", default="")
ap.add_argument("--max", type=int, default=12)
a = ap.parse_args()

p = ROOT / "work" / ("_fast_out_%s" % a.run) / "runs" / "stage1" / ("%s.stage1.json" % a.case)
d = json.loads(p.read_text("utf-8"))
print("== %s ==" % p)
print("top keys:", sorted(d.keys()))
comps = d.get("components") or []
for c in comps[:6]:
    print("  comp %-6s %-6s %-10s %s" % (c.get("id"), c.get("type"), c.get("source"),
                                         [round(float(v), 1) for v in (c.get("bbox_img") or [])]))
print("  ... 共 %d 个组件" % len(comps))
want = a.comp or next((str(c.get("id")) for c in comps
                       if str(c.get("type", "")).lower().startswith("box")), None)
print("聚焦组件 =", want)
for key in ("pin_color", "pin_mono", "pin_evidence", "passive_pin_report", "non_ic_pin_structure"):
    sec = d.get(key)
    if not isinstance(sec, dict) or want not in sec:
        print("-- %s: 无 %s 的记录" % (key, want))
        continue
    val = sec[want]
    print("-- %s[%s]:" % (key, want), type(val).__name__,
          ("keys=" + str(sorted(val.keys()))) if isinstance(val, dict) else ("len=%d" % len(val)))
    if isinstance(val, dict):
        for k, v in list(val.items())[:6]:
            if isinstance(v, list):
                print("     %-12s list(%d)" % (k, len(v)))
                for r in v[:a.max]:
                    if isinstance(r, dict):
                        keep = {kk: r.get(kk) for kk in
                                ("edge", "point_img", "red_tip_img", "green_start_img", "connected",
                                 "source", "confidence", "frame_confirmed", "normal_ink_run",
                                 "index", "role", "added", "calibrated", "point") if kk in r}
                        print("        ", json.dumps(keep, ensure_ascii=False))
                    else:
                        print("        ", r)
            else:
                print("     %-12s %s" % (k, json.dumps(v, ensure_ascii=False)[:160]))
    else:
        for r in val[:a.max]:
            print("     ", json.dumps(r, ensure_ascii=False)[:220])
pins = (d.get("pins") or {}).get(want) or {}
print("-- pins[%s] n=%d" % (want, len(pins)))
for k, v in list(pins.items())[:a.max]:
    if isinstance(v, dict):
        keep = {kk: v.get(kk) for kk in
                ("point", "name", "source", "evidence", "confidence", "side", "role") if kk in v}
        print("     %-8s %s" % (k, json.dumps(keep, ensure_ascii=False)))
