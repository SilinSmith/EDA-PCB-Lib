# -*- coding: utf-8 -*-
"""LED1 的脚为什么没进任何网：把该脚附近的**我方边端点**摊开看。

用法：python -X utf8 work/_ledpins.py [--pred work/_fast_out_finalrd8] [--cases 7,10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# archived under eda-pcb/tools/diag/ => repo root is 3 levels up
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402


def edges_of(net):
    out = []
    e = net.get("edges")
    if isinstance(e, dict):
        for k, seg in e.items():
            if isinstance(seg, list) and len(seg) == 2:
                out.append((k, (float(seg[0]["x"]), float(seg[0]["y"])),
                            (float(seg[1]["x"]), float(seg[1]["y"]))))
    return out


def seg_dist(px, py, p0, p1):
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    L2 = vx * vx + vy * vy
    if L2 <= 1e-9:
        return ((px - p0[0]) ** 2 + (py - p0[1]) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - p0[0]) * vx + (py - p0[1]) * vy) / L2))
    cx, cy = p0[0] + t * vx, p0[1] + t * vy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out_finalrd8"))
    ap.add_argument("--cases", default="7,10")
    ap.add_argument("--comp", default="LED1")
    ap.add_argument("--tol", type=float, default=30.0)
    a = ap.parse_args()
    want = set(a.cases.split(","))
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or "")
        if cid not in want:
            continue
        pp = Path(a.pred) / "final_results" / (cid + "_Pred.json")
        gt_doc = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pr_doc = json.loads(pp.read_text(encoding="utf-8"))
        for tag, doc in (("GT", gt_doc), ("PR", pr_doc)):
            comp = (doc.get("components") or {}).get(a.comp)
            pins = (doc.get("pins") or {}).get(a.comp) or {}
            print("== case %s %s %s bbox=%s" % (cid, tag, a.comp,
                                                (comp or {}).get("bbox")))
            for pk in sorted(pins):
                pt = (pins[pk] or {}).get("point") or {}
                px, py = float(pt.get("x", 0)), float(pt.get("y", 0))
                near = []
                for nm, net in (doc.get("nets") or {}).items():
                    for ek, p0, p1 in edges_of(net):
                        for pe in (p0, p1):
                            d = ((pe[0] - px) ** 2 + (pe[1] - py) ** 2) ** 0.5
                            if d <= a.tol:
                                near.append((d, nm, ek, pe))
                near.sort()
                print("   %s %-8s (%6.1f,%6.1f) 附近边端点 %d 个" % (tag, pk, px, py, len(near)))
                for d, nm, ek, pe in near[:4]:
                    net = (doc.get("nets") or {})[nm]
                    print("        d=%5.1f  %s.%s 端 (%6.1f,%6.1f)  该网 refs=%s"
                          % (d, nm, ek, pe[0], pe[1], net.get("hyperGraph")))
                segs_hit = []
                for nm, net in (doc.get("nets") or {}).items():
                    for ek, p0, p1 in edges_of(net):
                        d = seg_dist(px, py, p0, p1)
                        if d <= a.tol:
                            segs_hit.append((d, nm, ek, p0, p1))
                segs_hit.sort()
                for d, nm, ek, p0, p1 in segs_hit[:3]:
                    print("        线段距 %5.1f  %s.%s (%6.1f,%6.1f)-(%6.1f,%6.1f)"
                          % (d, nm, ek, p0[0], p0[1], p1[0], p1[1]))
            nets_with = [nm for nm, net in (doc.get("nets") or {}).items()
                         if a.comp in str(net.get("hyperGraph") or "")]
            print("   %s 提到 %s 的网：%s" % (tag, a.comp, nets_with or "（无）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
