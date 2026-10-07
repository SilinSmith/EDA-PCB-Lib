"""把 `_namefix.py` 找出的「名错」引脚连同它在 `_assigned` 里的来源文字框一起打出来。

用法：
    python work\\tools\\diag\\_nameboxes.py --case 8 [--pred work\\_gt10_vlmrun2\\runs\\stage2] [--tol 5]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT, os.path.join(ROOT, "eda-pcb")):
    if p not in sys.path:
        sys.path.insert(0, p)

from schematic_parser import metrics as M                     # noqa: E402
from schematic_parser.data import find_case, load_json        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pred", default=r"work\_gt10_vlmrun2\runs\stage2")
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()

    c = find_case(a.case)
    gt = load_json(c.json_path)
    pr = load_json(os.path.join(a.pred, "%s.json" % a.case))
    asg_path = os.path.join(a.pred, "_assigned", "%s.json" % a.case)
    asg = load_json(asg_path) if os.path.exists(asg_path) else {}
    cfg = M.EvalConfig()
    _f, km = M.assign_components(gt, pr, cfg)
    back = {g: k for k, g in km.items()}
    gcomp = gt.get("components") or {}
    gt_pins = gt.get("pins") or {}
    pr_pins = pr.get("pins") or {}
    # 反向：我们的元件键 -> stage1 的 cid（`assigned.comps` 的键）
    comps_asg = asg.get("comps") or {}
    # cid -> 我们的 key 需要自己重建：assemble 里 key 是按顺序发的；这里用 bbox 对齐
    cid_of_bbox = {}
    for cid, rec in comps_asg.items():
        bb = rec.get("bbox")
        if bb:
            cid_of_bbox[tuple(round(float(x), 1) for x in bb)] = cid

    for gk, pk in back.items():
        pu = gt_pins.get(gk) or {}
        qu = pr_pins.get(pk) or {}
        if not pu or not qu:
            continue
        gp = [(float(v["point"]["x"]), float(v["point"]["y"]), k, str(v.get("pinname") or ""))
              for k, v in pu.items()]
        qp = [(float(v["point"]["x"]), float(v["point"]["y"]), k, str(v.get("pinname") or ""))
              for k, v in qu.items()]
        for gx, gy, gkey, gname in sorted(gp, key=lambda t: (t[1], t[0])):
            best = None
            for qx, qy, qkey, qname in qp:
                d = math.hypot(gx - qx, gy - qy)
                if best is None or d < best[0]:
                    best = (d, qx, qy, qkey, qname)
            if best is None or best[0] > a.tol:
                continue
            if best[4] == gname:
                continue
            # 找这个名字来自哪个文字框
            box = None
            gv = gv_bb = None
            for cid, rec in comps_asg.items():
                rp = rec.get("pins") or {}
                for kk, rr in rp.items():
                    nm = (rr.get("name") or {})
                    if nm.get("text") == best[4]:
                        box = nm.get("box")
                        gv_bb = rr
                        break
                if box:
                    break
            print("GT %-10s %-8s %-14r | ours %-9s %-14r d=%5.1f box=%s"
                  % (gk, gkey, gname, best[3], best[4], best[0],
                     [round(float(x), 1) for x in box] if box else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
