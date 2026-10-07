# -*- coding: utf-8 -*-
"""把某个 case 的**全部**位号候选框按「前缀 + y 带」排成轴打印出来，
并给出每个框最近的真值键 —— 用来给 `fusion/axis.py` 的自检取真实夹具。"""
import json
import sys
from pathlib import Path

ROOT = Path("Y:/workspace/EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D  # noqa: E402
from PIL import Image  # noqa: E402

R = ROOT / "work/_submit_out220r36/runs"
CASES = {c.case_id: c for c in D.discover()}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dist_box(a, b):
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return (dx * dx + dy * dy) ** 0.5


def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "1"
    pref = sys.argv[2] if len(sys.argv) > 2 else "C"
    c = CASES[cid]
    gt = load(c.json_path)
    asg = load(R / "stage2/_assigned" / (cid + ".json"))
    H = float(Image.open(c.png_path).size[1])
    gtb = []
    for k, v in (gt.get("components") or {}).items():
        bb = v.get("bbox")
        if bb:
            g = [float(bb[0]), H - float(bb[3]), float(bb[2]), H - float(bb[1])]
            gtb.append((k, g))
    rows = []
    for cp, sl in (asg.get("desig_slots") or {}).items():
        for s in sl:
            t = s.get("text") or ""
            if not t.upper().startswith(pref):
                continue
            bx = [float(x) for x in s["box"]]
            d, gk = min(((dist_box(bx, g), k) for k, g in gtb))
            rows.append((round((bx[1] + bx[3]) / 2, 1), round(bx[0], 1), cp, s, gk, round(d, 1)))
    rows.sort()
    print("case %s 前缀 %s 的位号候选框 n=%d" % (cid, pref, len(rows)))
    print("%-8s %-8s %-7s %-9s %-8s %-6s %-6s %-6s %-6s %s"
          % ("y中位", "x0", "cid", "读法", "最近真值", "d", "w", "h", "w/h", "conf/分"))
    for yc, x0, cp, s, gk, d in rows:
        bx = [float(x) for x in s["box"]]
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        print("%-8s %-8s %-7s %-9s %-8s %-6s %-6s %-6s %-6s %.3f / %.3f"
              % (yc, x0, cp, s.get("text"), gk, d, w, h, round(w / h, 2),
                 s.get("conf", 0), s.get("score", 0)))


if __name__ == "__main__":
    main()
