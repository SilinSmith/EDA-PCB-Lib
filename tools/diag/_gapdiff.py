"""比较两跑的 stage1 段：某条竖线（默认 x≈843，case 1）上的跨度。

用法：`python -X utf8 work\\_gapdiff.py`（默认比 ckd0 基线与 gapb0 接线臂的 case 1）
"""
import json
import os

ROOT = r"Y:\workspace\EDA\work"


def verts(run, case, xc, half=5.0):
    p = os.path.join(ROOT, "_fast_out_" + run, "runs", "stage1", "%s.stage1.json" % case)
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    segs = (d.get("wire_detail") or {}).get("segments") or []
    out = []
    for s in segs:
        (x0, y0), (x1, y1) = s[0], s[-1]
        if abs(x0 - x1) > 2.5:
            continue
        if not (xc - half <= (x0 + x1) / 2.0 <= xc + half):
            continue
        out.append((round(min(y0, y1), 1), round(max(y0, y1), 1), round((x0 + x1) / 2.0, 1),
                    round(max(y0, y1) - min(y0, y1), 1)))
    out.sort()
    return len(segs), out


for run in ("ckd0", "gapb0"):
    n, vs = verts(run, "1", 843.0)
    print("=== %s：case 1 段数 %d，x≈843 竖段 %d 条 ===" % (run, n, len(vs)))
    for a, b, x, L in vs:
        print("   y=[%7.1f,%7.1f] x=%6.1f L=%5.1f" % (a, b, x, L))
