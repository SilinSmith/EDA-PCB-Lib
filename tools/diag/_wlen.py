# -*- coding: utf-8 -*-
"""量「位号框宽度 / 字高」能不能预测位号字符串长度。

动机：`fusion/priors.py:506 windows_for` 的连号窗口在纯文本上**不可分**
（case 1 真值 `C151..C156`，`[51,56]` 与 `[151,156]` 覆盖数完全相等）。
它自己的 docstring 说「能区分的是图」。但「图」里最便宜的判据就是
**框宽 / 字高 ≈ 字符数** —— 4 字符的 `C151` 比 3 字符的 `C51` 宽。
本脚本就是在真实数据上验这个判据。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("Y:/workspace/EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D  # noqa: E402

R = ROOT / "work/_submit_out220r36/runs"
CASES = {c.case_id: c for c in D.discover()}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def flip(b, H):
    return [b[0], H - b[3], b[2], H - b[1]]


def main():
    ids = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    # len(GT 键) -> [(w, h, w/h, 文本, GT 键, case)]
    tab = defaultdict(list)
    miss = defaultdict(int)
    for cid in ids:
        c = CASES.get(cid)
        if c is None:
            print("缺 case", cid); continue
        gt = load(c.json_path)
        asg = load(R / "stage2" / ("_assigned") / (cid + ".json"))
        from PIL import Image
        H = float(Image.open(c.png_path).size[1])
        gtb = []
        for k, v in (gt.get("components") or {}).items():
            bb = v.get("bbox")
            if bb:
                gtb.append((k, [float(x) for x in bb], flip([float(x) for x in bb], H)))
        n_ok = n_bad = 0
        for cidp, rec in ((asg.get("comps") or {}).items()):
            for ds in (rec.get("desig_slots") or []):
                box = ds.get("box") or ds.get("bbox")
                if not box:
                    continue
                bx = [float(x) for x in box]
                w = bx[2] - bx[0]
                h = bx[3] - bx[1]
                if h <= 0:
                    continue
                best, bi = 0.0, None
                for k, gj, gi in gtb:
                    v = iou(gi, bx)
                    if v > best:
                        best, bi = v, k
                if bi is None or best < 0.25:
                    miss[cid] += 1
                    continue
                tab[len(bi)].append((w, h, w / h, ds.get("text") or ds.get("canon") or "",
                                     bi, cid))
                n_ok += 1
        print("%-6s GT元件 %3d  位号框配上 %3d  配不上 %3d" % (cid, len(gtb), n_ok, miss[cid]))
    print()
    print("%-6s %5s %8s %8s %8s %8s  %s" % ("键长", "n", "宽中位", "高中位", "w/h中位", "w/h p10", "样例"))
    for L in sorted(tab):
        rows = sorted(tab[L], key=lambda r: r[2])
        ws = [r[0] for r in rows]; hs = [r[1] for r in rows]; rs = [r[2] for r in rows]
        med = lambda a: a[len(a) // 2]
        p10 = rs[max(0, int(0.10 * len(rs)))]
        ex = ", ".join("%s/%s(%.2f)" % (r[4], r[3], r[2]) for r in rows[:6])
        print("%-6d %5d %8.1f %8.1f %8.3f %8.3f  %s"
              % (L, len(rows), med(ws), med(hs), med(rs), p10, ex))


if __name__ == "__main__":
    main()
