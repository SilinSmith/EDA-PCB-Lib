# -*- coding: utf-8 -*-
"""位号框「宽/高」能不能预测真实位号的字符数 —— 决定连号窗口能不能靠几何定案。

`fusion/priors.py:506 windows_for` 的连号窗口在纯文本上不可分
（case 1 真值 `C151..C156`，`[51,56]` 与 `[151,156]` 覆盖数完全相等），
它自己的 docstring 说「能区分的是图」。这里验最便宜的几何判据：
**框宽 / 字高 ≈ 字符数**（KiCad 字形步进 ≈ 字高）。

做法：把 `_assigned.desig_slots[cid]` 的每个候选文字框（**图像空间**）
用「到 GT 元件框（**JSON 空间**，需翻转）的距离」配最近的那个 GT 元件，
然后分组统计 `len(GT键)` 与 `w/h`。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("Y:/workspace/EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D  # noqa: E402
from PIL import Image  # noqa: E402

R = ROOT / "work/_submit_out220r36/runs"
CASES = {c.case_id: c for c in D.discover()}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def flip(b, H):
    return [b[0], H - b[3], b[2], H - b[1]]


def dist_box(a, b):
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return (dx * dx + dy * dy) ** 0.5


def main():
    ids = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    if ids == ["--all"]:
        ids = [c.case_id for c in D.discover()
               if (R / "stage2/_assigned" / (c.case_id + ".json")).exists()]
        print("扫描 %d 个 case" % len(ids))
    tab = defaultdict(list)          # 真实字符数 -> [(w/h, GT键, 读法, case)]
    near = []
    for cid in ids:
        c = CASES.get(cid)
        if c is None:
            continue
        gt = load(c.json_path)
        asg = load(R / "stage2/_assigned" / (cid + ".json"))
        H = float(Image.open(c.png_path).size[1])
        gtb = []
        for k, v in (gt.get("components") or {}).items():
            bb = v.get("bbox")
            if bb:
                gtb.append((k, flip([float(x) for x in bb], H)))
        for cidp, sl in (asg.get("desig_slots") or {}).items():
            for s in sl:
                bx = [float(x) for x in s["box"]]
                w, h = bx[2] - bx[0], bx[3] - bx[1]
                if h <= 0:
                    continue
                d, gk = min(((dist_box(bx, g), k) for k, g in gtb))
                if d > 70:
                    continue
                tab[len(gk)].append((w / h, gk, s.get("text"), cid))
                if s.get("text") and s["text"].upper() != gk.upper():
                    near.append((cid, cidp, s.get("text"), gk, round(d, 1),
                                 round(w, 1), round(h, 1), round(w / h, 2)))
    print("== 真实位号字符数 -> 框宽/字高 ==")
    print("%-4s %4s %8s %8s %8s   %s" % ("字符", "n", "w/h中位", "w/h p10", "w/h p90", "样例(读法→真值)"))
    for L in sorted(tab):
        rs = sorted(r[0] for r in tab[L])
        med = lambda a: a[len(a) // 2] if a else 0
        ex = ", ".join("%s→%s" % (r[2], r[1]) for r in tab[L][:5])
        print("%-4d %4d %8.3f %8.3f %8.3f   %s"
              % (L, len(rs), med(rs), rs[max(0, len(rs) // 10)], rs[min(len(rs) - 1, len(rs) * 9 // 10)], ex))
    print()
    print("== **读错**的位号（读法与真值不同）n=%d ==" % len(near))
    for r in near[:60]:
        print("  case %-4s %-6s 读 %-8s 真值 %-8s d=%-6s 框 w=%-6s h=%-5s w/h=%s" % r)
    # 关键：读法比真值**短**时，框够不够宽
    print()
    print("== 读法比真值短（少字）时，框装得下真值那么长吗 ==")
    short = [r for r in near if len(r[2] or "") < len(r[3])]
    print("  共 %d 条。框 w/h 与「真值字符数」比较：" % len(short))
    for r in short[:40]:
        print("    case %-4s 读 %-7s(%-2d字) 真值 %-7s(%-2d字) 框 w/h=%-5s"
              % (r[0], r[2], len(r[2] or ""), r[3], len(r[3]), r[7]))


if __name__ == "__main__":
    main()
