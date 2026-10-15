# -*- coding: utf-8 -*-
"""**引脚号能不能靠几何推**：官方口径的引脚配对键是 `(元件键, 引脚号)`
（`schematic_parser/metrics.py:453`），而我们的引脚号靠 OCR 读符号旁的小字
（`_pindiag.py` 实测 38% 的 box 引脚记成 `pin_absent`）。如果能从引脚的**位置**
推出编号顺序，就能用「读到的少数几个 + 几何顺序」把整列号补齐/纠正。

本工具只查 GT（不碰我们的产物）：对每个 box 元件，把它的引脚按几种候选顺序排开，
看**引脚号是不是随顺序单调**。若大多数元件在某种顺序下单调，几何推断就成立。

    python -X utf8 work\\tools\\dev\\_pinnum.py --limit 60 --min-pins 4
    python -X utf8 work\\tools\\dev\\_pinnum.py --limit 60 --show 8
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
DR = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
SPLITS = ("200_train_cases", "10_GTcase", "10GT")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import metrics as M                       # noqa: E402
from schematic_parser.data import IC_TYPES, normalise_doc, pin_key_to_number  # noqa: E402


def find_gt(case: str):
    for sp in SPLITS:
        d = DR / sp / case
        if not d.is_dir():
            continue
        js = sorted(d.glob("%s*.json" % case))
        js = [p for p in js if "target_named" in p.name] or js
        for p in js:
            try:
                g = json.loads(p.read_text(encoding="utf-8"))
            except Exception:                                    # noqa: BLE001
                continue
            if g.get("components"):
                return sp, g
    return None, None


def pt(info):
    if not isinstance(info, dict):
        return None
    p = info.get("point")
    if isinstance(p, dict):
        try:
            return (float(p["x"]), float(p["y"]))
        except Exception:                                        # noqa: BLE001
            return None
    return None


def bbox_of(comp: dict):
    b = comp.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in b]
            return x1, y1, x2, y2
        except Exception:                                        # noqa: BLE001
            return None
    return None


def orders(points, nums, bbox):
    """返回 {顺序名: [(num, pt), ...]}，含按角度与按矩形周长的几种走法。"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    out = {}
    # 1) 绕中心的极角（逆时针 / 顺时针）
    ang = [(math.atan2(p[1] - cy, p[0] - cx), n, p) for n, p in zip(nums, points)]
    out["angle_ccw"] = [t[1:] for t in sorted(ang, key=lambda t: t[0])]
    out["angle_cw"] = [t[1:] for t in sorted(ang, key=lambda t: -t[0])]
    # 2) 按边分组、边上按坐标排（KiCad 常见：左从上到下、下从左到右、右从下到上、上从右到左）
    def side_of(p):
        # 点在最靠近哪条边（用框的四条边距离）
        d = {"L": abs(p[0] - x1), "R": abs(p[0] - x2), "B": abs(p[1] - y1), "T": abs(p[1] - y2)}
        return min(d, key=d.get)
    groups = collections.defaultdict(list)
    for n, p in zip(nums, points):
        groups[side_of(p)].append((n, p))
    seqs = {
        "L_B_R_T": [("L", lambda t: t[1][1]), ("B", lambda t: t[1][0]), ("R", lambda t: -t[1][1]), ("T", lambda t: -t[1][0])],
        "L_T_R_B": [("L", lambda t: t[1][1]), ("T", lambda t: t[1][0]), ("R", lambda t: -t[1][1]), ("B", lambda t: -t[1][0])],
        "T_R_B_L": [("T", lambda t: -t[1][0]), ("R", lambda t: t[1][1]), ("B", lambda t: t[1][0]), ("L", lambda t: -t[1][1])],
        "T_L_B_R": [("T", lambda t: t[1][0]), ("L", lambda t: t[1][1]), ("B", lambda t: -t[1][0]), ("R", lambda t: -t[1][1])],
    }
    for name, spec in seqs.items():
        seq = []
        for sd, keyf in spec:
            seq += [t for t in sorted(groups.get(sd, []), key=keyf)]
        out[name] = seq
    return out


def monotone_runs(seq, nums):
    """顺序里最长「连续编号递增」的一段长度（允许步长 1 的严格递增）。"""
    best = cur = 0
    prev = None
    for t in seq:
        n = t[0]
        if not n.isdigit():
            cur, prev = 0, None
            continue
        v = int(n)
        if prev is None or v == prev + 1:
            cur += 1
        elif v > (prev or 0):
            cur = 2          # 允许跳过几号：起一段新的
        else:
            cur = 1
        prev = v
        best = max(best, cur)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--min-pins", type=int, default=4)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    raw = Path(a.cases_file).read_text(encoding="utf-8-sig")
    cases = [t for t in re.split(r"[,\s]+", raw) if t]
    cases = [t for t in cases if re.fullmatch(r"[0-9A-Za-z_]+", t)][a.offset:a.offset + a.limit]

    win = collections.Counter()          # 每个顺序名赢下的元件数
    cover = collections.Counter()        # 每个顺序下「最长递增段」覆盖的引脚总数
    total_pins = 0
    n_comp = 0
    examples = []
    for case in cases:
        sp, g0 = find_gt(case)
        if g0 is None:
            continue
        gt = normalise_doc(g0)
        for key, comp in gt["components"].items():
            if M.canon_type((comp or {}).get("type")) not in IC_TYPES:
                continue
            group = gt["pins"].get(key) or {}
            pts, nums = [], []
            for pk, info in group.items():
                p = pt(info)
                if p is None:
                    continue
                pts.append(p); nums.append(str(pin_key_to_number(pk)))
            if len(pts) < a.min_pins or not all(n.isdigit() for n in nums):
                continue
            bx = bbox_of(comp)
            if bx is None:
                continue
            n_comp += 1
            total_pins += len(pts)
            best_name, best_len = None, -1
            for name, seq in orders(pts, nums, bx).items():
                L = monotone_runs(seq, nums)
                cover[name] += L
                if L > best_len:
                    best_name, best_len = name, L
            win[best_name] += 1
            if a.show and len(examples) < a.show:
                seq = orders(pts, nums, bx)[best_name]
                examples.append((case, key, len(pts),
                                 [(t[0], (round(t[1][0]), round(t[1][1]))) for t in seq]))

    print("== %d 张 case，%d 个 GT box 元件（>= %d 引脚），%d 个引脚 =="
          % (len(cases), n_comp, a.min_pins, total_pins))
    print("-- 哪个顺序最容易单调（赢下来的元件数）--")
    for k, v in win.most_common():
        print("   %-10s %5d  %5.1f%%   最长递增段累计 %d"
              % (k, v, 100.0 * v / max(n_comp, 1), cover[k]))
    if examples:
        print("-- 例子（case, 元件, 引脚数, 最优顺序下的 (号, 点)）--")
        for e in examples:
            print("   ", e[0], e[1], e[2], e[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
