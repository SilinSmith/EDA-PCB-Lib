"""彩色图上量「符号引脚桩」与「导线端点」相对 GT 引脚坐标的偏差。

目的：判断能否用颜色（红桩 / 绿线）推出引脚真实长度，取代写死的 PIN_OUT_BOX=12。

用法：
    python -X utf8 eda-pcb\\tools\\dev\\_pincolour.py --cases 2,3,4 --show 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
GT_DIR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"

RED_R = 25
GRN_R = 15
DARK = 200


def is_red(p) -> bool:
    r, g, b = int(p[0]), int(p[1]), int(p[2])
    return r > g + RED_R and r > b + RED_R and r > 60


def is_green(p) -> bool:
    r, g, b = int(p[0]), int(p[1]), int(p[2])
    return g > r + GRN_R and g > b + GRN_R and g > 60


def is_ink(p) -> bool:
    return max(int(p[0]), int(p[1]), int(p[2])) < DARK


def side_of(box, x, y):
    x0, y0, x1, y1 = box
    dl, dr = abs(x - x0), abs(x - x1)
    dt, db = abs(y - y0), abs(y - y1)
    m = min(dl, dr, dt, db)
    if m == dl:
        return "L", -1
    if m == dr:
        return "R", 1
    if m == dt:
        return "T", -1
    return "B", 1


def profile_rows(a, H, box, x, y, side, span=34):
    """在引脚行附近取最像引脚的一行，返回该行分类串与起点 x。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if side in ("L", "R"):
        edge = x0 if side == "L" else x1
        horiz = True
    else:
        edge = y1 if side == "T" else y0  # y_out 上边缘是大 y
        horiz = False
    best = None
    for dy in (-2, -1, 0, 1, 2):
        if horiz:
            yi = H - 1 - int(round(y)) + dy
            if not (0 <= yi < a.shape[0]):
                continue
            xs = range(edge - span, edge + span) if side == "L" else range(edge - span, edge + span)
            cells = [(xx, a[yi, xx]) for xx in xs if 0 <= xx < a.shape[1]]
        else:
            xi = int(round(x)) + dy
            if not (0 <= xi < a.shape[1]):
                continue
            ys = range(edge - span, edge + span)
            cells = [(H - 1 - yy, a[H - 1 - yy, xi]) for yy in ys if 0 <= H - 1 - yy < a.shape[0]]
        red = sum(1 for _p, px in cells if is_red(px))
        grn = sum(1 for _p, px in cells if is_green(px))
        sc = red * 2 + grn
        if best is None or sc > best[0]:
            best = (sc, cells)
    if not best:
        return []
    return best[1]


def run_len(cells, edge, pred, outward):
    """从框边向外连续满足 pred 的像素数（含抗锯齿跳空 ≤1px）。"""
    d = {p: px for p, px in cells}
    n = 0
    gap = 0
    s = edge
    while True:
        s += outward
        px = d.get(s)
        if px is None:
            break
        if pred(px):
            n += 1
            gap = 0
        else:
            gap += 1
            if gap > 1:
                break
    return n


def first_dist(cells, edge, pred, outward, limit=48):
    """框边向外第一个满足 pred 的像素距离（没有则 None）。"""
    d = {p: px for p, px in cells}
    for k in range(1, limit + 1):
        px = d.get(edge + outward * k)
        if px is None:
            break
        if pred(px):
            return k
    return None


def last_dist(cells, edge, pred, outward, limit=48):
    """框边向外最后一个满足 pred 的像素距离（没有则 None）。"""
    d = {p: px for p, px in cells}
    last = None
    for k in range(1, limit + 1):
        px = d.get(edge + outward * k)
        if px is None:
            break
        if pred(px):
            last = k
    return last


def analyse(cid, show=0):
    img_p = GT_DIR / cid / f"{cid}.png"
    im = Image.open(img_p).convert("RGB")
    a = np.array(im)
    H = a.shape[0]
    gt = json.loads((GT_DIR / cid / f"{cid}.json").read_text(encoding="utf-8"))
    comps = gt.get("components") or {}
    pins = gt.get("pins") or {}
    rows = []
    for key, pv in pins.items():
        c = comps.get(key)
        if not c:
            continue
        box = c["bbox"]
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            continue
        for pid, p in pv.items():
            x, y = p["point"]["x"], p["point"]["y"]
            side, out = side_of(box, x, y)
            edge = box[0] if side == "L" else box[2] if side == "R" else (box[3] if side == "T" else box[1])
            off_gt = abs(x - edge) if side in ("L", "R") else abs(y - edge)
            cells = profile_rows(a, H, box, x, y, side)
            if not cells:
                continue
            red = run_len(cells, int(round(edge)), is_red, out)
            red_end = last_dist(cells, int(round(edge)), is_red, out) or 0
            grn_first = first_dist(cells, int(round(edge)), is_green, out)
            grn_end = last_dist(cells, int(round(edge)), is_green, out) or 0
            ink_end = last_dist(cells, int(round(edge)), is_ink, out) or 0
            rows.append(dict(case=cid, key=key, pid=pid, side=side, off_gt=round(off_gt, 1),
                             red=red, red_end=red_end, green=grn_end,
                             green_first=(-1 if grn_first is None else grn_first),
                             ink=ink_end, ink0=run_len(cells, int(round(edge)), is_ink, out),
                             v=c.get("value")))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="2,3,4,5,6,7,8,9,10")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args(argv)
    allrows = []
    for cid in [c.strip() for c in args.cases.split(",") if c.strip()]:
        try:
            allrows += analyse(cid, args.show)
        except Exception as e:  # noqa: BLE001
            print(f"case {cid}: {e}")
    if not allrows:
        return 1
    print(f"\n共 {len(allrows)} 个「大框」GT 引脚\n")
    # 每「元件+边」聚合：GT 偏移是不是常量
    agg = defaultdict(list)
    for r in allrows:
        agg[(r["case"], r["key"], r["side"])].append(r)
    print(f"{'case':>4} {'comp':>7} {'side':>4} {'n':>3} {'off_gt':>7} {'red_end':>8} {'grn_first':>10} "
          f"{'grn_end':>8} {'ink_end':>8} {'ink0':>6} {'red-off':>8}")
    for k in sorted(agg):
        rs = agg[k]
        med = lambda key: float(np.median([r[key] for r in rs]))  # noqa: E731
        if len(rs) < 2 and k[0] not in ("1",):
            continue
        print(f"{k[0]:>4} {k[1]:>7} {k[2]:>4} {len(rs):>3} {med('off_gt'):>7.1f} {med('red_end'):>8.1f} "
              f"{med('green_first'):>10.1f} {med('green'):>8.1f} {med('ink'):>8.1f} {med('ink0'):>6.1f} "
              f"{med('red_end') - med('off_gt'):>8.1f}")
    print("\n全样本偏差（red_end − off_gt）:", dict(sorted(Counter(int(r["red_end"] - r["off_gt"]) for r in allrows).items())))
    print("全样本偏差（grn_first − off_gt）:", dict(sorted(Counter(int(r["green_first"] - r["off_gt"]) for r in allrows if r["green_first"] >= 0).items())))
    print("全样本偏差（ink_end − off_gt）:", dict(sorted(Counter(int(r["ink"] - r["off_gt"]) for r in allrows).items())))
    for name, fn in (("flat12", lambda r: 12.0), ("red_end", lambda r: r["red_end"]),
                     ("red_end-3", lambda r: r["red_end"] - 3), ("grn_first", lambda r: r["green_first"]),
                     ("ink_end", lambda r: r["ink"])):
        ok = sum(1 for r in allrows if r["green_first"] >= 0 and abs(r["off_gt"] - fn(r)) <= 5)
        n = sum(1 for r in allrows if r["green_first"] >= 0)
        print("  %-11s 5px 内命中 %.1f%% (%d/%d)" % (name, 100.0 * ok / max(1, n), ok, n))
    bad = [r for r in allrows if abs(r["off_gt"] - 12) > 5]
    print(f"\nflat12 超 5px 的 {len(bad)} 个（前 25）：")
    for r in bad[:25]:
        print("   case %-3s %-7s %-8s %s off_gt %5.1f red_end %2d grn_first %3d ink_end %2d  %s" % (
            r["case"], r["key"], r["pid"], r["side"], r["off_gt"], r["red_end"], r["green_first"], r["ink"], r["v"]))
    if args.show:
        print(f"\n前 {args.show} 行明细：")
        for r in allrows[: args.show]:
            print("   case %-3s %-7s %-8s %s off_gt %5.1f red_end %2d grn_first %3d ink_end %2d  %s" % (
                r["case"], r["key"], r["pid"], r["side"], r["off_gt"], r["red_end"], r["green_first"], r["ink"], r["v"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
