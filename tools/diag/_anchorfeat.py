# -*- coding: utf-8 -*-
"""锚点判别特征：**哪个图像可观测的量**能预测逐（元件,边）外伸量？

已知（`work\\_anchorfit.py`）：
  · 外伸量在（元件,边）内严格恒定 ⇒ 目标是每边一个标量；
  · 逐（元件,边）已知值 +2.71 分（引脚 F1 0.8166→0.9251），逐案常数 +1.06；
  · **尺度假设已排除**（`work\\_anchorscale.py`：框高/框宽/节距/脚数 相关都 <0.4）。

本探针量三个候选特征（都只用图像 + 我方 stage1 的 bbox/text 框，不用 GT 定位）：
  A) 引脚号文字框：该脚外侧最近文本框的「内边到框边的距离」`d_text_inner`、
     中心距离 `d_text_center`、字高 `text_h`（标注工具的号字锚点若是脚点，这三者会强相关）
  C) 图像剖面：从框边沿法向外推 40px 的像素分类（暗红桩/绿导线/白底），
     量 `stub_run`（暗红游程）、`green_start`（第一个绿像素距离）、`white_start`
逐组打印 GT 外伸量 + 特征中位，并给跨组相关与「偏离案例中位 >5px 的组能否被分开」。

用法：
    python -X utf8 work\\_anchorfeat.py [--cases 1,8] [--run sjndef]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc)
from schematic_parser.metrics import canon_type                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--run", default="sjndef")
ap.add_argument("--reach", type=int, default=40)
a = ap.parse_args()


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get("x"), raw.get("y"))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def box_of(comp):
    bb = (comp or {}).get("bbox")
    if isinstance(bb, dict):
        return (float(bb["x0"]), float(bb["y0"]), float(bb["x1"]), float(bb["y1"]))
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return tuple(float(v) for v in bb)
    return None


def edge_dist(p, box):
    x0, y0, x1, y1 = box
    return max([("x-", x0 - p[0]), ("x+", p[0] - x1), ("y-", y0 - p[1]), ("y+", p[1] - y1)],
               key=lambda kv: kv[1])


def classify(px):
    b, g, r = float(px[0]), float(px[1]), float(px[2])
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r > 90 and g < 115 and b < 115 and r - max(g, b) > 20:
        return "stub"          # 暗红引脚桩
    if g > r and g > b and g > 60:
        return "wire"          # 绿导线
    if max(r, g, b) < 120:
        return "dark"          # 其它暗墨（框线/文字）
    return "other"


def profile(img, x, y, side, reach):
    """从 (x,y) 沿法向朝**框外**走，返回各类首次出现的距离与暗红游程。"""
    h, w = img.shape[:2]
    dx, dy = {"x-": (-1, 0), "x+": (1, 0), "y-": (0, -1), "y+": (0, 1)}[side]
    seq = []
    for i in range(1, reach + 1):
        xx, yy = int(round(x + dx * i)), int(round(y + dy * i))
        if not (0 <= xx < w and 0 <= yy < h):
            break
        seq.append(classify(img[yy, xx]))
    first = {}
    for i, c in enumerate(seq, start=1):
        first.setdefault(c, i)
    stub = 0
    for i, c in enumerate(seq, start=1):
        if c == "stub":
            stub = i
        elif stub and c != "stub":
            break
    return first.get("stub", 0), stub, first.get("wire", 0), first.get("white", 0)


cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    if c.split == "10GT":
        cases[str(c.case_id)] = c

groups = defaultdict(list)
for cid in a.cases.split(","):
    c = cases.get(cid)
    if not c:
        continue
    gt = normalise_doc(json.load(open(str(c.json_path), encoding="utf-8")))
    stage1 = json.load(open(str(ROOT / "work" / ("_fast_out_%s" % a.run) / "runs" / "stage1" /
                                 ("%s.stage1.json" % cid)), encoding="utf-8"))
    texts = [t["bbox_img"] for t in (stage1.get("texts") or []) if t.get("bbox_img")]
    import cv2
    img = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
    H = img.shape[0]

    for key, comp in (gt.get("components") or {}).items():
        if canon_type((comp or {}).get("type")) not in IC_TYPES:
            continue
        bb = box_of(comp)
        if not bb:
            continue
        # GT 文档 y 向上 → 图像 y 向下：**框和脚点都要翻**（只翻脚点会把 y 边算成垃圾）
        x0, y0, x1, y1 = bb[0], H - bb[3], bb[2], H - bb[1]
        bb_img = (x0, y0, x1, y1)
        for pk, info in ((gt.get("pins") or {}).get(key) or {}).items():
            pd = _pt((info or {}).get("point"))
            if not pd:
                continue
            py = H - pd[1]
            px = pd[0]
            side, off = edge_dist((px, py), bb_img)
            # A) 号字框：该脚外侧、y 重叠 ±8 的最近文本框
            dt_inner = dt_center = th = None
            best = None
            for tx0, ty0, tx1, ty1 in texts:
                if (ty1 - ty0) > 22 or (tx1 - tx0) > 48:   # 号字尺寸（排除器件名/大标签）
                    continue
                tc = (ty0 + ty1) / 2.0
                if abs(tc - py) > 8:
                    continue
                if side == "x-" and tx1 <= x0:
                    d = x0 - tx1
                elif side == "x+" and tx0 >= x1:
                    d = tx0 - x1
                elif side == "y-" and ty1 <= y0:
                    d = y0 - ty1
                elif side == "y+" and ty0 >= y1:
                    d = ty0 - y1
                else:
                    continue
                if best is None or d < best[0]:
                    best = (d, (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0, ty1 - ty0)
            if best and best[0] > 45:                # 太远 ⇒ 不是这个脚的号字
                best = None
            if best:
                dt_inner = best[0]
                ctr = best[1] if side.startswith("x") else best[2]
                edge = x0 if side == "x-" else (x1 if side == "x+" else (y0 if side == "y-" else y1))
                sgn = -1.0 if side.endswith("-") else 1.0
                dt_center = (ctr - edge) * sgn
                th = best[3]
            # C) 图像剖面：**从框边起**沿法向外推（而非从 GT 点起，否则量的是“桩还剩多长”）
            ex = x0 if side == "x-" else (x1 if side == "x+" else px)
            ey = y0 if side == "y-" else (y1 if side == "y+" else py)
            stub_first, stub_run, wire_first, white_first = profile(img, ex, ey, side, a.reach)
            groups[(cid, key, side)].append({
                "off": off, "dt_inner": dt_inner, "dt_center": dt_center,
                "text_h": th, "stub_first": stub_first, "stub_run": stub_run,
                "wire_first": wire_first, "white_first": white_first,
            })

hdr = "%-4s %-5s %-4s %6s %6s %8s %8s %7s %7s %7s %7s" % (
    "case", "comp", "side", "GT外伸", "组n", "号字内边", "号字中心", "字高", "桩起", "绿起", "白起")
print("== 逐（元件,边）组：GT 外伸量 vs 三个候选特征（中位）==")
print(hdr)
print("-" * len(hdr))
feat = defaultdict(list)
for (cid, key, side), items in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1], kv[0][2])):
    med = lambda k: (statistics.median([it[k] for it in items if it[k] is not None])
                     if any(it[k] is not None for it in items) else None)
    off = statistics.median([it["off"] for it in items])
    vals = {k: med(k) for k in ("dt_inner", "dt_center", "text_h", "stub_first", "stub_run",
                                "wire_first", "white_first")}
    row = [off] + [vals[k] for k in ("dt_inner", "dt_center", "text_h", "stub_first",
                                     "stub_run", "wire_first", "white_first")]
    feat["off"].append(off)
    for k, v in zip(("dt_inner", "dt_center", "text_h", "stub_first", "stub_run",
                     "wire_first", "white_first"), row[1:]):
        feat[k].append(v)
    f = lambda v: ("%8.1f" % v) if isinstance(v, float) else "%8s" % "-"
    print("%-4s %-5s %-4s %6.1f %6d %s %s %7s %s %s %s" % (
        cid, key, side, off, len(items), f(vals["dt_inner"]), f(vals["dt_center"]),
        ("%7.1f" % vals["text_h"]) if vals["text_h"] is not None else "      -",
        f(vals["stub_first"]), f(vals["wire_first"]), f(vals["white_first"])))

off = np.array(feat["off"], dtype=float)
print("\n跨组相关（n=%d 组）：" % len(off))
for k, label in (("dt_inner", "号字内边距"), ("dt_center", "号字中心距"), ("text_h", "字高"),
                 ("stub_first", "桩首个暗红"), ("stub_run", "暗红游程"), ("wire_first", "绿起点"),
                 ("white_first", "白起点")):
    v = feat[k]
    ok = np.array([x is not None for x in v])
    x = np.array([x if x is not None else 0.0 for x in v], dtype=float)
    if ok.sum() < 5 or np.std(x[ok]) == 0:
        print("  %-10s 样本不足（n=%d）" % (label, int(ok.sum())))
        continue
    r = float(np.corrcoef(off[ok], x[ok])[0, 1])
    print("  %-10s Pearson %+.3f ｜ |相关|>0.5 的组 %d/%d ｜ GT外伸 中位 %.1f 特征 中位 %.1f" % (
        label, r, int((np.abs(x[ok] - np.median(x[ok])) > 0).sum()), int(ok.sum()),
        float(np.median(off[ok])), float(np.median(x[ok]))))

# 能否把「偏离案例中位 >5px」的组分开？
dev = {}
for (cid, key, side) in groups:
    pass
case_med = defaultdict(list)
for (cid, key, side), items in groups.items():
    case_med[cid].append(statistics.median([it["off"] for it in items]))
case_med = {k: statistics.median(v) for k, v in case_med.items()}
print("\n偏离案例中位 >5px 的组（就是引脚通道亏分的那几组）：")
for (cid, key, side), items in sorted(groups.items()):
    o = statistics.median([it["off"] for it in items])
    if o - case_med[cid] > 5:
        med = lambda k: (statistics.median([it[k] for it in items if it[k] is not None])
                         if any(it[k] is not None for it in items) else None)
        print("  case %-2s %-5s %-3s GT外伸 %5.1f（案例中位 %4.1f）｜ 号字内边 %s 中心 %s 桩游程 %s 绿起 %s" % (
            cid, key, side, o, case_med[cid],
            "%.1f" % med("dt_inner") if med("dt_inner") is not None else "-",
            "%.1f" % med("dt_center") if med("dt_center") is not None else "-",
            "%.1f" % med("stub_run") if med("stub_run") is not None else "-",
            "%.1f" % med("wire_first") if med("wire_first") is not None else "-"))
