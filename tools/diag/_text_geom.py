# -*- coding: utf-8 -*-
"""量清「文字框 ↔ 元件框」的几何关系。

两个问题：
  1. pinNumber / pinName 是画在元件框内还是框外？（决定第二阶段怎么分配角色）
  2. 位号（designator）相对元件框在哪个方位？距离多少？

数据来源两路：
  A. Dataset300 的 CVJsonStd：texts[].location 与图像严丝合缝、details 自带角色 ⇒ 几何真值
  B. 官方 220 的 GT：components 的 key 是位号、type 是类别 ⇒ 量 type→位号前缀先验
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CV_ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset300我不知道干嘛用的先放置")
GT_ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
ROLES = ("designator", "value", "pinName", "pinNumber", "netlabel", "netLabel")


def dist_to_rect(px, py, r):
    dx = max(r[0] - px, 0.0, px - r[2])
    dy = max(r[1] - py, 0.0, py - r[3])
    return (dx * dx + dy * dy) ** 0.5


def inside(px, py, r):
    return r[0] <= px <= r[2] and r[1] <= py <= r[3]


def _items(x):
    if isinstance(x, dict):
        return [v for v in x.values() if isinstance(v, dict)]
    if isinstance(x, list):
        return [v for v in x if isinstance(v, dict)]
    return []


def boxes_of(comp):
    for k in ("realBBox", "location"):
        b = comp.get(k)
        if isinstance(b, dict) and all(x in b for x in ("x1", "y1", "x2", "y2")):
            return [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]
    return None


def main():
    files = sorted(CV_ROOT.glob("*/*_CVJsonStd.json"))
    print("CVJsonStd 文件 %d 个" % len(files))
    agg = defaultdict(lambda: {"n": 0, "in_any": 0, "in_nearest": 0,
                               "d_nearest": [], "side": Counter()})
    first_comp = None
    canvas_mismatch = 0

    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        comps = []
        for c in _items(d.get("components")):
            b = boxes_of(c)
            if b:
                comps.append((b, c.get("category")))
        if first_comp is None and comps:
            first_comp = {"file": fp.name, "n_comps": len(comps),
                          "sample": json.loads(json.dumps(comps[:2]))}

        for t in _items(d.get("texts")):
            det = (t.get("details") or "").strip() or (t.get("category") or "")
            if det not in ROLES:
                continue
            loc = t.get("location") or {}
            if not all(k in loc for k in ("x1", "y1", "x2", "y2")):
                continue
            tb = [float(loc["x1"]), float(loc["y1"]), float(loc["x2"]), float(loc["y2"])]
            cx, cy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
            a = agg[det]
            a["n"] += 1
            if any(inside(cx, cy, b) for b, _ in comps):
                a["in_any"] += 1
            if not comps:
                continue
            j = min(range(len(comps)), key=lambda i: dist_to_rect(cx, cy, comps[i][0]))
            b = comps[j][0]
            a["in_nearest"] += int(inside(cx, cy, b))
            a["d_nearest"].append(dist_to_rect(cx, cy, b))
            w, h = b[2] - b[0], b[3] - b[1]
            if inside(cx, cy, b):
                a["side"]["in"] += 1
            else:
                if cx < b[0]:
                    a["side"]["left"] += 1
                elif cx > b[2]:
                    a["side"]["right"] += 1
                elif cy < b[1]:
                    a["side"]["above"] += 1
                elif cy > b[3]:
                    a["side"]["below"] += 1
                else:
                    a["side"]["other"] += 1

    print("\n=== A. CVJsonStd 文字 ↔ 元件几何 ===")
    print("%-11s %6s %8s %10s %9s %9s %s"
          % ("details", "n", "%在任一框内", "%在最近框内", "最近框d p50", "d p90", "方位"))
    for det in ROLES:
        a = agg.get(det)
        if not a or not a["n"]:
            continue
        ds = sorted(a["d_nearest"])
        p50 = ds[len(ds) // 2] if ds else -1
        p90 = ds[int(len(ds) * 0.9)] if ds else -1
        sides = " ".join("%s:%d" % kv for kv in a["side"].most_common(4))
        print("%-11s %6d %9.1f%% %10.1f%% %9.1f %9.1f %s"
              % (det, a["n"], 100.0 * a["in_any"] / a["n"],
                 100.0 * a["in_nearest"] / a["n"], p50, p90, sides))
    if first_comp:
        print("\n样例（%s，%d 个元件）: %s"
              % (first_comp["file"], first_comp["n_comps"],
                 json.dumps(first_comp["sample"], ensure_ascii=False)[:300]))

    # ------------------------------------------------------------------
    print("\n=== B. 官方 GT：type -> 位号前缀先验 ===")
    prior = defaultdict(Counter)
    n_comp = 0
    for split in ("10GT", "10_GTcase", "200_train_cases"):
        for jf in (GT_ROOT / split).glob("*/*.json"):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key, c in (d.get("components") or {}).items():
                n_comp += 1
                if key.startswith("∅") or key.startswith("∅"):
                    continue
                letters = "".join(ch for ch in key if ch.isalpha())
                prior[c.get("type")][letters or "?"] += 1
    print("元件总数 %d" % n_comp)
    print("%-16s %7s  %s" % ("type", "n", "位号前缀 top3"))
    for t, c in sorted(prior.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        top = " ".join("%s:%d" % kv for kv in c.most_common(3))
        print("%-16s %7d  %s" % (t, tot, top))


if __name__ == "__main__":
    main()
