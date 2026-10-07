# -*- coding: utf-8 -*-
"""`cvparts.textbox` 的框到底有多准？—— 决定「修框」还是「重写切分」。

对每个 CVJsonStd 的真值文字框，量：
  · IoU 最好的检出框（看是**偏了**还是**没检出**）
  · 与它有交集的所有检出框并起来后的 IoU（看是不是**被切碎了**）
再数检出框里有多少落不到任何真值框上（符号误检）。

    cd Y:\workspace\EDA\eda-pcb
    python Y:\workspace\EDA\work\_boxquality.py [--limit 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts.textbox import detect_boxes  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402

_YP = ROOT / "eda-pcb" / "yolo_part"
_DS = [p for p in _YP.iterdir() if p.is_dir() and p.name.startswith("Dataset300")]
D300 = _DS[0] if _DS else _YP / "Dataset300"


def cases(limit=0):
    out = []
    for d in sorted(D300.iterdir()) if D300.is_dir() else []:
        if not d.is_dir():
            continue
        png = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        js = sorted(d.glob("*_CVJsonStd.json"))
        if png and js:
            out.append((png[0], js[0]))
    return out[:limit] if limit else out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)

    best_iou, union_iou, n_over = [], [], []
    widths = []
    n_gt = n_fp = 0
    no_hit = 0
    for png, js in cases(args.limit):
        d = json.loads(Path(js).read_text(encoding="utf-8"))
        gts = []
        for t in (d.get("texts") or {}).values():
            loc = t.get("location") or {}
            c = (t.get("content") or "").strip()
            if None in (loc.get("x1"), loc.get("y1"), loc.get("x2"), loc.get("y2")) or not c:
                continue
            gts.append((float(loc["x1"]), float(loc["y1"]),
                        float(loc["x2"]), float(loc["y2"])))
        if not gts:
            continue
        bgr = imread_unicode(png)
        boxes = [tuple(float(v) for v in b) for b in detect_boxes(bgr)]
        widths += [b[2] - b[0] for b in boxes]

        hit = set()
        for g in gts:
            n_gt += 1
            ov = [(iou(g, b), i) for i, b in enumerate(boxes) if iou(g, b) > 0.05]
            if not ov:
                no_hit += 1
                best_iou.append(0.0)
                union_iou.append(0.0)
                n_over.append(0)
                continue
            ov.sort(reverse=True)
            best_iou.append(ov[0][0])
            n_over.append(len(ov))
            idx = [i for _, i in ov]
            u = (min(boxes[i][0] for i in idx), min(boxes[i][1] for i in idx),
                 max(boxes[i][2] for i in idx), max(boxes[i][3] for i in idx))
            union_iou.append(iou(g, u))
            hit.update(idx)
        n_fp += len(boxes) - len(hit)

    def rpt(name, v):
        v = np.array(v)
        print("%-22s n=%d  p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f  均值 %.3f  "
              "=0 占 %.1f%%  >=0.5 占 %.1f%%"
              % (name, len(v), *np.percentile(v, [10, 25, 50, 75, 90]), v.mean(),
                 100.0 * (v == 0).mean(), 100.0 * (v >= 0.5).mean()))

    print("真值文字框 %d 个，检出框 %d 个（其中落不到真值框上的 %d 个 = %.1f%%）"
          % (n_gt, n_gt + n_fp - n_gt + n_fp and 0 or 0, n_fp, 100.0 * n_fp / max(1, n_gt + n_fp)))
    print("没有任何检出框对上的真值框 %d 个（%.1f%%）" % (no_hit, 100.0 * no_hit / max(1, n_gt)))
    print()
    rpt("最好单框 IoU", best_iou)
    rpt("并集 IoU", union_iou)
    r = np.array(n_over)
    print("%-22s n=%d  p50 %.1f p90 %.1f  均值 %.2f   >=2 个的占 %.1f%%"
          % ("每个真值框压住几个检出框", len(r), *np.percentile(r, [50, 90]), r.mean(),
             100.0 * (r >= 2).mean()))
    w = np.array(widths)
    print("%-22s n=%d  p10 %.1f p50 %.1f p90 %.1f" % ("检出框宽度", len(w), *np.percentile(w, [10, 50, 90])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
