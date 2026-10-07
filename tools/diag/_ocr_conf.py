# -*- coding: utf-8 -*-
"""量「CRNN 置信度能不能把**文字框**与**符号误检框**分开」。

背景：`cvparts.textbox` 会把电阻锯齿、三极管、电池、地符号当成文字框检出，而这些框
常常正好落在元件中心 —— 于是「离元件最近的框」被符号抢走，位号赋值率只有 6%。

判据用 CVJsonStd 的 `texts[].location`（已目视确认与像素严丝合缝）：
  · 正样本 = 中心落在某个 location 内的检出框
  · 负样本 = 离所有 location 都 > 8px 的检出框

    cd Y:\workspace\EDA\eda-pcb
    python Y:\workspace\EDA\work\_ocr_conf.py [--limit 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts.textbox import detect_boxes  # noqa: E402
from stage2.text import TextEngine, imread_unicode  # noqa: E402

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


def gt_boxes(js_path):
    import json
    d = json.loads(Path(js_path).read_text(encoding="utf-8"))
    out = []
    for t in (d.get("texts") or {}).values():
        loc = t.get("location") or {}
        b = (loc.get("x1"), loc.get("y1"), loc.get("x2"), loc.get("y2"))
        if None in b:
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue
        out.append((tuple(float(v) for v in b), content))
    return out


def _inside(c, b):
    return b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args(argv)

    eng = TextEngine()
    print("[conf] %s" % eng.describe())
    cases_ = cases(args.limit)
    print("case %d 个" % len(cases_))

    pos, neg = [], []
    exact = {"pos": 0, "neg": 0}
    for png, js in cases_:
        gts = gt_boxes(js)
        if not gts:
            continue
        bgr = imread_unicode(png)
        boxes = detect_boxes(bgr)
        reads = eng.read(bgr, boxes)
        for r in reads:
            b = r["bbox"]
            c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            hit = None
            for gb, txt in gts:
                if _inside(c, gb):
                    hit = txt
                    break
            conf = max([v[1] for v in r["reads"].values()] or [0.0])
            txt = ""
            if r["reads"]:
                k = max(r["reads"], key=lambda kk: r["reads"][kk][1])
                txt = r["reads"][k][0]
            if hit is not None:
                pos.append((conf, txt, hit))
                if txt == hit:
                    exact["pos"] += 1
            else:
                far = all(min(abs(c[0] - gb[0]), abs(c[0] - gb[2])) > 8 or
                          min(abs(c[1] - gb[1]), abs(c[1] - gb[3])) > 8 for gb, _ in gts)
                if far:
                    neg.append((conf, txt, ""))

    def hist(rows, name):
        if not rows:
            print("%s: 空" % name)
            return
        cs = np.array([r[0] for r in rows])
        print("%s n=%d  conf  p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f  均值 %.3f"
              % (name, len(rows), *np.percentile(cs, [10, 25, 50, 75, 90]), cs.mean()))
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            print("   conf>=%.1f 留下 %5.1f%%" % (t, 100.0 * (cs >= t).mean()))

    print()
    hist(pos, "正样本（真文字）")
    hist(neg, "负样本（疑似符号）")

    if pos:
        print("\n真文字里读出原串的比例 %.3f (%d/%d)" % (exact["pos"] / len(pos), exact["pos"], len(pos)))
        # 交叉：用 0.5 做门槛时的保留率与正确率
        import collections
        for t in (0.3, 0.4, 0.5, 0.6):
            keep_pos = [p for p in pos if p[0] >= t]
            keep_neg = [n for n in neg if n[0] >= t]
            ok = sum(1 for p in keep_pos if p[1] == p[2])
            print("  thr=%.1f: 真文字留下 %d/%d（其中读对 %d，%.3f），符号误框留下 %d"
                  % (t, len(keep_pos), len(pos), ok,
                     ok / max(1, len(keep_pos)), len(keep_neg)))
    if neg:
        top = collections.Counter(n[1] for n in neg)
        print("\n负样本里最常见的读法: %s" % (top.most_common(12),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
