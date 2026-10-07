# -*- coding: utf-8 -*-
"""检出框该怎么修，才能让 CRNN 读对？

`_ocr_boxcmp.py` 的结论：CRNN 在 **GT 紧框**上 exact 0.4127，在 **`cvparts.textbox` 的检出框**
上只有 0.0266；检出框相对 GT 框中位**窄 5px、矮 6px**。所以瓶颈是框，不是识别。

本脚本在同一批 CVJsonStd 数据上扫几种「修框」臂，看哪种能把 exact 拉回来。

臂：
  B    检出框原样
  Pk   检出框外扩 k 像素（k=1..4）
  Uk   外扩 k 后，把**中心落在里面**的所有检出框并起来（治「一个字符串被切成几个框」）

    cd Y:\workspace\EDA\eda-pcb
    python Y:\workspace\EDA\work\_ocr_boxtune.py [--limit 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def pad(b, k, W, H):
    return (max(0.0, b[0] - k), max(0.0, b[1] - k), min(W, b[2] + k), min(H, b[3] + k))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)
    eng = TextEngine()
    print("[boxtune] %s" % eng.describe())

    arms = ["B", "P1", "P2", "P3", "P4", "U2", "U3", "U4"]
    ok = {a: 0 for a in arms}
    tot = {a: 0 for a in arms}
    shown = {a: 0 for a in arms}

    for png, js in cases(args.limit):
        d = json.loads(Path(js).read_text(encoding="utf-8"))
        gts = []
        for t in (d.get("texts") or {}).values():
            loc = t.get("location") or {}
            c = (t.get("content") or "").strip()
            if None in (loc.get("x1"), loc.get("y1"), loc.get("x2"), loc.get("y2")) or not c:
                continue
            gts.append(((float(loc["x1"]), float(loc["y1"]),
                         float(loc["x2"]), float(loc["y2"])), c))
        if not gts:
            continue
        bgr = imread_unicode(png)
        H, W = bgr.shape[:2]
        boxes = [tuple(float(v) for v in b) for b in detect_boxes(bgr)]

        jobs = []       # (gt_text, arm, box)
        for gb, txt in gts:
            cx, cy = (gb[0] + gb[2]) / 2.0, (gb[1] + gb[3]) / 2.0
            m = [b for b in boxes if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
            if not m:
                continue
            b = min(m, key=lambda b: abs((b[2] - b[0]) - (gb[2] - gb[0])))
            jobs.append((txt, "B", b))
            for k in (1, 2, 3, 4):
                jobs.append((txt, "P%d" % k, pad(b, k, W, H)))
            for k in (2, 3, 4):
                pb = pad(b, k, W, H)
                ins = [q for q in boxes
                       if pb[0] <= (q[0] + q[2]) / 2.0 <= pb[2]
                       and pb[1] <= (q[1] + q[3]) / 2.0 <= pb[3]]
                if not ins:
                    ins = [b]
                u = (min(q[0] for q in ins), min(q[1] for q in ins),
                     max(q[2] for q in ins), max(q[3] for q in ins))
                jobs.append((txt, "U%d" % k, u))

        # 一次性批量读（同一个 role）
        for arm in arms:
            sub = [(t, b) for (t, a, b) in jobs if a == arm]
            if not sub:
                continue
            reads = eng.read(bgr, [b for _, b in sub], ["designator"])
            for (t, b), r in zip(sub, reads):
                got, conf = r["reads"]["designator"]
                tot[arm] += 1
                if got == t:
                    ok[arm] += 1
                elif shown[arm] < 3:
                    shown[arm] += 1
                    print("   %-3s %-16r -> %-16r %.2f %s"
                          % (arm, t, got, conf, tuple(round(x) for x in b)))

    print("\n臂  exact    命中/总")
    for a in arms:
        if tot[a]:
            print("  %-3s %.4f  %d/%d" % (a, ok[a] / tot[a], ok[a], tot[a]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
