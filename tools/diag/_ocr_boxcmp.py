# -*- coding: utf-8 -*-
"""CRNN 到底行不行？把「在 GT 的紧框上读」与「在检测框上读」分开量。

`_ocr_conf.py` 测出「检出框里读出原串只有 4.6%」，而训练时的 val_exact 是 0.512。
两者差 11 倍，必须先分清是
  (A) CRNN 本身不行，还是
  (B) 训练用的是 CVJsonStd 的**紧框**，推理喂的是 `cvparts.textbox` 的**检测框**（尺寸/合并不同）。

    cd Y:\workspace\EDA\eda-pcb
    python Y:\workspace\EDA\work\_ocr_boxcmp.py [--limit 10]
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
from stage2.text import TextEngine, imread_unicode  # noqa: E402

_YP = ROOT / "eda-pcb" / "yolo_part"
_DS = [p for p in _YP.iterdir() if p.is_dir() and p.name.startswith("Dataset300")]
D300 = _DS[0] if _DS else _YP / "Dataset300"
ROLES = ("designator", "value", "pin_name", "pin_number", "net_label")


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


def read_box(eng, bgr, box, role="designator"):
    r = eng.read(bgr, [tuple(float(v) for v in box)], [role])
    if not r:
        return "", 0.0
    txt, conf = r[0]["reads"][role]
    return txt, conf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)
    eng = TextEngine()
    print("[boxcmp] %s" % eng.describe())

    a_tot = a_ok = 0
    b_tot = b_ok = 0
    d_w = []
    d_h = []
    rows = []
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
        boxes = [tuple(float(v) for v in b) for b in detect_boxes(bgr)]

        # 臂 A：直接在 GT 紧框上读
        for gb, txt in gts:
            got, conf = read_box(eng, bgr, gb)
            a_tot += 1
            if got == txt:
                a_ok += 1
            if len(rows) < 40 and got != txt:
                rows.append(("A", txt, got, conf, gb))

        # 臂 B：找到中心落在该 GT 框内的检出框，用它去读
        for gb, txt in gts:
            cx, cy = (gb[0] + gb[2]) / 2.0, (gb[1] + gb[3]) / 2.0
            m = [b for b in boxes if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
            if not m:
                continue
            b = min(m, key=lambda b: abs((b[2] - b[0]) - (gb[2] - gb[0])))
            got, conf = read_box(eng, bgr, b)
            b_tot += 1
            if got == txt:
                b_ok += 1
            d_w.append((b[2] - b[0]) - (gb[2] - gb[0]))
            d_h.append((b[3] - b[1]) - (gb[3] - gb[1]))
            if len(rows) < 80 and got != txt:
                rows.append(("B", txt, got, conf, b))

    print("\n臂 A（GT 紧框）  exact %.4f  (%d/%d)" % (a_ok / max(1, a_tot), a_ok, a_tot))
    print("臂 B（检测框）   exact %.4f  (%d/%d)" % (b_ok / max(1, b_tot), b_ok, b_tot))
    if d_w:
        w = np.array(d_w); h = np.array(d_h)
        print("\n检测框 − GT框  宽差 p10 %.1f p50 %.1f p90 %.1f ; 高差 p10 %.1f p50 %.1f p90 %.1f"
              % (*np.percentile(w, [10, 50, 90]), *np.percentile(h, [10, 50, 90])))
    print("\n抽样错例（臂 / 真值 / 读出 / conf / 框）:")
    for arm, t, g, c, b in rows[:30]:
        print("  %s  %-16r -> %-16r %.2f  %s" % (arm, t, g, c, tuple(round(x) for x in b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
