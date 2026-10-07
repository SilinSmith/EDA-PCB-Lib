# -*- coding: utf-8 -*-
"""检出框系统性偏小，怎么修？—— 定长外扩 vs 按高度比例外扩 vs 生长到墨迹边界。

真值来自 CVJsonStd 的 `texts[].location` + `content`。每个真值框取 IoU 最好的那个检出框，
对它做各臂变换，再交给 CRNN 读，比 exact。

    cd Y:\workspace\EDA\eda-pcb
    python Y:\workspace\EDA\work\_ocr_pad.py [--limit 10] [--engine]
        --engine 只跑「从墨迹生长」那一臂并打出样例（不带 = 跑全部臂）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts.textbox import INK, detect_boxes  # noqa: E402
from stage2.text import TextEngine, imread_unicode  # noqa: E402

_YP = ROOT / "eda-pcb" / "yolo_part"
_DS = [p for p in _YP.iterdir() if p.is_dir() and p.name.startswith("Dataset300")]
D300 = _DS[0] if _DS else _YP / "Dataset300"

ROLE_OF = {"designator": "designator", "value": "value", "pinName": "pin_name",
           "pinNumber": "pin_number", "netlabel": "net_label", "netLabel": "net_label"}


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
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def grow(gray, box, reach=8.0, ink=INK):
    """把框生长到「与框内墨迹连通」的墨迹边界（窗口 = 框外扩 reach）。"""
    W, H = gray.shape[1], gray.shape[0]
    x1 = int(max(0, np.floor(box[0] - reach)))
    y1 = int(max(0, np.floor(box[1] - reach)))
    x2 = int(min(W, np.ceil(box[2] + reach)))
    y2 = int(min(H, np.ceil(box[3] + reach)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return box
    win = (gray[y1:y2, x1:x2] < ink).astype(np.uint8)
    seed = np.zeros_like(win)
    sx1 = int(max(0, np.floor(box[0]) - x1))
    sy1 = int(max(0, np.floor(box[1]) - y1))
    sx2 = int(min(win.shape[1], np.ceil(box[2]) - x1))
    sy2 = int(min(win.shape[0], np.ceil(box[3]) - y1))
    seed[sy1:sy2, sx1:sx2] = 1
    seed &= win
    if not seed.any():
        return box
    n, lab = cv2.connectedComponents(win, connectivity=8)
    ids = np.unique(lab[seed > 0])
    ids = ids[ids > 0]
    if ids.size == 0:
        return box
    m = np.isin(lab, ids)
    ys, xs = np.nonzero(m)
    return (float(x1 + xs.min()), float(y1 + ys.min()),
            float(x1 + xs.max() + 1), float(y1 + ys.max() + 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--engine", action="store_true", help="只跑生长臂并打样例")
    args = ap.parse_args(argv)

    eng = TextEngine()
    eng.describe()

    FIX = [0, 1, 2, 3, 4, 6, 8]
    PROP = [0.10, 0.20, 0.30, 0.45, 0.60]
    arms = ([("fix%d" % k, lambda b, h, k=k: (b[0] - k, b[1] - k, b[2] + k, b[3] + k)) for k in FIX]
            + [("prop%.2f" % k, lambda b, h, k=k: (b[0] - k * h, b[1] - k * h, b[2] + k * h, b[3] + k * h))
               for k in PROP])
    arms.append(("grow", None))
    arms += [("max%d_30" % p,
              lambda b, h, p=p: (b[0] - max(p, 0.30 * h), b[1] - max(p, 0.30 * h),
                                  b[2] + max(p, 0.30 * h), b[3] + max(p, 0.30 * h)))
             for p in (2, 3, 4)]
    if args.engine:
        arms = [("grow", None)]

    hit = {n: 0 for n, _ in arms}
    tot = 0
    samples = []

    for png, js in cases(args.limit):
        d = json.loads(Path(js).read_text(encoding="utf-8"))
        gts = []
        for t in (d.get("texts") or {}).values():
            loc = t.get("location") or {}
            c = (t.get("content") or "").strip()
            role = ROLE_OF.get(str(t.get("details") or ""))
            if not role or not c or len(c) > 24:
                continue
            try:
                if None in (loc.get("x1"), loc.get("y1"), loc.get("x2"), loc.get("y2")):
                    continue
            except AttributeError:
                continue
            gts.append((tuple(float(loc[k]) for k in ("x1", "y1", "x2", "y2")), c, role))
        if not gts:
            continue
        bgr = imread_unicode(png)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = [tuple(float(v) for v in b) for b in detect_boxes(bgr)]

        # 每个真值框 → 最好检出框
        todo = []
        for g, c, role in gts:
            if not boxes:
                continue
            sc = [(iou(g, b), i) for i, b in enumerate(boxes)]
            sc.sort(reverse=True)
            if sc[0][0] < 0.05:
                continue
            b = boxes[sc[0][1]]
            h = max(1.0, b[3] - b[1])
            if args.engine:
                cand = [("grow", grow(gray, b))]
            else:
                cand = [(n, f(b, h)) for n, f in arms if f is not None]
                cand.append(("grow", grow(gray, b)))
            todo.append((g, c, role, cand))
            tot += 1

        # 逐臂批量读
        per_arm = {n: [] for n, _ in arms}
        for g, c, role, cand in todo:
            for n, bb in cand:
                per_arm.setdefault(n, []).append((bb, role, c))
        for n, items in per_arm.items():
            if not items:
                continue
            byrole = {}
            for i, (bb, role, c) in enumerate(items):
                byrole.setdefault(role, []).append(i)
            got = [None] * len(items)
            for role, idx in byrole.items():
                rr = eng.recognizer()
                outs = rr.read_many(bgr, [(items[i][0], role) for i in idx])
                for j, i in enumerate(idx):
                    got[i] = outs[j][0]
            for i, (bb, role, c) in enumerate(items):
                if got[i] is not None and got[i].strip() == c:
                    hit[n] += 1
                    if args.engine and len(samples) < 30 and n == "grow":
                        samples.append((c, got[i], bb))
                elif args.engine and n == "grow" and len(samples) < 30:
                    samples.append((c, got[i], bb))

    print("\n臂        exact     命中/总")
    for n, _ in arms:
        print("  %-8s %.4f    %d/%d" % (n, hit[n] / max(1, tot), hit[n], tot))
    if args.engine:
        print("\n生长臂样例（真值 -> 读出）:")
        for c, o, bb in samples[:30]:
            print("   %-14r -> %-14r  px=%s" % (c, o, tuple(round(v) for v in bb)))
    return 0


def _pil(bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    raise SystemExit(main())
