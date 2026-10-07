# -*- coding: utf-8 -*-
"""旧 `cvparts.textbox` vs 新 `cvparts.textseg`：检测质量（IoU）+ OCR exact。

真值来自 CVJsonStd 的 `texts[].location` + `content`（框与图上文字严丝合缝）。四个臂：

    old / new              用各自检测器的框直接裁
    old_pad / new_pad      再按 max(2, 0.30*h) 四边外扩

每个真值框取 IoU 最好的那个检出框（两个检测器各取各的），再交给 CRNN 读，比 exact。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_textseg_eval.py [--limit 10]
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

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import TextEngine, imread_unicode, pad_box  # noqa: E402

_YP = ROOT / "eda-pcb" / "yolo_part"
_DS = [p for p in _YP.iterdir() if p.is_dir() and p.name.startswith("Dataset300")]
D300 = _DS[0] if _DS else _YP / "Dataset300"

ROLE_OF = {"designator": "designator", "value": "value", "pinName": "pin_name",
           "pinNumber": "pin_number", "netlabel": "net_label", "netLabel": "net_label"}
ARMS = ("old", "old_pad", "new", "new_pad")


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


def iou_matrix(gt, det):
    g = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    d = np.asarray(det, dtype=np.float64).reshape(-1, 4)
    if not len(g) or not len(d):
        return np.zeros((len(g), len(d)))
    x1 = np.maximum(g[:, None, 0], d[None, :, 0])
    y1 = np.maximum(g[:, None, 1], d[None, :, 1])
    x2 = np.minimum(g[:, None, 2], d[None, :, 2])
    y2 = np.minimum(g[:, None, 3], d[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    ad = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    ua = ag[:, None] + ad[None, :] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def stat_line(tag, gt_best, det_best, n_gt, n_det):
    gb = np.concatenate(gt_best) if gt_best else np.zeros(0)
    db = np.concatenate(det_best) if det_best else np.zeros(0)
    print(f"  {tag:4s} 检出框 {n_det:5d} / 真值框 {n_gt:5d}")
    if gb.size:
        print(f"       真值框→最好检出框 IoU: p50 {np.median(gb):.3f} 均值 {gb.mean():.3f}"
              f"  >=0.5 {int((gb>=0.5).sum())}/{gb.size}={float((gb>=0.5).mean()):.3f}"
              f"  <0.1 {float((gb<0.1).mean()):.3f}")
    if db.size:
        print(f"       检出框→最好真值框 IoU: p50 {np.median(db):.3f} 均值 {db.mean():.3f}"
              f"  >=0.5 {int((db>=0.5).sum())}/{db.size}={float((db>=0.5).mean()):.3f}"
              f"  <0.1 {float((db<0.1).mean()):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)

    eng = TextEngine()

    hit = {a: 0 for a in ARMS}
    cnt = {a: 0 for a in ARMS}
    gt_best = {"old": [], "new": []}
    det_best = {"old": [], "new": []}
    n_det = {"old": 0, "new": 0}
    n_gt = 0

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
                b = tuple(float(loc[k]) for k in ("x1", "y1", "x2", "y2"))
            except (TypeError, KeyError):
                continue
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            gts.append((b, c, role))
        if not gts:
            continue
        bgr = imread_unicode(png)

        det = {"old": [tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)],
               "new": [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr)]}

        for k in ("old", "new"):
            n_det[k] += len(det[k])
            M = iou_matrix([g[0] for g in gts], det[k])
            if M.size:
                det_best[k].append(M.max(axis=0))
                gt_best[k].append(M.max(axis=1))
        n_gt += len(gts)

        items = []
        for b, c, role in gts:
            for k in ("old", "new"):
                if not det[k]:
                    continue
                M = iou_matrix([b], det[k])[0]
                j = int(np.argmax(M))
                if M[j] < 0.05:
                    continue
                items.append((k, det[k][j], role, c))
                items.append((k + "_pad", pad_box(det[k][j]), role, c))

        got = [None] * len(items)
        byrole = {}
        for i, (_a, _b, role, _c) in enumerate(items):
            byrole.setdefault(role, []).append(i)
        for role, idx in byrole.items():
            rr = eng.recognizer()
            outs = rr.read_many(bgr, [(items[i][1], role) for i in idx])
            for j, i in enumerate(idx):
                got[i] = outs[j][0]
        for i, (arm, _b, _role, c) in enumerate(items):
            cnt[arm] += 1
            if got[i] is not None and got[i].strip() == c:
                hit[arm] += 1

    print("\n=== 检测质量 ===")
    for k in ("old", "new"):
        stat_line(k, gt_best[k], det_best[k], n_gt, n_det[k])

    print("\n=== OCR exact（分母 = 该臂的样本数）===")
    for a in ARMS:
        print(f"  {a:8s} {hit[a]/max(1,cnt[a]):.4f}   ({hit[a]}/{cnt[a]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
