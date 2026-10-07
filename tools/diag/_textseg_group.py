# -*- coding: utf-8 -*-
"""文字行分组参数扫描。**关键指标是「原样裁（不 pad）的 exact」** —— 真值紧框原样读是
0.84，而任何检出框原样读只有 0.09；外扩能把检出框救到 0.40 但会把真值框毁到 0.20。
所以正确的靶子是「切出**紧**的框」，不是「切出框再外扩」。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_textseg_group.py [--limit 10]
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
from stage2.text import TextEngine, imread_unicode  # noqa: E402
from _textseg_eval import ROLE_OF, cases, iou_matrix  # noqa: E402

BASE = dict(ink_mode="fixed", min_dim=2, min_area=3, line_len=15,
            resid_min_h=4, resid_min_dim=1)
GROUPS = {
    "old": None,
    "g45": dict(gap_frac=0.45, v_overlap=0.40, h_ratio=2.2),
    "k170": dict(ink=170),
    "k190": dict(ink=190),
    "k210": dict(ink=210),
    "k225": dict(ink=225),
    "k235": dict(ink=235),
}


def gt_of(js):
    d = json.loads(Path(js).read_text(encoding="utf-8"))
    out = []
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
        out.append((b, c, role))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    names = [n for n in (args.only.split(",") if args.only else GROUPS) if n]
    rr = TextEngine().recognizer()

    ndet = {n: 0 for n in names}
    iou = {n: [] for n in names}
    gt5 = {n: 0 for n in names}
    raw_ok = {n: 0 for n in names}
    raw_cnt = {n: 0 for n in names}
    truth = {}
    gt_ok = gt_cnt = 0
    n_gt = 0

    for ci, (png, js) in enumerate(cases(args.limit)):
        gts = gt_of(js)
        if not gts:
            continue
        bgr = imread_unicode(png)
        n_gt += len(gts)

        # 天花板：真值紧框原样
        by = {}
        for i, (_b, _c, role) in enumerate(gts):
            by.setdefault(role, []).append(i)
        for role, idx in by.items():
            outs = rr.read_many(bgr, [(gts[i][0], role) for i in idx])
            for j, i in enumerate(idx):
                gt_cnt += 1
                if (outs[j][0] or "").strip() == gts[i][1]:
                    gt_ok += 1

        jobs = {}
        for n in names:
            if GROUPS[n] is None:
                det = [tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
            else:
                det = [tuple(float(v) for v in b)
                       for b in textseg.detect_boxes(bgr, **{**BASE, **GROUPS[n]})]
            ndet[n] += len(det)
            M = iou_matrix([g[0] for g in gts], det) if det else np.zeros((len(gts), 0))
            for i, (b, c, role) in enumerate(gts):
                row = M[i] if M.size else np.zeros(0)
                j = int(np.argmax(row)) if row.size else -1
                bi = float(row[j]) if j >= 0 else 0.0
                iou[n].append(bi)
                if bi >= 0.5:
                    gt5[n] += 1
                if bi >= 0.05:
                    jobs.setdefault((n, role), []).append((i, det[j], c))
        for (n, role), lst in jobs.items():
            outs = rr.read_many(bgr, [(b, role) for _i, b, _c in lst])
            for (i, _b, c), (txt, _cf) in zip(lst, outs):
                raw_cnt[n] += 1
                if (txt or "").strip() == c:
                    raw_ok[n] += 1

    print(f"\n真值框 {n_gt}（{args.limit} 张）")
    print(f"天花板 GT_raw  原样 exact {gt_ok/max(1,gt_cnt):.4f} ({gt_ok}/{gt_cnt})")
    print(f"\n{'臂':8s} {'检出':>6s} {'命中':>7s} {'IoU>=.5':>8s} {'IoU p50':>8s}"
          f" {'原样exact':>10s} {'端到端':>8s}")
    for n in names:
        g = np.asarray(iou[n])
        print(f"  {n:6s} {ndet[n]:6d} {int((g>=0.05).sum()):7d} "
              f"{gt5[n]:4d}/{n_gt:<3d} {np.median(g):8.3f} "
              f"{raw_ok[n]/max(1,raw_cnt[n]):10.4f} {raw_ok[n]/max(1,n_gt):8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
