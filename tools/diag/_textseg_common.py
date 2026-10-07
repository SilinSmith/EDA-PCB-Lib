# -*- coding: utf-8 -*-
"""文字框切分的严格对照：所有臂都在**同一批真值框**上比。

臂：
  ``GT_raw`` / ``GT_pad``   真值紧框原样 / 加 pad —— 这套 CRNN 的天花板
  ``old`` / ``old_pad``     旧 `cvparts.textbox`
  ``H`` / ``H_pad``         新 `cvparts.textseg`（当前推荐配置）

对每个臂记录「每个真值框读出来的串」，于是可以算：
  * 端到端 exact = 读对的真值框 / **全部**真值框（把漏检也算成错）
  * 命中集 exact = 读对的 / 该臂命中的真值框
  * **公共子集 exact** = 只在该臂与 old 都命中的真值框上比 —— 唯一公平的口径

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_textseg_sweep.py [--limit 10]
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
from _textseg_eval import ROLE_OF, cases, iou_matrix  # noqa: E402

H_CFG = dict(ink_mode="max", min_dim=2, min_area=3, line_len=15,
             resid_min_h=4, resid_min_dim=1)
DETS = {"old": None, "H": H_CFG}


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


def binarize(bgr, thr=200):
    """把所有非背景（灰 < thr）压成纯黑，其余压成纯白；返回 3 通道 BGR。"""
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bw = np.where(g < thr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--bw", type=int, default=-1,
                    help=">0 时额外加一组 *_bw 臂，用二值化后的图读")
    args = ap.parse_args(argv)
    BW = args.bw if args.bw > 0 else None

    rr = TextEngine().recognizer()
    res = {}          # arm -> {gi: text}
    match = {}        # arm -> set(gi)
    ndet = {n: 0 for n in DETS}
    iou = {n: [] for n in DETS}
    per_case = {}
    truth = {}
    n_gt = 0

    for ci, (png, js) in enumerate(cases(args.limit)):
        gts = gt_of(js)
        if not gts:
            continue
        bgr = imread_unicode(png)
        bgr_bw = binarize(bgr, BW) if BW else None
        gi_all = [(ci, i) for i in range(len(gts))]
        for k, gi in enumerate(gi_all):
            truth[gi] = gts[k][1]
            n_gt += 1

        jobs = {}      # (arm, role) -> [gi]

        def push(arm, gi, box):
            res.setdefault(arm, {})
            jobs.setdefault((arm, gts[gi[1]][2]), []).append((gi, box))
            if BW:
                res.setdefault(arm + "_bw", {})
                jobs.setdefault((arm + "_bw", gts[gi[1]][2]), []).append((gi, box))

        for gi, (b, _c, role) in zip(gi_all, gts):
            push("GT_raw", gi, b)
            push("GT_pad", gi, pad_box(b))

        for n, cfg in DETS.items():
            det = ([tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
                   if cfg is None else
                   [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **cfg)])
            ndet[n] += len(det)
            M = iou_matrix([g[0] for g in gts], det) if det else np.zeros((len(gts), 0))
            match.setdefault(n, set())
            match.setdefault(n + "_pad", set())
            for k, gi in enumerate(gi_all):
                row = M[k] if M.size else np.zeros(0)
                j = int(np.argmax(row)) if row.size else -1
                bi = float(row[j]) if j >= 0 else 0.0
                iou[n].append(bi)
                if bi >= 0.05:
                    match[n].add(gi)
                    match[n + "_pad"].add(gi)
                    push(n, gi, det[j])
                    push(n + "_pad", gi, pad_box(det[j]))
            per_case.setdefault(n, {})[png.parent.name] = (
                len(gts), len(match[n] & set(gi_all)))

        # 按 (arm, role) 批量读
        for (arm, role), lst in jobs.items():
            boxes = [b for _gi, b in lst]
            img = bgr_bw if (BW and arm.endswith("_bw")) else bgr
            outs = rr.read_many(img, [(b, role) for b in boxes])
            for (gi, _b), (txt, _cf) in zip(lst, outs):
                res[arm][gi] = (txt or "").strip()

    arms = sorted(res)
    print(f"\n真值框 {n_gt}（{len(cases(args.limit))} 张图）")
    print("\n=== 检出 ===")
    for n in DETS:
        g = np.asarray(iou[n])
        print(f"  {n:8s} 检出 {ndet[n]:5d}  命中 {len(match[n]):5d}/{n_gt} = "
              f"{len(match[n])/n_gt:.3f}  IoU p50 {np.median(g):.3f}")

    print("\n=== OCR exact ===")
    print(f"  {'臂':10s} {'命中':>6s} {'命中集':>8s} {'端到端':>8s}"
          f" {'公共(∩old)':>12s} {'公共(∩old_pad)':>14s}")
    base = match.get("old", set())
    basep = match.get("old_pad", set())
    for a in arms:
        m = match.get(a if not a.endswith("_bw") else a[:-3], set())
        if not m:
            # GT 臂：命中 = 全部
            m = set(truth)
        ok = sum(1 for gi in m if res[a].get(gi) == truth[gi])
        rec = res[a]
        com = m & base
        comok = sum(1 for gi in com if rec.get(gi) == truth[gi])
        comp = m & basep
        compok = sum(1 for gi in comp if rec.get(gi) == truth[gi])
        print(f"  {a:10s} {len(m):6d} {ok/max(1,len(m)):8.4f} {ok/max(1,n_gt):8.4f}"
              f" {comok:5d}/{len(com):<5d}={comok/max(1,len(com)):.4f}"
              f" {compok:6d}/{len(comp):<5d}={compok/max(1,len(comp)):.4f}")

    print("\n=== 逐 case 命中/真值 ===")
    keys = sorted({k for n in DETS for k in per_case[n]})
    print(f"  {'case':8s} " + "".join(f"{n:>12s}" for n in DETS))
    for k in keys:
        print(f"  {k:8s} " + "".join(
            f"{per_case[n].get(k, (0,0))[1]:6d}/{per_case[n].get(k, (0,0))[0]:<5d}"
            for n in DETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
