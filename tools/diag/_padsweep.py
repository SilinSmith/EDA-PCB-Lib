# -*- coding: utf-8 -*-
"""外扩参数扫描：在**同一批检出框**上比不同 (pad_frac, pad_min, 纵横比) 的 OCR exact。

动机（`_textseg_common.py` 的实测）：
    GT 紧框原样读          0.8430      <- 识别器天花板
    GT 紧框 + `pad_box`    0.1958      <- 当前外扩把正确的框毁掉
    old 检出框原样         0.0954
    old 检出框 + `pad_box` 0.4022      <- 外扩把偏小的检出框救回来
⇒ 外扩是**治标**：它救小框、毁准框。所以要找的是「对小框救得多、对准框毁得少」的
   外扩方式，而不是「要不要外扩」。

臂：
    ``<det>_raw``            检出框原样
    ``<det>_h<pf>``          传统：四边各扩 ``max(pad_min, pf*h)``
    ``<det>_hv<pf>_<pfy>``   纵横分离：左右扩 ``pf*h``、上下扩 ``pfy*h``
    ``GT_raw`` / ``GT_h*``   真值紧框（天花板 / 外扩对真值框的伤害）

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\tools\\diag\\_padsweep.py --limit 10
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import TextEngine, imread_unicode  # noqa: E402
from _textseg_eval import ROLE_OF, cases, iou_matrix  # noqa: E402

H_CFG = dict(ink_mode="max", min_dim=2, min_area=3, line_len=15,
             resid_min_h=4, resid_min_dim=1)
DETS = {"old": None, "H": H_CFG}

# (tag, pad_fx *框宽*, pad_fy *框高*, pad_min_x, pad_min_y)
CONFIGS = [
    ("raw",        0.00, 0.00, 0.0, 0.0),
    ("w0h0.43",    0.00, 0.43, 0.0, 0.0),
    ("w0.05h0.43", 0.05, 0.43, 0.0, 0.0),
    ("w0.10h0.43", 0.10, 0.43, 0.0, 0.0),
    ("w0.15h0.43", 0.15, 0.43, 0.0, 0.0),
    ("w0.20h0.43", 0.20, 0.43, 0.0, 0.0),
    ("w0.30h0.43", 0.30, 0.43, 0.0, 0.0),
    ("w0.15h0.30", 0.15, 0.30, 0.0, 0.0),
    ("w0.15h0.35", 0.15, 0.35, 0.0, 0.0),
    ("w0.15h0.50", 0.15, 0.50, 0.0, 0.0),
    ("w0.15h0.60", 0.15, 0.60, 0.0, 0.0),
    ("w0.20h0.60", 0.20, 0.60, 0.0, 0.0),
    ("w0.25h0.70", 0.25, 0.70, 0.0, 0.0),
    ("w0.15h0.43f1", 0.15, 0.43, 1.0, 1.0),
    ("w0.20h0.50f1", 0.20, 0.50, 1.0, 1.0),
    ("w0.15h0.43f2", 0.15, 0.43, 2.0, 2.0),
]


def grow(b, fx, fy, pmx=0.0, pmy=0.0):
    """fx 是**框宽**的比例，fy 是**框高**的比例。"""
    w = max(1.0, b[2] - b[0])
    h = max(1.0, b[3] - b[1])
    px = max(pmx, fx * w)
    py = max(pmy, fy * h)
    return (b[0] - px, b[1] - py, b[2] + px, b[3] + py)


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
    args = ap.parse_args(argv)

    rr = TextEngine().recognizer()
    truth = {}          # (base, gi) -> text
    jobs = {}           # (arm, role) -> [((base, gi), box)]
    n_gt = 0
    hit = {}

    for ci, (png, js) in enumerate(cases(args.limit)):
        gts = gt_of(js)
        if not gts:
            continue
        bgr = imread_unicode(png)
        dets = {}
        for n, cfg in DETS.items():
            dets[n] = ([tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
                       if cfg is None else
                       [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **cfg)])

        for gi, (b, c, role) in enumerate(gts):
            key = ("GT", (ci, gi))
            truth[key] = c
            n_gt += 1
            for tag, fx, fy, pmx, pmy in CONFIGS:
                jobs.setdefault(("GT_" + tag, role), []).append(
                    (key, grow(b, fx, fy, pmx, pmy)))
            for n in DETS:
                if not dets[n]:
                    continue
                M = iou_matrix([b], dets[n])[0]
                j = int(np.argmax(M))
                if M[j] < 0.05:
                    continue
                key = (n, (ci, gi))
                truth[key] = c
                for tag, fx, fy, pmx, pmy in CONFIGS:
                    jobs.setdefault((n + "_" + tag, role), []).append(
                        (key, grow(dets[n][j], fx, fy, pmx, pmy)))

        for (arm, role), lst in jobs.items():
            outs = rr.read_many(bgr, [(bx, role) for _k, bx in lst])
            for (key, _bx), (txt, _cf) in zip(lst, outs):
                a = hit.setdefault(arm, {})
                # 同一 key 多 case 会重复出现；只记第一次出现的所属 case
                if key[0] == "GT" or key[0] in DETS:
                    pass
                prev = a.get(key)
                cur = (txt or "").strip() == truth[key]
                a[key] = cur if prev is None else (prev or cur)
            jobs[(arm, role)] = []       # 释放

        # 每张图读完即清，避免跨图串味
        for k in list(jobs):
            if not jobs[k]:
                del jobs[k]

    print(f"\n真值框 {n_gt}（{len(cases(args.limit))} 张图）")
    print("\n=== 每个检出器自己的分母（命中该真值框的那些）===")
    for n in DETS:
        base = set(k for k in hit.get(n + "_raw", {}))
        print(f"  {n:4s} 命中 {len(base):5d}/{n_gt} = {len(base)/n_gt:.3f}")
        print(f"       {'臂':14s} {'命中集exact':>10s} {'端到端':>8s}")
        for tag, _fx, _fy, _pmx, _pmy in CONFIGS:
            a = hit.get(n + "_" + tag, {})
            ok = sum(1 for k in base if a.get(k))
            print(f"       {tag:14s} {ok/max(1,len(base)):10.4f} {ok/max(1,n_gt):8.4f}")

    gbase = set(k for k in hit.get("GT_raw", {}))
    print(f"\n=== 真值紧框（天花板；{len(gbase)}）===")
    for tag, _fx, _fy, _pmx, _pmy in CONFIGS:
        a = hit.get("GT_" + tag, {})
        ok = sum(1 for k in gbase if a.get(k))
        print(f"       {tag:14s} {ok/max(1,len(gbase)):10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
