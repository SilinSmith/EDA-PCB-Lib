# -*- coding: utf-8 -*-
"""验证 `stage1.pincol` 的引脚列枚举：拿真实 case 的 stage1 产物 + 原图跑一遍，
和 GT 的引脚点对账，看枚举出来的脚数是否够、落在哪。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_pincoltest.py --case 0143 --key U1
    python work\\tools\\diag\\_pincoltest.py --case 0143 --top 6      # 该 case 全部 IC
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

from schematic_parser.data import discover, load_json          # noqa: E402
from stage1 import pincol                                      # noqa: E402
from stage2.text import imread_unicode                         # noqa: E402

DEFAULT_DIRS = [
    ROOT / "work" / "_submit_out220r36" / "runs" / "stage1",
    ROOT / "work" / "_fusion_out220no4" / "runs" / "stage1",
]


def find_stage1(case: str) -> Path | None:
    for d in DEFAULT_DIRS:
        p = d / ("%s.stage1.json" % case)
        if p.exists():
            return p
    return None


def iou(a, b) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="0143")
    ap.add_argument("--key", default="U1")
    ap.add_argument("--top", type=int, default=1, help="只看 GT 引脚最多的前 N 个元件")
    ap.add_argument("--dir", default="")
    ap.add_argument("--vis", default="")
    a = ap.parse_args(argv)

    hit = None
    for c in discover():
        if c.case_id == a.case and c.json_path:
            hit = c
            break
    if hit is None:
        print("没找到 case", a.case); return 1
    gt = load_json(hit.json_path)
    bgr = imread_unicode(hit.png_path)
    H, W = bgr.shape[:2]

    sp = Path(a.dir) if a.dir else find_stage1(a.case)
    if sp is None or not sp.exists():
        print("没找到 stage1 产物"); return 1
    doc = json.loads(sp.read_text(encoding="utf-8"))
    comps = doc.get("components") or []
    texts = doc.get("texts") or []
    print("case %s  图 %dx%d  stage1 元件 %d  文字框 %d  (%s)"
          % (a.case, W, H, len(comps), len(texts), sp.name))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 170).astype(np.uint8)

    # GT 侧：按引脚数排 IC
    gtc = gt.get("components") or {}
    gtp = gt.get("pins") or {}
    rows = []
    for k, v in gtc.items():
        n = len(gtp.get(k) or {})
        if n:
            rows.append((n, k, v))
    rows.sort(reverse=True, key=lambda r: r[0])
    if a.key:
        rows = [r for r in rows if r[1] == a.key] or rows[: a.top]
    else:
        rows = rows[: a.top]

    cols = pincol.enumerate_pins(bgr, comps, texts, ink=ink)
    print(pincol.summarise(cols))
    print("")

    for n_gt, key, gv in rows:
        gbb_j = gv.get("bbox") or [0, 0, 0, 0]
        gbb = (gbb_j[0], H - gbb_j[3], gbb_j[2], H - gbb_j[1])   # JSON -> 图像
        gp = [((p["point"]["x"]), H - p["point"]["y"]) for p in (gtp.get(key) or {}).values()]
        best, bi = 0.0, None
        for c in comps:
            ib = c.get("bbox_img")
            if not ib:
                continue
            v = iou(gbb, [float(x) for x in ib])
            if v > best:
                best, bi = v, c
        if bi is None:
            print("%-4s GT 引脚 %-3d  →  **stage1 里没有框盖住它**" % (key, n_gt))
            continue
        recs = cols.get(str(bi.get("id")), [])
        pts = [(r["x"], r["y"]) for r in recs]
        near = 0
        for gx, gy in gp:
            if pts and min((gx - px) ** 2 + (gy - py) ** 2 for px, py in pts) <= 25:
                near += 1
        print("%-4s GT 引脚 %-3d  stage1 框 %s IoU %.2f type=%s"
              % (key, n_gt, bi.get("id"), best, bi.get("type")))
        print("     枚举出 %d 个脚（补格点 %d）；GT 点 5px 内命中 %d/%d"
              % (len(recs), sum(1 for r in recs if r["synth"]), near, n_gt))
        by_side = {}
        for r in recs:
            by_side.setdefault(r["side"], []).append(r)
        for s in ("L", "R", "T", "B"):
            rr = by_side.get(s) or []
            if not rr:
                continue
            pit = sorted({r["pitch"] for r in rr if r["pitch"]})
            print("     %s 边 %2d 个  pitch=%s  带号 %d / 带名 %d"
                  % (s, len(rr), pit, sum(1 for r in rr if r["out_texts"]),
                     sum(1 for r in rr if r["in_texts"])))
        # 落点抽查
        show = sorted(recs, key=lambda r: (r["side"], r["idx"]))
        for r in show[: a.top * 40 if a.top > 1 else 16]:
            print("       %s#%-3d (%7.1f,%7.1f) synth=%d out=%s in=%s"
                  % (r["side"], r["idx"], r["x"], r["y"], r["synth"],
                     r["out_texts"], r["in_texts"]))
        if a.vis:
            out = bgr.copy()
            for r in recs:
                col = (0, 0, 255) if r["synth"] else (0, 180, 0)
                cv2.circle(out, (int(r["x"]), int(r["y"])), 3, col, -1)
                for w, cc in ((r["out_win"], (255, 0, 0)), (r["in_win"], (0, 165, 255))):
                    cv2.rectangle(out, (int(w[0]), int(w[1])), (int(w[2]), int(w[3])), cc, 1)
            x0, y0, x1, y1 = [int(v) for v in bi["bbox_img"]]
            pad = 60
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
            crop = out[y0:y1, x0:x1]
            s = min(3.0, 1400.0 / max(1, crop.shape[1]))
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                              interpolation=cv2.INTER_CUBIC)
            p = Path(a.vis)
            cv2.imwrite(str(p), crop)
            print("     vis ->", p, crop.shape)
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
