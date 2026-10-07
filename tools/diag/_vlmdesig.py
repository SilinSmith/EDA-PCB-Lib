# -*- coding: utf-8 -*-
"""量 VLM 读「元器件位号」（IC 外框外的 U1 / U5 / IC2 / J3 …）的准确率。

为什么这是当前最关键的一步：`pin_f1`（权重 0.25）的 `flat()` 里有
`if comp not in box_comps: continue`，而 `box_comps` 取自 **GT** 的 box 元件键。
我们 box 元件的键命中率只有 **0.309**（GT 996 个 box，命中 308）⇒ 即使引脚
读得再准，键不对就一分不得。所以位号是 PinF1 的门票，也是 ComponentF1 的漏。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_vlmdesig.py --pred work\\stage2_v17 --limit 30 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from schematic_parser.data import discover, load_json  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _vlmread import ask_vlm  # noqa: E402

IC = ("box", "ic", "boxic")
PROMPT = (
    "这是电路原理图里的**一个元器件**（IC / 模块）。\n"
    "在该元器件外框**之外**（通常紧贴右上角或左上角）印着它的**位号**，"
    "形如「字母前缀 + 数字」，例如 U1、U5、IC2、J3、SW1。\n"
    "只回那个位号本身，不要解释、不要换行、不要任何其它字符。\n"
    "如果图里印的是 `U?`、`J?` 这种带问号的未标注形式，就原样回 `U?`。\n"
    "完全找不到就回 NONE。")


def norm(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s or "")).upper()


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def crop_pad(bgr, b, margin):
    h, w = bgr.shape[:2]
    x0 = max(0, int(b[0] - margin)); y0 = max(0, int(b[1] - margin))
    x1 = min(w, int(b[2] + margin)); y1 = min(h, int(b[3] + margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=60.0)
    ap.add_argument("--min-side", type=float, default=0.0, help=">0 时把短边补到该值（外扩）")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args(argv)

    pred_dir = Path(a.pred)
    jobs = []
    for c in discover():
        if not c.json_path:
            continue
        pf = pred_dir / ("%s.json" % c.case_id)
        if not pf.exists():
            continue
        pred = json.loads(pf.read_text(encoding="utf-8"))
        gt = load_json(c.json_path)
        gcomp = gt.get("components") or {}
        gbox = {k: v for k, v in gcomp.items()
                if str(v.get("type") or "").lower() in IC and v.get("bbox")}
        for key, v in (pred.get("components") or {}).items():
            if str(v.get("type") or "").lower() not in IC or not v.get("bbox"):
                continue
            best, bk = 0.0, None
            for gk, gv in gbox.items():
                s = iou(v["bbox"], gv["bbox"])
                if s > best:
                    best, bk = s, gk
            if best >= 0.5:
                jobs.append(dict(case=c.case_id, key=key, truth=bk, bbox=v["bbox"],
                                 png=c.png_path, Name=v.get("Name"), iou=best))
    if a.seed:
        import random
        random.Random(a.seed).shuffle(jobs)
    jobs = jobs[:a.limit] if a.limit else jobs
    print("待测 %d 个（预测键 vs GT 键，按 bbox IoU>=0.5 配对）" % len(jobs))
    if not jobs:
        return 1

    def work(j):
        bgr = imread_unicode(j["png"])
        H = float(bgr.shape[0])
        bb = j["bbox"]                                   # 预测的 bbox 是左下角 y 向上
        ib = (bb[0], H - bb[3], bb[2], H - bb[1])        # -> 图像坐标（y 向下）
        im = crop_pad(bgr, ib, a.margin)
        if im is None:
            return None
        if a.min_side and min(im.shape[:2]) < a.min_side:
            k = a.min_side / max(1, min(im.shape[:2]))
            im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        if a.render:
            cv2.imwrite(str(ROOT / "work" / "preview" / ("_desig_%s_%s.png" % (j["case"], j["key"]))), im)
        txt, dt = ask_vlm(im, prompt=PROMPT, model=a.model)
        got = (txt or "").strip().splitlines()[0].strip() if (txt or "").strip() else ""
        got = re.sub(r"[^0-9A-Za-z?]", "", got)
        return j, got, dt

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(work, jobs) if r]
    wall = time.time() - t0
    ok = exact = 0
    print("\n%-8s %-8s %-8s %-8s %6s  %s" % ("case", "pred键", "GT键", "VLM读", "IoU", "Name"))
    for j, got, dt in res:
        good = norm(got) == norm(j["truth"])
        exact += good
        if norm(got) and norm(got) != "NONE":
            ok += 1
        if not good and len(res) <= 60:
            print("%-8s %-8s %-8s %-8s %6.2f  %s"
                  % (j["case"], j["key"], j["truth"], got, j["iou"], str(j["Name"])[:16]))
    n = len(res)
    print("\n墙钟 %.1f s（%.2f s/个）" % (wall, wall / max(1, n)))
    print("  读出非空 %d/%d = %.4f" % (ok, n, ok / max(1, n)))
    print("  **与 GT 键一致** %d/%d = %.4f" % (exact, n, exact / max(1, n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
