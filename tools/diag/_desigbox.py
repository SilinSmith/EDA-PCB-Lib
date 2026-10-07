# -*- coding: utf-8 -*-
"""位号新机制试验：不再让 VLM 在整块区域里自己找位号，而是**把 stage1 已经切出来的
文字框逐个用红圈标给 VLM**，让它只读圈住的那一小块。

    python work\\tools\\diag\\_desigbox.py --stage1 work\\stage1_v6 --limit 60 --radius 130 --per 6

判据：GT box 元件的位号，是否出现在附近某个 stage1 文字框的「VLM 读数」里。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import discover, load_json  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from vlm import client  # noqa: E402

IC = ("box", "ic", "boxic")
DES_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")
PROMPT = (
    "这是一张由多个小格拼成的图。每一格是电路原理图的一个局部放大。\n"
    "每一格里有一个**红色圆圈**，圈住了一小段文字。\n"
    "请对每一格读出**红圈圈住的那段文字**，原样照抄，不要补全、不要猜测。\n"
    "按顺序每格回一行，格式 `格号:文字`（格号从 1 开始）。\n"
    "如果某一格的圈里看不到文字，就回 `格号:NONE`。\n"
    "不要解释、不要输出别的行。")
CROP_PROMPT = (
    "这是一张电路原理图的局部放大。请读出**画面中心那个器件**的位号"
    "（reference designator，例如 U1、R5、C12）。\n"
    "要求：\n"
    "1. 只回位号本身，一行，不要解释、不要加标点、不要补全。\n"
    "2. **不要回器件的型号/part number**（如 MAX232、AMS1117-3.3、78M12 都不是位号）。\n"
    "3. 如果画面里印的是 U? 这种问号，就原样回 `U?`。\n"
    "4. 如果确实找不到位号，回 `NONE`。")


def clean(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s or ""))


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def build(bgr, boxes, side=170, cols=3, pad=10, label_h=26):
    crops = []
    for bb in boxes:
        x0 = max(0, int(bb[0] - pad)); y0 = max(0, int(bb[1] - pad))
        x1 = min(bgr.shape[1], int(bb[2] + pad) + 1); y1 = min(bgr.shape[0], int(bb[3] + pad) + 1)
        im = bgr[y0:y1, x0:x1].copy()
        if im.size == 0:
            continue
        cx = (bb[0] + bb[2]) / 2.0 - x0
        cy = (bb[1] + bb[3]) / 2.0 - y0
        r = int(max(bb[2] - bb[0], bb[3] - bb[1]) / 2.0 * 1.25) + 3
        cv2.circle(im, (int(cx), int(cy)), r, (0, 0, 235), 2, cv2.LINE_AA)
        crops.append(im)
    if not crops:
        return None, 0
    up = []
    for im in crops:
        k = min(side / max(1, im.shape[1]), side / max(1, im.shape[0]))
        k = min(k, 4.0)
        up.append(cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC))
    rows = (len(up) + cols - 1) // cols
    cw = max(i.shape[1] for i in up); ch = max(i.shape[0] for i in up)
    canvas = np.full((rows * (ch + label_h), cols * cw, 3), 255, np.uint8)
    for i, im in enumerate(up):
        rr, cc = divmod(i, cols)
        y = rr * (ch + label_h) + label_h; x = cc * cw
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, str(i + 1), (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (180, 0, 0), 2, cv2.LINE_AA)
    return canvas, len(up)


def parse(txt, n):
    out = {}
    for ln in (txt or "").splitlines():
        m = re.match(r"^\s*(\d+)\s*[:：]\s*(.+?)\s*$", ln)
        if not m:
            continue
        i = int(m.group(1))
        if 1 <= i <= n:
            out[i] = m.group(2)
    return out


def rect_dist(b, r):
    """文字框中心到元件 bbox（矩形）的距离 —— 大 IC 的位号离中心很远，必须用矩形距离。"""
    cxb = (b[0] + b[2]) / 2.0; cyb = (b[1] + b[3]) / 2.0
    dx = max(r[0] - cxb, 0.0, cxb - r[2])
    dy = max(r[1] - cyb, 0.0, cyb - r[3])
    return (dx * dx + dy * dy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "work" / "stage1_v6"))
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--radius", type=float, default=160.0)
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--inside", action="store_true",
                    help="也允许元件框内部的文字框当候选（默认只要框外的）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dump", default="")
    ap.add_argument("--crop-margin", type=float, default=70.0,
                    help="同时跑一遍现行的「整块裁片直读位号」做对照；0 = 不跑")
    a = ap.parse_args()
    S1 = Path(a.stage1)
    jobs = []
    Hs = {}

    def img_h(png):
        if png not in Hs:
            Hs[png] = float(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE).shape[0])
        return Hs[png]

    for c in discover():
        if not c.json_path:
            continue
        f = S1 / ("%s.stage1.json" % c.case_id)
        if not f.exists():
            continue
        s1 = json.loads(f.read_text(encoding="utf-8"))
        tbs = [t["bbox_img"] for t in (s1.get("texts") or []) if t.get("bbox_img")]
        H = img_h(c.png_path)
        gt = load_json(c.json_path)
        for k, g in (gt.get("components") or {}).items():
            if str(g.get("type") or "").lower() not in IC or not g.get("bbox"):
                continue
            gb = g["bbox"]
            # GT bbox 是左下角 y 向上；stage1 的 bbox_img 是 y 向下 —— 必须先换算
            r = (gb[0], H - gb[3], gb[2], H - gb[1])
            cands = []
            for b in tbs:
                d = rect_dist(b, r)
                if d > a.radius:
                    continue
                inside = (r[0] <= (b[0] + b[2]) / 2.0 <= r[2]
                          and r[1] <= (b[1] + b[3]) / 2.0 <= r[3])
                if inside and not a.inside:
                    continue
                cands.append((d, b))
            cands.sort(key=lambda x: x[0])
            near = [b for _d, b in cands[:a.per]]
            if not near:
                continue
            jobs.append(dict(case=c.case_id, key=k, truth=k, boxes=near,
                             png=c.png_path, gt_bbox=gb))
            if a.limit and len(jobs) >= a.limit:
                break
        if a.limit and len(jobs) >= a.limit:
            break
    print("待测 GT box 元件 %d 个（每个最多 %d 个附近文字框，矩形距离 <= %.0f）"
          % (len(jobs), a.per, a.radius))

    def work(j):
        bgr = imread_unicode(j["png"])
        canvas, n = build(bgr, j["boxes"], cols=a.cols)
        if canvas is None:
            return dict(j, reads={})
        txt, _dt = client.ask(canvas, PROMPT, model=a.model)
        out = dict(j, reads=parse(txt, n))
        if a.crop_margin > 0:
            H = float(bgr.shape[0]); bb = j["gt_bbox"]
            ib = (bb[0], H - bb[3], bb[2], H - bb[1])
            m = a.crop_margin
            x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
            x1 = min(bgr.shape[1], int(ib[2] + m) + 1); y1 = min(bgr.shape[0], int(ib[3] + m) + 1)
            im = bgr[y0:y1, x0:x1]
            im = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            t2, _dt2 = client.ask(im, CROP_PROMPT, model=a.model)
            out["crop"] = (t2 or "").strip().splitlines()[0].strip() if (t2 or "").strip() else ""
        return out

    res = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, jobs):
            res.append(r)
    tot = len(res)
    any_read = sum(1 for r in res if r["reads"])
    hit = sum(1 for r in res if clean(r["truth"]) in [clean(v) for v in r["reads"].values()])
    hitlo = sum(1 for r in res if clean(r["truth"]).upper()
                in [clean(v).upper() for v in r["reads"].values()])
    shaping = sum(1 for r in res for v in r["reads"].values() if DES_RE.match(clean(v)))
    print("  读到至少一格     %d / %d = %.4f" % (any_read, tot, any_read / max(1, tot)))
    print("  精确命中 GT 键   %d / %d = %.4f" % (hit, tot, hit / max(1, tot)))
    print("  忽略大小写命中   %d / %d = %.4f" % (hitlo, tot, hitlo / max(1, tot)))
    print("  读成位号形状的格 %d 个" % shaping)

    def first_shape(r):
        for v in r["reads"].values():
            c = clean(v)
            if DES_RE.match(c):
                return c
        return ""

    def crop_read(r):
        return clean(r.get("crop") or "")

    def score(fn, name):
        taken = wrong = right = 0
        for r in res:
            g = fn(r)
            if not DES_RE.match(g or ""):
                continue
            taken += 1
            if g.upper() == clean(r["truth"]).upper():
                right += 1
            else:
                wrong += 1
        print("  %-22s 采纳 %3d 对 %3d 错 %3d 精度 %.3f 召回 %.3f"
              % (name, taken, right, wrong, right / max(1, taken), right / tot))

    print("--- 各条通道单独打分（只在读成位号形状时采纳）---")
    score(crop_read, "整块裁片直读")
    score(first_shape, "红圈标文字框首个")
    score(lambda r: first_shape(r) or crop_read(r), "文字框优先/裁片兜底")
    score(lambda r: crop_read(r) or first_shape(r), "裁片优先/文字框兜底")
    if a.dump:
        Path(a.dump).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print("  dump -> %s" % a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
