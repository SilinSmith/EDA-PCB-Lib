# -*- coding: utf-8 -*-
"""IC 位号读取：prompt / 裁片策略的多种写法对照。

`_desigwhy.py` 量出：996 个 GT box 元件里 414（0.416）键对上，
**311 个仍是兜底键 pNNN**（VLM 没给出可用的改名）、**262 个读成了别的键**。
后者的样例显示两类系统性错误：
  · 把**器件型号**当位号：`U2 -> MAX232`、`U3 -> STC11`
  · 读到**邻居**的位号：`U2 -> U1`、`U1 -> U4`、`U1 -> U2`
（`MAX232`/`STC11` 恰好符合 `^[A-Za-z]{1,3}\\d{1,4}$`，所以正则挡不住。）

这个脚本在同一批 GT box 元件上比几种写法，找最能压住这两类错误的那一种：

  base      现行 prompt + bbox 外扩 60
  nomodel   加「不许回型号、位号只有 2~5 个字符」的硬约束
  band      只裁器件框**上方**那条带（型号多印在下方，结构性排除）
  mask      整块裁，但把器件框内部涂成灰底并声明「灰底里不要读」
  tight     外扩缩到 28（减少邻居串入）

用法：
    python work\\tools\\diag\\_desigexp.py --limit 80 --variants base,nomodel,band,mask,tight
    python work\\tools\\diag\\_desigexp.py --limit 80 --dump work\\_desigexp.json
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
DESIG_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")

BASE = (
    "这是电路原理图里的**一个元器件**（IC / 模块）。\n"
    "在该元器件外框**之外**（通常紧贴右上角或左上角）印着它的**位号**，"
    "形如「字母前缀 + 数字」，例如 U1、U5、IC2、J3、SW1。\n"
    "只回那个位号本身，不要解释、不要换行、不要任何其它字符。\n"
    "如果图里印的是 `U?`、`J?` 这种带问号的未标注形式，就原样回 `U?`。\n"
    "完全找不到就回 NONE。")

NOMODEL = (
    "这是电路原理图里的**一个元器件**（IC / 模块），图里印着两种文字：\n"
    "  (a) 它的**位号**：短，2~5 个字符，「字母前缀 + 数字」，例如 U1、IC2、J3；\n"
    "  (b) 它的**型号 / 器件名**：长，通常是 4 个以上字符的字母数字串，"
    "例如 MAX232、STC11、AMS1117、TPS7A4701RGWR。\n"
    "**只回 (a) 位号**。\n"
    "千万不要回 (b) 型号 —— 哪怕型号看起来也像「字母+数字」。\n"
    "如果只找得到型号、找不到位号，回 NONE。\n"
    "如果位号印成 `U?`、`J?` 这种带问号的未标注形式，原样回 `U?`。\n"
    "只回那一个字符串，不要解释、不要换行。")

MASK_NOTE = (
    "\n图里被涂成**灰色底**的那块就是元器件的本体（外框内部），"
    "**灰底里面的文字（引脚名、型号）一律不要读**，只读灰底**外面**的位号。")


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


def crop_full(bgr, ib, margin):
    h, w = bgr.shape[:2]
    x0 = max(0, int(ib[0] - margin)); y0 = max(0, int(ib[1] - margin))
    x1 = min(w, int(ib[2] + margin) + 1); y1 = min(h, int(ib[3] + margin) + 1)
    if x1 <= x0 or y1 <= y0:
        return None, None
    return bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def crop_band(bgr, ib, margin, frac=1.15):
    """器件框上方那条带：高度取 body 高的 frac 倍，左右各外扩 margin。"""
    h, w = bgr.shape[:2]
    bh = ib[3] - ib[1]
    x0 = max(0, int(ib[0] - margin))
    x1 = min(w, int(ib[2] + margin) + 1)
    y0 = max(0, int(ib[1] - bh * frac))
    y1 = max(1, min(h, int(ib[1]) + 1))
    if x1 <= x0 or y1 <= y0:
        return None, None
    return bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def build(bgr, ib, variant, margin):
    if variant == "band":
        return crop_band(bgr, ib, margin)
    im, box = crop_full(bgr, ib, margin)
    if im is None:
        return None, None
    if variant == "mask":
        # 把 body 内部涂灰（保留外框线本身之外的邻域）
        x0, y0, x1, y1 = box
        ax0 = int(ib[0] - x0); ay0 = int(ib[1] - y0)
        ax1 = int(ib[2] - x0); ay1 = int(ib[3] - y0)
        ax0 = max(0, min(im.shape[1], ax0)); ax1 = max(0, min(im.shape[1], ax1))
        ay0 = max(0, min(im.shape[0], ay0)); ay1 = max(0, min(im.shape[0], ay1))
        if ax1 > ax0 and ay1 > ay0:
            im[ay0 + 2:ay1 - 1, ax0 + 2:ax1 - 1] = (228, 228, 228)
    return im, box


def prompt_of(variant):
    if variant == "base":
        return BASE
    if variant == "nomodel":
        return NOMODEL
    if variant == "mask":
        return NOMODEL + MASK_NOTE
    if variant == "band":
        return NOMODEL + ("\n（这张裁片只截取了器件外框**上方**的一条带。）")
    if variant == "tight":
        return NOMODEL
    raise SystemExit("未知 variant: %s" % variant)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=60.0)
    ap.add_argument("--tight-margin", type=float, default=28.0)
    ap.add_argument("--variants", default="base,nomodel,band,mask,tight")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dump", default="")
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
        gbox = {k: v for k, v in (gt.get("components") or {}).items()
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
                                 png=c.png_path, Name=v.get("Name")))
    if a.seed:
        import random
        random.Random(a.seed).shuffle(jobs)
    jobs = jobs[:a.limit] if a.limit else jobs
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    print("待测 %d 个 GT box 元件 × %d 种写法 = %d 次调用"
          % (len(jobs), len(variants), len(jobs) * len(variants)))
    if not jobs:
        return 1

    def work(arg):
        j, variant = arg
        bgr = imread_unicode(j["png"])
        H = float(bgr.shape[0])
        bb = j["bbox"]
        ib = (bb[0], H - bb[3], bb[2], H - bb[1])
        margin = a.tight_margin if variant == "tight" else a.margin
        im, box = build(bgr, ib, variant, margin)
        if im is None or im.size == 0:
            return dict(case=j["case"], key=j["key"], truth=j["truth"],
                        variant=variant, got="", err="empty", sec=0.0)
        if a.render and variant in ("base", "nomodel", "band", "mask"):
            cv2.imwrite(str(ROOT / "work" / "preview"
                            / ("_dx_%s_%s_%s.png" % (j["case"], j["key"], variant))), im)
        t0 = time.time()
        txt, _dt = ask_vlm(im, prompt=prompt_of(variant), model=a.model)
        sec = time.time() - t0
        got = (txt or "").strip().splitlines()[0].strip() if (txt or "").strip() else ""
        got = re.sub(r"[^0-9A-Za-z?]", "", got)
        return dict(case=j["case"], key=j["key"], truth=j["truth"],
                    variant=variant, got=got, Name=j.get("Name"), sec=sec)

    t0 = time.time()
    args = [(j, v) for j in jobs for v in variants]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(work, args))
    wall = time.time() - t0

    n = len(jobs)
    print("\n墙钟 %.1f s（%.2f s/次调用）\n" % (wall, wall / max(1, len(res))))
    print("%-8s %7s %7s %7s %7s %7s %7s" %
          ("写法", "一致", "一致率", "非空", "读到型号", "读到邻居", "NONE/其它"))
    gt_keys_by_case = {}
    for j in jobs:
        gt_keys_by_case.setdefault(j["case"], set()).add(norm(j["truth"]))
    for v in variants:
        rows = [r for r in res if r["variant"] == v]
        if not rows:
            continue
        exact = sum(1 for r in rows if norm(r["got"]) == norm(r["truth"]))
        nonempty = sum(1 for r in rows if norm(r["got"]) and norm(r["got"]) != "NONE")
        typ = sum(1 for r in rows
                  if norm(r["got"]) and norm(r["got"]) != norm(r["truth"])
                  and len(norm(r["got"])) >= 4 and not DESIG_RE.match(r["got"]))
        nb = sum(1 for r in rows
                 if norm(r["got"]) and norm(r["got"]) != norm(r["truth"])
                 and norm(r["got"]) in gt_keys_by_case.get(r["case"], set()))
        none = len(rows) - nonempty
        print("%-8s %7d %7.4f %7d %7d %7d %7d"
              % (v, exact, exact / len(rows), nonempty, typ, nb, none))
        if a.render:
            bad = ["%s %s->%s" % (r["case"], r["truth"], r["got"])
                   for r in rows if norm(r["got"]) != norm(r["truth"])]
            print("       错误样例：" + " | ".join(bad[:6]))
    if a.dump:
        Path(a.dump).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n明细 -> %s" % a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
