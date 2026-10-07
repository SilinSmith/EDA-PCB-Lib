# -*- coding: utf-8 -*-
"""VLM 读元件「值」（0.1uF / 10K / 15pF …）的可行性 —— 目前 `component.text_mismatch`
里最大的一桶（1029 条中 r/c 的值占 718：我们没写值 347、写了但写错 266、GT 本就是空 105）。

做法：对每个 r/c/l/d/fuse/crystal 预测元件，按 bbox 与 GT 配对（IoU>=0.5），
裁 `bbox + margin`，一次问回「位号=值」，与 GT 的 `value` 逐字比（去掉空白后精确比）。

用法：
    python work\\tools\\diag\\_vlmval.py --limit 100 --margin 25
    python work\\tools\\diag\\_vlmval.py --limit 100 --dump work\\_vlmval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
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

VAL_TYPES = ("r", "c", "l", "fuse", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin")

PROMPT = (
    "这是电路原理图里的**一个分立元件**（电阻 / 电容 / 电感 / 晶振 / 保险丝）的局部放大图。\n"
    "它旁边印着两行字：\n"
    "  · **位号**：短，字母前缀 + 数字，例如 R5、C12、L1、Y1；\n"
    "  · **值**：元件参数，例如 10K、4.7uF、0.1uF、15pF、100nF、10uH、1M。\n"
    "只回**一行**，形式为 `位号=值`，例如 `C12=0.1uF`。\n"
    "值里面的单位原样照抄（u / n / p / K / M / Ω 都要保留）。\n"
    "如果找不到值，就回 `位号=NONE`；如果什么都没有，回 NONE。\n"
    "不要解释、不要换行。")

# 第二版：加三条硬约束（照抄不规范化 / 不读容差耐压行 / 先定位元件自己的位号）
PROMPT2 = (
    "这是电路原理图里**一个分立元件**（电阻 R / 电容 C / 电感 L / 晶振 Y / 保险丝 F）的局部放大图。\n"
    "图里会有好几个类似的元件挨在一起，每个旁边印两行字：\n"
    "  第 1 行 = **位号**（短的「字母+数字」，如 C5、R12、L1）；\n"
    "  第 2 行 = **值**（元件参数，如 10K、4.7uF、0.1uF、15pF、100nF）。\n"
    "先找到**画面中心**那个元件，只报**它自己**的位号与值，不要报旁边元件的。\n"
    "值的字符**原样照抄，一个字符都不要改、不要规范化**：\n"
    "图上写 `100nF` 就写 `100nF`（不要写成 `0.1uF`，也不要写成 `10nF`）；\n"
    "图上写 `0.1uF` 就写 `0.1uF`。\n"
    "**绝对不要**把第三行（容差/耐压，如 `10%`、`16V`、`50V`）当成值。\n"
    "只回一行 `位号=值`。找不到值就回 `位号=NONE`；什么都没有就回 NONE。\n"
    "不要解释、不要换行。")

PROMPTS = {"v1": PROMPT, "v2": PROMPT2}


def norm(s):
    return re.sub(r"\s+", "", str(s or ""))


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def parse_val(txt):
    t = (txt or "").strip().splitlines()
    t = t[0].strip() if t else ""
    if "=" in t:
        lhs, rhs = t.split("=", 1)
        return re.sub(r"[^0-9A-Za-z\u00b5\u03a9\u002e]", "", lhs), rhs.strip()
    return "", t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=25.0)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--pv", default="v1", choices=("v1", "v2"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dump", default="")
    args = ap.parse_args(argv)

    pred_dir = Path(args.pred)
    jobs = []
    for c in discover():
        if not c.json_path:
            continue
        pf = pred_dir / ("%s.json" % c.case_id)
        if not pf.exists():
            continue
        pred = json.loads(pf.read_text(encoding="utf-8"))
        gt = load_json(c.json_path)
        gcomp = {k: v for k, v in (gt.get("components") or {}).items() if v.get("bbox")}
        for key, v in (pred.get("components") or {}).items():
            if str(v.get("type") or "").lower() not in VAL_TYPES or not v.get("bbox"):
                continue
            best, bk = 0.0, None
            for gk, gv in gcomp.items():
                if str(gv.get("type") or "").lower() not in VAL_TYPES:
                    continue
                s = iou(v["bbox"], gv["bbox"])
                if s > best:
                    best, bk = s, gk
            if best >= 0.5:
                gv = gcomp[bk]
                jobs.append(dict(case=c.case_id, key=key, truth_key=bk,
                                 gt_val=gv.get("value"), pred_val=v.get("value"),
                                 bbox=v["bbox"], png=c.png_path))
    if args.seed:
        import random
        random.Random(args.seed).shuffle(jobs)
    jobs = jobs[:args.limit] if args.limit else jobs
    have_gt = [j for j in jobs if j["gt_val"]]
    print("配对上的分立元件 %d 个，其中 GT 真值非空的 %d 个" % (len(jobs), len(have_gt)))
    if not jobs:
        return 1

    def work(j):
        bgr = imread_unicode(j["png"])
        H = float(bgr.shape[0])
        bb = j["bbox"]
        ib = (bb[0], H - bb[3], bb[2], H - bb[1])
        h, w = bgr.shape[:2]
        x0 = max(0, int(ib[0] - args.margin)); y0 = max(0, int(ib[1] - args.margin))
        x1 = min(w, int(ib[2] + args.margin) + 1); y1 = min(h, int(ib[3] + args.margin) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        im = bgr[y0:y1, x0:x1]
        if args.scale != 1.0:
            im = cv2.resize(im, None, fx=args.scale, fy=args.scale,
                            interpolation=cv2.INTER_CUBIC)
        t0 = time.time()
        txt, _dt = ask_vlm(im, prompt=PROMPTS[args.pv], model=args.model)
        sec = time.time() - t0
        dkey, dval = parse_val(txt)
        return dict(case=j["case"], key=j["key"], truth_key=j["truth_key"],
                    gt_val=j["gt_val"], pred_val=j["pred_val"],
                    read_key=dkey, read_val=dval, sec=sec)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = [r for r in ex.map(work, jobs) if r]
    wall = time.time() - t0

    # 只看 GT 真值非空的（空值那些算不出对错）
    sub = [r for r in res if r["gt_val"]]
    exact = sum(1 for r in sub if norm(r["read_val"]) == norm(r["gt_val"]))
    nonempty = sum(1 for r in sub if norm(r["read_val"])
                   and norm(r["read_val"]).upper() != "NONE")
    base_ok = sum(1 for r in sub if norm(r["pred_val"]) == norm(r["gt_val"]))
    print("\n墙钟 %.1f s（%.2f s/个）" % (wall, wall / max(1, len(res))))
    print("  GT 值非空 %d 个：" % len(sub))
    print("    现行管线写对        %4d  %.4f" % (base_ok, base_ok / max(1, len(sub))))
    print("    VLM 值读出非空      %4d  %.4f" % (nonempty, nonempty / max(1, len(sub))))
    print("    **VLM 值与 GT 逐字同** %4d  %.4f" % (exact, exact / max(1, len(sub))))
    c = Counter("%s -> %s" % (r["gt_val"], r["read_val"]) for r in sub
                if norm(r["read_val"]) != norm(r["gt_val"]))
    print("\n最常读错：")
    for k, v in c.most_common(18):
        print("    %-26s %d" % (k, v))
    if args.dump:
        Path(args.dump).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n明细 -> %s" % args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
