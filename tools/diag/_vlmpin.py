# -*- coding: utf-8 -*-
"""VLM 读 IC 引脚号/引脚名 —— 现在管线上「从来没从图上读」的那一项。

`PinF1` 权重 0.25，我们现在只拿 1.185/25（tp 69 / fp 1891 / fn 9287）。
`assemble.py` 的 box 分支要求 OCR 读出的号「全是 1..999 的互不相同整数」，
一颗 72 脚 IC 一个读废就整颗退回顺序号，所以真实引脚号/名几乎从没进过预测。

这里直接量：把 GT 的 box 元件框（外扩 margin）交给 VLM，让它列引脚，
与官方 GT 的 `pins[comp]`（`pin_<号>` -> `{pinname}`）比。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_vlmpin.py --splits 10GT --max-comp 6
    python work\\tools\\diag\\_vlmpin.py --splits 10GT,10_GTcase --max-comp 20 --workers 4
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

from schematic_parser.data import SPLITS, discover, load_json  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _vlmread import ask_vlm, upscale  # noqa: E402

PIN_PROMPT = (
    "这是电路原理图里的一个元器件（IC / 模块）。请读出它**全部的引脚**：\n"
    "- number：图中紧贴引脚、位于元器件外框之外的那串数字（引脚编号）\n"
    "- name：引脚旁边的信号名文字（如 OUT、GND、VCC、SPI_MOSI）\n"
    "输出一个 JSON 数组，每项 {\"number\": \"...\", \"name\": \"...\"}。\n"
    "读不到 number 就写 null。除了这个 JSON 数组不要输出任何其它内容。")

TERSE_PROMPT = (
    "这是电路原理图里的一个元器件（IC / 模块）。图里每个引脚旁边印着"
    "一个引脚编号（数字）和一个信号名。\n"
    "只回**一行**答案，格式是 `编号:名称` 之间用英文逗号分隔，例如：\n"
    "1:VIN,2:GND,3:EN\n"
    "不要 JSON、不要换行、不要解释、不要任何其它字符。")
MARGIN = 26.0


def crop_gt(bgr, bbox, margin=MARGIN):
    h, w = bgr.shape[:2]
    x0 = max(0, int(bbox[0] - margin))
    y0 = max(0, int(bbox[1] - margin))
    x1 = min(w, int(bbox[2] + margin))
    y1 = min(h, int(bbox[3] + margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]


def parse_pins(txt):
    t = (txt or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", t).strip()
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        # 退一步：整段按 `编号:名称` 逗号分隔解析（terse 格式）
        out = []
        for seg in re.split(r"[,\n;]+", t):
            mm = re.match(r"\s*[\[\(]?\s*([0-9A-Za-z]{1,4})\s*[:：]\s*([^\]\)]*?)\s*[\]\)]?\s*$",
                          seg)
            if mm:
                out.append((mm.group(1), mm.group(2)))
        return out
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    out = []
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                out.append((it.get("number"), it.get("name")))
    return out


def norm_pin_name(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s or "")).upper()


def norm_pin_num(s):
    if s is None:
        return ""
    return re.sub(r"[^0-9A-Za-z]", "", str(s)).upper()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="10GT")
    ap.add_argument("--max-comp", type=int, default=6)
    ap.add_argument("--min-pins", type=int, default=4)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--save", default="")
    ap.add_argument("--style", default="json", choices=("json", "terse"))
    a = ap.parse_args(argv)
    prompt = PIN_PROMPT if a.style == "json" else TERSE_PROMPT

    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    cs = discover(splits=splits)
    jobs = []
    for c in cs:
        if not c.json_path:
            continue
        gt = load_json(c.json_path)
        comps = gt.get("components") or {}
        pins = gt.get("pins") or {}
        for k, v in comps.items():
            if str(v.get("type") or "").lower() not in ("box", "ic", "boxic"):
                continue
            gp = pins.get(k) or {}
            if len(gp) < a.min_pins:
                continue
            jobs.append(dict(case=c.case_id, key=k, bbox=v.get("bbox"),
                             Name=v.get("Name"), gt=gp,
                             png=c.png_path))
            if len(jobs) >= a.max_comp:
                break
        if len(jobs) >= a.max_comp:
            break
    print("待测元件 %d 个（%s）" % (len(jobs), a.splits))
    if not jobs:
        return 1

    def work(j):
        bgr = imread_unicode(j["png"])
        H = bgr.shape[0]
        b = j["bbox"]
        ib = (b[0], H - b[3], b[2], H - b[1])          # GT 左下角 -> 图像左上角
        im = crop_gt(bgr, ib, a.margin)
        if im is None:
            return None
        im = upscale(im, a.scale)
        if a.save:
            cv2.imwrite(a.save.replace(".png", "_%s.png" % j["key"]), im)
        txt, dt = ask_vlm(im, prompt=prompt, model=a.model)
        return j, txt, dt, parse_pins(txt)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(work, jobs) if r]
    wall = time.time() - t0

    tot_gt = tot_num = tot_pair = tot_pred = 0
    print("\n%-6s %-6s %-5s %-5s %-5s %-6s %-7s %s"
          % ("case", "key", "GT", "pred", "num", "pair", "T(s)", "Name"))
    for j, txt, dt, pred in res:
        gp = j["gt"]
        gt_nums = {norm_pin_num(k.split("_", 1)[-1]) for k in gp}
        gt_pairs = {(norm_pin_num(k.split("_", 1)[-1]),
                     norm_pin_name(v.get("pinname"))) for k, v in gp.items()}
        gn = {norm_pin_num(x[0]) for x in pred if x[0]}
        gpn = {(norm_pin_num(x[0]), norm_pin_name(x[1])) for x in pred if x[0]}
        num_hit = len(gn & gt_nums)
        pair_hit = len(gpn & gt_pairs)
        tot_gt += len(gt_nums)
        tot_pred += len(gn)
        tot_num += num_hit
        tot_pair += pair_hit
        print("%-6s %-6s %-5d %-5d %-5d %-6d %-7.0f %s"
              % (j["case"], j["key"], len(gt_nums), len(gn), num_hit,
                 pair_hit, dt, str(j["Name"])[:20]))
        if os.environ.get("VLM_SHOW"):
            print("     raw:", (txt or "")[:400].replace("\n", " "))

    print("\n墙钟 %.1f s（%.1f s/元件）" % (wall, wall / max(1, len(res))))
    print("=== 引脚 ===")
    print("  GT 引脚数 %d，VLM 报出 %d" % (tot_gt, tot_pred))
    print("  引脚号对上   %5d/%d = %.4f" % (tot_num, tot_gt, tot_num / max(1, tot_gt)))
    print("  号+名都对上  %5d/%d = %.4f" % (tot_pair, tot_gt, tot_pair / max(1, tot_gt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
