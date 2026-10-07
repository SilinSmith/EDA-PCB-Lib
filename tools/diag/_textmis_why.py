# -*- coding: utf-8 -*-
"""把 `text_mismatch`（元件框对上了、type 也对，但 Name/value 不同）拆成可行动的几类。

用户 m06734 要的「把目前的各种错误列出来看看」如果只给一个 7644 的数字没有意义——
必须回答：**这 7644 条里，多少是「框没框对」、多少是「框对了但字认错」**。
只有后者才该去动识别器，前者该去动切分/生长。

分类规则（只看配对上的那一对 GT 框 g 与预测框 p）：
    empty     我们那颗元件 Name/value 是空的（根本没读到字）
    ocr       IoU>=0.5 且宽度比在 0.7~1.4 之间  -> 框是对的，字认错了
    narrow    p 明显比 g 窄（<0.7 倍）           -> 框被切碎，名字被截断
    wide      p 明显比 g 宽（>1.4 倍）           -> 框把邻居吃进来了
    shift     其余（位置偏移、缺角）

用法：
    python Y:\\workspace\\EDA\\work\\tools\\diag\\_textmis_why.py <预测目录> [--limit 220]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))

import comp_cases as CC  # noqa: E402

BUCKETS = ("empty", "ocr", "narrow", "wide", "shift")


def classify(g, p):
    tp, tg = CC.text_of(p), CC.text_of(g)
    if tp in (None, ""):
        return "empty"
    gw = max(1e-6, g["box"][2] - g["box"][0])
    pw = max(1e-6, p["box"][2] - p["box"][0])
    r = pw / gw
    v = CC.iou(g["box"], p["box"])
    if r < 0.7:
        return "narrow"
    if r > 1.4:
        return "wide"
    if v >= 0.5:
        return "ocr"
    return "shift"


def main(argv=None):
    ap = argparse.ArgumentParser(description="text_mismatch 归因")
    ap.add_argument("pred", default=r"Y:\workspace\EDA\work\stage3_best", nargs="?")
    ap.add_argument("--data-root", default=CC.DEFAULT_DATA_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    idx = CC.gt_index(args.data_root)
    files = sorted(Path(args.pred).glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if args.limit:
        files = files[:args.limit]

    tally = Counter()
    examples = {b: [] for b in BUCKETS}
    # 只对「GT 名字比我们长」这件事单独计一次：那才是「不可分就默认是长的」能救的
    g_longer = 0
    for f in files:
        case = f.name.replace(".json", "")
        rec = idx.get(case) or idx.get(re.sub(r"\D", "", case) or case)
        if rec is None:
            continue
        pdoc = json.loads(f.read_text(encoding="utf-8"))
        gdoc = CC.load_json(rec.json_path)
        H = 0
        # GT 的 y 要翻转，必须有图高；用 predictions 里最大的 bbox 上界估不出来，
        # 所以老老实实读图高（comp_cases.gt_components 需要它）。
        try:
            import cv2
            H = cv2.imdecode(__import__("numpy").fromfile(str(rec.png_path),
                                                        dtype="uint8"),
                             cv2.IMREAD_COLOR).shape[0]
        except Exception:
            continue
        gt = CC.gt_components(gdoc, H)
        pr = CC.pred_components(pdoc, H)
        pairs, _ug, _up = CC.pair(gt, pr)
        for i, j, _v in pairs:
            g, p = gt[i], pr[j]
            if str(g["type"]).lower() != str(p["type"]).lower():
                continue
            tg, tp = CC.text_of(g), CC.text_of(p)
            if tg is None and tp is None:
                continue
            if tg == tp:
                continue
            b = classify(g, p)
            tally[b] += 1
            if tg and tp and len(str(tg)) > len(str(tp)):
                g_longer += 1
            if len(examples[b]) < 6:
                examples[b].append((case, g["type"], tg, tp,
                                    round(p["box"][0] - g["box"][0], 1),
                                    round(p["box"][2] - g["box"][2], 1)))

    tot = sum(tally.values())
    print(f"\n配对上的 text_mismatch 共 {tot} 条：")
    for b in BUCKETS:
        n = tally[b]
        print(f"  {b:7s} {n:5d}  {100.0*n/max(1,tot):5.1f}%")
    print(f"\n其中「GT 文字比我们长」的：{g_longer} 条 "
          f"（{100.0*g_longer/max(1,tot):.1f}%）—— 「不可分就默认是长的」只可能动到这一批")
    for b in BUCKETS:
        print(f"\n--- {b} 例 ---")
        for case, ty, tg, tp, dx1, dx2 in examples[b]:
            print(f"  {case} {ty:10s} GT={tg!r:22s} 预测={tp!r:22s} 左边界差{dx1} 右边界差{dx2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
