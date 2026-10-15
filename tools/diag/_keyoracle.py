# -*- coding: utf-8 -*-
"""★ 关键实验：**位号（元件键）错误到底值多少分？**

动机：`metrics.evaluate_case` 的顺序是
    assign_components → rewrite_keys(用 keymap 改写预测键) → pin_f1 / net_f1 / line_f1
而
  * `pin_f1` 的键是 `(元件, 引脚号)`；
  * `net_hypergraph_f1`（micro）的 tp 是**引脚集合的交集**；
  * `net_line_f1` **只比较「网络配对成功」的那些网络的边**。
⇒ 一个元件键错，会同时污染 引脚 / 网络 / 导线 三个分项。本脚本量这个级联有多大。

做法：给预测做一次**「上帝视角」的键改写** —— 每个预测元件按 bbox 中心距去找
最近的 GT 元件（≤40px），把预测键换成 GT 键，然后重新打分。
这是「位号全对」的**上界**（同时也会顺带抬高元件分，所以元件分的提升要单独看）。

用法：
    python work\\tools\\diag\\_keyoracle.py --n 60
    python work\\tools\\diag\\_keyoracle.py --n 60 --each
    python work\\tools\\diag\\_keyoracle.py --n 220 --out work\\_full220
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (EvalConfig, _box, _centre, canon_type,  # noqa: E402
                                      evaluate_case, normalise_doc, rewrite_keys)
from stage2_eval import build_index, resolve                               # noqa: E402

METRICS = ["component", "pin", "net_hypergraph", "net_line"]
ZH = {"component": "元件", "pin": "引脚", "net_hypergraph": "网络", "net_line": "导线"}
GATE = 40.0


def lookup(cid):
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    case, _ = resolve(cid, exact, digits)
    if case is None:
        case, _ = resolve(cid.lstrip("0") or "0", exact, digits)
    return case


def pred_of(cid, out_dir):
    for name in (cid + "_Pred.json", cid + ".json"):
        q = Path(out_dir) / "final_results" / name
        if q.exists():
            return load_json(q)
    return None


def cheat_keymap(gt, pred):
    """预测元件 → 最近的 GT 元件键（≤40px）。只看「预测有 bbox」的。"""
    gctr = {}
    for k, v in (gt.get("components") or {}).items():
        c = _centre(_box((v or {}).get("bbox")))
        if c:
            gctr[k] = c
    km, n = {}, 0
    for k, v in (pred.get("components") or {}).items():
        c = _centre(_box((v or {}).get("bbox")))
        if not c:
            continue
        best, bd = None, 1e18
        for gk, gc in gctr.items():
            d = ((c[0] - gc[0]) ** 2 + (c[1] - gc[1]) ** 2) ** 0.5
            if d < bd:
                best, bd = gk, d
        if best is not None and bd <= GATE:
            km[k] = best
            n += 1
    return km, n


def run(n, out_dir, each, split):
    cfg = EvalConfig()
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == split][:n]
    base, orac = [], []
    print("%-8s %8s | %-31s | %-31s" % (
        "case", "改写键", "基线 元件/引脚/网络/导线", "上帝键 元件/引脚/网络/导线"))
    for c in cs:
        praw = pred_of(c.case_id, out_dir)
        if praw is None:
            continue
        gt = normalise_doc(load_json(Path(c.json_path)))
        pred = normalise_doc(praw)
        km, nk = cheat_keymap(gt, pred)
        b = evaluate_case(gt, pred, case_id=c.case_id, cfg=cfg)
        pred2 = rewrite_keys(pred, km)
        o = evaluate_case(gt, pred2, case_id=c.case_id, cfg=cfg)
        base.append(b)
        orac.append(o)
        if each:
            print("%-8s %8d | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f %8.4f" % (
                c.case_id, nk, b.component.f1, b.pin.f1, b.net_hypergraph.f1,
                b.net_line.f1, o.component.f1, o.pin.f1, o.net_hypergraph.f1,
                o.net_line.f1))
    if not base:
        print("没有数据")
        return 1
    N = len(base)
    print("-" * 96)
    print("case 数 %d（split=%s，out=%s）" % (N, split, out_dir))
    print("%-10s %10s %10s %10s" % ("分项", "基线", "上帝键", "增量"))
    sb = so = 0.0
    for m in METRICS:
        fb = sum(getattr(r, m).f1 for r in base) / N
        fo = sum(getattr(r, m).f1 for r in orac) / N
        print("%-10s %10.4f %10.4f %+10.4f" % (ZH[m], fb, fo, fo - fb))
    sb = sum(r.score for r in base) / N
    so = sum(r.score for r in orac) / N
    print("%-10s %10.4f %10.4f %+10.4f" % ("总分", sb, so, so - sb))
    tb = {m: sum(getattr(r, m).tp for r in base) for m in METRICS}
    to = {m: sum(getattr(r, m).tp for r in orac) for m in METRICS}
    print("tp 合计：" + "  ".join("%s %d→%d" % (ZH[m], tb[m], to[m]) for m in METRICS))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "work" / "_submit_out220r36"))
    ap.add_argument("--split", default="200_train_cases")
    ap.add_argument("--each", action="store_true")
    a = ap.parse_args()
    return run(a.n, a.out, a.each, a.split)


if __name__ == "__main__":
    raise SystemExit(main())
