# -*- coding: utf-8 -*-
"""引脚分的**结构上界**：如果「我们产出的引脚」能一一对上 GT 的引脚，能拿多少？

已知事实（`_pindist.py`）：
  * 在元件键已配对的前提下，40 张训练集里 GT 与预测**同时存在同一 (元件,引脚号)** 的
    只有 151 根（GT 侧 IC 引脚约 2000+ 根）；
  * 这 151 根里名字逐字符相等的只有 24.5%，坐标距 ≤5px 的只有 11/37。

所以问题分两层：(a) **找不到对应的那根引脚**（编号体系对不上）；(b) 找到了但名字/坐标不对。
本脚本把 (a) 用**上帝视角**消掉：对每个 GT IC 引脚，在**同一个元件键下**取预测引脚里
**离 GT `point` 最近的那一根**（点距就是官方 5px 判据用的量），再看名字对不对。

⇒ 得到「绑定问题解决后，名字+坐标还能拿到多少」的上界，用来判断下一步该修绑定还是修名字。

用法：
    python work\\tools\\diag\\_pinoracle.py --n 60
    python work\\tools\\diag\\_pinoracle.py --n 10 --split 10GT
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES, _point, canon_type,  # noqa: E402
                                      assign_components, normalise_doc, rewrite_keys)


def run(n, split, out_dir, examples):
    cfg = EvalConfig()
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == split][:n]
    st = Counter()
    ex = []
    for c in cs:
        praw = None
        for nm in (c.case_id + "_Pred.json", c.case_id + ".json"):
            q = Path(out_dir) / "final_results" / nm
            if q.exists():
                praw = load_json(q)
                break
        if praw is None:
            continue
        gt = normalise_doc(load_json(Path(c.json_path)))
        pred2 = rewrite_keys(normalise_doc(praw),
                             assign_components(gt, normalise_doc(praw), cfg)[1])
        box = {k for k, v in (gt.get("components") or {}).items()
               if canon_type((v or {}).get("type")) in IC_TYPES}
        for comp in box:
            gpin = gt.get("pins", {}).get(comp) or {}
            ppin = pred2.get("pins", {}).get(comp) or {}
            if not isinstance(gpin, dict) or not isinstance(ppin, dict):
                st["gt_pins"] += len(gpin) if isinstance(gpin, dict) else 0
                continue
            prs = []
            for pk, info in ppin.items():
                if isinstance(info, dict):
                    pt = _point(info.get("point"))
                    if pt:
                        prs.append((pt, info.get("pinname")))
            for pk, g in gpin.items():
                if not isinstance(g, dict):
                    continue
                st["gt_pins"] += 1
                a = _point(g.get("point"))
                if not prs or a is None:
                    st["no_pred_pin_on_comp"] += 1
                    continue
                st["has_pred_pin_on_comp"] += 1
                pt, pn = min(prs, key=lambda t: math.hypot(t[0][0] - a[0],
                                                           t[0][1] - a[1]))
                d = math.hypot(pt[0] - a[0], pt[1] - a[1])
                if pn == g.get("pinname"):
                    st["name_eq"] += 1
                    if d <= 5.0:
                        st["tp_oracle"] += 1
                    else:
                        st["name_eq_far"] += 1
                        if len(ex) < examples:
                            ex.append((c.case_id, comp, pk, g.get("pinname"),
                                       round(d, 1), a, pt))
                else:
                    st["name_ne"] += 1
                    if d <= 5.0:
                        st["pos_ok_name_bad"] += 1
    print("case 数 %d（split=%s，out=%s）" % (len(cs), split, Path(out_dir).name))
    g = st["gt_pins"]
    print("GT IC 引脚 %d 根" % g)
    print("  该元件键下**我们一根引脚都没有**：%d（%.1f%%）" % (
        st["no_pred_pin_on_comp"], 100.0 * st["no_pred_pin_on_comp"] / max(1, g)))
    print("  有引脚可比：%d" % st["has_pred_pin_on_comp"])
    print("      取最近的一根后，名字逐字符相等：%d（占 GT 的 %.1f%%）" % (
        st["name_eq"], 100.0 * st["name_eq"] / max(1, g)))
    print("          其中 距离 ≤5px（= 上界 tp）：%d" % st["tp_oracle"])
    print("          其中 名字对但距离 >5px：%d" % st["name_eq_far"])
    print("      名字不等：%d（其中位置倒是对的 %d）" % (st["name_ne"], st["pos_ok_name_bad"]))
    print("  ⇒ 绑定问题全解决后的引脚 F1 上界 ≈ %.4f（tp/(tp+fp+fn) 的粗估）" % (
        2.0 * st["tp_oracle"] / max(1, 2 * st["tp_oracle"]
                                    + (g - st["tp_oracle"]) + (g - st["tp_oracle"]))
        if st["tp_oracle"] else 0.0))
    if ex:
        print("\n名字对但距离 >5px 的样本（名对/距远）：")
        for cid, comp, pk, gn, d, a, pt in ex:
            print("  %-8s %-8s %-8s gt名=%-12s 距离%6.1f  gt%s pr%s" % (
                cid, comp, pk, str(gn)[:12], d, a, pt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="200_train_cases")
    ap.add_argument("--out", default=str(ROOT / "work" / "_submit_out220r36"))
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args()
    return run(a.n, a.split, a.out, a.examples)


if __name__ == "__main__":
    raise SystemExit(main())
