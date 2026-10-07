# -*- coding: utf-8 -*-
"""量一个事：**GT 的 `box`（大 IC）有多少是我们整颗没输出的。**

`pin_f1` 只统计 GT 里 `IC_TYPES` 的那些元件的引脚。只要我们对不上那颗 IC 的**键**，
它下面所有引脚就全变成 `pin.missing` ⇒ 那一张的引脚分（权重 0.25）直接归零。

这就是「某个小分爆 0」的主因，所以先把这个盘子量清楚。

用法：
    python work\tools\diag\_boxmiss.py [--pred work\\_submit_out220z\\final_results]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import IC_TYPES, EvalConfig, canon_type  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def find_pred(pred_dir, case_id):
    for name in (case_id + "_Pred.json", case_id + ".json"):
        p = pred_dir / name
        if p.exists():
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="GT box 元件的漏检盘子")
    ap.add_argument("--pred", default=str(ROOT / "work" / "_submit_out220z" / "final_results"))
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    pred_dir = Path(a.pred)
    rows, tot = [], collections.Counter()
    lost_pin_cases = 0
    missed_pins = 0
    for c in discover(a.data_root, SPLITS):
        if not c.json_path:
            continue
        pp = find_pred(pred_dir, c.case_id)
        if pp is None:
            continue
        gt = load_json(Path(c.json_path))
        pr = load_json(pp)
        gcomp = gt.get("components") or {}
        pkeys = set((pr.get("components") or {}).keys())
        boxes = [(k, v) for k, v in gcomp.items() if canon_type(v.get("type")) in IC_TYPES]
        if not boxes:
            continue
        hit = [k for k, _ in boxes if k in pkeys]
        miss = [k for k, _ in boxes if k not in pkeys]
        gpins = gt.get("pins") or {}
        mp = sum(len(gpins.get(k) or {}) for k in miss)
        rows.append((c.split, c.case_id, len(boxes), len(hit), len(miss), mp))
        tot["box"] += len(boxes)
        tot["hit"] += len(hit)
        tot["miss"] += len(miss)
        tot["miss_pins"] += mp
        if miss:
            lost_pin_cases += 1
        missed_pins += mp
        if a.limit and len(rows) >= a.limit:
            break

    rows.sort(key=lambda r: -r[5])
    print("有 GT box(IC_TYPES=%s) 的图：%d 张" % (sorted(IC_TYPES), len(rows)))
    print("GT box 元件 %d 个：键对上了 %d（%.3f）｜**整颗没输出 %d（%.3f）**"
          % (tot["box"], tot["hit"], tot["hit"] / max(1, tot["box"]),
             tot["miss"], tot["miss"] / max(1, tot["box"])))
    print("**这些漏掉的 box 一共带走 %d 个 GT 引脚**（平均 %.1f 个/张图）"
          % (tot["miss_pins"], tot["miss_pins"] / max(1, len(rows))))
    print("至少漏一颗 box 的图：%d / %d = %.3f" % (lost_pin_cases, len(rows),
                                                lost_pin_cases / max(1, len(rows))))
    print()
    print("漏得最狠的 25 张（case｜GT box｜对上｜漏｜带走的 GT 引脚数）：")
    for sp, cid, nb, nh, nm, mp in rows[:25]:
        print("  %-9s %-16s  %2d  %2d  %2d   %4d" % (cid, sp, nb, nh, nm, mp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
