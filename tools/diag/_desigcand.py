# -*- coding: utf-8 -*-
"""「GT 真位号读不出」的那 579 颗 IC：**正确位号到底在不在候选表里？**

`_icmisswhy.py` 已量到：220 张里没配上的 GT IC 元件有 **88% 的引脚质量属于
「GT 键是真位号、我们读不出来」**（579 颗 / 4587 个 GT 引脚）。这些元件我们**几何上是
检出了的**（重叠面积很大），只是发出去的键是兜底 `pNNN` 或读错的别的位号。

本脚本问一个决定性问题：对每一颗这样的 GT IC，找到我们**重叠最多**的那颗元件，
翻 `_assigned` 里的 `desig_slots[pid]`（stage2 为该元件保留的全部位号候选，
按 score 降序），看：

    HIT   —— 候选表里**有** GT 键（= 读对了但没选它，选法可修）
    MISS  —— 候选表里**没有** GT 键（= 真读不出，只能靠先验/重扫）

分类汇总后就知道「修选法」这条路的收益上限。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_desigcand.py [--pred 目录] [--assigned 目录] [--split ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES, assign_components,  # noqa: E402
                                      canon_type, is_unkeyed)
import stage2_eval as SE  # noqa: E402


def _ov(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_submit_out220r36"
                                          / "runs" / "stage2"))
    ap.add_argument("--assigned", default=str(ROOT / "work" / "_submit_out220r36"
                                              / "runs" / "stage2" / "_assigned"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    A = Path(a.assigned)
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]
    cnt = collections.Counter()
    pins = collections.Counter()
    pos = collections.Counter()      # HIT 时正确候选的排名
    rows = []

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None:
            continue
        if a.split and case.split != a.split:
            continue
        ap_ = A / ("%s.json" % f.stem)
        if not ap_.exists():
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        asg = json.loads(ap_.read_text(encoding="utf-8"))
        ds = asg.get("desig_slots") or {}
        _f1, km = assign_components(gt, pr, cfg, {})
        matched = set(km.values())
        used_p = set(km.keys())
        pcomps = pr.get("components") or {}
        free_p = [(k, v) for k, v in pcomps.items() if k not in used_p]
        for gk, gv in (gt.get("components") or {}).items():
            if canon_type((gv or {}).get("type")) not in IC_TYPES:
                continue
            if gk in matched or is_unkeyed(gk):
                continue
            gb = gv.get("bbox") or [0, 0, 0, 0]
            best, bestk = 0.0, None
            for pk, pv in free_p:
                o = _ov(gb, pv.get("bbox") or [0, 0, 0, 0])
                if o > best:
                    best, bestk = o, pk
            gp = len((gt.get("pins") or {}).get(gk) or {})
            want = str(gk)
            cands = [str(c.get("text") or "") for c in (ds.get(bestk) or [])] \
                if bestk else []
            if want in cands:
                cnt["HIT_候选表里有"] += 1
                pins["HIT_候选表里有"] += gp
                pos[cands.index(want)] += 1
                if len(rows) < a.top:
                    rows.append((case.case_id, gk, bestk, gp,
                                 [c for c in cands[:5]],
                                 [round(float(c.get("score") or 0), 2)
                                  for c in (ds.get(bestk) or [])[:5]]))
            else:
                cnt["MISS_候选表里没有"] += 1
                pins["MISS_候选表里没有"] += gp

    tot_c, tot_p = sum(cnt.values()), sum(pins.values())
    print("R2 类（GT 真位号、我们没读出来）共 %d 颗 / %d 个 GT 引脚" % (tot_c, tot_p))
    for k in sorted(cnt):
        print("  %-20s 元件 %5d (%5.1f%%)  引脚 %6d (%5.1f%%)"
              % (k, cnt[k], 100.0 * cnt[k] / max(1, tot_c),
                 pins[k], 100.0 * pins[k] / max(1, tot_p)))
    print("\nHIT 时正确候选在候选表里的排名分布（0 = 已经排第一）：%s"
          % dict(sorted(pos.items())))
    print("\n样例（case, GT键, 我方元件, GT引脚, 候选前5, 对应分数）：")
    for r in rows:
        print("   %s" % (r,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
