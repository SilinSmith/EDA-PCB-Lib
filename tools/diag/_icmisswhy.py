# -*- coding: utf-8 -*-
"""没配上的 GT IC 到底是哪一类 —— 决定「补 IC」这条路值不值得走。

`_pinzero.py` 已量到 220 张里 **63% 的引脚 fn 来自「整颗 GT IC 没配上」**（629 颗、
5210 个 GT 引脚）。本脚本把这 629 颗按**可恢复性**分成四类：

    R0  我们连重叠的元件都没有            -> 检测漏了（`stage1\\bigic.py` 的活）
    R1  GT 键是 `∅NNN`，我们也有重叠元件却仍没配上 -> 几何/类型门没过（融合层可修）
    R2  GT 键是**真位号**（`U1`），我们没发出这个键 -> 位号读不出（`_icread.py` 已证多半不可恢复）
    R3  其它（有重叠、键也被别人占了等）

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_icmisswhy.py --pred work\\_submit_out220r36\\runs\\stage2
    python work\\tools\\diag\\_icmisswhy.py --pred ... --split 200_train_cases
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import (DEFAULT_DATA_ROOT, SPLITS, discover,  # noqa: E402
                                   load_json)
from schematic_parser.metrics import (EvalConfig, IC_TYPES, assign_components,  # noqa: E402
                                      canon_type, is_unkeyed)
import stage2_eval as SE  # noqa: E402


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _overlap(a, b):
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_submit_out220r36"
                                          / "runs" / "stage2"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]
    cnt = collections.Counter()
    pins = collections.Counter()
    per_case = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None:
            continue
        if a.split and case.split != a.split:
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        _f1, km = assign_components(gt, pr, cfg, {})
        matched = set(km.values())
        gcomps = gt.get("components") or {}
        pcomps = pr.get("components") or {}
        # 我们的元件里没被键匹配占用的（可用来坐标匹配）
        used_p = set(km.keys())
        free_p = [(k, v) for k, v in pcomps.items() if k not in used_p]
        for gk, gv in gcomps.items():
            if canon_type((gv or {}).get("type")) not in IC_TYPES:
                continue
            if gk in matched:
                continue
            gp = len((gt.get("pins") or {}).get(gk) or {})
            gb = gv.get("bbox") or [0, 0, 0, 0]
            best, bestk = 0.0, None
            for pk, pv in free_p:
                ov = _overlap(gb, pv.get("bbox") or [0, 0, 0, 0])
                if ov > best:
                    best, bestk = ov, pk
            if bestk is None:
                cls = "R0_无重叠元件"
            elif is_unkeyed(gk):
                cls = "R1_GT是∅且我们有重叠"
            elif not is_unkeyed(gk) and str(gk) in pcomps:
                cls = "R3_键撞车"
            else:
                cls = "R2_GT真位号读不出"
            cnt[cls] += 1
            pins[cls] += gp
            per_case[case.case_id][cls] += gp
            if len(samples[cls]) < 8:
                samples[cls].append((case.case_id, gk, canon_type(gv.get("type")),
                                     gp, bestk, round(best, 1)))

    tot_c = sum(cnt.values())
    tot_p = sum(pins.values())
    print("没配上的 GT IC 元件 %d 颗、携带 GT 引脚 %d 个" % (tot_c, tot_p))
    for k in sorted(cnt):
        print("  %-22s 元件 %5d (%5.1f%%)   引脚 %6d (%5.1f%%)"
              % (k, cnt[k], 100.0 * cnt[k] / max(1, tot_c),
                 pins[k], 100.0 * pins[k] / max(1, tot_p)))
    for k in sorted(samples):
        print("\n%s 样例（case, GT键, 类型, GT引脚数, 最近我方空闲键, 重叠面积）：" % k)
        for s in samples[k]:
            print("   %s" % (s,))
    worst = sorted(per_case.items(), key=lambda kv: -sum(kv[1].values()))[:a.top]
    print("\n== 按 case（缺的 GT 引脚数）==")
    for cid, cc in worst:
        print("   %s  %s" % (cid, dict(cc)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
