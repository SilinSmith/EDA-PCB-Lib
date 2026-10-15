# -*- coding: utf-8 -*-
"""**引脚失分归因**：官方口径 `pin_f1` 只统计 GT 里 type 为 box 的元件
（`schematic_parser/metrics.py:441`），且配对键是 `(元件键, 引脚号)`
（`:453 pin_key_to_number`）——所以一颗 box 元件的键挂错，它的**全部**引脚都会
同时记成 `pin.missing`。本工具把每个 GT box 引脚的失分归到下面几类，用来判断
「引脚分低」到底是①元件没配上②我们没吐引脚③引脚号不对④引脚名读错⑤位置偏了：

    comp_unmatched   该 GT 元件没进 keymap（键与坐标都没配上）
    comp_no_group    配上了，但预测里这颗元件没有任何引脚组
    pin_absent       有引脚组，但没有这个引脚号
    num_shift        引脚号对不上，但两边的号集合互为平移（整体差一个常量）
    name_diff        号对上了，pinname 不同（读错或没读到）
    pos_far          号与名都对上了，点位差 > pin_tol
    ok               完全命中

用法：
    python -X utf8 work\\tools\\dev\\_pindiag.py --dir work\\_cnofix_mix --limit 40
    python -X utf8 work\\tools\\dev\\_pindiag.py --dir work\\_cnofix_real --limit 40 --show 12
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
DR = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
SPLITS = ("200_train_cases", "10_GTcase", "10GT")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from schematic_parser import metrics as M                      # noqa: E402
from schematic_parser.data import (IC_TYPES, normalise_doc,     # noqa: E402
                                   pin_key_to_number)
canon_type = M.canon_type


def find_gt(case: str):
    for sp in SPLITS:
        d = DR / sp / case
        if not d.is_dir():
            continue
        js = sorted(d.glob("%s*.json" % case))
        js = [p for p in js if "target_named" in p.name] or js
        for p in js:
            try:
                g = json.loads(p.read_text(encoding="utf-8"))
            except Exception:                                   # noqa: BLE001
                continue
            if g.get("components"):
                return sp, g
    return None, None


def flat(section: dict, box_comps):
    """与 metrics.pin_f1 的 flat() 同口径：只取 box 元件的引脚，键 = (元件, 引脚号)。"""
    out = {}
    for comp, group in section.items():
        if box_comps is not None and comp not in box_comps:
            continue
        if not isinstance(group, dict):
            continue
        for pk, info in group.items():
            out[(comp, pin_key_to_number(pk))] = (pk, info)
    return out


def point(info):
    if not isinstance(info, dict):
        return None
    p = info.get("point")
    if isinstance(p, dict):
        try:
            return (float(p["x"]), float(p["y"]))
        except Exception:                                       # noqa: BLE001
            return None
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try:
            return (float(p[0]), float(p[1]))
        except Exception:                                       # noqa: BLE001
            return None
    return None


def num_set(group: dict):
    out = set()
    for pk in group:
        n = pin_key_to_number(pk)
        if n:
            out.add(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(WORK / "_cnofix_mix"))
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pin-tol", type=float, default=5.0)
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条失分例子")
    a = ap.parse_args()

    raw = Path(a.cases_file).read_text(encoding="utf-8-sig")
    cases = [t for t in re.split(r"[,\s]+", raw) if t and "^" not in t]
    cases = [t for t in cases if re.fullmatch(r"[0-9A-Za-z_]+", t)]
    cases = cases[a.offset:a.offset + a.limit]

    cfg = M.EvalConfig()
    tally = collections.Counter()
    per_suite = collections.defaultdict(collections.Counter)
    numdep = collections.Counter()      # 引脚号字符串的长度分布（看 GT 用什么号）
    examples = []
    gt_pins_total = 0
    n_case = 0
    for case in cases:
        sp, gt_doc = find_gt(case)
        pred_p = Path(a.dir) / ("%s.json" % case)
        if gt_doc is None or not pred_p.exists():
            tally["缺产物/GT"] += 1
            continue
        n_case += 1
        gt = normalise_doc(gt_doc)
        pred = normalise_doc(json.loads(pred_p.read_text(encoding="utf-8")))
        _comp, keymap = M.assign_components(gt, pred, cfg, {})
        pred2 = M.rewrite_keys(pred, keymap)
        box = {k for k, v in gt["components"].items()
               if canon_type((v or {}).get("type")) in IC_TYPES}
        gf = flat(gt["pins"], box)
        pf = flat(pred2["pins"], None)      # 预测侧不过滤：要区分「没吐引脚」与「吐了但号不对」
        for (comp, num), (pk, g) in gf.items():
            gt_pins_total += 1
            numdep[len(str(num))] += 1
            tgt = keymap.get(comp, comp)
            group = pred2["pins"].get(tgt) or {}
            if comp not in keymap and not group:
                cause = "comp_unmatched"
            elif not isinstance(group, dict) or not group:
                cause = "comp_no_group"
            elif (tgt, num) not in pf:
                gs, ps = num_set(gt["pins"].get(comp, {})), num_set(group)
                numeric = gs and ps and not (gs & ps) \
                    and all(x.isdigit() for x in gs) and all(x.isdigit() for x in ps) \
                    and len(gs) == len(ps)
                cause = "num_shift" if numeric else "pin_absent"
            else:
                _ppk, p = pf[(tgt, num)]
                if not isinstance(p, dict):
                    cause = "malformed"
                elif p.get("pinname") != g.get("pinname"):
                    cause = "name_diff"
                else:
                    pa, pb = point(g), point(p)
                    if pa is None or pb is None:
                        cause = "no_point"
                    elif math.hypot(pa[0] - pb[0], pa[1] - pb[1]) > a.pin_tol:
                        cause = "pos_far"
                    else:
                        cause = "ok"
            tally[cause] += 1
            per_suite[sp or "?"][cause] += 1
            if cause != "ok" and len(examples) < a.show:
                examples.append((case, comp, num, cause,
                                 (g or {}).get("pinname"),
                                 (pf.get((tgt, num), (None, {}))[1] or {}).get("pinname"),
                                 len(gt["pins"].get(comp, {})), len(group)))

    print("== %s  (%d 张 case, %d 个 GT box 引脚) ==" % (a.dir, n_case, gt_pins_total))
    for k, v in tally.most_common():
        pct = 100.0 * v / gt_pins_total if gt_pins_total else 0.0
        print("  %-16s %6d  %5.1f%%" % (k, v, pct))
    print("-- 按 GT 套系 --")
    for sp, c in sorted(per_suite.items()):
        tot = sum(c.values())
        print("  %-18s n=%5d  ok %5.1f%%  主因: %s" % (
            sp, tot, 100.0 * c["ok"] / max(tot, 1),
            ", ".join("%s %d" % (k, v) for k, v in c.most_common(3) if k != "ok")))
    print("-- GT 引脚号长度分布 --", dict(sorted(numdep.items())))
    if examples:
        print("-- 失分例子（case, 元件, 号, 因, GT名, 我方名, GT引脚数, 我方引脚数）--")
        for e in examples:
            print("   ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
