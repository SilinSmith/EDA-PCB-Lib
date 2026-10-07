# -*- coding: utf-8 -*-
"""引脚失配归因：把 `pin_f1` 的损失拆成「键丢 / 号丢 / 名错 / 点错」。

`metrics.py:255` 的 `pin_f1` 把 `pins` 摊成 `(元件键, 引脚号)` 的扁平字典，
逐键比 `pinname`（**逐字符相等**）与 `point`（欧氏距离 <= `pin_tol`，默认 5px）。

所以每一笔丢分必然落在下面五类之一：

    key_miss    GT 有这颗元件，我们压根没输出这个键   -> 整颗元件的引脚全丢
    num_miss    键对上了，但这个引脚号我们没输出
    name_bad    号也对上了，pinname 不同
    point_bad   名也对上了，点位差 > tol
    ok          完全命中
    pred_extra  我们输出的引脚，GT 里没有对应键      -> pin.extra

用法：
    python work\\tools\\diag\\_pin_why.py [--pred work\\stage2_v5] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (  # noqa: E402
    DEFAULT_DATA_ROOT, SPLITS, discover, load_json, pin_key_to_number,
)


def gt_index(data_root):
    idx = {}
    for c in discover(data_root, SPLITS):
        if not c.json_path:
            continue
        idx.setdefault(c.case_id, c)
        idx.setdefault(re.sub(r"\D", "", c.case_id) or c.case_id, c)
    return idx


def flat(section):
    out = {}
    for comp, group in (section or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, info in group.items():
            out[(comp, pin_key_to_number(pk))] = info
    return out


def _point(p):
    if isinstance(p, dict):
        p = (p.get("x"), p.get("y"))
    if not isinstance(p, (list, tuple)) or len(p) < 2:
        return None
    try:
        return (float(p[0]), float(p[1]))
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="引脚失配归因")
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v5"))
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--pin-tol", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args(argv)

    idx = gt_index(args.data_root)
    files = [f for f in sorted(Path(args.pred).glob("*.json"))
             if not f.name.startswith("_")]
    if args.limit:
        files = files[:args.limit]

    cnt = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    gt_type_of = {}
    name_pairs = collections.Counter()
    samples = collections.defaultdict(list)

    for f in files:
        case = idx.get(f.stem) or idx.get(re.sub(r"\D", "", f.stem))
        if case is None:
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gc = gt.get("components") or {}
        for k, v in gc.items():
            gt_type_of[k] = (v or {}).get("type")
        gf, pf = flat(gt.get("pins")), flat(pred.get("pins"))
        pcomp = set(pred.get("components") or {})

        for (comp, num), g in gf.items():
            ty = gt_type_of.get(comp) or "?"
            p = pf.get((comp, num))
            if comp not in pcomp:
                cat = "key_miss"
            elif p is None:
                cat = "num_miss"
            elif not isinstance(p, dict):
                cat = "malformed"
            elif p.get("pinname") != g.get("pinname"):
                cat = "name_bad"
                name_pairs[(str(g.get("pinname"))[:24], str(p.get("pinname"))[:24])] += 1
            else:
                a, b = _point(g.get("point")), _point(p.get("point"))
                if a is None or b is None:
                    cat = "point_bad"
                elif math.hypot(a[0] - b[0], a[1] - b[1]) > args.pin_tol:
                    cat = "point_bad"
                else:
                    cat = "ok"
            cnt[cat] += 1
            by_type[ty][cat] += 1
            if cat != "ok" and len(samples[cat]) < args.samples:
                samples[cat].append("%s %s.%s gt_name=%r gt_pt=%r pred=%r"
                                    % (f.stem, comp, num, g.get("pinname"),
                                       g.get("point"), p))

        for (comp, num) in pf:
            if (comp, num) not in gf:
                cnt["pred_extra"] += 1
                by_type[gt_type_of.get(comp) or "?"]["pred_extra"] += 1

    tot_gt = sum(v for k, v in cnt.items() if k != "pred_extra")
    print("预测目录 %s   配对 %d 张   GT 引脚 %d" % (args.pred, len(files), tot_gt))
    print("-" * 68)
    for k in ("ok", "key_miss", "num_miss", "name_bad", "point_bad",
              "malformed", "pred_extra"):
        n = cnt.get(k, 0)
        print("  %-10s %-8d %.4f" % (k, n, n / max(1, tot_gt)))

    print("-" * 68)
    print("按 GT 元件类型（只看 ok 占比）：")
    rows = []
    for ty, c in by_type.items():
        t = sum(v for k, v in c.items() if k != "pred_extra")
        if t < 20:
            continue
        rows.append((t, ty, c))
    for t, ty, c in sorted(rows, reverse=True)[:16]:
        print("  %-16s n=%-6d ok %-6d %.3f | key_miss %-6d num_miss %-6d "
              "name_bad %-6d point_bad %-5d extra %d"
              % (ty, t, c.get("ok", 0), c.get("ok", 0) / t, c.get("key_miss", 0),
                 c.get("num_miss", 0), c.get("name_bad", 0), c.get("point_bad", 0),
                 c.get("pred_extra", 0)))

    if name_pairs:
        print("-" * 68)
        print("名错最多的 (gt_name, pred_name)：")
        for (g, p), n in name_pairs.most_common(15):
            print("  %-26s -> %-26s %d" % (g, p, n))

    for cat in ("num_miss", "name_bad", "point_bad"):
        if samples.get(cat):
            print("-" * 68)
            print("%s 样例：" % cat)
            for s in samples[cat]:
                print("  " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
