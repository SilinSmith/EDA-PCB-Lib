# -*- coding: utf-8 -*-
"""IC 引脚「点距」误差体检：PinF1 里 tp 要求 pinname 相等 **且** 点距 ≤ pin_tol(5 px)。

`_libopp.py` 量到：匹配上的 box/IC 里，有 58 个引脚「号对 + 名对」，但交付口径 PinF1 的 tp 只有
37 ⇒ 怀疑相当一部分是**坐标差过 5 px** 被判 position_mismatch。本脚本量这个：

  · 号能对上的引脚，点距分布（≤5 / ≤10 / ≤20 / >20 px）
  · 误差向量 (dx, dy) 的分布 —— 若是**系统性偏移**（外扩/内缩），就还有救
  · 按「名是否也对」分桶，看坐标与命名各占多少损失

用法：
    python -X utf8 work\\tools\\diag\\_pinpos.py --pred work\\_fnew --cases 1,2,3,4,5,7,8,9,10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.metrics import EvalConfig, assign_components, canon_type, rewrite_keys  # noqa: E402
from schematic_parser.data import IC_TYPES, pin_key_to_number  # noqa: E402

GT9 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
BUCKETS = ((5.0, "<=5 (tp)"), (10.0, "5~10"), (20.0, "10~20"), (50.0, "20~50"), (1e9, ">50"))


def bucket(d: float) -> str:
    for thr, name in BUCKETS:
        if d <= thr:
            return name
    return ">50"


def point_of(rec):
    p = (rec or {}).get("point") or {}
    return (float(p.get("x", 0.0)), float(p.get("y", 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--gt", default=str(GT9))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    cfg = EvalConfig()
    cfg.pin_tol = a.tol
    gtroot, predroot = Path(a.gt), Path(a.pred)
    name_ok = Counter()
    name_bad = Counter()
    dxy: dict = {k: [] for _t, k in BUCKETS}
    per_case = []

    for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
        gtf, pf = gtroot / c / ("%s.json" % c), predroot / ("%s.json" % c)
        if not (gtf.exists() and pf.exists()):
            continue
        gt = json.loads(gtf.read_text(encoding="utf-8"))
        pred = json.loads(pf.read_text(encoding="utf-8"))
        _f1, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)
        gpin, ppin = gt.get("pins", {}), pred2.get("pins", {})
        n_tp = n_pos = 0
        for k, rec in gt.get("components", {}).items():
            if canon_type(rec.get("type")) not in IC_TYPES or k not in gpin:
                continue
            if k not in pred2.get("components", {}):
                continue
            pmap = {pin_key_to_number(pk): pv for pk, pv in (ppin.get(k) or {}).items()}
            for pk, pv in gpin[k].items():
                num = pin_key_to_number(pk)
                if num not in pmap:
                    continue
                same_name = (pv.get("pinname") or "") == (pmap[num].get("pinname") or "")
                gx, gy = point_of(pv)
                px, py = point_of(pmap[num])
                d = math.dist((gx, gy), (px, py))
                b = bucket(d)
                (name_ok if same_name else name_bad)[b] += 1
                dxy[b].append((px - gx, py - gy))
                if same_name and d <= a.tol:
                    n_tp += 1
                elif same_name:
                    n_pos += 1
                    if a.verbose:
                        print("    [pos] %s %s.%s 误差 %.1f px  dx %.1f dy %.1f" % (c, k, num, d, px - gx, py - gy))
        per_case.append((c, n_tp, n_pos))
        print("  case %-3s tp %2d，名对但位置超差 %2d" % (c, n_tp, n_pos))

    print("\n===== 号能对上的引脚：名字 × 点距桶 =====")
    tot = sum(name_ok.values()) + sum(name_bad.values())
    for _t, b in BUCKETS:
        if tot:
            print("  %-9s 名对 %3d  名错 %3d" % (b, name_ok[b], name_bad[b]))
    print("  合计 %d（名对 %d / 名错 %d）" % (tot, sum(name_ok.values()), sum(name_bad.values())))
    print("\n===== 各桶的误差向量中位数（找系统性偏移）=====")
    for _t, b in BUCKETS:
        v = dxy[b]
        if v:
            mx = sorted(x for x, _y in v)[len(v) // 2]
            my = sorted(y for _x, y in v)[len(v) // 2]
            print("  %-9s n=%3d  中位 dx %+7.2f  dy %+7.2f" % (b, len(v), mx, my))


if __name__ == "__main__":
    main()
