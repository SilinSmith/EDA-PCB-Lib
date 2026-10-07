# -*- coding: utf-8 -*-
"""引脚窗口扫描：**在 GT 的引脚点上**读号/名，直接量「窗口几何」这一层的能力上限。

为什么单独做这一层：`_pinwhy.py` 已证明 10GT 上引脚爆 0 的主因是 `name_mismatch`
（判 55 条占 41 条），而引脚点其实有 83/94、引脚名一个都不为空。所以问题出在
**裁片窗口**（`enrich.adaptive_window` + `crop_dir`）与模型的配合上，不在检测上。
这里用 GT 的点当输入，把「点检测误差」这一项彻底拿掉，只看窗口怎么选最准。

用法::

    python work\\tools\\diag\\_pinsweep.py --cases 1,3,4,8,10
    python work\\tools\\diag\\_pinsweep.py --cases 8 --grid            # 扫 inner x cross
    python work\\tools\\diag\\_pinsweep.py --cases 8 --dump            # 存网格图

口径与官方 `pin_f1` 一致：**字符串严格相等**（`VIn` != `Vin`）。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # diag -> tools -> work -> EDA
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from vlm import enrich                      # noqa: E402
from stage2.text import imread_unicode      # noqa: E402
from stage2_eval import build_index, resolve, load_json  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT  # noqa: E402
from schematic_parser.metrics import (      # noqa: E402
    EvalConfig, IC_TYPES, assign_components, canon_type, normalise_doc,
    pin_key_to_number,
)


def gt_ic_pins(case, keymap):
    """返回 [(case_id, 元件键, bbox, [(引脚键, 号, 名, 点xy图像坐标), ...]), ...]

    只取「GT 是 IC 类型 且 键已被预测配上」的元件（`keymap` 方向是 预测键->GT键）。
    """
    gt = normalise_doc(load_json(case.json_path))
    H = None
    out = []
    matched = set(keymap.values())
    for k, v in (gt.get("components") or {}).items():
        if canon_type(v.get("type")) not in IC_TYPES or k not in matched:
            continue
        pins = (gt.get("pins") or {}).get(k) or {}
        items = []
        for pk, info in pins.items():
            if not isinstance(info, dict):
                continue
            pt = info.get("point") or {}
            if not isinstance(pt, dict):
                continue
            try:
                x, y = float(pt["x"]), float(pt["y"])
            except Exception:  # noqa: BLE001
                continue
            items.append((pk, str(pin_key_to_number(pk) or ""), str(info.get("pinname") or ""), (x, y)))
        if items:
            out.append((k, list(v.get("bbox") or []), items))
    return out


def read_at(bgr, bbox, items, opts, H, base):
    """按给定窗口基参读这批引脚，返回 {(号, 名): n} 与逐条明细。"""
    inner, cross, outer, ref = base
    win = enrich.adaptive_window(bbox, inner, cross, outer, ref)
    ctr = ((bbox[0] + bbox[2]) / 2.0, H - (bbox[1] + bbox[3]) / 2.0)
    pts = [(x, H - y) for (_pk, _n, _nm, (x, y)) in items]
    stat = {"grid": 0}
    ans = enrich.read_pins(bgr, pts, ctr, win, opts, stat)
    rows = []
    for i, (pk, num_g, name_g, _pt) in enumerate(items):
        num_p, name_p = ans.get(i, ("", ""))
        rows.append({
            "key": pk, "gt_num": num_g, "gt_name": name_g,
            "our_num": (num_p or "").strip(), "our_name": (name_p or "").strip(),
            "num_ok": (num_p or "").strip() == num_g,
            "name_ok": (name_p or "").strip() == name_g,
        })
    return rows, win, stat["grid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,3,4,8,10")
    ap.add_argument("--pred", default=str(ROOT / "work" / "_submit_out220z" / "final_results"))
    ap.add_argument("--inner", type=float, default=None)
    ap.add_argument("--cross", type=float, default=None)
    ap.add_argument("--outer", type=float, default=None)
    ap.add_argument("--ref", type=float, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--side", type=int, default=None)
    ap.add_argument("--mr", type=int, default=None)
    ap.add_argument("--grid", action="store_true", help="扫 inner x cross")
    ap.add_argument("--inners", default="", help="只扫这些 inner（逗号分隔），cross 用 --cross")
    ap.add_argument("--dump", action="store_true", help="存一版网格图到 work\\preview")
    ap.add_argument("--show", type=int, default=12, help="每张最多打印多少条错例")
    a = ap.parse_args()

    import cv2
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    opts = dict(enrich.DEFAULTS)
    if a.cols is not None:
        opts["cols"] = a.cols
    if a.side is not None:
        opts["side"] = a.side
    if a.mr is not None:
        opts["mr"] = a.mr
    base = (a.inner or opts["inner"], a.cross or opts["cross"],
            a.outer or opts["outer"], a.ref or opts["ref"])

    combos = [base]
    if a.inners:
        combos = [(float(x), base[1], base[2], base[3])
                  for x in a.inners.split(",") if x.strip()]
    elif a.grid:
        combos = [(i, c, base[2], base[3])
                  for i in (24.0, 40.0, 60.0, 78.0, 120.0)
                  for c in (6.0, 10.0, 16.0, 24.0, 36.0, 58.0)]

    pred_dir = Path(a.pred)
    agg = {}
    detail = {}
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        case, how = resolve(cid, exact, digits)
        if case is None:
            print("跳过（找不到）", cid); continue
        pf = pred_dir / ("%s_Pred.json" % case.case_id)
        if not pf.exists():
            pf = pred_dir / ("%s.json" % case.case_id)
        gt_doc = normalise_doc(load_json(case.json_path))
        pr_doc = normalise_doc(load_json(pf))
        _f1, keymap = assign_components(gt_doc, pr_doc, EvalConfig(), {})
        bgr = imread_unicode(case.png_path)
        H = float(bgr.shape[0])
        groups = gt_ic_pins(case, keymap)
        n_pin = sum(len(g[2]) for g in groups)
        print("== case %s（%s）IC 元件 %d 个 / GT 引脚 %d 个" % (case.case_id, how, len(groups), n_pin))
        for combo in combos:
            tot = {"n": 0, "num": 0, "name": 0, "both": 0, "calls": 0}
            rows_all = []
            for key, bbox, items in groups:
                rows, win, calls = read_at(bgr, bbox, items, opts, H, combo)
                rows_all += [(key, r) for r in rows]
                tot["calls"] += calls
                for r in rows:
                    tot["n"] += 1
                    tot["num"] += int(r["num_ok"])
                    tot["name"] += int(r["name_ok"])
                    tot["both"] += int(r["num_ok"] and r["name_ok"])
            agg.setdefault(combo, {"n": 0, "num": 0, "name": 0, "both": 0, "calls": 0})
            for k in ("n", "num", "name", "both", "calls"):
                agg[combo][k] += tot[k]
            if len(combos) == 1:
                detail[case.case_id] = rows_all
                for key, r in rows_all[:a.show]:
                    print("   %-6s %-6s GT %-10s/%-8s 我们 %-10s/%-8s  号%s 名%s"
                          % (key, r["key"], r["gt_num"], r["gt_name"],
                             r["our_num"], r["our_name"],
                             "O" if r["num_ok"] else "x", "O" if r["name_ok"] else "x"))
            print("   窗口 inner=%5.1f cross=%5.1f outer=%5.1f -> n=%3d 号 %.3f 名 %.3f 都 %.3f (calls %d)"
                  % (combo[0], combo[1], combo[2], tot["n"],
                     tot["num"] / max(1, tot["n"]), tot["name"] / max(1, tot["n"]),
                     tot["both"] / max(1, tot["n"]), tot["calls"]))

    if len(combos) > 1:
        print("\n== 汇总（按「名」降序）")
        rank = sorted(agg.items(), key=lambda kv: -kv[1]["name"])
        for combo, t in rank:
            print("   inner=%5.1f cross=%5.1f outer=%5.1f | n=%4d 号 %.3f 名 %.3f 都 %.3f calls=%d"
                  % (combo[0], combo[1], combo[2], t["n"],
                     t["num"] / max(1, t["n"]), t["name"] / max(1, t["n"]),
                     t["both"] / max(1, t["n"]), t["calls"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
