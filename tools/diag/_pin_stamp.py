# -*- coding: utf-8 -*-
"""引脚「图章点位」可行性核算。

背景（见 work/reports 与 stage2/README）：
  官方 GT 的引脚点位**不是走线接点**，而是「元件 bbox 中心沿符号主轴 ±SPAN」的理想点
  （schematic_parser/stamp.py 把每个类固化成一个小图章）。实测 r/c/l/fuse/crystal_2pin
  的 pin_1 恒在 −20、pin_2 恒在 +20（纯度 1.00）；d/led 反号（纯度 0.68~0.90）。

本脚本回答两个问题：
  1) 我们的 bbox 被 snap 成了正方形（r/c = 20x20），所以**不能**用它判断符号轴向；
     那用什么判断？—— 用我们自己输出的两个引脚点的相对位置（attach 点）。
  2) 照「中心 ±20」合成点位后，与 GT 点位的距离分布是多少（现在是多少）。

用法：
  python work\\tools\\diag\\_pin_stamp.py [--pred work\\stage2_v5] [--spans]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402

TWO = ("r", "c", "l", "fuse", "crystal_2pin", "d", "led")
PERP = frozenset({"crystal_2pin"})   # 引脚轴与 bbox 长短边“垂直”的那个类
POS = frozenset({"d", "led"})        # pin_1 在 +SPAN 一侧的类
DEFAULT_SPAN = 20.0


def _pt(p):
    if isinstance(p, dict):
        p = (p.get("x"), p.get("y"))
    try:
        return (float(p[0]), float(p[1]))
    except Exception:
        return None


def _bucket(d: float) -> str:
    for lim in (5, 10, 20, 40):
        if d <= lim:
            return "<=%d" % lim
    return ">40"


def _axis_from_bbox(bbox) -> bool:
    """True = 轴向 y。注意我们的 bbox 被 snap 成正方形，这个方法基本失效。"""
    return (bbox[3] - bbox[1]) >= (bbox[2] - bbox[0])


def _axis_from_pins(points) -> bool | None:
    """用我们自己两个引脚点的相对位置判断轴向（True = y）。"""
    if len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points[0], points[1]
    return abs(y2 - y1) >= abs(x2 - x1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=r"work\stage2_v5")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spans", action="store_true", help="打印每个类实测的中心到引脚距离")
    args = ap.parse_args()

    root = Path(r"Y:\workspace\EDA")
    pred_dir = Path(args.pred)
    if not pred_dir.is_absolute():
        pred_dir = root / pred_dir
    data_root = Path(args.data_root) if args.data_root else Path(DEFAULT_DATA_ROOT)

    idx: dict[str, object] = {}
    for c in discover(data_root, SPLITS):
        if c.json_path:
            idx.setdefault(c.case_id, c)
            idx.setdefault(re.sub(r"\D", "", c.case_id) or c.case_id, c)

    axis_ok = collections.Counter()      # 轴向判断命中率（用 attach 点）
    axis_bbox = collections.Counter()    # 轴向判断命中率（用 bbox 长短边，做对照）
    got_new = collections.Counter()      # 合成点位 vs GT 点位的距离
    got_old = collections.Counter()      # 现在点位 vs GT 点位的距离
    per_type = collections.defaultdict(collections.Counter)
    spans = collections.defaultdict(collections.Counter)

    files = sorted(f for f in pred_dir.glob("*.json") if not f.name.startswith("_"))
    if args.limit:
        files = files[: args.limit]

    for f in files:
        case = idx.get(f.stem) or idx.get(re.sub(r"\D", "", f.stem))
        if case is None:
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gp_all = gt.get("pins") or {}
        pp_all = pred.get("pins") or {}
        pc_all = pred.get("components") or {}
        for comp, c in (gt.get("components") or {}).items():
            ty = c.get("type")
            if ty not in TWO:
                continue
            gb = c.get("bbox")
            g = gp_all.get(comp)
            if not gb or not isinstance(g, dict) or len(g) != 2:
                continue
            ours = pp_all.get(comp) or {}
            mine = [_pt((ours.get(k) or {}).get("point")) for k in ("pin_1", "pin_2")]
            mine = [p for p in mine if p is not None]
            gtpts = {k: _pt((g.get(k) or {}).get("point")) for k in ("pin_1", "pin_2")}
            if gtpts["pin_1"] is None or gtpts["pin_2"] is None:
                continue

            gx = (gb[0] + gb[2]) / 2.0
            gy = (gb[1] + gb[3]) / 2.0
            # GT 的真实轴向（由 GT 自己的两个点决定）—— 用来评估我们的判轴准不准
            true_y = abs(gtpts["pin_1"][1] - gtpts["pin_2"][1]) >= abs(gtpts["pin_1"][0] - gtpts["pin_2"][0])
            for name, fn in (("axis_ok", _axis_from_pins(mine)),
                             ("axis_bbox", _axis_from_bbox(gb))):
                if fn is None:
                    continue
                got = fn if name == "axis_bbox" else bool(fn)
                (axis_ok if name == "axis_ok" else axis_bbox)["hit" if got == true_y else "miss"] += 1

            # GT 的 SPAN 实测
            for pk in ("pin_1", "pin_2"):
                dx = gtpts[pk][0] - gx
                dy = gtpts[pk][1] - gy
                if abs(dy) > abs(dx):
                    spans[(ty, "y", pk)][round(abs(dy))] += 1
                else:
                    spans[(ty, "x", pk)][round(abs(dx))] += 1

            pb = pc_all.get(comp)
            if not pb or not pb.get("bbox"):
                continue
            cx = (pb["bbox"][0] + pb["bbox"][2]) / 2.0
            cy = (pb["bbox"][1] + pb["bbox"][3]) / 2.0
            # 也考虑「GT 轴向」与「我们 attach 点判定」两种；正号按类表
            for tag, use_y in (("via_attach", _axis_from_pins(mine)),):
                if use_y is None:
                    continue
                if ty in PERP:
                    use_y = not use_y
                s1 = 1.0 if ty in POS else -1.0
                if use_y:
                    targets = {"pin_1": (cx, cy + s1 * DEFAULT_SPAN), "pin_2": (cx, cy - s1 * DEFAULT_SPAN)}
                else:
                    targets = {"pin_1": (cx + s1 * DEFAULT_SPAN, cy), "pin_2": (cx - s1 * DEFAULT_SPAN, cy)}
                for pk in ("pin_1", "pin_2"):
                    t = targets[pk]
                    d = math.hypot(gtpts[pk][0] - t[0], gtpts[pk][1] - t[1])
                    got_new[_bucket(d)] += 1
                    per_type[ty][_bucket(d)] += 1
                    op = (ours.get(pk) or {}).get("point")
                    op = _pt(op)
                    if op is not None:
                        got_old[_bucket(math.hypot(gtpts[pk][0] - op[0], gtpts[pk][1] - op[1]))] += 1

    def _rate(c: collections.Counter) -> str:
        n = sum(c.values()) or 1
        return "n=%-6d <=5px %.3f   %s" % (n, c["<=5"] / n,
                                           {k: c[k] for k in ("<=5", "<=10", "<=20", "<=40", ">40")})

    print("== 轴向判断（用我们自己的两个引脚点）==")
    print("   attach点 :", dict(axis_ok), "命中率 %.3f" % (axis_ok["hit"] / max(1, axis_ok["hit"] + axis_ok["miss"])))
    print("   bbox长短 :", dict(axis_bbox), "命中率 %.3f" % (axis_bbox["hit"] / max(1, axis_bbox["hit"] + axis_bbox["miss"])))
    print("== 点位命中 ==")
    print("   现在    :", _rate(got_old))
    print("   合成    :", _rate(got_new))
    for t in TWO:
        if per_type.get(t):
            print("     %-14s %s" % (t, _rate(per_type[t])))
    if args.spans:
        print("== 实测 SPAN（GT 点相对 GT 中心）==")
        for k in sorted(spans):
            print("   %-14s %s %-6s %s" % (k[0], k[1], k[2], spans[k].most_common(3)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
