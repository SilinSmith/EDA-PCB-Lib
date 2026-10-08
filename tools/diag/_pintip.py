# -*- coding: utf-8 -*-
"""_pintip.py — 引脚点几何体检：**tip vs base** 与 **邻脚错位** 两个假设的纯 CPU 判据。

背景（2026-10，见 docs\\COMPETITION_RULES.md §3 / DIAGNOSTICS 引脚节）：
  * 官方附录三说元件有 outerbbox/bodybbox，**pin tip 在外侧框边缘、base 连内侧框**；
  * 我们 `_pinpos.py` 量到超差引脚的误差中位数里，20~50 px 桶是 `dx +2.0, dy +29.0`
    ⇒ 两种解释：(H1) 我们发的是符号内侧的 base 点；(H2) 我们发的点其实是**隔壁脚**
    （号绑对了、几何绑错位）。

本脚本用「匹配上的 box 元件 + 名字相同的引脚」逐条量：
  sG = GT 点到 **GT bbox** 的有符号距离（>0 在框内、≈0 在框边缘、<0 在框外）
  sP = 我们的点到同一个 GT bbox 的有符号距离
  proj = (我们的点 − GT 点) 在「该 GT 点最近框边的**内法向**」上的投影（>0 = 我们更靠里）
  adj  = 我们的点是否落在**同元件另一个 GT 引脚点**的 5 px 内（H2 的直接证据）

用法（**从仓库根跑**）：
    python -X utf8 work\\tools\\diag\\_pintip.py --pred work\\_fnew
    python -X utf8 work\\tools\\diag\\_pintip.py --pred work\\_libpins_cpu_verify\\off
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES, _box, _point,  # noqa: E402
                                      assign_components, canon_type, pin_f1,
                                      pin_key_to_number, rewrite_keys)

CASES = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
_IN = {"l": (1.0, 0.0), "r": (-1.0, 0.0), "b": (0.0, 1.0), "t": (0.0, -1.0)}


def build_index(data_root):
    exact, digits = {}, {}
    for c in discover(data_root, SPLITS):
        if not c.json_path:
            continue
        exact.setdefault(c.case_id, c)
        d = re.sub(r"\D", "", c.case_id)
        if d:
            digits.setdefault(d, []).append(c)
    return exact, digits


def resolve(stem, exact, digits):
    if stem in exact:
        return exact[stem]
    d = re.sub(r"\D", "", stem) or stem
    cands = digits.get(d) or []
    if len(cands) == 1:
        return cands[0]
    pure = [c for c in cands if c.case_id.isdigit()]
    return pure[0] if len(pure) == 1 else None


def box_signed(pt, b):
    """>0 = 点在框内（值是到最近边的距离）；≈0 = 在框上；<0 = 在框外（值是到框的距离取负）。"""
    x, y = pt
    x1, y1, x2, y2 = b
    if x1 <= x <= x2 and y1 <= y <= y2:
        return min(x - x1, x2 - x, y - y1, y2 - y)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return -math.hypot(dx, dy)


def inward_at(pt, b):
    """返回 (内法向单位向量, 到最近边的距离, 边名) —— 以 pt 相对框 b 计。"""
    x, y = pt
    x1, y1, x2, y2 = b
    d = {"l": x - x1, "r": x2 - x, "b": y - y1, "t": y2 - y}
    k = min(d, key=lambda s: d[s])
    return _IN[k], d[k], k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="预测目录（含 <case>.json 或 <case>_Pred.json）")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--rows", type=int, default=30, help="最多打印多少行 >5px 的明细")
    a = ap.parse_args()
    preddir = pathlib.Path(a.pred)
    if not preddir.is_absolute():
        preddir = ROOT / preddir
    cfg = EvalConfig()
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    rows, miss_file = [], []
    agg = {"pairs": 0, "name_ok": 0, "dist_le5": 0, "buckets": {}, "adj": 0, "inward": 0,
           "sG": [], "sP": [], "cases": 0}

    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = resolve(cid, exact, digits)
        if case is None:
            miss_file.append(cid)
            continue
        f = preddir / ("%s.json" % cid)
        if not f.exists():
            f = preddir / ("%s_Pred.json" % cid)
        if not f.exists():
            miss_file.append(cid)
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        _f1, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)
        pf = pin_f1(gt, pred2, cfg)                       # 自检：应与 stage2_eval 一致
        box = {k for k, v in gt["components"].items()
               if canon_type((v or {}).get("type")) in IC_TYPES}
        agg["cases"] += 1
        n_le5 = n_bad = 0
        for comp in sorted(box):
            gb = _box((gt["components"].get(comp) or {}).get("bbox"))
            ggrp = (gt.get("pins") or {}).get(comp) or {}
            pgrp = (pred2.get("pins") or {}).get(comp) or {}
            gtpts = {}
            for pk, info in ggrp.items():
                p = _point((info or {}).get("point"))
                if p is not None:
                    gtpts[pin_key_to_number(pk)] = (p, (info or {}).get("pinname"))
            for pk, info in pgrp.items():
                n = pin_key_to_number(pk)
                if n not in gtpts:
                    continue
                p = _point((info or {}).get("point"))
                gpt, gname = gtpts[n]
                nm = (info or {}).get("pinname")
                if p is None or nm != gname:               # 只体检「名字对」的那些（号对已由 key 保证）
                    continue
                dist = math.hypot(p[0] - gpt[0], p[1] - gpt[1])
                agg["pairs"] += 1
                if dist <= cfg.pin_tol:
                    n_le5 += 1
                    agg["dist_le5"] += 1
                else:
                    n_bad += 1
                sG = sP = proj = tan = None
                side = ""
                if gb:
                    sG, sP = box_signed(gpt, gb), box_signed(p, gb)
                    inw, _de, side = inward_at(gpt, gb)
                    d = (p[0] - gpt[0], p[1] - gpt[1])
                    proj = d[0] * inw[0] + d[1] * inw[1]
                    tan = math.hypot(d[0] - proj * inw[0], d[1] - proj * inw[1])
                    agg["sG"].append(sG)
                    agg["sP"].append(sP)
                adj = None
                for n2, (g2, gname2) in sorted(gtpts.items()):
                    if n2 == n:
                        continue
                    d2 = math.hypot(p[0] - g2[0], p[1] - g2[1])
                    if d2 <= cfg.pin_tol and (adj is None or d2 < adj[3]):
                        adj = (n2, gname2, gname2 == nm, d2)
                if dist > cfg.pin_tol:
                    b = ("5-10" if dist <= 10 else "10-20" if dist <= 20
                         else "20-50" if dist <= 50 else ">50")
                    agg["buckets"][b] = agg["buckets"].get(b, 0) + 1
                    if adj:
                        agg["adj"] += 1
                    if proj is not None and dist > 0 and proj >= 0.7 * dist:
                        agg["inward"] += 1
                    rows.append((cid, comp, n, nm, dist, sG, sP, proj, tan, side, adj))
        print("case %-3s matched_name_ok=%d  <=5px %d  超差 %d   [自检 pin_f1 tp=%d fp=%d fn=%d %s]"
              % (cid, n_le5 + n_bad, n_le5, n_bad, pf.tp, pf.fp, pf.fn,
                 json.dumps(pf.detail, ensure_ascii=False) if pf.detail else ""))

    print("\n=== 超差明细（我们的点 − GT 点）===")
    print("%-4s %-6s %-5s %-10s %-6s %-7s %-7s %-7s %-6s %-4s %s"
          % ("case", "comp", "pin", "pinname", "|d|", "sG", "sP", "proj", "tan", "边", "adj(点落在别的GT脚上)"))
    for r in rows[: a.rows]:
        cid, comp, n, nm, dist, sG, sP, proj, tan, side, adj = r
        fmt = lambda v: ("%7.1f" % v) if v is not None else "      -"
        print("%-4s %-6s %-5s %-10s %6.1f %s %s %s %s %-4s %s"
              % (cid, comp, n, nm, dist, fmt(sG), fmt(sP), fmt(proj), fmt(tan), side or "-",
                 ("pin_%s(%s)%s |d|=%.1f" % (adj[0], adj[1], "同名" if adj[2] else "异名", adj[3]))
                 if adj else "-"))

    t = agg
    print("\n=== 汇总 ===")
    print("案例 %d 个；名字相同且号相同的引脚对 %d 个（<=5px %d，超差 %d）"
          % (t["cases"], t["pairs"], t["dist_le5"], t["pairs"] - t["dist_le5"]))
    print("超差分桶: %s" % json.dumps(t["buckets"], ensure_ascii=False))
    print("H2 邻脚证据（我们的点落在同元件另一个 GT 脚 5px 内）: %d" % t["adj"])
    print("H1 内法向证据（位移 ≥70%% 落在框的内法向上）: %d" % t["inward"])
    if t["sG"]:
        print("sG（GT 点到 GT bbox 的有符号距离）: 中位 %.1f 均值 %.1f  |sG|<=3 占 %d/%d"
              % (statistics.median(t["sG"]), statistics.mean(t["sG"]),
                 sum(1 for v in t["sG"] if abs(v) <= 3), len(t["sG"])))
        print("sP（我们的点到同一 GT bbox）      : 中位 %.1f 均值 %.1f  |sP|<=3 占 %d/%d"
              % (statistics.median(t["sP"]), statistics.mean(t["sP"]),
                 sum(1 for v in t["sP"] if abs(v) <= 3), len(t["sP"])))
        d = [p - g for p, g in zip(t["sP"], t["sG"])]
        print("sP − sG（我们比 GT 更靠里多少）    : 中位 %+.1f 均值 %+.1f   >0 占 %d/%d"
              % (statistics.median(d), statistics.mean(d), sum(1 for v in d if v > 0), len(d)))
    if miss_file:
        print("⚠️ 缺文件/配不上 GT 的案例: %s" % miss_file)


if __name__ == "__main__":
    main()
