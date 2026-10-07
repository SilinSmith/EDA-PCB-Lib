# -*- coding: utf-8 -*-
"""_keyfix.py —— 「检测对了、类型对了、中心 <=20px，只是 key 对不上」的那批元件，
把我们的预测键逐条归类，量化「修键」到底能不能修、怎么修。

用法：
    python work\\tools\\diag\\_keyfix.py --pred work\\_submit_out220x\\final_results
    python work\\tools\\diag\\_keyfix.py --pred work\\_v3w\\final_results --limit 40

分桶（只看 GT 带名、且我们有一个同类型、中心 <=20px 的未配对预测的那种）：
    P  我们输出的是兜底键 pNNN（压根没读出位号）
    W  我们输出的是另一个「合法位号形状」的键（读错成邻居/别的元件）
    G  我们输出的键其实等于另一颗 GT 的键（串位）
    X  其它（模型/库名/纯数字…）
另外统计：我们这个错的键，在 GT 里到底存不存在（不存在 = 白造一个键）。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # Y:\workspace\EDA
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import stage2_eval as SE                                     # noqa: E402
from schematic_parser import metrics as M                    # noqa: E402

DESIG_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")
FKEY_RE = re.compile(r"^p\d{3,4}$")
UNKEYED_RE = re.compile(r"^[∅\u2205]")


def is_unkeyed(k: str) -> bool:
    return bool(UNKEYED_RE.match(str(k))) or str(k).startswith("\u2205")


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def buckets(gt_path: Path, pred_path: Path, dist_max: float = 20.0):
    g = json.loads(gt_path.read_text(encoding="utf-8"))
    p = json.loads(pred_path.read_text(encoding="utf-8"))
    gc, pc = g.get("components") or {}, p.get("components") or {}
    cfg = M.EvalConfig()
    _f1, keymap = M.assign_components(g, p, cfg, {})
    used_p = set(keymap.keys())
    gkeys = set(gc)
    out = []
    for gk, gv in gc.items():
        if gk in keymap.values():
            continue
        gb = gv.get("bbox")
        if not gb:
            continue
        if is_unkeyed(gk):
            continue
        gty = M.canon_type(gv.get("type"))
        best = None
        for pk, pv in pc.items():
            if pk in used_p or not pv.get("bbox"):
                continue
            if M.canon_type(pv.get("type")) != gty:
                continue
            pb = pv.get("bbox")
            # GT 与预测的 bbox 都是左下原点，直接比
            d = math.hypot(center(gb)[0] - center(pb)[0], center(gb)[1] - center(pb)[1])
            if d <= dist_max and (best is None or d < best[1]):
                best = (pk, d)
        if best is None:
            continue
        pk = best[0]
        if FKEY_RE.match(str(pk)):
            tag = "P"          # 兜底键 pNNN
        elif DESIG_RE.match(str(pk)) and pk in gkeys:
            tag = "G"          # 是另一颗 GT 元件的键（串位）
        elif DESIG_RE.match(str(pk)):
            tag = "W"          # 位号形状，但 GT 里没有这个键（读错/造键）
        else:
            tag = "X"
        out.append({
            "case": pred_path.stem.replace("_Pred", ""),
            "gt": gk, "gt_type": gty, "pred": pk, "tag": tag, "d": round(best[1], 1),
            "pred_in_gt": str(pk) in gkeys,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="final_results 目录")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dist", type=float, default=20.0)
    ap.add_argument("--show", type=int, default=12, help="每桶打印多少条样例")
    a = ap.parse_args()

    root = Path(a.pred)
    exact, digits = SE.build_index(SE.DEFAULT_DATA_ROOT)
    rows = []
    files = sorted(root.glob("*_Pred.json")) or sorted(root.glob("*.json"))
    for f in files:
        stem = f.stem.replace("_Pred", "")
        try:
            case, _how = SE.resolve(stem, exact, digits)
        except Exception:
            continue
        rows += buckets(Path(case.json_path), f, a.dist)
        if a.limit and len(rows) >= a.limit:
            break

    n = len(rows)
    print(f"预测目录 {root}")
    print(f"扫了 {len(files)} 张，命中「检测对/类型对/中心<={a.dist:g}px 但 key 对不上」 {n} 个\n")
    cnt = Counter(r["tag"] for r in rows)
    name = {"P": "我们输出兜底键 pNNN（根本没读出位号）",
            "W": "我们输出了位号形状、但 GT 里没有的键（读错/新造）",
            "G": "我们的键其实是另一颗 GT 元件的键（串位）",
            "X": "其它（库名/型号/数字…）"}
    for t in ("P", "W", "G", "X"):
        c = cnt.get(t, 0)
        print(f"  {t}  {c:5d}  {c/max(1,n):6.1%}  {name[t]}")
    print()
    print("按 GT 类型 × 桶：")
    by = Counter((r["gt_type"], r["tag"]) for r in rows)
    types = sorted({r['gt_type'] for r in rows}, key=lambda t: -sum(v for (tt, _), v in by.items() if tt == t))
    for t in types:
        tot = sum(v for (tt, _), v in by.items() if tt == t)
        cells = "  ".join(f"{k}={by.get((t,k),0)}" for k in ("P", "W", "G", "X") if by.get((t, k), 0))
        print(f"  {t:12s} {tot:5d}   {cells}")
    print()
    for t in ("P", "W", "G", "X"):
        ex = [r for r in rows if r["tag"] == t][: a.show]
        if not ex:
            continue
        print(f"--- {t} 样例 ---")
        for r in ex:
            print(f"   {r['case']:>10s}  GT {r['gt']:>10s} <- 我们 {r['pred']:>10s}   d={r['d']:>5.1f}  pred_in_gt={r['pred_in_gt']}")
    print()
    # 我们的键是否「凭空造出来的」
    fake = sum(1 for r in rows if not r["pred_in_gt"])
    print(f"其中「我们输出的键在 GT 里根本不存在」 = {fake} / {n} = {fake/max(1,n):.1%}")
    real = n - fake
    print(f"     「我们输出的键确实是本图另一颗 GT 元件的键」 = {real} / {n} = {real/max(1,n):.1%}")


if __name__ == "__main__":
    main()
