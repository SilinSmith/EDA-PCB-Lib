"""量化「重复检测」对本方提交文档的影响 —— 纯 CPU，不动流水线。

背景：子代理实测 —— 497 个 FP 里 >=255 个（51.3%）不是错检，而是**同一颗器件被重复检测**
（已有另一个预测配对了那颗 GT，多出来的第 2 个框被记成 FP）。
本脚本直接问一个更狠的问题：**在我们交上去的 JSON 里，这种重复还剩多少、去重能拿多少分。**

用法：
    python work\\tools\\diag\\_dupscore.py [case ...]
默认走 19 张定点档；`--tol` 改判重的 corner 容差（默认 20，与官方 bbox_tol 同）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D                                        # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case                # noqa: E402

FAST = "1 2 3 4 5 7 8 9 10 0092 0048 0019 0024 0075 0086 0002 0001 0042 0103".split()
PRED_DIR = Path(r"Y:\workspace\EDA\work\_fusion_out220no4")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
TOL = 20.0
for a in sys.argv[1:]:
    if a.startswith("--tol"):
        TOL = float(a.split("=", 1)[1])


def cdev(a, b) -> float:
    return max(abs(float(a[i]) - float(b[i])) for i in range(4))


def dedup_components(doc: dict, tol: float):
    comps = doc.get("components") or {}
    keys = list(comps)
    keep, dropped = [], []
    for k in keys:
        c = comps[k]
        bb = c.get("bbox")
        if not bb:
            keep.append(k)
            continue
        dup_of = None
        for j in keep:
            cj = comps[j]
            if str(cj.get("type")) != str(c.get("type")):
                continue
            bj = cj.get("bbox")
            if not bj:
                continue
            if cdev(bb, bj) <= tol:
                dup_of = j
                break
        if dup_of is None:
            keep.append(k)
        else:
            dropped.append((k, dup_of))
    if not dropped:
        return doc, []
    out = dict(doc)
    out["components"] = {k: comps[k] for k in keep}
    pins = doc.get("pins") or {}
    out["pins"] = {k: pins[k] for k in keep if k in pins}
    return out, dropped


def dedup_pins(doc: dict, tol: float):
    """同一元件的同一点上若有多个引脚（同/异名），只留一个。"""
    pins = doc.get("pins") or {}
    out = {}
    ndrop = 0
    for k, plist in pins.items():
        if not isinstance(plist, dict):
            out[k] = plist
            continue
        keep = {}
        for pn, pv in plist.items():
            pt = (pv or {}).get("point") or {}
            dup = False
            for _k2, v2 in keep.items():
                p2 = (v2 or {}).get("point") or {}
                if abs(float(pt.get("x", 1e9)) - float(p2.get("x", 1e9))) <= tol and \
                   abs(float(pt.get("y", 1e9)) - float(p2.get("y", 1e9))) <= tol:
                    dup = True
                    break
            if dup:
                ndrop += 1
            else:
                keep[pn] = pv
        out[k] = keep
    if not ndrop:
        return doc, 0
    d = dict(doc)
    d["pins"] = out
    return d, ndrop


def main() -> None:
    cfg = EvalConfig()
    cases = args or FAST
    tot0 = tot1 = tot2 = 0.0
    n = 0
    print(f"判重容差 corner <= {TOL:.0f}px（官方 bbox_tol 同）")
    print(f"{'case':>6} {'基线':>7} {'去重元件':>8} {'再去重引脚':>10} {'丢元件':>6} {'丢引脚':>6}")
    for cid in cases:
        try:
            c = D.find_case(cid)
            gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
            doc = json.loads((PRED_DIR / f"{c.case_id}.json").read_text(encoding="utf-8"))
        except Exception as e:                                     # noqa: BLE001
            print(f"{cid:>6}  跳过: {type(e).__name__}: {e}")
            continue
        s0 = evaluate_case(gt, doc, c.case_id, cfg).score
        d1, dropped = dedup_components(doc, TOL)
        s1 = evaluate_case(gt, d1, c.case_id, cfg).score
        d2, npin = dedup_pins(d1, TOL)
        s2 = evaluate_case(gt, d2, c.case_id, cfg).score
        tot0 += s0
        tot1 += s1
        tot2 += s2
        n += 1
        print(f"{cid:>6} {s0:7.2f} {s1:8.2f} {s2:10.2f} {len(dropped):6d} {npin:6d}"
              + ("   " + ", ".join(f"{a}~{b}" for a, b in dropped[:6]) if dropped else ""))
    if n:
        print("-" * 60)
        print(f"均分 基线 {tot0/n:.3f} → 去重元件 {tot1/n:.3f} → 再去重引脚 {tot2/n:.3f}   "
              f"（净 {tot2/n - tot0/n:+.3f}，{n} 张）")


if __name__ == "__main__":
    main()
