"""引脚**位置**误差普查：同号引脚 GT 点 vs 我们的点到底差多少、往哪个方向差。

目的：判断 `pintrace` 的引脚桩是不是**系统性地短/长/偏**（`pin_tol = 5.0`，超一点就爆）。

用法（从 `Y:\\workspace\\EDA` 根跑）：
    python work\\tools\\diag\\_pinoff.py                       # 快测集 19 张，读 stage3_vlm
    python work\\tools\\diag\\_pinoff.py --pred work\\_submit_out220w\\runs\\stage2
    python work\\tools\\diag\\_pinoff.py --cases 1,2,9
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, SPLITS, discover,  # noqa: E402
                                   pin_key_to_number)
from schematic_parser.metrics import IC_TYPES, canon_type                # noqa: E402

FAST = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
        "0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]


def load_pred(d: Path, cid: str):
    for nm in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / nm
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8-sig"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="work/_submit_out220w/runs/stage3_vlm")
    ap.add_argument("--cases", default="")
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()
    cases = [c for c in a.cases.split(",") if c] or FAST
    pdir = ROOT / a.pred

    idx = {}
    for sp in SPLITS:
        for c in discover(DEFAULT_DATA_ROOT, splits=(sp,)):
            idx.setdefault(c.case_id, c)

    dists = []
    dx = Counter()
    dy = Counter()
    per_case = []
    for cid in cases:
        c = idx.get(cid)
        if c is None or not (pdir / ("%s.json" % cid)).exists() \
                and not (pdir / ("%s_Pred.json" % cid)).exists():
            continue
        gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pr = load_pred(pdir, cid)
        if pr is None:
            continue
        gkeys = {k for k, v in (gt.get("components") or {}).items()
                 if canon_type((v or {}).get("type")) in IC_TYPES}
        n_gt = n_pr = n_same = n_ok = 0
        for k in gkeys:
            if k not in (pr.get("components") or {}):
                continue
            gp = (gt.get("pins") or {}).get(k) or {}
            pp = (pr.get("pins") or {}).get(k) or {}
            pmap = {}
            for pk, pv in pp.items():
                pmap.setdefault(pin_key_to_number(pk), []).append(pv)
            for pk, pv in gp.items():
                num = pin_key_to_number(pk)
                n_gt += 1
                cand = pmap.get(num)
                if not cand:
                    continue
                n_same += 1
                ga = pv.get("point") or {}
                best = None
                for q in cand:
                    qa = q.get("point") or {}
                    d = math.hypot(ga.get("x", 0) - qa.get("x", 0),
                                   ga.get("y", 0) - qa.get("y", 0))
                    if best is None or d < best[0]:
                        best = (d, qa)
                if best is None:
                    continue
                d, qa = best
                dists.append(d)
                if d <= a.tol:
                    n_ok += 1
                dx[round((qa.get("x", 0) - ga.get("x", 0)) / 5.0) * 5] += 1
                dy[round((qa.get("y", 0) - ga.get("y", 0)) / 5.0) * 5] += 1
            n_pr += len(pp)
        if n_gt:
            per_case.append((cid, c.split, n_gt, n_same, n_ok))
    print("预测目录:", a.pred)
    print("%-8s %-18s %6s %8s %8s" % ("case", "split", "GTpin", "同号", "≤%.0fpx" % a.tol))
    for cid, sp, n_gt, n_same, n_ok in per_case:
        print("%-8s %-18s %6d %8d %8d" % (cid, sp, n_gt, n_same, n_ok))
    if dists:
        dists.sort()
        n = len(dists)
        print("---- 同号引脚点距离（n=%d）----" % n)
        for q in (0.5, 0.75, 0.9, 0.95, 1.0):
            print("   p%-5.0f = %7.1f px" % (q * 100, dists[min(n - 1, int(q * n))]))
        print("   ≤tol 命中率 = %.3f" % (sum(1 for d in dists if d <= a.tol) / n))
        for shift in (-20, -15, -10, -5, 0):
            pass
    print("---- 增量 dx（预测-GT，5px 分桶）----")
    print("  ", sorted(dx.items()))
    print("---- 增量 dy ----")
    print("  ", sorted(dy.items()))


if __name__ == "__main__":
    main()
