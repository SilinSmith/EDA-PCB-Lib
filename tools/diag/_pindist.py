# -*- coding: utf-8 -*-
"""引脚**位置**与**名字**的失分归因：GT 的 `point` 到底指哪里？我们给的又是什么？

`metrics.pin_f1` 的判据（逐字）：
    if p.get('pinname') != g.get('pinname'):  → name_mismatch
    if hypot(gt.point, pred.point) > 5.0:     → position_mismatch
其中 GT 的 `pins[comp]['pin_N']['point']` = **引脚名那段文字的位置**。
本脚本对「元件键已配对（走 assign→rewrite 正式链）」的引脚逐根量：
  * 名字是否逐字符相等；
  * 坐标距离落在哪个区间（5 / 15 / 30 / 60 / >60 px）。
还要把 pred 侧 `pinname` 的取值长什么样打出来（我们是不是直接用引脚号当名字）。

用法：
    python work\\tools\\diag\\_pindist.py --n 40
    python work\\tools\\diag\\_pindist.py --n 40 --split 10GT
    python work\\tools\\diag\\_pindist.py --n 40 --samples 12
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES, _point, canon_type,  # noqa: E402
                                      assign_components, normalise_doc, pin_key_to_number,
                                      rewrite_keys)


def flat(section, box):
    out = {}
    for comp, group in (section or {}).items():
        if comp not in box or not isinstance(group, dict):
            continue
        for pk, info in group.items():
            out[(comp, pin_key_to_number(pk))] = info
    return out


def run(n, split, out_dir, samples):
    cfg = EvalConfig()
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == split][:n]
    buckets = Counter()
    namestat = Counter()
    dist_all = []
    ex = []
    n_case = 0
    for c in cs:
        praw = None
        for nm in (c.case_id + "_Pred.json", c.case_id + ".json"):
            q = Path(out_dir) / "final_results" / nm
            if q.exists():
                praw = load_json(q)
                break
        if praw is None:
            continue
        n_case += 1
        gt = normalise_doc(load_json(Path(c.json_path)))
        pred = normalise_doc(praw)
        _comp, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)
        box = {k for k, v in (gt.get("components") or {}).items()
               if canon_type((v or {}).get("type")) in IC_TYPES}
        gf, pf = flat(gt.get("pins"), box), flat(pred2.get("pins"), box)
        for key in set(gf) & set(pf):
            g, p = gf[key], pf[key]
            if not isinstance(g, dict) or not isinstance(p, dict):
                continue
            neq = p.get("pinname") == g.get("pinname")
            a, b = _point(g.get("point")), _point(p.get("point"))
            d = math.hypot(a[0] - b[0], a[1] - b[1]) if (a and b) else float("inf")
            dist_all.append(d)
            namestat["name_eq" if neq else "name_ne"] += 1
            bkt = ("<=5" if d <= 5 else "5-15" if d <= 15 else "15-30" if d <= 30
                   else "30-60" if d <= 60 else ">60" if d < 1e9 else "无坐标")
            buckets[(bkt, "名对" if neq else "名错")] += 1
            if len(ex) < samples:
                ex.append((c.case_id, key, g.get("pinname"), p.get("pinname"),
                           round(d, 1), a, b))
    print("case 数 %d（split=%s）" % (n_case, split))
    print("两侧都存在的引脚对：%d 根" % sum(namestat.values()))
    print("  名字逐字符相等：%d（%.1f%%）" % (namestat["name_eq"],
                                       100.0 * namestat["name_eq"] / max(1, sum(namestat.values()))))
    print()
    print("%-8s %8s %8s" % ("距离区间", "名对", "名错"))
    for bkt in ("<=5", "5-15", "15-30", "30-60", ">60", "无坐标"):
        print("%-8s %8d %8d" % (bkt, buckets[(bkt, "名对")], buckets[(bkt, "名错")]))
    if dist_all:
        ds = sorted(x for x in dist_all if x < 1e9)
        if ds:
            print("距离分位：p10 %.1f  p50 %.1f  p90 %.1f  max %.1f" % (
                ds[len(ds) // 10], ds[len(ds) // 2], ds[int(len(ds) * 0.9)], ds[-1]))
    print()
    print("%-8s %-14s %-14s %-14s %6s" % ("case", "键", "GT名", "PR名", "距离"))
    for cid, key, gn, pn, d, a, b in ex:
        print("%-8s %-14s %-14s %-14s %6.1f   gt%s pr%s" % (
            cid, "%s/%s" % key, str(gn)[:14], str(pn)[:14], d, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="200_train_cases")
    ap.add_argument("--out", default=str(ROOT / "work" / "_submit_out220r36"))
    ap.add_argument("--samples", type=int, default=10)
    a = ap.parse_args()
    return run(a.n, a.split, a.out, a.samples)


if __name__ == "__main__":
    raise SystemExit(main())
