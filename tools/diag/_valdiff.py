# -*- coding: utf-8 -*-
"""对比两版 stage2 预测的 component.value 改动：每一处改动是变好还是变坏。

    python work\\tools\\diag\\_valdiff.py --a work\\stage2_v17 --b work\\stage2_v18
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
_sys_path_fix = ROOT / "work" / "tools" / "eval"
sys.path.insert(0, str(_sys_path_fix))
from stage2_eval import build_index, resolve  # noqa: E402


def norm(s):
    return "" if s is None else str(s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--b", default=str(ROOT / "work" / "stage2_v18"))
    ap.add_argument("--samples", type=int, default=25)
    a = ap.parse_args()
    A, B = Path(a.a), Path(a.b)
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    good = bad = same = 0
    chg = Counter()
    samples_good, samples_bad = [], []
    for f in sorted(B.glob("*.json")):
        stem = f.stem
        ca = A / f.name
        if not ca.exists():
            continue
        case, how = resolve(stem, exact, digits)
        if case is None:
            continue
        gt = load_json(case.json_path)
        gtc = gt.get("components") or {}
        pa = json.loads(ca.read_text(encoding="utf-8")).get("components") or {}
        pb = json.loads(f.read_text(encoding="utf-8")).get("components") or {}
        for k, g in gtc.items():
            if k not in pa or k not in pb:
                continue
            ga = norm((pa[k] or {}).get("value"))
            gb = norm((pb[k] or {}).get("value"))
            if ga == gb:
                same += 1
                continue
            gv = norm(g.get("value"))
            oka = (ga == gv)
            okb = (gb == gv)
            if okb and not oka:
                good += 1
                if len(samples_good) < a.samples:
                    samples_good.append((stem, k, gv, ga, gb))
            elif oka and not okb:
                bad += 1
                if len(samples_bad) < a.samples:
                    samples_bad.append((stem, k, gv, ga, gb))
            chg[(oka, okb)] += 1
    print("值被改动的元件 %d 个（未动 %d）" % (good + bad + sum(v for k2, v in chg.items()
                                                          if k2 not in ((False, True), (True, False))), same))
    print("  改好了 %d   改坏了 %d" % (good, bad))
    print("  四种状态 (A对?,B对?)：", dict(chg))
    # 策略对比：只在管线值为空时才覆盖
    print("\n=== 策略对比 ===")
    st = Counter()
    cas = Counter()
    for f in sorted(B.glob("*.json")):
        ca = A / f.name
        if not ca.exists():
            continue
        case, how = resolve(f.stem, exact, digits)
        if case is None:
            continue
        gt = load_json(case.json_path)
        gtc = gt.get("components") or {}
        pa = json.loads(ca.read_text(encoding="utf-8")).get("components") or {}
        pb = json.loads(f.read_text(encoding="utf-8")).get("components") or {}
        for k, g in gtc.items():
            if k not in pa or k not in pb:
                continue
            ga = norm((pa[k] or {}).get("value"))
            gb = norm((pb[k] or {}).get("value"))
            gv = norm(g.get("value"))
            same_ci = ga.upper() == gb.upper()
            if ga != gb:
                cas["moved"] += 1
                cas["pure_case" if same_ci else "substantive"] += 1
                if not same_ci and ga and gb:
                    cas["both_nonempty_diff"] += 1
                if ga and not gb:
                    cas["cleared"] += 1
                if not ga and gb:
                    cas["filled"] += 1
            effs = {
                "base": ga,
                "all": gb,
                "only_empty": (gb if not ga else ga),
                "keep_ci": (ga if same_ci else gb),
                "empty_or_ci": (ga if (same_ci or ga) else gb),
            }
            for who, eff in effs.items():
                st[(who, eff == gv)] += 1
    print("值发生过变化的：", dict(cas))
    for who in ("base", "all", "only_empty", "keep_ci", "empty_or_ci"):
        ok = st[(who, True)]; tot = ok + st[(who, False)]
        print("  %-12s 值正确 %d / %d = %.4f" % (who, ok, tot, ok / max(1, tot)))
    print("\n改好的样例（case key | GT | A | B）：")
    for s in samples_good:
        print("   %s %s | %r | %r -> %r" % s)
    print("\n改坏的样例（case key | GT | A | B）：")
    for s in samples_bad:
        print("   %s %s | %r | %r -> %r" % s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
