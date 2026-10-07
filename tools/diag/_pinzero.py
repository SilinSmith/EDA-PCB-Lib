# -*- coding: utf-8 -*-
"""`pin.missing` 的来源分解：是「整颗 IC 没配上」还是「配上了但引脚没有」。

`pin_box_only` 口径下 `pin_f1` 只看 GT 里 `canon_type ∈ IC_TYPES` 的元件，
所以每一条 GT IC 引脚的命运只有三种：
  A. 这颗 GT IC **根本没有对应的预测元件**（keymap 里没有）        -> 它的**全部**引脚都 missing
  B. 配上了，但**预测一侧这一段 `pins` 是空的**                    -> 同样全部 missing
  C. 配上了、也有引脚，只是号/名/坐标对不上                        -> extra / name / pos / missing

A 和 B 在官方 metrics 里**完全同分**（都是 fn），但修法截然不同：
A 要修位号读取或元件检出，B 要修引脚点检测。所以必须先分开数。

    python work\\tools\\diag\\_pinzero.py --pred work\\_s220_w16_1.5\\runs\\stage2
    python work\\tools\\diag\\_pinzero.py --pred work\\_gt10_vlmrun2\\final_results --top 20
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M          # noqa: E402
from schematic_parser.data import discover          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    cases = {c.case_id: c for c in discover()}
    want = [x.strip() for x in a.cases.split(",") if x.strip()] or None

    cfg = M.EvalConfig()
    IC = set(M.IC_TYPES)
    tot = collections.Counter()          # A/B/C 的引脚数
    ncomp = collections.Counter()        # A/B/C 的元件颗数
    noic = 0                             # GT 里一颗 IC 都没有的 case
    ic_cases = 0
    per_case = []
    by_pk = collections.defaultdict(collections.Counter)   # 预测键 -> 各桶
    gtype = collections.Counter()        # GT IC 的 type 分布

    for cid, c in sorted(cases.items()):
        if want and cid not in want:
            continue
        pj = pred_dir / ("%s.json" % cid)
        if not (pj.is_file() and c.json_path):
            continue
        gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pr = json.loads(pj.read_text(encoding="utf-8"))
        _f1, keymap = M.assign_components(gt, pr, cfg)
        g2p = {gk: pk for pk, gk in keymap.items()}

        gpins = gt.get("pins") or {}
        ppins = pr.get("pins") or {}
        gcomps = {k: v for k, v in (gt.get("components") or {}).items()
                  if str(v.get("type") or "").lower() in IC}
        if not gcomps:
            noic += 1
            continue
        ic_cases += 1
        cA = cB = cC = cM = 0
        for gk, gv in gcomps.items():
            gtype[str(gv.get("type"))] += 1
            gn = len(gpins.get(gk) or {})
            pk = g2p.get(gk)
            if not pk:
                tot["A_整颗IC没配上"] += gn
                ncomp["A_整颗IC没配上"] += 1
                cA += gn
                continue
            psec = ppins.get(pk) or {}
            if not psec:
                tot["B_配上了但引脚段为空"] += gn
                ncomp["B_配上了但引脚段为空"] += 1
                cB += gn
                by_pk[pk]["B_空"] += 1
                continue
            # C：两边都有引脚，逐号比
            gf, pf = {}, {}
            for k, v in (gpins.get(gk) or {}).items():
                gf[M.pin_key_to_number(k)] = v
            for k, v in psec.items():
                pf[M.pin_key_to_number(k)] = v
            miss = sum(1 for n in gf if n not in pf)
            cC += miss
            cM += len(gf)
            tot["C_有引脚但号对不上(missing)"] += miss
            for n, g in gf.items():
                p = pf.get(n)
                if p is None:
                    continue
                if p.get("pinname") != g.get("pinname"):
                    by_pk[pk]["name_mismatch"] += 1
                pa, ga = p.get("point") or {}, g.get("point") or {}
                try:
                    d = ((float(pa["x"]) - float(ga["x"])) ** 2
                         + (float(pa["y"]) - float(ga["y"])) ** 2) ** 0.5
                except Exception:  # noqa: BLE001
                    d = 0.0
                if d > cfg.pin_tol:
                    by_pk[pk]["position_mismatch"] += 1
        per_case.append((cid, cA, cB, cC, cM))

    gsum = sum(tot.values())
    print("有 GT IC 的 case: %d ｜ 一颗 IC 都没有的 case: %d" % (ic_cases, noic))
    print("GT IC 引脚总数（本口径下的 fn 上界）: %d" % gsum)
    print()
    for k in sorted(tot):
        print("  %-30s %6d  占 %5.1f%%   （%d 颗元件）"
              % (k, tot[k], 100.0 * tot[k] / max(gsum, 1), ncomp[k]))
    print()
    print("== 按 case 看 A 桶（整颗 IC 没配上）最惨的 %d 张 ==" % a.top)
    per_case.sort(key=lambda r: -r[1])
    for cid, cA, cB, cC, cM in per_case[:a.top]:
        print("   %-10s A=%4d  B=%4d  C=%4d   (GT 引脚共 %4d)" % (cid, cA, cB, cC, cM))
    print()
    print("== GT IC 的 type 分布 ==")
    print("  %s" % dict(gtype.most_common(10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
