# -*- coding: utf-8 -*-
"""**引脚失分账本**：把 `pin_f1` 的 fn/fp 逐条拆到「缺号 / 多号 / 名字错 / 位置错」，
再把「缺号」拆成「stage2 压根没读出这个号」和「读出来了但没被 emit」。

口径与官方一致：`metrics.pin_f1` **只统计 GT 里 type 为 box 的元件**。

    python work/tools/diag/_pinledger.py --run work/_fast_out_reg9 --split 10GT
    python work/tools/diag/_pinledger.py --run work/_full220 --split 10GT --show 20
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser import metrics as M                                  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402


def _pt(d):
    if not isinstance(d, dict):
        return None
    try:
        return float(d["x"]), float(d["y"])
    except Exception:
        return None


def _find_assigned(run: Path, cid: str):
    for p in (run / "runs" / "stage2" / "_assigned" / (cid + ".json"),
              run / "stage2" / "_assigned" / (cid + ".json"),
              run.parent / "runs" / "stage2" / "_assigned" / (cid + ".json")):
        if p.exists():
            return load_json(p)
    return None


def _find_pred(run: Path, cid: str):
    for p in (run / "final_results" / ("%s_Pred.json" % cid),
              run / "final_results" / ("%s.json" % cid),
              run / ("%s_Pred.json" % cid)):
        if p.exists():
            return load_json(p)
    return None


NEAR_PX = 45.0


def _ctr(box):
    if not box or len(box) < 4:
        return None
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def _near(a, b):
    """a、b 是否「像同一个号的误读」：互为前后缀，或编辑距离 ≤2。"""
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a):
        return True
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 2


def _find_stage1(run: Path, cid: str):
    for p in (run / "runs" / "stage1" / ("%s.stage1.json" % cid),
              run / "stage1" / ("%s.stage1.json" % cid)):
        if p.exists():
            return load_json(p)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条缺号明细")
    a = ap.parse_args()
    run = Path(a.run)

    cfg = M.EvalConfig()
    tot = collections.Counter()
    bykind = collections.Counter()
    miss_why = collections.Counter()
    near_pairs = collections.Counter()
    shown = 0
    print("%-5s %-6s | %-24s | %s" % ("case", "pinF1", "tp/fp/fn", "缺号 多号 名错 位置错"))
    print("-" * 78)
    for c in [x for x in discover(DEFAULT_DATA_ROOT) if x.split == a.split][:a.n]:
        raw = _find_pred(run, c.case_id)
        if raw is None:
            print("%-5s  !! 没有 _Pred.json" % c.case_id)
            continue
        gt = M.normalise_doc(load_json(Path(c.json_path)))
        pred = M.normalise_doc(raw)
        det = {}
        _comp, keymap = M.assign_components(gt, pred, cfg, det)
        pred2 = M.rewrite_keys(pred, keymap)
        box = {k for k, v in (gt.get("components") or {}).items()
               if M.canon_type((v or {}).get("type")) in M.IC_TYPES}
        asg = _find_assigned(run, c.case_id)
        s1 = _find_stage1(run, c.case_id) or {}
        H = float((s1.get("image") or {}).get("h") or 0)
        # 我方 slot：designator 文本 -> {"号": 槽位}
        slots = {}
        if asg:
            for _cid, cc in (asg.get("comps") or {}).items():
                key = ((cc.get("designator") or {}).get("text") or "").strip()
                b = slots.setdefault(key, {})
                for _s, rec in (cc.get("pins") or {}).items():
                    num = (((rec or {}).get("number") or {}).get("text") or "").strip()
                    if num:
                        b.setdefault(num, rec)
        st = collections.Counter()
        for comp in box:
            g = {M.pin_key_to_number(pk): v for pk, v in
                 ((gt.get("pins") or {}).get(comp) or {}).items()}
            p = {M.pin_key_to_number(pk): v for pk, v in
                 ((pred2.get("pins") or {}).get(comp) or {}).items()}
            for num in sorted(set(g) - set(p)):
                st["missing"] += 1
                sl = (slots.get(comp) or {}).get(str(num))
                if sl is not None:
                    why = "读出了号但 assemble 没 emit"
                else:
                    # 没有这个号 —— 是「压根没读」还是「读成了近失号」？
                    # 用位置找最近的**有号**槽位，比它的号与 GT 号的编辑距离。
                    ga = _pt((g[num] or {}).get("point"))
                    best = None
                    for onum, orec in (slots.get(comp) or {}).items():
                        ob = (orec.get("number") or {}).get("box")
                        oc = _ctr(ob)
                        if not oc or not ga or not H:
                            continue
                        d = math.hypot(ga[0] - oc[0], ga[1] - (H - oc[1]))
                        if best is None or d < best[0]:
                            best = (d, onum)
                    if best and best[0] <= NEAR_PX:
                        if best[1] == str(num):
                            why = "同号槽位存在（异常）"
                        elif _near(str(best[1]), str(num)):
                            why = "误读/截断（附近 %dpx 有近失号 %s→%s）" % (best[0], num, best[1])
                            near_pairs["%s→%s" % (num, best[1])] += 1
                        else:
                            why = "附近 %dpx 的号是別人的（%s→%s）" % (best[0], num, best[1])
                    else:
                        why = "附近 %dpx 内没有任何号槽位" % NEAR_PX
                miss_why[why] += 1
                if a.show and shown < a.show:
                    shown += 1
                    print("   缺号 %-5s %-6s pin_%-4s gt名=%-10s → %s" %
                          (c.case_id, comp, num, (g[num] or {}).get("pinname"), why))
            for num in sorted(set(p) - set(g)):
                st["extra"] += 1
            for num in sorted(set(g) & set(p)):
                gi, pi = g[num] or {}, p[num] or {}
                if gi.get("pinname") != pi.get("pinname"):
                    st["name"] += 1
                    bykind["名 %s→%s" % (pi.get("pinname"), gi.get("pinname"))] += 1
                    continue
                ga, pb = _pt(gi.get("point")), _pt(pi.get("point"))
                if ga and pb and math.hypot(ga[0] - pb[0], ga[1] - pb[1]) > cfg.pin_tol:
                    st["pos"] += 1
                else:
                    st["tp"] += 1
        f1 = 2.0 * st["tp"] / max(1, 2 * st["tp"] + st["extra"] + st["name"] * 2 +
                                  st["pos"] * 2 + st["missing"])
        print("%-5s %-6.4f | %2d/%2d/%2d           | %3d %3d %3d %3d" %
              (c.case_id, f1, st["tp"], st["extra"] + st["name"] + st["pos"],
               st["missing"] + st["name"] + st["pos"],
               st["missing"], st["extra"], st["name"], st["pos"]))
        tot.update(st)
    print("-" * 78)
    print("合计：tp=%d 缺号=%d 多号=%d 名错=%d 位置错=%d" %
          (tot["tp"], tot["missing"], tot["extra"], tot["name"], tot["pos"]))
    if tot["missing"]:
        print()
        print("「缺号」的成因：")
        for k, v in miss_why.most_common():
            print("   %-46s %d（占缺号 %.0f%%）" % (k, v, 100.0 * v / tot["missing"]))
    if near_pairs:
        print()
        print("近失号（GT→我方，前 16）：")
        for k, v in near_pairs.most_common(16):
            print("   %-16s %d" % (k, v))
    if bykind:
        print()
        print("名字错的形态（我方→GT，前 12）：")
        for k, v in bykind.most_common(12):
            print("   %-34s %d" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
