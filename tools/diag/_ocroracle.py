# -*- coding: utf-8 -*-
"""★★ **文字识别一共值多少分？**（写实上界，不改管线、不占显存）

问题：引脚失分里「号/名读错」占多大？如果 OCR 全对能涨多少？
这决定了「要不要花时间重训 CRNN」是否值得。

做法（对已有的 `_Pred.json` 做上帝视角改写，再重新打分）：
  * `base`  —— 原样；
  * `name`  —— 预测引脚里，**点已经落在 GT 引脚点 `pin_tol` 内**的那些，把 `pinname` 换成 GT 的；
              （键不动 ⇒ 只修 `name_mismatch`）
  * `num`   —— 在 `name` 基础上，**把引脚键也换成 GT 的键**，并把 `nets[*].hyperGraph`
              里的 `Comp.pin` 引用一起改写 ⇒ 「点和号都读对」。
⇒ `num - base` 就是「**在点已经找对的前提下**，把号与名都读对」能拿到的全部分数。
   OCR 再准也不可能超过它；而重训能动的只是其中的一部分。

    python work/tools/diag/_ocroracle.py --out work/_fast_out_reg9 --cases 1,2,3,4,5,7,8,9,10
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, build_hypergraph,    # noqa: E402
                                   discover, load_json, parse_hypergraph)
from schematic_parser.metrics import (EvalConfig, _point, assign_components,  # noqa: E402
                                      evaluate_case, normalise_doc, rewrite_keys)

METRICS = ["component", "pin", "net_hypergraph", "net_line"]
ZH = {"component": "元件", "pin": "引脚", "net_hypergraph": "网络", "net_line": "导线"}


def pred_of(cid, out_dir):
    for name in (cid + "_Pred.json", cid + ".json"):
        q = Path(out_dir) / "final_results" / name
        if q.exists():
            return load_json(q)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "work" / "_fast_out_reg9"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--each", action="store_true")
    a = ap.parse_args()
    cfg = EvalConfig()
    want = [s.strip() for s in a.cases.split(",") if s.strip()]
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split]
    if want:
        cs = [c for c in cs if c.case_id in want]
    if a.n:
        cs = cs[:a.n]

    rows = []
    for c in cs:
        praw = pred_of(c.case_id, a.out)
        if praw is None:
            print("!! 缺 %s" % c.case_id)
            continue
        gt = normalise_doc(load_json(Path(c.json_path)))
        pred = normalise_doc(praw)
        base = evaluate_case(gt, pred, case_id=c.case_id, cfg=cfg)
        _, km = assign_components(gt, pred, cfg, {})
        P = rewrite_keys(pred, km)

        Pn = copy.deepcopy(P)
        Pnum = copy.deepcopy(P)
        from schematic_parser.metrics import IC_TYPES, canon_type
        boxc = {k for k, v in (gt.get("components") or {}).items()
                if canon_type((v or {}).get("type")) in IC_TYPES}
        n_pin = n_hit = n_pt = n_num = 0
        nb_pin = nb_hit = 0
        for comp, group in (P.get("pins") or {}).items():
            if not isinstance(group, dict):
                continue
            gtgroup = (gt.get("pins") or {}).get(comp) or {}
            for pk, info in group.items():
                n_pin += 1
                if comp in boxc:
                    nb_pin += 1
                p = _point((info or {}).get("point"))
                best, bd = None, 1e18
                for gpk, ginfo in gtgroup.items():
                    q = _point((ginfo or {}).get("point"))
                    if p is None or q is None:
                        continue
                    d = math.hypot(p[0] - q[0], p[1] - q[1])
                    if d < bd:
                        best, bd = (gpk, ginfo), d
                if best is None or bd > cfg.pin_tol:
                    n_pt += 1
                    continue
                n_hit += 1
                if comp in boxc:
                    nb_hit += 1
                gpk, ginfo = best
                Pn["pins"][comp][pk]["pinname"] = (ginfo or {}).get("pinname")
                if gpk != pk:
                    n_num += 1
                Pnum["pins"][comp][gpk] = dict(Pnum["pins"][comp][pk])
                Pnum["pins"][comp][gpk]["pinname"] = (ginfo or {}).get("pinname")
                if gpk != pk:
                    Pnum["pins"][comp].pop(pk, None)
        # hyperGraph 引用一起改写
        for net in (Pnum.get("nets") or {}).values():
            if isinstance(net, dict) and net.get("hyperGraph"):
                refs = parse_hypergraph(net["hyperGraph"])
                # 用「预测引脚 → GT 引脚」的点匹配去改写
                m = {}
                for cname, group in (P.get("pins") or {}).items():
                    if not isinstance(group, dict):
                        continue
                    g2 = (gt.get("pins") or {}).get(cname) or {}
                    for pk, info in group.items():
                        p = _point((info or {}).get("point"))
                        best, bd = None, 1e18
                        for gpk, ginfo in g2.items():
                            q = _point((ginfo or {}).get("point"))
                            if p is None or q is None:
                                continue
                            d = math.hypot(p[0] - q[0], p[1] - q[1])
                            if d < bd:
                                best, bd = gpk, d
                        if best is not None and bd <= cfg.pin_tol:
                            m[(cname, pk[4:] if pk.startswith("pin_") else pk)] = \
                                best[4:] if best.startswith("pin_") else best
                net["hyperGraph"] = build_hypergraph(
                    [(cn, m.get((cn, pn), pn)) for cn, pn in refs])

        nm = evaluate_case(gt, Pn, case_id=c.case_id, cfg=cfg)
        nu = evaluate_case(gt, Pnum, case_id=c.case_id, cfg=cfg)
        rows.append((c.case_id, base, nm, nu, n_pin, n_hit, n_pt, n_num, nb_pin, nb_hit))
        if a.each:
            print("%-6s 点已对 %3d/%3d（点错 %3d，键要改 %3d）｜分 %6.2f → %6.2f → %6.2f"
                  % (c.case_id, n_hit, n_pin, n_pt, n_num,
                     base.score, nm.score, nu.score))
    if not rows:
        print("没有数据")
        return 1
    N = len(rows)
    print("=" * 100)
    print("out=%s  case 数 %d" % (a.out, N))
    print("%-8s %14s %14s %14s %10s" % ("分项", "base 原样", "name 名全对", "num 号+名全对", "num-base"))
    for m in METRICS:
        fb = sum(getattr(r[1], m).f1 for r in rows) / N
        fn = sum(getattr(r[2], m).f1 for r in rows) / N
        fu = sum(getattr(r[3], m).f1 for r in rows) / N
        print("%-8s %14.4f %14.4f %14.4f %+10.4f" % (ZH[m], fb, fn, fu, fu - fb))
    sb = sum(r[1].score for r in rows) / N
    sn = sum(r[2].score for r in rows) / N
    su = sum(r[3].score for r in rows) / N
    print("%-8s %14.4f %14.4f %14.4f %+10.4f" % ("总分", sb, sn, su, su - sb))
    for m in METRICS:
        tb = sum(getattr(r[1], m).tp for r in rows)
        tn = sum(getattr(r[2], m).tp for r in rows)
        tu = sum(getattr(r[3], m).tp for r in rows)
        print("  tp %s %d → %d → %d" % (ZH[m], tb, tn, tu))
    tot_pin = sum(r[4] for r in rows)
    tot_hit = sum(r[5] for r in rows)
    print("预测引脚合计 %d；**点落在 GT 点 %gpx 内** %d（%.0f%%）；点就没找对 %d（%.0f%%）"
          % (tot_pin, cfg.pin_tol, tot_hit, 100.0 * tot_hit / max(tot_pin, 1),
             sum(r[6] for r in rows), 100.0 * sum(r[6] for r in rows) / max(tot_pin, 1)))
    bp = sum(r[8] for r in rows)
    bh = sum(r[9] for r in rows)
    print("其中 **IC(box) 元件**的预测引脚 %d；点已对 %d（%.0f%%）；点没对 %d"
          % (bp, bh, 100.0 * bh / max(bp, 1), bp - bh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
