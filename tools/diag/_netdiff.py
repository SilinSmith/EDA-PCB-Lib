# -*- coding: utf-8 -*-
"""**网络失分账本**：把 `net_hypergraph_f1`（权重 0.35，micro 口径）的 fn/fp 逐条拆到
「哪个网络缺了哪些成员 / 多了哪些成员」，并按成员元件类型汇总。

口径与官方一致（`schematic_parser/metrics.py`）：
* `_net_sets(..., collapse_nonbox=cfg.net_pin_collapse_nonbox)`：**非 IC 元件**的引脚号
  折成通配 `(comp, '*')`，IC（box/boxic/ic）的引脚号逐位比较；
* `match_nets` 用匈牙利匹配（相似度 = 引脚集交集）；
* `net_hypergraph_f1` micro：`tp = Σ|p∩g|`、`fp = Σ|p| − tp`、`fn = Σ|g| − tp`。

    python -X utf8 eda-pcb/tools/diag/_netdiff.py --run Y:/workspace/EDA/work/_fast_out_dlv10
    python -X utf8 eda-pcb/tools/diag/_netdiff.py --run ... --case 1 --show 6
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser import metrics as M                                    # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json     # noqa: E402


def _find_pred(run: Path, cid: str):
    for p in (run / "final_results" / ("%s_Pred.json" % cid),
              run / "final_results" / ("%s.json" % cid)):
        if p.exists():
            return load_json(p)
    return None


def _ctype(comps, ref):
    v = (comps or {}).get(ref) or {}
    return M.canon_type(v.get("type")) or "?"


def one(case, run, cfg, show=0):
    raw = _find_pred(run, case.case_id)
    if raw is None:
        print("  !! 没有 _Pred.json")
        return None
    gt = M.normalise_doc(load_json(Path(case.json_path)))
    pred = M.normalise_doc(raw)
    det = {}
    _c, keymap = M.assign_components(gt, pred, cfg, det)
    pred2 = M.rewrite_keys(pred, keymap)
    det2 = {}
    net, pairs = M.net_hypergraph_f1(gt, pred2, cfg, det2)
    g = M._net_sets(gt["nets"], gt.get("components"), cfg.net_pin_collapse_nonbox)
    p = M._net_sets(pred2["nets"], pred2.get("components"),
                    cfg.net_pin_collapse_nonbox)
    print("  nets GT %d / pred %d ｜ 成员 GT %d / pred %d ｜ tp %d fp %d fn %d ｜ F1 %.4f"
          % (len(g), len(p), det2.get("net.gt_pins", 0), det2.get("net.pred_pins", 0),
             net.tp, net.fp, net.fn, net.f1))
    rows = []
    for i, j in pairs:
        gs, ps = g[j][1], p[i][1]
        rows.append((len(gs), g[j][0], p[i][0], gs, ps))
    rows.sort(key=lambda r: -r[0])
    for ng, gn, pn, gs, ps in rows[:max(0, show)]:
        only_g = sorted(gs - ps)
        only_p = sorted(ps - gs)
        print("   GT网 %-10s |%3d| ↔ 我方网 %-10s |%3d|  交 %3d" %
              (str(gn)[:10], len(gs), str(pn)[:10], len(ps), len(gs & ps)))
        if only_g:
            print("      GT独有(%d): %s" % (len(only_g), " ".join(
                "%s:%s" % (r, k) for r, k in only_g[:14])))
        if only_p:
            print("      我方独有(%d): %s" % (len(only_p), " ".join(
                "%s:%s" % (r, k) for r, k in only_p[:14])))
    matched_g = {j for _i, j in pairs}
    matched_p = {i for i, _j in pairs}
    for j, (nn, gs) in enumerate(g):
        if j not in matched_g and gs:
            print("   [未匹GT网] %-10s |%3d| %s" % (str(nn)[:10], len(gs),
                  " ".join("%s:%s" % r for r in sorted(gs)[:10])))
    for i, (nn, ps) in enumerate(p):
        if i not in matched_p and ps:
            print("   [未匹我方网] %-10s |%3d| %s" % (str(nn)[:10], len(ps),
                  " ".join("%s:%s" % r for r in sorted(ps)[:10])))
    # 按成员类型汇总（只统计已匹对的网）
    agg = collections.Counter()
    for i, j in pairs:
        gs, ps = g[j][1], p[i][1]
        for ref, _k in gs - ps:
            agg["fn:" + _ctype(gt.get("components"), ref)] += 1
        for ref, _k in ps - gs:
            agg["fp:" + _ctype(pred2.get("components"), ref)] += 1
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_dlv10")
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--case", default="")
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = M.EvalConfig()
    tot = collections.Counter()
    cases = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split]
    if a.case:
        cases = [c for c in cases if c.case_id == a.case]
    for c in cases[:a.n]:
        print("== case %s ==" % c.case_id)
        agg = one(c, run, cfg, a.show)
        if agg:
            tot.update(agg)
    print("\n== 汇总（已匹对网的成员差集，按类型）==")
    for k in sorted(tot):
        print("  %-12s %d" % (k, tot[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
