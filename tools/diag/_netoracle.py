# -*- coding: utf-8 -*-
"""`net_hypergraph` 的 tp 上限分解：我们的引脚引用里有多少「内容正确」的？

对每个 GT 网络，取「我们把元件键改写对、且引脚号也对」的成员数 —— 这是**分组完美**
时能拿到的 tp 上限。实际 tp 与它的差 = 纯粹的分组（过分割）损失。

    python work\\tools\\diag\\_netoracle.py --pred work\\_gt10_pb10\\runs\\stage2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M                    # noqa: E402
from schematic_parser.data import discover, net_pin_set       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    pred_dir = Path(a.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    cfg = M.EvalConfig()
    want = [x.strip() for x in a.cases.split(",") if x.strip()] or None
    tot = dict(gt_mem=0, gt_phi=0, oracle_tp=0, real_tp=0, pred_mem=0, pred_net=0, gt_net=0)
    for c in discover():
        if want and c.case_id not in want:
            continue
        pj = pred_dir / ("%s.json" % c.case_id)
        if not (pj.is_file() and c.json_path):
            continue
        gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pr = json.loads(pj.read_text(encoding="utf-8"))
        _f1, keymap = M.assign_components(gt, pr, cfg)
        pr2 = M.rewrite_keys(pr, keymap)
        gn = M._net_sets(gt["nets"])
        pn = M._net_sets(pr2["nets"])
        pairs = M.match_nets(gn, pn)
        real = sum(len(pn[i][1] & gn[j][1]) for i, j in pairs)
        # oracle：所有预测引脚引用的并集，与每条 GT 网络求交
        allpred = set()
        for _, s in pn:
            allpred |= s
        oracle = sum(len(allpred & g) for _, g in gn)
        tot["gt_mem"] += sum(len(g) for _, g in gn)
        tot["gt_phi"] += sum(1 for _, g in gn for (comp, _p) in g if comp.startswith("\u2205"))
        tot["pred_mem"] += sum(len(s) for _, s in pn)
        tot["pred_net"] += len(pn)
        tot["gt_net"] += len(gn)
        tot["oracle_tp"] += oracle
        tot["real_tp"] += real
    print(json.dumps(tot, ensure_ascii=False))
    print("oracle 上限 tp=%d  实际 tp=%d  ⇒ 分组损失 %d"
          % (tot["oracle_tp"], tot["real_tp"], tot["oracle_tp"] - tot["real_tp"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
