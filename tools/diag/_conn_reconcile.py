# -*- coding: utf-8 -*-
"""对账：为什么 60 张消融说 all12 的连通 F1 是 0.5421，而端到端评测器说 0.1457？

对**同一批产物、同一批 case**，用两种算法各算一遍：
  (a) `stage1_filter_ablation.conn_stats`（消融用的那份）
  (b) 端到端 `stage1_conn_eval` 的算法（已修 build_groups + 暴力绑定）
如果 (a) 和 (b) 在同一份段上给出同样的数，那差异只能来自「段的来源」；
如果不同，就是两个算法里还有一处不一致。
"""
import json
import os
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "eval"))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (r'Y:\workspace\EDA\eda-pcb', HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from schematic_parser.data import discover                       # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs       # noqa: E402
from stage1_conn_eval import (build_groups, gt_pins_and_nets,    # noqa: E402
                              _dist_pts_to_segs)
import stage1_filter_ablation as abl                             # noqa: E402

DATA_ROOT = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'
PRODS = [r'Y:\workspace\EDA\work\stage1_all11', r'Y:\workspace\EDA\work\stage1_all12']


def run_b(psegs, pins, nets, pin_tol=6.0, tol=1.0):
    """端到端 stage1_conn_eval 的算法。"""
    rootmap = build_groups(psegs, tol)
    keys = [k for k, _ in pins]
    pts = np.asarray([pt for _, pt in pins], np.float64).reshape(-1, 2)
    D = _dist_pts_to_segs(pts, psegs)
    jm = np.argmin(D, axis=1)
    dm = D[np.arange(len(keys)), jm]
    pr = {k: (int(rootmap[int(j)]) if float(d) <= pin_tol else ('solo', k))
          for k, j, d in zip(keys, jm, dm)}
    tp = fn = 0
    for _nid, ks in nets:
        ks = [k for k in ks if k in pr]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if pr[ks[i]] == pr[ks[j]]:
                    tp += 1
                else:
                    fn += 1
    return (tp / float(tp + fn) if tp + fn else 0.0), tp, fn


def main():
    cases = sorted(discover(DATA_ROOT), key=lambda c: c.case_id)[::20]   # 与消融同一批
    print(f'cases {len(cases)}')
    for prod in PRODS:
        ta = {'tp': 0, 'pos': 0, 'ns': 0, 'nt': 0}
        tb = [0, 0]
        nseg = 0
        for c in cases:
            sp = os.path.join(prod, c.case_id + '.stage1.json')
            if not os.path.exists(sp):
                continue
            sd = json.load(open(sp, encoding='utf-8'))
            gd = json.load(open(c.json_path, encoding='utf-8'))
            H = float(sd['image']['h'])
            segs = seg_arrays(sd['wire_detail']['segments'])
            nseg += len(segs)
            pins, nets = gt_pins_and_nets(gd, H)
            if not len(segs) or not pins:
                continue
            # (a) 消融的算法
            abl.PIN_MAP = dict(pins)
            tp, fn, ns, nt = abl.conn_stats(segs, pins, nets)
            ta['tp'] += tp
            ta['pos'] += tp + fn
            ta['ns'] += ns
            ta['nt'] += nt
            # (b) 端到端算法
            _r, tpb, fnb = run_b(segs, pins, nets)
            tb[0] += tpb
            tb[1] += fnb
        ra = ta['tp'] / float(ta['pos']) if ta['pos'] else 0.0
        rb = tb[0] / float(tb[0] + tb[1]) if (tb[0] + tb[1]) else 0.0
        print(f'{os.path.basename(prod)}  段 {nseg}')
        print(f'   (a) 消融 conn_stats : R {ra:.4f}  (TP {ta["tp"]} / pos {ta["pos"]})  '
              f'负样本误并 {ta["ns"]}/{ta["nt"]}')
        print(f'   (b) 端到端算法      : R {rb:.4f}  (TP {tb[0]} / pos {tb[0] + tb[1]})')


if __name__ == '__main__':
    main()
