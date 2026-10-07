# -*- coding: utf-8 -*-
"""定位「消融里的连通 R」和「stage1_conn_eval 里的连通 R」为什么差 2~9 倍。

对同一个产物目录，用两种引脚绑定方式各算一遍连通 R：
  A. 暴力最近段（我的消融用的）
  B. SegIndex.nearest（stage1_conn_eval.run 用的）
如果两者差很多，就是绑定方式的问题；如果都对不上消融里的数，那就是产物里的
`wire_detail["segments"]` 和 extract_wires 直接返回的 `w["segments"]` 不是一回事。
"""
import json
import os
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "eval"))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = r'Y:\workspace\EDA\eda-pcb'
for p in (EDA, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from schematic_parser.data import discover                      # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs      # noqa: E402
from stage1_conn_eval import (build_groups, gt_pins_and_nets,   # noqa: E402
                              SegIndex, gt_edge_segs)

DATA_ROOT = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'


def conn_r(psegs, pins, nets, mode, pin_tol=6.0, tol=1):
    if not len(psegs):
        return 0.0, 0, 0
    g_of = build_groups(psegs, tol)
    key2pt = dict(pins)

    if mode == 'brute':
        pin2grp = {}
        for k, pt in key2pt.items():
            d = dist_pts_to_segs(np.asarray([pt], np.float64), psegs)
            j = int(np.argmin(d))
            pin2grp[k] = g_of[j] if float(d[j]) <= pin_tol else ('solo', k)
    else:
        idx = SegIndex(psegs, pin_tol)
        pin2grp = {}
        for k, pt in key2pt.items():
            si, d = idx.nearest(pt[0], pt[1])
            if si >= 0 and d is not None and d <= pin_tol:
                pin2grp[k] = g_of.get(si, ('solo', si))
            else:
                pin2grp[k] = ('solo', k)

    tp = fn = 0
    for _nid, keys in nets:
        ks = [k for k in keys if k in key2pt]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if pin2grp[ks[i]] == pin2grp[ks[j]]:
                    tp += 1
                else:
                    fn += 1
    return (tp / float(tp + fn) if (tp + fn) else 0.0), tp, fn


def main():
    dirs = sys.argv[1:] or [r'Y:\workspace\EDA\work\stage1_all11',
                            r'Y:\workspace\EDA\work\stage1_all12']
    cases = sorted(discover(DATA_ROOT), key=lambda c: c.case_id)[::4]      # 55 张
    for d in dirs:
        tot = {'brute': [0, 0], 'grid': [0, 0]}
        nseg = 0
        for c in cases:
            sp = os.path.join(d, c.case_id + '.stage1.json')
            if not os.path.exists(sp):
                continue
            sd = json.load(open(sp, encoding='utf-8'))
            H = float(sd['image']['h'])
            segs = seg_arrays(sd['wire_detail']['segments'])
            nseg += len(segs)
            gdoc = json.load(open(c.json_path, encoding='utf-8'))
            pins, nets = gt_pins_and_nets(gdoc, H)
            for mode in ('brute', 'grid'):
                r, tp, fn = conn_r(segs, pins, nets, mode)
                tot[mode][0] += tp
                tot[mode][1] += fn
        print(f'{os.path.basename(d)}  段数合计 {nseg}  正样本对合计 {tot["brute"][0] + tot["brute"][1]}')
        for mode in ('brute', 'grid'):
            tp, fn = tot[mode]
            print(f'    {mode:5s} 绑定: 连通 R = {tp / float(tp + fn):.4f}   (TP {tp} / FN {fn})')


if __name__ == '__main__':
    main()
