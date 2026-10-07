# -*- coding: utf-8 -*-
"""连通分量的结构诊断：我的段图到底是不是「一坨」。

如果最大的分量装下了几乎所有段，那连通召回/精度这两个数就都失去意义
（把整张图并成一个分量，R 必为 1、P 必为 0）。
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (r'Y:\workspace\EDA\eda-pcb', HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage1_wires_eval import seg_arrays                    # noqa: E402
from stage1_conn_eval import (build_groups, gt_edge_segs,   # noqa: E402
                              _dist_pts_to_segs)

DATA_ROOT = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'
PROD = r'Y:\workspace\EDA\work\stage1_all12'
WANT = ['0001', '0050', '0101', '0107', '0059']


def selftest():
    """合成用例：三段两两相隔 90px 以上，tol=1 时必须还是 3 个分量。"""
    s = np.asarray([[0, 0, 10, 0], [100, 100, 110, 100], [500, 500, 510, 500]], np.float64)
    print('原始 build_groups(s,1.0) =', build_groups(s, 1.0))
    for t in (0.0, 0.5, 1.0, 4.0, 16.0):
        print(f'  tol={t:5.1f} -> {sorted(set(build_groups(s, t).values()))}')
    # 复刻规则 (1)：端点-端点
    e1 = np.concatenate([s[:, :2], s[:, 2:]], axis=0)
    own = np.repeat(np.arange(len(s)), 2)
    D1 = _dist_pts_to_segs(e1, s)
    print('  端点->段 距离矩阵 (自检用):')
    print(np.round(D1, 1))
    n1 = 0
    for m in range(len(e1)):
        for n in range(len(s)):
            if own[m] != n and D1[m][n] <= 1.0:
                n1 += 1
    print(f'  规则(1) 按「端点->段」判连边数 = {n1}')
    # 端点-端点 真距离
    from itertools import combinations
    mn = 1e18
    for a, b in combinations(range(len(e1)), 2):
        if own[a] == own[b]:
            continue
        d = float(np.hypot(e1[a][0] - e1[b][0], e1[a][1] - e1[b][1]))
        mn = min(mn, d)
    print(f'  端点-端点 最小跨段距离 = {mn:.1f} px')


def coincident(segs, tol):
    """统计「不同段之间端点距离 <= tol」的配对数 —— 这就是 (1) 准则下的连边数。"""
    e = np.concatenate([segs[:, :2], segs[:, 2:]], axis=0)
    own = np.repeat(np.arange(len(segs)), 2)
    D = _dist_pts_to_segs(e, segs)
    n = 0
    for m in range(len(e)):
        j = int(np.argmin(D[m]))
        if own[m] != j and float(D[m][j]) <= tol:
            n += 1
    return n


def main():
    selftest()
    from schematic_parser.data import discover
    allc = {c.case_id: c for c in discover(DATA_ROOT)}
    for cid in WANT:
        c = allc.get(cid)
        if c is None:
            continue
        sd = json.load(open(os.path.join(PROD, cid + '.stage1.json'), encoding='utf-8'))
        gd = json.load(open(c.json_path, encoding='utf-8'))
        H = float(sd['image']['h'])
        for name, segs in (('mine', seg_arrays(sd['wire_detail']['segments'])),
                           ('gt  ', gt_edge_segs(gd, H))):
            if not len(segs):
                print(f'{cid} {name}: 0 segs')
                continue
            g = build_groups(segs, 1.0)
            cnt = Counter(g.values())
            sizes = sorted(cnt.values(), reverse=True)
            nc = coincident(segs, 1.0)
            print(f'{cid} {name}: segs {len(segs)}  分量数 {len(cnt)}  '
                  f'最大分量 {sizes[0]} ({sizes[0] / float(len(segs)) * 100:.1f}%)  '
                  f'top5 {sizes[:5]}  (1)准则连边 {nc}')


if __name__ == '__main__':
    main()
