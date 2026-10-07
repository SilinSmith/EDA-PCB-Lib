"""stage1 导线过滤扫描：到底有多少「提取线段」是文字/符号/边框？
目的：骨架化对「所有墨迹」一视同仁，而电线与文字笔画都是细线 —— 所以提取总长是 GT 的 5 倍。
本脚本不重新提骨架，只在已有 segments 上做后验过滤，量 R/P 如何随过滤变化。

每张 case 只算一次距离矩阵，之后各过滤档位都是掩码操作，所以整轮扫描很便宜。

用法: python work\stage1_wire_filter.py [stage1产出目录] [--tol 5]
"""
import json
import os
import sys
import glob
import math
import collections
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'Y:\workspace\EDA\eda-pcb')

from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

NS = 8          # 每条线段的采样点数


def sample_polyline(p, q, n=NS):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p)[None, :] * (1 - t) + np.asarray(q)[None, :] * t


def seg_arrays(segs):
    a = np.asarray(segs, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != 2:
        return np.zeros((0, 4))
    return np.concatenate([a[:, 0, :], a[:, 1, :]], axis=1)


def pairwise(pts, segs):
    """pts (M,2) -> (M,N) 到每段的距离。分块。"""
    M, N = pts.shape[0], segs.shape[0]
    out = np.empty((M, N))
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    L2[L2 == 0] = 1e-12
    CH = 2048
    for s in range(0, M, CH):
        e = min(s + CH, M)
        px = pts[s:e, 0][:, None]
        py = pts[s:e, 1][:, None]
        t = ((px - x1) * dx + (py - y1) * dy) / L2
        np.clip(t, 0.0, 1.0, out=t)
        out[s:e] = np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
    return out


def seg_len(A):
    return np.hypot(A[:, 2] - A[:, 0], A[:, 3] - A[:, 1])


def axis_dev(A):
    """与最近的 0/45/90/135 度的偏差（度）。"""
    a = np.arctan2(A[:, 3] - A[:, 1], A[:, 2] - A[:, 0])
    m = np.mod(a, math.pi / 4.0)
    m = np.minimum(m, math.pi / 4.0 - m)
    return np.degrees(m)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else r'Y:\workspace\EDA\work\stage1_all2'
    tol = 5.0
    if '--tol' in sys.argv:
        tol = float(sys.argv[sys.argv.index('--tol') + 1])

    levels = []
    for L in (0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0):
        levels.append((f'len>={L:g}', (lambda L: (lambda A: seg_len(A) >= L))(L)))
    for deg in (20.0, 12.0):
        levels.append((f'axis<={deg:g}deg', (lambda d: (lambda A: axis_dev(A) <= d))(deg)))
    for L, deg in ((6.0, 20.0), (10.0, 20.0), (15.0, 20.0), (10.0, 12.0), (20.0, 20.0)):
        levels.append((f'len>={L:g}&axis<={deg:g}',
                       (lambda L, d: (lambda A: (seg_len(A) >= L) & (axis_dev(A) <= d)))(L, deg)))
    levels.append(('axis<=20&notiny', lambda A: (axis_dev(A) <= 20.0) & (seg_len(A) >= 3.0)))

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, '*.json')))

    # 累加器：[gtcov, gtedges, pcov, psegs, len_gt, len_pred]
    acc = {name: np.zeros(6) for name, _ in levels}
    src_acc = collections.defaultdict(lambda: {name: np.zeros(6) for name, _ in levels})
    base = np.zeros(6)

    for f in files:
        d = json.load(open(f, encoding='utf-8'))
        case = cases.get(d['case'])
        if case is None:
            for k, v in cases.items():
                if k.lstrip('0') == d['case'].lstrip('0'):
                    case = v
                    break
        if case is None:
            continue
        H = float(d['image']['h'])
        gt = json.load(open(case.json_path, encoding='utf-8'))
        gsegs = []
        for net in gt['nets'].values():
            for pts in (net.get('edges') or {}).values():
                for i in range(len(pts) - 1):
                    gsegs.append([[float(pts[i]['x']), H - float(pts[i]['y'])],
                                  [float(pts[i + 1]['x']), H - float(pts[i + 1]['y'])]])
        GA = seg_arrays(gsegs)
        PA = seg_arrays(d['wire_detail']['segments'])
        if PA.shape[0] == 0:
            continue

        # GT 采样点 -> 每段距离
        gpts = np.concatenate([sample_polyline(GA[i, :2], GA[i, 2:]) for i in range(GA.shape[0])])
        EM = pairwise(gpts, PA).reshape(GA.shape[0], NS, PA.shape[0]).min(axis=1)  # (E, N)
        # 提取段采样点 -> 每 GT 段距离，取每段最差
        ppts = np.concatenate([sample_polyline(PA[i, :2], PA[i, 2:]) for i in range(PA.shape[0])])
        PW = pairwise(ppts, GA).reshape(PA.shape[0], NS, GA.shape[0]).max(axis=1).min(axis=1)  # (N,)

        lg = float(seg_len(GA).sum())
        lp_all = float(seg_len(PA).sum())
        base += [0, 0, 0, PA.shape[0], lg, lp_all]

        png = Path(case.png_path).name.lower().replace(' ', '')
        src = ('kicad' if 'kicad' in png else 'altium' if 'altium' in png
               else 'datasheet' if 'datasheet' in png else 'jlc' if 'jlc' in png else 'other')

        for name, fn in levels:
            keep = fn(PA)
            n = int(keep.sum())
            if n:
                worst = EM[:, keep].min(axis=1)
                cov = int((worst <= tol).sum())
                pcov = int((PW[keep] <= tol).sum())
                lp = float(seg_len(PA)[keep].sum())
            else:
                cov = pcov = 0
                lp = 0.0
            row = np.array([cov, GA.shape[0], pcov, n, lg, lp])
            acc[name] += row
            src_acc[src][name] += row

    print(f'stage1 产物 {len(files)} 份   tol={tol}px')
    print(f'\n未过滤基线：提取段 {int(base[3])}  提取总长 {base[5]:.0f} / GT 总长 {base[4]:.0f} '
          f'= {base[5]/max(1,base[4]):.2f}x')

    print(f'\n{"过滤":22s} {"保留段":>8s} {"保留%":>7s} {"L/Lgt":>7s} {"R":>7s} {"P":>7s} {"F1":>7s}')
    rows = []
    for name, _ in levels:
        a = acc[name]
        if a[1] == 0:
            continue
        R = a[0] / a[1]
        P = a[2] / max(1, a[3])
        F1 = 2 * R * P / max(1e-9, R + P)
        rows.append((name, int(a[3]), a[3] / max(1, base[3]), a[5] / max(1, a[4]), R, P, F1))
    for name, nkeep, frac, lr, R, P, F1 in rows:
        print(f'{name:22s} {nkeep:8d} {frac*100:6.1f}% {lr:7.2f} {R:7.4f} {P:7.4f} {F1:7.4f}')

    best = max(rows, key=lambda t: t[6])
    print(f'\n最佳档位：{best[0]}  F1 {best[6]:.4f}  (R {best[4]:.4f} / P {best[5]:.4f})')

    print(f'\n=== 最佳档位 {best[0]} 按来源 ===')
    print(f'  {"来源":10s} {"GT段":>7s} {"覆盖":>7s} {"R":>7s} {"提取":>7s} {"贴住":>7s} {"P":>7s}')
    for s in sorted(src_acc):
        a = src_acc[s][best[0]]
        if a[1] == 0:
            continue
        print(f'  {s:10s} {int(a[1]):7d} {int(a[0]):7d} {a[0]/a[1]:7.3f} '
              f'{int(a[3]):7d} {int(a[2]):7d} {a[2]/max(1,a[3]):7.3f}')


if __name__ == '__main__':
    main()
