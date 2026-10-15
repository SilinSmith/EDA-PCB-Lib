"""stage1 导线产出 vs GT 的验收评测。

GT：每个 net 的 edges 是一组**线段**（json 坐标，左下原点、y 向上），形如
    "edge_1": [{"x":135.0,"y":684.0},{"x":135.0,"y":646.0}]
stage1：wire_detail.segments 是**图像坐标**（左上原点，y 向下）。

两个方向都要量：
  覆盖 (recall)  : 每条 GT 线段上的采样点，到最近提取线段的距离是否 <= tol
  精度 (precision): 每条提取线段的采样点，到最近 GT 线段的距离是否 <= tol
只报覆盖会掩盖「骨架碎成一地」的问题（碎片仍能贴住 GT 线）。

用法: python work\stage1_wires_eval.py [stage1产出目录] [--tol 5]
"""
import json
import os
import sys
import glob
import collections
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402


def seg_arrays(segs):
    """segs = list of [[x1,y1],[x2,y2]] -> (N,4) float64"""
    a = np.asarray(segs, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != 2:
        return np.zeros((0, 4))
    return np.concatenate([a[:, 0, :], a[:, 1, :]], axis=1)


def dist_pts_to_segs(pts, segs):
    """pts (M,2), segs (N,4) -> (M,) 每点到最近线段的距离。分块以免爆内存。"""
    M = pts.shape[0]
    N = segs.shape[0]
    out = np.full(M, np.inf)
    if N == 0:
        return out
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    L2[L2 == 0] = 1e-12
    bx0, bx1 = np.minimum(x1, x2), np.maximum(x1, x2)
    by0, by1 = np.minimum(y1, y2), np.maximum(y1, y2)
    CH = 4096
    for s in range(0, M, CH):
        e = min(s + CH, M)
        px = pts[s:e, 0][:, None]
        py = pts[s:e, 1][:, None]
        t = ((px - x1) * dx + (py - y1) * dy) / L2
        np.clip(t, 0.0, 1.0, out=t)
        cx = x1 + t * dx
        cy = y1 + t * dy
        d = np.hypot(px - cx, py - cy)
        out[s:e] = d.min(axis=1)
    return out


def sample_polyline(p, q, n=16):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p)[None, :] * (1 - t) + np.asarray(q)[None, :] * t


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / 'work' / 'stage1_all2')
    tol = 5.0
    if '--tol' in sys.argv:
        tol = float(sys.argv[sys.argv.index('--tol') + 1])

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, '*.json')))
    print(f'stage1 产物 {len(files)} 份   tol={tol}px')

    n_gt_edges = n_gt_cov = 0
    n_pred_segs = n_pred_cov = 0
    len_gt = len_pred = 0.0
    per_case = []
    src_stat = collections.defaultdict(lambda: [0, 0, 0, 0])   # gtcov, gt, pcov, p
    maxdist_gt = []

    for f in files:
        d = json.load(open(f, encoding='utf-8'))
        cid = d['case']
        case = cases.get(cid)
        if case is None:
            for k, v in cases.items():
                if k.lstrip('0') == cid.lstrip('0'):
                    case, cid = v, k
                    break
        if case is None:
            continue
        H = float(d['image']['h'])
        gt = json.load(open(case.json_path, encoding='utf-8'))

        # GT 线段 -> 图像坐标（y 翻转）
        gsegs = []
        for net in gt['nets'].values():
            for pts in (net.get('edges') or {}).values():
                if len(pts) < 2:
                    continue
                for i in range(len(pts) - 1):
                    gsegs.append([[float(pts[i]['x']), H - float(pts[i]['y'])],
                                  [float(pts[i + 1]['x']), H - float(pts[i + 1]['y'])]])
        GA = seg_arrays(gsegs)
        PA = seg_arrays(d['wire_detail']['segments'])

        # --- 覆盖：GT 采样点到最近提取线段
        if GA.shape[0] and PA.shape[0]:
            gt_pts = []
            gt_of = []
            for i in range(GA.shape[0]):
                gt_pts.append(sample_polyline(GA[i, :2], GA[i, 2:], 16))
                gt_of.append(np.full(16, i))
            gt_pts = np.concatenate(gt_pts, axis=0)
            gt_of = np.concatenate(gt_of)
            dist = dist_pts_to_segs(gt_pts, PA)
            worst = np.full(GA.shape[0], 0.0)
            np.maximum.at(worst, gt_of, dist)
            cov = int((worst <= tol).sum())
            maxdist_gt.append(float(np.median(worst)))
        else:
            cov = 0
            worst = np.zeros(GA.shape[0]) if GA.shape[0] else np.zeros(0)

        # --- 精度：提取线段采样点到最近 GT 线段
        if PA.shape[0] and GA.shape[0]:
            p_pts = []
            p_of = []
            for i in range(PA.shape[0]):
                p_pts.append(sample_polyline(PA[i, :2], PA[i, 2:], 16))
                p_of.append(np.full(16, i))
            p_pts = np.concatenate(p_pts, axis=0)
            p_of = np.concatenate(p_of)
            pdist = dist_pts_to_segs(p_pts, GA)
            pworst = np.full(PA.shape[0], 0.0)
            np.maximum.at(pworst, p_of, pdist)
            pcov = int((pworst <= tol).sum())
        else:
            pcov = 0

        lg = float(np.hypot(GA[:, 2] - GA[:, 0], GA[:, 3] - GA[:, 1]).sum()) if GA.shape[0] else 0.0
        lp = float(np.hypot(PA[:, 2] - PA[:, 0], PA[:, 3] - PA[:, 1]).sum()) if PA.shape[0] else 0.0

        n_gt_edges += GA.shape[0]
        n_gt_cov += cov
        n_pred_segs += PA.shape[0]
        n_pred_cov += pcov
        len_gt += lg
        len_pred += lp

        png = Path(case.png_path).name.lower().replace(' ', '')
        src = ('kicad' if 'kicad' in png else 'altium' if 'altium' in png
               else 'datasheet' if 'datasheet' in png else 'jlc' if 'jlc' in png else 'other')
        s = src_stat[src]
        s[0] += cov
        s[1] += GA.shape[0]
        s[2] += pcov
        s[3] += PA.shape[0]

        per_case.append((cid, GA.shape[0], cov, PA.shape[0], pcov, lg, lp))

    print('\n=== 汇总 ===')
    print(f'GT 线段   {n_gt_edges:7d}   覆盖 {n_gt_cov:7d}  recall {n_gt_cov/max(1,n_gt_edges):.4f}')
    print(f'提取线段  {n_pred_segs:7d}   贴住 {n_pred_cov:7d}  precision {n_pred_cov/max(1,n_pred_segs):.4f}')
    f1 = (2 * n_gt_cov / max(1, n_gt_edges) * n_pred_cov / max(1, n_pred_segs)
          / max(1e-9, (n_gt_cov / max(1, n_gt_edges) + n_pred_cov / max(1, n_pred_segs))))
    print(f'线段 F1   {f1:.4f}')
    print(f'GT 总线长 {len_gt:12.0f} px   提取总长 {len_pred:12.0f} px   比值 {len_pred/max(1.0,len_gt):.3f}')

    print('\n=== 按来源 ===')
    for s in sorted(src_stat):
        a = src_stat[s]
        print(f'  {s:10s} GT {a[1]:6d} 覆盖 {a[0]:6d} R {a[0]/max(1,a[1]):.3f}   '
              f'提取 {a[3]:6d} 贴住 {a[2]:6d} P {a[2]/max(1,a[3]):.3f}')

    per_case.sort(key=lambda t: t[2] / max(1, t[1]))
    print('\n=== 最差 12 张（按 GT 覆盖率）===')
    print(f'  {"case":8s} {"gtE":>5s} {"cov":>5s} {"R":>6s} {"predE":>6s} {"P":>6s} {"Lpred/Lgt":>10s}')
    for cid, g, cov, p, pcov, lg, lp in per_case[:12]:
        print(f'  {cid:8s} {g:5d} {cov:5d} {cov/max(1,g):6.3f} {p:6d} {pcov/max(1,p):6.3f} '
              f'{lp/max(1.0,lg):10.2f}')

    print('\n=== 最好 5 张 ===')
    for cid, g, cov, p, pcov, lg, lp in per_case[-5:]:
        print(f'  {cid:8s} {g:5d} {cov:5d} {cov/max(1,g):6.3f} {p:6d} {pcov/max(1,p):6.3f} '
              f'{lp/max(1.0,lg):10.2f}')

    if maxdist_gt:
        a = np.asarray(maxdist_gt)
        print(f'\n每张的「GT 线段最差点距离」中位数：p10 {np.percentile(a,10):.2f} '
              f'p50 {np.percentile(a,50):.2f} p90 {np.percentile(a,90):.2f}')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
