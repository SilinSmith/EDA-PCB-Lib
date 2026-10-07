"""导线提取的 A/B：无遮挡 / 遮挡元件 / 遮挡+边框与笔画过滤。
严格判据（与 stage1_wires_eval.py 一致）：
  覆盖 = GT 线段上**每个**采样点都能在 tol 内找到保留线段
  精度 = 保留线段上**每个**采样点都能在 tol 内找到 GT 线段
用法: python work\_wire_ab.py [case ...]
"""
import json
import os
import sys
import math

import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "eval"))
import cv2

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'Y:\workspace\EDA\eda-pcb')
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
from stage1.wires import extract_wires  # noqa: E402

TOL = 5.0
NS = 12
OUT = r'Y:\workspace\EDA\work\stage1_all2'

# (标签, 过滤档位)
BASE = dict(occlude='auto', min_len=0.0, max_axis_dev=12.0, drop_frame=True)
ARMS = [
    ('A 无遮挡',      dict(occlude=None, min_len=0.0, max_axis_dev=90.0, drop_frame=False)),
    ('C 遮挡+框+轴12', dict(BASE)),
    ('P ext>=30',     dict(BASE, min_extent=30.0)),
    ('Q ext>=50',     dict(BASE, min_extent=50.0)),
    ('R ext>=80',     dict(BASE, min_extent=80.0)),
    ('S ext>=120',    dict(BASE, min_extent=120.0)),
    ('U ext50 rc4',   dict(BASE, min_extent=50.0, rescue_pad=4.0)),
    ('V ext50 rc8',   dict(BASE, min_extent=50.0, rescue_pad=8.0)),
    ('W ext50 rc14',  dict(BASE, min_extent=50.0, rescue_pad=14.0)),
    ('X ext80 rc8',   dict(BASE, min_extent=80.0, rescue_pad=8.0)),
]


def seg_arrays(segs):
    a = np.asarray(segs, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != 2 or a.shape[0] == 0:
        return np.zeros((0, 4))
    return np.concatenate([a[:, 0, :], a[:, 1, :]], axis=1)


def dist_pts_to_segs(pts, segs):
    M, N = pts.shape[0], segs.shape[0]
    out = np.full(M, np.inf)
    if N == 0:
        return out
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    L2[L2 == 0] = 1e-12
    for s in range(0, M, 4096):
        e = min(s + 4096, M)
        px = pts[s:e, 0][:, None]
        py = pts[s:e, 1][:, None]
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / L2, 0, 1)
        out[s:e] = np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)).min(axis=1)
    return out


def sample(A, n=NS):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return A[:, None, :2] * (1 - t) + A[:, None, 2:] * t      # (N,n,2)


def measure(GA, PA):
    """返回 (recall, precision, len_ratio)。"""
    lg = float(np.hypot(GA[:, 2] - GA[:, 0], GA[:, 3] - GA[:, 1]).sum()) if GA.shape[0] else 0.0
    lp = float(np.hypot(PA[:, 2] - PA[:, 0], PA[:, 3] - PA[:, 1]).sum()) if PA.shape[0] else 0.0
    if GA.shape[0] == 0 or PA.shape[0] == 0:
        return 0.0, 0.0, (lp / lg if lg else 0.0)
    gp = sample(GA).reshape(-1, 2)
    d = dist_pts_to_segs(gp, PA).reshape(GA.shape[0], NS)
    R = float((d.max(axis=1) <= TOL).mean())
    pp = sample(PA).reshape(-1, 2)
    dp = dist_pts_to_segs(pp, GA).reshape(PA.shape[0], NS)
    P = float((dp.max(axis=1) <= TOL).mean())
    return R, P, (lp / lg if lg else 0.0)


def main():
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    ids = sys.argv[1:] or ['0101', '0107', '0001', '0059', '0050', '0011']
    for cid in ids:
        case = cases.get(cid) or cases.get(cid.zfill(4))
        if case is None:
            print('no case', cid); continue
        bgr = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
        H = bgr.shape[0]
        gt = json.load(open(case.json_path, encoding='utf-8'))
        gs = []
        for net in gt['nets'].values():
            for pts in (net.get('edges') or {}).values():
                for i in range(len(pts) - 1):
                    gs.append([[float(pts[i]['x']), H - float(pts[i]['y'])],
                               [float(pts[i + 1]['x']), H - float(pts[i + 1]['y'])]])
        GA = seg_arrays(gs)
        f = os.path.join(OUT, f'{case.case_id}.stage1.json')
        boxes = []
        if os.path.exists(f):
            d = json.load(open(f, encoding='utf-8'))
            boxes = [c['bbox_img'] for c in d['components']]
        src = os.path.basename(case.png_path).split('-')[-1].replace('.png', '')
        print(f'\n--- {case.case_id} ({src}) {bgr.shape[1]}x{H}  '
              f'GT段 {GA.shape[0]}  遮挡框 {len(boxes)} ---')
        print(f'  {"档位":16s} {"段数":>7s} {"真":>6s} {"覆盖":>7s} {"R":>7s} {"P":>7s} '
              f'{"F1":>7s} {"L/Lgt":>7s} {"耗时":>7s}')
        for name, kw in ARMS:
            k = dict(kw)
            k['occlude'] = boxes if k['occlude'] == 'auto' else None
            w, dt = extract_wires(bgr, **k)
            PA = seg_arrays(w['segments'])
            R, P, lr = measure(GA, PA)
            F1 = 2 * R * P / max(1e-9, R + P)
            print(f'  {name:16s} {w["n_segments"]:7d} {w["n_segments_raw"]:6d} '
                  f'{int(R*GA.shape[0]):7d} {R:7.4f} {P:7.4f} {F1:7.4f} {lr:7.2f} {dt:6.2f}s')


if __name__ == '__main__':
    main()
