"""提取线段按「是否真的压在 GT 网表线段上」分组，比较两组的图像颜色。
问题：骨架化对所有墨迹一视同仁，文字/符号/边框全被提取。
本脚本回答：真导线与假导线在颜色上分得开吗？按来源分开看。

用法: python work\stage1_wire_colour.py [stage1产出目录] [--tol 3]
"""
import json
import os
import sys
import glob
import math
import collections
from pathlib import Path

import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'Y:\workspace\EDA\eda-pcb')
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

NH, NS, NV = 18, 4, 4       # hue/sat/val 分箱


def src_of(png_name):
    p = png_name.lower().replace(' ', '')
    return ('kicad' if 'kicad' in p else 'altium' if 'altium' in p
            else 'datasheet' if 'datasheet' in p else 'jlc' if 'jlc' in p else 'other')


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else r'Y:\workspace\EDA\work\stage1_all2'
    tol = 3.0
    if '--tol' in sys.argv:
        tol = float(sys.argv[sys.argv.index('--tol') + 1])

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, '*.json')))
    print(f'stage1 产物 {len(files)} 份   tol={tol}px')

    # per source: 3D 计数 (tp, fp) + 长度统计
    cnt = collections.defaultdict(lambda: np.zeros((2, NH, NS, NV)))
    col_tp = collections.defaultdict(lambda: [[] for _ in range(3)])
    col_fp = collections.defaultdict(lambda: [[] for _ in range(3)])
    sev = collections.defaultdict(lambda: np.zeros((2, 2)))   # [tp/fp][len<15, len>=15]
    n_seg = collections.Counter()

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
        img = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H = img.shape[0]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gt = json.load(open(case.json_path, encoding='utf-8'))
        gsegs = []
        for net in gt['nets'].values():
            for pts in (net.get('edges') or {}).values():
                for i in range(len(pts) - 1):
                    gsegs.append((float(pts[i]['x']), H - float(pts[i]['y']),
                                  float(pts[i + 1]['x']), H - float(pts[i + 1]['y'])))
        GA = np.asarray(gsegs, dtype=np.float64) if gsegs else np.zeros((0, 4))
        src = src_of(Path(case.png_path).name)

        PA = np.asarray(d['wire_detail']['segments'], dtype=np.float64)
        if PA.ndim != 3 or PA.shape[0] == 0:
            continue
        segs = np.concatenate([PA[:, 0, :], PA[:, 1, :]], axis=1)
        n_seg[src] += segs.shape[0]
        if GA.shape[0] == 0:
            continue

        # 每段取 1/4、1/2、3/4 三点，读颜色中位数
        ts = np.array([0.25, 0.5, 0.75])[None, :, None]
        p = segs[:, None, :2] * (1 - ts) + segs[:, None, 2:] * ts      # (N,3,2)
        xs = np.clip(np.rint(p[:, :, 0]).astype(int), 0, img.shape[1] - 1)
        ys = np.clip(np.rint(p[:, :, 1]).astype(int), 0, img.shape[0] - 1)
        vals = hsv[ys, xs]                                            # (N,3,3)
        med = np.median(vals, axis=1).astype(int)                      # (N,3) H,S,V

        # 段中点到最近 GT 段距离
        mid = segs[:, :2] * 0.5 + segs[:, 2:] * 0.5
        x1, y1, x2, y2 = GA[:, 0], GA[:, 1], GA[:, 2], GA[:, 3]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        L2[L2 == 0] = 1e-12
        t = np.clip(((mid[:, 0][:, None] - x1) * dx + (mid[:, 1][:, None] - y1) * dy) / L2, 0, 1)
        dist = np.hypot(mid[:, 0][:, None] - (x1 + t * dx), mid[:, 1][:, None] - (y1 + t * dy))
        dmin = dist.min(axis=1)
        is_tp = dmin <= tol

        ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        hb = np.clip(med[:, 0] * NH // 180, 0, NH - 1)
        sb = np.clip(med[:, 1] * NS // 256, 0, NS - 1)
        vb = np.clip(med[:, 2] * NV // 256, 0, NV - 1)
        for k in range(segs.shape[0]):
            g = 0 if is_tp[k] else 1
            cnt[src][g, hb[k], sb[k], vb[k]] += 1
            sev[src][g, 0 if ln[k] < 15 else 1] += 1
            tgt = col_tp if is_tp[k] else col_fp
            for c in range(3):
                tgt[src][c].append(int(med[k, c]))

    print(f'\n{"来源":10s} {"提取段":>8s} {"真":>7s} {"假":>7s} {"P(基)":>7s} '
          f'{"饱和度p50真/假":>16s} {"明度p50真/假":>14s} {"色相p50真/假":>13s}')
    for s in sorted(n_seg):
        tp = int(cnt[s][0].sum())
        fp = int(cnt[s][1].sum())
        if tp + fp == 0:
            continue
        st = np.median(col_tp[s][1]) if col_tp[s][1] else -1
        sf = np.median(col_fp[s][1]) if col_fp[s][1] else -1
        vt = np.median(col_tp[s][2]) if col_tp[s][2] else -1
        vf = np.median(col_fp[s][2]) if col_fp[s][2] else -1
        ht = np.median(col_tp[s][0]) if col_tp[s][0] else -1
        hf = np.median(col_fp[s][0]) if col_fp[s][0] else -1
        print(f'{s:10s} {tp+fp:8d} {tp:7d} {fp:7d} {tp/(tp+fp):7.3f} '
              f'{st:7.0f}/{sf:<8.0f} {vt:6.0f}/{vf:<7.0f} {ht:6.0f}/{hf:<6.0f}')

    # 颜色分箱的「提升度」：挑出最富集的若干格，看能拿回多少真段
    print('\n=== 颜色格富集（按提升度排序，累计到覆盖 50% 真段）===')
    for s in sorted(n_seg):
        T = cnt[s][0]
        F = cnt[s][1]
        tot_t, tot_f = T.sum(), F.sum()
        if tot_t + tot_f == 0:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            lift = (T / max(1e-9, tot_t)) / np.maximum(1e-9, (T + F) / (tot_t + tot_f))
        order = np.argsort(-lift.ravel())
        flat_t, flat_f = T.ravel(), F.ravel()
        cum_t = cum_f = 0
        used = 0
        for idx in order:
            if flat_t[idx] == 0:
                continue
            cum_t += flat_t[idx]
            cum_f += flat_f[idx]
            used += 1
            if cum_t >= 0.5 * tot_t:
                break
        pur = cum_t / max(1, cum_t + cum_f)
        print(f'  {s:10s} 基线 P {tot_t/(tot_t+tot_f):.3f}  '
              f'用 {used:4d} 个颜色格覆盖 {cum_t/max(1,tot_t)*100:5.1f}% 真段 -> P {pur:.3f}  '
              f'（提升 {pur/max(1e-9,tot_t/(tot_t+tot_f)):.2f}x）')

    print('\n=== 长度门槛（真/假 在 len<15 与 len>=15 的比例）===')
    for s in sorted(n_seg):
        a = sev[s]
        if a.sum() == 0:
            continue
        print(f'  {s:10s} 真: <15 {int(a[0,0]):6d} >=15 {int(a[0,1]):6d}   '
              f'假: <15 {int(a[1,0]):6d} >=15 {int(a[1,1]):6d}   '
              f'len>=15 的纯度 {a[0,1]/max(1,a[0,1]+a[1,1]):.3f}')


if __name__ == '__main__':
    main()
