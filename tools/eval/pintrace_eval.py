"""贪心射线 vs 全局骨架 vs 两者取并 —— 同一把尺子的对照。

用法:
    python work\\pintrace_eval.py --skel Y:\workspace\EDA\work\stage1_all9 ^
                                 --rays Y:\workspace\EDA\work\pintrace_all [--tol 5] [--causes]

判据与 stage1_wires_eval.py 完全一致：
    recall    : 每条 GT 线段上的 16 个采样点，最差距离 <= tol 才算覆盖
    precision : 每条预测线段的 16 个采样点，最差距离 <= tol 才算贴住
"""
import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'Y:\workspace\EDA\eda-pcb')
sys.path.insert(0, r'Y:\workspace\EDA\work')

from schematic_parser.data import discover, DEFAULT_DATA_ROOT          # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs, sample_polyline  # noqa: E402


def gt_img_segs(gt, H):
    out = []
    for net in gt['nets'].values():
        for pts in (net.get('edges') or {}).values():
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                out.append([[float(pts[i]['x']), H - float(pts[i]['y'])],
                            [float(pts[i + 1]['x']), H - float(pts[i + 1]['y'])]])
    return out


def worst_per_seg(A, B):
    """A 的每条线段到 B 的最差采样距离（B 为空返回 inf）。"""
    n = A.shape[0]
    if not n:
        return np.zeros(0)
    if not B.shape[0]:
        return np.full(n, np.inf)
    pts, of = [], []
    for i in range(n):
        pts.append(sample_polyline(A[i, :2], A[i, 2:], 16))
        of.append(np.full(16, i))
    pts = np.concatenate(pts, axis=0)
    of = np.concatenate(of)
    d = dist_pts_to_segs(pts, B)
    w = np.zeros(n)
    np.maximum.at(w, of, d)
    return w


def arm_stats(GA, PA, tol):
    """-> (gt_n, cov, pred_n, pcov, pred_len, gt_len)"""
    if not GA.shape[0]:
        return 0, 0, 0, 0, 0.0, 0.0
    gw = worst_per_seg(GA, PA)
    cov = int((gw <= tol).sum())
    if PA.shape[0]:
        rw = worst_per_seg(PA, GA)
        pcov = int((rw <= tol).sum())
        lp = float(np.hypot(PA[:, 2] - PA[:, 0], PA[:, 3] - PA[:, 1]).sum())
    else:
        pcov, lp = 0, 0.0
    lg = float(np.hypot(GA[:, 2] - GA[:, 0], GA[:, 3] - GA[:, 1]).sum())
    return GA.shape[0], cov, PA.shape[0], pcov, lp, lg


def f1_of(gt, cov, pred, pcov):
    R = cov / max(1, gt)
    P = pcov / max(1, pred)
    return R, P, (2 * P * R / max(1e-9, P + R))


def classify(segs_bad, GA_keep, ink, tmask, cboxes, H, W):
    """给没被覆盖的 GT 线段归类。segs_bad: (n,4) 图像坐标。"""
    cnt = collections.Counter()
    for i in range(segs_bad.shape[0]):
        x1, y1, x2, y2 = segs_bad[i]
        t = np.linspace(0.05, 0.95, 12)[:, None]
        px = x1 * (1 - t[:, 0]) + x2 * t[:, 0]
        py = y1 * (1 - t[:, 0]) + y2 * t[:, 0]
        xi = np.clip(np.round(px).astype(int), 0, W - 1)
        yi = np.clip(np.round(py).astype(int), 0, H - 1)
        on_ink = ink[yi, xi].mean()
        in_txt = tmask[yi, xi].mean()
        in_cmp = False
        for b in cboxes:
            bx0, by0, bx1, by1 = b
            if np.any((px >= bx0) & (px <= bx1) & (py >= by0) & (py <= by1)):
                in_cmp = True
                break
        if in_txt > 0.30:
            cnt['in_text'] += 1
        elif in_cmp:
            cnt['in_comp'] += 1
        elif on_ink < 0.20:
            cnt['no_ink'] += 1
        else:
            cnt['other'] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skel', default=r'Y:\workspace\EDA\work\stage1_all9')
    ap.add_argument('--rays', default=r'Y:\workspace\EDA\work\pintrace_all')
    ap.add_argument('--tol', type=float, default=5.0)
    ap.add_argument('--causes', action='store_true', help='对骨架没覆盖的 GT 边做归因，看并集救回多少')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(args.skel, '*.json')))
    if args.limit:
        files = files[: args.limit]

    tot = collections.defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0])
    src_stat = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0]))
    cause_skel = collections.Counter()
    cause_union = collections.Counter()
    n_eval = 0

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
        W = float(d['image']['w'])
        gt = json.load(open(case.json_path, encoding='utf-8'))
        GA = seg_arrays(gt_img_segs(gt, H))
        SA = seg_arrays(d['wire_detail']['segments'])

        rf = os.path.join(args.rays, f'{cid}.json')
        RA = np.zeros((0, 4))
        if os.path.exists(rf):
            RA = seg_arrays(json.load(open(rf, encoding='utf-8'))['segments'])
        if SA.shape[0] and RA.shape[0]:
            UA = np.concatenate([SA, RA], axis=0)
        else:
            UA = SA if SA.shape[0] else RA

        if not GA.shape[0]:
            continue
        n_eval += 1
        png = Path(case.png_path).name.lower().replace(' ', '')
        src = ('kicad' if 'kicad' in png else 'altium' if 'altium' in png
               else 'datasheet' if 'datasheet' in png else 'jlc' if 'jlc' in png else 'other')

        for tag, PA in (('skel', SA), ('rays', RA), ('union', UA)):
            st = arm_stats(GA, PA, args.tol)
            for i in range(6):
                tot[tag][i] += st[i]
                src_stat[src][tag][i] += st[i]

        if args.causes:
            from cvparts.textbox import detect_boxes, text_mask
            img = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ink = (gray < 170).astype(np.uint8)
            tb = detect_boxes(img)
            tmask = text_mask(ink.shape, tb, pad=1.0)
            cboxes = [[c['bbox_img'][0], c['bbox_img'][1], c['bbox_img'][2], c['bbox_img'][3]]
                      for c in d['components']]
            wsz = worst_per_seg(GA, SA)
            bad = GA[wsz > args.tol]
            if bad.shape[0]:
                c1 = classify(bad, None, ink, tmask, cboxes, int(H), int(W))
                cause_skel.update(c1)
                wu = worst_per_seg(bad, UA)
                # 归因只看「并集也没救回来」的
                still = bad[wu > args.tol]
                if still.shape[0]:
                    c2 = classify(still, None, ink, tmask, cboxes, int(H), int(W))
                    cause_union.update(c2)

    print(f'\n评测 {n_eval} 张   tol={args.tol}px')
    print(f'{"臂":6s} {"GT":>7s} {"覆盖":>7s} {"R":>7s} {"预测":>7s} {"贴住":>7s} {"P":>7s} {"F1":>7s} {"L/Lgt":>7s}')
    for tag, k in (('骨架', 'skel'), ('射线', 'rays'), ('并集', 'union')):
        g, c, p, pc, lp, lg = tot[k]
        R, P, F = f1_of(g, c, p, pc)
        print(f'{tag:6s} {g:7d} {c:7d} {R:7.4f} {p:7d} {pc:7d} {P:7.4f} {F:7.4f} {lp/max(1.0,lg):7.3f}')

    print('\n=== 按来源 ===')
    for s in sorted(src_stat):
        row = []
        for tag in ('skel', 'rays', 'union'):
            g, c, p, pc, lp, lg = src_stat[s][tag]
            R, P, F = f1_of(g, c, p, pc)
            row.append(f'{tag} R{R:.3f}/P{P:.3f}/F{F:.3f}')
        print(f'  {s:10s} ' + '  '.join(row))

    if args.causes and cause_skel:
        print('\n=== 骨架没覆盖的 GT 边：归因（并集能救回多少）===')
        tot_bad = sum(cause_skel.values())
        print(f'  {"原因":12s} {"骨架漏":>7s} {"并集后仍漏":>10s} {"救回":>7s} {"救回率":>7s}')
        for k in ('in_text', 'in_comp', 'no_ink', 'other'):
            a = cause_skel.get(k, 0)
            b = cause_union.get(k, 0)
            print(f'  {k:12s} {a:7d} {b:10d} {a-b:7d} {(a-b)/max(1,a):7.3f}')
        print(f'  {"合计":12s} {tot_bad:7d} {sum(cause_union.values()):10d}')


if __name__ == '__main__':
    main()
