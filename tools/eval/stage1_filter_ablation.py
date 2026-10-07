# -*- coding: utf-8 -*-
"""导线过滤器消融：每一道过滤对「覆盖 F1」和「连通性」各值多少。

用户 m04510 授权的下一步：判明漏掉的那截线是被哪一道过滤删的。

做法不是去 wires.py 里逐段打标签（occlude/erase 在骨架化**之前**作用于图像，
本来就没有「被删的段」这个对象），而是**逐道开关后重测**：

  raw          什么都不开（裸骨架）
  +occ         遮挡已检出元件框
  +erase       再涂掉文字框
  +axis        max_axis_dev=12
  +frame       drop_frame
  full         min_extent=80 + rescue_pad=8（== 线上管线的配置）
  full-no-X    从 full 里单独去掉某一道

同批 case 上两个尺子一起报：
  覆盖 F1  —— GT 每条边的 16 个采样点都要有预测段在 tol 内（stage1_wires_eval 口径）
  连通 R   —— 同一 GT net 的引脚对，在预测段的并查集里是否同分量（stage1_conn_eval 口径）
两个尺子会打架：一道过滤可能救 F1、杀连通性，那正是要找的东西。

用法：
  python stage1_filter_ablation.py --cases 60 --workers 8
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = r'Y:\workspace\EDA\eda-pcb'
for p in (EDA, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage1.wires import extract_wires                      # noqa: E402
from schematic_parser.data import discover                  # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs, sample_polyline   # noqa: E402
from stage1_conn_eval import (build_groups, gt_pins_and_nets, gt_edge_segs,     # noqa: E402
                              _dist_pts_to_segs)

DATA_ROOT = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'
STAGE1 = r'Y:\workspace\EDA\work\stage1_all11'
TOL_COV = 5.0          # 覆盖判据容差
TOL_CONN = 1.0         # 连通并表容差
PIN_TOL = 6.0

ARMS = [
    ('raw',            dict()),
    ('+occ',           dict(occ=True)),
    ('+occ+erase',     dict(occ=True, erase=True)),
    ('+...+axis',      dict(occ=True, erase=True, axis=True)),
    ('+...+frame',     dict(occ=True, erase=True, axis=True, frame=True)),
    ('full',           dict(occ=True, erase=True, axis=True, frame=True, extent=True)),
    ('full-no-occ',    dict(occ=False, erase=True, axis=True, frame=True, extent=True)),
    ('full-no-erase',  dict(occ=True, erase=False, axis=True, frame=True, extent=True)),
    ('full-no-axis',   dict(occ=True, erase=True, axis=False, frame=True, extent=True)),
    ('full-no-frame',  dict(occ=True, erase=True, axis=True, frame=False, extent=True)),
    ('full-no-extent', dict(occ=True, erase=True, axis=True, frame=True, extent=False)),
    # 线上配置是「骨架 + 贪心射线」。射线跑在**原始墨迹**上，不受遮挡/涂字影响，
    # 所以它可能把被过滤删掉的那截中段补回来 —— 必须一起量。
    ('full+rays',          dict(occ=True, erase=True, axis=True, frame=True, extent=True, rays=True)),
    ('full-no-occ+rays',   dict(occ=False, erase=True, axis=True, frame=True, extent=True, rays=True)),
    ('full-no-erase+rays', dict(occ=True, erase=False, axis=True, frame=True, extent=True, rays=True)),
    ('full-no-axis+rays',  dict(occ=True, erase=True, axis=False, frame=True, extent=True, rays=True)),
    ('raw+rays',           dict(rays=True)),
    # 用户 m04510 要的正确操作化：射线碰到第一个锚点就截断，锚点 = 元件框 + 文字框。
    ('full+rays-clip',     dict(occ=True, erase=True, axis=True, frame=True, extent=True,
                                rays=True, rays_clip=True)),
    ('full+rays-anchor',   dict(occ=True, erase=True, axis=True, frame=True, extent=True,
                                rays=True, rays_clip=True, rays_anchor=True)),
    ('raw+rays-anchor',    dict(rays=True, rays_clip=True, rays_anchor=True)),
]


def cfg_kwargs(arm, occ_boxes, txt_boxes):
    kw = {}
    if arm.get('occ'):
        kw['occlude'] = occ_boxes
    if arm.get('erase'):
        kw['erase'] = txt_boxes
        kw['erase_pad'] = 1.0
    if arm.get('axis'):
        kw['max_axis_dev'] = 12.0
    if arm.get('frame'):
        kw['drop_frame'] = True
    if arm.get('extent'):
        kw['min_extent'] = 80.0
        kw['rescue_pad'] = 8.0
    return kw


def conn_stats(psegs, pins, nets):
    """连通性的 TP / FN / 负样本误并 —— 全部精确计数，O(引脚数)。

    只有「同 net 且同分量」是 TP。**必须同时报负样本误并**：把整张图并成一个
    分量可以把 R 刷到 1.0，那不是本事。负样本误并用「每个分量里有多少对异 net
    引脚」精确算：sum_r C(m_r,2) - (同分量里的正样本对数)。
    """
    P = len(PIN_MAP)
    if not len(psegs):
        return 0, 0, 0, 0
    rootmap = build_groups(psegs, TOL_CONN)
    # 每个引脚 -> 分量 id（接不上的各自单例）
    # **注意**：不能再用 `stage1_wires_eval.dist_pts_to_segs` —— 它返回的是
    # `(M,)`「每点到最近那一条段的距离」，不是 (M,N) 矩阵，于是 `np.argmin(d)`
    # 恒等于 0，**所有引脚都被绑到第 0 条段**，整张图塌成一个分量。
    # 症状：连通 R 恰好等于「两个引脚都绑上了」的比例（0.77 / 0.56），负样本误并 77%。
    keys = list(PIN_MAP)
    pts = np.asarray([PIN_MAP[k] for k in keys], np.float64).reshape(-1, 2)
    DM = _dist_pts_to_segs(pts, psegs)
    jm = np.argmin(DM, axis=1)
    dm = DM[np.arange(len(keys)), jm]
    pin_root = {}
    for k, j, d in zip(keys, jm, dm):
        pin_root[k] = int(rootmap[int(j)]) if float(d) <= PIN_TOL else ('solo', k)

    m = {}                       # 分量 -> 引脚数（全体引脚）
    for k, r in pin_root.items():
        m[r] = m.get(r, 0) + 1
    neg_same = sum(v * (v - 1) // 2 for v in m.values())

    tp = 0
    pos_tot = 0
    for _nid, keys in nets:
        buckets = {}
        for k in keys:
            if k not in pin_root:
                continue
            r = pin_root[k]
            buckets[r] = buckets.get(r, 0) + 1
        for v in buckets.values():
            tp += v * (v - 1) // 2
        n = sum(buckets.values())
        pos_tot += n * (n - 1) // 2
    neg_same -= tp
    neg_tot = P * (P - 1) // 2 - pos_tot
    return tp, pos_tot - tp, neg_same, neg_tot


PIN_MAP = {}


def job(spec):
    global PIN_MAP
    case_id, png, json_path, arm_name, arm = spec
    sp = os.path.join(STAGE1, case_id + '.stage1.json')
    if not os.path.exists(sp) or not os.path.exists(png):
        return None
    sd = json.load(open(sp, encoding='utf-8'))
    H = float(sd['image']['h'])
    occ_boxes = [c['bbox_img'] for c in sd['components']]
    txt_boxes = [t['bbox_img'] for t in (sd.get('texts') or [])]
    gdoc = json.load(open(json_path, encoding='utf-8'))

    bgr = cv2.imread(png, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    w, _sec = extract_wires(bgr, **cfg_kwargs(arm, occ_boxes, txt_boxes))
    if arm.get('rays'):
        from stage1 import pintrace as ray_mod
        ink = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < ray_mod.INK)
        if arm.get('rays_clip'):
            anchors = list(occ_boxes) + list(txt_boxes)
            w['segments'] = list(w['segments']) + ray_mod.trace_from_boxes(
                ink, occ_boxes, anchors=anchors, anchor_dist=24.0,
                clip_anchors=True, require_anchor=bool(arm.get('rays_anchor')))
        else:
            w['segments'] = list(w['segments']) + ray_mod.trace_from_boxes(ink, occ_boxes)
    psegs = seg_arrays(w['segments'])

    ge = gt_edge_segs(gdoc, H)
    pins, nets = gt_pins_and_nets(gdoc, H)
    PIN_MAP = dict(pins)

    # ---- 覆盖 F1 ----
    tp = fn = fp = 0
    if len(ge):
        for (x1, y1, x2, y2) in ge:
            pts = sample_polyline((x1, y1), (x2, y2), 16)
            d = dist_pts_to_segs(pts, psegs) if len(psegs) else np.full(16, np.inf)
            if float(np.max(d)) <= TOL_COV:
                tp += 1
            else:
                fn += 1
    if len(psegs):
        mid = np.stack([(psegs[:, 0] + psegs[:, 2]) / 2.0,
                        (psegs[:, 1] + psegs[:, 3]) / 2.0], axis=1)
        if len(ge):
            fp = int((dist_pts_to_segs(mid, ge) > TOL_COV).sum())
        else:
            fp = len(psegs)

    ct, cf, ns, nt = conn_stats(psegs, pins, nets)
    lgt = float(np.sum(np.hypot(ge[:, 2] - ge[:, 0], ge[:, 3] - ge[:, 1]))) if len(ge) else 0.0
    lpd = float(np.sum(np.hypot(psegs[:, 2] - psegs[:, 0], psegs[:, 3] - psegs[:, 1]))) if len(psegs) else 0.0
    return dict(case=case_id, arm=arm_name, n_seg=len(psegs), tp=tp, fp=fp, fn=fn,
                ct=ct, cf=cf, ns=ns, nt=nt, lpd=lpd, lgt=lgt)


def main():
    args = sys.argv[1:]

    def opt(name, dflt):
        return args[args.index(name) + 1] if name in args else dflt

    n_cases = int(opt('--cases', '60'))
    workers = int(opt('--workers', '8'))
    arms = ARMS
    if '--arms' in args:
        want = set(opt('--arms', '').split(','))
        arms = [a for a in ARMS if a[0] in want]

    cases = sorted(discover(DATA_ROOT), key=lambda c: c.case_id)
    step = max(1, len(cases) // n_cases)
    cases = cases[::step][:n_cases]
    print(f'case 数 {len(cases)} 臂数 {len(arms)} workers {workers}', flush=True)

    specs = [(c.case_id, c.png_path, c.json_path, name, arm)
             for c in cases for name, arm in arms]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(job, specs, chunksize=1)):
            if res:
                rows.append(res)
            if (k + 1) % 40 == 0:
                print(f'  {k+1}/{len(specs)}', flush=True)

    agg = {}
    for r in rows:
        a = agg.setdefault(r['arm'], dict(n=0, tp=0, fp=0, fn=0, ct=0, cf=0, ns=0, nt=0,
                                          lpd=0.0, lgt=0.0, nsg=0))
        a['n'] += 1
        a['tp'] += r['tp']; a['fp'] += r['fp']; a['fn'] += r['fn']
        a['ct'] += r['ct']; a['cf'] += r['cf']
        a['ns'] += r['ns']; a['nt'] += r['nt']
        a['lpd'] += r['lpd']; a['lgt'] += r['lgt']; a['nsg'] += r['n_seg']

    print()
    print('| 臂 | 段数 | 总长比 | 覆盖 R | 覆盖 P | 覆盖 F1 | 连通 R | 连通 P | 连通 F1 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, _ in arms:
        a = agg.get(name)
        if not a:
            continue
        cr = a['tp'] / float(a['tp'] + a['fn']) if (a['tp'] + a['fn']) else 0.0
        cp = a['tp'] / float(a['tp'] + a['fp']) if (a['tp'] + a['fp']) else 0.0
        c1 = 2 * cp * cr / (cp + cr) if (cp + cr) else 0.0
        kr = a['ct'] / float(a['ct'] + a['cf']) if (a['ct'] + a['cf']) else 0.0
        kp = 1.0 - (a['ns'] / float(a['nt'])) if a['nt'] else 0.0
        k1 = 2 * kp * kr / (kp + kr) if (kp + kr) else 0.0
        ratio = a['lpd'] / a['lgt'] if a['lgt'] else 0.0
        print(f"| {name} | {a['nsg']} | {ratio:.3f} | {cr:.4f} | {cp:.4f} | "
              f"{c1:.4f} | {kr:.4f} | {kp:.4f} | **{k1:.4f}** |")

    out = os.path.join(HERE, 'stage1_filter_ablation.json')
    json.dump(rows, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
