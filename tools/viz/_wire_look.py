"""看一眼导线的真相：原图 + 提取骨架段（绿） vs 原图 + GT 网表线段（红）。
用法: python work\_wire_look.py [case ...]
不带参数时自动挑：jlc 最差、kicad 0001、datasheet 最好、总体最好。
"""
import json
import os
import sys
import glob
from pathlib import Path

import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'Y:\workspace\EDA\eda-pcb')
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

OUT = os.environ.get('STAGE1_OUT', r'Y:\workspace\EDA\work\stage1_all2')
PREV = r'Y:\workspace\EDA\work\preview'


def gt_segs(case, H):
    gt = json.load(open(case.json_path, encoding='utf-8'))
    out = []
    for net in gt['nets'].values():
        for pts in (net.get('edges') or {}).values():
            for i in range(len(pts) - 1):
                out.append(([float(pts[i]['x']), H - float(pts[i]['y'])],
                            [float(pts[i + 1]['x']), H - float(pts[i + 1]['y'])]))
    return out


def render(case_id, cases, tag):
    f = os.path.join(OUT, f'{case_id}.stage1.json')
    if not os.path.exists(f):
        print('  skip', case_id)
        return None
    d = json.load(open(f, encoding='utf-8'))
    case = cases.get(case_id)
    if case is None:
        print('  no gt', case_id)
        return None
    img = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    g = gt_segs(case, H)

    panels = []
    for segs, colour, label in ((d['wire_detail']['segments'], (0, 255, 0), 'extracted'),
                                (g, (0, 0, 255), 'GT')):
        c = img.copy()
        for p, q in segs:
            cv2.line(c, (int(round(p[0])), int(round(p[1]))),
                     (int(round(q[0])), int(round(q[1]))), colour, 2, cv2.LINE_AA)
        cv2.putText(c, f'{case_id} {label}: {len(segs)} segs', (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(c, f'{case_id} {label}: {len(segs)} segs', (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        panels.append(c)

    # 左右拼，再压到宽 <= 1600
    both = np.concatenate([cv2.resize(p, (W // 2, H // 2)) for p in panels], axis=1)
    if both.shape[1] > 1700:
        s = 1700 / both.shape[1]
        both = cv2.resize(both, (int(both.shape[1] * s), int(both.shape[0] * s)))
    out = os.path.join(PREV, f'wire_{tag}.png')
    cv2.imwrite(out, both)
    print(f'  {case_id:12s} {tag:16s} extracted {len(d["wire_detail"]["segments"]):6d}  '
          f'GT {len(g):5d}  -> {out}')
    return out


def main():
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    # 算每张的严格覆盖率，用来挑样本
    sys.path.insert(0, r'Y:\workspace\EDA\work')
    args = sys.argv[1:]
    if args:
        for a in args:
            render(a, cases, a)
        return

    for cid, tag in (('0101', 'jlc_bad'), ('0107', 'jlc_bad2'), ('0001', 'kicad'),
                     ('0052', 'good'), ('0050', 'altium')):
        render(cid, cases, tag)


if __name__ == '__main__':
    main()
