"""§5.109 后续：case 1 x≈1432 上的「小岛」（11px）到底是 `extract_wires` 里哪一步造成的。

pskel 自己给出的 raw walks 是 GT 对齐的（[203,231]/[233,260]/[262,290]/[292,319]），
stage1 产物却只剩每段中间的 11px。这个探针按不同开关跑同一条线，把差别打出来。

用法：python -X utf8 work\_isldbg.py [case] [x0 x1 y0 y1]
"""
import json
import os
import sys

import cv2

from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover   # noqa: E402
from stage1 import wires as W                                   # noqa: E402

case = sys.argv[1] if len(sys.argv) > 1 else "1"
win = [int(v) for v in sys.argv[2:6]] if len(sys.argv) > 5 else [1428, 1436, 200, 340]
x0, x1, y0, y1 = win

cases = {}
for _c in discover(DEFAULT_DATA_ROOT):
    cases[str(getattr(_c, "case_id", "") or Path(getattr(_c, "json_path", "")).parent.name)] = _c
img = cv2.imread(str(cases[case].png_path), cv2.IMREAD_COLOR)
print("case", case, cases[case].png_path, img.shape[1], "x", img.shape[0],
      " DEFAULT_MAX_PIXELS=", getattr(W, "DEFAULT_MAX_PIXELS", None))

# 元件框 / 文字框：直接从既有 stage1 产物里取（pipeline 的 occlude/erase 输入）
s1p = ROOT / "work" / "_fast_out_gsd9" / "runs" / "stage1" / ("%s.stage1.json" % case)
if not s1p.exists():
    cand = sorted(ROOT.glob("work/*/runs/stage1/%s.stage1.json" % case))
    s1p = cand[-1] if cand else None
comps, texts = [], []
if s1p is not None and s1p.exists():
    d = json.load(open(s1p, encoding="utf-8"))
    comps = [c["bbox_img"] for c in d.get("components", [])]
    texts = [t["bbox_img"] for t in d.get("texts", [])]
    print("stage1:", s1p, " comps", len(comps), " texts", len(texts))
    print("--- 窗口内的文本框 / 元件框 (x0,y0,x1,y1) ---")
    for tag_, bs in (("text", texts), ("comp", comps)):
        for b in bs:
            bx1, by1, bx2, by2 = (float(v) for v in b[:4])
            if max(bx1, bx2) < x0 - 40 or min(bx1, bx2) > x1 + 40:
                continue
            if max(by1, by2) < y0 - 40 or min(by1, by2) > y1 + 40:
                continue
            print("   %-4s (%7.1f,%7.1f)-(%7.1f,%7.1f)  h=%5.1f" % (
                tag_, bx1, by1, bx2, by2, abs(by2 - by1)))

PIPE = dict(occlude=comps, erase=texts, erase_pad=1.0, drop_frame=True,
            max_axis_dev=12.0, min_extent=80.0, rescue_pad=8.0)
PLAIN = dict(occlude=None, erase=None, erase_pad=1.0, drop_frame=True,
             max_axis_dev=12.0, min_extent=80.0, rescue_pad=8.0)


def run(tag, kw, **env):
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = str(v)
    try:
        out, _dt = W.extract_wires(img, max_pixels=2_500_000, **kw)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    segs = out.get("segments") or []
    hits = []
    for s in segs:
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
            continue
        if abs(xs[0] - xs[-1]) > 2.5:          # 只挑竖段
            continue
        hits.append((round(min(ys), 1), round(max(ys), 1), round(xs[0], 1),
                     round(max(ys) - min(ys), 1)))
    hits.sort()
    print("== %-30s segs=%4d snaps=%3s tsnap=%3s 竖段 %d 条" % (
        tag, out.get("n_segments") or 0, out.get("n_gap_snaps"),
        out.get("n_text_snaps"), len(hits)))
    for a, b, cx, L in hits:
        print("      y[%7.1f,%7.1f] x=%6.1f L=%5.1f" % (a, b, cx, L))


run("A 不涂(无遮挡/无抹字)", PLAIN)
run("F pipeline 口径(occ+erase)", PIPE)
run("G 只抹字(occlude=[])", dict(PIPE, occlude=None))
run("H 只遮挡(erase=[])", dict(PIPE, erase=None))
run("I pipeline 口径 + SPLIT_JUNC=0", PIPE, STAGE1_SPLIT_JUNCTIONS="0")
run("J pipeline 口径 + GAP_SNAP=0", PIPE, STAGE1_WIRE_SNAP_GAPS="0")
run("K pipeline 口径 + CROSS_FILL=0", PIPE, STAGE1_WIRE_CROSS_FILL="0")
