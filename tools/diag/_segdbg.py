"""§5.109 后续取证：case 1 的 x≈1432 竖线上，骨架→raw walks→polylines→segments 到底哪一步把
连续骨架切成了「11px 小岛 + 18~19px 洞」（骨架与细笔画在该处都是连续的，见 work/_xmap.py）。

用法：python -X utf8 work\_segdbg.py [case] [x0 x1 y0 y1]
坐标全部是**图像帧（y 向下）**，与 work/_xmap.py 打印的一致。
"""
import sys
import cv2

from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from pipeline import pskel                      # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover   # noqa: E402

case = sys.argv[1] if len(sys.argv) > 1 else "1"
win = [int(v) for v in sys.argv[2:6]] if len(sys.argv) > 5 else [1428, 1436, 220, 330]
x0, x1, y0, y1 = win

cases = {}
for _c in discover(DEFAULT_DATA_ROOT):
    _cid = str(getattr(_c, "case_id", "") or Path(getattr(_c, "json_path", "")).parent.name)
    cases[_cid] = _c
img_path = str(cases[case].png_path)
print("case", case, "image", img_path)
img = cv2.imread(str(img_path))
H, W = img.shape[:2]
print("img %dx%d" % (W, H))

ex = pskel.WireExtractor(ink_mode="adaptive", thin_w=2.6, use_colour=True)
res = ex.extract(img)
print("pskel: polylines=%d segments=%d endpoints=%d junctions=%d nodes=%d branch_paths=%d"
      % (len(res.polylines), len(res.segments), len(res.endpoints),
         len(res.junctions), res.n_nodes, res.n_branch_paths))


def spans(items, label):
    out = []
    for it in items:
        xs = [p[0] for p in it]
        ys = [p[1] for p in it]
        if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
            continue
        out.append((min(ys), max(ys), min(xs), max(xs), len(it)))
    out.sort()
    print("== %s 在窗口内 %d 条" % (label, len(out)))
    for a, b, cx, dx, n in out:
        print("   y[%7.1f,%7.1f] 长%6.1f  x[%.1f,%.1f]  %d 点" % (a, b, b - a, cx, dx, n))


spans(res.polylines, "pskel.polylines")
spans(res.segments, "pskel.segments")

pts, deg, node_px, raw = pskel.build_graph(res.skel)
node_win = [p for p in node_px if x0 <= p[1] <= x1 and y0 <= p[0] <= y1]
print("== raw walks: %d 条；窗口内 node_px %d 个" % (len(raw), len(node_win)))
wout = []
for w in raw:
    xs = [c for r, c in w]
    ys = [r for r, c in w]
    if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
        continue
    wout.append((min(ys), max(ys), min(xs), max(xs), len(w)))
wout.sort()
for a, b, cx, dx, n in wout:
    print("   y[%7.1f,%7.1f] 长%6.1f  x[%.1f,%.1f]  %d 点" % (a, b, b - a, cx, dx, n))

# 竖直骨架在窗口内的连续性
col = res.skel[y0:y1 + 1, x0:x1 + 1]
rows_with = [y0 + i for i in range(col.shape[0]) if col[i].any()]
print("窗口内骨架有像素的行数 = %d / %d" % (len(rows_with), y1 - y0 + 1))
