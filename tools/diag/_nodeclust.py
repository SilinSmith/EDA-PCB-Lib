"""量 pskel 的结点聚簇尺寸 —— 验证「长斜线 = 跨大聚簇的隐式直连」假设。"""
import math
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from pipeline.pskel import WireExtractor, build_graph, neighbours, zhang_suen  # noqa: E402
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "0101"
cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
case = cases[CASE]
bgr = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
ex = WireExtractor()
thin, ink, chroma = ex.foreground(bgr)
skel = zhang_suen(thin, max_iter=ex.max_thin_iter)

pts, deg, node_px, raw = build_graph(skel)
node_list = sorted(node_px)
pos = {p: i for i, p in enumerate(node_list)}
parent = list(range(len(node_list)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


nset = set(node_list)
for p in node_list:
    for q in neighbours(nset, p):
        if q in pos:
            ra, rb = find(pos[p]), find(pos[q])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

clusters = {}
for p in node_list:
    clusters.setdefault(find(pos[p]), []).append(p)

sizes = []
for root, members in clusters.items():
    rs = [m[0] for m in members]
    cs = [m[1] for m in members]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    sizes.append((len(members), w, h, min(cs), min(rs), max(cs), max(rs)))

sizes.sort(key=lambda t: -math.hypot(t[1], t[2]))
print(f"{CASE}  骨架非零 {int((skel > 0).sum())}  结点像素 {len(node_list)}  聚簇 {len(clusters)}")
print(f"\n按包围盒对角线排序的前 12 个聚簇：")
print(f"  {'成员':>7s} {'w':>6s} {'h':>6s}  {'bbox(x1,y1,x2,y2)':>34s} {'对角':>8s}")
for n, w, h, x1, y1, x2, y2 in sizes[:12]:
    print(f"  {n:7d} {w:6d} {h:6d}  ({x1:5d},{y1:5d},{x2:5d},{y2:5d}) {math.hypot(w,h):8.1f}")

big = [s for s in sizes if math.hypot(s[1], s[2]) >= 50]
print(f"\n对角 >=50px 的聚簇：{len(big)} 个")
tot = sum(s[0] for s in sizes)
bigm = sum(s[0] for s in big)
print(f"它们占的结点像素 {bigm}/{tot} = {bigm/max(1,tot):.1%}")

# 用聚簇质心一起看：有没有一条 45 度跳线
print("\n对角 >=50px 的聚簇明细：")
for n, w, h, x1, y1, x2, y2 in big[:20]:
    print(f"  n={n:6d}  ({x1},{y1})-({x2},{y2})  w{w} h{h}")
