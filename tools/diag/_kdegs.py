"""量出 pskel._trace 里「每个节点的分支数 k」的分布 —— 决定 k^2 双循环代价。"""
import sys, time, math
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
import cv2, numpy as np
from schematic_parser import data as D
from pipeline import pskel

cid = sys.argv[1] if len(sys.argv) > 1 else "0107"
case = [c for c in D.discover(D.DEFAULT_DATA_ROOT) if c.case_id == cid][0]
bgr = cv2.imread(str(case.png_path))
print(f"{cid} {bgr.shape[1]}x{bgr.shape[0]}")

ext = pskel.WireExtractor()
fg, _ink, _chroma = ext.foreground(bgr)
skel = pskel.zhang_suen(fg, max_iter=ext.max_thin_iter)
print("骨架像素", int(skel.sum()))

t = time.time()
pts, deg, node_px, raw = pskel.build_graph(skel)
print(f"build_graph {time.time()-t:.2f}s  原始 walk {len(raw)}  结点像素 {len(node_px)}")

# 复刻 _trace 的并查集折叠
node_list = sorted(node_px)
pos = {p: i for i, p in enumerate(node_list)}
parent = list(range(len(node_list)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
for p in node_list:
    for q in pskel.neighbours(set(node_list), p):
        if q in pos:
            ra, rb = find(pos[p]), find(pos[q])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
clusters = {}
for p in node_list:
    clusters.setdefault(find(pos[p]), []).append(p)
node_of_px = {}
for i, members in enumerate(clusters.values()):
    for m in members:
        node_of_px[m] = i
print("折叠后结点数", len(clusters))

# keep 过滤（min_branch_len=3.0）
node_slots = defaultdict(list)
kept = 0
for walk in raw:
    if len(walk) < 2 or walk[0] not in node_of_px or walk[-1] not in node_of_px:
        continue
    xy = [(float(c), float(r)) for r, c in walk]
    L = math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
    if L < ext.min_branch_len:
        continue
    kept += 1
    node_slots[node_of_px[walk[0]]].append(1)
    node_slots[node_of_px[walk[-1]]].append(1)
print("keep 后的分支数（= len(half)/2）", kept)

ks = np.array([len(v) for v in node_slots.values()], float)
print(f"\n有分支的结点 {len(ks)}   k: max {ks.max():.0f}  p50 {np.percentile(ks,50):.0f} "
      f"p90 {np.percentile(ks,90):.0f} p99 {np.percentile(ks,99):.0f}")
big = ks[ks > 2]
print(f"k>2 的结点 {len(big)} 个   其中 k>10 的 {(big>10).sum()} 个   k>50 的 {(big>50).sum()} 个"
      f"   k>100 的 {(big>100).sum()} 个")
print(f"\n** sum(k^2) over k>2 = {int((big**2).sum()):,} **   <- 这就是那两个嵌套循环的迭代次数")
print(f"   sum(k^2) 最大的 8 个结点：")
for k in sorted(big, reverse=True)[:8]:
    print(f"      k={k:.0f}  k^2={k*k:,.0f}")
