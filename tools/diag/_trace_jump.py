"""复刻 pskel._trace 的并链循环并加日志：找出哪一次 append 产生了远距离跳变。"""
import math
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from pipeline.pskel import (WireExtractor, build_graph, neighbours,  # noqa: E402
                            _dir_at, rdp, snap_polyline, split_segments)

cases_png = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)\200_train_cases\0101\0101-jlc.png"
bgr = cv2.imread(cases_png, cv2.IMREAD_COLOR)
ex = WireExtractor()
thin, ink, chroma = ex.foreground(bgr)
from pipeline.pskel import zhang_suen  # noqa: E402
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
node_of_px, nodes = {}, []
import pipeline.pskel as P  # noqa: E402
Node = P.Node
Path = P.Path
HalfEdge = P.HalfEdge
for members in clusters.values():
    n = Node(idx=len(nodes), x=float(np.mean([m[1] for m in members])),
             y=float(np.mean([m[0] for m in members])))
    nodes.append(n)
    for m in members:
        node_of_px[m] = n.idx

paths, half = [], []
for walk in raw:
    if len(walk) < 2 or walk[0] not in node_of_px or walk[-1] not in node_of_px:
        continue
    xy = [(float(c), float(r)) for r, c in walk]
    L = math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
    p = Path(idx=len(paths), n0=node_of_px[walk[0]], n1=node_of_px[walk[-1]], pts=xy, length=L)
    paths.append(p)
keep = [p for p in paths if p.length >= ex.min_branch_len]
for p in keep:
    he0 = HalfEdge(path=p.idx, end=0, node=p.n0)
    he1 = HalfEdge(path=p.idx, end=1, node=p.n1)
    he0.partner, he1.partner = len(half) + 1, len(half)
    half.extend([he0, he1])

node_slots = {}
for hi, he in enumerate(half):
    node_slots.setdefault(he.node, []).append(hi)
for nid, hs in node_slots.items():
    n = nodes[nid]
    for hi in hs:
        he = half[hi]
        walk = [(int(y), int(x)) for x, y in paths[he.path].pts]
        d, _ = _dir_at(walk, he.end)
        n.slots.append(hi)
        n.dirs.append(d)

for nid, hs in node_slots.items():
    n = nodes[nid]
    k = len(hs)
    if k == 0 or k == 1:
        n.kind = 'E'; continue
    if k == 2:
        n.kind = 'C'; n.pair[0] = 1; n.pair[1] = 0; continue
    cost = np.full((k, k), 1e6, np.float64)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            vi, vj = n.dirs[i], n.dirs[j]
            cosv = -(vi[0] * vj[0] + vi[1] * vj[1])
            cost[i, j] = (1.0 - cosv) * 100.0
    from scipy.optimize import linear_sum_assignment
    if k % 2 == 0:
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            n.pair[int(i)] = int(j)
    else:
        big = np.full((k, k + 1), 1e6, np.float64)
        big[:, :k] = cost
        for i in range(k):
            big[i, k] = 56.0
        r, c = linear_sum_assignment(big)
        for i, j in zip(r, c):
            if j < k:
                n.pair[int(i)] = int(j)
    n.kind = 'T' if k == 3 else 'X'

print(f"paths {len(paths)} keep {len(keep)} half {len(half)} nodes {len(nodes)}")
print(f"各结点簇的槽位数分布: ", end="")
from collections import Counter  # noqa: E402
print(sorted(Counter(len(v) for v in node_slots.values()).items()))

# --- 并链并记录跳变
consumed = set()
jumps = []
for he in half:
    if he.partner in consumed:
        continue
    chain = []
    cur = he
    guard = 0
    while guard < 10000:
        guard += 1
        if cur.partner in consumed:
            break
        consumed.add(cur.partner)
        consumed.add(half[cur.partner].partner if half[cur.partner].partner >= 0 else -1)
        p = paths[cur.path]
        seg = p.pts if cur.end == 1 else list(reversed(p.pts))
        if chain:
            d = math.hypot(seg[0][0] - chain[-1][0], seg[0][1] - chain[-1][1])
            if d > 30.0:
                jumps.append((d, cur.path, cur.end, tuple(chain[-1]), tuple(seg[0]),
                              tuple(p.pts[0]), tuple(p.pts[-1]), p.n0, p.n1, cur.node))
            chain.extend(seg[1:])
        else:
            chain.extend(seg)
        exit_he = half[cur.partner]
        nid = exit_he.node
        n = nodes[nid]
        slot_pos = None
        for si, hi in enumerate(n.slots):
            if hi == cur.partner:
                slot_pos = si
                break
        if slot_pos is None:
            break
        nxt_pos = n.pair.get(slot_pos)
        if nxt_pos is None:
            break
        nxt_hi = n.slots[nxt_pos]
        if half[nxt_hi] is cur:
            break
        if consumed and nxt_hi in consumed:
            break
        cur = half[nxt_hi]
        if cur.partner in consumed:
            break
        consumed.add(nxt_hi)

jumps.sort(key=lambda r: -r[0])
print(f"\n并链时 >30px 的跳变：{len(jumps)} 次")
for d, pi, end, prev, start, p0, p1, n0, n1, entry_node in jumps[:15]:
    print(f"  d {d:7.1f}  path#{pi} end={end}  上一链尾 ({prev[0]:7.1f},{prev[1]:7.1f}) "
          f"-> seg[0] ({start[0]:7.1f},{start[1]:7.1f})")
    print(f"        path 端点 pts[0]=({p0[0]:7.1f},{p0[1]:7.1f}) pts[-1]=({p1[0]:7.1f},{p1[1]:7.1f})  "
          f"n0={n0} n1={n1}  入口结点={entry_node} ({nodes[entry_node].x:.1f},{nodes[entry_node].y:.1f})")
