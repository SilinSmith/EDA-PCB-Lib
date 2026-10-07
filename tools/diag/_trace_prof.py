"""把 pskel._trace 逐段计时，定位真正的瓶颈。用法：python _trace_prof.py 0107"""
import sys, math, time
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
import cv2, numpy as np
from schematic_parser import data as D
from pipeline import pskel
from pipeline.pskel import Node, Path, HalfEdge, _dir_at, build_graph, rdp, snap_polyline, split_segments

cid = sys.argv[1] if len(sys.argv) > 1 else "0107"
case = [c for c in D.discover(D.DEFAULT_DATA_ROOT) if c.case_id == cid][0]
bgr = cv2.imread(str(case.png_path))
ext = pskel.WireExtractor()
T = {}
t0 = time.perf_counter()
thin, ink, has_chroma = ext.foreground(bgr)
T["foreground"] = time.perf_counter() - t0
t0 = time.perf_counter()
skel = pskel.zhang_suen(thin, max_iter=ext.max_thin_iter)
T["zhang_suen"] = time.perf_counter() - t0
print(f"{cid} {bgr.shape[1]}x{bgr.shape[0]}  骨架像素(非零) {int((skel>0).sum()):,}")
for k, v in T.items():
    print(f"  {k:<14} {v:8.2f} s")


def timed_trace(skel):
    t0 = time.perf_counter()
    pts, deg, node_px, raw = build_graph(skel)
    print(f"  A build_graph        {time.perf_counter()-t0:8.2f} s   raw walks {len(raw)}  结点像素 {len(node_px)}")

    t0 = time.perf_counter()
    node_list = sorted(node_px)
    pos = {p: i for i, p in enumerate(node_list)}
    parent = list(range(len(node_list)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    _nset = set(node_list)          # <-- 修复：把循环不变量提到循环外
    for p in node_list:
        for q in pskel.neighbours(_nset, p):
            if q in pos:
                ra, rb = find(pos[p]), find(pos[q])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    clusters = {}
    for p in node_list:
        clusters.setdefault(find(pos[p]), []).append(p)
    node_of_px, nodes = {}, []
    for members in clusters.values():
        n = Node(idx=len(nodes), x=float(np.mean([m[1] for m in members])),
                 y=float(np.mean([m[0] for m in members])))
        nodes.append(n)
        for m in members:
            node_of_px[m] = n.idx
    print(f"  B 折叠并查集         {time.perf_counter()-t0:8.2f} s   结点 {len(nodes)}")

    t0 = time.perf_counter()
    paths, half = [], []
    tot_pts = 0
    for walk in raw:
        if len(walk) < 2 or walk[0] not in node_of_px or walk[-1] not in node_of_px:
            continue
        xy = [(float(c), float(r)) for r, c in walk]
        tot_pts += len(xy)
        L = math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
        paths.append(Path(idx=len(paths), n0=node_of_px[walk[0]], n1=node_of_px[walk[-1]],
                          pts=xy, length=L))
    print(f"  C 建 path            {time.perf_counter()-t0:8.2f} s   paths {len(paths)}  总点数 {tot_pts:,}")

    t0 = time.perf_counter()
    keep = [p for p in paths if p.length >= ext.min_branch_len]
    for p in keep:
        he0 = HalfEdge(path=p.idx, end=0, node=p.n0)
        he1 = HalfEdge(path=p.idx, end=1, node=p.n1)
        he0.partner, he1.partner = len(half) + 1, len(half)
        half.extend([he0, he1])
    print(f"  D keep/half          {time.perf_counter()-t0:8.2f} s   keep {len(keep)}  half {len(half)}")

    t0 = time.perf_counter()
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
    print(f"  E 结点方向           {time.perf_counter()-t0:8.2f} s")

    t0 = time.perf_counter()
    ksq = 0
    for nid, hs in node_slots.items():
        n = nodes[nid]
        k = len(hs)
        if k <= 1:
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
        ksq += k * k
        if k % 2 == 0:
            try:
                from scipy.optimize import linear_sum_assignment
                r, c = linear_sum_assignment(cost)
            except Exception:
                r, c = np.arange(k), np.arange(k)
            for i, j in zip(r, c):
                n.pair[int(i)] = int(j)
        else:
            big = np.full((k, k + 1), 1e6, np.float64)
            big[:, :k] = cost
            for i in range(k):
                big[i, k] = 56.0
            try:
                from scipy.optimize import linear_sum_assignment
                r, c = linear_sum_assignment(big)
            except Exception:
                r, c = np.arange(k), np.arange(k)
            for i, j in zip(r, c):
                if j < k:
                    n.pair[int(i)] = int(j)
        n.kind = 'T' if k == 3 else 'X'
    print(f"  F 结点配对(k^2)      {time.perf_counter()-t0:8.2f} s   sum k^2 {ksq:,}")

    t0 = time.perf_counter()
    consumed = set()
    polylines = []
    hops_total = 0
    stuck = 0
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
            chain.extend(seg if not chain else seg[1:])
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
        hops_total += guard
        if guard >= 10000:
            stuck += 1
        if len(chain) >= 2:
            polylines.append(chain)
    print(f"  G 走链               {time.perf_counter()-t0:8.2f} s   polylines {len(polylines)}  "
          f"总跳数 {hops_total:,}  触顶(10000)的链 {stuck}")

    t0 = time.perf_counter()
    segments = []
    for chain in polylines:
        poly = rdp(chain, ext.rdp_eps)
        if ext.snap:
            poly = snap_polyline(poly)
        segments.extend(split_segments(poly))
    print(f"  H rdp/snap/split     {time.perf_counter()-t0:8.2f} s   segments {len(segments)}")

    t0 = time.perf_counter()
    ep = [(nodes[hi.node].x, nodes[hi.node].y) for hi in half if len(nodes[hi.node].slots) <= 1]
    jn = [(nodes[hi.node].x, nodes[hi.node].y) for hi in half if len(nodes[hi.node].slots) >= 3]
    print(f"  I endpoints/junctions{time.perf_counter()-t0:8.2f} s   ep {len(ep)}  jn {len(jn)}")


timed_trace(skel)
