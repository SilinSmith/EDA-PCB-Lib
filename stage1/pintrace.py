"""从元件引脚出发的「贪心射线」导线发现（component-mediated wire tracing）。

动机（来自用户 m04178）：
    全局骨架化 + 过滤这条路，判据天生是「长 / 直 / 大连通」，于是
      · 被文字涂掉的走线在涂掉处断开 → 两截都变短 → 被 min_extent 丢掉
      · 走线穿过元件符号时被遮挡 → 同理
      · 小图上本来就短的走线 → 同理
    但**起点是确定的**：导线是用来连接元器件的，所以每条导线一定从某个引脚出发。
    于是反过来做：在元件（尤其是 box）四周找引脚，找到就沿引脚朝外的方向**直线外推**，
    一步一像素地跟着墨走，直到墨断掉（这就是「一直延伸下去直到有拐角」）。

和全局骨架的关键区别：**这条射线跑在原始墨迹上（不涂文字、不遮挡元件）**。
因为方向由引脚给定，文字不可能把它带偏 —— 文字遮挡是为了让*全局*方法少提假阳性，
而射线方法不需要它。

API:
    find_pins(ink, boxes, ...)   -> [Pin(box_idx, x, y, dx, dy)]
    extend_ray(ink, x, y, dx, dy, ...) -> (x2, y2, n_hit) | None
    trace_from_boxes(ink, boxes, ...)  -> list[[(x1,y1),(x2,y2)]]
    python stage1\\pintrace.py [--limit N] [--out DIR]   自检：在官方 220 张上量覆盖/精度
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

INK = 170
BAND = 16.0          # 在框外多远之内找引脚桩
MIN_RUN = 2          # 引脚桩沿框边方向至少几像素宽
MIN_STUB = 3.0       # 引脚桩朝外至少几像素长
GAP = 2              # 射线允许的连续空墨像素数
MAX_LEN = 900.0      # 单条射线最长像素
PERP_TOL = 1         # 垂直方向容差（1px 发丝线会有半像素抖动）
PROBE = 6            # 拐角续走时，在端点前方多少像素内找下一个方向


@dataclass
class Pin:
    box: int
    x: float
    y: float
    dx: float
    dy: float


# --------------------------------------------------------------- 找引脚
def find_pins(ink, boxes, band=BAND, min_run=MIN_RUN, min_stub=MIN_STUB):
    """在每个元件框的四条边外侧找「朝外的引脚桩」，返回 Pin 列表。

    做法：对某条边，逐行（或逐列）数一下框外 band 像素内的墨有多少，
    引脚桩那一行会有连续一长串墨（>= min_stub）；连续若干这样的行
    聚成一个引脚，取其中最靠外的那一列/行作为引脚尖端。
    """
    H, W = ink.shape
    pins: list[Pin] = []
    b = int(round(band))
    for bi, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = x1 - x0, y1 - y0
        if bw < 6 or bh < 6 or bw > 1400 or bh > 1400:
            continue

        # ---- 左右两条边：沿 y 扫，看每条 y 上框外 band 内的墨列数 ----
        for side, sx, dx in (("L", x0, -1.0), ("R", x1, 1.0)):
            xa = int(max(0, min(W - 1, sx + (2 if dx > 0 else -b))))
            xb = int(max(0, min(W, sx + (b if dx > 0 else -2))))
            if xb - xa < min_stub:
                continue
            ya, yb = int(max(0, y0 - 2)), int(min(H, y1 + 2))
            if yb - ya < 3:
                continue
            seg = ink[ya:yb, xa:xb]
            cnt = seg.sum(axis=1)
            for cy0, cy1 in _runs(cnt >= min_stub, min_run):
                yy = (cy0 + cy1) / 2.0 + ya
                # 尖端 = 该行带内最靠外的墨像素
                sub = ink[int(round(yy)), xa:xb]
                idx = np.nonzero(sub)[0]
                if idx.size == 0:
                    continue
                tip = xa + (idx.max() if dx > 0 else idx.min())
                x_start = sx + (4 if dx > 0 else -4)          # 从框边稍外开始，跳开框线本身
                if abs(tip - x_start) < min_stub:
                    continue
                pins.append(Pin(bi, float(x_start), float(yy), dx, 0.0))

        # ---- 上下两条边：沿 x 扫 ----
        for side, sy, dy in (("T", y0, -1.0), ("B", y1, 1.0)):
            ya = int(max(0, min(H - 1, sy + (2 if dy > 0 else -b))))
            yb = int(max(0, min(H, sy + (b if dy > 0 else -2))))
            if yb - ya < min_stub:
                continue
            xa, xb = int(max(0, x0 - 2)), int(min(W, x1 + 2))
            if xb - xa < 3:
                continue
            seg = ink[ya:yb, xa:xb]
            cnt = seg.sum(axis=0)
            for cx0, cx1 in _runs(cnt >= min_stub, min_run):
                xx = (cx0 + cx1) / 2.0 + xa
                sub = ink[ya:yb, int(round(xx))]
                idx = np.nonzero(sub)[0]
                if idx.size == 0:
                    continue
                tip = ya + (idx.max() if dy > 0 else idx.min())
                y_start = sy + (4 if dy > 0 else -4)
                if abs(tip - y_start) < min_stub:
                    continue
                pins.append(Pin(bi, float(xx), float(y_start), 0.0, dy))
    return pins


def _runs(mask, min_len):
    """布尔一维数组里的连续 True 段 -> [(start, end_exclusive)]，只留长度 >= min_len 的。"""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    gap = 1                                   # 允许 1px 的小断口
    if gap:
        s2, e2 = [], []
        for s, e in zip(starts, ends):
            if s2 and s - e2[-1] <= gap:
                e2[-1] = e
            else:
                s2.append(s)
                e2.append(e)
        starts, ends = np.asarray(s2), np.asarray(e2)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


# --------------------------------------------------------------- 外推射线
def extend_ray(ink, x, y, dx, dy, gap=GAP, max_len=MAX_LEN, perp_tol=PERP_TOL):
    """从 (x,y) 沿 (dx,dy) 逐步走，遇到连续 > gap 个空像素就停。

    垂直方向给 perp_tol 像素容差（1px 发丝线抗锯齿后会有半像素偏移）。
    返回 ((x2,y2), n_hit) —— x2,y2 是最后一个着墨的位置；长度不足则返回 None。
    """
    H, W = ink.shape
    px, py = -dy, dx                                   # 垂直方向
    hits = []
    miss = 0
    steps = int(max_len)
    for s in range(1, steps + 1):
        cx = x + dx * s
        cy = y + dy * s
        if not (0 <= cx < W and 0 <= cy < H):
            break
        found = False
        for o in range(-perp_tol, perp_tol + 1):
            xi = int(round(cx + px * o))
            yi = int(round(cy + py * o))
            if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]:
                hits.append((xi, yi))
                found = True
                break
        if found:
            miss = 0
        else:
            miss += 1
            if miss > gap:
                break
    if not hits:
        return None
    ex, ey = hits[-1]
    if len(hits) < 4:
        return None
    return (float(ex), float(ey)), len(hits)


def _probe(ink, x, y, dx, dy, dist, perp_tol=PERP_TOL):
    """(x,y) 沿 (dx,dy) 走 dist 像素处有没有墨（垂直方向给 perp_tol 容差）。"""
    H, W = ink.shape
    px, py = -dy, dx
    cx, cy = x + dx * dist, y + dy * dist
    for o in range(-perp_tol, perp_tol + 1):
        xi, yi = int(round(cx + px * o)), int(round(cy + py * o))
        if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]:
            return True
    return False


def extend_path(ink, x, y, dx, dy, gap=GAP, max_len=MAX_LEN, corners=0,
                probe=PROBE, perp_tol=PERP_TOL):
    """直线外推，遇到尽头就在 `probe` 像素内找拐角接着走。

    用户 m04178 的原话是「向外直接直线一路延伸下去**直到有拐角**」，m04360 又明确
    「确实应该给射线加拐角续走」。所以停下来的位置先看两个垂直方向有沒有墨：
    有就拐过去继续直线走（拐角数上限 `corners`）。
    返回折线点列 [(x0,y0), (x1,y1), ...]。
    """
    pts = [(float(x), float(y))]
    cx, cy, cdx, cdy = float(x), float(y), float(dx), float(dy)
    used = 0.0
    for ci in range(corners + 1):
        r = extend_ray(ink, cx, cy, cdx, cdy, gap=gap,
                       max_len=max(8.0, max_len - used), perp_tol=perp_tol)
        if r is None:
            break
        (ex, ey), _n = r
        step = float(np.hypot(ex - cx, ey - cy))
        if step < 2.0:
            break
        pts.append((ex, ey))
        used += step
        cx, cy = ex, ey
        if ci == corners or used >= max_len:
            break
        # 找拐角：两个垂直方向 + 一个「更大 gap 的直行」按优先级试
        nxt = None
        for nd in ((-cdy, cdx), (cdy, -cdx)):
            for k in range(1, probe + 1):
                if _probe(ink, cx, cy, nd[0], nd[1], k, perp_tol):
                    nxt = (nd, k)
                    break
            if nxt:
                break
        if nxt is None:
            break
        nd, k = nxt
        cx, cy = cx + nd[0] * k, cy + nd[1] * k
        pts.append((cx, cy))
        used += float(k)
        cdx, cdy = nd
    return pts


def _explode(pts):
    """折线 -> 一串 2 点线段（评测script 只吃 [[x,y],[x,y]] 这种形状）。"""
    return [[list(pts[i]), list(pts[i + 1])] for i in range(len(pts) - 1)]


def _near_box(pt, box, d):
    x, y = pt
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return dx * dx + dy * dy <= d * d


def _densify(pts, step=2.0):
    """折线按 ~step 像素加密成采样点列（用来沿射线找第一个锚点）。"""
    out = [pts[0]]
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        L = float(np.hypot(bx - ax, by - ay))
        n = max(1, int(L / step))
        for k in range(1, n + 1):
            out.append((ax + (bx - ax) * k / n, ay + (by - ay) * k / n))
    return out


def _anchor_mask(shape, boxes, dist):
    """「锚点附近」掩膜：查询 O(1)，避免 n_ray × n_anchor 的 Python 双层循环
    （一张图 4000 条射线 × 600 个锚点框 = 240 万次，直接查掩膜则是一次数组索引）。"""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    for ab in boxes:
        x0, y0, x1, y1 = ab
        cv2.rectangle(m, (int(np.floor(x0 - dist)), int(np.floor(y0 - dist))),
                      (int(np.ceil(x1 + dist)), int(np.ceil(y1 + dist))), 255, -1)
    return m


def _truncate(pts, dense, idx, step=2.0):
    """把折线截断到 dense[idx] 处，但**保留原始折线顶点**、只补一个精确截断点。

    直接返回 dense[:idx+1] 会把整条射线变成一串 2px 碎段（段数从 20k 涨到 32k），
    既拖慢下游建图，也对官方 net_line 的线段匹配不利。
    """
    cut = dense[idx]
    out = [pts[0]]
    acc = 0
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        n = max(1, int(float(np.hypot(bx - ax, by - ay)) / step))
        if acc + n <= idx:
            out.append((bx, by))
            acc += n
        else:
            break
    if out[-1] != cut:
        out.append(cut)
    return out


def trace_from_boxes(ink, boxes, min_pin_len=6.0, gap=GAP, max_len=MAX_LEN,
                     merge_tol=2.0, anchors=None, anchor_dist=24.0, corners=0,
                     probe=PROBE, clip_anchors=False, require_anchor=False):
    """在元件框周围找引脚并沿引脚方向直线外推，返回线段列表 [[p1,p2], ...]。

    anchors 非空时只保留「远端能落到某个别的锚点框附近」的射线 ——
    这直接编码了「导线是用来连接元器件的」：一条走线不可能凭空停在纸中间。
    anchor_dist=None 或 anchors 为空则不过滤。

    **clip_anchors=True 是这条原则的正确操作化**（用户 m04360/m04510）：
    射线一直走到「碰到第一个锚点」就**截断**，而不是走到底再问远端在不在锚点旁边。
    → 上一次的证伪就是操作化错误：走到底的射线早就冲过了标签，远端当然不在锚点旁；
      而锚点也不该只是「别的元件粗框」，必须包含**文字框（网络标签）**。
    require_anchor=True 时，走到尽头也没碰到任何锚点的射线整条丢掉。

    corners > 0 时射线在尽头会拐弯续走（用户 m04360 的要求），返回的折线会被
    拆成一串 2 点线段，相邻两段严格共端点 ⇒ 下游建图必然连通。
    """
    pins = find_pins(ink, boxes)
    amask = _anchor_mask(ink.shape, anchors, anchor_dist) if (anchors and clip_anchors) else None
    segs = []
    n_drop = 0
    for p in pins:
        pts = extend_path(ink, p.x, p.y, p.dx, p.dy, gap=gap, max_len=max_len,
                          corners=corners, probe=probe)
        if len(pts) < 2:
            continue

        if amask is not None:
            own = boxes[p.box] if (p.box is not None and 0 <= p.box < len(boxes)) else None
            idx = None
            H, W = ink.shape
            dense = _densify(pts, 2.0)
            for i, (x, y) in enumerate(dense):
                # 起点一定落在自己那个框的豁免圈里，先跳出去再找锚点
                if own is not None and _near_box((x, y), own, anchor_dist + 2.0):
                    continue
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H and amask[yi, xi]:
                    idx = i
                    break
            if idx is not None:
                pts = _truncate(pts, dense, idx, 2.0)
            elif require_anchor:
                n_drop += 1
                continue

        ex, ey = pts[-1]
        L = 0.0
        for i in range(len(pts) - 1):
            L += float(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
        if L < min_pin_len:
            continue
        if anchors and not clip_anchors:
            ok = False
            for ai, ab in enumerate(anchors):
                if ai == p.box:
                    continue
                if _near_box((ex, ey), ab, anchor_dist):
                    ok = True
                    break
            if not ok:
                n_drop += 1
                continue
        for piece in _explode(pts):
            segs.append(piece)
    return _dedup_segs(segs, tol=merge_tol)


def _dedup_segs(segs, tol=2.0):
    """把重合的射线去掉（同一个引脚桩会被相邻两条边各找到一次）。"""
    keep = []
    for s in sorted(segs, key=lambda t: -np.hypot(t[1][0] - t[0][0], t[1][1] - t[0][1])):
        (a, b) = s
        dup = False
        for (c, d) in keep:
            if (np.hypot(a[0] - c[0], a[1] - c[1]) <= tol and np.hypot(b[0] - d[0], b[1] - d[1]) <= tol) \
               or (np.hypot(a[0] - d[0], a[1] - d[1]) <= tol and np.hypot(b[0] - c[0], b[1] - c[1]) <= tol) \
               or (np.hypot(a[0] - c[0], a[1] - c[1]) <= tol and np.hypot(b[0] - c[0], b[1] - c[1]) <= tol):
                dup = True
                break
            # 一条被另一条完全包含
            for q in (a, b):
                t = ((q[0] - c[0]) * (d[0] - c[0]) + (q[1] - c[1]) * (d[1] - c[1]))
                L2 = (d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2 + 1e-9
                t = max(0.0, min(1.0, t / L2))
                if np.hypot(q[0] - (c[0] + t * (d[0] - c[0])), q[1] - (c[1] + t * (d[1] - c[1]))) <= tol:
                    dup = True
                    break
            if dup:
                break
        if not dup:
            keep.append(s)
    return keep


# --------------------------------------------------------------- 自检
def _main():
    from schematic_parser.data import discover, DEFAULT_DATA_ROOT
    sys.path.insert(0, r"Y:\workspace\EDA\work")
    from stage1_wires_eval import seg_arrays, dist_pts_to_segs, sample_polyline

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=r"Y:\workspace\EDA\work\stage1_pintrace")
    ap.add_argument("--save", action="store_true", help="把射线也写成 json 便于和骨架合起来评")
    ap.add_argument("--stage1", default="",
                    help="用这个 stage1 产物目录里的**检测框**（而不是 GT 框）来找引脚")
    ap.add_argument("--anchor", default="none", choices=("none", "comp", "both"),
                    help="远端必须落到锚点附近（comp=别的元件框，both=再加文字框）")
    ap.add_argument("--anchor-dist", type=float, default=24.0)
    ap.add_argument("--corners", type=int, default=0,
                    help="射线遇尽头后拐弯续走的次数上限（0=遇拐角即停）")
    ap.add_argument("--gap", type=int, default=GAP,
                    help="射线允许的连续空墨像素数（越大越能跨过断口/被擦掉的文字带）")
    ap.add_argument("--probe", type=int, default=PROBE,
                    help="拐角续走时在前方多少像素内找下一个方向")
    args = ap.parse_args()

    cases = sorted(discover(DEFAULT_DATA_ROOT), key=lambda c: c.case_id)
    if args.limit:
        cases = cases[: args.limit]
    os.makedirs(args.out, exist_ok=True)

    n_gt = n_cov = 0
    n_ray = n_ray_cov = 0
    len_gt = len_ray = 0.0
    t0 = time.time()
    npin_tot = 0
    for ci, case in enumerate(cases):
        img = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < INK).astype(np.uint8)
        det = json.load(open(case.json_path, encoding="utf-8"))
        if args.stage1:
            # 用 stage1 自己检出的元件框（bbox_img 已经是左上原点），这才是真实设置
            sf = None
            for cand in (f"{case.case_id}.stage1.json", f"{case.case_id}.json"):
                p = os.path.join(args.stage1, cand)
                if os.path.exists(p):
                    sf = p
                    break
            if sf is None:
                continue
            sd = json.load(open(sf, encoding="utf-8"))
            boxes = [[float(v) for v in c["bbox_img"]] for c in sd["components"]]
        else:
            # 先用 GT（量上限），后面再接 stage1 的检测结果
            boxes = [c["bbox"] for c in det["components"].values()]
            boxes = [[b[0], H - b[3], b[2], H - b[1]] for b in boxes]
        pins = find_pins(ink, boxes)
        npin_tot += len(pins)
        anchors = None
        if args.anchor != "none":
            anchors = [list(b) for b in boxes]
            if args.anchor == "both":
                from cvparts.textbox import detect_boxes
                anchors += [[float(v) for v in b] for b in detect_boxes(img)]
        segs = trace_from_boxes(ink, boxes, anchors=anchors,
                                anchor_dist=args.anchor_dist, corners=args.corners,
                                gap=args.gap, probe=args.probe)

        gt = json.load(open(case.json_path, encoding="utf-8"))
        gsegs = []
        for net in gt["nets"].values():
            for pts in (net.get("edges") or {}).values():
                if len(pts) < 2:
                    continue
                for i in range(len(pts) - 1):
                    gsegs.append([[float(pts[i]["x"]), H - float(pts[i]["y"])],
                                  [float(pts[i + 1]["x"]), H - float(pts[i + 1]["y"])]])
        GA = seg_arrays(gsegs)
        RA = seg_arrays(segs)
        if args.save:
            json.dump({"case": case.case_id,
                       "segments": [[list(s[0]), list(s[1])] for s in segs]},
                      open(os.path.join(args.out, f"{case.case_id}.json"), "w", encoding="utf-8"))
        if not (GA.shape[0] and RA.shape[0]):
            n_gt += GA.shape[0]
            continue
        tol = 5.0
        gp = np.concatenate([sample_polyline(GA[i, :2], GA[i, 2:], 16) for i in range(GA.shape[0])])
        go = np.concatenate([np.full(16, i) for i in range(GA.shape[0])])
        gd = dist_pts_to_segs(gp, RA)
        gw = np.zeros(GA.shape[0])
        np.maximum.at(gw, go, gd)
        rp = np.concatenate([sample_polyline(RA[i, :2], RA[i, 2:], 16) for i in range(RA.shape[0])])
        ro = np.concatenate([np.full(16, i) for i in range(RA.shape[0])])
        rd = dist_pts_to_segs(rp, GA)
        rw = np.zeros(RA.shape[0])
        np.maximum.at(rw, ro, rd)

        n_gt += GA.shape[0]
        n_cov += int((gw <= tol).sum())
        n_ray += RA.shape[0]
        n_ray_cov += int((rw <= tol).sum())
        len_gt += float(np.hypot(GA[:, 2] - GA[:, 0], GA[:, 3] - GA[:, 1]).sum())
        len_ray += float(np.hypot(RA[:, 2] - RA[:, 0], RA[:, 3] - RA[:, 1]).sum())
        if (ci + 1) % 25 == 0:
            print(f"  ... {ci+1}/{len(cases)}  {time.time()-t0:.0f}s")

    R = n_cov / max(1, n_gt)
    P = n_ray_cov / max(1, n_ray)
    F1 = 2 * P * R / max(1e-9, P + R)
    print(f"\n=== 纯贪心射线（框用 GT，只量「射线这条路本身」）===")
    print(f"  图 {len(cases)} 张   引脚 {npin_tot}")
    print(f"  GT 线段 {n_gt}  被射线覆盖 {n_cov}  recall {R:.4f}")
    print(f"  射线段  {n_ray}  贴住 GT   {n_ray_cov}  precision {P:.4f}")
    print(f"  F1 {F1:.4f}   总长比 {len_ray/max(1.0,len_gt):.3f}   耗时 {time.time()-t0:.1f}s")
    print(f"  产物 {args.out}")


if __name__ == "__main__":
    _main()
