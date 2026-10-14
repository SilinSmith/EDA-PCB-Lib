"""导线骨架提取的封装 —— 复用 `pipeline/pskel.py`，补三件它没有的事。

1. **降采样**：`pskel` 在最大的 `0171`（1675x1190）上单张要 88.5 s，而赛题预算是
   2.4~4 min/case（30~50 case / 2 小时）。`pskel` 是纯 Python 循环 + numpy，
   **GIL 绑死，多线程救不了它**（见 `pipeline.py` 用进程池）。这里先做最便宜的一刀：
   超过 `max_pixels` 的图先按面积比降采样跑骨架，坐标再乘回原图。

2. **遮挡已检出的元件（联合解算）**：这是第一阶段真正缺的一步。骨架化对「墨迹」
   一视同仁，而 IC 矩形边框、电容极板、电阻方框、引脚名文字全都是细笔画 —— 实测
   提取总长是 GT 网表线段的 **5.04 倍**，其中只有 6% 压在真导线上。既然元件框已经
   被 YOLO/CV 检出，就把这些框涂成背景色再跑骨架，让文字和符号体从一开始就不进入
   骨架。这不是后验过滤，是上游裁剪。

3. **图纸边框与笔画过滤**：图框是贴在页边缘的细长矩形，骨架一定会把它整条提出来；
   文字笔画短且方向杂乱。`drop_frame` / `min_len` / `max_axis_dev` 分别处理这三类。

输出坐标全部是**原图像素**（左上原点），与 `detect.py` 的元件框同一坐标系。
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
try:                                    # 作为包导入（python -m stage1 / submission\run.py）
    from envflag import env_on      # 「假值词表」唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage1\wires.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on

import cv2
import numpy as np

DEFAULT_MAX_PIXELS = 2_500_000


def _use_scale(w: int, h: int, max_pixels: int) -> float:
    px = float(w) * float(h)
    if px <= max_pixels:
        return 1.0
    return max(0.25, math.sqrt(max_pixels / px))


def _background_colour(bgr: np.ndarray) -> np.ndarray:
    """取四角小块的中位色当背景。比全图众数快且对黑底白线也成立。"""
    h, w = bgr.shape[:2]
    k = max(2, min(8, h // 8, w // 8))
    corners = np.concatenate([
        bgr[:k, :k].reshape(-1, 3), bgr[:k, -k:].reshape(-1, 3),
        bgr[-k:, :k].reshape(-1, 3), bgr[-k:, -k:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def occlude_boxes(bgr: np.ndarray, boxes, pad: float = 2.0) -> np.ndarray:
    """把给定的框（图像坐标 [x1,y1,x2,y2]）涂成背景色。

    ★ `STAGE1_OCC_LOCAL=1`：用**框周围环带的局部纸色**涂（纸色中值模糊图
    `pskel.estimate_bg_local` 在环带上的中值），而不是整张图一个常量。理由：白纸有
    渐晕/偏色，常量填充会在框边界留下一条与局部背景的**色差接缝**——灰度自适应
    阈值基本看不见它，但任何「与背景的偏离」判据（`bgdiff` / `bglab` / 彩色边缘）
    都会把这条接缝当墨迹，凭空长出骨架分支。默认 `0`（旧行为），A/B 后再定。
    """
    out = bgr.copy()
    if not boxes:
        return out
    h, w = out.shape[:2]
    bg = _background_colour(bgr)
    _loc = os.environ.get("STAGE1_OCC_LOCAL", "0").strip().lower() in ("1", "true", "on", "yes")
    bg_loc = None
    if _loc:
        try:
            from pipeline.pskel import estimate_bg_local
            bg_loc = estimate_bg_local(bgr, 51)
        except Exception:
            bg_loc = None
    for b in boxes:
        try:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
        except (TypeError, ValueError):
            continue
        x1 = int(max(0, math.floor(x1 - pad)))
        y1 = int(max(0, math.floor(y1 - pad)))
        x2 = int(min(w, math.ceil(x2 + pad)))
        y2 = int(min(h, math.ceil(y2 + pad)))
        if x2 > x1 and y2 > y1:
            fill = bg
            if bg_loc is not None:
                m = max(6, int(0.12 * max(x2 - x1, y2 - y1)))
                rx1, ry1 = max(0, x1 - m), max(0, y1 - m)
                rx2, ry2 = min(w, x2 + m), min(h, y2 + m)
                ring = bg_loc[ry1:ry2, rx1:rx2].reshape(-1, 3)
                box_px = (rx2 - rx1) * (ry2 - ry1)
                if ring.size and box_px > 0:
                    keep = np.ones(ring.shape[0], bool)
                    ix1, iy1 = x1 - rx1, y1 - ry1
                    ix2, iy2 = x2 - rx1, y2 - ry1
                    grid = np.zeros((ry2 - ry1, rx2 - rx1), bool)
                    grid[max(0, iy1):max(0, iy2), max(0, ix1):max(0, ix2)] = True
                    keep = ~grid.reshape(-1)
                    if keep.sum() >= 8:
                        fill = np.median(ring[keep], axis=0)
            out[y1:y2, x1:x2] = fill
    return out


def _seg_len(a, b) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _dist_to_boxes(seg, occlude=None, erase=None) -> float:
    """线段两端点到最近的「元件框 / 文字框」的距离（原图像素）。没有框时返回 1e9。

    用途：量化「这根短桩是在离所有框多远的地方被删掉的」——即 `rescue_pad` 要多大
    才能救它。用户 m42009 的链式单点失效诊断（`wires.py:534`）需要这个数。
    """
    boxes = list(occlude or []) + list(erase or [])
    if not boxes:
        return 1e9
    best = 1e9
    for p in seg:
        px, py = float(p[0]), float(p[1])
        for bx in boxes:
            try:
                x1, y1, x2, y2 = (float(v) for v in bx[:4])
            except (TypeError, ValueError):
                continue
            dx = max(x1 - px, 0.0, px - x2)
            dy = max(y1 - py, 0.0, py - y2)
            d = math.hypot(dx, dy)
            if d < best:
                best = d
    return best


def _dist_rect(p, rect) -> float:
    """点 `p` 到矩形（框外为正、框内为 0）的距离。"""
    x1, y1, x2, y2 = rect
    px, py = float(p[0]), float(p[1])
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _unit(a, b):
    L = _seg_len(a, b)
    if L <= 1e-6:
        return (0.0, 0.0)
    return ((float(b[0]) - float(a[0])) / L, (float(b[1]) - float(a[1])) / L)


def _ang_deg(u, v) -> float:
    nu, nv = math.hypot(u[0], u[1]), math.hypot(v[0], v[1])
    if nu <= 1e-6 or nv <= 1e-6:
        return 180.0
    c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(c))


def bridge_boxes(bgr, segments, boxes, *, near: float = 8.0, max_frac: float = 0.9,
                 angle: float = 30.0, ink_min: float = 0.5, de_min: float = 22.0,
                 min_side: float = 24.0, pad: float = 6.0, thick_max: float = 9.0):
    """★ 2026-10-10（用户 m42009 的「互相等待」）——框内导线的**交叉验证桥接**。

    病灶（实测，28 张）：`occlude_boxes` 在提线**之前**把**我们自己检出的元件框**
    涂成背景色 ⇒ 落在框内的 GT 导线边 662/3786 条里只有 17 条拿到覆盖
    （框内 R=0.0257 vs 框外 R=0.3444）。不是代码成环（`detect.py`/`bigic.py` 对
    ink/wire 零引用，元件先定稿、无任何回写），而是「拿假设去删证据」的自我实现：
    我认定这里有元件 ⇒ 把框内导线证据涂掉 ⇒ 框内那个连接永远得不到导线确认。

    修法不是「把框内像素全放回来」（首测证明会凭空长结构：元件本体轮廓/丝印都被
    当成导线），而是**只接受能双向交叉验证的桥**，四条证据同时成立才连：
      ① 两端各有一根**框外**已确认线段贴着框（≤ `near`）；
      ② 这两根的走向都要指向框内，且与桥的方向夹角 ≤ `angle`（共线）；
      ③ 桥的中点必须落在框内（确实穿过本体）；
      ④ 桥沿线必须真的「离背景有偏离」（Lab ΔE ≥ `de_min`）且**细**（距离变换 ≤
         `thick_max`，排除实心色块/元件填充）。
    框外线段 + 框内墨迹互为证据 —— 这就是「用其他手段交叉验证」。

    返回 `(bridges, info)`；`bridges` 是原图像素坐标的 `[[x1,y1],[x2,y2]]` 列表。
    默认**不启用**（`STAGE1_WIRE_BRIDGE`），由 A/B 决定是否转正。
    """
    info = {"n_box": 0, "n_attach": 0, "n_pair": 0, "n_ok": 0,
            "rej": {"len": 0, "mid": 0, "angle": 0, "ink": 0, "thick": 0}}
    bridges: list = []
    if not segments or not boxes:
        return bridges, info
    bgr = np.asarray(bgr)
    H, W = bgr.shape[:2]
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in _sys.path:
        _sys.path.insert(0, root)
    try:
        from pipeline import pskel
        bg = pskel.estimate_bg_local(bgr, k=51)
        dE = np.asarray(pskel.bg_distance(bgr, bg=bg, space="lab"), np.float32)
    except Exception:
        return bridges, info
    thin = None
    try:
        mast = (dE >= float(de_min)).astype(np.uint8)
        thin = cv2.distanceTransform(mast, cv2.DIST_L2, 3)
    except Exception:
        thin = None

    def _ix(p):
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        return min(max(x, 0), W - 1), min(max(y, 0), H - 1)

    def _side(p, rect):
        """贴框端点贴在哪条边上（必须在框外）：0 上 / 1 下 / 2 左 / 3 右 / -1 框内。"""
        x1, y1, x2, y2 = rect
        px, py = float(p[0]), float(p[1])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return -1
        if py < y1:
            return 0
        if py > y2:
            return 1
        if px < x1:
            return 2
        return 3

    for bx in boxes:
        try:
            x1, y1, x2, y2 = (float(v) for v in bx[:4])
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        w, h = x2 - x1, y2 - y1
        if min(w, h) < float(min_side):
            continue
        diag = math.hypot(w, h)
        rect = (x1, y1, x2, y2)
        info["n_box"] += 1
        att = []
        for si, (a, b) in enumerate(segments):
            for p, q in ((a, b), (b, a)):
                dd = _dist_rect(p, rect)
                if dd <= near:
                    sd = _side(p, rect)
                    if sd >= 0:
                        att.append((si, (float(p[0]), float(p[1])), _unit(q, p), sd, dd))
        info["n_attach"] += len(att)
        if len(att) < 2:
            continue
        cand = []
        for i in range(len(att)):
            for j in range(i + 1, len(att)):
                si, p1, u1, s1, d1 = att[i]
                sj, p2, u2, s2, d2 = att[j]
                if si == sj or s1 == s2:
                    # 同一条线段自己配自己、或两端贴同一条边（那只是沿边的小缺口，
                    # 不是「穿过元件本体」的桥）都跳过。
                    continue
                vx, vy = p2[0] - p1[0], p2[1] - p1[1]
                L = math.hypot(vx, vy)
                if L < 6.0 or L > max_frac * diag:
                    info["rej"]["len"] += 1
                    continue
                # 桥必须真的穿过本体：采样点里至少 60% 落在**严格框内**。
                npt, n_in, hit, fat = 11, 0, 0, 0
                for k in range(1, npt + 1):
                    t = k / (npt + 1.0)
                    px, py = p1[0] + vx * t, p1[1] + vy * t
                    if x1 < px < x2 and y1 < py < y2:
                        n_in += 1
                    ix, iy = _ix((px, py))
                    if dE[iy, ix] >= de_min:
                        hit += 1
                        if thin is not None and float(thin[iy, ix]) > thick_max:
                            fat += 1
                if n_in < 0.6 * npt:
                    info["rej"]["mid"] += 1
                    continue
                a1 = _ang_deg(u1, (vx, vy))
                a2 = _ang_deg(u2, (-vx, -vy))
                if a1 > angle or a2 > angle:
                    info["rej"]["angle"] += 1
                    continue
                sup = hit / float(npt)
                if sup < ink_min:
                    info["rej"]["ink"] += 1
                    continue
                if fat > 1:
                    info["rej"]["thick"] += 1
                    continue
                sc = sup * math.cos(math.radians(max(a1, a2)))
                cand.append((sc, i, j, p1, p2, sup, max(a1, a2), (n_in, npt, d1, d2)))
        info["n_pair"] += len(cand)
        used = set()
        for sc, i, j, p1, p2, sup, ang, extra in sorted(cand, key=lambda z: -z[0]):
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            bridges.append([list(p1), list(p2)])
            info["n_ok"] += 1
            try:
                from cvparts.hypo import LEDGER
                LEDGER.record("bridge", "wire:box", before=list(p1), after=list(p2),
                              reason=("框内交叉验证通过：沿线墨迹支持 %.2f（框内采样 %d/%d），"
                                      "最大共线偏差 %.1f°，贴框距离 %.1f/%.1f px，"
                                      "框 %.0f,%.0f,%.0f,%.0f"
                                      % (sup, extra[0], extra[1], ang, extra[2], extra[3],
                                         x1, y1, x2, y2)),
                              by="stage1/wires.py:bridge_boxes")
            except Exception:
                pass
    return bridges, info


def _axis_dev(a, b) -> float:
    """与最近的 0/45/90/135 度的偏差（度）。"""
    ang = math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))
    m = abs(ang) % (math.pi / 4.0)
    m = min(m, math.pi / 4.0 - m)
    return math.degrees(m)


def _is_frame(a, b, w: int, h: int, margin_frac: float, len_frac: float) -> bool:
    """贴近页面边缘、且很长的轴对齐线段 —— 判为图纸边框。"""
    L = _seg_len(a, b)
    if L < len_frac * min(w, h):
        return False
    if _axis_dev(a, b) > 6.0:
        return False
    m = margin_frac * min(w, h)
    xs = (float(a[0]), float(b[0]))
    ys = (float(a[1]), float(b[1]))
    near_x = max(xs) < m or min(xs) > w - m
    near_y = max(ys) < m or min(ys) > h - m
    return near_x or near_y


def _poly_len(pl) -> float:
    return sum(_seg_len(pl[i], pl[i + 1]) for i in range(len(pl) - 1))


def _split_at_points(segments, pts, tol: float = 1.5):
    """把每条线段在 **给定点集** `pts` 落在其内部（垂距 ≤ tol）处切开。

    返回 (新线段表, 被切过的线段条数)。与 `_split_at_nodes` 的唯一区别是切点来源：
    这里可以喂 `pskel` 真正的**结点**（degree ≥ 3 的骨架簇），而不是「别人的线段端点」。
    """
    out, n_split = [], 0
    for a, b in segments:
        ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        cuts = []
        if L2 > 1e-6:
            for px, py in pts:
                t = ((px - ax) * dx + (py - ay) * dy) / L2
                if t <= 1e-3 or t >= 1.0 - 1e-3:
                    continue
                qx, qy = ax + t * dx, ay + t * dy
                if math.hypot(px - qx, py - qy) <= tol:
                    cuts.append((t, qx, qy))
        if not cuts:
            out.append([a, b])
            continue
        cuts.sort()
        n_split += 1
        prev = (ax, ay)
        for _t, qx, qy in cuts:
            if math.hypot(qx - prev[0], qy - prev[1]) <= tol:
                continue
            out.append([[prev[0], prev[1]], [qx, qy]])
            prev = (qx, qy)
        if math.hypot(bx - prev[0], by - prev[1]) > tol:
            out.append([[prev[0], prev[1]], [bx, by]])
    return out, n_split


def _split_at_nodes(segments, tol: float = 1.5):
    """把「别的线段的端点落在自己身上」的线段在那些点切开（T 形分叉处切段）。

    返回 (新线段表, 被切过的线段条数)。端点集合取自**原始** segments 的所有端点；
    切出来的新端点不会再参与（避免级联），但一拍足以把「长直线穿过分叉」解决。
    """
    pts = []
    for a, b in segments:
        pts.append((float(a[0]), float(a[1])))
        pts.append((float(b[0]), float(b[1])))
    out, n_split = [], 0
    for a, b in segments:
        ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        cuts = []
        if L2 > 1e-6:
            for px, py in pts:
                t = ((px - ax) * dx + (py - ay) * dy) / L2
                if t <= 1e-3 or t >= 1.0 - 1e-3:
                    continue
                qx, qy = ax + t * dx, ay + t * dy
                if math.hypot(px - qx, py - qy) <= tol:
                    cuts.append((t, qx, qy))
        if not cuts:
            out.append([a, b])
            continue
        cuts.sort()
        n_split += 1
        prev = (ax, ay)
        for _t, qx, qy in cuts:
            if math.hypot(qx - prev[0], qy - prev[1]) <= tol:
                continue
            out.append([[prev[0], prev[1]], [qx, qy]])
            prev = (qx, qy)
        if math.hypot(bx - prev[0], by - prev[1]) > tol:
            out.append([[prev[0], prev[1]], [bx, by]])
    return out, n_split


def _rebuild_nodes(segments, tol: float = 1.0):
    """从过滤后的线段重建端点与结点，保证三者互相一致。

    degree==1 -> 端点；同一个位置出现 >=3 条线段的端点 -> 结点；==2 是过路点。
    """
    buckets: dict[tuple[int, int], list] = {}
    for a, b in segments:
        for p in (a, b):
            key = (int(round(float(p[0]) / tol)), int(round(float(p[1]) / tol)))
            buckets.setdefault(key, []).append(p)
    endpoints, junctions = [], []
    for key, pts in buckets.items():
        if len(pts) == 1:
            endpoints.append([float(pts[0][0]), float(pts[0][1])])
        elif len(pts) >= 3:
            ax = sum(float(p[0]) for p in pts) / len(pts)
            ay = sum(float(p[1]) for p in pts) / len(pts)
            junctions.append([ax, ay])
    return endpoints, junctions


def _comp_stats(segments, tol: float = 1.0):
    """把线段按端点相接关系聚成连通分量，返回 [(extent, total_len, segs), ...]。

    extent = 分量所有端点的包围盒对角线长度。这是区分「导线」与「文字」的关键量：
    一个字母/数字只有 10~25px 的 extent，而一条走线要么延伸很远，要么在 L/T/十
    拐角处继续延伸 —— 「十」这个字只有 10px，但「十」字形分叉可以延伸几百像素。
    """
    def key(p):
        return (int(round(float(p[0]) / tol)), int(round(float(p[1]) / tol)))

    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in segments:
        ka, kb = key(a), key(b)
        parent.setdefault(ka, ka)
        parent.setdefault(kb, kb)
        union(ka, kb)

    groups: dict = {}
    for a, b in segments:
        groups.setdefault(find(key(a)), []).append((a, b))

    out = []
    for segs in groups.values():
        xs = [float(p[0]) for s in segs for p in s]
        ys = [float(p[1]) for s in segs for p in s]
        ext = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        tot = sum(_seg_len(a, b) for a, b in segs)
        out.append((ext, tot, segs))
    return out


# ---------------------------------------------------------------- know: 导线颜色
#
# 「这张原理图的导线是什么颜色」是**规则判不出来**的（试过四个逐图判据全灭：保留比例、
# 离元件框净像素、连通 >=2 元件、图框以内长直线 —— 图框自身的长直线永远比导线多）。
# 但模型一眼就能答对，而且只要 1 次调用。这是 21 节点架构里 `know` 节点的第一个落地项。
WIRE_HUE_PROMPT = (
    "这是一张电路原理图。请只回答一个问题：**图中连接元件的导线（wire）是什么颜色？**\n"
    "注意：图框、表格线、元件符号、文字的颜色不算，只看连接元件的那些线。\n"
    "请只回一个颜色词，从这些里选：红 / 绿 / 蓝 / 黑 / 青 / 紫 / 橙 / 棕 / 灰。找不到就回 未知。"
)
# OpenCV H 是 0..179。只映射到实测验证过的几个：KiCad 导线 h~85、Altium 绿-青 h~55/65。
_HUE_WORDS = (
    ("红", 5.0), ("橙", 18.0), ("黄", 30.0), ("黄绿", 45.0),
    ("绿", 55.0), ("青", 95.0), ("蓝", 125.0), ("紫", 145.0),
)
_HINT_STAT = {"calls": 0, "hit": 0, "words": {}}
# ★ 2026-10-08：给色相询问一个**自己的、短的**超时 + 熔断闸。
#   根因：`vlm/client.py:38 TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "600"))` ——
#   默认 **600 秒**，而且 `ask()` 最多重试 4 次；`ask()` 把一切异常吞成 `"ERR:..."` 字符串，
#   于是「端点慢」在 stage1 里只表现为「这张图莫名多花了 5 分钟」。问一个颜色词不需要 10 分钟。
HUE_TIMEOUT = float(os.environ.get("STAGE1_WIRE_HUE_TIMEOUT", "30"))
HUE_MAXFAIL = int(os.environ.get("STAGE1_WIRE_HUE_MAXFAIL", "3"))
_HUE_FAIL = {"n": 0, "off": False}


def _hue_small(bgr: np.ndarray) -> np.ndarray:
    """★ 2026-10-08：**问色相之前先把图缩到最大边 `STAGE1_WIRE_HUE_MAXSIDE`（默认 1024）**。

    原来这里把**整张原图**（14 MPx 的图 base64 之后十几 MB）直接发给端点，实测
    `0003`（4491x3173）这一次调用耗 **303 s**（cProfile：`wires.py:230 → vlm/client.py:161
    → urllib urlopen → socket.recv_into` 303.4 s），而 `timing` 里 yolo/cv/wires/rays/
    text/bigic/pins 加起来只有 4.3 s —— **stage1 的「黑洞」就是这一次 VLM 调用**。
    更糟的是 `VLM_TIMEOUT` 默认 600 s、`ask()` 还有最多 4 次重试，`ask()` 内部又把
    一切异常吞成 `"ERR:..."` 字符串 ⇒ 失败只会表现为「这张图慢了 5 分钟」。

    我们问的只是「导线是什么颜色」这一个词，1024 边长足够（`pskel.chroma` 的判据
    本来就是颜色是否成片出现）。缩放不改变 `_HINT_STAT` 的语义。
    """
    try:
        cap = int(os.environ.get("STAGE1_WIRE_HUE_MAXSIDE", "1024"))
    except Exception:
        cap = 1024
    if cap <= 0:
        return bgr
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= cap:
        return bgr
    s = cap / float(m)
    return cv2.resize(bgr, (max(8, int(round(w * s))), max(8, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def _hint_wire_hue(bgr: np.ndarray) -> None:
    """`STAGE1_WIRE_COLOUR=confirm`（默认）或 `hint` 时问一次模型，把色相写进
    `pipeline.pskel.HINT_HUE`；两者的区别在末端的掩膜（`confirm` 要求该色相在本图调色板
    里被证实，否则不删任何像素），发问这一步完全一样。

    ★ 2026-10-08 两处加固（都是为了「2 小时 / 50 张」的时间预算）：
    ① 发出去的是 `_hue_small(bgr)`（最大边 1024），不是整图；
    ② 用 `STAGE1_WIRE_HUE_TIMEOUT`（默认 **30 s**）而不是 `VLM_TIMEOUT`（默认 600 s），
       并且**连续失败 `STAGE1_WIRE_HUE_MAXFAIL`（默认 3）次后就整进程关掉**这一路 ——
       端点没起来时，220 张图不该每张都去等一次超时。
    `_HUE_STAT["skipped"]` 记熔断后跳过的张数，便于事后确认「是省了时间还是丢了几张的提示」。
    """
    if os.environ.get("STAGE1_WIRE_COLOUR", "confirm").strip().lower() not in (
            "hint", "confirm", "hint_confirm", "hint+confirm", "verify"):
        return
    if _HUE_FAIL["off"]:
        _HINT_STAT["skipped"] = int(_HINT_STAT.get("skipped", 0)) + 1
        return
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pipeline import pskel
    pskel.HINT_HUE = None
    # 灰度图（没有彩度）问也白问，而且掩膜本来就会自己退 —— 先花 0 成本筛掉。
    # ⚠️ 判据只能用**彩度像素的绝对个数**，不能用分位数：像 `0103` 这种稀疏图纸，
    # 绿导线占全图不到 1%，`percentile(chroma, 90)` 只有个位数，会被误判成灰度图。
    try:
        if int((pskel.chroma(bgr) >= 22.0).sum()) < 200:
            return
    except Exception:
        pass
    try:
        from vlm import client
        r = client.ask(_hue_small(bgr), WIRE_HUE_PROMPT, timeout=HUE_TIMEOUT,
                       max_tokens=24)
        txt = r[0] if isinstance(r, (tuple, list)) else r
    except Exception:
        txt = "ERR:raise"
    txt = "".join(str(txt).split())
    if txt.startswith("ERR:") or not txt:
        _HUE_FAIL["n"] += 1
        _HINT_STAT["fail"] = int(_HINT_STAT.get("fail", 0)) + 1
        _HINT_STAT["last_err"] = txt[:60]
        if _HUE_FAIL["n"] >= HUE_MAXFAIL and not _HUE_FAIL["off"]:
            _HUE_FAIL["off"] = True
        return
    _HINT_STAT["words"][txt] = _HINT_STAT["words"].get(txt, 0) + 1
    for w, hue in _HUE_WORDS:
        if w in txt:
            pskel.HINT_HUE = hue
            _HINT_STAT["hit"] += 1
            return
    # 答「黑 / 灰 / 未知」= 这张图没有彩色导线，退回 dominant（掩膜本来也会自己退）


def extract_wires(bgr: np.ndarray, max_pixels: int = DEFAULT_MAX_PIXELS,
                  ink_mode: str = "adaptive", use_colour: bool = True,
                  thin_w: float = 2.6, rdp_eps: float = 1.6, snap: bool = True,
                  scale: float | None = None,
                  occlude=None, occ_pad: float = 2.0,
                  erase=None, erase_pad: float = 1.0,
                  min_len: float = 0.0, max_axis_dev: float = 90.0,
                  min_poly_len: float = 0.0,
                  min_extent: float = 0.0, rescue_pad: float = 0.0,
                  drop_frame: bool = False,
                  frame_margin_frac: float = 0.03, frame_len_frac: float = 0.30):
    """返回 (dict, seconds)。dict 的坐标全是原图像素。

    occlude: 已检出元件的框（原图坐标），跑骨架前先涂掉。这是第一阶段的关键一步。
    erase:   额外要涂掉的东西（实际用文字框）。与 occlude 分开的原因：occlude 还要
        给 rescue_pad 当「元件附近」的判据，把文字框混进去会让豁免半径失去含义。
        文字是导线提取里最大的一坨假阳性（README 第 7 条限制之首），涂掉它直接提精度。
    min_extent: 连通分量包围盒对角线下限 —— 大尺度连通先验，专治文字笔画。
    rescue_pad: 豁免半径。分量即使小于 min_extent，只要线段端点落在「已检出元件框
        外扩 occ_pad+rescue_pad」内就仍然保留 —— 导线是用来连接元器件的。
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # ★ 2026-07 审计项⑤：`_HINT_STAT` 是模块级累加器，`submission\run.py` 在同进程
    #   里连跑 220 张，以前**从不重置** ⇒ `words` 字典无界增长（每张图都会往里塞
    #   模型吐的颜色词），而且统计值跨图混在一起没有意义。这里每张图重置一次。
    _HINT_STAT["calls"] = 0
    _HINT_STAT["hit"] = 0
    _HINT_STAT["words"].clear()
    from pipeline.pskel import WireExtractor


    H, W = bgr.shape[:2]
    s = _use_scale(W, H, max_pixels) if scale is None else float(scale)

    work = bgr
    if occlude:
        work = occlude_boxes(work, occlude, pad=occ_pad)
    if erase:
        work = occlude_boxes(work, erase, pad=erase_pad)
    src = work
    if s != 1.0:
        src = cv2.resize(work, (max(8, int(round(W * s))), max(8, int(round(H * s)))),
                         interpolation=cv2.INTER_AREA)
    _hint_wire_hue(bgr)
    # ★ 2026-10-08 新增：导线骨架「空壳」修正旋钮（病灶见 `pipeline/pskel.py:118`
    #   `thin_stroke_mask` 的 `dt <= max_half_width`）：一根 4px 宽的导线，横截面上
    #   dt = 1,2,2,1 ⇒ `dt <= 1.6` **只留下两条外沿**，Zhang-Suen 于是给出**两条平行骨架**
    #   （镜像出两条边 2~3px 相望）。GT 的 `edges` 是**一条中心线**，而 `net_line` 是一对一派
    #   ⇒ 每条 GT 边最多命中我们两条里的一条，另一条必成 fp。实测官方 9 张：配对网内
    #   我们 993 条 edge / GT 523 条（1.90×），其中**856 对**是「平行且端点都近」的双胞胎。
    #   2026-10-09 定案：**默认已改为 2.6**（真交付口径 A/B：导线 F1 0.1929→0.2317，
    #   总分 66.33/66.52 → 66.79/66.60，九张里 8 张涨；导线通道两次重复逐位一致 ⇒ 增量确定性，
    #   总分里的 ±0.19 是 VLM 噪声。表见 `docs/RESULTS.md` §4.8 与 `docs/DIAGNOSTICS.md` 的 `net_line` 小节）。
    #   三个旋钮：
    #     STAGE1_WIRE_THIN_W     半宽阈值（默认 **2.6**；1.6 = 历史值，会把 4px 描边掏空）
    #     STAGE1_WIRE_CLOSE_THIN 1 = 3x3 形态学闭运算把两条外沿之间的 2px 缝焊上（再骨架化）
    #     STAGE1_WIRE_DILATE     N>0 = 把「离细笔画 N px 内的墨」补回来（N=2 可填满 4~5px 描边，与 2.6 打平）
    try:
        _thin_w = float(os.environ.get("STAGE1_WIRE_THIN_W", "") or thin_w)
    except ValueError:
        _thin_w = thin_w
    _close_thin = os.environ.get("STAGE1_WIRE_CLOSE_THIN", "0").strip().lower() in (
        "1", "true", "on", "yes")
    # ★ 2026-10-10（用户 m40034：「背景是白的，导线颜色不确定，只要从背景过渡到任意颜色
    #   都有可能是导线」）：前景判据可选。`bglab` = 与**局部背景**的 Lab ΔE + Otsu，
    #   亮度/彩度/色相的偏离等权当候选 —— 灰度自适应看不见「等亮度不同色」的边界。
    #   `STAGE1_WIRE_INK_MODE` 覆盖调用方传入的 `ink_mode`；默认空 = 用调用方的值（adaptive）。
    #   ★ 名字必须带 `_MODE`：`STAGE1_WIRE_INK` 已被 `stage1/labels.py:133` 占用（那是个
    #     布尔闸门，`env_on`）。曾经本开关叫 `STAGE1_WIRE_INK`，A/B 臂设 `=bglab` 时顺带
    #     把 labels 的焊接墨迹判据关掉了（`env_on("…","bglab")` = 假）⇒ 读数被污染。
    _ink_env = os.environ.get("STAGE1_WIRE_INK_MODE", "").strip().lower()
    if _ink_env in ("adaptive", "otsu", "bgdiff", "bglab"):
        ink_mode = _ink_env
    try:
        _dilate = int(os.environ.get("STAGE1_WIRE_DILATE", "0"))
    except ValueError:
        _dilate = 0
    ex = WireExtractor(ink_mode=ink_mode, thin_w=_thin_w, use_colour=use_colour,
                       rdp_eps=rdp_eps, snap=snap, close_thin=_close_thin,
                       dilate_to=_dilate)
    t0 = time.time()
    r = ex.extract(src)
    dt = time.time() - t0

    inv = 1.0 / s
    # ★ 2026-10（本轮）：`pskel` 其实**已经算出**真正的骨架结点（degree ≥ 3 的像素簇，
    #   `SkeletonResult.junctions`，n_nodes 也一并返回），但 `wires.py` 一直没用它 ——
    #   `_rebuild_nodes` 那套「≥3 条线段共用同一个 1px 桶」的判据在官方 10 张上只数出
    #   0~4 个结点（case 2/9 = 0，case 1 = 4），而按 netL 账本每张图需要 ~11~37 个切点
    #   （`work/_splitoracle.py`：切在 GT 端点的天花板是 netL 0.5157→0.6684 ≈ **+1.53 分**）。
    #   ⇒ 把 pskel 的结点接出来（`STAGE1_SPLIT_JUNC_NODES=1` 时直接在结点处切段），
    #   默认 0 = 行为不变。
    jnodes = [[float(x) * inv, float(y) * inv]
              for x, y in (getattr(r, "junctions", None) or [])]
    blines = [[[float(x) * inv, float(y) * inv] for x, y in pl] for pl in r.polylines]
    bsegs = [[[float(a[0]) * inv, float(a[1]) * inv],
              [float(b[0]) * inv, float(b[1]) * inv]] for a, b in r.segments]
    n_raw = len(bsegs)

    # --- polyline 级过滤（正确的粒度）---
    # pskel 的 snap 会把一条长走线切成很多 <10px 的短段，所以在 segment 级按长度过滤
    # 会把真导线一起删掉（0101 上 F1 从 0.19 直接归零）。真正的「枝」是 polyline：
    # 文字笔画的 polyline 短，导线的 polyline 长。先按 polyline 长度筛，再只保留
    # 落在保留下来的 polyline 上的 segment。
    n_pl_dropped = 0
    if min_poly_len > 0:
        plen = [_poly_len(pl) for pl in blines]
        keep_pl = [i for i, L in enumerate(plen) if L >= min_poly_len]
        n_pl_dropped = len(blines) - len(keep_pl)
        owner: dict = {}
        for i in keep_pl:
            for x, y in blines[i]:
                owner.setdefault((int(round(float(x))), int(round(float(y)))), set()).add(i)

        def _on_kept(a, b):
            ka = (int(round(float(a[0]))), int(round(float(a[1]))))
            kb = (int(round(float(b[0]))), int(round(float(b[1]))))
            return bool(owner.get(ka, frozenset()) & owner.get(kb, frozenset()))

        bsegs = [s for s in bsegs if _on_kept(s[0], s[1])]
        blines = [blines[i] for i in keep_pl]

    # --- 大尺度连通过滤（用户的核心洞察）---
    # 文字笔画是孤立的小团：每个字母/数字自成 ~10~25px 的连通分量。
    # 导线是一个大尺度的连通体：要么一路延伸很远，要么在 L/T/十 拐角处继续延伸 ——
    # 「十」这个字只有 10px，但「十」字形分叉可以延伸几百像素。
    #
    # 实测（0101 jlc）：墨迹掩膜 740 个分量，extent 中位数 10px、p90 18px，
    # 只有 19 个 >=50px；0001 KiCad 2096 个分量只有 6 个 >=50px。先验成立。
    #
    # 两个必须踩对的细节：
    # 1) 分量要在**墨迹掩膜**上算，不能在骨架掩膜上算 —— zhang_suen 会把 1px 发丝线
    #    在交叉/毛刺处断连，骨架分量比墨迹分量碎得多。
    # 2) 采样要打**膨胀后的墨迹**，不能打骨架中心线 —— RDP 之后的线段是折线的弦，
    #    中点常常偏离 1px 宽的骨架，采不到标签就会把真导线整条丢掉（0101 上 F1 归零）。
    n_comp = n_comp_dropped = n_comp_cand = 0
    stubs_latent: list = []
    if min_extent > 0:
        _t, ink, _c = ex.foreground(src)
        ink = cv2.dilate((np.asarray(ink) > 0).astype(np.uint8),
                         np.ones((3, 3), np.uint8), iterations=1)
        n_lab, lab, cstats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        diag = np.zeros(n_lab, np.float64)
        for i in range(1, n_lab):
            diag[i] = math.hypot(float(cstats[i][2]), float(cstats[i][3]))
        n_comp = n_lab - 1
        # ⚠️ 这个数只是「对角 < min_extent 的**分量个数**」，即**候选**；下面的
        # `_near()` 豁免会把其中相当一部分线段留下来。真正被删掉的线段数在循环末尾记
        # （`n_comp_dropped`）。2026-07 审计项⑤之前这里被直接当成「被删数」用，
        # 拿它调参会得到错误结论。
        n_comp_cand = int((diag[1:] < min_extent).sum())
        hh, ww = ink.shape[:2]

        # 豁免区：贴着「已检出元件框」或「文字框」的短分量。用户的第二句话 ——
        # 导线是用来连接元器件的。jlc 那种靠 net label 表达连通的风格里，GT 的边就是
        # 「引脚到标签」的短桩，本身落在小分量里；只要它从一个元件或一个标签出发，
        # 它就还是导线。把文字框也纳入豁免区是必须的：文字一旦被涂掉，导线就在那里
        # 断开，两截都可能小于 min_extent 而被整条丢掉（KiCad 上 R 0.219→0.129 就是这么来的）。
        boxmask = None
        if rescue_pad > 0 and (occlude or erase):
            boxmask = np.zeros((hh, ww), np.uint8)
            pad = occ_pad + rescue_pad
            for bx in (occlude or []):
                x1, y1, x2, y2 = [float(v) * s for v in bx]
                cv2.rectangle(boxmask,
                              (int(math.floor(x1 - pad)), int(math.floor(y1 - pad))),
                              (int(math.ceil(x2 + pad)), int(math.ceil(y2 + pad))),
                              255, -1)
            pad2 = erase_pad + rescue_pad
            for bx in (erase or []):
                x1, y1, x2, y2 = [float(v) * s for v in bx]
                cv2.rectangle(boxmask,
                              (int(math.floor(x1 - pad2)), int(math.floor(y1 - pad2))),
                              (int(math.ceil(x2 + pad2)), int(math.ceil(y2 + pad2))),
                              255, -1)

        def _near(pt):
            if boxmask is None:
                return False
            ix = int(round(float(pt[0]) * s))
            iy = int(round(float(pt[1]) * s))
            return 0 <= iy < hh and 0 <= ix < ww and boxmask[iy, ix] > 0

        keep = []
        _n_bsegs = len(bsegs)
        stubs_latent = []
        for a, b in bsegs:
            votes: dict = {}
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                ix = int(round((float(a[0]) + (float(b[0]) - float(a[0])) * t) * s))
                iy = int(round((float(a[1]) + (float(b[1]) - float(a[1])) * t) * s))
                if 0 <= iy < hh and 0 <= ix < ww:
                    v = int(lab[iy, ix])
                    if v > 0:
                        votes[v] = votes.get(v, 0) + 1
            _v = max(votes, key=votes.get) if votes else None
            big = bool(votes) and diag[_v] >= min_extent
            if big or not votes or _near(a) or _near(b):
                keep.append([a, b])        # 采不到标签时不删，宁可保守
            else:
                # ★ 2026-10-10（用户 m42009：「导线在等元件确认、元件在等导线确认」）：
                #   这条线段被「小分量 + 不贴任何框」删掉了。它以前只进 `n_comp_dropped`
                #   这个**计数器**（`wires.py:540`），没有任何留痕 —— 于是链断在这里：
                #   元件框被漏检 ⇒ 短桩不贴框 ⇒ 桩删 ⇒ 该元件没有导线触点 ⇒
                #   `pins.py:251` 返回空 ⇒ 网消失。**改成「降权 + 保留池」**：
                #   桩本身不进本轮 `segments`（保持旧行为、分数逐位不变），但记下
                #   端点、分量对角线与「离最近框多远」，供**第二轮交叉验证**用
                #   （`pipeline.py` 的 latent 促进步：引脚/文字/网络证据反过来救桩）。
                _dg = float(diag[_v]) * inv if _v is not None else 0.0
                _db = _dist_to_boxes([a, b], occlude, erase)
                stubs_latent.append({"seg": [[float(a[0]), float(a[1])],
                                             [float(b[0]), float(b[1])]],
                                     "diag": round(_dg, 1), "dbox": round(_db, 1)})
        bsegs = keep
        if stubs_latent:
            try:
                from cvparts.hypo import LEDGER, DEMOTE_HARD
                for _st in stubs_latent:
                    LEDGER.demote(where="wire:stub", value=("%.0f,%.0f" % (_st["seg"][0][0],
                                                                           _st["seg"][0][1])),
                                  score_before=1.0, score_after=DEMOTE_HARD,
                                  reason=("小分量(diag=%.0f<min_extent=%.0f)且不贴任何框"
                                          "(dbox=%.1f)" % (_st["diag"], min_extent, _st["dbox"])),
                                  by="stage1/wires.py:534")
            except Exception:
                pass
        # ★ 审计项⑤：`n_dropped_comp` 现在记的是**真正被删掉的线段数**（分量太小、
        #   且不贴着任何元件/文字框因而没被 `_near()` 豁免）。候选分量数另记
        #   `n_comp_cand`。以前直接拿「候选分量个数」当删除数，调参会看错。
        n_comp_dropped = _n_bsegs - len(bsegs)

    # --- 上游裁剪 ---
    # ★ 2026-10-10（用户 m42009「两边互相等」）：这三道闸门以前只留一个**计数器**
    #   （`n_dropped_len/axis/frame`），被删的线段本身没有任何留痕 ⇒ 一条真导线因为
    #   角度偏心或贴边被删掉时，下游「引脚在等导线」永远等不到、而且查不出原因。
    #   现在每一条都进 `stubs_pruned`（含原因与到最近框的距离）+ 账本降权记录，
    #   供第二轮交叉验证（引脚/文字/网络证据反过来救桩）使用。**不改 `segments`**。
    n_len = n_axis = n_frame = 0
    stubs_pruned: list = []
    segments = []
    for a, b in bsegs:
        L = _seg_len(a, b)
        if L < min_len:
            n_len += 1
            stubs_pruned.append({"seg": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
                                 "reason": "len", "L": round(float(L), 1),
                                 "dbox": round(_dist_to_boxes([a, b], occlude, erase), 1)})
            continue
        dev = _axis_dev(a, b)
        if dev > max_axis_dev:
            n_axis += 1
            stubs_pruned.append({"seg": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
                                 "reason": "axis", "dev": round(float(dev), 1),
                                 "dbox": round(_dist_to_boxes([a, b], occlude, erase), 1)})
            continue
        if drop_frame and _is_frame(a, b, W, H, frame_margin_frac, frame_len_frac):
            n_frame += 1
            stubs_pruned.append({"seg": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
                                 "reason": "frame", "L": round(float(L), 1),
                                 "dbox": round(_dist_to_boxes([a, b], occlude, erase), 1)})
            continue
        segments.append([a, b])
    if stubs_pruned:
        try:
            from cvparts.hypo import LEDGER, DEMOTE_HARD
            for _sp in stubs_pruned:
                LEDGER.demote(where="wire:prune", value=("%.0f,%.0f" % (_sp["seg"][0][0],
                                                                        _sp["seg"][0][1])),
                              score_before=1.0, score_after=DEMOTE_HARD,
                              reason=("上游裁剪 %s（%s，dbox=%.1f）"
                                      % (_sp["reason"],
                                         ("L=%.1f<min_len=%.1f" % (_sp["L"], min_len))
                                         if _sp["reason"] == "len" else
                                         (("dev=%.1f>max_axis_dev=%.1f" % (_sp["dev"], max_axis_dev))
                                          if _sp["reason"] == "axis" else
                                          ("贴边图框线 L=%.1f" % _sp["L"])),
                                         _sp["dbox"])),
                              by="stage1/wires.py:prune")
        except Exception:
            pass

    # 折线：整条丢掉才丢，只要还有一段保留就保留（保守）
    kept = {(round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1))
            for a, b in segments}
    polylines = [pl for pl in blines
                 if any((round(pl[i][0], 1), round(pl[i][1], 1),
                         round(pl[i + 1][0], 1), round(pl[i + 1][1], 1)) in kept
                        for i in range(len(pl) - 1))]

    # ★ 2026-10-10（用户 m42009 的「互相等待」）：框内交叉验证桥接。
    #   默认关（`STAGE1_WIRE_BRIDGE=0`）。理由见 `bridge_boxes` 文档：框内导线证据
    #   被我们自己涂掉，28 张实测框内 R=0.0257 vs 框外 R=0.3444 —— 但「放回全部框内
    #   像素」会凭空长结构，所以只接受双向交叉验证通过的桥。
    bridges: list = []
    b_info: dict = {}
    if env_on("STAGE1_WIRE_BRIDGE", "0"):
        bridges, b_info = bridge_boxes(
            bgr, segments, occlude,
            near=float(os.environ.get("STAGE1_WIRE_BRIDGE_NEAR", "8")),
            max_frac=float(os.environ.get("STAGE1_WIRE_BRIDGE_FRAC", "0.9")),
            angle=float(os.environ.get("STAGE1_WIRE_BRIDGE_ANGLE", "30")),
            ink_min=float(os.environ.get("STAGE1_WIRE_BRIDGE_INK", "0.5")),
            de_min=float(os.environ.get("STAGE1_WIRE_BRIDGE_DE", "22")),
            min_side=float(os.environ.get("STAGE1_WIRE_BRIDGE_MIN_SIDE", "24")),
            thick_max=float(os.environ.get("STAGE1_WIRE_BRIDGE_THICK", "9")))
        if bridges:
            segments = list(segments) + [[list(b[0]), list(b[1])] for b in bridges]

    # --- 在 T 形分叉处切段（`STAGE1_SPLIT_JUNCTIONS=1`）---
    # 病根：pskel 的 snap 会把一条走线接成一整段，即使它中间经过了一个 T 形分叉。
    # 于是 `_rebuild_nodes` 数出 `junctions: 0`（没有 >=3 条线段共用端点），
    # 而官方 GT 的边是在分叉处切开的（`0001` GT 340 条边 vs 我们 153 条）。
    # 切法：任何**别的线段的端点**落在本条线段内部（垂距 <= tol）就在那里切开。
    # 官方 `net_line` 判据是「同一 net 内两条边的端点距离 <= 5px」，切开只会让端点
    # 变多、不会让已有的端点移动，所以这是**严格增加候选**的一步。
    # 220 张干净 A/B（同源检测，只换线段）：`work\stage1_v6` 23.9559 →
    # `work\_s1js` **24.0117（+0.0558）**，`component` 一项逐位不变，增益全在
    # `net_line`（F1 0.0197 → 0.0241、tp 578 → 876、precision 0.0186 → 0.0207）。
    n_split = 0
    if env_on("STAGE1_SPLIT_JUNCTIONS", "1"):
        segments, n_split = _split_at_nodes(
            segments, tol=float(os.environ.get("STAGE1_SPLIT_TOL", "1.5")))

    # --- 在 pskel 的真结点处切段（`STAGE1_SPLIT_JUNC_NODES=1`）---
    # 与上面那条的区别：切点来自**骨架结点**（degree≥3 的像素簇），不是「别的线段端点」。
    # 骨架结点才是图上的 T/十 分叉，GT 的边正是在那里断开的。
    n_jsplit = 0
    if env_on("STAGE1_SPLIT_JUNC_NODES", "0") and jnodes:
        segments, n_jsplit = _split_at_points(
            segments, jnodes, tol=float(os.environ.get("STAGE1_SPLIT_JUNC_TOL", "2.5")))

    endpoints, junctions = _rebuild_nodes(segments, tol=1.0)
    if not segments:
        endpoints, junctions = [], []

    out = {
        "scale": round(s, 4),
        "n_occluded": len(occlude) if occlude else 0,
        "n_erased": len(erase) if erase else 0,
        "n_polylines": len(polylines),
        "n_segments": len(segments),
        "n_segments_raw": n_raw,
        "n_dropped_poly": n_pl_dropped,
        "n_dropped_len": n_len,
        "n_dropped_axis": n_axis,
        "n_dropped_frame": n_frame,
        "n_segments_split": n_split,
        "n_junction_nodes": len(jnodes),
        "n_segments_junc_split": n_jsplit,
        "junction_nodes": jnodes,
        "n_gap_bridges": int(getattr(r, "n_gap_bridges", 0)),
        "n_bridges": len(bridges),
        "bridges": bridges,
        "bridge_info": b_info,
        "n_components": n_comp,
        "n_dropped_comp": n_comp_dropped,
        "n_comp_cand": n_comp_cand,
        # ★ 保留池：被「小分量 + 不贴框」删掉的短桩（端点/分量对角线/离最近框多远）。
        #   不进本轮 `segments`（旧行为不变），留给 `pipeline.py` 的**第二轮交叉验证**
        #   在引脚/文字/网络证据齐了之后反过来救它们。
        "stubs_latent": stubs_latent,
        "n_stubs_latent": len(stubs_latent),
        # ★ 2026-10-10（用户 m42009）：被上游三道裁剪（min_len / max_axis_dev / _is_frame）
        #   删掉的线段也进保留池，附原因与到最近框的距离。以前只有计数器，删了就没了。
        "stubs_pruned": stubs_pruned,
        "n_stubs_pruned": len(stubs_pruned),
        "n_endpoints": len(endpoints),
        "n_junctions": len(junctions),
        "n_nodes": int(r.n_nodes),
        "n_branch_paths": int(r.n_branch_paths),
        "has_chroma": bool(r.has_chroma),
        "n_thin_px": int(r.thin.sum()),
        "polylines": polylines,
        "segments": segments,
        # ★ 审计项⑤：把「问色相」的统计接出来（以前 `_HINT_STAT` 只写不读，
        #   没人看得见 hint 到底命中没有）。`calls`/`hit`/`words` 每张图重置。
        "hint": {"calls": _HINT_STAT["calls"], "hit": _HINT_STAT["hit"],
                 "words": dict(_HINT_STAT["words"])},
        "endpoints": endpoints,
        "junctions": junctions,
    }
    return out, dt


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import schematic_parser as S
    args = sys.argv[1:]
    ids = args or ["0050", "0059", "0011", "0102"]
    for cid in ids:
        c = S.find_case(cid, S.DEFAULT_DATA_ROOT)
        if c is None:
            print(f"{cid}: 找不到"); continue
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        w, t = extract_wires(bgr)
        print(f"{cid:<10}{bgr.shape[1]}x{bgr.shape[0]}  scale {w['scale']}  "
              f"折线 {w['n_polylines']:>5}  线段 {w['n_segments']:>5}  "
              f"端点 {w['n_endpoints']:>5}  结点 {w['n_junctions']:>4}  "
              f"细笔画像素 {w['n_thin_px']:>7}  用时 {t:>6.2f} s")
