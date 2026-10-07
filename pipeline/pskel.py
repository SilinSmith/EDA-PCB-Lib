# -*- coding: utf-8 -*-
"""Wire skeleton extraction: raster -> drawing graph -> axis-aligned polyline segments.

    ink -> thin-stroke mask (wire/pin strokes only) -> Zhang-Suen skeleton
        -> 8-connected graph, pixels collapsed into nodes (endpoint / T / X / corner)
        -> path per skeleton branch, Douglas-Peucker + 0/45/90 lattice snap
        -> junction resolution: pair branches "straight through" (crossing) vs. unpaired (T)
        -> trace half-edges through the pairings -> maximal polylines
        -> split into two-point edges and union-find them into nets

Everything is in image coordinates (top-left origin); the single y-flip to competition
JSON happens in ``config.img_to_json``.

The stroke-width gate is what makes this work on a greyscale datasheet where colour is
useless: wires are ~2 px strokes, glyphs and symbol bodies are thicker.
"""
from __future__ import annotations

import math
import os

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- morphology

def zhang_suen(img: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen thinning on a uint8 0/255 mask.

    8 shift-add planes and one boolean remove-mask per sub-iteration; returns as soon as a
    sub-iteration deletes nothing, which is the common case for schematic line work (a
    skeleton of a 1675x1190 sheet converges in ~20 sweeps).
    """
    I = (img > 0).astype(np.uint8)
    # cache the eight neighbours once per *pixel*; they only change where we delete
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = np.pad(I, 1)
            p2 = P[0:-2, 1:-1]; p3 = P[0:-2, 2:]; p4 = P[1:-1, 2:]
            p5 = P[2:, 2:];     p6 = P[2:, 1:-1]; p7 = P[2:, 0:-2]
            p8 = P[1:-1, 0:-2]; p9 = P[0:-2, 0:-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            A = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            A += ((p3 == 0) & (p4 == 1))
            A += ((p4 == 0) & (p5 == 1))
            A += ((p5 == 0) & (p6 == 1))
            A += ((p6 == 0) & (p7 == 1))
            A += ((p7 == 0) & (p8 == 1))
            A += ((p8 == 0) & (p9 == 1))
            A += ((p9 == 0) & (p2 == 1))
            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            rm = (I == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if rm.any():
                I[rm] = 0
                changed = True
        if not changed:
            break
    return (I * 255).astype(np.uint8)


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbours(pts: set, p) -> List[Tuple[int, int]]:
    x, y = p
    return [(x + dx, y + dy) for dx, dy in _NB if (x + dx, y + dy) in pts]


# ---------------------------------------------------------------- stroke masks

def estimate_bg(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    ring = np.concatenate([bgr[0:2].reshape(-1, 3), bgr[h - 2:h].reshape(-1, 3),
                           bgr[:, 0:2].reshape(-1, 3), bgr[:, w - 2:w].reshape(-1, 3)])
    return np.median(ring, axis=0).astype(np.float32)


def ink_mask(bgr: np.ndarray, mode: str = 'adaptive', k: int = 31,
             diff_tol: int = 26, c: int = 8) -> np.ndarray:
    """Foreground = everything that is not the sheet background.

    ``adaptive`` cv2 adaptive mean threshold on the inverted grey level (handles the
                 vignetting / gradient backgrounds of the photographed sheets)
    ``otsu``     global Otsu on the inverted grey level
    ``bgdiff``   per-channel distance from the border-ring background colour
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - g
    if mode == 'otsu':
        _, m = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return m
    if mode == 'bgdiff':
        bg = estimate_bg(bgr)
        d = np.abs(bgr.astype(np.float32) - bg).max(axis=2)
        return ((d > diff_tol) * 255).astype(np.uint8)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, k, c)


# `know` 节点用：由 `stage1/wires.py` 问一次模型「导线是什么颜色」后写进来的强制色相。
HINT_HUE = None


def chroma(bgr: np.ndarray) -> np.ndarray:
    b = bgr.astype(np.int16)
    return (b.max(axis=2) - b.min(axis=2)).astype(np.float32)


def thin_stroke_mask(bgr: np.ndarray, ink: np.ndarray, max_half_width: float = 1.6):
    """Keep only ink whose half stroke width (2*EDT) is <= *max_half_width*."""
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    thin = (ink > 0) & (dt <= max_half_width)
    return (thin.astype(np.uint8) * 255), dt


def wire_colour_mask(bgr: np.ndarray, ink: np.ndarray, hue_tol: float = 42.0):
    """Pixels whose hue matches the dominant ink hue (the wire colour), or ``None``.

    ``None`` means "no usable chroma" (greyscale datasheet) -- the caller then keeps the
    stroke-width route only.

    模式由环境变量 ``STAGE1_WIRE_COLOUR`` 控制（**默认 ``hint``**）：

    * ``hint``（默认）—— 由 `stage1/wires.py` 在提线前先问模型一次「这张图的导线是什么
      颜色」（灰度图零成本跳过），把色相写进模块级 ``HINT_HUE``；**只在主色相是红系
      （``h0 < 25`` 或 ``h0 > 155`` = 图框/符号色）时才采纳**，否则维持 ``dominant``。
      19 张快测：24.87 vs ``dominant`` 24.81，导线 F1 0.0146→0.0201、导线 tp 29→52。
    * ``h<N>`` —— 实验用：强制指定导线色相（如 ``h55``）。
    * ``dominant``（旧默认）—— 取 ink 里**像素数最多**的那个色相，只留 ±``hue_tol``。
      **这个假设在多色原理图上会翻车**：KiCad 导出的图里图框与元件符号是暗红
      （色相 ~5°）、导线是绿（~85°）、网络标签是蓝（~125°），红色像素数最多 ⇒
      掩膜把**整张图的导线全删掉**。实测 case ``0001``：`thin` 掩膜剩 67042 px，
      其中绿色只剩 **6 px**（导线全灭）。
    * ``chroma`` —— 只丢掉**没有彩度**的墨迹（文字/灰度底图），保留所有有彩度的
      像素（导线、符号、标签都留）。实测 case ``0001`` 保住绿色 12248/13122 = 0.933。
    * ``auto`` —— **逐图**选导线色：在**图框以内**做 >=40px 的长直线（横/竖开运算），
      选长直线像素最多的那个候选色相。先决条件是「导线是图里唯一的长直线」，
      而图框本身的长直线全在 4.5% 的边带上，排除掉才不会让「框的颜色」赢。
      **实测失效**：它给 ``0001`` 选的还是红 h35（图框内的分隔线比导线更长更多）。
    * ``off`` —— 直接不做颜色过滤。
    """
    # 默认 `hint`：由 `stage1/wires.py` 先问一次模型「这张图的导线是什么颜色」（灰度图会
    # 零成本跳过）。19 张快测实测 `hint` 24.87 > `dominant` 24.81，导线 F1 0.0146→0.0201、
    # 导线 tp 29→52、导线爆 0 从 9 张降到 8 张，官方 10GT(9) 40.13→40.14 不动。
    # 模型答不出来 / 主色相不是红系时都会自动退回 `dominant`，所以是安全默认。
    mode = os.environ.get("STAGE1_WIRE_COLOUR", "hint").strip().lower()
    hint_want = None
    if mode == "hint":
        # `know` 节点：`stage1/wires.py` 先问一次模型「这张图的导线是什么颜色」，写进 `HINT_HUE`。
        # **只在「主色相是红系（图框/符号色）」时才有资格改写** —— 主色相本来就是绿系
        # （比如 `9` 的 h65），说明掩膜没删错东西，别去动它。
        hint_want = HINT_HUE
        mode = "dominant"
    if mode in ("off", "none", "0", "false"):
        return None
    ch = chroma(bgr)
    v = (ink > 0)
    if not v.any() or float(np.percentile(ch[v], 90)) < 22.0:
        return None
    if mode == "chroma":
        return (((ch >= 22.0) & v) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    hist = np.bincount((h[v] // 10).astype(np.int32), minlength=18)
    if mode.startswith("h") and mode[1:].isdigit():
        # 实验用：强制指定导线色相，例如 STAGE1_WIRE_COLOUR=h55 / h85
        h0 = float(mode[1:])
        d = np.abs(h - h0)
        d = np.minimum(d, 180.0 - d)
        return (((d <= hue_tol) & v) * 255).astype(np.uint8)
    if mode == "auto":
        # 逐图选「导线色」：**图框以内**的长直线（>=40px 的横/竖开运算）最多的那个色相。
        # 两个要点：①只在**细笔画**上判（粗的实心填充/图框会淹没直方图）；
        # ②排除 4.5% 的图框边带，否则「框的颜色」永远赢。
        _t, _ = thin_stroke_mask(bgr, ink, 1.6)
        _t = _t > 0
        if not _t.any():
            return None
        h_all = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float32)
        hist_t = np.bincount((h_all[_t] // 10).astype(np.int32), minlength=18)
        H_, W_ = bgr.shape[:2]
        ey, ex = int(H_ * 0.045), int(W_ * 0.045)
        inner = np.zeros((H_, W_), bool)
        inner[ey:H_ - ey, ex:W_ - ex] = True
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        best_h, best_n = None, -1
        cand = [b_ for b_ in range(18) if hist_t[b_] >= max(1, hist_t.max() * 0.02)]
        for b_ in cand:
            h0 = float(b_) * 10.0 + 5.0
            dd = np.abs(h_all - h0)
            dd = np.minimum(dd, 180.0 - dd)
            mm = (((dd <= hue_tol) & _t).astype(np.uint8)) * 255
            lr = (cv2.morphologyEx(mm, cv2.MORPH_OPEN, kh) > 0) | \
                 (cv2.morphologyEx(mm, cv2.MORPH_OPEN, kv) > 0)
            n = int((lr & inner).sum())
            if n > best_n:
                best_h, best_n = h0, n
        h0 = best_h if best_h is not None else float(np.argmax(hist)) * 10.0 + 5.0
    else:
        h0 = float(np.argmax(hist)) * 10.0 + 5.0
    if hint_want is not None:
        # 主色相是「红系」（图框暗红 h~5）才接受模型的答案；否则维持 dominant。
        if h0 < 25.0 or h0 > 155.0:
            h0 = float(hint_want)
        if os.environ.get("STAGE1_WIRE_DEBUG"):
            print("  [wire] hint=%s dominant=%.0f -> h0=%.0f"
                  % (hint_want, float(np.argmax(hist)) * 10.0 + 5.0, h0), flush=True)
    d = np.abs(h - h0)
    d = np.minimum(d, 180.0 - d)
    return (((d <= hue_tol) & v) * 255).astype(np.uint8)



# ---------------------------------------------------------------- graph

@dataclass
class Path:
    idx: int
    n0: int
    n1: int
    pts: List[Tuple[float, float]]                 # xy, from n0 to n1
    length: float = 0.0


@dataclass
class Node:
    idx: int
    x: float
    y: float
    slots: List[int] = field(default_factory=list)      # half-edge ids in this node
    dirs: List[Tuple[float, float]] = field(default_factory=list)
    pair: Dict[int, int] = field(default_factory=dict)  # slot pos -> slot pos
    kind: str = 'E'


@dataclass
class HalfEdge:
    path: int
    end: int                                        # 0 = at path.n0, 1 = at path.n1
    node: int
    partner: int = -1                               # the half-edge from the other end


def build_graph(skel: np.ndarray):
    pts = {(int(y), int(x)) for y, x in zip(*np.nonzero(skel))}
    deg = {}
    for p in pts:
        deg[p] = sum(1 for q in neighbours(pts, p) if q in pts)
    node_px = {p for p in pts if deg[p] != 2}

    used = set()
    raw: List[List[Tuple[int, int]]] = []
    for p in node_px:
        for q in neighbours(pts, p):
            if (p, q) in used:
                continue
            used.add((p, q))
            walk = [p, q]
            prev, cur = p, q
            while cur not in node_px:
                nxt = [r for r in neighbours(pts, cur) if r in pts and r != prev]
                if not nxt:
                    break
                n2 = nxt[0]
                used.add((cur, n2)); used.add((n2, cur))
                walk.append(n2)
                prev, cur = cur, n2
            used.add((cur, prev)); used.add((prev, cur))
            raw.append(walk)
    return pts, deg, node_px, raw


def _dir_at(walk: Sequence[Tuple[int, int]], from_end: int, k: int = 7):
    """Unit direction *leaving* the path: from_end=0 -> leaving walk[0]."""
    w = walk if from_end == 0 else list(reversed(walk))
    if len(w) < 2:
        return (0.0, 0.0), None
    p0 = np.array(w[0], dtype=float)
    p1 = np.array(w[min(k, len(w) - 1)], dtype=float)
    v = p1 - p0
    n = float(np.linalg.norm(v))
    return ((float(v[0] / n), float(v[1] / n)) if n > 0 else (0.0, 0.0)), w


# ---------------------------------------------------------------- fitting

def rdp(points: Sequence[Tuple[float, float]], eps: float = 1.6) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    p0 = np.asarray(points[0], dtype=float)
    p1 = np.asarray(points[-1], dtype=float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1e-9:
        dists = [float(np.hypot(*(np.asarray(p, dtype=float) - p0))) for p in points]
    else:
        dists = [abs(float(np.cross(d, np.asarray(p, dtype=float) - p0))) / L for p in points]
    i = int(np.argmax(dists))
    if dists[i] <= eps:
        return [points[0], points[-1]]
    return rdp(points[:i + 1], eps)[:-1] + rdp(points[i:], eps)


_SNAP_TARGETS = (0.0, 45.0, 90.0, 135.0)


def snap_polyline(poly: Sequence[Tuple[float, float]], tol_deg: float = 9.0):
    """Snap segments to the 0/45/90 lattice (97.1% of 8439 GT edges are exactly axis aligned)."""
    if len(poly) < 2:
        return [tuple(map(float, p)) for p in poly]
    out = [(float(poly[0][0]), float(poly[0][1]))]
    for b in poly[1:]:
        ax, ay = out[-1]
        bx, by = float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        tgt = min(_SNAP_TARGETS, key=lambda t: min(abs(t - ang), 180 - abs(t - ang)))
        if min(abs(tgt - ang), 180 - abs(tgt - ang)) > tol_deg:
            out.append((bx, by)); continue
        if tgt == 0.0:
            out.append((bx, ay))
        elif tgt == 90.0:
            out.append((ax, by))
        elif tgt == 45.0:
            m = (abs(dx) + abs(dy)) / 2.0
            out.append((ax + math.copysign(m, dx), ay + math.copysign(m, dy)))
        else:
            m = (abs(dx) + abs(dy)) / 2.0
            out.append((ax + math.copysign(m, dx), ay - math.copysign(m, dy)))
    return out


def split_segments(poly, min_len: float = 2.0):
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_len:
            out.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return out


# ---------------------------------------------------------------- extractor

@dataclass
class SkeletonResult:
    skel: np.ndarray
    thin: np.ndarray
    ink: np.ndarray
    polylines: List[List[Tuple[float, float]]] = field(default_factory=list)
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    endpoints: List[Tuple[float, float]] = field(default_factory=list)
    junctions: List[Tuple[float, float]] = field(default_factory=list)
    n_nodes: int = 0
    n_branch_paths: int = 0
    has_chroma: bool = False


class WireExtractor:
    def __init__(self, ink_mode: str = 'adaptive', thin_w: float = 1.6,
                 use_colour: bool = True, rdp_eps: float = 1.6, snap: bool = True,
                 min_branch_len: float = 3.0, close_ink: bool = False,
                 close_thin: bool = False, dilate_to: int = 0, adap_c: int = 8,
                 adap_k: int = 31, scale: float = 1.0, max_thin_iter: int = 200):
        self.ink_mode = ink_mode
        self.thin_w = thin_w
        self.use_colour = use_colour
        self.rdp_eps = rdp_eps
        self.snap = snap
        self.min_branch_len = min_branch_len
        self.close_ink = close_ink
        self.close_thin = close_thin
        self.dilate_to = dilate_to
        self.adap_c = adap_c
        self.adap_k = adap_k
        self.scale = scale
        self.max_thin_iter = max_thin_iter

    def foreground(self, bgr: np.ndarray):
        if self.scale and self.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_CUBIC)
        ink = ink_mask(bgr, self.ink_mode, k=self.adap_k, c=self.adap_c)
        if self.close_ink:
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        thin, dt = thin_stroke_mask(bgr, ink, self.thin_w)
        has_chroma = False
        if self.use_colour:
            cm = wire_colour_mask(bgr, ink)
            if cm is not None:
                has_chroma = True
                thin = cv2.bitwise_and(thin, cm)
        if self.close_thin:
            # repair the 1px gaps thinning leaves at stroke crossings --
            # a closing keeps a pixel only when its whole 3x3 neighbourhood is set
            thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if self.dilate_to:
            # conservative "shell" tolerance: a pixel that has a thin-stroke pixel nearby is
            # kept as a wire pixel too, which also repairs the gaps
            near = cv2.dilate(thin, np.ones((self.dilate_to * 2 + 1,) * 2, np.uint8))
            thin = cv2.bitwise_and(near, ink)
        return thin, ink, has_chroma

    def extract(self, bgr: np.ndarray) -> SkeletonResult:
        thin, ink, has_chroma = self.foreground(bgr)
        skel = zhang_suen(thin, max_iter=self.max_thin_iter)
        res = SkeletonResult(skel=skel, thin=thin, ink=ink, has_chroma=has_chroma)
        self._trace(skel, res)
        return res

    # ---------------------------------------------------------- internals
    def _trace(self, skel: np.ndarray, res: SkeletonResult):
        pts, deg, node_px, raw = build_graph(skel)

        # --- collapse adjacent non-degree-2 pixels into one node
        node_list = sorted(node_px)
        pos = {p: i for i, p in enumerate(node_list)}
        parent = list(range(len(node_list)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        # 注意：node_list 在这个循环里不变，集合必须提到循环外。
        # 原先写成 neighbours(set(node_list), p) —— 每次迭代都重建一个 |node_list|
        # 元素的集合，整体代价 O(n^2)。在 0107 上（39,605 个结点像素）这一行独占
        # 92.9 s / 104 s，提前后降到 0.14 s，输出完全一致。
        nset = set(node_list)
        for p in node_list:
            for q in neighbours(nset, p):
                if q in pos:
                    ra, rb = find(pos[p]), find(pos[q])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        clusters: Dict[int, List[Tuple[int, int]]] = {}
        for p in node_list:
            clusters.setdefault(find(pos[p]), []).append(p)
        node_of_px, nodes = {}, []
        for members in clusters.values():
            n = Node(idx=len(nodes), x=float(np.mean([m[1] for m in members])),
                     y=float(np.mean([m[0] for m in members])))
            nodes.append(n)
            for m in members:
                node_of_px[m] = n.idx

        # --- paths
        paths: List[Path] = []
        half: List[HalfEdge] = []
        for walk in raw:
            if len(walk) < 2 or walk[0] not in node_of_px or walk[-1] not in node_of_px:
                continue
            xy = [(float(c), float(r)) for r, c in walk]
            L = math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
            p = Path(idx=len(paths), n0=node_of_px[walk[0]], n1=node_of_px[walk[-1]],
                     pts=xy, length=L)
            paths.append(p)
        res.n_branch_paths = len(paths)

        keep = [p for p in paths if p.length >= self.min_branch_len]
        for p in keep:
            he0 = HalfEdge(path=p.idx, end=0, node=p.n0)
            he1 = HalfEdge(path=p.idx, end=1, node=p.n1)
            he0.partner, he1.partner = len(half) + 1, len(half)
            half.extend([he0, he1])

        # --- node slot directions
        node_slots: Dict[int, List[int]] = {}
        for he in half:
            node_slots.setdefault(he.node, []).append(len([h for h in half[:0]]))
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

        # --- pair branches at every node by "straight through" preference
        for nid, hs in node_slots.items():
            n = nodes[nid]
            k = len(hs)
            if k == 0:
                n.kind = 'E'
                continue
            if k == 1:
                n.kind = 'E'
                continue
            if k == 2:
                n.kind = 'C'
                n.pair[0] = 1
                n.pair[1] = 0
                continue
            cost = np.full((k, k), 1e6, np.float64)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    vi, vj = n.dirs[i], n.dirs[j]
                    cosv = -(vi[0] * vj[0] + vi[1] * vj[1])
                    cost[i, j] = (1.0 - cosv) * 100.0
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
                    big[i, k] = 56.0            # cost of leaving branch i unpaired
                try:
                    from scipy.optimize import linear_sum_assignment
                    r, c = linear_sum_assignment(big)
                except Exception:
                    r, c = np.arange(k), np.arange(k)
                for i, j in zip(r, c):
                    if j < k:
                        n.pair[int(i)] = int(j)
            n.kind = 'T' if k == 3 else 'X'

        # --- trace maximal polylines by jumping across the pairings
        consumed = set()
        polylines: List[List[Tuple[float, float]]] = []
        for he in half:
            if he.partner in consumed:
                continue
            # start at `he`, i.e. enter the path at its `he.end` side
            chain: List[Tuple[float, float]] = []
            cur = he
            guard = 0
            while guard < 10000:
                guard += 1
                if cur.partner in consumed:
                    break
                consumed.add(cur.partner)
                consumed.add(half[cur.partner].partner if half[cur.partner].partner >= 0 else -1)
                p = paths[cur.path]
                # `cur.end` 是我们进入这条 path 的那一端（见 _dir_at 的约定：
                # from_end=0 -> 从 walk[0] 出发）。所以 end==0 时要按原始顺序走。
                # 2026-02 修：这里原先写成 `if cur.end == 1`，方向正好反了 ——
                # 每进入一条 path 都会先瞬移到它的**另一端**再倒着走回来，于是
                # 每条 chain 在每个结点处都产生一个等于该 path 长度的跳变
                # （0101 单张 >30px 的跳变 131 次）。轴对齐的跳变表现为凭空多出
                # 的图纸边框长线（总长比 2.5×），斜向的跳变又被 snap_polyline
                # 吸附成精确 45°，伪装成"大尺度斜线"并躲过 max_axis_dev 过滤。
                seg = p.pts if cur.end == 0 else list(reversed(p.pts))
                chain.extend(seg if not chain else seg[1:])
                exit_he = half[cur.partner]
                nid = exit_he.node
                n = nodes[nid]
                # which slot position does exit_he occupy at this node?
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
                # continue
                consumed.add(nxt_hi)
            if len(chain) >= 2:
                polylines.append(chain)

        res.polylines = polylines
        for chain in polylines:
            poly = rdp(chain, self.rdp_eps)
            if self.snap:
                poly = snap_polyline(poly)
            res.segments.extend(split_segments(poly))
        # ★ 2026-07 审计项⑤：`half` 是**半边表** —— 一个 k 度结点挂着 k 条半边，
        #   所以原来那两条按半边列表推导的写法会把同一个结点坐标在 `junctions`
        #   里**重复 k 次**（`endpoints` 因为条件是 `len(slots) <= 1`，最多命中一次，
        #   不受影响）。目前 `stage1\wires.py:447` 是用 `_rebuild_nodes(segments)`
        #   **重算** endpoints/junctions 的，`pskel` 这两个字段并没有被下游读，
        #   所以这个重复不影响得分；但它会误导任何直接看 `pskel` 输出的人，按结点去重。
        _ep, _jn = [], []
        _seen_ep, _seen_jn = set(), set()
        for hi in half:
            nd = nodes[hi.node]
            if len(nd.slots) <= 1:
                if hi.node not in _seen_ep:
                    _seen_ep.add(hi.node)
                    _ep.append((nd.x, nd.y))
            elif len(nd.slots) >= 3 and hi.node not in _seen_jn:
                _seen_jn.add(hi.node)
                _jn.append((nd.x, nd.y))
        res.endpoints = _ep
        res.junctions = _jn
        res.n_nodes = len(nodes)
        return res


# ---------------------------------------------------------------- net grouping

def group_segments(segs, agg: float = 2.6):
    """Union-find two-point segments into connected wires by endpoint adjacency."""
    n = len(segs)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ends = [(np.array(a, float), np.array(b, float)) for a, b in segs]
    cell: Dict[Tuple[int, int], List[int]] = {}
    for i, (a, b) in enumerate(ends):
        for p in (a, b):
            k = (int(p[0] // agg), int(p[1] // agg))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cell.get((k[0] + dx, k[1] + dy), []):
                        if min(float(np.hypot(*(ends[j][0] - p))),
                               float(np.hypot(*(ends[j][1] - p)))) <= agg:
                            ra, rb = find(i), find(j)
                            if ra != rb:
                                parent[max(ra, rb)] = min(ra, rb)
            cell.setdefault(k, []).append(i)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [[segs[i] for i in g] for g in groups.values()]
