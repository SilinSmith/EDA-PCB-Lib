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
    """把给定的框（图像坐标 [x1,y1,x2,y2]）涂成背景色。"""
    out = bgr.copy()
    if not boxes:
        return out
    h, w = out.shape[:2]
    bg = _background_colour(bgr)
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
            out[y1:y2, x1:x2] = bg
    return out


def _seg_len(a, b) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


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
    """`STAGE1_WIRE_COLOUR=hint` 时问一次模型，把色相写进 `pipeline.pskel.HINT_HUE`。

    ★ 2026-10-08 两处加固（都是为了「2 小时 / 50 张」的时间预算）：
    ① 发出去的是 `_hue_small(bgr)`（最大边 1024），不是整图；
    ② 用 `STAGE1_WIRE_HUE_TIMEOUT`（默认 **30 s**）而不是 `VLM_TIMEOUT`（默认 600 s），
       并且**连续失败 `STAGE1_WIRE_HUE_MAXFAIL`（默认 3）次后就整进程关掉**这一路 ——
       端点没起来时，220 张图不该每张都去等一次超时。
    `_HUE_STAT["skipped"]` 记熔断后跳过的张数，便于事后确认「是省了时间还是丢了几张的提示」。
    """
    if os.environ.get("STAGE1_WIRE_COLOUR", "hint").strip().lower() != "hint":
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
                  thin_w: float = 1.6, rdp_eps: float = 1.6, snap: bool = True,
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
    ex = WireExtractor(ink_mode=ink_mode, thin_w=thin_w, use_colour=use_colour,
                       rdp_eps=rdp_eps, snap=snap)
    t0 = time.time()
    r = ex.extract(src)
    dt = time.time() - t0

    inv = 1.0 / s
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
        for a, b in bsegs:
            votes: dict = {}
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                ix = int(round((float(a[0]) + (float(b[0]) - float(a[0])) * t) * s))
                iy = int(round((float(a[1]) + (float(b[1]) - float(a[1])) * t) * s))
                if 0 <= iy < hh and 0 <= ix < ww:
                    v = int(lab[iy, ix])
                    if v > 0:
                        votes[v] = votes.get(v, 0) + 1
            big = bool(votes) and diag[max(votes, key=votes.get)] >= min_extent
            if big or not votes or _near(a) or _near(b):
                keep.append([a, b])        # 采不到标签时不删，宁可保守
        bsegs = keep
        # ★ 审计项⑤：`n_dropped_comp` 现在记的是**真正被删掉的线段数**（分量太小、
        #   且不贴着任何元件/文字框因而没被 `_near()` 豁免）。候选分量数另记
        #   `n_comp_cand`。以前直接拿「候选分量个数」当删除数，调参会看错。
        n_comp_dropped = _n_bsegs - len(bsegs)

    # --- 上游裁剪 ---
    n_len = n_axis = n_frame = 0
    segments = []
    for a, b in bsegs:
        L = _seg_len(a, b)
        if L < min_len:
            n_len += 1
            continue
        dev = _axis_dev(a, b)
        if dev > max_axis_dev:
            n_axis += 1
            continue
        if drop_frame and _is_frame(a, b, W, H, frame_margin_frac, frame_len_frac):
            n_frame += 1
            continue
        segments.append([a, b])

    # 折线：整条丢掉才丢，只要还有一段保留就保留（保守）
    kept = {(round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1))
            for a, b in segments}
    polylines = [pl for pl in blines
                 if any((round(pl[i][0], 1), round(pl[i][1], 1),
                         round(pl[i + 1][0], 1), round(pl[i + 1][1], 1)) in kept
                        for i in range(len(pl) - 1))]

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
        "n_components": n_comp,
        "n_dropped_comp": n_comp_dropped,
        "n_comp_cand": n_comp_cand,
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
