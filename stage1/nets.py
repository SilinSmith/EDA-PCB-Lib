"""第一阶段：把「引脚点」和「导线图」合成**网络**。

`stage1` 之前只输出元件框 —— `pin 0.25 + net_hypergraph 0.35 + net_line 0.10`
= **0.70 分在结构上拿不到**，因为产物里既没有引脚点也没有网络。这个模块补上那一步：

    段  --(labels.wire_groups)-->  连通分量
    元件框 --(pins.find_pins)-->  引脚点
    引脚点 --(最近段)--> 挂到某个分量
    一个分量 + 挂上来的引脚 = 一个 net

**已知的结构性天花板（必须记住）**：官方 `pin_f1` 的键是 `(component, pin_number)`，
`net_hypergraph` 的键也是 `(component, pin_number)`；只有 `pin_f1` 还比较 `pinname`。
**引脚号/引脚名只能靠 OCR**（第二阶段 `eda-pcb\\textrec\\`），位号（`key`）同理。
所以这里产出的是**位置正确、编号自造**的引脚和网络：几何部分是对的，能拿分的部分
要等第二阶段把编号和名字填进来。别把这个模块的输出当成可以直接提交的东西。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

try:                                    # 作为包导入（python -m stage1）
    from . import labels as labels_mod
    from . import pins as pins_mod
except ImportError:                     # 直接跑脚本（python stage1\nets.py 自检）
    import labels as labels_mod         # noqa: F401
    import pins as pins_mod             # noqa: F401

# 引脚点挂到导线图上时，允许的最大距离（`STAGE1_PIN_JOIN_DIST`，默认 12）。
# ★ 这是一道**硬截断**：`build_nets` 里 `if dist[m] > join_dist: continue` 会把
# 「离任何导线段都超过这个距离」的引脚**整个丢掉**（当成悬空脚、不进任何网络）。
# case `1` 实测 `U1` 的 34 个脚到最近导线段的距离 p50 = 45.6px、p90 = 173px，
# 而只有 8/34 落在 5px 以内 ⇒ 12px 这道闸门很可能是 `net_hypergraph` 召回的大漏勺。
PIN_JOIN_DIST = float(os.environ.get("STAGE1_PIN_JOIN_DIST", "12"))
# ★★ 「不丢弃，只降权」—— owner 的工程哲学落在这里（泛化，不是特例）：
#   上面那道 `join_dist` 是**硬截断**，超出的引脚在 `_pick` 返回 -1 后由
#   `if g < 0: continue` 直接抹掉，**下游再也看不到它存在过**。这违反 owner
#   反复讲的「可以削弱内容的置信度，但不能轻易覆盖 / 丢弃，不然很多关键信息
#   都被覆盖了」。真正的问题不是「半径该多大」（那是拍脑袋调参），而是
#   **判据不该是二值的**：远就是「证据弱」，不是「证据不存在」。
#   所以：引脚**永远**挂到最近的分量上，同时把距离记下来；
#     * `d <= join_dist`      ⇒ strong（和原来一样进网）
#     * `join_dist < d <= far`⇒ weak（进网，但带 `d`，下游可降权/复核）
#     * `d > PIN_JOIN_FAR` 或无线段 ⇒ 进 `report["orphans"]`（仍然**不消失**）
#   `PIN_JOIN_FAR = 0` 时退化成历史行为（weak 区间为空），便于对照。
PIN_JOIN_FAR = float(os.environ.get("STAGE1_PIN_JOIN_FAR", "0"))
# ★★ 「并行导线等距对齐」的紧半径（`STAGE1_PIN_JOIN_SYNTH`，默认 3px）。
#   只作用于 `pincol` **按格点补出来**的那些脚（`pin_synth` 里标 1）。理由：
#   IC 引脚列的实测间距是 **10.0px**（0143 的 U1：GT 64 脚 = 左 32 + 右 32，
#   同一个 y 序列，间距精确 10.0），而 `PIN_JOIN_DIST=12` 比这一格还大 ——
#   一个**悬空**的补脚（框边外 3px 内没有墨）会把隔壁脚的导线认领过来，
#   于是两条本该分开的网络被并成一条。3px 的意义：真实接点就在框边往外
#   0~3px 内（`pincol.CONNECT=3` 就是这条证据的定义），超过就说明它不属于
#   这一脚。**这条不是调参，是把「没有导线」这件事表达了出来的唯一办法。**
#   设 `STAGE1_PIN_JOIN_SYNTH` 为负数或 ≥ `PIN_JOIN_DIST` 即退回历史行为。
PIN_JOIN_SYNTH = float(os.environ.get("STAGE1_PIN_JOIN_SYNTH", "3"))
# ★ 引脚归属模式（`STAGE1_PIN_ATTR`）：
#   `nearest`（默认）= 挂到「最近的那一条段」所属的连通分量（历史行为）。
#   `longest`        = 在 `join_dist + PIN_ATTR_SLACK` 半径内，改挂到**总长最大**的分量。
# 动机：短桩（引脚引线、符号笔画）经常自成一个分量，`nearest` 会把引脚挂到这个
# 「短路组」上，而它真正连的是旁边那条长轨道（case 1 的 29 条网里 12 条段数为 0、
# 右侧 5 条本该被同一根竖线并起来 —— 见 `work\README.md`）。
# `longest` 只是「在同样够得着的前提下，优先相信更长的那条线」，不动任何阈值。
PIN_ATTR = os.environ.get("STAGE1_PIN_ATTR", "nearest").strip().lower()
PIN_ATTR_SLACK = float(os.environ.get("STAGE1_PIN_ATTR_SLACK", "6"))
# 少于这么多引脚的「网络」丢掉（孤立毛刺）。注意它同时也就是「空网丢弃」的开关：
# `MIN_NET_PINS = 1` 已经把 0 引脚的网挡在 `build_nets` 外面，所以不再需要单独的
# `DROP_EMPTY` 常量（2026-07 审计删掉了那个只写不读的装饰常量）。
MIN_NET_PINS = 1


def build_pins(components, segments, texts=None, **kw):
    """`pins.find_pins` 的薄包装（保留一个统一入口，方便以后换实现）。"""
    return pins_mod.find_pins(components, segments, texts=texts, **kw)


def _nearest_seg(pts: np.ndarray, segs: np.ndarray, full: bool = False):
    """每个点到最近段的距离与段下标。返回 (dist (M,), idx (M,))。

    `full=True` 时额外返回完整的 `(N,M)` 距离矩阵（`PIN_ATTR=longest` 要用）。
    """
    if not len(pts) or not len(segs):
        z = np.full(len(pts), np.inf), np.zeros(len(pts), int)
        return z + (np.zeros((len(segs), len(pts))),) if full else z
    a = segs[:, None, :2]                      # (N,1,2)
    b = segs[:, None, 2:]                      # (N,1,2)
    p = pts[None, :, :]                        # (1,M,2)
    ab = b - a
    L2 = (ab ** 2).sum(-1)                     # (N,1)
    L2 = np.where(L2 < 1e-9, 1e-9, L2)
    t = ((p - a) * ab).sum(-1) / L2            # (N,M)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab               # (N,M,2)
    d = np.hypot(*(p - proj).transpose(2, 0, 1))   # (N,M)
    j = np.argmin(d, axis=0)
    out = (d[j, np.arange(len(pts))], j)
    return out + (d,) if full else out


def _component_boxes(components):
    """Read optional image-space boxes without depending on detector classes."""
    boxes = []
    for comp in components or ():
        try:
            box = comp.get("bbox_img") if isinstance(comp, dict) else comp
            x1, y1, x2, y2 = map(float, box)
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _crosses_box_interior(start, end, box):
    """Whether an attachment bridge enters a rectangle's open interior.

    Touching an edge or corner is allowed.  A zero-length bridge adds no new
    connection, even when a detection box overlaps an existing wire.
    """
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False
    lo, hi = 0.0, 1.0
    for origin, step, lower, upper in (
            (start[0], dx, box[0], box[2]),
            (start[1], dy, box[1], box[3])):
        if abs(step) < 1e-9:
            if not lower < origin < upper:
                return False
            continue
        t1, t2 = (lower - origin) / step, (upper - origin) / step
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        if hi - lo <= 1e-9:
            return False
    return hi - lo > 1e-9


def _guard_distances(pts, segs, distances, boxes, radii):
    """Mask local distances in place within the caller's existing search radii."""
    guarded = distances
    blocked = {}
    if not boxes:
        return guarded, blocked
    for m, point in enumerate(pts):
        for k in np.flatnonzero(distances[:, m] <= radii[m]):
            a, b = segs[k, :2], segs[k, 2:]
            delta = b - a
            length2 = float(np.dot(delta, delta))
            t = float(np.clip(np.dot(point - a, delta) / length2, 0.0, 1.0)) if length2 else 0.0
            projection = a + t * delta
            if any(_crosses_box_interior(point, projection, box) for box in boxes):
                guarded[k, m] = np.inf
                blocked.setdefault(m, []).append(int(k))
    return guarded, blocked


def build_nets(comp_pins, segments, tol=None, ext=None, ext_nearest=None,
               weld=None, perp_tol=None, cos_tol=None, ink=None, over=None,
               join_dist: float = PIN_JOIN_DIST, min_pins: int = MIN_NET_PINS,
               join_far: float = PIN_JOIN_FAR, report=None,
               pin_synth=None, join_synth: float = PIN_JOIN_SYNTH,
               components=None, guard_attachments: bool = False,
               pin_connected=None):
    """返回 (nets, seg_groups)。

    `comp_pins`: `{comp_id: [(x, y), ...]}`（图像坐标）
    `nets`: `[{"id": "net_1", "pins": [[comp_id, pin_index], ...],
               "n_seg": k, "seg_index": [...]}]`
    `seg_groups`: 每条段的组号（组号 = 组内最小段下标），空段时是 `[]`。

    `weld/perp_tol/cos_tol` 是「方向性桥接」的三个旋钮，默认取
    `labels_mod.WIRE_WELD / WIRE_PERP / WIRE_COS`（前两个由环境变量
    `STAGE1_WIRE_WELD / STAGE1_WIRE_PERP / STAGE1_WIRE_COS` 控制）。

    `ink` 是**图像坐标（y 向下）**的墨迹掩膜（`uint8`，非 0 为墨）。给了它，
    `weld` 那条路会额外要求两端之间的直线上有墨 —— 这是把 `weld` 放宽
    （`WIRE_WELD=24 / WIRE_PERP=14`）变成只赚不亏的关键（见 `labels._ink_along`）。

    ★ `pin_synth` 是「并行导线等距对齐」的入口：`{comp_id: [0/1, ...]}`，
    与 `comp_pins` 逐位对齐，`1` = 这一脚是 `pincol` **按格点补出来**的
    （框边外侧没有墨 ⇒ 没有导线接上去）。对这些脚改用 `join_synth`（默认 3px）
    而不是 `join_dist`（默认 12px）去挑分段。**为什么必须分开**：IC 的引脚列
    间距只有 ~10px，而 `join_dist=12` 比它还大，一个悬空的补脚会把**隔壁脚的
    导线**认领过来，于是两条本该分开的网络被并成一条 —— 症状是 `net_hypergraph`
    的 P 虚高、R 被拖平。给了 `pin_synth=None`（默认）时行为与历史完全一致。

    `guard_attachments=True` opts into conservative attachment checks. Optional
    `components` contain `bbox_img` rectangles (or are bare image-space boxes):
    a bridge to a wire cannot pass through a component's interior. Equal-distance
    choices in different wire groups are kept in `report["ambiguous"]`, rather
    than resolved by segment order. Same-group ties are harmless. Neither check
    expands a search radius or changes wire grouping; the default remains off.

    `pin_connected` is optional measured color evidence, aligned by component
    and pin index. Only explicit False prevents network attachment (a measured
    floating red stub or an NC mark). Missing/None evidence keeps the legacy
    behavior; nearby skeleton noise must not connect a known floating pin.
    """
    segs = segments if isinstance(segments, np.ndarray) else labels_mod._as_segs(segments)
    tol = labels_mod.WIRE_TOL if tol is None else tol
    ext = labels_mod.WIRE_EXT if ext is None else ext
    if ext_nearest is None:
        ext_nearest = labels_mod.WIRE_EXT_NEAREST
    if weld is None:
        weld = getattr(labels_mod, "WIRE_WELD", 0.0)
    if perp_tol is None:
        perp_tol = getattr(labels_mod, "WIRE_PERP", 1.5)
    if cos_tol is None:
        cos_tol = getattr(labels_mod, "WIRE_COS", -0.75)
    if over is None:
        over = getattr(labels_mod, "WIRE_OVER", 0.0)

    if not len(segs):
        if report is not None:
            for cid, plist in comp_pins.items():
                for k, point in enumerate(plist):
                    report.setdefault("orphans", []).append({
                        "comp": cid, "pin": k,
                        "x": round(float(point[0]), 1), "y": round(float(point[1]), 1),
                        "d": None})
        return [], []

    seg_groups = labels_mod.wire_groups(
        segs, tol=tol, ext=ext, ext_nearest=ext_nearest,
        weld=weld, perp_tol=perp_tol, cos_tol=cos_tol, over=over,
        ink=(ink if getattr(labels_mod, "WIRE_INK", True) else None),
        ink_frac=getattr(labels_mod, "WIRE_INK_FRAC", 0.5))

    # 收集所有引脚点，并记住它属于哪个元件、第几个脚
    pts, owner, jd, strict_synth, disconnected = [], [], [], [], []
    synth_radius = float(join_synth)
    for cid, plist in comp_pins.items():
        sf = (pin_synth or {}).get(cid) or []
        cf = (pin_connected or {}).get(cid) or []
        for k, p in enumerate(plist):
            pts.append((float(p[0]), float(p[1])))
            owner.append((cid, k))
            disconnected.append(k < len(cf) and cf[k] is False)
            # 「补出来的脚」用紧半径挑导线，其余用 `join_dist`。见 `PIN_JOIN_SYNTH`。
            # Negative or nonrestrictive values disable the synthetic-pin guard,
            # as documented by PIN_JOIN_SYNTH. A strict guard also applies to
            # weak attachment: join_far must not resurrect a known unconnected pin.
            restricted = bool(k < len(sf) and sf[k] and 0 <= synth_radius < float(join_dist))
            strict_synth.append(restricted)
            jd.append(synth_radius if restricted else float(join_dist))
    if not pts:
        return [], seg_groups
    P = np.asarray(pts, np.float64)
    JD = np.asarray(jd, np.float64)
    use_longest = (PIN_ATTR == "longest")
    blocked, ambiguous = {}, {}
    if use_longest or guard_attachments:
        dist, sidx, D = _nearest_seg(P, segs, full=True)
    else:
        dist, sidx = _nearest_seg(P, segs)

    if guard_attachments:
        radii = np.where(np.asarray(strict_synth), JD, np.maximum(JD, float(join_far)))
        D, blocked = _guard_distances(P, segs, D, _component_boxes(components), radii)
        sidx = np.argmin(D, axis=0)
        dist = D[sidx, np.arange(len(P))]
        for m, nearest in enumerate(dist):
            if not np.isfinite(nearest) or nearest > radii[m]:
                continue
            tied = np.flatnonzero(np.abs(D[:, m] - nearest) <= 1e-6)
            groups = sorted({int(seg_groups[k]) for k in tied})
            if len(groups) > 1:
                ambiguous[m] = groups
        if report is not None:
            for m, indices in blocked.items():
                report.setdefault("blocked", []).append({
                    "comp": owner[m][0], "pin": owner[m][1], "segments": indices})
            for m, groups in ambiguous.items():
                report.setdefault("ambiguous", []).append({
                    "comp": owner[m][0], "pin": owner[m][1],
                    "x": round(float(P[m][0]), 1), "y": round(float(P[m][1]), 1),
                    "d": round(float(dist[m]), 1), "groups": groups})

    # 每个分量的段下标 + 总长（`longest` 模式的排序键）
    groups_idx: Dict[int, List[int]] = {}
    groups_len: Dict[int, float] = {}
    if use_longest:
        for k in range(len(segs)):
            g = int(seg_groups[k])
            groups_idx.setdefault(g, []).append(k)
            groups_len[g] = groups_len.get(g, 0.0) + float(
                math.hypot(segs[k][2] - segs[k][0], segs[k][3] - segs[k][1]))
        g_keys = sorted(groups_idx)

    def _pick(m: int) -> int:
        """给第 m 个引脚挑一个分量号（返回 -1 = 悬空）。"""
        if disconnected[m]:
            return -1
        # 每脚自己的接线半径：补出来的脚只有 `join_synth`（见 `PIN_JOIN_SYNTH`）。
        # `pin_synth` 没给时 `JD[m]` 恒等于参数 `join_dist`，行为与历史一致。
        jdm = float(JD[m])
        if m in ambiguous:
            return -1
        if use_longest:
            near = float(dist[m])
            # ★ 2026-07 审计修复：这两处原来写的是模块常量 `PIN_JOIN_DIST`，
            #   而 `nearest` 支用的是**参数** `join_dist`。只要有人给
            #   `build_nets(..., join_dist=X)` 传了非默认值又开了
            #   `STAGE1_PIN_ATTR=longest`，这个参数就会被**静默忽略**，
            #   两条分支用不同的接线半径。当前调用点（`stage1\pipeline.py:254`）
            #   没传 join_dist，所以行为不变，但这颗雷必须拆。
            if near > jdm:
                return -1
            # 先看「最近的那条」；再在 slack 半径内找总长最大的分量
            best_g, best_len = int(seg_groups[int(sidx[m])]), float(
                groups_len[int(seg_groups[int(sidx[m])])])
            lim = min(jdm, near + PIN_ATTR_SLACK)
            for g in g_keys:
                dm = float(D[groups_idx[g], m].min())
                if dm > lim:
                    continue
                if groups_len[g] > best_len:
                    best_g, best_len = g, groups_len[g]
            return best_g
        if dist[m] > jdm:
            return -1
        return int(seg_groups[int(sidx[m])])

    buckets: Dict[int, List] = {}
    _far = float(join_far)
    for m in range(len(P)):
        g = _pick(m)
        if g < 0:
            d = float(dist[m])
            rec = {"comp": owner[m][0], "pin": owner[m][1],
                   "x": round(float(P[m][0]), 1), "y": round(float(P[m][1]), 1),
                   "d": (round(d, 1) if np.isfinite(d) else None)}
            if disconnected[m]:
                rec["reason"] = "color_stub_without_connected_wire"
            # ★ 「不丢弃，只降权」：`join_far` 给了就说明调用方要的是**软判据** ——
            #   这一段距离里的引脚照样挂上（挂最近的那个分量），只是记进
            #   `report["weak"]` 让下游能降权/复核。`join_far=0`（默认）时
            #   weak 区间为空，行为与历史完全一致。
            if (not disconnected[m] and not strict_synth[m] and m not in ambiguous
                    and _far > 0 and np.isfinite(d) and d <= _far):
                g = int(seg_groups[int(sidx[m])])
                if report is not None:
                    report.setdefault("weak", []).append(rec)
                buckets.setdefault(g, []).append([owner[m][0], owner[m][1]])
                continue
            # 真的够不着：**仍然不消失**，进 `orphans` 留痕（下游/复核轮可见）
            if report is not None:
                report.setdefault("orphans", []).append(rec)
            continue                       # 无接线（悬空引脚）——但已留痕
        buckets.setdefault(g, []).append([owner[m][0], owner[m][1]])

    nets = []
    for i, g in enumerate(sorted(buckets)):
        pin_list = buckets[g]
        if len(pin_list) < min_pins:
            continue
        seg_index = [int(k) for k in range(len(segs)) if int(seg_groups[k]) == g]
        nets.append({"id": "net_%d" % (i + 1), "pins": pin_list,
                     "n_seg": len(seg_index), "seg_index": seg_index})
    # ★ 2026-07 审计修复：这里原来有一句
    #     `if DROP_EMPTY: pass`
    #   它是**装饰**——`DROP_EMPTY` 恒为 True 却什么都不做，真正的「空网丢弃」
    #   由上一条 `if len(pin_list) < min_pins: continue` 完成（`MIN_NET_PINS = 1`）。
    #   读代码的人会以为这儿有个开关，其实没有。常量一并删掉。
    return nets, seg_groups


def summarise(comp_pins, nets) -> str:
    n_pin = sum(len(v) for v in comp_pins.values())
    n_used = sum(len(n["pins"]) for n in nets)
    return ("引脚 %d（挂在网络上 %d，悬空 %d）/ 网络 %d"
            % (n_pin, n_used, n_pin - n_used, len(nets)))


# ------------------------------------------------------------------ 自检
def _selftest() -> int:
    """合成自检：两条分开的走线 + 四个元件引脚，应该得到 2 个网络、4 个引脚。"""
    fails = []

    def chk(name, got, want):
        ok = got == want
        if not ok:
            fails.append(name)
        print("  [%s] %-34s got %s want %s" % ("OK" if ok else "FAIL", name, got, want))

    print("1) 两条互不相干的走线，各挂 2 个引脚 -> 2 个网络")
    segs = [[(0, 0), (100, 0)], [(0, 500), (100, 500)]]
    cp = {"a": [(50.0, 0.0), (10.0, 0.0)], "b": [(60.0, 500.0), (20.0, 500.0)]}
    nets, groups = build_nets(cp, segs, tol=1.0, ext=0.0)
    chk("网络数", len(nets), 2)
    chk("挂在网络上的引脚数", sum(len(n["pins"]) for n in nets), 4)
    chk("每条段自成一组", sorted(set(groups)), sorted(groups))

    print("2) 两条段端点相接 -> 并成一个网络")
    segs2 = [[(0, 0), (100, 0)], [(100, 0), (100, 80)]]
    cp2 = {"a": [(50.0, 0.0)], "b": [(100.0, 80.0)]}
    nets2, _ = build_nets(cp2, segs2, tol=1.5, ext=0.0)
    chk("拐角并成 1 个网络", len(nets2), 1)
    chk("该网络 2 个引脚", len(nets2[0]["pins"]), 2)

    print("3) 悬空引脚（离导线 100px）不进任何网络")
    cp3 = {"a": [(50.0, 0.0)], "z": [(50.0, 100.0)]}
    nets3, _ = build_nets(cp3, [[(0, 0), (100, 0)]], tol=1.0, ext=0.0)
    chk("只有 1 个引脚进网", sum(len(n["pins"]) for n in nets3), 1)

    print("4) 断了 8px 的两条段：ext=0 是 2 个网络，ext=8 应并成 1 个")
    segs4 = [[(0, 0), (100, 0)], [(108, 0), (200, 0)]]
    cp4 = {"a": [(50.0, 0.0)], "b": [(150.0, 0.0)]}
    n0, _ = build_nets(cp4, segs4, tol=1.0, ext=0.0, weld=0.0)
    n1, _ = build_nets(cp4, segs4, tol=1.0, ext=8.0, weld=0.0)
    chk("ext=0", len(n0), 2)
    chk("ext=8 朝前延伸接上", len(n1), 1)

    print("")
    if fails:
        print("**有 %d 个 FAIL，禁止把结果当真**：%s" % (len(fails), fails))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(_selftest())
