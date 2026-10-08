"""第一阶段：引脚点提取（纯几何，不训练、不 OCR）。

**为什么必须有这个模块**：官方评分里 pin 0.25 + net_hypergraph 0.35 + net_line 0.10
= **0.70** 全部以「引脚点」为最小单位，而 stage1 之前只输出元件框、一个引脚点都没有
—— 这三块分数在结构上就拿不到。

## 几何依据（220 张 GT 实测，`work\\gt_pin_geom.py`）

* 引脚**从不在框内**（0.0%）；
* 引脚到框的距离 p25 / p50 / p75 = 9 / 10 / 10 px，<=12px 占 **86.1%**
  ⇒ 默认就是「**框外 10px**」；
* `box`（IC）的引脚贴在左右两条**长边**外侧、沿边分布（left 4268 / right 3577）；
* 两脚元件（r/c/l/d/led）的两个引脚在**长轴方向**、离中心 `L/2 + 10`：
  GT 例 R5 bbox [208,340,216,360] → 引脚 y=330/370，中心 y=350，20/2+10 = 20 ✓；
  `∅202` 方框 10×10 → 引脚离中心 15 = 10/2+10 ✓；
* 一脚符号（gnd/v/net_input/testpoint）只有 1 个脚，上下左右不定。

## 主判据：导线触到框边的位置

原理图的基本画法就是「导线画到引脚处就停」，所以**导线与框边的接触点**就是引脚位置。
类型先验只在「一根线都没触到」时兜底（只有一半引脚会接线）。

**接触点必须朝框外推 10px 才是引脚**：接触点在框边上（误差 0px），而 GT 的引脚在框外
10px —— 不推的话每个引脚都差 10px，而 `pin_tol` 默认只有 5px，会全判错。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# GT 实测的默认外推距离；<=12px 覆盖 86.1%
PIN_OUT = 10.0
# ★ 大 IC（`box`）的引脚桩长度。
# 官方 10GT 上同号引脚点误差普查（`work\tools\diag\_pinpos.py`）显示 GT 的 `box`
# 引脚点离框边**左边 14px、右边 20px**，看起来「推到 15px 更对」—— **但那是假的**：
# 逐张量下来只有 case 1 的框检测偏了，其它图的 GT 引脚点就在 ~10px 处。
# 真正的判据是端到端 A/B（`run.py -input work\_gt10_in` 后 `stage2_eval.py`，
# 见 `work\README.md` 的「引脚桩长度」一节）：
#     PIN_OUT_BOX = 10 -> 42.5191（pin F1 0.1394、tp 26、position_mismatch 12）
#     PIN_OUT_BOX = 15 -> 41.4275（pin F1 0.1021、tp 19、position_mismatch 16）
#     PIN_OUT_BOX = 20 -> 40.0155（pin F1 0.0276、tp  5、position_mismatch 31）
# ⇒ 上面那次 A/B 用的是**很旧的代码**。2025-11 在当前代码上重扫了一遍
#   （`work\tools\eval\fast.py --no-vlm`，`work\_fast_base\od*.json`）：
#     10GT 9 张：8 -> pin 0.0998（42.35? 41.35）｜10 -> 0.1492（42.67）
#                ｜**11 -> 0.1586（42.91）**｜**12 -> 0.1602（42.88）**
#                ｜13 -> 0.1413（42.29）｜14 -> 0.1171（41.69）
#     19 张快测：10 -> pin 0.0737（25.99）｜**12 -> pin 0.0789（26.09）**
#   ⇒ **12.0 在两个集合上都赢**（相对 10：10GT pin +7.4%、19 张 pin +7.1%），
#     所以默认改成 12.0。GT 的引脚外推距离实测 p50 = 10 / p75 = 13（`_pincov.py`），
#     12 正好落在两者之间，符合预期。
# 教训：`STAGE1_PIN_OUT_BOX` 曾经被写成默认 15.0，让整条管线白亏 1.09（10GT）。
PIN_OUT_BOX = float(os.environ.get("STAGE1_PIN_OUT_BOX", "12.0"))
# 「触到框边」的判定半径
TOUCH_TOL = 6.0
# 同一引脚上多个接触点的合并半径（沿框边弧长）
CLUSTER_R = 12.0
# ★ IC（`box`）单独用更小的合并半径。
# 220 张实测 GT 的 `box` 引脚**沿框边的间距 p25=10 p50=10 p75=15 px**
# （`work\tools\diag\_pincov.py`），而 `CLUSTER_R=12` 比这个间距还大 ⇒ 相邻引脚
# 被并成一个簇。实例：`0113` 的 `U4`（MAX7219，24 脚）我们只输出 **7** 个点
# （`work\_submit_out220r21\runs\stage2\0113.json`），而点的位置其实准到 1px
# （我们 x=185.1/305.5 vs GT x=186.0/306.0）——**亏的是覆盖率，不是精度**。
# ⚠️ 默认值仍是 12.0（= `CLUSTER_R`，即行为不变）。虽然理论上「12 > 引脚间距 10」
# 会并掉邻脚，但**实测三个臂都是净亏**（10GT 9 张，`--no-vlm`，`work\_fast_base\`）：
#     12.0（默认）42.67 ｜ 6.0 41.68 ｜ 5.0 41.47 ｜ 4.0 41.47
# 19 张快测：12.0 25.99 ｜ 4.0 25.57（pin 0.0737 → 0.0649）。
# 原因：`pin_f1` 的 tp 要「号 + 名 + 点」全对，**多出来的引脚绝大多数只能变成
# `pin.extra`（fp）**，把 `fn` 换成 `fp+fn` 是亏的。拆分只在「名和号本来就读对了」
# 的引脚上才有收益，而那部分覆盖率极低（`_pinname.py`：`equal` 只有 5.5%）。
# 旋钮留着，等引脚名/号通道修好之后再回来扫。
CLUSTER_IC = float(os.environ.get("STAGE1_PIN_CLUSTER_IC", "12.0"))
# 采样步长（像素）
SAMPLE_STEP = 3.0
# 每个元件最多输出多少个引脚（防毛刺炸开）
MAX_PINS = 64

# ★ 几何引脚桩补检（`STAGE1_PIN_STUB`，默认关）。
# 主判据「导线触到框边」**只能看见接了线的引脚**，而 GT 列出符号的**全部**引脚。
# 220 张实测（`work\tools\diag\_pincov.py`，只算官方 matcher 配上的 358 颗 IC）：
#     GT 引脚 4187 ｜ 我们的引脚点 1901（0.45）
#     `point@5px` 只有 **16.8%**（scorer 的 `cfg.pin_tol` 就是 5.0）
# 而最惨的几张（`0087/U24` 143 脚、`0080/U18` 51 脚）落进了 `pins_from_prior`
# 兜底，只剩 12 个罐头点 —— 因为导线停在引脚桩的**外端**，根本没碰到框。
# `pintrace.find_pins` 找的是「框外朝外的几何小桩」，跟接不接线无关，正好补这一块。
# 代价：桩比真引脚多（40 张抽样 999 桩 vs 750 GT 脚），所以默认关，靠 A/B 决定。
# `union`（把桩并进接触点）实测有**净亏**：19 张 25.99 → 24.68、10GT 42.67 → 39.77
# —— 桩不是真的接线接触点，多出来的点会把网络图搞乱（fp 涨得比 tp 快）。
# ⇒ 只保留 `fill`：**仅在接触点为空**（正在走 `pins_from_prior` 兜底）时才改用桩，
#    这样正好命中 `0087/U24`(143 脚) / `0080/U18`(51 脚) 那类「导线停在桩外端、
#    一个都碰不到框」的符号，不动接触点已经好用的那些 IC。
# `STAGE1_PIN_STUB` = `sparse`（**当前默认**，见下面 ★★）| `0`/`off`（关闭补桩）
#                  | `1`/`fill`（仅兜底时替换）| `union`（全并）。
#
# ★ 2025-10-05 重测（**带 VLM + 当前四旋钮默认**，`fast.py --save …`）：
#     `union` 不再像老读数那样净亏 —— 因为 VLM 的引脚网格通道能从桩上读出真号真名。
#       10GT(9)：默认 58.05 → `union`+`mixed` **58.53**（+0.48，netH 0.4971→0.5051）
#       19 张  ：默认 34.56 → `union`+`mixed` **34.63**（+0.07）
#       隔离臂（10GT）：`DROP_UNREAD=all` 单独 58.05（与 `mixed` 逐位相同）；
#                      `union`+`all` 58.23 < `union`+`mixed` 58.53。
#     `--no-vlm` 下 `union` 仍是亏的（10GT 48.15→47.53、19 张 28.60→28.47）——
#     没有 VLM 时桩只会被 `rebind` 套上错号。**所以 `union` 的价值取决于 VLM 通道。**
#
# ★★ 2025-10-05 「稀疏 IC 才补桩」成为默认（`sparse`，`STAGE1_PIN_STUB_RATIO=0.6`）。
#     触发条件：`len(接触点) < 0.6 * len(几何桩)` —— 只动**明显欠检**的 IC。
#     A/B（全在四旋钮默认之上）：
#       | 集合      | 基线             | `sparse 0.6`      |
#       | 10GT(9) 带VLM | 58.05 (pin .4808) | **59.00 (pin .5040)** |
#       | 19 张 带VLM   | 34.56 (pin .2294) | **35.03 (pin .2357)** |
#       | 10GT(9) 无VLM | 48.15 (pin .2720) | **48.35 (pin .2651)** |
#       | 19 张 无VLM   | 28.60 (pin .1320) | **28.85 (pin .1329)** |
#     比例扫描（10GT 带 VLM）：0.3 → 58.05（等于没触发）｜0.4 → 57.79｜0.5 → 58.23｜
#     **0.6 → 59.00**｜0.7 → 58.18｜0.8 → 57.98｜1.0 → 58.74。0.6 之外都不稳。
_STUB_MODE = os.environ.get("STAGE1_PIN_STUB", "sparse").strip().lower()
STUB_FILL = _STUB_MODE not in ("", "0", "false", "off", "none")
STUB_UNION = _STUB_MODE in ("union", "all", "add")
STUB_SPARSE = _STUB_MODE in ("sparse", "gap", "short")
STUB_SPARSE_RATIO = float(os.environ.get("STAGE1_PIN_STUB_RATIO", "0.6"))

# 两脚元件：引脚在长轴两端
TWO_PIN = frozenset({"r", "c", "l", "d", "led", "fuse", "crystal_2pin", "battery",
                     "battary", "net_short", "switch", "buzzer", "amp"})
# 一脚符号
# ★ `"tp"` 是 `STAGE1_MERGE_TP=1` 时 `stage1/detect.py` 把 `pin`/`testpoint` 合并出的
#   **内部类**（见 detect.py:100-124）。少了它，开关打开后 `pins_from_prior` 在
#   `:263` 匹配不上、一路掉到 `:274 return []` —— 而它的两个父类都在表里，
#   于是「合并」会**顺手删掉测试点/焊盘的兜底脚**（脚不进 pin F1、但进 net 超图 0.35）。
#   两个父类都按一脚处理，合并类必须同款，否则开关打开前后不是同一套语义。
ONE_PIN = frozenset({"gnd", "v", "vcc", "vbus_in", "net_input", "net_output", "pin",
                     "testpoint", "tp", "net_bidirection", "block", "other"})


def sample_segments(segments: Sequence[Sequence[Sequence[float]]],
                    step: float = SAMPLE_STEP) -> np.ndarray:
    """把线段按 `step` 像素密集采样成一个点云 (M,2)。

    采样而不是只取端点：导线常常**从元件框中间穿过去**（串联元件），那时接触点
    在段中间，只看端点会漏掉。
    """
    pts: List[Tuple[float, float]] = []
    for s in segments:
        try:
            ax, ay = float(s[0][0]), float(s[0][1])
            bx, by = float(s[-1][0]), float(s[-1][1])
        except (TypeError, IndexError, ValueError):
            continue
        if not (math.isfinite(ax) and math.isfinite(ay)
                and math.isfinite(bx) and math.isfinite(by)):
            continue
        L = math.hypot(bx - ax, by - ay)
        k = max(1, int(math.ceil(L / max(0.5, step))))
        for i in range(k + 1):
            t = i / k
            pts.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    if not pts:
        return np.zeros((0, 2), np.float64)
    return np.asarray(pts, np.float64)


def _edges(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # 顺序固定：0=上(y1) 1=右(x2) 2=下(y2) 3=左(x1)；图像坐标 y 向下
    return [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]


def _normals(bbox):
    """四条边朝**框外**的单位法线（图像坐标，y 向下）。"""
    return [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]


def outline_hits(pts: np.ndarray, bbox, tol: float = TOUCH_TOL):
    """返回 (edge_idx, arc_len, dist, point) —— 落在框边 `tol` 内的采样点。

    用「到**框边**（不是实心矩形）的距离」：只算实心矩形距离的话，框内部远处的
    点距离也是 0，会把整个框内的东西全当成接触点。
    """
    if not len(pts):
        return np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros((0, 2))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # 每条边算一次：参数 t、垂距 d、弧长 s
    best_d = np.full(len(pts), np.inf)
    best_e = np.zeros(len(pts), int)
    best_s = np.zeros(len(pts))
    for e, ((ax, ay), (bx, by)) in enumerate(_edges(bbox)):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            continue
        t = ((pts[:, 0] - ax) * dx + (pts[:, 1] - ay) * dy) / L2
        t = np.clip(t, 0.0, 1.0)
        cx = ax + t * dx
        cy = ay + t * dy
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        s = t * math.sqrt(L2)
        upd = d < best_d
        best_d = np.where(upd, d, best_d)
        best_e = np.where(upd, e, best_e)
        best_s = np.where(upd, s, best_s)
    keep = best_d <= tol
    return best_e[keep], best_s[keep], best_d[keep], pts[keep]


def _cluster(edges: np.ndarray, arc: np.ndarray, gap: float):
    """按 (边号, 沿边弧长) 聚类；同一条边上弧长间距 > gap 就断开。

    返回 [(edge_idx, [(arc, ...)], ...)]，每组内按弧长排序。
    """
    order = np.lexsort((arc, edges))
    out = []
    cur_e = None
    cur_s: List[float] = []
    for k in order:
        e, s = int(edges[k]), float(arc[k])
        if cur_e is None or e != cur_e or (s - cur_s[-1]) > gap:
            if cur_s:
                out.append((cur_e, cur_s))
            cur_e, cur_s = e, [s]
        else:
            cur_s.append(s)
    if cur_s:
        out.append((cur_e, cur_s))
    return out


def _point_on_edge(bbox, e: int, s: float):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if e == 0:
        return (x1 + s, y1)
    if e == 1:
        return (x2, y1 + s)
    if e == 2:
        return (x2 - s, y2)
    return (x1, y2 - s)


def pins_from_touch(bbox, pts: np.ndarray, tol: float = TOUCH_TOL,
                    cluster_r: float = CLUSTER_R, out_dist: float = PIN_OUT):
    """导线接触点 -> 引脚（朝框外推 `out_dist`）。"""
    e, s, _d, _p = outline_hits(pts, bbox, tol)
    if not len(e):
        return []
    normals = _normals(bbox)
    pins = []
    for ed, arcs in _cluster(e, s, cluster_r):
        smid = float(np.median(arcs))
        bx, by = _point_on_edge(bbox, ed, smid)
        nx, ny = normals[ed]
        pins.append((bx + nx * out_dist, by + ny * out_dist))
    return pins


def pins_from_prior(bbox, ctype: str):
    """没有任何导线触到时，按类型先验兜底。"""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = abs(x2 - x1), abs(y2 - y1)
    t = (ctype or "").split(":")[0]
    if t in TWO_PIN:
        # 引脚在长轴两端，离中心 L/2 + 10（GT 实测）
        if h >= w:
            d = h / 2.0 + PIN_OUT
            return [(cx, cy - d), (cx, cy + d)]
        d = w / 2.0 + PIN_OUT
        return [(cx - d, cy), (cx + d, cy)]
    if t in ONE_PIN:
        # 一脚符号：贴较长的那条边中点朝外；gt 里上下不定，取较长的边更常见
        if h >= w:
            return [(cx, y1 - PIN_OUT)]
        return [(cx, y2 + PIN_OUT)]
    # 2026-07 审计修复：这里原来是 `if t in ONE_PIN or True:`，`or True` 让 ONE_PIN 完全失效、
    # 本行 return [] 永不可达，于是**所有**不在 TWO_PIN 的类（实际就是 box 大 IC）在没有导线
    # 触框时被凭空造出 1 个引脚（长边中点外 10px）。那个假脚随后变成一条单脚网络
    # （nets.py 的 MIN_NET_PINS = 1 挡不住），还可能落进 join_dist(12) 被绑到邻近真网上。
    # 自检抓不到：_selftest 用的 "r"/"v" 本来就在 TWO_PIN/ONE_PIN 里。
    # box 的引脚应由 pins_from_touch / _merge_stubs(几何桩) 给出，先验兜底对 box 无意义。
    return []


def _stub_points(bb, ink, out_dist):
    """`pintrace.find_pins` 的几何引脚桩，统一重排到「框外 `out_dist` px」。"""
    try:
        from . import pintrace as _pt
    except ImportError:                                   # pragma: no cover
        import pintrace as _pt
    x0, y0, x1, y1 = [float(v) for v in bb]
    out = []
    for p in _pt.find_pins(ink, [list(bb)]):
        if p.dx < 0:
            q = (x0 - out_dist, float(p.y))
        elif p.dx > 0:
            q = (x1 + out_dist, float(p.y))
        elif p.dy < 0:
            q = (float(p.x), y0 - out_dist)
        else:
            q = (float(p.x), y1 + out_dist)
        if not any(math.hypot(q[0] - r[0], q[1] - r[1]) <= 3.0 for r in out):
            out.append(q)
    return out


def _merge_stubs(pins, bb, ink, out_dist):
    """把 `pintrace.find_pins` 找到的几何引脚桩并进 `pins`（3px 内去重）。

    桩的原始点是「框外 4px」，这里一律重排到 GT 的约定位置（框外 `out_dist` px），
    和 `pins_from_touch` 的外推保持一致 —— GT 实测引脚到框距离 p50 = 10px，
    且 100% 在框外。
    """
    out = list(pins)
    for q in _stub_points(bb, ink, out_dist):
        if not any(math.hypot(q[0] - r[0], q[1] - r[1]) <= 3.0 for r in out):
            out.append(q)
    return out


def find_pins(components, segments, texts=None, touch_tol: float = TOUCH_TOL,
              cluster_r: float = CLUSTER_R, out_dist: float = PIN_OUT,
              sample_step: float = SAMPLE_STEP, use_prior: bool = True,
              ink=None):
    """给每个元件找引脚。返回 {comp_id: [(x, y), ...]}（图像坐标）。

    `texts` 目前不参与（引脚号/引脚名要 OCR，属于第二阶段），留着是为了以后把
    「框边外的文字框」当成一个额外的引脚证据。

    `ink` 非空且 `STAGE1_PIN_STUB=1` 时，对 `box`/`bigic` 追加几何引脚桩。
    """
    pts = sample_segments(segments, sample_step)
    out: Dict[str, List[Tuple[float, float]]] = {}
    for c in components:
        bb = c.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        _ct = str(c.get("type") or "").lower()
        _is_ic = ("box" in _ct or "bigic" in _ct)
        _od = PIN_OUT_BOX if _is_ic else out_dist
        # IC 的引脚间距只有 ~10px，不能用 12px 的合并半径（会把邻脚并掉）
        _cr = CLUSTER_IC if _is_ic else cluster_r
        pins = pins_from_touch(bb, pts, touch_tol, _cr, _od)
        if not pins and use_prior:
            pins = pins_from_prior(bb, c.get("type") or "")
        if STUB_FILL and ink is not None and _is_ic:
            _sp = _stub_points(bb, ink, _od)
            _few = (STUB_SPARSE and _sp
                    and len(pins) < STUB_SPARSE_RATIO * len(_sp))
            if STUB_UNION or not pins or _few:
                pins = _merge_stubs(pins, bb, ink, _od)
        # 去重（同一位置只留一个），再截断
        dedup: List[Tuple[float, float]] = []
        for p in pins:
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) <= 3.0 for q in dedup):
                dedup.append(p)
        out[c["id"]] = dedup[:MAX_PINS]
    return out



# ------------------------------------------------------------------ 自检
def _selftest() -> int:
    """合成自检。**每次改这个模块都先跑**：本项目已经在「评测函数没自检」上栽过
    三次（网格索引静默漏找、`argmin` 恒为 0、段形状 (M,) vs (M,N)），三次都只有
    真实数据给一个说不清对错的数字。
    """
    fails = []

    def chk(name, got, want, tol=1e-6):
        ok = len(got) == len(want) and all(
            abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
            for a, b in zip(sorted(got), sorted(want)))
        if not ok:
            fails.append(name)
        print("  [%s] %-40s got %s" % ("OK" if ok else "FAIL", name, got))

    box = (100.0, 100.0, 200.0, 300.0)
    print("1) 一根线从左边接到框边中点 -> 引脚应朝框外推 10px")
    chk("左边中点", pins_from_touch(box, sample_segments([[(100, 200), (0, 200)]])),
        [(90.0, 200.0)])

    print("2) 左边两根线，分别在 y=150 / y=250")
    chk("左边两点", pins_from_touch(box, sample_segments([[(100, 150), (0, 150)],
                                                          [(100, 250), (0, 250)]])),
        [(90.0, 150.0), (90.0, 250.0)])

    print("3) 一根线**穿过**框（串联元件）-> 左右各一个引脚")
    chk("穿过", pins_from_touch(box, sample_segments([[(0, 200), (300, 200)]])),
        [(90.0, 200.0), (210.0, 200.0)])

    print("4) 线离框 15px -> 不算触到")
    chk("差 15px", pins_from_touch(box, sample_segments([[(85, 200), (0, 200)]])), [])

    print("5) 线**沿着**左边界画（y 120..200）-> 整段贴边，算 1 个引脚、取中点 y=160")
    # 注意期望值是 160 不是 200：整段都贴边时它本来就是一个「沿边的长接触」，
    # 取这段的中点是合理行为。我第一版把期望写成 200（线的起点），是测试写错了。
    chk("沿边画", pins_from_touch(box, sample_segments([[(100, 200), (100, 120)]])),
        [(90.0, 160.0)])

    print("6) 上边 + 右边各一根 -> 法线方向要对（上边推 -y，右边推 +x）")
    chk("上+右", pins_from_touch(box, sample_segments([[(150, 100), (150, 0)],
                                                       [(200, 200), (300, 200)]])),
        [(150.0, 90.0), (210.0, 200.0)])

    print("7) 类型先验兜底：两脚元件引脚在长轴两端、离中心 L/2+10")
    rbox = (208.0, 340.0, 216.0, 360.0)     # GT 例 R5：引脚 y=330 / 370
    chk("R5 长轴两端", pins_from_prior(rbox, "r"), [(212.0, 330.0), (212.0, 370.0)])

    print("8) 方框电源符号（GT: ∅202 bbox 10x10 -> 引脚离中心 15）")
    chk("方框 1 脚", pins_from_prior((1074.0, 286.0, 1084.0, 296.0), "v"),
        [(1079.0, 276.0)])

    print("9) 有导线时不走先验（即使类型是 r）")
    comps = [{"id": "c1", "type": "r", "bbox_img": [100, 100, 200, 300]}]
    got = find_pins(comps, [[[100, 200], [0, 200]]], use_prior=True)["c1"]
    chk("触到就不兜底", got, [(90.0, 200.0)])

    print("10) 一根线都没有时走先验")
    got = find_pins(comps, [], use_prior=True)["c1"]
    chk("没线走先验", got, [(150.0, 90.0), (150.0, 310.0)])

    print("11) 回归：大 IC（box）没有导线触框时**不得**凭空造脚")
    # 这里以前是 `if t in ONE_PIN or True:`，box 会拿到 1 个假脚（长边中点外 10px），
    # 接着变成一条单脚网络、还可能被 join_dist 绑到邻近真网上。见 pins_from_prior 的注释。
    chk("box 不兜底", pins_from_prior((100.0, 100.0, 200.0, 300.0), "box"), [])
    chk("bigic 不兜底", pins_from_prior((100.0, 100.0, 200.0, 300.0), "bigic"), [])
    _b = [{"id": "b1", "type": "box", "bbox_img": [100, 100, 200, 300]}]
    chk("box 无线无墨 -> 0 脚", find_pins(_b, [], use_prior=True)["b1"], [])

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
